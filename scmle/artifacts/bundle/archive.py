"""Writing and loading bundles: verified before anything is unpickled."""

import contextlib
import dataclasses
import json
import os
import pathlib
import tempfile
from typing import Any
from typing import Self

import joblib
import pandas

from scmle import durability
from scmle import envelope
from scmle.artifacts.bundle import core
from scmle.artifacts.bundle import layout
from scmle.artifacts.bundle import provenance
from scmle.artifacts.bundle import seal
from scmle.contracts import artifacts
from scmle.contracts import errors
from scmle.contracts import signing
import scmle.data.reference
import scmle.evaluation.conformal


def unpack(
    root: pathlib.Path,
) -> tuple[dict[str, Any], artifacts.Interface]:
    """Reads the manifest and the schema of a bundle directory.

    Args:
        root: The bundle directory.

    Returns:
        The manifest as a dict and the parsed schema.

    Raises:
        ArtifactError: If a file is missing or the metadata is invalid.
    """
    try:
        manifest = json.loads(
            (root / layout.MANIFEST_FILE).read_text(encoding="utf-8")
        )
        schema = artifacts.Interface.from_dict(
            json.loads((root / layout.SCHEMA_FILE).read_text(encoding="utf-8"))
        )
    except FileNotFoundError as error:
        raise errors.ArtifactError(
            f"not a model bundle (missing {error.filename})"
        ) from error
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise errors.ArtifactError("invalid model bundle metadata") from error
    return manifest, schema


def admit(
    manifest: dict[str, Any], trusted: bool, allow_legacy_format: bool
) -> tuple[provenance.Manifest, bool]:
    """Accepts a bundle's format and reads its manifest.

    Args:
        manifest: The manifest as a dict.
        trusted: Whether the caller vouches for the bundle.
        allow_legacy_format: Whether a trusted format 1 bundle may load.

    Returns:
        The manifest and whether the bundle is format 1, which has no schema
        checksum.

    Raises:
        ArtifactError: If the format is not this build's (a trusted format 1
            bundle aside, when allowed) or the schema checksum is missing.
    """
    legacy = manifest.get("format") == 1
    if manifest.get("format") != layout.BUNDLE_FORMAT and not (
        legacy and allow_legacy_format and trusted
    ):
        raise errors.ArtifactError(
            f"unsupported bundle format {manifest.get('format')}: this "
            f"build reads format {layout.BUNDLE_FORMAT}. A bundle stores "
            "the module path and class name of the objects inside it, and "
            "those names changed, so a bundle written by an earlier "
            "build cannot be loaded; retrain and save it again"
        )
    info = provenance.Manifest.read(manifest)
    if layout.SCHEMA_FILE not in info.files and not legacy:
        raise errors.ArtifactError(
            "bundle is missing a schema checksum; re-export it with "
            f"bundle format {layout.BUNDLE_FORMAT}"
        )
    return info, legacy


def authenticate(
    root: pathlib.Path,
    info: provenance.Manifest,
    legacy: bool,
    trusted: bool,
    signing_key: str | bytes | None,
) -> None:
    """Authenticates a bundle and checks its model and schema checksums.

    Nothing is unpickled here. `trusted` vouches for a missing signature
    only: a signature that cannot be checked, because no key is set, may be
    an invalid one, so it is refused like one.

    Args:
        root: The bundle directory.
        info: The bundle's manifest.
        legacy: Whether the bundle is format 1, which has no schema checksum.
        trusted: Whether the caller vouches for an unsigned bundle.
        signing_key: HMAC key; defaults to `$SCMLE_SIGNING_KEY`.

    Raises:
        ArtifactError: If the bundle is unauthenticated, signed but no key is
            set to check it, lacks its model file, or a checksum differs.
    """
    authentic = seal.verify_signature(root, signing_key)
    if not authentic and not trusted:
        raise errors.ArtifactError(
            "refusing to load an unauthenticated bundle: it is pickle-"
            "based and loading executes code. Sign bundles with "
            f"${signing.SIGNING_KEY_ENV}, or pass trusted=True if you trust "
            "its source."
        )
    if not authentic and (root / seal.SIGNATURE_FILE).exists():
        raise errors.ArtifactError(
            "refusing a signed bundle with no signing key set to verify it: "
            "trusted=True vouches for a missing signature, never for one "
            f"that may be invalid. Set ${signing.SIGNING_KEY_ENV} or pass "
            "signing_key."
        )
    model_path = root / layout.MODEL_FILE
    if not model_path.exists():
        raise errors.ArtifactError(f"missing {layout.MODEL_FILE}")
    if layout.shasum(model_path) != info.files.get(layout.MODEL_FILE):
        raise errors.ArtifactError(
            "checksum mismatch: model file was modified or corrupted"
        )
    schema_path = root / layout.SCHEMA_FILE
    if not legacy and layout.shasum(schema_path) != info.files.get(
        layout.SCHEMA_FILE
    ):
        raise errors.ArtifactError(
            "checksum mismatch: schema file was modified or corrupted"
        )


def resolve(
    root: pathlib.Path, info: provenance.Manifest, name: str, modified: str
) -> pathlib.Path | None:
    """Returns the path of a listed bundle member whose checksum matches.

    Args:
        root: The bundle directory.
        info: The bundle's manifest.
        name: File name of the member.
        modified: Words that finish "checksum mismatch: " in the error.

    Returns:
        The member's path, or None if the manifest does not list it.

    Raises:
        ArtifactError: If the member is missing or its checksum differs.
    """
    if name not in info.files:
        return None
    path = root / name
    if not path.exists() or layout.shasum(path) != info.files[name]:
        raise errors.ArtifactError(f"checksum mismatch: {modified}")
    return path


def unpickle(path: pathlib.Path) -> Any:
    """Loads one verified, pickled member of a bundle.

    Args:
        path: The member file, already checksummed.

    Returns:
        The stored object.

    Raises:
        ArtifactError: If the member names a module or class this build
            does not have, as one written by another build does.
    """
    try:
        return joblib.load(path)
    except (ModuleNotFoundError, AttributeError) as error:
        raise errors.ArtifactError(
            f"{path.name} names a module or class this build does not have "
            f"({error}); the bundle was written by another build, so retrain "
            "and save it again"
        ) from error


def hydrate(
    root: pathlib.Path, info: provenance.Manifest
) -> tuple[Any, Any, Any, Any, Any]:
    """Loads the verified reference, history, calibrator, conformal, novelty.

    Args:
        root: The bundle directory.
        info: The bundle's manifest.

    Returns:
        The reference profile, series history, calibrator, conformal
        calibrator and novelty detector, each None when the bundle lacks it.

    Raises:
        ArtifactError: If a listed member is missing or its checksum differs.
    """
    reference = None
    path = resolve(
        root, info, layout.REFERENCE_FILE, "reference profile was modified"
    )
    if path is not None:
        reference = scmle.data.reference.Baseline.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
    history = None
    path = resolve(
        root, info, layout.HISTORY_FILE, "series history was modified"
    )
    if path is not None:
        history = pandas.read_parquet(path)
    calibrator = None
    path = resolve(
        root, info, layout.CALIBRATOR_FILE, "calibrator was modified"
    )
    if path is not None:
        calibrator = unpickle(path)
    calibrated = None
    path = resolve(
        root, info, layout.CONFORMAL_FILE, "conformal scores were modified"
    )
    if path is not None:
        calibrated = scmle.evaluation.conformal.ConformalCalibrator.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
    detector = None
    path = resolve(
        root, info, layout.NOVELTY_FILE, "novelty detector was modified"
    )
    if path is not None:
        detector = unpickle(path)
    return reference, history, calibrator, calibrated, detector


def restore(
    root: pathlib.Path, info: provenance.Manifest
) -> tuple[Any, Any, Any, Any]:
    """Loads the verified quantile, CV+, coverage and shift-weight members.

    Args:
        root: The bundle directory.
        info: The bundle's manifest.

    Returns:
        The quantile models, CV+ fold models, coverage report and shift
        weights, each None when the bundle lacks it.

    Raises:
        ArtifactError: If a listed member is missing or its checksum differs.
    """
    intervals = None
    path = resolve(
        root, info, layout.QUANTILE_FILE, "quantile models were modified"
    )
    if path is not None:
        intervals = unpickle(path)
    cv_plus_models = None
    path = resolve(
        root, info, layout.CV_PLUS_FILE, "CV+ fold models were modified"
    )
    if path is not None:
        cv_plus_models = unpickle(path)
    stored_coverage = None
    path = resolve(
        root, info, layout.COVERAGE_FILE, "coverage report was modified"
    )
    if path is not None:
        stored_coverage = json.loads(path.read_text(encoding="utf-8"))
    stored_weights = None
    path = resolve(
        root, info, layout.SHIFT_WEIGHTS_FILE, "shift weights were modified"
    )
    if path is not None:
        stored_weights = unpickle(path)
    return intervals, cv_plus_models, stored_coverage, stored_weights


class Archive(core.Core):
    """Writing, re-signing and loading a bundle directory."""

    def persist(self, root: pathlib.Path, files: dict[str, str]) -> None:
        """Writes the reference, calibrator, conformal and novelty members.

        A stale calibrator file is deleted when the bundle has no calibrator.

        Args:
            root: The bundle directory.
            files: The checksum table of the written files; extended in place
                in the order the members are written.
        """
        if self.reference is not None:
            reference_path = root / layout.REFERENCE_FILE
            reference_path.write_text(
                json.dumps(self.reference.to_dict()),
                encoding="utf-8",
                newline="\n",
            )
            files[layout.REFERENCE_FILE] = layout.shasum(reference_path)
        if self.calibrator is not None:
            calibrator_path = root / layout.CALIBRATOR_FILE
            joblib.dump(self.calibrator, calibrator_path)
            files[layout.CALIBRATOR_FILE] = layout.shasum(calibrator_path)
        elif (root / layout.CALIBRATOR_FILE).exists():
            (root / layout.CALIBRATOR_FILE).unlink()
        if self.conformal is not None:
            conformal_path = root / layout.CONFORMAL_FILE
            conformal_path.write_text(
                json.dumps(self.conformal.to_dict()),
                encoding="utf-8",
                newline="\n",
            )
            files[layout.CONFORMAL_FILE] = layout.shasum(conformal_path)
        if self.novelty is not None:
            novelty_path = root / layout.NOVELTY_FILE
            joblib.dump(self.novelty, novelty_path)
            files[layout.NOVELTY_FILE] = layout.shasum(novelty_path)

    def write_interval_members(
        self, root: pathlib.Path, files: dict[str, str]
    ) -> None:
        """Writes the quantile, CV+, coverage, shift-weight and history members.

        A stale history file is deleted when the bundle has no history.

        Args:
            root: The bundle directory.
            files: The checksum table of the written files; extended in place
                in the order the members are written.
        """
        if self.quantile is not None:
            quantile_path = root / layout.QUANTILE_FILE
            joblib.dump(self.quantile, quantile_path)
            files[layout.QUANTILE_FILE] = layout.shasum(quantile_path)
        if self.cv_plus is not None:
            cv_plus_path = root / layout.CV_PLUS_FILE
            joblib.dump(self.cv_plus, cv_plus_path)
            files[layout.CV_PLUS_FILE] = layout.shasum(cv_plus_path)
        if self.coverage_report is not None:
            coverage_path = root / layout.COVERAGE_FILE
            coverage_path.write_text(
                json.dumps(self.coverage_report, sort_keys=True),
                encoding="utf-8",
                newline="\n",
            )
            files[layout.COVERAGE_FILE] = layout.shasum(coverage_path)
        if self.shift_weights is not None:
            shift_path = root / layout.SHIFT_WEIGHTS_FILE
            joblib.dump(self.shift_weights, shift_path)
            files[layout.SHIFT_WEIGHTS_FILE] = layout.shasum(shift_path)
        if self.history is not None:
            history_path = root / layout.HISTORY_FILE
            self.history.to_parquet(history_path, index=False)
            files[layout.HISTORY_FILE] = layout.shasum(history_path)
        elif (root / layout.HISTORY_FILE).exists():
            (root / layout.HISTORY_FILE).unlink()

    def save(
        self,
        directory: str | pathlib.Path,
        signing_key: str | bytes | None = None,
    ) -> pathlib.Path:
        """Writes the bundle, signing it when a key is available.

        Args:
            directory: Destination directory (created if needed).
            signing_key: HMAC key; defaults to `$SCMLE_SIGNING_KEY`. An
                unsigned bundle can only be loaded with `trusted=True`.
                Without a key, an existing `signature.json` is deleted.

        Returns:
            The directory path.
        """
        root = pathlib.Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        model_path = root / layout.MODEL_FILE
        joblib.dump(self.pipeline, model_path)
        (root / layout.SCHEMA_FILE).write_text(
            json.dumps(self.schema.to_dict(), indent=2),
            encoding="utf-8",
            newline="\n",
        )
        schema_path = root / layout.SCHEMA_FILE
        files = {
            layout.MODEL_FILE: layout.shasum(model_path),
            layout.SCHEMA_FILE: layout.shasum(schema_path),
        }
        self.persist(root, files)
        self.write_interval_members(root, files)
        info = dataclasses.replace(
            self.info, format=layout.BUNDLE_FORMAT, files=files
        )
        manifest_text = json.dumps(info.to_dict(), indent=2)
        (root / layout.MANIFEST_FILE).write_text(
            manifest_text, encoding="utf-8", newline="\n"
        )
        signature_path = root / seal.SIGNATURE_FILE
        key = signing.keying(signing_key)
        if key is not None:
            signature = seal.sign(manifest_text.encode(), key)
            signature_path.write_text(
                json.dumps(
                    {
                        "algorithm": seal.SIGNATURE_ALGORITHM,
                        "signature": signature,
                    }
                ),
                encoding="utf-8",
                newline="\n",
            )
        elif signature_path.exists():
            signature_path.unlink()
        return root

    @classmethod
    def rotate_signing_key(
        cls,
        directory: str | pathlib.Path,
        current_key: str | bytes,
        new_key: str | bytes,
    ) -> pathlib.Path:
        """Re-signs an existing bundle without reserializing its model.

        The current signature must verify before the new signature is written.
        Rotation changes only ``signature.json`` and uses an atomic replace, so
        a failed rotation leaves the previously valid signature in place.

        Args:
            directory: Bundle directory.
            current_key: The key the bundle is currently signed with.
            new_key: The key to sign with from now on.

        Returns:
            The bundle directory.

        Raises:
            ArtifactError: If either key is empty, the manifest cannot be read,
                or the bundle's current signature is absent. A signature that
                exists but does not verify raises the same error from
                `verify_signature`.
        """
        root = pathlib.Path(directory)
        try:
            old = signing.keying(current_key)
            replacement = signing.keying(new_key)
        except ValueError:
            # handled: invalid key becomes None, then ArtifactError raised just
            # below
            old = replacement = None
        if old is None or replacement is None:
            raise errors.ArtifactError(
                "signing-key rotation requires non-empty current and new keys"
            )
        manifest_path = root / layout.MANIFEST_FILE
        try:
            manifest_text = manifest_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise errors.ArtifactError(
                "cannot rotate an incomplete model bundle"
            ) from error
        if not seal.verify_signature(root, old):
            raise errors.ArtifactError(
                "cannot rotate an unauthenticated model bundle"
            )
        signature_path = root / seal.SIGNATURE_FILE
        fd, temporary_name = tempfile.mkstemp(
            dir=root, prefix=f".{seal.SIGNATURE_FILE}.", text=True
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    {
                        "algorithm": seal.SIGNATURE_ALGORITHM,
                        "signature": seal.sign(
                            manifest_text.encode(), replacement
                        ),
                    },
                    handle,
                )
                handle.flush()
                os.fsync(handle.fileno())
            pathlib.Path(temporary_name).replace(signature_path)
            durability.sync_directory(signature_path.parent)
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                pathlib.Path(temporary_name).unlink()
            raise
        return root

    @classmethod
    def load(
        cls,
        directory: str | pathlib.Path,
        trusted: bool = False,
        signing_key: str | bytes | None = None,
        allow_version_mismatch: bool = False,
        allow_legacy_format: bool = False,
    ) -> Self:
        """Loads and verifies a bundle.

        Bundles are pickle-based, and unpickling executes code, so loading is
        refused unless the bundle is authentic: either its HMAC signature
        verifies against `signing_key` / `$SCMLE_SIGNING_KEY`, or the caller
        vouches for its source with `trusted=True`. Checksums and signatures
        are verified *before* anything is unpickled. An invalid signature is
        always an error, even with `trusted=True`, and `trusted=True` is
        refused outright when `$SCMLE_ENV` is `production`.

        Args:
            directory: Bundle directory.
            trusted: Load an unsigned bundle you vouch for.
            signing_key: HMAC key used to verify the signature; defaults to
                `$SCMLE_SIGNING_KEY`.
            allow_version_mismatch: Load despite differing dependency
                versions.
            allow_legacy_format: Load a trusted format 1 bundle so it can be
                re-saved as the current authenticated format. This is refused
                unless `trusted=True`.

        Returns:
            The verified bundle.

        Raises:
            ArtifactError: If the bundle is unauthenticated, tampered with,
                incomplete, or incompatible.
        """
        envelope.vouch(trusted)
        root = pathlib.Path(directory)
        manifest, schema = unpack(root)
        info, legacy = admit(manifest, trusted, allow_legacy_format)
        authenticate(root, info, legacy, trusted, signing_key)
        problems = info.incompatibilities()
        if problems and not allow_version_mismatch:
            raise errors.ArtifactError(
                "bundle was built with different dependencies: "
                + "; ".join(problems),
                {
                    "mismatches": problems,
                    "environment_lock": info.environment_lock(),
                },
            )
        reference, history, calibrator, calibrated, detector = hydrate(
            root, info
        )
        intervals, cv_plus_models, stored_coverage, stored_weights = restore(
            root, info
        )
        loaded = cls(
            unpickle(root / layout.MODEL_FILE),
            schema,
            info,
            reference,
            history,
            calibrator,
            calibrated,
            detector,
            intervals,
        )
        loaded.cv_plus = cv_plus_models
        loaded.shift_weights = stored_weights
        loaded.coverage_report = stored_coverage
        return loaded
