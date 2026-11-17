"""Verified packages of a fitted image classifier."""

import hashlib
import io
import json
import os
import pathlib
import shutil
import tempfile
from typing import Self

import joblib

from scmle import durability
from scmle import envelope
from scmle.contracts import errors
from scmle.models.images.estimator import Estimator
from scmle.models.images.records import Package


class Archive(Estimator):
    """Saves and loads the fitted estimator as a verified package."""

    def save(
        self,
        path: str | pathlib.Path,
        signing_key: str | bytes | None = None,
    ) -> Package:
        """Atomically saves this fitted classifier as a verified package.

        Args:
            path: Directory to create; it must not exist yet.
            signing_key: HMAC key for the manifest, which records the
                payload digest; the configured key is used when None. An
                unsigned package loads only with `trusted=True` (SEC-2).

        Returns:
            The package path and the digest of its model payload.

        Raises:
            FileExistsError: If `path` already exists.
        """
        self.preflight()
        target = pathlib.Path(path).expanduser()
        if target.exists():
            raise FileExistsError(f"model package already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_directory: pathlib.Path | None = None
        try:
            temporary_directory = pathlib.Path(
                tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}.")
            )
            model_path = temporary_directory / "model.joblib"
            joblib.dump(self, model_path, compress=3)
            durability.sync(model_path)
            manifest = {
                "schema_version": 1,
                "model_path": "model.joblib",
                "artifact_digest": file_digest(model_path),
                "model_digest": self.model_digest,
                "config": self.config.to_dict(),
            }
            manifest = dict(envelope.sign(manifest, signing_key))
            manifest_path = temporary_directory / "manifest.json"
            with manifest_path.open(
                "w", encoding="utf-8", newline="\n"
            ) as stream:
                json.dump(
                    manifest,
                    stream,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            durability.sync_directory(temporary_directory)
            pathlib.Path(temporary_directory).replace(target)
            durability.sync_directory(target.parent)
        except Exception:
            if temporary_directory is not None:
                shutil.rmtree(temporary_directory, ignore_errors=True)
            raise
        return Package(str(target), str(manifest["artifact_digest"]))

    @classmethod
    def load(
        cls,
        path: str | pathlib.Path,
        trusted: bool = False,
        signing_key: str | bytes | None = None,
    ) -> Self:
        """Loads a package, authenticating it before anything is unpickled.

        Args:
            path: Package directory written by `save`.
            trusted: Whether the caller vouches for an unsigned package;
                a signed package that verifies needs no vouching.
            signing_key: HMAC key; the configured key is used when None.

        Returns:
            The fitted classifier stored in the package.

        Raises:
            ArtifactError: If the manifest's signature does not verify, or
                it is unsigned (or no key is set) and `trusted` is false,
                or `trusted` is true and `$SCMLE_ENV` is `production`.
            FileNotFoundError: If `path` is not a directory.
            ValueError: If the manifest or payload is missing, the version
                is unsupported, or a digest does not match.
            TypeError: If the payload is not an `Classifier`.
        """
        target = pathlib.Path(path).expanduser()
        if not target.is_dir():
            raise FileNotFoundError(target)
        manifest_path = target / "manifest.json"
        if not manifest_path.is_file():
            raise errors.InputError("image model package has no manifest")
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("schema_version") != 1:
            raise errors.InputError("unsupported image model package version")
        model_path = target / str(manifest.get("model_path", ""))
        if not model_path.is_file():
            raise errors.InputError("image model package has no model payload")
        envelope.check(manifest, target, signing_key, trusted)
        payload = envelope.fetch(model_path, manifest.get("artifact_digest"))
        loaded = joblib.load(io.BytesIO(payload))
        if not isinstance(loaded, cls):
            raise errors.KindError("package does not contain an Classifier")
        loaded.preflight()
        if loaded.model_digest != manifest.get("model_digest"):
            raise errors.InputError(
                "image model digest does not match manifest"
            )
        return loaded


def file_digest(path: pathlib.Path) -> str:
    """Computes a SHA-256 digest for a package artifact.

    Args:
        path: File to hash.

    Returns:
        The hexadecimal digest of the file contents.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
