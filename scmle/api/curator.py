"""Curation-only runs: signed evidence about a modality, with no model."""

from collections.abc import Mapping
import contextlib
import json
import os
import pathlib
import tempfile
from typing import Any

from scmle import durability
from scmle.contracts import errors
from scmle.contracts import modalities
from scmle.contracts import signing
import scmle.contracts.data
from scmle.data.modalities import curation
from scmle.data.modalities import document
from scmle.pipeline import modality_lifecycle


def emplace(path: pathlib.Path, content: bytes) -> None:
    """Publishes curation bytes atomically without replacing an artifact.

    A hard link from a fully flushed sibling file gives the destination a
    no-clobber, atomic first publication.  ``FileExistsError`` is deliberately
    preserved for the caller so a concurrent publisher can verify and reuse its
    result instead of replacing it.

    Args:
        path: Destination of the artifact; its directory is created.
        content: The bytes to publish.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, path)
        durability.sync_directory(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            pathlib.Path(temporary_name).unlink()


def conform(
    manifest: scmle.contracts.data.Passport,
    modality: str,
    dataset_hash: str,
    sample_count: int,
) -> None:
    """Checks that a modality manifest describes the adapter's dataset.

    Args:
        manifest: The manifest to check.
        modality: The adapter's modality name.
        dataset_hash: The adapter's dataset hash.
        sample_count: The adapter's number of samples.

    Raises:
        ValueError: If the manifest's modality, content hash or sample count
            differs from the adapter's.
    """
    if manifest.modality != modality:
        raise errors.InputError("manifest modality must match adapter")
    if manifest.digests["content"] != dataset_hash:
        raise errors.InputError("manifest content_hash must match adapter")
    if manifest.samples != sample_count:
        raise errors.InputError("manifest sample_count must match adapter")


def verify(
    path: pathlib.Path,
    signing_key: str | bytes,
    candidate: modality_lifecycle.Curation,
) -> modality_lifecycle.Curation:
    """Verifies an existing curation artifact against a new candidate.

    Args:
        path: Location of the existing artifact.
        signing_key: Key that authenticates the evidence.
        candidate: The curation the current adapter produced.

    Returns:
        The verified existing curation.

    Raises:
        ValueError: If the artifact is unreadable, is not an object, or
            differs from the candidate in modality, dataset hash, sample count
            or canonical content.
    """
    existing = modality_lifecycle.Curation.read_verified(path, signing_key)
    if existing.modality != candidate.modality:
        raise errors.InputError(
            "existing curation modality does not match current adapter"
        ) from None
    if existing.digests != candidate.digests:
        raise errors.InputError(
            "existing curation dataset_hash does not match current adapter"
        ) from None
    if existing.samples != candidate.samples:
        raise errors.InputError(
            "existing curation sample_count does not match current adapter"
        ) from None
    stored = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.loads(json.dumps(existing.to_dict()))
    if stored != canonical or existing.digest != candidate.digest:
        raise errors.InputError(
            "existing curation canonical content does not match current adapter"
        ) from None
    return existing


class Curator:
    """Curates and documents a modality without forcing a model fit.

    This is the data-understanding entry point for unlabeled, unsupervised, or
    domain-specific datasets. It persists signed curation and dataset-card
    evidence, but it never invents labels, features, metrics, or a model.

    Attributes:
        adapter: Adapter that owns source identity and curation policy.
        signing_key: Key that authenticates the evidence.
        output_dir: Directory where curation evidence is written.
        manifest: Source, lineage, privacy and task contract, if any.
    """

    def __init__(
        self,
        adapter: modality_lifecycle.Curator,
        signing_key: str | bytes,
        *,
        output_dir: str | pathlib.Path = "runs",
        manifest: scmle.contracts.data.Passport | None = None,
    ) -> None:
        """Defines a curation-only run over a typed modality adapter.

        Args:
            adapter: Adapter that owns source identity and curation policy.
            signing_key: Non-empty key used to authenticate the evidence.
            output_dir: Directory where curation evidence is written.
            manifest: Optional source, lineage, privacy, and task contract.

        Raises:
            TypeError: If the adapter or manifest has the wrong type.
            ValueError: If the signing key is empty.
        """
        if not isinstance(adapter, modality_lifecycle.Curator):
            raise errors.KindError("adapter must be a Curator")
        if not signing_key:
            raise errors.InputError("signing_key must not be empty")
        if manifest is not None and not isinstance(
            manifest, scmle.contracts.data.Passport
        ):
            raise errors.KindError("manifest must be Passport")
        self.adapter = adapter
        self.signing_key = signing_key
        self.output_dir = pathlib.Path(output_dir)
        self.manifest = manifest

    def examine(
        self, modality: str, dataset_hash: str, sample_count: int
    ) -> Mapping[str, Any]:
        """Curates the adapter's dataset and checks the report describes it.

        Args:
            modality: The adapter's serialized modality.
            dataset_hash: The adapter's dataset hash.
            sample_count: The adapter's number of samples.

        Returns:
            The curation report as a JSON-native mapping.

        Raises:
            ValueError: If the manifest or the curation report disagrees with
                the adapter.
            TypeError: If the adapter's `curate()` does not return a modality
                curation report.
        """
        if self.manifest is not None:
            conform(self.manifest, modality, dataset_hash, sample_count)
        report = self.adapter.curate()
        if not isinstance(report, (curation.Curation, document.Curation)):
            raise errors.KindError(
                "adapter curate() must return a modality curation report"
            )
        curation_payload = report.to_dict()
        digest, samples = curation.identify(curation_payload)
        if digest != dataset_hash:
            raise errors.InputError("curation dataset_hash must match adapter")
        if samples != sample_count:
            raise errors.InputError("curation sample_count must match adapter")
        return curation_payload

    def run(self) -> modality_lifecycle.Curation:
        """Persists signed curation evidence without entering modelling.

        If an artifact already exists it is verified and reused when it matches
        the current adapter, instead of being replaced. An existing artifact
        whose signature does not verify raises from `read_verified`.

        Returns:
            The signed curation result, either newly written or the matching
            existing one.

        Raises:
            ValueError: If `output_dir` exists and is not a directory, the
                manifest or the curation report disagrees with the adapter, or
                an existing artifact is unreadable, is not an object, or differs
                from the current adapter's modality, dataset hash, sample count
                or content.
            TypeError: If the adapter's `curate()` does not return a modality
                curation report.
        """
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise errors.InputError("output_dir must be a directory")
        sample_count = len(self.adapter.samples)
        modality = modalities.serialise(self.adapter.modality)
        dataset_hash = self.adapter.digest
        curation_payload = self.examine(modality, dataset_hash, sample_count)
        card = curation.Datasheet.from_report(
            modality,
            curation_payload,
            self.adapter.remarks(),
            manifest=self.manifest,
            task="not_declared",
        )
        unsigned: dict[str, object] = {
            "schema_version": modality_lifecycle.CURATION_SCHEMA_VERSION,
            "record_type": "modality_curation",
            "modality": modality,
            "digests": {"dataset": dataset_hash},
            "samples": sample_count,
            "curation": curation_payload,
            "card": card.to_dict(),
        }
        signature = signing.mac(unsigned, self.signing_key)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "modality_curation.json"
        candidate = modality_lifecycle.Curation(
            modality=modality,
            digests={"dataset": dataset_hash},
            samples=sample_count,
            curation=curation_payload,
            card=card.to_dict(),
            path=str(path),
            signature=signature,
        )
        payload = candidate.to_dict()
        serialized_payload = (
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        try:
            emplace(path, serialized_payload)
        except FileExistsError:
            # handled: existing artifact verified and returned (idempotent
            # publish)
            return verify(path, self.signing_key, candidate)
        return candidate
