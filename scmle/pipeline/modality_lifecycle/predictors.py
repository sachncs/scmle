"""Verified library inference for a signed modality package."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any
from typing import cast
from typing import Self
from typing import TYPE_CHECKING

from scmle.contracts import errors
from scmle.contracts import modalities
from scmle.governance import attestation
from scmle.pipeline.modality_lifecycle import containers
from scmle.pipeline.modality_lifecycle import curators
from scmle.pipeline.modality_lifecycle import reporting
from scmle.pipeline.modality_lifecycle import verification

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy
    from sklearn.linear_model import LogisticRegression

    # Predictor's fields name these two bare, as the public API snapshot
    # records them (tests/fixtures/public_api.json).
    from scmle.pipeline.modality_lifecycle.containers import Package
    from scmle.pipeline.modality_lifecycle.curators import Preparer


@dataclasses.dataclass(frozen=True, slots=True)
class Predictor:
    """Verified library inference for a signed modality lifecycle package.

    The caller supplies an adapter configured with the decoder and
    representation policy required for new samples. This class does not
    infer a codec, deserialize arbitrary raw transport payloads, or expose an
    HTTP server; it keeps those deployment decisions outside the package.

    Attributes:
        adapter: Adapter that turns new samples into features.
        model: Verified classifier loaded from the package.
        package: Identity of the package the model came from.
        names: Feature names the package was trained on.
        dataset: Hash of the training dataset recorded in the
            package.
        report: Digest of the verified lifecycle report, or
            None if no report was checked.
    """

    adapter: Preparer
    model: LogisticRegression = dataclasses.field(repr=False)
    package: Package
    names: tuple[str, ...]
    dataset: str
    report: str | None = None

    @staticmethod
    def identify(
        report: reporting.Record,
        adapter: curators.Preparer,
        package_path: Path,
        identity: tuple[str, str, str],
    ) -> None:
        """Checks that a verified lifecycle report names this exact package.

        Args:
            report: The verified lifecycle report.
            adapter: Adapter for new samples.
            package_path: Package directory.
            identity: Dataset hash, model digest and signature value that the
                package manifest records.

        Raises:
            ValueError: If the report's modality, dataset, package path, model
                digest or package signature differ.
        """
        dataset_hash, model_digest, signature_value = identity
        if report.modality != modalities.serialise(adapter.modality):
            raise errors.InputError("lifecycle report modality does not match")
        if report.digests["dataset"] != dataset_hash:
            raise errors.InputError(
                "lifecycle report dataset identity does not match"
            )
        if Path(report.package.path).resolve() != package_path.resolve():
            raise errors.InputError(
                "lifecycle report package path does not match"
            )
        if report.package.model_sha256 != model_digest:
            raise errors.InputError(
                "lifecycle report model digest does not match"
            )
        if report.package.signature != signature_value:
            raise errors.InputError(
                "lifecycle report package signature does not match"
            )

    @classmethod
    def load(
        cls,
        adapter: curators.Preparer,
        package_path: str | Path,
        signing_key: str | bytes,
        lifecycle_report_path: str | Path | None = None,
    ) -> Self:
        """Loads and validates a signed package for one modality adapter.

        When ``lifecycle_report_path`` is supplied, the signed report must
        identify this exact modality, dataset, package path, model digest,
        and package signature. This creates an auditable package-to-serving
        boundary without changing the experimental modality status.

        Args:
            adapter: Adapter for new samples; its modality must match the
                package.
            package_path: Package directory written by the lifecycle.
            signing_key: Key that the package and report signatures are checked
                with.
            lifecycle_report_path: Signed lifecycle report to cross-check with
                the package, if any.

        Returns:
            A predictor holding the verified model and its feature schema.

        Raises:
            ValueError: If the manifest, modality, feature names or signature
                are invalid or mismatch the adapter or the model, or the
                lifecycle report does not identify this package.
        """
        model = verification.load(package_path, signing_key)
        feature_names, identity = declared(Path(package_path), adapter, model)
        dataset_hash, model_digest, signature_value = identity
        report_digest: str | None = None
        if lifecycle_report_path is not None:
            report = reporting.Record.load(lifecycle_report_path, signing_key)
            cls.identify(report, adapter, Path(package_path), identity)
            report_digest = attestation.checksum(report.preimage())
        package = containers.Package(
            path=str(package_path),
            model_sha256=model_digest,
            signature=signature_value,
        )
        return cls(
            adapter=adapter,
            model=model,
            package=package,
            names=feature_names,
            dataset=dataset_hash,
            report=report_digest,
        )

    def featurize(self, samples: Sequence[Any]) -> containers.Batch:
        """Prepares and validates features for new modality samples.

        Args:
            samples: New samples to featurize.

        Returns:
            The feature batch for the samples, in order.

        Raises:
            ValueError: If there are no samples or the adapter's feature names
                differ from the package's.
        """
        values = tuple(samples)
        if not values:
            raise errors.InputError("samples must not be empty")
        batch = self.adapter.features(values)
        if batch.feature_names != self.names:
            raise errors.InputError(
                "new sample feature schema does not match package"
            )
        return batch

    def predict(self, samples: Sequence[Any]) -> numpy.ndarray:
        """Predicts labels after validating the signed feature contract.

        Args:
            samples: New samples to classify.

        Returns:
            Array of predicted labels, one per sample.
        """
        return cast(
            "numpy.ndarray",
            self.model.predict(self.featurize(samples).values),
        )

    def predict_proba(self, samples: Sequence[Any]) -> numpy.ndarray:
        """Predicts class probabilities after feature-contract validation.

        Args:
            samples: New samples to classify.

        Returns:
            Array with one row per sample and one column per class, in the
            model's class order.
        """
        probabilities: numpy.ndarray = self.model.predict_proba(
            self.featurize(samples).values
        )
        return probabilities

    def health(self) -> dict[str, object]:
        """Reports machine-readable package and inference readiness evidence.

        Returns:
            Mapping with the verification flag, modality, training dataset hash,
            feature names and model digest, plus the lifecycle report digest and
            flag if a report was checked.
        """
        health: dict[str, object] = {
            "package_verified": True,
            "modality": modalities.serialise(self.adapter.modality),
            "training_dataset_hash": self.dataset,
            "feature_names": list(self.names),
            "model_sha256": self.package.model_sha256,
        }
        if self.report is not None:
            health.update(
                {
                    "lifecycle_report_verified": True,
                    "lifecycle_report_digest": self.report,
                }
            )
        return health


def declared(
    package_path: Path, adapter: curators.Preparer, model: LogisticRegression
) -> tuple[tuple[str, ...], tuple[str, str, str]]:
    """Reads what a verified package manifest declares and checks it.

    Args:
        package_path: Package directory whose signature was verified.
        adapter: Adapter for new samples; its modality must match.
        model: The verified classifier loaded from the package.

    Returns:
        The feature names, and the dataset hash, model digest and signature
        value the manifest records.

    Raises:
        ValueError: If the manifest is not an object, its modality differs
            from the adapter's, its feature names are invalid or do not fit
            the model, or its dataset hash, model digest or signature is
            invalid.
    """
    manifest_path = package_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise errors.InputError("modality package manifest must be an object")
    if manifest.get("modality") != modalities.serialise(adapter.modality):
        raise errors.InputError("package modality does not match adapter")
    raw_features = manifest.get("feature_names")
    if not isinstance(raw_features, list) or not all(
        isinstance(name, str) and name.strip() for name in raw_features
    ):
        raise errors.InputError("package feature_names are invalid")
    feature_names = tuple(raw_features)
    if getattr(model, "n_features_in_", len(feature_names)) != len(
        feature_names
    ):
        raise errors.InputError("package model and feature schema do not match")
    dataset_hash = manifest.get("dataset_hash")
    model_digest = manifest.get("model_sha256")
    signature = manifest.get("signature")
    if not isinstance(dataset_hash, str) or not dataset_hash.strip():
        raise errors.InputError("package dataset_hash is invalid")
    if not isinstance(model_digest, str) or not model_digest.strip():
        raise errors.InputError("package model_sha256 is invalid")
    value = signature.get("value") if isinstance(signature, dict) else None
    if not isinstance(value, str):
        raise errors.InputError("package signature is invalid")
    return feature_names, (dataset_hash, model_digest, value)
