"""Complete, bounded baseline lifecycles for non-tabular datasets.

The lifecycle is deliberately explicit about what a modality baseline sees.
Each adapter owns the modality-specific validation, split, and representation
policy; the stage runner owns the shared evidence, model, packaging,
inference, monitoring, documentation, and recovery contracts. The baseline
model is a regularized linear classifier. It is useful for proving lifecycle
integrity, not for claiming state-of-the-art representation quality.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from datetime import datetime
import hashlib
import hmac
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import tempfile
import threading
import time
from typing import Any
from typing import cast
from typing import ClassVar
from typing import Literal
from typing import Self
import wave

import joblib
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

from scmle import durability
from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import events
from scmle.contracts import modalities
from scmle.contracts import performance
from scmle.core import limits
from scmle.data.modalities import audio
from scmle.data.modalities import curation
from scmle.data.modalities import document
from scmle.data.modalities import geospatial
from scmle.data.modalities import graph
from scmle.data.modalities import image
from scmle.data.modalities import video
from scmle.explainability import quality
from scmle.governance import attestation
from scmle.models import documents
from scmle.models import images

MAX_REPEATED_EXPLANATION_SAMPLES = 128
MAX_RAW_GRAPH_FEATURES = 64
MAX_GRAPH_CONTEXT_NODES = 64
MAX_GRAPH_CONTEXT_EDGES = 128
PACKAGE_SCHEMA_VERSION = 1
CURATION_SCHEMA_VERSION = 1
SHA256_LENGTH = 64


class LifecycleCancelledError(errors.CancelledError):
    """Raised when a modality lifecycle receives a cancellation request."""

    def __init__(self, message: str) -> None:
        """Creates a cancellation error with an actionable message.

        Args:
            message: Explanation of why and where the run was cancelled.
        """
        super().__init__(message, {"scope": "modality_lifecycle"})


class LifecycleCancellation:
    """Thread-safe cooperative cancellation signal for one lifecycle run."""

    def __init__(self) -> None:
        """Creates a cancellation signal in the not-cancelled state."""
        self.signal = threading.Event()

    def cancel(self) -> None:
        """Requests cancellation at the next lifecycle stage boundary."""
        self.signal.set()

    def is_cancelled(self) -> bool:
        """Reports whether cancellation has been requested.

        Returns:
            True once `cancel` has been called, otherwise False.
        """
        return self.signal.is_set()


@dataclasses.dataclass(frozen=True, slots=True)
class FeatureBatch:
    """A finite, bounded feature matrix tied to stable sample identities.

    Attributes:
        sample_ids: Unique identifier of each row, in row order.
        feature_names: Non-empty name of each column, in column order.
        values: Read-only float matrix with one row per sample ID and one column
            per feature name; every value is finite.
    """

    sample_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: numpy.ndarray = dataclasses.field(repr=False)

    def __post_init__(self) -> None:
        """Validates shape, finiteness, and immutable identity metadata.

        Raises:
            ValueError: If the values are not a two-dimensional matrix, the row
                or column counts do not match the IDs or names, a value is not
                finite, sample IDs repeat, or a feature name is blank.
        """
        matrix = numpy.asarray(self.values, dtype=numpy.float64)
        if matrix.ndim != 2:
            raise ValueError("feature values must be a two-dimensional matrix")
        if matrix.shape[0] != len(self.sample_ids):
            raise ValueError("feature rows must match sample IDs")
        if matrix.shape[1] != len(self.feature_names):
            raise ValueError("feature columns must match feature names")
        if not numpy.isfinite(matrix).all():
            raise ValueError("feature values must be finite")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample IDs must be unique within a feature batch")
        if any(not name.strip() for name in self.feature_names):
            raise ValueError("feature names must not be empty")
        frozen = matrix.copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "sample_ids", tuple(self.sample_ids))
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "values", frozen)


@dataclasses.dataclass(frozen=True, slots=True)
class DatasetPartitions:
    """Leakage-aware train, validation, and test partitions.

    Attributes:
        train: Samples used to fit the model.
        validation: Samples held out for validation.
        test: Samples held out for the final evaluation.
        groups: Mapping of partition name to the sorted group identifiers it
            contains.
    """

    train: tuple[Any, ...]
    validation: tuple[Any, ...]
    test: tuple[Any, ...]
    groups: dict[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        """Normalizes partitions to tuples and sorts each group list."""
        object.__setattr__(self, "train", tuple(self.train))
        object.__setattr__(self, "validation", tuple(self.validation))
        object.__setattr__(self, "test", tuple(self.test))
        normalized = {
            name: tuple(sorted(values)) for name, values in self.groups.items()
        }
        object.__setattr__(self, "groups", normalized)


@dataclasses.dataclass(frozen=True, slots=True)
class LifecyclePackage:
    """Digest and signature identity of an atomically written package.

    Attributes:
        path: Package directory.
        model_sha256: SHA-256 digest of the serialized model.
        signature: Signature value of the package manifest.
        schema_version: Version of the package layout.
    """

    path: str
    model_sha256: str
    signature: str
    schema_version: int = PACKAGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validates package identity before it enters lifecycle evidence.

        Raises:
            ValueError: If the path or signature is blank, the model digest is
                not a SHA-256 digest, or the schema version is unsupported.
        """
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("package path must be non-empty")
        if not isinstance(self.model_sha256, str) or not is_sha256(
            self.model_sha256
        ):
            raise ValueError("package model_sha256 must be a SHA-256 digest")
        if not isinstance(self.signature, str) or not self.signature.strip():
            raise ValueError("package signature must be non-empty")
        if self.schema_version != PACKAGE_SCHEMA_VERSION:
            raise ValueError("unsupported modality package schema version")

    def to_dict(self) -> dict[str, object]:
        """Converts the package identity to JSON-native values.

        Returns:
            Mapping of field name to value.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class ModalityPredictor:
    """Verified library inference for a signed modality lifecycle package.

    The caller supplies an adapter configured with the decoder and
    representation policy required for new samples. This class does not
    infer a codec, deserialize arbitrary raw transport payloads, or expose an
    HTTP server; it keeps those deployment decisions outside the package.

    Attributes:
        adapter: Adapter that turns new samples into features.
        model: Verified classifier loaded from the package.
        package: Identity of the package the model came from.
        feature_names: Feature names the package was trained on.
        training_dataset_hash: Hash of the training dataset recorded in the
            package.
        lifecycle_report_digest: Digest of the verified lifecycle report, or
            None if no report was checked.
    """

    adapter: ModalityAdapter
    model: LogisticRegression = dataclasses.field(repr=False)
    package: LifecyclePackage
    feature_names: tuple[str, ...]
    training_dataset_hash: str
    lifecycle_report_digest: str | None = None

    @classmethod
    def from_package(
        cls,
        adapter: ModalityAdapter,
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
        model = load_package(package_path, signing_key)
        manifest_path = Path(package_path) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("modality package manifest must be an object")
        if manifest.get("modality") != modalities.modality_value(
            adapter.modality
        ):
            raise ValueError("package modality does not match adapter")
        raw_features = manifest.get("feature_names")
        if not isinstance(raw_features, list) or not all(
            isinstance(name, str) and name.strip() for name in raw_features
        ):
            raise ValueError("package feature_names are invalid")
        feature_names = tuple(raw_features)
        if getattr(model, "n_features_in_", len(feature_names)) != len(
            feature_names
        ):
            raise ValueError("package model and feature schema do not match")
        dataset_hash = manifest.get("dataset_hash")
        model_digest = manifest.get("model_sha256")
        signature = manifest.get("signature")
        if not isinstance(dataset_hash, str) or not dataset_hash.strip():
            raise ValueError("package dataset_hash is invalid")
        if not isinstance(model_digest, str) or not model_digest.strip():
            raise ValueError("package model_sha256 is invalid")
        if not isinstance(signature, dict) or not isinstance(
            signature.get("value"), str
        ):
            raise ValueError("package signature is invalid")
        report_digest: str | None = None
        if lifecycle_report_path is not None:
            report = LifecycleReport.read_verified(
                lifecycle_report_path, signing_key
            )
            if report.modality != modalities.modality_value(adapter.modality):
                raise ValueError("lifecycle report modality does not match")
            if report.dataset_hash != dataset_hash:
                raise ValueError(
                    "lifecycle report dataset identity does not match"
                )
            if (
                Path(report.package.path).resolve()
                != Path(package_path).resolve()
            ):
                raise ValueError("lifecycle report package path does not match")
            if report.package.model_sha256 != model_digest:
                raise ValueError("lifecycle report model digest does not match")
            if report.package.signature != signature["value"]:
                raise ValueError(
                    "lifecycle report package signature does not match"
                )
            report_digest = attestation.canonical_digest(report.unsigned_dict())
        package = LifecyclePackage(
            path=str(package_path),
            model_sha256=model_digest,
            signature=signature["value"],
        )
        return cls(
            adapter=adapter,
            model=model,
            package=package,
            feature_names=feature_names,
            training_dataset_hash=dataset_hash,
            lifecycle_report_digest=report_digest,
        )

    def features_for(self, samples: Sequence[Any]) -> FeatureBatch:
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
            raise ValueError("samples must not be empty")
        batch = self.adapter.features(values)
        if batch.feature_names != self.feature_names:
            raise ValueError("new sample feature schema does not match package")
        return batch

    def predict(self, samples: Sequence[Any]) -> numpy.ndarray:
        """Predicts labels after validating the signed feature contract.

        Args:
            samples: New samples to classify.

        Returns:
            Array of predicted labels, one per sample.
        """
        return self.model.predict(self.features_for(samples).values)

    def predict_proba(self, samples: Sequence[Any]) -> numpy.ndarray:
        """Predicts class probabilities after feature-contract validation.

        Args:
            samples: New samples to classify.

        Returns:
            Array with one row per sample and one column per class, in the
            model's class order.
        """
        return self.model.predict_proba(self.features_for(samples).values)

    def health(self) -> dict[str, object]:
        """Reports machine-readable package and inference readiness evidence.

        Returns:
            Mapping with the verification flag, modality, training dataset hash,
            feature names and model digest, plus the lifecycle report digest and
            flag if a report was checked.
        """
        health = {
            "package_verified": True,
            "modality": modalities.modality_value(self.adapter.modality),
            "training_dataset_hash": self.training_dataset_hash,
            "feature_names": list(self.feature_names),
            "model_sha256": self.package.model_sha256,
        }
        if self.lifecycle_report_digest is not None:
            health.update(
                {
                    "lifecycle_report_verified": True,
                    "lifecycle_report_digest": self.lifecycle_report_digest,
                }
            )
        return health


@dataclasses.dataclass(frozen=True, slots=True)
class LifecycleReport:
    """All evidence emitted by one completed modality lifecycle.

    Attributes:
        modality: Modality value the lifecycle ran for.
        dataset_hash: Hash identifying the dataset.
        stage_evidence: Evidence of every required lifecycle stage, in order.
        evaluation: Held-out evaluation evidence.
        performance: Resource and timing evidence.
        package: Identity of the written model package.
        inference: Inference smoke-test evidence.
        monitoring: Drift-monitoring evidence.
        documentation: Documentation evidence.
        recovery: Recovery and resource evidence.
        signature: SHA-256 signature of the report content, or None if unsigned.
    """

    modality: str
    dataset_hash: str
    stage_evidence: tuple[modalities.StageEvidence, ...]
    evaluation: dict[str, object]
    performance: dict[str, object]
    package: LifecyclePackage
    inference: dict[str, object]
    monitoring: dict[str, object]
    documentation: dict[str, object]
    recovery: dict[str, object]
    signature: str | None = None

    def __post_init__(self) -> None:
        """Validates the report identity and optional signature.

        The six evidence mappings (`evaluation`, `performance`, `inference`,
        `monitoring`, `documentation`, `recovery`) are also normalized to
        sorted JSON-native dictionaries; a value that is not a mapping raises
        TypeError and one that is not finite JSON raises ValueError, both from
        `normalize_json_object`.

        Raises:
            ValueError: If the modality or dataset hash is blank, the stage
                evidence is not every required stage once in order or lacks
                evidence links, or the signature is not a SHA-256 digest.
            TypeError: If the package is not a `LifecyclePackage` or the stage
                evidence holds other objects.
        """
        for field_name in ("modality", "dataset_hash"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        if not isinstance(self.package, LifecyclePackage):
            raise TypeError("package must be LifecyclePackage")
        stages = tuple(self.stage_evidence)
        if any(
            not isinstance(evidence, modalities.StageEvidence)
            for evidence in stages
        ):
            raise TypeError("stage_evidence must contain StageEvidence records")
        expected_stages = modalities.REQUIRED_LIFECYCLE_STAGES
        actual_stages = tuple(evidence.stage for evidence in stages)
        if actual_stages != expected_stages:
            raise ValueError(
                "lifecycle report stage evidence must contain every required "
                "stage exactly once in order"
            )
        if any(not evidence.evidenced for evidence in stages):
            raise ValueError(
                "lifecycle report stages must contain evidence links"
            )
        object.__setattr__(self, "stage_evidence", stages)
        for field_name in (
            "evaluation",
            "performance",
            "inference",
            "monitoring",
            "documentation",
            "recovery",
        ):
            object.__setattr__(
                self,
                field_name,
                normalize_json_object(getattr(self, field_name), field_name),
            )
        if self.signature is not None and not is_sha256(self.signature):
            raise ValueError("signature must be a SHA-256 digest")

    def unsigned_dict(self) -> dict[str, object]:
        """Builds the exact report content covered by the signature.

        Returns:
            JSON-native mapping of every field except the signature.
        """
        return {
            "schema_version": 2,
            "modality": self.modality,
            "dataset_hash": self.dataset_hash,
            "stage_evidence": [
                evidence.to_dict() for evidence in self.stage_evidence
            ],
            "evaluation": self.evaluation,
            "performance": self.performance,
            "package": self.package.to_dict(),
            "inference": self.inference,
            "monitoring": self.monitoring,
            "documentation": self.documentation,
            "recovery": self.recovery,
        }

    def sign(self, signing_key: str | bytes) -> Self:
        """Signs the report over its exact serialized evidence.

        Args:
            signing_key: Key to sign with.

        Returns:
            A copy of this report carrying the signature.
        """
        return dataclasses.replace(
            self,
            signature=attestation.sign_payload(
                self.unsigned_dict(), signing_key
            ),
        )

    def verify_signature(self, signing_key: str | bytes) -> bool:
        """Checks the report signature against its current content.

        Args:
            signing_key: Key the signature should have been made with.

        Returns:
            True if the report is signed and the signature matches; False if it
            is unsigned or does not match.
        """
        if self.signature is None:
            return False
        expected = attestation.sign_payload(self.unsigned_dict(), signing_key)
        return hmac.compare_digest(self.signature, expected)

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> Self:
        """Restores a versioned report without trusting its evidence claims.

        Args:
            record: Mapping produced by `to_dict`.

        Returns:
            The restored report.

        Raises:
            ValueError: If the schema version is unsupported or any field is
                missing or has the wrong shape.
        """
        if record.get("schema_version") != 2:
            raise ValueError("unsupported lifecycle report schema version")

        raw_stage_evidence = record.get("stage_evidence")
        if not isinstance(raw_stage_evidence, list):
            raise ValueError("lifecycle report stage_evidence must be a list")
        stage_evidence: list[modalities.StageEvidence] = []
        for item in raw_stage_evidence:
            if not isinstance(item, Mapping):
                raise ValueError("lifecycle report stage evidence is invalid")
            stage = item.get("stage")
            evidence_links = item.get("evidence_links")
            evidence = item.get("evidence")
            notes = item.get("notes")
            if not isinstance(stage, str):
                raise ValueError("lifecycle report stage is invalid")
            if not isinstance(evidence_links, list) or not all(
                isinstance(value, str) for value in evidence_links
            ):
                raise ValueError("lifecycle report evidence_links are invalid")
            if not isinstance(evidence, Mapping):
                raise ValueError("lifecycle report evidence is invalid")
            if not isinstance(notes, list) or not all(
                isinstance(value, str) for value in notes
            ):
                raise ValueError("lifecycle report notes are invalid")
            stage_evidence.append(
                modalities.StageEvidence(
                    stage=modalities.LifecycleStage(stage),
                    evidence_links=tuple(evidence_links),
                    evidence=evidence,
                    notes=tuple(notes),
                )
            )

        raw_package = record.get("package")
        if not isinstance(raw_package, Mapping):
            raise ValueError("lifecycle report package is invalid")
        package_values = {
            "path": raw_package.get("path"),
            "model_sha256": raw_package.get("model_sha256"),
            "signature": raw_package.get("signature"),
            "schema_version": raw_package.get("schema_version"),
        }
        if not all(
            isinstance(package_values[field_name], str)
            for field_name in ("path", "model_sha256", "signature")
        ):
            raise ValueError("lifecycle report package identity is invalid")
        if not isinstance(package_values["schema_version"], int):
            raise ValueError("lifecycle report package schema is invalid")

        def mapping_field(name: str) -> dict[str, object]:
            value = record.get(name)
            if not isinstance(value, Mapping):
                raise ValueError(f"lifecycle report {name} must be an object")
            return dict(value)

        raw_signature = record.get("signature")
        signature: str | None
        if raw_signature is None:
            signature = None
        elif isinstance(raw_signature, Mapping):
            if (
                raw_signature.get("algorithm")
                != attestation.SIGNATURE_ALGORITHM
            ):
                raise ValueError("unsupported lifecycle report signature")
            signature_value = raw_signature.get("value")
            if (
                not isinstance(signature_value, str)
                or not signature_value.strip()
            ):
                raise ValueError("lifecycle report signature is invalid")
            signature = signature_value
        else:
            raise ValueError("lifecycle report signature is invalid")

        modality = record.get("modality")
        dataset_hash = record.get("dataset_hash")
        if not isinstance(modality, str) or not isinstance(dataset_hash, str):
            raise ValueError("lifecycle report identity is invalid")
        return cls(
            modality=modality,
            dataset_hash=dataset_hash,
            stage_evidence=tuple(stage_evidence),
            evaluation=mapping_field("evaluation"),
            performance=mapping_field("performance"),
            package=LifecyclePackage(
                path=cast(str, package_values["path"]),
                model_sha256=cast(str, package_values["model_sha256"]),
                signature=cast(str, package_values["signature"]),
                schema_version=cast(int, package_values["schema_version"]),
            ),
            inference=mapping_field("inference"),
            monitoring=mapping_field("monitoring"),
            documentation=mapping_field("documentation"),
            recovery=mapping_field("recovery"),
            signature=signature,
        )

    @classmethod
    def read(cls, path: str | Path) -> Self:
        """Reads and parses a persisted lifecycle report.

        Args:
            path: JSON file written from `to_dict`.

        Returns:
            The parsed report (signature not yet verified).

        Raises:
            ValueError: If the file does not hold a JSON object.
        """
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(record, Mapping):
            raise ValueError("lifecycle report must be a JSON object")
        return cls.from_dict(record)

    @classmethod
    def read_verified(cls, path: str | Path, signing_key: str | bytes) -> Self:
        """Reads a report and rejects it unless its content signature verifies.

        Args:
            path: JSON file written from `to_dict`.
            signing_key: Key the signature should have been made with.

        Returns:
            The parsed report with a verified signature.

        Raises:
            ValueError: If the signature is missing or does not verify.
        """
        report = cls.read(path)
        if not report.verify_signature(signing_key):
            raise ValueError("lifecycle report signature does not verify")
        return report

    def to_dict(self) -> dict[str, object]:
        """Converts the report to a deterministic JSON-compatible record.

        Returns:
            The unsigned content plus a `signature` entry that is None if
            unsigned, otherwise the algorithm and value.
        """
        record = self.unsigned_dict()
        record["signature"] = (
            None
            if self.signature is None
            else {
                "algorithm": attestation.SIGNATURE_ALGORITHM,
                "value": self.signature,
            }
        )
        return record


@dataclasses.dataclass(frozen=True, slots=True)
class LifecycleCheckpoint:
    """Signed durable boundary for one modality lifecycle replay.

    A checkpoint records only JSON-native stage identity and evidence digests.
    Resume reconstructs the in-memory context by replaying completed stages
    against the caller-provided adapter, then verifies that every replayed
    stage is byte-for-byte equivalent to the checkpoint. This avoids trusting
    a serialized estimator or arbitrary adapter object as recovery state.

    Attributes:
        modality: Modality value of the run.
        dataset_hash: Hash identifying the training dataset.
        observed_dataset_hash: Hash identifying the dataset used for
            monitoring.
        package_path: Where the model package is written.
        report_path: Where the lifecycle report is written.
        seed: Random seed of the run.
        completed_stages: Values of the lifecycle stages finished so far, a
            prefix of the required order.
        stage_digests: Digest of each completed stage's evidence, in the same
            order.
        require_independent_ground_truth: Whether the run demands independently
            authored explanation ground truth.
        manifest_digest: Digest of the package manifest once it exists, else
            None.
        status: "running" until every stage is done, then "completed".
        signature: Signature value over the content, or None if unsigned.
        schema_version: Version of the checkpoint layout.
    """

    modality: str
    dataset_hash: str
    observed_dataset_hash: str
    package_path: str
    report_path: str
    seed: int
    completed_stages: tuple[str, ...]
    stage_digests: tuple[str, ...]
    require_independent_ground_truth: bool
    manifest_digest: str | None = None
    status: Literal["running", "completed"] = "running"
    signature: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        """Rejects incomplete, non-prefix, or unsigned-identity state.

        Raises:
            ValueError: If an identity field is blank, the status is unknown, a
                digest or signature is blank, the completed stages are not an
                ordered prefix of the lifecycle, the digests do not match the
                stages, or a completed checkpoint lacks a stage.
            TypeError: If the seed is not an integer or the ground-truth
                requirement is not a boolean.
        """
        for field_name in (
            "modality",
            "dataset_hash",
            "observed_dataset_hash",
            "package_path",
            "report_path",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not isinstance(self.require_independent_ground_truth, bool):
            raise TypeError("require_independent_ground_truth must be bool")
        if self.status not in ("running", "completed"):
            raise ValueError("checkpoint status must be running or completed")
        if self.manifest_digest is not None and not (
            self.manifest_digest.strip()
        ):
            raise ValueError("manifest_digest must be non-empty when provided")
        if self.signature is not None and not self.signature.strip():
            raise ValueError("signature must be non-empty when provided")
        expected_stages = tuple(
            stage.value for stage in modalities.REQUIRED_LIFECYCLE_STAGES
        )[: len(self.completed_stages)]
        if tuple(self.completed_stages) != expected_stages:
            raise ValueError(
                "completed stages must be an ordered lifecycle prefix"
            )
        if len(self.stage_digests) != len(self.completed_stages):
            raise ValueError("stage digests must match completed stages")
        if any(not digest.strip() for digest in self.stage_digests):
            raise ValueError("stage digests must not be empty")
        if self.status == "completed" and len(self.completed_stages) != len(
            modalities.REQUIRED_LIFECYCLE_STAGES
        ):
            raise ValueError(
                "completed checkpoint must contain every lifecycle stage"
            )

    def unsigned_dict(self) -> dict[str, object]:
        """Builds the exact checkpoint content covered by its signature.

        Returns:
            JSON-native mapping of every field except the signature, plus a
            fixed `record_type` entry.
        """
        return {
            "schema_version": self.schema_version,
            "record_type": "modality_lifecycle_checkpoint",
            "modality": self.modality,
            "dataset_hash": self.dataset_hash,
            "observed_dataset_hash": self.observed_dataset_hash,
            "package_path": self.package_path,
            "report_path": self.report_path,
            "seed": self.seed,
            "completed_stages": list(self.completed_stages),
            "stage_digests": list(self.stage_digests),
            "require_independent_ground_truth": (
                self.require_independent_ground_truth
            ),
            "manifest_digest": self.manifest_digest,
            "status": self.status,
        }

    def sign(self, signing_key: str | bytes) -> Self:
        """Signs the checkpoint over its exact serialized identity.

        Args:
            signing_key: Key to sign with.

        Returns:
            A copy of this checkpoint carrying the signature.
        """
        return dataclasses.replace(
            self,
            signature=attestation.sign_payload(
                self.unsigned_dict(), signing_key
            ),
        )

    def verify_signature(self, signing_key: str | bytes) -> bool:
        """Checks the persisted checkpoint signature.

        Args:
            signing_key: Key the signature should have been made with.

        Returns:
            True if the checkpoint is signed and the signature matches; False if
            it is unsigned or does not match.
        """
        if self.signature is None:
            return False
        expected = attestation.sign_payload(self.unsigned_dict(), signing_key)
        return hmac.compare_digest(self.signature, expected)

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> Self:
        """Restores a checkpoint without trusting its serialized claims.

        Args:
            record: Mapping produced by `to_dict`.

        Returns:
            The restored checkpoint.

        Raises:
            ValueError: If the schema version, record type, signature or any
                field is unsupported, missing or malformed.
        """
        if record.get("schema_version") != 1:
            raise ValueError("unsupported lifecycle checkpoint schema version")
        if record.get("record_type") != "modality_lifecycle_checkpoint":
            raise ValueError("invalid lifecycle checkpoint record type")
        completed = record.get("completed_stages")
        digests = record.get("stage_digests")
        if not isinstance(completed, list) or not all(
            isinstance(stage, str) for stage in completed
        ):
            raise ValueError("checkpoint completed_stages must be a list")
        if not isinstance(digests, list) or not all(
            isinstance(digest, str) for digest in digests
        ):
            raise ValueError("checkpoint stage_digests must be a list")
        seed = record.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("checkpoint seed is invalid")
        require_ground_truth = record.get("require_independent_ground_truth")
        if not isinstance(require_ground_truth, bool):
            raise ValueError("checkpoint ground-truth requirement is invalid")
        status = record.get("status")
        if status not in ("running", "completed"):
            raise ValueError("checkpoint status is invalid")
        signature = record.get("signature")
        if not isinstance(signature, Mapping):
            raise ValueError("checkpoint signature is missing")
        if signature.get("algorithm") != attestation.SIGNATURE_ALGORITHM:
            raise ValueError("unsupported lifecycle checkpoint signature")
        signature_value = signature.get("value")
        if not isinstance(signature_value, str) or not signature_value.strip():
            raise ValueError("lifecycle checkpoint signature is invalid")
        manifest_digest = record.get("manifest_digest")
        if manifest_digest is not None and not isinstance(manifest_digest, str):
            raise ValueError("checkpoint manifest_digest is invalid")
        return cls(
            modality=str(record.get("modality", "")),
            dataset_hash=str(record.get("dataset_hash", "")),
            observed_dataset_hash=str(record.get("observed_dataset_hash", "")),
            package_path=str(record.get("package_path", "")),
            report_path=str(record.get("report_path", "")),
            seed=seed,
            completed_stages=tuple(completed),
            stage_digests=tuple(digests),
            require_independent_ground_truth=require_ground_truth,
            manifest_digest=manifest_digest,
            status=cast(Literal["running", "completed"], status),
            signature=signature_value,
            schema_version=1,
        )

    def to_dict(self) -> dict[str, object]:
        """Converts the checkpoint to deterministic JSON-native evidence.

        Returns:
            The unsigned content plus a `signature` entry holding the algorithm
            and value.
        """
        record = self.unsigned_dict()
        record["signature"] = {
            "algorithm": attestation.SIGNATURE_ALGORITHM,
            "value": self.signature,
        }
        return record

    @classmethod
    def read(cls, path: str | Path) -> Self:
        """Reads a persisted lifecycle checkpoint.

        Args:
            path: JSON file written from `to_dict`.

        Returns:
            The parsed checkpoint (signature not yet verified).

        Raises:
            ValueError: If the file does not hold a JSON object.
        """
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(record, Mapping):
            raise ValueError("lifecycle checkpoint must be a JSON object")
        return cls.from_dict(record)

    @classmethod
    def read_verified(cls, path: str | Path, signing_key: str | bytes) -> Self:
        """Reads a checkpoint and rejects it unless its signature verifies.

        Args:
            path: JSON file written from `to_dict`.
            signing_key: Key the signature should have been made with.

        Returns:
            The parsed checkpoint with a verified signature.

        Raises:
            ValueError: If the signature does not verify.
        """
        checkpoint = cls.read(path)
        if not checkpoint.verify_signature(signing_key):
            raise ValueError("lifecycle checkpoint signature does not verify")
        return checkpoint


@dataclasses.dataclass(frozen=True, slots=True)
class ModalityCurationResult:
    """Signed curation evidence for a dataset without a model claim.

    ``content_digest`` is the SHA-256 digest of the canonical unsigned
    evidence. It is deliberately independent of the signing key so that
    consumers can compare curation content across attestations.

    Attributes:
        modality: Modality value of the dataset.
        dataset_hash: Hash identifying the dataset.
        sample_count: Number of samples curated; non-negative.
        curation: Curation report evidence as a JSON-native mapping.
        dataset_card: Dataset card evidence as a JSON-native mapping.
        path: Where the evidence file is written.
        signature: SHA-256 signature of the unsigned evidence.
    """

    modality: str
    dataset_hash: str
    sample_count: int
    curation: Mapping[str, object]
    dataset_card: Mapping[str, object]
    path: str
    signature: str

    def __post_init__(self) -> None:
        """Validates curation evidence identity before it is published.

        Raises:
            ValueError: If an identity field is blank, the sample count is not a
                non-negative integer, the signature is not a SHA-256 digest, or
                the curation or dataset card disagrees with the modality,
                dataset hash or sample count.
        """
        for field_name in ("modality", "dataset_hash", "path"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty")
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 0
        ):
            raise ValueError("sample_count must be a non-negative integer")
        if not isinstance(self.signature, str) or not is_sha256(self.signature):
            raise ValueError("signature must be a SHA-256 digest")
        normalized_curation = normalize_json_object(self.curation, "curation")
        normalized_card = normalize_json_object(
            self.dataset_card, "dataset_card"
        )
        for name, record in (
            ("curation", normalized_curation),
            ("dataset_card", normalized_card),
        ):
            if "modality" in record and record.get("modality") != self.modality:
                raise ValueError(f"{name} does not match modality")
            if record.get("dataset_hash") != self.dataset_hash:
                raise ValueError(f"{name} does not match dataset_hash")
            if record.get("sample_count") != self.sample_count:
                raise ValueError(f"{name} does not match sample_count")
        object.__setattr__(self, "curation", normalized_curation)
        object.__setattr__(self, "dataset_card", normalized_card)

    def unsigned_dict(self) -> dict[str, object]:
        """Builds the exact curation content covered by the signature.

        Returns:
            JSON-native mapping of the record type, identity, curation and
            dataset card.
        """
        return {
            "schema_version": CURATION_SCHEMA_VERSION,
            "record_type": "modality_curation",
            "modality": self.modality,
            "dataset_hash": self.dataset_hash,
            "sample_count": self.sample_count,
            "curation": dict(self.curation),
            "dataset_card": dict(self.dataset_card),
        }

    @property
    def content_digest(self) -> str:
        """The deterministic digest of the unsigned evidence."""
        return attestation.canonical_digest(self.unsigned_dict())

    def verify_signature(self, signing_key: str | bytes) -> bool:
        """Checks the persisted curation signature.

        An empty string key raises ValueError from the signing helper.

        Args:
            signing_key: Key the signature should have been made with.

        Returns:
            True if the signature matches the unsigned evidence, else False.
        """
        expected = attestation.sign_payload(self.unsigned_dict(), signing_key)
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> dict[str, object]:
        """Converts the result to JSON-native curation evidence.

        Returns:
            The unsigned content plus the content digest, path and a `signature`
            entry holding the algorithm and value.
        """
        record = self.unsigned_dict()
        record["content_digest"] = self.content_digest
        record["path"] = self.path
        record["signature"] = {
            "algorithm": attestation.SIGNATURE_ALGORITHM,
            "value": self.signature,
        }
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> Self:
        """Restores a curation result without trusting its signature claim.

        Args:
            record: Mapping produced by `to_dict`.

        Returns:
            The restored result.

        Raises:
            ValueError: If the schema version, record type, identity, evidence,
                signature or content digest is unsupported, malformed or does
                not match.
        """
        if record.get("schema_version") != CURATION_SCHEMA_VERSION:
            raise ValueError("unsupported modality curation schema version")
        if record.get("record_type") != "modality_curation":
            raise ValueError("invalid modality curation record type")
        modality = record.get("modality")
        dataset_hash = record.get("dataset_hash")
        path = record.get("path")
        sample_count = record.get("sample_count")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (modality, dataset_hash, path)
        ):
            raise ValueError("modality curation identity is invalid")
        if isinstance(sample_count, bool) or not isinstance(sample_count, int):
            raise ValueError("modality curation sample_count is invalid")
        curation = record.get("curation")
        dataset_card = record.get("dataset_card")
        if not isinstance(curation, Mapping) or not isinstance(
            dataset_card, Mapping
        ):
            raise ValueError("modality curation evidence is invalid")
        signature = record.get("signature")
        if not isinstance(signature, Mapping):
            raise ValueError("modality curation signature is missing")
        if signature.get("algorithm") != attestation.SIGNATURE_ALGORITHM:
            raise ValueError("unsupported modality curation signature")
        signature_value = signature.get("value")
        if not isinstance(signature_value, str) or not is_sha256(
            signature_value
        ):
            raise ValueError("modality curation signature is invalid")
        result = cls(
            modality=cast(str, modality),
            dataset_hash=cast(str, dataset_hash),
            sample_count=sample_count,
            curation=curation,
            dataset_card=dataset_card,
            path=cast(str, path),
            signature=signature_value,
        )
        content_digest = record.get("content_digest")
        if content_digest is not None:
            if not isinstance(content_digest, str) or not is_sha256(
                content_digest
            ):
                raise ValueError("modality curation content_digest is invalid")
            if not hmac.compare_digest(content_digest, result.content_digest):
                raise ValueError(
                    "modality curation content_digest does not match evidence"
                )
        return result

    @classmethod
    def read(cls, path: str | Path) -> Self:
        """Reads a persisted curation result.

        Args:
            path: JSON file written from `to_dict`.

        Returns:
            The parsed result (signature not yet verified).

        Raises:
            ValueError: If the file does not hold a JSON object.
        """
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(record, Mapping):
            raise ValueError("modality curation record must be an object")
        return cls.from_dict(record)

    @classmethod
    def read_verified(cls, path: str | Path, signing_key: str | bytes) -> Self:
        """Reads curation evidence and requires a valid content signature.

        Args:
            path: JSON file written from `to_dict`.
            signing_key: Key the signature should have been made with.

        Returns:
            The parsed result with a verified signature.

        Raises:
            ValueError: If the signature does not verify.
        """
        result = cls.read(path)
        if not result.verify_signature(signing_key):
            raise ValueError("modality curation signature does not verify")
        return result


class ModalityCurationAdapter(abc.ABC):
    """Polymorphic boundary for source identity and data understanding."""

    @property
    @abc.abstractmethod
    def modality(self) -> modalities.ModalityIdentity:
        """The declared dataset family."""

    @property
    @abc.abstractmethod
    def dataset_hash(self) -> str:
        """The immutable dataset identity."""

    @property
    @abc.abstractmethod
    def samples(self) -> tuple[Any, ...]:
        """The samples in canonical source order."""

    @abc.abstractmethod
    def curate(
        self,
    ) -> curation.ModalityCurationReport | document.DocumentCurationReport:
        """Builds reviewable data-understanding evidence.

        Returns:
            The curation report for the dataset.
        """

    @abc.abstractmethod
    def modality_notes(self) -> tuple[str, ...]:
        """Lists representation and curation limits.

        Returns:
            Plain statements of what the representation and curation do not
            cover.
        """


@dataclasses.dataclass(frozen=True, slots=True)
class ExtensionCurationAdapter(ModalityCurationAdapter):
    """Adapts arbitrary caller-owned records to the curation-only contract.

    The callback owns domain-specific profiling and validation. This adapter
    binds its result to the extension identity, immutable dataset identity,
    and sample count before the signed curation project can publish evidence.
    It intentionally does not provide split, feature, label, or model hooks.

    Attributes:
        extension: Namespaced extension modality identity.
        sample_values: Caller-owned records in canonical order.
        dataset_hash_value: Caller-owned immutable dataset identity.
        curation_callback: Callable that returns the curation report.
        limitations: Non-empty statements of the dataset's boundaries.
    """

    extension: modalities.ExtensionModality
    sample_values: tuple[Any, ...]
    dataset_hash_value: str
    curation_callback: Callable[[], curation.ModalityCurationReport]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validates the caller-owned curation boundary.

        Raises:
            TypeError: If `extension` is not an `ExtensionModality` or the
                callback is not callable.
            ValueError: If the dataset hash is blank or the limitations are
                empty or contain blank values.
        """
        if not isinstance(self.extension, modalities.ExtensionModality):
            raise TypeError("extension must be ExtensionModality")
        if not self.dataset_hash_value.strip():
            raise ValueError("dataset_hash_value must not be empty")
        if not callable(self.curation_callback):
            raise TypeError("curation_callback must be callable")
        if not self.limitations or any(
            not value.strip() for value in self.limitations
        ):
            raise ValueError("limitations must contain non-empty values")
        object.__setattr__(self, "sample_values", tuple(self.sample_values))
        object.__setattr__(self, "limitations", tuple(self.limitations))

    @property
    def modality(self) -> modalities.ExtensionModality:
        """The namespaced extension identity."""
        return self.extension

    @property
    def dataset_hash(self) -> str:
        """The caller-owned immutable dataset identity."""
        return self.dataset_hash_value

    @property
    def samples(self) -> tuple[Any, ...]:
        """The records in caller-defined canonical order."""
        return self.sample_values

    def curate(
        self,
    ) -> curation.ModalityCurationReport:
        """Runs and validates the caller-owned curation callback.

        Returns:
            The report returned by the callback.

        Raises:
            TypeError: If the callback does not return a curation report.
            ValueError: If the report's modality, dataset hash or sample count
                does not match the extension.
        """
        report = self.curation_callback()
        if not isinstance(report, curation.ModalityCurationReport):
            raise TypeError("curation_callback must return a curation report")
        expected_modality = modalities.modality_value(self.extension)
        if report.modality != expected_modality:
            raise ValueError(
                "curation report modality does not match extension"
            )
        if report.dataset_hash != self.dataset_hash_value:
            raise ValueError("curation report hash does not match extension")
        if report.sample_count != len(self.sample_values):
            raise ValueError("curation report count does not match extension")
        return report

    def modality_notes(self) -> tuple[str, ...]:
        """Lists the explicit caller-owned boundaries for the dataset card.

        Returns:
            The limitations given at construction.
        """
        return self.limitations


class ModalityAdapter(ModalityCurationAdapter):
    """Polymorphic boundary between a dataset family and the lifecycle."""

    modality: ClassVar[modalities.ModalityIdentity]

    @abc.abstractmethod
    def split(self, seed: int) -> DatasetPartitions:
        """Splits the dataset into leakage-aware partitions.

        Args:
            seed: Random seed for the split.

        Returns:
            Train, validation and test partitions.
        """

    @abc.abstractmethod
    def features(self, samples: Sequence[Any]) -> FeatureBatch:
        """Builds bounded features for the supplied samples.

        Args:
            samples: Samples to featurize.

        Returns:
            Feature batch with one row per sample, in order.
        """

    @abc.abstractmethod
    def sample_id(self, sample: Any) -> str:
        """Gives the stable identity of one sample.

        Args:
            sample: Sample to identify.

        Returns:
            The sample's identifier.
        """

    @abc.abstractmethod
    def label(self, sample: Any) -> object | None:
        """Gives the supervised label of one sample.

        Args:
            sample: Sample to label.

        Returns:
            The label, or None if the sample has none.
        """

    @abc.abstractmethod
    def monitor(self, observed: ModalityAdapter) -> dict[str, object]:
        """Compares this reference adapter with an observed adapter.

        Args:
            observed: Adapter over the newly observed data.

        Returns:
            Drift evidence as a JSON-native mapping.
        """

    def modality_specific_explanation(
        self,
        sample: Any,
        model: LogisticRegression,
        feature_names: Sequence[str],
        coefficients: numpy.ndarray,
    ) -> dict[str, object]:
        """Builds explicit modality-specific explanation evidence.

        The default is an honest unavailable result. Adapters override this
        hook only when they own a decoder or perturbation policy that can be
        represented and tested without inventing raw-modality semantics.

        Args:
            sample: Sample to explain.
            model: Fitted classifier.
            feature_names: Names of the model's features.
            coefficients: Linear coefficients for the predicted class, one per
                feature.

        Returns:
            Mapping with a "status" key; in this base class always
            "not_available" with the reason and limitations.
        """
        return {
            "status": "not_available",
            "modality": modalities.modality_value(self.modality),
            "reason": "the adapter has no modality-specific explanation policy",
            "limitations": [
                "The lifecycle explanation remains in feature space.",
                "No raw-modality or causal interpretation is provided.",
            ],
        }

    def explanation_ground_truth(
        self,
        sample: Any,
        feature_names: Sequence[str],
    ) -> quality.ExplanationGroundTruth | None:
        """Gives caller-owned independent explanation evidence, if known.

        Built-in adapters return no reference by default because raw-modality
        semantics cannot be inferred from a feature matrix. A domain adapter
        may override this method when it carries an independently authored
        annotation, benchmark, or synthetic oracle.

        Args:
            sample: Sample being explained.
            feature_names: Names of the model's features.

        Returns:
            The ground truth for the sample, or None if none is known (always
            None in this base class).
        """
        del sample, feature_names
        return None

    def labels(self, samples: Sequence[Any]) -> tuple[object, ...]:
        """Gives the labels of several samples.

        Args:
            samples: Samples to label.

        Returns:
            Labels in sample order.
        """
        return tuple(self.label(sample) for sample in samples)

    def group_ids(self, samples: Sequence[Any]) -> tuple[str, ...]:
        """Gives the stable grouping identities used for split evidence.

        Args:
            samples: Samples to identify.

        Returns:
            The sample identifier of each sample, in order.
        """
        return tuple(self.sample_id(sample) for sample in samples)


class DocumentAdapter(ModalityAdapter):
    """Adapter for UTF-8, Markdown, and caller-decoded document samples.

    Attributes:
        dataset: Immutable document dataset the adapter wraps.
    """

    modality = modalities.Modality.DOCUMENT

    def __init__(self, dataset: document.DocumentDataset) -> None:
        """Binds an immutable document dataset.

        Args:
            dataset: Documents the lifecycle runs on.
        """
        self.dataset = dataset

    @property
    def dataset_hash(self) -> str:
        """The document corpus identity."""
        return self.dataset.content_hash

    @property
    def samples(self) -> tuple[document.DocumentSample, ...]:
        """The document samples in the dataset."""
        return self.dataset.samples

    def curate(self) -> document.DocumentCurationReport:
        """Builds document profiling and label evidence.

        Returns:
            The curation report for the dataset.
        """
        return self.dataset.curate()

    def split(self, seed: int) -> DatasetPartitions:
        """Splits by document identity to prevent chunk leakage.

        Args:
            seed: Random seed for the split.

        Returns:
            Train, validation and test partitions, with the sorted document IDs
            of each under `groups`.
        """
        result = self.dataset.split(seed=seed)
        return DatasetPartitions(
            result.train,
            result.validation,
            result.test,
            {
                "train": tuple(
                    sorted({sample.document_id for sample in result.train})
                ),
                "validation": tuple(
                    sorted({sample.document_id for sample in result.validation})
                ),
                "test": tuple(
                    sorted({sample.document_id for sample in result.test})
                ),
            },
        )

    def features(
        self, samples: Sequence[document.DocumentSample]
    ) -> FeatureBatch:
        """Prepares bounded lexical features without using labels.

        Args:
            samples: Documents to featurize.

        Returns:
            Batch with one row per document and two columns, "character_count"
            and "token_count".
        """
        profiles = document.DocumentDataset(tuple(samples)).profile()
        rows = numpy.asarray(
            [
                (profile.character_count, profile.token_count)
                for profile in profiles
            ],
            dtype=numpy.float64,
        )
        return FeatureBatch(
            tuple(self.sample_id(sample) for sample in samples),
            ("character_count", "token_count"),
            rows,
        )

    def sample_id(self, sample: document.DocumentSample) -> str:
        """Gives a stable identity for one document sample or chunk.

        Args:
            sample: Document sample.

        Returns:
            The document ID and content hash joined by a colon.
        """
        return f"{sample.document_id}:{sample.content_hash}"

    def label(self, sample: document.DocumentSample) -> object | None:
        """Gives the document label.

        Args:
            sample: Document sample.

        Returns:
            The sample's label, or None if it has none.
        """
        return sample.label

    def modality_specific_explanation(
        self,
        sample: Any,
        model: LogisticRegression,
        feature_names: Sequence[str],
        coefficients: numpy.ndarray,
    ) -> dict[str, object]:
        """Builds page and section provenance for a document explanation.

        Args:
            sample: Document sample to explain.
            model: Fitted classifier.
            feature_names: Names of the model's features (unused).
            coefficients: Linear coefficients (unused).

        Returns:
            Mapping with the document's source identity, citations, provenance
            spans, source-span occlusion scores and limitations.

        Raises:
            TypeError: If `sample` is not a `DocumentSample`.
        """
        if not isinstance(sample, document.DocumentSample):
            raise TypeError("sample must be a DocumentSample")
        raw_occlusion = self.raw_span_occlusion_scores(sample, model)
        return {
            "status": "available",
            "method": "document_span_provenance",
            "document_id": sample.document_id,
            "source_uri": sample.source_uri,
            "source_digest": sample.source_digest,
            "format": sample.format.value,
            "content_hash": sample.content_hash,
            "citations": [
                citation.to_dict() for citation in sample.citations()
            ],
            "spans": [
                {
                    "kind": span.kind,
                    "start": span.start,
                    "end": span.end,
                    "label": span.label,
                    "page_number": span.page_number,
                }
                for span in sample.provenance
            ],
            "raw_input_occlusion": raw_occlusion,
            "limitations": [
                "Span provenance and occlusion identify source context; they "
                "are not OCR confidence or citation grounding.",
                "The lifecycle model uses bounded lexical counts.",
                "No causal claim is made.",
            ],
        }

    def raw_span_occlusion_scores(
        self,
        sample: document.DocumentSample,
        model: LogisticRegression,
    ) -> dict[str, object]:
        """Scores source-character masks while preserving span offsets.

        Args:
            sample: Document sample to explain.
            model: Fitted classifier.

        Returns:
            Mapping with one entry per provenance span holding the change in the
            model's decision score when the span's characters are replaced by
            spaces (full score minus masked score), plus the baseline and
            limitations.
        """
        original_features = self.features((sample,)).values
        original_decision = numpy.asarray(
            model.decision_function(original_features)
        )
        if original_decision.ndim == 1:
            selected_class = 0
            original_score = float(original_decision[0])
        else:
            predicted = model.predict(original_features)[0]
            selected_class = int(
                numpy.flatnonzero(model.classes_ == predicted)[0]
            )
            original_score = float(original_decision[0, selected_class])
        spans: list[dict[str, object]] = []
        citations = sample.citations()
        for provenance in sample.provenance:
            characters = list(sample.content)
            characters[provenance.start : provenance.end] = " " * (
                provenance.end - provenance.start
            )
            masked_content = "".join(characters)
            if not masked_content.strip():
                masked_content = "."
            masked_sample = document.DocumentSample(
                document_id=sample.document_id,
                content=masked_content,
                format=sample.format,
                source_uri=sample.source_uri,
                encoding=sample.encoding,
                source_digest=sample.source_digest,
                label=sample.label,
                provenance=sample.provenance,
            )
            masked_features = self.features((masked_sample,)).values
            masked_decision = numpy.asarray(
                model.decision_function(masked_features)
            )
            if masked_decision.ndim == 1:
                masked_score = float(masked_decision[0])
            else:
                masked_score = float(masked_decision[0, selected_class])
            spans.append(
                {
                    "kind": provenance.kind,
                    "start": provenance.start,
                    "end": provenance.end,
                    "page_number": provenance.page_number,
                    "citation": next(
                        (
                            citation.to_dict()
                            for citation in citations
                            if citation.start == provenance.start
                            and citation.end == provenance.end
                        ),
                        None,
                    ),
                    "score_delta_from_full": original_score - masked_score,
                    "masked_content_hash": masked_sample.content_hash,
                }
            )
        return {
            "status": "available",
            "method": "raw_document_span_occlusion",
            "spans": spans,
            "citations": [citation.to_dict() for citation in citations],
            "baseline": "space replacement in source character spans",
            "feature_space_only": False,
            "limitations": [
                "The model consumes bounded character and token counts after "
                "masking.",
                "Span masks do not establish citation grounding, OCR "
                "confidence, or semantic causality.",
            ],
        }

    def monitor(self, observed: ModalityAdapter) -> dict[str, object]:
        """Builds the document profile drift report.

        Args:
            observed: Adapter over the newly observed documents.

        Returns:
            The drift report as a JSON-native mapping.

        Raises:
            TypeError: If `observed` is not a `DocumentAdapter`.
        """
        if not isinstance(observed, DocumentAdapter):
            raise TypeError("observed adapter must be a DocumentAdapter")
        current = self.dataset.curate()
        report = documents.DocumentMonitor(current).observe(observed.dataset)
        return report.to_dict()

    def modality_notes(self) -> tuple[str, ...]:
        """Describes the bounded document representation.

        Returns:
            Plain statements of what the representation does not cover.
        """
        return (
            "The lifecycle baseline uses character and token counts, not OCR, "
            "layout, or semantic embeddings.",
            "Term-level document explanations require the separate "
            "DocumentClassifier path.",
        )


class TextAdapter(DocumentAdapter):
    """Document adapter with an explicit text modality declaration."""

    modality = modalities.Modality.TEXT


class ImageAdapter(ModalityAdapter):
    """Adapter for bounded, decoded image samples.

    Attributes:
        dataset: Immutable image dataset the adapter wraps.
    """

    modality = modalities.Modality.IMAGE

    def __init__(self, dataset: image.ImageDataset) -> None:
        """Binds an immutable image dataset.

        Args:
            dataset: Images the lifecycle runs on.
        """
        self.dataset = dataset

    @property
    def dataset_hash(self) -> str:
        """The image corpus identity."""
        return self.dataset.content_hash

    @property
    def samples(self) -> tuple[image.ImageSample, ...]:
        """The image samples in the dataset."""
        return self.dataset.samples

    def curate(self) -> curation.ModalityCurationReport:
        """Builds image dimension, encoding, duplicate, and label evidence.

        Returns:
            The curation report for the dataset.
        """
        return self.dataset.curate()

    def split(self, seed: int) -> DatasetPartitions:
        """Splits by capture group to prevent identity leakage.

        Args:
            seed: Random seed for the split.

        Returns:
            Train, validation and test partitions, with the dataset's group IDs
            under `groups`.
        """
        result = self.dataset.split(seed=seed)
        return DatasetPartitions(
            result.train,
            result.validation,
            result.test,
            result.group_ids(),
        )

    def features(self, samples: Sequence[image.ImageSample]) -> FeatureBatch:
        """Prepares deterministic resized-pixel features.

        Args:
            samples: Images to featurize.

        Returns:
            Batch with one row per image and 768 columns, one per pixel of a
            16 x 16 RGB image, named "pixel[row,column,channel]".
        """
        dataset = image.ImageDataset(tuple(samples))
        values = dataset.prepare_tabular_features()
        width, height = (16, 16)
        channels = 3
        names = tuple(
            f"pixel[{row},{column},{channel}]"
            for row in range(height)
            for column in range(width)
            for channel in range(channels)
        )
        return FeatureBatch(
            tuple(self.sample_id(sample) for sample in samples), names, values
        )

    def sample_id(self, sample: image.ImageSample) -> str:
        """Gives a source-and-content identity for one image sample.

        Args:
            sample: Image sample.

        Returns:
            The image source and content hash joined by a colon.
        """
        return f"{sample.source}:{sample.content_hash}"

    def label(self, sample: image.ImageSample) -> object | None:
        """Gives the image label.

        Args:
            sample: Image sample.

        Returns:
            The sample's label, or None if it has none.
        """
        return sample.label

    def modality_specific_explanation(
        self,
        sample: Any,
        model: LogisticRegression,
        feature_names: Sequence[str],
        coefficients: numpy.ndarray,
    ) -> dict[str, object]:
        """Builds bounded occlusion evidence on transformed RGB pixels.

        Args:
            sample: Image sample to explain.
            model: Fitted classifier.
            feature_names: Names of the model's features (unused).
            coefficients: Linear coefficients (unused).

        Returns:
            Mapping with the ten 4 x 4 pixel regions whose occlusion changes the
            decision score most (score minus occluded score), the source-pixel
            occlusion evidence and limitations.

        Raises:
            TypeError: If `sample` is not an `ImageSample`.
            ValueError: If the features are not 16 x 16 RGB.
        """
        if not isinstance(sample, image.ImageSample):
            raise TypeError("sample must be an ImageSample")
        features = self.features((sample,)).values
        if features.shape[1] != 16 * 16 * 3:
            raise ValueError("image lifecycle expects 16x16 RGB features")
        decision = numpy.asarray(model.decision_function(features))
        if decision.ndim == 1:
            selected_class = 0
            original_score = float(decision[0])
        else:
            predicted = model.predict(features)[0]
            selected_class = int(
                numpy.flatnonzero(model.classes_ == predicted)[0]
            )
            original_score = float(decision[0, selected_class])
        terms: list[tuple[str, float]] = []
        width = 16
        height = 16
        channels = 3
        for row in range(0, height, 4):
            for column in range(0, width, 4):
                end_row = min(row + 4, height)
                end_column = min(column + 4, width)
                indexes = [
                    (pixel_row * width + pixel_column) * channels + channel
                    for pixel_row in range(row, end_row)
                    for pixel_column in range(column, end_column)
                    for channel in range(channels)
                ]
                occluded = features.copy()
                occluded[0, indexes] = 0.0
                occluded_decision = numpy.asarray(
                    model.decision_function(occluded)
                )
                if occluded_decision.ndim == 1:
                    occluded_score = float(occluded_decision[0])
                else:
                    occluded_score = float(occluded_decision[0, selected_class])
                terms.append(
                    (
                        f"region[{row}:{end_row},{column}:{end_column}]",
                        original_score - occluded_score,
                    )
                )
        selected = sorted(
            terms,
            key=lambda item: (-abs(item[1]), item[0]),
        )[:10]
        raw_terms = self.raw_pixel_occlusion_terms(
            sample, model, selected_class, original_score
        )
        return {
            "status": "available",
            "method": "transformed_pixel_region_occlusion",
            "terms": [[name, float(value)] for name, value in selected],
            "baseline": "zero-valued transformed RGB pixels",
            "feature_space_only": True,
            "raw_input_occlusion": raw_terms,
            "limitations": [
                "Occlusion occurs after resizing and RGB conversion.",
                "Raw input occlusion is bounded pixel evidence, not object "
                "localization or validated saliency.",
                "Score changes are associational and not causal.",
            ],
        }

    def raw_pixel_occlusion_terms(
        self,
        sample: image.ImageSample,
        model: LogisticRegression,
        selected_class: int,
        original_score: float,
    ) -> dict[str, object]:
        """Scores bounded source-pixel masks before the resize transform.

        The model still consumes the declared 16x16 RGB representation. The
        difference is that each mask is applied to the validated source
        pixels first, so the evidence records a source-space perturbation and
        the exact image dimensions used to construct it.

        Args:
            sample: Image sample to explain.
            model: Fitted classifier.
            selected_class: Index of the class whose score is tracked.
            original_score: Decision score of the unmasked image for that class.

        Returns:
            Mapping with the ten source regions (a 4 x 4 grid over the source
            image) whose masking changes the score most (original minus masked),
            the source dimensions as [width, height], the baseline and
            limitations.

        Raises:
            ValueError: If the decoded image does not have three channels.
        """
        source_pixels = sample.decode("RGB")
        source_height, source_width, source_channels = source_pixels.shape
        if source_channels != 3:
            raise ValueError("decoded RGB image must have three channels")
        image_library = image.require_pillow("raw image explanation")
        row_step = max(1, int(numpy.ceil(source_height / 4)))
        column_step = max(1, int(numpy.ceil(source_width / 4)))
        terms: list[tuple[str, float]] = []
        for row in range(0, source_height, row_step):
            for column in range(0, source_width, column_step):
                end_row = min(row + row_step, source_height)
                end_column = min(column + column_step, source_width)
                masked_pixels = source_pixels.copy()
                masked_pixels[row:end_row, column:end_column, :] = 0
                resized = image_library.fromarray(masked_pixels).resize(
                    (16, 16), image_library.Resampling.BILINEAR
                )
                features = (
                    numpy.asarray(resized, dtype=numpy.float32).reshape(1, -1)
                    / 255.0
                )
                decision = numpy.asarray(model.decision_function(features))
                if decision.ndim == 1:
                    masked_score = float(decision[0])
                else:
                    masked_score = float(decision[0, selected_class])
                terms.append(
                    (
                        f"source_region[{row}:{end_row},{column}:{end_column}]",
                        original_score - masked_score,
                    )
                )
        selected = sorted(
            terms,
            key=lambda item: (-abs(item[1]), item[0]),
        )[:10]
        return {
            "method": "raw_pixel_region_occlusion",
            "terms": [[name, float(value)] for name, value in selected],
            "source_dimensions": [source_width, source_height],
            "baseline": "zero-valued decoded RGB source pixels",
            "feature_space_only": False,
            "limitations": [
                "Source regions are pixel masks, not semantic objects.",
                "The model still consumes the resized RGB representation.",
                "Score changes are associational and not causal.",
            ],
        }

    def monitor(self, observed: ModalityAdapter) -> dict[str, object]:
        """Builds the mean-pixel and duplicate drift evidence.

        Args:
            observed: Adapter over the newly observed images.

        Returns:
            The drift report as a JSON-native mapping.

        Raises:
            TypeError: If `observed` is not an `ImageAdapter`.
        """
        if not isinstance(observed, ImageAdapter):
            raise TypeError("observed adapter must be an ImageAdapter")
        report = images.ImageMonitor(self.dataset.curate()).observe(
            observed.dataset
        )
        return report.to_dict()

    def modality_notes(self) -> tuple[str, ...]:
        """Describes image lifecycle limits.

        Returns:
            Plain statements of what the representation does not claim.
        """
        return (
            "The baseline uses resized RGB pixels and does not claim "
            "localization, saliency validation, or object detection.",
        )


class AudioAdapter(ModalityAdapter):
    """Adapter for WAV-capable audio samples and explicit codec probes.

    Attributes:
        dataset: Immutable audio dataset the adapter wraps.
    """

    modality = modalities.Modality.AUDIO

    def __init__(self, dataset: audio.AudioDataset) -> None:
        """Binds an immutable audio dataset.

        Args:
            dataset: Audio the lifecycle runs on.
        """
        self.dataset = dataset

    @property
    def dataset_hash(self) -> str:
        """The audio corpus identity."""
        return self.dataset.content_hash

    @property
    def samples(self) -> tuple[audio.AudioSample, ...]:
        """The audio samples in the dataset."""
        return self.dataset.samples

    def curate(self) -> curation.ModalityCurationReport:
        """Builds codec, signal shape, duration, and label evidence.

        Returns:
            The curation report for the dataset.
        """
        return self.dataset.curate()

    def split(self, seed: int) -> DatasetPartitions:
        """Splits by speaker/session group.

        Args:
            seed: Random seed for the split.

        Returns:
            Train, validation and test partitions, with the dataset's group IDs
            under `groups`.
        """
        result = self.dataset.split(seed=seed)
        return DatasetPartitions(
            result.train, result.validation, result.test, result.group_ids()
        )

    def features(self, samples: Sequence[audio.AudioSample]) -> FeatureBatch:
        """Prepares bounded WAV signal statistics.

        Args:
            samples: Audio samples to featurize.

        Returns:
            Batch with one row per sample, identified by its source.
        """
        prepared = audio.AudioDataset(tuple(samples)).prepare_features()
        return FeatureBatch(
            tuple(prepared.sources), prepared.feature_names, prepared.to_numpy()
        )

    def sample_id(self, sample: audio.AudioSample) -> str:
        """Gives the audio source identity.

        Args:
            sample: Audio sample.

        Returns:
            The sample's source.
        """
        return sample.source

    def label(self, sample: audio.AudioSample) -> object | None:
        """Gives the audio label.

        Args:
            sample: Audio sample.

        Returns:
            The sample's label, or None if it has none.
        """
        return sample.label

    def monitor(self, observed: ModalityAdapter) -> dict[str, object]:
        """Builds curation identity and profile drift evidence.

        Args:
            observed: Adapter over the newly observed audio.

        Returns:
            Mapping with both dataset hashes, the sample count change (observed
            minus reference) and, per profile key, whether its value differs.

        Raises:
            TypeError: If `observed` is not an `AudioAdapter`.
        """
        if not isinstance(observed, AudioAdapter):
            raise TypeError("observed adapter must be an AudioAdapter")
        current = observed.dataset.curate()
        reference = self.dataset.curate()
        return {
            "reference_hash": reference.dataset_hash,
            "observed_hash": current.dataset_hash,
            "sample_count_delta": current.sample_count - reference.sample_count,
            "profile_delta": {
                key: current.profile.get(key) != reference.profile.get(key)
                for key in sorted(set(current.profile) | set(reference.profile))
            },
        }

    def modality_notes(self) -> tuple[str, ...]:
        """Describes audio lifecycle limits.

        Returns:
            Plain statements of what the representation does not claim.
        """
        return (
            "Signal features are bounded WAV statistics; non-WAV codecs "
            "require an explicit decoder adapter.",
            "The baseline does not claim transcript, speaker, or time-local "
            "explanations.",
        )

    def modality_specific_explanation(
        self,
        sample: Any,
        model: LogisticRegression,
        feature_names: Sequence[str],
        coefficients: numpy.ndarray,
    ) -> dict[str, object]:
        """Builds bounded time-local score evidence for WAV samples.

        Args:
            sample: Audio sample to explain.
            model: Fitted classifier.
            feature_names: Names of the model's features (unused).
            coefficients: Linear coefficients (unused).

        Returns:
            Mapping with the model's decision score for up to eight consecutive
            time windows of the sample, raw-waveform occlusion scores and a
            noise robustness probe; or a mapping with status "not_available" if
            the sample is not WAV.

        Raises:
            TypeError: If `sample` is not an `AudioSample`.
        """
        if not isinstance(sample, audio.AudioSample):
            raise TypeError("sample must be an AudioSample")
        if sample.metadata.format.lower() != "wav":
            return {
                "status": "not_available",
                "method": "time_local_wav_window_score",
                "reason": "an explicit WAV decoder is required",
            }
        window_count = min(8, sample.metadata.frame_count)
        window_frames = max(
            1, int(numpy.ceil(sample.metadata.frame_count / window_count))
        )
        full_features = self.features((sample,)).values[0]
        full_decision = numpy.asarray(
            model.decision_function(full_features[None])
        )
        full_score = float(full_decision.reshape(-1)[0])
        windows: list[dict[str, object]] = []
        for start_frame in range(0, sample.metadata.frame_count, window_frames):
            prepared = audio.AudioDataset((sample,)).prepare_features(
                max_frames=window_frames,
                start_frame=start_frame,
            )
            values = prepared.to_numpy()[0]
            decision = numpy.asarray(model.decision_function(values[None]))
            windows.append(
                {
                    "start_seconds": start_frame / sample.metadata.sample_rate,
                    "end_seconds": min(
                        sample.metadata.frame_count,
                        start_frame + prepared.frames_observed[0],
                    )
                    / sample.metadata.sample_rate,
                    "start_frame": start_frame,
                    "frames_observed": prepared.frames_observed[0],
                    "score": float(decision.reshape(-1)[0]),
                    "score_delta_from_full": float(
                        decision.reshape(-1)[0] - full_score
                    ),
                }
            )
        raw_windows = self.raw_waveform_window_scores(
            sample, model, full_score, window_frames
        )
        noise_robustness = self.background_noise_robustness(
            sample, model, full_score
        )
        return {
            "status": "available",
            "method": "time_local_wav_window_score",
            "window_count": len(windows),
            "windows": windows,
            "feature_space_only": True,
            "raw_input_occlusion": raw_windows,
            "background_noise_robustness": noise_robustness,
            "limitations": [
                "Raw waveform windows are bounded occlusion evidence, not "
                "validated waveform saliency.",
                "The score is associational and does not establish a causal "
                "audio event.",
            ],
        }

    def background_noise_robustness(
        self,
        sample: audio.AudioSample,
        model: LogisticRegression,
        original_score: float,
        snr_db: tuple[float, ...] = (20.0, 10.0, 0.0),
        seed: int = 0,
    ) -> dict[str, object]:
        """Measures bounded score changes under deterministic additive noise.

        This is a reproducible perturbation probe for the current linear
        baseline. It is not a perceptual speech-quality test, a denoising
        system, or evidence that the model is robust to real recording noise.

        Args:
            sample: WAV audio sample to perturb.
            model: Fitted classifier.
            original_score: Decision score of the unperturbed sample.
            snr_db: Requested signal-to-noise ratios in decibels; one to eight
                finite values in [0, 120].
            seed: Seed of the noise generator.

        Returns:
            Mapping with one scenario per requested ratio holding the observed
            ratio, the score, its change from `original_score` and the share
            of clipped values; or status "not_available" if the sample is not
            WAV or is silent.

        Raises:
            ValueError: If `snr_db` is empty, has more than eight values, or
                holds a value outside [0, 120].
        """
        if sample.metadata.format.lower() != "wav":
            return {
                "status": "not_available",
                "method": "deterministic_additive_background_noise",
                "reason": "an explicit WAV decoder is required",
            }
        if not snr_db or len(snr_db) > 8:
            raise ValueError("snr_db must contain between one and eight values")
        if any(
            not numpy.isfinite(value) or value < 0.0 or value > 120.0
            for value in snr_db
        ):
            raise ValueError("snr_db values must be finite and in [0, 120]")
        parameters, values = audio.wav_pcm_values(sample.encoded_bytes)
        signal_rms = float(numpy.sqrt(numpy.mean(numpy.square(values))))
        if signal_rms <= 0.0:
            return {
                "status": "not_available",
                "method": "deterministic_additive_background_noise",
                "reason": "source signal is silent",
            }
        randomizer = numpy.random.default_rng(seed)
        scenarios: list[dict[str, object]] = []
        for requested_snr in snr_db:
            noise_rms = signal_rms / (10.0 ** (requested_snr / 20.0))
            noise = randomizer.normal(0.0, noise_rms, size=values.shape)
            mixed = values + noise
            clipping_fraction = float(numpy.mean(numpy.abs(mixed) > 1.0))
            payload = audio.wav_payload_from_pcm(parameters, mixed)
            perturbed = audio.AudioSample.from_bytes(
                payload,
                source=f"{sample.source}#noise-{requested_snr:g}",
                label=sample.label,
                group_id=sample.group_id,
                transcript=sample.transcript,
            )
            features = self.features((perturbed,)).values
            score = float(
                numpy.asarray(model.decision_function(features)).reshape(-1)[0]
            )
            observed_noise = mixed - values
            observed_noise_rms = float(
                numpy.sqrt(numpy.mean(numpy.square(observed_noise)))
            )
            observed_snr = 20.0 * math.log10(signal_rms / observed_noise_rms)
            scenarios.append(
                {
                    "requested_snr_db": requested_snr,
                    "observed_snr_db": observed_snr,
                    "score": score,
                    "score_delta_from_clean": score - original_score,
                    "clipping_fraction": clipping_fraction,
                }
            )
        return {
            "status": "available",
            "method": "deterministic_additive_background_noise",
            "seed": seed,
            "scenarios": scenarios,
            "feature_space_only": False,
            "limitations": [
                "Synthetic white noise is a repeatable stressor, not a model "
                "of every recording environment.",
                "Score changes do not establish perceptual quality or causal "
                "robustness.",
                "Clipping is measured after mixing and must be reviewed "
                "before interpreting low-SNR results.",
            ],
        }

    def raw_waveform_window_scores(
        self,
        sample: audio.AudioSample,
        model: LogisticRegression,
        original_score: float,
        window_frames: int,
    ) -> dict[str, object]:
        """Scores source PCM windows after replacing them with silence.

        Args:
            sample: WAV audio sample to explain.
            model: Fitted classifier.
            original_score: Decision score of the unmasked sample.
            window_frames: Number of PCM frames in each window.

        Returns:
            Mapping with one entry per window holding its frame and second
            bounds and the score change (masked score minus original); or
            status "not_available" if the sample width is unknown.
        """
        if sample.metadata.sample_width is None:
            return {
                "status": "not_available",
                "method": "raw_wav_waveform_window_occlusion",
                "reason": "sample width is required to mask PCM frames",
            }
        with wave.open(io.BytesIO(sample.encoded_bytes), "rb") as opened:
            parameters = opened.getparams()
            raw_frames = bytearray(
                opened.readframes(sample.metadata.frame_count)
            )
        frame_width = parameters.sampwidth * parameters.nchannels
        silence_value = 128 if parameters.sampwidth == 1 else 0
        silence_frame = bytes([silence_value]) * frame_width
        windows: list[dict[str, object]] = []
        for start_frame in range(0, sample.metadata.frame_count, window_frames):
            end_frame = min(
                sample.metadata.frame_count, start_frame + window_frames
            )
            masked_frames = raw_frames.copy()
            start_byte = start_frame * frame_width
            end_byte = end_frame * frame_width
            masked_frames[start_byte:end_byte] = silence_frame * (
                end_frame - start_frame
            )
            payload = io.BytesIO()
            with wave.open(payload, "wb") as writer:
                writer.setparams(parameters)
                writer.writeframes(bytes(masked_frames))
            masked_sample = audio.AudioSample.from_bytes(
                payload.getvalue(),
                source=sample.source,
                label=sample.label,
                group_id=sample.group_id,
            )
            features = self.features((masked_sample,)).values
            score = float(
                numpy.asarray(model.decision_function(features)).reshape(-1)[0]
            )
            windows.append(
                {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_seconds": start_frame / sample.metadata.sample_rate,
                    "end_seconds": end_frame / sample.metadata.sample_rate,
                    "score_delta_from_full": score - original_score,
                }
            )
        return {
            "status": "available",
            "method": "raw_wav_waveform_window_occlusion",
            "windows": windows,
            "sample_rate": sample.metadata.sample_rate,
            "frame_count": sample.metadata.frame_count,
            "baseline": "silence replacement in source PCM frames",
            "feature_space_only": False,
            "limitations": [
                "The model consumes bounded aggregate signal features after "
                "masking.",
                "This does not establish transcript, speaker, event, or "
                "causal attribution.",
            ],
        }


class VideoAdapter(ModalityAdapter):
    """Adapter for probed video metadata or an explicit frame decoder.

    Attributes:
        dataset: Immutable video dataset the adapter wraps.
        frame_decoder: Caller-supplied frame decoder, or None to use container
            metadata only.
        max_frames: Maximum number of frames sampled per video.
        max_frame_pixels: Maximum number of pixels per decoded frame.
        temporal_cutoff: Canonical observation time that separates training
            from later data, or None to split by group.
    """

    modality = modalities.Modality.VIDEO

    def __init__(
        self,
        dataset: video.VideoDataset,
        frame_decoder: video.VideoFrameDecoder | None = None,
        max_frames: int = 8,
        max_frame_pixels: int = 1_000_000,
        temporal_cutoff: datetime | str | None = None,
    ) -> None:
        """Binds a dataset, decoder, and optional explicit temporal policy.

        Args:
            dataset: Videos the lifecycle runs on.
            frame_decoder: Decoder that turns encoded bytes and a frame index
                into pixels; if None, only container metadata is used.
            max_frames: Maximum number of frames sampled per video.
            max_frame_pixels: Maximum number of pixels per decoded frame.
            temporal_cutoff: Time that separates training from later data, if
                the split must be temporal.

        Raises:
            ValueError: If `max_frames` or `max_frame_pixels` is below 1.
        """
        if max_frames < 1 or max_frame_pixels < 1:
            raise ValueError("frame bounds must be positive")
        self.dataset = dataset
        self.frame_decoder = frame_decoder
        self.max_frames = max_frames
        self.max_frame_pixels = max_frame_pixels
        self.temporal_cutoff = (
            video.canonical_observation_time(temporal_cutoff)
            if temporal_cutoff is not None
            else None
        )

    @property
    def dataset_hash(self) -> str:
        """The video corpus identity."""
        return self.dataset.content_hash

    @property
    def samples(self) -> tuple[video.VideoSample, ...]:
        """The video samples in the dataset."""
        return self.dataset.samples

    def curate(self) -> curation.ModalityCurationReport:
        """Builds encoding, shape, timing, and label evidence.

        Returns:
            The curation report for the dataset.
        """
        return self.dataset.curate()

    def split(self, seed: int) -> DatasetPartitions:
        """Splits by subject/session/camera group, or by time with a cutoff.

        Args:
            seed: Random seed for the group split; unused for a temporal split.

        Returns:
            Train, validation and test partitions with their group IDs. With a
            temporal cutoff, validation is empty and test holds the later data.
        """
        if self.temporal_cutoff is not None:
            temporal_result = self.dataset.split_temporal(self.temporal_cutoff)

            def group_ids(
                samples: Sequence[video.VideoSample],
            ) -> tuple[str, ...]:
                return tuple(
                    sorted(
                        {sample.group_id or sample.source for sample in samples}
                    )
                )

            return DatasetPartitions(
                temporal_result.train,
                (),
                temporal_result.future,
                {
                    "train": group_ids(temporal_result.train),
                    "validation": (),
                    "test": group_ids(temporal_result.future),
                },
            )
        group_result = self.dataset.split(seed=seed)
        return DatasetPartitions(
            group_result.train,
            group_result.validation,
            group_result.test,
            group_result.group_ids(),
        )

    def features(self, samples: Sequence[video.VideoSample]) -> FeatureBatch:
        """Prepares metadata or explicitly decoded temporal features.

        Args:
            samples: Videos to featurize.

        Returns:
            Batch with one row per video, identified by its source; decoded
            frame statistics if a decoder is set, otherwise container metadata.
        """
        dataset = video.VideoDataset(tuple(samples))
        prepared = (
            dataset.prepare_frame_features(
                self.frame_decoder,
                max_frames=self.max_frames,
                max_frame_pixels=self.max_frame_pixels,
            )
            if self.frame_decoder is not None
            else dataset.prepare_metadata_features()
        )
        return FeatureBatch(
            tuple(prepared.sources), prepared.feature_names, prepared.to_numpy()
        )

    def sample_id(self, sample: video.VideoSample) -> str:
        """Gives the video source identity.

        Args:
            sample: Video sample.

        Returns:
            The sample's source.
        """
        return sample.source

    def label(self, sample: video.VideoSample) -> object | None:
        """Gives the video label.

        Args:
            sample: Video sample.

        Returns:
            The sample's label, or None if it has none.
        """
        return sample.label

    def monitor(self, observed: ModalityAdapter) -> dict[str, object]:
        """Builds curation identity and metadata drift evidence.

        Args:
            observed: Adapter over the newly observed videos.

        Returns:
            Mapping with both dataset hashes, the sample count change (observed
            minus reference) and, per profile key, whether its value differs.

        Raises:
            TypeError: If `observed` is not a `VideoAdapter`.
        """
        if not isinstance(observed, VideoAdapter):
            raise TypeError("observed adapter must be a VideoAdapter")
        reference = self.dataset.curate()
        current = observed.dataset.curate()
        return {
            "reference_hash": reference.dataset_hash,
            "observed_hash": current.dataset_hash,
            "sample_count_delta": current.sample_count - reference.sample_count,
            "profile_delta": {
                key: current.profile.get(key) != reference.profile.get(key)
                for key in sorted(set(current.profile) | set(reference.profile))
            },
        }

    def modality_notes(self) -> tuple[str, ...]:
        """Describes video lifecycle limits.

        Returns:
            Plain statements of the representation in use and what it does not
            cover.
        """
        representation = (
            "The baseline uses caller-decoded, evenly sampled frame "
            "statistics with a bounded pixel budget."
            if self.frame_decoder is not None
            else "The baseline uses probed container metadata and never "
            "silently decodes frames."
        )
        return (
            representation,
            "Temporal, audio-video synchronization, and frame-local "
            "explanations require a decoder policy.",
        )

    def modality_specific_explanation(
        self,
        sample: Any,
        model: LogisticRegression,
        feature_names: Sequence[str],
        coefficients: numpy.ndarray,
    ) -> dict[str, object]:
        """Builds bounded temporal frame-statistics occlusion evidence.

        Args:
            sample: Video sample to explain.
            model: Fitted classifier.
            feature_names: Names of the model's features (unused).
            coefficients: Linear coefficients (unused).

        Returns:
            Mapping with the score change (zeroed minus full) per sampled frame,
            largest absolute change first, plus raw-frame occlusion evidence; or
            status "not_available" if no frame decoder is set.

        Raises:
            TypeError: If `sample` is not a `VideoSample`.
        """
        if not isinstance(sample, video.VideoSample):
            raise TypeError("sample must be a VideoSample")
        if self.frame_decoder is None:
            return {
                "status": "not_available",
                "method": "temporal_frame_statistics_occlusion",
                "reason": "an explicit frame decoder is required",
            }
        prepared = self.features((sample,))
        values = prepared.values[0]
        full_score = float(
            numpy.asarray(model.decision_function(values[None])).reshape(-1)[0]
        )
        terms: list[dict[str, object]] = []
        slot_width = 5
        frame_indices = self.dataset.frame_indices_for(sample, self.max_frames)
        for slot, frame_index in enumerate(frame_indices):
            occluded = values.copy()
            start = slot * slot_width
            occluded[start : start + slot_width] = 0.0
            score = float(
                numpy.asarray(model.decision_function(occluded[None])).reshape(
                    -1
                )[0]
            )
            terms.append(
                {
                    "frame_index": frame_index,
                    "time_seconds": frame_index
                    / sample.metadata.frames_per_second,
                    "score_delta": score - full_score,
                }
            )
        terms.sort(
            key=lambda item: (
                -abs(float(cast(float, item["score_delta"]))),
                int(cast(int, item["frame_index"])),
            )
        )
        raw_windows = self.raw_frame_occlusion_scores(sample, model, full_score)
        return {
            "status": "available",
            "method": "temporal_frame_statistics_occlusion",
            "terms": terms,
            "sampled_frame_indices": list(frame_indices),
            "feature_space_only": True,
            "raw_input_occlusion": raw_windows,
            "limitations": [
                "Raw frame occlusion is bounded decoded-pixel evidence, not "
                "validated motion or semantic saliency.",
                "Audio-video synchronization and temporal event localization "
                "are not established.",
                "Score changes are associational and not causal.",
            ],
        }

    def raw_frame_occlusion_scores(
        self,
        sample: video.VideoSample,
        model: LogisticRegression,
        original_score: float,
    ) -> dict[str, object]:
        """Scores sampled decoded frames after replacing each with zeros.

        Args:
            sample: Video sample to explain.
            model: Fitted classifier.
            original_score: Decision score with no frame masked.

        Returns:
            Mapping with one entry per sampled frame holding its index, time in
            seconds and score change (masked minus original); or status
            "not_available" if no frame decoder is set.
        """
        if self.frame_decoder is None:
            return {
                "status": "not_available",
                "method": "raw_video_frame_occlusion",
                "reason": "an explicit frame decoder is required",
            }
        decoder = self.frame_decoder
        frame_indices = self.dataset.frame_indices_for(sample, self.max_frames)
        windows: list[dict[str, object]] = []
        for target_frame in frame_indices:

            def masked_decoder(
                payload: bytes,
                frame_index: int,
                target_frame_index: int = target_frame,
            ) -> numpy.ndarray:
                frame = numpy.asarray(decoder(payload, frame_index))
                if frame_index == target_frame_index:
                    return numpy.zeros_like(frame)
                return frame

            prepared = video.VideoDataset((sample,)).prepare_frame_features(
                masked_decoder,
                max_frames=self.max_frames,
                max_frame_pixels=self.max_frame_pixels,
            )
            values = prepared.to_numpy()[0]
            score = float(
                numpy.asarray(model.decision_function(values[None])).reshape(
                    -1
                )[0]
            )
            windows.append(
                {
                    "frame_index": target_frame,
                    "time_seconds": target_frame
                    / sample.metadata.frames_per_second,
                    "score_delta_from_full": score - original_score,
                }
            )
        return {
            "status": "available",
            "method": "raw_video_frame_occlusion",
            "windows": windows,
            "sampled_frame_indices": list(frame_indices),
            "baseline": "zero-valued decoded source frames",
            "feature_space_only": False,
            "limitations": [
                "The caller-owned decoder defines frame pixels and codec "
                "semantics.",
                "Frame masks do not establish motion, synchronization, or "
                "semantic localization.",
                "Score changes are associational and not causal.",
            ],
        }


class GraphAdapter(ModalityAdapter):
    """Adapter for inductive node classification on graph node features.

    Attributes:
        dataset: Validated graph with declared node features.
    """

    modality = modalities.Modality.GRAPH

    def __init__(self, dataset: graph.GraphData) -> None:
        """Binds a validated graph with declared node features.

        Args:
            dataset: Graph the lifecycle runs on.
        """
        self.dataset = dataset

    @property
    def dataset_hash(self) -> str:
        """The full graph identity."""
        return self.dataset.identity

    @property
    def samples(self) -> tuple[graph.GraphNode, ...]:
        """The canonical graph nodes."""
        return self.dataset.nodes

    def curate(self) -> curation.ModalityCurationReport:
        """Builds topology, feature, and label evidence.

        Returns:
            The curation report for the graph.
        """
        return self.dataset.curate()

    def split(self, seed: int) -> DatasetPartitions:
        """Splits nodes inductively and retains partition-local edge evidence.

        Args:
            seed: Random seed for the split.

        Returns:
            Train, validation and test partitions of nodes, with the node IDs of
            each under `groups`.
        """
        result = graph.inductive_node_split(self.dataset, seed=seed)
        by_id = {node.node_id: node for node in self.dataset.nodes}
        return DatasetPartitions(
            tuple(by_id[node_id] for node_id in result.train_nodes),
            tuple(by_id[node_id] for node_id in result.validation_nodes),
            tuple(by_id[node_id] for node_id in result.test_nodes),
            {
                "train": result.train_nodes,
                "validation": result.validation_nodes,
                "test": result.test_nodes,
            },
        )

    def features(self, samples: Sequence[graph.GraphNode]) -> FeatureBatch:
        """Prepares declared node features without cross-partition topology.

        Args:
            samples: Nodes to featurize.

        Returns:
            Batch with one row per node, identified by its node ID, and one
            column per declared node feature.

        Raises:
            ValueError: If the graph declares no node features.
        """
        if not self.dataset.feature_names:
            raise ValueError("graph baseline requires declared node features")
        rows = numpy.asarray(
            [sample.features for sample in samples], dtype=float
        )
        return FeatureBatch(
            tuple(sample.node_id for sample in samples),
            self.dataset.feature_names,
            rows,
        )

    def sample_id(self, sample: graph.GraphNode) -> str:
        """Gives a node identity.

        Args:
            sample: Graph node.

        Returns:
            The node ID.
        """
        return sample.node_id

    def label(self, sample: graph.GraphNode) -> object | None:
        """Gives the node label.

        Args:
            sample: Graph node.

        Returns:
            The node's label, or None if it has none.
        """
        return sample.label

    def one_hop_subgraph(
        self,
        node_id: str,
        max_nodes: int = MAX_GRAPH_CONTEXT_NODES,
        max_edges: int = MAX_GRAPH_CONTEXT_EDGES,
    ) -> dict[str, object]:
        """Builds bounded, deterministic one-hop topology evidence.

        The result is review context for a node-feature baseline. It does not
        claim that the model used topology or that edge removal is a faithful
        graph-neural-network attribution method.

        Args:
            node_id: Node at the center of the subgraph.
            max_nodes: Maximum number of nodes returned, including the center.
            max_edges: Maximum number of edges returned.

        Returns:
            Mapping with the center node, the selected node IDs (center first,
            then neighbors in sorted order), the edges among them, the limits,
            flags for whether nodes or edges were truncated, and the observed
            counts of incident edges and neighbors.

        Raises:
            ValueError: If `node_id` is blank or unknown, or a limit is below 1.
        """
        if not node_id.strip():
            raise ValueError("node_id must not be empty")
        if max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        if max_edges < 1:
            raise ValueError("max_edges must be positive")
        if node_id not in self.dataset.node_ids:
            raise ValueError(f"unknown graph node: {node_id}")

        incident_edges = tuple(
            edge
            for edge in self.dataset.edges
            if edge.source == node_id or edge.target == node_id
        )
        neighbor_ids = tuple(
            sorted(
                {
                    edge.target if edge.source == node_id else edge.source
                    for edge in incident_edges
                }
            )
        )
        selected_neighbors = neighbor_ids[: max_nodes - 1]
        selected_node_ids = (node_id, *selected_neighbors)
        selected_node_set = set(selected_node_ids)
        candidate_edges = tuple(
            edge
            for edge in incident_edges
            if edge.source in selected_node_set
            and edge.target in selected_node_set
        )
        selected_edges = candidate_edges[:max_edges]
        return {
            "center_node_id": node_id,
            "node_ids": list(selected_node_ids),
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "weight": edge.weight,
                    "relation": edge.relation,
                }
                for edge in selected_edges
            ],
            "limits": {"max_nodes": max_nodes, "max_edges": max_edges},
            "truncated": {
                "nodes": len(neighbor_ids) > len(selected_neighbors),
                "edges": len(candidate_edges) > len(selected_edges),
            },
            "observed_counts": {
                "incident_edges": len(incident_edges),
                "neighbors": len(neighbor_ids),
            },
        }

    def modality_specific_explanation(
        self,
        sample: Any,
        model: LogisticRegression,
        feature_names: Sequence[str],
        coefficients: numpy.ndarray,
    ) -> dict[str, object]:
        """Builds one-hop graph context and node-feature evidence.

        Args:
            sample: Graph node to explain.
            model: Fitted classifier.
            feature_names: Names of the model's features.
            coefficients: Linear coefficients for the model's features, one per
                feature name.

        Returns:
            Mapping with the one-hop subgraph, the ten largest feature
            contributions (value times coefficient), the decision score and the
            change in score when each of up to 64 features is zeroed.

        Raises:
            TypeError: If `sample` is not a `GraphNode`.
        """
        if not isinstance(sample, graph.GraphNode):
            raise TypeError("sample must be a GraphNode")
        subgraph = self.one_hop_subgraph(sample.node_id)
        subgraph_node_ids = cast(list[str], subgraph["node_ids"])
        neighbors = subgraph_node_ids[1:]
        edges = cast(list[dict[str, object]], subgraph["edges"])
        values = numpy.asarray(sample.features, dtype=numpy.float64)
        decision = numpy.asarray(model.decision_function(values[None]))
        if decision.ndim == 1:
            selected_class = 0
            original_score = float(decision[0])
        else:
            predicted = model.predict(values[None])[0]
            selected_class = int(
                numpy.flatnonzero(model.classes_ == predicted)[0]
            )
            original_score = float(decision[0, selected_class])
        feature_terms = sorted(
            (
                (name, float(values[index] * coefficients[index]))
                for index, name in enumerate(feature_names)
            ),
            key=lambda item: (-abs(item[1]), item[0]),
        )[:10]
        raw_features: list[dict[str, object]] = []
        feature_limit = min(len(feature_names), MAX_RAW_GRAPH_FEATURES)
        for index in range(feature_limit):
            masked_values = list(sample.features)
            masked_values[index] = 0.0
            masked_sample = graph.GraphNode(
                node_id=sample.node_id,
                features=tuple(masked_values),
                label=sample.label,
            )
            masked_features = self.features((masked_sample,)).values
            masked_decision = numpy.asarray(
                model.decision_function(masked_features)
            )
            if masked_decision.ndim == 1:
                masked_score = float(masked_decision[0])
            else:
                masked_score = float(masked_decision[0, selected_class])
            raw_features.append(
                {
                    "feature": feature_names[index],
                    "feature_index": index,
                    "score_delta_from_full": original_score - masked_score,
                }
            )
        return {
            "status": "available",
            "method": "one_hop_graph_context_and_node_feature_contribution",
            "node_id": sample.node_id,
            "neighbors": neighbors,
            "edges": edges,
            "subgraph": subgraph,
            "feature_terms": [[name, value] for name, value in feature_terms],
            "decision_score": float(decision.reshape(-1)[0]),
            "model_uses_topology": False,
            "raw_input_occlusion": {
                "status": "available",
                "method": "raw_graph_node_feature_occlusion",
                "features": raw_features,
                "baseline": "zero replacement in source node features",
                "feature_space_only": False,
                "bounded_by": MAX_RAW_GRAPH_FEATURES,
                "limitations": [
                    "The perturbation masks declared node features, not graph "
                    "topology or neighbor identity.",
                    "Topology remains review context because the baseline does "
                    "not consume edges.",
                    "Score changes are associational and not causal.",
                ],
            },
            "limitations": [
                "The baseline uses declared node features, not graph topology.",
                "One-hop context is evidence for review, not GNN attribution.",
                "No causal or counterfactual graph claim is made.",
            ],
        }

    def monitor(self, observed: ModalityAdapter) -> dict[str, object]:
        """Builds graph identity and profile drift evidence.

        Args:
            observed: Adapter over the newly observed graph.

        Returns:
            Mapping with both graph identities and, per profile key, whether its
            value differs.

        Raises:
            TypeError: If `observed` is not a `GraphAdapter`.
        """
        if not isinstance(observed, GraphAdapter):
            raise TypeError("observed adapter must be a GraphAdapter")
        reference = self.dataset.profile().to_dict()
        current = observed.dataset.profile().to_dict()
        return {
            "reference_hash": self.dataset.identity,
            "observed_hash": observed.dataset.identity,
            "profile_delta": {
                key: current.get(key) != reference.get(key)
                for key in sorted(set(current) | set(reference))
            },
        }

    def modality_notes(self) -> tuple[str, ...]:
        """Describes graph lifecycle limits.

        Returns:
            Plain statements of what the baseline does and does not claim.
        """
        return (
            "The baseline is inductive node classification on declared node "
            "features.",
            "It excludes topology-derived features across partitions and "
            "does not claim graph-neural-network support.",
        )


class GeospatialAdapter(ModalityAdapter):
    """Adapter for CRS-declared vector feature classification.

    Attributes:
        dataset: Geospatial dataset with a declared coordinate reference system.
        block_size: Side of the spatial split blocks, in CRS units.
    """

    modality = modalities.Modality.GEOSPATIAL

    def __init__(
        self, dataset: geospatial.GeospatialDataset, block_size: float = 1.0
    ) -> None:
        """Binds a geospatial dataset and an explicit CRS-unit block size.

        Args:
            dataset: Vector features the lifecycle runs on.
            block_size: Side of the spatial split blocks, in CRS units.

        Raises:
            ValueError: If `block_size` is not positive.
        """
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.dataset = dataset
        self.block_size = block_size

    @property
    def dataset_hash(self) -> str:
        """The vector dataset identity."""
        return self.dataset.content_hash

    @property
    def samples(self) -> tuple[geospatial.GeoFeature, ...]:
        """The geospatial features in the dataset."""
        return self.dataset.features

    def curate(self) -> curation.ModalityCurationReport:
        """Builds geometry, CRS, bounds, and label evidence.

        Returns:
            The curation report with spatial boundary leakage added to its
            profile, and a warning finding if geometries cross block envelopes.
        """
        report = self.dataset.curate()
        profile = dict(report.profile)
        quality_findings = list(report.quality_findings)
        if not self.dataset.crs.strip():
            profile["spatial_boundary_leakage"] = {
                "status": "blocked_missing_crs",
                "block_size": self.block_size,
            }
        else:
            leakage = self.dataset.boundary_leakage(self.block_size)
            profile["spatial_boundary_leakage"] = leakage.to_dict()
            if leakage.crossing_feature_ids:
                quality_findings.append(
                    curation.build_finding(
                        "quality.geospatial-boundary-leakage",
                        data.FindingSeverity.WARNING,
                        {
                            "dataset": self.dataset_hash,
                            "block_size": self.block_size,
                        },
                        leakage.to_dict(),
                        "Review geometries crossing spatial block envelopes "
                        "before using centroid-based evaluation.",
                        False,
                    )
                )
        return dataclasses.replace(
            report,
            profile=profile,
            quality_findings=tuple(quality_findings),
        )

    def split(self, seed: int) -> DatasetPartitions:
        """Splits by spatial blocks in declared CRS units.

        Args:
            seed: Random seed for the split.

        Returns:
            Train, validation and test partitions, with the block IDs of each
            under `groups`.
        """
        result = self.dataset.split_spatial(self.block_size, seed=seed)
        return DatasetPartitions(
            result.train,
            result.validation,
            result.test,
            result.block_ids(),
        )

    def features(
        self, samples: Sequence[geospatial.GeoFeature]
    ) -> FeatureBatch:
        """Prepares bounded geometry envelope and centroid features.

        The dataset is rebuilt with the adapter's CRS and source; a missing
        CRS or a non-finite prepared value raises GeospatialValidationError
        from the dataset.

        Args:
            samples: Features to convert.

        Returns:
            Batch with one row per feature, identified by its feature ID.
        """
        dataset = geospatial.GeospatialDataset(
            tuple(samples), crs=self.dataset.crs, source=self.dataset.source
        )
        prepared = dataset.prepare_features()
        return FeatureBatch(
            tuple(prepared.feature_ids),
            prepared.feature_names,
            prepared.to_numpy(),
        )

    def sample_id(self, sample: geospatial.GeoFeature) -> str:
        """Gives a feature identity.

        Args:
            sample: Geospatial feature.

        Returns:
            The feature ID.
        """
        return sample.feature_id

    def label(self, sample: geospatial.GeoFeature) -> object | None:
        """Gives the feature label.

        Args:
            sample: Geospatial feature.

        Returns:
            The feature's label, or None if it has none.
        """
        return sample.label

    def modality_specific_explanation(
        self,
        sample: Any,
        model: LogisticRegression,
        feature_names: Sequence[str],
        coefficients: numpy.ndarray,
    ) -> dict[str, object]:
        """Builds CRS-bound geometry and feature-contribution evidence.

        Args:
            sample: Geospatial feature to explain; it must be in the dataset.
            model: Fitted classifier.
            feature_names: Names of the model's features.
            coefficients: Linear coefficients for the model's features, one per
                feature name.

        Returns:
            Mapping with the feature's CRS, geometry type and bounds, the ten
            largest feature contributions (value times coefficient), the
            decision score and the score change when each coordinate axis is
            replaced by the bounds midpoint.

        Raises:
            TypeError: If `sample` is not a `GeoFeature`.
        """
        if not isinstance(sample, geospatial.GeoFeature):
            raise TypeError("sample must be a GeoFeature")
        feature_index = next(
            index
            for index, item in enumerate(self.dataset.features)
            if item.feature_id == sample.feature_id
        )
        prepared = self.dataset.prepare_features()
        values = numpy.asarray(
            prepared.to_numpy()[feature_index], dtype=numpy.float64
        )
        decision = numpy.asarray(model.decision_function(values[None]))
        if decision.ndim == 1:
            selected_class = 0
            original_score = float(decision[0])
        else:
            predicted = model.predict(values[None])[0]
            selected_class = int(
                numpy.flatnonzero(model.classes_ == predicted)[0]
            )
            original_score = float(decision[0, selected_class])
        feature_terms = sorted(
            (
                (name, float(values[index] * coefficients[index]))
                for index, name in enumerate(feature_names)
            ),
            key=lambda item: (-abs(item[1]), item[0]),
        )[:10]
        minimum_x, minimum_y, maximum_x, maximum_y = sample.bounds
        baselines = ("x", "y")
        baseline_values = (
            (minimum_x + maximum_x) / 2,
            (minimum_y + maximum_y) / 2,
        )
        raw_axes: list[dict[str, object]] = []
        for axis, axis_name, baseline in zip(
            (0, 1), baselines, baseline_values, strict=True
        ):
            masked_sample = sample.masked_geometry_axis(axis, baseline)
            masked_values = self.features((masked_sample,)).values
            masked_decision = numpy.asarray(
                model.decision_function(masked_values)
            )
            if masked_decision.ndim == 1:
                masked_score = float(masked_decision[0])
            else:
                masked_score = float(masked_decision[0, selected_class])
            raw_axes.append(
                {
                    "axis": axis_name,
                    "baseline_value": baseline,
                    "score_delta_from_full": original_score - masked_score,
                    "masked_geometry_hash": masked_sample.geometry_hash,
                }
            )
        return {
            "status": "available",
            "method": "crs_bound_geometry_and_feature_contribution",
            "feature_id": sample.feature_id,
            "crs": self.dataset.crs,
            "geometry_type": sample.geometry_type,
            "bounds": list(sample.bounds),
            "feature_terms": [[name, value] for name, value in feature_terms],
            "decision_score": float(decision.reshape(-1)[0]),
            "spatial_units_preserved": True,
            "raw_input_occlusion": {
                "status": "available",
                "method": "raw_geospatial_coordinate_axis_occlusion",
                "axes": raw_axes,
                "crs": self.dataset.crs,
                "baseline": "centroid replacement in source CRS units",
                "feature_space_only": False,
                "limitations": [
                    "Axis masks are bounded geometry perturbations, not a map, "
                    "region, or spatial-semantic attribution.",
                    "Geometry validity beyond the declared source contract is "
                    "not inferred from score changes.",
                    "Score changes are associational and not causal.",
                ],
            },
            "limitations": [
                "Coordinates remain in the declared CRS and are not "
                "reprojected.",
                "The result is not a map attribution, spatial join, or causal "
                "claim.",
            ],
        }

    def monitor(self, observed: ModalityAdapter) -> dict[str, object]:
        """Builds geospatial identity, CRS, and profile drift evidence.

        Args:
            observed: Adapter over the newly observed features.

        Returns:
            Mapping with both dataset hashes and flags for whether the CRS and
            the bounds differ.

        Raises:
            TypeError: If `observed` is not a `GeospatialAdapter`.
        """
        if not isinstance(observed, GeospatialAdapter):
            raise TypeError("observed adapter must be a GeospatialAdapter")
        reference = self.dataset.profile().to_dict()
        current = observed.dataset.profile().to_dict()
        return {
            "reference_hash": self.dataset.content_hash,
            "observed_hash": observed.dataset.content_hash,
            "crs_changed": current["crs"] != reference["crs"],
            "bounds_changed": current["bounds"] != reference["bounds"],
        }

    def modality_notes(self) -> tuple[str, ...]:
        """Describes geospatial lifecycle limits.

        Returns:
            Plain statements of what the baseline does and does not claim.
        """
        return (
            "The baseline uses geometry envelopes and centroids in the "
            "declared CRS.",
            "It does not claim reprojection, raster, spatial-join, or causal "
            "spatial inference support.",
        )


class RasterAdapter(ModalityAdapter):
    """Adapter for labelled raster metadata or bounded decoded pixels.

    Attributes:
        dataset: Raster samples, in the order given.
        labels_by_source: Caller-owned label of each raster, keyed by source.
        decoder: Caller-owned pixel decoder, or None for metadata only.
        decoder_name: Stripped name and version of the decoder, or None.
        max_pixels: Maximum number of pixels decoded per raster.
    """

    modality = modalities.Modality.RASTER

    def __init__(
        self,
        samples: Sequence[geospatial.RasterSample],
        labels: Mapping[str, object],
        decoder: geospatial.RasterDecoder | None = None,
        decoder_name: str | None = None,
        max_pixels: int = 4_000_000,
    ) -> None:
        """Binds raster identities and caller-owned labels.

        Raster payload decoding remains caller-owned. When supplied, the
        decoder name and pixel budget become part of the representation
        contract. The label sidecar is keyed by the canonical raster source
        and is part of the caller's dataset evidence.

        Args:
            samples: Rasters to run on; at least one, with unique sources.
            labels: Label of each raster, keyed by its source.
            decoder: Callable that decodes a raster payload to pixels, or None
                to use metadata only.
            decoder_name: Name and version that identify the decoder; required
                with a decoder and not allowed without one.
            max_pixels: Maximum number of pixels decoded per raster.

        Raises:
            ValueError: If there are no samples, sources repeat, a source has no
                label, `max_pixels` is below 1, or the decoder and its name are
                not given together.
            TypeError: If `decoder` is not callable.
        """
        normalized = tuple(samples)
        if not normalized:
            raise ValueError("raster samples must not be empty")
        if decoder is not None and not callable(decoder):
            raise TypeError("decoder must be callable")
        if decoder is None and decoder_name is not None:
            raise ValueError("decoder_name requires a caller-owned decoder")
        if decoder is not None and not decoder_name:
            raise ValueError(
                "decoder_name must identify a caller-owned decoder and version"
            )
        if max_pixels < 1:
            raise ValueError("max_pixels must be positive")
        sources = tuple(sample.source for sample in normalized)
        if len(set(sources)) != len(sources):
            raise ValueError("raster sources must be unique")
        if any(source not in labels for source in sources):
            raise ValueError("raster labels must cover every sample source")
        self.dataset = normalized
        self.labels_by_source = dict(labels)
        self.decoder = decoder
        self.decoder_name = decoder_name.strip() if decoder_name else None
        self.max_pixels = max_pixels

    @property
    def dataset_hash(self) -> str:
        """The raster identity and label-sidecar digest."""
        identities = (
            f"{sample.source}\0{sample.file_hash}\0"
            f"{sample.nodata_value}\0{sample.nodata_fraction}\0"
            f"{self.labels_by_source[sample.source]}\0"
            f"decoder={self.decoder_name or 'metadata-only'}\0"
            f"max_pixels={self.max_pixels}"
            for sample in self.dataset
        )
        return curation.dataset_hash(identities)

    @property
    def samples(self) -> tuple[geospatial.RasterSample, ...]:
        """The raster samples in the dataset."""
        return self.dataset

    def curate(self) -> curation.ModalityCurationReport:
        """Builds raster shape, CRS, transform, and label evidence.

        Returns:
            The curation report, with decoded pixel statistics if a decoder is
            set (otherwise a metadata-only marker) and warnings for mixed
            shapes, CRS or data types and for unknown nodata handling.
        """
        dimensions = sorted(
            {
                (sample.width, sample.height, sample.bands)
                for sample in self.dataset
            }
        )
        crs_values = sorted({sample.crs for sample in self.dataset})
        dtypes = sorted({sample.dtype for sample in self.dataset})
        nodata_profile: dict[str, object] = {
            "statuses": sorted(
                {sample.nodata_status for sample in self.dataset}
            ),
            "declared_values": sorted(
                {
                    sample.nodata_value
                    for sample in self.dataset
                    if sample.nodata_value is not None
                }
            ),
            "fraction_min": min(
                (
                    sample.nodata_fraction
                    for sample in self.dataset
                    if sample.nodata_fraction is not None
                ),
                default=None,
            ),
            "fraction_max": max(
                (
                    sample.nodata_fraction
                    for sample in self.dataset
                    if sample.nodata_fraction is not None
                ),
                default=None,
            ),
        }
        profile = {
            "dimensions": [list(value) for value in dimensions],
            "crs": crs_values,
            "dtypes": dtypes,
            "transform_lengths": sorted(
                {len(sample.transform) for sample in self.dataset}
            ),
            "total_bytes": sum(sample.byte_size for sample in self.dataset),
            "nodata": nodata_profile,
            "label_counts": curation.label_counts(
                self.labels_by_source.values()
            ),
        }
        if self.decoder is None:
            profile["pixels"] = {
                "status": "metadata_only",
                "decoder": None,
                "max_pixels": self.max_pixels,
            }
        else:
            decoded = [
                sample.decode(self.decoder, max_pixels=self.max_pixels)
                for sample in self.dataset
            ]
            profile["pixels"] = {
                "status": "decoded",
                "decoder": self.decoder_name,
                "max_pixels": self.max_pixels,
                "shapes": [
                    list(shape)
                    for shape in sorted(
                        {tuple(values.shape) for values in decoded}
                    )
                ],
                "minimum": min(float(values.min()) for values in decoded),
                "maximum": max(float(values.max()) for values in decoded),
                "measured_nodata_fraction": sorted(
                    value
                    for value in (
                        sample.measured_nodata_fraction(
                            self.decoder, max_pixels=self.max_pixels
                        )
                        for sample in self.dataset
                        if sample.nodata_value is not None
                    )
                    if value is not None
                ),
            }
        quality_findings: list[data.CurationFinding] = []
        if len(crs_values) > 1 or len(dimensions) > 1 or len(dtypes) > 1:
            quality_findings.append(
                curation.build_finding(
                    "quality.raster-shape-or-crs-drift",
                    data.FindingSeverity.WARNING,
                    {"dataset": self.dataset_hash},
                    {
                        "dimensions": profile["dimensions"],
                        "crs": crs_values,
                        "dtypes": dtypes,
                    },
                    "Record an explicit raster normalization policy before "
                    "promotion.",
                    False,
                )
            )
        if any(sample.nodata_status == "unknown" for sample in self.dataset):
            quality_findings.append(
                curation.build_finding(
                    "quality.raster-nodata-unknown",
                    data.FindingSeverity.WARNING,
                    {"dataset": self.dataset_hash},
                    {"statuses": nodata_profile["statuses"]},
                    "Declare or measure nodata handling before interpreting "
                    "raster values.",
                    False,
                )
            )
        return curation.ModalityCurationReport(
            modality="geospatial-raster",
            dataset_hash=self.dataset_hash,
            sample_count=len(self.dataset),
            profile=profile,
            quality_findings=tuple(quality_findings),
        )

    def split(self, seed: int) -> DatasetPartitions:
        """Splits by source identity because the core lacks raster decoding.

        Args:
            seed: Random seed for the shuffle.

        Returns:
            Train, validation and test partitions where validation and test each
            hold a fifth of the rasters (at least one), with the sources of each
            under `groups`.
        """
        generator = numpy.random.default_rng(seed)
        order = generator.permutation(len(self.dataset))
        test_count = max(1, len(self.dataset) // 5)
        validation_count = max(1, len(self.dataset) // 5)
        test_indices = {int(value) for value in order[:test_count]}
        validation_indices = {
            int(value)
            for value in order[test_count : test_count + validation_count]
        }
        train = tuple(
            sample
            for index, sample in enumerate(self.dataset)
            if index not in test_indices | validation_indices
        )
        validation = tuple(
            sample
            for index, sample in enumerate(self.dataset)
            if index in validation_indices
        )
        test = tuple(
            sample
            for index, sample in enumerate(self.dataset)
            if index in test_indices
        )
        return DatasetPartitions(
            train,
            validation,
            test,
            {
                "train": tuple(sample.source for sample in train),
                "validation": tuple(sample.source for sample in validation),
                "test": tuple(sample.source for sample in test),
            },
        )

    def features(
        self, samples: Sequence[geospatial.RasterSample]
    ) -> FeatureBatch:
        """Prepares raster metadata or bounded decoded pixel features.

        Args:
            samples: Rasters to featurize.

        Returns:
            Batch with one row per raster, identified by its source: per-band
            pixel statistics if a decoder is set, otherwise width, height, band
            count, byte size and six transform values.

        Raises:
            GeospatialValidationError: If decoding is on and the rasters do not
                share one band count (an empty sequence counts as not sharing
                one).
        """
        if self.decoder is not None:
            band_counts = {sample.bands for sample in samples}
            if len(band_counts) != 1:
                raise geospatial.GeospatialValidationError(
                    "decoded raster features require a consistent band count",
                    {"code": "pixel_band_mismatch"},
                )
            band_count = next(iter(band_counts))
            names = self.pixel_feature_names(band_count)
            rows = []
            for sample in samples:
                pixels = sample.decode(self.decoder, max_pixels=self.max_pixels)
                rows.append(self.pixel_feature_values(pixels, band_count))
            return FeatureBatch(
                tuple(sample.source for sample in samples),
                names,
                numpy.asarray(rows, dtype=numpy.float64),
            )
        names = (
            "width",
            "height",
            "bands",
            "byte_size",
            "transform_0",
            "transform_1",
            "transform_2",
            "transform_3",
            "transform_4",
            "transform_5",
        )
        rows = []
        for sample in samples:
            transform = tuple(sample.transform) + (0.0,) * 6
            rows.append(
                (
                    float(sample.width),
                    float(sample.height),
                    float(sample.bands),
                    float(sample.byte_size),
                    *transform[:6],
                )
            )
        return FeatureBatch(
            tuple(sample.source for sample in samples),
            names,
            numpy.asarray(rows),
        )

    @staticmethod
    def pixel_feature_names(band_count: int) -> tuple[str, ...]:
        """Names the bounded decoded-pixel summary features.

        Args:
            band_count: Number of raster bands.

        Returns:
            For each band in order, the names of its mean, standard deviation,
            minimum and maximum.

        Raises:
            ValueError: If `band_count` is below 1.
        """
        if band_count < 1:
            raise ValueError("band_count must be positive")
        return tuple(
            f"pixel_band_{band}_{statistic}"
            for band in range(band_count)
            for statistic in ("mean", "std", "min", "max")
        )

    @staticmethod
    def pixel_feature_values(
        pixels: numpy.ndarray, band_count: int
    ) -> tuple[float, ...]:
        """Computes finite per-band summary values for decoded pixels.

        Args:
            pixels: Decoded array of shape (rows, columns) for one band or
                (rows, columns, bands).
            band_count: Number of bands the raster declares.

        Returns:
            For each band in order, the mean, standard deviation, minimum and
            maximum of its values.

        Raises:
            GeospatialValidationError: If the pixel shape does not match the
                declared band count.
        """
        values = numpy.asarray(pixels)
        if values.ndim == 2:
            if band_count != 1:
                raise geospatial.GeospatialValidationError(
                    "single-band decoded pixels must declare one band",
                    {"code": "pixel_band_mismatch"},
                )
            channels = values[:, :, None]
        elif values.ndim == 3 and values.shape[2] == band_count:
            channels = values
        else:
            raise geospatial.GeospatialValidationError(
                "decoded pixels do not match the declared band count",
                {"code": "pixel_band_mismatch"},
            )
        return tuple(
            statistic_value
            for band in range(band_count)
            for statistic_value in (
                float(channels[:, :, band].mean()),
                float(channels[:, :, band].std()),
                float(channels[:, :, band].min()),
                float(channels[:, :, band].max()),
            )
        )

    def sample_id(self, sample: geospatial.RasterSample) -> str:
        """Gives the canonical raster source identity.

        Args:
            sample: Raster sample.

        Returns:
            The raster's source.
        """
        return sample.source

    def label(self, sample: geospatial.RasterSample) -> object | None:
        """Gives the caller-owned raster label.

        Args:
            sample: Raster sample.

        Returns:
            The label recorded for the raster's source.
        """
        return self.labels_by_source[sample.source]

    def modality_specific_explanation(
        self,
        sample: Any,
        model: LogisticRegression,
        feature_names: Sequence[str],
        coefficients: numpy.ndarray,
    ) -> dict[str, object]:
        """Builds bounded occlusion evidence for decoded raster pixels.

        Args:
            sample: Raster sample to explain.
            model: Fitted classifier.
            feature_names: Names of the model's features; they must be the
                raster's pixel feature names.
            coefficients: Linear coefficients, one per feature.

        Returns:
            Mapping with the ten pixel windows (a 4 x 4 grid) whose masking
            changes the decision score most (original minus masked), all
            windows with their spatial bounds, and limitations; or status
            "not_available" if no decoder is set.

        Raises:
            TypeError: If `sample` is not a `RasterSample`.
            ValueError: If the feature names or coefficients do not match the
                raster's pixel features.
        """
        if not isinstance(sample, geospatial.RasterSample):
            raise TypeError("sample must be a RasterSample")
        if self.decoder is None:
            return {
                "status": "not_available",
                "modality": modalities.modality_value(self.modality),
                "reason": "an explicit named raster decoder is required",
                "limitations": [
                    "Metadata-only rasters cannot produce pixel-region "
                    "evidence.",
                    "Raw encoded-payload occlusion is outside this contract.",
                ],
            }
        pixels = sample.decode(self.decoder, max_pixels=self.max_pixels)
        original_values = numpy.asarray(
            self.pixel_feature_values(pixels, sample.bands), dtype=numpy.float64
        )
        if tuple(feature_names) != self.pixel_feature_names(sample.bands):
            raise ValueError("raster explanation feature schema does not match")
        if original_values.shape[0] != coefficients.shape[0]:
            raise ValueError(
                "raster explanation coefficients do not match features"
            )
        original_features = original_values.reshape(1, -1)
        decision = numpy.asarray(model.decision_function(original_features))
        if decision.ndim == 1:
            selected_class = 0
            original_score = float(decision[0])
        else:
            predicted = model.predict(original_features)[0]
            selected_class = int(
                numpy.flatnonzero(model.classes_ == predicted)[0]
            )
            original_score = float(decision[0, selected_class])
        height, width = pixels.shape[:2]
        row_step = max(1, int(numpy.ceil(height / 4)))
        column_step = max(1, int(numpy.ceil(width / 4)))
        terms: list[tuple[str, float]] = []
        windows: list[dict[str, object]] = []
        for row in range(0, height, row_step):
            for column in range(0, width, column_step):
                end_row = min(row + row_step, height)
                end_column = min(column + column_step, width)
                masked_pixels = pixels.copy()
                if masked_pixels.ndim == 2:
                    masked_pixels[row:end_row, column:end_column] = 0.0
                else:
                    masked_pixels[row:end_row, column:end_column, :] = 0.0
                masked_features = numpy.asarray(
                    self.pixel_feature_values(masked_pixels, sample.bands),
                    dtype=numpy.float64,
                ).reshape(1, -1)
                masked_decision = numpy.asarray(
                    model.decision_function(masked_features)
                )
                if masked_decision.ndim == 1:
                    masked_score = float(masked_decision[0])
                else:
                    masked_score = float(masked_decision[0, selected_class])
                region_name = f"region[{row}:{end_row},{column}:{end_column}]"
                score_delta = original_score - masked_score
                spatial_bounds = sample.pixel_window_bounds(
                    row, end_row, column, end_column
                )
                terms.append((region_name, score_delta))
                windows.append(
                    {
                        "pixel_window": [row, end_row, column, end_column],
                        "spatial_bounds": (
                            list(spatial_bounds)
                            if spatial_bounds is not None
                            else None
                        ),
                        "score_delta": float(score_delta),
                    }
                )
        selected = sorted(
            terms,
            key=lambda item: (-abs(item[1]), item[0]),
        )[:10]
        return {
            "status": "available",
            "method": "decoded_pixel_region_occlusion",
            "source": sample.source,
            "decoder": self.decoder_name,
            "decoded_dimensions": [int(width), int(height), int(sample.bands)],
            "crs": sample.crs,
            "window_count": len(terms),
            "terms": [[name, float(value)] for name, value in selected],
            "windows": windows,
            "baseline": "zero-valued decoded raster pixels",
            "feature_space_only": False,
            "limitations": [
                "Occlusion is applied after the caller-owned decoder and is "
                "not evidence about encoded bytes or decoder failure modes.",
                "Regions are pixel windows, not semantic objects or validated "
                "geospatial causal explanations.",
                "Score changes are associational and not causal.",
            ],
        }

    def monitor(self, observed: ModalityAdapter) -> dict[str, object]:
        """Builds raster profile drift evidence.

        Args:
            observed: Adapter over the newly observed rasters.

        Returns:
            Mapping with both dataset hashes and whether the profile differs.

        Raises:
            TypeError: If `observed` is not a `RasterAdapter`.
        """
        if not isinstance(observed, RasterAdapter):
            raise TypeError("observed adapter must be a RasterAdapter")
        reference = self.curate()
        current = observed.curate()
        return {
            "reference_hash": reference.dataset_hash,
            "observed_hash": current.dataset_hash,
            "profile_changed": current.profile != reference.profile,
        }

    def modality_notes(self) -> tuple[str, ...]:
        """Describes raster lifecycle limits.

        Returns:
            Plain statements of the representation in use and what it does not
            cover.
        """
        representation = (
            "The baseline uses caller-decoded, bounded pixel summary "
            "features with an explicit decoder policy."
            if self.decoder is not None
            else "The baseline uses raster metadata and a caller-owned label "
            "sidecar; it does not decode pixels."
        )
        return (
            representation,
            "Raster transforms, reprojection, window sampling, and spatial "
            "joins require an explicit codec policy; nodata metadata is "
            "caller-owned unless measured through the decoder.",
        )


@dataclasses.dataclass
class LifecycleContext:
    """Mutable products shared by the fifteen lifecycle stages.

    Attributes:
        adapter: Modality adapter the lifecycle runs on.
        package_path: Directory the model package is written to.
        signing_key: Key that signs the package and report.
        seed: Random seed of the run.
        observed_adapter: Adapter over the data used for monitoring.
        manifest: Modality manifest, if the caller supplied one.
        require_independent_ground_truth: Whether explanation quality needs
            independently authored ground truth.
        resume_replay: Whether the run replays stages already checkpointed.
        stage_records: Result of each completed stage.
        curation_report: Curation report produced by the curation stage.
        partitions: Train, validation and test partitions, once split.
        feature_batches: Feature batch per partition name.
        model: Fitted classifier, once trained.
        evaluation: Held-out evaluation evidence.
        performance_observations: Measured operations, one per kind.
        data_explanation: Data explainability evidence.
        model_explanation: Model explainability evidence.
        package: Identity of the written package, once packaged.
        inference: Inference smoke-test evidence.
        monitoring: Drift-monitoring evidence.
        documentation: Documentation evidence.
        recovery: Recovery and resource evidence.
    """

    adapter: ModalityAdapter
    package_path: Path
    signing_key: str | bytes
    seed: int
    observed_adapter: ModalityAdapter
    manifest: data.ModalityManifest | None = None
    require_independent_ground_truth: bool = False
    resume_replay: bool = False
    stage_records: dict[modalities.LifecycleStage, dict[str, object]] = (
        dataclasses.field(default_factory=dict)
    )
    curation_report: Any = None
    partitions: DatasetPartitions | None = None
    feature_batches: dict[str, FeatureBatch] = dataclasses.field(
        default_factory=dict
    )
    model: LogisticRegression | None = None
    evaluation: dict[str, object] = dataclasses.field(default_factory=dict)
    performance_observations: dict[
        performance.PerformanceOperation,
        performance.PerformanceObservation,
    ] = dataclasses.field(default_factory=dict)
    data_explanation: dict[str, object] = dataclasses.field(
        default_factory=dict
    )
    model_explanation: dict[str, object] = dataclasses.field(
        default_factory=dict
    )
    package: LifecyclePackage | None = None
    inference: dict[str, object] = dataclasses.field(default_factory=dict)
    monitoring: dict[str, object] = dataclasses.field(default_factory=dict)
    documentation: dict[str, object] = dataclasses.field(default_factory=dict)
    recovery: dict[str, object] = dataclasses.field(default_factory=dict)

    def record(
        self,
        stage: modalities.LifecycleStage,
        payload: dict[str, object],
    ) -> None:
        """Records one stage result for the final evidence index.

        Args:
            stage: Stage that produced the result.
            payload: JSON-native result of the stage.
        """
        self.stage_records[stage] = payload

    def record_performance(
        self,
        operation: performance.PerformanceOperation,
        elapsed_seconds: float,
        sample_count: int,
    ) -> None:
        """Stores one measured operation with machine and timestamp context.

        The observation also carries the process's peak memory, the platform
        and Python version, and a UTC timestamp. It replaces any earlier
        observation of the same operation.

        Args:
            operation: Operation that was measured.
            elapsed_seconds: Wall-clock duration of the operation.
            sample_count: Number of samples the operation processed.
        """
        self.performance_observations[operation] = (
            performance.PerformanceObservation(
                operation=operation,
                elapsed_seconds=elapsed_seconds,
                peak_memory_gb=limits.peak_memory_gb(),
                sample_count=sample_count,
                machine=platform.platform(),
                observed_at=events.utc_timestamp(),
                software=(
                    f"{platform.python_implementation()} "
                    f"{platform.python_version()}"
                ),
            )
        )


class ModalityLifecycleStage(abc.ABC):
    """One polymorphic stage in a modality lifecycle.

    Attributes:
        stage: Lifecycle stage this class implements.
        performance_operation: Operation whose duration is measured for this
            stage, or None if it is not timed.
    """

    stage: ClassVar[modalities.LifecycleStage]
    performance_operation: ClassVar[performance.PerformanceOperation | None] = (
        None
    )

    @abc.abstractmethod
    def execute(self, context: LifecycleContext) -> None:
        """Executes the stage and records its evidence.

        Args:
            context: Shared lifecycle state, updated in place.
        """


class IngestionStage(ModalityLifecycleStage):
    """Confirm source identity and sample cardinality."""

    stage = modalities.LifecycleStage.INGESTION
    performance_operation = performance.PerformanceOperation.INGEST

    def execute(self, context: LifecycleContext) -> None:
        """Records immutable source identity.

        Args:
            context: Shared lifecycle state, updated in place.

        Raises:
            ValueError: If the dataset has no samples.
        """
        if not context.adapter.samples:
            raise ValueError("modality dataset must not be empty")
        context.record(
            self.stage,
            {
                "dataset_hash": context.adapter.dataset_hash,
                "sample_count": len(context.adapter.samples),
            },
        )


class ValidationStage(ModalityLifecycleStage):
    """Reject missing labels and malformed feature prerequisites."""

    stage = modalities.LifecycleStage.VALIDATION

    def execute(self, context: LifecycleContext) -> None:
        """Validates labels and dataset-level curation state.

        Args:
            context: Shared lifecycle state, updated in place.

        Raises:
            ValueError: If any sample lacks a label or fewer than two classes
                are present.
        """
        labels = context.adapter.labels(context.adapter.samples)
        if any(label is None for label in labels):
            raise ValueError("complete modality lifecycle requires labels")
        if len(set(labels)) < 2:
            raise ValueError("classification lifecycle requires two labels")
        context.record(
            self.stage,
            {"label_count": len(labels), "class_count": len(set(labels))},
        )


class ProfilingStage(ModalityLifecycleStage):
    """Capture modality-specific profile evidence."""

    stage = modalities.LifecycleStage.PROFILING

    def execute(self, context: LifecycleContext) -> None:
        """Persists the curation/profile object in lifecycle state.

        Args:
            context: Shared lifecycle state, updated in place.
        """
        context.curation_report = context.adapter.curate()
        context.record(
            self.stage, {"profile": context.curation_report.to_dict()}
        )


class CurationStage(ModalityLifecycleStage):
    """Block unresolved data-quality findings before representation."""

    stage = modalities.LifecycleStage.CURATION

    def execute(self, context: LifecycleContext) -> None:
        """Requires a release-ready curation report.

        Args:
            context: Shared lifecycle state, updated in place.

        Raises:
            ValueError: If there is no curation report or it has unresolved
                blocking findings.
        """
        report = context.curation_report
        if report is None or not report.release_ready:
            findings = (
                []
                if report is None
                else [
                    finding.finding_id
                    for finding in report.unresolved_blocking_findings
                ]
            )
            raise ValueError(
                f"unresolved blocking curation findings: {findings}"
            )
        context.record(self.stage, {"release_ready": True, "unresolved": []})


class SplittingStage(ModalityLifecycleStage):
    """Create leakage-aware train, validation, and test partitions."""

    stage = modalities.LifecycleStage.SPLITTING

    def execute(self, context: LifecycleContext) -> None:
        """Splits and rejects overlapping sample identities.

        Args:
            context: Shared lifecycle state, updated in place.

        Raises:
            ValueError: If the train or test partition is empty or two
                partitions share a group.
        """
        partitions = context.adapter.split(context.seed)
        if not partitions.train or not partitions.test:
            raise ValueError(
                "lifecycle split requires non-empty train and test partitions"
            )
        groups = [
            set(partitions.groups.get(name, ()))
            for name in ("train", "validation", "test")
        ]
        if any(
            left & right
            for index, left in enumerate(groups)
            for right in groups[index + 1 :]
        ):
            raise ValueError("lifecycle split contains overlapping groups")
        context.partitions = partitions
        context.record(
            self.stage,
            {
                "counts": {
                    "train": len(partitions.train),
                    "validation": len(partitions.validation),
                    "test": len(partitions.test),
                },
                "groups": partitions.groups,
            },
        )


class FeaturePreparationStage(ModalityLifecycleStage):
    """Prepare partition-local finite feature matrices."""

    stage = modalities.LifecycleStage.FEATURE_PREPARATION

    def execute(self, context: LifecycleContext) -> None:
        """Prepares and aligns features without fitting on held-out rows.

        Args:
            context: Shared lifecycle state, updated in place.

        Raises:
            RuntimeError: If the data has not been split yet.
            ValueError: If the partitions' feature names differ.
        """
        partitions = context.partitions
        if partitions is None:
            raise RuntimeError("splitting must precede feature preparation")
        batches = {
            "train": context.adapter.features(partitions.train),
            "validation": context.adapter.features(partitions.validation),
            "test": context.adapter.features(partitions.test),
        }
        names = batches["train"].feature_names
        if any(batch.feature_names != names for batch in batches.values()):
            raise ValueError("partition feature schemas do not match")
        context.feature_batches = batches
        context.record(
            self.stage,
            {
                "feature_names": list(names),
                "shape": {
                    name: list(batch.values.shape)
                    for name, batch in batches.items()
                },
            },
        )


class ModellingStage(ModalityLifecycleStage):
    """Fit the bounded regularized linear baseline on training features."""

    stage = modalities.LifecycleStage.MODELLING
    performance_operation = performance.PerformanceOperation.FIT

    def execute(self, context: LifecycleContext) -> None:
        """Fits only on the training partition.

        Args:
            context: Shared lifecycle state, updated in place.
        """
        batch = context.feature_batches["train"]
        labels = context.adapter.labels(
            context.partitions.train if context.partitions else ()
        )
        model = LogisticRegression(max_iter=500, random_state=context.seed)
        model.fit(batch.values, labels)
        context.model = model
        context.record(
            self.stage,
            {
                "model_family": "regularized_linear_classifier",
                "classes": [str(value) for value in model.classes_],
            },
        )


class EvaluationStage(ModalityLifecycleStage):
    """Evaluate once on the untouched test partition."""

    stage = modalities.LifecycleStage.EVALUATION

    def execute(self, context: LifecycleContext) -> None:
        """Computes deterministic classification metrics and predictions.

        The metrics are computed on the test partition's feature batch and
        stored as `context.evaluation`, which is also recorded as the stage
        evidence.

        Args:
            context: Shared lifecycle state, updated in place.

        Raises:
            RuntimeError: If the model or partitions are missing.
        """
        model = context.model
        partitions = context.partitions
        if model is None or partitions is None:
            raise RuntimeError(
                "modelling and splitting must precede evaluation"
            )
        batch = context.feature_batches["test"]
        actual = context.adapter.labels(partitions.test)
        predicted = model.predict(batch.values)
        probabilities = model.predict_proba(batch.values)
        context.evaluation = {
            "sample_count": len(actual),
            "accuracy": float(accuracy_score(actual, predicted)),
            "macro_f1": float(f1_score(actual, predicted, average="macro")),
            "log_loss": float(
                log_loss(actual, probabilities, labels=list(model.classes_))
            ),
            "labels": [str(value) for value in model.classes_],
            "predictions": [str(value) for value in predicted],
        }
        context.record(self.stage, context.evaluation.copy())


class DataExplainabilityStage(ModalityLifecycleStage):
    """Explain source composition, split boundaries, and feature ranges."""

    stage = modalities.LifecycleStage.DATA_EXPLAINABILITY

    def execute(self, context: LifecycleContext) -> None:
        """Records data lineage and non-causal feature summaries.

        Args:
            context: Shared lifecycle state, updated in place.

        Raises:
            RuntimeError: If the data has not been split yet.
        """
        batch = context.feature_batches["train"]
        partitions = context.partitions
        if partitions is None:
            raise RuntimeError("splitting must precede data explanation")
        labels = context.adapter.labels(partitions.train)
        label_counts: dict[str, int] = {}
        for label in labels:
            key = str(label)
            label_counts[key] = label_counts.get(key, 0) + 1
        statistics: dict[str, dict[str, float | int]] = {}
        for index, name in enumerate(batch.feature_names):
            values = batch.values[:, index]
            statistics[name] = {
                "min": float(numpy.min(values)),
                "max": float(numpy.max(values)),
                "mean": float(numpy.mean(values)),
                "std": float(numpy.std(values)),
                "q05": float(numpy.quantile(values, 0.05)),
                "median": float(numpy.median(values)),
                "q95": float(numpy.quantile(values, 0.95)),
                "nonfinite_count": int(
                    numpy.count_nonzero(~numpy.isfinite(values))
                ),
            }
        partition_samples = {
            "train": partitions.train,
            "validation": partitions.validation,
            "test": partitions.test,
        }
        sample_lineage = [
            {
                "sample_id": context.adapter.sample_id(sample),
                "partition": partition,
            }
            for partition, samples in partition_samples.items()
            for sample in samples
        ]
        context.data_explanation = {
            "dataset_hash": context.adapter.dataset_hash,
            "sample_ids": list(batch.sample_ids),
            "sample_lineage": sample_lineage,
            "feature_ranges": {
                name: {
                    "min": float(batch.values[:, index].min()),
                    "max": float(batch.values[:, index].max()),
                }
                for index, name in enumerate(batch.feature_names)
            },
            "feature_statistics": statistics,
            "label_distribution": {
                "counts": dict(sorted(label_counts.items())),
                "sample_count": len(labels),
            },
            "partition_counts": {
                "train": len(partitions.train),
                "validation": len(partitions.validation),
                "test": len(partitions.test),
            },
            "split_groups": partitions.groups,
            "limitations": [
                "Feature summaries describe association and coverage, not "
                "causation.",
                *context.adapter.modality_notes(),
            ],
        }
        context.record(self.stage, context.data_explanation.copy())


class ModelExplainabilityStage(ModalityLifecycleStage):
    """Explain linear weights and test feature-space stability."""

    stage = modalities.LifecycleStage.MODEL_EXPLAINABILITY

    def execute(self, context: LifecycleContext) -> None:
        """Records signed coefficient contributions with limitations.

        Args:
            context: Shared lifecycle state, updated in place.

        Raises:
            RuntimeError: If the model, partitions or test features are missing,
                or no explanation report could be produced.
            ValueError: If the linear reference does not match the model's
                decision output, or independent ground truth is required and the
                suite is unavailable or failed.
        """
        model = context.model
        batch = context.feature_batches["test"]
        partitions = context.partitions
        if model is None or partitions is None or not len(batch.sample_ids):
            raise RuntimeError(
                "model, partitions, and test features are required"
            )
        strength = numpy.mean(numpy.abs(model.coef_), axis=0)
        global_terms = sorted(
            (
                (name, float(strength[index]))
                for index, name in enumerate(batch.feature_names)
            ),
            key=lambda item: (-item[1], item[0]),
        )[:10]
        decision_values = numpy.asarray(model.decision_function(batch.values))
        reference_values = batch.values @ model.coef_.T + model.intercept_
        if reference_values.ndim == 2 and reference_values.shape[1] == 1:
            reference_values = reference_values[:, 0]
        if decision_values.ndim == 2 and decision_values.shape[1] == 1:
            decision_values = decision_values[:, 0]
        if decision_values.shape != reference_values.shape:
            raise ValueError(
                "linear explanation reference shape does not match model "
                "decision output"
            )
        fidelity = quality.fidelity_score(
            decision_values.astype(float).reshape(-1).tolist(),
            reference_values.astype(float).reshape(-1).tolist(),
        )
        train_values = context.feature_batches["train"].values
        feature_scales = numpy.maximum(numpy.std(train_values, axis=0), 1.0)
        perturbation_size = feature_scales * 1e-6
        repeated_stabilities: list[float] = []
        local_report: dict[str, object] | None = None
        explanation_reports: list[dict[str, object]] = []
        perturbation_reports: list[dict[str, object]] = []
        coefficients = model.coef_[0]
        stability = 0.0
        evaluated_sample_count = min(
            len(batch.sample_ids), MAX_REPEATED_EXPLANATION_SAMPLES
        )
        for sample_index in range(evaluated_sample_count):
            sample_row = batch.values[sample_index]
            sample_decision = numpy.asarray(
                model.decision_function(sample_row[None])
            )
            sample_class = 0
            if sample_decision.ndim > 1:
                sample_class = int(numpy.argmax(sample_decision[0]))
            sample_coefficients = model.coef_[sample_class]
            sample_report = quality.linear_contribution_report(
                batch.sample_ids[sample_index],
                batch.feature_names,
                sample_row.tolist(),
                sample_coefficients.tolist(),
                top_k=min(10, len(batch.feature_names)),
            )
            sample_reports = [sample_report]
            for feature_index in range(len(batch.feature_names)):
                for direction in (-1.0, 1.0):
                    perturbed_row = sample_row.copy()
                    perturbed_row[feature_index] += (
                        direction * perturbation_size[feature_index]
                    )
                    sample_reports.append(
                        quality.linear_contribution_report(
                            batch.sample_ids[sample_index],
                            batch.feature_names,
                            perturbed_row.tolist(),
                            sample_coefficients.tolist(),
                            top_k=min(10, len(batch.feature_names)),
                        )
                    )
            explanation_reports.append(sample_report)
            sample_stability = quality.ranking_stability(
                tuple(sample_reports),
                top_k=min(10, len(batch.feature_names)),
            )
            repeated_stabilities.append(sample_stability)
            if sample_index == 0:
                coefficients = sample_coefficients
                local_report = sample_report
                perturbation_reports = sample_reports
                perturbation_stability = sample_stability
                stability_decision = sample_decision
                nonzero = numpy.flatnonzero(
                    numpy.abs(sample_row * sample_coefficients) > 1e-12
                )
                if len(nonzero):
                    perturbed = numpy.repeat(
                        sample_row[None, :], len(nonzero), axis=0
                    )
                    perturbed[numpy.arange(len(nonzero)), nonzero] = 0.0
                    perturbed_decisions = numpy.asarray(
                        model.decision_function(perturbed)
                    )
                    if perturbed_decisions.ndim == 1:
                        observed_change = (
                            perturbed_decisions - stability_decision[0]
                        )
                    else:
                        observed_change = (
                            perturbed_decisions[:, sample_class]
                            - stability_decision[0, sample_class]
                        )
                    expected_change = -(
                        sample_row[nonzero] * sample_coefficients[nonzero]
                    )
                    stability = float(
                        numpy.mean(
                            numpy.sign(observed_change)
                            == numpy.sign(expected_change)
                        )
                    )
        if local_report is None or not perturbation_reports:
            raise RuntimeError(
                "repeated explanation evaluation produced no report"
            )
        references = []
        for sample_index in range(evaluated_sample_count):
            ground_truth = context.adapter.explanation_ground_truth(
                partitions.test[sample_index], batch.feature_names
            )
            if ground_truth is not None:
                references.append(
                    (
                        ground_truth,
                        explanation_reports[sample_index],
                        batch.sample_ids[sample_index],
                    )
                )
        ground_truth_evidence = {
            **quality.evaluate_ground_truth_suite(
                evaluated_sample_count, references
            ).to_dict(),
            "feature_space_only": True,
            "limitations": [
                "Feature-space stability and fidelity are not independent "
                "semantic validation.",
                "Raw-modality ground truth remains caller-owned evidence.",
            ],
        }
        if context.require_independent_ground_truth and (
            ground_truth_evidence["status"] != "available"
            or ground_truth_evidence["passed"] is not True
        ):
            raise ValueError(
                "independent explanation ground-truth suite is required "
                "before packaging"
            )
        repeated_stability = min(repeated_stabilities)
        quality_evaluation = quality.evaluate(
            method="linear_reference_zero_ablation_and_perturbation",
            sample_count=len(batch.sample_ids),
            thresholds=quality.QualityThresholds(
                stability_minimum=0.9,
                fidelity_minimum=0.999999,
            ),
            stability=repeated_stability,
            fidelity=fidelity,
        )
        context.model_explanation = {
            "method": "linear_coefficient_and_local_contribution",
            "model_family": "regularized_linear_classifier",
            "global_terms": [list(value) for value in global_terms],
            "local": {
                "sample_id": batch.sample_ids[0],
                "terms": local_report["terms"],
            },
            "zero_ablation_direction_consistency": stability,
            "perturbation_stability": {
                "score": perturbation_stability,
                "report_count": len(perturbation_reports),
                "relative_step": 1e-6,
                "feature_space_only": True,
            },
            "repeated_perturbation": {
                "score": repeated_stability,
                "sample_count": len(batch.sample_ids),
                "evaluated_sample_count": evaluated_sample_count,
                "minimum_across_samples": repeated_stability,
                "threshold": 0.9,
                "feature_space_only": True,
                "bounded_by": MAX_REPEATED_EXPLANATION_SAMPLES,
                "limitations": [
                    "The suite perturbs the declared feature representation; "
                    "it does not establish raw-modality saliency.",
                    "The sample bound is a lifecycle resource limit, not a "
                    "production-scale stress guarantee.",
                ],
            },
            "independent_ground_truth": ground_truth_evidence,
            "modality_specific": context.adapter.modality_specific_explanation(
                partitions.test[0],
                model,
                batch.feature_names,
                coefficients,
            ),
            "quality": quality_evaluation.to_dict(),
            "limitations": [
                "Associational explanation only.",
                "Input representation and linear approximation constrain "
                "interpretation.",
                "Quality checks validate the feature-space linear baseline; "
                "they do not validate raw-modality causality.",
            ],
        }
        context.record(self.stage, context.model_explanation.copy())


class PackagingStage(ModalityLifecycleStage):
    """Writes and signs a digest-bound model package atomically."""

    stage = modalities.LifecycleStage.PACKAGING
    performance_operation = performance.PerformanceOperation.PACKAGE

    def execute(self, context: LifecycleContext) -> None:
        """Persists the estimator and signed manifest.

        When replaying a checkpointed run onto an existing package, the package
        is verified against the replayed model instead of being rewritten.

        Args:
            context: Shared lifecycle state, updated in place.

        Raises:
            RuntimeError: If there is no fitted model.
            ValueError: If an existing package does not match the resumed run.
            FileExistsError: If the package path already exists and the run is
                not replaying a checkpoint.
        """
        model = context.model
        if model is None:
            raise RuntimeError("model is required before packaging")
        target = context.package_path
        if context.resume_replay and target.exists():
            predictor = ModalityPredictor.from_package(
                context.adapter, target, context.signing_key
            )
            expected_features = context.feature_batches["train"].feature_names
            if predictor.training_dataset_hash != context.adapter.dataset_hash:
                raise ValueError(
                    "checkpoint package dataset identity does not match "
                    "the resumed adapter"
                )
            if predictor.feature_names != expected_features:
                raise ValueError(
                    "checkpoint package feature schema does not match "
                    "the resumed adapter"
                )
            if not (
                numpy.array_equal(predictor.model.classes_, model.classes_)
                and numpy.array_equal(predictor.model.coef_, model.coef_)
                and numpy.array_equal(
                    predictor.model.intercept_, model.intercept_
                )
            ):
                raise ValueError(
                    "checkpoint package model does not match deterministic "
                    "stage replay"
                )
            context.package = predictor.package
            context.record(self.stage, context.package.to_dict())
            return
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent)
        )
        try:
            model_path = temporary / "model.joblib"
            joblib.dump(model, model_path, compress=3)
            sync_file(model_path)
            unsigned = {
                "schema_version": 1,
                "modality": modalities.modality_value(context.adapter.modality),
                "dataset_hash": context.adapter.dataset_hash,
                "feature_names": list(
                    context.feature_batches["train"].feature_names
                ),
                "model_sha256": file_digest(model_path),
                "evaluation": {
                    key: value
                    for key, value in context.evaluation.items()
                    if key != "predictions"
                },
            }
            signature = attestation.sign_payload(unsigned, context.signing_key)
            manifest = {
                **unsigned,
                "signature": {
                    "algorithm": attestation.SIGNATURE_ALGORITHM,
                    "value": signature,
                },
            }
            manifest_path = temporary / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            sync_file(manifest_path)
            sync_directory(temporary)
            os.replace(temporary, target)
            sync_directory(target.parent)
            context.package = LifecyclePackage(
                str(target), str(unsigned["model_sha256"]), signature
            )
            context.record(self.stage, context.package.to_dict())
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise


class InferenceStage(ModalityLifecycleStage):
    """Verify the package before loading and reproducing predictions."""

    stage = modalities.LifecycleStage.INFERENCE

    def execute(self, context: LifecycleContext) -> None:
        """Loads trusted bytes and compares predictions with the fitted model.

        Args:
            context: Shared lifecycle state, updated in place.

        Raises:
            RuntimeError: If the package or partitions are missing.
            ValueError: If the packaged model's predictions differ from the
                evaluation predictions.
        """
        if context.package is None or context.partitions is None:
            raise RuntimeError(
                "package and partitions are required before inference"
            )
        started = time.perf_counter()
        loaded = load_package(context.package.path, context.signing_key)
        context.record_performance(
            performance.PerformanceOperation.LOAD,
            time.perf_counter() - started,
            len(context.feature_batches["test"].sample_ids),
        )
        started = time.perf_counter()
        predictions = loaded.predict(context.feature_batches["test"].values)
        context.record_performance(
            performance.PerformanceOperation.INFERENCE,
            time.perf_counter() - started,
            len(context.feature_batches["test"].sample_ids),
        )
        expected = context.evaluation["predictions"]
        actual = [str(value) for value in predictions]
        if actual != expected:
            raise ValueError(
                "packaged inference does not reproduce evaluation predictions"
            )
        context.inference = {
            "package_verified": True,
            "prediction_count": len(actual),
            "predictions_match": True,
        }
        context.record(self.stage, context.inference.copy())


class MonitoringStage(ModalityLifecycleStage):
    """Emit reference-to-observed data drift evidence."""

    stage = modalities.LifecycleStage.MONITORING

    def execute(self, context: LifecycleContext) -> None:
        """Records modality-specific monitoring output.

        Args:
            context: Shared lifecycle state, updated in place.
        """
        context.monitoring = context.adapter.monitor(context.observed_adapter)
        context.record(self.stage, context.monitoring.copy())


class DocumentationStage(ModalityLifecycleStage):
    """Build a truthful dataset and model card for the baseline."""

    stage = modalities.LifecycleStage.DOCUMENTATION

    def execute(self, context: LifecycleContext) -> None:
        """Records support status, intended use, and limitations.

        Args:
            context: Shared lifecycle state, updated in place.

        Raises:
            RuntimeError: If the data has not been profiled yet.
        """
        if context.curation_report is None:
            raise RuntimeError("profiling must precede documentation")
        dataset_card = curation.ModalityDatasetCard.from_report(
            modalities.modality_value(context.adapter.modality),
            context.curation_report.to_dict(),
            context.adapter.modality_notes(),
            manifest=context.manifest,
        )
        context.documentation = {
            "modality": modalities.modality_value(context.adapter.modality),
            "status": (
                "extension-only"
                if isinstance(
                    context.adapter.modality, modalities.ExtensionModality
                )
                else "experimental"
            ),
            "intended_use": (
                "bounded lifecycle validation and baseline comparison"
            ),
            "dataset_hash": context.adapter.dataset_hash,
            "model_family": "regularized_linear_classifier",
            "dataset_card": dataset_card.to_dict(),
            "limitations": list(context.adapter.modality_notes()),
            "promotion_requirements": [
                "independent metric recomputation",
                "modality-specific explanation validation",
                "external serving and resource-fault evidence",
            ],
        }
        context.record(self.stage, context.documentation.copy())


class RecoveryAndResourceStage(ModalityLifecycleStage):
    """Prove bounded resource claims and reject a tampered package."""

    stage = modalities.LifecycleStage.RECOVERY_AND_RESOURCE_TESTS

    def execute(self, context: LifecycleContext) -> None:
        """Tampers with a copy of the package and verifies it is rejected.

        Args:
            context: Shared lifecycle state, updated in place.

        Raises:
            RuntimeError: If there is no package.
            ValueError: If a tampered copy of the package is not rejected.
        """
        if context.package is None:
            raise RuntimeError("package is required before recovery tests")
        temporary = Path(tempfile.mkdtemp(prefix="scmle-lifecycle-recovery-"))
        tampered = temporary / "package"
        try:
            shutil.copytree(context.package.path, tampered)
            manifest_path = tampered / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["dataset_hash"] = "tampered"
            manifest_path.write_text(
                json.dumps(manifest), encoding="utf-8", newline="\n"
            )
            rejected = False
            try:
                load_package(tampered, context.signing_key)
            except (TypeError, ValueError):
                rejected = True
            if not rejected:
                raise ValueError("tampered package was not rejected")
            context.recovery = {
                "tampered_manifest_rejected": True,
                "feature_rows": int(
                    context.feature_batches["train"].values.shape[0]
                ),
                "feature_columns": int(
                    context.feature_batches["train"].values.shape[1]
                ),
                "bounded": True,
            }
            context.record(self.stage, context.recovery.copy())
        finally:
            shutil.rmtree(temporary, ignore_errors=True)


class ModalityLifecycle:
    """Executes the fifteen shared lifecycle stages in deterministic order.

    Attributes:
        stages: The stage objects, in required lifecycle order.
    """

    stages: ClassVar[tuple[ModalityLifecycleStage, ...]] = (
        IngestionStage(),
        ValidationStage(),
        ProfilingStage(),
        CurationStage(),
        SplittingStage(),
        FeaturePreparationStage(),
        ModellingStage(),
        EvaluationStage(),
        DataExplainabilityStage(),
        ModelExplainabilityStage(),
        PackagingStage(),
        InferenceStage(),
        MonitoringStage(),
        DocumentationStage(),
        RecoveryAndResourceStage(),
    )

    def run(
        self,
        adapter: ModalityAdapter,
        package_path: str | Path,
        signing_key: str | bytes,
        seed: int = 0,
        observed_adapter: ModalityAdapter | None = None,
        report_path: str | Path | None = None,
        performance_budget: performance.PerformanceBudget | None = None,
        performance_baseline: performance.PerformanceBaseline | None = None,
        manifest: data.ModalityManifest | None = None,
        require_independent_ground_truth: bool = False,
        cancellation_check: Callable[[], bool] | None = None,
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
    ) -> LifecycleReport:
        """Runs every stage and optionally persists its evidence report.

        A supplied manifest is checked against the adapter before any stage
        executes. This keeps source, task, privacy, and lineage claims bound
        to the exact modality content used by the lifecycle.

        If `cancellation_check` returns True at a stage boundary, artifacts
        created by this run are removed and `LifecycleCancelledError` is
        raised.

        Args:
            adapter: Adapter for the reference dataset.
            package_path: Directory the model package is written to; it must not
                exist unless a checkpoint replay is resuming onto it.
            signing_key: Key that signs the package, report and checkpoints.
            seed: Random seed for the split and the model.
            observed_adapter: Adapter over the data to monitor against the
                reference; defaults to `adapter`. Its modality must match.
            report_path: File the signed report is written to, if any.
            performance_budget: Limits the measured operations must meet, if
                any.
            performance_baseline: Earlier run the measurements are compared
                with, if any.
            manifest: Modality manifest to check against the adapter, if any.
            require_independent_ground_truth: Whether to fail unless the
                independent explanation ground-truth suite is available and
                passes.
            cancellation_check: Callable polled at stage boundaries that returns
                True to cancel the run.
            checkpoint_path: File where signed stage checkpoints are kept, if
                any; requires `report_path` and must differ from it.
            resume: Whether to resume from an existing checkpoint at
                `checkpoint_path`, replaying its completed stages and verifying
                their digests.

        Returns:
            The signed lifecycle report. If a completed checkpoint is resumed,
            the previously written report after verification.

        Raises:
            TypeError: If `require_independent_ground_truth` or `resume` is
                not a bool, or `cancellation_check` is not callable.
            ValueError: If the option combination is invalid, the observed
                modality or manifest does not match the adapter, a checkpoint or
                report does not match the resumed run, a stage fails, or the
                performance budget or baseline is violated.
            RuntimeError: If the stage order is invalid or packaging produced no
                package.
            FileExistsError: If the checkpoint exists and `resume` is False.
            FileNotFoundError: If `resume` is True and the checkpoint does not
                exist.
        """
        if not isinstance(require_independent_ground_truth, bool):
            raise TypeError("require_independent_ground_truth must be bool")
        if cancellation_check is not None and not callable(cancellation_check):
            raise TypeError("cancellation_check must be callable")
        if not isinstance(resume, bool):
            raise TypeError("resume must be bool")
        package_target = Path(package_path)
        report_target = Path(report_path) if report_path is not None else None
        checkpoint_target = (
            Path(checkpoint_path) if checkpoint_path is not None else None
        )
        if resume and checkpoint_target is None:
            raise ValueError("resume requires checkpoint_path")
        if checkpoint_target is not None and report_target is None:
            raise ValueError(
                "checkpoint_path requires report_path so completion can be "
                "verified"
            )
        if checkpoint_target is not None and checkpoint_target == report_target:
            raise ValueError("checkpoint_path and report_path must differ")
        package_preexisting = package_target.exists()
        report_preexisting = (
            report_target.exists() if report_target is not None else False
        )

        def ensure_not_cancelled() -> None:
            """Stops before publication and removes this run's artifacts.

            Does nothing when no check was given or it reports no
            cancellation. On cancellation it deletes the package directory and
            the report file unless they existed before the run.

            Raises:
                LifecycleCancelledError: If `cancellation_check` reports
                    cancellation.
            """
            if cancellation_check is None or not cancellation_check():
                return
            if not package_preexisting:
                shutil.rmtree(package_target, ignore_errors=True)
            if report_target is not None and not report_preexisting:
                report_target.unlink(missing_ok=True)
            raise LifecycleCancelledError(
                "modality lifecycle cancelled at a stage boundary"
            )

        ensure_not_cancelled()
        observed = observed_adapter or adapter
        if observed.modality != adapter.modality:
            raise ValueError(
                "observed adapter modality must match reference modality"
            )
        if manifest is not None:
            if manifest.modality != modalities.modality_value(adapter.modality):
                raise ValueError("manifest modality must match adapter")
            if manifest.content_hash != adapter.dataset_hash:
                raise ValueError("manifest content_hash must match adapter")
            if manifest.sample_count != len(adapter.samples):
                raise ValueError("manifest sample_count must match adapter")
        manifest_digest = (
            attestation.canonical_digest(manifest.to_dict())
            if manifest is not None
            else None
        )
        stage_names = tuple(stage.stage.value for stage in self.stages)
        required_stage_names = tuple(
            stage.value for stage in modalities.REQUIRED_LIFECYCLE_STAGES
        )
        if stage_names != required_stage_names:
            raise RuntimeError("modality lifecycle stage order is invalid")
        resume_count = 0
        resume_digests: tuple[str, ...] = ()
        checkpoint: LifecycleCheckpoint | None = None

        if checkpoint_target is not None:
            if checkpoint_target.exists() and not resume:
                raise FileExistsError(checkpoint_target)
            if resume and not checkpoint_target.exists():
                raise FileNotFoundError(checkpoint_target)
            if resume:
                checkpoint = LifecycleCheckpoint.read_verified(
                    checkpoint_target, signing_key
                )
                expected_identity = {
                    "modality": modalities.modality_value(adapter.modality),
                    "dataset_hash": adapter.dataset_hash,
                    "observed_dataset_hash": observed.dataset_hash,
                    "package_path": str(package_target),
                    "report_path": str(report_target),
                    "seed": seed,
                    "require_independent_ground_truth": (
                        require_independent_ground_truth
                    ),
                    "manifest_digest": manifest_digest,
                }
                actual_identity = {
                    "modality": checkpoint.modality,
                    "dataset_hash": checkpoint.dataset_hash,
                    "observed_dataset_hash": checkpoint.observed_dataset_hash,
                    "package_path": checkpoint.package_path,
                    "report_path": checkpoint.report_path,
                    "seed": checkpoint.seed,
                    "require_independent_ground_truth": (
                        checkpoint.require_independent_ground_truth
                    ),
                    "manifest_digest": checkpoint.manifest_digest,
                }
                if actual_identity != expected_identity:
                    raise ValueError(
                        "lifecycle checkpoint identity does not match the "
                        "requested resume"
                    )
                if (
                    checkpoint.completed_stages
                    != stage_names[: len(checkpoint.completed_stages)]
                ):
                    raise ValueError(
                        "lifecycle checkpoint stage order does not match "
                        "the current lifecycle"
                    )
                if checkpoint.status == "completed":
                    if report_target is None or not report_target.is_file():
                        raise ValueError(
                            "completed lifecycle checkpoint has no report"
                        )
                    completed_report = LifecycleReport.read_verified(
                        report_target, signing_key
                    )
                    if (
                        completed_report.modality
                        != modalities.modality_value(adapter.modality)
                        or completed_report.dataset_hash != adapter.dataset_hash
                        or Path(completed_report.package.path) != package_target
                    ):
                        raise ValueError(
                            "completed lifecycle report does not match "
                            "the requested resume"
                        )
                    ModalityPredictor.from_package(
                        adapter,
                        completed_report.package.path,
                        signing_key,
                        lifecycle_report_path=report_target,
                    )
                    return completed_report
                resume_count = len(checkpoint.completed_stages)
                resume_digests = checkpoint.stage_digests

        def persist_checkpoint(
            completed_stages: Sequence[str],
            stage_digests: Sequence[str],
            status: Literal["running", "completed"],
        ) -> None:
            """Atomically persists a signed stage boundary.

            Args:
                completed_stages: Values of the stages finished so far.
                stage_digests: Digest of each finished stage's evidence.
                status: "running" or "completed".
            """
            if checkpoint_target is None:
                return
            record = LifecycleCheckpoint(
                modality=modalities.modality_value(adapter.modality),
                dataset_hash=adapter.dataset_hash,
                observed_dataset_hash=observed.dataset_hash,
                package_path=str(package_target),
                report_path=str(report_target),
                seed=seed,
                completed_stages=tuple(completed_stages),
                stage_digests=tuple(stage_digests),
                require_independent_ground_truth=(
                    require_independent_ground_truth
                ),
                manifest_digest=manifest_digest,
                status=status,
            ).sign(signing_key)
            durability.write_atomically(
                checkpoint_target,
                (
                    json.dumps(record.to_dict(), indent=2, sort_keys=True)
                    + "\n"
                ).encode("utf-8"),
            )

        completed_stages = list(
            checkpoint.completed_stages if checkpoint is not None else ()
        )
        stage_digests = list(
            checkpoint.stage_digests if checkpoint is not None else ()
        )
        persist_checkpoint(completed_stages, stage_digests, "running")
        context = LifecycleContext(
            adapter=adapter,
            package_path=package_target,
            signing_key=signing_key,
            seed=seed,
            observed_adapter=observed,
            manifest=manifest,
            require_independent_ground_truth=require_independent_ground_truth,
        )
        for stage_index, stage in enumerate(self.stages):
            ensure_not_cancelled()
            context.resume_replay = stage_index < resume_count
            started = time.perf_counter()
            stage.execute(context)
            operation = stage.performance_operation
            if operation is not None:
                if operation is performance.PerformanceOperation.INGEST:
                    sample_count = len(context.adapter.samples)
                elif operation is performance.PerformanceOperation.FIT:
                    sample_count = len(
                        context.feature_batches["train"].sample_ids
                    )
                else:
                    sample_count = len(
                        context.feature_batches["train"].sample_ids
                    )
                context.record_performance(
                    operation,
                    time.perf_counter() - started,
                    sample_count,
                )
            stage_digest = attestation.canonical_digest(
                context.stage_records[stage.stage]
            )
            if stage_index < resume_count:
                if stage_digest != resume_digests[stage_index]:
                    raise ValueError(
                        "lifecycle checkpoint replay diverged at stage "
                        f"{stage.stage.value}"
                    )
            else:
                completed_stages.append(stage.stage.value)
                stage_digests.append(stage_digest)
                persist_checkpoint(completed_stages, stage_digests, "running")
            context.resume_replay = False
            ensure_not_cancelled()
        ensure_not_cancelled()
        records = tuple(
            modalities.StageEvidence(
                stage=stage,
                evidence_links=("modality_lifecycle.json",),
                evidence=context.stage_records[stage],
                notes=(f"completed {stage.value}",),
            )
            for stage in modalities.REQUIRED_LIFECYCLE_STAGES
        )
        package = context.package
        if package is None:
            raise RuntimeError("packaging stage did not produce a package")
        observations = tuple(
            context.performance_observations[operation]
            for operation in performance.PerformanceOperation
            if operation in context.performance_observations
        )
        performance_run = performance.PerformanceRun(
            run_id=f"{adapter.dataset_hash}:{seed}",
            observations=observations,
        )
        performance_result = (
            performance_budget.evaluate(observations)
            if performance_budget is not None
            else None
        )
        performance_comparison = (
            performance_baseline.evaluate(performance_run)
            if performance_baseline is not None
            else None
        )
        performance_evidence: dict[str, object] = {
            "run": performance_run.to_dict(),
            "observations": [
                observation.to_dict() for observation in observations
            ],
            "budget": (
                performance_budget.to_dict()
                if performance_budget is not None
                else None
            ),
            "result": (
                performance_result.to_dict()
                if performance_result is not None
                else None
            ),
            "baseline": (
                performance_baseline.to_dict()
                if performance_baseline is not None
                else None
            ),
            "comparison": (
                performance_comparison.to_dict()
                if performance_comparison is not None
                else None
            ),
        }
        if performance_result is not None and not performance_result.passed:
            if not package_preexisting:
                shutil.rmtree(package_target, ignore_errors=True)
            raise ValueError(
                "performance budget failed: "
                + "; ".join(performance_result.violations)
            )
        if (
            performance_comparison is not None
            and not performance_comparison.passed
        ):
            if not package_preexisting:
                shutil.rmtree(package_target, ignore_errors=True)
            raise ValueError(
                "performance baseline failed: "
                + "; ".join(performance_comparison.violations)
            )
        report = LifecycleReport(
            modality=modalities.modality_value(adapter.modality),
            dataset_hash=adapter.dataset_hash,
            stage_evidence=records,
            evaluation=context.evaluation,
            performance=performance_evidence,
            package=package,
            inference=context.inference,
            monitoring=context.monitoring,
            documentation=context.documentation,
            recovery=context.recovery,
        ).sign(signing_key)
        if report_target is not None:
            durability.write_atomically(
                report_target,
                (
                    json.dumps(report.to_dict(), indent=2, sort_keys=True)
                    + "\n"
                ).encode("utf-8"),
            )
            ensure_not_cancelled()
        if checkpoint_target is not None:
            persist_checkpoint(
                completed_stages,
                stage_digests,
                "completed",
            )
        return report

    def declaration(
        self, report: LifecycleReport
    ) -> modalities.ModalityDeclaration:
        """Converts a report into an explicit experimental declaration.

        Args:
            report: Completed lifecycle report.

        Returns:
            A modality declaration with the report's stage evidence and
            limitations; its status is extension-only for extension modalities
            and experimental otherwise.
        """
        return modalities.ModalityDeclaration(
            modality=modalities.modality_from_value(report.modality),
            status=(
                modalities.ModalityStatus.EXTENSION_ONLY
                if report.modality.startswith("extension:")
                else modalities.ModalityStatus.EXPERIMENTAL
            ),
            stage_evidence=report.stage_evidence,
            required_extras=("core",),
            failure_modes=(
                "malformed or unsupported modality input",
                "representation or codec dependency unavailable",
                "resource limits exceeded",
                "package signature or inference parity failure",
            ),
            limitations=tuple(
                str(value)
                for value in cast(
                    Sequence[object], report.documentation["limitations"]
                )
            ),
        )


def file_digest(path: Path) -> str:
    """Computes the SHA-256 digest of a file.

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


def sync_file(path: Path) -> None:
    """Flushes a file before it becomes visible through an atomic rename.

    Args:
        path: File to flush.
    """
    durability.sync_file(path)


def sync_directory(path: Path) -> None:
    """Flushes a directory entry after an atomic replacement.

    Args:
        path: Directory whose entry must reach stable storage.
    """
    durability.sync_directory(path)


def load_package(
    path: str | Path, signing_key: str | bytes
) -> LogisticRegression:
    """Verifies a signed package before deserializing its estimator.

    Args:
        path: Package directory holding `manifest.json` and `model.joblib`.
        signing_key: Key the manifest signature is checked with.

    Returns:
        The classifier stored in the package.

    Raises:
        ValueError: If the package is incomplete, its manifest, signature,
            digest or feature schema is invalid, or the classifier has fewer
            than two classes or non-finite values.
        TypeError: If the payload is not a logistic regression.
    """
    target = Path(path)
    manifest_path = target / "manifest.json"
    model_path = target / "model.joblib"
    if not manifest_path.is_file() or not model_path.is_file():
        raise ValueError("modality package is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("modality package manifest must be an object")
    unsigned = dict(manifest)
    signature = unsigned.pop("signature", None)
    schema_version = unsigned.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != (
        PACKAGE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported modality package schema version")
    modality = unsigned.get("modality")
    if not isinstance(modality, str) or not modality.strip():
        raise ValueError("modality package modality is invalid")
    dataset_hash = unsigned.get("dataset_hash")
    if not isinstance(dataset_hash, str) or not dataset_hash.strip():
        raise ValueError("modality package dataset_hash is invalid")
    feature_names = unsigned.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("modality package feature_names are invalid")
    if any(
        not isinstance(name, str) or not name.strip() for name in feature_names
    ) or len(set(feature_names)) != len(feature_names):
        raise ValueError("modality package feature_names are invalid")
    model_digest = unsigned.get("model_sha256")
    if not isinstance(model_digest, str) or not is_sha256(model_digest):
        raise ValueError("modality package model_sha256 is invalid")
    if not isinstance(unsigned.get("evaluation"), Mapping):
        raise ValueError("modality package evaluation is invalid")
    if (
        not isinstance(signature, dict)
        or signature.get("algorithm") != attestation.SIGNATURE_ALGORITHM
    ):
        raise ValueError("modality package signature is missing or unsupported")
    value = signature.get("value")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("modality package signature is empty")
    expected = attestation.sign_payload(unsigned, signing_key)
    if not hmac.compare_digest(value, expected):
        raise ValueError("modality package signature does not verify")
    if file_digest(model_path) != model_digest:
        raise ValueError("modality package model digest does not verify")
    loaded = joblib.load(model_path)
    if not isinstance(loaded, LogisticRegression):
        raise TypeError(
            "modality package does not contain the expected classifier"
        )
    if getattr(loaded, "n_features_in_", None) != len(feature_names):
        raise ValueError(
            "modality package model and feature schema do not match"
        )
    if len(getattr(loaded, "classes_", ())) < 2:
        raise ValueError(
            "modality package classifier has fewer than two classes"
        )
    if (
        not numpy.isfinite(loaded.coef_).all()
        or not numpy.isfinite(loaded.intercept_).all()
    ):
        raise ValueError(
            "modality package classifier contains non-finite values"
        )
    return loaded


def is_sha256(value: str) -> bool:
    """Checks for a canonical lowercase SHA-256 digest.

    Args:
        value: Text to check.

    Returns:
        True if `value` is 64 lowercase hexadecimal characters, else False.
    """
    return len(value) == SHA256_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


def normalize_json_object(
    value: Mapping[str, object], name: str
) -> dict[str, object]:
    """Normalizes a finite JSON object or rejects the evidence boundary.

    Args:
        value: Mapping to normalize.
        name: Field name used in error messages.

    Returns:
        A new dictionary of JSON-native values with sorted keys.

    Raises:
        TypeError: If `value` is not a mapping.
        ValueError: If `value` is not JSON-native, has non-finite numbers, or
            does not serialize to an object.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        normalized = json.loads(
            json.dumps(dict(value), allow_nan=False, sort_keys=True)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be JSON-native") from error
    if not isinstance(normalized, dict):
        raise ValueError(f"{name} must serialize to an object")
    return normalized


__all__ = [
    "AudioAdapter",
    "DataExplainabilityStage",
    "DatasetPartitions",
    "DocumentAdapter",
    "FeatureBatch",
    "GeospatialAdapter",
    "GraphAdapter",
    "ImageAdapter",
    "InferenceStage",
    "LifecycleCancellation",
    "LifecycleCancelledError",
    "LifecycleCheckpoint",
    "LifecyclePackage",
    "LifecycleReport",
    "ModalityAdapter",
    "ModalityCurationResult",
    "ModalityLifecycle",
    "ModalityPredictor",
    "ModellingStage",
    "RasterAdapter",
    "TextAdapter",
    "VideoAdapter",
    "file_digest",
    "is_sha256",
    "load_package",
    "normalize_json_object",
]
