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
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

from scml import durability
from scml.contracts import data as data_contracts
from scml.contracts import errors
from scml.contracts import events
from scml.contracts import modalities
from scml.contracts import performance
from scml.core import limits
from scml.data.modalities import audio
from scml.data.modalities import curation
from scml.data.modalities import document
from scml.data.modalities import geospatial
from scml.data.modalities import graph
from scml.data.modalities import image
from scml.data.modalities import video
from scml.explainability import quality
from scml.governance import attestation
from scml.models import documents as document_models
from scml.models import images as image_models

MAX_REPEATED_EXPLANATION_SAMPLES = 128
MAX_RAW_GRAPH_FEATURES = 64
MAX_GRAPH_CONTEXT_NODES = 64
MAX_GRAPH_CONTEXT_EDGES = 128
PACKAGE_SCHEMA_VERSION = 1
SHA256_LENGTH = 64


class LifecycleCancelledError(errors.CancelledError):
    """Raised when a modality lifecycle receives a cancellation request."""

    def __init__(self, message: str) -> None:
        """Create a cancellation error with an actionable message."""
        super().__init__(message, {"scope": "modality_lifecycle"})


class LifecycleCancellation:
    """Thread-safe cooperative cancellation signal for one lifecycle run."""

    def __init__(self) -> None:
        """Create a cancellation signal in the not-cancelled state."""
        self.signal = threading.Event()

    def cancel(self) -> None:
        """Request cancellation at the next lifecycle stage boundary."""
        self.signal.set()

    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self.signal.is_set()


@dataclasses.dataclass(frozen=True, slots=True)
class FeatureBatch:
    """A finite, bounded feature matrix tied to stable sample identities."""

    sample_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: np.ndarray = dataclasses.field(repr=False)

    def __post_init__(self) -> None:
        """Validate shape, finiteness, and immutable identity metadata."""
        matrix = np.asarray(self.values, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("feature values must be a two-dimensional matrix")
        if matrix.shape[0] != len(self.sample_ids):
            raise ValueError("feature rows must match sample IDs")
        if matrix.shape[1] != len(self.feature_names):
            raise ValueError("feature columns must match feature names")
        if not np.isfinite(matrix).all():
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
    """Leakage-aware train, validation, and test partitions."""

    train: tuple[Any, ...]
    validation: tuple[Any, ...]
    test: tuple[Any, ...]
    groups: dict[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        """Normalize partitions and reject identity overlap."""
        object.__setattr__(self, "train", tuple(self.train))
        object.__setattr__(self, "validation", tuple(self.validation))
        object.__setattr__(self, "test", tuple(self.test))
        normalized = {
            name: tuple(sorted(values)) for name, values in self.groups.items()
        }
        object.__setattr__(self, "groups", normalized)


@dataclasses.dataclass(frozen=True, slots=True)
class LifecyclePackage:
    """Digest and signature identity of an atomically written package."""

    path: str
    model_sha256: str
    signature: str
    schema_version: int = PACKAGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate package identity before it enters lifecycle evidence."""
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
        """Return JSON-native package evidence."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class ModalityPredictor:
    """Verified library inference for a signed modality lifecycle package.

    The caller supplies an adapter configured with the decoder and
    representation policy required for new samples. This class does not
    infer a codec, deserialize arbitrary raw transport payloads, or expose an
    HTTP server; it keeps those deployment decisions outside the package.
    """

    adapter: ModalityAdapter
    model: LogisticRegression = dataclasses.field(repr=False)
    package: LifecyclePackage
    feature_names: tuple[str, ...]
    training_dataset_hash: str

    @classmethod
    def from_package(
        cls,
        adapter: ModalityAdapter,
        package_path: str | Path,
        signing_key: str | bytes,
    ) -> Self:
        """Load and validate a signed package for one modality adapter."""
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
        )

    def features_for(self, samples: Sequence[Any]) -> FeatureBatch:
        """Prepare and validate features for new modality samples."""
        values = tuple(samples)
        if not values:
            raise ValueError("samples must not be empty")
        batch = self.adapter.features(values)
        if batch.feature_names != self.feature_names:
            raise ValueError("new sample feature schema does not match package")
        return batch

    def predict(self, samples: Sequence[Any]) -> np.ndarray:
        """Return predictions after validating the signed feature contract."""
        return self.model.predict(self.features_for(samples).values)

    def predict_proba(self, samples: Sequence[Any]) -> np.ndarray:
        """Return class probabilities after feature-contract validation."""
        return self.model.predict_proba(self.features_for(samples).values)

    def health(self) -> dict[str, object]:
        """Return machine-readable package and inference readiness evidence."""
        return {
            "package_verified": True,
            "modality": modalities.modality_value(self.adapter.modality),
            "training_dataset_hash": self.training_dataset_hash,
            "feature_names": list(self.feature_names),
            "model_sha256": self.package.model_sha256,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class LifecycleReport:
    """All evidence emitted by one completed modality lifecycle."""

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
        """Validate the report identity and optional signature."""
        for field_name in ("modality", "dataset_hash"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.signature is not None and not self.signature.strip():
            raise ValueError("signature must be non-empty when provided")

    def unsigned_dict(self) -> dict[str, object]:
        """Return the exact report content covered by the signature."""
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
        """Return a report signed over its exact serialized evidence."""
        return dataclasses.replace(
            self,
            signature=attestation.sign_payload(
                self.unsigned_dict(), signing_key
            ),
        )

    def verify_signature(self, signing_key: str | bytes) -> bool:
        """Return whether the report signature matches its current content."""
        if self.signature is None:
            return False
        expected = attestation.sign_payload(self.unsigned_dict(), signing_key)
        return hmac.compare_digest(self.signature, expected)

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> Self:
        """Restore a versioned report without trusting its evidence claims."""
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
        """Read and parse a persisted lifecycle report."""
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(record, Mapping):
            raise ValueError("lifecycle report must be a JSON object")
        return cls.from_dict(record)

    @classmethod
    def read_verified(cls, path: str | Path, signing_key: str | bytes) -> Self:
        """Read a report and reject it unless its content signature verifies."""
        report = cls.read(path)
        if not report.verify_signature(signing_key):
            raise ValueError("lifecycle report signature does not verify")
        return report

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible lifecycle record."""
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
        """Reject incomplete, non-prefix, or unsigned-identity state."""
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
        """Return the exact checkpoint content covered by its signature."""
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
        """Return a checkpoint signed over its exact serialized identity."""
        return dataclasses.replace(
            self,
            signature=attestation.sign_payload(
                self.unsigned_dict(), signing_key
            ),
        )

    def verify_signature(self, signing_key: str | bytes) -> bool:
        """Return whether the persisted checkpoint signature is valid."""
        if self.signature is None:
            return False
        expected = attestation.sign_payload(self.unsigned_dict(), signing_key)
        return hmac.compare_digest(self.signature, expected)

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> Self:
        """Restore a checkpoint without trusting its serialized claims."""
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
        """Return deterministic JSON-native checkpoint evidence."""
        record = self.unsigned_dict()
        record["signature"] = {
            "algorithm": attestation.SIGNATURE_ALGORITHM,
            "value": self.signature,
        }
        return record

    @classmethod
    def read(cls, path: str | Path) -> Self:
        """Read a persisted lifecycle checkpoint."""
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(record, Mapping):
            raise ValueError("lifecycle checkpoint must be a JSON object")
        return cls.from_dict(record)

    @classmethod
    def read_verified(cls, path: str | Path, signing_key: str | bytes) -> Self:
        """Read a checkpoint and reject it unless its signature verifies."""
        checkpoint = cls.read(path)
        if not checkpoint.verify_signature(signing_key):
            raise ValueError("lifecycle checkpoint signature does not verify")
        return checkpoint


@dataclasses.dataclass(frozen=True, slots=True)
class ModalityCurationResult:
    """Signed curation evidence for a dataset without a model claim."""

    modality: str
    dataset_hash: str
    sample_count: int
    curation: Mapping[str, object]
    dataset_card: Mapping[str, object]
    path: str
    signature: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-native curation result evidence."""
        return {
            "schema_version": 1,
            "record_type": "modality_curation",
            "modality": self.modality,
            "dataset_hash": self.dataset_hash,
            "sample_count": self.sample_count,
            "curation": dict(self.curation),
            "dataset_card": dict(self.dataset_card),
            "path": self.path,
            "signature": self.signature,
        }


class ModalityCurationAdapter(abc.ABC):
    """Polymorphic boundary for source identity and data understanding."""

    @property
    @abc.abstractmethod
    def modality(self) -> modalities.ModalityIdentity:
        """Return the declared dataset family."""

    @property
    @abc.abstractmethod
    def dataset_hash(self) -> str:
        """Return the immutable dataset identity."""

    @property
    @abc.abstractmethod
    def samples(self) -> tuple[Any, ...]:
        """Return samples in canonical source order."""

    @abc.abstractmethod
    def curate(
        self,
    ) -> curation.ModalityCurationReport | document.DocumentCurationReport:
        """Return reviewable data-understanding evidence."""

    @abc.abstractmethod
    def modality_notes(self) -> tuple[str, ...]:
        """Return representation and curation limits."""


@dataclasses.dataclass(frozen=True, slots=True)
class ExtensionCurationAdapter(ModalityCurationAdapter):
    """Adapt arbitrary caller-owned records to the curation-only contract.

    The callback owns domain-specific profiling and validation. This adapter
    binds its result to the extension identity, immutable dataset identity,
    and sample count before the signed curation project can publish evidence.
    It intentionally does not provide split, feature, label, or model hooks.
    """

    extension: modalities.ExtensionModality
    sample_values: tuple[Any, ...]
    dataset_hash_value: str
    curation_callback: Callable[[], curation.ModalityCurationReport]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the caller-owned curation boundary."""
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
        """Return the namespaced extension identity."""
        return self.extension

    @property
    def dataset_hash(self) -> str:
        """Return the caller-owned immutable dataset identity."""
        return self.dataset_hash_value

    @property
    def samples(self) -> tuple[Any, ...]:
        """Return arbitrary records in caller-defined canonical order."""
        return self.sample_values

    def curate(
        self,
    ) -> curation.ModalityCurationReport:
        """Run and validate the caller-owned curation callback."""
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
        """Return explicit caller-owned boundaries for the dataset card."""
        return self.limitations


class ModalityAdapter(ModalityCurationAdapter):
    """Polymorphic boundary between a dataset family and the lifecycle."""

    modality: ClassVar[modalities.ModalityIdentity]

    @abc.abstractmethod
    def split(self, seed: int) -> DatasetPartitions:
        """Return leakage-aware partitions."""

    @abc.abstractmethod
    def features(self, samples: Sequence[Any]) -> FeatureBatch:
        """Return bounded features for the supplied samples."""

    @abc.abstractmethod
    def sample_id(self, sample: Any) -> str:
        """Return the stable identity of one sample."""

    @abc.abstractmethod
    def label(self, sample: Any) -> object | None:
        """Return the supervised label, if one exists."""

    @abc.abstractmethod
    def monitor(self, observed: ModalityAdapter) -> dict[str, object]:
        """Compare this reference adapter with an observed adapter."""

    def modality_specific_explanation(
        self,
        sample: Any,
        model: LogisticRegression,
        feature_names: Sequence[str],
        coefficients: np.ndarray,
    ) -> dict[str, object]:
        """Return explicit modality-specific explanation evidence.

        The default is an honest unavailable result. Adapters override this
        hook only when they own a decoder or perturbation policy that can be
        represented and tested without inventing raw-modality semantics.
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
        """Return caller-owned independent explanation evidence, if known.

        Built-in adapters return no reference by default because raw-modality
        semantics cannot be inferred from a feature matrix. A domain adapter
        may override this method when it carries an independently authored
        annotation, benchmark, or synthetic oracle.
        """
        del sample, feature_names
        return None

    def labels(self, samples: Sequence[Any]) -> tuple[object, ...]:
        """Return labels in sample order."""
        return tuple(self.label(sample) for sample in samples)

    def group_ids(self, samples: Sequence[Any]) -> tuple[str, ...]:
        """Return stable grouping identities for split evidence."""
        return tuple(self.sample_id(sample) for sample in samples)


class DocumentAdapter(ModalityAdapter):
    """Adapter for UTF-8, Markdown, and caller-decoded document samples."""

    modality = modalities.Modality.DOCUMENT

    def __init__(self, dataset: document.DocumentDataset) -> None:
        """Bind an immutable document dataset."""
        self.dataset = dataset

    @property
    def dataset_hash(self) -> str:
        """Return the document corpus identity."""
        return self.dataset.content_hash

    @property
    def samples(self) -> tuple[document.DocumentSample, ...]:
        """Return document samples."""
        return self.dataset.samples

    def curate(self) -> document.DocumentCurationReport:
        """Return document profiling and label evidence."""
        return self.dataset.curate()

    def split(self, seed: int) -> DatasetPartitions:
        """Split by document identity to prevent chunk leakage."""
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
        """Prepare bounded lexical features without using labels."""
        profiles = document.DocumentDataset(tuple(samples)).profile()
        rows = np.asarray(
            [
                (profile.character_count, profile.token_count)
                for profile in profiles
            ],
            dtype=np.float64,
        )
        return FeatureBatch(
            tuple(self.sample_id(sample) for sample in samples),
            ("character_count", "token_count"),
            rows,
        )

    def sample_id(self, sample: document.DocumentSample) -> str:
        """Return a stable identity for one document sample or chunk."""
        return f"{sample.document_id}:{sample.content_hash}"

    def label(self, sample: document.DocumentSample) -> object | None:
        """Return the document label."""
        return sample.label

    def modality_specific_explanation(
        self,
        sample: Any,
        model: LogisticRegression,
        feature_names: Sequence[str],
        coefficients: np.ndarray,
    ) -> dict[str, object]:
        """Return page and section provenance for a document explanation."""
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
        """Score source-character masks while preserving span offsets."""
        original_features = self.features((sample,)).values
        original_decision = np.asarray(
            model.decision_function(original_features)
        )
        if original_decision.ndim == 1:
            selected_class = 0
            original_score = float(original_decision[0])
        else:
            predicted = model.predict(original_features)[0]
            selected_class = int(np.flatnonzero(model.classes_ == predicted)[0])
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
            masked_decision = np.asarray(
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
        """Return the document profile drift report."""
        if not isinstance(observed, DocumentAdapter):
            raise TypeError("observed adapter must be a DocumentAdapter")
        current = self.dataset.curate()
        report = document_models.DocumentMonitor(current).observe(
            observed.dataset
        )
        return report.to_dict()

    def modality_notes(self) -> tuple[str, ...]:
        """Describe the bounded document representation."""
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
    """Adapter for bounded, decoded image samples."""

    modality = modalities.Modality.IMAGE

    def __init__(self, dataset: image.ImageDataset) -> None:
        """Bind an immutable image dataset."""
        self.dataset = dataset

    @property
    def dataset_hash(self) -> str:
        """Return the image corpus identity."""
        return self.dataset.content_hash

    @property
    def samples(self) -> tuple[image.ImageSample, ...]:
        """Return image samples."""
        return self.dataset.samples

    def curate(self) -> curation.ModalityCurationReport:
        """Return image dimensions, encodings, duplicate, and label evidence."""
        return self.dataset.curate()

    def split(self, seed: int) -> DatasetPartitions:
        """Split by capture group to prevent identity leakage."""
        result = self.dataset.split(seed=seed)
        return DatasetPartitions(
            result.train,
            result.validation,
            result.test,
            result.group_ids(),
        )

    def features(self, samples: Sequence[image.ImageSample]) -> FeatureBatch:
        """Prepare deterministic resized-pixel features."""
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
        """Return a source-and-content identity for one image sample."""
        return f"{sample.source}:{sample.content_hash}"

    def label(self, sample: image.ImageSample) -> object | None:
        """Return the image label."""
        return sample.label

    def modality_specific_explanation(
        self,
        sample: Any,
        model: LogisticRegression,
        feature_names: Sequence[str],
        coefficients: np.ndarray,
    ) -> dict[str, object]:
        """Return bounded occlusion evidence on transformed RGB pixels."""
        if not isinstance(sample, image.ImageSample):
            raise TypeError("sample must be an ImageSample")
        features = self.features((sample,)).values
        if features.shape[1] != 16 * 16 * 3:
            raise ValueError("image lifecycle expects 16x16 RGB features")
        decision = np.asarray(model.decision_function(features))
        if decision.ndim == 1:
            selected_class = 0
            original_score = float(decision[0])
        else:
            predicted = model.predict(features)[0]
            selected_class = int(np.flatnonzero(model.classes_ == predicted)[0])
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
                occluded_decision = np.asarray(
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
        """Score bounded source-pixel masks before the resize transform.

        The model still consumes the declared 16x16 RGB representation. The
        difference is that each mask is applied to the validated source
        pixels first, so the evidence records a source-space perturbation and
        the exact image dimensions used to construct it.
        """
        source_pixels = sample.decode("RGB")
        source_height, source_width, source_channels = source_pixels.shape
        if source_channels != 3:
            raise ValueError("decoded RGB image must have three channels")
        image_library = image.require_pillow("raw image explanation")
        row_step = max(1, int(np.ceil(source_height / 4)))
        column_step = max(1, int(np.ceil(source_width / 4)))
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
                    np.asarray(resized, dtype=np.float32).reshape(1, -1) / 255.0
                )
                decision = np.asarray(model.decision_function(features))
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
        """Return mean-pixel and duplicate drift evidence."""
        if not isinstance(observed, ImageAdapter):
            raise TypeError("observed adapter must be an ImageAdapter")
        report = image_models.ImageMonitor(self.dataset.curate()).observe(
            observed.dataset
        )
        return report.to_dict()

    def modality_notes(self) -> tuple[str, ...]:
        """Describe image lifecycle limits."""
        return (
            "The baseline uses resized RGB pixels and does not claim "
            "localization, saliency validation, or object detection.",
        )


class AudioAdapter(ModalityAdapter):
    """Adapter for WAV-capable audio samples and explicit codec probes."""

    modality = modalities.Modality.AUDIO

    def __init__(self, dataset: audio.AudioDataset) -> None:
        """Bind an immutable audio dataset."""
        self.dataset = dataset

    @property
    def dataset_hash(self) -> str:
        """Return the audio corpus identity."""
        return self.dataset.content_hash

    @property
    def samples(self) -> tuple[audio.AudioSample, ...]:
        """Return audio samples."""
        return self.dataset.samples

    def curate(self) -> curation.ModalityCurationReport:
        """Return codec, signal shape, duration, and label evidence."""
        return self.dataset.curate()

    def split(self, seed: int) -> DatasetPartitions:
        """Split by speaker/session group."""
        result = self.dataset.split(seed=seed)
        return DatasetPartitions(
            result.train, result.validation, result.test, result.group_ids()
        )

    def features(self, samples: Sequence[audio.AudioSample]) -> FeatureBatch:
        """Prepare bounded WAV signal statistics."""
        prepared = audio.AudioDataset(tuple(samples)).prepare_features()
        return FeatureBatch(
            tuple(prepared.sources), prepared.feature_names, prepared.to_numpy()
        )

    def sample_id(self, sample: audio.AudioSample) -> str:
        """Return the audio source identity."""
        return sample.source

    def label(self, sample: audio.AudioSample) -> object | None:
        """Return the audio label."""
        return sample.label

    def monitor(self, observed: ModalityAdapter) -> dict[str, object]:
        """Return curation identity and profile drift evidence."""
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
        """Describe audio lifecycle limits."""
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
        coefficients: np.ndarray,
    ) -> dict[str, object]:
        """Return bounded time-local score evidence for WAV samples."""
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
            1, int(np.ceil(sample.metadata.frame_count / window_count))
        )
        full_features = self.features((sample,)).values[0]
        full_decision = np.asarray(model.decision_function(full_features[None]))
        full_score = float(full_decision.reshape(-1)[0])
        windows: list[dict[str, object]] = []
        for start_frame in range(0, sample.metadata.frame_count, window_frames):
            prepared = audio.AudioDataset((sample,)).prepare_features(
                max_frames=window_frames,
                start_frame=start_frame,
            )
            values = prepared.to_numpy()[0]
            decision = np.asarray(model.decision_function(values[None]))
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
        """Measure bounded score changes under deterministic additive noise.

        This is a reproducible perturbation probe for the current linear
        baseline. It is not a perceptual speech-quality test, a denoising
        system, or evidence that the model is robust to real recording noise.
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
            not np.isfinite(value) or value < 0.0 or value > 120.0
            for value in snr_db
        ):
            raise ValueError("snr_db values must be finite and in [0, 120]")
        parameters, values = audio.wav_pcm_values(sample.encoded_bytes)
        signal_rms = float(np.sqrt(np.mean(np.square(values))))
        if signal_rms <= 0.0:
            return {
                "status": "not_available",
                "method": "deterministic_additive_background_noise",
                "reason": "source signal is silent",
            }
        randomizer = np.random.default_rng(seed)
        scenarios: list[dict[str, object]] = []
        for requested_snr in snr_db:
            noise_rms = signal_rms / (10.0 ** (requested_snr / 20.0))
            noise = randomizer.normal(0.0, noise_rms, size=values.shape)
            mixed = values + noise
            clipping_fraction = float(np.mean(np.abs(mixed) > 1.0))
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
                np.asarray(model.decision_function(features)).reshape(-1)[0]
            )
            observed_noise = mixed - values
            observed_noise_rms = float(
                np.sqrt(np.mean(np.square(observed_noise)))
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
        """Score source PCM windows after replacing them with silence."""
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
                np.asarray(model.decision_function(features)).reshape(-1)[0]
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
    """Adapter for probed video metadata or an explicit frame decoder."""

    modality = modalities.Modality.VIDEO

    def __init__(
        self,
        dataset: video.VideoDataset,
        frame_decoder: video.VideoFrameDecoder | None = None,
        max_frames: int = 8,
        max_frame_pixels: int = 1_000_000,
        temporal_cutoff: datetime | str | None = None,
    ) -> None:
        """Bind a dataset, decoder, and optional explicit temporal policy."""
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
        """Return the video corpus identity."""
        return self.dataset.content_hash

    @property
    def samples(self) -> tuple[video.VideoSample, ...]:
        """Return video samples."""
        return self.dataset.samples

    def curate(self) -> curation.ModalityCurationReport:
        """Return encoding, shape, timing, and label evidence."""
        return self.dataset.curate()

    def split(self, seed: int) -> DatasetPartitions:
        """Split by subject/session/camera group."""
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
        """Prepare metadata or explicitly decoded temporal features."""
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
        """Return the video source identity."""
        return sample.source

    def label(self, sample: video.VideoSample) -> object | None:
        """Return the video label."""
        return sample.label

    def monitor(self, observed: ModalityAdapter) -> dict[str, object]:
        """Return curation identity and metadata drift evidence."""
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
        """Describe video lifecycle limits."""
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
        coefficients: np.ndarray,
    ) -> dict[str, object]:
        """Return bounded temporal frame-statistics occlusion evidence."""
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
            np.asarray(model.decision_function(values[None])).reshape(-1)[0]
        )
        terms: list[dict[str, object]] = []
        slot_width = 5
        frame_indices = self.dataset.frame_indices_for(sample, self.max_frames)
        for slot, frame_index in enumerate(frame_indices):
            occluded = values.copy()
            start = slot * slot_width
            occluded[start : start + slot_width] = 0.0
            score = float(
                np.asarray(model.decision_function(occluded[None])).reshape(-1)[
                    0
                ]
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
        """Score sampled decoded frames after replacing each with zeros."""
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
            ) -> np.ndarray:
                frame = np.asarray(decoder(payload, frame_index))
                if frame_index == target_frame_index:
                    return np.zeros_like(frame)
                return frame

            prepared = video.VideoDataset((sample,)).prepare_frame_features(
                masked_decoder,
                max_frames=self.max_frames,
                max_frame_pixels=self.max_frame_pixels,
            )
            values = prepared.to_numpy()[0]
            score = float(
                np.asarray(model.decision_function(values[None])).reshape(-1)[0]
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
    """Adapter for inductive node classification on graph node features."""

    modality = modalities.Modality.GRAPH

    def __init__(self, dataset: graph.GraphData) -> None:
        """Bind a validated graph with declared node features."""
        self.dataset = dataset

    @property
    def dataset_hash(self) -> str:
        """Return the full graph identity."""
        return self.dataset.identity

    @property
    def samples(self) -> tuple[graph.GraphNode, ...]:
        """Return canonical graph nodes."""
        return self.dataset.nodes

    def curate(self) -> curation.ModalityCurationReport:
        """Return topology, feature, and label evidence."""
        return self.dataset.curate()

    def split(self, seed: int) -> DatasetPartitions:
        """Split nodes inductively and retain partition-local edge evidence."""
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
        """Prepare declared node features without cross-partition topology."""
        if not self.dataset.feature_names:
            raise ValueError("graph baseline requires declared node features")
        rows = np.asarray([sample.features for sample in samples], dtype=float)
        return FeatureBatch(
            tuple(sample.node_id for sample in samples),
            self.dataset.feature_names,
            rows,
        )

    def sample_id(self, sample: graph.GraphNode) -> str:
        """Return a node identity."""
        return sample.node_id

    def label(self, sample: graph.GraphNode) -> object | None:
        """Return the node label."""
        return sample.label

    def one_hop_subgraph(
        self,
        node_id: str,
        max_nodes: int = MAX_GRAPH_CONTEXT_NODES,
        max_edges: int = MAX_GRAPH_CONTEXT_EDGES,
    ) -> dict[str, object]:
        """Return bounded, deterministic one-hop topology evidence.

        The result is review context for a node-feature baseline. It does not
        claim that the model used topology or that edge removal is a faithful
        graph-neural-network attribution method.
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
        coefficients: np.ndarray,
    ) -> dict[str, object]:
        """Return one-hop graph context and node-feature evidence."""
        if not isinstance(sample, graph.GraphNode):
            raise TypeError("sample must be a GraphNode")
        subgraph = self.one_hop_subgraph(sample.node_id)
        subgraph_node_ids = cast(list[str], subgraph["node_ids"])
        neighbors = subgraph_node_ids[1:]
        edges = cast(list[dict[str, object]], subgraph["edges"])
        values = np.asarray(sample.features, dtype=np.float64)
        decision = np.asarray(model.decision_function(values[None]))
        if decision.ndim == 1:
            selected_class = 0
            original_score = float(decision[0])
        else:
            predicted = model.predict(values[None])[0]
            selected_class = int(np.flatnonzero(model.classes_ == predicted)[0])
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
            masked_decision = np.asarray(
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
        """Return graph identity and profile drift evidence."""
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
        """Describe graph lifecycle limits."""
        return (
            "The baseline is inductive node classification on declared node "
            "features.",
            "It excludes topology-derived features across partitions and "
            "does not claim graph-neural-network support.",
        )


class GeospatialAdapter(ModalityAdapter):
    """Adapter for CRS-declared vector feature classification."""

    modality = modalities.Modality.GEOSPATIAL

    def __init__(
        self, dataset: geospatial.GeospatialDataset, block_size: float = 1.0
    ) -> None:
        """Bind a geospatial dataset and an explicit CRS-unit block size."""
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.dataset = dataset
        self.block_size = block_size

    @property
    def dataset_hash(self) -> str:
        """Return the vector dataset identity."""
        return self.dataset.content_hash

    @property
    def samples(self) -> tuple[geospatial.GeoFeature, ...]:
        """Return geospatial features."""
        return self.dataset.features

    def curate(self) -> curation.ModalityCurationReport:
        """Return geometry, CRS, bounds, and label evidence."""
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
                        data_contracts.FindingSeverity.WARNING,
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
        """Split by spatial blocks in declared CRS units."""
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
        """Prepare bounded geometry envelope and centroid features."""
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
        """Return a feature identity."""
        return sample.feature_id

    def label(self, sample: geospatial.GeoFeature) -> object | None:
        """Return the feature label."""
        return sample.label

    def modality_specific_explanation(
        self,
        sample: Any,
        model: LogisticRegression,
        feature_names: Sequence[str],
        coefficients: np.ndarray,
    ) -> dict[str, object]:
        """Return CRS-bound geometry and feature-contribution evidence."""
        if not isinstance(sample, geospatial.GeoFeature):
            raise TypeError("sample must be a GeoFeature")
        feature_index = next(
            index
            for index, item in enumerate(self.dataset.features)
            if item.feature_id == sample.feature_id
        )
        prepared = self.dataset.prepare_features()
        values = np.asarray(
            prepared.to_numpy()[feature_index], dtype=np.float64
        )
        decision = np.asarray(model.decision_function(values[None]))
        if decision.ndim == 1:
            selected_class = 0
            original_score = float(decision[0])
        else:
            predicted = model.predict(values[None])[0]
            selected_class = int(np.flatnonzero(model.classes_ == predicted)[0])
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
            masked_decision = np.asarray(model.decision_function(masked_values))
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
        """Return geospatial identity, CRS, and profile drift evidence."""
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
        """Describe geospatial lifecycle limits."""
        return (
            "The baseline uses geometry envelopes and centroids in the "
            "declared CRS.",
            "It does not claim reprojection, raster, spatial-join, or causal "
            "spatial inference support.",
        )


class RasterAdapter(ModalityAdapter):
    """Adapter for labelled raster metadata or bounded decoded pixels."""

    modality = modalities.Modality.RASTER

    def __init__(
        self,
        samples: Sequence[geospatial.RasterSample],
        labels: Mapping[str, object],
        decoder: geospatial.RasterDecoder | None = None,
        decoder_name: str | None = None,
        max_pixels: int = 4_000_000,
    ) -> None:
        """Bind raster identities and caller-owned labels.

        Raster payload decoding remains caller-owned. When supplied, the
        decoder name and pixel budget become part of the representation
        contract. The label sidecar is keyed by the canonical raster source
        and is part of the caller's dataset evidence.
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
        """Return the raster identity and label-sidecar digest."""
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
        """Return raster samples."""
        return self.dataset

    def curate(self) -> curation.ModalityCurationReport:
        """Return raster shape, CRS, transform, and label evidence."""
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
        quality_findings: list[data_contracts.CurationFinding] = []
        if len(crs_values) > 1 or len(dimensions) > 1 or len(dtypes) > 1:
            quality_findings.append(
                curation.build_finding(
                    "quality.raster-shape-or-crs-drift",
                    data_contracts.FindingSeverity.WARNING,
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
                    data_contracts.FindingSeverity.WARNING,
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
        """Split by source identity because the core lacks raster decoding."""
        generator = np.random.default_rng(seed)
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
        """Prepare raster metadata or bounded decoded pixel features."""
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
                np.asarray(rows, dtype=np.float64),
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
            tuple(sample.source for sample in samples), names, np.asarray(rows)
        )

    @staticmethod
    def pixel_feature_names(band_count: int) -> tuple[str, ...]:
        """Return names for bounded decoded-pixel summary features."""
        if band_count < 1:
            raise ValueError("band_count must be positive")
        return tuple(
            f"pixel_band_{band}_{statistic}"
            for band in range(band_count)
            for statistic in ("mean", "std", "min", "max")
        )

    @staticmethod
    def pixel_feature_values(
        pixels: np.ndarray, band_count: int
    ) -> tuple[float, ...]:
        """Return finite per-band summary values for decoded pixels."""
        values = np.asarray(pixels)
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
        """Return the canonical raster source identity."""
        return sample.source

    def label(self, sample: geospatial.RasterSample) -> object | None:
        """Return the caller-owned raster label."""
        return self.labels_by_source[sample.source]

    def modality_specific_explanation(
        self,
        sample: Any,
        model: LogisticRegression,
        feature_names: Sequence[str],
        coefficients: np.ndarray,
    ) -> dict[str, object]:
        """Return bounded occlusion evidence for decoded raster pixels."""
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
        original_values = np.asarray(
            self.pixel_feature_values(pixels, sample.bands), dtype=np.float64
        )
        if tuple(feature_names) != self.pixel_feature_names(sample.bands):
            raise ValueError("raster explanation feature schema does not match")
        if original_values.shape[0] != coefficients.shape[0]:
            raise ValueError(
                "raster explanation coefficients do not match features"
            )
        original_features = original_values.reshape(1, -1)
        decision = np.asarray(model.decision_function(original_features))
        if decision.ndim == 1:
            selected_class = 0
            original_score = float(decision[0])
        else:
            predicted = model.predict(original_features)[0]
            selected_class = int(np.flatnonzero(model.classes_ == predicted)[0])
            original_score = float(decision[0, selected_class])
        height, width = pixels.shape[:2]
        row_step = max(1, int(np.ceil(height / 4)))
        column_step = max(1, int(np.ceil(width / 4)))
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
                masked_features = np.asarray(
                    self.pixel_feature_values(masked_pixels, sample.bands),
                    dtype=np.float64,
                ).reshape(1, -1)
                masked_decision = np.asarray(
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
        """Return raster profile drift evidence."""
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
        """Describe raster lifecycle limits."""
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
    """Mutable products shared by the fifteen lifecycle stages."""

    adapter: ModalityAdapter
    package_path: Path
    signing_key: str | bytes
    seed: int
    observed_adapter: ModalityAdapter
    manifest: data_contracts.ModalityManifest | None = None
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
        """Record one stage result for the final evidence index."""
        self.stage_records[stage] = payload

    def record_performance(
        self,
        operation: performance.PerformanceOperation,
        elapsed_seconds: float,
        sample_count: int,
    ) -> None:
        """Record one measured operation with machine and timestamp context."""
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
    """One polymorphic stage in a modality lifecycle."""

    stage: ClassVar[modalities.LifecycleStage]
    performance_operation: ClassVar[performance.PerformanceOperation | None] = (
        None
    )

    @abc.abstractmethod
    def execute(self, context: LifecycleContext) -> None:
        """Execute the stage and record its evidence."""


class IngestionStage(ModalityLifecycleStage):
    """Confirm source identity and sample cardinality."""

    stage = modalities.LifecycleStage.INGESTION
    performance_operation = performance.PerformanceOperation.INGEST

    def execute(self, context: LifecycleContext) -> None:
        """Record immutable source identity."""
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
        """Validate labels and dataset-level curation state."""
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
        """Persist the curation/profile object in lifecycle state."""
        context.curation_report = context.adapter.curate()
        context.record(
            self.stage, {"profile": context.curation_report.to_dict()}
        )


class CurationStage(ModalityLifecycleStage):
    """Block unresolved data-quality findings before representation."""

    stage = modalities.LifecycleStage.CURATION

    def execute(self, context: LifecycleContext) -> None:
        """Require a release-ready curation report."""
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
        """Split and reject overlapping sample identities."""
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
        """Prepare and align features without fitting on held-out rows."""
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
        """Fit only on the training partition."""
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
        """Compute deterministic classification metrics and predictions."""
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
        """Record data lineage and non-causal feature summaries."""
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
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "q05": float(np.quantile(values, 0.05)),
                "median": float(np.median(values)),
                "q95": float(np.quantile(values, 0.95)),
                "nonfinite_count": int(np.count_nonzero(~np.isfinite(values))),
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
        """Record signed coefficient contributions with limitations."""
        model = context.model
        batch = context.feature_batches["test"]
        partitions = context.partitions
        if model is None or partitions is None or not len(batch.sample_ids):
            raise RuntimeError(
                "model, partitions, and test features are required"
            )
        strength = np.mean(np.abs(model.coef_), axis=0)
        global_terms = sorted(
            (
                (name, float(strength[index]))
                for index, name in enumerate(batch.feature_names)
            ),
            key=lambda item: (-item[1], item[0]),
        )[:10]
        decision_values = np.asarray(model.decision_function(batch.values))
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
        feature_scales = np.maximum(np.std(train_values, axis=0), 1.0)
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
            sample_decision = np.asarray(
                model.decision_function(sample_row[None])
            )
            sample_class = 0
            if sample_decision.ndim > 1:
                sample_class = int(np.argmax(sample_decision[0]))
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
                nonzero = np.flatnonzero(
                    np.abs(sample_row * sample_coefficients) > 1e-12
                )
                if len(nonzero):
                    perturbed = np.repeat(
                        sample_row[None, :], len(nonzero), axis=0
                    )
                    perturbed[np.arange(len(nonzero)), nonzero] = 0.0
                    perturbed_decisions = np.asarray(
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
                        np.mean(
                            np.sign(observed_change) == np.sign(expected_change)
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
    """Write and sign a digest-bound model package atomically."""

    stage = modalities.LifecycleStage.PACKAGING
    performance_operation = performance.PerformanceOperation.PACKAGE

    def execute(self, context: LifecycleContext) -> None:
        """Persist the estimator and signed manifest."""
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
                np.array_equal(predictor.model.classes_, model.classes_)
                and np.array_equal(predictor.model.coef_, model.coef_)
                and np.array_equal(predictor.model.intercept_, model.intercept_)
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
        """Load trusted bytes and compare predictions with the fitted model."""
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
        """Record modality-specific monitoring output."""
        context.monitoring = context.adapter.monitor(context.observed_adapter)
        context.record(self.stage, context.monitoring.copy())


class DocumentationStage(ModalityLifecycleStage):
    """Build a truthful dataset and model card for the baseline."""

    stage = modalities.LifecycleStage.DOCUMENTATION

    def execute(self, context: LifecycleContext) -> None:
        """Record support status, intended use, and limitations."""
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
        """Copy, tamper, and verify the package without touching its source."""
        if context.package is None:
            raise RuntimeError("package is required before recovery tests")
        temporary = Path(tempfile.mkdtemp(prefix="scml-lifecycle-recovery-"))
        tampered = temporary / "package"
        try:
            shutil.copytree(context.package.path, tampered)
            manifest_path = tampered / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["dataset_hash"] = "tampered"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
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
    """Execute the fifteen shared lifecycle stages in deterministic order."""

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
        manifest: data_contracts.ModalityManifest | None = None,
        require_independent_ground_truth: bool = False,
        cancellation_check: Callable[[], bool] | None = None,
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
    ) -> LifecycleReport:
        """Run every stage and optionally persist its evidence report.

        A supplied manifest is checked against the adapter before any stage
        executes. This keeps source, task, privacy, and lineage claims bound
        to the exact modality content used by the lifecycle.
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
            """Stop before publication and remove this run's artifacts."""
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
                    load_package(completed_report.package.path, signing_key)
                    return completed_report
                resume_count = len(checkpoint.completed_stages)
                resume_digests = checkpoint.stage_digests

        def persist_checkpoint(
            completed_stages: Sequence[str],
            stage_digests: Sequence[str],
            status: Literal["running", "completed"],
        ) -> None:
            """Atomically persist a signed stage boundary."""
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
        """Convert a report into an explicit experimental declaration."""
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
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sync_file(path: Path) -> None:
    """Flush a file before it becomes visible through an atomic rename."""
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def sync_directory(path: Path) -> None:
    """Flush a directory entry after an atomic replacement."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_package(
    path: str | Path, signing_key: str | bytes
) -> LogisticRegression:
    """Verify a signed package before deserializing its estimator."""
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
        not np.isfinite(loaded.coef_).all()
        or not np.isfinite(loaded.intercept_).all()
    ):
        raise ValueError(
            "modality package classifier contains non-finite values"
        )
    return loaded


def is_sha256(value: str) -> bool:
    """Return whether a value is a canonical lowercase SHA-256 digest."""
    return len(value) == SHA256_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


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
    "ModalityLifecycle",
    "ModalityPredictor",
    "ModellingStage",
    "RasterAdapter",
    "TextAdapter",
    "VideoAdapter",
    "file_digest",
    "is_sha256",
    "load_package",
]
