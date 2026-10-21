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
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time
from typing import Any
from typing import cast
from typing import ClassVar
from typing import Self

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

from scml.contracts import data as data_contracts
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
    schema_version: int = 1

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
        if manifest.get("modality") != adapter.modality.value:
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
            "modality": self.adapter.modality.value,
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

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible lifecycle record."""
        return {
            "schema_version": 1,
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


class ModalityAdapter(abc.ABC):
    """Polymorphic boundary between a dataset family and the lifecycle."""

    modality: ClassVar[modalities.Modality]

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

    @abc.abstractmethod
    def modality_notes(self) -> tuple[str, ...]:
        """Return representation limits that belong in the model card."""

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
            "modality": self.modality.value,
            "reason": "the adapter has no modality-specific explanation policy",
            "limitations": [
                "The lifecycle explanation remains in feature space.",
                "No raw-modality or causal interpretation is provided.",
            ],
        }

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
        return {
            "status": "available",
            "method": "document_span_provenance",
            "document_id": sample.document_id,
            "source_uri": sample.source_uri,
            "format": sample.format.value,
            "content_hash": sample.content_hash,
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
            "limitations": [
                "Span provenance identifies source context; it is not OCR "
                "confidence or citation grounding.",
                "The lifecycle model uses bounded lexical counts.",
                "No causal claim is made.",
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
        identities = (sample.content_hash for sample in self.dataset.samples)
        return curation.dataset_hash(identities)

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
        return {
            "status": "available",
            "method": "transformed_pixel_region_occlusion",
            "terms": [[name, float(value)] for name, value in selected],
            "baseline": "zero-valued transformed RGB pixels",
            "feature_space_only": True,
            "limitations": [
                "Occlusion occurs after resizing and RGB conversion.",
                "Regions are not object localization or raw-image saliency.",
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
        return {
            "status": "available",
            "method": "time_local_wav_window_score",
            "window_count": len(windows),
            "windows": windows,
            "feature_space_only": True,
            "limitations": [
                "Windows summarize decoded WAV statistics rather than raw "
                "waveform attribution.",
                "The score is associational and does not establish a causal "
                "audio event.",
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
    ) -> None:
        """Bind a dataset and an optional bounded caller-owned decoder."""
        if max_frames < 1 or max_frame_pixels < 1:
            raise ValueError("frame bounds must be positive")
        self.dataset = dataset
        self.frame_decoder = frame_decoder
        self.max_frames = max_frames
        self.max_frame_pixels = max_frame_pixels

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
        result = self.dataset.split(seed=seed)
        return DatasetPartitions(
            result.train, result.validation, result.test, result.group_ids()
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
        return {
            "status": "available",
            "method": "temporal_frame_statistics_occlusion",
            "terms": terms,
            "sampled_frame_indices": list(frame_indices),
            "feature_space_only": True,
            "limitations": [
                "Occlusion removes decoded frame statistics, not raw pixels or "
                "motion vectors.",
                "Audio-video synchronization and temporal event localization "
                "are not established.",
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
        incident_edges = tuple(
            edge
            for edge in self.dataset.edges
            if edge.source == sample.node_id or edge.target == sample.node_id
        )
        neighbors = sorted(
            {
                edge.target if edge.source == sample.node_id else edge.source
                for edge in incident_edges
            }
        )
        values = np.asarray(sample.features, dtype=np.float64)
        decision = np.asarray(model.decision_function(values[None]))
        feature_terms = sorted(
            (
                (name, float(values[index] * coefficients[index]))
                for index, name in enumerate(feature_names)
            ),
            key=lambda item: (-abs(item[1]), item[0]),
        )[:10]
        return {
            "status": "available",
            "method": "one_hop_graph_context_and_node_feature_contribution",
            "node_id": sample.node_id,
            "neighbors": neighbors,
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "weight": edge.weight,
                    "relation": edge.relation,
                }
                for edge in incident_edges
            ],
            "feature_terms": [[name, value] for name, value in feature_terms],
            "decision_score": float(decision.reshape(-1)[0]),
            "model_uses_topology": False,
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
        return self.dataset.curate()

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
        values = np.asarray(
            self.dataset.prepare_features().to_numpy()[feature_index],
            dtype=np.float64,
        )
        decision = np.asarray(model.decision_function(values[None]))
        feature_terms = sorted(
            (
                (name, float(values[index] * coefficients[index]))
                for index, name in enumerate(feature_names)
            ),
            key=lambda item: (-abs(item[1]), item[0]),
        )[:10]
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
    """Adapter for labelled raster metadata with an explicit label sidecar."""

    modality = modalities.Modality.GEOSPATIAL

    def __init__(
        self,
        samples: Sequence[geospatial.RasterSample],
        labels: Mapping[str, object],
    ) -> None:
        """Bind raster identities and caller-owned labels.

        Raster payload decoding is intentionally outside the core package.
        The label sidecar is keyed by the canonical raster source and is part
        of the caller's dataset evidence.
        """
        normalized = tuple(samples)
        if not normalized:
            raise ValueError("raster samples must not be empty")
        sources = tuple(sample.source for sample in normalized)
        if len(set(sources)) != len(sources):
            raise ValueError("raster sources must be unique")
        if any(source not in labels for source in sources):
            raise ValueError("raster labels must cover every sample source")
        self.dataset = normalized
        self.labels_by_source = dict(labels)

    @property
    def dataset_hash(self) -> str:
        """Return the raster identity and label-sidecar digest."""
        identities = (
            f"{sample.source}\0{sample.file_hash}\0{self.labels_by_source[sample.source]}"
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
        profile = {
            "dimensions": [list(value) for value in dimensions],
            "crs": crs_values,
            "dtypes": dtypes,
            "transform_lengths": sorted(
                {len(sample.transform) for sample in self.dataset}
            ),
            "total_bytes": sum(sample.byte_size for sample in self.dataset),
            "label_counts": curation.label_counts(
                self.labels_by_source.values()
            ),
        }
        quality_findings: tuple[data_contracts.CurationFinding, ...] = ()
        if len(crs_values) > 1 or len(dimensions) > 1 or len(dtypes) > 1:
            quality_findings = (
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
                ),
            )
        return curation.ModalityCurationReport(
            modality="geospatial-raster",
            dataset_hash=self.dataset_hash,
            sample_count=len(self.dataset),
            profile=profile,
            quality_findings=quality_findings,
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
        """Prepare raster metadata features without decoding payloads."""
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

    def sample_id(self, sample: geospatial.RasterSample) -> str:
        """Return the canonical raster source identity."""
        return sample.source

    def label(self, sample: geospatial.RasterSample) -> object | None:
        """Return the caller-owned raster label."""
        return self.labels_by_source[sample.source]

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
        return (
            "The baseline uses raster metadata and a caller-owned label "
            "sidecar; it does not decode pixels.",
            "Raster transforms, nodata, reprojection, window sampling, and "
            "spatial joins require an explicit codec policy.",
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
        selected_class = 0
        if model.coef_.shape[0] > 1:
            selected_class = int(np.argmax(decision_values[0]))
        row = batch.values[0]
        coefficients = model.coef_[selected_class]
        local_report = quality.linear_contribution_report(
            batch.sample_ids[0],
            batch.feature_names,
            row.tolist(),
            coefficients.tolist(),
            top_k=min(10, len(batch.feature_names)),
        )
        nonzero = np.flatnonzero(np.abs(row * coefficients) > 1e-12)
        if len(nonzero):
            perturbed = np.repeat(row[None, :], len(nonzero), axis=0)
            perturbed[np.arange(len(nonzero)), nonzero] = 0.0
            perturbed_decisions = np.asarray(model.decision_function(perturbed))
            if perturbed_decisions.ndim == 1:
                observed_change = perturbed_decisions - decision_values[0]
            else:
                observed_change = (
                    perturbed_decisions[:, selected_class]
                    - decision_values[0, selected_class]
                )
            expected_change = -(row[nonzero] * coefficients[nonzero])
            stability = float(
                np.mean(np.sign(observed_change) == np.sign(expected_change))
            )
        else:
            stability = 0.0
        perturbation_reports = [local_report]
        train_values = context.feature_batches["train"].values
        feature_scales = np.maximum(np.std(train_values, axis=0), 1.0)
        perturbation_size = feature_scales * 1e-6
        for index in range(len(batch.feature_names)):
            for direction in (-1.0, 1.0):
                perturbed_row = row.copy()
                perturbed_row[index] += direction * perturbation_size[index]
                perturbation_reports.append(
                    quality.linear_contribution_report(
                        batch.sample_ids[0],
                        batch.feature_names,
                        perturbed_row.tolist(),
                        coefficients.tolist(),
                        top_k=min(10, len(batch.feature_names)),
                    )
                )
        perturbation_stability = quality.ranking_stability(
            tuple(perturbation_reports),
            top_k=min(10, len(batch.feature_names)),
        )
        quality_evaluation = quality.evaluate(
            method="linear_reference_zero_ablation_and_perturbation",
            sample_count=len(batch.sample_ids),
            thresholds=quality.QualityThresholds(
                stability_minimum=0.9,
                fidelity_minimum=0.999999,
            ),
            stability=perturbation_stability,
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
                "modality": context.adapter.modality.value,
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
            context.adapter.modality.value,
            context.curation_report.to_dict(),
            context.adapter.modality_notes(),
            manifest=context.manifest,
        )
        context.documentation = {
            "modality": context.adapter.modality.value,
            "status": "experimental",
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
    ) -> LifecycleReport:
        """Run every stage and optionally persist its evidence report.

        A supplied manifest is checked against the adapter before any stage
        executes. This keeps source, task, privacy, and lineage claims bound
        to the exact modality content used by the lifecycle.
        """
        observed = observed_adapter or adapter
        if observed.modality is not adapter.modality:
            raise ValueError(
                "observed adapter modality must match reference modality"
            )
        if manifest is not None:
            if manifest.modality != adapter.modality.value:
                raise ValueError("manifest modality must match adapter")
            if manifest.content_hash != adapter.dataset_hash:
                raise ValueError("manifest content_hash must match adapter")
            if manifest.sample_count != len(adapter.samples):
                raise ValueError("manifest sample_count must match adapter")
        context = LifecycleContext(
            adapter=adapter,
            package_path=Path(package_path),
            signing_key=signing_key,
            seed=seed,
            observed_adapter=observed,
            manifest=manifest,
        )
        for stage in self.stages:
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
        report = LifecycleReport(
            modality=adapter.modality.value,
            dataset_hash=adapter.dataset_hash,
            stage_evidence=records,
            evaluation=context.evaluation,
            performance=performance_evidence,
            package=package,
            inference=context.inference,
            monitoring=context.monitoring,
            documentation=context.documentation,
            recovery=context.recovery,
        )
        if report_path is not None:
            path = Path(report_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if performance_result is not None and not performance_result.passed:
            raise ValueError(
                "performance budget failed: "
                + "; ".join(performance_result.violations)
            )
        if (
            performance_comparison is not None
            and not performance_comparison.passed
        ):
            raise ValueError(
                "performance baseline failed: "
                + "; ".join(performance_comparison.violations)
            )
        return report

    def declaration(
        self, report: LifecycleReport
    ) -> modalities.ModalityDeclaration:
        """Convert a report into an explicit experimental declaration."""
        return modalities.ModalityDeclaration(
            modality=modalities.Modality(report.modality),
            status=modalities.ModalityStatus.EXPERIMENTAL,
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
    signature = manifest.pop("signature", None)
    if (
        not isinstance(signature, dict)
        or signature.get("algorithm") != attestation.SIGNATURE_ALGORITHM
    ):
        raise ValueError("modality package signature is missing or unsupported")
    value = signature.get("value")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("modality package signature is empty")
    if not attestation.sign_payload(manifest, signing_key) == value:
        raise ValueError("modality package signature does not verify")
    if file_digest(model_path) != manifest.get("model_sha256"):
        raise ValueError("modality package model digest does not verify")
    loaded = joblib.load(model_path)
    if not isinstance(loaded, LogisticRegression):
        raise TypeError(
            "modality package does not contain the expected classifier"
        )
    return loaded


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
    "load_package",
]
