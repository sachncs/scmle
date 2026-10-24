"""Bounded non-tabular modelling through existing adapter representations.

This module is deliberately narrower than :class:`ModalityProject`. It does
not discover decoders, select foundation models, or accept arbitrary adapter
implementations. The public project accepts only the built-in adapters whose
``features`` method already defines a finite representation. Every accepted
path remains experimental until the external release gates are satisfied.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from pathlib import Path
from typing import Any
from typing import cast
from typing import Literal

import numpy as np

from scmle.contracts import data as data_contracts
from scmle.contracts import errors
from scmle.contracts import modalities
from scmle.contracts import performance
from scmle.models import modality_lifecycle

NON_TABULAR_MODEL_REPORT = "modality_lifecycle.json"
NON_TABULAR_MODEL_DIRECTORY = "model"
NON_TABULAR_MODEL_FAMILY = "regularized_linear_classifier"


class UnsupportedNonTabularModelError(errors.ConfigError):
    """Raised when a non-tabular model path cannot cross its gate."""

    category = "non_tabular_model"


@dataclasses.dataclass(frozen=True, slots=True)
class NonTabularModelCapability:
    """The bounded representation and claim for one accepted adapter."""

    modality: str
    representation: str
    decoder_policy: str
    model_family: str = NON_TABULAR_MODEL_FAMILY
    status: Literal["experimental"] = "experimental"
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the capability record before it is persisted or returned."""
        for name in (
            "modality",
            "representation",
            "decoder_policy",
            "model_family",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.status != "experimental":
            raise ValueError("non-tabular model paths must remain experimental")
        if not self.limitations or any(
            not isinstance(value, str) or not value.strip()
            for value in self.limitations
        ):
            raise ValueError("limitations must contain non-empty strings")
        object.__setattr__(self, "limitations", tuple(self.limitations))

    def to_dict(self) -> dict[str, object]:
        """Return JSON-native capability evidence."""
        return {
            "modality": self.modality,
            "representation": self.representation,
            "decoder_policy": self.decoder_policy,
            "model_family": self.model_family,
            "status": self.status,
            "limitations": list(self.limitations),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class NonTabularModelResult:
    """Verified model, lifecycle evidence, and representation declaration."""

    report: modality_lifecycle.LifecycleReport
    predictor: modality_lifecycle.ModalityPredictor
    capability: NonTabularModelCapability

    def __post_init__(self) -> None:
        """Validate the result's package and capability identities."""
        if self.report.modality != self.capability.modality:
            raise ValueError("report and capability modalities do not match")
        if self.predictor.training_dataset_hash != self.report.dataset_hash:
            raise ValueError("predictor and report dataset identities differ")

    @property
    def modality(self) -> str:
        """Return the declared modality."""
        return self.capability.modality

    @property
    def dataset_hash(self) -> str:
        """Return the immutable training snapshot identity."""
        return self.report.dataset_hash

    def predict(self, samples: Sequence[Any]) -> np.ndarray:
        """Predict labels after validating the stored representation schema."""
        return self.predictor.predict(samples)

    def predict_proba(self, samples: Sequence[Any]) -> np.ndarray:
        """Return probabilities after validating the stored schema."""
        return self.predictor.predict_proba(samples)

    def health(self) -> dict[str, object]:
        """Return verified inference health and the capability boundary."""
        return {
            **self.predictor.health(),
            "support_status": self.capability.status,
            "representation": self.capability.representation,
            "decoder_policy": self.capability.decoder_policy,
            "model_family": self.capability.model_family,
            "limitations": list(self.capability.limitations),
        }


def capability_for_adapter(
    adapter: modality_lifecycle.ModalityAdapter,
) -> NonTabularModelCapability:
    """Return the explicit capability for one built-in adapter.

    Exact built-in types are required. A caller-defined subclass can change
    validation, splitting, feature construction, or decoder semantics and is
    therefore rejected by this production-oriented boundary. Such adapters
    remain usable through the broader experimental lifecycle.

    Raises:
        TypeError: If ``adapter`` is not a modelling adapter.
        UnsupportedNonTabularModelError: If its exact type is not allowlisted.
    """
    if not isinstance(adapter, modality_lifecycle.ModalityAdapter):
        raise TypeError("adapter must be a ModalityAdapter")
    adapter_type = type(adapter)
    if adapter_type in {
        modality_lifecycle.DocumentAdapter,
        modality_lifecycle.TextAdapter,
    }:
        return NonTabularModelCapability(
            modality=modalities.modality_value(adapter.modality),
            representation="bounded_character_and_token_counts",
            decoder_policy="caller-decoded text or document samples",
            limitations=(
                "This is a lexical baseline, not OCR, layout, table, or "
                "semantic document understanding.",
                "The model is associational and not a causal explanation.",
            ),
        )
    if adapter_type is modality_lifecycle.ImageAdapter:
        return NonTabularModelCapability(
            modality="image",
            representation="decoded_resized_rgb_pixels",
            decoder_policy="built-in bounded RGB decoder",
            limitations=(
                "This is a pixel baseline, not object detection, semantic "
                "segmentation, or validated saliency.",
                "The model is associational and not a causal explanation.",
            ),
        )
    if adapter_type is modality_lifecycle.AudioAdapter:
        return NonTabularModelCapability(
            modality="audio",
            representation="bounded_decoded_wav_signal_statistics",
            decoder_policy="built-in WAV decoder; other codecs are rejected",
            limitations=(
                "This representation does not provide transcript, speaker, "
                "or semantic event understanding.",
                "The model is associational and not a causal explanation.",
            ),
        )
    if adapter_type is modality_lifecycle.VideoAdapter:
        video_adapter = cast(modality_lifecycle.VideoAdapter, adapter)
        decoder_policy = (
            "caller-owned bounded frame decoder"
            if video_adapter.frame_decoder is not None
            else "probed container metadata; no frame decoder"
        )
        representation = (
            "bounded_decoded_frame_statistics"
            if video_adapter.frame_decoder is not None
            else "bounded_container_metadata"
        )
        return NonTabularModelCapability(
            modality="video",
            representation=representation,
            decoder_policy=decoder_policy,
            limitations=(
                "Metadata-only video does not claim visual modelling; a "
                "frame decoder is required for frame evidence.",
                "This is not audio-video understanding, motion recognition, "
                "or temporal event localization.",
            ),
        )
    if adapter_type is modality_lifecycle.GraphAdapter:
        return NonTabularModelCapability(
            modality="graph",
            representation="declared_inductive_node_features",
            decoder_policy="caller-declared graph nodes and features",
            limitations=(
                "Edges are retained for evidence but are not model inputs; "
                "this is not GNN or topology attribution support.",
                "The model is associational and not a causal explanation.",
            ),
        )
    if adapter_type is modality_lifecycle.GeospatialAdapter:
        return NonTabularModelCapability(
            modality="geospatial",
            representation="declared_crs_geometry_features",
            decoder_policy="caller-declared geometries and CRS",
            limitations=(
                "This is a bounded geometry baseline, not a geospatial "
                "foundation model or spatial causal analysis.",
                "CRS and spatial split policy remain domain responsibilities.",
            ),
        )
    if adapter_type is modality_lifecycle.RasterAdapter:
        raster_adapter = cast(modality_lifecycle.RasterAdapter, adapter)
        decoder_policy = (
            f"caller-owned raster decoder {raster_adapter.decoder_name}"
            if raster_adapter.decoder is not None
            else "metadata-only raster representation"
        )
        representation = (
            "bounded_decoded_pixel_summary"
            if raster_adapter.decoder is not None
            else "bounded_raster_metadata"
        )
        return NonTabularModelCapability(
            modality="raster",
            representation=representation,
            decoder_policy=decoder_policy,
            limitations=(
                "Metadata-only rasters do not claim pixel modelling; a "
                "decoder is required for pixel evidence.",
                "This is not raster foundation-model, reprojection, or "
                "spatial causal support.",
            ),
        )
    raise UnsupportedNonTabularModelError(
        "adapter type is not allowlisted for bounded non-tabular modelling; "
        "exact built-in ModalityAdapter types are required",
        {
            "adapter_type": (
                f"{adapter_type.__module__}.{adapter_type.__qualname__}"
            ),
            "modality": modalities.modality_value(adapter.modality),
            "supported_boundary": "exact built-in ModalityAdapter types",
        },
    )


class NonTabularModelProject:
    """Run a bounded, explicitly experimental non-tabular model lifecycle.

    This project is production-oriented in its artifact and failure boundary,
    not production-supported in its model claims. It requires an explicit
    ``experimental_opt_in`` because the repository has not completed the
    independent review, external fault, hosted serving, and modality-specific
    fidelity gates needed for a supported production declaration.
    """

    def __init__(
        self,
        adapter: modality_lifecycle.ModalityAdapter,
        signing_key: str | bytes,
        *,
        experimental_opt_in: bool = False,
        output_dir: str | Path = "runs",
        seed: int = 42,
        observed_adapter: modality_lifecycle.ModalityAdapter | None = None,
        performance_budget: performance.PerformanceBudget | None = None,
        performance_baseline: performance.PerformanceBaseline | None = None,
        manifest: data_contracts.ModalityManifest | None = None,
        require_independent_ground_truth: bool = False,
        cancellation: modality_lifecycle.LifecycleCancellation | None = None,
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
    ) -> None:
        """Define a bounded non-tabular model run.

        Raises:
            UnsupportedNonTabularModelError: If opt-in is absent or the
                adapter is outside the exact built-in boundary.
        """
        self.capability = capability_for_adapter(adapter)
        if not isinstance(experimental_opt_in, bool):
            raise TypeError("experimental_opt_in must be bool")
        if not experimental_opt_in:
            raise UnsupportedNonTabularModelError(
                "non-tabular modelling is experimental; set "
                "experimental_opt_in=True explicitly",
                {
                    "modality": self.capability.modality,
                    "status": self.capability.status,
                },
            )
        if not signing_key:
            raise ValueError("signing_key must not be empty")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if not isinstance(require_independent_ground_truth, bool):
            raise TypeError("require_independent_ground_truth must be bool")
        if not isinstance(resume, bool):
            raise TypeError("resume must be bool")
        if observed_adapter is not None:
            observed_capability = capability_for_adapter(observed_adapter)
            if observed_capability != self.capability:
                raise UnsupportedNonTabularModelError(
                    "observed adapter must use the same bounded capability",
                    {
                        "reference": self.capability.to_dict(),
                        "observed": observed_capability.to_dict(),
                    },
                )
            if isinstance(adapter, modality_lifecycle.VideoAdapter) and (
                adapter.frame_decoder
                is not cast(
                    modality_lifecycle.VideoAdapter, observed_adapter
                ).frame_decoder
            ):
                raise UnsupportedNonTabularModelError(
                    "reference and observed video adapters must share the "
                    "same frame decoder object",
                    {"modality": self.capability.modality},
                )
            if isinstance(adapter, modality_lifecycle.RasterAdapter) and (
                adapter.decoder
                is not cast(
                    modality_lifecycle.RasterAdapter, observed_adapter
                ).decoder
            ):
                raise UnsupportedNonTabularModelError(
                    "reference and observed raster adapters must share the "
                    "same decoder object",
                    {"modality": self.capability.modality},
                )
        if cancellation is not None and not isinstance(
            cancellation, modality_lifecycle.LifecycleCancellation
        ):
            raise TypeError("cancellation must be a LifecycleCancellation")
        self.adapter = adapter
        self.signing_key = signing_key
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.observed_adapter = observed_adapter
        self.performance_budget = performance_budget
        self.performance_baseline = performance_baseline
        self.manifest = manifest
        self.require_independent_ground_truth = require_independent_ground_truth
        self.cancellation = (
            cancellation or modality_lifecycle.LifecycleCancellation()
        )
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else None
        )
        self.resume = resume
        self.lifecycle = modality_lifecycle.ModalityLifecycle()
        self.result: NonTabularModelResult | None = None

    @property
    def report_path(self) -> Path:
        """Return the signed lifecycle report location."""
        return self.output_dir / NON_TABULAR_MODEL_REPORT

    @property
    def package_path(self) -> Path:
        """Return the signed model package location."""
        return self.output_dir / NON_TABULAR_MODEL_DIRECTORY

    def run(self) -> NonTabularModelResult:
        """Run, verify, and return the complete bounded lifecycle."""
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise ValueError("output_dir must be a directory")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lifecycle.run(
            self.adapter,
            self.package_path,
            self.signing_key,
            seed=self.seed,
            observed_adapter=self.observed_adapter,
            report_path=self.report_path,
            performance_budget=self.performance_budget,
            performance_baseline=self.performance_baseline,
            manifest=self.manifest,
            require_independent_ground_truth=(
                self.require_independent_ground_truth
            ),
            cancellation_check=self.cancellation.is_cancelled,
            checkpoint_path=self.checkpoint_path,
            resume=self.resume,
        )
        return self.load_result()

    def load_result(self) -> NonTabularModelResult:
        """Load and verify the signed report and package for this project."""
        report = modality_lifecycle.LifecycleReport.read_verified(
            self.report_path, self.signing_key
        )
        if report.modality != self.capability.modality:
            raise ValueError("lifecycle report modality does not match project")
        if report.dataset_hash != self.adapter.dataset_hash:
            raise ValueError("lifecycle report dataset does not match project")
        predictor = modality_lifecycle.ModalityPredictor.from_package(
            self.adapter,
            report.package.path,
            self.signing_key,
            lifecycle_report_path=self.report_path,
        )
        self.result = NonTabularModelResult(
            report=report,
            predictor=predictor,
            capability=self.capability,
        )
        return self.result

    def cancel(self) -> None:
        """Request cancellation at the next lifecycle stage boundary."""
        self.cancellation.cancel()

    @classmethod
    def capability_for_adapter(
        cls, adapter: modality_lifecycle.ModalityAdapter
    ) -> NonTabularModelCapability:
        """Return the capability without opting into a model run."""
        return capability_for_adapter(adapter)


__all__ = [
    "NON_TABULAR_MODEL_DIRECTORY",
    "NON_TABULAR_MODEL_FAMILY",
    "NON_TABULAR_MODEL_REPORT",
    "NonTabularModelCapability",
    "NonTabularModelProject",
    "NonTabularModelResult",
    "UnsupportedNonTabularModelError",
    "capability_for_adapter",
]
