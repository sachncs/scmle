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
    """The bounded representation and claim for one accepted adapter.

    Attributes:
        modality: The modality name.
        representation: How samples are turned into model inputs.
        decoder_policy: Who decodes the samples and how.
        model_family: The model family fitted on the representation.
        status: Always `experimental`.
        limitations: Non-empty statements of what the path does not claim.
    """

    modality: str
    representation: str
    decoder_policy: str
    model_family: str = NON_TABULAR_MODEL_FAMILY
    status: Literal["experimental"] = "experimental"
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the capability record before it is persisted or returned.

        Raises:
            ValueError: If a text field is empty, the status is not
                `experimental`, or the limitations are empty or contain a blank
                entry.
        """
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
        """Return JSON-native capability evidence.

        Returns:
            A dict of all fields, with the limitations as a list.
        """
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
    """Verified model, lifecycle evidence, and representation declaration.

    Attributes:
        report: The verified lifecycle report.
        predictor: The predictor loaded from the signed package.
        capability: The bounded capability the model was built under.
    """

    report: modality_lifecycle.LifecycleReport
    predictor: modality_lifecycle.ModalityPredictor
    capability: NonTabularModelCapability

    def __post_init__(self) -> None:
        """Validate the result's package and capability identities.

        Raises:
            ValueError: If the report and capability modalities differ, or the
                predictor was trained on a different dataset than the report.
        """
        if self.report.modality != self.capability.modality:
            raise ValueError("report and capability modalities do not match")
        if self.predictor.training_dataset_hash != self.report.dataset_hash:
            raise ValueError("predictor and report dataset identities differ")

    @property
    def modality(self) -> str:
        """The declared modality."""
        return self.capability.modality

    @property
    def dataset_hash(self) -> str:
        """The immutable training snapshot identity."""
        return self.report.dataset_hash

    def predict(self, samples: Sequence[Any]) -> np.ndarray:
        """Predict labels after validating the stored representation schema.

        Args:
            samples: The samples to score, in the adapter's input form.

        Returns:
            The predicted labels, one per sample.
        """
        return self.predictor.predict(samples)

    def predict_proba(self, samples: Sequence[Any]) -> np.ndarray:
        """Return probabilities after validating the stored schema.

        Args:
            samples: The samples to score, in the adapter's input form.

        Returns:
            The class probabilities, one row per sample.
        """
        return self.predictor.predict_proba(samples)

    def health(self) -> dict[str, object]:
        """Return verified inference health and the capability boundary.

        Returns:
            The predictor's health record plus the support status,
            representation, decoder policy, model family and limitations.
        """
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

    Args:
        adapter: The adapter to describe.

    Returns:
        The capability of the adapter's exact type, including whether a video
        frame decoder or raster decoder makes the representation pixel- or
        frame-based or metadata-only.

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

    Attributes:
        capability: The bounded capability of the adapter.
        adapter: The adapter holding the training samples.
        signing_key: HMAC key that signs the package and the report.
        output_dir: Directory for the report and the package.
        seed: Random seed.
        observed_adapter: Adapter holding later observed samples, if any.
        performance_budget: Performance limits to enforce, if any.
        performance_baseline: Performance baseline to compare with, if any.
        manifest: The modality manifest, if any.
        require_independent_ground_truth: Whether independent ground truth is
            required.
        cancellation: Cancellation signal checked by the lifecycle.
        checkpoint_path: Where progress is checkpointed, if anywhere.
        resume: Whether to resume from the checkpoint.
        lifecycle: The lifecycle that runs the model.
        result: The latest result run or loaded, or None.
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

        Args:
            adapter: Exact built-in adapter holding the training samples.
            signing_key: Non-empty HMAC key that signs the package and report.
            experimental_opt_in: Must be True to acknowledge the experimental
                status.
            output_dir: Directory for the report and the package.
            seed: Random seed (an integer, not a bool).
            observed_adapter: Adapter of later observed samples; it must have
                the same capability and share the decoder object of `adapter`.
            performance_budget: Performance limits to enforce.
            performance_baseline: Performance baseline to compare with.
            manifest: The modality manifest.
            require_independent_ground_truth: Whether independent ground truth
                is required.
            cancellation: Cancellation signal; a new one if omitted.
            checkpoint_path: Where to checkpoint progress.
            resume: Whether to resume from the checkpoint.

        Raises:
            UnsupportedNonTabularModelError: If opt-in is absent, the adapter is
                outside the exact built-in boundary, or the observed adapter
                differs in capability or decoder.
            TypeError: If a flag, the seed or the cancellation has the wrong
                type.
            ValueError: If the signing key is empty.
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
        """The signed lifecycle report location."""
        return self.output_dir / NON_TABULAR_MODEL_REPORT

    @property
    def package_path(self) -> Path:
        """The signed model package location."""
        return self.output_dir / NON_TABULAR_MODEL_DIRECTORY

    def run(self) -> NonTabularModelResult:
        """Run, verify, and return the complete bounded lifecycle.

        Returns:
            The verified result, loaded back from the written report and
            package.

        Raises:
            ValueError: If `output_dir` exists and is not a directory.
        """
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
        """Load and verify the signed report and package for this project.

        Returns:
            The verified result, which is also kept as `result`.

        Raises:
            ValueError: If the report's modality or dataset does not match the
                project.
        """
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
        """Request cancellation at the next lifecycle check."""
        self.cancellation.cancel()

    @classmethod
    def capability_for_adapter(
        cls, adapter: modality_lifecycle.ModalityAdapter
    ) -> NonTabularModelCapability:
        """Return the capability without opting into a model run.

        Args:
            adapter: The adapter to describe.

        Returns:
            The capability of the adapter's exact type.
        """
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
