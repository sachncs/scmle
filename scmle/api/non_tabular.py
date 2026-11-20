"""Bounded non-tabular modelling through existing adapter representations.

This module is deliberately narrower than :class:`ModalityProject`. It does
not discover decoders, select foundation models, or accept arbitrary adapter
implementations. The public project accepts only the built-in adapters whose
``features`` method already defines a finite representation. Every accepted
path remains experimental until the external release gates are satisfied.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any
from typing import cast
from typing import ClassVar
from typing import Literal
from typing import TYPE_CHECKING

from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import modalities
from scmle.contracts import performance
from scmle.contracts import records
from scmle.pipeline import modality_lifecycle

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy

NON_TABULAR_MODEL_REPORT = "modality_lifecycle.json"
NON_TABULAR_MODEL_DIRECTORY = "model"
NON_TABULAR_MODEL_FAMILY = "regularized_linear_classifier"


class UnsupportedError(errors.ConfigError):
    """Raised when a non-tabular model path cannot cross its gate."""

    category = "non_tabular_model"


@dataclasses.dataclass(frozen=True, slots=True)
class Scope:
    """The bounded representation and claim for one accepted adapter.

    Attributes:
        modality: The modality name.
        representation: How samples are turned into model inputs.
        policy: Who decodes the samples and how.
        family: The model family fitted on the representation.
        status: Always `experimental`.
        limitations: Non-empty statements of what the path does not claim.
        KEYS: Version-1 keys to the written ones (`decoder`, `family`);
            `downgrade` restores version 1.
    """

    KEYS: ClassVar[records.Migration] = records.Migration(
        renames={"decoder_policy": "decoder", "model_family": "family"}
    )

    modality: str
    representation: str
    policy: str
    family: str = NON_TABULAR_MODEL_FAMILY
    status: Literal["experimental"] = "experimental"
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validates the capability record before it is persisted or returned.

        Raises:
            ValueError: If a text field is empty, the status is not
                `experimental`, or the limitations are empty or contain a blank
                entry.
        """
        for name in (
            "modality",
            "representation",
            "policy",
            "family",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise errors.InputError(f"{name} must be a non-empty string")
        if self.status != "experimental":
            raise errors.InputError(
                "non-tabular model paths must remain experimental"
            )
        limitations: tuple[object, ...] = self.limitations
        if not limitations or any(
            not isinstance(value, str) or not value.strip()
            for value in limitations
        ):
            raise errors.InputError(
                "limitations must contain non-empty strings"
            )
        object.__setattr__(self, "limitations", tuple(self.limitations))

    def to_dict(self) -> dict[str, object]:
        """Returns JSON-native capability evidence.

        Returns:
            A dict of all fields under their written keys, with the
            limitations as a list.
        """
        return self.KEYS.upgrade(
            {
                "modality": self.modality,
                "representation": self.representation,
                "decoder_policy": self.policy,
                "model_family": self.family,
                "status": self.status,
                "limitations": list(self.limitations),
            }
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Predictor:
    """Verified model, lifecycle evidence, and representation declaration.

    Attributes:
        report: The verified lifecycle report.
        predictor: The predictor loaded from the signed package.
        capability: The bounded capability the model was built under.
    """

    report: modality_lifecycle.Record
    predictor: modality_lifecycle.Predictor
    capability: Scope

    def __post_init__(self) -> None:
        """Validates the result's package and capability identities.

        Raises:
            ValueError: If the report and capability modalities differ, or the
                predictor was trained on a different dataset than the report.
        """
        if self.report.modality != self.capability.modality:
            raise errors.InputError(
                "report and capability modalities do not match"
            )
        if self.predictor.dataset != self.report.digests["dataset"]:
            raise errors.InputError(
                "predictor and report dataset identities differ"
            )

    @property
    def modality(self) -> str:
        """The declared modality."""
        return self.capability.modality

    @property
    def digest(self) -> str:
        """The immutable training snapshot identity."""
        return self.report.digests["dataset"]

    def predict(self, samples: Sequence[Any]) -> numpy.ndarray:
        """Predicts labels after validating the stored representation schema.

        Args:
            samples: The samples to score, in the adapter's input form.

        Returns:
            The predicted labels, one per sample.
        """
        return self.predictor.predict(samples)

    def predict_proba(self, samples: Sequence[Any]) -> numpy.ndarray:
        """Returns probabilities after validating the stored schema.

        Args:
            samples: The samples to score, in the adapter's input form.

        Returns:
            The class probabilities, one row per sample.
        """
        return self.predictor.predict_proba(samples)

    def health(self) -> dict[str, object]:
        """Returns verified inference health and the capability boundary.

        Returns:
            The predictor's health record plus the support status,
            representation, decoder policy, model family and limitations.
        """
        return {
            **self.predictor.health(),
            "support_status": self.capability.status,
            "representation": self.capability.representation,
            "decoder_policy": self.capability.policy,
            "model_family": self.capability.family,
            "limitations": list(self.capability.limitations),
        }


def qualify(adapter: modality_lifecycle.Footage) -> Scope:
    """Returns the capability of a video adapter.

    Args:
        adapter: The video adapter to describe.

    Returns:
        A scope that depends on whether a frame decoder is set.
    """
    decoder_policy = (
        "caller-owned bounded frame decoder"
        if adapter.decoder is not None
        else "probed container metadata; no frame decoder"
    )
    representation = (
        "bounded_decoded_frame_statistics"
        if adapter.decoder is not None
        else "bounded_container_metadata"
    )
    return Scope(
        modality="video",
        representation=representation,
        policy=decoder_policy,
        limitations=(
            (
                "Metadata-only video does not claim visual modelling; a "
                "frame decoder is required for frame evidence."
            ),
            (
                "This is not audio-video understanding, motion recognition, "
                "or temporal event localization."
            ),
        ),
    )


def delimit(adapter: modality_lifecycle.Imagery) -> Scope:
    """Returns the capability of a raster adapter.

    Args:
        adapter: The raster adapter to describe.

    Returns:
        A scope that depends on whether a raster decoder is set.
    """
    decoder_policy = (
        f"caller-owned raster decoder {adapter.codec}"
        if adapter.decoder is not None
        else "metadata-only raster representation"
    )
    representation = (
        "bounded_decoded_pixel_summary"
        if adapter.decoder is not None
        else "bounded_raster_metadata"
    )
    return Scope(
        modality="raster",
        representation=representation,
        policy=decoder_policy,
        limitations=(
            (
                "Metadata-only rasters do not claim pixel modelling; a "
                "decoder is required for pixel evidence."
            ),
            (
                "This is not raster foundation-model, reprojection, or "
                "spatial causal support."
            ),
        ),
    )


ASSOCIATIONAL = "The model is associational and not a causal explanation."
SCOPES: dict[type, Scope] = {
    modality_lifecycle.Image: Scope(
        modality="image",
        representation="decoded_resized_rgb_pixels",
        policy="built-in bounded RGB decoder",
        limitations=(
            (
                "This is a pixel baseline, not object detection, semantic "
                "segmentation, or validated saliency."
            ),
            ASSOCIATIONAL,
        ),
    ),
    modality_lifecycle.Audio: Scope(
        modality="audio",
        representation="bounded_decoded_wav_signal_statistics",
        policy="built-in WAV decoder; other codecs are rejected",
        limitations=(
            (
                "This representation does not provide transcript, "
                "speaker, or semantic event understanding."
            ),
            ASSOCIATIONAL,
        ),
    ),
    modality_lifecycle.Graph: Scope(
        modality="graph",
        representation="declared_inductive_node_features",
        policy="caller-declared graph nodes and features",
        limitations=(
            (
                "Edges are retained for evidence but are not model "
                "inputs; this is not GNN or topology attribution support."
            ),
            ASSOCIATIONAL,
        ),
    ),
    modality_lifecycle.Geospatial: Scope(
        modality="geospatial",
        representation="declared_crs_geometry_features",
        policy="caller-declared geometries and CRS",
        limitations=(
            (
                "This is a bounded geometry baseline, not a geospatial "
                "foundation model or spatial causal analysis."
            ),
            "CRS and spatial split policy remain domain responsibilities.",
        ),
    ),
}
"""The capability of each exact built-in adapter type whose scope is fixed."""


def invariant(
    adapter: modality_lifecycle.Preparer,
) -> Scope | None:
    """Returns the capability of an adapter whose scope never varies.

    Args:
        adapter: The adapter to describe.

    Returns:
        The capability for a document, text, image, audio, graph or
        geospatial adapter of an exact built-in type, or None for any
        other type.
    """
    adapter_type = type(adapter)
    if adapter_type in {
        modality_lifecycle.Document,
        modality_lifecycle.Plaintext,
    }:
        return Scope(
            modality=modalities.serialise(adapter.modality),
            representation="bounded_character_and_token_counts",
            policy="caller-decoded text or document samples",
            limitations=(
                (
                    "This is a lexical baseline, not OCR, layout, table, or "
                    "semantic document understanding."
                ),
                ASSOCIATIONAL,
            ),
        )
    return SCOPES.get(adapter_type)


def entitle(
    adapter: modality_lifecycle.Preparer,
) -> Scope:
    """Returns the explicit capability for one built-in adapter.

    Exact built-in types are required. A caller-defined subclass can change
    validation, splitting, feature construction, or decoder semantics and is
    therefore rejected by this production-oriented boundary. Such adapters
    remain usable through the broader experimental lifecycle.

    Args:
        adapter: The adapter to describe.

    Returns:
        The capability of the adapter's exact type. For video and raster
        adapters the representation and decoder policy depend on whether a
        frame decoder or raster decoder is set (decoded statistics) or not
        (container or raster metadata only).

    Raises:
        TypeError: If ``adapter`` is not a ``Preparer``.
        UnsupportedNonTabularModelError: If its exact type is not allowlisted.
    """
    if not isinstance(adapter, modality_lifecycle.Preparer):
        raise errors.KindError("adapter must be a Preparer")
    adapter_type = type(adapter)
    if adapter_type is modality_lifecycle.Footage:
        return qualify(cast("modality_lifecycle.Footage", adapter))
    if adapter_type is modality_lifecycle.Imagery:
        return delimit(cast("modality_lifecycle.Imagery", adapter))
    scope = invariant(adapter)
    if scope is not None:
        return scope
    raise UnsupportedError(
        "adapter type is not allowlisted for bounded non-tabular modelling; "
        "exact built-in Preparer types are required",
        {
            "adapter_type": (
                f"{adapter_type.__module__}.{adapter_type.__qualname__}"
            ),
            "modality": modalities.serialise(adapter.modality),
            "supported_boundary": "exact built-in Preparer types",
        },
    )


def conform(
    adapter: modality_lifecycle.Preparer,
    observed_adapter: modality_lifecycle.Preparer,
    capability: Scope,
) -> None:
    """Checks that an observed adapter matches the reference adapter.

    Args:
        adapter: The reference adapter holding the training samples.
        observed_adapter: The adapter of later observed samples.
        capability: The capability of the reference adapter.

    Raises:
        UnsupportedNonTabularModelError: If the observed adapter has another
            capability or, for video and raster adapters, another decoder
            object.
    """
    observed_capability = entitle(observed_adapter)
    if observed_capability != capability:
        raise UnsupportedError(
            "observed adapter must use the same bounded capability",
            {
                "reference": capability.to_dict(),
                "observed": observed_capability.to_dict(),
            },
        )
    if isinstance(adapter, modality_lifecycle.Footage) and (
        adapter.decoder
        is not cast("modality_lifecycle.Footage", observed_adapter).decoder
    ):
        raise UnsupportedError(
            "reference and observed video adapters must share the "
            "same frame decoder object",
            {"modality": capability.modality},
        )
    if isinstance(adapter, modality_lifecycle.Imagery) and (
        adapter.decoder
        is not cast("modality_lifecycle.Imagery", observed_adapter).decoder
    ):
        raise UnsupportedError(
            "reference and observed raster adapters must share the "
            "same decoder object",
            {"modality": capability.modality},
        )


def vet(
    signing_key: str | bytes,
    seed: int,
    require_independent_ground_truth: bool,
    resume: bool,
) -> None:
    """Checks the run arguments every modality lifecycle surface shares.

    Args:
        signing_key: Key that authenticates the package.
        seed: Seed of the split and the model.
        require_independent_ground_truth: Whether an independent explanation
            reference suite is required.
        resume: Whether to resume from the checkpoint.

    Raises:
        ValueError: If the signing key is empty.
        TypeError: If the seed is not an integer (a bool is refused) or a
            flag is not a bool.
    """
    if not signing_key:
        raise errors.InputError("signing_key must not be empty")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise errors.KindError("seed must be an integer")
    if not isinstance(require_independent_ground_truth, bool):
        raise errors.KindError("require_independent_ground_truth must be bool")
    if not isinstance(resume, bool):
        raise errors.KindError("resume must be bool")


def consent(experimental_opt_in: bool, capability: Scope) -> None:
    """Checks that the caller acknowledged the experimental status.

    Args:
        experimental_opt_in: The caller's acknowledgement.
        capability: The capability the run would be built under.

    Raises:
        TypeError: If the acknowledgement is not a bool.
        UnsupportedNonTabularModelError: If it is not given.
    """
    if not isinstance(experimental_opt_in, bool):
        raise errors.KindError("experimental_opt_in must be bool")
    if not experimental_opt_in:
        raise UnsupportedError(
            "non-tabular modelling is experimental; set "
            "experimental_opt_in=True explicitly",
            {"modality": capability.modality, "status": capability.status},
        )


class Modeler:
    """Runs a bounded, explicitly experimental non-tabular model lifecycle.

    This project is production-oriented in its artifact and failure boundary,
    not production-supported in its model claims. It requires an explicit
    ``experimental_opt_in`` because the repository has not completed the
    independent review, external fault, hosted serving, and modality-specific
    fidelity gates needed for a supported production declaration.

    Attributes:
        capability: The bounded capability of the adapter.
        adapter: The adapter holding the training samples.
        key: HMAC key that signs the package and the report.
        directory: Directory for the report and the package.
        seed: Random seed.
        observed: Adapter holding later observed samples, if any.
        budget: Performance limits to enforce, if any.
        baseline: Performance baseline to compare with, if any.
        manifest: The modality manifest, if any.
        independent: Whether independent ground truth is
            required.
        cancellation: Cancellation signal checked by the lifecycle.
        checkpoint: Where progress is checkpointed, if anywhere.
        resume: Whether to resume from the checkpoint.
        lifecycle: The lifecycle that runs the model.
        result: The latest result run or loaded, or None.
    """

    def __init__(
        self,
        adapter: modality_lifecycle.Preparer,
        key: str | bytes,
        *,
        experimental_opt_in: bool = False,
        directory: str | Path = "runs",
        seed: int = 42,
        observed: modality_lifecycle.Preparer | None = None,
        budget: performance.Budget | None = None,
        baseline: performance.Baseline | None = None,
        manifest: data.Passport | None = None,
        independent: bool = False,
        cancellation: modality_lifecycle.Cancellation | None = None,
        checkpoint: str | Path | None = None,
        resume: bool = False,
    ) -> None:
        """Defines a bounded non-tabular model run.

        Args:
            adapter: Exact built-in adapter holding the training samples.
            key: Non-empty HMAC key that signs the package and report.
            experimental_opt_in: Must be True to acknowledge the experimental
                status.
            directory: Directory for the report and the package; its `model`
                subdirectory must not already exist when the lifecycle runs.
            seed: Random seed (an integer, not a bool).
            observed: Adapter of later observed samples; it must have
                the same capability and, for video and raster adapters, share
                the decoder object of `adapter`.
            budget: Performance limits to enforce.
            baseline: Performance baseline to compare with.
            manifest: The modality manifest.
            independent: Whether independent ground truth
                is required.
            cancellation: Cancellation signal; a new one if omitted.
            checkpoint: Where to checkpoint progress.
            resume: Whether to resume from the checkpoint.

        Raises:
            UnsupportedNonTabularModelError: If opt-in is absent, the adapter is
                outside the exact built-in boundary, or the observed adapter
                differs in capability or decoder.
            TypeError: If an adapter is not a `Preparer`, or a flag, the
                seed or the cancellation has the wrong type.
            ValueError: If the signing key is empty.
        """
        self.capability = entitle(adapter)
        consent(experimental_opt_in, self.capability)
        vet(key, seed, independent, resume)
        if observed is not None:
            conform(adapter, observed, self.capability)
        if cancellation is not None and not isinstance(
            cancellation, modality_lifecycle.Cancellation
        ):
            raise errors.KindError("cancellation must be a Cancellation")
        self.adapter = adapter
        self.key = key
        self.directory = Path(directory)
        self.seed = seed
        self.observed = observed
        self.budget = budget
        self.baseline = baseline
        self.manifest = manifest
        self.independent = independent
        self.cancellation = cancellation or modality_lifecycle.Cancellation()
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        self.resume = resume
        self.lifecycle = modality_lifecycle.Lifecycle()
        self.result: Predictor | None = None

    @property
    def report(self) -> Path:
        """The signed lifecycle report location."""
        return self.directory / NON_TABULAR_MODEL_REPORT

    @property
    def package(self) -> Path:
        """The signed model package location."""
        return self.directory / NON_TABULAR_MODEL_DIRECTORY

    def run(self) -> Predictor:
        """Runs, verifies, and returns the complete bounded lifecycle.

        The lifecycle writes the signed report and package under `directory`.
        Its failures, including cancellation, propagate unchanged.

        Returns:
            The verified result, loaded back from the written report and
            package.

        Raises:
            ValueError: If `directory` exists and is not a directory. Loading
                the result back raises it too when the report does not match
                the project.
        """
        if self.directory.exists() and not self.directory.is_dir():
            raise errors.InputError("output_dir must be a directory")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lifecycle.run(
            self.adapter,
            self.package,
            self.key,
            seed=self.seed,
            observed_adapter=self.observed,
            report_path=self.report,
            performance_budget=self.budget,
            performance_baseline=self.baseline,
            manifest=self.manifest,
            require_independent_ground_truth=(self.independent),
            cancellation_check=self.cancellation.is_cancelled,
            checkpoint_path=self.checkpoint,
            resume=self.resume,
        )
        return self.recall()

    def recall(self) -> Predictor:
        """Loads and verifies the signed report and package for this project.

        Returns:
            The verified result, which is also kept as `result`.

        Raises:
            ValueError: If the report's modality or dataset does not match the
                project.
        """
        report = modality_lifecycle.Record.load(self.report, self.key)
        if report.modality != self.capability.modality:
            raise errors.InputError(
                "lifecycle report modality does not match project"
            )
        if report.digests["dataset"] != self.adapter.digest:
            raise errors.InputError(
                "lifecycle report dataset does not match project"
            )
        predictor = modality_lifecycle.Predictor.load(
            self.adapter,
            report.package.path,
            self.key,
            lifecycle_report_path=self.report,
        )
        self.result = Predictor(
            report=report,
            predictor=predictor,
            capability=self.capability,
        )
        return self.result

    def cancel(self) -> None:
        """Requests cancellation at the next lifecycle check."""
        self.cancellation.cancel()


__all__ = [
    "NON_TABULAR_MODEL_DIRECTORY",
    "NON_TABULAR_MODEL_FAMILY",
    "NON_TABULAR_MODEL_REPORT",
    "Modeler",
    "Predictor",
    "Scope",
    "UnsupportedError",
    "entitle",
]
