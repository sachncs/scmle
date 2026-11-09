"""Explicit lifecycle contracts for the supported PyTorch model families.

The contracts in this module are deliberately narrower than a general
multimodal foundation-model API.  They describe exactly what the current
worker-backed implementations can prove: finite, dense, two-dimensional
numeric tabular input; local library inference; and feature-level attribution
for the same numeric representation seen by the model.
"""

import dataclasses
from typing import Any
from typing import Self

import numpy as np

from scmle.contracts import capabilities


@dataclasses.dataclass(frozen=True, slots=True)
class CheckpointContract:
    """Serialization boundary for a fitted model family.

    Attributes:
        format: Identifier of the checkpoint format.
        schema_version: Positive version of the checkpoint schema.
        trusted_local_storage: Whether checkpoints may only be loaded from
            trusted local storage.
        atomic_replace: Whether saving replaces the file atomically.
    """

    format: str
    schema_version: int
    trusted_local_storage: bool
    atomic_replace: bool

    def __post_init__(self) -> None:
        """Validates the checkpoint declaration.

        Raises:
            ValueError: If the format is blank or the schema version is below 1.
        """
        if not self.format.strip():
            raise ValueError("checkpoint format must not be empty")
        if self.schema_version < 1:
            raise ValueError("checkpoint schema version must be positive")

    def to_dict(self) -> dict[str, object]:
        """Returns a stable JSON-compatible description."""
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "trusted_local_storage": self.trusted_local_storage,
            "atomic_replace": self.atomic_replace,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class InferenceContract:
    """Input and output behavior promised by a model family.

    Attributes:
        modality: Input modality the family accepts.
        input_rank: Number of dimensions of the input array.
        input_dtype: Name of the dtype the input is converted to.
        methods: Inference methods offered, sorted and de-duplicated.
        output: Description of what the methods return.
        deterministic: Whether repeated inference gives identical output.
    """

    modality: capabilities.Modality
    input_rank: int
    input_dtype: str
    methods: tuple[str, ...]
    output: str
    deterministic: bool

    def __post_init__(self) -> None:
        """Validates the inference declaration.

        Raises:
            ValueError: If the input rank is below 1, the dtype or output is
                blank, or no method (or a blank method) is declared.
        """
        if self.input_rank < 1:
            raise ValueError("inference input rank must be positive")
        if not self.input_dtype.strip() or not self.output.strip():
            raise ValueError("inference dtype and output must not be empty")
        if not self.methods or any(
            not method.strip() for method in self.methods
        ):
            raise ValueError("inference must declare at least one method")
        object.__setattr__(self, "methods", tuple(sorted(set(self.methods))))

    def to_dict(self) -> dict[str, object]:
        """Returns a stable JSON-compatible description."""
        return {
            "modality": self.modality.value,
            "input_rank": self.input_rank,
            "input_dtype": self.input_dtype,
            "methods": list(self.methods),
            "output": self.output,
            "deterministic": self.deterministic,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class ExplanationContract:
    """Local explanation behavior promised by a model family.

    Attributes:
        method: Name of the attribution method.
        scope: Scope of the explanation, for example `local`.
        output: Description of what an explanation contains.
        limitations: Caveats of the method, sorted and de-duplicated.
    """

    method: str
    scope: str
    output: str
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validates the explanation declaration.

        Raises:
            ValueError: If the method, scope or output is blank, or any
                limitation is blank.
        """
        if not self.method.strip() or not self.scope.strip():
            raise ValueError("explanation method and scope must not be empty")
        if not self.output.strip():
            raise ValueError("explanation output must not be empty")
        if any(not item.strip() for item in self.limitations):
            raise ValueError("explanation limitations must not be empty")
        object.__setattr__(
            self, "limitations", tuple(sorted(set(self.limitations)))
        )

    def to_dict(self) -> dict[str, object]:
        """Returns a stable JSON-compatible description."""
        return {
            "method": self.method,
            "scope": self.scope,
            "output": self.output,
            "limitations": list(self.limitations),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class FeatureExplanation:
    """Feature-level attribution returned by a fitted tabular model.

    Attributes:
        values: Attribution matrix of shape `(n_rows, feature_count)`, stored
            as float64.
        method: Name of the attribution method.
        target: What the attribution is taken with respect to.
        feature_count: Number of input features, equal to the column count.
    """

    values: np.ndarray
    method: str
    target: str
    feature_count: int

    def __post_init__(self) -> None:
        """Validates the returned attribution matrix.

        Raises:
            ValueError: If `values` is not two-dimensional, its column count
                differs from `feature_count`, it contains non-finite numbers, or
                the method or target is blank.
        """
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(
                "explanation values must be a two-dimensional matrix"
            )
        if values.shape[1] != self.feature_count:
            raise ValueError("explanation feature count does not match values")
        if not np.isfinite(values).all():
            raise ValueError("explanation values must be finite")
        if not self.method.strip() or not self.target.strip():
            raise ValueError("explanation method and target must not be empty")
        object.__setattr__(self, "values", values)

    def to_dict(self) -> dict[str, object]:
        """Returns JSON-compatible metadata without silently losing values."""
        return {
            "method": self.method,
            "target": self.target,
            "feature_count": self.feature_count,
            "row_count": int(self.values.shape[0]),
            "values": self.values.tolist(),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class ModelFamilyContract:
    """Reusable capability boundary for one deep-learning model family.

    Attributes:
        model_family: Name of the model family.
        status: Support status of the family.
        modalities: Modalities the family accepts, sorted by value.
        required_extras: Optional install extras the family needs.
        checkpoint: Serialization contract.
        inference: Input and output contract.
        explanation: Explanation contract, or None when the family offers no
            explanations.
        limitations: Caveats of the family, sorted and de-duplicated.
    """

    model_family: str
    status: capabilities.CapabilityStatus
    modalities: tuple[capabilities.Modality, ...]
    required_extras: tuple[str, ...]
    checkpoint: CheckpointContract
    inference: InferenceContract
    explanation: ExplanationContract | None
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validates and canonicalizes the family declaration.

        Raises:
            ValueError: If the family name is blank, no modality is declared, or
                an extra or limitation is blank.
            TypeError: If a modality is not a `Modality` value.
        """
        if not self.model_family.strip():
            raise ValueError("model family must not be empty")
        if not self.modalities:
            raise ValueError("model family must declare a modality")
        if any(
            not isinstance(item, capabilities.Modality)
            for item in self.modalities
        ):
            raise TypeError("model family modalities must use Modality values")
        if any(not item.strip() for item in self.required_extras):
            raise ValueError("required extras must not be empty")
        if any(not item.strip() for item in self.limitations):
            raise ValueError("model family limitations must not be empty")
        object.__setattr__(
            self,
            "modalities",
            tuple(sorted(set(self.modalities), key=lambda item: item.value)),
        )
        object.__setattr__(
            self, "required_extras", tuple(sorted(set(self.required_extras)))
        )
        object.__setattr__(
            self, "limitations", tuple(sorted(set(self.limitations)))
        )

    def supports_modality(self, modality: capabilities.Modality | str) -> bool:
        """Returns whether this family accepts the declared modality.

        Args:
            modality: A `Modality` or its string value.

        Returns:
            True if the modality is among the family's declared modalities.
        """
        value = capabilities.Modality(modality)
        return value in self.modalities

    def require_modality(self, modality: capabilities.Modality | str) -> None:
        """Fail closed when a caller requests an undeclared modality.

        Args:
            modality: A `Modality` or its string value.

        Raises:
            ValueError: If the family does not declare the modality.
        """
        value = capabilities.Modality(modality)
        if not self.supports_modality(value):
            supported = ", ".join(item.value for item in self.modalities)
            raise ValueError(
                f"{self.model_family} supports only {supported} modality; "
                f"received {value.value}"
            )

    def validate_features(
        self, features: Any, expected_width: int | None = None
    ) -> np.ndarray:
        """Converts and validates the numeric tabular inference boundary.

        Args:
            features: Dense or sparse array-like of model inputs.
            expected_width: Required number of columns; None skips the check.

        Returns:
            The features as a float32 array with the declared input rank.

        Raises:
            ValueError: If the family has no tabular input contract, the input
                cannot be converted to a numeric array, its rank or width is
                wrong, or it contains non-finite values.
        """
        if not self.supports_modality(capabilities.Modality.TABULAR):
            raise ValueError(
                f"{self.model_family} has no numeric tabular input contract"
            )
        try:
            if hasattr(features, "toarray"):
                features = features.toarray()
            matrix = np.asarray(features, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{self.model_family} requires a finite numeric 2-D matrix"
            ) from error
        if matrix.ndim != self.inference.input_rank:
            raise ValueError(
                f"{self.model_family} requires a {self.inference.input_rank}-D "
                f"numeric matrix, received {matrix.ndim}-D input"
            )
        if expected_width is not None and matrix.shape[1] != expected_width:
            raise ValueError(
                f"{self.model_family} expected {expected_width} features, "
                f"received {matrix.shape[1]}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"{self.model_family} requires finite features")
        return matrix

    def to_dict(self) -> dict[str, object]:
        """Returns the complete declaration for evidence and diagnostics."""
        return {
            "model_family": self.model_family,
            "status": self.status.value,
            "modalities": [item.value for item in self.modalities],
            "required_extras": list(self.required_extras),
            "checkpoint": self.checkpoint.to_dict(),
            "inference": self.inference.to_dict(),
            "explanation": (
                None if self.explanation is None else self.explanation.to_dict()
            ),
            "limitations": list(self.limitations),
        }

    @classmethod
    def unsupported_multimodal(cls) -> Self:
        """Returns the explicit fail-closed boundary for multimodal models."""
        return cls(
            model_family=capabilities.ModelFamily.MULTIMODAL_TRANSFORMER.value,
            status=capabilities.CapabilityStatus.EXTENSION_ONLY,
            modalities=(capabilities.Modality.TABULAR,),
            required_extras=("deeplearning",),
            checkpoint=CheckpointContract(
                "scmle.multimodal-transformer.unsupported", 1, True, True
            ),
            inference=InferenceContract(
                capabilities.Modality.TABULAR,
                2,
                "float32",
                ("reject",),
                "unsupported",
                False,
            ),
            explanation=None,
            limitations=(
                "no text, image, audio, video, graph, or geospatial encoder",
                "no production multimodal fusion or pretrained checkpoint",
                "must not be selected by the project API",
            ),
        )


TABULAR_TRANSFORMER_CONTRACT = ModelFamilyContract(
    model_family=capabilities.ModelFamily.TRANSFORMER.value,
    status=capabilities.CapabilityStatus.EXPERIMENTAL,
    modalities=(capabilities.Modality.TABULAR,),
    required_extras=("deeplearning",),
    checkpoint=CheckpointContract(
        "scmle.tabular-transformer.v1", 1, True, True
    ),
    inference=InferenceContract(
        capabilities.Modality.TABULAR,
        2,
        "float32",
        ("predict", "predict_proba"),
        "prediction or class probability",
        True,
    ),
    explanation=ExplanationContract(
        "gradient_x_input",
        "local",
        "one signed attribution per numeric input feature",
        (
            "attributions are local sensitivity scores, not causal effects",
            "feature names and preprocessing semantics are caller-owned",
        ),
    ),
    limitations=(
        "numeric tabular input only",
        "experimental opt-in only",
        "library inference only",
        "not a text, image, audio, video, or multimodal transformer",
    ),
)


TABULAR_JEPA_CONTRACT = ModelFamilyContract(
    model_family=capabilities.ModelFamily.JEPA.value,
    status=capabilities.CapabilityStatus.EXPERIMENTAL,
    modalities=(capabilities.Modality.TABULAR,),
    required_extras=("deeplearning",),
    checkpoint=CheckpointContract("scmle.tabular-jepa.v2", 1, True, True),
    inference=InferenceContract(
        capabilities.Modality.TABULAR,
        2,
        "float32",
        ("transform", "encode"),
        "context embedding",
        True,
    ),
    explanation=ExplanationContract(
        "embedding_gradient_x_input",
        "local",
        "one signed sensitivity attribution per numeric input feature",
        (
            "attributions describe embedding sensitivity, not label effects",
            "feature names and preprocessing semantics are caller-owned",
        ),
    ),
    limitations=(
        "numeric tabular representation only",
        "experimental opt-in only",
        "not a pretrained or multimodal JEPA",
        "downstream uncertainty is not included in the embedding",
    ),
)


TABULAR_TORCH_CONTRACT = ModelFamilyContract(
    model_family="deep_learning",
    status=capabilities.CapabilityStatus.SUPPORTED,
    modalities=(capabilities.Modality.TABULAR,),
    required_extras=("deeplearning",),
    checkpoint=CheckpointContract("scmle.tabular-torch.v1", 1, True, True),
    inference=InferenceContract(
        capabilities.Modality.TABULAR,
        2,
        "float32",
        ("predict", "predict_proba"),
        "prediction or class probability",
        True,
    ),
    explanation=ExplanationContract(
        "gradient_x_input",
        "local",
        "one signed attribution per numeric input feature",
        (
            "attributions are local sensitivity scores, not causal effects",
            "feature names and preprocessing semantics are caller-owned",
        ),
    ),
    limitations=("numeric tabular input only", "library inference only"),
)


def model_family_contract(architecture: str) -> ModelFamilyContract:
    """Resolves the explicit contract for a worker-backed architecture.

    Args:
        architecture: Architecture name, for example `ft_transformer`.

    Returns:
        The transformer contract for `ft_transformer`, otherwise the generic
        tabular PyTorch contract.
    """
    if architecture == "ft_transformer":
        return TABULAR_TRANSFORMER_CONTRACT
    return TABULAR_TORCH_CONTRACT


__all__ = [
    "TABULAR_JEPA_CONTRACT",
    "TABULAR_TORCH_CONTRACT",
    "TABULAR_TRANSFORMER_CONTRACT",
    "CheckpointContract",
    "ExplanationContract",
    "FeatureExplanation",
    "InferenceContract",
    "ModelFamilyContract",
    "model_family_contract",
]
