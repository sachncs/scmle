"""Explicit lifecycle contracts for the supported PyTorch model families.

The contracts in this module are deliberately narrower than a general
multimodal foundation-model API.  They describe exactly what the current
worker-backed implementations can prove: finite, dense, two-dimensional
numeric tabular input; local library inference; and feature-level attribution
for the same numeric representation seen by the model.
"""

import dataclasses
from typing import Any
from typing import ClassVar

import numpy

from scmle.contracts import capabilities
from scmle.contracts import errors
from scmle.contracts import records

ATTRIBUTION_MATRIX_RANK = 2


@dataclasses.dataclass(frozen=True, slots=True)
class Checkpoint:
    """Serialization boundary for a fitted model family.

    Attributes:
        format: Identifier of the checkpoint format.
        schema_version: Positive version of the checkpoint schema.
        trusted: Whether checkpoints may only be loaded from
            trusted local storage.
        replace: Whether saving replaces the file atomically.
        migrations: The written layout's version steps, oldest first;
            version 2 writes `trusted` and `atomic`. `schema_version` is the
            checkpoint file's version, not this declaration's.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={
                "trusted_local_storage": "trusted",
                "atomic_replace": "atomic",
            }
        ),
    )

    format: str
    schema_version: int
    trusted: bool
    replace: bool

    def __post_init__(self) -> None:
        """Validates the checkpoint declaration.

        Raises:
            ValueError: If the format is blank or the schema version is below 1.
        """
        if not self.format.strip():
            raise errors.InputError("checkpoint format must not be empty")
        if self.schema_version < 1:
            raise errors.InputError(
                "checkpoint schema version must be positive"
            )

    def to_dict(self) -> dict[str, object]:
        """Returns a stable JSON-compatible description."""
        return records.upgrade(
            self.migrations,
            {
                "format": self.format,
                "schema_version": self.schema_version,
                "trusted_local_storage": self.trusted,
                "atomic_replace": self.replace,
            },
            records.ORIGIN,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Contract:
    """Input and output behavior promised by a model family.

    Attributes:
        modality: Input modality the family accepts.
        rank: Number of dimensions of the input array.
        dtype: Name of the dtype the input is converted to.
        methods: Inference methods offered, sorted and de-duplicated.
        output: Description of what the methods return.
        deterministic: Whether repeated inference gives identical output.
        migrations: The written layout's version steps, oldest first;
            version 2 writes `rank` and `dtype`.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={"input_rank": "rank", "input_dtype": "dtype"}
        ),
    )

    modality: capabilities.Modality
    rank: int
    dtype: str
    methods: tuple[str, ...]
    output: str
    deterministic: bool

    def __post_init__(self) -> None:
        """Validates the inference declaration.

        Raises:
            ValueError: If the input rank is below 1, the dtype or output is
                blank, or no method (or a blank method) is declared.
        """
        if self.rank < 1:
            raise errors.InputError("inference input rank must be positive")
        if not self.dtype.strip() or not self.output.strip():
            raise errors.InputError(
                "inference dtype and output must not be empty"
            )
        if not self.methods or any(
            not method.strip() for method in self.methods
        ):
            raise errors.InputError(
                "inference must declare at least one method"
            )
        object.__setattr__(self, "methods", tuple(sorted(set(self.methods))))

    def to_dict(self) -> dict[str, object]:
        """Returns a stable JSON-compatible description."""
        return records.upgrade(
            self.migrations,
            {
                "modality": self.modality.value,
                "input_rank": self.rank,
                "input_dtype": self.dtype,
                "methods": list(self.methods),
                "output": self.output,
                "deterministic": self.deterministic,
            },
            records.ORIGIN,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Explainability:
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
            raise errors.InputError(
                "explanation method and scope must not be empty"
            )
        if not self.output.strip():
            raise errors.InputError("explanation output must not be empty")
        if any(not item.strip() for item in self.limitations):
            raise errors.InputError("explanation limitations must not be empty")
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
class Contribution:
    """Feature-level attribution returned by a fitted tabular model.

    Attributes:
        values: Attribution matrix of shape `(n_rows, feature_count)`, stored
            as float64.
        method: Name of the attribution method.
        target: What the attribution is taken with respect to.
        features: Number of input features, equal to the column count.
        migrations: The written layout's version steps, oldest first;
            version 2 writes `features` as `features`.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(renames={"feature_count": "features"}),
    )

    values: numpy.ndarray
    method: str
    target: str
    features: int

    def __post_init__(self) -> None:
        """Validates the returned attribution matrix.

        Raises:
            ValueError: If `values` is not two-dimensional, its column count
                differs from `features`, it contains non-finite numbers, or
                the method or target is blank.
        """
        values = numpy.asarray(self.values, dtype=numpy.float64)
        if values.ndim != ATTRIBUTION_MATRIX_RANK:
            raise errors.InputError(
                "explanation values must be a two-dimensional matrix"
            )
        if values.shape[1] != self.features:
            raise errors.InputError(
                "explanation feature count does not match values"
            )
        if not numpy.isfinite(values).all():
            raise errors.InputError("explanation values must be finite")
        if not self.method.strip() or not self.target.strip():
            raise errors.InputError(
                "explanation method and target must not be empty"
            )
        object.__setattr__(self, "values", values)

    def to_dict(self) -> dict[str, object]:
        """Returns JSON-compatible metadata without silently losing values."""
        return records.upgrade(
            self.migrations,
            {
                "method": self.method,
                "target": self.target,
                "feature_count": self.features,
                "row_count": int(self.values.shape[0]),
                "values": self.values.tolist(),
            },
            records.ORIGIN,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Boundary:
    """Reusable capability boundary for one deep-learning model family.

    Attributes:
        family: Name of the model family.
        status: Support status of the family.
        modalities: Modalities the family accepts, sorted by value.
        extras: Optional install extras the family needs.
        checkpoint: Serialization contract.
        inference: Input and output contract.
        explanation: Explanation contract, or None when the family offers no
            explanations.
        limitations: Caveats of the family, sorted and de-duplicated.
        migrations: The written layout's version steps, oldest first;
            version 2 writes `family`, `extras`, and the checkpoint and
            inference contracts at their version 2.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            {"model_family": "family", "required_extras": "extras"},
            parts={
                "checkpoint": Checkpoint.migrations[0],
                "inference": Contract.migrations[0],
            },
        ),
    )

    family: str
    status: capabilities.Status
    modalities: tuple[capabilities.Modality, ...]
    extras: tuple[str, ...]
    checkpoint: Checkpoint
    inference: Contract
    explanation: Explainability | None
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validates and canonicalizes the family declaration.

        Raises:
            ValueError: If the family name is blank, no modality is declared, or
                an extra or limitation is blank.
            TypeError: If a modality is not a `Modality` value.
        """
        if not self.family.strip():
            raise errors.InputError("model family must not be empty")
        if not self.modalities:
            raise errors.InputError("model family must declare a modality")
        if any(
            not isinstance(item, capabilities.Modality)
            for item in self.modalities
        ):
            raise errors.KindError(
                "model family modalities must use Modality values"
            )
        if any(not item.strip() for item in self.extras):
            raise errors.InputError("required extras must not be empty")
        if any(not item.strip() for item in self.limitations):
            raise errors.InputError(
                "model family limitations must not be empty"
            )
        object.__setattr__(
            self,
            "modalities",
            tuple(sorted(set(self.modalities), key=lambda item: item.value)),
        )
        object.__setattr__(self, "extras", tuple(sorted(set(self.extras))))
        object.__setattr__(
            self, "limitations", tuple(sorted(set(self.limitations)))
        )

    def accommodates(self, modality: capabilities.Modality | str) -> bool:
        """Returns whether this family accepts the declared modality.

        Args:
            modality: A `Modality` or its string value.

        Returns:
            True if the modality is among the family's declared modalities.
        """
        value = capabilities.Modality(modality)
        return value in self.modalities

    def validate(
        self, features: Any, expected_width: int | None = None
    ) -> numpy.ndarray:
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
        if not self.accommodates(capabilities.Modality.TABULAR):
            raise errors.InputError(
                f"{self.family} has no numeric tabular input contract"
            )
        try:
            if hasattr(features, "toarray"):
                features = features.toarray()
            matrix = numpy.asarray(features, dtype=numpy.float32)
        except (TypeError, ValueError) as error:
            raise errors.InputError(
                f"{self.family} requires a finite numeric 2-D matrix"
            ) from error
        if matrix.ndim != self.inference.rank:
            raise errors.InputError(
                f"{self.family} requires a {self.inference.rank}-D "
                f"numeric matrix, received {matrix.ndim}-D input"
            )
        if expected_width is not None and matrix.shape[1] != expected_width:
            raise errors.InputError(
                f"{self.family} expected {expected_width} features, "
                f"received {matrix.shape[1]}"
            )
        if not numpy.isfinite(matrix).all():
            raise errors.InputError(f"{self.family} requires finite features")
        return matrix

    def to_dict(self) -> dict[str, object]:
        """Returns the complete declaration for evidence and diagnostics."""
        return {
            "family": self.family,
            "status": self.status.value,
            "modalities": [item.value for item in self.modalities],
            "extras": list(self.extras),
            "checkpoint": self.checkpoint.to_dict(),
            "inference": self.inference.to_dict(),
            "explanation": (
                None if self.explanation is None else self.explanation.to_dict()
            ),
            "limitations": list(self.limitations),
        }


TABULAR_TRANSFORMER_CONTRACT = Boundary(
    family=capabilities.Family.TRANSFORMER.value,
    status=capabilities.Status.EXPERIMENTAL,
    modalities=(capabilities.Modality.TABULAR,),
    extras=("deeplearning",),
    checkpoint=Checkpoint(
        format="scmle.tabular-transformer.v1",
        schema_version=1,
        trusted=True,
        replace=True,
    ),
    inference=Contract(
        modality=capabilities.Modality.TABULAR,
        rank=2,
        dtype="float32",
        methods=("predict", "predict_proba"),
        output="prediction or class probability",
        deterministic=True,
    ),
    explanation=Explainability(
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


TABULAR_JEPA_CONTRACT = Boundary(
    family=capabilities.Family.JEPA.value,
    status=capabilities.Status.EXPERIMENTAL,
    modalities=(capabilities.Modality.TABULAR,),
    extras=("deeplearning",),
    checkpoint=Checkpoint(
        format="scmle.tabular-jepa.v2",
        schema_version=1,
        trusted=True,
        replace=True,
    ),
    inference=Contract(
        modality=capabilities.Modality.TABULAR,
        rank=2,
        dtype="float32",
        methods=("transform", "encode"),
        output="context embedding",
        deterministic=True,
    ),
    explanation=Explainability(
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


TABULAR_TORCH_CONTRACT = Boundary(
    family="deep_learning",
    status=capabilities.Status.SUPPORTED,
    modalities=(capabilities.Modality.TABULAR,),
    extras=("deeplearning",),
    checkpoint=Checkpoint(
        format="scmle.tabular-torch.v1",
        schema_version=1,
        trusted=True,
        replace=True,
    ),
    inference=Contract(
        modality=capabilities.Modality.TABULAR,
        rank=2,
        dtype="float32",
        methods=("predict", "predict_proba"),
        output="prediction or class probability",
        deterministic=True,
    ),
    explanation=Explainability(
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


def resolve(architecture: str) -> Boundary:
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
    "Boundary",
    "Checkpoint",
    "Contract",
    "Contribution",
    "Explainability",
    "resolve",
]
