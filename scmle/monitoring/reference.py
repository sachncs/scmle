"""Training-time reference distributions that production data is compared to.

Reference profiles are computed once from the development data (and the
model's held-out predictions) and stored in the bundle, so monitoring needs no
access to the training data.
"""

import abc
import dataclasses
from typing import Any
from typing import ClassVar
from typing import Self

import numpy as np
import pandas as pd

from scmle.contracts import data as data_contracts
from scmle.data.validation import columns

DECILES = tuple(np.linspace(0.1, 0.9, 9))
MAX_CATEGORIES = 50
OTHER = "__other__"
EPSILON = 1e-4
NOISE_TAIL = 1.5


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray
) -> float:
    """Computes PSI between two discrete distributions.

    Args:
        expected: Reference proportions.
        actual: Observed proportions (same bins).

    Returns:
        The PSI; ~0 for identical distributions, >0.25 is a large shift.
    """
    e = np.clip(np.asarray(expected, dtype=float), EPSILON, None)
    a = np.clip(np.asarray(actual, dtype=float), EPSILON, None)
    return float(np.sum((a - e) * np.log(a / e)))


@dataclasses.dataclass(frozen=True)
class DistributionReference(abc.ABC):
    """Reference distribution of one column or score.

    Attributes:
        missing_rate: Share of missing values in the reference data.
        rows: Number of non-missing reference observations (drives the
            sampling-noise floor of PSI).
        kind: Discriminator used when persisting.
    """

    missing_rate: float
    rows: int = 0
    kind: ClassVar[str]

    @abc.abstractmethod
    def proportions(self, values: pd.Series) -> np.ndarray:
        """Bins new values into this reference's bins as proportions.

        Args:
            values: New observations.

        Returns:
            The share of the (non-missing) values in each bin, in the order of
            `expected()`; all zeros if there are no values.
        """

    @abc.abstractmethod
    def expected(self) -> np.ndarray:
        """Returns the reference proportions.

        Returns:
            The reference share of each bin.
        """

    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping.

        Returns:
            A dict that `reference_from_dict` can rebuild this reference from.
        """

    def psi(self, values: pd.Series) -> float:
        """Computes the raw PSI of new values against this reference.

        Args:
            values: New observations.

        Returns:
            The population stability index, without the noise floor removed.
        """
        return population_stability_index(
            self.expected(), self.proportions(values)
        )

    def noise_floor(self, new_rows: int) -> float:
        """Estimates the PSI level that sampling noise alone produces.

        With few observations PSI is positive even without drift: its mean is
        about `(bins - 1) * (1/n_reference + 1/n_new)` (verified by simulation)
        and its 95th percentile about twice that, so the floor is scaled by
        `NOISE_TAIL`. Comparing PSI to fixed thresholds without removing this
        floor causes false alarms.

        Args:
            new_rows: Number of non-missing new observations.

        Returns:
            The floor; 0.0 if either sample size is not positive.
        """
        if self.rows <= 0 or new_rows <= 0:
            return 0.0
        bins = len(self.expected())
        return NOISE_TAIL * (bins - 1) * (1.0 / self.rows + 1.0 / new_rows)

    def adjusted_psi(self, values: pd.Series) -> float:
        """Computes PSI minus its sampling-noise floor, never below zero.

        Args:
            values: New observations.

        Returns:
            The adjusted PSI.
        """
        raw = self.psi(values)
        floor = self.noise_floor(int(values.notna().sum()))
        return max(raw - floor, 0.0)

    def unseen_rate(self, values: pd.Series) -> float:
        """Computes the share of new values outside the reference domain.

        Args:
            values: New observations.

        Returns:
            0.0 here; subclasses with a defined domain override it.
        """
        return 0.0


@dataclasses.dataclass(frozen=True)
class NumericReference(DistributionReference):
    """Decile-binned numeric distribution.

    Attributes:
        edges: Interior bin edges.
        fractions: Reference proportion per bin.
    """

    edges: tuple[float, ...] = ()
    fractions: tuple[float, ...] = ()
    kind: ClassVar[str] = "numeric"

    @classmethod
    def fit(cls, values: pd.Series) -> Self:
        """Learns decile edges from reference values.

        Args:
            values: Reference observations; non-finite values count as missing.

        Returns:
            The reference, with the unique decile edges and the share of values
            in each bin.
        """
        numeric = columns.to_finite(values)
        present = numeric.dropna().to_numpy(dtype=float)
        edges = (
            tuple(np.unique(np.quantile(present, DECILES)))
            if len(present)
            else ()
        )
        missing = float(numeric.isna().mean())
        reference = cls(missing, len(present), edges, ())
        fractions = tuple(reference.proportions(numeric))
        return cls(missing, len(present), edges, fractions)

    def proportions(self, values: pd.Series) -> np.ndarray:
        """Bins values by the stored edges.

        Args:
            values: New observations; values that are not numeric are ignored.

        Returns:
            The share of numeric values in each bin.
        """
        numeric = (
            pd.to_numeric(values, errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        bins = np.searchsorted(np.asarray(self.edges), numeric, side="right")
        counts = np.bincount(bins, minlength=len(self.edges) + 1).astype(float)
        total = counts.sum()
        return counts / total if total else counts

    def expected(self) -> np.ndarray:
        """Returns the stored proportions.

        Returns:
            The reference share of each bin.
        """
        return np.asarray(self.fractions, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping.

        Returns:
            A dict with the kind, missing rate, row count, edges and fractions.
        """
        return {
            "kind": self.kind,
            "missing_rate": self.missing_rate,
            "rows": self.rows,
            "edges": list(self.edges),
            "fractions": list(self.fractions),
        }


@dataclasses.dataclass(frozen=True)
class CategoricalReference(DistributionReference):
    """Category frequency distribution.

    Attributes:
        frequencies: Reference proportion per category (plus `__other__`).
    """

    frequencies: dict[str, float] = dataclasses.field(default_factory=dict)
    kind: ClassVar[str] = "categorical"

    @classmethod
    def fit(cls, values: pd.Series) -> Self:
        """Learns category frequencies from reference values.

        Args:
            values: Reference observations.

        Returns:
            The reference, keeping the `MAX_CATEGORIES` most frequent categories
            and pooling the rest under `__other__`.
        """
        present = values.dropna().astype(str)
        top = present.value_counts(normalize=True).head(MAX_CATEGORIES)
        frequencies = {str(k): float(v) for k, v in top.items()}
        frequencies[OTHER] = max(0.0, 1.0 - sum(frequencies.values()))
        return cls(float(values.isna().mean()), len(present), frequencies)

    def levels(self) -> list[str]:
        """Lists the category names in a stable order.

        Returns:
            The sorted category names, including `__other__`.
        """
        return sorted(self.frequencies)

    def proportions(self, values: pd.Series) -> np.ndarray:
        """Counts new values per reference category (unknown -> other).

        Args:
            values: New observations.

        Returns:
            The share of values in each category, in `levels()` order.
        """
        known = set(self.frequencies)
        mapped = (
            values.dropna()
            .astype(str)
            .map(lambda v: v if v in known else OTHER)
        )
        counts = mapped.value_counts()
        vector = np.array(
            [float(counts.get(level, 0.0)) for level in self.levels()]
        )
        total = vector.sum()
        return vector / total if total else vector

    def expected(self) -> np.ndarray:
        """Returns reference proportions in `levels()` order.

        Returns:
            The reference share of each category.
        """
        return np.array([self.frequencies[level] for level in self.levels()])

    def unseen_rate(self, values: pd.Series) -> float:
        """Computes the share of new values with a category unseen in training.

        Args:
            values: New observations.

        Returns:
            The share of non-missing values not among the reference categories;
            0.0 if there are none.
        """
        present = values.dropna().astype(str)
        if present.empty:
            return 0.0
        return float((~present.isin(set(self.frequencies) - {OTHER})).mean())

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping.

        Returns:
            A dict with the kind, missing rate, row count and frequencies.
        """
        return {
            "kind": self.kind,
            "missing_rate": self.missing_rate,
            "rows": self.rows,
            "frequencies": self.frequencies,
        }


REFERENCE_KINDS: dict[str, type[DistributionReference]] = {
    cls.kind: cls for cls in (NumericReference, CategoricalReference)
}


def reference_from_dict(record: dict[str, Any]) -> DistributionReference:
    """Rebuilds a reference from `to_dict` output.

    Args:
        record: A dict produced by `to_dict`.

    Returns:
        A numeric reference if its kind is `numeric`, otherwise a categorical
        one.
    """
    kind = record["kind"]
    if kind == "numeric":
        return NumericReference(
            record["missing_rate"],
            record["rows"],
            tuple(record["edges"]),
            tuple(record["fractions"]),
        )
    return CategoricalReference(
        record["missing_rate"], record["rows"], record["frequencies"]
    )


@dataclasses.dataclass(frozen=True)
class ReferenceProfile:
    """Reference distributions of every feature and of the model's output.

    Attributes:
        features: Reference per feature column.
        output: Reference of the model's score / predicted class.
        output_name: What `output` describes.
        reference_metrics: Held-out metrics at training time.
    """

    features: dict[str, DistributionReference]
    output: DistributionReference
    output_name: str
    reference_metrics: dict[str, float]

    @classmethod
    def build(
        cls,
        features: pd.DataFrame,
        plan: data_contracts.FeaturePlan,
        output: pd.Series,
        output_name: str,
        reference_metrics: dict[str, float],
    ) -> Self:
        """Builds the profile from reference data.

        Args:
            features: Reference feature rows.
            plan: Feature columns by kind.
            output: The model's score (numeric) or predicted class (categorical)
                on held-out rows.
            output_name: Description of `output`.
            reference_metrics: Held-out metrics at training time.

        Returns:
            The profile, with a reference for each numeric and
            categorical column of `plan` present in `features`, and one for
            the output.
        """
        references: dict[str, DistributionReference] = {}
        for column in plan.numeric:
            if column in features.columns:
                references[column] = NumericReference.fit(features[column])
        for column in plan.categorical:
            if column in features.columns:
                references[column] = CategoricalReference.fit(features[column])
        numeric_output = pd.api.types.is_numeric_dtype(output)
        output_reference = (
            NumericReference.fit(output)
            if numeric_output
            else CategoricalReference.fit(output)
        )
        return cls(references, output_reference, output_name, reference_metrics)

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping.

        Returns:
            A dict with the feature references, the output reference, its name
            and the reference metrics.
        """
        return {
            "features": {k: v.to_dict() for k, v in self.features.items()},
            "output": self.output.to_dict(),
            "output_name": self.output_name,
            "reference_metrics": self.reference_metrics,
        }

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> Self:
        """Rebuilds a profile from `to_dict` output.

        Args:
            record: A dict produced by `to_dict`.

        Returns:
            The profile.
        """
        return cls(
            {k: reference_from_dict(v) for k, v in record["features"].items()},
            reference_from_dict(record["output"]),
            record["output_name"],
            record["reference_metrics"],
        )
