"""Synthetic tabular data: generator contract, copula generator, evaluator.

`SyntheticGenerator` is the plugin contract (fit on real rows, sample new
ones); `FidelityEvaluator` is its evaluation counterpart. The shipped
`GaussianCopulaGenerator` preserves each numeric column's marginal
distribution and the rank correlations between columns, and samples
categorical columns from their empirical frequencies. Synthetic rows only
ever enter a *training fold*; whether they help is decided on the original
validation distribution.
"""

import abc
import dataclasses
from typing import Any
from typing import ClassVar
from typing import Self

import numpy as np
import pandas as pd
from scipy import special

from scmle.sampling import resampling

MIN_FIT_ROWS = 5
RIDGE = 1e-6


class SyntheticGenerator(abc.ABC):
    """Learns the distribution of real rows and samples new ones.

    Attributes:
        name: Identifier.
    """

    name: ClassVar[str]

    @abc.abstractmethod
    def fit(self, features: pd.DataFrame) -> Self:
        """Learns the distribution from real rows (training data only).

        Args:
            features: The real rows.

        Returns:
            The fitted generator.
        """

    @abc.abstractmethod
    def sample(self, rows: int, rng: np.random.Generator) -> pd.DataFrame:
        """Draws synthetic rows with the training columns.

        Args:
            rows: Number of rows to draw.
            rng: Source of randomness.

        Returns:
            A frame of `rows` synthetic rows.
        """


class GaussianCopulaGenerator(SyntheticGenerator):
    """Gaussian copula over numeric columns, empirical categoricals.

    Attributes:
        numeric: Numeric column names.
        categorical: Other column names.
        sorted_values: Sorted non-missing values per numeric column.
        correlation: Correlation of the numeric columns' normal scores.
        missing_rates: Missing share per column.
        frequencies: Category proportions per categorical column.
        dtypes: Original dtypes, restored on sampling.
    """

    name = "gaussian_copula"

    def fit(self, features: pd.DataFrame) -> Self:
        """Estimates the marginals and the normal-score correlation.

        Args:
            features: The real rows. Numeric non-boolean columns are modeled
                with the copula; all other columns by their frequencies.

        Returns:
            The fitted generator.
        """
        self.numeric = [
            c
            for c in features.columns
            if pd.api.types.is_numeric_dtype(features[c])
            and not pd.api.types.is_bool_dtype(features[c])
        ]
        self.categorical = [
            c for c in features.columns if c not in self.numeric
        ]
        self.dtypes = features.dtypes.to_dict()
        self.missing_rates = features.isna().mean().to_dict()
        self.sorted_values = {
            c: np.sort(features[c].dropna().to_numpy(dtype=float))
            for c in self.numeric
        }
        self.frequencies = {
            c: features[c].value_counts(normalize=True, dropna=True)
            for c in self.categorical
        }
        self.correlation = self.estimate_correlation(features)
        return self

    def estimate_correlation(self, features: pd.DataFrame) -> np.ndarray:
        """Estimates the correlation of the numeric columns' normal scores.

        Args:
            features: The real rows.

        Returns:
            The correlation matrix of the numeric columns in `numeric`
            order, with eigenvalues floored so it is positive definite and
            rescaled to a unit diagonal; a 0 by 0 matrix without numeric
            columns.
        """
        if not self.numeric:
            return np.zeros((0, 0))
        scores = pd.DataFrame(index=features.index)
        for column in self.numeric:
            ranks = features[column].rank(method="average")
            valid = features[column].notna()
            count = int(valid.sum())
            scores[column] = np.where(
                valid, special.ndtri((ranks - 0.5) / max(count, 1)), np.nan
            )
        correlation = scores.corr().fillna(0.0).to_numpy().copy()
        np.fill_diagonal(correlation, 1.0)
        values, vectors = np.linalg.eigh(correlation)
        values = np.clip(values, RIDGE, None)
        repaired = (vectors * values) @ vectors.T
        scale = np.sqrt(np.diag(repaired))
        return repaired / np.outer(scale, scale)

    def sample(self, rows: int, rng: np.random.Generator) -> pd.DataFrame:
        """Draws synthetic rows through the copula.

        Args:
            rows: Number of rows to draw.
            rng: Source of randomness.

        Returns:
            A frame of `rows` rows in the training column order. Missing
            values are reinserted at each column's training rate, and
            integer columns are rounded and stored as `Float64`.
        """
        output: dict[str, Any] = {}
        if self.numeric:
            normal = rng.multivariate_normal(
                np.zeros(len(self.numeric)), self.correlation, size=rows
            )
            uniform = special.ndtr(normal)
            for index, column in enumerate(self.numeric):
                values = self.sorted_values[column]
                if len(values) == 0:
                    output[column] = np.full(rows, np.nan)
                    continue
                grid = np.linspace(0.0, 1.0, len(values))
                output[column] = np.interp(uniform[:, index], grid, values)
        for column in self.categorical:
            freq = self.frequencies[column]
            if len(freq) == 0:
                output[column] = np.full(rows, np.nan, dtype=object)
            else:
                output[column] = rng.choice(
                    freq.index.to_numpy(dtype=object),
                    size=rows,
                    p=(freq / freq.sum()).to_numpy(),
                )
        frame = pd.DataFrame(output)[list(self.dtypes)]
        for key, rate in self.missing_rates.items():
            if rate > 0:
                frame.loc[rng.random(rows) < rate, key] = np.nan
        for key, dtype in self.dtypes.items():
            if key in self.numeric and pd.api.types.is_integer_dtype(dtype):
                frame[key] = frame[key].round().astype("Float64")
        return frame


class GeneratorOverSampler(resampling.Resampler):
    """Oversamples minority classes with synthetic rows from a generator.

    Attributes:
        generator_class: The `SyntheticGenerator` fitted per minority class.
    """

    name = "generator_oversample"
    generator_class: type[SyntheticGenerator] = GaussianCopulaGenerator

    def resample(
        self, features: pd.DataFrame, target: pd.Series, seed: int
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Adds synthetic rows until every class matches the largest.

        Classes with fewer than `MIN_FIT_ROWS` rows are left as they are.

        Args:
            features: Training features.
            target: Training labels.
            seed: Seed for the generator sampling and the final shuffle.

        Returns:
            The original plus synthetic features and labels, shuffled, with
            a fresh index.
        """
        rng = np.random.default_rng(seed)
        counts = target.value_counts()
        goal = int(counts.max())
        parts, labels = [features], [target]
        for key, count in counts.items():
            label = key
            missing = goal - int(count)
            members = features[(target == label).to_numpy()]
            if missing <= 0 or len(members) < MIN_FIT_ROWS:
                continue
            synthetic = self.generator_class().fit(members).sample(missing, rng)
            parts.append(
                synthetic.astype(features.dtypes.to_dict(), errors="ignore")
            )
            labels.append(pd.Series([label] * missing))
        combined = pd.concat(parts, ignore_index=True)
        combined_target = pd.concat(labels, ignore_index=True)
        order = rng.permutation(len(combined))
        return combined.iloc[order].reset_index(
            drop=True
        ), combined_target.iloc[order].reset_index(drop=True)


class CopulaOverSampler(GeneratorOverSampler):
    """Gaussian-copula oversampling of minority classes."""

    name = "copula_oversample"
    generator_class = GaussianCopulaGenerator


@dataclasses.dataclass(frozen=True)
class FidelityReport:
    """How closely synthetic rows resemble the real ones.

    Attributes:
        max_marginal_ks: Largest two-sample KS distance over numeric columns.
        max_category_distance: Largest total-variation distance over
            categorical columns.
        correlation_error: Largest absolute difference between the numeric
            correlation matrices.
        duplicate_share: Share of synthetic rows that exactly copy a real row
            (a privacy / overfitting red flag).
    """

    max_marginal_ks: float
    max_category_distance: float
    correlation_error: float
    duplicate_share: float


class FidelityEvaluator:
    """Scores synthetic data against the real data it imitates."""

    def evaluate(
        self, real: pd.DataFrame, synthetic: pd.DataFrame
    ) -> FidelityReport:
        """Compares marginals, dependence structure and verbatim copying.

        Args:
            real: Real rows.
            synthetic: Synthetic rows with the same columns.

        Returns:
            The fidelity measures; see `FidelityReport` for their meaning.
        """
        numeric = [
            str(c)
            for c in real.columns
            if pd.api.types.is_numeric_dtype(real[c])
            and not pd.api.types.is_bool_dtype(real[c])
        ]
        categorical = [str(c) for c in real.columns if c not in numeric]
        ks = 0.0
        for column in numeric:
            a = np.sort(real[column].dropna().to_numpy(dtype=float))
            b = np.sort(synthetic[column].dropna().to_numpy(dtype=float))
            if len(a) and len(b):
                grid = np.union1d(a, b)
                ks = max(
                    ks,
                    float(
                        np.max(
                            np.abs(
                                np.searchsorted(a, grid, side="right") / len(a)
                                - np.searchsorted(b, grid, side="right")
                                / len(b)
                            )
                        )
                    ),
                )
        tv = 0.0
        for column in categorical:
            p = real[column].value_counts(normalize=True)
            q = synthetic[column].value_counts(normalize=True)
            levels = p.index.union(q.index)
            tv = max(
                tv,
                float(
                    0.5
                    * np.abs(
                        p.reindex(levels, fill_value=0)
                        - q.reindex(levels, fill_value=0)
                    ).sum()
                ),
            )
        error = 0.0
        if len(numeric) > 1:
            real_corr = (
                real[numeric].astype(float).corr(method="spearman").to_numpy()
            )
            fake_corr = (
                synthetic[numeric]
                .astype(float)
                .corr(method="spearman")
                .to_numpy()
            )
            error = float(np.nanmax(np.abs(real_corr - fake_corr)))
        real_keys = set(map(tuple, real.astype(str).to_numpy().tolist()))
        copies = sum(
            tuple(row) in real_keys
            for row in synthetic.astype(str).to_numpy().tolist()
        )
        return FidelityReport(ks, tv, error, copies / max(len(synthetic), 1))
