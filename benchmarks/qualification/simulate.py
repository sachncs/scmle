"""Semi-synthetic worlds with known truth.

A world keeps a real feature matrix, so correlation, skew and cardinality are
realistic, and draws the outcome from a known function. In an alternative
world a named set of columns has planted effects and the rest have none; in a
null world the outcome is a permutation of a real outcome, so the features
keep their dependence structure and no feature is related to the outcome
beyond the structure the permutation preserves (groups or time blocks).

Running a test over many worlds gives its empirical size and power with
Monte Carlo intervals, and running a pipeline over them gives its empirical
false-discovery proportion.
"""

from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from scmle.contracts import qualification
from scmle.core import randomness
from scmle.data.detectors import certificates

TASKS = ("regression", "binary")
# Coefficients on standardized features, in units of the noise standard
# deviation (regression) or log-odds per standard deviation (binary).
EFFECT_GRID = (0.1, 0.2, 0.3, 0.5, 0.8)
SIZE_BAND_SIGMAS = 2.0
DEFAULT_SLACK = 0.01


@dataclasses.dataclass(frozen=True, eq=False)
class World:
    """Features and an outcome with known truth.

    Attributes:
        features: Real feature matrix, unchanged.
        outcome: Outcome drawn from the world's function or permuted.
        kind: `alternative` (planted effects) or `null` (none).
        effects: Planted coefficient per feature; zero means no effect.
        seed_path: (run seed, component, index) of the random stream.
    """

    features: pd.DataFrame
    outcome: pd.Series
    kind: str
    effects: Mapping[str, float]
    seed_path: qualification.SeedPath

    @property
    def signal_columns(self) -> tuple[str, ...]:
        """Columns with a planted effect, in feature order."""
        return tuple(c for c in self.features.columns if self.effects[c] != 0)

    @property
    def null_columns(self) -> tuple[str, ...]:
        """Columns with no planted effect, in feature order."""
        return tuple(c for c in self.features.columns if self.effects[c] == 0)


def check_features(features: pd.DataFrame) -> None:
    """Requires finite numeric columns that each have spread.

    Raises:
        ValueError: On an empty frame, a non-numeric or boolean column, a
            missing or infinite value, a constant column or a repeated name.
    """
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("features must be a non-empty DataFrame")
    if not features.columns.is_unique:
        raise ValueError("feature names must be unique")
    for name in features.columns:
        column = features[name]
        if not pd.api.types.is_numeric_dtype(
            column
        ) or pd.api.types.is_bool_dtype(column):
            raise ValueError(f"feature '{name}' must be numeric")
        values = column.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"feature '{name}' must be finite")
        if float(np.std(values)) == 0.0:
            raise ValueError(f"feature '{name}' has no spread")


def standardize(features: pd.DataFrame) -> pd.DataFrame:
    """Z-scores each column (sample standard deviation)."""
    return (features - features.mean()) / features.std()


def planted_world(
    features: pd.DataFrame,
    effects: Mapping[str, float],
    seed: int,
    index: int = 0,
    task: str = "regression",
    noise_sd: float = 1.0,
) -> World:
    """Draws an outcome from planted effects on real features.

    Regression: `y = sum(effect_j * z_j) + noise_sd * N(0, 1)`. Binary:
    `y ~ Bernoulli(sigmoid(sum(effect_j * z_j)))`. `z_j` is the standardized
    feature.

    Args:
        features: Real numeric feature matrix.
        effects: Coefficient per planted column; at least one nonzero.
        seed: Run seed.
        index: Repeat counter for the random stream.
        task: `regression` or `binary`.
        noise_sd: Noise standard deviation, positive (regression only uses
            it; it is validated for both).

    Returns:
        An alternative world; unlisted columns have effect zero.

    Raises:
        ValueError: On invalid features, effects, task or noise.
    """
    check_features(features)
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    sigma = qualification.finite(noise_sd, "noise_sd")
    if sigma <= 0.0:
        raise ValueError("noise_sd must be positive")
    unknown = sorted(set(effects) - set(features.columns))
    if unknown:
        raise ValueError(f"effects name unknown columns: {unknown}")
    full = dict.fromkeys(features.columns, 0.0)
    for name, value in effects.items():
        full[name] = qualification.finite(value, f"effect of '{name}'")
    if not any(full.values()):
        raise ValueError("an alternative world needs a nonzero effect")
    rng = randomness.component_rng(seed, "simulate.planted", index)
    z = standardize(features)
    linear = z.to_numpy() @ np.array(list(full.values()))
    if task == "regression":
        values = linear + sigma * rng.normal(size=len(features))
    else:
        probability = 1.0 / (1.0 + np.exp(-linear))
        values = (rng.random(len(features)) < probability).astype(int)
    return World(
        features=features,
        outcome=pd.Series(values, index=features.index, name="y"),
        kind="alternative",
        effects=full,
        seed_path=randomness.seed_path(seed, "simulate.planted", index),
    )


def random_effects(
    columns: Sequence[str],
    n_signal: int,
    size: float,
    seed: int,
    index: int = 0,
) -> dict[str, float]:
    """Picks `n_signal` columns and gives each effect `+size` or `-size`.

    Raises:
        ValueError: If `n_signal` is not between 1 and the column count or
            `size` is not positive.
    """
    names = list(columns)
    if not 1 <= n_signal <= len(names):
        raise ValueError("n_signal must be between 1 and the column count")
    magnitude = qualification.finite(size, "size")
    if magnitude <= 0.0:
        raise ValueError("size must be positive")
    rng = randomness.component_rng(seed, "simulate.effects", index)
    chosen = np.sort(rng.choice(len(names), n_signal, replace=False))
    signs = rng.choice([-1.0, 1.0], size=n_signal)
    return {
        names[i]: float(s) * magnitude
        for i, s in zip(chosen, signs, strict=True)
    }


def permuted(
    values: np.ndarray,
    rng: np.random.Generator,
    groups: pd.Series | None,
    block_size: int | None,
) -> np.ndarray:
    """Permutes `values` globally, within groups or within time blocks."""
    out = values.copy()
    if groups is not None:
        for _, positions in sorted(
            groups.groupby(groups.to_numpy(), sort=True).indices.items()
        ):
            out[positions] = values[rng.permutation(positions)]
    elif block_size is not None:
        for start in range(0, len(values), block_size):
            positions = np.arange(start, min(start + block_size, len(values)))
            out[positions] = values[rng.permutation(positions)]
    else:
        out = values[rng.permutation(len(values))]
    return out


def check_permutation_scope(
    rows: int, groups: pd.Series | None, block_size: int | None
) -> None:
    """Validates the grouping arguments of a null world."""
    if groups is not None and block_size is not None:
        raise ValueError("use groups or block_size, not both")
    if groups is not None and len(groups) != rows:
        raise ValueError("groups must have one entry per row")
    if block_size is not None:
        qualification.count(block_size, "block_size", minimum=2)


def null_world(
    features: pd.DataFrame,
    outcome: pd.Series,
    seed: int,
    index: int = 0,
    groups: pd.Series | None = None,
    block_size: int | None = None,
) -> World:
    """A world with no feature-outcome relation beyond the kept structure.

    The outcome is permuted: within `groups` (so group-level association is
    kept), within consecutive blocks of `block_size` rows (so slow time
    structure is kept), or globally when neither is given.

    Args:
        features: Real numeric feature matrix.
        outcome: Real outcome, one value per row.
        seed: Run seed.
        index: Repeat counter for the random stream.
        groups: Group label per row.
        block_size: Rows per time block, at least 2.

    Returns:
        A null world with all effects zero.

    Raises:
        ValueError: On invalid features, a length mismatch or both scopes.
    """
    check_features(features)
    return build_null(features, outcome, seed, index, groups, block_size)


def build_null(
    features: pd.DataFrame,
    outcome: pd.Series,
    seed: int,
    index: int,
    groups: pd.Series | None,
    block_size: int | None,
) -> World:
    """Builds a null world from already validated features."""
    if len(outcome) != len(features):
        raise ValueError("outcome needs one value per feature row")
    check_permutation_scope(len(features), groups, block_size)
    rng = randomness.component_rng(seed, "simulate.null", index)
    values = permuted(outcome.to_numpy(), rng, groups, block_size)
    return World(
        features=features,
        outcome=pd.Series(values, index=features.index, name=outcome.name),
        kind="null",
        effects=dict.fromkeys(features.columns, 0.0),
        seed_path=randomness.seed_path(seed, "simulate.null", index),
    )


def null_worlds(
    features: pd.DataFrame,
    outcome: pd.Series,
    runs: int,
    seed: int,
    groups: pd.Series | None = None,
    block_size: int | None = None,
) -> Iterator[World]:
    """Yields `runs` independent null worlds (stream index 0..runs-1)."""
    qualification.count(runs, "runs", minimum=1)
    check_features(features)
    for run in range(runs):
        yield build_null(features, outcome, seed, run, groups, block_size)


def size_band(alpha: float, runs: int) -> tuple[float, float]:
    """`alpha +/- 2 * sqrt(alpha * (1 - alpha) / runs)`.

    A correct level-`alpha` test falls inside this band in about 95 percent
    of Monte Carlo experiments of `runs` worlds.
    """
    level = qualification.probability(alpha, "alpha")
    qualification.count(runs, "runs", minimum=1)
    half = SIZE_BAND_SIGMAS * math.sqrt(level * (1.0 - level) / runs)
    return level - half, level + half


@dataclasses.dataclass(frozen=True)
class RejectionRate:
    """How often a test rejected over a set of worlds.

    Attributes:
        rejections: Worlds in which the test rejected.
        runs: Worlds tested.
        alpha: Level used to decide a rejection.
    """

    rejections: int
    runs: int
    alpha: float

    @property
    def rate(self) -> float:
        """Share of worlds that rejected."""
        return self.rejections / self.runs

    @property
    def interval(self) -> tuple[float, float]:
        """Wilson interval for the rejection probability."""
        return certificates.wilson_interval(self.rejections, self.runs)

    @property
    def within_size_band(self) -> bool:
        """Whether the rate is inside `size_band(alpha, runs)`."""
        low, high = size_band(self.alpha, self.runs)
        return low <= self.rate <= high


def rejection_rate(
    test: Callable[[World], float], worlds: Iterable[World], alpha: float
) -> RejectionRate:
    """Applies a p-value test to each world and counts rejections.

    Args:
        test: Returns a p-value for a world.
        worlds: Worlds to test.
        alpha: Rejection level in [0, 1].

    Returns:
        The rejection count over the worlds. On null worlds this is the
        empirical size; on alternative worlds it is the empirical power.

    Raises:
        ValueError: If there are no worlds or a p-value is outside [0, 1].
    """
    level = qualification.probability(alpha, "alpha")
    rejections = 0
    runs = 0
    for world in worlds:
        p_value = qualification.probability(test(world), "p-value")
        rejections += int(p_value <= level)
        runs += 1
    if runs == 0:
        raise ValueError("no worlds to test")
    return RejectionRate(rejections, runs, level)


def power_curve(
    test: Callable[[World], float],
    features: pd.DataFrame,
    column: str,
    grid: Sequence[float],
    runs: int,
    alpha: float,
    seed: int,
    task: str = "regression",
    noise_sd: float = 1.0,
) -> dict[float, RejectionRate]:
    """Empirical power of `test` at each effect size on one column.

    Args:
        test: Returns a p-value for a world.
        features: Real numeric feature matrix.
        column: Column that carries the planted effect.
        grid: Positive effect sizes (see `EFFECT_GRID`).
        runs: Worlds per effect size.
        alpha: Rejection level.
        seed: Run seed.
        task: `regression` or `binary`.
        noise_sd: Noise standard deviation for regression.

    Returns:
        Rejection rate per effect size.
    """
    qualification.count(runs, "runs", minimum=1)
    curve: dict[float, RejectionRate] = {}
    for position, size in enumerate(grid):
        if qualification.finite(size, "effect size") <= 0.0:
            raise ValueError("effect sizes must be positive")
        worlds = (
            planted_world(
                features,
                {column: size},
                seed,
                index=position * runs + run,
                task=task,
                noise_sd=noise_sd,
            )
            for run in range(runs)
        )
        curve[size] = rejection_rate(test, worlds, alpha)
    return curve


def coefficient_pvalues(world: World) -> pd.Series:
    """Two-sided p-values of the OLS coefficient of each feature.

    The planted truth is conditional: a null column has a zero coefficient
    given the others. With correlated features a marginal test on such a
    column still rejects, because it shares signal with a planted column, so
    the conditional hypothesis needs a conditional test like this one.

    Args:
        world: A world with more rows than features plus one.

    Returns:
        A p-value per feature (the intercept is fitted, not returned).

    Raises:
        ValueError: If there are too few rows for the fit.
    """
    z = standardize(world.features).to_numpy()
    rows, columns = z.shape
    if rows <= columns + 1:
        raise ValueError("coefficient tests need more rows than columns + 1")
    design = np.column_stack([np.ones(rows), z])
    outcome = world.outcome.to_numpy(dtype=float)
    solution, *_ = np.linalg.lstsq(design, outcome, rcond=None)
    residual = outcome - design @ solution
    freedom = rows - columns - 1
    variance = float(residual @ residual) / freedom
    covariance = variance * np.linalg.inv(design.T @ design)
    errors = np.sqrt(np.diag(covariance))[1:]
    t_values = solution[1:] / errors
    p_values = 2.0 * stats.t.sf(np.abs(t_values), freedom)
    return pd.Series(p_values, index=world.features.columns)


def false_discovery_proportion(
    rejected: Collection[str], world: World
) -> float:
    """Share of rejected columns that have no planted effect.

    Zero when nothing is rejected.

    Raises:
        ValueError: If a rejected name is not a feature.
    """
    unknown = sorted(set(rejected) - set(world.features.columns))
    if unknown:
        raise ValueError(f"rejected names unknown columns: {unknown}")
    if not rejected:
        return 0.0
    false = len(set(rejected) & set(world.null_columns))
    return false / len(set(rejected))


@dataclasses.dataclass(frozen=True)
class FalseDiscoverySummary:
    """Empirical FDR over simulated worlds.

    Attributes:
        mean: Average false-discovery proportion, the empirical FDR.
        upper: One-sided upper confidence limit of the mean (normal
            approximation; 1.0 with fewer than two runs).
        runs: Worlds averaged.
        confidence: One-sided level of `upper`.
    """

    mean: float
    upper: float
    runs: int
    confidence: float

    def supports(self, alpha: float, slack: float = DEFAULT_SLACK) -> bool:
        """Whether `upper <= alpha + slack`, the F7 acceptance rule."""
        return self.upper <= alpha + slack


def summarize_false_discoveries(
    proportions: Sequence[float], confidence: float = 0.95
) -> FalseDiscoverySummary:
    """Averages false-discovery proportions with an upper limit.

    Raises:
        ValueError: On no proportions, a value outside [0, 1] or a
            confidence outside (0, 1).
    """
    if not proportions:
        raise ValueError("no proportions to summarize")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be within (0, 1)")
    values: Any = np.array(
        [qualification.probability(p, "proportion") for p in proportions]
    )
    mean = float(values.mean())
    runs = len(values)
    if runs < 2:
        upper = 1.0
    else:
        z = float(stats.norm.ppf(confidence))
        upper = min(1.0, mean + z * float(values.std(ddof=1)) / math.sqrt(runs))
    return FalseDiscoverySummary(mean, upper, runs, confidence)
