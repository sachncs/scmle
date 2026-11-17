"""The survival rule and the estimates of one finding per specification."""

from collections.abc import Sequence
import dataclasses
import itertools
import math
import operator
from typing import Any

import numpy
from scipy import stats

from scmle.assurance.charter import contracts
from scmle.assurance.robustness import multiverse
from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.runtime import randomness

DETECTOR_ID = "f6-stability"
DEFECT_OUTLIER = "fragile_outlier_driven"
DEFECT_REVERSAL = "fragile_inclusion_reversal"
DEFECT_NULL = "no_effect"
DEFECT_TYPES = (DEFECT_OUTLIER, DEFECT_REVERSAL, DEFECT_NULL)
STABLE = "stable"
FRAGILE = "fragile"
NOT_EVALUABLE = "not_evaluable"
IMPUTATION = "imputation"
OUTLIER_RULE = "outlier_rule"
INCLUSION = "inclusion"
MIN_CHOICE_LEVELS = 2
IMPLEMENTED: dict[str, tuple[str, ...]] = {
    IMPUTATION: ("median", "mean", "zero", "complete_case"),
    OUTLIER_RULE: ("none", "winsorize", "iqr_drop"),
    INCLUSION: ("all", "first_half", "second_half"),
}
MIN_PAIRS = 10
MIN_SPECS = 4
BOOTSTRAP = 100
CONFIDENCE = 0.95
WINSOR_TAILS = (1.0, 99.0)
IQR_FENCE = 1.5
SIGNIFICANCE = 0.05


@dataclasses.dataclass(frozen=True, slots=True)
class Thresholds:
    """The survival rule, fixed before any result exists.

    Attributes:
        s_min: Smallest share of specifications whose effect has the sign of
            the median.
        band: Largest relative distance of the median effect from the
            headline effect.
        q_min: Smallest share of specifications whose perturbation interval
            excludes zero.
    """

    s_min: float = 0.8
    band: float = 0.5
    q_min: float = 0.8

    def __post_init__(self) -> None:
        """Validates the thresholds.

        Raises:
            ValueError: If `s_min` or `q_min` is outside (0, 1], or `band` is
                not positive.
        """
        for name in ("s_min", "q_min"):
            share = qualification.finite(getattr(self, name), name)
            if not 0.0 < share <= 1.0:
                raise errors.InputError(f"{name} must be within (0, 1]")
        if qualification.finite(self.band, "band") <= 0.0:
            raise errors.InputError("band must be positive")

    def to_dict(self) -> dict[str, float]:
        """Converts the rule to a JSON-ready dictionary.

        Returns:
            The thresholds `s_min`, `band` and `q_min` by name.
        """
        return dataclasses.asdict(self)


DEFAULT_RULE = Thresholds()


def actionable(
    choices: Sequence[contracts.Choice],
) -> tuple[tuple[contracts.Choice, ...], tuple[str, ...]]:
    """The declared choices this estimator can act on, and what was dropped.

    Args:
        choices: The charter's multiverse.

    Returns:
        (choices restricted to implemented options, plain-language notes
        about everything that was left out).
    """
    kept: list[contracts.Choice] = []
    notes: list[str] = []
    for choice in choices:
        known = IMPLEMENTED.get(choice.name)
        if known is None:
            notes.append(
                f"choice '{choice.name}' has no effect on a rank-correlation "
                "finding here: not run"
            )
            continue
        options = tuple(o for o in choice.options if o in known)
        left_out = [o for o in choice.options if o not in known]
        if left_out:
            notes.append(
                f"options {left_out} of '{choice.name}' are not implemented: "
                "not run"
            )
        if choice.default not in options:
            notes.append(
                f"default '{choice.default}' of '{choice.name}' is not "
                "implemented: choice not run"
            )
            continue
        kept.append(contracts.Choice(choice.name, options, choice.default))
    return tuple(kept), tuple(notes)


def prepare(
    x: numpy.ndarray,
    y: numpy.ndarray,
    positions: numpy.ndarray,
    rows: int,
    spec: multiverse.Specification,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Applies a specification to a (possibly resampled) feature and target.

    Args:
        x: Feature values, NaN where missing.
        y: Target codes.
        positions: Original row position of each entry, so `inclusion`
            refers to the stored order even after resampling.
        rows: Number of rows in the full data.
        spec: The specification.

    Returns:
        The feature and target after inclusion, outlier rule and
        imputation, as float arrays of equal length. Rows dropped by the
        outlier rule or by `complete_case`, and rows whose target is not
        finite, are removed from both.
    """
    inclusion = spec.to_dict().get(INCLUSION, "all")
    keep = numpy.ones(len(x), dtype=bool)
    if inclusion == "first_half":
        keep = positions < rows / 2.0
    elif inclusion == "second_half":
        keep = positions >= rows / 2.0
    values = numpy.array(x[keep], dtype=float)
    target = numpy.array(y[keep], dtype=float)
    observed = values[numpy.isfinite(values)]
    rule = spec.to_dict().get(OUTLIER_RULE, "none")
    if rule != "none" and observed.size:
        if rule == "winsorize":
            low, high = numpy.percentile(observed, WINSOR_TAILS)
            values = numpy.where(
                numpy.isfinite(values), numpy.clip(values, low, high), values
            )
        else:
            q1, q3 = numpy.percentile(observed, (25.0, 75.0))
            fence = IQR_FENCE * (q3 - q1)
            outside = numpy.isfinite(values) & (
                (values < q1 - fence) | (values > q3 + fence)
            )
            values, target = values[~outside], target[~outside]
    method = spec.to_dict().get(IMPUTATION, "median")
    missing = ~numpy.isfinite(values)
    if method == "complete_case":
        values, target = values[~missing], target[~missing]
    elif missing.any():
        seen = values[~missing]
        fill = 0.0
        if method == "median" and seen.size:
            fill = float(numpy.median(seen))
        elif method == "mean" and seen.size:
            fill = float(seen.mean())
        values = numpy.where(missing, fill, values)
    ok = numpy.isfinite(target)
    return values[ok], target[ok]


def rankcorr(x: numpy.ndarray, y: numpy.ndarray) -> float:
    """Computes Spearman's rank correlation of two arrays.

    Args:
        x: The first array.
        y: The second array.

    Returns:
        Spearman's rho, or NaN when it is not defined: fewer than
        `MIN_PAIRS` pairs, arrays of different length, or either array
        constant.
    """
    if len(x) < MIN_PAIRS or len(y) != len(x):
        return math.nan
    ranks_x, ranks_y = stats.rankdata(x), stats.rankdata(y)
    if ranks_x.std() == 0.0 or ranks_y.std() == 0.0:
        return math.nan
    return float(numpy.corrcoef(ranks_x, ranks_y)[0, 1])


def resample(
    rows: int, strata: numpy.ndarray | None, rng: numpy.random.Generator
) -> numpy.ndarray:
    """Draws bootstrap row positions.

    Args:
        rows: Number of rows.
        strata: A label per row, or None. When given, whole strata are
            drawn with replacement instead of single rows.
        rng: Random generator.

    Returns:
        An integer array of row positions: `rows` of them without strata,
        otherwise the concatenated members of the drawn strata (so its
        length varies with the strata drawn).
    """
    if strata is None:
        return rng.integers(0, rows, size=rows)
    labels = numpy.asarray(strata)
    names, inverse = numpy.unique(labels, return_inverse=True)
    members = [numpy.flatnonzero(inverse == k) for k in range(len(names))]
    drawn = rng.integers(0, len(names), size=len(names))
    return numpy.concatenate([members[k] for k in drawn])


@dataclasses.dataclass(frozen=True, slots=True)
class Variant:
    """One specification's estimate of a finding.

    Attributes:
        sha256: Specification hash.
        choices: The specification as choice to option.
        effect: Spearman's rho, or None when not defined.
        low: Lower end of the perturbation interval, or None.
        high: Upper end of the perturbation interval, or None.
    """

    sha256: str
    choices: dict[str, str]
    effect: float | None
    low: float | None
    high: float | None

    @property
    def conclusive(self) -> bool:
        """Whether the perturbation interval is defined and excludes zero."""
        return (
            self.low is not None
            and self.high is not None
            and (self.low > 0.0 or self.high < 0.0)
        )

    def to_dict(self) -> dict[str, Any]:
        """Converts the record to a JSON-ready dictionary.

        Returns:
            The fields by name.
        """
        return {
            "sha256": self.sha256,
            "choices": self.choices,
            "effect": self.effect,
            "low": self.low,
            "high": self.high,
        }


def estimate(
    x: numpy.ndarray,
    y: numpy.ndarray,
    spec: multiverse.Specification,
    seed: int,
    finding: str,
    strata: numpy.ndarray | None,
    bootstrap: int,
) -> dict[str, float]:
    """Estimates one specification's effect and its bootstrap interval.

    Args:
        x: Feature values, NaN where missing.
        y: Target codes, one per row.
        spec: The specification.
        seed: Run seed.
        finding: Name of the finding, part of the bootstrap stream's name.
        strata: Group or block labels for resampling, or None.
        bootstrap: Number of resamples.

    Returns:
        ``effect``, ``low`` and ``high``; a key is omitted when it is not
        defined (so the dictionary can be cached as numbers only). The
        interval is left out when fewer than half the resamples (and fewer
        than two) give a defined effect.
    """
    rows = len(x)
    positions = numpy.arange(rows)
    effect = rankcorr(*prepare(x, y, positions, rows, spec))
    out: dict[str, float] = {}
    if not math.isfinite(effect):
        return out
    out["effect"] = effect
    rng = randomness.component_rng(
        seed, f"f6.stability.{finding}.{spec.sha256}", 0
    )
    picks = list(
        map(
            operator.call,
            itertools.repeat(lambda: resample(rows, strata, rng), bootstrap),
        )
    )
    values = [
        rankcorr(*prepare(x[pick], y[pick], pick, rows, spec)) for pick in picks
    ]
    draws = [v for v in values if math.isfinite(v)]
    if len(draws) >= max(bootstrap // 2, 2):
        tail = (1.0 - CONFIDENCE) / 2.0 * 100.0
        low, high = numpy.percentile(draws, (tail, 100.0 - tail))
        out["low"], out["high"] = float(low), float(high)
    return out


def verdict(
    effects: Sequence[Variant], headline: float | None, rule: Thresholds
) -> tuple[str, str, dict[str, float | None]]:
    """Applies the survival rule to per-specification effects.

    Args:
        effects: One entry per evaluated specification.
        headline: The default specification's effect, or None.
        rule: Thresholds.

    Returns:
        (status, reason, numbers). The status is `not_evaluable` when there
        is no headline or fewer than `MIN_SPECS` defined effects, then
        `fragile` if any threshold fails, otherwise `stable`. The numbers
        hold `median`, `sign_share`, `size_ratio` (infinite when the
        headline is zero) and `interval_share`, all None when not
        evaluable.
    """
    defined = [e.effect for e in effects if e.effect is not None]
    numbers: dict[str, float | None] = {
        "median": None,
        "sign_share": None,
        "size_ratio": None,
        "interval_share": None,
    }
    if headline is None or len(defined) < MIN_SPECS:
        return (
            NOT_EVALUABLE,
            (
                f"fewer than {MIN_SPECS} specifications gave an estimate "
                "(or the headline specification did not)"
            ),
            numbers,
        )
    median = float(numpy.median(defined))
    sign = float(numpy.sign(median))
    sign_share = (
        0.0
        if sign == 0.0
        else float(numpy.mean([numpy.sign(v) == sign for v in defined]))
    )
    size_ratio = (
        math.inf if headline == 0.0 else abs(median - headline) / abs(headline)
    )
    interval_share = float(
        numpy.mean([e.conclusive for e in effects if e.effect is not None])
    )
    numbers = {
        "median": median,
        "sign_share": sign_share,
        "size_ratio": size_ratio,
        "interval_share": interval_share,
    }
    failures = []
    if sign_share < rule.s_min:
        failures.append(
            f"the sign agrees in {sign_share:.0%} of "
            f"specifications (need {rule.s_min:.0%})"
        )
    if size_ratio > rule.band:
        failures.append(
            f"the median effect is {size_ratio:.0%} away from "
            f"the headline (limit {rule.band:.0%})"
        )
    if interval_share < rule.q_min:
        failures.append(
            f"the interval excludes zero in {interval_share:.0%} "
            f"of specifications (need {rule.q_min:.0%})"
        )
    if failures:
        return FRAGILE, "; ".join(failures), numbers
    return (
        STABLE,
        "sign, size and interval hold across the declared grid",
        numbers,
    )


def attribution(
    effects: Sequence[Variant],
) -> dict[str, dict[str, Any]]:
    """Attributes the spread of the estimates to the declared choices.

    For each varied choice, the share of the variance of the estimates
    between specifications that its options explain (one-way eta squared),
    with the mean estimate under each option.

    Args:
        effects: One entry per evaluated specification.

    Returns:
        Choice name to a dictionary with `option_means` (option to mean
        estimate), `spread` (largest minus smallest option mean) and
        `eta_squared` (0 when the estimates do not vary). Choices with fewer
        than two options among the defined estimates are left out; the result
        is empty when no estimate is defined.
    """
    rows = [e for e in effects if e.effect is not None]
    out: dict[str, dict[str, Any]] = {}
    if not rows:
        return out
    values = numpy.array([e.effect for e in rows], dtype=float)
    total = float(((values - values.mean()) ** 2).sum())
    for name in sorted({k for e in rows for k in e.choices}):
        groups: dict[str, list[float]] = {}
        for e, value in zip(rows, values, strict=True):
            groups.setdefault(e.choices[name], []).append(float(value))
        if len(groups) < MIN_CHOICE_LEVELS:
            continue
        means = {k: float(numpy.mean(v)) for k, v in sorted(groups.items())}
        between = sum(
            len(v) * (numpy.mean(v) - values.mean()) ** 2
            for v in groups.values()
        )
        out[name] = {
            "option_means": means,
            "spread": max(means.values()) - min(means.values()),
            "eta_squared": float(between / total) if total > 0 else 0.0,
        }
    return out
