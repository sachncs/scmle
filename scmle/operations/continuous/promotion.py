"""Whether a promoted model helped: period and holdback comparisons."""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import hashlib
import math
from typing import Any

import numpy
from scipy import stats

from scmle.contracts import errors


@dataclasses.dataclass(frozen=True, slots=True)
class Indicator:
    """A metric watched after a promotion.

    Attributes:
        name: Metric name.
        kind: ``operational`` or ``business``.
        better: ``higher`` or ``lower``.
        margin: Smallest difference that matters, in the metric's unit.
        primary: Whether the model must improve it to count as helping.
    """

    __module__ = "scmle.operations.continuous"

    name: str
    kind: str
    better: str
    margin: float = 0.0
    primary: bool = False

    def __post_init__(self) -> None:
        """Validates the choices.

        Raises:
            ValueError: If the kind is not operational or business, `better`
                is not higher or lower, or the margin is negative or not finite.
        """
        if self.kind not in {"operational", "business"}:
            raise errors.InputError("kind must be operational or business")
        if self.better not in {"higher", "lower"}:
            raise errors.InputError("better must be higher or lower")
        if self.margin < 0 or not math.isfinite(self.margin):
            raise errors.InputError("margin must be finite and not negative")


def contrast(
    before: numpy.ndarray,
    after: numpy.ndarray,
    alpha: float,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Percentile interval of mean(after) - mean(before).

    Args:
        before: Observations of the earlier period.
        after: Observations of the later period.
        alpha: Error level; the interval spans the alpha/2 to 1 - alpha/2
            percentiles.
        resamples: Number of bootstrap resamples.
        seed: Seed of the resampling.

    Returns:
        The lower and upper bounds of the interval.
    """
    rng = numpy.random.default_rng(seed)
    first = rng.choice(before, size=(resamples, len(before))).mean(axis=1)
    second = rng.choice(after, size=(resamples, len(after))).mean(axis=1)
    gap = second - first
    low, high = numpy.quantile(gap, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def judge(
    spec: Indicator,
    old: numpy.ndarray,
    new: numpy.ndarray,
    adjusted: float,
    min_samples: int,
    resamples: int,
    seed: int,
) -> Mapping[str, Any]:
    """The verdict and interval of one metric for `post_promotion_check`.

    Args:
        spec: The metric.
        old: Its observations in the earlier period.
        new: Its observations in the later period.
        adjusted: The metric's share of the joint error level.
        min_samples: Fewest observations in each period for a verdict.
        resamples: Number of bootstrap resamples.
        seed: Seed of the resampling.

    Returns:
        The metric's record: verdict, interval and sample counts, or a reason
        and no interval when a period has fewer than `min_samples`.
    """
    if min(len(old), len(new)) < min_samples:
        return {
            "metric": spec.name,
            "kind": spec.kind,
            "verdict": "inconclusive",
            "reason": f"fewer than {min_samples} observations in a period",
            "samples_before": len(old),
            "samples_after": len(new),
            "interval": None,
        }
    low, high = contrast(old, new, adjusted, resamples, seed)
    sign = 1.0 if spec.better == "higher" else -1.0
    good_low, good_high = sorted((sign * low, sign * high))
    if good_low > spec.margin:
        verdict = "improved"
    elif good_high < -spec.margin:
        verdict = "worse"
    else:
        verdict = "inconclusive"
    return {
        "metric": spec.name,
        "kind": spec.kind,
        "better": spec.better,
        "margin": spec.margin,
        "primary": spec.primary,
        "verdict": verdict,
        "interval": [round(low, 6), round(high, 6)],
        "confidence": round(1.0 - adjusted, 4),
        "samples_before": len(old),
        "samples_after": len(new),
        "descriptive_difference": round(float(new.mean() - old.mean()), 6),
    }


def post_promotion_check(
    specs: Sequence[Indicator],
    before: Mapping[str, Sequence[float]],
    after: Mapping[str, Sequence[float]],
    *,
    alpha: float = 0.05,
    min_samples: int = 30,
    resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Did the promoted model help? An interval and a verdict per metric.

    The error level is split over the metrics (Bonferroni), so the intervals
    hold together at ``1 - alpha``. A metric is ``improved`` or ``worse`` only
    when its whole interval lies beyond ``margin`` on that side; otherwise it
    is ``inconclusive``, as it is when either period has fewer than
    ``min_samples`` observations. The overall verdict is ``harmed`` if any
    metric is worse, ``helped`` if there is at least one primary metric, every
    primary metric improved and none is worse, otherwise ``inconclusive``.

    This is a comparison of two periods. It cannot separate the model from
    anything else that changed between them; the statement says so.

    Args:
        specs: The metrics to judge.
        before: Observations of each metric in the earlier period.
        after: Observations of each metric in the later period.
        alpha: Joint error level over all metrics.
        min_samples: Fewest observations in each period for a verdict.
        resamples: Number of bootstrap resamples.
        seed: Seed of the resampling.

    Returns:
        The overall verdict, one record per metric in the order of `specs`
        (verdict, interval and sample counts; a metric with too few samples
        has a reason and no interval instead of the other fields) and a
        statement of what the comparison shows.

    Raises:
        ValueError: If a metric has no values for a period or ``specs`` is
            empty.
    """
    if not specs:
        raise errors.InputError("name at least one metric")
    adjusted = alpha / len(specs)
    results = []
    for spec in specs:
        if spec.name not in before or spec.name not in after:
            raise errors.InputError(
                f"no values for {spec.name} in both periods"
            )
        old = numpy.asarray(before[spec.name], dtype=float)
        new = numpy.asarray(after[spec.name], dtype=float)
        results.append(
            judge(spec, old, new, adjusted, min_samples, resamples, seed)
        )
    verdicts = {r["metric"]: r["verdict"] for r in results}
    primaries = [s.name for s in specs if s.primary]
    if "worse" in verdicts.values():
        overall = "harmed"
    elif primaries and all(verdicts[n] == "improved" for n in primaries):
        overall = "helped"
    else:
        overall = "inconclusive"
    return {
        "overall": overall,
        "metrics": results,
        "statement": (
            f"Overall: {overall}. Each interval is the change in the mean "
            f"from the period before to the period after, at joint "
            f"confidence {1 - alpha:.0%} over {len(specs)} metrics. This "
            "compares two periods; it cannot separate the model from "
            "seasonality, traffic mix or other changes, and a holdback or "
            "randomised split would be needed to attribute the change to the "
            "model."
        ),
    }


def reserve(
    unit_ids: Sequence[Any], fraction: float, salt: str
) -> numpy.ndarray:
    """Which units stay on the previous model: a keyed hash of the unit id.

    The side of a unit depends only on its id and the `salt` (for example the
    release id), so every replica of the service and any later audit agree
    without sharing state, and a unit keeps its side for the life of the
    release. The hash is uniform, so each unit is held back independently
    with probability `fraction`.

    Args:
        unit_ids: Identifiers of the units.
        fraction: Share of units to hold back, strictly between 0 and 1.
        salt: Value mixed into the hash, for example the release id.

    Returns:
        A boolean array, True for each unit that stays on the previous
        model.

    Raises:
        ValueError: If `fraction` is not strictly between 0 and 1.
    """
    if not 0.0 < fraction < 1.0:
        raise errors.InputError("fraction must lie strictly between 0 and 1")
    draws = numpy.array(
        [
            int.from_bytes(
                hashlib.sha256(f"{salt}:{unit}".encode()).digest()[:8], "big"
            )
            / 2.0**64
            for unit in unit_ids
        ]
    )
    return draws < fraction


def balance(
    specs: Sequence[Indicator],
    holdback: Mapping[str, Sequence[float]],
    promoted: Mapping[str, Sequence[float]],
    expected_fraction: float,
    ratio_level: float,
) -> Mapping[str, Any]:
    """The sample-ratio check of a holdback split.

    An exact two-sided binomial test of the observed split against
    `expected_fraction`; the group sizes are those of the first metric.

    Args:
        specs: The metrics; the first one gives the group sizes.
        holdback: Observations of each metric for the held-back units.
        promoted: Observations of each metric for the promoted units.
        expected_fraction: The declared share of units held back.
        ratio_level: Level of the test.

    Returns:
        The group sizes, the expected fraction, the p-value, the level and
        whether the split is flagged as not matching.
    """
    first = specs[0].name if specs else ""
    held = len(holdback.get(first, ()))
    moved = len(promoted.get(first, ()))
    total = held + moved
    p_value = (
        float(stats.binomtest(held, total, expected_fraction).pvalue)
        if total
        else 1.0
    )
    return {
        "holdback_units": held,
        "promoted_units": moved,
        "expected_fraction": expected_fraction,
        "p_value": p_value,
        "level": ratio_level,
        "flag": p_value < ratio_level,
    }


def attribute(
    overall: str, alpha: float, count: int, check: Mapping[str, Any]
) -> str:
    """The statement of what a holdback comparison shows.

    Args:
        overall: The overall verdict.
        alpha: Joint error level over all metrics.
        count: Number of metrics judged.
        check: The sample-ratio check of `balance`.

    Returns:
        The statement, saying whether the split matched the declared one.
    """
    return (
        f"Overall: {overall}. Each interval is the difference in the mean "
        "between units served by the promoted model and units kept on the "
        f"previous model at the same time, at joint confidence "
        f"{1 - alpha:.0%} over {count} metrics. "
        + (
            "The observed split differs from the declared fraction "
            f"(p = {check['p_value']:.2g}), so the assignment was not what was "
            "declared and no verdict is given. "
            if check["flag"]
            else "The change is attributable to the model if units were "
            "assigned at random and do not influence one another. "
        )
        + "The check does not look at metrics that were not selected."
    )


def holdback_check(
    specs: Sequence[Indicator],
    holdback: Mapping[str, Sequence[float]],
    promoted: Mapping[str, Sequence[float]],
    *,
    expected_fraction: float,
    alpha: float = 0.05,
    min_samples: int = 30,
    resamples: int = 2000,
    seed: int = 0,
    ratio_level: float = 0.001,
) -> dict[str, Any]:
    """Did the promoted model help, against units kept on the old model?

    The comparison is concurrent: units were assigned to the previous model
    (`holdback`) or to the promoted one at random by `reserve`,
    so seasonality, traffic mix and other changes affect both sides alike and
    the difference is attributable to the model, to the extent that the
    assignment was random and the units do not influence each other.

    The check first compares the observed split with `expected_fraction` by an
    exact two-sided binomial test at `ratio_level` (a sample-ratio check). The
    group sizes are taken from the first metric in `specs`. A split that does
    not match means the assignment was not what was declared: the overall
    verdict is then capped at ``inconclusive`` (the per-metric verdicts are
    left as computed) and the statement says so.
    Otherwise the metric intervals and verdicts are those of
    `post_promotion_check`, with the holdback as the reference period.

    Args:
        specs: The metrics to judge.
        holdback: Observations of each metric for the units kept on the
            previous model.
        promoted: Observations of each metric for the units on the promoted
            model.
        expected_fraction: The declared share of units held back, strictly
            between 0 and 1.
        alpha: Joint error level over all metrics.
        min_samples: Fewest observations in each group for a verdict.
        resamples: Number of bootstrap resamples.
        seed: Seed of the resampling.
        ratio_level: Level of the sample-ratio test.

    Returns:
        The result of `post_promotion_check` with the design, the
        sample-ratio check and a statement added; the overall verdict is
        capped at inconclusive when the split does not match.

    Raises:
        ValueError: As `post_promotion_check`, and for an
            ``expected_fraction`` outside the open unit interval.
    """
    if not 0.0 < expected_fraction < 1.0:
        raise errors.InputError(
            "expected_fraction must lie strictly between 0 and 1"
        )
    check = balance(specs, holdback, promoted, expected_fraction, ratio_level)
    result = post_promotion_check(
        specs,
        holdback,
        promoted,
        alpha=alpha,
        min_samples=min_samples,
        resamples=resamples,
        seed=seed,
    )
    overall = "inconclusive" if check["flag"] else result["overall"]
    result["design"] = "concurrent_holdback"
    result["overall"] = overall
    result["ratio_check"] = check
    result["statement"] = attribute(overall, alpha, len(specs), check)
    return result
