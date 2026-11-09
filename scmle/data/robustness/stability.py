"""Multiverse stability of an exploration finding (PCS).

A finding is the rank correlation (Spearman's rho) between one feature and the
target on the exploration set `X`. The declared specification grid
(`multiverse`) varies the judgment calls the charter named; this module
implements the ones that act on a feature-target pair:

* `imputation`: `median`, `mean`, `zero` or `complete_case` for missing values
  of the feature;
* `outlier_rule`: `none`, `winsorize` (clip to the 1st and 99th percentile)
  or `iqr_drop` (drop values beyond 1.5 interquartile ranges of the quartiles);
* `inclusion`: `all`, `first_half` or `second_half` of the rows in their
  stored order (time order when the data are sorted by time).

Any other declared choice or option is reported as not applicable and is not
run; it never counts towards stability.

For every specification the finding is re-estimated and perturbed by
bootstrap resamples (whole groups or blocks when `strata` is given), which
gives the PCS-style perturbation interval of Yu and Kumbier (2020, PNAS
117:3920-3929). The survival rule, fixed here and not tuned to results:
a finding is `stable` iff

* the sign of the median effect agrees in at least `s_min` of the
  specifications,
* the median effect is within `band` of the headline (default specification)
  effect, and
* the perturbation interval excludes zero in at least `q_min` of them.

Otherwise it is `fragile`. A joint specification-curve test (Simonsohn,
Simmons and Nelson, 2020, Nature Human Behaviour 4:1208-1214) permutes the
target within strata and recomputes the whole curve, asking whether the curve
is more than chance without treating specifications as independent.

Stability is relative to the declared set. It says nothing about analyses that
were not declared, and a `stable` verdict is not a confirmation: confirmation
comes only from the single test on `C`. The rule's operating characteristics
are in `benchmarks/qualification/f6_certificates.json`; until a current
certificate passes, a `fragile` verdict is a warning, not a quarantine.
"""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
import math
import sys
from typing import Any

import numpy
from scipy import stats

from scmle.contracts import qualification
from scmle.core import randomness
from scmle.data.charter import contracts
from scmle.data.detectors import certificates
from scmle.data.ledger import bivariate
from scmle.data.ledger import controls
from scmle.data.robustness import multiverse

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
class Rule:
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
                raise ValueError(f"{name} must be within (0, 1]")
        if qualification.finite(self.band, "band") <= 0.0:
            raise ValueError("band must be positive")

    def to_dict(self) -> dict[str, float]:
        """Converts the rule to a JSON-ready dictionary.

        Returns:
            The thresholds `s_min`, `band` and `q_min` by name.
        """
        return dataclasses.asdict(self)


DEFAULT_RULE = Rule()


def applicable_choices(
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
    inclusion = spec.option(INCLUSION, "all")
    keep = numpy.ones(len(x), dtype=bool)
    if inclusion == "first_half":
        keep = positions < rows / 2.0
    elif inclusion == "second_half":
        keep = positions >= rows / 2.0
    values = numpy.array(x[keep], dtype=float)
    target = numpy.array(y[keep], dtype=float)
    observed = values[numpy.isfinite(values)]
    rule = spec.option(OUTLIER_RULE, "none")
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
    method = spec.option(IMPUTATION, "median")
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


def rank_effect(x: numpy.ndarray, y: numpy.ndarray) -> float:
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
class SpecEffect:
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
    def excludes_zero(self) -> bool:
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


def estimate_spec(
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
    effect = rank_effect(*prepare(x, y, positions, rows, spec))
    out: dict[str, float] = {}
    if not math.isfinite(effect):
        return out
    out["effect"] = effect
    rng = randomness.component_rng(
        seed, f"f6.stability.{finding}.{spec.sha256}", 0
    )
    picks = [resample(rows, strata, rng) for index in range(bootstrap)]
    values = [
        rank_effect(*prepare(x[pick], y[pick], pick, rows, spec))
        for pick in picks
    ]
    draws = [v for v in values if math.isfinite(v)]
    if len(draws) >= max(bootstrap // 2, 2):
        tail = (1.0 - CONFIDENCE) / 2.0 * 100.0
        low, high = numpy.percentile(draws, (tail, 100.0 - tail))
        out["low"], out["high"] = float(low), float(high)
    return out


def verdict(
    effects: Sequence[SpecEffect], headline: float | None, rule: Rule
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
            f"fewer than {MIN_SPECS} specifications gave an estimate "
            "(or the headline specification did not)",
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
        numpy.mean([e.excludes_zero for e in effects if e.effect is not None])
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
    effects: Sequence[SpecEffect],
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
        if len(groups) < 2:
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


@dataclasses.dataclass(frozen=True, slots=True)
class CurveTest:
    """Joint test of the whole specification curve against permuted targets.

    Attributes:
        permutations: Number of permuted curves.
        median_abs_effect: Observed median absolute effect.
        share_significant: Observed share of specifications with p <= 0.05.
        p_median: Permutation p-value of the median absolute effect:
            ``(1 + #{permuted >= observed}) / (permutations + 1)``.
        p_share: The same for the share significant.
    """

    permutations: int
    median_abs_effect: float
    share_significant: float
    p_median: float
    p_share: float

    def to_dict(self) -> dict[str, Any]:
        """Converts the test to a JSON-ready dictionary.

        Returns:
            The fields by name.
        """
        return dataclasses.asdict(self)


def curve_statistics(
    x: numpy.ndarray, y: numpy.ndarray, grid: multiverse.Grid
) -> tuple[float, float]:
    """Summarises the specification curve.

    Specifications with fewer than `MIN_PAIRS` pairs, or a constant feature
    or target, are left out.

    Args:
        x: Feature values, NaN where missing.
        y: Target codes, one per row.
        grid: The specifications.

    Returns:
        (median absolute Spearman rho, share of specifications with
        p <= `SIGNIFICANCE`). Both are 0 when no specification is usable.
    """
    rows = len(x)
    positions = numpy.arange(rows)
    effects, flags = [], []
    for spec in grid.specs:
        px, py = prepare(x, y, positions, rows, spec)
        if len(px) < MIN_PAIRS or numpy.ptp(px) == 0.0 or numpy.ptp(py) == 0.0:
            continue
        outcome = bivariate.spearman(px, py)
        effects.append(abs(outcome.effect))
        flags.append(outcome.p <= SIGNIFICANCE)
    if not effects:
        return 0.0, 0.0
    return float(numpy.median(effects)), float(numpy.mean(flags))


def curve_test(
    x: numpy.ndarray,
    y: numpy.ndarray,
    grid: multiverse.Grid,
    seed: int,
    finding: str,
    *,
    permutations: int,
    strata: numpy.ndarray | None = None,
) -> CurveTest:
    """Specification-curve test by permuting the target within strata.

    The permutation is exact under exchangeability within strata (global
    when `strata` is None); clustered or temporal data must pass strata.

    Args:
        x: Feature values, NaN where missing.
        y: Target codes, one per row.
        grid: The specifications.
        seed: Run seed.
        finding: Name of the finding, part of the permutation stream's name.
        permutations: Number of permuted curves. A value that is not an
            integer of at least 1 raises `ValueError`.
        strata: Labels within which the target is permuted, or None.

    Returns:
        The observed curve statistics and their permutation p-values, each
        ``(1 + #{permuted >= observed}) / (permutations + 1)``.
    """
    qualification.count(permutations, "permutations", minimum=1)
    median, share = curve_statistics(x, y, grid)
    rng = randomness.component_rng(seed, f"f6.curve.{finding}", 0)
    curves = [
        curve_statistics(x, controls.permute_within(y, strata, rng), grid)
        for index in range(permutations)
    ]
    medians = [m for m, share in curves]
    shares = [share for m, share in curves]
    return CurveTest(
        permutations,
        median,
        share,
        (1.0 + sum(m >= median - 1e-12 for m in medians))
        / (permutations + 1.0),
        (1.0 + sum(s >= share - 1e-12 for s in shares)) / (permutations + 1.0),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class Stability:
    """The stability verdict for one finding.

    Attributes:
        name: The finding (feature name).
        status: `stable`, `fragile` or `not_evaluable`.
        reason: Why, in one sentence.
        headline: Effect under the default specification, or None.
        numbers: Median, sign share, size ratio and interval share.
        specs: Per-specification estimates (the specification curve).
        skipped: (specification hash, reason) for specifications not run.
        attribution: Per-choice option means and eta squared.
        curve: Joint specification-curve test, or None.
        check: The `Check` record, or None when not evaluable.
    """

    name: str
    status: str
    reason: str
    headline: float | None
    numbers: dict[str, float | None]
    specs: tuple[SpecEffect, ...]
    skipped: tuple[tuple[str, str], ...]
    attribution: dict[str, dict[str, Any]]
    curve: CurveTest | None
    check: qualification.Check | None

    def to_dict(self) -> dict[str, Any]:
        """Converts the verdict to a JSON-ready dictionary.

        Returns:
            The fields by name, with non-finite `numbers` written as None,
            `specs` and `skipped` as lists and nested records as
            dictionaries (or None).
        """
        numbers = {
            k: (v if v is not None and math.isfinite(v) else None)
            for k, v in self.numbers.items()
        }
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "headline": self.headline,
            "numbers": numbers,
            "specs": [s.to_dict() for s in self.specs],
            "skipped": [list(item) for item in self.skipped],
            "attribution": self.attribution,
            "curve": None if self.curve is None else self.curve.to_dict(),
            "check": None if self.check is None else self.check.to_dict(),
        }


def assess_finding(
    name: str,
    x: numpy.ndarray,
    y: numpy.ndarray,
    grid: multiverse.Grid,
    seed: int,
    *,
    rule: Rule = DEFAULT_RULE,
    bootstrap: int = BOOTSTRAP,
    strata: numpy.ndarray | None = None,
    curve_permutations: int = 0,
    cache: multiverse.ResultCache | None = None,
    budget_seconds: float | None = None,
    clock: Callable[[], float] | None = None,
    gate: certificates.GateStatus = certificates.GateStatus.DESCRIPTIVE,
) -> Stability:
    """Re-runs a finding over the grid and applies the survival rule.

    Args:
        name: The finding, used in keys and reports.
        x: Feature values (NaN where missing), one per exploration row.
        y: Target codes, same length.
        grid: The specifications.
        seed: Run seed.
        rule: Survival thresholds.
        bootstrap: Resamples per specification.
        strata: Group or block labels for the bootstrap and the curve test.
        curve_permutations: Permuted curves for the joint test; 0 skips it.
        cache: Optional result cache, for resuming.
        budget_seconds: Time allowed for this finding.
        clock: Monotonic clock, replaceable in tests.
        gate: Whether a current certificate lets a `fragile` verdict
            quarantine the finding from confirmation.

    Returns:
        The verdict. Fewer than `MIN_SPECS` estimates give `not_evaluable`,
        with no curve test and no `Check`.

    Raises:
        ValueError: If `x` and `y` differ in length.
    """
    values = numpy.asarray(x, dtype=float)
    codes = numpy.asarray(y, dtype=float)
    if len(values) != len(codes):
        raise ValueError("x and y must have one entry per row")
    options = {} if clock is None else {"clock": clock}
    ran = multiverse.run_grid(
        grid,
        lambda spec: estimate_spec(
            values, codes, spec, seed, name, strata, bootstrap
        ),
        finding=name,
        seed=seed,
        cache=cache,
        budget_seconds=budget_seconds,
        **options,
    )
    effects = tuple(
        SpecEffect(
            spec.sha256,
            spec.to_dict(),
            ran.results[spec.sha256].get("effect"),
            ran.results[spec.sha256].get("low"),
            ran.results[spec.sha256].get("high"),
        )
        for spec in grid.specs
        if spec.sha256 in ran.results
    )
    headline_row = next(
        (e for e in effects if e.sha256 == grid.default.sha256), None
    )
    headline = None if headline_row is None else headline_row.effect
    status, reason, numbers = verdict(effects, headline, rule)
    curve = None
    if curve_permutations > 0 and status != NOT_EVALUABLE:
        curve = curve_test(
            values,
            codes,
            grid,
            seed,
            name,
            permutations=curve_permutations,
            strata=strata,
        )
    check = None
    if status != NOT_EVALUABLE:
        lows = [e.low for e in effects if e.low is not None]
        highs = [e.high for e in effects if e.high is not None]
        envelope = (min(lows), max(highs)) if lows and highs else None
        check = qualification.Check(
            check_id=f"{DETECTOR_ID}:{name}",
            null_hypothesis="the finding's sign and size are unaffected by "
            "the declared analysis choices",
            statistic=float(numbers["sign_share"] or 0.0),
            p_value=None if curve is None else curve.p_median,
            interval=envelope,
            effect=float(numbers["median"] or 0.0),
            n=len(values),
            decision=(
                qualification.Decision.PASS
                if status == STABLE
                else certificates.demote(
                    qualification.Decision.QUARANTINE, gate
                )
            ),
            set_id=qualification.SetId.EXPLORATION,
            seed_path=randomness.seed_path(seed, f"f6.stability.{name}", 0),
            tier=qualification.Tier.UNVERIFIED,
        )
        if check.p_value is None and check.interval is None:
            check = dataclasses.replace(
                check, interval=(check.effect, check.effect)
            )
    return Stability(
        name,
        status,
        reason,
        headline,
        numbers,
        effects,
        ran.skipped,
        attribution(effects),
        curve,
        check,
    )


def code_digest() -> str:
    """Hashes the multiverse and stability sources.

    Returns:
        The code hash that keys the stability certificates.
    """
    return certificates.code_hash(multiverse, sys.modules[__name__])


def gate_status(
    store: certificates.CertificateStore | None,
) -> certificates.GateStatus:
    """Decides whether stability verdicts may act.

    Args:
        store: The certificate store, or None when there is none.

    Returns:
        `GATING` only when every defect type in `DEFECT_TYPES` has a passing
        certificate for the current code; `DESCRIPTIVE` otherwise, including
        when `store` is None.
    """
    if store is None:
        return certificates.GateStatus.DESCRIPTIVE
    digest = code_digest()
    for defect in DEFECT_TYPES:
        if store.status(DETECTOR_ID, digest, defect) is not (
            certificates.GateStatus.GATING
        ):
            return certificates.GateStatus.DESCRIPTIVE
    return certificates.GateStatus.GATING
