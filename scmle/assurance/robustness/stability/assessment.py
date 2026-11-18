"""The stability verdict of a finding and the certificate gate."""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
import math
import sys
from typing import Any

import numpy

from scmle.assurance.detectors import certificates
from scmle.assurance.robustness import multiverse
from scmle.assurance.robustness.stability import estimation
from scmle.assurance.robustness.stability import permutation
from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.runtime import randomness


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
    specs: tuple[estimation.Variant, ...]
    skipped: tuple[tuple[str, str], ...]
    attribution: dict[str, dict[str, Any]]
    curve: permutation.Curve | None
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
    rule: estimation.Thresholds = estimation.DEFAULT_RULE,
    bootstrap: int = estimation.BOOTSTRAP,
    strata: numpy.ndarray | None = None,
    curve_permutations: int = 0,
    cache: multiverse.Memo | None = None,
    budget_seconds: float | None = None,
    clock: Callable[[], float] | None = None,
    gate: certificates.Authority = certificates.Authority.DESCRIPTIVE,
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
        raise errors.InputError("x and y must have one entry per row")
    options = {} if clock is None else {"clock": clock}
    ran = multiverse.run_grid(
        grid,
        lambda spec: estimation.estimate(
            values, codes, spec, seed, name, strata, bootstrap
        ),
        finding=name,
        seed=seed,
        cache=cache,
        budget_seconds=budget_seconds,
        **options,
    )
    return judge(
        name,
        values,
        codes,
        grid,
        seed,
        ran,
        rule=rule,
        strata=strata,
        curve_permutations=curve_permutations,
        gate=gate,
    )


def judge(
    name: str,
    values: numpy.ndarray,
    codes: numpy.ndarray,
    grid: multiverse.Grid,
    seed: int,
    ran: multiverse.Sweep,
    *,
    rule: estimation.Thresholds,
    strata: numpy.ndarray | None,
    curve_permutations: int,
    gate: certificates.Authority,
) -> Stability:
    """Applies the survival rule to the estimates the grid produced.

    Args:
        name: The finding, used in keys and reports.
        values: Feature values (NaN where missing), one per exploration row.
        codes: Target codes, same length.
        grid: The specifications.
        seed: Run seed.
        ran: The per-specification estimates and the skipped specifications.
        rule: Survival thresholds.
        strata: Group or block labels for the curve test.
        curve_permutations: Permuted curves for the joint test; 0 skips it.
        gate: Whether a current certificate lets a `fragile` verdict
            quarantine the finding from confirmation.

    Returns:
        The verdict, with the curve test and the `Check` unless it is
        `not_evaluable`.
    """
    effects = tuple(
        estimation.Variant(
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
    status, reason, numbers = estimation.verdict(effects, headline, rule)
    curve = None
    if curve_permutations > 0 and status != estimation.NOT_EVALUABLE:
        curve = permutation.specify(
            values,
            codes,
            grid,
            seed,
            name,
            permutations=curve_permutations,
            strata=strata,
        )
    return Stability(
        name,
        status,
        reason,
        headline,
        numbers,
        effects,
        ran.skipped,
        estimation.attribution(effects),
        curve,
        certify(name, effects, numbers, status, curve, values, seed, gate),
    )


def certify(
    name: str,
    effects: Sequence[estimation.Variant],
    numbers: dict[str, float | None],
    status: str,
    curve: permutation.Curve | None,
    values: numpy.ndarray,
    seed: int,
    gate: certificates.Authority,
) -> qualification.Check | None:
    """Builds the `Check` record of a stability verdict.

    Args:
        name: The finding.
        effects: One entry per evaluated specification.
        numbers: Median, sign share, size ratio and interval share.
        status: `stable`, `fragile` or `not_evaluable`.
        curve: The joint specification-curve test, or None.
        values: Feature values, one per exploration row.
        seed: Run seed.
        gate: Whether a current certificate lets a `fragile` verdict
            quarantine the finding from confirmation.

    Returns:
        None when the finding is `not_evaluable`; otherwise the check, whose
        interval is the envelope of the perturbation intervals, or the
        median effect itself when neither an interval nor a p-value exists.
    """
    if status == estimation.NOT_EVALUABLE:
        return None
    lows = [e.low for e in effects if e.low is not None]
    highs = [e.high for e in effects if e.high is not None]
    envelope = (min(lows), max(highs)) if lows and highs else None
    check = qualification.Check(
        identifier=f"{estimation.DETECTOR_ID}:{name}",
        null="the finding's sign and size are unaffected by "
        "the declared analysis choices",
        statistic=float(numbers["sign_share"] or 0.0),
        p_value=None if curve is None else curve.p_median,
        interval=envelope,
        effect=float(numbers["median"] or 0.0),
        n=len(values),
        decision=(
            qualification.Decision.PASS
            if status == estimation.STABLE
            else certificates.demote(qualification.Decision.QUARANTINE, gate)
        ),
        subset=qualification.Subset.EXPLORATION,
        seeds=randomness.seed_path(seed, f"f6.stability.{name}", 0),
        tier=qualification.Tier.UNVERIFIED,
    )
    if check.p_value is None and check.interval is None:
        check = dataclasses.replace(
            check, interval=(check.effect, check.effect)
        )
    return check


def code_digest() -> str:
    """Hashes the multiverse and stability sources.

    The stability sources are its estimation, permutation and assessment
    modules.

    Returns:
        The code hash that keys the stability certificates.
    """
    return certificates.code_hash(
        multiverse, estimation, permutation, sys.modules[__name__]
    )


def adjudication(
    store: certificates.Store | None,
) -> certificates.Authority:
    """Decides whether stability verdicts may act.

    Args:
        store: The certificate store, or None when there is none.

    Returns:
        `GATING` only when every defect type in `DEFECT_TYPES` has a passing
        certificate for the current code; `DESCRIPTIVE` otherwise, including
        when `store` is None.
    """
    if store is None:
        return certificates.Authority.DESCRIPTIVE
    digest = code_digest()
    for defect in estimation.DEFECT_TYPES:
        if store.status(estimation.DETECTOR_ID, digest, defect) is not (
            certificates.Authority.GATING
        ):
            return certificates.Authority.DESCRIPTIVE
    return certificates.Authority.GATING
