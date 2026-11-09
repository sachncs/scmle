"""Sensitivity of a causal estimate to unmeasured confounding.

E-value (VanderWeele and Ding 2017). For a risk ratio `RR >= 1` the minimum
strength of association, on the risk-ratio scale, that an unmeasured
confounder would need with both treatment and outcome to explain the
association away is `RR + sqrt(RR (RR - 1))`; for `RR < 1` the reciprocal is
used. For the confidence limit nearest the null the same formula applies, and
the E-value is 1 when the interval contains 1. A standardised mean difference
`d` is converted by `RR ~ exp(0.91 d)` (VanderWeele and Ding). Checked here
against the paper's worked example (`RR = 3.9` gives 7.26); no frozen output
of the R package `EValue` was available.

Robustness value and partial-R-squared bounds (Cinelli and Hazlett 2020,
JRSS-B 82(1)) for a linear model `Y ~ D + X`. With `t` the t statistic of `D`
and `df` the residual degrees of freedom, `f = t / sqrt(df)`:

    RV_q       = (sqrt(f_q^4 + 4 f_q^2) - f_q^2) / 2,   f_q = q |f|,
    RV_{q,a}   = the same with f_q = q |f| - t*_{a, df-1} / sqrt(df - 1),
    |bias|     = se sqrt(df) sqrt(R2yz R2dz / (1 - R2dz)),
    se_adj     = se sqrt((1 - R2yz) / (1 - R2dz)) sqrt(df / (df - 1)).

The formulas of the robustness value (eq. 18 to 20), the standard-error
adjustment (eq. 12) and the bound were read in the paper on 2026-11-05 and
agree. The paper's Table 1 for the same example (estimate 0.097, `t = 4.18`,
`df = 783`: partial R2 2.2%, `RV_1 = 13.9%`, `RV_{1,0.05} = 7.6%`) is
reproduced to the printed digits; the R package itself was not run here. One
rule of the paper is NOT implemented: when `RV_{q,a} <= 1 - 1/f_q^2` (which
needs `f_q > 1`) the paper sets `RV_{q,a} = (f_q^2 - f*^2) / (1 + f_q^2)`
with `f* = t*_{a,df-1} / sqrt(df - 1)`, so that no weaker confounder
overturns the conclusion; the code applies eq. 20 unchanged.

Benchmarking against an observed covariate `x_j`: the confounder is allowed
to be `kd` times as strongly related to the treatment and `ky` times as
strongly related to the outcome as `x_j`. The conversions used are

    R2dz   = kd f2_dxj,    f2_dxj = R2_{D~xj|X-j} / (1 - R2_{D~xj|X-j}),
    R2zxj  = kd R2_dxj^2 / ((1 - kd R2_dxj)(1 - R2_dxj)),
    R2yz   = ((sqrt(ky) + sqrt(R2zxj)) / sqrt(1 - R2zxj))^2
             * R2_{Y~xj|D,X} / (1 - R2_{Y~xj|D,X}),

checked on 2026-11-05 against the paper's Supplement B (eq. 76 to 84:
`eta = (sqrt(kY) + f_KD f_D) / sqrt(1 - f_KD^2 f_D^2)`, `R2zxj` equal to
`f_KD^2 f_D^2`, and the bound on `R2yz` equal to `eta^2` times the covariate's
partial Cohen's f^2) and found to agree algebraically; the paper defines `kD`
and `kY` on the partial R2 of `Xj` given `X-j` (eq. 21) and states the bound
as an upper bound that is tight. The package was not run. The simulation
certificate measures whether the resulting bound covers
the truth, and a failure there is published, not tuned away. Benchmarking is
an assumption about the unobserved, and the report says so. The bound applies
to the linear-model coefficient of `D`; for an AIPW estimate it is a
description of how fragile that linear summary is.
"""

import dataclasses
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

D_TO_RR = 0.91
INFEASIBLE = 0.999


def evalue(rr: float) -> float:
    """Computes the E-value of a risk ratio (reciprocal taken below 1).

    Args:
        rr: The risk ratio, which must be positive.

    Returns:
        The minimum confounder strength, on the risk-ratio scale, that
        explains the association away: ``RR + sqrt(RR (RR - 1))`` after
        taking the reciprocal of a ratio below 1.

    Raises:
        ValueError: If `rr` is not positive.
    """
    if not rr > 0:
        raise ValueError("a risk ratio must be positive")
    ratio = rr if rr >= 1 else 1.0 / rr
    return ratio + math.sqrt(ratio * (ratio - 1.0))


def evalue_interval(
    estimate: float, low: float, high: float
) -> dict[str, float]:
    """Computes the E-values of a risk ratio and of its limit nearest null.

    Args:
        estimate: The risk ratio.
        low: Lower confidence limit.
        high: Upper confidence limit.

    Returns:
        A mapping with `point`, the E-value of the estimate, and `limit`,
        the E-value of the limit nearest 1 (1.0 when the interval contains
        1).

    Raises:
        ValueError: If the estimate does not lie within its limits.
    """
    if not low <= estimate <= high:
        raise ValueError("the estimate must lie within its limits")
    nearest = low if estimate >= 1 else high
    return {
        "point": evalue(estimate),
        "limit": 1.0 if low <= 1.0 <= high else evalue(nearest),
    }


def smd_to_rr(d: float) -> float:
    """Converts a standardised mean difference to a risk ratio.

    Args:
        d: The standardised mean difference.

    Returns:
        The approximate risk ratio ``exp(0.91 d)``.
    """
    return math.exp(D_TO_RR * d)


@dataclasses.dataclass(frozen=True)
class OlsFit:
    """The coefficient of `D` in `Y ~ D + X` and what the bounds need.

    Attributes:
        estimate: Coefficient of `D`.
        se: Its conventional standard error.
        dof: Residual degrees of freedom.
        t: The t statistic.
        r2yd: Partial R-squared of `D` with `Y` given `X`.
        benchmarks: Per covariate, `(R2_{D~xj|X-j}, R2_{Y~xj|D,X})`.
    """

    estimate: float
    se: float
    dof: int
    t: float
    r2yd: float
    benchmarks: dict[str, tuple[float, float]]


def partial_r2(t: float, dof: int) -> float:
    """Computes a partial R-squared from a t statistic.

    Args:
        t: The t statistic of a coefficient.
        dof: Residual degrees of freedom.

    Returns:
        ``t^2 / (t^2 + dof)``.
    """
    return t * t / (t * t + dof)


def ols_fit(
    frame: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: tuple[str, ...],
) -> OlsFit:
    """Fits `Y ~ D + X` and the benchmark partial R-squareds.

    Covariates must be numeric (encode categories first).

    Args:
        frame: The data.
        treatment: Treatment column `D`.
        outcome: Outcome column `Y`.
        covariates: Numeric adjustment columns `X`.

    Returns:
        The coefficient of `D` with its standard error, t statistic and
        residual degrees of freedom, and the benchmark partial R-squareds of
        each covariate.

    Raises:
        ValueError: If fewer than three residual degrees of freedom remain.
    """
    n = len(frame)
    x = (
        frame[list(covariates)].to_numpy(dtype=float)
        if covariates
        else np.empty((n, 0))
    )
    d = frame[treatment].to_numpy(dtype=float)
    y = frame[outcome].to_numpy(dtype=float)
    full = np.column_stack([np.ones(n), d, x])
    dof = n - full.shape[1]
    if dof < 3:
        raise ValueError("too few rows for the regression")
    coef = np.linalg.lstsq(full, y, rcond=None)[0]
    resid = y - full @ coef
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(full.T @ full)
    se = float(math.sqrt(cov[1, 1]))
    t = float(coef[1] / se)
    benchmarks = {}
    for j, name in enumerate(covariates):
        t_y = float(coef[2 + j] / math.sqrt(cov[2 + j, 2 + j]))
        r2y = partial_r2(t_y, dof)
        design = np.column_stack([np.ones(n), x])
        cd = np.linalg.lstsq(design, d, rcond=None)[0]
        rd = d - design @ cd
        dof_d = n - design.shape[1]
        cov_d = float(rd @ rd) / dof_d * np.linalg.inv(design.T @ design)
        t_d = float(cd[1 + j] / math.sqrt(cov_d[1 + j, 1 + j]))
        benchmarks[name] = (partial_r2(t_d, dof_d), r2y)
    return OlsFit(float(coef[1]), se, dof, t, partial_r2(t, dof), benchmarks)


def robustness_value(
    t: float, dof: int, q: float = 1.0, alpha: float | None = None
) -> float:
    """Computes the robustness value of an estimate.

    It is the strength a confounder needs, with both treatment and outcome,
    to cut the estimate by a fraction `q`.

    Args:
        t: The t statistic of the treatment coefficient.
        dof: Residual degrees of freedom.
        q: Fraction of the estimate to explain away (1 for all of it).
        alpha: Significance level for the version that makes the estimate
            lose significance, or None for the point-estimate version.

    Returns:
        The robustness value as a partial R-squared; 0.0 when `alpha` is
        given and the estimate is not significant at that level.
    """
    f = abs(t) / math.sqrt(dof)
    fq = q * f
    if alpha is not None:
        crit = float(stats.t.ppf(1 - alpha / 2, dof - 1))
        fq -= crit / math.sqrt(dof - 1)
        if fq <= 0:
            return 0.0
    return 0.5 * (math.sqrt(fq**4 + 4 * fq**2) - fq**2)


def bias_bound(fit: OlsFit, r2yz: float, r2dz: float) -> float:
    """Computes the largest absolute bias of a confounder.

    Args:
        fit: The fitted regression.
        r2yz: Partial R-squared of the confounder with the outcome, in
            [0, 1).
        r2dz: Partial R-squared of the confounder with the treatment, in
            [0, 1).

    Returns:
        ``se * sqrt(dof) * sqrt(r2yz * r2dz / (1 - r2dz))``.

    Raises:
        ValueError: If either partial R-squared is outside [0, 1).
    """
    if not (0 <= r2yz < 1 and 0 <= r2dz < 1):
        raise ValueError("partial R-squareds must lie in [0, 1)")
    return fit.se * math.sqrt(fit.dof) * math.sqrt(r2yz * r2dz / (1 - r2dz))


def adjusted_interval(
    fit: OlsFit, r2yz: float, r2dz: float, level: float = 0.95
) -> tuple[float, float]:
    """Computes the interval over both directions of a confounder.

    Args:
        fit: The fitted regression.
        r2yz: Partial R-squared of the confounder with the outcome.
        r2dz: Partial R-squared of the confounder with the treatment.
        level: Confidence level.

    Returns:
        The lower and upper limits, the estimate moved by the bias bound and
        the adjusted standard error times a t quantile. Partial R-squareds
        outside [0, 1) are refused with a `ValueError` by `bias_bound`.
    """
    bias = bias_bound(fit, r2yz, r2dz)
    se_adj = (
        fit.se
        * math.sqrt((1 - r2yz) / (1 - r2dz))
        * math.sqrt(fit.dof / (fit.dof - 1))
    )
    z = float(stats.t.ppf(0.5 + level / 2, fit.dof - 1))
    return fit.estimate - bias - z * se_adj, fit.estimate + bias + z * se_adj


def benchmark_strength(
    r2dxj: float, r2yxj: float, kd: float, ky: float
) -> tuple[float, float]:
    """Converts benchmark multiples into confounder strengths.

    Args:
        r2dxj: Partial R-squared of the benchmark covariate with the
            treatment given the others.
        r2yxj: Partial R-squared of the benchmark covariate with the
            outcome given the treatment and the others.
        kd: How many times as strongly related to the treatment as the
            benchmark.
        ky: How many times as strongly related to the outcome as the
            benchmark.

    Returns:
        ``(R2yz, R2dz)``, the confounder's partial R-squareds with the
        outcome and with the treatment.

    Raises:
        ValueError: If a multiple is so large that the implied strength
            reaches 1 or more.
    """
    r2dz = kd * r2dxj / (1 - r2dxj)
    if r2dz >= 1:
        raise ValueError(
            "the benchmark multiple is too large for this covariate"
        )
    r2zx = kd * r2dxj**2 / ((1 - kd * r2dxj) * (1 - r2dxj))
    if r2zx >= 1:
        raise ValueError(
            "the benchmark multiple is too large for this covariate"
        )
    r2yz = (
        ((math.sqrt(ky) + math.sqrt(r2zx)) / math.sqrt(1 - r2zx)) ** 2
        * r2yxj
        / (1 - r2yxj)
    )
    if r2yz >= 1:
        raise ValueError(
            "the benchmark multiple is too large for this covariate"
        )
    return r2yz, r2dz


@dataclasses.dataclass(frozen=True)
class SensitivityReport:
    """Result of the sensitivity gate.

    Attributes:
        estimate: Linear-model coefficient of `D`.
        rv: Robustness value for a full reduction (`q = 1`).
        rv_alpha: Robustness value for losing significance.
        benchmark: What bounded the confounder (words).
        r2yz: Assumed partial R2 of the confounder with the outcome.
        r2dz: Assumed partial R2 with the treatment.
        bias: Largest absolute bias at that strength.
        low: Lower limit of the adjusted interval.
        high: Upper limit.
        holds: Whether the adjusted interval still excludes zero.
        covariate: The covariate that set the strength, if benchmarked.
    """

    estimate: float
    rv: float
    rv_alpha: float
    benchmark: str
    r2yz: float
    r2dz: float
    bias: float
    low: float
    high: float
    holds: bool
    covariate: str | None

    def to_dict(self) -> dict[str, Any]:
        """Builds the JSON-ready record.

        Returns:
            A mapping of every field.
        """
        return dataclasses.asdict(self)


def assess(
    frame: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: tuple[str, ...],
    benchmark: str = "strongest_observed",
    strength: float = 1.0,
    level: float = 0.95,
) -> SensitivityReport:
    """Runs the charter's sensitivity gate for the linear effect summary.

    `strongest_observed` takes the covariate whose benchmarked bound is
    widest, with multiples `kd = ky = strength`; `fixed` uses `strength` as
    both partial R-squareds.

    Args:
        frame: The data.
        treatment: Treatment column.
        outcome: Outcome column.
        covariates: Numeric adjustment columns.
        benchmark: `strongest_observed` (default) or `fixed`.
        strength: The multiple (or, for `fixed`, the partial R-squared).
        level: Confidence level.

    Returns:
        The report: robustness values, the assumed strengths and their
        bound, the adjusted interval, and whether it still excludes zero.
        When a multiple is infeasible for a covariate its strengths are
        set to 0.999 and the gate fails.

    Raises:
        ValueError: If benchmarking is requested without covariates.
    """
    fit = ols_fit(frame, treatment, outcome, covariates)
    chosen: str | None = None
    if benchmark == "fixed":
        r2yz = r2dz = float(strength)
        words = f"fixed partial R2 {strength} with treatment and outcome"
    else:
        if not covariates:
            raise ValueError("benchmarking needs at least one covariate")
        best = -1.0
        r2yz = r2dz = 0.0
        for name, (r2d, r2y) in fit.benchmarks.items():
            try:
                a, b = benchmark_strength(r2d, r2y, strength, strength)
            except ValueError:
                a, b = INFEASIBLE, INFEASIBLE
            width = bias_bound(fit, min(a, INFEASIBLE), min(b, INFEASIBLE))
            if width > best:
                best, r2yz, r2dz, chosen = width, a, b, name
        words = (
            f"a confounder {strength} times as strong as {chosen}, the "
            "observed covariate with the widest bound (an assumption about "
            "the unobserved)"
        )
    if INFEASIBLE in (r2yz, r2dz):
        words += (
            "; the multiple exceeds what the data allow for that covariate, "
            f"so both strengths are set to {INFEASIBLE} and the gate fails"
        )
    r2yz, r2dz = min(r2yz, INFEASIBLE), min(r2dz, INFEASIBLE)
    low, high = adjusted_interval(fit, r2yz, r2dz, level)
    return SensitivityReport(
        estimate=fit.estimate,
        rv=robustness_value(fit.t, fit.dof, 1.0),
        rv_alpha=robustness_value(fit.t, fit.dof, 1.0, 1 - level),
        benchmark=words,
        r2yz=r2yz,
        r2dz=r2dz,
        bias=bias_bound(fit, r2yz, r2dz),
        low=low,
        high=high,
        holds=bool(low > 0 or high < 0),
        covariate=chosen,
    )
