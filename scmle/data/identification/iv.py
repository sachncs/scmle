"""Instrumental variables: just-identified 2SLS with weak-instrument care (F10).

One endogenous regressor `d`, one instrument `z` and optional exogenous
controls `W` (with an intercept). All variables are residualised on `W`; then

    beta   = sum(z y) / sum(z d)                       (the Wald ratio),
    Var    = sum(z^2 u^2) / sum(z d)^2,   u = y - beta d  (robust, HC1),
    F      = pi^2 / Var(pi),   pi = sum(z d) / sum(z z)   (robust first stage).

With one instrument the robust first-stage F is the effective F statistic of
Olea and Pflueger (2013); that equality is from memory and not re-verified.

The Anderson-Rubin set inverts the robust test of `beta = b0`: the statistic
`(pi_y - b0 pi_d)^2 / (v_yy - 2 b0 v_yd + b0^2 v_dd)` is below the chi-square
quantile on a set that solves a quadratic in `b0`; it is a bounded interval, a
union of two rays, or the whole line, and its coverage does not depend on the
instrument's strength. The 2SLS interval is reported as the headline only when
`F >= F_THRESHOLD`; otherwise the headline is the Anderson-Rubin set. The
threshold 10 is the common rule of thumb, not a Stock-Yogo critical value;
what it buys is measured in the certificate.

Not done: overidentification tests (one instrument means the model is just
identified and `exclusion` cannot be tested), several instruments, several
endogenous regressors. The estimand is the local average treatment effect for
compliers and needs monotonicity; none of exclusion, independence or
monotonicity is checked here.
"""

import dataclasses
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

F_THRESHOLD = 10.0


@dataclasses.dataclass(frozen=True)
class IvResult:
    """A just-identified IV fit.

    Attributes:
        estimate: The 2SLS (Wald) estimate.
        se: Robust standard error.
        wald_low: Lower limit of the 2SLS interval.
        wald_high: Upper limit of the 2SLS interval.
        first_stage: First-stage coefficient.
        f_stat: Robust first-stage F.
        weak: Whether `f_stat < F_THRESHOLD`.
        ar_kind: `bounded`, `complement` or `whole_line`.
        ar_low: Lower end (bounded) or end of the left ray (complement).
        ar_high: Upper end (bounded) or start of the right ray.
        low: Headline lower limit (finite only when not unbounded).
        high: Headline upper limit.
        headline: `wald` or `anderson_rubin`.
        level: Confidence level.
        n: Rows.
    """

    estimate: float
    se: float
    wald_low: float
    wald_high: float
    first_stage: float
    f_stat: float
    weak: bool
    ar_kind: str
    ar_low: float
    ar_high: float
    low: float
    high: float
    headline: str
    level: float
    n: int

    def excludes(self, value: float) -> bool:
        """Whether the headline set excludes `value`."""
        if self.headline == "wald":
            return not self.wald_low <= value <= self.wald_high
        if self.ar_kind == "bounded":
            return not self.ar_low <= value <= self.ar_high
        if self.ar_kind == "complement":
            return self.ar_low < value < self.ar_high
        return False

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record (infinite ends are written as strings)."""
        record = dataclasses.asdict(self)
        for key, value in record.items():
            if isinstance(value, float) and not math.isfinite(value):
                record[key] = str(value)
        return record


def residualise(
    columns: list[np.ndarray], controls: np.ndarray
) -> list[np.ndarray]:
    """Residuals of each column on the controls (with an intercept)."""
    design = np.column_stack([np.ones(len(controls)), controls])
    out = []
    for column in columns:
        coef, *_ = np.linalg.lstsq(design, column, rcond=None)
        out.append(column - design @ coef)
    return out


def anderson_rubin(
    z: np.ndarray, y: np.ndarray, d: np.ndarray, level: float
) -> tuple[str, float, float]:
    """The Anderson-Rubin set for one instrument as `(kind, low, high)`."""
    szz = float(z @ z)
    pi_y, pi_d = float(z @ y) / szz, float(z @ d) / szz
    r_y, r_d = y - pi_y * z, d - pi_d * z
    v_yy = float(np.sum(z**2 * r_y**2)) / szz**2
    v_dd = float(np.sum(z**2 * r_d**2)) / szz**2
    v_yd = float(np.sum(z**2 * r_y * r_d)) / szz**2
    c = float(stats.chi2.ppf(level, 1))
    a = pi_d**2 - c * v_dd
    b = -2.0 * (pi_y * pi_d - c * v_yd)
    k = pi_y**2 - c * v_yy
    if a > 0:
        disc = b * b - 4 * a * k
        if disc <= 0:
            return "bounded", float("nan"), float("nan")
        root = math.sqrt(disc)
        return "bounded", (-b - root) / (2 * a), (-b + root) / (2 * a)
    if a < 0:
        disc = b * b - 4 * a * k
        if disc <= 0:
            return "whole_line", -math.inf, math.inf
        root = math.sqrt(disc)
        left, right = sorted(((-b - root) / (2 * a), (-b + root) / (2 * a)))
        return "complement", left, right
    return "whole_line", -math.inf, math.inf


def estimate(
    frame: pd.DataFrame,
    treatment: str,
    outcome: str,
    instrument: str,
    controls: tuple[str, ...] = (),
    level: float = 0.95,
) -> IvResult:
    """Fits the just-identified model.

    Raises:
        ValueError: If the instrument has no variation after the controls or
            the data hold missing values.
    """
    names = [treatment, outcome, instrument, *controls]
    data = frame[names].to_numpy(dtype=float)
    if not np.isfinite(data).all():
        raise ValueError("IV columns must not hold missing values")
    n = len(data)
    d, y, z = data[:, 0], data[:, 1], data[:, 2]
    w = data[:, 3:] if controls else np.empty((n, 0))
    d, y, z = residualise([d, y, z], w)
    szd, szz = float(z @ d), float(z @ z)
    if szz <= 1e-12 or abs(szd) <= 1e-12:
        raise ValueError("the instrument has no first-stage variation")
    scale = n / (n - 2 - w.shape[1])
    beta = float(z @ y) / szd
    u = y - beta * d
    se = math.sqrt(scale * float(np.sum(z**2 * u**2))) / abs(szd)
    pi = szd / szz
    v = d - pi * z
    f_stat = pi**2 / (scale * float(np.sum(z**2 * v**2)) / szz**2)
    quantile = float(stats.norm.ppf(0.5 + level / 2))
    wald = (beta - quantile * se, beta + quantile * se)
    kind, ar_low, ar_high = anderson_rubin(z, y, d, level)
    weak = f_stat < F_THRESHOLD
    if weak:
        headline = "anderson_rubin"
        low = ar_low if kind == "bounded" else -math.inf
        high = ar_high if kind == "bounded" else math.inf
    else:
        headline, low, high = "wald", wald[0], wald[1]
    return IvResult(
        estimate=beta,
        se=se,
        wald_low=wald[0],
        wald_high=wald[1],
        first_stage=pi,
        f_stat=float(f_stat),
        weak=bool(weak),
        ar_kind=kind,
        ar_low=ar_low,
        ar_high=ar_high,
        low=low,
        high=high,
        headline=headline,
        level=level,
        n=n,
    )


def exclusion_bound(result: IvResult, direct_effect: float) -> float:
    """Bias of the 2SLS estimate if the instrument had a direct effect.

    A direct effect `gamma` of the instrument on the outcome moves the Wald
    ratio by `gamma / pi`.
    """
    return direct_effect / result.first_stage
