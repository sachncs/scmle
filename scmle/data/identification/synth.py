"""Synthetic control with placebo-in-space inference (F10).

One treated unit, many donors, a pre-period and a post-period. The weights
are non-negative, sum to one and minimise the squared distance between the
treated unit's pre-period outcomes and the weighted donors' (Abadie, Diamond
and Hainmueller 2010, from memory; outcomes only, no covariates). The effect
is the mean post-period gap. Inference re-assigns the treatment to each
donor in turn (the treated unit is dropped from the pool), refits, and ranks
the post/pre root-mean-squared prediction error ratio:

    p = (1 + #{donors whose ratio >= the treated unit's}) / (J + 1).

The smallest attainable p-value is `1 / (J + 1)`, so at level `alpha` at least
`ceil(1 / alpha) - 1` donors are needed and the module refuses with fewer.
With one treated unit there is no confirmation sample to hold out; the result
is a single-unit inference whose validity rests on the donors being
exchangeable with the treated unit apart from the treatment, and the tier rule
caps such a claim at `attested`.
"""

import dataclasses
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize


@dataclasses.dataclass(frozen=True)
class SynthResult:
    """A synthetic control fit.

    Attributes:
        effect: Mean post-period gap (treated minus synthetic).
        gaps: Gap per post period.
        weights: Donor to weight (non-zero weights only).
        pre_rmspe: Pre-period root mean squared prediction error.
        post_rmspe: Post-period root mean squared gap.
        ratio: Post over pre RMSPE.
        p_value: Placebo-in-space rank p-value.
        donors: Donors in the pool.
        min_p: Smallest attainable p-value, `1 / (donors + 1)`.
        placebo_ratios: Ratio per donor.
        alpha: Level the donor count was checked against.
    """

    effect: float
    gaps: tuple[float, ...]
    weights: dict[str, float]
    pre_rmspe: float
    post_rmspe: float
    ratio: float
    p_value: float
    donors: int
    min_p: float
    placebo_ratios: dict[str, float]
    alpha: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)


def fit_weights(target: np.ndarray, donors: np.ndarray) -> np.ndarray:
    """Simplex-constrained least-squares weights (donors are columns)."""
    k = donors.shape[1]
    start = np.full(k, 1.0 / k)

    def loss(w: np.ndarray) -> float:
        return float(np.sum((target - donors @ w) ** 2))

    def gradient(w: np.ndarray) -> np.ndarray:
        return -2.0 * donors.T @ (target - donors @ w)

    result = optimize.minimize(
        loss,
        start,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    weights = np.clip(result.x, 0.0, None)
    return weights / weights.sum()


def ratio_for(
    pre: np.ndarray, post: np.ndarray, treated: int, pool: list[int]
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Fit unit `treated` on `pool`; returns RMSPEs, ratio, gaps, weights."""
    w = fit_weights(pre[:, treated], pre[:, pool])
    pre_gap = pre[:, treated] - pre[:, pool] @ w
    post_gap = post[:, treated] - post[:, pool] @ w
    pre_rmspe = math.sqrt(float(np.mean(pre_gap**2)))
    post_rmspe = math.sqrt(float(np.mean(post_gap**2)))
    ratio = post_rmspe / pre_rmspe if pre_rmspe > 0 else math.inf
    return pre_rmspe, post_rmspe, ratio, post_gap, w


def min_donors(alpha: float) -> int:
    """Donors needed so that `1 / (J + 1) <= alpha` (19 at 0.05)."""
    return math.ceil(1.0 / alpha) - 1


def estimate(
    frame: pd.DataFrame,
    unit: str,
    time: str,
    outcome: str,
    treated_unit: Any,
    first_treated: Any,
    alpha: float = 0.05,
) -> SynthResult:
    """Fits the synthetic control and its placebo distribution.

    Raises:
        ValueError: If the panel is unbalanced, a period range is empty, or
            fewer than `min_donors(alpha)` donors exist.
    """
    wide = frame.pivot(index=time, columns=unit, values=outcome).sort_index()
    if wide.isna().any().any():
        raise ValueError("the panel must be balanced with no missing outcomes")
    if treated_unit not in wide.columns:
        raise ValueError("the treated unit is not in the panel")
    is_pre = wide.index < first_treated
    if is_pre.sum() < 2 or (~is_pre).sum() < 1:
        raise ValueError("need two pre-periods and one post-period")
    units = list(wide.columns)
    donors = [u for u in units if u != treated_unit]
    if len(donors) < min_donors(alpha):
        raise ValueError(
            f"{len(donors)} donors cannot reach p <= {alpha}: at least "
            f"{min_donors(alpha)} are needed"
        )
    values = wide.to_numpy(dtype=float)
    pre, post = values[is_pre], values[~is_pre]
    index = {u: i for i, u in enumerate(units)}
    pool = [index[u] for u in donors]
    pre_r, post_r, ratio, gap, w = ratio_for(
        pre, post, index[treated_unit], pool
    )
    placebos = {}
    for donor in donors:
        others = [index[u] for u in donors if u != donor]
        placebos[str(donor)] = ratio_for(pre, post, index[donor], others)[2]
    exceed = sum(1 for r in placebos.values() if r >= ratio)
    p_value = (1 + exceed) / (len(donors) + 1)
    return SynthResult(
        effect=float(gap.mean()),
        gaps=tuple(float(g) for g in gap),
        weights={
            str(d): float(x) for d, x in zip(donors, w, strict=True) if x > 1e-6
        },
        pre_rmspe=pre_r,
        post_rmspe=post_r,
        ratio=ratio,
        p_value=p_value,
        donors=len(donors),
        min_p=1.0 / (len(donors) + 1),
        placebo_ratios=placebos,
        alpha=alpha,
    )
