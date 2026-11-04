"""Back-door adjustment by augmented inverse-probability weighting (F10).

Nuisances (propensity `e(x)` and the two outcome regressions `m_t(x)`) are
fitted once on the exploration set `X` and frozen. The score on the
confirmation set `C` is then a mean of independent terms given the frozen
nuisances, so no cross-fitting is needed on `C` and the interval is valid for
`C` however flexible the learners are; a poor fit only widens it. The score is
the doubly robust (orthogonal) score of Chernozhukov et al., "Double/Debiased
Machine Learning for Treatment and Causal Parameters" (arXiv:1608.00060):

    psi_i = m1(x_i) - m0(x_i)
            + T_i (Y_i - m1(x_i)) / e(x_i)
            - (1 - T_i)(Y_i - m0(x_i)) / (1 - e(x_i)).

For the effect on the treated the score below was derived from the
efficient influence function by hand (not copied from a source) and is
checked against a planted effect in the certificate:

    theta_att = mean(T (Y - m0) - (1 - T) e/(1 - e) (Y - m0)) / p,
    phi_i     = (T_i (Y_i - m0 - theta_att)
                 - (1 - T_i) e/(1 - e) (Y_i - m0)) / p,   p = mean(T).

The standard error of either estimand is `sd(score) / sqrt(n)`. Validity needs
the declared identification assumptions (unconfoundedness given the
declared covariates, positivity, no interference); none of them is checked
by this module.
"""

import dataclasses
import hashlib
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import ensemble
from sklearn import linear_model
from sklearn import preprocessing

from scmle.evaluation import sequential

LEARNERS = ("linear", "boosted")
OVERLAP_EPS = 0.05
OVERLAP_SHARE = 0.05
MIN_ARM = 10
CLIP = 1e-6


@dataclasses.dataclass(frozen=True, slots=True)
class Design:
    """A frozen map from covariate columns to a numeric matrix.

    Numeric columns pass through; other columns become indicator columns for
    the levels seen when the design was built (an unseen level is all zero).
    """

    numeric: tuple[str, ...]
    levels: tuple[tuple[str, tuple[str, ...]], ...]

    @classmethod
    def build(cls, frame: pd.DataFrame, columns: tuple[str, ...]) -> "Design":
        """Reads the column kinds from `frame`."""
        numeric = []
        levels = []
        for name in columns:
            if pd.api.types.is_numeric_dtype(frame[name]) and not (
                pd.api.types.is_bool_dtype(frame[name])
            ):
                numeric.append(name)
            else:
                seen = sorted(frame[name].astype(str).unique())
                levels.append((name, tuple(seen)))
        return cls(tuple(numeric), tuple(levels))

    def matrix(self, frame: pd.DataFrame) -> np.ndarray:
        """The numeric matrix of `frame` under this design."""
        parts = [frame[list(self.numeric)].to_numpy(dtype=float)]
        for name, seen in self.levels:
            column = frame[name].astype(str).to_numpy()
            parts.append(
                np.column_stack([(column == v).astype(float) for v in seen])
            )
        matrix = np.hstack(parts) if parts else np.empty((len(frame), 0))
        if not np.isfinite(matrix).all():
            raise ValueError("covariates must not hold missing values")
        return matrix


@dataclasses.dataclass(frozen=True)
class Nuisances:
    """Frozen propensity and outcome models, fitted on `X` only.

    Attributes:
        design: Covariate encoding.
        scaler: Standardiser fitted on `X`.
        propensity: Model of `P(T = 1 | x)`.
        control: Model of `E[Y | T = 0, x]`.
        treated: Model of `E[Y | T = 1, x]`.
        learner: `linear` or `boosted`.
        digest: SHA-256 of the models' predictions on the fitting rows.
    """

    design: Design
    scaler: preprocessing.StandardScaler
    propensity: Any
    control: Any
    treated: Any
    learner: str
    digest: str

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, ...]:
        """Returns `(e, m0, m1)` for the rows of `frame`."""
        matrix = self.scaler.transform(self.design.matrix(frame))
        e = self.propensity.predict_proba(matrix)[:, 1]
        return (
            np.clip(e, CLIP, 1 - CLIP),
            self.control.predict(matrix),
            self.treated.predict(matrix),
        )


def make_learners(learner: str, seed: int) -> tuple[Any, Any]:
    """A propensity classifier and an outcome regressor."""
    if learner == "linear":
        return (
            linear_model.LogisticRegression(C=10.0, max_iter=2000),
            linear_model.Ridge(alpha=1.0),
        )
    if learner == "boosted":
        return (
            ensemble.GradientBoostingClassifier(
                n_estimators=100, max_depth=2, random_state=seed
            ),
            ensemble.GradientBoostingRegressor(
                n_estimators=100, max_depth=2, random_state=seed
            ),
        )
    raise ValueError(f"learner must be one of {LEARNERS}")


def binary(values: pd.Series, name: str) -> np.ndarray:
    """The 0/1 array of a two-valued column."""
    array = values.to_numpy(dtype=float)
    if not np.isin(array, (0.0, 1.0)).all():
        raise ValueError(f"{name} must be binary (0 or 1)")
    return array


def fit_nuisances(
    frame: pd.DataFrame,
    covariates: tuple[str, ...],
    treatment: str,
    outcome: str,
    seed: int = 0,
    learner: str = "linear",
) -> Nuisances:
    """Fits and freezes the nuisance models on the exploration rows."""
    t = binary(frame[treatment], treatment)
    y = frame[outcome].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("the outcome must not hold missing values")
    if min(t.sum(), len(t) - t.sum()) < MIN_ARM:
        raise ValueError(f"each arm needs at least {MIN_ARM} rows")
    design = Design.build(frame, covariates)
    raw = design.matrix(frame)
    if raw.shape[1] == 0:
        raw = np.zeros((len(frame), 1))
    scaler = preprocessing.StandardScaler().fit(raw)
    matrix = scaler.transform(raw)
    classifier = make_learners(learner, seed)[0].fit(matrix, t)
    control = make_learners(learner, seed)[1].fit(matrix[t == 0], y[t == 0])
    treated = make_learners(learner, seed)[1].fit(matrix[t == 1], y[t == 1])
    e = classifier.predict_proba(matrix)[:, 1]
    probe = np.round(
        np.concatenate([e, control.predict(matrix), treated.predict(matrix)]),
        10,
    )
    digest = hashlib.sha256(probe.tobytes()).hexdigest()
    return Nuisances(
        design, scaler, classifier, control, treated, learner, digest
    )


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size `(sum w)^2 / sum w^2`."""
    return sequential.effective_sample_size(np.asarray(weights, dtype=float))


def weighted_smd(
    matrix: np.ndarray, t: np.ndarray, e: np.ndarray
) -> np.ndarray:
    """Largest-first absolute standardized mean differences after IPW."""
    w = np.where(t == 1, 1.0 / e, 1.0 / (1.0 - e))
    out = np.zeros(matrix.shape[1])
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        m1 = np.average(col[t == 1], weights=w[t == 1])
        m0 = np.average(col[t == 0], weights=w[t == 0])
        pooled = np.sqrt((col[t == 1].var() + col[t == 0].var()) / 2)
        out[j] = abs(m1 - m0) / pooled if pooled > 0 else 0.0
    return out


def overlap_report(
    e: np.ndarray,
    t: np.ndarray,
    matrix: np.ndarray | None = None,
    eps: float = OVERLAP_EPS,
) -> dict[str, Any]:
    """Propensity range, extreme share, Kish ESS and weighted balance.

    `passed` is the pre-registered overlap gate: at most `OVERLAP_SHARE` of
    the units may have a propensity outside `[eps, 1 - eps]`.
    """
    outside = (e < eps) | (e > 1 - eps)
    w = np.where(t == 1, 1.0 / e, 1.0 / (1.0 - e))
    report: dict[str, Any] = {
        "min_propensity": float(e.min()),
        "max_propensity": float(e.max()),
        "eps": eps,
        "share_outside": float(outside.mean()),
        "gate_share": OVERLAP_SHARE,
        "ess_treated": effective_sample_size(w[t == 1]),
        "ess_control": effective_sample_size(w[t == 0]),
        "n": len(e),
        "passed": bool(outside.mean() <= OVERLAP_SHARE),
    }
    if matrix is not None and matrix.shape[1]:
        smd = weighted_smd(matrix, t, e)
        report["max_weighted_smd"] = float(smd.max())
    return report


@dataclasses.dataclass(frozen=True)
class AipwResult:
    """An AIPW estimate on the confirmation rows.

    Attributes:
        estimand: `ATE` or `ATT`.
        estimate: The point estimate.
        se: Standard error of the score mean.
        low: Lower interval limit.
        high: Upper interval limit.
        level: Confidence level.
        n: Rows scored (after trimming).
        n_input: Rows offered.
        overlap: The overlap report on the scored rows.
        population: Words naming the estimand actually estimated.
        nuisance_digest: Digest of the frozen models.
    """

    estimand: str
    estimate: float
    se: float
    low: float
    high: float
    level: float
    n: int
    n_input: int
    overlap: dict[str, Any]
    population: str
    nuisance_digest: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)


def scores(
    estimand: str,
    t: np.ndarray,
    y: np.ndarray,
    e: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
) -> tuple[float, np.ndarray]:
    """The point estimate and the per-row influence values."""
    if estimand == "ATE":
        psi = m1 - m0 + t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e)
        return float(psi.mean()), psi - psi.mean()
    if estimand == "ATT":
        p = t.mean()
        raw = t * (y - m0) - (1 - t) * e / (1 - e) * (y - m0)
        theta = float(raw.mean() / p)
        phi = (t * (y - m0 - theta) - (1 - t) * e / (1 - e) * (y - m0)) / p
        return theta, phi
    raise ValueError("estimand must be ATE or ATT")


def estimate(
    nuisances: Nuisances,
    frame: pd.DataFrame,
    treatment: str,
    outcome: str,
    estimand: str = "ATE",
    level: float = 0.95,
    trim: float = 0.0,
    min_share: float = 0.5,
) -> AipwResult:
    """Scores the frozen nuisances on `frame` (the confirmation rows).

    Args:
        nuisances: Models fitted on `X`.
        frame: Rows never used to fit them.
        treatment: Binary treatment column.
        outcome: Outcome column.
        estimand: `ATE` or `ATT`.
        level: Confidence level of the Wald interval.
        trim: Drop units whose propensity is outside `[trim, 1 - trim]`.
        min_share: Smallest share of rows that may remain after trimming.

    Raises:
        ValueError: If trimming leaves less than `min_share` of the rows.
    """
    t = binary(frame[treatment], treatment)
    y = frame[outcome].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("the outcome must not hold missing values")
    e, m0, m1 = nuisances.predict(frame)
    keep = (e >= trim) & (e <= 1 - trim) if trim > 0 else np.ones(len(e), bool)
    if keep.mean() < min_share:
        raise ValueError(
            f"trimming at {trim} keeps {keep.mean():.3f} of the rows, below "
            f"the declared minimum share {min_share}"
        )
    if min(t[keep].sum(), keep.sum() - t[keep].sum()) < 2:
        raise ValueError("each arm needs at least two scored rows")
    t, y, e, m0, m1 = (a[keep] for a in (t, y, e, m0, m1))
    theta, influence = scores(estimand, t, y, e, m0, m1)
    n = int(keep.sum())
    se = float(np.sqrt(influence.var(ddof=1) / n))
    z = float(stats.norm.ppf(0.5 + level / 2))
    matrix = nuisances.scaler.transform(nuisances.design.matrix(frame))[keep]
    population = "the full population of the rows"
    if trim > 0:
        population = (
            f"the overlap population (propensity within [{trim}, {1 - trim}]);"
            " the estimand changed by trimming"
        )
    return AipwResult(
        estimand=estimand,
        estimate=theta,
        se=se,
        low=theta - z * se,
        high=theta + z * se,
        level=level,
        n=n,
        n_input=len(frame),
        overlap=overlap_report(e, t, matrix),
        population=population,
        nuisance_digest=nuisances.digest,
    )


@dataclasses.dataclass(frozen=True)
class DiffResult:
    """A difference in means (randomized assignment).

    Attributes:
        estimate: Treated minus control mean.
        se: Neyman standard error.
        low: Lower limit (Welch degrees of freedom).
        high: Upper limit.
        level: Confidence level.
        n: Rows.
    """

    estimate: float
    se: float
    low: float
    high: float
    level: float
    n: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)


def difference_in_means(
    frame: pd.DataFrame, treatment: str, outcome: str, level: float = 0.95
) -> DiffResult:
    """Treated minus control mean with the Neyman variance estimate."""
    t = binary(frame[treatment], treatment)
    y = frame[outcome].to_numpy(dtype=float)
    y1, y0 = y[t == 1], y[t == 0]
    if min(len(y1), len(y0)) < 2:
        raise ValueError("each arm needs at least two rows")
    v1, v0 = y1.var(ddof=1) / len(y1), y0.var(ddof=1) / len(y0)
    se = float(np.sqrt(v1 + v0))
    df = (v1 + v0) ** 2 / (v1**2 / (len(y1) - 1) + v0**2 / (len(y0) - 1))
    quantile = float(stats.t.ppf(0.5 + level / 2, df))
    diff = float(y1.mean() - y0.mean())
    return DiffResult(
        diff, se, diff - quantile * se, diff + quantile * se, level, len(y)
    )
