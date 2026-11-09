"""Back-door adjustment by augmented inverse-probability weighting.

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

import numpy
import pandas
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
MIN_SCORED_ARM_ROWS = 2


@dataclasses.dataclass(frozen=True, slots=True)
class Design:
    """A frozen map from covariate columns to a numeric matrix.

    Numeric non-boolean columns pass through; other columns (including
    boolean ones) become indicator columns for
    the levels seen when the design was built (an unseen level is all zero).

    Attributes:
        numeric: Names of the columns that pass through unchanged.
        levels: For each non-numeric column, its name and the levels seen
            when the design was built, in sorted order.
    """

    numeric: tuple[str, ...]
    levels: tuple[tuple[str, tuple[str, ...]], ...]

    @classmethod
    def build(
        cls, frame: pandas.DataFrame, columns: tuple[str, ...]
    ) -> "Design":
        """Reads the column kinds from `frame`.

        Args:
            frame: The rows to read the column kinds and levels from.
            columns: The covariate columns, in order.

        Returns:
            A design in which numeric non-boolean columns pass through and
            every other column is encoded by its levels.
        """
        numeric = []
        levels = []
        for name in columns:
            if pandas.api.types.is_numeric_dtype(frame[name]) and not (
                pandas.api.types.is_bool_dtype(frame[name])
            ):
                numeric.append(name)
            else:
                seen = sorted(frame[name].astype(str).unique())
                levels.append((name, tuple(seen)))
        return cls(tuple(numeric), tuple(levels))

    def matrix(self, frame: pandas.DataFrame) -> numpy.ndarray:
        """Encodes the rows of `frame` under this design.

        Args:
            frame: Rows holding the design's columns.

        Returns:
            A float matrix: the numeric columns, then one indicator column
            per level of each non-numeric column.

        Raises:
            ValueError: If a numeric covariate holds a missing or non-finite
                value. Missing values in other columns are encoded as the
                level "nan" instead.
        """
        parts = [frame[list(self.numeric)].to_numpy(dtype=float)]
        for name, seen in self.levels:
            column = frame[name].astype(str).to_numpy()
            parts.append(
                numpy.column_stack([(column == v).astype(float) for v in seen])
            )
        matrix = numpy.hstack(parts) if parts else numpy.empty((len(frame), 0))
        if not numpy.isfinite(matrix).all():
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

    def predict(self, frame: pandas.DataFrame) -> tuple[numpy.ndarray, ...]:
        """Predicts the nuisances for the rows of `frame`.

        Args:
            frame: Rows holding the covariate columns.

        Returns:
            The propensity `e` (clipped away from 0 and 1), the control
            outcome prediction `m0` and the treated outcome prediction `m1`,
            one entry per row.
        """
        matrix = self.scaler.transform(self.design.matrix(frame))
        e = self.propensity.predict_proba(matrix)[:, 1]
        return (
            numpy.clip(e, CLIP, 1 - CLIP),
            self.control.predict(matrix),
            self.treated.predict(matrix),
        )


def make_learners(learner: str, seed: int) -> tuple[Any, Any]:
    """Builds an unfitted propensity classifier and outcome regressor.

    Args:
        learner: `linear` (logistic and ridge regression) or `boosted`
            (gradient boosting).
        seed: Random seed of the boosted learners.

    Returns:
        The propensity classifier and the outcome regressor.

    Raises:
        ValueError: If `learner` is not a known learner.
    """
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


def binary(values: pandas.Series, name: str) -> numpy.ndarray:
    """Reads a binary column as a 0/1 float array.

    Args:
        values: The column.
        name: Column name used in the error message.

    Returns:
        The values as floats.

    Raises:
        ValueError: If a value is neither 0 nor 1.
    """
    array = values.to_numpy(dtype=float)
    if not numpy.isin(array, (0.0, 1.0)).all():
        raise ValueError(f"{name} must be binary (0 or 1)")
    return array


def fit_nuisances(
    frame: pandas.DataFrame,
    covariates: tuple[str, ...],
    treatment: str,
    outcome: str,
    seed: int = 0,
    learner: str = "linear",
) -> Nuisances:
    """Fits and freezes the nuisance models on the exploration rows.

    Args:
        frame: Exploration rows.
        covariates: Adjustment columns; with none, a constant column is used.
        treatment: Binary treatment column.
        outcome: Outcome column.
        seed: Random seed of the boosted learners.
        learner: `linear` or `boosted`.

    Returns:
        The frozen models, their encoding and scaler, and a digest of their
        predictions on the fitting rows.

    Raises:
        ValueError: If the outcome has missing values or an arm has fewer
            than the minimum number of rows. A non-binary treatment, an
            unknown learner or missing covariates are also refused with a
            `ValueError`.
    """
    t = binary(frame[treatment], treatment)
    y = frame[outcome].to_numpy(dtype=float)
    if not numpy.isfinite(y).all():
        raise ValueError("the outcome must not hold missing values")
    if min(t.sum(), len(t) - t.sum()) < MIN_ARM:
        raise ValueError(f"each arm needs at least {MIN_ARM} rows")
    design = Design.build(frame, covariates)
    raw = design.matrix(frame)
    if raw.shape[1] == 0:
        raw = numpy.zeros((len(frame), 1))
    scaler = preprocessing.StandardScaler().fit(raw)
    matrix = scaler.transform(raw)
    classifier = make_learners(learner, seed)[0].fit(matrix, t)
    control = make_learners(learner, seed)[1].fit(matrix[t == 0], y[t == 0])
    treated = make_learners(learner, seed)[1].fit(matrix[t == 1], y[t == 1])
    e = classifier.predict_proba(matrix)[:, 1]
    probe = numpy.round(
        numpy.concatenate(
            [e, control.predict(matrix), treated.predict(matrix)]
        ),
        10,
    )
    digest = hashlib.sha256(probe.tobytes()).hexdigest()
    return Nuisances(
        design, scaler, classifier, control, treated, learner, digest
    )


def effective_sample_size(weights: numpy.ndarray) -> float:
    """Computes the Kish effective sample size `(sum w)^2 / sum w^2`.

    Args:
        weights: The weights.

    Returns:
        The squared sum of the weights over the sum of their squares, or 0.0
        when the sum of squares is not positive.
    """
    return sequential.effective_sample_size(numpy.asarray(weights, dtype=float))


def weighted_smd(
    matrix: numpy.ndarray, t: numpy.ndarray, e: numpy.ndarray
) -> numpy.ndarray:
    """Computes absolute standardized mean differences after IPW.

    Args:
        matrix: Covariate matrix, one column per covariate.
        t: Binary treatment, one entry per row.
        e: Propensity score, one entry per row.

    Returns:
        One value per column: the absolute difference of the weighted
        treated and control means (each weighted by the inverse probability
        of its arm) divided by the pooled unweighted standard deviation (0
        when that is zero).
    """
    w = numpy.where(t == 1, 1.0 / e, 1.0 / (1.0 - e))
    out = numpy.zeros(matrix.shape[1])
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        m1 = numpy.average(col[t == 1], weights=w[t == 1])
        m0 = numpy.average(col[t == 0], weights=w[t == 0])
        pooled = numpy.sqrt((col[t == 1].var() + col[t == 0].var()) / 2)
        out[j] = abs(m1 - m0) / pooled if pooled > 0 else 0.0
    return out


def overlap_report(
    e: numpy.ndarray,
    t: numpy.ndarray,
    matrix: numpy.ndarray | None = None,
    eps: float = OVERLAP_EPS,
) -> dict[str, Any]:
    """Reports propensity range, extreme share, Kish ESS and weighted balance.

    `passed` is the pre-registered overlap gate: at most `OVERLAP_SHARE` of
    the units may have a propensity outside `[eps, 1 - eps]`.

    Args:
        e: Propensity scores.
        t: Binary treatment.
        matrix: Covariate matrix for the weighted balance, or None.
        eps: Lower propensity bound of the gate (the upper is `1 - eps`).

    Returns:
        A mapping with the propensity range, the share outside the bounds,
        the effective sample sizes of the arms, the row count, `passed`, and
        the largest weighted standardized difference (`max_weighted_smd`) only
        when a matrix with at least one column is given.
    """
    outside = (e < eps) | (e > 1 - eps)
    w = numpy.where(t == 1, 1.0 / e, 1.0 / (1.0 - e))
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
class Estimate:
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
        """Builds the JSON-ready record.

        Returns:
            A mapping of every field.
        """
        return dataclasses.asdict(self)


def scores(
    estimand: str,
    t: numpy.ndarray,
    y: numpy.ndarray,
    e: numpy.ndarray,
    m0: numpy.ndarray,
    m1: numpy.ndarray,
) -> tuple[float, numpy.ndarray]:
    """Computes the point estimate and the per-row influence values.

    Args:
        estimand: `ATE` or `ATT`.
        t: Binary treatment.
        y: Outcome.
        e: Propensity scores.
        m0: Control outcome predictions.
        m1: Treated outcome predictions.

    Returns:
        The estimate and the per-row influence values (the centred scores
        for `ATE`).

    Raises:
        ValueError: If `estimand` is neither `ATE` nor `ATT`.
    """
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
    frame: pandas.DataFrame,
    treatment: str,
    outcome: str,
    estimand: str = "ATE",
    level: float = 0.95,
    trim: float = 0.0,
    min_share: float = 0.5,
) -> Estimate:
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

    Returns:
        The estimate with its Wald interval, the overlap report on the
        scored rows and the digest of the frozen models.

    Raises:
        ValueError: If trimming leaves less than `min_share` of the rows,
            an arm has fewer than two scored rows, or the outcome has missing
            values. A non-binary treatment or an unknown estimand is also
            refused with a `ValueError`.
    """
    t = binary(frame[treatment], treatment)
    y = frame[outcome].to_numpy(dtype=float)
    if not numpy.isfinite(y).all():
        raise ValueError("the outcome must not hold missing values")
    e, m0, m1 = nuisances.predict(frame)
    keep = (
        (e >= trim) & (e <= 1 - trim) if trim > 0 else numpy.ones(len(e), bool)
    )
    if keep.mean() < min_share:
        raise ValueError(
            f"trimming at {trim} keeps {keep.mean():.3f} of the rows, below "
            f"the declared minimum share {min_share}"
        )
    if min(t[keep].sum(), keep.sum() - t[keep].sum()) < MIN_SCORED_ARM_ROWS:
        raise ValueError("each arm needs at least two scored rows")
    t, y, e, m0, m1 = (a[keep] for a in (t, y, e, m0, m1))
    theta, influence = scores(estimand, t, y, e, m0, m1)
    n = int(keep.sum())
    se = float(numpy.sqrt(influence.var(ddof=1) / n))
    z = float(stats.norm.ppf(0.5 + level / 2))
    matrix = nuisances.scaler.transform(nuisances.design.matrix(frame))[keep]
    population = "the full population of the rows"
    if trim > 0:
        population = (
            f"the overlap population (propensity within [{trim}, {1 - trim}]);"
            " the estimand changed by trimming"
        )
    return Estimate(
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
class Difference:
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
        """Builds the JSON-ready record.

        Returns:
            A mapping of every field.
        """
        return dataclasses.asdict(self)


def difference_in_means(
    frame: pandas.DataFrame, treatment: str, outcome: str, level: float = 0.95
) -> Difference:
    """Computes the treated minus control mean with the Neyman variance.

    Args:
        frame: Rows with the treatment and outcome.
        treatment: Binary treatment column.
        outcome: Outcome column.
        level: Confidence level.

    Returns:
        The difference, its Neyman standard error and a Welch t interval.

    Raises:
        ValueError: If an arm has fewer than two rows. A non-binary
            treatment is also refused with a `ValueError`.
    """
    t = binary(frame[treatment], treatment)
    y = frame[outcome].to_numpy(dtype=float)
    y1, y0 = y[t == 1], y[t == 0]
    if min(len(y1), len(y0)) < MIN_SCORED_ARM_ROWS:
        raise ValueError("each arm needs at least two rows")
    v1, v0 = y1.var(ddof=1) / len(y1), y0.var(ddof=1) / len(y0)
    se = float(numpy.sqrt(v1 + v0))
    df = (v1 + v0) ** 2 / (v1**2 / (len(y1) - 1) + v0**2 / (len(y0) - 1))
    quantile = float(stats.t.ppf(0.5 + level / 2, df))
    diff = float(y1.mean() - y0.mean())
    return Difference(
        diff, se, diff - quantile * se, diff + quantile * se, level, len(y)
    )
