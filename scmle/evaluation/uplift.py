"""Scoring uplift models, and checking that the treatment looks randomized.

A treatment effect is never observed for one row (a row is treated or not),
so an uplift model is judged on groups: rows are ranked by predicted uplift
and the Qini curve accumulates how much better the treated did than the
controls, as the ranking is walked from the most to the least promising row.
A good model's curve rises above the straight line a random ranking gives.
"""

import dataclasses
from typing import Any

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics as sk_metrics
from sklearn import model_selection

from scmle.contracts import metrics as metric_contracts

TOP_SHARE = 0.3
PROPENSITY_AUC_LIMIT = 0.6
MIN_ARM_SHARE = 0.05
ASSUMPTION = (
    "Uplift estimates assume that, given the features, treatment was "
    "assigned independently of the outcome (no unmeasured confounding). A "
    "randomized experiment guarantees this; observational data do not."
)


def qini_curve(
    outcome: np.ndarray, treatment: np.ndarray, score: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative incremental outcome along the uplift ranking.

    Returns:
        The number of rows walked and the incremental outcome there:
        treated outcomes minus control outcomes rescaled to the treated
        group's size.
    """
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    y = np.asarray(outcome, dtype=float)[order]
    t = np.asarray(treatment).astype(int)[order]
    treated = np.cumsum(t)
    control = np.cumsum(1 - t)
    gain_treated = np.cumsum(y * t)
    gain_control = np.cumsum(y * (1 - t))
    scale = np.divide(treated, control, out=np.zeros(len(y)), where=control > 0)
    return np.arange(1, len(y) + 1), gain_treated - gain_control * scale


def qini_coefficient(
    outcome: np.ndarray, treatment: np.ndarray, score: np.ndarray
) -> float:
    """Area between the model's Qini curve and the random-ranking line.

    Normalized by the number of rows, so it is in outcome units; larger is
    better and a random ranking scores about zero.
    """
    rows, curve = qini_curve(outcome, treatment, score)
    if len(rows) < 2:
        return float("nan")
    line = rows / rows[-1] * curve[-1]
    return float(np.trapezoid(curve - line, rows) / (rows[-1] ** 2))


def uplift_at(
    outcome: np.ndarray,
    treatment: np.ndarray,
    score: np.ndarray,
    share: float = TOP_SHARE,
) -> float:
    """Treated-minus-control outcome rate in the top `share` of the ranking."""
    count = max(int(len(score) * share), 1)
    top = np.argsort(-np.asarray(score, dtype=float), kind="stable")[:count]
    y = np.asarray(outcome, dtype=float)[top]
    t = np.asarray(treatment).astype(int)[top]
    if t.sum() == 0 or (1 - t).sum() == 0:
        return float("nan")
    return float(y[t == 1].mean() - y[t == 0].mean())


def average_effect(outcome: np.ndarray, treatment: np.ndarray) -> float:
    """Overall treated-minus-control difference in the outcome."""
    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treatment).astype(int)
    return float(y[t == 1].mean() - y[t == 0].mean())


@dataclasses.dataclass(frozen=True)
class Propensity:
    """How well the features predict who was treated.

    Attributes:
        treated_share: Share of rows treated.
        auc: Cross-validated ROC-AUC of predicting treatment from features
            (0.5 in a randomized experiment).
        randomized: Whether the treatment looks randomized.
        note: What to make of it.
    """

    treated_share: float
    auc: float
    randomized: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)


def check_propensity(
    features: pd.DataFrame, treatment: np.ndarray, seed: int = 0
) -> Propensity:
    """Judges from the covariates whether treatment looks randomized."""
    arm = np.asarray(treatment).astype(int)
    share = float(arm.mean())
    if share < MIN_ARM_SHARE or share > 1 - MIN_ARM_SHARE:
        return Propensity(
            share,
            float("nan"),
            False,
            f"only {min(share, 1 - share):.1%} of rows are in one arm",
        )
    numeric = features.select_dtypes("number").fillna(0.0).to_numpy(dtype=float)
    if numeric.shape[1] == 0:
        return Propensity(share, 0.5, True, "no numeric covariates to check")
    scores = model_selection.cross_val_predict(
        linear_model.LogisticRegression(max_iter=500),
        (numeric - numeric.mean(0))
        / np.where(numeric.std(0) > 0, numeric.std(0), 1),
        arm,
        cv=model_selection.StratifiedKFold(3, shuffle=True, random_state=seed),
        method="predict_proba",
    )[:, 1]
    auc = float(sk_metrics.roc_auc_score(arm, scores))
    randomized = auc <= PROPENSITY_AUC_LIMIT
    note = (
        "covariates do not predict treatment; consistent with randomization"
        if randomized
        else f"covariates predict treatment (AUC {auc:.2f}): assignment was "
        "not random, so the estimates may be confounded"
    )
    return Propensity(share, auc, randomized, note)


@dataclasses.dataclass(frozen=True, slots=True)
class UpliftMetric(metric_contracts.Metric):
    """A metric of how well predicted uplift ranks real incremental response.

    Attributes:
        kind: `qini` or `uplift_at_30`.
    """

    kind: str = "qini"

    def accepts_weights(self) -> bool:
        """These scores are unweighted."""
        return False

    def scorer(self, positive: Any = None) -> Any:
        """Builds a scorer `(estimator, features, outcome)`."""
        kind = self.kind

        def score(
            estimator: Any,
            features: pd.DataFrame,
            target: pd.Series,
            sample_weight: Any = None,
        ) -> float:
            arm = features[estimator.treatment_column].to_numpy(dtype=int)
            predicted = np.asarray(estimator.predict(features), dtype=float)
            outcome = np.asarray(target, dtype=float)
            if kind == "qini":
                return qini_coefficient(outcome, arm, predicted)
            return uplift_at(outcome, arm, predicted)

        return score
