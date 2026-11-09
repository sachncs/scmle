"""Scoring uplift models, and checking that the treatment looks randomized.

A treatment effect is never observed for one row (a row is treated or not),
so an uplift model is judged on groups: rows are ranked by predicted uplift
and the Qini curve accumulates how much better the treated did than the
controls, as the ranking is walked from the most to the least promising row.
A good model's curve rises above the straight line a random ranking gives.
"""

import dataclasses
from typing import Any

import numpy
import pandas
from sklearn import linear_model
from sklearn import metrics as sk_metrics
from sklearn import model_selection

import scmle.contracts.metrics
from scmle.evaluation import balance

TOP_SHARE = 0.3
PROPENSITY_AUC_LIMIT = 0.6
MIN_ARM_SHARE = 0.05
ASSUMPTION = (
    "Uplift estimates assume that, given the features, treatment was "
    "assigned independently of the outcome (no unmeasured confounding). A "
    "randomized experiment guarantees this; observational data do not."
)


def qini_curve(
    outcome: numpy.ndarray, treatment: numpy.ndarray, score: numpy.ndarray
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Computes the cumulative incremental outcome along the uplift ranking.

    Args:
        outcome: Observed outcome of each row.
        treatment: 1 where the row was treated, 0 for control.
        score: Predicted uplift of each row; higher ranks earlier.

    Returns:
        The number of rows walked (1 to n) and the incremental outcome
        there: cumulative treated outcomes minus cumulative control
        outcomes rescaled to the treated group's size (the control term is
        0 until a control row has been walked).
    """
    order = numpy.argsort(-numpy.asarray(score, dtype=float), kind="stable")
    y = numpy.asarray(outcome, dtype=float)[order]
    t = numpy.asarray(treatment).astype(int)[order]
    treated = numpy.cumsum(t)
    control = numpy.cumsum(1 - t)
    gain_treated = numpy.cumsum(y * t)
    gain_control = numpy.cumsum(y * (1 - t))
    scale = numpy.divide(
        treated, control, out=numpy.zeros(len(y)), where=control > 0
    )
    return numpy.arange(1, len(y) + 1), gain_treated - gain_control * scale


def qini_coefficient(
    outcome: numpy.ndarray, treatment: numpy.ndarray, score: numpy.ndarray
) -> float:
    """Computes the area between the Qini curve and the random-ranking line.

    Normalized by the number of rows, so it is in outcome units; larger is
    better and a random ranking scores about zero.

    Args:
        outcome: Observed outcome of each row.
        treatment: 1 where the row was treated, 0 for control.
        score: Predicted uplift of each row; higher ranks earlier.

    Returns:
        The coefficient, or NaN with fewer than two rows.
    """
    rows, curve = qini_curve(outcome, treatment, score)
    if len(rows) < 2:
        return float("nan")
    line = rows / rows[-1] * curve[-1]
    return float(numpy.trapezoid(curve - line, rows) / (rows[-1] ** 2))


def uplift_at(
    outcome: numpy.ndarray,
    treatment: numpy.ndarray,
    score: numpy.ndarray,
    share: float = TOP_SHARE,
) -> float:
    """Computes treated-minus-control outcome in the top `share` of the ranking.

    Args:
        outcome: Observed outcome of each row.
        treatment: 1 where the row was treated, 0 for control.
        score: Predicted uplift of each row; higher ranks earlier.
        share: Fraction of rows, by rank, that counts as the top.

    Returns:
        The mean outcome of the treated minus that of the controls among the
        top rows (at least one row); NaN if either arm is absent there.
    """
    count = max(int(len(score) * share), 1)
    top = numpy.argsort(-numpy.asarray(score, dtype=float), kind="stable")[
        :count
    ]
    y = numpy.asarray(outcome, dtype=float)[top]
    t = numpy.asarray(treatment).astype(int)[top]
    if t.sum() == 0 or (1 - t).sum() == 0:
        return float("nan")
    return float(y[t == 1].mean() - y[t == 0].mean())


def average_effect(outcome: numpy.ndarray, treatment: numpy.ndarray) -> float:
    """Computes the overall treated-minus-control difference in the outcome.

    Args:
        outcome: Observed outcome of each row.
        treatment: 1 where the row was treated, 0 for control.

    Returns:
        The mean outcome of the treated minus that of the controls; NaN
        if either arm is empty.
    """
    y = numpy.asarray(outcome, dtype=float)
    t = numpy.asarray(treatment).astype(int)
    if not (t == 1).any() or not (t == 0).any():
        return float("nan")
    return float(y[t == 1].mean() - y[t == 0].mean())


@dataclasses.dataclass(frozen=True)
class Propensity:
    """How well the features predict who was treated.

    Attributes:
        treated_share: Share of rows treated.
        auc: Cross-validated ROC-AUC of predicting treatment from features
            (0.5 in a randomized experiment); a description, not the rule.
        randomized: Whether the treatment looks randomized: the permutation
            balance test of `balance.balance_test` does not reject at its
            level (an exact test for randomized assignment). When it cannot
            run (no numeric covariate with non-zero spread) the AUC cutoff
            decides: randomized if `auc` is at most `PROPENSITY_AUC_LIMIT`.
            An arm with under `MIN_ARM_SHARE` of the rows is never
            randomized, and no numeric covariate at all counts as randomized.
        note: Plain-language reading of the result.
        balance_p_value: Permutation p-value of the balance test, or None
            when the test did not run.
    """

    treated_share: float
    auc: float
    randomized: bool
    note: str
    balance_p_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Returns the JSON-ready record.

        Returns:
            A mapping of the fields.
        """
        return dataclasses.asdict(self)


def check_propensity(
    features: pandas.DataFrame, treatment: numpy.ndarray, seed: int = 0
) -> Propensity:
    """Judges from the covariates whether treatment looks randomized.

    Args:
        features: Covariates of each row.
        treatment: 1 where the row was treated, 0 for control.
        seed: Seed of the cross-validation and the permutation test.

    Returns:
        The treated share, the cross-validated AUC, the verdict, a note and
        the permutation-test p-value when there is one. The verdict follows
        the permutation balance test when it is available, otherwise the
        AUC against `PROPENSITY_AUC_LIMIT`. An arm with under
        `MIN_ARM_SHARE` of the rows gives a NaN AUC and not randomized; no
        numeric covariates gives AUC 0.5 and randomized.
    """
    arm = numpy.asarray(treatment).astype(int)
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
        / numpy.where(numeric.std(0) > 0, numeric.std(0), 1),
        arm,
        cv=model_selection.StratifiedKFold(3, shuffle=True, random_state=seed),
        method="predict_proba",
    )[:, 1]
    auc = float(sk_metrics.roc_auc_score(arm, scores))
    found = balance.balance_test(features, arm, seed)
    p_value = found["p_value"]
    evidence = (
        "AUC rule" if p_value is None else f"permutation p = {p_value:.3g}"
    )
    randomized = (
        bool(found["balanced"])
        if found["balanced"] is not None
        else auc <= PROPENSITY_AUC_LIMIT
    )
    note = (
        "covariates are balanced across the arms (permutation test); "
        "consistent with randomization"
        if randomized
        else f"covariates differ across the arms ({evidence}, AUC "
        f"{auc:.2f}): assignment was not random, so the estimates may be "
        "confounded"
    )
    return Propensity(share, auc, randomized, note, p_value)


@dataclasses.dataclass(frozen=True, slots=True)
class UpliftMetric(scmle.contracts.metrics.Metric):
    """A metric of how well predicted uplift ranks real incremental response.

    Attributes:
        kind: `qini` or `uplift_at_30`.
    """

    kind: str = "qini"

    def accepts_weights(self) -> bool:
        """Returns whether row weights are accepted; these scores are not.

        Returns:
            Always False.
        """
        return False

    def scorer(self, positive: Any = None) -> Any:
        """Builds a scorer called as `(estimator, features, outcome)`.

        Args:
            positive: Unused; there is no positive class.

        Returns:
            A function returning the Qini coefficient (`qini`) or the uplift
            in the top `TOP_SHARE` of the ranking (otherwise) of the
            estimator's predictions. The treatment indicator is read from the
            estimator's `treatment_column`; sample weights are ignored.
        """
        kind = self.kind

        def score(
            estimator: Any,
            features: pandas.DataFrame,
            target: pandas.Series,
            sample_weight: Any = None,
        ) -> float:
            arm = features[estimator.treatment_column].to_numpy(dtype=int)
            predicted = numpy.asarray(estimator.predict(features), dtype=float)
            outcome = numpy.asarray(target, dtype=float)
            if kind == "qini":
                return qini_coefficient(outcome, arm, predicted)
            return uplift_at(outcome, arm, predicted)

        return score
