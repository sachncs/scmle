"""Label noise: two detectors that must agree, and a noise-rate interval.

Detector 1 is confident learning (Northcutt, Jiang and Chuang 2021, JAIR 70):
out-of-fold class probabilities, a per-class threshold t_j equal to the mean
probability of class j over rows labelled j, and the off-diagonal cells of
the confident joint. A row is an issue when the class it is confidently
counted into differs from its given label. Definitions were checked against
cleanlab's documentation (`filter_by="confident_learning"`), not against the
paper text.

Detector 2 is an in-sample area under the margin in the spirit of Pleiss et
al. (2020, NeurIPS): the margin of the given label's score over the best
other score, averaged over the boosting stages of a small model fitted on all
rows; a negative area flags the row. This is NOT the published procedure: the
original averages over neural-network epochs and reads its cutoff from
"threshold samples" (99th percentile). That cutoff was tried on boosting and
flagged most clean rows because the trees memorise the threshold class, so it
was dropped. The zero cutoff and the small model are design choices whose
operating characteristics rest on the certificate alone.

A row is a suspect only if both detectors flag it. The noise rate has two
forms: with an audited random sample, a prediction-powered interval
(Angelopoulos et al. 2023, Science 382; the estimator "mean of predictions
over unlabeled data minus the rectifier" was verified against secondary
descriptions, the CLT interval is the standard one and is checked by a
coverage simulation); without an audit, a detector-only flag rate with a
model-dependence caveat and tier `unverified`.
"""

from collections.abc import Sequence
import dataclasses
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import ensemble
from sklearn import model_selection

from scmle.core import randomness
from scmle.data.detectors import certificates

MIN_ROWS = 50
FOLDS = 5
MAX_ITER = 50
MAX_LEAF_NODES = 31
MAX_ROWS = 5000
MARGIN_LEAVES = 4
SAMPLING_CONFIDENCE = 0.95


@dataclasses.dataclass(frozen=True)
class NoiseRate:
    """Estimate of the label-noise rate.

    Attributes:
        estimate: Point estimate.
        interval: (low, high) at `confidence`.
        method: `prediction_powered` or `detector_only`.
        confidence: Level of the interval.
        tier: `unverified` unless an audit supplied the truth.
        caveat: What the number does not mean.
        audited: Size of the audited sample, 0 if none.
    """

    estimate: float
    interval: tuple[float, float]
    method: str
    confidence: float
    tier: str
    caveat: str
    audited: int


@dataclasses.dataclass(frozen=True)
class LabelReport:
    """Result of the two-detector label-noise check.

    Attributes:
        confident_learning: Row positions flagged by confident learning.
        margin: Row positions flagged by the margin detector.
        suspects: Row positions flagged by both, ascending.
        agreement: Jaccard overlap of the two flag sets (1.0 if both empty).
        rows_checked: Rows used after the cap.
        input_rows: Rows given.
        sampling_bound: Hoeffding half-width of a rate estimated from the
            capped sample (0.0 when no sampling happened).
        rate: Noise-rate estimate for the suspects.
        thresholds: Confident-learning class thresholds.
    """

    confident_learning: tuple[int, ...]
    margin: tuple[int, ...]
    suspects: tuple[int, ...]
    agreement: float
    rows_checked: int
    input_rows: int
    sampling_bound: float
    rate: NoiseRate
    thresholds: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)


def encode(labels: Sequence[Any] | pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Class values and integer codes of a label vector."""
    classes, codes = np.unique(np.asarray(labels), return_inverse=True)
    return classes, np.asarray(codes).ravel()


def model(
    seed: int, leaves: int = MAX_LEAF_NODES
) -> ensemble.HistGradientBoostingClassifier:
    """The boosting model with `leaves` leaves per tree."""
    return ensemble.HistGradientBoostingClassifier(
        max_iter=MAX_ITER,
        max_leaf_nodes=leaves,
        early_stopping=False,
        random_state=seed,
    )


def out_of_fold_probabilities(
    features: pd.DataFrame, codes: np.ndarray, seed: int
) -> np.ndarray:
    """Class probabilities from models that never saw the row.

    Raises:
        ValueError: With too few rows, one class, or a class of one row.
    """
    counts = np.bincount(codes)
    if len(codes) < MIN_ROWS or len(counts) < 2:
        raise ValueError(
            f"label-noise detection needs {MIN_ROWS}+ rows and two classes"
        )
    folds = min(FOLDS, int(np.min(counts)))
    if folds < 2:
        raise ValueError("every class needs at least two rows")
    return model_selection.cross_val_predict(
        model(seed),
        features,
        codes,
        cv=model_selection.StratifiedKFold(
            folds,
            shuffle=True,
            random_state=randomness.derive_int_seed(seed, "labels.folds", 0),
        ),
        method="predict_proba",
    )


def class_thresholds(
    probabilities: np.ndarray, codes: np.ndarray
) -> np.ndarray:
    """t_j: mean probability of class j over rows labelled j."""
    return np.array(
        [
            probabilities[codes == j, j].mean()
            for j in range(probabilities.shape[1])
        ]
    )


def confident_assignment(
    probabilities: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    """Class each row is confidently counted into, or -1.

    Among classes whose probability reaches their threshold, the one with the
    largest probability.
    """
    eligible = probabilities >= thresholds
    masked = np.where(eligible, probabilities, -np.inf)
    best = masked.argmax(axis=1)
    return np.where(eligible.any(axis=1), best, -1)


def confident_learning_flags(
    probabilities: np.ndarray, codes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rows in the off-diagonal cells of the confident joint.

    Args:
        probabilities: Out-of-fold probabilities, shape (n, K).
        codes: Given label codes.

    Returns:
        A boolean flag per row and the class thresholds.
    """
    thresholds = class_thresholds(probabilities, codes)
    assigned = confident_assignment(probabilities, thresholds)
    return (assigned >= 0) & (assigned != codes), thresholds


def mean_margins(
    features: pd.DataFrame, codes: np.ndarray, seed: int
) -> np.ndarray:
    """Area under the margin over boosting stages, for the training rows."""
    fitted = model(seed, MARGIN_LEAVES).fit(features, codes)
    total = np.zeros(len(codes))
    stages = 0
    rows = np.arange(len(codes))
    for scores in fitted.staged_decision_function(features):
        scores = np.asarray(scores)
        if scores.ndim == 1:
            scores = np.column_stack([-scores / 2.0, scores / 2.0])
        own = scores[rows, codes]
        rest = scores.copy()
        rest[rows, codes] = -np.inf
        total += own - rest.max(axis=1)
        stages += 1
    return total / stages


def margin_flags(
    features: pd.DataFrame, codes: np.ndarray, seed: int
) -> np.ndarray:
    """Rows whose given label loses on average over the boosting stages.

    The model is fitted on all rows with their given labels and kept small
    (`MARGIN_LEAVES`) so that it cannot memorise wrong labels; a row is
    flagged when its area under the margin is negative, that is, the model
    scores some other class above the given one on average.
    """
    area = mean_margins(features, codes, seed)
    return area < 0.0


def ppi_mean(
    predicted_unlabeled: np.ndarray,
    predicted_labeled: np.ndarray,
    truth_labeled: np.ndarray,
    confidence: float = SAMPLING_CONFIDENCE,
) -> tuple[float, tuple[float, float]]:
    """Prediction-powered confidence interval for a mean.

    The estimate is the mean of the predictions over the unlabeled rows
    minus the rectifier, the mean of (prediction - truth) over the labeled
    rows. Its variance is estimated as s_f^2 / N + s_rect^2 / n and the
    interval is normal, clipped to [0, 1] by the caller's use.

    Args:
        predicted_unlabeled: Predictions on N rows without a truth value.
        predicted_labeled: Predictions on the n audited rows.
        truth_labeled: Audited truth on those rows.
        confidence: Interval level in (0, 1).

    Returns:
        (estimate, (low, high)).

    Raises:
        ValueError: With fewer than 2 audited rows or no unlabeled rows.
    """
    f_big = np.asarray(predicted_unlabeled, dtype=float)
    f_small = np.asarray(predicted_labeled, dtype=float)
    y_small = np.asarray(truth_labeled, dtype=float)
    n, big = len(f_small), len(f_big)
    if n < 2 or big < 2 or len(y_small) != n:
        raise ValueError("need 2+ audited and 2+ unaudited rows")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be within (0, 1)")
    rectifier = f_small - y_small
    estimate = float(f_big.mean() - rectifier.mean())
    variance = f_big.var(ddof=1) / big + rectifier.var(ddof=1) / n
    half = float(stats.norm.ppf(0.5 + confidence / 2.0) * math.sqrt(variance))
    return estimate, (estimate - half, estimate + half)


def noise_rate(
    flags: np.ndarray,
    audit: pd.Series | None = None,
    confidence: float = SAMPLING_CONFIDENCE,
) -> NoiseRate:
    """Noise-rate interval from detector flags and an optional audit.

    Args:
        flags: Boolean suspect flag per row (positional).
        audit: Boolean Series whose index holds row positions that were
            audited at random and whose value is True when the label is
            wrong. The sample must be random; a convenience sample voids the
            interval.
        confidence: Interval level.

    Returns:
        With an audit of 2+ rows (and 2+ others), a prediction-powered
        interval, tier `attested`; otherwise the detector-only flag rate with
        a Wilson interval for the flag *rate* and tier `unverified`.
    """
    flags = np.asarray(flags, dtype=bool)
    n = len(flags)
    if audit is not None and 2 <= len(audit) <= n - 2:
        positions = np.asarray(audit.index, dtype=int)
        if len(np.unique(positions)) != len(positions):
            raise ValueError("audit positions must be unique")
        mask = np.zeros(n, dtype=bool)
        mask[positions] = True
        estimate, (low, high) = ppi_mean(
            flags[~mask].astype(float),
            flags[positions].astype(float),
            audit.to_numpy(dtype=float),
            confidence,
        )
        return NoiseRate(
            estimate=min(1.0, max(0.0, estimate)),
            interval=(max(0.0, low), min(1.0, high)),
            method="prediction_powered",
            confidence=confidence,
            tier="attested",
            caveat=(
                "valid only if the audited rows are a random sample and the "
                "audit is correct"
            ),
            audited=len(audit),
        )
    low, high = certificates.wilson_interval(int(flags.sum()), n, confidence)
    return NoiseRate(
        estimate=float(flags.mean()),
        interval=(low, high),
        method="detector_only",
        confidence=confidence,
        tier="unverified",
        caveat=(
            "share of rows flagged by the detectors, not the noise rate: "
            "it is biased by their recall and false-alarm rate and depends "
            "on the model"
        ),
        audited=0,
    )


def hoeffding_bound(
    rows: int, confidence: float = SAMPLING_CONFIDENCE
) -> float:
    """Half-width of a Hoeffding interval for a rate from `rows` draws."""
    return math.sqrt(math.log(2.0 / (1.0 - confidence)) / (2.0 * rows))


def detect(
    features: pd.DataFrame,
    labels: pd.Series,
    seed: int = 0,
    audit: pd.Series | None = None,
) -> LabelReport:
    """Runs both detectors and reports where they agree.

    Args:
        features: Numeric feature frame (exploration rows only).
        labels: Class labels aligned with `features`.
        seed: Run seed.
        audit: Optional audited sample, see `noise_rate`; positions refer to
            the rows of `features` as given.

    Returns:
        The report; positions refer to the capped sample, which keeps the
        audited rows. When rows are capped, `sampling_bound` records the
        Hoeffding half-width of a rate from the sample.

    Raises:
        ValueError: With too few rows or classes (see
            `out_of_fold_probabilities`).
    """
    input_rows = len(features)
    positions = np.arange(input_rows)
    if input_rows > MAX_ROWS:
        rng = randomness.component_rng(seed, "labels.cap", 0)
        audited = (
            np.asarray(audit.index, dtype=int) if audit is not None else None
        )
        others = (
            positions if audited is None else np.setdiff1d(positions, audited)
        )
        room = MAX_ROWS - (0 if audited is None else len(audited))
        take = rng.choice(others, max(room, 0), replace=False)
        positions = np.sort(
            take if audited is None else np.concatenate([take, audited])
        )
        features = features.iloc[positions]
        labels = labels.iloc[positions]
        if audit is not None:
            remap = {int(p): i for i, p in enumerate(positions)}
            audit = pd.Series(
                audit.to_numpy(),
                index=[remap[int(p)] for p in audit.index],
            )
    codes = encode(labels)[1]
    probabilities = out_of_fold_probabilities(features, codes, seed)
    cl_flags, thresholds = confident_learning_flags(probabilities, codes)
    aum_flags = margin_flags(features, codes, seed)
    both = cl_flags & aum_flags
    union = int((cl_flags | aum_flags).sum())
    agreement = 1.0 if union == 0 else float(both.sum()) / union
    capped = input_rows > len(features)
    return LabelReport(
        confident_learning=tuple(int(i) for i in np.flatnonzero(cl_flags)),
        margin=tuple(int(i) for i in np.flatnonzero(aum_flags)),
        suspects=tuple(int(i) for i in np.flatnonzero(both)),
        agreement=agreement,
        rows_checked=len(features),
        input_rows=input_rows,
        sampling_bound=hoeffding_bound(len(features)) if capped else 0.0,
        rate=noise_rate(both, audit),
        thresholds=tuple(float(t) for t in thresholds),
    )
