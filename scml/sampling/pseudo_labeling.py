"""Self-training on unlabeled rows, kept only when it measurably helps.

A model fitted on the labeled rows labels the unlabeled rows it is confident
about; those rows join the next fit. Self-training can reinforce a model's own
mistakes, so the technique is judged the way every other strategy is: inside
each cross-validation fold, the augmented fit is scored on held-out labeled
rows against the plain fit, and it is used only when it wins on average.
"""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
from typing import Any

import numpy as np
import pandas as pd
from sklearn import base

DEFAULT_CONFIDENCE = 0.9
MIN_PSEUDO_ROWS = 20


@dataclasses.dataclass(frozen=True)
class SelfTrainingOutcome:
    """What cross-validation said about self-training.

    Attributes:
        gain: Mean held-out score change from adding pseudo-labels, in the
            scorer's larger-is-better orientation.
        folds: Number of folds judged.
        pseudo_rows: Average pseudo-labeled rows added per fold.
        accepted: Whether self-training will be used.
    """

    gain: float
    folds: int
    pseudo_rows: float
    accepted: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)


def confident_rows(
    fitted: Any, unlabeled: pd.DataFrame, confidence: float
) -> tuple[pd.DataFrame, pd.Series]:
    """Unlabeled rows the fitted classifier labels with enough confidence.

    Args:
        fitted: Fitted classifier with `predict_proba` and `classes_`.
        unlabeled: Rows without a label.
        confidence: Smallest top-class probability that is trusted.

    Returns:
        The trusted rows and the labels the model gave them.
    """
    if unlabeled.empty:
        return unlabeled, pd.Series(dtype=object)
    probabilities = np.asarray(fitted.predict_proba(unlabeled))
    top = probabilities.argmax(axis=1)
    keep = probabilities.max(axis=1) >= confidence
    classes = np.asarray(fitted.classes_)
    return unlabeled[keep], pd.Series(
        classes[top[keep]], index=unlabeled.index[keep]
    )


def augmented_fit(
    plain: Any,
    features: pd.DataFrame,
    target: pd.Series,
    unlabeled: pd.DataFrame,
    confidence: float,
) -> tuple[Any, int]:
    """Labels confident rows with a fitted model, then refits on both.

    Args:
        plain: Classifier already fitted on `features` and `target`.
        features: Labeled training features.
        target: Their labels.
        unlabeled: Rows without a label.
        confidence: Smallest top-class probability that is trusted.

    Returns:
        A fresh fit on labeled plus pseudo-labeled rows and the number of
        pseudo-labeled rows (`plain` itself when too few qualified).
    """
    rows, labels = confident_rows(plain, unlabeled, confidence)
    if len(rows) < MIN_PSEUDO_ROWS:
        return plain, 0
    both = pd.concat([features, rows], ignore_index=True)
    labelled = pd.concat(
        [target.reset_index(drop=True), labels.reset_index(drop=True)],
        ignore_index=True,
    )
    return base.clone(plain).fit(both, labelled), len(rows)


def judge(
    template: Any,
    features: pd.DataFrame,
    target: pd.Series,
    folds: Sequence[tuple[Sequence[int], Sequence[int]]],
    unlabeled: pd.DataFrame,
    scorer: Callable[..., float],
    confidence: float = DEFAULT_CONFIDENCE,
    should_stop: Callable[[], bool] = lambda: False,
) -> SelfTrainingOutcome | None:
    """Compares plain and self-trained fits on held-out labeled rows.

    Returns:
        The outcome, or None when time ran out or no fold had enough
        confident rows to add.
    """
    gains: list[float] = []
    added: list[int] = []
    for fit_positions, evaluate_positions in folds:
        if should_stop():
            return None
        fit = np.asarray(fit_positions)
        held = np.asarray(evaluate_positions)
        plain = base.clone(template).fit(features.iloc[fit], target.iloc[fit])
        augmented, count = augmented_fit(
            plain,
            features.iloc[fit],
            target.iloc[fit],
            unlabeled,
            confidence,
        )
        if count == 0:
            continue
        gains.append(
            float(scorer(augmented, features.iloc[held], target.iloc[held]))
            - float(scorer(plain, features.iloc[held], target.iloc[held]))
        )
        added.append(count)
    if not gains:
        return None
    gain = float(np.mean(gains))
    return SelfTrainingOutcome(
        gain, len(gains), float(np.mean(added)), gain > 0
    )
