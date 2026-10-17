"""Which labelled rows look mislabelled?

Confident learning in its simplest form: every row is scored by a model that
never saw it (out-of-fold). A row whose given label receives a low
probability while another label receives a high one is a suspect. The
per-class threshold is the mean probability the model gives that class to
rows carrying the label, so a rare class is not flagged just for being rare.
Suspects are candidates for review, not proof of error.
"""

import dataclasses
from typing import Any

import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import model_selection

MIN_ROWS = 50
FOLDS = 5


@dataclasses.dataclass(frozen=True)
class LabelNoiseReport:
    """Rows whose label disagrees with a confident out-of-fold prediction.

    Attributes:
        suspects: Positional row indices ordered by how confident the
            disagreement is, most confident first.
        suggested: Label the model would give each suspect.
        confidence: Probability of the suggested label for each suspect.
        rate: Share of rows flagged.
    """

    suspects: tuple[int, ...]
    suggested: tuple[Any, ...]
    confidence: tuple[float, ...]
    rate: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)


def find_label_noise(
    features: pd.DataFrame, labels: pd.Series, seed: int = 0
) -> LabelNoiseReport:
    """Flags rows whose label a model that never saw them contradicts.

    Args:
        features: Numeric feature frame.
        labels: Class labels aligned with the features.
        seed: Seed for folds and the model.

    Returns:
        The suspects, most confident first.

    Raises:
        ValueError: With fewer than `MIN_ROWS` rows or a single class.
    """
    values = np.asarray(labels)
    classes, encoded = np.unique(values, return_inverse=True)
    if len(values) < MIN_ROWS or len(classes) < 2:
        raise ValueError(
            f"label-noise detection needs {MIN_ROWS}+ rows and two classes"
        )
    folds = min(FOLDS, int(np.bincount(encoded).min()))
    if folds < 2:
        raise ValueError("every class needs at least two rows")
    probabilities = model_selection.cross_val_predict(
        ensemble.HistGradientBoostingClassifier(random_state=seed),
        features,
        encoded,
        cv=model_selection.StratifiedKFold(
            folds, shuffle=True, random_state=seed
        ),
        method="predict_proba",
    )
    own = probabilities[np.arange(len(encoded)), encoded]
    thresholds = np.array(
        [
            probabilities[encoded == index, index].mean()
            for index in range(len(classes))
        ]
    )
    best = probabilities.argmax(axis=1)
    flagged = (own < thresholds[encoded]) & (best != encoded)
    flagged &= probabilities[np.arange(len(encoded)), best] >= thresholds[best]
    order = np.flatnonzero(flagged)
    order = order[np.argsort(-probabilities[order, best[order]])]
    return LabelNoiseReport(
        suspects=tuple(int(index) for index in order),
        suggested=tuple(classes[best[order]].tolist()),
        confidence=tuple(float(probabilities[i, best[i]]) for i in order),
        rate=float(flagged.mean()),
    )
