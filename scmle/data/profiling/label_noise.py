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
from typing import cast

import numpy
import pandas
from sklearn import ensemble
from sklearn import model_selection

from scmle.contracts import data

MIN_CLASSES = 2
MIN_FOLDS = 2
MIN_ROWS = 50
FOLDS = 5
ESTIMATOR_MAX_ITER = 50
ESTIMATOR_MAX_LEAF_NODES = 31
MAX_ROWS = 5000
NOTE_RATE = 0.05


@dataclasses.dataclass(frozen=True)
class LabelNoise:
    """Rows whose label disagrees with a confident out-of-fold prediction.

    Attributes:
        suspects: Positional row indices ordered by how confident the
            disagreement is, most confident first.
        suggested: Label the model would give each suspect.
        confidence: Probability of the suggested label for each suspect.
        rate: Share of rows flagged.
        class_thresholds: Mean own-label probabilities by class.
        folds: Number of out-of-fold partitions used.
    """

    suspects: tuple[int, ...]
    suggested: tuple[Any, ...]
    confidence: tuple[float, ...]
    rate: float
    class_thresholds: tuple[float, ...] = ()
    folds: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Builds the JSON-ready record.

        Returns:
            A dict with the suspects, suggested labels, confidences and the
            flagged rate. `class_thresholds` and `folds` are not included.
        """
        return {
            "suspects": self.suspects,
            "suggested": self.suggested,
            "confidence": self.confidence,
            "rate": self.rate,
        }

    def quality_evidence(
        self,
        input_rows: int,
        rows_checked: int,
        seed: int,
        sample_limit: int,
    ) -> dict[str, object]:
        """Returns reproducible evidence for this label-noise check.

        Args:
            input_rows: Rows available before the bounded sample was taken.
            rows_checked: Rows actually used by the check.
            seed: Seed controlling sampling, folds, and the estimator.
            sample_limit: Maximum rows permitted by the check.

        Returns:
            JSON-native quality evidence.
        """
        return {
            "observation": "label_noise",
            "status": "flagged" if self.rate >= NOTE_RATE else "clear",
            "method": "out_of_fold_confident_disagreement",
            "thresholds": {
                "minimum_rows": MIN_ROWS,
                "minimum_classes": 2,
                "maximum_folds": FOLDS,
                "minimum_finding_rate": NOTE_RATE,
                "class_own_label_probability": list(self.class_thresholds),
            },
            "sample_bounds": {
                "input_rows": input_rows,
                "rows_checked": rows_checked,
                "maximum_rows": sample_limit,
                "sampled": input_rows > rows_checked,
            },
            "seed": seed,
            "metrics": {
                "suspect_rows": len(self.suspects),
                "suspect_rate": self.rate,
                "classes": len(self.class_thresholds),
                "folds": self.folds,
            },
            "limitations": [
                "Suspects are candidates for label review, not proof of error.",
                (
                    "The classifier and confidence thresholds can miss "
                    "systematic or ambiguous label errors."
                ),
            ],
        }


def find_label_noise(
    features: pandas.DataFrame, labels: pandas.Series, seed: int = 0
) -> LabelNoise:
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
    values = numpy.asarray(labels)
    classes, encoded = numpy.unique(values, return_inverse=True)
    if len(values) < MIN_ROWS or len(classes) < MIN_CLASSES:
        raise ValueError(
            f"label-noise detection needs {MIN_ROWS}+ rows and two classes"
        )
    counts = list(numpy.bincount(numpy.asarray(encoded).ravel()))
    folds = min(FOLDS, int(min(counts)))
    if folds < MIN_FOLDS:
        raise ValueError("every class needs at least two rows")
    probabilities = cast(
        "numpy.ndarray",
        model_selection.cross_val_predict(
            ensemble.HistGradientBoostingClassifier(
                max_iter=ESTIMATOR_MAX_ITER,
                max_leaf_nodes=ESTIMATOR_MAX_LEAF_NODES,
                random_state=seed,
            ),
            features,
            encoded,
            cv=model_selection.StratifiedKFold(
                folds, shuffle=True, random_state=seed
            ),
            method="predict_proba",
        ),
    )
    own = probabilities[numpy.arange(len(encoded)), encoded]
    thresholds = numpy.array(
        [
            probabilities[encoded == index, index].mean()
            for index in range(len(classes))
        ]
    )
    best = probabilities.argmax(axis=1)
    flagged = (own < thresholds[encoded]) & (best != encoded)
    flagged &= (
        probabilities[numpy.arange(len(encoded)), best] >= thresholds[best]
    )
    order = numpy.flatnonzero(flagged)
    order = order[numpy.argsort(-probabilities[order, best[order]])]
    return LabelNoise(
        suspects=tuple(int(index) for index in order),
        suggested=tuple(classes[best[order]].tolist()),
        confidence=tuple(float(probabilities[i, best[i]]) for i in order),
        rate=float(flagged.mean()),
        class_thresholds=tuple(float(value) for value in thresholds),
        folds=folds,
    )


def assess_label_noise(
    features: pandas.DataFrame, labels: pandas.Series, seed: int = 0
) -> data.QualityObservation:
    """Runs bounded label-noise detection and retains its evidence.

    Args:
        features: Numeric feature frame.
        labels: Class labels aligned with the features.
        seed: Seed for bounded sampling, folds, and the model.

    Returns:
        A quality observation even when the check cannot be evaluated.
    """
    input_rows = len(features)
    if input_rows > MAX_ROWS:
        rows = features.sample(MAX_ROWS, random_state=seed).index
        features, labels = features.loc[rows], labels.loc[rows]
    rows_checked = len(features)
    try:
        report = find_label_noise(features, labels, seed)
    except ValueError as error:
        return data.QualityObservation(
            note=None,
            evidence={
                "observation": "label_noise",
                "status": "not_evaluated",
                "method": "out_of_fold_confident_disagreement",
                "thresholds": {
                    "minimum_rows": MIN_ROWS,
                    "minimum_classes": 2,
                    "maximum_folds": FOLDS,
                    "minimum_finding_rate": NOTE_RATE,
                },
                "sample_bounds": {
                    "input_rows": input_rows,
                    "rows_checked": rows_checked,
                    "maximum_rows": MAX_ROWS,
                    "sampled": input_rows > rows_checked,
                },
                "seed": seed,
                "metrics": {"error": str(error)},
                "limitations": [
                    (
                        "The check was not evaluated because its data contract "
                        "was not met."
                    ),
                ],
            },
        )
    note = None
    if report.rate >= NOTE_RATE:
        note = (
            f"label noise: {report.rate:.0%} of {rows_checked} checked rows "
            "have a label that a model trained without them contradicts "
            "confidently; review "
            "before trusting scores"
        )
    return data.QualityObservation(
        note=note,
        evidence=report.quality_evidence(
            input_rows, rows_checked, seed, MAX_ROWS
        ),
    )
