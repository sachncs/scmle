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
import threadpoolctl

from scmle.contracts import data
from scmle.contracts import errors

MIN_CLASSES = 2
MIN_FOLDS = 2
MIN_ROWS = 50
FOLDS = 5
ESTIMATOR_MAX_ITER = 50
ESTIMATOR_MAX_LEAF_NODES = 31
MAX_ROWS = 5000
NOTE_RATE = 0.05
BINARY_CLASSES = 2
# Cost model of `contradict` (seconds, one thread):
#   seconds = BASE + CLASS x C' + CELL x rows x columns x C',
# C' the classes above two, else 1: the boosted model grows one tree per
# class above two in every round, so the cell term grows with C' too.
# BASE and CLASS were fitted 2026-11-17 on narrow evidence tables, where
# the fixed cost dominates. CELL was fitted on narrow tables too (1.5e-7)
# and so priced wide ones at a twentieth of their cost (gina: 8.8 s on
# 2,522 x 970 cells, predicted 1.1 s). Remeasured 2026-11-19
# (OMP_NUM_THREADS=1, CPU seconds, at most 4,000 rows) the seconds above
# the fixed cost were 1.3e-6 to 3.14e-6 per cell and class way on wilt,
# qsar-biodeg, APSFailure, Bioresponse, christine, mfeat-factors and
# semeion (3.2e-6 by least squares on the profiler's runs, which bound at
# 3.6e-6: `scmle.pipeline.profiler.NOISE_CELL`), so CELL is 3.6e-6. The
# largest measured to predicted ratio is then 1.06 (the 1,600-row,
# 30-column binary table, 0.36 s against 0.34 s), so MARGIN = 1.25 keeps
# the prediction an upper bound on every measured table. A larger thread
# cap only finishes sooner.
BASE_SECONDS = 0.05
CLASS_SECONDS = 0.12
CELL_SECONDS = 3.6e-6
MARGIN = 1.25
# Share of the time budget the check may take. The costliest table of the
# 2026-11-17 fit (1132 rows, 111 columns, 8 classes: 1.16 s) is predicted
# at 5.8 s, past the 3 s this share gives a 60 s budget, so B = 60 checks
# 434 of its rows; a shorter budget checks fewer, and a wide table below
# `MIN_ROWS` affordable rows is skipped on record.
SHARE = 0.05


@dataclasses.dataclass(frozen=True)
class Noise:
    """Rows whose label disagrees with a confident out-of-fold prediction.

    Attributes:
        suspects: Positional row indices ordered by how confident the
            disagreement is, most confident first.
        suggested: Label the model would give each suspect.
        confidence: Probability of the suggested label for each suspect.
        rate: Share of rows flagged.
        cutoffs: Mean own-label probabilities by class.
        folds: Number of out-of-fold partitions used.
        rows: Rows checked (fewer than given when an allowance subsampled).
    """

    suspects: tuple[int, ...]
    suggested: tuple[Any, ...]
    confidence: tuple[float, ...]
    rate: float
    cutoffs: tuple[float, ...] = ()
    folds: int = 0
    rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Builds the JSON-ready record.

        Returns:
            A dict with the suspects, suggested labels, confidences and the
            flagged rate. `cutoffs` and `folds` are not included.
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
                "class_own_label_probability": list(self.cutoffs),
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
                "classes": len(self.cutoffs),
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


def ways(classes: int) -> int:
    """Counts the one-vs-rest problems the cost model charges for.

    Args:
        classes: Classes of the target.

    Returns:
        `classes` above two, else 1.
    """
    return classes if classes > BINARY_CLASSES else 1


def cost(rows: int, columns: int, classes: int) -> float:
    """Predicts an upper bound on the seconds of `contradict`.

    Args:
        rows: Rows checked.
        columns: Feature columns.
        classes: Classes of the target.

    Returns:
        `MARGIN x (BASE + CLASS x C' + CELL x rows x columns x C')`.
    """
    return MARGIN * (
        BASE_SECONDS
        + CLASS_SECONDS * ways(classes)
        + CELL_SECONDS * rows * columns * ways(classes)
    )


def affordable(columns: int, classes: int, allowance: float) -> int:
    """Finds the most rows whose predicted `cost` fits an allowance.

    Args:
        columns: Feature columns.
        classes: Classes of the target.
        allowance: Seconds the check may take.

    Returns:
        The rows, 0 when not even the fixed cost fits.
    """
    spare = allowance / MARGIN - BASE_SECONDS - CLASS_SECONDS * ways(classes)
    if spare <= 0:
        return 0
    # The relative nudge keeps an allowance computed by `cost` for some rows
    # from losing one row to rounding.
    per = CELL_SECONDS * max(columns, 1) * ways(classes)
    return int(spare / per * (1 + 1e-9))


def contradict(
    features: pandas.DataFrame,
    labels: pandas.Series,
    seed: int = 0,
) -> Noise:
    """Flags rows whose label a model that never saw them contradicts.

    Fitting the time budget is `inspect`'s job: it samples the rows down to
    what its allowance affords, or skips the check on record.

    Args:
        features: Numeric feature frame.
        labels: Class labels aligned with the features.
        seed: Seed for folds and the model.

    Returns:
        The suspects, most confident first, as positions in `features`.

    Raises:
        InputError: With fewer than `MIN_ROWS` rows or a single class.
    """
    values = numpy.asarray(labels)
    classes, encoded = numpy.unique(values, return_inverse=True)
    if len(values) < MIN_ROWS or len(classes) < MIN_CLASSES:
        raise errors.InputError(
            f"label-noise detection needs {MIN_ROWS}+ rows and two classes"
        )
    counts = list(numpy.bincount(numpy.asarray(encoded).ravel()))
    folds = min(FOLDS, int(min(counts)))
    if folds < MIN_FOLDS:
        raise errors.InputError("every class needs at least two rows")
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
    return Noise(
        suspects=tuple(int(index) for index in order),
        suggested=tuple(classes[best[order]].tolist()),
        confidence=tuple(float(probabilities[i, best[i]]) for i in order),
        rate=float(flagged.mean()),
        cutoffs=tuple(float(value) for value in thresholds),
        folds=folds,
        rows=len(encoded),
    )


def unevaluated(
    note: str | None,
    status: str,
    reason: str,
    why: str,
    seed: int,
    rows: tuple[int, int, int],
) -> data.Observation:
    """Builds the observation of a check that did not run.

    Args:
        note: The finding text, or None.
        status: `not_evaluated` or `skipped`.
        reason: The error or skip reason.
        why: The limitation sentence.
        seed: Seed of the check.
        rows: Rows given, rows used and the most rows the check could use.

    Returns:
        The observation, with JSON-native quality evidence.
    """
    given, checked, limit = rows
    key = "reason" if status == "skipped" else "error"
    return data.Observation(
        note=note,
        evidence={
            "observation": "label_noise",
            "status": status,
            "method": "out_of_fold_confident_disagreement",
            "thresholds": {
                "minimum_rows": MIN_ROWS,
                "minimum_classes": 2,
                "maximum_folds": FOLDS,
                "minimum_finding_rate": NOTE_RATE,
            },
            "sample_bounds": {
                "input_rows": given,
                "rows_checked": checked,
                "maximum_rows": limit,
                "sampled": given > checked,
            },
            "seed": seed,
            "metrics": {key: reason},
            "limitations": [why],
        },
    )


def inspect(
    features: pandas.DataFrame,
    labels: pandas.Series,
    seed: int = 0,
    allowance: float | None = None,
    threads: int | None = None,
) -> data.Observation:
    """Runs bounded label-noise detection and retains its evidence.

    Args:
        features: Numeric feature frame.
        labels: Class labels aligned with the features.
        seed: Seed for bounded sampling, folds, and the model.
        allowance: Seconds the check may take by `cost`; the rows are
            sampled down to what fits, and the check is skipped, with the
            reason in its evidence and note, when `MIN_ROWS` rows do not
            fit. None bounds the rows by `MAX_ROWS` only.
        threads: Math-library threads the model may use (the run's
            `cpu_limit`); None leaves the libraries' default.

    Returns:
        A quality observation even when the check cannot be evaluated.
    """
    input_rows = len(features)
    limit = MAX_ROWS
    if allowance is not None:
        width = features.shape[1]
        classes = int(labels.nunique())
        fits = affordable(width, classes, allowance)
        if fits < MIN_ROWS:
            reason = (
                f"{MIN_ROWS} rows need about "
                f"{cost(MIN_ROWS, width, classes):.2f} s; the allowance is "
                f"{allowance:.2f} s ({SHARE:.0%} of the time budget)"
            )
            return unevaluated(
                f"label noise not checked: {reason}",
                "skipped",
                reason,
                "The check was skipped because it did not fit its share of "
                "the time budget.",
                seed,
                (input_rows, 0, fits),
            )
        limit = min(MAX_ROWS, fits)
    if input_rows > limit:
        rows = features.sample(limit, random_state=seed).index
        features, labels = features.loc[rows], labels.loc[rows]
    rows_checked = len(features)
    try:
        with threadpoolctl.threadpool_limits(limits=threads):
            report = contradict(features, labels, seed)
    except ValueError as error:
        # handled: not_evaluated observation carrying the error message
        return unevaluated(
            None,
            "not_evaluated",
            str(error),
            "The check was not evaluated because its data contract was not "
            "met.",
            seed,
            (input_rows, rows_checked, limit),
        )
    note = None
    if report.rate >= NOTE_RATE:
        note = (
            f"label noise: {report.rate:.0%} of {rows_checked} checked rows "
            "have a label that a model trained without them contradicts "
            "confidently; review "
            "before trusting scores"
        )
    return data.Observation(
        note=note,
        evidence=report.quality_evidence(input_rows, rows_checked, seed, limit),
    )
