"""Split-conformal uncertainty: prediction intervals and sets with coverage.

Given out-of-sample errors of a model on data it was not trained on, the
finite-sample quantile of those errors yields a margin such that a new
exchangeable row falls inside it with at least the requested probability.
Regression gets an interval around the prediction; classification gets the
set of labels whose probability is not too low. The calibrator keeps the
scores themselves, so any coverage level can be asked for later.

Besides the pooled margin, a classification calibrator keeps the scores of
each true class so that sets can be class-conditional (Mondrian): the margin
for class `k` comes from the calibration rows of class `k` only, so coverage
holds within each class given exchangeability inside the class. A class with
too few calibration rows gets an infinite margin, which puts the label in
every set; the report of that is the caller's duty (`thin_classes`).
`weighted_quantile` is the weighted conformal quantile for covariate shift
with a known or estimated likelihood ratio (Tibshirani, Barber, Candes and
Ramdas, arXiv:1904.06019). `cv_plus_bounds` is the CV+ interval of Barber,
Candes, Ramdas and Tibshirani (arXiv:1905.02928) from stored fold models. Both
papers were read on 2026-11-05. Weighted conformal (their Corollary 1) gives
coverage at least `1 - alpha` under covariate shift when the weights are the
exact likelihood ratio `dP_test(x) / dP_train(x)` (or proportional to it), with
the test point's mass at plus infinity as here; label shift or an estimated
ratio is not covered by that statement. CV+ (Theorem 4) guarantees coverage
at least `1 - 2 alpha - min(2 (1 - 1/K) / (n/K + 1), (1 - K/n) / (K + 1))`,
hence at least `1 - 2 alpha - sqrt(2/n)`, for exchangeable data and an
algorithm that treats training points symmetrically; the paper constructs
examples showing the factor 2 cannot be removed for the jackknife+ without
further assumptions, and "near `1 - alpha`" is its empirical expectation, not
a theorem. The measured coverage in
`benchmarks/qualification/uncertainty_certify.py` is what this package relies
on.
"""

from collections.abc import Sequence
import dataclasses
import math
from typing import Any
from typing import Self

import numpy

MIN_CALIBRATION_ROWS = 20
DEFAULT_COVERAGE = 0.9


def finite_sample_quantile(scores: numpy.ndarray, coverage: float) -> float:
    """Conformal quantile of nonconformity scores.

    Args:
        scores: Nonconformity scores on held-out rows.
        coverage: Desired coverage in (0, 1).

    Returns:
        The smallest score such that at least `coverage` of new rows are
        expected to score no higher. Infinity when there are too few scores
        for the requested coverage.
    """
    count = len(scores)
    rank = math.ceil((count + 1) * coverage)
    if rank > count:
        return math.inf
    return float(numpy.sort(scores)[rank - 1])


def weighted_quantile(
    scores: numpy.ndarray,
    weights: numpy.ndarray,
    test_weight: float,
    coverage: float,
) -> float:
    """Weighted conformal quantile for covariate shift.

    Calibration point `i` carries mass `w_i / (sum w + w_test)` and the
    test point carries `w_test / (sum w + w_test)` at plus infinity. The
    result is the smallest score whose cumulative mass reaches `coverage`.

    Args:
        scores: Nonconformity scores on held-out rows.
        weights: Likelihood ratios (target over calibration density) of
            those rows; non-negative.
        test_weight: The likelihood ratio at the test point.
        coverage: Desired coverage in (0, 1).

    Returns:
        The weighted quantile; infinity when the calibration mass cannot
        reach `coverage` or when the total mass, test weight included, is
        not positive.

    Raises:
        ValueError: On a length mismatch or a negative weight.
    """
    values = numpy.asarray(scores, dtype=float)
    mass = numpy.asarray(weights, dtype=float)
    if len(values) != len(mass):
        raise ValueError("scores and weights must have the same length")
    if (mass < 0).any() or test_weight < 0:
        raise ValueError("weights must be non-negative")
    total = float(mass.sum()) + float(test_weight)
    if total <= 0:
        return math.inf
    order = numpy.argsort(values, kind="stable")
    cumulative = numpy.cumsum(mass[order]) / total
    reached = numpy.flatnonzero(cumulative >= coverage - 1e-12)
    if reached.size == 0:
        return math.inf
    return float(values[order][reached[0]])


def weighted_margins(
    scores: numpy.ndarray,
    weights: numpy.ndarray,
    test_weights: numpy.ndarray,
    coverage: float,
) -> numpy.ndarray:
    """`weighted_quantile` for many test weights at once.

    Equal to calling `weighted_quantile` once per test weight (the tests
    check this), except when every calibration weight and the test weight
    are zero: `weighted_quantile` then gives infinity, this function the
    smallest score. It sorts the calibration scores a single time.

    Args:
        scores: Nonconformity scores on held-out rows.
        weights: Likelihood ratios of those rows; non-negative.
        test_weights: Likelihood ratios at the test points; non-negative.
        coverage: Desired coverage in (0, 1).

    Returns:
        One margin per test weight: the weighted quantile, or infinity where
        the calibration mass cannot reach `coverage`.

    Raises:
        ValueError: If `scores` and `weights` differ in length, or any weight
            is negative.
    """
    values = numpy.asarray(scores, dtype=float)
    mass = numpy.asarray(weights, dtype=float)
    ratios = numpy.asarray(test_weights, dtype=float)
    if len(values) != len(mass):
        raise ValueError("scores and weights must have the same length")
    if (mass < 0).any() or (ratios < 0).any():
        raise ValueError("weights must be non-negative")
    order = numpy.argsort(values, kind="stable")
    ordered = values[order]
    cumulative = numpy.cumsum(mass[order])
    needed = coverage * (cumulative[-1] + ratios) - 1e-12 * (
        cumulative[-1] + ratios
    )
    position = numpy.searchsorted(cumulative, needed, side="left")
    margins = numpy.full(len(ratios), math.inf)
    reachable = position < len(ordered)
    margins[reachable] = ordered[position[reachable]]
    return margins


def cv_plus_bounds(
    fold_predictions: numpy.ndarray,
    fold_of_row: numpy.ndarray,
    residuals: numpy.ndarray,
    coverage: float = DEFAULT_COVERAGE,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """CV+ prediction intervals from the K fold models.

    With `a = 1 - coverage`, the upper limit is the
    `ceil((1 - a) * (n + 1))`-th smallest of `mu_{k(i)}(x) + R_i` and the
    lower limit is the `floor(a * (n + 1))`-th smallest of
    `mu_{k(i)}(x) - R_i`, where `mu_k` was fit without fold `k`, `k(i)` is
    the fold of calibration row `i` and `R_i` its absolute out-of-fold error.
    A rank outside `1..n` gives an unbounded side.

    Args:
        fold_predictions: `(rows, K)` predictions of each fold model.
        fold_of_row: Fold index (0..K-1) of each of the `n` calibration rows.
        residuals: Absolute out-of-fold error of each calibration row.
        coverage: Desired coverage in (0, 1).

    Returns:
        Lower and upper limits, one per prediction row.

    Raises:
        ValueError: If `coverage` is not strictly between 0 and 1,
            `fold_of_row` and `residuals` differ in length, or a fold index
            lies outside `0 .. K-1`.
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must lie strictly between 0 and 1")
    predictions = numpy.atleast_2d(numpy.asarray(fold_predictions, dtype=float))
    folds = numpy.asarray(fold_of_row, dtype=int)
    errors = numpy.asarray(residuals, dtype=float)
    if len(folds) != len(errors):
        raise ValueError("fold_of_row and residuals must have equal length")
    if folds.size and (folds.min() < 0 or folds.max() >= predictions.shape[1]):
        raise ValueError("fold_of_row names a fold without a model")
    count = len(errors)
    centres = predictions[:, folds]
    upper_rank = math.ceil(coverage * (count + 1))
    lower_rank = math.floor((1.0 - coverage) * (count + 1))
    rows = len(predictions)
    upper = numpy.full(rows, math.inf)
    lower = numpy.full(rows, -math.inf)
    if 1 <= upper_rank <= count:
        upper = numpy.sort(centres + errors, axis=1)[:, upper_rank - 1]
    if 1 <= lower_rank <= count:
        lower = numpy.sort(centres - errors, axis=1)[:, lower_rank - 1]
    return lower, upper


def one_row_per_entity(
    entities: Sequence[Any], rng: numpy.random.Generator
) -> list[int]:
    """Positions of one randomly chosen row of each entity, in entity order.

    For a new entity the exchangeable unit is the entity, not the row: rows of
    one entity share a random effect, so pooling them overstates how many
    independent scores there are. Taking one row per entity gives scores that
    are exchangeable with the score of a new entity (the coverage
    certification measures it).

    Args:
        entities: Entity label of each calibration row.
        rng: Source of the choice.

    Returns:
        Positions into `entities`, one per distinct entity, ordered by the
        entity's first appearance.
    """
    positions: dict[Any, list[int]] = {}
    for index, entity in enumerate(entities):
        positions.setdefault(entity, []).append(index)
    return [
        int(rows[int(rng.integers(len(rows)))]) for rows in positions.values()
    ]


@dataclasses.dataclass(frozen=True)
class ConformalCalibrator:
    """Calibration scores for intervals (regression) or sets (classification).

    Attributes:
        kind: `regression` or `classification`.
        scores: Nonconformity scores of held-out rows: absolute errors
            (regression) or one minus the probability of the true class
            (classification).
        classes: Class labels, in probability-column order (classification).
        class_scores: For classification, the scores of the rows whose true
            class is `classes[k]`, one tuple per class (empty when the
            calibrator predates class-conditional sets).
    """

    kind: str
    scores: tuple[float, ...]
    classes: tuple[Any, ...] = ()
    class_scores: tuple[tuple[float, ...], ...] = ()

    @classmethod
    def for_regression(
        cls, observed: Sequence[float], predicted: Sequence[float]
    ) -> Self:
        """Calibrates from out-of-sample predictions.

        Args:
            observed: True values of held-out rows.
            predicted: Out-of-sample predictions for the same rows.

        Returns:
            A regression calibrator whose scores are the absolute errors.

        Fewer than `MIN_CALIBRATION_ROWS` rows are rejected by `checked`
        with a `ValueError`.
        """
        errors = numpy.abs(numpy.asarray(observed) - numpy.asarray(predicted))
        return cls("regression", cls.checked(errors))

    @classmethod
    def for_classification(
        cls,
        labels: Sequence[Any],
        probabilities: numpy.ndarray,
        classes: Sequence[Any],
    ) -> Self:
        """Calibrates from out-of-sample class probabilities.

        Args:
            labels: True labels of held-out rows.
            probabilities: Out-of-sample probabilities, one column per class.
            classes: Class labels in column order.

        Returns:
            A classification calibrator holding the pooled scores (one minus
            the probability of the true class) and the scores grouped by
            true class.

        Raises:
            ValueError: If a label is outside `classes`. Fewer than
                `MIN_CALIBRATION_ROWS` rows are also rejected, by `checked`.
        """
        position = {label: index for index, label in enumerate(classes)}
        if any(label not in position for label in labels):
            raise ValueError("labels outside the known classes")
        true_probability = numpy.asarray(probabilities)[
            numpy.arange(len(labels)), [position[label] for label in labels]
        ]
        scores = cls.checked(1.0 - true_probability)
        by_class = tuple(
            tuple(
                float(score)
                for score, label in zip(scores, labels, strict=True)
                if label == known
            )
            for known in classes
        )
        return cls("classification", scores, tuple(classes), by_class)

    @staticmethod
    def checked(scores: numpy.ndarray) -> tuple[float, ...]:
        """Validates calibration input and freezes it.

        Args:
            scores: Nonconformity scores.

        Returns:
            The scores as a tuple of floats.

        Raises:
            ValueError: If there are fewer than `MIN_CALIBRATION_ROWS`
                scores.
        """
        if len(scores) < MIN_CALIBRATION_ROWS:
            raise ValueError(
                f"need at least {MIN_CALIBRATION_ROWS} calibration rows"
            )
        return tuple(float(s) for s in scores)

    def margin(self, coverage: float = DEFAULT_COVERAGE) -> float:
        """The margin guaranteeing `coverage` for exchangeable new rows.

        Args:
            coverage: Desired coverage, strictly between 0 and 1.

        Returns:
            The finite-sample quantile of the pooled scores; infinity when
            there are too few scores for `coverage`.

        Raises:
            ValueError: If coverage is not strictly between 0 and 1.
        """
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must lie strictly between 0 and 1")
        return finite_sample_quantile(numpy.asarray(self.scores), coverage)

    def interval(
        self,
        predictions: Sequence[float] | numpy.ndarray,
        coverage: float = DEFAULT_COVERAGE,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Prediction intervals for a regression model.

        Args:
            predictions: Point predictions.
            coverage: Desired coverage in (0, 1).

        Returns:
            The lower and upper limits: each prediction minus and plus the
            margin.

        Raises:
            ValueError: For a classification calibrator or a coverage not
                strictly between 0 and 1.
        """
        if self.kind != "regression":
            raise ValueError("intervals need a regression calibrator")
        values = numpy.asarray(predictions, dtype=float)
        margin = self.margin(coverage)
        return values - margin, values + margin

    def class_margins(
        self, coverage: float = DEFAULT_COVERAGE
    ) -> tuple[float, ...]:
        """One margin per class from that class's calibration rows only.

        A class with fewer than `ceil(1 / (1 - coverage)) - 1` rows has an
        infinite margin: its label is kept in every set.

        Args:
            coverage: Desired coverage, strictly between 0 and 1.

        Returns:
            The margin of each class, in the order of `classes`.

        Raises:
            ValueError: Without per-class scores or with a bad coverage.
        """
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must lie strictly between 0 and 1")
        if self.kind != "classification" or not self.class_scores:
            raise ValueError("this calibrator has no per-class scores")
        return tuple(
            finite_sample_quantile(numpy.asarray(scores), coverage)
            for scores in self.class_scores
        )

    def thin_classes(self, coverage: float = DEFAULT_COVERAGE) -> list[Any]:
        """Classes too rare for a finite class-conditional margin.

        Args:
            coverage: Desired coverage, strictly between 0 and 1.

        Returns:
            The labels whose class-conditional margin is infinite, in the
            order of `classes`. A calibrator without per-class scores or an
            invalid coverage is rejected by `class_margins` with a
            `ValueError`.
        """
        margins = self.class_margins(coverage)
        return [
            label
            for label, margin in zip(self.classes, margins, strict=True)
            if math.isinf(margin)
        ]

    def weighted_interval(
        self,
        predictions: Sequence[float] | numpy.ndarray,
        calibration_weights: Sequence[float] | numpy.ndarray,
        test_weights: Sequence[float] | numpy.ndarray,
        coverage: float = DEFAULT_COVERAGE,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Intervals under covariate shift (weighted conformal).

        Args:
            predictions: Point predictions at the test rows.
            calibration_weights: Likelihood ratios of the calibration rows,
                in the order the scores were stored.
            test_weights: Likelihood ratios of the test rows.
            coverage: Desired coverage in (0, 1).

        Returns:
            The lower and upper limits: each prediction minus and plus its
            own weighted margin (infinite where the calibration mass cannot
            reach `coverage`).

        Raises:
            ValueError: For a classification calibrator, if `predictions`
                and `test_weights` or the scores and `calibration_weights`
                differ in length, or if a weight is negative.
        """
        if self.kind != "regression":
            raise ValueError("intervals need a regression calibrator")
        values = numpy.asarray(predictions, dtype=float)
        ratios = numpy.asarray(test_weights, dtype=float)
        if len(values) != len(ratios):
            raise ValueError("predictions and test_weights differ in length")
        margins = weighted_margins(
            numpy.asarray(self.scores),
            numpy.asarray(calibration_weights),
            ratios,
            coverage,
        )
        return values - margins, values + margins

    def sets(
        self,
        probabilities: numpy.ndarray,
        coverage: float = DEFAULT_COVERAGE,
        conditional: str | None = None,
    ) -> list[list[Any]]:
        """Prediction sets for a classifier.

        Args:
            probabilities: Probabilities, one column per class.
            coverage: Desired coverage in (0, 1).
            conditional: `None` for the pooled margin, `class` for one
                margin per true class (coverage within each class).

        Returns:
            For each row, the labels whose score `1 - probability` does not
            exceed the margin, in the order of `classes`.

        Raises:
            ValueError: For a regression calibrator, an unknown
                `conditional`, or (for `class`) a calibrator without
                per-class scores; an invalid coverage is also rejected.
        """
        if self.kind != "classification":
            raise ValueError("sets need a classification calibrator")
        if conditional not in (None, "class"):
            raise ValueError("conditional must be None or 'class'")
        margin: float | numpy.ndarray = (
            numpy.asarray(self.class_margins(coverage))
            if conditional == "class"
            else self.margin(coverage)
        )
        keep = 1.0 - numpy.asarray(probabilities) <= margin
        return [
            [
                label
                for label, flag in zip(self.classes, row, strict=True)
                if flag
            ]
            for row in keep
        ]

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping.

        Returns:
            The kind, scores, classes and per-class scores, with tuples as
            lists.
        """
        return {
            "kind": self.kind,
            "scores": list(self.scores),
            "classes": list(self.classes),
            "class_scores": [list(scores) for scores in self.class_scores],
        }

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> Self:
        """Rebuilds a calibrator from `to_dict` output.

        Args:
            record: A mapping with `kind`, `scores`, `classes` and
                optionally `class_scores`.

        Returns:
            The rebuilt calibrator.
        """
        return cls(
            record["kind"],
            tuple(record["scores"]),
            tuple(record["classes"]),
            tuple(tuple(scores) for scores in record.get("class_scores", ())),
        )
