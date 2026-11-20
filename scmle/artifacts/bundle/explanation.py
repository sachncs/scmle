"""Explaining predictions, recourse and what to label next."""

import pathlib
from typing import Any

import numpy
import pandas

from scmle.artifacts.bundle import core
from scmle.contracts import errors
from scmle.explainability import local
import scmle.explainability.counterfactual
import scmle.inference.mapping

BINARY_CLASSES = 2


class Label:
    """A wanted class label, the goal of a classifier's recourse search.

    Attributes:
        bundle: The bundle that predicts.
        desired: The wanted class label.
        column: Its probability column.
    """

    def __init__(self, bundle: core.Core, desired: Any) -> None:
        """Binds the bundle and the wanted class.

        Args:
            bundle: The bundle that predicts.
            desired: The wanted class label, one of the bundle's classes.
        """
        self.bundle = bundle
        self.desired = desired
        self.column = list(bundle.schema.classes or ()).index(desired)

    def progress(self, rows: pandas.DataFrame) -> numpy.ndarray:
        """Measures how close rows are to the wanted class.

        Args:
            rows: Candidate rows.

        Returns:
            The predicted probability of the wanted class.
        """
        return numpy.asarray(self.bundle.predict_proba(rows))[:, self.column]

    def reached(self, rows: pandas.DataFrame) -> numpy.ndarray:
        """Tests whether rows are predicted as the wanted class.

        Args:
            rows: Candidate rows.

        Returns:
            A boolean per row.
        """
        matches: numpy.ndarray = (
            numpy.asarray(self.bundle.predict(rows)) == self.desired
        )
        return matches


class Range:
    """A wanted range of values, the goal of a regressor's recourse search.

    Attributes:
        bundle: The bundle that predicts.
        low: The lowest wanted value.
        high: The highest wanted value.
    """

    def __init__(self, bundle: core.Core, desired: Any) -> None:
        """Binds the bundle and the wanted range.

        Args:
            bundle: The bundle that predicts.
            desired: The wanted `(low, high)` range.
        """
        self.bundle = bundle
        self.low, self.high = desired

    def progress(self, rows: pandas.DataFrame) -> numpy.ndarray:
        """Measures how close rows are to the wanted range.

        Args:
            rows: Candidate rows.

        Returns:
            Minus the distance of each prediction from the wanted range (0
            inside it).
        """
        values = numpy.asarray(self.bundle.predict(rows), dtype=float)
        distance: numpy.ndarray = -(
            numpy.maximum(self.low - values, 0)
            + numpy.maximum(values - self.high, 0)
        )
        return distance

    def reached(self, rows: pandas.DataFrame) -> numpy.ndarray:
        """Tests whether rows have a prediction inside the wanted range.

        Args:
            rows: Candidate rows.

        Returns:
            A boolean per row.
        """
        values = numpy.asarray(self.bundle.predict(rows), dtype=float)
        inside: numpy.ndarray = (values >= self.low) & (values <= self.high)
        return inside


class Explanation(core.Core):
    """Explanations, counterfactual recourse and labelling priorities."""

    def explain(
        self,
        data: str | pathlib.Path | pandas.DataFrame,
        background: str | pathlib.Path | pandas.DataFrame,
        mapping: scmle.inference.mapping.Mapper | None = None,
        class_label: Any = None,
    ) -> pandas.DataFrame:
        """Explains individual predictions against a background sample.

        Args:
            data: Rows to explain (DataFrame or path).
            background: Rows describing typical input, for example a sample
                of the training data.
            mapping: Explicit column-name policy; exact names if omitted.
            class_label: Class whose probability is explained; the positive
                class when omitted for binary models.

        Returns:
            A frame indexed like the prepared rows, with a `prediction`
            column (probability of the class, or the regression value) and
            one occlusion contribution column per feature: the prediction
            minus its mean after replacing that feature with background
            values (at most the first rows of `background` are used).

        Raises:
            ArtifactError: For a model without exactly two classes when
                `class_label` is omitted. A `class_label` that is not a class
                raises `ValueError` from the lookup.
        """
        rows = self.prepare(data, mapping)
        reference = self.prepare(background, mapping)
        if self.schema.task == "regression":
            score = self.pipeline.predict
        else:
            classes = list(self.classes or ())
            if class_label is None:
                if len(classes) != BINARY_CLASSES:
                    raise errors.ArtifactError(
                        "a multiclass model needs class_label to explain"
                    )
                class_label = classes[1]
            column = classes.index(class_label)
            estimator = self.calibrator or self.pipeline

            def score(frame: pandas.DataFrame) -> numpy.ndarray:
                """Scores rows for the explained output.

                Args:
                    frame: Rows to score.

                Returns:
                    The probability of the explained class (calibrated if a
                    calibrator is attached).
                """
                return numpy.asarray(estimator.predict_proba(frame))[:, column]

        return local.Occlusion(score, self.schema.features).explain(
            rows, reference
        )

    def counterfactual(
        self,
        row: pandas.DataFrame | dict[str, Any],
        desired: Any,
        actionable: list[str],
        background: str | pathlib.Path | pandas.DataFrame,
        max_changes: int = 3,
    ) -> scmle.explainability.counterfactual.Counterfactual:
        """Searches greedily for a small change that reaches the wanted outcome.

        Only the columns in `actionable` are ever changed, and only to
        candidate values drawn from `background` (quantiles of numeric
        columns, the most frequent categories). Unnecessary changes are
        dropped afterwards, but the change is not guaranteed to be the
        smallest possible.

        Args:
            row: One row (a dict or a one-row frame).
            desired: The wanted class label for classifiers, or a
                `(low, high)` range for regression.
            actionable: Columns the search may change.
            background: Typical rows, the source of candidate values.
            max_changes: Most columns it may change.

        Returns:
            The search result; `found` says whether the outcome was reached.

        Raises:
            ArtifactError: For unknown or non-feature actionable columns, a
                `row` that is not exactly one row, or a `desired` class this
                model does not have. A model without probabilities raises it
                from `predict_proba` when classifying.
        """
        features = [*self.schema.numeric, *self.schema.categorical]
        unknown = [c for c in actionable if c not in features]
        if unknown:
            raise errors.ArtifactError(
                f"not model features, so they cannot be changed: {unknown}"
            )
        frame = pandas.DataFrame([row] if isinstance(row, dict) else row)
        if len(frame) != 1:
            raise errors.ArtifactError("give exactly one row")
        reference = self.read(background)
        fixed = {c: frame.iloc[0][c] for c in frame.columns}
        goal: Label | Range
        if self.schema.classes is not None:
            if desired not in self.schema.classes:
                raise errors.ArtifactError(
                    f"{desired!r} is not one of {list(self.schema.classes)}"
                )
            goal = Label(self, desired)
        else:
            goal = Range(self, desired)
        search = scmle.explainability.counterfactual.Recourse(
            goal.progress,
            goal.reached,
            self.predict,
            reference,
            actionable,
            max_changes,
        )
        return search.search(fixed)

    def prioritize(
        self,
        data: str | pathlib.Path | pandas.DataFrame,
        count: int = 10,
        mapping: scmle.inference.mapping.Mapper | None = None,
    ) -> pandas.DataFrame:
        """Picks the unlabeled rows a person should label next.

        Classification ranks by the smallest gap between the two most likely
        classes (the model cannot tell them apart), using the pipeline's
        uncalibrated probabilities. Regression ranks by the width of the
        uncalibrated quantile band (the model is least sure there).

        Args:
            data: Unlabeled rows (DataFrame or supported tabular path).
            count: How many rows to return.
            mapping: Explicit column-name policy; exact names if omitted.

        Returns:
            Up to `count` of the most informative rows: `position` in `data`
            and an `uncertainty` score in [0, 1] for classification (a width
            in target units for regression), most uncertain first.

        Raises:
            ArtifactError: For a regression bundle without quantile models.
        """
        frame = self.prepare(data, mapping)
        if self.classes is not None:
            ordered = numpy.sort(
                numpy.asarray(self.pipeline.predict_proba(frame)), axis=1
            )
            top = ordered[:, -1]
            runner_up = ordered[:, -2] if ordered.shape[1] > 1 else 0.0
            uncertainty = 1.0 - (top - runner_up)
        else:
            if self.quantile is None:
                raise errors.ArtifactError(
                    "this bundle has no quantile models to rank by"
                )
            lower, upper = self.quantile.native(frame)
            uncertainty = upper - lower
        order = numpy.argsort(-uncertainty, kind="stable")[:count]
        return pandas.DataFrame(
            {"position": order, "uncertainty": uncertainty[order]}
        )
