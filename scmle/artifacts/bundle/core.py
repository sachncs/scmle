"""`Core`, the state of a bundle and its plain predictions."""

import pathlib
import threading
from typing import Any
from typing import TYPE_CHECKING

import numpy
import pandas

from scmle.artifacts.bundle import provenance
from scmle.contracts import artifacts
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data import feature_graph
from scmle.data.readers import interchange
from scmle.data.readers import tabular
import scmle.data.reference
import scmle.evaluation.conformal
import scmle.evaluation.quantile
from scmle.inference import validation
import scmle.inference.mapping
import scmle.inference.novelty

if TYPE_CHECKING:
    from scmle.evaluation import cv_plus
    from scmle.evaluation import shift_weights


class Core:
    """The state of a bundle and the predictions every task shares.

    `model.Bundle` documents the attributes. The archive, intervals,
    explanation and output parts of a bundle build on this one.
    """

    def __init__(
        self,
        pipeline: Any,
        schema: artifacts.Interface,
        info: provenance.Manifest,
        reference: scmle.data.reference.Baseline | None = None,
        history: pandas.DataFrame | None = None,
        calibrator: Any = None,
        conformal: scmle.evaluation.conformal.ConformalCalibrator | None = None,
        novelty: scmle.inference.novelty.Strangeness | None = None,
        quantile: scmle.evaluation.quantile.QuantileIntervals | None = None,
    ) -> None:
        """Initializes the bundle.

        Args:
            pipeline: Fitted preprocessing + estimator pipeline.
            schema: Input contract.
            info: Provenance metadata.
            reference: Training-time distributions for monitoring.
            history: Recent rows per entity that series features look back
                on, so records scored alone still get lag / rolling values.
            calibrator: Probability calibrator fitted on labeled rows after
                training (see `calibrate`), applied by `predict_proba`.
            conformal: Out-of-fold nonconformity scores that turn
                predictions into intervals or sets with a coverage level.
            novelty: Detector that flags rows unlike the training data.
            quantile: Low and high quantile models with calibration, for
                intervals whose width follows the row (regression).
        """
        self.pipeline = pipeline
        self.schema = schema
        self.info = info
        self.reference = reference
        self.history = history
        self.calibrator = calibrator
        self.conformal = conformal
        self.novelty = novelty
        self.quantile = quantile
        # The fold models behind `predict_interval(method="cv_plus")`; set
        # by the run or by `load`, kept out of the constructor signature.
        self.cv_plus: cv_plus.CvPlusEnsemble | None = None
        # Calibration weights for covariate shift, attached on demand by
        # `tilt`.
        self.shift_weights: shift_weights.StoredShiftWeights | None = None
        # Measured coverage of the stored intervals or sets on the final test
        # rows (`coverage_run.run_report`), set by the run.
        self.coverage_report: dict[str, Any] | None = None
        # Serializes `remember`, which rewrites `history` from itself.
        self.history_lock = threading.RLock()

    @property
    def features(self) -> list[str]:
        """Feature columns expected at inference."""
        return self.schema.features

    @property
    def classes(self) -> tuple[Any, ...] | None:
        """Class labels, or None for regression."""
        return self.schema.classes

    def read_frame(
        self, data: str | pathlib.Path | pandas.DataFrame
    ) -> pandas.DataFrame:
        """Loads inference input from a path or converts an in-memory table.

        Args:
            data: A supported tabular path, or an in-memory table (a pandas,
                Polars or Arrow table).

        Returns:
            The data as a pandas frame (a copy for pandas input).
        """
        if isinstance(data, (str, pathlib.Path)):
            return tabular.read_table(data)
        return interchange.to_pandas(data)

    def validate(
        self,
        data: str | pathlib.Path | pandas.DataFrame,
        mapping: scmle.inference.mapping.Mapper | None = None,
    ) -> artifacts.Conformance:
        """Checks input against the schema without predicting.

        Args:
            data: DataFrame or path to a supported tabular dataset.
            mapping: Explicit column-name policy; exact names if omitted.

        Returns:
            A report of missing and extra columns, missing-value counts and
            unseen categories; problems are reported, not raised.
        """
        frame = self.read_frame(data)
        if mapping is not None:
            frame = mapping.apply(frame, [*self.features, self.schema.target])
        return validation.Validator(self.schema).report(frame)

    def prepare(
        self,
        data: str | pathlib.Path | pandas.DataFrame,
        mapping: scmle.inference.mapping.Mapper | None = None,
    ) -> pandas.DataFrame:
        """Validates input and returns the feature columns.

        Missing columns, or non-numeric values in numeric columns, raise
        `SchemaError`.

        Args:
            data: DataFrame or path to a supported tabular dataset.
            mapping: Explicit column-name policy; exact names if omitted.

        Returns:
            The pipeline's input columns in schema order (with series features
            computed), followed by any optional inputs that are present.
        """
        frame = self.read_frame(data)
        if mapping is not None:
            frame = mapping.apply(frame, [*self.features, self.schema.target])
        validation.Validator(self.schema).check(frame)
        if self.schema.series:
            frame = self.assemble().apply(frame)
        wanted = [
            *self.schema.consumed,
            *(
                c
                for c in self.schema.optional_inputs
                if c in frame.columns and c not in self.schema.consumed
            ),
        ]
        return frame[wanted]

    def predict(
        self,
        data: str | pathlib.Path | pandas.DataFrame,
        mapping: scmle.inference.mapping.Mapper | None = None,
    ) -> numpy.ndarray:
        """Predicts labels or values through the full fitted pipeline.

        Args:
            data: DataFrame or path to a supported tabular dataset.
            mapping: Explicit column-name policy; exact names if omitted.

        Returns:
            One prediction per row (labels for classifiers, class labels mapped
            back from codes for ordinal tasks, values for regression, one column
            per target for multi-target models).
        """
        predictions: numpy.ndarray = self.pipeline.predict(
            self.prepare(data, mapping)
        )
        if self.schema.targets:
            return numpy.asarray(predictions)
        if self.schema.task == tasks.Task.ORDINAL.value:
            labels = numpy.asarray(self.schema.classes, dtype=object)
            decoded: numpy.ndarray = labels[
                numpy.asarray(predictions).astype(int)
            ]
            return decoded
        return predictions

    def predict_proba(
        self,
        data: str | pathlib.Path | pandas.DataFrame,
        mapping: scmle.inference.mapping.Mapper | None = None,
    ) -> numpy.ndarray:
        """Predicts class probabilities.

        Args:
            data: DataFrame or path to a supported tabular dataset.
            mapping: Explicit column-name policy; exact names if omitted.

        Returns:
            One row per input row and one column per class, calibrated if a
            calibrator is attached.

        Raises:
            ArtifactError: For regression models or estimators without
                probabilities.
        """
        if self.schema.task in (
            "regression",
            tasks.Task.MULTIOUTPUT.value,
        ) or not hasattr(self.pipeline, "predict_proba"):
            raise errors.ArtifactError(
                "this model does not provide probabilities"
            )
        frame = self.prepare(data, mapping)
        probabilities: numpy.ndarray
        if self.calibrator is not None:
            probabilities = self.calibrator.predict_proba(frame)
        else:
            probabilities = self.pipeline.predict_proba(frame)
        return probabilities

    def assemble(self) -> feature_graph.Temporal:
        """Builds the series feature builder for this model, with its history.

        Returns:
            A builder over the schema's series features, group and time columns
            and the stored history.
        """
        return feature_graph.Temporal(
            [tasks.Node(**node) for node in self.schema.series],
            self.schema.group_column,
            self.schema.time_column,
            self.history,
        )

    def remember(self, data: str | pathlib.Path | pandas.DataFrame) -> None:
        """Adds scored rows to the history later records will look back on.

        Call this after scoring when the rows are real, ordered observations
        (as the serving adapters do). Rows with the same entity and time as
        stored ones replace them. Nothing happens for models without series
        features.

        Args:
            data: The rows that were scored.
        """
        if not self.schema.series:
            return
        with self.history_lock:  # Read-modify-write of `history`.
            builder = self.assemble()
            frame = self.read_frame(data)
            columns = builder.historical
            parts = [frame[columns]]
            if self.history is not None:
                parts.insert(0, self.history)
            combined = pandas.concat(parts, ignore_index=True)
            key = [
                c
                for c in (self.schema.group_column, self.schema.time_column)
                if c
            ]
            combined = combined.drop_duplicates(subset=key, keep="last")
            self.history = builder.retain(combined)

    def atypical(
        self,
        data: str | pathlib.Path | pandas.DataFrame,
        mapping: scmle.inference.mapping.Mapper | None = None,
    ) -> pandas.DataFrame:
        """Flags rows that look unlike the training data.

        Args:
            data: DataFrame or path to a supported tabular dataset.
            mapping: Explicit column-name policy; exact names if omitted.

        Returns:
            A frame with `novelty_score`, `out_of_range_features` and
            `out_of_distribution` per row.

        Raises:
            ArtifactError: If the bundle has no novelty detector.
        """
        if self.novelty is None:
            raise errors.ArtifactError("this bundle has no novelty detector")
        frame = self.prepare(data, mapping)
        return self.novelty.score(frame)
