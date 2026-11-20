"""Prediction intervals and sets with a stated coverage."""

from collections.abc import Sequence
import pathlib
from typing import Any

import numpy
import pandas

from scmle.artifacts.bundle import core
from scmle.contracts import errors
from scmle.evaluation import cv_plus
from scmle.evaluation import shift_weights
import scmle.evaluation.conformal
import scmle.inference.mapping


class Intervals(core.Core):
    """Prediction intervals and sets from the stored calibration."""

    def predict_interval(
        self,
        data: str | pathlib.Path | pandas.DataFrame,
        coverage: float = scmle.evaluation.conformal.DEFAULT_COVERAGE,
        mapping: scmle.inference.mapping.Mapper | None = None,
        method: str = "residual",
        shift: bool = False,
    ) -> pandas.DataFrame:
        """Returns prediction intervals with a stated coverage (regression).

        Args:
            data: DataFrame or path to a supported tabular dataset.
            coverage: Share of new rows the interval should contain.
            mapping: Explicit column-name policy; exact names if omitted.
            method: `residual` for a margin of the same width everywhere, or
                `quantile` for conformalized quantile regression, whose
                width follows how noisy each row's neighbourhood is, or
                `cv_plus` for the CV+ interval from the stored fold models
                (kept only when the run had fewer than 1000 calibration rows,
                each held out in exactly one fold, and the models are small
                enough). `quantile` bounds are widened to include the point
                prediction.
            shift: Use the weighted conformal interval for covariate shift
                with the weights attached by `tilt`
                (`residual` method only). The ratio is estimated, so the
                coverage is approximate; a limit is infinite where the
                calibration weight cannot reach `coverage`.

        Returns:
            Columns `prediction`, `lower` and `upper`.

        Raises:
            ArtifactError: For classification models, unknown methods, or
                bundles saved without the needed calibration.
        """
        if shift and method != "residual":
            raise errors.ArtifactError(
                "shift-weighted intervals exist for the 'residual' method only"
            )
        if method == "quantile":
            if self.quantile is None:
                raise errors.ArtifactError(
                    "this bundle has no quantile calibration"
                )
            frame = self.prepare(data, mapping)
            lower, upper = self.quantile.interval(frame, coverage)
            predictions = numpy.asarray(
                self.pipeline.predict(frame), dtype=float
            )
            return pandas.DataFrame(
                {
                    "prediction": predictions,
                    "lower": numpy.minimum(lower, predictions),
                    "upper": numpy.maximum(upper, predictions),
                }
            )
        if method == "cv_plus":
            return self.jackknife(data, coverage, mapping)
        if method != "residual":
            raise errors.ArtifactError(
                f"unknown interval method '{method}'; use 'residual', "
                "'quantile' or 'cv_plus'"
            )
        calibrator = self.require("regression")
        if shift:
            return self.reweight(calibrator, data, coverage, mapping)
        predictions = numpy.asarray(self.predict(data, mapping), dtype=float)
        lower, upper = calibrator.interval(predictions, coverage)
        return pandas.DataFrame(
            {"prediction": predictions, "lower": lower, "upper": upper}
        )

    def jackknife(
        self,
        data: str | pathlib.Path | pandas.DataFrame,
        coverage: float,
        mapping: scmle.inference.mapping.Mapper | None,
    ) -> pandas.DataFrame:
        """Returns the CV+ interval from the stored fold models.

        Args:
            data: DataFrame or path to a supported tabular dataset.
            coverage: Share of new rows the interval should contain.
            mapping: Explicit column-name policy; exact names if omitted.

        Returns:
            Columns `prediction`, `lower` and `upper`.

        Raises:
            ArtifactError: If the bundle has no CV+ fold models.
        """
        if self.cv_plus is None:
            raise errors.ArtifactError(
                "this bundle has no CV+ fold models; the run stores them "
                "for regression with fewer than "
                f"{cv_plus.MIN_ROWS_FOR_SPLIT} calibration rows "
                "held out once each (see the cv_plus.skipped event)"
            )
        frame = self.prepare(data, mapping)
        lower, upper = self.cv_plus.interval(frame, coverage)
        predictions = numpy.asarray(self.pipeline.predict(frame), dtype=float)
        return pandas.DataFrame(
            {"prediction": predictions, "lower": lower, "upper": upper}
        )

    def tilt(
        self,
        calibration_rows: pandas.DataFrame,
        target_rows: pandas.DataFrame,
        columns: Sequence[str] | None = None,
        clip: float = shift_weights.DEFAULT_CLIP,
    ) -> shift_weights.Weights:
        """Estimates and stores likelihood ratios for weighted conformal.

        A discriminator is fitted between the calibration rows and the target
        rows. An effective sample size that is too small refuses the weights
        with `RefusedError` (a `ValueError`, not an `ArtifactError`)
        and none are stored.

        Args:
            calibration_rows: The rows the stored conformal scores came from, in
                score order (development rows in the order of the held-out
                positions of the split's folds).
            target_rows: Unlabeled rows of the population to predict for.
            columns: Numeric feature columns for the discriminator; all numeric
                schema features if omitted.
            clip: Upper bound on the ratio.

        Returns:
            The stored weights (also kept on the bundle and saved with it).

        Raises:
            ArtifactError: Without a regression conformal calibration, on a row
                count that differs from the stored scores, or on columns that
                are not numeric features of the schema.
        """
        calibrator = self.require("regression")
        if len(calibration_rows) != len(calibrator.scores):
            raise errors.ArtifactError(
                f"{len(calibration_rows)} calibration rows given but "
                f"{len(calibrator.scores)} conformal scores are stored"
            )
        chosen = tuple(
            columns
            if columns is not None
            else (
                c
                for c in self.features
                if pandas.api.types.is_numeric_dtype(calibration_rows[c])
            )
        )
        if not chosen or any(
            c not in self.features
            or not pandas.api.types.is_numeric_dtype(calibration_rows[c])
            for c in chosen
        ):
            raise errors.ArtifactError(
                "shift weights need numeric feature columns of the schema"
            )
        found = shift_weights.weigh(
            calibration_rows[list(chosen)].to_numpy(float),
            target_rows[list(chosen)].to_numpy(float),
            clip,
        )
        self.shift_weights = shift_weights.Weights(found, chosen)
        return self.shift_weights

    def reweight(
        self,
        calibrator: scmle.evaluation.conformal.ConformalCalibrator,
        data: str | pathlib.Path | pandas.DataFrame,
        coverage: float,
        mapping: scmle.inference.mapping.Mapper | None,
    ) -> pandas.DataFrame:
        """Computes weighted conformal intervals with the stored shift weights.

        Args:
            calibrator: The regression conformal calibrator.
            data: DataFrame or path to a supported tabular dataset.
            coverage: Share of new rows the interval should contain.
            mapping: Explicit column-name policy; exact names if omitted.

        Returns:
            Columns `prediction`, `lower` and `upper`.

        Raises:
            ArtifactError: If no shift weights are attached.
        """
        if self.shift_weights is None:
            raise errors.ArtifactError(
                "this bundle has no shift weights; call tilt"
            )
        frame = self.prepare(data, mapping)
        predictions = numpy.asarray(self.pipeline.predict(frame), dtype=float)
        lower, upper = calibrator.reweight(
            predictions,
            self.shift_weights.weights.calibration,
            self.shift_weights.test(frame),
            coverage,
        )
        return pandas.DataFrame(
            {"prediction": predictions, "lower": lower, "upper": upper}
        )

    def predict_set(
        self,
        data: str | pathlib.Path | pandas.DataFrame,
        coverage: float = scmle.evaluation.conformal.DEFAULT_COVERAGE,
        mapping: scmle.inference.mapping.Mapper | None = None,
        conditional: str | None = None,
    ) -> list[list[Any]]:
        """Returns prediction sets with a stated coverage (classification).

        Regression models, or bundles saved without conformal scores, raise
        `ArtifactError`. An unknown `conditional` mode or a coverage outside
        (0, 1) raises `ValueError` from the calibrator.

        Args:
            data: DataFrame or path to a supported tabular dataset.
            coverage: Share of new rows whose true label the set contains.
            mapping: Explicit column-name policy; exact names if omitted.
            conditional: `None` for a pooled margin (coverage on average over
                classes) or `class` for one margin per true class (coverage
                within each class). A class with too few calibration rows is
                kept in every set.

        Returns:
            For each row the labels that cannot be ruled out.
        """
        calibrator = self.require("classification")
        return calibrator.sets(
            self.predict_proba(data, mapping), coverage, conditional
        )

    def require(
        self, kind: str
    ) -> scmle.evaluation.conformal.ConformalCalibrator:
        """Returns the calibrator of the right kind or explains its absence.

        Args:
            kind: `regression` or `classification`.

        Returns:
            The conformal calibrator.

        Raises:
            ArtifactError: If the bundle has no calibrator of that kind (the run
                skips it when the time budget is spent or too few held-out rows
                exist).
        """
        if self.conformal is None or self.conformal.kind != kind:
            raise errors.ArtifactError(
                f"this bundle has no {kind} conformal calibration; the run "
                "skips it when the time budget is spent or too few held-out "
                "rows exist (see the conformal.skipped event in events.jsonl)"
            )
        return self.conformal
