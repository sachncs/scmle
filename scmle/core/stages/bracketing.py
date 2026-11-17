"""Quantile brackets and interval coverage of the package."""

from typing import Any

import numpy

import scmle.artifacts.bundle
from scmle.contracts import tasks
import scmle.core.context
from scmle.core.stages import cost
from scmle.core.stages import diagnosis
from scmle.core.stages import profiling
from scmle.core.stages import scoring
from scmle.evaluation import conformal
from scmle.evaluation import coverage_confirm
from scmle.evaluation import coverage_run
from scmle.evaluation import coverage_slices
from scmle.evaluation import quantile
from scmle.preparation import preprocessing


class Bracketing(diagnosis.Diagnosis):
    """Quantile brackets, interval coverage and the opening of `C`.

    Part of `PackageStage`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    def settle(
        self,
        context: scmle.core.context.Context,
        bundle: scmle.artifacts.bundle.Bundle,
    ) -> None:
        """Opens `C` once the bundle and its calibration are frozen.

        Only for a charter that asked for the coverage claim on `C`: the
        ledger was held back, and now registers the coverage claim with the
        others and tests everything in the one opening. If the coverage claim
        cannot be built (no registered claim hash or no conformal calibration
        on the bundle), its document is written as not evaluated. Does nothing
        when no ledger is pending.

        Args:
            context: The run context, holding the pending ledger.
            bundle: The frozen model bundle, whose conformal calibration the
                coverage claim tests.
        """
        pending = context.pending_ledger
        if pending is None:
            return
        context.pending_ledger = None
        task = scmle.core.context.require(context.task, "task")
        runner = None
        if context.coverage_claim_sha256 is not None:
            runner = coverage_confirm.rig(
                bundle, task.value, context.coverage_claim_sha256
            )
        if runner is None:
            context.write_json(
                coverage_confirm.DOCUMENT,
                coverage_confirm.not_evaluated(
                    "the bundle holds no conformal calibration, so there is "
                    "no interval or set to test on C"
                ),
            )
        profiling.ProfileStage().conclude(
            context, pending, pending.subject, runner
        )

    def bracket(
        self, context: scmle.core.context.Context
    ) -> quantile.QuantileIntervals | None:
        """Fits and calibrates the low and high quantile models (regression).

        Both models are boosted trees on the standard preprocessing. Each
        cross-validation fold fits them on its training rows and records by
        how much each held-out row misses the band (negative inside it); the
        final pair is fit on all development rows.

        Args:
            context: The run context, holding the split, plan and budget.

        Returns:
            The intervals, or None for classification, when the time budget
            is spent or too few held-out rows exist.
        """
        task = scmle.core.context.require(context.task, "task")
        if task is not tasks.Task.REGRESSION:
            return None
        selected = scmle.core.context.require(context.selected, "selected")
        if cost.halt(context, selected, "quantile intervals"):
            return None
        split = scmle.core.context.require(context.split, "split")
        plan = scmle.core.context.require(
            context.profile, "profile"
        ).feature_plan
        adapter = context.registry.get("hist_gradient_boosting")
        seed = context.spec.validation.seed

        def build(level: float) -> Any:
            return preprocessing.instantiate(
                plan,
                adapter,
                task,
                seed,
                {"loss": "quantile", "quantile": level},
            )

        observed: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for fit_positions, evaluate_positions in split.manifest.folds:
            if cost.halt(context, selected, "quantile intervals"):
                return None
            fit = numpy.asarray(fit_positions)
            evaluate = numpy.asarray(evaluate_positions)
            train_x = split.dev_features.iloc[fit]
            train_y = split.dev_target.iloc[fit]
            low, high = quantile.corridor(
                build(quantile.LOWER_LEVEL).fit(train_x, train_y),
                build(quantile.UPPER_LEVEL).fit(train_x, train_y),
                split.dev_features.iloc[evaluate],
            )
            observed.extend(split.dev_target.iloc[evaluate])
            lows.extend(low)
            highs.extend(high)
        if len(observed) < conformal.MIN_CALIBRATION_ROWS:
            return None
        return quantile.QuantileIntervals(
            build(quantile.LOWER_LEVEL).fit(
                split.dev_features, split.dev_target
            ),
            build(quantile.UPPER_LEVEL).fit(
                split.dev_features, split.dev_target
            ),
            quantile.Band.calibrate(observed, lows, highs),
        )

    def gauge(
        self,
        context: scmle.core.context.Context,
        bundle: scmle.artifacts.bundle.Bundle,
        evaluation: dict[str, Any],
    ) -> None:
        """Measures the coverage of the bundle's intervals on the test rows.

        Does nothing without held-out test rows. Adds `quantile_coverage` to
        the evaluation when the bundle has quantile models, and stores and
        writes the coverage report when one is measured.

        Args:
            context: The run context, holding the test rows and the claim.
            bundle: The model bundle; receives the coverage report.
            evaluation: The evaluation document; receives the quantile
                coverage curve.
        """
        if bundle.quantile is not None and context.test_rows is not None:
            evaluation["quantile_coverage"] = bundle.quantile.covering(
                *context.test_rows, scoring.COVERAGE_LEVELS
            )
        if context.test_rows is not None:
            # Each declared slice column becomes a label series aligned with
            # the test rows: a numeric column with many values is banded at
            # quantiles of the development rows, any other keeps its values as
            # text, and a column the rows lack maps to None.
            split = scmle.core.context.require(context.split, "split")
            test_features = context.test_rows[0]
            slices = {
                column: (
                    coverage_slices.bucket(
                        test_features[column], split.dev_features.get(column)
                    )
                    if column in test_features.columns
                    else None
                )
                for column in scoring.slicers(context)
            }
            measured = coverage_run.run_report(
                bundle,
                *context.test_rows,
                scoring.COVERAGE_LEVELS,
                claim_sha256=context.coverage_claim_sha256,
                slices=slices,
            )
            if measured is not None:
                bundle.coverage_report = measured
                context.write_json("coverage_report.json", measured)
                context.emit(
                    "coverage.reported",
                    "coverage of the stored intervals or sets was measured "
                    "on the final test rows",
                    ("coverage_report.json",),
                    {"methods": sorted(measured["methods"])},
                )
