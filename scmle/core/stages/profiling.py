"""The profile stage: task, split and development profile."""

from typing import ClassVar

import numpy
import pandas

from scmle.contracts import tasks
import scmle.core.context
from scmle.core.stages import impounding
from scmle.core.stages import ingestion
import scmle.core.state
from scmle.data.splitting import splitters
from scmle.evaluation import backtest
from scmle.intelligence import knowledge


class ProfileStage(impounding.Impounding):
    """Resolves the task, splits the data, and profiles the development rows.

    Only the target is looked at to resolve the task and to split (class
    counts for stratification, groups and time for the other protocols). Every
    statistic that drives a modelling decision (column statistics, feature
    selection, leakage hypotheses, the fingerprint) is then computed on the
    development partition alone, so the locked test rows never influence
    them.
    """

    requires: ClassVar[tuple[type, ...]] = (ingestion.IngestStage,)
    state: ClassVar[scmle.core.state.Progress] = (
        scmle.core.state.Progress.PROFILING
    )

    def describe(
        self,
        context: scmle.core.context.Context,  # noqa: ARG002 -- Stage.describe
    ) -> str:
        """Names the stage.

        Args:
            context: The run context; not consulted.

        Returns:
            The text "profile dataset".
        """
        return "profile dataset"

    def execute(self, context: scmle.core.context.Context) -> None:
        """Resolves the task, splits the data and profiles the development set.

        Writes the profile, curation report and dataset card, enforces the
        leakage policy and, in a charter run, then runs the claim ledger. The
        validation helpers for the survival, uplift, multilabel, multioutput,
        forecast and ordinal targets raise their own errors.

        Args:
            context: The run context; receives the task, metric, profile,
                curation report and the artifacts written for them.

        Raises:
            BlockedError: If curation leaves unresolved blocking findings, or
                `leakage_policy` is `block` and a leakage hypothesis was raised
                or a screened column could not be checked.
            ConfigError: If `pseudo_label` is used for a non-classification
                task, the approved plan's task differs from the one the data
                resolves to, or a success criterion names a metric that is not
                reported.
            DataError: If a declared group, time or weight column is missing
                from the data.
        """
        spec = context.spec
        frame = scmle.core.context.require(context.frame, "frame")
        frame, task = self.infer(context, frame)
        self.vet(spec, frame)
        frame = self.synthesize(context, frame)
        split = self.divide(context, task, frame)
        spec = context.spec  # reweighing may have set the weight column
        profile, development = self.survey(context, spec, task, frame, split)
        self.curate(context, frame, development, profile)
        context.emit(
            "features.selected",
            "feature set and exclusions fixed from the profile",
            ("dataset_profile.json",),
            {
                "numeric": list(profile.feature_plan.numeric),
                "categorical": list(profile.feature_plan.categorical),
                "dropped": dict(profile.feature_plan.dropped),
            },
        )
        context.track(
            fingerprint=knowledge.fingerprint(profile, task), task=task.value
        )
        self.enforce_leakage_policy(spec, profile)
        self.run_claim_ledger(context, task)

    def divide(
        self,
        context: scmle.core.context.Context,
        task: tasks.Task,
        frame: pandas.DataFrame,
    ) -> splitters.Split:
        """Splits the dataset and records the manifest.

        Also computes the seasonal baseline for a forecast with a seasonal
        period, seals the confirmation set under a charter and applies
        reweighing.

        Args:
            context: The run context; receives the split and its manifest.
            task: The resolved task.
            frame: The data to split.

        Returns:
            The split; under a charter, or when resources are hard-enforced, its
            locked rows are sealed in a separate vault process.
        """
        split = splitters.Splitter(context.spec, task).split(frame)
        data = context.spec.data
        if task is tasks.Task.FORECAST and data.seasonal_period:
            # Computed now: under hard isolation the frame held afterwards has
            # no locked rows, but the seasonal baseline needs their history.
            context.seasonal_baseline = backtest.seasonal_naive(
                frame,
                numpy.asarray(split.manifest.test_ids),
                data.value_column or "",
                data.group_column,
                len(data.targets),
                data.seasonal_period,
            )
        chartered = bool(context.spec.charter_path)
        if chartered:
            split = self.impound(context, task, frame, split)
        self.reweigh(context, task, split)
        isolation = "in-process protocol"
        # A charter requests custody regardless of the resource-enforcement
        # mode: sealing the rows must not depend on trial supervision.
        if context.regulator.hard or chartered:
            split.locked_test = self.seal(split.locked_test)
            context.frame = split.dev_features.assign(
                **{context.spec.data.target: split.dev_target}
            )
            isolation = "separate vault process"
        context.split = split
        context.write_json("validation_manifest.json", split.manifest.to_dict())
        context.track(split_fingerprint=split.manifest.fingerprint)
        context.emit(
            "split.created",
            f"{split.manifest.strategy} split",
            ("validation_manifest.json",),
            {
                "fingerprint": split.manifest.fingerprint,
                "strategy": split.manifest.strategy,
                "development_rows": len(split.manifest.train_ids),
                "test_rows": len(split.manifest.test_ids),
                "test_set_isolation": isolation,
                "notes": list(split.manifest.notes),
            },
        )
        return split
