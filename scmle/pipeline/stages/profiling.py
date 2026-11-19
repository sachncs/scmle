"""The profile stage: task, split and development profile."""

import dataclasses
from typing import ClassVar

import numpy
import pandas

from scmle.contracts import tasks
import scmle.contracts.data
import scmle.contracts.metrics
from scmle.data.splitting import splitters
from scmle.evaluation import backtest
from scmle.pipeline import engine
from scmle.pipeline import kinds
from scmle.pipeline import knowledge
from scmle.pipeline import products
from scmle.pipeline.stages import impounding
from scmle.pipeline.stages import ingestion
from scmle.pipeline.stages import labelling
from scmle.pipeline.stages import surveying
import scmle.runtime.state


@dataclasses.dataclass(frozen=True)
class Partition(products.Derived):
    """The task, the split and the development profile; made again on resume.

    It holds the locked test set, which is never written in clear.

    Attributes:
        task: The resolved task.
        metric: The primary metric.
        profile: The development-set profile.
        curation: The curation report.
        split: The split, with the locked test set.
        frame: The rows the later stages use.
        propensity: For an uplift task, how well the covariates predict
            treatment, else None.
        seasonal: Seasonal-naive forecasts for the locked rows of a
            forecast with a seasonal period, else None.
    """

    task: tasks.Task
    metric: scmle.contracts.metrics.Metric
    profile: scmle.contracts.data.Portrait
    curation: scmle.contracts.data.Curation | None
    split: splitters.Split
    frame: pandas.DataFrame
    propensity: dict[str, float | bool | str | None] | None = None
    seasonal: numpy.ndarray | None = None

    def summary(self) -> object:
        """The task, metric, split fingerprint and profile.

        Returns:
            The identity of the partition.
        """
        return {
            "task": self.task,
            "metric": self.metric.name,
            "split": self.split.manifest.fingerprint,
            "profile": self.profile.to_dict(),
        }

    def close(self) -> None:
        """Closes the locked test set the partition holds."""
        self.split.locked_test.close()


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
    makes: ClassVar[type[products.Product] | None] = Partition
    state: ClassVar[scmle.runtime.state.Progress] = (
        scmle.runtime.state.Progress.PROFILING
    )

    def describe(
        self,
        board: engine.Board,  # noqa: ARG002 -- Stage.describe
    ) -> str:
        """Names the stage.

        Args:
            board: The run's board; not consulted.

        Returns:
            The text "profile dataset".
        """
        return "profile dataset"

    def execute(self, board: engine.Board) -> Partition:
        """Resolves the task, splits the data and profiles the development set.

        Writes the profile, curation report and dataset card, enforces the
        leakage policy and, in a charter run, then runs the claim ledger. The
        validation helpers for the survival, uplift, multilabel, multioutput,
        forecast and ordinal targets raise their own errors.

        The resolution, the screen and the portrait are applied to the
        board as soon as they are made, because the claim ledger still
        reads them there; the claim ledger may amend the portrait, so the
        partition takes the profile and curation the board holds last.
        Until the partition is returned the stage owns the split: a failure
        after it closes the locked rows.

        The rows are the ingested ones (`ingestion.Intake`), read from the
        product the board holds.

        Args:
            board: The run's board; the partition is applied to it, so a
                caller that runs the stage outside the engine finds it there.

        Returns:
            The partition: the task, the metric, the split, the profile and
            curation report, and the rows the later stages use.

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
        intake = board.get(ingestion.Intake)
        resolution = self.infer(board, intake.frame)
        resolution.apply(board)
        self.vet(board.services.spec, resolution.frame)
        screen = self.synthesize(board, resolution.frame)
        screen.apply(board)
        split, rows, seasonal = self.divide(
            board, intake.kind, resolution.task, screen.frame
        )
        try:
            made = self.portray(
                board, resolution, screen.frame, split, rows, seasonal
            )
        except BaseException:
            split.locked_test.close()
            raise
        made.apply(board)
        return made

    def portray(
        self,
        board: engine.Board,
        resolution: labelling.Resolution,
        frame: pandas.DataFrame,
        split: splitters.Split,
        rows: pandas.DataFrame,
        seasonal: numpy.ndarray | None,
    ) -> Partition:
        """Profiles and curates the development rows, then screens them.

        Args:
            board: The run's board; receives the portrait.
            resolution: The resolved task and labelled rows.
            frame: The whole data, with the synthesised features.
            split: The split just made.
            rows: The rows the later stages use.
            seasonal: The seasonal-naive forecasts of the locked rows, if any.

        Returns:
            The partition.
        """
        spec = board.services.spec  # reweighing may have set the weight
        task = resolution.task
        profile, development, propensity = self.survey(
            board, spec, task, frame, split
        )
        portrait = self.curate(board, frame, development, profile)
        portrait.apply(board)
        board.journal.emit(
            "features.selected",
            "feature set and exclusions fixed from the profile",
            ("dataset_profile.json",),
            {
                "numeric": list(profile.feature_plan.numeric),
                "categorical": list(profile.feature_plan.categorical),
                "dropped": dict(profile.feature_plan.dropped),
            },
        )
        board.journal.track(
            fingerprint=knowledge.fingerprint(profile, task), task=task.value
        )
        self.enforce_leakage_policy(spec, profile)
        self.ledger(board, task).apply(board)
        latest = board.get(surveying.Portrait)
        return Partition(
            task,
            resolution.metric,
            latest.profile,
            latest.curation,
            split,
            rows,
            propensity,
            seasonal,
        )

    def divide(
        self,
        board: engine.Board,
        kind: kinds.Kind,
        task: tasks.Task,
        frame: pandas.DataFrame,
    ) -> tuple[splitters.Split, pandas.DataFrame, numpy.ndarray | None]:
        """Splits the data under its kind's constraints; records the manifest.

        Also computes the seasonal baseline for a forecast with a seasonal
        period, seals the confirmation set under a charter and applies
        reweighing.

        Args:
            board: The run's board; receives the split's manifest.
            kind: The kind of the rows, whose split constraints apply
                (`kinds.Kind.constrain`).
            task: The resolved task.
            frame: The data to split.

        Returns:
            The split; under a charter, or when resources are hard-enforced,
            its locked rows are sealed in a separate vault process. Then the
            rows the later stages use: the development rows when sealed,
            else `frame`. Last, the seasonal-naive forecasts of the locked
            rows, or None.
        """
        split = kind.constrain(board.services.spec, task).split(frame)
        data = board.services.spec.data
        seasonal = None
        if task is tasks.Task.FORECAST and data.seasonal_period:
            # Computed now: under hard isolation the frame held afterwards has
            # no locked rows, but the seasonal baseline needs their history.
            seasonal = backtest.seasonal_naive(
                frame,
                numpy.asarray(split.manifest.test_ids),
                data.value_column or "",
                data.group_column,
                len(data.targets),
                data.seasonal_period,
            )
        chartered = bool(board.services.spec.charter_path)
        if chartered:
            custody = self.sequester(board, task, frame, split)
            custody.apply(board)  # from here `Board.close` closes the vault
            split = custody.outcome.split
        self.reweigh(board, task, split)
        isolation = "in-process protocol"
        # A charter requests custody regardless of the resource-enforcement
        # mode: sealing the rows must not depend on trial supervision.
        if board.services.governor.hard or chartered:
            split.locked_test = self.seal(split.locked_test)
            frame = split.dev_features.assign(
                **{board.services.spec.data.target: split.dev_target}
            )
            isolation = "separate vault process"
        board.journal.write_json(
            "validation_manifest.json", split.manifest.to_dict()
        )
        board.journal.track(split_fingerprint=split.manifest.fingerprint)
        board.journal.emit(
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
        return split, frame, seasonal
