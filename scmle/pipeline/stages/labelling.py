"""Task resolution, the first part of the profile stage."""

import dataclasses
from typing import ClassVar

import pandas

from scmle.contracts import errors
from scmle.contracts import tasks
import scmle.contracts.metrics
import scmle.data.validation.target
import scmle.pipeline.context
from scmle.pipeline.stages import bridge
from scmle.pipeline.stages import interface


@dataclasses.dataclass(frozen=True)
class Resolution(bridge.Bridge):
    """The resolved task and the labelled rows it is learned from.

    Attributes:
        task: The resolved task.
        metric: The primary metric.
        frame: The labelled rows (an ordinal target encoded as ranks).
        unlabeled: Rows without a target set aside by the `unlabeled`
            policy, or None when every row has one.
        labels: For an ordinal task, the labels from lowest to highest,
            else None (the context's `ordinal_labels`).
        notes: The intake's repairs, and the rows set aside (the context's
            `cleaning_notes`).
    """

    names: ClassVar[dict[str, str]] = {
        "labels": "ordinal_labels",
        "notes": "cleaning_notes",
    }

    task: tasks.Task
    metric: scmle.contracts.metrics.Metric
    frame: pandas.DataFrame
    unlabeled: pandas.DataFrame | None
    labels: tuple[str | int, ...] | None
    notes: list[str]

    def summary(self) -> object:
        """The task, the metric and the rows set aside.

        Returns:
            The identity of the resolution.
        """
        return {
            "task": self.task,
            "metric": self.metric.name,
            "unlabeled": None
            if self.unlabeled is None
            else len(self.unlabeled),
            "ordinal": self.labels,
            "notes": list(self.notes),
        }


class Labelling(interface.Stage):
    """The unlabelled rows set aside, the task inferred, the target vetted.

    Part of `ProfileStage`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    def set_aside_unlabeled(
        self, context: scmle.pipeline.context.Context, frame: pandas.DataFrame
    ) -> tuple[pandas.DataFrame, pandas.DataFrame]:
        """Applies the `unlabeled` policy to rows with a missing target.

        Args:
            context: The run context, holding the data specification.
            frame: The data, some of whose target values are missing.

        Returns:
            The labeled rows, and the others without the target column
            (their weight, if the data has one, defaults to 1).

        Raises:
            BlockedError: Under the default `block` policy.
        """
        data = context.services.spec.data
        missing = frame[data.target].isna()
        if data.unlabeled == "block":
            raise errors.BlockedError(
                f"target '{data.target}' has {int(missing.sum())} missing "
                "values; set data.unlabeled to 'ignore' or 'pseudo_label' to "
                "use the unlabeled rows"
            )
        rows = frame[missing].drop(columns=[data.target]).reset_index(drop=True)
        if data.weight_column and data.weight_column in rows:
            rows[data.weight_column] = rows[data.weight_column].fillna(1.0)
        return frame[~missing].reset_index(drop=True), rows

    def infer(
        self, context: scmle.pipeline.context.Context, frame: pandas.DataFrame
    ) -> Resolution:
        """Resolves the task and validates the target columns it needs.

        Args:
            context: The run context, holding the intake's repairs.
            frame: The loaded data.

        Returns:
            The resolution: the task, the metric, the frame (with rows
            lacking a target set aside, or an ordinal target encoded), the
            rows set aside and the ordinal labels.

        Raises:
            ConfigError: If `pseudo_label` is used for a non-classification
                task, the approved plan's task differs from the one the data
                resolves to, or a success criterion names a metric that is not
                reported.
        """
        spec = context.services.spec
        notes = list(context.cleaning_notes)
        unlabeled = None
        if frame[spec.data.target].isna().any():
            frame, unlabeled = self.set_aside_unlabeled(context, frame)
            notes.append(
                f"{len(unlabeled)} row(s) without a target were set aside "
                f"(policy '{spec.data.unlabeled}')"
            )
        task = scmle.data.validation.target.TaskResolver(spec).resolve(
            frame[spec.data.target]
        )
        if (
            unlabeled is not None
            and spec.data.unlabeled == "pseudo_label"
            and not task.is_classification
        ):
            raise errors.ConfigError(
                "unlabeled='pseudo_label' supports classification only"
            )
        approved = context.services.approved_plan
        if approved is not None and approved.task != task.value:
            raise errors.ConfigError(
                f"the approved plan is for {approved.task} but the data "
                f"resolves to {task.value}"
            )
        self.confirm(spec, frame, task)
        labels = None
        if task is tasks.Task.ORDINAL:
            labels = tuple(spec.data.ordinal_order)
            ranks = {label: rank for rank, label in enumerate(labels)}
            frame = frame.copy()
            frame[spec.data.target] = (
                frame[spec.data.target].map(ranks).astype(int)
            )
        metric = context.services.catalog.resolve(spec.primary_metric, task)
        unknown = sorted(
            set(spec.problem.success_criteria)
            - set(context.services.catalog.report_names[task])
        )
        if unknown:
            raise errors.ConfigError(
                f"success criteria name unreported metrics {unknown}; "
                "available: "
                f"{sorted(context.services.catalog.report_names[task])}"
            )
        return Resolution(task, metric, frame, unlabeled, labels, notes)

    def confirm(
        self, spec: tasks.Config, frame: pandas.DataFrame, task: tasks.Task
    ) -> None:
        """Checks the event, treatment or target columns a task needs.

        Args:
            spec: Project specification naming the columns.
            frame: The loaded data.
            task: The resolved task.
        """
        if task is tasks.Task.SURVIVAL:
            scmle.data.validation.target.check_events(
                frame, spec.data.event_column
            )
        if task is tasks.Task.UPLIFT:
            scmle.data.validation.target.check_treatment(
                frame, spec.data.treatment_column
            )
        if task in (
            tasks.Task.MULTILABEL,
            tasks.Task.MULTIOUTPUT,
            tasks.Task.FORECAST,
        ):
            scmle.data.validation.target.check_targets(
                frame, spec.data.targets, task
            )

    def vet(self, spec: tasks.Config, frame: pandas.DataFrame) -> None:
        """Checks that the declared group, time and weight columns exist.

        Args:
            spec: The project specification.
            frame: The loaded data.

        Raises:
            DataError: If a declared group, time or weight column is missing
                from the data.
        """
        for role, column in (
            ("group", spec.data.group_column),
            ("time", spec.data.time_column),
            ("weight", spec.data.weight_column),
        ):
            if column and column not in frame.columns:
                raise errors.DataError(
                    f"{role} column '{column}' not found; "
                    f"columns: {list(frame.columns)[:20]}"
                )
