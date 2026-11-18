"""Task resolution, the first part of the profile stage."""

import pandas

from scmle.contracts import errors
from scmle.contracts import tasks
import scmle.data.validation.target
import scmle.pipeline.context
from scmle.pipeline.stages import interface


class Labelling(interface.Stage):
    """The unlabelled rows set aside, the task inferred, the target vetted.

    Part of `ProfileStage`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    def set_aside_unlabeled(
        self, context: scmle.pipeline.context.Context, frame: pandas.DataFrame
    ) -> pandas.DataFrame:
        """Applies the `unlabeled` policy to rows with a missing target.

        Args:
            context: The run context; receives the unlabeled rows and the
                labeled frame.
            frame: The data, some of whose target values are missing.

        Returns:
            The labeled rows. The others wait in `context.unlabeled` (their
            weight, if the data has one, defaults to 1).

        Raises:
            BlockedError: Under the default `block` policy.
        """
        data = context.spec.data
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
        context.unlabeled = rows
        context.cleaning_notes.append(
            f"{len(rows)} row(s) without a target were set aside "
            f"(policy '{data.unlabeled}')"
        )
        context.frame = frame[~missing].reset_index(drop=True)
        return context.frame

    def infer(
        self, context: scmle.pipeline.context.Context, frame: pandas.DataFrame
    ) -> tuple[pandas.DataFrame, tasks.Task]:
        """Resolves the task and validates the target columns it needs.

        Sets the context's task and metric (and, for an ordinal task, the
        encoded frame and the labels).

        Args:
            context: The run context.
            frame: The loaded data.

        Returns:
            The frame (with rows lacking a target set aside, or an ordinal
            target encoded) and the resolved task.

        Raises:
            ConfigError: If `pseudo_label` is used for a non-classification
                task, the approved plan's task differs from the one the data
                resolves to, or a success criterion names a metric that is not
                reported.
        """
        spec = context.spec
        target = frame[spec.data.target]
        if target.isna().any():
            frame = self.set_aside_unlabeled(context, frame)
            target = frame[spec.data.target]
        task = scmle.data.validation.target.TaskResolver(spec).resolve(target)
        if (
            context.unlabeled is not None
            and spec.data.unlabeled == "pseudo_label"
            and not task.is_classification
        ):
            raise errors.ConfigError(
                "unlabeled='pseudo_label' supports classification only"
            )
        approved = context.approved_plan
        if approved is not None and approved.task != task.value:
            raise errors.ConfigError(
                f"the approved plan is for {approved.task} but the data "
                f"resolves to {task.value}"
            )
        self.confirm(spec, frame, task)
        if task is tasks.Task.ORDINAL:
            ranks = {
                label: rank
                for rank, label in enumerate(spec.data.ordinal_order)
            }
            frame = frame.copy()
            frame[spec.data.target] = (
                frame[spec.data.target].map(ranks).astype(int)
            )
            context.frame = frame
            context.ordinal_labels = tuple(spec.data.ordinal_order)
        context.task = task
        context.metric = context.catalog.resolve(spec.primary_metric, task)
        unknown = sorted(
            set(spec.problem.success_criteria)
            - set(context.catalog.report_names[task])
        )
        if unknown:
            raise errors.ConfigError(
                f"success criteria name unreported metrics {unknown}; "
                f"available: {sorted(context.catalog.report_names[task])}"
            )
        return frame, task

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
