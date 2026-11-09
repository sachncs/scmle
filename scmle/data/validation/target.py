"""Target validation: resolving the task from the actual target values."""

import pandas

from scmle.contracts import errors
from scmle.contracts import tasks

TEXT_TARGET_MIN_CLASSES = 20
TEXT_TARGET_MAX_RATIO = 0.5
AMBIGUOUS_MAX_VALUES = 10
AMBIGUOUS_MAX_RATIO = 0.05


def is_numeric_target(target: pandas.Series) -> bool:
    """Reports whether the target is numeric and not boolean.

    Args:
        target: The target column.

    Returns:
        True for a numeric, non-boolean dtype.
    """
    return pandas.api.types.is_numeric_dtype(
        target
    ) and not pandas.api.types.is_bool_dtype(target)


MIN_EVENTS = 10
MIN_ARM_ROWS = 20


def check_events(frame: pandas.DataFrame, column: str | None) -> None:
    """Requires a complete 0/1 event column with enough observed events.

    Args:
        frame: Data containing the event column.
        column: Name of the event column, or None if not configured.

    Raises:
        BlockedError: If the column is missing, not 0/1, has gaps, or records
            fewer than `MIN_EVENTS` events (censoring everything leaves
            nothing to learn from).
    """
    if column is None or column not in frame.columns:
        raise errors.BlockedError(f"event column '{column}' not found")
    values = pandas.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not set(values.unique()) <= {0, 1}:
        raise errors.BlockedError(
            f"event column '{column}' must hold 0 (censored) or 1 (event) in "
            "every row"
        )
    if int(values.sum()) < MIN_EVENTS:
        raise errors.BlockedError(
            f"only {int(values.sum())} event(s) observed; survival models "
            f"need at least {MIN_EVENTS}"
        )


def encode_ordinal(
    frame: pandas.DataFrame, column: str, order: list[str | int]
) -> pandas.DataFrame:
    """Copies `frame`, replacing ordinal labels by integer ranks.

    Args:
        frame: Data containing the ordinal column.
        column: Name of the ordinal column.
        order: Labels from lowest to highest; a label's rank is its index.

    Returns:
        A copy of `frame` whose `column` holds integer ranks.
    """
    ranks = {label: rank for rank, label in enumerate(order)}
    encoded = frame.copy()
    encoded[column] = frame[column].map(ranks).astype(int)
    return encoded


def check_treatment(frame: pandas.DataFrame, column: str | None) -> None:
    """Requires a complete 0/1 treatment column with both arms present.

    Args:
        frame: Data containing the treatment column.
        column: Name of the treatment column, or None if not configured.

    Raises:
        BlockedError: If the column is missing, not 0/1, has gaps, or one
            arm is empty or smaller than 20 rows.
    """
    if column is None or column not in frame.columns:
        raise errors.BlockedError(f"treatment column '{column}' not found")
    values = pandas.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not values.isin([0, 1]).all():
        raise errors.BlockedError(
            f"treatment column '{column}' must hold 0 (control) or 1 "
            "(treated) in every row"
        )
    if min(int(values.sum()), int((1 - values).sum())) < MIN_ARM_ROWS:
        raise errors.BlockedError(
            f"each arm needs at least {MIN_ARM_ROWS} rows"
        )


def check_targets(
    frame: pandas.DataFrame, columns: list[str], task: tasks.TaskType
) -> None:
    """Requires every target column to exist, be complete and fit the task.

    Args:
        frame: Data containing the target columns.
        columns: Names of the target columns.
        task: Resolved task type; multilabel columns must hold 0/1, all
            others must be numeric and non-boolean.

    Raises:
        BlockedError: For a missing column, gaps, labels that are not 0/1
            (multilabel) or values that are not numeric (multioutput).
    """
    for column in columns:
        if column not in frame.columns:
            raise errors.BlockedError(f"target column '{column}' not found")
        values = frame[column]
        if values.isna().any():
            raise errors.BlockedError(
                f"target column '{column}' has missing values"
            )
        if task is tasks.TaskType.MULTILABEL:
            if not values.isin([0, 1]).all():
                raise errors.BlockedError(
                    f"label column '{column}' must hold 0 or 1"
                )
        elif not is_numeric_target(values):
            raise errors.BlockedError(
                f"target column '{column}' must be numeric"
            )


class TaskResolver:
    """Resolves the requested task against the actual target values.

    Attributes:
        request: The task requested in the project specification.
        order: Listed ordinal label order (empty when none is given).
    """

    def __init__(self, spec: tasks.ProjectSpec) -> None:
        """Initializes the resolver.

        Args:
            spec: Project specification carrying the task request.
        """
        self.request = spec.task
        self.order = list(spec.data.ordinal_order)

    def resolve(self, target: pandas.Series) -> tasks.TaskType:
        """Returns the concrete task type.

        Args:
            target: The target column.

        Returns:
            The concrete task type: the validated requested type, or for
            `auto` the inferred binary, multiclass or regression type.

        Raises:
            BlockedError: If the target has no non-missing values, the
                request and the data are inconsistent, or an `auto` request
                is ambiguous.
        """
        if target.isna().all():
            raise errors.BlockedError(
                f"target '{target.name}' has no non-missing values"
            )
        if self.request == "survival":
            return self.resolve_survival(target)
        if self.request == "ordinal":
            return self.resolve_ordinal(target)
        if self.request == "ranking":
            return self.resolve_ranking(target)
        if self.request == "uplift":
            return self.resolve_uplift(target)
        if self.request in ("multilabel", "multioutput", "forecast"):
            return tasks.TaskType(self.request)
        if self.request == "regression":
            return self.resolve_regression(target)
        if self.request == "auto":
            return self.resolve_auto(target)
        return self.resolve_classification(target)

    def resolve_survival(self, target: pandas.Series) -> tasks.TaskType:
        """Validates a survival request: durations must be positive numbers.

        Args:
            target: The duration column.

        Returns:
            `TaskType.SURVIVAL`.

        Raises:
            BlockedError: If the target is not numeric or has a value that
                is not greater than zero.
        """
        if not is_numeric_target(target) or (target <= 0).any():
            raise errors.BlockedError(
                "survival needs a numeric duration target greater than zero"
            )
        return tasks.TaskType.SURVIVAL

    def resolve_uplift(self, target: pandas.Series) -> tasks.TaskType:
        """The outcome must be numeric (0/1 labels or a quantity).

        Args:
            target: The outcome column.

        Returns:
            `TaskType.UPLIFT`.

        Raises:
            BlockedError: If the outcome is not numeric.
        """
        if not is_numeric_target(target):
            raise errors.BlockedError("uplift needs a numeric outcome")
        return tasks.TaskType.UPLIFT

    def resolve_ranking(self, target: pandas.Series) -> tasks.TaskType:
        """Resolves a ranking target, requiring non-negative whole grades.

        Args:
            target: The relevance column.

        Returns:
            `TaskType.RANKING`.

        Raises:
            BlockedError: If any grade is missing, unparseable, negative or
                not a whole number.
        """
        values = pandas.to_numeric(target, errors="coerce")
        if values.isna().any() or (values < 0).any() or (values % 1 != 0).any():
            raise errors.BlockedError(
                "ranking needs a relevance target of non-negative whole "
                "numbers (0 = not relevant)"
            )
        return tasks.TaskType.RANKING

    def resolve_ordinal(self, target: pandas.Series) -> tasks.TaskType:
        """Requires the listed order to match the labels in the data.

        Args:
            target: The ordinal label column.

        Returns:
            `TaskType.ORDINAL`.

        Raises:
            BlockedError: If a label in the data is missing from the listed
                order, or a listed label has no rows.
        """
        present = set(target.dropna().unique())
        listed = set(self.order)
        if present - listed:
            raise errors.BlockedError(
                "ordinal_order omits label(s) present in the data: "
                f"{sorted(map(str, present - listed))}"
            )
        if listed - present:
            raise errors.BlockedError(
                "ordinal_order lists label(s) with no rows: "
                f"{sorted(map(str, listed - present))}"
            )
        return tasks.TaskType.ORDINAL

    def resolve_regression(self, target: pandas.Series) -> tasks.TaskType:
        """Validates an explicit regression request.

        Args:
            target: The target column.

        Returns:
            `TaskType.REGRESSION`.

        Raises:
            BlockedError: If the target is not numeric.
        """
        if not is_numeric_target(target):
            raise errors.BlockedError(
                "regression requested but the target is not numeric"
            )
        return tasks.TaskType.REGRESSION

    def resolve_classification(self, target: pandas.Series) -> tasks.TaskType:
        """Validates an explicit (possibly generic) classification request.

        Args:
            target: The target column.

        Returns:
            `TaskType.BINARY` for two classes, otherwise
            `TaskType.MULTICLASS`.

        Raises:
            BlockedError: If the target has fewer than two classes, a
                `binary_classification` request does not have exactly two,
                or a `multiclass_classification` request has only two.
        """
        classes = target.nunique(dropna=True)
        if classes < 2:
            raise errors.BlockedError(
                f"classification needs >=2 classes; target has {classes}"
            )
        binary = classes == 2
        if self.request == "binary_classification" and not binary:
            raise errors.BlockedError(
                f"binary_classification requested but target has {classes} "
                "classes"
            )
        if self.request == "multiclass_classification" and binary:
            raise errors.BlockedError(
                "multiclass_classification requested but target has 2 classes"
            )
        return tasks.TaskType.BINARY if binary else tasks.TaskType.MULTICLASS

    def resolve_auto(self, target: pandas.Series) -> tasks.TaskType:
        """Infers the task; blocks when the choice is genuinely ambiguous.

        Args:
            target: The target column.

        Returns:
            `TaskType.BINARY` for two distinct values, `TaskType.MULTICLASS`
            for a non-numeric target with more, otherwise
            `TaskType.REGRESSION`.

        Raises:
            BlockedError: If the target has fewer than two distinct values,
                is non-numeric and looks like an identifier or free text,
                or is a small-cardinality integer for which classification
                versus regression is ambiguous.
        """
        classes = target.nunique(dropna=True)
        if classes < 2:
            raise errors.BlockedError(
                f"target '{target.name}' has {classes} distinct value(s); "
                "there is nothing to learn"
            )
        if (
            not is_numeric_target(target)
            and classes > TEXT_TARGET_MIN_CLASSES
            and classes / max(len(target), 1) > TEXT_TARGET_MAX_RATIO
        ):
            raise errors.BlockedError(
                f"target '{target.name}' is non-numeric with {classes} "
                "distinct values: it looks like an identifier or free text, "
                "not a class label; choose another target"
            )
        if not is_numeric_target(target):
            return (
                tasks.TaskType.BINARY
                if classes == 2
                else tasks.TaskType.MULTICLASS
            )
        if classes == 2:
            return tasks.TaskType.BINARY
        values = target.dropna()
        integral = bool((values % 1 == 0).all())
        if (
            integral
            and classes <= AMBIGUOUS_MAX_VALUES
            and classes / max(len(target), 1) < AMBIGUOUS_MAX_RATIO
        ):
            raise errors.BlockedError(
                f"target '{target.name}' looks like a small-cardinality "
                f"integer ({classes} values): classification vs regression "
                "is ambiguous; set task= explicitly"
            )
        return tasks.TaskType.REGRESSION
