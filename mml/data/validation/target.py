"""Target validation: resolving the task from the actual target values."""

import pandas as pd

from mml.contracts import errors
from mml.contracts import tasks

TEXT_TARGET_MIN_CLASSES = 20
TEXT_TARGET_MAX_RATIO = 0.5
AMBIGUOUS_MAX_VALUES = 10
AMBIGUOUS_MAX_RATIO = 0.05


def is_numeric_target(target: pd.Series) -> bool:
    """Whether the target is numeric and not boolean."""
    return pd.api.types.is_numeric_dtype(
        target
    ) and not pd.api.types.is_bool_dtype(target)


MIN_EVENTS = 10


def check_events(frame: pd.DataFrame, column: str | None) -> None:
    """Requires a complete 0/1 event column with enough observed events.

    Raises:
        BlockedError: If the column is missing, not 0/1, has gaps, or records
            fewer than `MIN_EVENTS` events (censoring everything leaves
            nothing to learn from).
    """
    if column is None or column not in frame.columns:
        raise errors.BlockedError(f"event column '{column}' not found")
    values = pd.to_numeric(frame[column], errors="coerce")
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
    frame: pd.DataFrame, column: str, order: list[str | int]
) -> pd.DataFrame:
    """Copy of `frame` whose ordinal labels are replaced by integer ranks."""
    ranks = {label: rank for rank, label in enumerate(order)}
    encoded = frame.copy()
    encoded[column] = frame[column].map(ranks).astype(int)
    return encoded


def check_targets(
    frame: pd.DataFrame, columns: list[str], task: tasks.TaskType
) -> None:
    """Requires every target column to exist, be complete and fit the task.

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
    """Resolves the requested task against the actual target values."""

    def __init__(self, spec: tasks.ProjectSpec) -> None:
        """Initializes the resolver.

        Args:
            spec: Project specification carrying the task request.
        """
        self.request = spec.task
        self.order = list(spec.data.ordinal_order)

    def resolve(self, target: pd.Series) -> tasks.TaskType:
        """Returns the concrete task type.

        Args:
            target: The target column.

        Returns:
            Binary, multiclass or regression.

        Raises:
            BlockedError: If the request and the data are inconsistent or the
                task is ambiguous.
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
        if self.request in ("multilabel", "multioutput"):
            return tasks.TaskType(self.request)
        if self.request == "regression":
            return self.resolve_regression(target)
        if self.request == "auto":
            return self.resolve_auto(target)
        return self.resolve_classification(target)

    def resolve_survival(self, target: pd.Series) -> tasks.TaskType:
        """Validates a survival request: durations must be positive numbers."""
        if not is_numeric_target(target) or (target <= 0).any():
            raise errors.BlockedError(
                "survival needs a numeric duration target greater than zero"
            )
        return tasks.TaskType.SURVIVAL

    def resolve_ranking(self, target: pd.Series) -> tasks.TaskType:
        """Relevance grades must be non-negative whole numbers."""
        values = pd.to_numeric(target, errors="coerce")
        if values.isna().any() or (values < 0).any() or (values % 1 != 0).any():
            raise errors.BlockedError(
                "ranking needs a relevance target of non-negative whole "
                "numbers (0 = not relevant)"
            )
        return tasks.TaskType.RANKING

    def resolve_ordinal(self, target: pd.Series) -> tasks.TaskType:
        """Requires the listed order to match the labels in the data."""
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

    def resolve_regression(self, target: pd.Series) -> tasks.TaskType:
        """Validates an explicit regression request."""
        if not is_numeric_target(target):
            raise errors.BlockedError(
                "regression requested but the target is not numeric"
            )
        return tasks.TaskType.REGRESSION

    def resolve_classification(self, target: pd.Series) -> tasks.TaskType:
        """Validates an explicit (possibly generic) classification request."""
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

    def resolve_auto(self, target: pd.Series) -> tasks.TaskType:
        """Infers the task; blocks when the choice is genuinely ambiguous."""
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
