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


class TaskResolver:
    """Resolves the requested task against the actual target values."""

    def __init__(self, spec: tasks.ProjectSpec) -> None:
        """Initializes the resolver.

        Args:
            spec: Project specification carrying the task request.
        """
        self.request = spec.task

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
        if self.request == "regression":
            return self.resolve_regression(target)
        if self.request == "auto":
            return self.resolve_auto(target)
        return self.resolve_classification(target)

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
