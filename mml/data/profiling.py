"""Dataset profiling: task resolution, feature typing, findings.

Profiling never fits model state. Risk findings are hypotheses, not verdicts:
a feature strongly associated with the target may be a good predictor or a
leak, and statistics alone cannot tell which.
"""

import pandas as pd

from mml.contracts import data as data_contracts
from mml.contracts import errors
from mml.contracts import tasks
from mml.data import columns

LEAK_ASSOCIATION = 0.98
MISSINGNESS_LEAK = 0.9
HIGH_MISSING_DROP = 0.95
HIGH_MISSING_WARN = 0.30
SEVERE_IMBALANCE = 10.0
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


def column_exclusion_reason(
    kind: columns.ColumnKind, series: pd.Series, stats: dict
) -> str | None:
    """Returns why a column cannot be a feature, or None."""
    if stats["missing_rate"] > HIGH_MISSING_DROP:
        return "missing >95%"
    if stats["n_unique"] <= 1:
        return "constant"
    return kind.exclusion_reason(series)


def summarize_target(
    target: pd.Series, task: tasks.TaskType
) -> tuple[dict, list[str]]:
    """Summarizes the target distribution and imbalance risk."""
    if not task.is_classification:
        moments = target.astype(float).agg(["mean", "std", "skew"])
        stats = {str(k): float(v) for k, v in moments.to_dict().items()}
        return {"task": task.value, **stats}, []
    counts = target.value_counts()
    ratio = float(counts.iloc[0] / counts.iloc[-1])
    summary = {
        "task": task.value,
        "n_classes": len(counts),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "imbalance_ratio": ratio,
    }
    risks = []
    if ratio >= SEVERE_IMBALANCE:
        risks.append(f"severe class imbalance (ratio {ratio:.1f}:1)")
    return summary, risks


class DatasetProfiler:
    """Builds a `DatasetProfile` for a dataset under a project spec."""

    def __init__(self, spec: tasks.ProjectSpec) -> None:
        """Initializes the profiler.

        Args:
            spec: Project specification describing column roles.
        """
        self.spec = spec

    def role_columns(self, frame: pd.DataFrame) -> dict[str, str]:
        """Maps configured non-feature columns to why they are excluded.

        Raises:
            DataError: If a configured column is absent from the data.
        """
        data = self.spec.data
        roles = dict.fromkeys(data.id_columns, "id column")
        roles |= dict.fromkeys(data.excluded_columns, "excluded by user")
        if data.group_column:
            roles[data.group_column] = "group column (used for splitting only)"
        if data.time_column:
            roles[data.time_column] = "time column (used for splitting only)"
        for column in roles:
            if column not in frame.columns:
                raise errors.DataError(
                    f"configured column '{column}' not found in dataset"
                )
        return roles

    def inspect_columns(
        self, frame: pd.DataFrame, roles: dict[str, str]
    ) -> tuple[dict, dict[str, str], dict[str, list[str]], list[str]]:
        """Computes statistics and decides which columns become features.

        Args:
            frame: The dataset.
            roles: Columns already excluded by configuration.

        Returns:
            Per-column statistics, dropped-column reasons, feature columns
            grouped by kind name, and quality findings.
        """
        statistics: dict[str, dict] = {}
        dropped = dict(roles)
        features: dict[str, list[str]] = {"numeric": [], "categorical": []}
        quality: list[str] = []
        for name in frame.columns:
            if name == self.spec.data.target:
                continue
            series = frame[name]
            kind = columns.kind_of(series)
            stats = kind.statistics(series)
            statistics[name] = stats
            if name in dropped:
                continue
            reason = column_exclusion_reason(kind, series, stats)
            if reason is not None:
                dropped[name] = reason
                quality.append(f"column '{name}' dropped: {reason}")
                continue
            features[kind.name].append(name)
            if stats["missing_rate"] > HIGH_MISSING_WARN:
                quality.append(
                    f"column '{name}' is {stats['missing_rate']:.0%} missing"
                )
        return statistics, dropped, features, quality

    def duplicate_findings(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
        task: tasks.TaskType,
    ) -> list[str]:
        """Reports duplicate rows and, for classification, label conflicts."""
        data = self.spec.data
        deduplicated = frame.drop(columns=list(data.id_columns))
        duplicates = int(deduplicated.duplicated().sum())
        if not duplicates:
            return []
        findings = [f"{duplicates} duplicate rows"]
        if feature_columns and task.is_classification:
            labels = frame.groupby(
                feature_columns, dropna=False, observed=True
            )[data.target].nunique()
            conflicts = int((labels > 1).sum())
            if conflicts:
                findings.append(
                    f"{conflicts} feature combinations have inconsistent labels"
                )
        return findings

    def leakage_findings(
        self,
        frame: pd.DataFrame,
        statistics: dict[str, dict],
        feature_columns: list[str],
        task: tasks.TaskType,
    ) -> list[str]:
        """Flags features that are almost perfectly tied to the target."""
        target = frame[self.spec.data.target]
        findings = []
        for name in feature_columns:
            series = frame[name]
            kind = columns.kind_of(series)
            score = kind.association(series, target, task)
            missing_score = kind.missingness_association(series, target, task)
            statistics[name]["target_association"] = round(score, 4)
            statistics[name]["missingness_association"] = round(
                missing_score, 4
            )
            if score >= LEAK_ASSOCIATION:
                findings.append(
                    f"feature '{name}' is almost perfectly associated with "
                    f"the target ({score:.3f}): possible leakage (or an "
                    "exceptionally strong predictor); review before trusting"
                )
            if missing_score >= MISSINGNESS_LEAK:
                findings.append(
                    f"whether feature '{name}' is recorded almost determines "
                    f"the target ({missing_score:.3f}): possible leakage "
                    "(information captured only for some outcomes); review "
                    "before trusting"
                )
        return findings

    def profile(
        self,
        frame: pd.DataFrame,
        manifest: data_contracts.DatasetManifest,
        task: tasks.TaskType,
    ) -> data_contracts.DatasetProfile:
        """Profiles the dataset.

        Args:
            frame: The ingested dataset.
            manifest: Its identity.
            task: Resolved task type.

        Returns:
            The profile including the resulting feature plan.

        Raises:
            BlockedError: If the target has missing values or no usable
                feature remains.
            DataError: If a configured column is absent.
        """
        target = frame[self.spec.data.target]
        if target.isna().any():
            raise errors.BlockedError(
                f"target '{target.name}' has {int(target.isna().sum())} "
                "missing values"
            )
        roles = self.role_columns(frame)
        statistics, dropped, features, quality = self.inspect_columns(
            frame, roles
        )
        usable = features["numeric"] + features["categorical"]
        if not usable:
            raise errors.BlockedError(
                "no usable feature columns remain after exclusions"
            )
        quality += self.duplicate_findings(frame, usable, task)
        risks = self.leakage_findings(frame, statistics, usable, task)
        target_summary, target_risks = summarize_target(target, task)
        return data_contracts.DatasetProfile(
            manifest=manifest,
            column_statistics=statistics,
            quality_findings=tuple(quality),
            risk_findings=tuple(risks + target_risks),
            target_summary=target_summary,
            feature_plan=data_contracts.FeaturePlan(
                numeric=tuple(features["numeric"]),
                categorical=tuple(features["categorical"]),
                dropped=dropped,
            ),
        )
