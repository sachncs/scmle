"""The column checks of a dataset profile.

`Inspection` decides which columns become features and why the others do
not, checks the weights, counts duplicates and label conflicts, derives
calendar features, describes each feature's association with the target
and runs the leakage screen. `profiler.Profiler` assembles the portrait
from them.
"""

import numpy
import pandas

from scmle.assurance.admissibility import screen
from scmle.contracts import errors
from scmle.contracts import records
from scmle.contracts import tasks
from scmle.data.readers import streaming
from scmle.data.validation import columns

HIGH_MISSING_DROP = 0.95
HIGH_MISSING_WARN = 0.30
DATETIME_PARTS = (
    "year",
    "month",
    "dayofweek",
    "hour",
    "dayofyear",
    "timestamp",
)
OPTIONAL_COLUMN_ROLES = (
    ("treatment_column", "treatment indicator (used by uplift models only)"),
    (
        "baseline_column",
        "current process decision (used by the baseline claim only)",
    ),
    (
        "logged_action_column",
        "logged action (used by off-policy evaluation only)",
    ),
    (
        "propensity_column",
        "logged propensity (used by off-policy evaluation only)",
    ),
    ("query_column", "query identifier (ranking only)"),
    ("event_column", "event indicator (used by survival models only)"),
    ("weight_column", "sample weight (used for training only)"),
)


def disqualify(
    kind: columns.Family, series: pandas.Series, stats: records.Layout
) -> str | None:
    """Finds why a column cannot be a feature.

    Args:
        kind: The column's kind.
        series: The column.
        stats: The column's statistics, with `missing_rate` and `n_unique`.

    Returns:
        `missing >95%` when the missing rate exceeds `HIGH_MISSING_DROP`,
        `constant` for a column with at most one distinct value, otherwise
        the reason given by `kind.rationale`; None when the column
        can be a feature.
    """
    if stats["missing_rate"] > HIGH_MISSING_DROP:
        return "missing >95%"
    if stats["n_unique"] <= 1:
        return "constant"
    return kind.rationale(series)


MIN_BALANCE_ROWS = 100
MIN_BALANCE_CLASSES = 2


class Inspection:
    """The column checks of a profiler.

    Attributes:
        spec: Project specification describing column roles.
    """

    spec: tasks.Config

    def exclude(self, frame: pandas.DataFrame) -> dict[str, str]:
        """Maps configured non-feature columns to why they are excluded.

        Args:
            frame: The dataset.

        Returns:
            A dict from column name to the reason it is not a feature.

        Raises:
            DataError: If a configured column is absent from the data, or
                the weight column fails `check_weights`.
        """
        data = self.spec.data
        roles = dict.fromkeys(data.id_columns, "id column")
        roles |= dict.fromkeys(data.excluded_columns, "excluded by user")
        roles |= dict.fromkeys(
            data.unavailable_at_prediction,
            "not available at prediction time (feature-availability contract)",
        )
        if data.group_column:
            roles[data.group_column] = "group column (used for splitting only)"
        if data.time_column and not data.time_as_feature:
            roles[data.time_column] = "time column (used for splitting only)"
        for extra in data.targets[1:]:
            roles[extra] = "additional target (multi-target task)"
        if streaming.SPLIT_COLUMN in frame.columns:
            roles[streaming.SPLIT_COLUMN] = "split side chosen while streaming"
        for attribute, reason in OPTIONAL_COLUMN_ROLES:
            column = getattr(data, attribute)
            if column:
                roles[column] = reason
        for column in roles:
            if column not in frame.columns:
                raise errors.DataError(
                    f"configured column '{column}' not found in dataset"
                )
        if data.weight_column:
            self.check_weights(frame[data.weight_column])
        return roles

    def check_weights(self, weights: pandas.Series) -> None:
        """Requires finite, non-negative weights with a positive total.

        Args:
            weights: The weight column.

        Raises:
            DataError: If the weight column cannot weight training rows.
        """
        numeric = pandas.to_numeric(weights, errors="coerce").to_numpy(float)
        name = self.spec.data.weight_column
        if not numpy.isfinite(numeric).all():
            raise errors.DataError(
                f"weight column '{name}' has missing or non-numeric values"
            )
        if (numeric < 0).any() or numeric.sum() <= 0:
            raise errors.DataError(
                f"weight column '{name}' must be non-negative with a "
                "positive total"
            )

    def inspect(
        self, frame: pandas.DataFrame, roles: dict[str, str]
    ) -> tuple[
        dict[str, records.Layout],
        dict[str, str],
        dict[str, list[str]],
        list[str],
    ]:
        """Computes statistics and decides which columns become features.

        Args:
            frame: The dataset.
            roles: Columns already excluded by configuration.

        Returns:
            Per-column statistics, dropped-column reasons, feature columns
            grouped by kind name, and quality findings.
        """
        statistics: dict[str, records.Layout] = {}
        dropped = dict(roles)
        features: dict[str, list[str]] = {
            "numeric": [],
            "categorical": [],
            "datetime": [],
            "text": [],
        }
        quality: list[str] = []
        for name in frame.columns:
            if name == self.spec.data.target:
                continue
            series = frame[name]
            kind = columns.categorize(series)
            stats = kind.statistics(series)
            statistics[name] = stats
            if name in dropped:
                continue
            reason = disqualify(kind, series, stats)
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

    @staticmethod
    def derive(
        frame: pandas.DataFrame, stamps: tuple[str, ...]
    ) -> list[tasks.Node]:
        """Builds calendar and trend features for each datetime column.

        Only parts that vary in the data are produced: a column holding one
        year yields no year feature. The `timestamp` part carries the trend.

        Args:
            frame: The dataset.
            stamps: Names of the datetime columns.

        Returns:
            One `datepart` feature node per varying part and column, named
            `<column>__<part>`.
        """
        nodes = []
        for column in stamps:
            values = pandas.to_datetime(frame[column], errors="coerce")
            for part in DATETIME_PARTS:
                if part == "timestamp":
                    varies = values.nunique() > 1  # noqa: PD101 -- counts distinct non-missing values; an equality check against the first value would treat NaT as different
                else:
                    varies = getattr(values.dt, part).nunique() > 1  # noqa: PD101 -- counts distinct non-missing values; an equality check against the first value would treat NaT as different
                if varies:
                    nodes.append(
                        tasks.Node(
                            name=f"{column}__{part}",
                            op="datepart",
                            inputs=[column],
                            params={"part": part},
                        )
                    )
        return nodes

    def duplicate_count(self, frame: pandas.DataFrame) -> int:
        """Counts rows duplicated apart from their identifier columns.

        Args:
            frame: The dataset.

        Returns:
            The number of rows that repeat an earlier row once the
            identifier columns are dropped.
        """
        data = self.spec.data
        deduplicated = frame.drop(columns=list(data.id_columns))
        return int(deduplicated.duplicated().sum())

    @staticmethod
    def balance(target: pandas.Series, task: tasks.Task) -> list[str]:
        """Warns when classes are exactly equal in size.

        Real data are rarely perfectly balanced, so equal counts usually mean
        the table was down-sampled. Scores on it then describe a world with
        equal prevalence, not the one the model will serve.

        Args:
            target: The target column.
            task: Resolved task type.

        Returns:
            One warning sentence when there are at least
            `MIN_BALANCE_CLASSES` classes, at least `MIN_BALANCE_ROWS` rows
            and all classes have the same count; otherwise an empty list.
        """
        if not task.is_classification or len(target) < MIN_BALANCE_ROWS:
            return []
        counts = target.value_counts()
        if (
            len(counts) >= MIN_BALANCE_CLASSES
            and (counts == counts.iloc[0]).all()
        ):
            return [
                (
                    f"every class has exactly {int(counts.iloc[0])} rows: "
                    "if the table was balanced by sampling, test scores "
                    "reflect equal prevalence, not the real-world mix"
                )
            ]
        return []

    def dupes(
        self,
        frame: pandas.DataFrame,
        feature_columns: list[str],
        task: tasks.Task,
    ) -> list[str]:
        """Reports duplicate rows and, for classification, label conflicts.

        Args:
            frame: The dataset.
            feature_columns: Feature columns that define a combination.
            task: Resolved task type.

        Returns:
            An empty list without duplicate rows; otherwise the duplicate
            count and, for classification, the number of feature
            combinations that carry more than one label.
        """
        data = self.spec.data
        duplicates = self.duplicate_count(frame)
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

    def leakage_screen(
        self,
        frame: pandas.DataFrame,
        feature_columns: list[str],
        task: tasks.Task,
    ) -> screen.Screen | None:
        """Screens the training rows for leakage suspects, or None.

        There is no custody in a run without a charter, so the rows are the
        development rows given to the profiler. The screen controls the false
        discovery rate of its significant list; see `screen` for what that
        does and does not mean.

        Args:
            frame: The development rows.
            feature_columns: Feature columns to screen.
            task: Resolved task type.

        Returns:
            The screen report; None when the task is not classification and
            the target is not numeric.
        """
        data = self.spec.data
        target = frame[data.target]
        if task.is_classification:
            kind = "classification"
        elif pandas.api.types.is_numeric_dtype(target):
            kind = "regression"
        else:
            return None
        return screen.screen_table(
            frame,
            data.target,
            kind,
            feature_columns,
            self.spec.validation.seed,
            data.group_column,
            data.time_column,
            time_budget_seconds=self.spec.resources.time_budget_seconds,
        )
