"""Dataset profiling: task resolution, feature typing, findings.

Profiling never fits model state. Risk findings are hypotheses, not verdicts:
a feature strongly associated with the target may be a good predictor or a
leak, and statistics alone cannot tell which. Leakage suspicion comes from
`scmle.assurance.admissibility.screen`, a permutation test with multiplicity
control, not from a fixed cutoff.
"""

from collections.abc import Mapping
import dataclasses
from typing import Any

import numpy
import pandas
import threadpoolctl

from scmle.assurance.admissibility import screen
from scmle.contracts import errors
from scmle.contracts import tasks
import scmle.contracts.data
from scmle.data.profiling import label_noise
from scmle.data.profiling import missingness
from scmle.data.readers import streaming
from scmle.data.validation import columns
from scmle.pipeline import fingerprint

HIGH_MISSING_DROP = 0.95
HIGH_MISSING_WARN = 0.30
SEVERE_IMBALANCE = 10.0
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
    kind: columns.Family, series: pandas.Series, stats: dict[str, Any]
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


def summarize(
    target: pandas.Series, task: tasks.Task
) -> tuple[dict[str, Any], list[str]]:
    """Summarizes the target distribution and imbalance risk.

    Args:
        target: The target column.
        task: Resolved task type.

    Returns:
        A pair of the summary dict and the risk sentences. For
        classification the summary holds class counts and the imbalance
        ratio (largest to smallest class) and the risks include a warning
        when the ratio reaches `SEVERE_IMBALANCE`. Otherwise the summary
        holds mean, std and skew, and the risks are empty.
    """
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


MIN_BALANCE_ROWS = 100
MIN_BALANCE_CLASSES = 2
# Profiling outside the leakage screen, the baselines and the label-noise
# check (ingest, statistics, missingness, fingerprint), fitted 2026-11-17 on
# the 15 evidence tables (OMP_NUM_THREADS=1, two runs at a time):
# seconds = REST_BASE + REST_ROW x rows. The largest measured to predicted
# ratio was 1.30 (1.34 s against 1.03 s on a 1132-row, 111-column table);
# REST_MARGIN = 1.5 keeps a fifth more headroom than that, because the model
# has no column term.
REST_BASE = 1.0
REST_ROW = 3e-5
REST_MARGIN = 1.5


def rest(rows: int) -> float:
    """Predicts an upper bound on the seconds of the uncapped profiling.

    Args:
        rows: Rows profiled.

    Returns:
        `REST_MARGIN x (REST_BASE + REST_ROW x rows)`.
    """
    return REST_MARGIN * (REST_BASE + REST_ROW * rows)


def estimate(rows: int, columns: int, classes: int, budget: float) -> float:
    """Predicts an upper bound on the seconds of data preparation.

    Preparation is reading and profiling the development rows: the leakage
    screen and its probe, the difficulty baselines and, for classification,
    the label-noise check. Each of the three is capped at its share of the
    budget (`screen.SCREEN_SHARE`, `fingerprint.SHARE`, `label_noise.SHARE`)
    and reduced or skipped to fit it, so its part is the smaller of its
    predicted cost and its share; the rest is `rest`. Every cost model is
    an upper bound on the measured single-thread seconds.

    Args:
        rows: Development rows to be profiled.
        columns: Feature columns.
        classes: Classes of the target; below two means regression.
        budget: The run's time budget in seconds.

    Returns:
        The predicted seconds.

    Raises:
        InputError: If a count is negative or the budget is not positive.
    """
    if min(rows, columns, classes) < 0 or budget <= 0:
        raise errors.InputError(
            "estimate needs non-negative rows, columns and classes and a "
            f"positive budget, got {rows}, {columns}, {classes}, {budget}"
        )
    total = rest(rows)
    total += min(
        screen.bound(rows, columns, classes), budget * screen.SCREEN_SHARE
    )
    total += min(
        fingerprint.cost(rows, columns, classes), budget * fingerprint.SHARE
    )
    if classes >= MIN_BALANCE_CLASSES:
        noisy = min(rows, label_noise.MAX_ROWS)
        total += min(
            label_noise.cost(noisy, columns, classes),
            budget * label_noise.SHARE,
        )
    return total


class Profiler:
    """Builds a `Portrait` for a dataset under a project spec."""

    def __init__(self, spec: tasks.Config) -> None:
        """Initializes the profiler.

        Args:
            spec: Project specification describing column roles.
        """
        self.spec = spec

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
        dict[str, dict[str, Any]],
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
        statistics: dict[str, dict[str, Any]] = {}
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

    def monotonic_features(self, numeric: list[str]) -> dict[str, int]:
        """Validates the monotonic constraints against the numeric features.

        Args:
            numeric: Names of the numeric feature columns.

        Returns:
            The monotonic constraints from the project specification, as a
            dict from column name to direction.

        Raises:
            BlockedError: If a constrained column is not a numeric feature.
        """
        wanted = self.spec.data.monotonic_constraints
        unusable = sorted(set(wanted) - set(numeric))
        if unusable:
            raise errors.BlockedError(
                f"monotonic constraints need numeric feature columns: "
                f"{unusable} are missing, excluded or not numeric"
            )
        return dict(wanted)

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

    def describe(
        self,
        frame: pandas.DataFrame,
        statistics: dict[str, dict[str, Any]],
        feature_columns: list[str],
        task: tasks.Task,
    ) -> None:
        """Records each feature's descriptive association with the target.

        These two numbers feed the dataset fingerprint only. They carry no
        cutoff and raise no finding; leakage suspicion comes from
        `leakage_screen`.

        Args:
            frame: The dataset.
            statistics: Per-column statistics; updated in place with
                `target_association` and `missingness_association`.
            feature_columns: Feature columns to describe.
            task: Resolved task type.
        """
        target = frame[self.spec.data.target]
        for name in feature_columns:
            series = frame[name]
            kind = columns.categorize(series)
            statistics[name]["target_association"] = round(
                kind.association(series, target, task), 4
            )
            statistics[name]["missingness_association"] = round(
                kind.missingness_association(series, target, task), 4
            )

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

    def profile(
        self,
        frame: pandas.DataFrame,
        manifest: scmle.contracts.data.DatasetManifest,
        task: tasks.Task,
        read_seconds: float = 0.0,
        total_rows: int | None = None,
    ) -> scmle.contracts.data.Portrait:
        """Profiles the dataset.

        A configured column that is absent from the data raises DataError
        from `exclude`.

        Args:
            frame: The rows to profile (the development partition when the
                locked test rows must not influence any statistic).
            manifest: Identity of the whole dataset.
            task: Resolved task type.
            read_seconds: Time spent reading the data (operational cost).
            total_rows: Rows in the whole dataset when `frame` is a subset.

        Returns:
            The profile including the resulting feature plan.

        Raises:
            BlockedError: If the target has missing values, no usable
                feature remains, or a monotonic constraint names a column
                that is not a numeric feature.
        """
        with threadpoolctl.threadpool_limits(
            limits=self.spec.resources.cpu_limit
        ):
            return self.portray(frame, manifest, task, read_seconds, total_rows)

    def draft(
        self,
        target: pandas.Series,
        features: dict[str, list[str]],
        statistics: Mapping[str, Mapping[str, Any]],
        dropped: dict[str, str],
    ) -> scmle.contracts.data.Blueprint:
        """Builds the feature plan before derived features are added.

        Args:
            target: The target column.
            features: Usable columns by kind (`numeric`, `categorical`,
                `datetime`, `text`).
            statistics: Column statistics by column.
            dropped: Excluded columns and the reason for each.

        Returns:
            The plan of the usable columns and the configured roles.

        Raises:
            BlockedError: If a monotonic constraint names a column that is
                not a numeric feature (`monotonic_features`).
        """
        data = self.spec.data
        return scmle.contracts.data.Blueprint(
            numeric=tuple(features["numeric"]),
            categorical=tuple(features["categorical"]),
            text=tuple(features["text"]),
            monotonic=self.monotonic_features(features["numeric"]),
            dropped=dropped,
            group_column=data.group_column,
            time_column=data.time_column,
            weight_column=data.weight_column,
            event_column=data.event_column,
            query_column=data.query_column,
            target_columns=tuple(data.targets),
            value_column=data.value_column,
            treatment_column=data.treatment_column,
            binary_outcome=bool(set(pandas.unique(target.dropna())) <= {0, 1}),
            text_encoding=data.text_encoding,
            embedding_model=data.embedding_model,
            ordinal_levels=len(data.ordinal_order) or None,
            high_cardinality=tuple(
                c
                for c in features["categorical"]
                if statistics[c]["n_unique"]
                >= scmle.contracts.data.HIGH_CARDINALITY_LEVELS
            ),
        )

    @staticmethod
    def associate(
        frame: pandas.DataFrame,
        usable: list[str],
        target: pandas.Series,
        task: tasks.Task,
    ) -> tuple[list[missingness.Finding], list[str]]:
        """Tests whether each column's missingness is tied to the target.

        Args:
            frame: The rows to profile.
            usable: Numeric and categorical feature columns.
            target: The target column.
            task: Resolved task type.

        Returns:
            Every finding, and a quality note for each informative one.
        """
        found = missingness.assess(
            frame, usable, target, task.is_classification
        )
        return found, [
            f"missingness of '{f.column}' is associated with the target "
            f"({f.test}, effect {f.effect:.2f}, p={f.p_value:.1e})"
            for f in found
            if f.informative
        ]

    @staticmethod
    def expose(report: screen.Screen | None) -> tuple[list[str], list[str]]:
        """Turns a leakage screen report into risk findings and notes.

        Args:
            report: The screen report, or None when no screen ran.

        Returns:
            The risk findings (the warning first, if any) and the quality
            notes of the screen.
        """
        if report is None:
            return [], []
        risks = list(report.sentences())
        if report.warning:
            risks.insert(0, report.warning)
        return risks, [f"leakage screen: {note}" for note in report.notes]

    def portray(
        self,
        frame: pandas.DataFrame,
        manifest: scmle.contracts.data.DatasetManifest,
        task: tasks.Task,
        read_seconds: float,
        total_rows: int | None,
    ) -> scmle.contracts.data.Portrait:
        """Profiles the dataset under the thread cap `profile` applies.

        Args:
            frame: The rows to profile.
            manifest: Identity of the whole dataset.
            task: Resolved task type.
            read_seconds: Time spent reading the data.
            total_rows: Rows in the whole dataset when `frame` is a subset.

        Returns:
            The profile, as `profile` describes.

        Raises:
            BlockedError: As `profile` describes.
        """
        target = frame[self.spec.data.target]
        if target.isna().any():
            raise errors.BlockedError(
                f"target '{target.name}' has {int(target.isna().sum())} "
                "missing values"
            )
        roles = self.exclude(frame)
        statistics, dropped, features, quality = self.inspect(frame, roles)
        usable = features["numeric"] + features["categorical"]
        if not (usable or features["datetime"] or features["text"]):
            raise errors.BlockedError(
                "no usable feature columns remain after exclusions"
            )
        stamps = tuple(features["datetime"])
        derived = tuple(self.derive(frame, stamps))
        quality += self.dupes(frame, usable, task) + self.balance(target, task)
        gaps, notes = self.associate(frame, usable, target, task)
        quality += notes
        self.describe(frame, statistics, usable, task)
        report = self.leakage_screen(
            frame, usable + features["datetime"] + features["text"], task
        )
        risks, notes = self.expose(report)
        quality += notes
        target_summary, target_risks = summarize(target, task)
        base_plan = self.draft(target, features, statistics, dropped)
        dataset_statistics = fingerprint.imprint(
            frame,
            self.spec,
            task,
            base_plan,
            statistics,
            self.duplicate_count(frame),
            read_seconds,
            len(risks),
            total_rows,
            self.spec.resources.time_budget_seconds * fingerprint.SHARE,
        )
        difficulty = dataset_statistics["ml_difficulty"]
        if difficulty.get("skipped"):
            quality.append(f"ml_difficulty skipped: {difficulty['reason']}")
        feature_plan = dataclasses.replace(
            base_plan,
            numeric=base_plan.numeric + tuple(n.name for n in derived),
            derived=derived,
            extra_inputs=tuple(c for c in stamps if derived) + base_plan.text,
            informative_missing=tuple(f.column for f in gaps if f.informative),
        )
        return scmle.contracts.data.Portrait(
            manifest=manifest,
            column_statistics=statistics,
            quality_findings=tuple(quality),
            risk_findings=tuple(risks + target_risks),
            target_summary=target_summary,
            feature_plan=feature_plan,
            dataset_statistics=dataset_statistics,
            missingness=tuple(f.to_dict() for f in gaps),
            leakage={} if report is None else report.to_dict(),
        )
