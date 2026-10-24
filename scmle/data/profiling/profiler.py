"""Dataset profiling: task resolution, feature typing, findings.

Profiling never fits model state. Risk findings are hypotheses, not verdicts:
a feature strongly associated with the target may be a good predictor or a
leak, and statistics alone cannot tell which.
"""

import dataclasses

import numpy as np
import pandas as pd

from scmle.contracts import data as data_contracts
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.profiling import missingness
from scmle.data.readers import streaming
from scmle.data.validation import columns
from scmle.intelligence import fingerprint

LEAK_ASSOCIATION = 0.98
MISSINGNESS_LEAK = 0.9
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


MIN_BALANCE_ROWS = 100
MIN_BALANCE_CLASSES = 2


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
        if data.treatment_column:
            roles[data.treatment_column] = (
                "treatment indicator (used by uplift models only)"
            )
        if data.query_column:
            roles[data.query_column] = "query identifier (ranking only)"
        if data.event_column:
            roles[data.event_column] = (
                "event indicator (used by survival models only)"
            )
        if data.weight_column:
            roles[data.weight_column] = "sample weight (used for training only)"
        for column in roles:
            if column not in frame.columns:
                raise errors.DataError(
                    f"configured column '{column}' not found in dataset"
                )
        if data.weight_column:
            self.check_weights(frame[data.weight_column])
        return roles

    def check_weights(self, weights: pd.Series) -> None:
        """Requires finite, non-negative weights with a positive total.

        Raises:
            DataError: If the weight column cannot weight training rows.
        """
        numeric = pd.to_numeric(weights, errors="coerce").to_numpy(float)
        name = self.spec.data.weight_column
        if not np.isfinite(numeric).all():
            raise errors.DataError(
                f"weight column '{name}' has missing or non-numeric values"
            )
        if (numeric < 0).any() or numeric.sum() <= 0:
            raise errors.DataError(
                f"weight column '{name}' must be non-negative with a "
                "positive total"
            )

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

    def monotonic_features(self, numeric: list[str]) -> dict[str, int]:
        """Validates the monotonic constraints against the numeric features.

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
    def datetime_nodes(
        frame: pd.DataFrame, stamps: tuple[str, ...]
    ) -> list[tasks.FeatureNode]:
        """Calendar and trend features for each datetime column.

        Only parts that vary in the data are produced: a column holding one
        year yields no year feature. The elapsed-days part carries the trend.
        """
        nodes = []
        for column in stamps:
            values = pd.to_datetime(frame[column], errors="coerce")
            for part in DATETIME_PARTS:
                if part == "timestamp":
                    varies = values.nunique() > 1
                else:
                    varies = getattr(values.dt, part).nunique() > 1
                if varies:
                    nodes.append(
                        tasks.FeatureNode(
                            name=f"{column}__{part}",
                            op="datepart",
                            inputs=[column],
                            params={"part": part},
                        )
                    )
        return nodes

    def duplicate_count(self, frame: pd.DataFrame) -> int:
        """Counts rows duplicated apart from their identifier columns."""
        data = self.spec.data
        deduplicated = frame.drop(columns=list(data.id_columns))
        return int(deduplicated.duplicated().sum())

    @staticmethod
    def balance_findings(target: pd.Series, task: tasks.TaskType) -> list[str]:
        """Warns when classes are exactly equal in size.

        Real data are rarely perfectly balanced, so equal counts usually mean
        the table was down-sampled. Scores on it then describe a world with
        equal prevalence, not the one the model will serve.
        """
        if not task.is_classification or len(target) < MIN_BALANCE_ROWS:
            return []
        counts = target.value_counts()
        if len(counts) >= MIN_BALANCE_CLASSES and counts.nunique() == 1:
            return [
                f"every class has exactly {int(counts.iloc[0])} rows: if the "
                "table was balanced by sampling, test scores reflect equal "
                "prevalence, not the real-world mix"
            ]
        return []

    def duplicate_findings(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
        task: tasks.TaskType,
    ) -> list[str]:
        """Reports duplicate rows and, for classification, label conflicts."""
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
        read_seconds: float = 0.0,
        total_rows: int | None = None,
    ) -> data_contracts.DatasetProfile:
        """Profiles the dataset.

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
        if not (usable or features["datetime"] or features["text"]):
            raise errors.BlockedError(
                "no usable feature columns remain after exclusions"
            )
        stamps = tuple(features["datetime"])
        derived = tuple(self.datetime_nodes(frame, stamps))
        quality += self.duplicate_findings(frame, usable, task)
        quality += self.balance_findings(target, task)
        gaps = missingness.assess(frame, usable, target, task.is_classification)
        informative = tuple(f.column for f in gaps if f.informative)
        quality += [
            f"missingness of '{f.column}' is associated with the target "
            f"({f.test}, effect {f.effect:.2f}, p={f.p_value:.1e})"
            for f in gaps
            if f.informative
        ]
        risks = self.leakage_findings(frame, statistics, usable, task)
        target_summary, target_risks = summarize_target(target, task)
        base_plan = data_contracts.FeaturePlan(
            numeric=tuple(features["numeric"]),
            categorical=tuple(features["categorical"]),
            text=tuple(features["text"]),
            monotonic=self.monotonic_features(features["numeric"]),
            dropped=dropped,
            group_column=self.spec.data.group_column,
            time_column=self.spec.data.time_column,
            weight_column=self.spec.data.weight_column,
            event_column=self.spec.data.event_column,
            query_column=self.spec.data.query_column,
            target_columns=tuple(self.spec.data.targets),
            value_column=self.spec.data.value_column,
            treatment_column=self.spec.data.treatment_column,
            binary_outcome=bool(set(pd.unique(target.dropna())) <= {0, 1}),
            text_encoding=self.spec.data.text_encoding,
            embedding_model=self.spec.data.embedding_model,
            ordinal_levels=len(self.spec.data.ordinal_order) or None,
            high_cardinality=tuple(
                c
                for c in features["categorical"]
                if statistics[c]["n_unique"]
                >= data_contracts.HIGH_CARDINALITY_LEVELS
            ),
        )
        dataset_statistics = fingerprint.build_fingerprint(
            frame,
            self.spec,
            task,
            base_plan,
            statistics,
            self.duplicate_count(frame),
            read_seconds,
            len(risks),
            total_rows,
        )
        feature_plan = dataclasses.replace(
            base_plan,
            numeric=base_plan.numeric + tuple(n.name for n in derived),
            derived=derived,
            extra_inputs=tuple(c for c in stamps if derived) + base_plan.text,
            informative_missing=informative,
        )
        return data_contracts.DatasetProfile(
            manifest=manifest,
            column_statistics=statistics,
            quality_findings=tuple(quality),
            risk_findings=tuple(risks + target_risks),
            target_summary=target_summary,
            feature_plan=feature_plan,
            dataset_statistics=dataset_statistics,
            missingness=tuple(f.to_dict() for f in gaps),
        )
