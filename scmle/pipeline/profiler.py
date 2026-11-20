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

import pandas
import threadpoolctl

from scmle.assurance.admissibility import screen
from scmle.contracts import errors
from scmle.contracts import tasks
import scmle.contracts.data
from scmle.data.profiling import label_noise
from scmle.data.profiling import missingness
from scmle.data.validation import columns
from scmle.pipeline import fingerprint
from scmle.pipeline import inspection

SEVERE_IMBALANCE = 10.0


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
    if not task.classification:
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


# Preparation outside the leakage screen, the baselines and the label-noise
# check (split, ingest, statistics, the per-column target association of
# `describe`, missingness, custody), measured 2026-11-19 as the seconds to
# the first trial less those three parts, on 30 runs of 24 OpenML tables
# (800-3,200 rows, 5-970 columns, budgets 30 and 60 s, OMP_NUM_THREADS=1,
# two runs at a time; `scratchpad/c14/prep.py`): seconds = REST_BASE +
# REST_ROW x rows + REST_CELL x rows x columns. On wide tables the cell term
# dominates: `describe` fits one small tree per numeric column (3.0 s of
# gina's 4.7 s). Least squares gives 0.85, 2.3e-5 and 1.6e-6; the stated
# coefficients round them to 1.0, 3e-5 and 1.5e-6, under which the largest
# measured to predicted ratio is 1.13 (philippine, 3,200 x 308), so
# REST_MARGIN = 1.5 keeps a third more headroom than any measured table.
REST_BASE = 1.0
REST_ROW = 3e-5
REST_CELL = 1.5e-6
REST_MARGIN = 1.5


def rest(rows: int, columns: int = 0) -> float:
    """Predicts an upper bound on the seconds of the uncapped profiling.

    Args:
        rows: Rows profiled.
        columns: Feature columns.

    Returns:
        `REST_MARGIN x (REST_BASE + REST_ROW x rows + REST_CELL x rows x
        columns)`.
    """
    return REST_MARGIN * (
        REST_BASE + REST_ROW * rows + REST_CELL * rows * columns
    )


def noise(rows: int, columns: int, classes: int, budget: float) -> float:
    """Predicts the seconds of the label-noise check as it will run.

    The check samples the rows its allowance (`label_noise.SHARE` of the
    budget) admits by `label_noise.cost` and skips below
    `label_noise.MIN_ROWS`; those rows are priced by that same cost model,
    the single one the admission and this estimate share.

    Args:
        rows: Development rows.
        columns: Feature columns.
        classes: Classes of the target.
        budget: The run's time budget in seconds.

    Returns:
        `label_noise.cost` of the rows checked, or 0.0 when the check will
        be skipped.
    """
    admitted = label_noise.affordable(
        columns, classes, budget * label_noise.SHARE
    )
    if admitted < label_noise.MIN_ROWS:
        return 0.0
    checked = min(rows, label_noise.MAX_ROWS, admitted)
    return label_noise.cost(checked, columns, classes)


def estimate(rows: int, columns: int, classes: int, budget: float) -> float:
    """Predicts an upper bound on the seconds of data preparation.

    Preparation is reading and profiling the development rows: the leakage
    screen and its probe, the difficulty baselines and, for classification,
    the label-noise check. The screen and the baselines are capped at their
    share of the budget (`screen.SCREEN_SHARE`, `fingerprint.SHARE`), so
    their part is the smaller of their predicted cost and their share; the
    label-noise check is priced as it will run (`noise`); the rest, which
    grows with rows x columns, is `rest`.

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
    total = rest(rows, columns)
    total += min(
        screen.bound(rows, columns, classes), budget * screen.SCREEN_SHARE
    )
    total += min(
        fingerprint.cost(rows, columns, classes), budget * fingerprint.SHARE
    )
    if classes >= inspection.MIN_BALANCE_CLASSES:
        total += noise(rows, columns, classes, budget)
    return total


class Profiler(inspection.Inspection):
    """Builds a `Portrait` for a dataset under a project spec.

    The column checks are `inspection.Inspection`'s; the assembly of the
    portrait, its feature plan and the monotonic constraints are here.
    """

    def __init__(self, spec: tasks.Config, texts: tuple[str, ...] = ()) -> None:
        """Initializes the profiler.

        Args:
            spec: Project specification describing column roles.
            texts: Columns a data kind declares free text: they are
                profiled as text however few rows there are, so a short
                corpus keeps its text instead of dropping it as an id.
        """
        self.spec = spec
        self.texts = texts

    def constrain(self, numeric: list[str]) -> dict[str, int]:
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
                kind.discriminate(series, target, task), 4
            )

    def profile(
        self,
        frame: pandas.DataFrame,
        manifest: scmle.contracts.data.Manifest,
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
                not a numeric feature (`constrain`).
        """
        data = self.spec.data
        return scmle.contracts.data.Blueprint(
            numeric=tuple(features["numeric"]),
            categorical=tuple(features["categorical"]),
            text=tuple(features["text"]),
            monotonic=self.constrain(features["numeric"]),
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
            cardinal=tuple(
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
        found = missingness.assess(frame, usable, target, task.classification)
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
        manifest: scmle.contracts.data.Manifest,
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
        # ISO date text is profiled, screened and expanded as the dates it
        # is, so the screen's `cutoff` reading sees it.
        frame = columns.temporal(frame)
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
            self.count(frame),
            read_seconds,
            len(risks),
            total_rows,
            self.spec.resources.time_budget_seconds * fingerprint.SHARE,
        )
        difficulty = dataset_statistics["ml_difficulty"]
        if difficulty.get("skipped"):
            quality.append(f"ml_difficulty skipped: {difficulty['reason']}")
        feature_plan = extend(base_plan, derived, stamps, gaps)
        return scmle.contracts.data.Portrait(
            manifest=manifest,
            statistics=statistics,
            quality=tuple(quality),
            risk_findings=tuple(risks + target_risks),
            summary=target_summary,
            plan=feature_plan,
            overview=dataset_statistics,
            missingness=tuple(f.to_dict() for f in gaps),
            leakage={} if report is None else report.to_dict(),
        )


def extend(
    plan: scmle.contracts.data.Blueprint,
    derived: tuple[tasks.Node, ...],
    stamps: tuple[str, ...],
    gaps: Any,
) -> scmle.contracts.data.Blueprint:
    """Adds the derived calendar features and informative gaps to a plan.

    Args:
        plan: The drafted feature plan.
        derived: The calendar and trend features.
        stamps: The datetime columns they derive from.
        gaps: The missingness findings.

    Returns:
        The plan with the derived features as numeric inputs, their source
        columns as extra inputs, and the informatively missing columns.
    """
    return dataclasses.replace(
        plan,
        numeric=plan.numeric + tuple(n.name for n in derived),
        derived=derived,
        extras=tuple(c for c in stamps if derived) + plan.text,
        informative_missing=tuple(f.column for f in gaps if f.informative),
    )
