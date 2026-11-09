"""Dataset fingerprint: the dimensions that drive decision intelligence.

Covers shape, types, missingness, target, features, relationships, data
structure, quality, operational cost and risk. Statistics preserve
uncertainty: they describe the data and raise hypotheses, never verdicts.
"""

import itertools
import time
from typing import Any

import numpy
import pandas
from sklearn import dummy
from sklearn import ensemble
from sklearn import linear_model
from sklearn import model_selection
from sklearn import pipeline

from scmle.contracts import tasks
import scmle.contracts.data
from scmle.data.curation import preprocessing
from scmle.data.validation import columns
from scmle.monitoring import reference

BYTES_PER_MB = 1024 * 1024
SAMPLE_ROWS = 5000
MAX_CORRELATION_COLUMNS = 150
MIN_CORRELATION_COLUMNS = 2
MIN_SPREADS = 2
REDUNDANT_CORRELATION = 0.95
MAX_REDUNDANT_PAIRS = 20
THIRDS = 3
CO_MISSING = 0.8
TEXT_MEAN_LENGTH = 40
DIFFICULTY_ROWS = 2000
DIFFICULTY_FOLDS = 3


def redundant_pairs(
    frame: pandas.DataFrame, numeric: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Finds strongly correlated numeric feature pairs.

    Args:
        frame: The dataset.
        numeric: Numeric feature columns.

    Returns:
        Up to `MAX_REDUNDANT_PAIRS` pairs with |Pearson r| above the
        threshold, strongest first.
    """
    names = list(numeric)[:MAX_CORRELATION_COLUMNS]
    if len(names) < MIN_CORRELATION_COLUMNS:
        return []
    sample = frame[names].astype(float)
    if len(sample) > SAMPLE_ROWS:
        sample = sample.sample(SAMPLE_ROWS, random_state=0)
    matrix = sample.corr().abs().to_numpy()
    pairs: list[dict[str, Any]] = [
        {
            "a": names[i],
            "b": names[j],
            "correlation": float(matrix[i, j]),
        }
        for i, j in itertools.combinations(range(len(names)), 2)
        if matrix[i, j] >= REDUNDANT_CORRELATION
    ]
    pairs.sort(key=lambda pair: float(pair["correlation"]), reverse=True)
    return pairs[:MAX_REDUNDANT_PAIRS]


def order_drift(frame: pandas.DataFrame, numeric: tuple[str, ...]) -> float:
    """Measures drift along row order.

    Compares the first and last third of the rows. A high value means the data's
    distribution changes along row order, hinting at temporal shift worth a
    temporal validation scheme.

    Args:
        frame: The dataset.
        numeric: Numeric feature columns.

    Returns:
        The largest adjusted PSI over the numeric columns; 0.0 if there are
        fewer than 90 rows or no numeric columns.
    """
    if len(frame) < 3 * 30 or not numeric:
        return 0.0
    size = len(frame) // THIRDS
    head, tail = frame.iloc[:size], frame.iloc[-size:]
    worst = 0.0
    for column in numeric:
        ref = reference.NumericReference.fit(head[column])
        worst = max(worst, ref.adjusted_psi(tail[column]))
    return float(worst)


def missingness_patterns(features: pandas.DataFrame) -> dict[str, Any]:
    """Summarizes how values go missing together.

    Args:
        features: The feature columns.

    Returns:
        The number of distinct row-level missingness patterns, the share of the
        most common non-empty pattern, and how many column pairs go missing
        together (indicator correlation above 0.8).
    """
    missing = features.isna()
    holes = missing.loc[:, missing.any()]
    if holes.empty:
        return {"pattern_count": 1, "top_pattern_share": 0.0, "co_missing": 0}
    keys = holes.astype(int).astype(str).agg("".join, axis=1)
    counts = keys.value_counts()
    silent = "0" * holes.shape[1]
    partial = counts[counts.index != silent]
    pairs = 0
    if holes.shape[1] > 1:
        corr = holes.astype(float).corr().abs().to_numpy()
        pairs = int(
            ((corr > CO_MISSING) & ~numpy.eye(len(corr), dtype=bool)).sum() // 2
        )
    return {
        "pattern_count": len(counts),
        "top_pattern_share": float(partial.iloc[0] / len(keys))
        if len(partial)
        else 0.0,
        "co_missing": pairs,
    }


def text_like_columns(
    frame: pandas.DataFrame, plan: scmle.contracts.data.FeaturePlan
) -> list[str]:
    """Finds categorical columns holding free text rather than short labels.

    Args:
        frame: The dataset.
        plan: The feature plan, naming categorical, text and dropped columns.

    Returns:
        The non-numeric columns among those whose mean string length exceeds
        `TEXT_MEAN_LENGTH`.
    """
    found = []
    for name in (*plan.categorical, *plan.text, *plan.dropped):
        if name not in frame.columns or pandas.api.types.is_numeric_dtype(
            frame[name]
        ):
            continue
        lengths = frame[name].dropna().astype(str).str.len()
        if len(lengths) and lengths.mean() > TEXT_MEAN_LENGTH:
            found.append(name)
    return found


def scale_range(frame: pandas.DataFrame, numeric: tuple[str, ...]) -> float:
    """Measures the orders of magnitude between the widest and narrowest column.

    Args:
        frame: The dataset.
        numeric: Numeric feature columns.

    Returns:
        The base-10 logarithm of the ratio of the largest to the smallest
        positive standard deviation; 0.0 with fewer than two such columns.
    """
    spreads = [
        float(frame[c].astype(float).std())
        for c in numeric
        if frame[c].astype(float).std() > 0
    ]
    if len(spreads) < MIN_SPREADS:
        return 0.0
    return float(numpy.log10(max(spreads) / min(spreads)))


def ml_difficulty(
    frame: pandas.DataFrame,
    spec: tasks.Config,
    task: tasks.Task,
    plan: scmle.contracts.data.FeaturePlan,
) -> dict[str, Any]:
    """Runs cheap baselines that show how hard the problem looks.

    A few-second cross-validation on a subsample compares a constant predictor,
    a linear model and a boosted-tree model, so the planner and the user can see
    the headroom: baseline performance, the gain from non-linearity, and (for
    classification) how separable the classes are. Baselines are hypotheses
    about difficulty, not model selection.

    Args:
        frame: The dataset.
        spec: Project specification.
        task: Resolved task type.
        plan: Feature plan.

    Returns:
        A mapping with `available` False and a `reason` if a baseline cannot be
        scored. Otherwise `available` True, the scoring metric, the rows used,
        the three baseline scores, the nonlinearity gain, the headroom over the
        constant baseline, the separability (the linear score for
        classification, else None) and the seconds taken.
    """
    sample = frame
    if len(sample) > DIFFICULTY_ROWS:
        sample = sample.sample(DIFFICULTY_ROWS, random_state=0)
    features = sample[list(plan.features)]
    target = sample[spec.data.target]
    if task.is_classification:
        scoring = (
            "roc_auc" if task is tasks.Task.BINARY else ("balanced_accuracy")
        )
        estimators = {
            "constant": dummy.DummyClassifier(strategy="prior"),
            "linear": linear_model.LogisticRegression(max_iter=300),
            "nonlinear": ensemble.HistGradientBoostingClassifier(
                max_iter=60, random_state=0
            ),
        }
    else:
        scoring = "r2"
        estimators = {
            "constant": dummy.DummyRegressor(),
            "linear": linear_model.Ridge(),
            "nonlinear": ensemble.HistGradientBoostingRegressor(
                max_iter=60, random_state=0
            ),
        }
    options = scmle.contracts.data.Preprocessing(scale=True)
    scores: dict[str, float] = {}
    started = time.monotonic()
    try:
        for name, estimator in estimators.items():
            model = pipeline.Pipeline(
                [
                    ("prep", preprocessing.build_preprocessor(plan, options)),
                    ("model", estimator),
                ]
            )
            result = model_selection.cross_val_score(
                model, features, target, cv=DIFFICULTY_FOLDS, scoring=scoring
            )
            finite = numpy.asarray(result, dtype=float)
            finite = finite[numpy.isfinite(finite)]
            if not finite.size:
                return {
                    "available": False,
                    "reason": f"{name} baseline produced no finite scores",
                }
            scores[name] = float(numpy.mean(finite))
    except ValueError as error:
        return {"available": False, "reason": str(error)[:200]}
    gain = scores["nonlinear"] - scores["linear"]
    return {
        "available": True,
        "metric": scoring,
        "rows": len(sample),
        "constant_baseline": scores["constant"],
        "linear_baseline": scores["linear"],
        "nonlinear_baseline": scores["nonlinear"],
        "nonlinearity_gain": gain,
        "headroom_over_constant": scores["nonlinear"] - scores["constant"],
        "separability": scores["linear"] if task.is_classification else None,
        "seconds": time.monotonic() - started,
    }


def structure(frame: pandas.DataFrame, spec: tasks.Config) -> dict[str, Any]:
    """Describes group and temporal structure.

    Args:
        frame: The dataset.
        spec: Project specification, naming the group and time columns.

    Returns:
        A mapping with `groups` (group count, mean rows per group, share of
        repeated entities) and `time` (range, unparseable count and whether the
        rows are in time order) for the columns that are configured.
    """
    data = spec.data
    result: dict[str, Any] = {}
    if data.group_column:
        counts = frame[data.group_column].value_counts()
        result["groups"] = {
            "column": data.group_column,
            "n_groups": len(counts),
            "mean_rows_per_group": float(counts.mean()),
            "repeated_entity_share": float((counts > 1).mean()),
        }
    if data.time_column:
        times = pandas.to_datetime(frame[data.time_column], errors="coerce")
        valid = times.dropna()
        result["time"] = {
            "column": data.time_column,
            "unparseable": int(times.isna().sum()),
            "start": str(valid.min()) if len(valid) else None,
            "end": str(valid.max()) if len(valid) else None,
            "monotonic": bool(valid.is_monotonic_increasing),
        }
    return result


def build_fingerprint(
    frame: pandas.DataFrame,
    spec: tasks.Config,
    task: tasks.Task,
    plan: scmle.contracts.data.FeaturePlan,
    column_statistics: dict[str, dict[str, Any]],
    duplicate_rows: int,
    read_seconds: float,
    leakage_flags: int,
    total_rows: int | None = None,
) -> dict[str, Any]:
    """Assembles the dataset-level fingerprint.

    Args:
        frame: The dataset.
        spec: Project specification.
        task: Resolved task type.
        plan: Feature plan.
        column_statistics: Per-column statistics already computed.
        duplicate_rows: Number of duplicate rows.
        read_seconds: Time spent reading the data.
        leakage_flags: Number of leakage hypotheses raised.
        total_rows: Rows in the whole dataset when `frame` is only the
            development partition.

    Returns:
        A JSON-serializable mapping grouped by fingerprint dimension.
    """
    features = frame[list(plan.features)]
    target = frame[spec.data.target]
    numeric = (
        features[list(plan.numeric)] if plan.numeric else features.iloc[:, :0]
    )
    zero_share = (
        float((numeric.fillna(1) == 0).to_numpy().mean())
        if numeric.size
        else 0.0
    )
    missing = features.isna()
    kinds = {
        kind: sum(1 for s in column_statistics.values() if s["kind"] == kind)
        for kind in ("numeric", "categorical", "datetime")
    }
    kinds["text_like"] = len(text_like_columns(frame, plan))
    kinds["id_like"] = sum(
        1 for reason in plan.dropped.values() if reason.startswith("id-like")
    ) + len(spec.data.id_columns)
    target_block: dict[str, Any] = {"cardinality": int(target.nunique())}
    if task.is_classification:
        counts = target.value_counts()
        target_block |= {
            "entropy_bits": columns.entropy(target),
            "min_class_count": int(counts.min()),
            "imbalance_ratio": float(counts.iloc[0] / counts.iloc[-1]),
        }
    else:
        values = target.astype(float)
        target_block |= {
            "skew": float(numpy.nan_to_num(values.agg("skew"))),
            "outlier_rate": columns.outlier_rate(values),
        }
    return {
        "shape": {
            "rows": len(frame),
            "total_rows": total_rows if total_rows is not None else len(frame),
            "columns": frame.shape[1],
            "effective_samples": len(frame) - duplicate_rows,
            "sparsity": zero_share,
        },
        "types": kinds,
        "missingness": {
            "overall_rate": float(missing.to_numpy().mean())
            if missing.size
            else 0.0,
            "columns_with_missing": int(missing.any().sum()),
            "rows_with_any_missing": float(missing.any(axis=1).mean())
            if missing.size
            else 0.0,
            "patterns": missingness_patterns(features),
        },
        "target": target_block,
        "features": {
            "max_cardinality": max(
                (column_statistics[c]["n_unique"] for c in plan.categorical),
                default=0,
            ),
            "mean_outlier_rate": float(
                numpy.mean(
                    [
                        column_statistics[c].get("outlier_rate", 0.0)
                        for c in plan.numeric
                    ]
                )
            )
            if plan.numeric
            else 0.0,
            "scale_orders_of_magnitude": scale_range(frame, plan.numeric),
            "mean_categorical_entropy_bits": float(
                numpy.mean(
                    [
                        column_statistics[c]["entropy_bits"]
                        for c in plan.categorical
                    ]
                )
            )
            if plan.categorical
            else 0.0,
        },
        "relationships": {
            "redundant_numeric_pairs": redundant_pairs(frame, plan.numeric),
            "max_target_association": max(
                (
                    s.get("target_association", 0.0)
                    for s in column_statistics.values()
                ),
                default=0.0,
            ),
        },
        "ml_difficulty": ml_difficulty(frame, spec, task, plan),
        "structure": structure(frame, spec),
        "quality": {
            "duplicate_rows": duplicate_rows,
            "duplicate_share": duplicate_rows / max(len(frame), 1),
        },
        "operational": {
            "memory_mb": float(
                frame.memory_usage(deep=True).sum() / BYTES_PER_MB
            ),
            "read_seconds": read_seconds,
            "feature_count": len(plan.features),
        },
        "risk": {
            "leakage_hypotheses": leakage_flags,
            "order_drift_psi": order_drift(frame, plan.numeric),
            "rare_classes": task.is_classification
            and target_block["min_class_count"] < spec.validation.folds * 2,
        },
    }
