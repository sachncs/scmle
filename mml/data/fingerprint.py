"""Dataset fingerprint: the dimensions that drive decision intelligence.

Covers shape, types, missingness, target, features, relationships, data
structure, quality, operational cost and risk. Statistics preserve
uncertainty: they describe the data and raise hypotheses, never verdicts.
"""

import itertools
from typing import Any

import numpy as np
import pandas as pd

from mml.contracts import data as data_contracts
from mml.contracts import tasks
from mml.data import columns
from mml.monitoring import reference

BYTES_PER_MB = 1024 * 1024
SAMPLE_ROWS = 5000
MAX_CORRELATION_COLUMNS = 150
REDUNDANT_CORRELATION = 0.95
MAX_REDUNDANT_PAIRS = 20
THIRDS = 3


def redundant_pairs(
    frame: pd.DataFrame, numeric: tuple[str, ...]
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
    if len(names) < 2:
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


def order_drift(frame: pd.DataFrame, numeric: tuple[str, ...]) -> float:
    """Largest adjusted PSI between the first and last third of the rows.

    A high value means the data's distribution changes along row order,
    hinting at temporal shift worth a temporal validation scheme.
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


def structure(frame: pd.DataFrame, spec: tasks.ProjectSpec) -> dict[str, Any]:
    """Describes group and temporal structure."""
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
        times = pd.to_datetime(frame[data.time_column], errors="coerce")
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
    frame: pd.DataFrame,
    spec: tasks.ProjectSpec,
    task: tasks.TaskType,
    plan: data_contracts.FeaturePlan,
    column_statistics: dict[str, dict[str, Any]],
    duplicate_rows: int,
    read_seconds: float,
    leakage_flags: int,
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
            "skew": float(np.nan_to_num(values.agg("skew"))),
            "outlier_rate": columns.outlier_rate(values),
        }
    return {
        "shape": {
            "rows": len(frame),
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
        },
        "target": target_block,
        "features": {
            "max_cardinality": max(
                (column_statistics[c]["n_unique"] for c in plan.categorical),
                default=0,
            ),
            "mean_outlier_rate": float(
                np.mean(
                    [
                        column_statistics[c].get("outlier_rate", 0.0)
                        for c in plan.numeric
                    ]
                )
            )
            if plan.numeric
            else 0.0,
            "mean_categorical_entropy_bits": float(
                np.mean(
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
