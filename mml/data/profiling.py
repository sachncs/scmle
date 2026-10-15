"""Dataset profiling: task resolution, feature typing, quality and leakage-risk findings.

Profiling never fits model state. Risk findings are hypotheses, not verdicts: a feature
strongly associated with the target may be a good predictor or a leak.
"""

from __future__ import annotations

import pandas as pd

from mml.contracts.data import DatasetManifest, DatasetProfile, FeaturePlan
from mml.contracts.errors import BlockedError, DataError
from mml.contracts.tasks import ProjectSpec, TaskType

ID_UNIQUE_RATIO = 0.95
LEAK_ASSOC = 0.98


def resolve_task(spec: ProjectSpec, y: pd.Series) -> TaskType:
    """Resolve the requested task against the actual target; BLOCK when ambiguous."""
    if y.isna().all():
        raise BlockedError(f"target '{y.name}' has no non-missing values")
    req = spec.task
    n_classes = y.nunique(dropna=True)
    numeric = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
    if req == "regression":
        if not numeric:
            raise BlockedError("regression requested but the target is not numeric")
        return TaskType.REGRESSION
    if req in ("binary_classification", "multiclass_classification", "classification"):
        if n_classes < 2:
            raise BlockedError(f"classification needs >=2 classes; target has {n_classes}")
        binary = n_classes == 2
        if req == "binary_classification" and not binary:
            raise BlockedError(
                f"binary_classification requested but target has {n_classes} classes"
            )
        if req == "multiclass_classification" and binary:
            raise BlockedError("multiclass_classification requested but target has 2 classes")
        return TaskType.BINARY if binary else TaskType.MULTICLASS
    # auto
    if not numeric or pd.api.types.is_bool_dtype(y):
        return TaskType.BINARY if n_classes == 2 else TaskType.MULTICLASS
    is_int = pd.api.types.is_integer_dtype(y) or bool((y.dropna() % 1 == 0).all())
    if n_classes == 2:
        return TaskType.BINARY
    if is_int and n_classes <= 10 and n_classes / max(len(y), 1) < 0.05:
        raise BlockedError(
            f"target '{y.name}' looks like a small-cardinality integer ({n_classes} values): "
            "classification vs regression is ambiguous; set task= explicitly"
        )
    return TaskType.REGRESSION


def _kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    return "categorical"


def _association(x: pd.Series, y: pd.Series, task: TaskType) -> float:
    """Single-feature association with the target in [0, 1] (cheap leakage heuristic)."""
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if len(x) < 10 or x.nunique() < 2:
        return 0.0
    kind = _kind(x)
    if kind == "numeric" and not task.is_classification:
        return float(abs(x.astype(float).corr(y.astype(float), method="spearman")) or 0.0)
    if task.is_classification:
        # purity: share of rows whose value maps to one class (category / binned numeric)
        if kind == "categorical" or x.nunique() <= 20:
            key = x
        else:
            key = pd.qcut(x.astype(float), 20, duplicates="drop")
        if key.nunique() > 0.5 * len(x):
            return 0.0  # near-unique keys trivially "pure"; handled as id-like elsewhere
        purity = y.groupby(key, observed=True).agg(lambda v: v.value_counts().iloc[0]).sum()
        base = y.value_counts().iloc[0]
        return float((purity - base) / max(len(y) - base, 1))
    return 0.0


def profile_dataset(
    df: pd.DataFrame, spec: ProjectSpec, manifest: DatasetManifest, task: TaskType
) -> DatasetProfile:
    d = spec.data
    y = df[d.target]
    if y.isna().any():
        raise BlockedError(f"target '{d.target}' has {int(y.isna().sum())} missing values")

    structural = {d.target, *d.id_columns, *d.excluded_columns}
    structural |= {c for c in (d.group_column, d.time_column) if c}
    for c in structural - {d.target}:
        if c not in df.columns:
            raise DataError(f"configured column '{c}' not found in dataset")

    dropped: dict[str, str] = {}
    for c in d.id_columns:
        dropped[c] = "id column"
    for c in d.excluded_columns:
        dropped[c] = "excluded by user"
    if d.group_column:
        dropped[d.group_column] = "group column (used for splitting only)"
    if d.time_column:
        dropped[d.time_column] = "time column (used for splitting only)"

    stats: dict[str, dict] = {}
    quality: list[str] = []
    risk: list[str] = []
    numeric: list[str] = []
    categorical: list[str] = []
    n = len(df)

    for c in df.columns:
        if c == d.target:
            continue
        s = df[c]
        kind = _kind(s)
        info: dict = {
            "kind": kind,
            "dtype": str(s.dtype),
            "missing_rate": float(s.isna().mean()),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if kind == "numeric" and not pd.api.types.is_bool_dtype(s):
            f = s.astype(float)
            info |= {
                "mean": float(f.mean()),
                "std": float(f.std()),
                "min": float(f.min()),
                "max": float(f.max()),
            }
        elif kind == "categorical":
            info["top"] = {str(k): int(v) for k, v in s.value_counts().head(5).items()}
        stats[c] = info
        if c in dropped:
            continue
        if info["n_unique"] <= 1:
            dropped[c] = "constant"
            quality.append(f"column '{c}' is constant; dropped")
            continue
        if info["missing_rate"] > 0.95:
            dropped[c] = "missing >95%"
            quality.append(f"column '{c}' is >95% missing; dropped")
            continue
        if kind == "datetime":
            dropped[c] = "datetime (unsupported in v0.1; set as time_column or engineer features)"
            quality.append(f"datetime column '{c}' excluded from features")
            continue
        if kind == "categorical" and info["n_unique"] / n > ID_UNIQUE_RATIO:
            dropped[c] = "id-like (near-unique text)"
            quality.append(f"column '{c}' looks like an identifier; dropped")
            continue
        (numeric if kind == "numeric" else categorical).append(c)
        if info["missing_rate"] > 0.3:
            quality.append(f"column '{c}' is {info['missing_rate']:.0%} missing")

    dup = int(df.drop(columns=[c for c in df.columns if c in d.id_columns]).duplicated().sum())
    if dup:
        quality.append(f"{dup} duplicate rows")
        feats = numeric + categorical
        if feats:
            conflicts = int(
                (df.groupby(feats, dropna=False, observed=True)[d.target].nunique() > 1).sum()
            )
            if conflicts and task.is_classification:
                quality.append(f"{conflicts} feature combinations have inconsistent labels")

    for c in numeric + categorical:
        a = _association(df[c], y, task)
        stats[c]["target_association"] = round(a, 4)
        if a >= LEAK_ASSOC:
            risk.append(
                f"feature '{c}' is almost perfectly associated with the target ({a:.3f}): "
                "possible leakage (or an exceptionally strong predictor); review before trusting"
            )

    if task.is_classification:
        counts = y.value_counts()
        target_summary = {
            "task": task.value,
            "n_classes": int(len(counts)),
            "class_counts": {str(k): int(v) for k, v in counts.items()},
            "imbalance_ratio": float(counts.iloc[0] / counts.iloc[-1]),
        }
        if target_summary["imbalance_ratio"] >= 10:
            risk.append(f"severe class imbalance (ratio {target_summary['imbalance_ratio']:.1f}:1)")
    else:
        target_summary = {
            "task": task.value,
            "mean": float(y.mean()),
            "std": float(y.std()),
            "skew": float(y.skew()),
        }
    if not (numeric or categorical):
        raise BlockedError("no usable feature columns remain after exclusions")

    return DatasetProfile(
        manifest=manifest,
        column_statistics=stats,
        quality_findings=tuple(quality),
        risk_findings=tuple(risk),
        target_summary=target_summary,
        feature_plan=FeaturePlan(tuple(numeric), tuple(categorical), dropped),
    )
