"""Dataset ingestion (CSV / Parquet) and manifest construction."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from mml.contracts.data import DatasetManifest
from mml.contracts.errors import DataError

_READERS = {
    ".csv": pd.read_csv,
    ".parquet": pd.read_parquet,
    ".pq": pd.read_parquet,
}


def read_table(uri: str | Path) -> pd.DataFrame:
    """Read a CSV/Parquet file. Raises DataError with a precise reason on failure."""
    path = Path(uri)
    if not path.exists():
        raise DataError(f"dataset not found: {path}")
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        raise DataError(f"unsupported file type '{path.suffix}'; use one of {sorted(_READERS)}")
    try:
        df = reader(path)
    except Exception as exc:  # parser errors vary by engine
        raise DataError(f"failed to read {path.name}: {exc}") from exc
    if df.empty:
        raise DataError(f"{path.name} contains no rows")
    if df.columns.duplicated().any():
        dupes = sorted(set(df.columns[df.columns.duplicated()].astype(str)))
        raise DataError(f"duplicate column names: {dupes}")
    return df


def schema_hash(df: pd.DataFrame) -> str:
    payload = "|".join(f"{c}:{t}" for c, t in zip(df.columns, df.dtypes.astype(str), strict=True))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def data_hash(df: pd.DataFrame) -> str:
    row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()[:16]


def build_manifest(df: pd.DataFrame, target: str, uri: str = "") -> DatasetManifest:
    if target not in df.columns:
        raise DataError(f"target column '{target}' not found; columns: {list(df.columns)[:20]}")
    s, d = schema_hash(df), data_hash(df)
    return DatasetManifest(
        dataset_id=f"ds_{d[:8]}",
        schema_hash=s,
        data_hash=d,
        row_count=len(df),
        column_count=df.shape[1],
        target=target,
        uri=str(uri),
    )
