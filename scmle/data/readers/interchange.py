"""Tabular interchange: pandas, Arrow and Polars in, Arrow out.

Internally the framework computes on pandas frames, but the public surface is
Arrow-compatible: any object implementing the Arrow PyCapsule stream protocol
(`pyarrow.Table`, `polars.DataFrame`, ...) or exposing `to_pandas()` is
accepted wherever data is, and results can be exported as Arrow tables.
"""

import pathlib
from typing import Any

import pandas as pd
import pyarrow as pa

from scmle.contracts import errors

ARROW_STREAM = "__arrow_c_stream__"


def is_tabular(data: Any) -> bool:
    """Reports whether `data` looks like an in-memory table (not a path).

    Args:
        data: Any object.

    Returns:
        True if `data` is a pandas DataFrame or exposes the Arrow stream
        protocol, `to_pandas()` or `collect()`; False otherwise.
    """
    return (
        isinstance(data, pd.DataFrame)
        or hasattr(data, ARROW_STREAM)
        or hasattr(data, "to_pandas")
        or hasattr(data, "collect")
    )


def to_pandas(data: Any) -> pd.DataFrame:
    """Converts any supported in-memory table to a pandas DataFrame.

    Args:
        data: A pandas DataFrame, `pyarrow.Table`, Polars DataFrame or
            LazyFrame, or any object with the Arrow stream protocol.

    Returns:
        A pandas DataFrame (a copy for pandas input).

    Raises:
        DataError: If the object is not tabular.
    """
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if hasattr(data, "collect"):  # Polars LazyFrame and similar.
        data = data.collect()
    if isinstance(data, pa.Table):
        return data.to_pandas()
    if hasattr(data, ARROW_STREAM):
        return pa.table(data).to_pandas()
    if hasattr(data, "to_pandas"):
        return data.to_pandas()
    if isinstance(data, (str, pathlib.Path)):
        raise errors.DataError("paths are read with the dataset loader")
    raise errors.DataError(f"unsupported table type {type(data).__name__}")


def to_arrow(frame: pd.DataFrame) -> pa.Table:
    """Exports a frame as an Arrow table (zero-copy where possible).

    Args:
        frame: The pandas DataFrame to export.

    Returns:
        An Arrow table with the frame's columns; the index is not included.
    """
    return pa.Table.from_pandas(frame, preserve_index=False)
