"""Observers that turn supported Python data into catalog evidence."""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import datetime
import hashlib
from typing import Any

import numpy
import pandas

from scmle.contracts import errors
from scmle.data.catalog import canon
from scmle.data.catalog import evidence
from scmle.data.catalog import identity

MATRIX_DIMENSIONS = 2


@dataclasses.dataclass(frozen=True, slots=True)
class Observation:
    """Intermediate evidence produced by the supported data observers."""

    data_kind: str
    shape: tuple[int, ...]
    schema: tuple[evidence.Field, ...]
    sample_count: int
    missingness: evidence.Missingness
    duplicates: evidence.Duplication
    label: evidence.Labeling
    content_identity: str
    limitations: tuple[str, ...]
    release_blockers: tuple[str, ...]


def sequential(value: Any) -> bool:
    """Returns whether a value can represent an ordered column.

    Args:
        value: Any object.

    Returns:
        True for sequences and NumPy arrays other than strings, bytes,
        byte arrays, and mappings.
    """
    return isinstance(value, (Sequence, numpy.ndarray)) and not isinstance(
        value, (str, bytes, bytearray, Mapping)
    )


def measure(frame: pandas.DataFrame) -> tuple[int, ...]:
    """Reads the dimensions of a frame as plain integers.

    Args:
        frame: The frame to measure.

    Returns:
        The frame's shape.
    """
    return tuple(int(value) for value in frame.shape)


def frame_for_pandas(
    data: pandas.DataFrame | pandas.Series,
) -> tuple[pandas.DataFrame | None, str, tuple[int, ...]]:
    """Converts a pandas object to an observation frame.

    Args:
        data: A DataFrame or a Series.

    Returns:
        A tuple ``(frame, data_kind, shape)``; a Series becomes a one-column
        frame named after the Series, or ``value`` when it has no name.
    """
    if isinstance(data, pandas.DataFrame):
        return data, "pandas_dataframe", measure(data)
    name = str(data.name) if data.name is not None else "value"
    frame = data.to_frame(name=name)
    return frame, "pandas_series", measure(frame)


def tabulate(
    data: numpy.ndarray,
) -> tuple[pandas.DataFrame | None, str, tuple[int, ...]]:
    """Converts a NumPy array to an observation frame.

    Args:
        data: The array.

    Returns:
        A tuple ``(frame, data_kind, shape)``; the frame is None for an
        array with more than two dimensions.
    """
    if data.ndim == 0:
        frame = pandas.DataFrame({"value": [data.item()]})
        return frame, "numpy_scalar", ()
    shape = tuple(int(value) for value in data.shape)
    if data.ndim == 1:
        frame = pandas.DataFrame({"value": data})
        return frame, "numpy_array", shape
    if data.ndim == MATRIX_DIMENSIONS:
        frame = pandas.DataFrame(
            data, columns=[str(value) for value in range(data.shape[1])]
        )
        return frame, "numpy_array", shape
    return None, "numpy_array", shape


def frame_for_mapping(
    data: Mapping[Any, Any],
) -> tuple[pandas.DataFrame | None, str, tuple[int, ...]]:
    """Converts a mapping to an observation frame.

    Args:
        data: A mapping of columns, or a single record.

    Returns:
        A tuple ``(frame, data_kind, shape)``; a mapping whose values are
        all sequences is a set of columns, any other mapping is one record.

    Raises:
        ValueError: If a mapping of columns has columns of unequal length.
    """
    values = list(data.values())
    if values and all(sequential(value) for value in values):
        lengths = {len(value) for value in values}
        if len(lengths) != 1:
            raise errors.InputError("mapping columns must have equal lengths")
        frame = pandas.DataFrame(data)
        return frame, "column_mapping", measure(frame)
    frame = pandas.DataFrame([dict(data)])
    return frame, "record_mapping", measure(frame)


def coerce(
    data: Any,
) -> tuple[pandas.DataFrame | None, str, tuple[int, ...]]:
    """Converts supported record-shaped data to a bounded observation frame.

    Args:
        data: The object to convert.

    Returns:
        A tuple ``(frame, data_kind, shape)``.  ``frame`` is a DataFrame,
        or None for byte payloads, NumPy arrays with more than two
        dimensions, and opaque objects.  ``data_kind`` names the detected
        form (the type name for opaque objects).  ``shape`` is the
        dimensions of the data, ``()`` for scalars and opaque objects.

    Raises:
        ValueError: If a mapping of columns has columns of unequal length.
    """
    if isinstance(data, (pandas.DataFrame, pandas.Series)):
        return frame_for_pandas(data)
    if isinstance(data, numpy.ndarray):
        return tabulate(data)
    if isinstance(data, Mapping):
        return frame_for_mapping(data)
    if isinstance(data, (list, tuple)):
        if data and all(isinstance(value, Mapping) for value in data):
            frame = pandas.DataFrame.from_records(data)
            return frame, "record_sequence", measure(frame)
        frame = pandas.DataFrame({"value": list(data)})
        return frame, "sequence", measure(frame)
    return fallback(data)


def fallback(
    data: Any,
) -> tuple[pandas.DataFrame | None, str, tuple[int, ...]]:
    """Describes bytes, scalars and opaque objects.

    Args:
        data: The object.

    Returns:
        A tuple ``(frame, data_kind, shape)``; the frame is None for bytes
        and opaque objects, whose kind is the type name.
    """
    if isinstance(data, (bytes, bytearray, memoryview)):
        return None, "bytes", (len(data),)
    if isinstance(data, (str, int, float, bool, datetime.date)) or data is None:
        frame = pandas.DataFrame({"value": [data]})
        return frame, "scalar", ()
    return None, type(data).__name__, ()


def describe(frame: pandas.DataFrame) -> tuple[evidence.Field, ...]:
    """Returns ordered field metadata, preserving duplicate column positions.

    Args:
        frame: The frame whose columns are described.

    Returns:
        One field per column, in column order, with the dtype as a string.
    """
    return tuple(
        evidence.Field(str(name), str(data_type))
        for name, data_type in zip(
            frame.columns, frame.dtypes.astype(str), strict=True
        )
    )


def tally(frame: pandas.DataFrame) -> evidence.Missingness:
    """Measures missingness over a frame without imputing or modifying it.

    Args:
        frame: The frame to inspect.

    Returns:
        An observed summary.  Per-field counts are keyed by column name,
        with ``name[position]`` used for names that occur more than once.
    """
    missing = frame.isna()
    counts = missing.sum(axis=0)
    missing_values = int(counts.sum())
    names = [str(name) for name in frame.columns]
    field_counts = {
        (name if names.count(name) == 1 else f"{name}[{position}]"): int(count)
        for position, (name, count) in enumerate(
            zip(names, counts, strict=True)
        )
    }
    return evidence.Missingness(
        status="observed",
        total=int(frame.shape[0] * frame.shape[1]),
        missing=missing_values,
        incomplete=int(missing.any(axis=1).sum()),
        fields=field_counts,
    )


def dedupe(frame: pandas.DataFrame) -> evidence.Duplication:
    """Measures duplicate rows and preserves unsupported cases as evidence.

    Args:
        frame: The frame to inspect.

    Returns:
        A summary counting every row that has at least one identical row
        (all copies, not only the repeats).  If row equality is not
        supported for some field, a ``not_observed`` summary with zero
        duplicates.
    """
    try:
        duplicate_mask = frame.duplicated(keep=False)
    except (TypeError, ValueError):
        # handled: returned as not_observed summary; documented
        return evidence.Duplication(
            status="not_observed",
            rows=0,
            fraction=0.0,
            method="row equality was not supported for one or more fields",
        )
    duplicate_rows = int(duplicate_mask.sum())
    fraction = duplicate_rows / len(frame) if len(frame) else 0.0
    return evidence.Duplication(
        status="observed",
        rows=duplicate_rows,
        fraction=fraction,
        method="pandas.DataFrame.duplicated(keep=False)",
    )


def summarize(
    frame: pandas.DataFrame | None, label_column: str | None
) -> tuple[evidence.Labeling, tuple[str, ...], tuple[str, ...]]:
    """Summarize only an explicitly named label column.

    Args:
        frame: The observation frame, or None when there is none.
        label_column: Name of the declared label column, or None.

    Returns:
        A tuple ``(summary, limitations, blockers)``.  The summary is
        ``not_declared`` without a label column, ``not_observed`` when the
        column is absent or appears more than once, and otherwise
        ``missing_all``, ``constant``, or ``observed`` with a distribution
        keyed by canonical JSON of each value.
    """
    if label_column is None:
        return (
            evidence.Labeling("not_declared", None, 0, 0, 0, {}),
            ("label_was_not_declared",),
            ("label_contract_not_declared_for_supervised_use",),
        )
    if frame is None or label_column not in frame.columns:
        return (
            evidence.Labeling("not_observed", label_column, 0, 0, 0, {}),
            ("declared_label_could_not_be_observed",),
            ("declared_label_field_is_not_available",),
        )
    if list(frame.columns).count(label_column) != 1:
        return (
            evidence.Labeling("not_observed", label_column, 0, 0, 0, {}),
            ("declared_label_field_is_ambiguous",),
            ("declared_label_field_is_ambiguous",),
        )
    series = frame[label_column]
    missing_count = int(series.isna().sum())
    observed = series[~series.isna()]
    distribution: dict[str, int] = {}
    for value in observed.tolist():
        bucket = canon.canonize(canon.canonicalize(value))
        distribution[bucket] = distribution.get(bucket, 0) + 1
    observed_count = len(observed)
    distinct_count = len(distribution)
    status = (
        "missing_all"
        if not observed_count
        else ("constant" if distinct_count == 1 else "observed")
    )
    return (
        evidence.Labeling(
            status,
            label_column,
            observed_count,
            missing_count,
            distinct_count,
            distribution,
        ),
        (),
        ("label_is_constant",) if status == "constant" else (),
    )


def frame_content(
    frame: pandas.DataFrame, data_kind: str, shape: tuple[int, ...]
) -> dict[str, Any]:
    """Returns canonical content for a frame-backed observation.

    Args:
        frame: The observed frame.
        data_kind: Name of the detected data form.
        shape: Dimensions of the original data.

    Returns:
        A JSON-native dictionary of the kind, shape, schema, and all rows
        in order, with each value canonicalized.
    """
    rows = [
        [canon.canonicalize(value) for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    return {
        "data_kind": data_kind,
        "shape": list(shape),
        "schema": [field.to_dict() for field in describe(frame)],
        "rows": rows,
    }


def scrutinize(
    frame: pandas.DataFrame,
    data_kind: str,
    shape: tuple[int, ...],
    label_column: str | None,
    supplied_identity: str | None,
) -> Observation:
    """Observes frame-backed data: schema, quality, label and identity.

    Args:
        frame: The observation frame.
        data_kind: Name of the detected data form.
        shape: Dimensions of the original data.
        label_column: Name of the declared label column, or None.
        supplied_identity: Caller-supplied SHA-256 digest of the content,
            or None to digest the canonical frame content.

    Returns:
        The observation, with missing values and duplicate rows reported as
        limitations and release blockers, not repaired.
    """
    limitations: list[str] = []
    blockers: list[str] = []
    schema = describe(frame)
    sample_count = int(frame.shape[0])
    missingness = tally(frame)
    duplicates = dedupe(frame)
    label, label_limitations, label_blockers = summarize(frame, label_column)
    limitations.extend(label_limitations)
    blockers.extend(label_blockers)
    if missingness.missing:
        limitations.append("missing_values_are_reported_not_repaired")
        blockers.append("missingness_policy_not_declared")
    if duplicates.rows:
        limitations.append("duplicate_rows_are_reported_not_removed")
        blockers.append("duplicate_policy_not_declared")
    if supplied_identity is None:
        content_identity = canon.digest_value(
            frame_content(frame, data_kind, shape)
        )
    else:
        content_identity = identity.assay(supplied_identity, "content_identity")
        limitations.append("content_identity_was_supplied_by_caller")
    return Observation(
        data_kind=data_kind,
        shape=shape,
        schema=schema,
        sample_count=sample_count,
        missingness=missingness,
        duplicates=duplicates,
        label=label,
        content_identity=content_identity,
        limitations=tuple(dict.fromkeys(limitations)),
        release_blockers=tuple(dict.fromkeys(blockers)),
    )


def weigh(
    data: bytes | bytearray | memoryview,
    data_kind: str,
    shape: tuple[int, ...],
    label_column: str | None,
    supplied_identity: str | None,
) -> Observation:
    """Observes a byte payload without decoding it.

    Args:
        data: The payload.
        data_kind: Name of the detected data form.
        shape: Dimensions of the payload.
        label_column: Name of the declared label column, or None.
        supplied_identity: Caller-supplied SHA-256 digest of the content,
            or None to digest the bytes.

    Returns:
        The observation: one payload field, one sample, and no missingness,
        duplicate or label evidence.
    """
    limitations: list[str] = []
    blockers: list[str] = []
    schema = (evidence.Field("payload", "bytes"),)
    content_identity = (
        f"sha256:{hashlib.sha256(bytes(data)).hexdigest()}"
        if supplied_identity is None
        else identity.assay(supplied_identity, "content_identity")
    )
    if supplied_identity is not None:
        limitations.append("content_identity_was_supplied_by_caller")
    limitations.append("modality_specific_decoding_was_not_attempted")
    label, label_limitations, label_blockers = summarize(None, label_column)
    limitations.extend(label_limitations)
    blockers.extend(label_blockers)
    blockers.append("modality_specific_decoder_not_declared")
    return Observation(
        data_kind=data_kind,
        shape=shape,
        schema=schema,
        sample_count=1,
        missingness=evidence.Missingness("not_observed", 0, 0, 0, {}),
        duplicates=evidence.Duplication(
            "not_observed", 0, 0.0, "raw bytes have no row identity"
        ),
        label=label,
        content_identity=content_identity,
        limitations=tuple(limitations),
        release_blockers=tuple(blockers),
    )


def examine(
    data: Any,
    label_column: str | None,
    supplied_identity: str | None,
) -> Observation:
    """Observes a supported object and produces structured quality evidence.

    Args:
        data: The object to observe.
        label_column: Name of the declared label column, or None.
        supplied_identity: Caller-supplied SHA-256 digest of the content,
            or None to compute one when the content is observable.

    Returns:
        The observation.  Frame-backed data gets schema, missingness,
        duplicate, and label evidence; bytes get a single payload field;
        opaque objects get only the supplied identity and a
        single-sample placeholder.

    Raises:
        TypeError: If the object is opaque (neither frame-backed nor
            bytes) and no ``supplied_identity`` is given.
    """
    frame, data_kind, shape = coerce(data)
    limitations: list[str] = []
    blockers: list[str] = []
    if frame is not None:
        return scrutinize(
            frame, data_kind, shape, label_column, supplied_identity
        )
    if isinstance(data, (bytes, bytearray, memoryview)):
        return weigh(data, data_kind, shape, label_column, supplied_identity)
    if supplied_identity is None:
        raise errors.KindError(
            "opaque data needs a caller-supplied SHA-256 content_identity"
        )
    content_identity = identity.assay(supplied_identity, "content_identity")
    label, label_limitations, label_blockers = summarize(None, label_column)
    limitations.extend(
        (
            "schema_and_quality_metrics_were_not_observed",
            "content_identity_was_supplied_by_caller",
            *label_limitations,
        )
    )
    blockers.extend(("schema_adapter_not_declared", *label_blockers))
    return Observation(
        data_kind=data_kind,
        shape=shape,
        schema=(evidence.Field("value", type(data).__name__),),
        sample_count=1,
        missingness=evidence.Missingness("not_observed", 0, 0, 0, {}),
        duplicates=evidence.Duplication(
            "not_observed", 0, 0.0, "opaque object has no row identity"
        ),
        label=label,
        content_identity=content_identity,
        limitations=tuple(limitations),
        release_blockers=tuple(blockers),
    )
