"""ISO 8601 date text.

A text column is read as dates only when no reader could take it two ways:
every value is an ISO 8601 date (year, month, day in that order) in one
layout and one zone. The datetime column kind asks this module; day-first
or month-first text stays categorical rather than risk a silent misreading.
"""

import re

import pandas

# ISO 8601 calendar date, optionally with a time (minutes, seconds,
# fractions) and a zone. Year, month, day in that order is the one layout
# no reader takes two ways; day-first or month-first text is not matched.
ISO_DATE = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
    r"(?P<zone>Z|[+-]\d{2}(?::?\d{2})?)?)?"
)


def iso(series: pandas.Series) -> bool:
    """Whether a text column holds ISO 8601 dates in a single layout.

    Every distinct non-missing value must match `ISO_DATE` with the digits
    in the same places and the same zone (so every parser reads the column
    the same way, and as one clock), and `pandas.to_datetime`, the parser
    the calendar features use, must read each as a moment.

    Args:
        series: Column values.

    Returns:
        True for a text, object or categorical column whose non-missing
        values are all ISO dates of one layout and one zone; False
        otherwise, including for a column with no non-missing value.
    """
    values = texts(series)
    if (
        values is None
        or values.empty
        or not all(isinstance(v, str) for v in values)
        or not values.str.fullmatch(ISO_DATE.pattern).all()
    ):
        return False
    layouts = values.str.replace(r"\d", "0", regex=True)
    if not (layouts == layouts.iloc[0]).all():
        return False
    # Two offsets in one layout (`+01:00`, `+02:00`) are not one clock; the
    # parser refuses them, and a column of them is not read as dates.
    zones = values.str.extract(ISO_DATE.pattern)["zone"].fillna("")
    if not (zones == zones.iloc[0]).all():
        return False
    parsed = pandas.to_datetime(values, errors="coerce")
    return bool(
        pandas.api.types.is_datetime64_any_dtype(parsed)
        and parsed.notna().all()
    )


def texts(series: pandas.Series) -> pandas.Series | None:
    """The distinct non-missing values of a text-like column.

    Args:
        series: Column values.

    Returns:
        The categories of a categorical column, the distinct non-missing
        values of an object or string column, or None for any other dtype.
    """
    if isinstance(series.dtype, pandas.CategoricalDtype):
        return pandas.Series(series.cat.categories)
    if pandas.api.types.is_object_dtype(
        series
    ) or pandas.api.types.is_string_dtype(series):
        return pandas.Series(series.dropna().unique())
    return None
