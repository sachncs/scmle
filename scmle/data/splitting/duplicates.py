"""Finds rows that repeat one record.

Rows whose features match (numbers compared to six decimals, text exactly)
form a group. The splitter counts test rows that repeat a development record
(`splitters.Splitter.disclose`); a charter run's custody carve keeps each
group on one side; the duplicate detector reports them.
"""

import numpy
import pandas

DECIMALS = 6
NOT_DUPLICATED = -1


def duplicate_groups(
    frame: pandas.DataFrame, columns: list[str]
) -> numpy.ndarray:
    """Labels each row with the id of its duplicate group.

    Args:
        frame: The full dataset.
        columns: Feature columns that define a record.

    Returns:
        An integer array aligned with the frame's rows. Rows with no copy
        get `NOT_DUPLICATED`; rows whose features match share one group id.
        An empty `columns` list marks every row `NOT_DUPLICATED`.
    """
    if not columns:
        return numpy.full(len(frame), NOT_DUPLICATED)
    keyed = frame[columns].copy()
    for column in keyed.columns:
        if pandas.api.types.is_float_dtype(keyed[column]):
            keyed[column] = keyed[column].round(DECIMALS)
        elif not pandas.api.types.is_numeric_dtype(keyed[column]):
            keyed[column] = keyed[column].astype(str)
    hashes = pandas.util.hash_pandas_object(keyed, index=False).to_numpy()
    codes, uniques = pandas.factorize(hashes)
    sizes = numpy.bincount(codes, minlength=len(uniques))
    return numpy.where(sizes[codes] > 1, codes, NOT_DUPLICATED)
