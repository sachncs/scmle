"""The planted target encoding is one a column can carry."""

import numpy
import pandas

from benchmarks.evidence import leakage


def test_a_target_encoding_is_a_function_of_its_source_column():
    # A column of few values with many ties, as blood transfusion's `V4`:
    # rows tied on the source value share one level, so the encoding is
    # the column's own (no bin cuts a value by row position).
    rng = numpy.random.default_rng(0)
    frame = pandas.DataFrame(
        {"x": rng.integers(0, 40, 400), "y": rng.integers(0, 2, 400)}
    )
    table = leakage.Table("toy", "toy", "y", leakage.CLASSIFY)
    encoded = leakage.plant(frame, table, "target_encoding", 0)
    per = encoded.groupby("x")["level_target_mean"].nunique()
    assert (per == 1).all()
    assert encoded["level_target_mean"].nunique() > 2


def test_a_target_encoding_of_distinct_values_has_levels_of_level_rows():
    rng = numpy.random.default_rng(1)
    frame = pandas.DataFrame(
        {"x": rng.normal(size=400), "y": rng.integers(0, 2, 400)}
    )
    table = leakage.Table("toy", "toy", "y", leakage.CLASSIFY)
    encoded = leakage.plant(frame, table, "target_encoding", 0)
    order = encoded.sort_values("x")["level_target_mean"].to_numpy()
    runs = 1 + numpy.count_nonzero(order[1:] != order[:-1])
    # Neighbouring levels may share a mean, never split one.
    assert runs <= len(frame) // leakage.LEVEL
