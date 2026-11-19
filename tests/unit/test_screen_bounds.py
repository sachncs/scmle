"""The screen survives any column name and keeps to its share of the budget."""

import dataclasses

import numpy
import pandas
import pytest

from scmle.assurance.admissibility import leakage
from scmle.assurance.admissibility import plants
from scmle.assurance.admissibility import screen
from scmle.assurance.admissibility.screen import planning
from scmle.assurance.admissibility.screen import records


def noise(rows=200, seed=0):
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame(
        {
            "leak_copy": rng.normal(size=rows),
            "leak_duplicates": rng.normal(size=rows),
            "x": rng.normal(size=rows),
        }
    )
    frame["y"] = (frame["x"] + rng.normal(size=rows) > 0).astype(int)
    return frame


def test_a_user_column_named_as_a_plant_is_screened_not_refused():
    # A user table may hold a column named like one of the probe's plants;
    # the probe names its plants apart and screens the user's column.
    frame = noise()
    features = ["leak_copy", "leak_duplicates", "x"]
    report = screen.screen_table(frame, "y", "classification", features, 0)
    assert set(report.probe) == set(plants.KINDS)
    assert report.probe["leak_copy"]
    assert "leak_copy" not in {f.feature for f in report.suspects}


def reference(values, orders):
    # The loop `records.runs` replaced, one sort per order.
    codes, levels = pandas.factorize(values)
    shares = numpy.bincount(codes, minlength=len(levels)) / len(values)
    best = 1.0 - float((shares**2).sum())
    labels = None
    for key in orders:
        order = numpy.lexsort((values, key))
        held = values[order]
        changed = numpy.r_[True, held[1:] != held[:-1]]
        moved = numpy.r_[True, key[order][1:] != key[order][:-1]]
        share = changed.sum() / (changed | moved).sum()
        if share < best:
            best = share
            labels = numpy.empty(len(values), dtype="int64")
            labels[order] = numpy.cumsum(changed) - 1
    return labels


def test_runs_group_the_rows_as_one_sort_per_order_did():
    rng = numpy.random.default_rng(3)
    for trial in range(30):
        rows = int(rng.integers(5, 300))
        frame = pandas.DataFrame(
            {
                f"c{j}": rng.integers(0, int(rng.integers(1, 40)), rows)
                for j in range(int(rng.integers(1, 9)))
            }
        )
        groups = numpy.arange(rows) // int(rng.integers(1, 8))
        means = pandas.Series(rng.random(rows)).groupby(groups)
        values = means.transform("mean").to_numpy()
        if trial % 3 == 0:
            values = rng.integers(0, 3, rows).astype(float)
        orders = records.keys(frame, list(frame.columns))
        found = records.runs(values, orders)
        expected = reference(values, orders)
        if expected is None:
            assert found is None
        else:
            assert numpy.array_equal(found, expected)
    assert records.runs(numpy.zeros(4), []) is None


def test_a_date_column_written_as_iso_text_is_read_against_the_cutoff():
    frame = pandas.DataFrame(
        {
            "closed": ["2020-01-03", "2019-12-30", "2020-02-01", None],
            "note": ["2020-01-03", "soon", "2020-02-01", "x"],
            "days": [1.0, 2.0, 3.0, 4.0],
        }
    )
    found = records.dated(frame, list(frame.columns), "2020-01-01")
    assert [f.feature for f in found] == ["closed"]
    assert found[0].statistic == pytest.approx(0.5)
    stamped = frame.assign(closed=pandas.to_datetime(frame["closed"]))
    again = records.dated(stamped, ["closed"], "2020-01-01")
    assert again[0].statistic == found[0].statistic


def test_a_target_mean_with_gaps_is_tested_where_it_is_recorded():
    # An encoding of a source with missing values leaves those rows blank;
    # it is still a target mean on the rows where it is filled.
    rng = numpy.random.default_rng(5)
    rows = 600
    source = rng.normal(size=rows)
    y = (source + rng.normal(size=rows) > 0).astype(int)
    level = numpy.argsort(numpy.argsort(source)) // 5
    mean = pandas.Series(y).groupby(level).transform("mean").to_numpy()
    blank = rng.random(rows) < 0.2
    frame = pandas.DataFrame(
        {
            "source": numpy.where(blank, numpy.nan, source),
            "encoded": numpy.where(blank, numpy.nan, mean),
            "y": y,
        }
    )
    cfg = dataclasses.replace(leakage.CERTIFIED, seed=0)
    names = ["source", "encoded"]
    found = planning.ready(frame, "y", "classification", cfg, names, None, None)
    sample, problem = found[0], found[3]
    assert records.candidates(sample, names, problem) == [1]
    (found,) = records.encodings(sample, names, problem, cfg)
    assert found.feature == "encoded" and found.p_value < 1e-6
    assert found.bound > screen.NATURAL


def binary_table(rows=2000, columns=300, seed=0):
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame(
        rng.integers(0, 2, (rows, columns)),
        columns=[f"b{j}" for j in range(columns)],
    )
    frame["y"] = rng.integers(0, 2, rows)
    return frame


def test_proposing_and_records_are_priced_from_the_shape_alone():
    frame = binary_table()
    names = [c for c in frame.columns if c != "y"]
    cfg = dataclasses.replace(leakage.CERTIFIED, seed=0)
    before = planning.foresee(
        frame, "y", "classification", cfg, names, None, None
    )
    proposing = 300 * (screen.COLUMN_SECONDS + screen.WAY_SECONDS)
    records_part = screen.RECORD_SECONDS * 2000 * 300 * 300
    assert before == pytest.approx(
        screen.SCAN_SECONDS + proposing + records_part
    )
    assert (
        planning.foresee(
            frame, "absent", "classification", cfg, names, None, None
        )
        is None
    )


def test_a_wide_table_is_declined_before_proposing_with_the_reason(monkeypatch):
    def refuse(*args, **options):
        raise AssertionError("proposed although the scan cannot fit")

    monkeypatch.setattr(screen, "propose", refuse)
    frame = binary_table()
    names = [c for c in frame.columns if c != "y"]
    report = screen.screen_table(
        frame, "y", "classification", names, 0, time_budget_seconds=30
    )
    assert report.unchecked == tuple(names)
    assert "work limit for a 30 s time budget" in report.warning
    assert report.notes[0].startswith("proposing the hypotheses of 300 columns")
    assert report.notes[-1] == "probe not run: the screen itself did not run"


def test_a_scan_that_fits_records_its_predicted_seconds():
    frame = noise()
    names = ["leak_copy", "leak_duplicates", "x"]
    report = screen.screen_table(
        frame, "y", "classification", names, 0, time_budget_seconds=900
    )
    assert report.unchecked == () and report.probe
    assert 0 < report.seconds <= screen.share(900)
