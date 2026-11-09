"""Rows without a target are refused, ignored or put to use on request."""

import numpy
import pandas
import pytest
import support

from scmle.contracts import errors


def partly_labeled(rows=900, unlabeled=300, seed=0):
    rng = numpy.random.default_rng(seed)
    x = rng.normal(size=rows)
    other = rng.normal(size=rows)
    y = (x + 0.5 * other + rng.normal(0, 0.6, rows) > 0).astype(float)
    frame = pandas.DataFrame({"x": x, "other": other, "y": y})
    frame.loc[rng.choice(rows, unlabeled, replace=False), "y"] = numpy.nan
    return frame


def run(tmp_path, frame, **kwargs):
    return support.make_project(
        frame, "y", time_budget=300, output_dir=tmp_path, **kwargs
    ).run()


def test_unlabeled_rows_are_refused_by_default(tmp_path):
    with pytest.raises(errors.BlockedError, match=r"data\.unlabeled"):
        run(tmp_path, partly_labeled())


def test_ignored_rows_are_reported_and_kept_out_of_the_split(tmp_path):
    result = run(tmp_path, partly_labeled(), unlabeled="ignore")

    assert result.evaluation["n_dev"] + result.evaluation["n_test"] == 600
    assert any(
        "300 row(s) without a target" in finding
        for finding in result.profile.quality_findings
    )
    assert result.evaluation["semi_supervised"] is None


def test_pseudo_labeling_is_judged_and_recorded(tmp_path):
    result = run(tmp_path, partly_labeled(), unlabeled="pseudo_label")

    record = result.evaluation["semi_supervised"]
    assert record["unlabeled_rows"] == 300
    assert record["used"] == bool(
        record["outcome"] and record["outcome"]["accepted"]
    )


def test_regression_cannot_be_pseudo_labeled(tmp_path):
    frame = partly_labeled()
    frame["y"] = frame["x"] * 2 + frame["y"].fillna(0)
    frame.loc[:200, "y"] = numpy.nan

    with pytest.raises(errors.ConfigError, match="classification only"):
        run(tmp_path, frame, unlabeled="pseudo_label", task="regression")


def test_labeling_candidates_are_the_rows_near_the_decision_boundary(tmp_path):
    result = run(tmp_path, partly_labeled(), unlabeled="ignore")
    pool = pandas.DataFrame(
        {"x": [-4.0, 0.0, 4.0, 0.05], "other": [0.0, 0.0, 0.0, 0.0]}
    )

    ranked = result.bundle.rank_for_labeling(pool, count=2)

    assert set(ranked["position"]) == {1, 3}
    assert ranked["uncertainty"].is_monotonic_decreasing
