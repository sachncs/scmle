"""Tests for the Step 2 run diagnostics, on a real tiny SCMLE run."""

import numpy
import pandas
import pytest

from benchmarks.sota import diagnostics
import scmle.api.project


@pytest.fixture(scope="module")
def summary(tmp_path_factory):
    rng = numpy.random.default_rng(0)
    data = pandas.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    data["y"] = numpy.where(data["a"] + rng.normal(size=200) > 0, "p", "n")
    root = tmp_path_factory.mktemp("runs")
    result = scmle.api.project.Project(
        data, "y", time_budget=15, seed=3, output_dir=root
    ).run()
    return diagnostics.summarize_run(result.run_dir)


def test_the_budget_and_time_used_are_reported(summary):
    assert summary["budget_s"] == 15
    assert summary["seconds_events"] > 0
    assert summary["fraction_used"] == pytest.approx(
        summary["seconds_events"] / 15, abs=0.01
    )


def test_every_trial_status_is_counted(summary):
    assert set(summary["trials"]) == {"ok", "failed", "pruned", "skipped"}
    assert summary["trials"]["ok"] > 0
    assert sum(summary["models"].values()) == summary["trials"]["ok"]


def test_the_selected_model_ran(summary):
    assert summary["selected"] in summary["models"]


def test_skipped_stages_are_listed_with_a_reason(summary):
    for stage in summary["skipped_stages"]:
        assert stage["event"].endswith(".skipped")
        assert "reason" in stage
