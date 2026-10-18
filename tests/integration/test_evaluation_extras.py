"""Nested validation, prevalence notes and the equal-class warning."""

import numpy as np
import pandas as pd
import support

from mml.contracts import tasks


def frame(rows=600, seed=0):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {"a": rng.normal(size=rows), "b": rng.normal(size=rows)}
    )
    data["y"] = (data.a + rng.normal(0, 0.8, rows) > 0).astype(int)
    return data


def test_nested_validation_reports_the_optimism_of_the_winning_score(tmp_path):
    result = support.make_project(
        frame(),
        "y",
        time_budget=300,
        output_dir=tmp_path,
        selection=tasks.SelectionSpec(nested_cv=True),
    ).run()

    nested = result.evaluation["nested_cv"]
    assert nested["folds"] >= 2 and nested["configurations"] >= 2
    assert "optimism_natural" in nested and 0.4 < nested["mean_natural"] <= 1.0


def test_nested_validation_is_off_unless_asked(tmp_path):
    result = support.make_project(
        frame(), "y", time_budget=300, output_dir=tmp_path
    ).run()

    assert result.evaluation["nested_cv"] is None
    assert result.evaluation["evaluation_prevalence"]["natural"]


def test_perfectly_equal_classes_get_a_down_sampling_warning(tmp_path):
    data = frame(400)
    data["y"] = np.tile([0, 1], 200)

    result = support.make_project(
        data, "y", time_budget=300, output_dir=tmp_path
    ).run()

    assert any(
        "exactly 160 rows" in finding
        for finding in result.profile.quality_findings
    )


def test_the_bias_variance_report_names_the_dominant_error(tmp_path):
    rng = np.random.default_rng(2)
    data = pd.DataFrame({"x": rng.uniform(-3, 3, 800)})
    data["y"] = np.sin(2 * data.x) + rng.normal(0, 0.3, 800)

    result = support.make_project(
        data,
        "y",
        task="regression",
        time_budget=300,
        output_dir=tmp_path,
        selection=tasks.SelectionSpec(bias_variance_replicates=15),
    ).run()

    report = result.evaluation["bias_variance"]
    assert report["kind"] == "squared_error" and report["replicates"] == 15
    assert report["verdict"] in {
        "bias-dominated",
        "variance-dominated",
        "balanced",
    }
    assert report["advice"]


def test_the_bias_variance_report_is_off_by_default(tmp_path):
    result = support.make_project(
        frame(), "y", time_budget=300, output_dir=tmp_path
    ).run()

    assert result.evaluation["bias_variance"] is None


def test_the_sensitivity_report_separates_training_split_and_noise(tmp_path):
    result = support.make_project(
        frame(),
        "y",
        time_budget=300,
        output_dir=tmp_path,
        selection=tasks.SelectionSpec(sensitivity_repeats=4),
    ).run()

    report = result.evaluation["sensitivity"]
    assert report["dominant"] in {"seed", "resample", "split"}
    assert len(report["split"]["scores"]) == 4
    assert report["noise"]["0.0"] >= report["noise"]["1.0"]
    assert report["decay"] <= 0.0


def test_the_sensitivity_report_is_off_by_default(tmp_path):
    result = support.make_project(
        frame(), "y", time_budget=300, output_dir=tmp_path
    ).run()

    assert result.evaluation["sensitivity"] is None
