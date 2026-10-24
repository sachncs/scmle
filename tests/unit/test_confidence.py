"""Cross-validation uncertainty accounts for fold dependence."""

import pytest

from scmle.evaluation import confidence


def test_corrected_error_is_wider_than_the_naive_one():
    scores = [0.80, 0.82, 0.78, 0.81, 0.79]
    result = confidence.cv_interval(scores, 400, 100, "random")
    assert result["corrected_standard_error"] > result["naive_standard_error"]
    low, high = result["interval"]
    assert low < result["mean"] < high
    assert result["mean"] == pytest.approx(0.80)
    ratio = result["corrected_standard_error"] / result["naive_standard_error"]
    assert ratio == pytest.approx((1 + 5 * 100 / 400) ** 0.5)


def test_temporal_validation_is_flagged_as_optimistic():
    scores = [0.7, 0.72, 0.69]
    note = confidence.cv_interval(scores, 300, 100, "temporal")["note"]
    assert "optimistic" in note


def test_a_single_fold_has_no_interval():
    assert (
        confidence.cv_interval([0.9], 100, 10, "random")["available"] is False
    )


def test_runs_report_the_interval(binary_frame, tmp_path):
    import support

    result = support.make_project(binary_frame, "y", output_dir=tmp_path).run()
    record = result.evaluation["cv_confidence"]
    assert record["available"] and record["folds"] == 5
    assert record["interval"][0] <= result.evaluation["selected_cv_score"]
