"""Tests for the benchmark regression gate."""

import json

from benchmarks import check_regression

BASELINE = {
    "a": {"metric": "roc_auc", "bound": 0.9},
    "b": {"metric": "rmse", "bound": 10.0},
}


def record(name, metric, score, status="ok"):
    return {
        "name": name,
        "metric": metric,
        "test_score": score,
        "status": status,
    }


def test_passes_when_every_bound_holds():
    results = [record("a", "roc_auc", 0.95), record("b", "rmse", 9.0)]
    assert check_regression.violations(results, BASELINE) == []


def test_direction_follows_the_metric():
    results = [record("a", "roc_auc", 0.85), record("b", "rmse", 12.0)]
    problems = check_regression.violations(results, BASELINE)
    assert len(problems) == 2 and "breaches" in problems[0]
    assert (
        check_regression.violations(
            [record("a", "roc_auc", 0.9), record("b", "rmse", 10.0)], BASELINE
        )
        == []
    )


def test_missing_failed_and_changed_metric_are_violations():
    results = [record("a", "roc_auc", 0.0, status="error")]
    problems = check_regression.violations(results, BASELINE)
    assert "run failed" in problems[0] and "no result" in problems[1]
    swapped = [record("a", "pr_auc", 0.99), record("b", "rmse", 1.0)]
    assert "metric changed" in check_regression.violations(swapped, BASELINE)[0]


def test_command_line_exit_codes(tmp_path, caplog):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(BASELINE))
    good = tmp_path / "good.json"
    good.write_text(
        json.dumps(
            {
                "results": [
                    record("a", "roc_auc", 0.95),
                    record("b", "rmse", 1.0),
                ]
            }
        )
    )
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {"results": [record("a", "roc_auc", 0.5), record("b", "rmse", 1.0)]}
        )
    )
    assert check_regression.main([str(good), "--baseline", str(baseline)]) == 0
    assert check_regression.main([str(bad), "--baseline", str(baseline)]) == 1
    assert "REGRESSION a" in caplog.text


def repeated(name, metric, scores):
    return {
        "name": name,
        "metric": metric,
        "test_score": sum(scores) / len(scores),
        "test_scores": scores,
        "seeds": len(scores),
        "status": "ok",
    }


def test_noise_around_the_bound_does_not_fail_repeated_runs():
    noisy = repeated("a", "roc_auc", [0.86, 0.95, 0.88, 0.96])

    assert check_regression.violations([noisy], {"a": BASELINE["a"]}) == []


def test_a_whole_interval_beyond_the_bound_fails_repeated_runs():
    worse = repeated("a", "roc_auc", [0.80, 0.81, 0.79, 0.80])

    assert check_regression.violations([worse], {"a": BASELINE["a"]})
