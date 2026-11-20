"""The HTML run report built from a completed run."""

import types

import pandas

from scmle.artifacts import report


def run(evaluation=None, diagnostics=None, warning=None, importance=None):
    profile = types.SimpleNamespace(
        quality=("missing <a>", "warn"),
        risk_findings=("risky",),
        admonition=warning,
    )
    plan = types.SimpleNamespace(summary=lambda: "line one\nline <two>")
    return types.SimpleNamespace(
        identifier="run_<1>",
        evaluation={"metrics": {"accuracy": 0.91234, "f1": 0.5}}
        | (evaluation or {}),
        diagnostics=diagnostics or {},
        profile=profile,
        plan=plan,
        salience=lambda: (
            importance
            if importance is not None
            else pandas.DataFrame({"feature": [], "importance": []})
        ),
        summary=lambda: "done & dusted",
        model="ridge",
    )


LEADERBOARD = pandas.DataFrame({"model": ["ridge"], "score": [0.9]})


def test_the_page_escapes_and_lists_the_run_facts():
    page = report.Renderer().render(run(), LEADERBOARD)
    assert page.startswith("<!doctype html><meta charset=utf-8>")
    assert "<title>run_&lt;1&gt;</title><h1>run_&lt;1&gt;</h1>" in page
    assert "<p><b>done &amp; dusted</b></p>" in page
    assert (
        "<h2>Findings</h2><ul><li>missing &lt;a&gt;</li><li>warn</li>"
        "<li>risky</li></ul>"
    ) in page
    assert "<tr><td>accuracy</td><td>0.9123</td></tr>" in page
    assert "<tr><td>f1</td><td>0.5000</td></tr>" in page
    assert "<h2>Final test metrics (ridge)</h2>" in page
    assert "<h2>Feature importance (permutation)</h2><p>unavailable</p>" in page
    assert "<ul><li>unavailable</li></ul>" in page
    assert "<p>line one<br>line &lt;two&gt;</p></details>" in page
    assert "<p>no checks were available</p>" in page
    assert "Brier" not in page


def test_the_screening_warning_comes_first_and_bold():
    page = report.Renderer().render(run(warning="warn"), LEADERBOARD)
    assert (
        "<ul><li><b>WARNING warn</b></li><li>missing &lt;a&gt;</li>"
        "<li>risky</li></ul>"
    ) in page


def test_a_lone_warning_replaces_the_none_item():
    result = run(warning="only")
    result.profile.quality = ("only",)
    result.profile.risk_findings = ()
    page = report.Renderer().render(result, LEADERBOARD)
    assert "<ul><li><b>WARNING only</b></li></ul>" in page
    result.profile.admonition = None
    result.profile.quality = ()
    page = report.Renderer().render(result, LEADERBOARD)
    assert "<h2>Findings</h2><ul><li>none</li></ul>" in page


def test_importance_calibration_and_explanations_are_shown():
    importance = pandas.DataFrame(
        {"feature": [f"f{i}" for i in range(12)], "importance": [0.123456] * 12}
    )
    result = run(
        evaluation={"explanations": {"shap": {}, "lime": {}}},
        diagnostics={
            "calibration": {
                "brier_score": 0.12345,
                "expected_calibration_error": 0.05,
            }
        },
        importance=importance,
    )
    page = report.Renderer().render(result, LEADERBOARD)
    assert "<td>0.1235</td>" in page
    assert "f9" in page
    assert "f10" not in page
    assert (
        "<p>Brier score 0.1235, expected calibration error 0.0500</p>" in page
    )
    assert (
        '<li><a href="explanations/lime.html">lime</a></li>'
        '<li><a href="explanations/shap.html">shap</a></li>'
    ) in page


def test_reliability_lists_every_check_with_results():
    result = run(
        evaluation={
            "health": {"verdict": "ok", "note": "fine"},
            "learning": {
                "verdict": "flat",
                "gains": {"last": 0.01, "total": -0.2},
            },
        },
        diagnostics={
            "adversarial_validation": {
                "available": True,
                "verdict": "similar",
                "auc": 0.514,
            },
            "robustness": {"scenarios": [1], "fragile_under": ["noise", "<x>"]},
            "slice_performance": {
                "worst": [
                    {"feature": "age", "slice": "old", "rows": 7, "gap": -0.1}
                ]
            },
            "fairness": {
                "flagged": ["sex"],
                "columns": {"sex": {"flags": ["gap", "ratio"]}},
            },
        },
    )
    assert report.Renderer.reliability(result) == (
        "<p>Fit health: <b>ok</b> (fine)</p>"
        "<p>Learning curve: <b>flat</b> (last step +0.0100, whole curve "
        "-0.2000)</p>"
        "<p>Development versus test rows: similar (separability AUC 0.51)</p>"
        "<p>Robustness: fragile under noise, &lt;x&gt;</p>"
        "<p>Weakest slice: age = old (7 rows, -0.100 versus overall)</p>"
        "<p>Fairness (sex): gap; ratio</p>"
    )


def test_reliability_skips_checks_without_results():
    result = run(
        diagnostics={
            "adversarial_validation": {"available": False},
            "robustness": {"scenarios": [1], "fragile_under": []},
            "slice_performance": {"worst": []},
        }
    )
    assert report.Renderer.reliability(result) == (
        "<p>Robustness: fragile under none</p>"
    )
