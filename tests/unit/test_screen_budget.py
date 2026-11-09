"""The leakage screen is bounded by the declared time budget and loud."""

import json

import numpy
import pandas
import pytest
import support

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.admissibility import screen
from scmle.data.profiling import profiler
from scmle.data.readers import tabular
import scmle.data.validation.target

SMALL_BUDGET = 2
LARGE_BUDGET = 900


def wide(seed=0, rows=300, columns=10, classes=3):
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame(
        {f"x{i}": rng.normal(size=rows) for i in range(columns)}
    )
    frame["y"] = rng.choice([f"c{k}" for k in range(classes)], rows)
    return frame


def run_screen(frame, budget):
    return screen.screen_table(
        frame,
        "y",
        "classification",
        [c for c in frame.columns if c != "y"],
        7,
        time_budget_seconds=budget,
    )


def test_the_work_cap_is_derived_from_the_budget_alone():
    assert screen.work_cap(None) is None
    assert screen.work_cap(120) == 450_000
    assert screen.work_cap(1800) == 6_750_000
    assert screen.SCREEN_SHARE == 0.15


def test_a_wide_multiclass_table_is_declined_small_and_screened_large():
    frame = wide()
    declined = run_screen(frame, SMALL_BUDGET)
    assert declined.permutations == 0 and declined.findings == ()
    assert declined.unchecked == tuple(f"x{i}" for i in range(10))
    assert "NOT RUN for 10 columns" in declined.warning
    assert "work limit for a 2 s time budget" in declined.warning
    assert "unchecked, not clean" in declined.warning
    screened = run_screen(frame, LARGE_BUDGET)
    assert screened.permutations > 0 and screened.unchecked == ()
    assert screened.warning == ""
    again = run_screen(frame, LARGE_BUDGET)
    assert [f.significant for f in again.findings] == [
        f.significant for f in screened.findings
    ]
    assert again.to_dict() == screened.to_dict()


def test_no_budget_means_no_cap_beyond_the_permutation_limit():
    assert run_screen(wide(columns=4), None).unchecked == ()


def test_a_table_declined_for_too_many_permutations_is_loud(monkeypatch):
    monkeypatch.setattr(screen, "MAX_PERMUTATIONS", 300)
    report = run_screen(wide(), None)
    assert report.unchecked and "permutation limit" in report.warning


def profile_of(frame, budget):
    spec = tasks.ProjectSpec(
        data=tasks.DataSpec(uri="m", target="y"),
        resources=tasks.ResourceSpec(time_budget_seconds=budget),
    )
    resolved = scmle.data.validation.target.TaskResolver(spec).resolve(
        frame["y"]
    )
    manifest = tabular.build_manifest(frame, "y")
    return profiler.DatasetProfiler(spec).profile(frame, manifest, resolved)


def test_the_profiler_passes_the_projects_budget_and_warns_first():
    result = profile_of(wide(), SMALL_BUDGET)
    assert result.unchecked_columns == tuple(f"x{i}" for i in range(10))
    assert result.risk_findings[0] == result.screen_warning
    assert result.leakage["warning"] == result.screen_warning
    assert profile_of(wide(), LARGE_BUDGET).unchecked_columns == ()


def test_an_incomplete_screen_is_in_summary_report_log_and_plan(tmp_path):
    result = support.make_project(
        wide(), "y", output_dir=tmp_path, time_budget=SMALL_BUDGET
    ).run()
    warning = result.profile.screen_warning
    assert warning.startswith("leakage screen NOT RUN for 10 columns")
    summary = result.summary()
    assert "PROVISIONAL: leakage screen incomplete" in summary
    assert summary.endswith("WARNING " + warning)
    page = (result.run_dir / "report.html").read_text()
    assert page.index("WARNING leakage screen NOT RUN") < page.index(
        "<h2>Leaderboard"
    )
    log = json.loads(
        (result.run_dir / "diagnostics" / "decision_log.json").read_text()
    )
    assert any(
        d["technique"] == "leakage_screen" and warning in d["reason"]
        for d in log["plan"]
    )
    assert f"WARNING: {warning}" in result.plan.summary()


def test_block_refuses_an_unchecked_screen_and_warn_proceeds(tmp_path):
    with pytest.raises(errors.BlockedError, match="NOT RUN") as raised:
        support.make_project(
            wide(),
            "y",
            output_dir=tmp_path,
            time_budget=SMALL_BUDGET,
            leakage_policy="block",
        ).run()
    assert raised.value.details["leakage"]["unchecked"]
    assert "'warn'" in str(raised.value)
    ok = support.make_project(
        wide(),
        "y",
        output_dir=tmp_path,
        time_budget=LARGE_BUDGET,
        leakage_policy="block",
    ).run()
    assert "NOT RUN" not in ok.summary()
