"""The leakage screen is bounded by the declared time budget and loud."""

import dataclasses
import json

import numpy
import pandas
import pytest
import support

from scmle.assurance.admissibility import leakage
from scmle.assurance.admissibility import plants
from scmle.assurance.admissibility import screen
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.readers import tabular
import scmle.data.validation.target
from scmle.pipeline import profiler

SMALL_BUDGET = 2
LARGE_BUDGET = 900
# A whole run needs more than a 2 s budget, so the run-level tests use a
# table whose screen (about 11 s predicted for 40 columns, 3 classes, 300
# rows) does not fit the 3 s share of a 20 s budget that a run fits in.
WIDE = 40
RUN_BUDGET = 20


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


def test_the_share_is_a_fraction_of_the_budget_in_seconds():
    assert screen.share(None) is None
    assert screen.share(120) == pytest.approx(120 * screen.SCREEN_SHARE)
    assert screen.SCREEN_SHARE == 0.15


def test_the_cost_of_a_scan_grows_with_its_rows():
    small = screen.cost("classification", 1000, 50_000)
    large = screen.cost("classification", 2000, 50_000)
    assert large - screen.SCAN_SECONDS == pytest.approx(
        2 * (small - screen.SCAN_SECONDS)
    )
    assert screen.cost("regression", 2000, 50_000) < large
    # The old cap let 0.15 B x 25,000 fits through whatever the rows; at
    # 2000 rows that work is predicted to take longer than the share.
    fits = int(60 * screen.SCREEN_SHARE * screen.FITS_PER_SECOND)
    assert screen.cost("classification", 2000, fits) > 60 * 0.15


def demands(frame):
    cfg = dataclasses.replace(leakage.CERTIFIED, run_seed=7)
    features = [c for c in frame.columns if c != "y"]
    real = screen.measure(
        frame, "y", "classification", cfg, features, None, None
    )
    copy = plants.table(frame, "y", 7, continuous=False)
    names = [*features, *plants.DEFECTS]
    probe = screen.measure(copy, "y", "classification", cfg, names, None, None)
    return real, probe


def test_measure_counts_the_work_the_scan_draws():
    frame = wide(classes=2, columns=6)
    real = demands(frame)[0]
    report = run_screen(frame, None)
    assert real.permutations == report.permutations
    assert real.family == report.family
    assert real.rows == report.rows


def test_the_real_and_probe_scans_share_one_allowance():
    frame = wide(classes=2, columns=6)
    real, probe = demands(frame)
    total = real.seconds + probe.seconds
    assert total <= screen.share(LARGE_BUDGET)
    assert run_screen(frame, LARGE_BUDGET).probe
    squeezed = (real.seconds + probe.seconds / 2) / screen.SCREEN_SHARE
    partial = run_screen(frame, squeezed)
    assert partial.unchecked == () and partial.probe == {}
    assert any(note.startswith("probe not run") for note in partial.notes)
    alone = run_screen(frame, None)
    assert [(f.feature, f.p_value) for f in partial.findings] == [
        (f.feature, f.p_value) for f in alone.findings
    ]
    short = (real.seconds / 2) / screen.SCREEN_SHARE
    assert run_screen(frame, short).unchecked


def test_a_declined_screen_runs_no_probe():
    declined = run_screen(wide(), SMALL_BUDGET)
    assert declined.probe == {}
    assert not any("did not find the planted" in n for n in declined.notes)


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
    spec = tasks.Config(
        data=tasks.Dataset(uri="m", target="y"),
        resources=tasks.Resources(time_budget_seconds=budget),
    )
    resolved = scmle.data.validation.target.TaskResolver(spec).resolve(
        frame["y"]
    )
    manifest = tabular.manifested(frame, "y")
    return profiler.Profiler(spec).profile(frame, manifest, resolved)


def test_the_profiler_passes_the_projects_budget_and_warns_first():
    result = profile_of(wide(), SMALL_BUDGET)
    assert result.untested == tuple(f"x{i}" for i in range(10))
    assert result.risk_findings[0] == result.admonition
    assert result.leakage["warning"] == result.admonition
    assert profile_of(wide(), LARGE_BUDGET).untested == ()


def test_an_incomplete_screen_is_in_summary_report_log_and_plan(tmp_path):
    # Enforces docs/LEAKAGE.md:188 and docs/LEAKAGE.md:194.
    result = support.make_project(
        wide(columns=WIDE), "y", output_dir=tmp_path, time_budget=RUN_BUDGET
    ).run()
    warning = result.profile.admonition
    assert warning.startswith("leakage screen NOT RUN for 40 columns")
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
    # Enforces docs/ARCHITECTURE.md:134.
    # Enforces docs/LEAKAGE.md:194.
    with pytest.raises(errors.BlockedError, match="NOT RUN") as raised:
        support.make_project(
            wide(columns=WIDE),
            "y",
            output_dir=tmp_path,
            time_budget=RUN_BUDGET,
            leakage_policy="block",
        ).run()
    assert raised.value.details["leakage"]["unchecked"]
    assert "'warn'" in str(raised.value)
    # Only the binary screen has a passing certificate, so a screened run
    # under `block` is a binary one (ERR-1).
    ok = support.make_project(
        wide(classes=2),
        "y",
        output_dir=tmp_path,
        time_budget=LARGE_BUDGET,
        leakage_policy="block",
    ).run()
    assert "NOT RUN" not in ok.summary()
