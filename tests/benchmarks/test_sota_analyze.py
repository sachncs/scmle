"""Tests for the comparison statistics, against hand-computed values."""

import pytest

from benchmarks.sota import analyze
from benchmarks.sota import records


def run(system, task, score, metric="roc_auc", status="ok", budget=60):
    return records.RunRecord(
        system=system,
        budget_s=budget,
        task_id=task,
        fold=0,
        repeat=0,
        metric=metric,
        score=None if status != "ok" else score,
        seconds=1.0,
        status=status,
    )


def test_lower_is_better_metrics_are_flipped_so_higher_is_always_better():
    assert analyze.oriented(run("a", 1, 0.3, "log_loss")) == -0.3
    assert analyze.oriented(run("a", 1, 0.9, "roc_auc")) == 0.9
    assert analyze.oriented(run("a", 1, 0, status="failed")) is None


def test_normalization_gives_best_one_worst_zero_and_failures_zero():
    found = [
        run("scmle", 1, 0.80),
        run("ag", 1, 0.90),
        run("lgbm", 1, 0.70),
        run("flaml", 1, 0.0, status="failed"),
    ]
    systems = ["ag", "flaml", "lgbm", "scmle"]

    normalized = analyze.normalize(analyze.task_scores(found, 60), systems)

    assert normalized[1] == pytest.approx(
        {"ag": 1.0, "flaml": 0.0, "lgbm": 0.0, "scmle": 0.5}
    )


def test_a_task_where_every_system_ties_scores_one_for_all():
    found = [run("scmle", 1, 0.5), run("ag", 1, 0.5)]

    normalized = analyze.normalize(
        analyze.task_scores(found, 60), ["ag", "scmle"]
    )

    assert normalized[1] == {"ag": 1.0, "scmle": 1.0}


def test_a_close_system_meets_the_claim_and_a_far_one_does_not():
    close = [
        run(s, t, v)
        for t in range(1, 21)
        for s, v in (("ag", 0.90), ("scmle", 0.899), ("lgbm", 0.80))
    ]
    far = [
        run(s, t, v)
        for t in range(1, 21)
        for s, v in (("ag", 0.90), ("scmle", 0.82), ("lgbm", 0.80))
    ]

    good = analyze.summarize(close, 60)
    bad = analyze.summarize(far, 60)

    assert good["best_reference"] == "ag"
    assert good["claim_met"] is True
    assert bad["claim_met"] is False
    assert bad["wins_losses_ties"]["losses"] == 20


def test_the_bootstrap_interval_contains_the_point_estimate_and_is_stable():
    reference = [1.0, 1.0, 0.9, 1.0, 0.8, 1.0]
    ours = [0.9, 1.0, 0.7, 0.8, 0.8, 0.9]

    first = analyze.bootstrap_gap(reference, ours)
    second = analyze.bootstrap_gap(reference, ours)

    assert first == second
    assert first[1] <= first[0] <= first[2]


def test_holm_rejects_in_order_and_stops_at_the_first_failure():
    pairs = {("a", "b"): 0.001, ("a", "c"): 0.04, ("b", "c"): 0.6}

    verdict = analyze.holm(pairs)

    assert verdict == {("a", "b"): True, ("a", "c"): False, ("b", "c"): False}


def test_failures_are_counted_in_the_summary():
    found = [
        run("scmle", 1, 0.8),
        run("ag", 1, 0.9),
        run("ag", 2, 0.0, status="timeout"),
        run("scmle", 2, 0.7),
    ]

    summary = analyze.summarize(found, 60)

    assert summary["failures"] == {"ag": 1, "scmle": 0}
