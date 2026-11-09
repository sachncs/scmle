"""Exact replay vs statistically comparable reruns."""

import copy

import support

import scmle.api.project
from scmle.artifacts import reproducibility


def run(frame, tmp_path, **kwargs):
    return support.make_project(
        frame, "y", time_budget=300, output_dir=tmp_path, **kwargs
    ).run()


def test_a_deterministic_rerun_is_an_exact_replay(binary_frame, tmp_path):
    original = run(binary_frame, tmp_path)
    verdict = scmle.api.project.Project.verify_replay(
        original.run_dir, data=binary_frame
    )
    assert verdict.level == reproducibility.EXACT
    assert verdict.reasons == [] and verdict.max_abs_difference == 0.0
    assert verdict.trials_compared == len(original.trials)


def test_a_different_seed_is_a_different_run(binary_frame, tmp_path):
    first = reproducibility.load_run(run(binary_frame, tmp_path / "a").run_dir)
    second = reproducibility.load_run(
        run(binary_frame, tmp_path / "b", seed=7).run_dir
    )
    verdict = reproducibility.compare_runs(first, second)
    assert verdict.level == reproducibility.DIFFERENT
    assert any(
        "splits differ" in r or "configurations differ" in r
        for r in verdict.reasons
    )


def test_changed_data_is_detected(binary_frame, tmp_path):
    first = reproducibility.load_run(run(binary_frame, tmp_path / "a").run_dir)
    altered = binary_frame.copy()
    altered.loc[1, "a"] += 1.0
    second = reproducibility.load_run(run(altered, tmp_path / "b").run_dir)
    assert (
        "datasets differ" in reproducibility.compare_runs(first, second).reasons
    )


def make_facts(scores, std=0.02, selected="logistic_regression"):
    trial = {
        "trial_id": "trial_001",
        "model_name": "logistic_regression",
        "strategy": "default",
        "params": {},
        "status": "ok",
        "fold_scores": scores,
        "score_mean": sum(scores) / len(scores),
        "score_std": std,
    }
    return {
        "config": {"seed": 42},
        "dataset": {"data_hash": "h"},
        "split": "s",
        "trials": [trial],
        "selected": (selected, "default", "{}"),
        "environment": "env",
    }


def test_tiny_nondeterminism_is_comparable_not_exact():
    base = make_facts([0.80, 0.82, 0.81])
    jittered = make_facts([0.8002, 0.8199, 0.8101])
    verdict = reproducibility.compare_runs(base, jittered)
    assert verdict.level == reproducibility.COMPARABLE
    assert 0 < verdict.max_abs_difference < 0.001
    assert (
        reproducibility.compare_runs(base, copy.deepcopy(base)).level
        == reproducibility.EXACT
    )


def test_large_drift_or_a_different_winner_is_different():
    base = make_facts([0.80, 0.82, 0.81])
    moved = make_facts([0.60, 0.62, 0.61])
    assert (
        reproducibility.compare_runs(base, moved).level
        == reproducibility.DIFFERENT
    )
    other = make_facts([0.80, 0.82, 0.81], selected="random_forest")
    verdict = reproducibility.compare_runs(base, other)
    assert verdict.level == reproducibility.DIFFERENT
    assert "a different candidate was selected" in verdict.reasons


def test_environment_differences_are_noted_not_fatal():
    base = make_facts([0.8, 0.8, 0.8])
    other = copy.deepcopy(base)
    other["environment"] = "different pins"
    verdict = reproducibility.compare_runs(base, other)
    assert verdict.level == reproducibility.EXACT
    assert any("environments differ" in note for note in verdict.notes)
