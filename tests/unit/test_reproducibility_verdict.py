"""Tests for classifying a rerun as exact, comparable or different."""

import copy
import json
import math
import pathlib
from typing import Any

import numpy as np

from scmle.artifacts import reproducibility
from scmle.evaluation import parameter_estimation


def trial(
    trial_id: str = "t1",
    scores: tuple[float, ...] = (0.8, 0.9, 1.0),
    status: str = "ok",
    std: float = 0.1,
) -> dict[str, Any]:
    """Return one trial record."""
    return {
        "trial_id": trial_id,
        "model_name": "ridge",
        "strategy": "single",
        "params": {"alpha": 1.0},
        "status": status,
        "fold_scores": list(scores),
        "score_mean": sum(scores) / len(scores) if scores else 0.0,
        "score_std": std,
    }


def facts() -> dict[str, Any]:
    """Return the comparable facts of a small two-trial run."""
    return {
        "config": {"seed": 1},
        "dataset": {"data_hash": "abc"},
        "split": "split-1",
        "trials": [trial("t1"), trial("t2", (0.5, 0.6, 0.7))],
        "selected": ("ridge", "single", '{"alpha": 1.0}'),
        "environment": "numpy==2.0",
        "parameter_estimation": None,
    }


def test_identical_runs_are_exact():
    verdict = reproducibility.compare_runs(facts(), facts())

    assert verdict.level == reproducibility.EXACT
    assert verdict.reasons == []
    assert verdict.max_abs_difference == 0.0
    assert verdict.trials_compared == 2
    assert verdict.notes == []


def test_environment_difference_is_a_note_only():
    other = facts()
    other["environment"] = "numpy==2.1"

    verdict = reproducibility.compare_runs(facts(), other)

    assert verdict.level == reproducibility.EXACT
    assert len(verdict.notes) == 1
    assert "environments differ" in verdict.notes[0]


def test_small_score_drift_is_comparable():
    other = facts()
    other["trials"][0] = trial("t1", (0.81, 0.9, 0.99))

    verdict = reproducibility.compare_runs(facts(), other)

    assert verdict.level == reproducibility.COMPARABLE
    assert math.isclose(verdict.max_abs_difference, 0.01)
    assert verdict.reasons == ["fold scores differ within tolerance"]


def test_large_score_drift_is_different():
    other = facts()
    other["trials"][1] = trial("t2", (0.1, 0.2, 0.3))

    verdict = reproducibility.compare_runs(facts(), other)

    assert verdict.level == reproducibility.DIFFERENT
    assert verdict.trials_compared == 2
    assert any("t2: mean score moved" in r for r in verdict.reasons)


def test_tolerance_scales_the_comparable_band():
    other = facts()
    other["trials"][0] = trial("t1", (0.6, 0.7, 0.8))

    strict = reproducibility.compare_runs(facts(), other, tolerance_std=1.0)
    loose = reproducibility.compare_runs(facts(), other, tolerance_std=3.0)

    assert strict.level == reproducibility.DIFFERENT
    assert loose.level == reproducibility.COMPARABLE


def test_zero_std_uses_the_floor():
    first, second = facts(), facts()
    first["trials"] = [trial("t1", (0.5, 0.5), std=0.0)]
    second["trials"] = [trial("t1", (0.5, 0.5000001), std=0.0)]

    verdict = reproducibility.compare_runs(first, second)

    assert verdict.level == reproducibility.DIFFERENT


def test_status_change_is_different():
    other = facts()
    other["trials"][0] = trial("t1", (), status="failed")

    verdict = reproducibility.compare_runs(facts(), other)

    assert verdict.level == reproducibility.DIFFERENT
    assert "t1: status ok -> failed" in verdict.reasons


def test_matching_failed_trials_without_scores_are_exact():
    first, second = facts(), facts()
    first["trials"] = [trial("t1", (), status="failed")]
    second["trials"] = [trial("t1", (), status="failed")]
    first["trials"][0]["score_mean"] = 0.0

    verdict = reproducibility.compare_runs(first, second)

    assert verdict.level == reproducibility.EXACT


def test_different_selection_is_different_even_with_equal_scores():
    other = facts()
    other["selected"] = ("lasso", "single", "{}")

    verdict = reproducibility.compare_runs(facts(), other)

    assert verdict.level == reproducibility.DIFFERENT
    assert "a different candidate was selected" in verdict.reasons


def test_structural_differences_short_circuit():
    other = copy.deepcopy(facts())
    other["config"] = {"seed": 2}
    other["dataset"] = {"data_hash": "xyz"}
    other["split"] = "split-2"
    other["parameter_estimation"] = "digest"
    other["trials"] = other["trials"][:1]

    verdict = reproducibility.compare_runs(facts(), other)

    assert verdict.level == reproducibility.DIFFERENT
    assert verdict.reasons == [
        "configurations differ",
        "datasets differ",
        "splits differ",
        "parameter-estimation evidence differs",
        "the executed trials differ",
    ]
    assert math.isnan(verdict.max_abs_difference)
    assert verdict.trials_compared == 0


def test_changed_trial_params_are_detected():
    other = facts()
    other["trials"][0]["params"] = {"alpha": 2.0}

    verdict = reproducibility.compare_runs(facts(), other)

    assert verdict.reasons == ["the executed trials differ"]


def write_json(path: pathlib.Path, record: dict[str, Any]) -> None:
    """Write a JSON record, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


def write_run(root: pathlib.Path, with_extras: bool = False) -> None:
    """Write the files of a finished run into a directory."""
    write_json(root / "config.json", {"seed": 1})
    write_json(root / "dataset_manifest.json", {"data_hash": "abc"})
    write_json(
        root / "evaluation.json",
        {
            "split_fingerprint": "split-1",
            "selected_model": "ridge",
            "selected_strategy": "single",
            "selected_params": {"b": 2, "a": 1},
        },
    )
    write_json(root / "trials" / "trial_2.json", trial("t2"))
    write_json(root / "trials" / "trial_1.json", trial("t1"))
    if with_extras:
        (root / "environment.lock").write_text("numpy==2.0", encoding="utf-8")


def test_load_run_reads_sorted_trials_and_defaults(tmp_path):
    write_run(tmp_path)

    loaded = reproducibility.load_run(str(tmp_path))

    assert [t["trial_id"] for t in loaded["trials"]] == ["t1", "t2"]
    assert loaded["config"] == {"seed": 1}
    assert loaded["dataset"] == {"data_hash": "abc"}
    assert loaded["split"] == "split-1"
    assert loaded["selected"] == (
        "ridge",
        "single",
        '{"a": 1, "b": 2}',
    )
    assert loaded["environment"] == ""
    assert loaded["parameter_estimation"] is None


def test_load_run_reads_environment_lock(tmp_path):
    write_run(tmp_path, with_extras=True)

    loaded = reproducibility.load_run(tmp_path)

    assert loaded["environment"] == "numpy==2.0"


def test_load_run_digests_parameter_estimation(tmp_path):
    write_run(tmp_path)
    rng = np.random.default_rng(0)
    features = rng.normal(size=(60, 2))
    outcome = features @ np.array([1.0, -0.5]) + rng.normal(size=60) * 0.1
    result = parameter_estimation.estimate_parameters(
        features,
        outcome,
        parameter_estimation.EstimandSpec(
            name="effect",
            outcome="y",
            population="synthetic rows",
            likelihood="gaussian",
            feature_names=("x1", "x2"),
            assumptions=("independent errors",),
        ),
        parameter_estimation.EstimationPolicy(estimator="linear"),
    )
    expected = result.write(tmp_path / "parameter_estimation.json")

    loaded = reproducibility.load_run(tmp_path)

    assert loaded["parameter_estimation"] == expected


def test_loaded_reruns_compare_exact_and_detect_new_evidence(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    write_run(first, with_extras=True)
    write_run(second, with_extras=True)

    same = reproducibility.compare_runs(
        reproducibility.load_run(first), reproducibility.load_run(second)
    )
    write_json(second / "config.json", {"seed": 2})
    changed = reproducibility.compare_runs(
        reproducibility.load_run(first), reproducibility.load_run(second)
    )

    assert same.level == reproducibility.EXACT
    assert same.trials_compared == 2
    assert changed.level == reproducibility.DIFFERENT
    assert changed.reasons == ["configurations differ"]
