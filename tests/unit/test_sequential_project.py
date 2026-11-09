"""Unit tests for the durable sequential estimation project API."""

import dataclasses

import numpy
import pandas
import pytest

from scmle.api.sequential import resolve_tabular_input
from scmle.api.sequential import SequentialEstimationProject
from scmle.evaluation.sequential import SequentialEstimand
from scmle.evaluation.sequential import SequentialEstimationPolicy


def make_estimand() -> SequentialEstimand:
    """Return a complete sequential causal declaration."""
    return SequentialEstimand(
        name="sustained_treatment_effect",
        subject_identity="subject_id",
        time_column="time",
        treatment_column="treatment",
        outcome="outcome",
        baseline_covariates=("baseline_score",),
        time_varying_covariates=("state",),
        population="subjects with complete follow-up",
        assumptions=(
            "consistency",
            "positivity",
            "sequential_exchangeability",
            "no_unmeasured_time_varying_confounding",
            "time_varying_covariates_are_pre_treatment",
            "complete_follow_up",
        ),
    )


def make_policy() -> SequentialEstimationPolicy:
    """Return a bounded policy suitable for the small test panel."""
    return SequentialEstimationPolicy(
        minimum_subjects=60,
        minimum_effective_sample_size=2.0,
    )


def make_frame() -> pandas.DataFrame:
    """Return a deterministic balanced panel with treatment variation."""
    generator = numpy.random.default_rng(23)
    rows = []
    for subject_id in range(80):
        baseline_score = float(generator.normal())
        previous_treatment = 0
        for time in range(3):
            state = float(
                0.25 * baseline_score
                + 0.3 * previous_treatment
                + generator.normal(scale=0.4)
            )
            probability = 1.0 / (
                1.0 + numpy.exp(-(-0.1 + 0.2 * baseline_score + 0.2 * state))
            )
            treatment = int(generator.random() < probability)
            outcome = float(
                1.0
                + 1.3 * treatment
                + 0.3 * state
                + 0.2 * baseline_score
                + generator.normal(scale=0.4)
            )
            rows.append(
                {
                    "subject_id": subject_id,
                    "time": time,
                    "treatment": treatment,
                    "outcome": outcome,
                    "baseline_score": baseline_score,
                    "state": state,
                }
            )
            previous_treatment = treatment
    return pandas.DataFrame(rows)


def test_project_requires_explicit_estimator_declarations() -> None:
    """The project cannot silently choose a sequential estimand or policy."""
    frame = make_frame()
    with pytest.raises(TypeError, match="SequentialEstimand"):
        SequentialEstimationProject(frame, object(), make_policy())
    with pytest.raises(TypeError, match="SequentialEstimationPolicy"):
        SequentialEstimationProject(frame, make_estimand(), object())
    with pytest.raises(TypeError, match="supported tabular"):
        SequentialEstimationProject(object(), make_estimand(), make_policy())


def test_resolve_tabular_input_copies_frames_and_loads_supported_paths(
    tmp_path,
) -> None:
    """Input resolution owns in-memory data and delegates file formats."""
    frame = make_frame()
    resolved = resolve_tabular_input(frame)
    resolved.loc[0, "outcome"] = -999.0
    assert frame.loc[0, "outcome"] != -999.0

    path = tmp_path / "panel.csv"
    frame.to_csv(path, index=False)
    loaded = resolve_tabular_input(path)
    assert loaded.shape == frame.shape
    assert list(loaded.columns) == list(frame.columns)


def test_project_result_can_be_loaded_from_a_verified_artifact(
    tmp_path,
) -> None:
    """A new project instance can expose its digest-verified durable result."""
    project = SequentialEstimationProject(
        make_frame(), make_estimand(), make_policy(), output_dir=tmp_path
    )
    result = project.run()
    restored = SequentialEstimationProject(
        make_frame(), make_estimand(), make_policy(), output_dir=tmp_path
    ).load_result()

    assert result.digest() == restored.digest()
    assert project.artifact_path.name == "sequential_estimation.json"


def test_replay_detects_policy_changes(tmp_path) -> None:
    """Replay verification reports a changed policy instead of hiding it."""
    policy = make_policy()
    project = SequentialEstimationProject(
        make_frame(), make_estimand(), policy, output_dir=tmp_path
    )
    project.run()
    changed = dataclasses.replace(policy, maximum_weight=10.0)
    replay = SequentialEstimationProject(
        make_frame(), make_estimand(), changed, output_dir=project.output_dir
    ).verify_replay()

    assert replay.level == "different"
    assert "policy" in " ".join(replay.reasons)
