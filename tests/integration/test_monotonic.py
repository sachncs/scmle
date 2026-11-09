"""Monotonic constraints are enforced, not just requested."""

import numpy
import pandas
import pytest
import support

from scmle.contracts import errors
from scmle.contracts import tasks


def noisy_frame(rows=900, seed=0):
    """Risk rises with `missed` overall, but the data has a local dip."""
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame(
        {
            "missed": rng.uniform(0, 10, rows),
            "age": rng.uniform(20, 70, rows),
        }
    )
    dip = -1.5 * numpy.exp(-((frame.missed - 5) ** 2))
    score = 0.6 * frame.missed + dip + rng.normal(0, 1.2, rows) - 3
    frame["y"] = (score > 0).astype(int)
    return frame


def sweep(result, column="missed"):
    grid = pandas.DataFrame({column: numpy.linspace(0, 10, 60)})
    grid["age"] = 40.0
    return result.bundle.predict_proba(grid)[:, 1]


def run(frame, tmp_path, **kwargs):
    return support.make_project(
        frame, "y", time_budget=300, output_dir=tmp_path, **kwargs
    ).run()


def test_constrained_predictions_never_fall_as_the_feature_rises(tmp_path):
    result = run(
        noisy_frame(),
        tmp_path,
        monotonic_constraints={"missed": 1},
        models=tasks.ModelSpec(exclude_families=["deep_learning"]),
    )

    curve = sweep(result)

    assert numpy.all(numpy.diff(curve) >= -1e-9)
    assert result.selected_model in {
        "hist_gradient_boosting",
        "lightgbm",
        "xgboost",
        "dummy",
    }


def test_only_models_that_can_enforce_the_constraint_are_planned(tmp_path):
    project = support.make_project(
        noisy_frame(),
        "y",
        time_budget=300,
        output_dir=tmp_path,
        monotonic_constraints={"missed": 1},
    )

    plan = project.plan()

    assert "logistic_regression" not in plan.candidate_models
    assert "stacking" not in plan.candidate_models
    assert "hist_gradient_boosting" in plan.candidate_models
    reasons = {d.technique: d.reason for d in plan.decisions}
    assert reasons["random_forest"] == "cannot enforce monotonic constraints"


def test_a_decreasing_constraint_is_respected_too(tmp_path):
    frame = noisy_frame()
    frame["y"] = 1 - frame["y"]

    result = run(frame, tmp_path, monotonic_constraints={"missed": -1})

    assert numpy.all(numpy.diff(sweep(result)) <= 1e-9)


def test_constraints_on_unusable_columns_or_multiclass_tasks_are_refused(
    tmp_path,
):
    with pytest.raises(errors.BlockedError, match="numeric feature"):
        support.make_project(
            noisy_frame(), "y", monotonic_constraints={"nope": 1}
        ).plan()
    with pytest.raises(ValueError, match=r"\+1 or -1"):
        tasks.ProjectSpec(
            data=tasks.DataSpec(
                uri="m", target="y", monotonic_constraints={"a": 2}
            )
        )
    multi = noisy_frame()
    multi["y"] = pandas.cut(multi.missed, 3, labels=False)
    with pytest.raises(errors.BlockedError, match="multiclass"):
        support.make_project(
            multi,
            "y",
            task="classification",
            monotonic_constraints={"missed": 1},
        ).plan()
