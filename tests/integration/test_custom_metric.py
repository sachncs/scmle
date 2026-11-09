"""User-defined metrics drive search, selection and reporting."""

import metric_functions
import pytest
import support

import scmle
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.evaluation import metrics


@pytest.fixture
def registered():
    metric = scmle.register_metric(
        "business_cost",
        metric_functions.business_cost,
        greater_is_better=False,
        task_types=(tasks.TaskType.BINARY,),
    )
    yield metric
    metrics.CATALOG.unregister("business_cost")


def test_a_custom_cost_metric_is_optimised_and_reported(
    registered, binary_frame, tmp_path
):
    result = support.make_project(
        binary_frame,
        "y",
        metric="business_cost",
        time_budget=300,
        output_dir=tmp_path,
    ).run()

    assert result.evaluation["metric"] == "business_cost"
    assert result.evaluation["metric_greater_is_better"] is False
    assert "business_cost" in result.evaluation["test_metrics"]
    board = result.leaderboard()
    best = board[board["status"] == "ok"]["score_mean"]
    assert best.iloc[0] == best.min()


def test_registering_twice_needs_replace_and_unknown_tasks_are_rejected(
    registered,
):
    with pytest.raises(errors.ConfigError, match="already exists"):
        scmle.register_metric(
            "business_cost", metric_functions.business_cost, False
        )
    scmle.register_metric(
        "business_cost",
        metric_functions.business_cost,
        False,
        task_types=(tasks.TaskType.BINARY,),
        replace=True,
    )
    with pytest.raises(errors.ConfigError, match="not defined"):
        metrics.CATALOG.resolve("business_cost", tasks.TaskType.REGRESSION)


def test_a_regression_metric_can_be_user_defined(tmp_path):
    import numpy
    import pandas

    rng = numpy.random.default_rng(2)
    frame = pandas.DataFrame({"x": rng.uniform(1, 5, 400)})
    frame["y"] = 3 * frame.x + rng.normal(0, 0.3, 400)
    scmle.register_metric(
        "mape",
        metric_functions.mean_absolute_percentage,
        greater_is_better=False,
        task_types=(tasks.TaskType.REGRESSION,),
    )
    try:
        result = support.make_project(
            frame, "y", metric="mape", time_budget=300, output_dir=tmp_path
        ).run()
    finally:
        metrics.CATALOG.unregister("mape")

    assert result.evaluation["test_metrics"]["mape"] < 0.2
