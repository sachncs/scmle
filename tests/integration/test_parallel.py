"""Parallel trial execution."""

import pytest
import support

import scml
from scml.contracts import errors
from scml.contracts import tasks
from scml.core import scheduler


def square(value):
    return value * value


def test_backends_preserve_order_and_results():
    items = list(range(10))
    assert scheduler.SequentialBackend().map(square, items) == [
        i * i for i in items
    ]
    assert scheduler.ProcessBackend(2).map(square, items) == [
        i * i for i in items
    ]
    assert scheduler.ProcessBackend(2).map(square, [3]) == [9]
    assert scheduler.ProcessBackend(2).map(square, []) == []


def test_backend_factory_and_validation():
    assert isinstance(scheduler.create_backend(1), scheduler.SequentialBackend)
    pool = scheduler.create_backend(3)
    assert isinstance(pool, scheduler.ProcessBackend) and pool.workers == 3
    with pytest.raises(errors.ConfigError):
        scheduler.ProcessBackend(1)


def test_parallel_run_matches_sequential_run(binary_frame, tmp_path):
    search = tasks.SearchSpec(top_models=1, configs_per_model=4)

    def run(workers):
        return support.make_project(
            binary_frame,
            "y",
            time_budget=300,
            workers=workers,
            search=search,
            output_dir=tmp_path / f"w{workers}",
        ).run()

    sequential, parallel = run(1), run(2)
    assert [t.trial_id for t in sequential.trials] == [
        t.trial_id for t in parallel.trials
    ]
    assert [t.params for t in sequential.trials] == [
        t.params for t in parallel.trials
    ]
    for a, b in zip(sequential.trials, parallel.trials, strict=True):
        assert a.status == b.status
        assert a.score_mean == pytest.approx(b.score_mean, nan_ok=True)
    assert sequential.selected_model == parallel.selected_model
    assert (
        sequential.evaluation["selected_params"]
        == parallel.evaluation["selected_params"]
    )
    assert sequential.evaluation["test_metrics"] == pytest.approx(
        parallel.evaluation["test_metrics"]
    )


def test_workers_participate_in_the_config_hash(binary_frame):
    one = scml.MLProject(binary_frame, "y").spec.config_hash()
    two = scml.MLProject(binary_frame, "y", workers=2).spec.config_hash()
    assert one != two
