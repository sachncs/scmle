"""Trials spread over a multi-node Ray cluster give single-node results."""

import dataclasses
import importlib.util
import os
import sys

import pytest
import support

from mml.integrations import ray_backend
from mml.models import registry as registry_module

if importlib.util.find_spec("ray") is not None:
    import ray.cloudpickle

    # Workers cannot import this test module, so ship its functions by value.
    ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ray") is None, reason="ray is not installed"
)


def square(value):
    return value * value


def die_on_three(value):
    if value == 3:
        os._exit(1)  # The worker process vanishes without a word.
    return value * 10


@pytest.fixture(scope="module")
def cluster():
    """Two Ray nodes on this machine, joined through one head."""
    import ray
    from ray.cluster_utils import Cluster

    if ray.is_initialized():
        ray.shutdown()
    tests_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [tests_dir, os.environ.get("PYTHONPATH", "")]
    )
    fake = Cluster(
        initialize_head=True,
        head_node_args={"num_cpus": 1, "include_dashboard": False},
    )
    fake.add_node(num_cpus=1)
    yield fake
    ray.shutdown()
    fake.shutdown()


@pytest.fixture
def backend(cluster):
    import ray

    if ray.is_initialized():
        ray.shutdown()
    return ray_backend.RayBackend(2, address=cluster.address)


def test_tasks_run_on_every_node_and_come_back_in_order(backend, cluster):
    import ray

    assert len([n for n in ray.nodes() if n["Alive"]]) == 2

    assert backend.map(square, [1, 2, 3, 4, 5, 6]) == [1, 4, 9, 16, 25, 36]


def test_a_lost_worker_becomes_a_failed_item_not_a_dead_run(backend):
    seen = []

    def lost(item, error):
        seen.append((item, type(error).__name__))
        return -1

    outputs = backend.map(die_on_three, [1, 2, 3, 4], on_failure=lost)

    assert outputs == [10, 20, -1, 40]
    assert seen and seen[0][0] == 3


def test_without_a_failure_handler_the_error_surfaces(backend):
    import ray

    with pytest.raises(ray.exceptions.RayError):
        backend.map(die_on_three, [3])


def test_a_multi_node_run_reproduces_the_single_node_trials(
    binary_frame, tmp_path, cluster
):
    import ray

    if ray.is_initialized():
        ray.shutdown()
    local = support.make_project(
        binary_frame, "y", time_budget=300, output_dir=tmp_path / "local"
    ).run()
    spread = support.make_project(
        binary_frame,
        "y",
        time_budget=300,
        output_dir=tmp_path / "ray",
        executor="ray",
        cluster_address=cluster.address,
        workers=2,
    ).run()

    def scores(result):
        return {
            (t.model_name, t.strategy): t.fold_scores
            for t in result.trials
            if t.fold_scores
        }

    assert scores(local) == pytest.approx(scores(spread))
    assert local.selected_model == spread.selected_model
    assert local.evaluation["execution_backend"] == "sequential"
    assert spread.evaluation["execution_backend"] == "ray"


def test_a_trial_past_the_memory_limit_is_killed_on_a_ray_worker(
    binary_frame, tmp_path, cluster
):
    import ray

    if ray.is_initialized():
        ray.shutdown()
    project = support.make_project(
        binary_frame,
        "y",
        time_budget=300,
        output_dir=tmp_path,
        executor="ray",
        cluster_address=cluster.address,
        workers=2,
        memory_limit_gb=1.0,
        enforcement="hard",
    )
    registry = registry_module.ModelRegistry.with_builtins()
    registry.register(support.HungryAdapter())
    project.orchestrator.registry = registry
    plan = dataclasses.replace(
        project.plan(), candidate_models=("dummy", "hungry")
    )

    result = project.run(plan)

    hungry = [t for t in result.trials if t.model_name == "hungry"]
    assert hungry and hungry[0].status == "failed"
    assert "memory limit" in hungry[0].error
    assert result.state == "COMPLETED" and result.selected_model == "dummy"
