"""Survival adapters build Cox and discrete-time hazard models."""

import numpy

from scmle.contracts import search
from scmle.contracts import tasks
from scmle.models.survival import adapters
from scmle.models.survival import estimators


def test_a_cox_adapter_builds_a_penalised_model():
    adapter = adapters.Cox()
    built = adapter.build(tasks.Task.SURVIVAL, 9, {"penalty": 3.0})
    assert isinstance(built, estimators.Hazards)
    assert built.penalty == 3.0
    assert adapter.build(tasks.Task.SURVIVAL, 9).penalty == 1.0
    assert adapter.name == "cox_ph"
    assert adapter.capabilities.needs_scaling is True
    assert adapter.capabilities.cost == 2
    assert adapter.search_space.names() == ("penalty",)
    assert isinstance(adapter.search_space.parameters[0], search.Real)


def test_a_hazard_adapter_builds_seeded_discrete_time_boosting():
    adapter = adapters.Hazard()
    built = adapter.build(tasks.Task.SURVIVAL, 4, {"learning_rate": 0.05})
    assert isinstance(built, estimators.DiscreteTimeBoosting)
    assert built.random_state == 4
    assert built.learning_rate == 0.05
    assert adapter.name == "discrete_time_boosting"
    assert adapter.capabilities.handles_missing is False
    assert adapter.capabilities.cost == 4
    assert adapter.capabilities.memory_factor == 4.0
    assert (adapters.Cox, adapters.Hazard) == adapters.SURVIVAL_ADAPTERS
    assert {tasks.Task.SURVIVAL} == adapters.SURVIVAL


def test_a_cox_model_fits_a_known_hazard_direction():
    rng = numpy.random.default_rng(0)
    x = rng.normal(size=(200, 1))
    duration = rng.exponential(numpy.exp(-x[:, 0]))
    model = (
        adapters.Cox()
        .build(tasks.Task.SURVIVAL, 0)
        .fit(x, duration, numpy.ones(200))
    )
    risk = model.predict(x)
    assert numpy.corrcoef(risk, x[:, 0])[0, 1] > 0.9
