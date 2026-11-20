"""Deep-learning adapters without PyTorch: refusals, settings, pipelines."""

import pytest

from scmle.contracts import data
from scmle.contracts import models
from scmle.contracts import tasks
from scmle.models.deeplearning import estimators
from scmle.models.deeplearning import networks
from scmle.runtime import worker

TOO_MANY = 10**9


class Probe:
    """A worker client stub that sees CUDA only when asked for `probe`.

    Attributes:
        calls: The requests it received.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def request(self, name: str, args: tuple) -> bool:
        """Records the request and answers it.

        Args:
            name: The worker command.
            args: Its arguments.

        Returns:
            True only for `probe` with no arguments.
        """
        self.calls.append((name, args))
        return name == "probe" and args == ()


@pytest.fixture
def installed(monkeypatch):
    """Makes every adapter see its dependencies as installed."""
    monkeypatch.setattr(models.Model, "is_available", lambda self: True)


@pytest.fixture
def probe(monkeypatch, installed):
    found = Probe()
    monkeypatch.setattr(worker, "client", lambda: found)
    return found


def test_network_keeps_the_given_declaration() -> None:
    network = networks.Network(
        name="custom", architecture="mlp", capability_family="transformer"
    )
    assert (network.name, network.architecture) == ("custom", "mlp")
    assert network.boundary == "transformer"
    assert network.contract is networks.Network.contract


@pytest.mark.parametrize("model", [networks.MLP, networks.Tabpfn()])
def test_accelerated_asks_the_worker_to_probe(probe, model) -> None:
    assert model.accelerated() is True
    assert probe.calls == [("probe", ())]


def test_jepa_reports_a_missing_dependency_first(monkeypatch) -> None:
    monkeypatch.setattr(models.Model, "is_available", lambda self: False)
    assert networks.Jepa().objection(tasks.Task.BINARY, 4) == (
        "missing dependency: torch"
    )


def test_jepa_refuses_too_few_rows(installed) -> None:
    assert networks.Jepa().objection(tasks.Task.BINARY, 15) == (
        "tabular JEPA requires at least 16 rows for its training and "
        "validation boundaries"
    )
    assert networks.Jepa().objection(tasks.Task.BINARY, 16) is None


def test_tabpfn_reports_the_base_reason_first(installed) -> None:
    found = networks.Tabpfn().objection(tasks.Task.BINARY, TOO_MANY)
    assert found is not None
    assert found.startswith("dataset too large")


def test_tabpfn_is_extension_only(installed) -> None:
    # Enforces website/src/pages/docs/modeling.astro:51.
    assert networks.Tabpfn().objection(tasks.Task.BINARY, 10) == (
        "foundation model is extension-only until accepted weight license, "
        "token custody, and complete release evidence exist"
    )


@pytest.mark.parametrize(
    ("task", "head"),
    [
        (tasks.Task.BINARY, estimators.TorchClassifier),
        (tasks.Task.REGRESSION, estimators.TorchRegressor),
    ],
)
def test_jepa_shares_its_settings_between_steps(installed, task, head) -> None:
    params = {
        "batch_size": 64,
        "variance_weight": 0.2,
        "learning_rate": 0.01,
        "dropout": 0.3,
    }
    found = networks.Jepa().build(task, 7, params)
    representation = found.named_steps["representation"].get_params()
    model = found.named_steps["model"]
    assert isinstance(model, head)
    assert (
        representation["batch_size"],
        representation["variance_weight"],
        representation["learning_rate"],
        representation["random_state"],
    ) == (64, 0.2, 0.01, 7)
    settings = model.get_params()
    assert (
        settings["batch_size"],
        settings["learning_rate"],
        settings["hidden_sizes"],
        settings["random_state"],
        settings["dropout"],
    ) == (64, 0.01, (32,), 7, 0.1)


def test_jepa_assembles_the_front_end_first(installed) -> None:
    plan = data.Blueprint(("x",), ())
    found = networks.Jepa().assemble(plan, tasks.Task.REGRESSION, 3)
    assert [step[0] for step in found.steps] == [
        "prep",
        "representation",
        "model",
    ]


def test_keras_seeds_its_estimator(installed) -> None:
    found = networks.Keras().build(tasks.Task.REGRESSION, 5)
    assert found.get_params()["random_state"] == 5
