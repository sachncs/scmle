"""The PyTorch training engines, run on a numpy stand-in for torch.

PyTorch is not installed in the framework process, so the engines run on a
recording stand-in: tensors are numpy arrays with the few torch methods the
engines call, and the network is a fixed linear map, so every loss, weight,
learning rate and stopping decision can be checked exactly.
"""

import contextlib
import dataclasses
import math
import pickle
import sys
import types

import numpy
import pytest
from scipy import sparse

from scmle.contracts import errors
from scmle.models.deeplearning import architectures
from scmle.models.deeplearning.engines import descent
from scmle.models.deeplearning.engines import heads
from scmle.models.deeplearning.engines import network
from scmle.models.deeplearning.engines import trainer
import scmle.models.deeplearning.training


class Tensor(numpy.ndarray):
    """A numpy array with the torch tensor methods the engines call."""

    backwards: list[float] = []

    def dim(self):
        return self.ndim

    def unsqueeze(self, dim):
        return numpy.expand_dims(self, dim).view(Tensor)

    def repeat_interleave(self, count):
        return numpy.repeat(numpy.asarray(self), count).view(Tensor)

    def mean(
        self,
        axis=None,
        dtype=None,
        out=None,
        keepdims=False,
        *,
        dim=None,
        where=True,
    ):
        axis = dim if dim is not None else axis
        return (
            numpy.asarray(self).mean(axis=axis, keepdims=keepdims).view(Tensor)
        )

    def argmax(
        self, axis=None, out=None, *, dim=None, keepdim=False, keepdims=False
    ):
        axis = dim if dim is not None else axis
        return (
            numpy.asarray(self)
            .argmax(axis=axis, keepdims=keepdim or keepdims)
            .view(Tensor)
        )

    def gather(self, dim, index):
        return numpy.take_along_axis(
            numpy.asarray(self), numpy.asarray(index), axis=dim
        ).view(Tensor)

    def backward(self):
        Tensor.backwards.append(float(self))

    def detach(self):
        return self

    def numpy(self):
        return numpy.asarray(self)

    def to(self, device):
        return self

    def requires_grad_(self, requires_grad):
        assert requires_grad is True
        return self


def tensor(values, dtype=float):
    return numpy.asarray(values, dtype=dtype).view(Tensor)


class Net:
    """A linear network `x @ weight` that records how it is driven.

    Attributes:
        weight: The `(inputs, outputs)` matrix.
        modes: Every `train`/`eval` switch, in order.
        devices: Every device it was moved to.
        loaded: Every state dict loaded into it.
    """

    def __init__(self, inputs, outputs, members=0):
        self.weight = numpy.zeros((inputs, outputs))
        self.members = members
        self.modes = []
        self.devices = []
        self.loaded = []

    def __call__(self, features):
        outputs = numpy.asarray(features) @ self.weight
        if self.members:
            outputs = numpy.repeat(outputs[:, None, :], self.members, axis=1)
        return outputs.view(Tensor)

    def parameters(self):
        return ["weight"]

    def to(self, device):
        self.devices.append(device)
        return self

    def train(self):
        self.modes.append("train")
        return self

    def eval(self):
        self.modes.append("eval")
        return self

    def cpu(self):
        self.devices.append("cpu")
        return self

    def state_dict(self):
        return {"weight": self.weight.copy().view(Tensor)}

    def load_state_dict(self, state):
        self.loaded.append(state)
        self.weight = numpy.asarray(state["weight"]).copy()


class Linear(architectures.Architecture):
    """An architecture that builds `Net` and records its arguments."""

    name = "linear"

    def __init__(self):
        self.calls = []

    def build(self, torch, inputs, outputs, hidden_sizes, dropout):
        self.calls.append((inputs, outputs, hidden_sizes, dropout))
        return Net(inputs, outputs)


class Optimizer:
    """A stand-in optimizer with two parameter groups.

    Attributes:
        kind: The constructor name.
        parameters: The parameters it was given.
        options: The constructor keyword arguments.
        zeroed: How often `zero_grad` ran.
        steps: How often `step` ran.
    """

    def __init__(self, kind, parameters, **options):
        self.kind = kind
        self.parameters = parameters
        self.options = options
        self.param_groups = [{"lr": None}, {"lr": None}]
        self.zeroed = 0
        self.steps = 0
        self.rates = []

    def zero_grad(self):
        self.zeroed += 1

    def step(self, closure=None):
        self.steps += 1
        self.rates.append(self.param_groups[0]["lr"])
        if closure is not None:
            return closure()
        return None


class Clip:
    """`clip_grad_norm_`: records the limit and returns the next norm."""

    def __init__(self, norms):
        self.norms = list(norms)
        self.limits = []

    def __call__(self, parameters, limit):
        self.limits.append(limit)
        return self.norms.pop(0) if len(self.norms) > 1 else self.norms[0]


def stand_in(norms=(1.0,), cuda=False):
    """A fresh stand-in for the torch module."""
    seeds = []
    graded = []

    def optimizer(kind):
        return lambda parameters, **options: Optimizer(
            kind, parameters, **options
        )

    def cross_entropy(outputs, targets, reduction, label_smoothing):
        assert reduction == "none"
        graded.append(("cross_entropy", label_smoothing))
        shifted = numpy.asarray(outputs) - numpy.asarray(outputs).max(
            axis=1, keepdims=True
        )
        log = shifted - numpy.log(numpy.exp(shifted).sum(axis=1, keepdims=True))
        return (-log[numpy.arange(len(targets)), numpy.asarray(targets)]).view(
            Tensor
        )

    def huber_loss(outputs, targets, reduction, delta):
        assert reduction == "none"
        graded.append(("huber", delta, numpy.asarray(outputs).shape))
        error = numpy.abs(numpy.asarray(outputs) - numpy.asarray(targets))
        return numpy.where(
            error < delta, 0.5 * error**2, delta * (error - 0.5 * delta)
        ).view(Tensor)

    def grad(output, inputs):
        graded.append(("grad", float(output)))
        return (tensor(numpy.full(inputs.shape, 2.0)),)

    module = types.ModuleType("torch")
    module.seeds = seeds
    module.graded = graded
    module.clip = Clip(norms)
    module.manual_seed = seeds.append
    module.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    module.from_numpy = lambda array: numpy.asarray(array).view(Tensor)
    module.no_grad = contextlib.nullcontext
    module.exp = numpy.exp
    module.autograd = types.SimpleNamespace(grad=grad)
    module.optim = types.SimpleNamespace(
        Adam=optimizer("Adam"),
        AdamW=optimizer("AdamW"),
        SGD=optimizer("SGD"),
        RMSprop=optimizer("RMSprop"),
        LBFGS=optimizer("LBFGS"),
    )
    module.nn = types.SimpleNamespace(
        utils=types.SimpleNamespace(clip_grad_norm_=module.clip),
        functional=types.SimpleNamespace(
            cross_entropy=cross_entropy, huber_loss=huber_loss
        ),
    )
    return module


@pytest.fixture
def torch(monkeypatch):
    module = stand_in()
    monkeypatch.setitem(sys.modules, "torch", module)
    Tensor.backwards = []
    return module


@pytest.fixture
def linear(monkeypatch):
    architecture = Linear()
    monkeypatch.setitem(architectures.ARCHITECTURES, "linear", architecture)
    return architecture


def alternating(rows):
    """Values whose standardization is exactly -1, +1, -1, ..."""
    return numpy.array([0.0, 2.0] * (rows // 2))


# network


def test_the_network_constants_are_fixed():
    assert network.MIN_ROWS_FOR_VALIDATION == 30
    assert network.SINGLE_MEMBER_OUTPUT_DIMS == 2


def test_a_plan_is_frozen():
    plan = network.Plan(8, False, 2, 3, 30, 3)

    assert dataclasses.asdict(plan) == {
        "rows": 8,
        "order": False,
        "accumulate": 2,
        "updates": 3,
        "steps": 30,
        "warmup": 3,
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.rows = 9


def test_the_engines_keep_every_hyperparameter():
    found = heads.Classifier().get_params()

    assert found == {
        "architecture": "mlp",
        "hidden_sizes": (64,),
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "epochs": 100,
        "batch_size": 128,
        "schedule": "cosine",
        "warmup_epochs": 1,
        "max_grad_norm": 5.0,
        "patience": 10,
        "validation_fraction": 0.1,
        "device": "auto",
        "random_state": 0,
        "optimizer": "adam",
        "momentum": 0.9,
        "loss": "default",
        "label_smoothing": 0.0,
        "focal_gamma": 2.0,
        "huber_delta": 1.0,
        "accumulation_steps": 1,
        "class_weight": None,
    }
    regressor = heads.Regressor(
        optimizer="sgd", momentum=0.5, loss="huber", huber_delta=2.0
    )
    assert regressor.get_params()["optimizer"] == "sgd"
    assert regressor.get_params()["momentum"] == 0.5
    assert regressor.get_params()["loss"] == "huber"
    assert regressor.get_params()["huber_delta"] == 2.0
    assert "class_weight" not in regressor.get_params()


def test_the_engines_report_the_public_module():
    assert heads.Classifier.__module__ == "scmle.models.deeplearning.engines"
    assert heads.Regressor.__module__ == "scmle.models.deeplearning.engines"
    assert trainer.Trainer.__module__ == "scmle.models.deeplearning.engines"
    assert heads.Classifier.losses == ("default", "focal")
    assert heads.Regressor.losses == ("default", "huber")


@pytest.mark.parametrize(
    ("device", "cuda", "expected"),
    [
        ("auto", True, "cuda"),
        ("auto", False, "cpu"),
        ("cpu", True, "cpu"),
        ("cuda", False, "cuda"),
    ],
)
def test_place_resolves_auto_by_cuda_availability(device, cuda, expected):
    assert heads.Regressor(device=device).place(stand_in(cuda=cuda)) == expected


def test_dense_turns_sparse_and_integer_input_into_float32():
    found = network.Network.dense(sparse.csr_matrix([[0, 1], [2, 0]]))

    assert found.dtype == numpy.float32
    assert found.tolist() == [[0.0, 1.0], [2.0, 0.0]]
    assert network.Network.dense([[1, 2]]).dtype == numpy.float32


def test_construct_sizes_the_architecture_from_the_fit(linear):
    engine = heads.Classifier(
        architecture="linear", hidden_sizes=[3, 2], dropout=0.4
    )
    engine.vectorize(["a", "b", "c"])
    engine.input_width_ = 5

    built = engine.construct(None)

    assert linear.calls == [(5, 3, (3, 2), 0.4)]
    assert built.weight.shape == (5, 3)


def test_a_pickle_stores_weights_as_arrays_and_rebuilds_the_network(
    torch, linear
):
    engine = heads.Regressor(architecture="linear")
    engine.input_width_ = 2
    engine.network_ = Net(2, 1)
    engine.network_.weight = numpy.array([[1.5], [-2.0]])

    state = engine.__getstate__()

    assert "network_" not in state
    assert type(state["weights_"]["weight"]) is numpy.ndarray
    assert state["weights_"]["weight"].tolist() == [[1.5], [-2.0]]
    restored = pickle.loads(pickle.dumps(engine))
    assert restored.network_.weight.tolist() == [[1.5], [-2.0]]
    assert restored.network_.modes == ["eval"]
    assert len(restored.network_.loaded) == 1


def test_an_unfitted_engine_pickles_without_weights():
    engine = heads.Regressor(epochs=3)

    assert "weights_" not in engine.__getstate__()
    restored = pickle.loads(pickle.dumps(engine))
    assert restored.epochs == 3
    assert not hasattr(restored, "network_")


def test_the_weighted_loss_averages_rows_by_their_weights(torch):
    engine = heads.Regressor()
    net = Net(1, 1)
    net.weight = numpy.array([[1.0]])
    tensors = {
        "x": tensor([[1.0], [2.0], [3.0]]),
        "y": tensor([0.0, 0.0, 1.0]),
        "w": tensor([1.0, 3.0, 100.0]),
    }

    found = engine.weigh(torch, net, tensors, numpy.array([0, 1]))

    assert float(found) == pytest.approx((1.0 * 1.0 + 3.0 * 4.0) / 4.0)
    everything = engine.weigh(torch, net, tensors, numpy.array([0, 1, 2]))
    assert float(everything) == pytest.approx((1.0 + 12.0 + 400.0) / 104.0)


def test_the_weighted_loss_averages_ensemble_members_first(torch):
    engine = heads.Regressor()
    net = Net(1, 1, members=3)
    net.weight = numpy.array([[2.0]])
    tensors = {
        "x": tensor([[1.0], [1.0]]),
        "y": tensor([0.0, 1.0]),
        "w": tensor([1.0, 1.0]),
    }

    found = engine.weigh(torch, net, tensors, numpy.array([0, 1]))

    assert float(found) == pytest.approx((4.0 + 1.0) / 2.0)


# descent


def test_an_apportion_cuts_epochs_into_accumulated_updates():
    engine = heads.Regressor(
        batch_size=10, accumulation_steps=2, epochs=3, warmup_epochs=2
    )

    assert engine.apportion(25) == network.Plan(10, False, 2, 2, 6, 4)
    assert engine.apportion(20) == network.Plan(10, False, 2, 1, 3, 2)
    assert engine.apportion(21) == network.Plan(10, False, 2, 2, 6, 4)


def test_an_apportion_uses_one_full_batch_for_lbfgs():
    engine = heads.Regressor(
        optimizer="lbfgs",
        batch_size=4,
        accumulation_steps=5,
        epochs=7,
        warmup_epochs=0,
    )

    assert engine.apportion(25) == network.Plan(25, True, 1, 1, 7, 0)


def test_an_apportion_of_no_rows_still_plans_one_update():
    engine = heads.Regressor(batch_size=10, epochs=2)

    assert engine.apportion(0) == network.Plan(1, False, 1, 1, 2, 1)


def test_an_apportion_sizes_auto_batches_from_the_rows():
    engine = heads.Regressor(batch_size="auto")

    plan = engine.apportion(4096)

    assert (
        plan.rows
        == scmle.models.deeplearning.training.batching("auto", 4096)
        == 128
    )
    assert plan.updates == 32


def test_prepare_multiplies_class_and_sample_weights():
    engine = heads.Classifier(class_weight="balanced")

    matrix, encoded, weights = engine.prepare(
        [[1, 2], [3, 4], [5, 6], [7, 8]], ["a", "a", "a", "b"], [1, 2, 3, 4]
    )

    assert matrix.dtype == numpy.float32
    assert encoded.tolist() == [0, 0, 0, 1]
    assert weights.dtype == numpy.float32
    assert weights.tolist() == pytest.approx([2 / 3, 4 / 3, 2.0, 8.0])
    assert engine.prepare([[1]], ["a"], None)[2].tolist() == [1.0]


def test_adjust_sets_the_scheduled_rate_on_every_group():
    engine = heads.Regressor(learning_rate=0.1, schedule="cosine")
    optimizer = Optimizer("Adam", [])
    plan = network.Plan(8, False, 1, 5, 10, 2)

    engine.adjust(optimizer, 0, plan)
    assert [g["lr"] for g in optimizer.param_groups] == pytest.approx(
        [0.05, 0.05]
    )
    engine.adjust(optimizer, 6, plan)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.05)
    engine.adjust(optimizer, 10, plan)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)


def test_vet_accepts_known_settings():
    heads.Classifier(
        optimizer="lbfgs", loss="focal", accumulation_steps=1
    ).vet()
    heads.Regressor(optimizer="rmsprop", loss="huber").vet()
    assert descent.Descent.OPTIMIZERS == (
        "adam",
        "adamw",
        "sgd",
        "rmsprop",
        "lbfgs",
    )
    assert descent.Descent.CLASSIFIER_LOSSES == ("default", "focal")
    assert descent.Descent.REGRESSOR_LOSSES == ("default", "huber")
    assert descent.Descent.losses == ("default",)


@pytest.mark.parametrize(
    ("engine", "message"),
    [
        (
            heads.Regressor(optimizer="adagrad"),
            "unknown optimizer 'adagrad'; use one of ['adam', 'adamw', 'sgd', 'rmsprop', 'lbfgs']",
        ),
        (
            heads.Regressor(loss="focal"),
            "unknown loss 'focal' for this task; use one of ['default', 'huber']",
        ),
        (
            heads.Classifier(loss="huber"),
            "unknown loss 'huber' for this task; use one of ['default', 'focal']",
        ),
        (
            heads.Regressor(accumulation_steps=0),
            "accumulation_steps must be at least 1",
        ),
    ],
)
def test_vet_rejects_unknown_settings(engine, message):
    with pytest.raises(errors.ConfigError) as caught:
        engine.vet()

    assert str(caught.value) == message


@pytest.mark.parametrize(
    ("settings", "kind", "options"),
    [
        ({"optimizer": "adam"}, "Adam", {"lr": 0.1, "weight_decay": 0.01}),
        ({"optimizer": "adamw"}, "AdamW", {"lr": 0.1, "weight_decay": 0.01}),
        (
            {"optimizer": "rmsprop"},
            "RMSprop",
            {"lr": 0.1, "weight_decay": 0.01},
        ),
        (
            {"optimizer": "sgd", "momentum": 0.5},
            "SGD",
            {
                "lr": 0.1,
                "momentum": 0.5,
                "nesterov": True,
                "weight_decay": 0.01,
            },
        ),
        (
            {"optimizer": "sgd", "momentum": 0.0},
            "SGD",
            {
                "lr": 0.1,
                "momentum": 0.0,
                "nesterov": False,
                "weight_decay": 0.01,
            },
        ),
        (
            {"optimizer": "lbfgs"},
            "LBFGS",
            {"lr": 1.0, "max_iter": 20, "line_search_fn": "strong_wolfe"},
        ),
        ({"optimizer": "other"}, "Adam", {"lr": 0.1, "weight_decay": 0.01}),
    ],
)
def test_instantiate_builds_the_configured_optimizer(settings, kind, options):
    engine = heads.Regressor(learning_rate=0.1, weight_decay=0.01, **settings)

    found = engine.instantiate(stand_in(), Net(1, 1))

    assert found.kind == kind
    assert found.options == options
    assert found.parameters == ["weight"]


def test_an_lbfgs_step_returns_the_starting_loss_and_the_unclipped_norm(torch):
    torch.clip.norms = [3.5]
    engine = heads.Regressor(optimizer="lbfgs")
    optimizer = Optimizer("LBFGS", [])
    tensors = {"x": tensor([[1.0]]), "y": tensor([2.0]), "w": tensor([1.0])}

    loss, norm = engine.lbfgs_step(
        torch, Net(1, 1), optimizer, tensors, numpy.array([0])
    )

    assert (loss, norm) == (4.0, 3.5)
    assert optimizer.zeroed == 1
    assert Tensor.backwards == [4.0]
    assert torch.clip.limits == [math.inf]


def epoch_inputs(rows):
    tensors = {
        "x": tensor(numpy.ones((rows, 1))),
        "y": tensor(numpy.arange(rows, dtype=float)),
        "w": tensor(numpy.ones(rows)),
    }
    return tensors, tensor(numpy.arange(rows), dtype=int)


def test_an_epoch_accumulates_batches_and_steps_on_the_last_partial_one(torch):
    torch.clip.norms = [7.0]
    engine = heads.Regressor(
        max_grad_norm=5.0, learning_rate=0.1, schedule="constant"
    )
    optimizer = Optimizer("Adam", [])
    monitor = scmle.models.deeplearning.training.Monitor()
    plan = network.Plan(2, False, 2, 2, 20, 0)
    tensors, rows = epoch_inputs(5)

    loss, alive, step = engine.train(
        torch, Net(1, 1), optimizer, tensors, monitor, plan, (rows, 3)
    )

    assert alive is True
    assert step == 5
    assert optimizer.steps == 2
    assert optimizer.zeroed == 2
    assert optimizer.rates == [0.1, 0.1]
    assert Tensor.backwards == pytest.approx([0.25, 6.5 / 2, 16.0 / 2])
    assert loss == pytest.approx((0.5 + 6.5) / 2 + 16.0)
    assert torch.clip.limits == [5.0, 5.0]
    assert monitor.norms == [7.0, 7.0]
    assert monitor.clipped == 2


def test_an_epoch_without_clipping_uses_an_infinite_limit(torch):
    engine = heads.Regressor(max_grad_norm=None)
    monitor = scmle.models.deeplearning.training.Monitor()
    tensors, rows = epoch_inputs(2)

    loss, alive, step = engine.train(
        torch,
        Net(1, 1),
        Optimizer("Adam", []),
        tensors,
        monitor,
        network.Plan(2, False, 1, 1, 1, 0),
        (rows, 0),
    )

    assert (loss, alive, step) == (0.5, True, 1)
    assert torch.clip.limits == [math.inf]
    assert monitor.clipped == 0


def test_an_epoch_stops_at_a_non_finite_gradient(torch):
    torch.clip.norms = [1.0, math.nan, 1.0]
    engine = heads.Regressor()
    optimizer = Optimizer("Adam", [])
    monitor = scmle.models.deeplearning.training.Monitor()
    tensors, rows = epoch_inputs(3)

    loss, alive, step = engine.train(
        torch,
        Net(1, 1),
        optimizer,
        tensors,
        monitor,
        network.Plan(1, False, 1, 3, 3, 0),
        (rows, 0),
    )

    assert alive is False
    assert step == 1
    assert loss == 0.0
    assert optimizer.steps == 1
    assert monitor.nonfinite is True


def test_a_second_order_epoch_takes_one_lbfgs_step_per_batch(torch):
    torch.clip.norms = [2.0]
    engine = heads.Regressor(optimizer="lbfgs")
    optimizer = Optimizer("LBFGS", [])
    monitor = scmle.models.deeplearning.training.Monitor()
    tensors, rows = epoch_inputs(3)

    loss, alive, step = engine.train(
        torch,
        Net(1, 1),
        optimizer,
        tensors,
        monitor,
        network.Plan(3, True, 1, 1, 1, 0),
        (rows, 4),
    )

    assert (alive, step) == (True, 5)
    assert loss == pytest.approx(5.0 / 3.0)
    assert optimizer.steps == 1
    assert monitor.norms == [2.0]
    assert monitor.clipped == 0


def test_a_second_order_epoch_stops_when_the_loss_diverges(torch):
    torch.clip.norms = [math.inf]
    engine = heads.Regressor(optimizer="lbfgs")
    tensors, rows = epoch_inputs(2)

    loss, alive, step = engine.train(
        torch,
        Net(1, 1),
        Optimizer("LBFGS", []),
        tensors,
        scmle.models.deeplearning.training.Monitor(),
        network.Plan(1, True, 1, 2, 2, 0),
        (rows, 0),
    )

    assert (loss, alive, step) == (0.0, False, 0)


# heads


def test_a_classifier_encodes_labels_and_sizes_one_unit_per_class():
    engine = heads.Classifier()

    codes = engine.vectorize(["b", "a", "c", "a"])

    assert codes.dtype == numpy.int64
    assert codes.tolist() == [1, 0, 2, 0]
    assert engine.classes_.tolist() == ["a", "b", "c"]
    assert engine.dimensionality() == 3


def test_a_classifier_balances_weights_by_inverse_class_frequency():
    engine = heads.Classifier(class_weight="balanced")
    engine.vectorize(["a", "b", "c"])

    found = engine.weighting(numpy.array([0, 0, 1]))

    assert found.dtype == numpy.float32
    assert found.tolist() == pytest.approx([0.5, 0.5, 1.0])
    assert heads.Classifier().weighting(numpy.array([0, 0, 1])).tolist() == [
        1.0,
        1.0,
        1.0,
    ]


def test_a_classifier_penalizes_with_smoothed_or_focal_cross_entropy():
    torch = stand_in()
    outputs = tensor([[0.0, math.log(3.0)], [0.0, 0.0]])
    targets = numpy.array([1, 0])
    plain = heads.Classifier(label_smoothing=0.2).penalize(
        torch, outputs, targets
    )

    assert plain.tolist() == pytest.approx([-math.log(0.75), math.log(2.0)])
    assert torch.graded == [("cross_entropy", 0.2)]
    focal = heads.Classifier(loss="focal", focal_gamma=3.0).penalize(
        torch, outputs, targets
    )
    assert focal.tolist() == pytest.approx(
        [0.25**3 * -math.log(0.75), 0.5**3 * math.log(2.0)]
    )


def test_a_classifier_averages_member_probabilities(monkeypatch):
    engine = heads.Classifier()
    engine.vectorize(["no", "yes"])
    raw = numpy.array(
        [[[0.0, math.log(3.0)], [0.0, 0.0]], [[1000.0, 0.0], [1000.0, 0.0]]]
    )
    monkeypatch.setattr(engine, "emit", lambda features: raw)

    found = engine.predict_proba([[0.0]])

    assert found.shape == (2, 2)
    assert found[0].tolist() == pytest.approx([0.375, 0.625])
    assert found[1].tolist() == pytest.approx([1.0, 0.0])
    assert engine.predict([[0.0]]).tolist() == ["yes", "no"]


def test_a_regressor_standardizes_its_target():
    engine = heads.Regressor()

    encoded = engine.vectorize([1.0, 3.0, 5.0])

    assert encoded.dtype == numpy.float32
    assert encoded.tolist() == pytest.approx(
        [-math.sqrt(1.5), 0.0, math.sqrt(1.5)]
    )
    assert engine.target_mean_ == 3.0
    assert engine.target_scale_ == pytest.approx(math.sqrt(8 / 3))
    assert engine.dimensionality() == 1
    assert engine.weighting(encoded).tolist() == [1.0, 1.0, 1.0]
    assert engine.weighting(encoded).dtype == numpy.float32


def test_a_regressor_uses_unit_scale_for_a_constant_target():
    engine = heads.Regressor()

    assert engine.vectorize([2.0, 2.0]).tolist() == [0.0, 0.0]
    assert engine.target_scale_ == 1.0


def test_a_regressor_penalizes_with_squared_or_huber_error():
    torch = stand_in()
    outputs = tensor([[0.0], [3.0]])
    targets = tensor([0.5, 0.0])

    squared = heads.Regressor().penalize(torch, outputs, targets)
    huber = heads.Regressor(loss="huber", huber_delta=1.0).penalize(
        torch, outputs, targets
    )

    assert squared.tolist() == [0.25, 9.0]
    assert huber.tolist() == [0.125, 2.5]
    assert torch.graded == [("huber", 1.0, (2,))]


def test_a_regressor_restores_the_original_units(monkeypatch):
    engine = heads.Regressor()
    engine.vectorize([0.0, 4.0])
    raw = numpy.array([[[1.0, 9.0], [3.0, 9.0]], [[-1.0, 9.0], [-1.0, 9.0]]])
    monkeypatch.setattr(engine, "emit", lambda features: raw)

    assert engine.predict([[0.0]]).tolist() == [6.0, 0.0]


# trainer


def test_divide_holds_out_a_share_only_from_thirty_rows():
    engine = heads.Regressor(validation_fraction=0.1, random_state=3)

    held, kept = engine.divide(30)
    order = numpy.random.default_rng(3).permutation(30)
    assert held.tolist() == order[:3].tolist()
    assert kept.tolist() == order[3:].tolist()

    small, rest = engine.divide(29)
    assert small.tolist() == []
    assert sorted(rest.tolist()) == list(range(29))


def test_a_validation_score_runs_the_network_in_evaluation_mode(torch):
    engine = heads.Regressor()
    net = Net(1, 1)
    tensors = {
        "x": tensor([[1.0], [1.0]]),
        "y": tensor([1.0, 3.0]),
        "w": tensor([1.0, 1.0]),
    }

    found = engine.evaluate(torch, net, tensors, numpy.array([0, 1]))

    assert found == 5.0
    assert type(found) is float
    assert net.modes == ["eval"]


def test_fit_rejects_an_unknown_architecture_before_importing_torch(
    monkeypatch,
):
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(errors.ConfigError, match="unknown architecture 'nope'"):
        heads.Regressor(architecture="nope").fit([[1.0]], [1.0])
    with pytest.raises(errors.ConfigError, match="unknown optimizer"):
        heads.Regressor(optimizer="nope").fit([[1.0]], [1.0])


def test_fit_stops_early_when_validation_stops_improving(torch, linear):
    engine = heads.Regressor(
        architecture="linear",
        epochs=20,
        patience=2,
        batch_size=9,
        random_state=5,
        device="auto",
    )

    fitted = engine.fit(numpy.ones((30, 2)), alternating(30))

    assert fitted is engine
    assert torch.seeds == [5]
    assert engine.input_width_ == 2
    assert engine.n_iter_ == 3
    net = engine.network_
    assert net.devices == ["cpu", "cpu"]
    assert len(net.loaded) == 1
    assert net.modes[-1] == "eval"
    assert net.modes.count("train") == 3
    assert engine.training_report_["verdict"] == "stalled"
    assert engine.training_report_["steps"] == 9
    assert engine.training_report_["loss_improvement"] == 0.0


def test_fit_runs_every_epoch_without_a_validation_split(torch, linear):
    engine = heads.Regressor(architecture="linear", epochs=4, batch_size=10)

    engine.fit(numpy.ones((10, 1)), alternating(10))

    assert engine.n_iter_ == 4
    assert engine.network_.loaded == []
    assert engine.training_report_["steps"] == 4
    assert engine.training_report_["verdict"] == "stalled"


def test_fit_with_patience_one_keeps_training_while_scores_improve(
    torch, linear, monkeypatch
):
    scores = iter([3.0, 2.0, 1.0, 1.5, 0.5])
    monkeypatch.setattr(
        heads.Regressor, "evaluate", lambda self, *args: next(scores)
    )
    engine = heads.Regressor(architecture="linear", epochs=5, patience=1)

    engine.fit(numpy.ones((30, 1)), alternating(30))

    assert engine.n_iter_ == 4
    assert len(engine.network_.loaded) == 1


def test_fit_stops_at_a_diverged_epoch(torch, linear):
    torch.clip.norms = [math.nan]
    engine = heads.Regressor(architecture="linear", epochs=5)

    engine.fit(numpy.ones((4, 1)), alternating(4))

    assert engine.n_iter_ == 0
    assert engine.training_report_["verdict"] == "diverged"
    assert engine.network_.modes == ["train", "eval"]


def test_a_fitted_classifier_predicts_its_own_labels(torch, linear):
    engine = heads.Classifier(architecture="linear", epochs=1)
    engine.fit(numpy.eye(2), ["left", "right"])
    engine.network_.weight = numpy.array([[1.0, 0.0], [0.0, 1.0]])

    assert engine.predict(numpy.eye(2)[::-1]).tolist() == ["right", "left"]
    assert engine.emit(numpy.eye(2)).shape == (2, 1, 2)


def test_raw_outputs_keep_member_outputs(torch):
    engine = heads.Regressor()
    engine.network_ = Net(2, 1, members=3)
    engine.network_.weight = numpy.array([[1.0], [2.0]])

    found = engine.emit(sparse.csr_matrix([[1.0, 1.0]]))

    assert type(found) is numpy.ndarray
    assert found.shape == (1, 3, 1)
    assert found.ravel().tolist() == [3.0, 3.0, 3.0]


def test_explain_multiplies_the_gradient_by_the_input(torch):
    engine = heads.Regressor()
    engine.network_ = Net(2, 1)
    engine.network_.weight = numpy.array([[1.0], [2.0]])

    found = engine.explain([[1.0, -3.0], [0.5, 0.0]])

    assert type(found) is numpy.ndarray
    assert found.tolist() == [[2.0, -6.0], [1.0, 0.0]]
    assert torch.graded == [("grad", -5.0 + 0.5)]


def test_explain_selects_the_predicted_class_logit(torch):
    engine = heads.Classifier()
    engine.network_ = Net(2, 2, members=2)
    engine.network_.weight = numpy.array([[1.0, 4.0], [3.0, 0.0]])

    engine.explain([[1.0, 0.0], [0.0, 1.0]])

    assert torch.graded == [("grad", 4.0 + 3.0)]
