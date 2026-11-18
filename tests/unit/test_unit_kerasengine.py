"""The Keras training engines, on a recording stand-in for Keras."""

import os
import pickle
import sys
import types

import numpy
import pytest

from scmle.models.deeplearning import keras_engine


class History:
    def __init__(self, losses):
        self.history = {"loss": list(losses)}


class Network:
    """A stand-in `keras.Sequential` that records how it is used.

    Attributes:
        layers: The layer specifications it was built from.
        compiled: The keyword arguments of `compile`.
        fitted: The positional and keyword arguments of `fit`.
        weights: The weights last set.
    """

    losses = [1.0, 0.5, 0.25]
    outputs = numpy.zeros((1, 1))

    def __init__(self, layers):
        self.layers = list(layers)
        self.compiled = None
        self.fitted = None
        self.weights = [numpy.array([1.0, 2.0]), numpy.array([[3.0]])]

    def compile(self, **kwargs):
        self.compiled = kwargs

    def fit(self, *args, **kwargs):
        self.fitted = (args, kwargs)
        return History(Network.losses)

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def predict(self, features, verbose):
        assert verbose == 0
        assert features.dtype == numpy.float32
        return Network.outputs


def stand_in():
    seeds = []
    module = types.ModuleType("keras")
    module.seeds = seeds
    module.Sequential = Network
    module.regularizers = types.SimpleNamespace(l2=lambda value: ("l2", value))
    module.layers = types.SimpleNamespace(
        Input=lambda shape: ("Input", shape),
        Dense=lambda units, activation=None, kernel_regularizer=None: (
            "Dense",
            units,
            activation,
            kernel_regularizer,
        ),
        Dropout=lambda rate: ("Dropout", rate),
    )
    module.utils = types.SimpleNamespace(set_random_seed=seeds.append)
    module.optimizers = types.SimpleNamespace(Adam=lambda rate: ("Adam", rate))
    module.callbacks = types.SimpleNamespace(
        TerminateOnNaN=lambda: ("TerminateOnNaN",),
        EarlyStopping=lambda patience, restore_best_weights: (
            "EarlyStopping",
            patience,
            restore_best_weights,
        ),
    )
    module.losses = types.SimpleNamespace(
        SparseCategoricalCrossentropy=lambda from_logits: (
            "sparse",
            from_logits,
        )
    )
    return module


@pytest.fixture
def keras(monkeypatch):
    module = stand_in()
    monkeypatch.setitem(sys.modules, "keras", module)
    monkeypatch.setenv("KERAS_BACKEND", "torch")
    Network.losses = [1.0, 0.5, 0.25]
    Network.outputs = numpy.zeros((1, 1))
    return module


def test_the_engine_thresholds_are_fixed():
    assert keras_engine.STALL_IMPROVEMENT == 0.01
    assert keras_engine.MIN_ROWS_FOR_VALIDATION == 30
    assert keras_engine.MIN_EPOCHS_FOR_TREND == 3


def test_load_selects_the_torch_backend_unless_one_is_set(keras, monkeypatch):
    monkeypatch.delenv("KERAS_BACKEND")

    assert keras_engine.load() is keras
    assert os.environ["KERAS_BACKEND"] == "torch"

    monkeypatch.setenv("KERAS_BACKEND", "jax")
    keras_engine.load()
    assert os.environ["KERAS_BACKEND"] == "jax"


def test_the_engine_keeps_its_hyperparameters():
    engine = keras_engine.Trainer()

    assert engine.get_params() == {
        "hidden_sizes": (64,),
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "l2": 1e-5,
        "epochs": 100,
        "batch_size": 128,
        "patience": 10,
        "validation_fraction": 0.1,
        "random_state": 0,
    }
    assert engine.dimensionality() == 1
    assert engine.objective() == "mse"
    found = engine.vectorize([1, 2])
    assert found.dtype == numpy.float32
    assert found.tolist() == [1.0, 2.0]


def test_build_stacks_regularized_relu_layers_and_dropout(keras):
    engine = keras_engine.Trainer(hidden_sizes=(8, 4), dropout=0.3, l2=0.5)

    network = engine.build(keras, 5)

    assert network.layers == [
        ("Input", (5,)),
        ("Dense", 8, "relu", ("l2", 0.5)),
        ("Dropout", 0.3),
        ("Dense", 4, "relu", ("l2", 0.5)),
        ("Dropout", 0.3),
        ("Dense", 1, None, None),
    ]


def test_fit_holds_out_rows_for_early_stopping_from_thirty_rows(keras):
    engine = keras_engine.Trainer(
        epochs=7,
        batch_size=16.0,
        patience=3,
        validation_fraction=0.2,
        random_state=4,
    )
    features = numpy.ones((30, 2))

    fitted = engine.fit(features, numpy.arange(30), sample_weight=[1] * 30)

    assert fitted is engine
    assert keras.seeds == [4]
    assert engine.input_width_ == 2
    network = engine.network_
    assert network.compiled == {"optimizer": ("Adam", 1e-3), "loss": "mse"}
    args, kwargs = network.fitted
    assert args[0].dtype == numpy.float32
    assert args[1].tolist() == list(range(30))
    assert kwargs["sample_weight"].dtype == numpy.float32
    assert kwargs["epochs"] == 7
    assert kwargs["batch_size"] == 16
    assert isinstance(kwargs["batch_size"], int)
    assert kwargs["validation_split"] == 0.2
    assert kwargs["callbacks"] == [
        ("TerminateOnNaN",),
        ("EarlyStopping", 3, True),
    ]
    assert kwargs["verbose"] == 0
    assert kwargs["shuffle"] is True
    assert engine.n_iter_ == 3
    assert engine.training_report_ == {
        "verdict": "ok",
        "steps": 3,
        "improvement": 0.75,
    }


def test_fit_trains_on_every_row_below_thirty_rows(keras):
    engine = keras_engine.Trainer()

    engine.fit(numpy.ones((29, 2)), numpy.zeros(29))

    kwargs = engine.network_.fitted[1]
    assert kwargs["validation_split"] == 0.0
    assert kwargs["callbacks"] == [("TerminateOnNaN",)]
    assert kwargs["sample_weight"] is None


@pytest.mark.parametrize(
    ("losses", "expected"),
    [
        ([1.0, float("nan")], {"verdict": "diverged", "steps": 2}),
        ([1.0, float("inf"), 0.5], {"verdict": "diverged", "steps": 3}),
        ([1.0, 0.5], {"verdict": "ok", "steps": 2, "improvement": 0.0}),
        (
            [1.0, 1.0, 0.995],
            {
                "verdict": "stalled",
                "steps": 3,
                "improvement": pytest.approx(0.005),
            },
        ),
        (
            [1.0, 1.0, 0.99],
            {"verdict": "ok", "steps": 3, "improvement": pytest.approx(0.01)},
        ),
        (
            [0.0, 0.0, 0.0],
            {"verdict": "stalled", "steps": 3, "improvement": 0.0},
        ),
        ([-2.0, -2.0, -3.0], {"verdict": "ok", "steps": 3, "improvement": 0.5}),
        ([], {"verdict": "ok", "steps": 0, "improvement": 0.0}),
    ],
)
def test_the_verdict_names_divergence_and_stalls(losses, expected):
    assert keras_engine.Trainer.verdict(losses) == expected


def test_a_pickle_keeps_weights_and_rebuilds_the_network(keras):
    engine = keras_engine.Trainer(hidden_sizes=(3,)).fit(
        numpy.ones((4, 2)), numpy.zeros(4)
    )
    engine.network_.weights = [numpy.array([7.0])]

    state = engine.__getstate__()

    assert "network_" not in state
    assert [w.tolist() for w in state["weights_"]] == [[7.0]]
    restored = pickle.loads(pickle.dumps(engine))
    assert restored.network_.weights[0].tolist() == [7.0]
    assert restored.network_.layers[0] == ("Input", (2,))
    assert restored.network_.layers[1] == ("Dense", 3, "relu", ("l2", 1e-5))


def test_an_unfitted_engine_pickles_without_weights():
    engine = keras_engine.Trainer(epochs=3)

    state = engine.__getstate__()

    assert "weights_" not in state
    assert pickle.loads(pickle.dumps(engine)).epochs == 3


def test_a_classifier_encodes_labels_and_uses_sparse_cross_entropy(keras):
    engine = keras_engine.Classifier()

    codes = engine.vectorize(["b", "a", "c", "a"])

    assert codes.dtype == numpy.int64
    assert codes.tolist() == [1, 0, 2, 0]
    assert engine.classes_.tolist() == ["a", "b", "c"]
    assert engine.dimensionality() == 3
    assert engine.objective() == ("sparse", True)


def test_a_classifier_turns_scores_into_probabilities_and_labels(keras):
    engine = keras_engine.Classifier().fit(numpy.ones((2, 1)), ["no", "yes"])
    Network.outputs = numpy.array([[0.0, numpy.log(3.0)], [1000.0, 0.0]])

    probabilities = engine.predict_proba([[1.0], [2.0]])

    assert probabilities[0].tolist() == pytest.approx([0.25, 0.75])
    assert probabilities[1].tolist() == pytest.approx([1.0, 0.0])
    assert engine.predict([[1.0], [2.0]]).tolist() == ["yes", "no"]
    assert engine.network_.compiled["loss"] == ("sparse", True)
    assert engine.network_.layers[-1] == ("Dense", 2, None, None)


def test_a_regressor_standardizes_and_restores_the_target(keras):
    engine = keras_engine.Regressor()

    encoded = engine.vectorize([1.0, 3.0])

    assert encoded.dtype == numpy.float32
    assert encoded.tolist() == [-1.0, 1.0]
    assert engine.target_mean_ == 2.0
    assert engine.target_scale_ == 1.0
    engine.network_ = Network([])
    Network.outputs = numpy.array([[0.5, 9.0], [-2.0, 9.0]])
    assert engine.predict([[0.0], [0.0]]).tolist() == [2.5, 0.0]


def test_a_regressor_uses_unit_scale_for_a_constant_target():
    engine = keras_engine.Regressor()

    assert engine.vectorize([4.0, 4.0]).tolist() == [0.0, 0.0]
    assert engine.target_mean_ == 4.0
    assert engine.target_scale_ == 1.0


def test_raw_outputs_ask_the_network_quietly_in_float32(keras):
    engine = keras_engine.Trainer()
    engine.network_ = Network([])
    Network.outputs = [[1.5]]

    found = engine.raw_outputs([[1, 2]])

    assert isinstance(found, numpy.ndarray)
    assert found.tolist() == [[1.5]]
