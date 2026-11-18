"""The tabular network architectures, built on a numpy stand-in for torch.

PyTorch is not installed in the framework process, so each architecture is
built from a recording stand-in whose layers compute with numpy: the layer
plan and the forward arithmetic are checked exactly.
"""

import math
import types

import numpy
import pytest

from scmle.models.deeplearning import architectures


class Tensor(numpy.ndarray):
    """A numpy array with the few torch tensor methods the networks call."""

    def unsqueeze(self, dim):
        return numpy.expand_dims(self, dim).view(Tensor)

    def expand(self, *sizes):
        shape = tuple(
            current if size == -1 else size
            for size, current in zip(sizes, self.shape, strict=True)
        )
        return numpy.broadcast_to(self, shape).view(Tensor)

    def float(self):
        return self.astype(numpy.float32).view(Tensor)


def tensor(values):
    return numpy.asarray(values, dtype=float).view(Tensor)


class Module:
    """The stand-in `torch.nn.Module`: calling it runs `forward`."""

    def __init__(self):
        self.spec = (type(self).__name__,)

    def __call__(self, features):
        return self.forward(features)


class Linear(Module):
    """Sums its inputs into every output unit (an all-ones weight)."""

    def __init__(self, width_in, width_out):
        self.spec = ("Linear", width_in, width_out)
        self.weight = numpy.ones((width_in, width_out))

    def forward(self, features):
        return (features @ self.weight).view(Tensor)


class ReLU(Module):
    def __init__(self):
        self.spec = ("ReLU",)

    def forward(self, features):
        return numpy.maximum(features, 0.0).view(Tensor)


class PReLU(Module):
    def __init__(self):
        self.spec = ("PReLU",)

    def forward(self, features):
        return numpy.where(features > 0, features, 0.25 * features).view(Tensor)


class Dropout(Module):
    def __init__(self, probability):
        self.spec = ("Dropout", probability)

    def forward(self, features):
        return features


class LayerNorm(Module):
    def __init__(self, width):
        self.spec = ("LayerNorm", width)

    def forward(self, features):
        return features


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)
        self.spec = ("Sequential", len(layers))

    def forward(self, features):
        for layer in self.layers:
            features = layer(features)
        return features


class EncoderLayer:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class Encoder(Module):
    def __init__(self, layer, count, **kwargs):
        self.layer = layer
        self.count = count
        self.kwargs = kwargs

    def forward(self, features):
        return features


class Initializer:
    """Records `kaiming_uniform_` calls and fills the weight with ones."""

    def __init__(self):
        self.calls = []

    def kaiming_uniform_(self, weight, a):
        self.calls.append((weight.shape, a))
        weight[...] = 1.0
        return weight


def randint(low, high, shape):
    """Alternates 0 and 1, so the member signs alternate -1 and +1."""
    assert (low, high) == (0, 2)
    return (numpy.arange(math.prod(shape)) % 2).reshape(shape).view(Tensor)


def fake():
    """A fresh stand-in for the torch module."""
    initializer = Initializer()
    nn = types.SimpleNamespace(
        Module=Module,
        Linear=Linear,
        ReLU=ReLU,
        PReLU=PReLU,
        Dropout=Dropout,
        LayerNorm=LayerNorm,
        Sequential=Sequential,
        TransformerEncoderLayer=EncoderLayer,
        TransformerEncoder=Encoder,
        Parameter=lambda value: numpy.array(value, dtype=float).view(Tensor),
        init=initializer,
    )
    return types.SimpleNamespace(
        nn=nn,
        empty=lambda *shape: numpy.zeros(shape).view(Tensor),
        ones=lambda *shape: numpy.ones(shape).view(Tensor),
        zeros=lambda *shape: numpy.zeros(shape).view(Tensor),
        randn=lambda *shape: numpy.ones(shape).view(Tensor),
        randint=randint,
        relu=lambda value: numpy.maximum(value, 0.0).view(Tensor),
        sqrt=numpy.sqrt,
        cat=lambda parts, dim: numpy.concatenate(parts, axis=dim).view(Tensor),
    )


def specs(network):
    return [layer.spec for layer in network.layers]


def test_the_registry_names_every_architecture_by_its_own_name():
    assert {
        name: type(value) for name, value in architectures.ARCHITECTURES.items()
    } == {
        "mlp": architectures.Mlp,
        "residual": architectures.Residual,
        "tabm": architectures.Ensemble,
        "ft_transformer": architectures.Transformer,
        "robust_mlp": architectures.Robust,
    }
    assert architectures.ENSEMBLE_MEMBERS == 8
    assert architectures.ATTENTION_HEADS == 4


def test_an_architecture_cannot_be_instantiated_without_build():
    with pytest.raises(TypeError):
        architectures.Architecture()


def test_an_mlp_stacks_linear_relu_and_dropout_per_hidden_size():
    network = architectures.Mlp().build(fake(), 3, 2, (4, 5), 0.3)

    assert specs(network) == [
        ("Linear", 3, 4),
        ("ReLU",),
        ("Dropout", 0.3),
        ("Linear", 4, 5),
        ("ReLU",),
        ("Dropout", 0.3),
        ("Linear", 5, 2),
    ]
    assert network(tensor([[1.0, -2.0, 3.0]])).tolist() == [[40.0, 40.0]]
    assert network(tensor([[-1.0, -1.0, -1.0]])).tolist() == [[0.0, 0.0]]


def test_an_mlp_without_hidden_sizes_is_one_linear_layer():
    network = architectures.Mlp().build(fake(), 3, 2, (), 0.0)

    assert specs(network) == [("Linear", 3, 2)]


def test_a_residual_network_adds_one_block_per_nonzero_hidden_size():
    network = architectures.Residual().build(fake(), 3, 2, (4, 0, 6), 0.2)

    names = [layer.spec[0] for layer in network.layers]
    assert names == ["Linear", "Block", "Block", "LayerNorm", "Linear"]
    assert network.layers[0].spec == ("Linear", 3, 4)
    assert network.layers[3].spec == ("LayerNorm", 4)
    assert network.layers[4].spec == ("Linear", 4, 2)
    block = network.layers[1]
    assert block.norm.spec == ("LayerNorm", 4)
    assert block.linear.spec == ("Linear", 4, 4)
    assert block.drop.spec == ("Dropout", 0.2)


def test_a_residual_block_adds_its_transformed_input_back():
    block = architectures.Residual().build(fake(), 4, 1, (4,), 0.0).layers[1]

    found = block(tensor([[1.0, -2.0, 0.0, 3.0]]))

    assert found.tolist() == [[5.0, 2.0, 4.0, 7.0]]


def test_an_ensemble_returns_one_output_per_member():
    torch = fake()
    network = architectures.Ensemble().build(torch, 3, 2, (3,), 0.1)

    kinds = [layer.spec[0] for layer in network.layers.layers]
    assert kinds == ["Affine", "ReLU", "Dropout", "Affine"]
    assert network.layers.layers[2].spec == ("Dropout", 0.1)
    assert torch.nn.init.calls == [((3, 3), 5**0.5), ((3, 2), 5**0.5)]
    first = network.layers.layers[0]
    assert first.weight.shape == (3, 3)
    assert first.out_scale.tolist() == numpy.ones((8, 3)).tolist()
    assert first.bias.tolist() == numpy.zeros((8, 3)).tolist()
    assert first.in_scale[0].tolist() == [-1.0, 1.0, -1.0]
    assert first.in_scale[1].tolist() == [1.0, -1.0, 1.0]

    found = network(tensor([[1.0, 2.0, 3.0]]))

    assert found.shape == (1, 8, 2)
    assert found[0, 0].tolist() == [0.0, 0.0]
    assert found[0, 1].tolist() == [2.0, 2.0]


def test_an_affine_layer_scales_inputs_and_outputs_per_member():
    affine = architectures.Ensemble.affine(fake())(2, 3)
    affine.out_scale = tensor(numpy.full((8, 3), 2.0))
    affine.bias = tensor(numpy.full((8, 3), 0.5))
    features = tensor(numpy.ones((1, 8, 2)))

    found = affine(features)

    assert found.shape == (1, 8, 3)
    assert found[0, 0].tolist() == [0.5, 0.5, 0.5]
    assert found[0, 1].tolist() == [0.5, 0.5, 0.5]
    affine.in_scale = tensor(numpy.ones((8, 2)))
    assert affine(features)[0, 0].tolist() == [4.5, 4.5, 4.5]


@pytest.mark.parametrize(("width", "heads"), [(8, 4), (12, 4), (6, 1), (4, 4)])
def test_a_transformer_uses_four_heads_only_when_the_width_divides(
    width, heads
):
    network = architectures.Transformer().build(
        fake(), 3, 2, (width, width), 0.25
    )

    layer = network.encoder.layer
    assert layer.args == (width, heads)
    assert layer.kwargs == {
        "dim_feedforward": 2 * width,
        "dropout": 0.25,
        "activation": "gelu",
        "batch_first": True,
        "norm_first": True,
    }
    assert network.encoder.count == 2
    assert network.encoder.kwargs == {"enable_nested_tensor": False}


def test_a_transformer_embeds_each_feature_and_reads_the_class_token():
    network = architectures.Transformer().build(fake(), 3, 2, (4,), 0.0)

    assert network.weight.tolist() == numpy.full((3, 4), 0.5).tolist()
    assert network.bias.tolist() == numpy.zeros((3, 4)).tolist()
    assert network.cls.shape == (1, 1, 4)
    assert network.encoder.count == 1
    assert [layer.spec for layer in network.head.layers] == [
        ("LayerNorm", 4),
        ("ReLU",),
        ("Linear", 4, 2),
    ]
    assert network(tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).tolist() == [
        [0.0, 0.0],
        [0.0, 0.0],
    ]
    network.cls = tensor(numpy.ones((1, 1, 4)))
    assert network(tensor([[1.0, 2.0, 3.0]])).tolist() == [[4.0, 4.0]]


def test_a_robust_mlp_scales_saturates_then_uses_parametric_activations():
    network = architectures.Robust().build(fake(), 3, 2, (4,), 0.1)

    assert specs(network) == [
        ("Scaling",),
        ("Saturation",),
        ("Linear", 3, 4),
        ("PReLU",),
        ("Dropout", 0.1),
        ("Linear", 4, 2),
    ]
    assert network.layers[0].scale.tolist() == [1.0, 1.0, 1.0]


def test_a_robust_saturation_maps_large_values_smoothly_inside_three():
    network = architectures.Robust().build(fake(), 3, 1, (), 0.0)
    scaling, saturation = network.layers[0], network.layers[1]

    found = saturation(tensor([3.0, 0.0, -4.0]))

    assert found.tolist() == pytest.approx([3.0 / math.sqrt(2.0), 0.0, -2.4])
    assert abs(float(saturation(tensor([1e3]))[0])) == pytest.approx(
        1e3 / math.sqrt(1.0 + (1e3 / 3.0) ** 2)
    )
    assert float(network(tensor([[3.0, 0.0, -4.0]]))[0, 0]) == pytest.approx(
        3.0 / math.sqrt(2.0) - 2.4
    )
    scaling.scale = tensor([2.0, -1.0, 0.5])
    assert scaling(tensor([1.0, 1.0, 4.0])).tolist() == [2.0, -1.0, 2.0]
