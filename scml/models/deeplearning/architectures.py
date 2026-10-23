"""Experimental tabular network architectures, built lazily from PyTorch.

The transformer architecture in this module is an FT-Transformer over a
dense numeric tabular matrix. Its name does not imply text, image, audio,
video, or multimodal transformer support.
"""

import abc
from typing import Any


class Architecture(abc.ABC):
    """A way to turn a feature vector into class scores or a value.

    Subclasses are stateless; the estimator owns training.
    """

    name: str

    @abc.abstractmethod
    def build(
        self,
        torch: Any,
        inputs: int,
        outputs: int,
        hidden_sizes: tuple[int, ...],
        dropout: float,
    ) -> Any:
        """Creates an untrained `torch.nn.Module`.

        Args:
            torch: The imported torch module.
            inputs: Width of the input matrix.
            outputs: Number of output units.
            hidden_sizes: Layer widths.
            dropout: Dropout probability.
        """


class MlpArchitecture(Architecture):
    """Fully connected layers with ReLU and dropout."""

    name = "mlp"

    def build(
        self,
        torch: Any,
        inputs: int,
        outputs: int,
        hidden_sizes: tuple[int, ...],
        dropout: float,
    ) -> Any:
        """Stacks linear, ReLU and dropout layers."""
        layers: list[Any] = []
        width = inputs
        for size in hidden_sizes:
            layers += [
                torch.nn.Linear(width, size),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ]
            width = size
        layers.append(torch.nn.Linear(width, outputs))
        return torch.nn.Sequential(*layers)


class ResidualArchitecture(Architecture):
    """An input projection, residual blocks, then a linear head.

    Every block is `x + Dropout(Linear(ReLU(LayerNorm(x))))`; the width is
    the first entry of `hidden_sizes` and its length is the block count.
    """

    name = "residual"

    def build(
        self,
        torch: Any,
        inputs: int,
        outputs: int,
        hidden_sizes: tuple[int, ...],
        dropout: float,
    ) -> Any:
        """Builds the residual network."""
        width = hidden_sizes[0]

        class Block(torch.nn.Module):
            """One pre-normalized residual block."""

            def __init__(self) -> None:
                super().__init__()
                self.norm = torch.nn.LayerNorm(width)
                self.linear = torch.nn.Linear(width, width)
                self.drop = torch.nn.Dropout(dropout)

            def forward(self, features: Any) -> Any:
                """Adds the transformed input back to itself."""
                hidden = self.linear(torch.relu(self.norm(features)))
                return features + self.drop(hidden)

        blocks = [Block() for size in hidden_sizes if size]
        return torch.nn.Sequential(
            torch.nn.Linear(inputs, width),
            *blocks,
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, outputs),
        )


ENSEMBLE_MEMBERS = 8
ATTENTION_HEADS = 4


class EnsembleArchitecture(Architecture):
    """A BatchEnsemble MLP in the spirit of TabM.

    Several sub-networks share one weight matrix per layer and differ only in
    cheap per-member scalings and biases, so an ensemble costs little more
    than a single MLP. The network returns one output per member and the
    estimator averages them.
    """

    name = "tabm"

    def build(
        self,
        torch: Any,
        inputs: int,
        outputs: int,
        hidden_sizes: tuple[int, ...],
        dropout: float,
    ) -> Any:
        """Builds the shared-weight ensemble."""
        members = ENSEMBLE_MEMBERS

        class EnsembleLinear(torch.nn.Module):
            """Shared weight with per-member input/output scales and bias."""

            def __init__(self, width_in: int, width_out: int) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.empty(width_in, width_out)
                )
                torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)
                signs = torch.randint(0, 2, (members, width_in)) * 2 - 1
                self.in_scale = torch.nn.Parameter(signs.float())
                self.out_scale = torch.nn.Parameter(
                    torch.ones(members, width_out)
                )
                self.bias = torch.nn.Parameter(torch.zeros(members, width_out))

            def forward(self, features: Any) -> Any:
                """Applies the member-specific adaptation of the layer."""
                return (
                    features * self.in_scale
                ) @ self.weight * self.out_scale + self.bias

        class Network(torch.nn.Module):
            """Repeats the input for every member and stacks the layers."""

            def __init__(self) -> None:
                super().__init__()
                layers: list[Any] = []
                width = inputs
                for size in hidden_sizes:
                    layers += [
                        EnsembleLinear(width, size),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(dropout),
                    ]
                    width = size
                layers.append(EnsembleLinear(width, outputs))
                self.layers = torch.nn.Sequential(*layers)

            def forward(self, features: Any) -> Any:
                """Returns member outputs of shape (rows, members, outputs)."""
                repeated = features.unsqueeze(1).expand(-1, members, -1)
                return self.layers(repeated)

        return Network()


class TransformerArchitecture(Architecture):
    """An experimental tabular-only FT-Transformer.

    Each scalar feature is embedded as `value * weight + bias`, a learned
    classification token is prepended, a stack of pre-norm transformer
    encoder layers mixes the tokens, and the classification token feeds the
    output head. The first hidden size is the embedding width and the number
    of hidden sizes is the number of layers.
    """

    name = "ft_transformer"

    def build(
        self,
        torch: Any,
        inputs: int,
        outputs: int,
        hidden_sizes: tuple[int, ...],
        dropout: float,
    ) -> Any:
        """Builds the tokenizer, encoder and head."""
        width = hidden_sizes[0]
        heads = ATTENTION_HEADS if width % ATTENTION_HEADS == 0 else 1

        class Network(torch.nn.Module):
            """Feature tokenizer, transformer encoder and head."""

            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.randn(inputs, width) * width**-0.5
                )
                self.bias = torch.nn.Parameter(torch.zeros(inputs, width))
                self.cls = torch.nn.Parameter(torch.zeros(1, 1, width))
                layer = torch.nn.TransformerEncoderLayer(
                    width,
                    heads,
                    dim_feedforward=2 * width,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = torch.nn.TransformerEncoder(
                    layer, len(hidden_sizes), enable_nested_tensor=False
                )
                self.head = torch.nn.Sequential(
                    torch.nn.LayerNorm(width),
                    torch.nn.ReLU(),
                    torch.nn.Linear(width, outputs),
                )

            def forward(self, features: Any) -> Any:
                """Scores each row from its classification token."""
                tokens = features.unsqueeze(-1) * self.weight + self.bias
                cls = self.cls.expand(tokens.shape[0], -1, -1)
                encoded = self.encoder(torch.cat([cls, tokens], dim=1))
                return self.head(encoded[:, 0])

        return Network()


class RobustMlpArchitecture(Architecture):
    """An MLP with RealMLP-style numerics: learned scaling, smooth clipping.

    A learned per-feature scale feeds a smooth clip `x / sqrt(1 + (x/3)^2)`
    that tames outliers without hard truncation, followed by linear layers
    with parametric (PReLU) activations.
    """

    name = "robust_mlp"

    def build(
        self,
        torch: Any,
        inputs: int,
        outputs: int,
        hidden_sizes: tuple[int, ...],
        dropout: float,
    ) -> Any:
        """Builds the scaling, clipping and parametric-activation stack."""

        class Scaling(torch.nn.Module):
            """Learned per-feature scale."""

            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(inputs))

            def forward(self, features: Any) -> Any:
                """Multiplies each feature by its scale."""
                return features * self.scale

        class SmoothClip(torch.nn.Module):
            """Soft saturation of large magnitudes."""

            def forward(self, features: Any) -> Any:
                """Maps R to (-3, 3) smoothly."""
                return features / torch.sqrt(1.0 + (features / 3.0) ** 2)

        layers: list[Any] = [Scaling(), SmoothClip()]
        width = inputs
        for size in hidden_sizes:
            layers += [
                torch.nn.Linear(width, size),
                torch.nn.PReLU(),
                torch.nn.Dropout(dropout),
            ]
            width = size
        layers.append(torch.nn.Linear(width, outputs))
        return torch.nn.Sequential(*layers)


ARCHITECTURES: dict[str, Architecture] = {
    architecture.name: architecture
    for architecture in (
        MlpArchitecture(),
        ResidualArchitecture(),
        EnsembleArchitecture(),
        TransformerArchitecture(),
        RobustMlpArchitecture(),
    )
}
