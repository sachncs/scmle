"""Network architectures for tabular data, built lazily from PyTorch."""

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

        blocks = [Block() for __ in hidden_sizes]
        return torch.nn.Sequential(
            torch.nn.Linear(inputs, width),
            *blocks,
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, outputs),
        )


ARCHITECTURES: dict[str, Architecture] = {
    architecture.name: architecture
    for architecture in (MlpArchitecture(), ResidualArchitecture())
}
