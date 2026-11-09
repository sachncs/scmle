"""Hyperparameter search-space contracts.

Each `Parameter` subclass knows how to sample itself, so spaces are
composed polymorphically instead of branching on parameter type.
"""

import abc
import dataclasses
import math
import random
from typing import Any


@dataclasses.dataclass(frozen=True, slots=True)
class Parameter(abc.ABC):
    """One tunable hyperparameter.

    Attributes:
        name: Constructor argument name of the estimator.
    """

    name: str

    @abc.abstractmethod
    def sample(self, rng: random.Random) -> Any:
        """Draws a value.

        Args:
            rng: Source of randomness.

        Returns:
            A value inside the parameter's domain.
        """


@dataclasses.dataclass(frozen=True, slots=True)
class FloatParameter(Parameter):
    """A float in `[low, high]`, optionally log-uniform.

    Attributes:
        low: Lower bound (> 0 when `log`).
        high: Upper bound.
        log: Sample uniformly in log space.
    """

    low: float
    high: float
    log: bool = False

    def sample(self, rng: random.Random) -> float:
        """Draws a float.

        Args:
            rng: The random generator.

        Returns:
            A value between low and high, log-uniform if `log` is set.
        """
        if self.log:
            return math.exp(
                rng.uniform(math.log(self.low), math.log(self.high))
            )
        return rng.uniform(self.low, self.high)


@dataclasses.dataclass(frozen=True, slots=True)
class IntParameter(Parameter):
    """An integer in `[low, high]` (inclusive), optionally log-uniform.

    Attributes:
        low: Lower bound (> 0 when `log`).
        high: Upper bound.
        log: Sample uniformly in log space, then truncate to an integer.
    """

    low: int
    high: int
    log: bool = False

    def sample(self, rng: random.Random) -> int:
        """Draws an integer.

        Args:
            rng: The random generator.

        Returns:
            An integer between low and high inclusive, log-uniform if `log`
            is set.
        """
        if self.log:
            value = math.exp(
                rng.uniform(math.log(self.low), math.log(self.high + 1))
            )
            return min(self.high, max(self.low, int(value)))
        return rng.randint(self.low, self.high)


@dataclasses.dataclass(frozen=True, slots=True)
class ChoiceParameter(Parameter):
    """One of a fixed set of values.

    Attributes:
        choices: Allowed values.
    """

    choices: tuple[Any, ...]

    def sample(self, rng: random.Random) -> Any:
        """Draws one of the choices.

        Args:
            rng: The random generator.

        Returns:
            One of the choices, uniformly.
        """
        return rng.choice(self.choices)


@dataclasses.dataclass(frozen=True, slots=True)
class SearchSpace:
    """A set of parameters tuned together.

    Attributes:
        parameters: The tunable parameters.
    """

    parameters: tuple[Parameter, ...]

    def sample(self, rng: random.Random) -> dict[str, Any]:
        """Draws one full configuration.

        Args:
            rng: The random generator.

        Returns:
            A mapping from parameter name to a sampled value.
        """
        return {p.name: p.sample(rng) for p in self.parameters}

    def names(self) -> tuple[str, ...]:
        """Returns the parameter names."""
        return tuple(p.name for p in self.parameters)
