"""Optimizers with an ask/tell interface.

The framework owns orchestration (budgets, pruning, isolation); an optimizer
only proposes configurations and learns from scores, so a library such as
Optuna never becomes the framework's architecture.
"""

import abc
import functools
import importlib.util
import itertools
import math
import random
from typing import Any
from typing import ClassVar

from scmle.contracts import errors
from scmle.contracts import search


class Optimizer(abc.ABC):
    """Proposes configurations from a search space.

    Attributes:
        name: Name used in `SearchSpec.optimizer`.
        requires: Importable modules the optimizer needs.
    """

    name: ClassVar[str]
    requires: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        space: search.SearchSpace,
        seed: int,
        greater_is_better: bool = True,
    ) -> None:
        """Initializes the optimizer.

        Args:
            space: Parameters to tune.
            seed: Random seed.
            greater_is_better: Orientation of the scores passed to `tell`.
        """
        self.space = space
        self.seed = seed
        self.greater_is_better = greater_is_better

    @classmethod
    def is_available(cls) -> bool:
        """Whether every required module is importable."""
        return all(importlib.util.find_spec(m) for m in cls.requires)

    @abc.abstractmethod
    def ask(self) -> dict[str, Any]:
        """Proposes the next configuration."""

    @abc.abstractmethod
    def tell(self, params: dict[str, Any], score: float) -> None:
        """Reports the score achieved by a proposed configuration.

        Args:
            params: A configuration previously returned by `ask`.
            score: Its (possibly partial-fidelity) score.
        """


class RandomOptimizer(Optimizer):
    """Seeded random search; dependency-free default."""

    name = "random"

    def __init__(
        self,
        space: search.SearchSpace,
        seed: int,
        greater_is_better: bool = True,
    ) -> None:
        """Initializes the sampler and the observation history."""
        super().__init__(space, seed, greater_is_better)
        self.rng = random.Random(seed)  # nosec B311 - seeded search, not security
        self.history: list[tuple[dict[str, Any], float]] = []

    def ask(self) -> dict[str, Any]:
        """Draws a configuration uniformly from the space."""
        return self.space.sample(self.rng)

    def tell(self, params: dict[str, Any], score: float) -> None:
        """Records the observation."""
        self.history.append((params, score))


@functools.singledispatch
def suggest(parameter: search.Parameter, trial: Any) -> Any:
    """Asks an Optuna trial for a value of `parameter` (dispatch on type)."""
    raise errors.ConfigError(
        f"unsupported parameter type {type(parameter).__name__}"
    )


@suggest.register
def suggest_float(parameter: search.FloatParameter, trial: Any) -> float:
    """Suggests a float."""
    return trial.suggest_float(
        parameter.name, parameter.low, parameter.high, log=parameter.log
    )


@suggest.register
def suggest_int(parameter: search.IntParameter, trial: Any) -> int:
    """Suggests an integer."""
    return trial.suggest_int(
        parameter.name, parameter.low, parameter.high, log=parameter.log
    )


@suggest.register
def suggest_choice(parameter: search.ChoiceParameter, trial: Any) -> Any:
    """Suggests one of the choices."""
    return trial.suggest_categorical(parameter.name, list(parameter.choices))


class GridOptimizer(Optimizer):
    """Walks a coarse grid: low, middle and high of each numeric parameter.

    Choices contribute every option (up to four). The grid is shuffled with
    the seed, so even a short run touches distant corners rather than one
    neighbourhood, and random draws take over once the grid is exhausted.
    """

    name = "grid"
    LEVELS = 3
    MAX_CHOICES = 4

    def __init__(
        self,
        space: search.SearchSpace,
        seed: int,
        greater_is_better: bool = True,
    ) -> None:
        """Builds and shuffles the grid."""
        super().__init__(space, seed, greater_is_better)
        axes = [self.levels(p) for p in space.parameters]
        points = [
            dict(zip(space.names(), combo, strict=True))
            for combo in itertools.product(*axes)
        ]
        self.rng = random.Random(seed)  # nosec B311 - seeded, not security
        self.rng.shuffle(points)
        self.points = points
        self.position = 0

    @classmethod
    def levels(cls, parameter: search.Parameter) -> list[Any]:
        """The grid values of one parameter."""
        if isinstance(parameter, search.ChoiceParameter):
            return list(parameter.choices)[: cls.MAX_CHOICES]
        if isinstance(parameter, search.IntParameter):
            whole_low, whole_high = parameter.low, parameter.high
            if parameter.log:
                whole_mid = round(math.sqrt(whole_low * whole_high))
            else:
                whole_mid = (whole_low + whole_high) // 2
            return sorted({whole_low, whole_mid, whole_high})
        assert isinstance(parameter, search.FloatParameter)  # nosec B101
        low, high = parameter.low, parameter.high
        middle = math.sqrt(low * high) if parameter.log else (low + high) / 2
        return [low, middle, high]

    def ask(self) -> dict[str, Any]:
        """The next grid point, then random draws."""
        if self.position < len(self.points):
            point = self.points[self.position]
            self.position += 1
            return dict(point)
        return self.space.sample(self.rng)

    def tell(self, params: dict[str, Any], score: float) -> None:
        """A grid keeps no state beyond its position."""


class OptunaOptimizer(Optimizer):
    """TPE search through Optuna (optional `optimization` extra)."""

    name = "optuna"
    requires = ("optuna",)

    def __init__(
        self,
        space: search.SearchSpace,
        seed: int,
        greater_is_better: bool = True,
    ) -> None:
        """Creates the study.

        Raises:
            ConfigError: If Optuna is not installed.
        """
        super().__init__(space, seed, greater_is_better)
        if not self.is_available():
            raise errors.ConfigError(
                "optimizer 'optuna' needs the 'optimization' extra"
            )
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study = optuna.create_study(
            direction="maximize" if greater_is_better else "minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        self.pending: list[tuple[dict[str, Any], Any]] = []

    def ask(self) -> dict[str, Any]:
        """Asks the study for the next configuration."""
        trial = self.study.ask()
        params = {p.name: suggest(p, trial) for p in self.space.parameters}
        self.pending.append((params, trial))
        return params

    def tell(self, params: dict[str, Any], score: float) -> None:
        """Feeds the score back to the matching pending trial."""
        for index, (candidate, trial) in enumerate(self.pending):
            if candidate == params:
                self.study.tell(trial, score)
                del self.pending[index]
                return


OPTIMIZERS: dict[str, type[Optimizer]] = {
    optimizer.name: optimizer
    for optimizer in (RandomOptimizer, GridOptimizer, OptunaOptimizer)
}


def create_optimizer(
    name: str,
    space: search.SearchSpace,
    seed: int,
    greater_is_better: bool,
) -> Optimizer:
    """Instantiates an optimizer by name.

    Args:
        name: `random`, `grid` or `optuna`.
        space: Parameters to tune.
        seed: Random seed.
        greater_is_better: Orientation of the told scores.

    Returns:
        The optimizer.

    Raises:
        ConfigError: If the name is unknown or its dependency is missing.
    """
    if name not in OPTIMIZERS:
        raise errors.ConfigError(
            f"unknown optimizer '{name}'; available: {sorted(OPTIMIZERS)}"
        )
    return OPTIMIZERS[name](space, seed, greater_is_better)
