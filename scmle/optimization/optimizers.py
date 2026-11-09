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
        name: Name used in `Search.optimizer`.
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
        """Reports whether every required module is importable.

        Returns:
            True if each module named in `requires` can be found (always True
            when `requires` is empty), otherwise False.
        """
        return all(importlib.util.find_spec(m) for m in cls.requires)

    @abc.abstractmethod
    def ask(self) -> dict[str, Any]:
        """Proposes the next configuration.

        Returns:
            A mapping from parameter name to the proposed value, covering every
            parameter in the search space.
        """

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
        """Initializes the sampler and the observation history.

        Args:
            space: Parameters to tune.
            seed: Seed of the random generator.
            greater_is_better: Orientation of the scores passed to `tell`.
        """
        super().__init__(space, seed, greater_is_better)
        self.rng = random.Random(seed)  # nosec B311  # noqa: S311 -- seeded search, not security
        self.history: list[tuple[dict[str, Any], float]] = []

    def ask(self) -> dict[str, Any]:
        """Draws a configuration uniformly from the space.

        Returns:
            A mapping from parameter name to the sampled value.
        """
        return self.space.sample(self.rng)

    def tell(self, params: dict[str, Any], score: float) -> None:
        """Records the observation.

        Args:
            params: A configuration previously returned by `ask`.
            score: Its score.
        """
        self.history.append((params, score))


@functools.singledispatch
def suggest(
    parameter: search.Parameter,
    trial: Any,  # noqa: ARG001 -- dispatch signature shared by every type
) -> Any:
    """Asks an Optuna trial for a value of `parameter` (dispatch on type).

    This is the fallback of a single-dispatch function; the registered
    implementations below handle the supported parameter types.

    Args:
        parameter: The search-space parameter to suggest a value for.
        trial: An Optuna trial.

    Raises:
        ConfigError: If the parameter's type has no registered implementation.
    """
    raise errors.ConfigError(
        f"unsupported parameter type {type(parameter).__name__}"
    )


@suggest.register
def suggest_float(parameter: search.FloatParameter, trial: Any) -> float:
    """Suggests a float.

    Args:
        parameter: The float parameter, with its bounds and log flag.
        trial: An Optuna trial.

    Returns:
        A float within the parameter's bounds, on a log scale if the parameter
        is logarithmic.
    """
    return float(
        trial.suggest_float(
            parameter.name, parameter.low, parameter.high, log=parameter.log
        )
    )


@suggest.register
def suggest_int(parameter: search.IntParameter, trial: Any) -> int:
    """Suggests an integer.

    Args:
        parameter: The integer parameter, with its bounds and log flag.
        trial: An Optuna trial.

    Returns:
        An integer within the parameter's bounds, on a log scale if the
        parameter is logarithmic.
    """
    return int(
        trial.suggest_int(
            parameter.name, parameter.low, parameter.high, log=parameter.log
        )
    )


@suggest.register
def suggest_choice(parameter: search.ChoiceParameter, trial: Any) -> Any:
    """Suggests one of the choices.

    Args:
        parameter: The categorical parameter, with its allowed choices.
        trial: An Optuna trial.

    Returns:
        One of the parameter's choices.
    """
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
        """Builds and shuffles the grid.

        Args:
            space: Parameters to tune.
            seed: Seed used to shuffle the grid and to draw once it is
                exhausted.
            greater_is_better: Orientation of the scores passed to `tell`.
        """
        super().__init__(space, seed, greater_is_better)
        axes = [self.levels(p) for p in space.parameters]
        points = [
            dict(zip(space.names(), combo, strict=True))
            for combo in itertools.product(*axes)
        ]
        self.rng = random.Random(seed)  # nosec B311  # noqa: S311 -- seeded, not security
        self.rng.shuffle(points)
        self.points = points
        self.position = 0

    @classmethod
    def levels(cls, parameter: search.Parameter) -> list[Any]:
        """Lists the grid values of one parameter.

        Args:
            parameter: The search-space parameter.

        Returns:
            The first `MAX_CHOICES` choices for a categorical parameter. For an
            integer parameter, the sorted distinct values among low, middle and
            high (geometric middle if logarithmic). For a float parameter, the
            list low, middle, high.
        """
        if isinstance(parameter, search.ChoiceParameter):
            return list(parameter.choices)[: cls.MAX_CHOICES]
        if isinstance(parameter, search.IntParameter):
            whole_low, whole_high = parameter.low, parameter.high
            if parameter.log:
                whole_mid = round(math.sqrt(whole_low * whole_high))
            else:
                whole_mid = (whole_low + whole_high) // 2
            return sorted({whole_low, whole_mid, whole_high})
        assert isinstance(parameter, search.FloatParameter)  # nosec B101  # noqa: S101 -- narrows the type for the last branch
        low, high = parameter.low, parameter.high
        middle = math.sqrt(low * high) if parameter.log else (low + high) / 2
        return [low, middle, high]

    def ask(self) -> dict[str, Any]:
        """Returns the next grid point, then random draws.

        Returns:
            The next unvisited point of the shuffled grid, as a copy; once the
            grid is exhausted, a random sample from the space.
        """
        if self.position < len(self.points):
            point = self.points[self.position]
            self.position += 1
            return dict(point)
        return self.space.sample(self.rng)

    def tell(self, params: dict[str, Any], score: float) -> None:
        """Ignores the observation: a grid keeps no state beyond its position.

        Args:
            params: A configuration previously returned by `ask`. Unused.
            score: Its score. Unused.
        """


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

        Args:
            space: Parameters to tune.
            seed: Seed of the TPE sampler.
            greater_is_better: Whether the study maximizes (True) or minimizes
                (False) the told scores.

        Raises:
            ConfigError: If Optuna is not installed.
        """
        super().__init__(space, seed, greater_is_better)
        if not self.is_available():
            raise errors.ConfigError(
                "optimizer 'optuna' needs the 'optimization' extra"
            )
        import optuna  # noqa: PLC0415 -- optional dependency

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study = optuna.create_study(
            direction="maximize" if greater_is_better else "minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        self.pending: list[tuple[dict[str, Any], Any]] = []

    def ask(self) -> dict[str, Any]:
        """Asks the study for the next configuration.

        Returns:
            A mapping from parameter name to the value suggested by the study's
            trial. The trial is held pending until `tell` receives its score.
        """
        trial = self.study.ask()
        params = {p.name: suggest(p, trial) for p in self.space.parameters}
        self.pending.append((params, trial))
        return params

    def tell(self, params: dict[str, Any], score: float) -> None:
        """Feeds the score back to the matching pending trial.

        Args:
            params: A configuration previously returned by `ask`. If no pending
                trial has equal parameters, the call does nothing.
            score: Its score.
        """
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
        The optimizer registered under `name`.

    Raises:
        ConfigError: If the name is unknown, or if the optimizer's constructor
            raises it because its dependency is missing.
    """
    if name not in OPTIMIZERS:
        raise errors.ConfigError(
            f"unknown optimizer '{name}'; available: {sorted(OPTIMIZERS)}"
        )
    return OPTIMIZERS[name](space, seed, greater_is_better)
