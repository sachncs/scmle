"""The stage interface every workflow stage implements."""

import abc
from typing import ClassVar

import scmle.pipeline.context
import scmle.runtime.state


class Stage(abc.ABC):
    """One step of the workflow.

    Attributes:
        state: Lifecycle state the run is in while the stage executes.
        requires: Stages that must run before this one (the workflow is
            ordered by `engine.order`).
    """

    state: ClassVar[scmle.runtime.state.Progress]
    requires: ClassVar[tuple[type, ...]] = ()

    @abc.abstractmethod
    def describe(self, context: scmle.pipeline.context.Context) -> str:
        """Returns the reason recorded on entering the stage.

        Args:
            context: The run context.

        Returns:
            The reason text attached to the stage's state transition.
        """

    @abc.abstractmethod
    def execute(self, context: scmle.pipeline.context.Context) -> None:
        """Performs the stage, reading and writing the context.

        Args:
            context: The run context; earlier stages' products are read from it
                and this stage's products are written to it.
        """
