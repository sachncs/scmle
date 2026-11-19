"""The stage interface every workflow stage implements."""

import abc
from typing import ClassVar

from scmle.pipeline import products
import scmle.pipeline.context
import scmle.runtime.state


class Stage(abc.ABC):
    """One step of the workflow.

    Attributes:
        state: Lifecycle state the run is in while the stage executes.
        requires: Stages that must run before this one (the workflow is
            ordered by `engine.order`).
        makes: What the stage makes (`products.Product`): read from the
            context after the stage, kept with the stage record, and applied
            instead of running the stage again when a run resumes. None
            for a stage that declares none: it, and every stage after it,
            then runs again on resume.
    """

    state: ClassVar[scmle.runtime.state.Progress]
    requires: ClassVar[tuple[type, ...]] = ()
    makes: ClassVar[type[products.Product] | None] = None

    @abc.abstractmethod
    def describe(self, context: scmle.pipeline.context.Context) -> str:
        """Returns the reason recorded on entering the stage.

        Args:
            context: The run context.

        Returns:
            The reason text attached to the stage's state transition.
        """

    @abc.abstractmethod
    def execute(
        self, context: scmle.pipeline.context.Context
    ) -> products.Product | None:
        """Performs the stage, reading the context.

        Args:
            context: The run context; earlier stages' products are read from
                it.

        Returns:
            What the stage made, which the engine applies to the context and
            keeps with the stage record; None for a stage that writes the
            context itself.
        """
