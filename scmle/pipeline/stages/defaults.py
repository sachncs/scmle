"""The standard stage sequence."""

from scmle.pipeline.stages import assessment
from scmle.pipeline.stages import ingestion
from scmle.pipeline.stages import interface
from scmle.pipeline.stages import packaging
from scmle.pipeline.stages import planning
from scmle.pipeline.stages import profiling
from scmle.pipeline.stages import reservation
from scmle.pipeline.stages import selecting
from scmle.pipeline.stages import training
from scmle.runtime import engine
from scmle.runtime import plugins


def default_stages() -> tuple[interface.Stage, ...]:
    """Returns the standard stage sequence, in lifecycle order.

    Several stages may share a lifecycle state: the reservation runs in the
    validating state, right after ingestion. Stages enrolled in the plugin
    registry (kind `stages`; a stage, or a stage class built here) join the
    sequence where their `requires` and state place them.

    Returns:
        The stages, ordered by `engine.order`.
    """
    enrolled = tuple(
        item() if isinstance(item, type) else item
        for item in plugins.REGISTRY.entries(plugins.Kind.STAGES).values()
    )
    return engine.order(
        (
            ingestion.IngestStage(),
            reservation.Reservation(),
            profiling.ProfileStage(),
            planning.Planning(),
            training.TrainStage(),
            selecting.SelectStage(),
            assessment.FinalEvaluationStage(),
            packaging.PackageStage(),
            *enrolled,
        )
    )
