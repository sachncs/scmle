"""The standard stage sequence."""

from scmle.pipeline import engine
from scmle.pipeline.stages import assessment
from scmle.pipeline.stages import ingestion
from scmle.pipeline.stages import interface
from scmle.pipeline.stages import packaging
from scmle.pipeline.stages import planning
from scmle.pipeline.stages import profiling
from scmle.pipeline.stages import reservation
from scmle.pipeline.stages import selecting
from scmle.pipeline.stages import training


def default_stages() -> tuple[interface.Stage, ...]:
    """Returns the standard stage sequence, in lifecycle order.

    Several stages may share a lifecycle state: the reservation runs in the
    validating state, right after ingestion.
    """
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
        )
    )
