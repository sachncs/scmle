"""The standard stage sequence."""

from scmle.core import compiler
from scmle.core.stages import assessment
from scmle.core.stages import ingestion
from scmle.core.stages import interface
from scmle.core.stages import packaging
from scmle.core.stages import planning
from scmle.core.stages import profiling
from scmle.core.stages import selecting
from scmle.core.stages import training


def default_stages() -> tuple[interface.Stage, ...]:
    """Returns the standard stage sequence, one per lifecycle state."""
    return compiler.assemble(
        (
            ingestion.IngestStage(),
            profiling.ProfileStage(),
            planning.Planning(),
            training.TrainStage(),
            selecting.SelectStage(),
            assessment.FinalEvaluationStage(),
            packaging.PackageStage(),
        )
    )
