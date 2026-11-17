"""The standard stage sequence."""

from scmle.pipeline import compiler
from scmle.pipeline.stages import assessment
from scmle.pipeline.stages import ingestion
from scmle.pipeline.stages import interface
from scmle.pipeline.stages import packaging
from scmle.pipeline.stages import planning
from scmle.pipeline.stages import profiling
from scmle.pipeline.stages import selecting
from scmle.pipeline.stages import training


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
