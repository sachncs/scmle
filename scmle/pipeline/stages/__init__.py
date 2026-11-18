"""Workflow stages.

Each `Stage` subclass owns one state of the run lifecycle and implements
`execute`; the orchestrator treats them uniformly. Because stages only read
and write a `Context`, the ingest/profile prefix is reused for the
side-effect-free plan preview.
"""

import logging

from scmle.pipeline.stages.assessment import FinalEvaluationStage
from scmle.pipeline.stages.cost import affordable
from scmle.pipeline.stages.cost import dimensions
from scmle.pipeline.stages.cost import FIT_MARGIN
from scmle.pipeline.stages.cost import foldcost
from scmle.pipeline.stages.cost import halt
from scmle.pipeline.stages.cost import late
from scmle.pipeline.stages.cost import meter
from scmle.pipeline.stages.cost import refitcost
from scmle.pipeline.stages.cost import TAIL_SECONDS
from scmle.pipeline.stages.defaults import default_stages
from scmle.pipeline.stages.examination import DIAGNOSTIC_BUDGET_SHARE
from scmle.pipeline.stages.examination import DIAGNOSTIC_MIN_SECONDS
from scmle.pipeline.stages.exploration import Escrow
from scmle.pipeline.stages.exploration import governing
from scmle.pipeline.stages.ingestion import IngestStage
from scmle.pipeline.stages.interface import Stage
from scmle.pipeline.stages.packaging import PackageStage
from scmle.pipeline.stages.packaging import PARITY_ROWS
from scmle.pipeline.stages.planning import advise
from scmle.pipeline.stages.planning import devise
from scmle.pipeline.stages.planning import Planning
from scmle.pipeline.stages.planning import requisition
from scmle.pipeline.stages.profiling import ProfileStage
from scmle.pipeline.stages.scoring import appraise
from scmle.pipeline.stages.scoring import choose_decision_threshold
from scmle.pipeline.stages.scoring import COVERAGE_LEVELS
from scmle.pipeline.stages.scoring import coverage_on_confirmation
from scmle.pipeline.stages.scoring import decisional
from scmle.pipeline.stages.scoring import freeze
from scmle.pipeline.stages.scoring import MATRIX_DIMENSIONS
from scmle.pipeline.stages.scoring import pin
from scmle.pipeline.stages.scoring import reference_scores
from scmle.pipeline.stages.scoring import slicers
from scmle.pipeline.stages.selecting import Candidate
from scmle.pipeline.stages.selecting import frontier
from scmle.pipeline.stages.selecting import SelectStage
from scmle.pipeline.stages.summary import BASELINE
from scmle.pipeline.stages.surveying import graft
from scmle.pipeline.stages.training import TrainStage

__all__ = [
    "BASELINE",
    "COVERAGE_LEVELS",
    "DIAGNOSTIC_BUDGET_SHARE",
    "DIAGNOSTIC_MIN_SECONDS",
    "FIT_MARGIN",
    "LOGGER",
    "MATRIX_DIMENSIONS",
    "PARITY_ROWS",
    "TAIL_SECONDS",
    "Candidate",
    "Escrow",
    "FinalEvaluationStage",
    "IngestStage",
    "PackageStage",
    "Planning",
    "ProfileStage",
    "SelectStage",
    "Stage",
    "TrainStage",
    "advise",
    "affordable",
    "appraise",
    "choose_decision_threshold",
    "coverage_on_confirmation",
    "decisional",
    "default_stages",
    "devise",
    "dimensions",
    "foldcost",
    "freeze",
    "frontier",
    "governing",
    "graft",
    "halt",
    "late",
    "meter",
    "pin",
    "reference_scores",
    "refitcost",
    "requisition",
    "slicers",
]

LOGGER = logging.getLogger(__name__)
