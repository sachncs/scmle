"""Workflow stages.

Each `Stage` subclass owns one state of the run lifecycle and implements
`execute`; the orchestrator treats them uniformly. Because stages only read
and write a `Context`, the ingest/profile prefix is reused for the
side-effect-free plan preview.
"""

import logging

from scmle.core.stages.assessment import BASELINE
from scmle.core.stages.assessment import FinalEvaluationStage
from scmle.core.stages.cost import affordable
from scmle.core.stages.cost import dimensions
from scmle.core.stages.cost import FIT_MARGIN
from scmle.core.stages.cost import foldcost
from scmle.core.stages.cost import halt
from scmle.core.stages.cost import late
from scmle.core.stages.cost import meter
from scmle.core.stages.cost import refitcost
from scmle.core.stages.cost import TAIL_SECONDS
from scmle.core.stages.defaults import default_stages
from scmle.core.stages.examination import DIAGNOSTIC_BUDGET_SHARE
from scmle.core.stages.examination import DIAGNOSTIC_MIN_SECONDS
from scmle.core.stages.exploration import Escrow
from scmle.core.stages.exploration import governing
from scmle.core.stages.ingestion import IngestStage
from scmle.core.stages.interface import Stage
from scmle.core.stages.packaging import PackageStage
from scmle.core.stages.packaging import PARITY_ROWS
from scmle.core.stages.planning import advise
from scmle.core.stages.planning import devise
from scmle.core.stages.planning import Planning
from scmle.core.stages.planning import requisition
from scmle.core.stages.profiling import ProfileStage
from scmle.core.stages.scoring import appraise
from scmle.core.stages.scoring import choose_decision_threshold
from scmle.core.stages.scoring import COVERAGE_LEVELS
from scmle.core.stages.scoring import coverage_on_confirmation
from scmle.core.stages.scoring import decisional
from scmle.core.stages.scoring import freeze
from scmle.core.stages.scoring import MATRIX_DIMENSIONS
from scmle.core.stages.scoring import pin
from scmle.core.stages.scoring import reference_scores
from scmle.core.stages.scoring import slicers
from scmle.core.stages.selecting import Candidate
from scmle.core.stages.selecting import frontier
from scmle.core.stages.selecting import SelectStage
from scmle.core.stages.surveying import graft
from scmle.core.stages.training import TrainStage

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
