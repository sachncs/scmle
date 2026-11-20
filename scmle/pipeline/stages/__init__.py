"""Workflow stages.

Each `Stage` subclass owns one state of the run lifecycle and implements
`execute`; the orchestrator treats them uniformly. Because stages read
their inputs from the run's board (`engine.Board`) and return their
products, the ingest/profile prefix is reused for the side-effect-free plan
preview.
"""

import logging

from scmle.pipeline.stages.assessment import Assessment
from scmle.pipeline.stages.cost import affordable
from scmle.pipeline.stages.cost import dimensions
from scmle.pipeline.stages.cost import FIT_MARGIN
from scmle.pipeline.stages.cost import foldcost
from scmle.pipeline.stages.cost import halt
from scmle.pipeline.stages.cost import late
from scmle.pipeline.stages.cost import meter
from scmle.pipeline.stages.cost import refitcost
from scmle.pipeline.stages.cost import TAIL_SECONDS
from scmle.pipeline.stages.defaults import sequence
from scmle.pipeline.stages.examination import DIAGNOSTIC_BUDGET_SHARE
from scmle.pipeline.stages.examination import DIAGNOSTIC_MIN_SECONDS
from scmle.pipeline.stages.exploration import coverage_on_confirmation
from scmle.pipeline.stages.exploration import Escrow
from scmle.pipeline.stages.exploration import governing
from scmle.pipeline.stages.ingestion import Ingest
from scmle.pipeline.stages.interface import Stage
from scmle.pipeline.stages.packaging import Packaging
from scmle.pipeline.stages.packaging import PARITY_ROWS
from scmle.pipeline.stages.planning import advise
from scmle.pipeline.stages.planning import devise
from scmle.pipeline.stages.planning import Planning
from scmle.pipeline.stages.planning import requisition
from scmle.pipeline.stages.profiling import Profiling
from scmle.pipeline.stages.scoring import anchor
from scmle.pipeline.stages.scoring import appraise
from scmle.pipeline.stages.scoring import choose
from scmle.pipeline.stages.scoring import COVERAGE_LEVELS
from scmle.pipeline.stages.scoring import decisional
from scmle.pipeline.stages.scoring import MATRIX_DIMENSIONS
from scmle.pipeline.stages.scoring import pin
from scmle.pipeline.stages.scoring import slicers
from scmle.pipeline.stages.selecting import Candidate
from scmle.pipeline.stages.selecting import frontier
from scmle.pipeline.stages.selecting import Selection
from scmle.pipeline.stages.summary import BASELINE
from scmle.pipeline.stages.surveying import graft
from scmle.pipeline.stages.training import Training

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
    "Assessment",
    "Candidate",
    "Escrow",
    "Ingest",
    "Packaging",
    "Planning",
    "Profiling",
    "Selection",
    "Stage",
    "Training",
    "advise",
    "affordable",
    "anchor",
    "appraise",
    "choose",
    "coverage_on_confirmation",
    "decisional",
    "devise",
    "dimensions",
    "foldcost",
    "frontier",
    "governing",
    "graft",
    "halt",
    "late",
    "meter",
    "pin",
    "refitcost",
    "requisition",
    "sequence",
    "slicers",
]

LOGGER = logging.getLogger(__name__)
