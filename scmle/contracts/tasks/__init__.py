"""Typed, versioned project specification (pydantic).

`kinds` holds the task kinds and the literal choices, `graph` the derived
feature graph, `sections` the other configuration sections and `config` the
configuration itself with the rules that span sections.
"""

from scmle.contracts.tasks.config import Config
from scmle.contracts.tasks.config import CONFIG_VERSION
from scmle.contracts.tasks.config import FAIRNESS_WEIGHT
from scmle.contracts.tasks.config import FAST_BUDGET_SECONDS
from scmle.contracts.tasks.graph import Derivation
from scmle.contracts.tasks.graph import GRAPH_VERSION
from scmle.contracts.tasks.graph import horizons
from scmle.contracts.tasks.graph import Node
from scmle.contracts.tasks.kinds import DatasetModality
from scmle.contracts.tasks.kinds import STANDARD_TASKS
from scmle.contracts.tasks.kinds import Task
from scmle.contracts.tasks.kinds import TaskRequest
from scmle.contracts.tasks.sections import Candidates
from scmle.contracts.tasks.sections import Dataset
from scmle.contracts.tasks.sections import Parity
from scmle.contracts.tasks.sections import Problem
from scmle.contracts.tasks.sections import Resources
from scmle.contracts.tasks.sections import Search
from scmle.contracts.tasks.sections import Selection
from scmle.contracts.tasks.sections import Stacking
from scmle.contracts.tasks.sections import Strategies
from scmle.contracts.tasks.sections import Threshold
from scmle.contracts.tasks.sections import Validation

__all__ = [
    "CONFIG_VERSION",
    "FAIRNESS_WEIGHT",
    "FAST_BUDGET_SECONDS",
    "GRAPH_VERSION",
    "STANDARD_TASKS",
    "Candidates",
    "Config",
    "Dataset",
    "DatasetModality",
    "Derivation",
    "Node",
    "Parity",
    "Problem",
    "Resources",
    "Search",
    "Selection",
    "Stacking",
    "Strategies",
    "Task",
    "TaskRequest",
    "Threshold",
    "Validation",
    "horizons",
]
