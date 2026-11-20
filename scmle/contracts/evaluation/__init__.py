"""Task-aware evaluation and release policy contracts.

The contract is intentionally independent from metric and diagnostic
implementations. It describes what a run promises to measure and what
evidence is required before a model may be released. A policy is not inferred
from the presence of an estimator or a diagnostic class: every supported
combination must be declared explicitly.

`support` holds what each task supports, `parts` the pieces of a policy and
`policy` the versioned `Criteria` record and its registry. Load a stored
policy with `Criteria.model_validate` and write one with
`model_dump(mode="json")`.
"""

from scmle.contracts.evaluation.parts import Calibration
from scmle.contracts.evaluation.parts import Declaration
from scmle.contracts.evaluation.parts import Equity
from scmle.contracts.evaluation.parts import Explanations
from scmle.contracts.evaluation.parts import Fairness
from scmle.contracts.evaluation.parts import Gates
from scmle.contracts.evaluation.parts import Hurdle
from scmle.contracts.evaluation.parts import Requirement
from scmle.contracts.evaluation.parts import Subgroups
from scmle.contracts.evaluation.parts import Uncertainty
from scmle.contracts.evaluation.policy import Criteria
from scmle.contracts.evaluation.policy import Registry
from scmle.contracts.evaluation.support import Explicability
from scmle.contracts.evaluation.support import Method
from scmle.contracts.evaluation.support import METRIC_DIRECTIONS
from scmle.contracts.evaluation.support import Orientation
from scmle.contracts.evaluation.support import Procedure
from scmle.contracts.evaluation.support import Remedy
from scmle.contracts.evaluation.support import Technique
from scmle.contracts.evaluation.support import UndeclaredError
from scmle.contracts.evaluation.support import UnsupportedError

__all__ = [
    "METRIC_DIRECTIONS",
    "Calibration",
    "Criteria",
    "Declaration",
    "Equity",
    "Explanations",
    "Explicability",
    "Fairness",
    "Gates",
    "Hurdle",
    "Method",
    "Orientation",
    "Procedure",
    "Registry",
    "Remedy",
    "Requirement",
    "Subgroups",
    "Technique",
    "Uncertainty",
    "UndeclaredError",
    "UnsupportedError",
]
