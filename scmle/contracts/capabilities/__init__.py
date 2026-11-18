"""Capability contracts for pre-training support decisions.

The registry is deliberately independent from model adapters. A planner can
ask whether a complete capability is eligible before it constructs a model,
loads an optional dependency, or starts training.

`axes` holds what a capability is declared along, `declarations` the
family, environment and capability records, `catalog` the registered
declarations and their resource checks and `registry` the registry.
"""

from scmle.contracts.capabilities.axes import Evaluation
from scmle.contracts.capabilities.axes import Explainability
from scmle.contracts.capabilities.axes import Fairness
from scmle.contracts.capabilities.axes import Inference
from scmle.contracts.capabilities.axes import Modality
from scmle.contracts.capabilities.axes import ModelFamily
from scmle.contracts.capabilities.axes import Status
from scmle.contracts.capabilities.axes import Uncertainty
from scmle.contracts.capabilities.declarations import Capability
from scmle.contracts.capabilities.declarations import Coordinate
from scmle.contracts.capabilities.declarations import Declaration
from scmle.contracts.capabilities.declarations import (
    DEFAULT_MODEL_FAMILY_DECLARATIONS,
)
from scmle.contracts.capabilities.declarations import Environment
from scmle.contracts.capabilities.declarations import Reason
from scmle.contracts.capabilities.declarations import Resources
from scmle.contracts.capabilities.declarations import UnsupportedError
from scmle.contracts.capabilities.registry import CAPABILITY_SCHEMA_VERSION
from scmle.contracts.capabilities.registry import Registry

__all__ = [
    "CAPABILITY_SCHEMA_VERSION",
    "DEFAULT_MODEL_FAMILY_DECLARATIONS",
    "Capability",
    "Coordinate",
    "Declaration",
    "Environment",
    "Evaluation",
    "Explainability",
    "Fairness",
    "Inference",
    "Modality",
    "ModelFamily",
    "Reason",
    "Registry",
    "Resources",
    "Status",
    "Uncertainty",
    "UnsupportedError",
]
