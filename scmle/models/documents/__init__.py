"""A bounded, explainable baseline lifecycle for labelled documents.

This module covers documents only. It uses the validated document contracts,
keeps feature preparation and modelling in one immutable configuration, and
refuses to imply that a serialized model is safe to load without an explicit
trust decision.
"""

from scmle.models.documents.archive import file_digest
from scmle.models.documents.classifier import Classifier
from scmle.models.documents.monitor import Monitor
from scmle.models.documents.records import Config
from scmle.models.documents.records import Drift
from scmle.models.documents.records import Evaluation
from scmle.models.documents.records import Explanation
from scmle.models.documents.records import Package

__all__ = [
    "Classifier",
    "Config",
    "Drift",
    "Evaluation",
    "Explanation",
    "Monitor",
    "Package",
    "file_digest",
]
