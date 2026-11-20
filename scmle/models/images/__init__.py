"""A bounded, explainable baseline lifecycle for labelled images.

The classifier deliberately uses deterministic resized pixels rather than
claiming a production computer-vision model. It is an experimental library
path that makes its curation, feature, evaluation, explanation, packaging,
and drift boundaries explicit.
"""

from scmle.models.images.archive import digest
from scmle.models.images.classifier import Classifier
from scmle.models.images.monitor import Monitor
from scmle.models.images.records import Config
from scmle.models.images.records import Drift
from scmle.models.images.records import Evaluation
from scmle.models.images.records import Explanation
from scmle.models.images.records import Package

__all__ = [
    "Classifier",
    "Config",
    "Drift",
    "Evaluation",
    "Explanation",
    "Monitor",
    "Package",
    "digest",
]
