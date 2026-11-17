"""PyTorch training engines. They run only inside the worker process.

Importing this module needs PyTorch, so nothing outside `worker` loads it;
the estimators users see are the proxies in `estimators`.
"""

from scmle.models.deeplearning.engines.heads import Classifier
from scmle.models.deeplearning.engines.heads import Regressor
from scmle.models.deeplearning.engines.network import MIN_ROWS_FOR_VALIDATION
from scmle.models.deeplearning.engines.network import Plan
from scmle.models.deeplearning.engines.network import SINGLE_MEMBER_OUTPUT_DIMS
from scmle.models.deeplearning.engines.trainer import Trainer

__all__ = [
    "MIN_ROWS_FOR_VALIDATION",
    "SINGLE_MEMBER_OUTPUT_DIMS",
    "Classifier",
    "Plan",
    "Regressor",
    "Trainer",
]
