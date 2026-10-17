"""magicml: autonomous, leakage-safe ML lifecycle framework."""

__version__ = "0.1.0b1"

from mml.api.predict import load_model
from mml.api.predict import predict
from mml.api.project import MLProject
from mml.api.results import RunResult
from mml.artifacts.bundle import ModelBundle
from mml.evaluation.metrics import register_metric
from mml.models.custom import register_estimator
from mml.models.custom import unregister_estimator

__all__ = [
    "MLProject",
    "ModelBundle",
    "RunResult",
    "__version__",
    "load_model",
    "predict",
    "register_estimator",
    "register_metric",
    "unregister_estimator",
]
