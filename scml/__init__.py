"""scml: autonomous, leakage-safe ML lifecycle framework."""

__version__ = "0.1.0b1"

from scml.api.predict import load_model
from scml.api.predict import predict
from scml.api.project import MLProject
from scml.api.project import ModalityProject
from scml.api.results import RunResult
from scml.artifacts.bundle import ModelBundle
from scml.evaluation.metrics import register_metric
from scml.models.custom import register_estimator
from scml.models.custom import unregister_estimator

__all__ = [
    "MLProject",
    "ModalityProject",
    "ModelBundle",
    "RunResult",
    "__version__",
    "load_model",
    "predict",
    "register_estimator",
    "register_metric",
    "unregister_estimator",
]
