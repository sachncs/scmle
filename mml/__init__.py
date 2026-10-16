"""magicml: autonomous, leakage-safe ML lifecycle framework."""

__version__ = "0.1.0b1"

from mml.api.predict import load_model
from mml.api.predict import predict
from mml.api.project import MLProject
from mml.api.results import RunResult
from mml.artifacts.bundle import ModelBundle

__all__ = [
    "MLProject",
    "ModelBundle",
    "RunResult",
    "__version__",
    "load_model",
    "predict",
]
