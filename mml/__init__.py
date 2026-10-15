"""magicml: autonomous, leakage-safe ML lifecycle framework."""

__version__ = "0.1.0.dev0"

from mml.api.project import MLProject
from mml.api.results import RunResult
from mml.artifacts.bundle import ModelBundle

__all__ = ["MLProject", "ModelBundle", "RunResult", "__version__"]
