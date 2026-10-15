"""magicml: autonomous, leakage-safe ML lifecycle framework."""

__version__ = "0.1.0.dev0"

from mml.api.project import MLProject  # noqa: E402
from mml.api.results import RunResult  # noqa: E402
from mml.artifacts.bundle import ModelBundle  # noqa: E402

__all__ = ["MLProject", "ModelBundle", "RunResult", "__version__"]
