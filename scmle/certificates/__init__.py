"""Certificate stores shipped with the package.

Each store is a JSON file written by the qualification harness
(`python -m benchmarks.qualification`) and read by the gate it certifies.
The stores live inside the package, so an installed wheel gates exactly as
a source checkout does: a gate never turns descriptive because the harness
folder is absent (ERR-1).
"""

import importlib.resources
import pathlib

DIRECTORY = pathlib.Path(str(importlib.resources.files(__name__)))
