"""Task kinds and the literal choices the configuration accepts."""

import enum
from typing import Literal


class Task(enum.StrEnum):
    """Supervised learning task kinds."""

    BINARY = "binary_classification"
    MULTICLASS = "multiclass_classification"
    REGRESSION = "regression"
    SURVIVAL = "survival"
    ORDINAL = "ordinal"
    RANKING = "ranking"
    MULTILABEL = "multilabel"
    MULTIOUTPUT = "multioutput"
    FORECAST = "forecast"
    UPLIFT = "uplift"

    @property
    def classification(self) -> bool:
        """Whether the task predicts discrete classes."""
        return self in (Task.BINARY, Task.MULTICLASS)


STANDARD_TASKS = frozenset({Task.BINARY, Task.MULTICLASS, Task.REGRESSION})
"""Tasks scikit-learn style estimators handle; survival needs its own models."""

TaskRequest = Literal[
    "auto",
    "classification",
    "regression",
    "survival",
    "ordinal",
    "ranking",
    "multilabel",
    "multioutput",
    "forecast",
    "uplift",
    "binary_classification",
    "multiclass_classification",
]
"""Public task requests; `classification` is resolved from the target."""

DatasetModality = Literal[
    "tabular",
    "text",
    "document",
    "image",
    "audio",
    "video",
    "graph",
    "geospatial",
    "raster",
]
"""Dataset families accepted by the configuration contract.

The modality is an explicit declaration, not an inference from file suffixes.
Each declared modality must still cross the capability registry before a run
can train a model.
"""
