"""Open-source benchmark datasets.

Each `BenchmarkDataset` knows how to load itself and how to describe the
problem; `SklearnDataset` ships with scikit-learn (offline) while
`OpenMlDataset` downloads from OpenML (cached by scikit-learn).
"""

import abc
import dataclasses
from typing import ClassVar

import pandas as pd
from sklearn import datasets

from mml.contracts import tasks


@dataclasses.dataclass(frozen=True)
class BenchmarkDataset(abc.ABC):
    """A named supervised problem.

    Attributes:
        name: Display name.
        target: Target column name.
        task: Task request passed to `MLProject`.
        metric: Primary metric.
        max_rows: Stratification-free row cap for slow downloads (None = all).
        online: Whether loading needs the network.
    """

    name: str
    target: str
    task: tasks.TaskRequest
    metric: str
    max_rows: int | None = None
    online: ClassVar[bool] = False

    @abc.abstractmethod
    def fetch(self) -> pd.DataFrame:
        """Downloads/loads the raw table, including the target column."""

    def load(self) -> pd.DataFrame:
        """Loads the table, subsampling deterministically if capped."""
        frame = self.fetch()
        if self.max_rows is not None and len(frame) > self.max_rows:
            frame = frame.sample(self.max_rows, random_state=0)
        return frame.reset_index(drop=True)


@dataclasses.dataclass(frozen=True)
class SklearnDataset(BenchmarkDataset):
    """A dataset bundled with scikit-learn.

    Attributes:
        loader: Name of the `sklearn.datasets.load_*` function.
    """

    loader: str = ""

    def fetch(self) -> pd.DataFrame:
        """Loads the bundled dataset as a DataFrame."""
        bunch = getattr(datasets, self.loader)(as_frame=True)
        frame = bunch.frame.copy()
        if self.target not in frame.columns:
            frame[self.target] = bunch.target
        return frame


@dataclasses.dataclass(frozen=True)
class OpenMlDataset(BenchmarkDataset):
    """A dataset downloaded from OpenML.

    Attributes:
        openml_name: Dataset name on OpenML.
        version: Dataset version.
    """

    openml_name: str = ""
    version: int = 1
    online: ClassVar[bool] = True

    def fetch(self) -> pd.DataFrame:
        """Downloads the dataset (cached after the first call)."""
        bunch = datasets.fetch_openml(
            self.openml_name, version=self.version, as_frame=True, parser="auto"
        )
        frame = bunch.frame.copy()
        for column in frame.select_dtypes("category").columns:
            frame[column] = frame[column].astype(object)
        return frame


OFFLINE = (
    SklearnDataset(
        "breast_cancer",
        "target",
        "classification",
        "roc_auc",
        loader="load_breast_cancer",
    ),
    SklearnDataset(
        "wine", "target", "classification", "log_loss", loader="load_wine"
    ),
    SklearnDataset(
        "digits", "target", "classification", "log_loss", loader="load_digits"
    ),
    SklearnDataset(
        "diabetes_progression",
        "target",
        "regression",
        "rmse",
        loader="load_diabetes",
    ),
)

ONLINE = (
    OpenMlDataset(
        "credit-g", "class", "classification", "roc_auc", openml_name="credit-g"
    ),
    OpenMlDataset(
        "titanic",
        "survived",
        "classification",
        "roc_auc",
        openml_name="titanic",
    ),
    OpenMlDataset(
        "kc1_defects", "defects", "classification", "pr_auc", openml_name="kc1"
    ),
    OpenMlDataset(
        "adult_income",
        "class",
        "classification",
        "roc_auc",
        max_rows=12000,
        openml_name="adult",
    ),
    OpenMlDataset(
        "california_housing",
        "median_house_value",
        "regression",
        "rmse",
        max_rows=10000,
        openml_name="california_housing",
    ),
)

ALL = OFFLINE + ONLINE
