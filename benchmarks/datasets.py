"""Open-source benchmark datasets.

Each `BenchmarkDataset` knows how to load itself and how to describe the
problem; `SklearnDataset` ships with scikit-learn (offline) while
`OpenMlDataset` downloads from OpenML (cached by scikit-learn).
"""

import abc
import dataclasses
from typing import ClassVar

import pandas
from sklearn import datasets

from scmle.contracts import tasks


@dataclasses.dataclass(frozen=True)
class BenchmarkDataset(abc.ABC):
    """A named supervised problem.

    Attributes:
        name: Display name.
        target: Target column name.
        task: Task request passed to `Project`.
        metric: Primary metric.
        max_rows: Stratification-free row cap for slow downloads (None = all).
        drop: Columns that are identifiers or free text, never features.
        numeric_target: The target is stored as text (an OpenML category) but
            is a number.
        time_column: Column ordering the rows, for chronological validation.
        time_as_feature: Whether the model may also use that column.
        online: Whether loading needs the network.
    """

    name: str
    target: str
    task: tasks.TaskRequest
    metric: str
    max_rows: int | None = None
    drop: tuple[str, ...] = ()
    numeric_target: bool = False
    time_column: str | None = None
    time_as_feature: bool = False
    online: ClassVar[bool] = False

    @abc.abstractmethod
    def fetch(self) -> pandas.DataFrame:
        """Downloads/loads the raw table, including the target column.

        Returns:
            The full table before any dropping or subsampling.
        """

    def load(self) -> pandas.DataFrame:
        """Loads the table, subsampling deterministically if capped.

        Drops the `drop` columns, converts the target to numbers when
        `numeric_target` is set, and samples `max_rows` rows with a fixed seed
        when the table is longer than that.

        Returns:
            The table with a fresh 0..n-1 index.
        """
        frame = self.fetch().drop(columns=list(self.drop))
        if self.numeric_target:
            frame[self.target] = pandas.to_numeric(frame[self.target])
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

    def fetch(self) -> pandas.DataFrame:
        """Loads the bundled dataset as a DataFrame.

        Returns:
            The scikit-learn frame; when it has no target column, the bunch's
            target is added under `target`.
        """
        bunch = getattr(datasets, self.loader)(as_frame=True)
        frame: pandas.DataFrame = bunch.frame.copy()
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

    def fetch(self) -> pandas.DataFrame:
        """Downloads the dataset (cached after the first call).

        Returns:
            The OpenML frame with categorical columns converted to object dtype.
        """
        bunch = datasets.fetch_openml(
            self.openml_name, version=self.version, as_frame=True, parser="auto"
        )
        frame: pandas.DataFrame = bunch.frame.copy()
        for column in frame.select_dtypes("category").columns:
            frame[column] = frame[column].astype(object)
        return frame


@dataclasses.dataclass(frozen=True)
class HuggingFaceDataset(BenchmarkDataset):
    """A table stored in a Hugging Face dataset repository.

    Attributes:
        repo: Repository id, for example `inria-soda/tabular-benchmark`.
        filename: CSV or Parquet file inside the repository.
    """

    repo: str = ""
    filename: str = ""
    online: ClassVar[bool] = True

    def fetch(self) -> pandas.DataFrame:
        """Downloads the file (cached by `huggingface_hub`) and reads it.

        Returns:
            The table, read as Parquet when the file name ends in `.parquet` and
            as CSV otherwise.
        """
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(self.repo, self.filename, repo_type="dataset")
        if self.filename.endswith(".parquet"):
            return pandas.read_parquet(path)
        return pandas.read_csv(path)


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


def open_ml(
    name: str,
    openml_name: str,
    target: str,
    task: tasks.TaskRequest,
    metric: str,
    version: int = 1,
    max_rows: int | None = None,
    drop: tuple[str, ...] = (),
    time_column: str | None = None,
    time_as_feature: bool = False,
    numeric_target: bool = False,
) -> OpenMlDataset:
    """Describes an OpenML dataset (shorthand for the extended suite).

    Args:
        name: Display name.
        openml_name: Dataset name on OpenML.
        target: Target column name.
        task: Task request passed to `Project`.
        metric: Primary metric.
        version: OpenML dataset version.
        max_rows: Row cap applied when loading (None = all rows).
        drop: Columns that are identifiers or free text, never features.
        time_column: Column ordering the rows, for chronological validation.
        time_as_feature: Whether the model may also use that column.
        numeric_target: The target is stored as text but is a number.

    Returns:
        The dataset description; nothing is downloaded until `fetch`.
    """
    return OpenMlDataset(
        name,
        target,
        task,
        metric,
        max_rows=max_rows,
        drop=drop,
        numeric_target=numeric_target,
        time_column=time_column,
        time_as_feature=time_as_feature,
        openml_name=openml_name,
        version=version,
    )


EXTENDED = (
    # Binary classification: tiny and imbalanced, zeros-as-missing, many
    # numeric features, categorical-heavy, extreme imbalance, mixed types,
    # chronological order, missing values.
    open_ml(
        "blood_transfusion",
        "blood-transfusion-service-center",
        "Class",
        "classification",
        "roc_auc",
    ),
    open_ml("pima_diabetes", "diabetes", "class", "classification", "roc_auc"),
    open_ml("phoneme", "phoneme", "Class", "classification", "roc_auc"),
    open_ml("spambase", "spambase", "class", "classification", "roc_auc"),
    open_ml(
        "bank_marketing",
        "bank-marketing",
        "Class",
        "classification",
        "roc_auc",
        max_rows=15000,
    ),
    open_ml(
        "credit_card_fraud",
        "creditcard",
        "Class",
        "classification",
        "pr_auc",
        max_rows=40000,
    ),
    open_ml("telco_churn", "churn", "class", "classification", "roc_auc"),
    open_ml(
        "electricity",
        "electricity",
        "class",
        "classification",
        "roc_auc",
        max_rows=12000,
        time_column="date",
        time_as_feature=True,
    ),
    open_ml(
        "credit_approval",
        "credit-approval",
        "class",
        "classification",
        "roc_auc",
    ),
    open_ml(
        "horse_colic",
        "colic",
        "surgical_lesion",
        "classification",
        "roc_auc",
        version=2,
    ),
    # Multiclass: few and many classes, small and large.
    open_ml("vehicle", "vehicle", "Class", "classification", "log_loss"),
    open_ml("segment", "segment", "class", "classification", "log_loss"),
    open_ml(
        "letter",
        "letter",
        "class",
        "classification",
        "log_loss",
        max_rows=10000,
    ),
    open_ml(
        "covertype",
        "covertype",
        "class",
        "classification",
        "log_loss",
        version=3,
        max_rows=20000,
    ),
    # Regression: mixed types, counts, heavy-tailed prices, smooth functions.
    open_ml(
        "house_sales",
        "house_sales",
        "price",
        "regression",
        "rmse",
        version=3,
        max_rows=12000,
    ),
    open_ml(
        "bike_sharing",
        "Bike_Sharing_Demand",
        "count",
        "regression",
        "rmse",
        version=2,
    ),
    open_ml(
        "abalone",
        "abalone",
        "Class_number_of_rings",
        "regression",
        "rmse",
        numeric_target=True,
    ),
    open_ml(
        "diamonds",
        "diamonds",
        "price",
        "regression",
        "rmse",
        max_rows=15000,
    ),
    open_ml("cpu_act", "cpu_act", "usr", "regression", "rmse"),
    open_ml("wine_quality", "wine_quality", "quality", "regression", "rmse"),
    open_ml("kin8nm", "kin8nm", "y", "regression", "rmse"),
)


HUB_REPO = "inria-soda/tabular-benchmark"


def hugging_face(
    name: str,
    filename: str,
    target: str,
    task: tasks.TaskRequest,
    metric: str,
    repo: str = HUB_REPO,
    max_rows: int | None = None,
    drop: tuple[str, ...] = (),
) -> HuggingFaceDataset:
    """Describes a Hugging Face table (shorthand for the suite below).

    Args:
        name: Display name.
        filename: CSV or Parquet file inside the repository.
        target: Target column name.
        task: Task request passed to `Project`.
        metric: Primary metric.
        repo: Repository id of the dataset repository.
        max_rows: Row cap applied when loading (None = all rows).
        drop: Columns that are identifiers or free text, never features.

    Returns:
        The dataset description; nothing is downloaded until `fetch`.
    """
    return HuggingFaceDataset(
        name,
        target,
        task,
        metric,
        max_rows=max_rows,
        drop=drop,
        repo=repo,
        filename=filename,
    )


def classification(
    name: str,
    filename: str,
    target: str,
    repo: str = HUB_REPO,
    max_rows: int | None = None,
) -> HuggingFaceDataset:
    """A binary classification table of the curated benchmark.

    Args:
        name: Display name.
        filename: CSV or Parquet file inside the repository.
        target: Target column name.
        repo: Repository id of the dataset repository.
        max_rows: Row cap applied when loading (None = all rows).

    Returns:
        The dataset description, scored by ROC AUC.
    """
    return hugging_face(
        name,
        filename,
        target,
        "classification",
        "roc_auc",
        repo,
        max_rows,
    )


def regression(
    name: str,
    filename: str,
    target: str,
    repo: str = HUB_REPO,
    max_rows: int | None = None,
    drop: tuple[str, ...] = (),
) -> HuggingFaceDataset:
    """A regression table of the curated benchmark.

    Args:
        name: Display name.
        filename: CSV or Parquet file inside the repository.
        target: Target column name.
        repo: Repository id of the dataset repository.
        max_rows: Row cap applied when loading (None = all rows).
        drop: Columns that are identifiers or free text, never features.

    Returns:
        The dataset description, scored by RMSE.
    """
    return hugging_face(
        name, filename, target, "regression", "rmse", repo, max_rows, drop
    )


# The curated tabular benchmark of Grinsztajn et al. plus scikit-learn's
# small tables, all downloaded from the Hugging Face Hub.
HUGGING_FACE = (
    classification("hf_compas", "clf_cat/compas-two-years.csv", "twoyearrecid"),
    classification(
        "hf_credit_default", "clf_cat/default-of-credit-card-clients.csv", "y"
    ),
    classification("hf_eye_movements", "clf_cat/eye_movements.csv", "label"),
    classification("hf_albert", "clf_cat/albert.csv", "class", max_rows=15000),
    classification("hf_bioresponse", "clf_num/Bioresponse.csv", "target"),
    classification("hf_magic_telescope", "clf_num/MagicTelescope.csv", "class"),
    classification("hf_heloc", "clf_num/heloc.csv", "RiskPerformance"),
    classification("hf_pol", "clf_num/pol.csv", "binaryClass"),
    classification("hf_jannis", "clf_num/jannis.csv", "class", max_rows=15000),
    classification(
        "hf_adult_census",
        "adult.csv",
        "income",
        repo="scikit-learn/adult-census-income",
        max_rows=15000,
    ),
    regression(
        "hf_brazilian_houses", "reg_cat/Brazilian_houses.csv", "total_(BRL)"
    ),
    regression(
        "hf_mercedes",
        "reg_cat/Mercedes_Benz_Greener_Manufacturing.csv",
        "y",
    ),
    regression(
        "hf_medical_charges",
        "reg_cat/medical_charges.csv",
        "AverageTotalPayments",
        max_rows=15000,
    ),
    regression("hf_ailerons", "reg_num/Ailerons.csv", "goal"),
    regression("hf_miami_housing", "reg_num/MiamiHousing2016.csv", "SALEPRC"),
    regression("hf_elevators", "reg_num/elevators.csv", "Goal"),
    regression(
        "hf_superconduct",
        "reg_num/superconduct.csv",
        "criticaltemp",
        max_rows=12000,
    ),
    regression("hf_sulfur", "reg_num/sulfur.csv", "y1"),
    regression(
        "hf_auto_mpg",
        "auto-mpg.csv",
        "mpg",
        repo="scikit-learn/auto-mpg",
        drop=("car name",),
    ),
    regression(
        "hf_fish_weight", "Fish.csv", "Weight", repo="scikit-learn/Fish"
    ),
)

ALL = OFFLINE + ONLINE
SUITES = {
    "core": ALL,
    "extended": EXTENDED,
    "huggingface": HUGGING_FACE,
    "everything": ALL + EXTENDED + HUGGING_FACE,
}
