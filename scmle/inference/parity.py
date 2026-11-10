"""Golden-record parity: the same records give the same outputs everywhere.

The reference model is written out by hand (`model_spec.json`): a logistic
function with fixed medians, scalings and coefficients, so nothing is fitted
and no platform can train a slightly different model. The expected outputs
are computed in closed form with numpy alone, independently of scikit-learn,
and compared with what a serving path returns under tolerances that were
declared before the comparison (`tolerances.json`).
"""

import dataclasses
import json
import pathlib
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

import numpy
import pandas
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from scmle.artifacts import bundle
from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import serving

MISSING_CATEGORY = "__missing__"
POSITIVE_CLASS_THRESHOLD = 0.5
REMOTE_URL_SCHEMES = ("http", "https")


def read_json(path: str | pathlib.Path) -> dict[str, Any]:
    """Reads a JSON object from `path`.

    Args:
        path: The JSON file.

    Returns:
        The parsed top-level object.

    Raises:
        ArtifactError: If the file cannot be read or decoded, or does not
            hold a JSON object.
    """
    try:
        record = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise errors.ArtifactError(f"cannot read {path}") from error
    if not isinstance(record, dict):
        raise errors.ArtifactError(f"{path} is not a JSON object")
    return record


@dataclasses.dataclass(frozen=True, slots=True)
class Tolerance:
    """How closely one environment must match the reference.

    Attributes:
        probability_abs: Largest allowed absolute difference in the class-1
            probability, in [0, 1).
        label: Label comparison mode; only "exact" is supported.
    """

    probability_abs: float
    label: str = "exact"

    def __post_init__(self) -> None:
        """Validates the tolerance.

        Raises:
            ConfigError: If `label` is not "exact" or `probability_abs` is
                outside [0, 1).
        """
        if self.label != "exact":
            raise errors.ConfigError("labels are compared exactly")
        if not (0 <= self.probability_abs < 1):
            raise errors.ConfigError("probability_abs must be in [0, 1)")


@dataclasses.dataclass(frozen=True, slots=True)
class GoldenSet:
    """Reference model, records and declared tolerances.

    Attributes:
        spec: The hand-written model specification.
        records: The input records.
        same_process: Tolerance for a comparison within one process.
        cross_environment: Tolerance for a comparison across environments.
    """

    spec: dict[str, Any]
    records: list[dict[str, Any]]
    same_process: Tolerance
    cross_environment: Tolerance


@dataclasses.dataclass(frozen=True, slots=True)
class Parity:
    """The outcome of one comparison.

    Attributes:
        passed: Whether labels and probabilities were all within tolerance.
        records: Number of expected records.
        max_probability_error: Largest absolute class-1 probability
            difference; infinite when the output count was wrong.
        tolerance: The probability tolerance applied.
        mismatches: Description of each difference found.
    """

    passed: bool
    records: int
    max_probability_error: float
    tolerance: float
    mismatches: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Converts the report to a JSON-compatible record.

        Returns:
            A dictionary with one entry per attribute; the mismatches are a
            list.
        """
        return {
            "passed": self.passed,
            "records": self.records,
            "max_probability_error": self.max_probability_error,
            "tolerance": self.tolerance,
            "mismatches": list(self.mismatches),
        }


def load_golden(directory: str | pathlib.Path) -> GoldenSet:
    """Loads the golden files from `directory`.

    Args:
        directory: Folder with `tolerances.json`, `model_spec.json` and
            `records.json`.

    Returns:
        The golden set. A file that cannot be read or is not a JSON object
        makes `read_json` raise `ArtifactError`; a missing key in a file
        raises `KeyError`.
    """
    root = pathlib.Path(directory)
    tolerances = read_json(root / "tolerances.json")
    return GoldenSet(
        spec=read_json(root / "model_spec.json"),
        records=read_json(root / "records.json")["records"],
        same_process=Tolerance(**tolerances["same_process"]),
        cross_environment=Tolerance(**tolerances["cross_environment"]),
    )


def to_float(frame: Any) -> Any:
    """Casts a frame to float; a named function so the pipeline can pickle.

    Args:
        frame: Array-like data.

    Returns:
        A data frame of floats, NaN where a value is not numeric.
    """
    return pandas.DataFrame(frame).apply(pandas.to_numeric, errors="coerce")


def to_object(frame: Any) -> Any:
    """Casts a frame to object; a named function so the pipeline can pickle.

    Args:
        frame: Array-like data.

    Returns:
        A data frame with object dtype.
    """
    return pandas.DataFrame(frame).astype(object)


def reference_bundle(spec: dict[str, Any]) -> bundle.Bundle:
    """Builds the hand-written model as a bundle.

    The pipeline has the structure of a trained one (median imputation,
    scaling, one-hot encoding, logistic regression). It is fitted on two rows
    only to create its objects; every learned number is then overwritten with
    the value from `spec`.

    Args:
        spec: The model specification: numeric columns, categorical
            columns (only the first is used), its two categories, medians,
            means, scales, coefficients, intercept, classes, target and model
            name and version.

    Returns:
        The bundle holding the hand-written pipeline.

    Raises:
        ArtifactError: If `spec` lists more than two categories. The
            encoder is fitted on the first two only, so its categories then
            differ from the sorted list in `spec`.
    """
    numeric = list(spec["numeric"])
    categorical = list(spec["categorical"])
    categories = list(spec["categories"])
    frame = pandas.DataFrame(
        {
            **{name: [0.0, 1.0] for name in numeric},
            categorical[0]: categories[:2],
        }
    )
    prep = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("cast", FunctionTransformer(to_float)),
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("cast", FunctionTransformer(to_object)),
                        (
                            "impute",
                            SimpleImputer(
                                strategy="constant",
                                fill_value=MISSING_CATEGORY,
                            ),
                        ),
                        (
                            "encode",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )
    pipeline = Pipeline([("prep", prep), ("model", LogisticRegression())])
    pipeline.fit(frame, [0, 1])
    numeric_steps = prep.named_transformers_["numeric"]
    numeric_steps["impute"].statistics_ = numpy.array(
        [spec["median"][name] for name in numeric]
    )
    scaler = numeric_steps["scale"]
    scaler.mean_ = numpy.array([spec["mean"][name] for name in numeric])
    scaler.scale_ = numpy.array([spec["scale"][name] for name in numeric])
    scaler.var_ = scaler.scale_**2
    encoder = prep.named_transformers_["categorical"]["encode"]
    if [str(item) for item in encoder.categories_[0]] != sorted(categories):
        raise errors.ArtifactError("reference categories must be sorted")
    weights = [spec["coefficients"][name] for name in numeric]
    weights += [
        spec["coefficients"][f"{categorical[0]}={item}"]
        for item in sorted(categories)
    ]
    model = pipeline.named_steps["model"]
    model.coef_ = numpy.array([weights])
    model.intercept_ = numpy.array([spec["intercept"]])
    model.classes_ = numpy.array(spec["classes"])
    plan = data.FeaturePlan(tuple(numeric), tuple(categorical))
    return bundle.Bundle.create(
        pipeline,
        plan,
        frame,
        spec["target"],
        "binary_classification",
        "roc_auc",
        spec["model_name"],
        model_version=spec["model_version"],
    )


def expected_outputs(
    spec: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Computes labels and probabilities in closed form with numpy only.

    Args:
        spec: The model specification.
        records: Input records; a missing numeric value is replaced by the
            median, and an unknown category adds nothing.

    Returns:
        The `classes`, the predicted `labels` (1 when the probability
        exceeds 0.5) and the `probability_of_class_1` per record.
    """
    numeric = list(spec["numeric"])
    category_name = spec["categorical"][0]
    categories = sorted(spec["categories"])
    probabilities = []
    for record in records:
        logit = float(spec["intercept"])
        for name in numeric:
            raw = record.get(name)
            value = spec["median"][name] if raw is None else float(raw)
            z = (value - spec["mean"][name]) / spec["scale"][name]
            logit += spec["coefficients"][name] * z
        chosen = record.get(category_name)
        if chosen in categories:
            logit += spec["coefficients"][f"{category_name}={chosen}"]
        probabilities.append(1.0 / (1.0 + float(numpy.exp(-logit))))
    return {
        "classes": list(spec["classes"]),
        "labels": [int(p > POSITIVE_CLASS_THRESHOLD) for p in probabilities],
        "probability_of_class_1": probabilities,
    }


def compare(
    expected: dict[str, Any],
    predictions: list[Any],
    probabilities: list[list[float]],
    tolerance: Tolerance,
) -> Parity:
    """Compares a serving path's outputs with the expected ones.

    Args:
        expected: Output of `expected_outputs`.
        predictions: Labels returned by the serving path.
        probabilities: Class probabilities per record, with class 1 in
            column 1.
        tolerance: The tolerance to apply.

    Returns:
        The report; failed with an infinite error if the output count
        differs from the expected one.
    """
    mismatches = []
    count = len(expected["labels"])
    if len(predictions) != count or len(probabilities) != count:
        return Parity(
            passed=False,
            records=count,
            max_probability_error=float("inf"),
            tolerance=tolerance.probability_abs,
            mismatches=(f"expected {count} outputs, got {len(predictions)}",),
        )
    worst = 0.0
    for index in range(count):
        if int(predictions[index]) != expected["labels"][index]:
            mismatches.append(f"record {index}: label differs")
        error = abs(
            float(probabilities[index][1])
            - expected["probability_of_class_1"][index]
        )
        worst = max(worst, error)
        if error > tolerance.probability_abs:
            mismatches.append(f"record {index}: probability off by {error:.3g}")
    return Parity(
        passed=not mismatches,
        records=count,
        max_probability_error=worst,
        tolerance=tolerance.probability_abs,
        mismatches=tuple(mismatches),
    )


def check_remote(
    url: str,
    directory: str | pathlib.Path,
    level: str = "cross_environment",
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Sends the golden records to a running service and compares.

    Posts the records to `<url>/predict` and compares the answer with
    `expected.json` from `directory`.

    Args:
        url: Base URL of the service.
        directory: Folder with the golden files and `expected.json`.
        level: `same_process` selects that tolerance; any other value
            selects the `cross_environment` tolerance.
        timeout: Seconds to wait for the answer.

    Returns:
        The parity report as a dictionary, plus the model version, artifact
        digest and level. The report is marked failed when the response
        lacks any key in `serving_contract.RESPONSE_KEYS`. On an HTTP error
        status, or a response that is not an object with `predictions` and
        `probabilities`, only `passed` (False) and `mismatches` are returned
        (with `level` for the latter). An unreadable golden file raises
        `ArtifactError` from `read_json`; other network errors propagate.

    Raises:
        ValueError: If `url` does not use http or https.
    """
    if urllib.parse.urlsplit(url).scheme not in REMOTE_URL_SCHEMES:
        raise ValueError(f"url must use http or https, got {url!r}")
    root = pathlib.Path(directory)
    golden = load_golden(root)
    expected = read_json(root / "expected.json")
    request = urllib.request.Request(  # noqa: S310 -- scheme checked above
        url.rstrip("/") + "/predict",
        json.dumps({"records": golden.records}).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(  # noqa: S310 -- scheme checked above  # nosec B310
            request, timeout=timeout
        ) as reply:
            body = json.loads(reply.read())
    except urllib.error.HTTPError as error:
        return {"passed": False, "mismatches": [f"HTTP {error.code}"]}
    tolerance = (
        golden.same_process
        if level == "same_process"
        else golden.cross_environment
    )
    if (
        not isinstance(body, dict)
        or "predictions" not in body
        or "probabilities" not in body
    ):
        return {
            "passed": False,
            "mismatches": ["response lacks predictions or probabilities"],
            "level": level,
        }
    report = compare(
        expected, body["predictions"], body["probabilities"], tolerance
    ).to_dict()
    missing = [key for key in serving.RESPONSE_KEYS if key not in body]
    if missing:
        report["passed"] = False
        report["mismatches"].append(f"response lacks {missing}")
    report.update(
        model_version=body.get("model_version"),
        artifact_digest=body.get("artifact_digest"),
        level=level,
    )
    return report
