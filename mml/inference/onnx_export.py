"""ONNX export of a bundle's fitted pipeline, with a parity check.

The exported graph reproduces the pipeline's preprocessing and model for
inputs prepared the same way the framework prepares them: numeric columns as
float32, categorical columns as strings. Models or preprocessing steps the
converter cannot express are refused with a message naming them, never
exported approximately.
"""

import copy
import importlib
import pathlib
from typing import Any

import numpy as np
import pandas as pd
from sklearn import impute
from sklearn import pipeline
from sklearn import preprocessing

from mml.contracts import errors
from mml.data.validation import columns
from mml.models import weighting
from mml.sampling import resampling

OPSET = 17
DEFAULT_TOLERANCE = 1e-4
INSTALL_HINT = "install the 'onnx' extra (skl2onnx, onnxruntime)"
WRAPPERS = (
    weighting.WeightedEstimator,
    resampling.ResamplingClassifier,
    resampling.DensityWeightedRegressor,
)


def load_converter() -> tuple[Any, Any]:
    """Imports the converter and runtime, or explains how to install them."""
    try:
        convert = importlib.import_module("skl2onnx")
        runtime = importlib.import_module("onnxruntime")
    except ImportError as error:
        raise errors.ArtifactError(f"ONNX export: {INSTALL_HINT}") from error
    return convert, runtime


def unwrap(fitted: Any) -> pipeline.Pipeline:
    """The fitted scikit-learn pipeline inside the framework's wrappers.

    Raises:
        ArtifactError: If the model is not a plain pipeline underneath.
    """
    current = fitted
    while isinstance(current, WRAPPERS):
        current = current.estimator_
    if not isinstance(current, pipeline.Pipeline):
        raise errors.ArtifactError(
            f"ONNX export is not supported for {type(current).__name__}"
        )
    return current


def is_token_imputer(step: Any) -> bool:
    """Whether `step` fills missing text with a constant token."""
    return (
        isinstance(step, impute.SimpleImputer)
        and step.strategy == "constant"
        and isinstance(step.fill_value, str)
    )


def passthrough(step: Any) -> Any:
    """An identity that stands in for steps done before the graph runs."""
    return preprocessing.FunctionTransformer().fit(np.zeros((1, 1)))


def without_casts(model: Any) -> Any:
    """Copy of `model` with its type casts and text imputers made identities.

    Casting columns to float or string, and filling missing text with the
    missing token, happen before the graph runs (see `feed`), because the
    converter cannot express them.
    """
    clone = copy.deepcopy(model)
    stack: list[Any] = [clone]
    while stack:
        node = stack.pop()
        if isinstance(node, pipeline.Pipeline):
            for index, (name, step) in enumerate(node.steps):
                if isinstance(
                    step, preprocessing.FunctionTransformer
                ) or is_token_imputer(step):
                    node.steps[index] = (name, passthrough(step))
                else:
                    stack.append(step)
        elif hasattr(node, "transformers_"):
            for index, (name, step, cols) in enumerate(node.transformers_):
                if isinstance(
                    step, preprocessing.FunctionTransformer
                ) or is_token_imputer(step):
                    node.transformers_[index] = (name, passthrough(step), cols)
                elif not isinstance(step, str):
                    stack.append(step)
    return clone


def input_types(bundle: Any, types: Any) -> list[tuple[str, Any]]:
    """ONNX input declarations: floats for numeric, strings for categorical."""
    schema = bundle.schema
    return [
        *((c, types.FloatTensorType([None, 1])) for c in schema.numeric),
        *((c, types.StringTensorType([None, 1])) for c in schema.categorical),
    ]


def export(bundle: Any, path: str | pathlib.Path) -> pathlib.Path:
    """Writes the bundle's pipeline as an ONNX file.

    Args:
        bundle: A fitted `ModelBundle`.
        path: Destination file.

    Returns:
        The path written.

    Raises:
        ArtifactError: For models that cannot be exported faithfully, when
            an inference-time calibrator is attached (the graph would not
            reproduce `predict_proba`), or when the extra is not installed.
    """
    converter = load_converter()[0]
    types = importlib.import_module("skl2onnx.common.data_types")
    task = bundle.schema.task
    if task not in {
        "binary_classification",
        "multiclass_classification",
        "regression",
    }:
        raise errors.ArtifactError(
            f"ONNX export does not support {task} models"
        )
    if bundle.calibrator is not None:
        raise errors.ArtifactError(
            "ONNX export would not reproduce the attached probability "
            "calibrator; export before calibrating"
        )
    fitted = without_casts(unwrap(bundle.pipeline))
    options = (
        {id(fitted.steps[-1][1]): {"zipmap": False}}
        if bundle.classes is not None
        else None
    )
    try:
        model = converter.convert_sklearn(
            fitted,
            initial_types=input_types(bundle, types),
            options=options,
            target_opset=OPSET,
        )
    except Exception as error:  # The converter raises many types.
        raise errors.ArtifactError(
            f"ONNX export is not supported for model "
            f"'{bundle.info.model_name}': {error}"
        ) from error
    destination = pathlib.Path(path)
    destination.write_bytes(model.SerializeToString())
    return destination


def feed(bundle: Any, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Inputs for the exported graph from a validated frame."""
    prepared = bundle.prepare_frame(frame)
    schema = bundle.schema
    inputs = {
        c: pd.to_numeric(prepared[c], errors="coerce")
        .to_numpy(np.float32)
        .reshape(-1, 1)
        for c in schema.numeric
    }
    inputs |= {
        c: prepared[c]
        .astype(object)
        .where(prepared[c].notna(), columns.MISSING_TOKEN)
        .astype(str)
        .to_numpy()
        .reshape(-1, 1)
        for c in schema.categorical
    }
    return inputs


def verify(
    bundle: Any,
    path: str | pathlib.Path,
    frame: pd.DataFrame,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, float]:
    """Compares the exported graph with the in-process pipeline.

    Returns:
        `max_difference` (probabilities or values) and `label_agreement`
        (share of identical predicted labels, 1.0 for regression).

    Raises:
        ArtifactError: If the difference exceeds `tolerance`.
    """
    runtime = load_converter()[1]
    session = runtime.InferenceSession(str(path))
    outputs = session.run(None, feed(bundle, frame))
    prepared = bundle.prepare_frame(frame)
    if bundle.classes is not None:
        expected = np.asarray(bundle.pipeline.predict_proba(prepared))
        difference = float(np.abs(np.asarray(outputs[1]) - expected).max())
        agreement = float(
            np.mean(
                np.asarray(outputs[0]).astype(str)
                == np.asarray(bundle.pipeline.predict(prepared)).astype(str)
            )
        )
    else:
        expected = np.asarray(bundle.pipeline.predict(prepared), dtype=float)
        scale = max(float(np.abs(expected).max()), 1.0)
        difference = float(
            np.abs(np.asarray(outputs[0]).ravel() - expected).max() / scale
        )
        agreement = 1.0
    if difference > tolerance:
        raise errors.ArtifactError(
            f"the ONNX graph differs from the pipeline by {difference:.2e} "
            f"(tolerance {tolerance:.0e})"
        )
    return {"max_difference": difference, "label_agreement": agreement}


class OnnxScorer:
    """Scores frames with an exported graph, without the framework pipeline."""

    def __init__(self, bundle: Any, path: str | pathlib.Path) -> None:
        """Opens the graph.

        Args:
            bundle: The bundle the graph was exported from (for its schema).
            path: ONNX file.
        """
        runtime = load_converter()[1]
        self.bundle = bundle
        self.session = runtime.InferenceSession(str(path))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Labels or values."""
        outputs = self.session.run(None, feed(self.bundle, frame))
        return np.asarray(outputs[0]).ravel()

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Class probabilities."""
        return np.asarray(self.session.run(None, feed(self.bundle, frame))[1])
