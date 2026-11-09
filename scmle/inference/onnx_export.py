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

from scmle import durability
from scmle.contracts import errors
from scmle.data.validation import columns
from scmle.models import weighting
from scmle.models import wrappers
from scmle.models.boosting import early_stopping
from scmle.sampling import resampling

OPSET = 17
ML_OPSET = 3
DEFAULT_TOLERANCE = 1e-4
INSTALL_HINT = "install the 'onnx' extra (skl2onnx, onnxruntime)"
WRAPPERS = (
    weighting.WeightedEstimator,
    resampling.ResamplingClassifier,
    resampling.DensityWeightedRegressor,
)


def load_converter() -> tuple[Any, Any]:
    """Import the converter and runtime, or explain how to install them.

    Returns:
        The `skl2onnx` module and the `onnxruntime` module.

    Raises:
        ArtifactError: If either module cannot be imported.
    """
    try:
        convert = importlib.import_module("skl2onnx")
        runtime = importlib.import_module("onnxruntime")
    except ImportError as error:
        raise errors.ArtifactError(
            f"ONNX export: {INSTALL_HINT}; bundle.export_service(directory) "
            "packages any model as a ready-to-run service without it"
        ) from error
    return convert, runtime


MODEL_WRAPPERS = (
    early_stopping.HoldoutEarlyStopping,
    wrappers.LabelEncodedClassifier,
    wrappers.FlatPredictionClassifier,
    wrappers.FitParamsClassifier,
    wrappers.FitParamsRegressor,
    wrappers.MonotoneEstimator,
)
CODES_KEY = "label_codes"


def inner_model(step: Any) -> tuple[Any, bool]:
    """Find the fitted model inside the framework's model wrappers.

    Args:
        step: A fitted estimator, possibly wrapped.

    Returns:
        The innermost estimator and whether it was trained on integer codes
        for the class labels (so its outputs must be mapped back).
    """
    encoded = False
    while isinstance(step, MODEL_WRAPPERS):
        encoded = encoded or hasattr(step, "encoder_")
        step = step.estimator_
    return step, encoded


def register_booster_converters() -> None:
    """Register LightGBM and XGBoost converters when onnxmltools exists.

    Nothing is registered when any of the needed modules cannot be imported.
    """
    try:
        registry = importlib.import_module("skl2onnx")
        shapes = importlib.import_module(
            "skl2onnx.shape_calculators.linear_classifier"
        )
        regression_shapes = importlib.import_module(
            "skl2onnx.shape_calculators.linear_regressor"
        )
        lightgbm = importlib.import_module("lightgbm")
        xgboost = importlib.import_module("xgboost")
        light = importlib.import_module(
            "onnxmltools.convert.lightgbm.operator_converters.LightGbm"
        )
        xgb = importlib.import_module(
            "onnxmltools.convert.xgboost.operator_converters.XGBoost"
        )
    except ImportError:
        return
    options = {"nocl": [True, False], "zipmap": [True, False, "columns"]}
    registry.update_registered_converter(
        lightgbm.LGBMClassifier,
        "LightGbmLGBMClassifier",
        shapes.calculate_linear_classifier_output_shapes,
        light.convert_lightgbm,
        options=options,
    )
    registry.update_registered_converter(
        lightgbm.LGBMRegressor,
        "LightGbmLGBMRegressor",
        regression_shapes.calculate_linear_regressor_output_shapes,
        light.convert_lightgbm,
    )
    registry.update_registered_converter(
        xgboost.XGBClassifier,
        "XGBoostXGBClassifier",
        shapes.calculate_linear_classifier_output_shapes,
        xgb.convert_xgboost,
        options=options,
    )
    registry.update_registered_converter(
        xgboost.XGBRegressor,
        "XGBoostXGBRegressor",
        regression_shapes.calculate_linear_regressor_output_shapes,
        xgb.convert_xgboost,
    )


def unwrap(fitted: Any) -> pipeline.Pipeline:
    """Find the fitted scikit-learn pipeline inside the framework's wrappers.

    Args:
        fitted: A fitted estimator, possibly wrapped.

    Returns:
        The pipeline underneath the wrappers.

    Raises:
        ArtifactError: If the model is not a plain pipeline underneath.
    """
    current = fitted
    while isinstance(current, WRAPPERS):
        current = current.estimator_
    if not isinstance(current, pipeline.Pipeline):
        raise errors.ArtifactError(
            f"ONNX export is not supported for {type(current).__name__}; "
            "bundle.export_service(directory) packages any model as a "
            "ready-to-run service instead"
        )
    return current


def is_token_imputer(step: Any) -> bool:
    """Check whether `step` fills missing text with a constant token.

    Args:
        step: A pipeline step.

    Returns:
        True for a constant-strategy simple imputer with a string fill value.
    """
    return (
        isinstance(step, impute.SimpleImputer)
        and step.strategy == "constant"
        and isinstance(step.fill_value, str)
    )


def passthrough(step: Any) -> Any:
    """Make an identity that stands in for steps done before the graph runs.

    Args:
        step: The step being replaced; unused.

    Returns:
        A fitted identity function transformer.
    """
    return preprocessing.FunctionTransformer().fit(np.zeros((1, 1)))


def without_casts(model: Any) -> Any:
    """Copy a model, replacing its type casts and text imputers by identities.

    Casting columns to float or string, and filling missing text with the
    missing token, happen before the graph runs (see `feed`), because the
    converter cannot express them.

    Args:
        model: A fitted pipeline.

    Returns:
        A deep copy in which every function transformer and constant text
        imputer is an identity; `model` is not modified.
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
    """Declare the ONNX inputs: floats for numeric, strings for categorical.

    Args:
        bundle: A bundle whose schema lists the numeric and categorical
            columns.
        types: The `skl2onnx.common.data_types` module.

    Returns:
        One (column name, tensor type) pair per input, numeric columns first.
    """
    schema = bundle.schema
    return [
        *((c, types.FloatTensorType([None, 1])) for c in schema.numeric),
        *((c, types.StringTensorType([None, 1])) for c in schema.categorical),
    ]


def export(bundle: Any, path: str | pathlib.Path) -> pathlib.Path:
    """Write the bundle's pipeline as an ONNX file.

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
    register_booster_converters()
    fitted = without_casts(unwrap(bundle.pipeline))
    model_step, encoded = inner_model(fitted.steps[-1][1])
    fitted.steps[-1] = (fitted.steps[-1][0], model_step)
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
            target_opset={"": OPSET, "ai.onnx.ml": ML_OPSET},
        )
    except Exception as error:  # The converter raises many types.
        raise errors.ArtifactError(
            f"ONNX export is not supported for model "
            f"'{bundle.info.model_name}': {error}. "
            "bundle.export_service(directory) packages any model as a "
            "ready-to-run service instead"
        ) from error
    if encoded:
        entry = model.metadata_props.add()
        entry.key, entry.value = CODES_KEY, "1"
    destination = pathlib.Path(path)
    durability.write_atomically(destination, model.SerializeToString())
    return destination


def decode(bundle: Any, session: Any, labels: np.ndarray) -> np.ndarray:
    """Map integer class codes back to labels when the graph was trained so.

    Args:
        bundle: The bundle the graph came from; supplies the class labels.
        session: The ONNX runtime session of the graph.
        labels: The graph's label output.

    Returns:
        The class labels when the graph carries the code marker, else
        `labels` as an array.
    """
    meta = session.get_modelmeta().custom_metadata_map
    if meta.get(CODES_KEY) == "1":
        return np.asarray(bundle.classes, dtype=object)[
            np.asarray(labels).astype(int)
        ]
    return np.asarray(labels)


def feed(bundle: Any, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build the exported graph's inputs from a validated frame.

    Args:
        bundle: The bundle the graph came from.
        frame: Incoming rows.

    Returns:
        One column array of shape (rows, 1) per input: float32 for numeric
        columns and strings for categorical ones, with missing categories
        replaced by the missing token.
    """
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
    """Compare the exported graph with the in-process pipeline.

    Args:
        bundle: The bundle the graph came from.
        path: The ONNX file.
        frame: Rows to compare on.
        tolerance: Largest allowed difference.

    Returns:
        `max_difference` (largest absolute probability difference, or for
        regression the largest absolute difference relative to the larger of
        the largest absolute value and 1) and `label_agreement` (share of
        identical predicted labels, 1.0 for regression).

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
                decode(bundle, session, outputs[0]).astype(str)
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
        """Open the graph.

        Args:
            bundle: The bundle the graph was exported from (for its schema).
            path: ONNX file.
        """
        runtime = load_converter()[1]
        self.bundle = bundle
        self.session = runtime.InferenceSession(str(path))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict labels or values.

        Args:
            frame: Incoming rows.

        Returns:
            A one-dimensional array of labels or values.
        """
        outputs = self.session.run(None, feed(self.bundle, frame))
        return decode(self.bundle, self.session, outputs[0]).ravel()

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.

        Args:
            frame: Incoming rows.

        Returns:
            An array of shape (rows, classes).
        """
        return np.asarray(self.session.run(None, feed(self.bundle, frame))[1])
