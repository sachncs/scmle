"""ONNX export of a bundle's fitted pipeline, with a divergence gate.

The exported graph reproduces the pipeline's preprocessing and model for
inputs prepared the same way the framework prepares them: numeric columns as
float32 with unparseable and infinite values missing, categorical columns as
strings. Models or preprocessing steps the converter cannot express are
refused with a message naming them, never exported approximately. `verify`
is the gate: it blocks a graph whose predictions on held-out rows differ from
the native model beyond the float32 rounding bound.
"""

import copy
import hashlib
import importlib
import json
import math
import pathlib
from typing import Any

import numpy
import pandas
from sklearn import impute
from sklearn import pipeline
from sklearn import preprocessing

from scmle import durability
from scmle.contracts import errors
from scmle.data.validation import columns
from scmle.models.boosting import early_stopping
from scmle.models.sklearn import trees
from scmle.preparation import resampling
from scmle.preparation import weighting
from scmle.preparation import wrappers

OPSET = 17
ML_OPSET = 3
UNIT = float(numpy.finfo(numpy.float32).eps) / 2
"""Unit roundoff of float32, the arithmetic the graph runs in."""
OPERATIONS = 1024
"""Roundings along one prediction: a dot product or tree-ensemble sum over
at most this many terms, then one link function."""
DEFAULT_TOLERANCE = OPERATIONS * UNIT / (1 - OPERATIONS * UNIT)
"""gamma_n = n u / (1 - n u), the bound on the relative error of a float32
sum of n = OPERATIONS terms (Higham, Accuracy and Stability of Numerical
Algorithms, Lemma 3.1); about 6.1e-5, tighter than the former fixed 1e-4."""
RATE = 0.01
"""Share of diverging rows the gate rules out."""
CONFIDENCE = 0.95
"""Confidence with which the gate rules out `RATE`."""
ROWS = math.ceil(math.log(1 - CONFIDENCE) / math.log(1 - RATE))
"""Held-out rows needed: with none of `ROWS` rows diverging, a divergence
rate of `RATE` or more has probability (1 - RATE)**ROWS <= 1 - CONFIDENCE
(exact one-sided binomial bound); 299 rows for 1% at 95%."""
CASTS = (columns.floating, columns.objectify)
"""Functions `feed` applies before the graph runs; any other function
transformer is refused, never replaced by an identity."""
INSTALL_HINT = "install the 'onnx' extra (skl2onnx, onnxruntime)"
WRAPPERS = (
    weighting.Reweighter,
    resampling.ResamplingClassifier,
    resampling.Density,
)


def load() -> tuple[Any, Any]:
    """Imports the converter and runtime, or explains how to install them.

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
            f"ONNX export: {INSTALL_HINT}; "
            "scmle.deploy.packaging.scaffold(bundle, directory) packages "
            "any model as a ready-to-run service without it"
        ) from error
    return convert, runtime


MODEL_WRAPPERS = (
    early_stopping.Holdout,
    wrappers.Encoded,
    wrappers.Flattener,
    wrappers.Forwarder,
    wrappers.Curried,
    wrappers.MonotoneEstimator,
)
CODES_KEY = "label_codes"
MANIFEST = ".manifest.json"
"""Suffix of the export manifest written beside the graph."""


def enrol() -> None:
    """Registers LightGBM and XGBoost converters when onnxmltools exists.

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
        # handled: optional onnxmltools stack absent; nothing registered,
        # documented
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
    """Finds the fitted scikit-learn pipeline inside the framework's wrappers.

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
            "scmle.deploy.packaging.scaffold(bundle, directory) packages "
            "any model as a ready-to-run service instead"
        )
    return current


def imputes(step: Any) -> bool:
    """Checks whether `step` fills missing text with a constant token.

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


def passthrough() -> Any:
    """Returns an identity that stands in for steps done before the graph.

    Returns:
        A fitted identity function transformer.
    """
    return preprocessing.FunctionTransformer().fit(numpy.zeros((1, 1)))


def casts(step: Any) -> bool:
    """Checks whether `feed` already does what `step` does.

    Args:
        step: A pipeline step.

    Returns:
        True for a cast in `CASTS`, an identity function transformer, or a
        constant text imputer.

    Raises:
        ArtifactError: For a function transformer whose function the graph
            cannot express; replacing it by an identity would diverge.
    """
    if not isinstance(step, preprocessing.FunctionTransformer):
        return imputes(step)
    if step.func is None or step.func in CASTS:
        return True
    raise errors.ArtifactError(
        "ONNX export cannot express the function transformer "
        f"'{getattr(step.func, '__name__', step.func)}'; "
        "scmle.deploy.packaging.scaffold(bundle, directory) packages any "
        "model as a ready-to-run service instead"
    )


def uncast(model: Any) -> Any:
    """Copies a model, replacing its type casts and text imputers by identities.

    Casting columns to float (infinite values to missing) or to objects, and
    filling missing text with the missing token, happen before the graph runs
    (see `feed`), because the converter cannot express them. Only pipelines
    and column transformers (found through `transformers_`) are searched,
    nested to any depth.

    Args:
        model: A fitted pipeline.

    Returns:
        A deep copy in which every cast and constant text imputer is an
        identity; `model` is not modified.

    Raises:
        ArtifactError: For any other function transformer (see `casts`).
    """
    clone = copy.deepcopy(model)
    stack: list[Any] = [clone]
    while stack:
        node = stack.pop()
        if isinstance(node, pipeline.Pipeline):
            for index, (name, step) in enumerate(node.steps):
                if casts(step):
                    node.steps[index] = (name, passthrough())
                else:
                    stack.append(step)
        elif hasattr(node, "transformers_"):
            for index, (name, step, cols) in enumerate(node.transformers_):
                if casts(step):
                    node.transformers_[index] = (name, passthrough(), cols)
                elif not isinstance(step, str):
                    stack.append(step)
    return clone


def export(bundle: Any, path: str | pathlib.Path) -> pathlib.Path:
    """Writes the bundle's pipeline as an ONNX file.

    Registers LightGBM and XGBoost converters in the converter's global
    registry when `onnxmltools` is available. Classifiers are exported
    without the ZipMap output. A model trained on integer class codes gets a
    `label_codes` entry in the file's metadata so that scoring maps the codes
    back to labels. The file is written atomically.

    Args:
        bundle: A fitted `Bundle`.
        path: Destination file.

    Returns:
        The path written.

    Raises:
        ArtifactError: For models that cannot be exported faithfully, when
            an inference-time calibrator is attached (the graph would not
            reproduce `predict_proba`), or when the extra is not installed.
    """
    converter = load()[0]
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
    enrol()
    fitted = uncast(unwrap(bundle.pipeline))
    model_step = fitted.steps[-1][1]
    encoded = False
    while isinstance(model_step, MODEL_WRAPPERS):
        encoded = encoded or hasattr(model_step, "encoder_")
        model_step = model_step.estimator_
    # The converter looks estimators up by exact type.
    fitted.steps[-1] = (fitted.steps[-1][0], trees.stock(model_step))
    options = (
        {id(fitted.steps[-1][1]): {"zipmap": False}}
        if bundle.classes is not None
        else None
    )
    try:
        model = converter.convert_sklearn(
            fitted,
            initial_types=[
                *(
                    (c, types.FloatTensorType([None, 1]))
                    for c in bundle.schema.numeric
                ),
                *(
                    (c, types.StringTensorType([None, 1]))
                    for c in bundle.schema.categorical
                ),
            ],
            options=options,
            target_opset={"": OPSET, "ai.onnx.ml": ML_OPSET},
        )
    except Exception as error:  # The converter raises many types.
        raise errors.ArtifactError(
            f"ONNX export is not supported for model "
            f"'{bundle.info.model}': {error}. "
            "scmle.deploy.packaging.scaffold(bundle, directory) packages "
            "any model as a ready-to-run service instead"
        ) from error
    if encoded:
        entry = model.metadata_props.add()
        entry.key, entry.value = CODES_KEY, "1"
    destination = pathlib.Path(path)
    durability.write(destination, model.SerializeToString())
    return destination


def decode(bundle: Any, session: Any, labels: numpy.ndarray) -> numpy.ndarray:
    """Maps integer class codes back to labels when the graph was trained so.

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
        decoded: numpy.ndarray = numpy.asarray(bundle.classes, dtype=object)[
            numpy.asarray(labels).astype(int)
        ]
        return decoded
    return numpy.asarray(labels)


def feed(bundle: Any, frame: pandas.DataFrame) -> dict[str, numpy.ndarray]:
    """Builds the exported graph's inputs from a validated frame.

    Args:
        bundle: The bundle the graph came from.
        frame: Incoming rows.

    Returns:
        One column array of shape (rows, 1) per input, keyed by column name:
        float32 for numeric columns (values that are not numbers and
        infinite values become NaN, as `columns.floating` does in the
        pipeline) and strings for categorical ones, with missing categories
        replaced by the missing token.
    """
    prepared = bundle.prepare(frame)
    schema = bundle.schema
    inputs = {
        c: columns.scrub(prepared[c]).to_numpy(numpy.float32).reshape(-1, 1)
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


def enough(frame: pandas.DataFrame | None) -> pandas.DataFrame:
    """Blocks a held-out set too small for the divergence gate.

    It runs before conversion too, so a bad held-out set is refused
    before the converter is imported or run.

    Args:
        frame: Held-out rows; None means none were given.

    Returns:
        The rows, at least `ROWS` of them.

    Raises:
        BlockedError: With no rows or fewer than `ROWS`.
    """
    rows = 0 if frame is None else len(frame)
    if frame is None or rows < ROWS:
        raise errors.BlockedError(
            f"the ONNX divergence gate needs at least {ROWS} rows held out "
            f"from training to rule out a {RATE:.0%} divergence rate at "
            f"{CONFIDENCE:.0%} confidence; got {rows}"
        )
    return frame


def verify(
    bundle: Any,
    path: str | pathlib.Path,
    frame: pandas.DataFrame | None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, float]:
    """Blocks a graph whose predictions diverge from the native model.

    The gate scores held-out rows with both the graph and the in-process
    pipeline. Classes may disagree only on rows whose native top-two
    probability margin is within twice `tolerance`, the most a difference of
    `tolerance` per class can move the decision.

    Args:
        bundle: The bundle the graph came from.
        path: The ONNX file.
        frame: Held-out rows to compare on, at least `ROWS` of them; None
            (no rows given) blocks.
        tolerance: Largest allowed difference; a larger value is the
            override.

    Returns:
        `rows`, `max_difference` (largest absolute probability difference,
        or for regression the largest absolute difference relative to the
        larger of the largest absolute value and 1) and `label_agreement`
        (share of identical predicted labels, 1.0 for regression).

    Raises:
        BlockedError: With no rows or fewer than `ROWS`, when the difference
            exceeds `tolerance` or is NaN, or when a label differs on a row
            whose margin exceeds twice `tolerance`.
    """
    held = enough(frame)
    rows = len(held)
    runtime = load()[1]
    session = runtime.InferenceSession(str(path))
    outputs = session.run(None, feed(bundle, held))
    prepared = bundle.prepare(held)
    agreement, disagreements = 1.0, 0
    if bundle.classes is not None:
        expected = numpy.asarray(bundle.pipeline.predict_proba(prepared))
        difference = float(
            numpy.abs(numpy.asarray(outputs[1]) - expected).max()
        )
        agrees = decode(bundle, session, outputs[0]).astype(str) == (
            numpy.asarray(bundle.pipeline.predict(prepared)).astype(str)
        )
        agreement = float(numpy.mean(agrees))
        ranked = numpy.sort(expected, axis=1)
        clear = ranked[:, -1] - ranked[:, -2] > 2 * tolerance
        disagreements = int(numpy.sum(clear & ~agrees))
    else:
        expected = numpy.asarray(bundle.pipeline.predict(prepared), dtype=float)
        scale = max(float(numpy.abs(expected).max()), 1.0)
        difference = float(
            numpy.abs(numpy.asarray(outputs[0]).ravel() - expected).max()
            / scale
        )
    if not difference <= tolerance:
        raise errors.BlockedError(
            f"the ONNX graph differs from the pipeline by {difference:.2e} "
            f"on {rows} held-out rows (tolerance {tolerance:.1e})"
        )
    if disagreements:
        raise errors.BlockedError(
            f"the ONNX graph's labels differ from the pipeline's on "
            f"{disagreements} of {rows} held-out rows whose margin exceeds "
            f"{2 * tolerance:.1e}"
        )
    return {
        "rows": float(rows),
        "max_difference": difference,
        "label_agreement": agreement,
    }


def record(
    bundle: Any,
    path: pathlib.Path,
    report: dict[str, float] | None,
    tolerance: float,
) -> pathlib.Path:
    """Writes the export manifest beside the graph.

    The manifest names the graph by digest and says whether the divergence
    gate passed on it or was skipped on request.

    Args:
        bundle: The bundle the graph came from.
        path: The ONNX file.
        report: What `verify` returned, or None when the check was skipped.
        tolerance: The tolerance the gate used.

    Returns:
        The manifest's path: the graph's path with `MANIFEST` appended.
    """
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    document = {
        "graph_sha256": f"sha256:{digest}",
        "model_name": bundle.info.model,
        "verified": report is not None,
        "tolerance": tolerance,
        "rows": None if report is None else int(report["rows"]),
        "max_difference": None if report is None else report["max_difference"],
        "label_agreement": (
            None if report is None else report["label_agreement"]
        ),
    }
    manifest = path.with_name(path.name + MANIFEST)
    content = json.dumps(document, indent=2, sort_keys=True) + "\n"
    durability.write(manifest, content.encode("utf-8"))
    return manifest
