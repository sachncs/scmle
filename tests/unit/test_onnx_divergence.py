"""The ONNX divergence gate blocks an export that differs from the model."""

import importlib.util
import math

import numpy
import pytest
from sklearn import preprocessing
import test_onnx_export

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.inference import onnx_export

needs_onnx = pytest.mark.skipif(
    importlib.util.find_spec("skl2onnx") is None
    or importlib.util.find_spec("onnxruntime") is None,
    reason="the onnx extra is not installed",
)


def held_out(rows=onnx_export.ROWS):
    """Rows drawn apart from the training rows (another seed)."""
    return test_onnx_export.frame(rows=rows, seed=7)


def test_the_tolerance_is_the_float32_rounding_bound_and_not_weaker():
    unit = numpy.finfo(numpy.float32).eps / 2
    gamma = onnx_export.OPERATIONS * unit
    assert pytest.approx(gamma / (1 - gamma)) == onnx_export.DEFAULT_TOLERANCE
    # The previous fixed tolerance was 1e-4; the derived one is tighter.
    assert onnx_export.DEFAULT_TOLERANCE <= 1e-4


def test_the_row_minimum_rules_out_the_divergence_rate_at_the_confidence():
    miss = 1 - onnx_export.RATE
    assert miss**onnx_export.ROWS <= 1 - onnx_export.CONFIDENCE
    assert miss ** (onnx_export.ROWS - 1) > 1 - onnx_export.CONFIDENCE
    assert (
        math.ceil(math.log(1 - onnx_export.CONFIDENCE) / math.log(miss))
        == onnx_export.ROWS
    )


@needs_onnx
def test_held_out_rows_with_infinities_reach_the_graph_as_missing(tmp_path):
    bundle = test_onnx_export.make_bundle(
        "logistic_regression", tasks.Task.BINARY
    )[0]
    path = bundle.export_onnx(tmp_path / "m.onnx", unverified=True)
    rows = held_out()
    rows.loc[::20, "num"] = numpy.inf
    rows.loc[1::20, "num"] = -numpy.inf

    report = onnx_export.verify(bundle, path, rows)

    assert report["max_difference"] <= onnx_export.DEFAULT_TOLERANCE
    assert report["label_agreement"] == 1.0


@needs_onnx
@pytest.mark.parametrize(
    "task", [tasks.Task.BINARY, tasks.Task.REGRESSION], ids=lambda t: t.value
)
def test_an_export_that_drifted_from_the_model_is_blocked(task, tmp_path):
    name = "logistic_regression" if task.is_classification else "ridge"
    bundle = test_onnx_export.make_bundle(name, task)[0]
    path = bundle.export_onnx(tmp_path / "m.onnx", unverified=True)
    model = bundle.pipeline.steps[-1][1]
    while not hasattr(model, "coef_"):
        model = model.estimator_
    model.coef_ = model.coef_ * 1.5

    with pytest.raises(errors.BlockedError, match="differs"):
        onnx_export.verify(bundle, path, held_out())


@needs_onnx
def test_a_larger_tolerance_is_the_override(tmp_path):
    bundle = test_onnx_export.make_bundle(
        "logistic_regression", tasks.Task.BINARY
    )[0]
    path = bundle.export_onnx(tmp_path / "m.onnx", unverified=True)
    bundle.pipeline.steps[-1][1].coef_ = (
        bundle.pipeline.steps[-1][1].coef_ * 1.01
    )

    report = onnx_export.verify(bundle, path, held_out(), tolerance=1.0)

    assert 0 < report["max_difference"] <= 1.0


@needs_onnx
def test_fewer_held_out_rows_than_the_minimum_are_blocked(tmp_path):
    bundle = test_onnx_export.make_bundle(
        "logistic_regression", tasks.Task.BINARY
    )[0]
    path = bundle.export_onnx(tmp_path / "m.onnx", unverified=True)

    with pytest.raises(errors.BlockedError, match=f"{onnx_export.ROWS} rows"):
        onnx_export.verify(bundle, path, held_out(onnx_export.ROWS - 1))


@needs_onnx
def test_labels_that_disagree_away_from_the_boundary_are_blocked(
    monkeypatch, tmp_path
):
    bundle = test_onnx_export.make_bundle(
        "logistic_regression", tasks.Task.BINARY
    )[0]
    path = bundle.export_onnx(tmp_path / "m.onnx", unverified=True)
    flipped = {"no": "yes", "yes": "no"}
    monkeypatch.setattr(
        onnx_export,
        "decode",
        lambda bundle, session, labels: numpy.array(
            [flipped[str(label)] for label in labels], dtype=object
        ),
    )

    with pytest.raises(errors.BlockedError, match="labels"):
        onnx_export.verify(bundle, path, held_out())


@needs_onnx
def test_a_function_the_graph_cannot_express_is_refused_by_name(tmp_path):
    bundle = test_onnx_export.make_bundle(
        "logistic_regression", tasks.Task.BINARY
    )[0]
    numeric = bundle.pipeline.steps[0][1].transformers_[0][1]
    numeric.steps[0] = (
        "cast",
        preprocessing.FunctionTransformer(numpy.log1p).fit(numpy.ones((1, 2))),
    )

    with pytest.raises(errors.ArtifactError, match="log1p"):
        bundle.export_onnx(tmp_path / "m.onnx", check=held_out())


def test_too_few_held_out_rows_are_refused_before_the_converter_loads(
    monkeypatch, tmp_path
):
    bundle = test_onnx_export.make_bundle(
        "logistic_regression", tasks.Task.BINARY
    )[0]
    loads = []
    monkeypatch.setattr(onnx_export, "load", lambda: loads.append(1) or ())

    for rows in (None, held_out(onnx_export.ROWS - 1)):
        with pytest.raises(errors.BlockedError, match="at least"):
            bundle.export_onnx(tmp_path / "m.onnx", check=rows)

    assert loads == []
    assert not (tmp_path / "m.onnx").exists()
