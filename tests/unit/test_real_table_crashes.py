"""Two shapes that crashed runs on real tables now fit.

OpenML table 315 has a categorical column with about 1,200 levels, past
the 255 that HistGradientBoosting accepts natively. Table 363711 has a class
so rare that a slice of the training rows holds fewer examples of it than
the three calibration folds ask for.
"""

import pickle

import numpy
import pandas
import pytest

from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.splitting import inner_cv
from scmle.models.sklearn import classical
from scmle.preparation import preprocessing


def wide_frame(rows=3000, levels=1200, seed=0):
    rng = numpy.random.default_rng(seed)
    code = rng.integers(0, levels, rows)
    x = rng.normal(size=rows)
    frame = pandas.DataFrame({"code": [f"c{i}" for i in code], "x": x})
    target = pandas.Series((x + (code % 7) / 7 > 0.5).astype(int))
    return frame, target


def rare_frame(rare, rows=60, seed=0):
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame(
        {"x": rng.normal(size=rows), "z": rng.normal(size=rows)}
    )
    labels = numpy.where(frame.x > 0, "a", "b").astype(object)
    labels[:rare] = "rare"
    return frame, pandas.Series(labels)


PLAN = data.Blueprint(
    numeric=("x",), categorical=("code",), high_cardinality=("code",)
)


@pytest.mark.parametrize("task", [tasks.Task.BINARY, tasks.Task.REGRESSION])
def test_native_categoricals_past_the_level_limit_fit(task):
    frame, target = wide_frame()
    model = preprocessing.instantiate(
        PLAN, classical.HIST_GRADIENT_BOOSTING, task, 0
    )
    model.fit(frame, target)
    encoder = (
        model.named_steps["prep"]
        .named_transformers_["high_cardinality"]
        .named_steps["encode"]
    )
    # The pooled levels are the recorded decision.
    assert len(encoder.infrequent_categories_[0]) > 0
    codes = model.named_steps["prep"].transform(frame)[:, 0]
    assert codes.max() < preprocessing.NATIVE_LEVELS


@pytest.mark.parametrize("rare", [1, 2])
def test_svm_calibration_adapts_folds_to_a_rare_class(rare):
    frame, target = rare_frame(rare)
    model = classical.SUPPORT_VECTOR.build(tasks.Task.MULTICLASS, 0)
    model.fit(frame, target)
    assert model.predict_proba(frame).shape == (len(frame), 3)
    assert model.folds_ == max(inner_cv.MIN_FOLDS, rare)


@pytest.mark.parametrize("task", [tasks.Task.MULTICLASS, tasks.Task.REGRESSION])
def test_hist_gradient_boosting_stops_early_like_a_practitioner(task):
    # Without a stopping rule it runs a fixed 100 rounds: too many for a
    # small table (log loss overfits), too few for a smooth one.
    model = classical.HIST_GRADIENT_BOOSTING.build(task, 0)
    assert model.get_params()["early_stopping"] is True
    assert model.get_params()["max_iter"] == classical.ROUNDS
    assert model.get_params()["n_iter_no_change"] == classical.PATIENCE


def test_hist_gradient_boosting_adapter_pickles_for_the_trial_process():
    # Trials run in a separate process, which receives the adapter pickled.
    adapter = pickle.loads(pickle.dumps(classical.HIST_GRADIENT_BOOSTING))
    assert adapter.fixed_params == classical.STOPPING


@pytest.mark.parametrize("rare", [1, 2])
def test_hist_gradient_boosting_fits_with_a_single_row_class(rare):
    # A stratified stopping split needs two rows of every class.
    frame, target = rare_frame(rare, rows=200)
    model = classical.HIST_GRADIENT_BOOSTING.build(tasks.Task.MULTICLASS, 0)
    model.fit(frame, target)
    assert model.predict_proba(frame).shape == (len(frame), 3)
    assert model.get_params()["early_stopping"] is True
    assert model.stopped_ is (rare >= classical.STRATIFIED)


def test_inner_splitter_calibration_adapts_folds():
    frame, target = rare_frame(2)
    model = inner_cv.Calibrated(
        classical.LOGISTIC_REGRESSION.build(tasks.Task.MULTICLASS, 0),
        method="sigmoid",
        cv=inner_cv.Splitter(3, seed=0),
    )
    model.fit(frame, target)
    assert model.folds_ == 2
    assert model.cv.n_splits == 3  # The configured value is kept.


def test_calibration_keeps_folds_when_classes_suffice():
    frame, target = rare_frame(10)
    model = classical.SUPPORT_VECTOR.build(tasks.Task.MULTICLASS, 0)
    model.fit(frame, target)
    assert model.folds_ == classical.CALIBRATION_FOLDS


def test_calibration_refuses_a_single_class():
    frame = pandas.DataFrame({"x": [0.0, 1.0, 2.0]})
    model = classical.SUPPORT_VECTOR.build(tasks.Task.BINARY, 0)
    with pytest.raises(errors.InputError, match="two classes"):
        model.fit(frame, pandas.Series(["a", "a", "a"]))
