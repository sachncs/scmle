"""Awkward but legitimate datasets must work or be refused clearly."""

import numpy as np
import pandas as pd
import support

from mml.contracts import errors


def base(rows=200, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {"a": rng.normal(size=rows), "b": rng.normal(size=rows)}
    )
    frame["y"] = (frame.a + rng.normal(scale=0.5, size=rows) > 0).astype(int)
    return frame


def run(frame, tmp_path, target="y", **kwargs):
    return support.make_project(
        frame, target, time_budget=300, output_dir=tmp_path, **kwargs
    ).run()


def check_inference(result, frame, target="y"):
    features = frame.drop(columns=[target])
    predictions = result.predict(features)
    assert len(predictions) == len(features)
    return predictions


def test_tiny_dataset(tmp_path):
    frame = base(60)
    result = run(frame, tmp_path, folds=3)
    check_inference(result, frame)


def test_single_feature(tmp_path):
    frame = base()[["a", "y"]]
    check_inference(run(frame, tmp_path), frame)


def test_all_missing_and_constant_columns_are_dropped(tmp_path):
    frame = base().assign(empty=np.nan, same=7)
    result = run(frame, tmp_path)
    dropped = result.profile.feature_plan.dropped
    assert dropped["empty"] == "missing >95%" and dropped["same"] == "constant"
    check_inference(result, frame)


def test_unicode_and_spaced_column_names(tmp_path):
    frame = base().rename(
        columns={"a": "año de ingreso", "b": "col with space"}
    )
    result = run(frame, tmp_path)
    assert "año de ingreso" in result.bundle.features
    check_inference(result, frame)


def test_boolean_target_and_features(tmp_path):
    frame = base()
    frame["flag"] = frame.a > 0.2
    frame["y"] = frame["y"].astype(bool)
    result = run(frame, tmp_path)
    assert set(result.predict(frame.drop(columns="y"))) <= {False, True}


def test_pandas_nullable_and_string_dtypes(tmp_path):
    frame = base()
    frame["count"] = pd.array(np.arange(len(frame)) % 7, dtype="Int64")
    frame.loc[::9, "count"] = pd.NA
    frame["label"] = pd.array(np.where(frame.a > 0, "p", "q"), dtype="string")
    frame["ok"] = pd.array(frame.b > 0, dtype="boolean")
    result = run(frame, tmp_path)
    check_inference(result, frame)


def test_categorical_dtype_column(tmp_path):
    frame = base()
    frame["segment"] = pd.Categorical(np.where(frame.a > 0, "hi", "lo"))
    check_inference(run(frame, tmp_path), frame)


def test_very_high_cardinality_categorical(tmp_path):
    rng = np.random.default_rng(1)
    frame = base(400)
    frame["code"] = rng.choice([f"c{i}" for i in range(150)], 400)
    result = run(frame, tmp_path)
    check_inference(result, frame)


def test_infinite_values_are_treated_as_missing(tmp_path):
    frame = base()
    frame.loc[::17, "a"] = np.inf
    frame.loc[::23, "b"] = -np.inf
    result = run(frame, tmp_path)
    check_inference(result, frame)
    notes = [q for q in result.profile.quality_findings if "infinite" in q]
    assert len(notes) == 2 and "treated as missing" in notes[0]
    assert np.isfinite(result.predict_proba(frame.drop(columns="y"))).all()


def test_duplicate_rows_and_conflicting_labels(tmp_path):
    frame = pd.concat([base(120), base(120)], ignore_index=True)
    frame.loc[:20, "y"] = 1 - frame.loc[:20, "y"]
    result = run(frame, tmp_path)
    assert any("duplicate" in q for q in result.profile.quality_findings)


def test_wide_dataset_with_few_rows(tmp_path):
    rng = np.random.default_rng(2)
    frame = pd.DataFrame(
        rng.normal(size=(150, 200)), columns=[f"f{i}" for i in range(200)]
    )
    frame["y"] = (frame.f0 + frame.f1 > 0).astype(int)
    result = run(frame, tmp_path)
    check_inference(result, frame)


def test_datetime_feature_is_excluded_not_fatal(tmp_path):
    frame = base().assign(
        when=pd.date_range("2024-01-01", periods=200, freq="D")
    )
    result = run(frame, tmp_path)
    assert "datetime" in result.profile.feature_plan.dropped["when"]
    check_inference(result, frame)


def test_rare_class_is_blocked_or_reduced_clearly(tmp_path):
    frame = base(300)
    frame["y"] = np.where(
        np.arange(300) < 2, "rare", np.where(frame.a > 0, "x", "z")
    )
    try:
        result = run(frame, tmp_path, task="classification")
    except errors.BlockedError as error:
        assert "class" in str(error)
    else:
        check_inference(result, frame)


def test_regression_with_outliers_and_nans(tmp_path):
    frame = base()
    frame["y"] = frame.a * 3
    frame.loc[::31, "y"] = 1e6
    frame.loc[::13, "b"] = np.nan
    result = run(frame, tmp_path, task="regression")
    check_inference(result, frame)


def test_numeric_strings_in_object_columns(tmp_path):
    frame = base().assign(code=lambda f: (f.index % 5).astype(str))
    check_inference(run(frame, tmp_path), frame)


def test_target_named_like_a_feature_prefix(tmp_path):
    frame = (
        base()
        .rename(columns={"y": "outcome"})
        .assign(outcome_date_id=lambda f: f.index)
    )
    result = run(frame, tmp_path, target="outcome")
    check_inference(result, frame, "outcome")


def test_shuffled_columns_at_inference(tmp_path):
    frame = base()
    result = run(frame, tmp_path)
    features = frame.drop(columns="y")[["b", "a"]]
    np.testing.assert_array_equal(
        result.predict(features), result.predict(frame.drop(columns="y"))
    )


def test_dataset_with_index_not_starting_at_zero(tmp_path):
    frame = base()
    frame.index = frame.index + 1000
    check_inference(run(frame, tmp_path), frame)
