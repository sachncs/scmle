"""Tests for column kinds."""

import numpy
import pandas
import pytest

from scmle.contracts import data
from scmle.contracts import tasks
from scmle.data.validation import columns

OPTIONS_SCALED = data.Preprocessing(scale=True)
OPTIONS_PLAIN = data.Preprocessing()


def test_kind_detection():
    frame = pandas.DataFrame(
        {
            "n": [1.0, 2.0],
            "b": [True, False],
            "s": ["a", "b"],
            "t": pandas.to_datetime(["2024-01-01", "2024-01-02"]),
        }
    )
    kinds = {c: columns.kind_of(frame[c]).name for c in frame}
    assert kinds == {
        "n": "numeric",
        "b": "numeric",
        "s": "categorical",
        "t": "datetime",
    }


def test_statistics_per_kind():
    numeric = columns.NumericKind().statistics(
        pandas.Series([1.0, 3.0, numpy.nan])
    )
    assert numeric["mean"] == 2.0 and numeric["missing_rate"] == pytest.approx(
        1 / 3
    )
    assert "mean" not in columns.NumericKind().statistics(
        pandas.Series([True, False])
    )
    top = columns.CategoricalKind().statistics(pandas.Series(list("aab")))[
        "top"
    ]
    assert top == {"a": 2, "b": 1}


def test_exclusion_rules():
    ids = pandas.Series([f"u{i}" for i in range(50)])
    assert "id-like" in columns.CategoricalKind().exclusion_reason(ids)
    assert (
        columns.CategoricalKind().exclusion_reason(
            pandas.Series(list("ab" * 25))
        )
        is None
    )
    # Datetimes are kept: the profiler expands them into features.
    assert columns.DatetimeKind().exclusion_reason(pandas.Series([1])) is None
    assert columns.NumericKind().exclusion_reason(pandas.Series([1])) is None


def test_transformers_exist_only_for_feature_kinds():
    assert columns.NumericKind().transformer(OPTIONS_SCALED) is not None
    assert columns.CategoricalKind().transformer(OPTIONS_PLAIN) is not None
    assert columns.DatetimeKind().transformer(OPTIONS_SCALED) is None


def test_scale_flag_controls_scaler():
    scaled = columns.NumericKind().transformer(OPTIONS_SCALED)
    unscaled = columns.NumericKind().transformer(OPTIONS_PLAIN)
    assert "scale" in scaled.named_steps and "scale" not in unscaled.named_steps


def test_association_detects_perfect_predictors():
    rng = numpy.random.default_rng(0)
    target = pandas.Series(rng.integers(0, 2, 200))
    leak = target * 5.0
    noise = pandas.Series(rng.normal(size=200))
    kind = columns.NumericKind()
    clf = tasks.Task.BINARY
    assert kind.association(leak, target, clf) == pytest.approx(1.0)
    assert kind.association(noise, target, clf) < 0.5
    cat = columns.CategoricalKind()
    labels = target.map({0: "no", 1: "yes"})
    assert cat.association(labels, target, clf) == pytest.approx(1.0)
    assert cat.association(labels, target, tasks.Task.REGRESSION) == 0.0


def test_regression_association_uses_rank_correlation():
    x = pandas.Series(numpy.arange(100.0))
    assert columns.NumericKind().association(
        x, x**3, tasks.Task.REGRESSION
    ) == pytest.approx(1.0)


def test_association_with_too_few_rows_is_zero():
    kind = columns.NumericKind()
    short = pandas.Series([1.0, 2.0, 3.0])
    assert (
        kind.association(short, pandas.Series([0, 1, 0]), tasks.Task.BINARY)
        == 0.0
    )


def test_numeric_statistics_include_skew():
    values = pandas.Series(
        numpy.r_[numpy.ones(95), [50.0, 60.0, 70.0, 80.0, 90.0]]
    )
    stats = columns.NumericKind().statistics(values)
    assert stats["skew"] > 1.0 and stats["max"] == 90.0


def test_numeric_transformer_composes_optional_steps():
    full = columns.NumericKind().transformer(
        data.Preprocessing(scale=True, log_skewed=True, interactions=True)
    )
    assert list(full.named_steps) == [
        "cast",
        "impute",
        "log",
        "scale",
        "interactions",
    ]


def make_outcome_recorded_only_frame(rows=400):
    rng = numpy.random.default_rng(0)
    target = pandas.Series(rng.integers(0, 2, rows))
    informative = pandas.Series(
        numpy.where(target == 1, rng.normal(size=rows), numpy.nan)
    )
    return target, informative


def test_missingness_leak_detected_when_recorded_for_one_outcome_only():
    target, informative = make_outcome_recorded_only_frame()
    kind = columns.NumericKind()
    score = kind.missingness_association(informative, target, tasks.Task.BINARY)
    assert score >= 0.9


def test_one_sided_presence_is_flagged_even_when_mostly_missing():
    rng = numpy.random.default_rng(1)
    target = pandas.Series(rng.integers(0, 2, 500))
    recorded = pandas.Series(numpy.nan, index=target.index)
    died = target.index[(target == 0)][:60]  # recorded only for outcome 0
    recorded[died] = 1.0
    score = columns.NumericKind().missingness_association(
        recorded, target, tasks.Task.BINARY
    )
    assert score == 1.0


def test_random_missingness_is_not_flagged():
    rng = numpy.random.default_rng(2)
    target = pandas.Series(rng.integers(0, 2, 500))
    values = pandas.Series(rng.normal(size=500))
    values[rng.random(500) < 0.4] = numpy.nan
    kind = columns.NumericKind()
    assert kind.missingness_association(values, target, tasks.Task.BINARY) < 0.2


def test_rare_or_nearly_complete_features_are_not_judged():
    target = pandas.Series([0, 1] * 100)
    complete = pandas.Series(numpy.arange(200.0))
    assert (
        columns.NumericKind().missingness_association(
            complete, target, tasks.Task.BINARY
        )
        == 0.0
    )
    sparse = pandas.Series([numpy.nan] * 190 + [1.0] * 10)
    assert (
        columns.NumericKind().missingness_association(
            sparse, target, tasks.Task.BINARY
        )
        == 0.0
    )


def test_regression_missingness_uses_rank_correlation():
    rng = numpy.random.default_rng(3)
    target = pandas.Series(rng.normal(size=400))
    values = pandas.Series(numpy.where(target > 0.5, 1.0, numpy.nan))
    score = columns.NumericKind().missingness_association(
        values, target, tasks.Task.REGRESSION
    )
    assert score > 0.5


def test_to_finite_maps_text_and_infinities_to_nan():
    values = pandas.Series([1.0, numpy.inf, -numpy.inf, "x", None, "3"])
    result = columns.to_finite(values)
    assert result.isna().tolist() == [False, True, True, True, True, False]
    assert result.iloc[5] == 3.0


def test_all_missing_columns_are_reported_as_missing_not_constant():
    from scmle.data.profiling import profiler

    assert (
        profiler.column_exclusion_reason(
            columns.NumericKind(),
            pandas.Series([numpy.nan] * 10),
            {"n_unique": 0, "missing_rate": 1.0},
        )
        == "missing >95%"
    )


def test_threshold_association_finds_sharp_boundaries_that_bins_blur():
    rng = numpy.random.default_rng(0)
    values = pandas.Series(rng.normal(size=300))
    separable = (values > 0.123).astype(int)
    assert columns.threshold_association(values, separable) == pytest.approx(
        1.0
    )
    assert (
        columns.NumericKind().association(values, separable, tasks.Task.BINARY)
        >= 0.98
    )
    noise = pandas.Series(rng.integers(0, 2, 300))
    assert columns.threshold_association(values, noise) < 0.5


def test_threshold_association_handles_degenerate_labels():
    values = pandas.Series(numpy.arange(50.0))
    assert columns.threshold_association(values, pandas.Series([1] * 50)) == 0.0
    multi = pandas.Series(numpy.repeat([0, 1, 2], [20, 15, 15]))
    assert columns.threshold_association(values, multi) == pytest.approx(1.0)


def sentences(count):
    return pandas.Series(
        [f"customer number {i} wrote a long review today" for i in range(count)]
    )


def test_free_text_is_its_own_kind_and_short_labels_are_not():
    assert columns.kind_of(sentences(60)).name == "text"
    labels = pandas.Series(["red", "green", "blue"] * 30)
    assert columns.kind_of(labels).name == "categorical"
    cities = pandas.Series(["New York City", "Los Angeles CA"] * 40)
    assert columns.kind_of(cities).name == "categorical"
    ids = pandas.Series([f"u{i}" for i in range(60)])
    assert columns.kind_of(ids).name == "categorical"


def test_the_text_transformer_learns_its_vocabulary_per_fit():
    kind = columns.TextKind()
    train = pandas.DataFrame(
        {"t": ["good fast service"] * 20 + ["bad slow service"] * 20}
    )
    encoder = kind.transformer(OPTIONS_PLAIN)

    matrix = encoder.fit_transform(train)
    terms = list(encoder.named_steps["tfidf"].get_feature_names_out())

    assert matrix.shape[0] == 40 and "service" in terms and "good" in terms
    unseen = encoder.transform(
        pandas.DataFrame({"t": ["totally new words only"]})
    )
    assert unseen.shape == (1, len(terms)) and unseen.sum() == 0


def test_text_statistics_report_words_per_value():
    stats = columns.TextKind().statistics(sentences(40))

    assert stats["kind"] == "text" and stats["mean_words"] == 8.0
