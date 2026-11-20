"""Behavioural tests for the data strategies: applicability and builds."""

import types

import numpy
import pandas
import pytest
from sklearn import compose

from scmle.contracts import errors
from scmle.contracts import experiments
from scmle.contracts import tasks
from scmle.data.readers import tabular
from scmle.data.splitting import inner_cv
import scmle.models.registry
from scmle.pipeline import profiler
from scmle.preparation import resampling
from scmle.preparation.strategies import features
from scmle.preparation.strategies import interface
from scmle.preparation.strategies import rebalancing
from scmle.preparation.strategies import registry

CATALOG = scmle.models.registry.Catalog.stock()


def plan(numeric=(), categorical=(), high=(), monotonic=None, informative=None):
    return types.SimpleNamespace(
        numeric=list(numeric),
        categorical=list(categorical),
        cardinal=list(high),
        monotonic=monotonic or {},
        informative_missing=informative,
    )


def portrait(target=None, stats=None, **kwargs):
    return types.SimpleNamespace(
        summary=target or {},
        statistics=stats or {},
        plan=plan(**kwargs),
    )


def why(strategy, profile, model, task=tasks.Task.BINARY):
    return strategy.applicability(profile, task, CATALOG.get(model))


def test_the_registry_lists_every_strategy():
    assert sorted(registry.STRATEGIES) == sorted(
        [
            experiments.DEFAULT_STRATEGY,
            "class_weight",
            "random_oversample",
            "random_undersample",
            "smote",
            "borderline_smote",
            "adasyn",
            "jitter_augment",
            "copula_augment",
            "calibrated",
            "log_skewed",
            "interactions",
            "construct_features",
            "select_features",
            "missing_indicators",
            "iterative_imputation",
            "target_transform",
            "target_encoding",
            "rare_target_weighting",
        ]
    )
    assert experiments.DEFAULT_STRATEGY not in registry.EXPERIMENTAL
    assert len(registry.EXPERIMENTAL) == 18
    assert frozenset({"jitter_augment", "copula_augment"}) == registry.OPT_IN
    assert "jitter_augment" not in registry.DEFAULT_NAMES
    assert len(registry.DEFAULT_NAMES) == 16
    assert isinstance(registry.lookup("smote"), rebalancing.PlainSmote)
    with pytest.raises(errors.ConfigError, match="unknown strategy 'nope'"):
        registry.lookup("nope")


def test_the_thresholds_have_their_documented_values():
    assert interface.MIN_IMBALANCE_RATIO == 3.0
    assert interface.TARGET_SKEW_THRESHOLD == 1.0
    assert interface.SKEW_THRESHOLD == 1.0
    assert interface.MIN_INTERACTION_FEATURES == 2
    assert interface.MAX_INTERACTION_FEATURES == 12
    assert interface.MIN_SELECTION_FEATURES == 6
    assert interface.MIN_MISSING_RATE == 0.01
    assert interface.MAX_ITERATIVE_FEATURES == 50
    assert interface.CALIBRATION_FOLDS == 3


def test_the_default_strategy_always_applies():
    default = interface.Default()
    assert why(default, portrait(), "dummy") is None
    assert default.name == experiments.DEFAULT_STRATEGY


def test_every_other_strategy_refuses_the_baseline():
    for name in registry.EXPERIMENTAL:
        assert why(registry.lookup(name), portrait(), "dummy") == (
            "the baseline model is never transformed"
        )


def test_rebalancing_needs_skewed_classification():
    weighting = rebalancing.Weighting()
    skewed = portrait({"imbalance_ratio": 3.0})
    assert why(weighting, skewed, "logistic_regression") is None
    assert why(weighting, skewed, "ridge", tasks.Task.REGRESSION) == (
        "rebalancing is a classification technique"
    )
    assert why(weighting, portrait({"imbalance_ratio": 2.9}), "ridge") == (
        "classes are balanced enough (2.9:1 < 3:1)"
    )
    assert why(weighting, portrait(), "ridge") == (
        "classes are balanced enough (1.0:1 < 3:1)"
    )
    assert why(weighting, skewed, "knn") == "knn has no class_weight parameter"
    assert why(rebalancing.Oversampling(), skewed, "knn") is None


def test_smote_needs_the_extra_and_numeric_features(monkeypatch):
    smote = rebalancing.PlainSmote()
    skewed = {"imbalance_ratio": 5.0}
    monkeypatch.setattr(
        rebalancing.importlib.util, "find_spec", lambda name: None
    )
    assert why(smote, portrait(skewed, numeric=["a", "b"]), "knn") == (
        "needs the 'sampling' extra (imbalanced-learn)"
    )
    monkeypatch.setattr(
        rebalancing.importlib.util, "find_spec", lambda name: object()
    )
    assert why(
        smote, portrait(skewed, numeric=["a", "b"], categorical=["c"]), "knn"
    ) == ("interpolating one-hot categories is not meaningful")
    assert why(smote, portrait(skewed, numeric=["a"]), "knn") == (
        "needs at least two numeric features"
    )
    assert why(smote, portrait(skewed, numeric=["a", "b"]), "knn") is None
    assert rebalancing.Borderline.sampler == "BorderlineSMOTE"
    assert rebalancing.Adasyn.sampler == "ADASYN"
    assert rebalancing.Smote.minimum == 2


def test_recalibration_needs_an_uncalibrated_classifier():
    calibration = rebalancing.Recalibration()
    assert why(calibration, portrait(), "random_forest") is None
    assert why(calibration, portrait(), "ridge", tasks.Task.REGRESSION) == (
        "calibration applies to classification"
    )
    assert why(calibration, portrait(), "logistic_regression") == (
        "logistic regression is already well calibrated"
    )
    blind = types.SimpleNamespace(
        family="tree",
        name="blind",
        capabilities=types.SimpleNamespace(probabilities=False),
    )
    assert calibration.applicability(portrait(), tasks.Task.BINARY, blind) == (
        "blind does not provide probabilities"
    )


def test_logarithm_needs_a_skewed_non_negative_feature():
    logarithm = features.Logarithm()
    stats = {"a": {"skew": 1.5, "min": 0.0}}
    assert why(logarithm, portrait(stats=stats, numeric=["a"]), "ridge") is None
    assert why(
        logarithm, portrait(stats=stats, numeric=["a"]), "random_forest"
    ) == ("monotone transforms do not change tree splits")
    assert why(logarithm, portrait(stats=stats, numeric=["a"]), "lightgbm") == (
        "monotone transforms do not change tree splits"
    )
    for column in (
        {"skew": 1.0, "min": 0.0},
        {"skew": 2.0, "min": -0.1},
        {"skew": 2.0},
        {"min": 0.0},
    ):
        profile = portrait(stats={"a": column}, numeric=["a"])
        assert why(logarithm, profile, "ridge") == (
            "no right-skewed non-negative numeric feature"
        )
    assert why(logarithm, portrait(stats=stats, numeric=["b"]), "ridge") == (
        "no right-skewed non-negative numeric feature"
    )


def test_rarity_needs_a_long_tailed_regression_target():
    rarity = features.Rarity()
    tail = {"skew": -1.0}
    assert why(rarity, portrait(tail), "ridge", tasks.Task.REGRESSION) is None
    assert why(rarity, portrait(tail), "ridge", tasks.Task.BINARY) == (
        "regression only (classification uses class weighting)"
    )
    assert why(
        rarity, portrait({"skew": 0.5}), "ridge", tasks.Task.REGRESSION
    ) == ("target is not long-tailed (|skew| 0.50)")
    assert why(rarity, portrait(tail), "knn", tasks.Task.REGRESSION) == (
        "knn does not accept sample weights"
    )


def test_transformation_needs_plain_regression():
    transform = features.Transformation()
    tail = {"skew": 2.0}
    assert (
        why(transform, portrait(tail), "ridge", tasks.Task.REGRESSION) is None
    )
    assert why(transform, portrait(tail), "ridge", tasks.Task.BINARY) == (
        "target transforms are a regression technique"
    )
    assert why(
        transform, portrait({"skew": 0.99}), "ridge", tasks.Task.REGRESSION
    ) == ("target is not long-tailed (|skew| 0.99)")


def test_encoding_needs_a_high_cardinality_column():
    encoding = features.Encoding()
    assert why(encoding, portrait(high=["c"]), "ridge") is None
    assert why(encoding, portrait(), "ridge") == (
        "no high-cardinality categorical feature"
    )


@pytest.mark.parametrize(
    ("count", "reason"),
    [
        (1, "1 numeric features is outside [2, 12]"),
        (2, None),
        (12, None),
        (13, "13 numeric features is outside [2, 12]"),
    ],
)
def test_interactions_need_a_moderate_linear_model(count, reason):
    profile = portrait(numeric=[str(i) for i in range(count)])
    assert why(features.Interactions(), profile, "ridge") == reason


def test_interactions_refuse_non_linear_models():
    profile = portrait(numeric=["a", "b"])
    assert why(features.Interactions(), profile, "knn") == (
        "non-linear models learn interactions themselves"
    )


def test_construction_needs_two_numeric_features():
    construction = features.Construction()
    assert why(construction, portrait(numeric=["a", "b"]), "knn") is None
    assert why(construction, portrait(numeric=["a"]), "knn") == (
        "fewer than two numeric features to combine"
    )


def test_selection_needs_enough_unconstrained_features():
    selection = features.Selection()
    six = portrait(numeric=list("abcd"), categorical=list("ef"))
    assert why(selection, six, "knn") is None
    five = portrait(numeric=list("abc"), categorical=list("ef"))
    assert why(selection, five, "knn") == "fewer than 6 features"
    bound = portrait(numeric=list("abcdef"), monotonic={"a": 1})
    assert why(selection, bound, "knn") == (
        "feature selection would shift the monotonic constraints"
    )


def test_imputation_needs_gaps_the_model_cannot_handle():
    iterative = features.IterativeImputation()
    gaps = {"a": {"missing_rate": 0.01}}
    assert why(iterative, portrait(stats=gaps, numeric=["a"]), "knn") is None
    assert why(iterative, portrait(stats=gaps, numeric=["a"]), "lightgbm") == (
        "the model handles missing values natively"
    )
    few = {"a": {"missing_rate": 0.009}}
    assert why(iterative, portrait(stats=few, numeric=["a"]), "knn") == (
        "no numeric feature is missing in >= 1%"
    )
    wide = [str(i) for i in range(51)]
    stats = {name: {"missing_rate": 0.5} for name in wide}
    assert why(iterative, portrait(stats=stats, numeric=wide), "knn") == (
        "51 numeric features is too many to impute iteratively"
    )
    fifty = wide[:50]
    assert why(iterative, portrait(stats=stats, numeric=fifty), "knn") is None


def test_indicators_need_informative_gaps():
    indicators = features.Indicators()
    gaps = {"a": {"missing_rate": 0.2}}
    unassessed = portrait(stats=gaps, numeric=["a"])
    assert why(indicators, unassessed, "knn") is None
    flagged = portrait(stats=gaps, numeric=["a"], informative=["a"])
    assert why(indicators, flagged, "knn") is None
    silent = portrait(stats=gaps, numeric=["a"], informative=["z"])
    assert why(indicators, silent, "knn") == (
        "no numeric feature's gaps are associated with the target"
    )


def test_feature_strategy_flags():
    assert features.Logarithm.skewed
    assert features.Interactions.interactions
    assert features.Construction.construct
    assert features.Selection.select
    assert features.Indicators.missing == "indicator"
    assert features.IterativeImputation.missing == "iterative"
    assert features.Features.missing == "median"


@pytest.fixture(scope="module")
def real_plan():
    rng = numpy.random.default_rng(0)
    frame = pandas.DataFrame(
        {"a": rng.exponential(2.0, 200), "b": rng.normal(size=200)}
    )
    frame["y"] = (rng.random(200) < 0.2).astype(int)
    spec = tasks.Config(data=tasks.Dataset(uri="m", target="y"))
    manifest = tabular.manifested(frame, "y")
    return (
        profiler.Profiler(spec).profile(frame, manifest, tasks.Task.BINARY).plan
    )


def test_builds_wrap_the_pipeline_as_each_strategy_requires(real_plan):
    def build(name, model="logistic_regression", task=tasks.Task.BINARY):
        return registry.lookup(name).build(
            real_plan, CATALOG.get(model), task, 0
        )

    weighted = build("class_weight")
    assert weighted.get_params()["model__class_weight"] == "balanced"
    oversampled = build("random_oversample")
    assert isinstance(oversampled, resampling.ResamplingClassifier)
    assert isinstance(oversampled.resampler, resampling.Oversampler)
    undersampled = build("random_undersample")
    assert isinstance(undersampled.resampler, resampling.Undersampler)
    assert isinstance(build("calibrated"), inner_cv.Calibrated)
    transformed = build("target_transform", "ridge", tasks.Task.REGRESSION)
    assert isinstance(transformed, compose.TransformedTargetRegressor)
    weighted_rows = build(
        "rare_target_weighting", "ridge", tasks.Task.REGRESSION
    )
    assert isinstance(weighted_rows, resampling.Density)
    for name in ("default", "log_skewed", "select_features", "target_encoding"):
        assert build(name).get_params()
