"""Tests for data strategies."""

import numpy as np
import pandas as pd
import pytest

from mml.contracts import errors
from mml.contracts import tasks
from mml.data.profiling import profiler
from mml.data.readers import tabular
from mml.models import registry as registry_module
from mml.sampling import resampling
from mml.sampling import strategies

REGISTRY = registry_module.ModelRegistry.with_builtins()
BINARY = tasks.TaskType.BINARY


def profile_of(frame, task=BINARY):
    spec = tasks.ProjectSpec(data=tasks.DataSpec(uri="m", target="y"))
    manifest = tabular.build_manifest(frame, "y")
    return profiler.DatasetProfiler(spec).profile(frame, manifest, task)


@pytest.fixture
def skewed_imbalanced():
    rng = np.random.default_rng(0)
    rows = 400
    frame = pd.DataFrame(
        {
            "spend": rng.exponential(5.0, rows),
            "age": rng.normal(40, 10, rows),
            "tenure": rng.exponential(2.0, rows),
        }
    )
    frame["y"] = (rng.random(rows) < 0.1).astype(int)
    return frame


@pytest.fixture
def balanced():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    frame["y"] = rng.integers(0, 2, 300)
    return frame


def reason(name, profile, model, task=BINARY):
    return strategies.get_strategy(name).applicability(
        profile, task, REGISTRY.get(model)
    )


def test_imbalance_strategies_apply_only_to_skewed_classification(
    skewed_imbalanced, balanced
):
    skewed, even = profile_of(skewed_imbalanced), profile_of(balanced)
    for name in ("class_weight", "random_oversample", "random_undersample"):
        assert reason(name, skewed, "logistic_regression") is None
        assert "balanced enough" in reason(name, even, "logistic_regression")
    regression = skewed_imbalanced.assign(y=skewed_imbalanced.spend * 2)
    assert "classification" in reason(
        "random_oversample",
        profile_of(regression, tasks.TaskType.REGRESSION),
        "ridge",
        tasks.TaskType.REGRESSION,
    )


def test_class_weight_requires_estimator_support(skewed_imbalanced):
    from sklearn import dummy

    from mml.contracts import models

    class NoWeights(models.ModelAdapter):
        name = "no_weights"
        family = "test"
        capabilities = models.Capabilities(frozenset({BINARY}))

        def build(self, task, seed, params=None):
            return dummy.DummyClassifier(strategy="prior")

    profile = profile_of(skewed_imbalanced)
    assert reason("class_weight", profile, "random_forest") is None
    strategy = strategies.get_strategy("class_weight")
    found = strategy.applicability(profile, BINARY, NoWeights())
    assert found == "no_weights has no class_weight parameter"


def test_baseline_is_never_transformed(skewed_imbalanced):
    profile = profile_of(skewed_imbalanced)
    for name in strategies.EXPERIMENTAL:
        assert "baseline" in reason(name, profile, "dummy")
    assert reason("default", profile, "dummy") is None


def test_feature_strategies_follow_model_family(skewed_imbalanced):
    profile = profile_of(skewed_imbalanced)
    assert reason("log_skewed", profile, "logistic_regression") is None
    assert "tree splits" in reason("log_skewed", profile, "random_forest")
    assert reason("interactions", profile, "logistic_regression") is None
    assert "non-linear" in reason("interactions", profile, "random_forest")


def test_log_skewed_needs_a_skewed_feature(balanced):
    assert "no right-skewed" in reason(
        "log_skewed", profile_of(balanced), "logistic_regression"
    )


def test_interaction_feature_count_limits():
    rng = np.random.default_rng(2)
    wide = pd.DataFrame({f"f{i}": rng.normal(size=100) for i in range(14)})
    wide["y"] = rng.integers(0, 2, 100)
    assert "outside" in reason(
        "interactions", profile_of(wide), "logistic_regression"
    )
    narrow = wide[["f0", "y"]]
    assert "outside" in reason(
        "interactions", profile_of(narrow), "logistic_regression"
    )


def test_builds_produce_fittable_candidates(skewed_imbalanced):
    import importlib.util

    profile = profile_of(skewed_imbalanced)
    plan = profile.feature_plan
    features = skewed_imbalanced.drop(columns="y")
    target = skewed_imbalanced["y"]
    has_sampling = importlib.util.find_spec("imblearn") is not None
    for name in strategies.STRATEGIES:
        if name in {"smote", "borderline_smote", "adasyn"} and not has_sampling:
            continue
        if name in {"rare_target_weighting", "target_transform"}:
            continue  # regression only; covered by their own tests
        adapter = REGISTRY.get("logistic_regression")
        candidate = strategies.get_strategy(name).build(
            plan, adapter, BINARY, 0
        )
        candidate.fit(features, target)
        assert candidate.predict_proba(features).shape == (400, 2), name


def test_resampling_strategy_wraps_the_pipeline(skewed_imbalanced):
    plan = profile_of(skewed_imbalanced).feature_plan
    candidate = strategies.get_strategy("random_undersample").build(
        plan, REGISTRY.get("random_forest"), BINARY, 3
    )
    assert isinstance(candidate, resampling.ResamplingClassifier)
    assert candidate.random_state == 3


def test_class_weight_is_injected_over_other_params(skewed_imbalanced):
    plan = profile_of(skewed_imbalanced).feature_plan
    candidate = strategies.get_strategy("class_weight").build(
        plan, REGISTRY.get("logistic_regression"), BINARY, 0, {"C": 0.5}
    )
    model = candidate.named_steps["model"]
    assert model.class_weight == "balanced" and model.C == 0.5


def test_registry_and_unknown_strategy():
    assert strategies.EXPERIMENTAL == (
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
    )
    assert "jitter_augment" not in strategies.DEFAULT_NAMES
    with pytest.raises(errors.ConfigError, match="unknown strategy"):
        strategies.get_strategy("nope")


def smote_profile():
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    frame["y"] = (rng.random(500) < 0.08).astype(int)
    frame.loc[frame.y == 1, "a"] += 1.5
    return frame, profile_of(frame)


def test_smote_family_needs_the_sampling_extra_numeric_features_and_imbalance(
    balanced,
):
    imblearn = pytest.importorskip("imblearn")
    assert imblearn
    frame, profile = smote_profile()
    for name in ("smote", "borderline_smote", "adasyn"):
        assert reason(name, profile, "random_forest") is None
        assert "balanced enough" in reason(
            name, profile_of(balanced), "random_forest"
        )
    mixed = frame.assign(cat=np.where(frame.a > 0, "p", "q"))
    assert "one-hot" in reason("smote", profile_of(mixed), "random_forest")
    single = frame[["a", "y"]]
    assert "two numeric" in reason("smote", profile_of(single), "random_forest")


def test_smote_without_the_extra_explains_itself(monkeypatch):
    profile = smote_profile()[1]
    monkeypatch.setattr(
        strategies.importlib.util, "find_spec", lambda name: None
    )
    assert "sampling" in reason("smote", profile, "random_forest")


@pytest.mark.parametrize("name", ["smote", "borderline_smote", "adasyn"])
def test_smote_candidates_fit_and_resample_only_the_training_fold(name):
    pytest.importorskip("imblearn")
    frame, profile = smote_profile()
    adapter = REGISTRY.get("logistic_regression")
    candidate = strategies.get_strategy(name).build(
        profile.feature_plan, adapter, BINARY, 0
    )
    assert isinstance(candidate, resampling.PreprocessedResamplingClassifier)
    features, target = frame.drop(columns="y"), frame["y"]
    candidate.fit(features.iloc[:350], target.iloc[:350])
    held = features.iloc[350:]
    assert candidate.predict_proba(held).shape == (150, 2)
    assert candidate.predict(held).shape == (150,)


def test_jitter_is_opt_in_and_only_perturbs_synthetic_numeric_rows():
    assert "jitter_augment" in strategies.OPT_IN
    assert "jitter_augment" not in strategies.DEFAULT_NAMES
    rng = np.random.default_rng(6)
    features = pd.DataFrame(
        {"a": rng.normal(size=200), "c": rng.choice(["x", "y"], 200)}
    )
    target = pd.Series([1] * 20 + [0] * 180)
    sampled, labels = resampling.JitterOverSampler(0.1).resample(
        features, target, 0
    )
    assert labels.value_counts().to_dict() == {0: 180, 1: 180}
    originals = set(features["a"].round(12))
    synthetic = ~sampled["a"].round(12).isin(originals)
    assert synthetic.sum() >= 150  # nearly all extra minority rows are jittered
    assert set(sampled["c"]) <= {"x", "y"}  # categoricals copied untouched
    assert (labels[synthetic] == 1).all()
    assert sampled["a"].std() < 1.5 * features["a"].std()


def ratio_problem(rows=900, seed=3):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "price": rng.uniform(1, 10, rows),
            "quantity": rng.uniform(1, 10, rows),
            "noise": rng.normal(size=rows),
        }
    )
    frame["y"] = (frame.price / frame.quantity > 1).astype(int)
    return frame


def wide_problem(rows=800, noise_columns=10, seed=4):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {f"n{i}": rng.normal(size=rows) for i in range(noise_columns)}
    )
    frame["s0"] = rng.normal(size=rows)
    frame["s1"] = rng.normal(size=rows)
    frame["y"] = (frame.s0 + frame.s1 + rng.normal(0, 0.5, rows) > 0).astype(
        int
    )
    return frame


def cross_validated(frame, strategy_name, model):
    from sklearn import model_selection

    profile = profile_of(frame)
    strategy = strategies.get_strategy(strategy_name)
    estimator = strategy.build(
        profile.feature_plan, REGISTRY.get(model), BINARY, 0
    )
    return model_selection.cross_val_score(
        estimator,
        frame.drop(columns="y"),
        frame["y"],
        cv=4,
        scoring="roc_auc",
    ).mean()


def test_constructed_ratios_let_a_linear_model_learn_a_ratio_target():
    from sklearn import model_selection

    rng = np.random.default_rng(5)
    frame = pd.DataFrame(
        {
            "price": rng.uniform(1, 10, 900),
            "quantity": rng.uniform(1, 10, 900),
            "noise": rng.normal(size=900),
        }
    )
    frame["y"] = frame.price / frame.quantity + rng.normal(0, 0.05, 900)
    spec = tasks.ProjectSpec(data=tasks.DataSpec(uri="m", target="y"))
    profile = profiler.DatasetProfiler(spec).profile(
        frame,
        tabular.build_manifest(frame, "y"),
        tasks.TaskType.REGRESSION,
    )

    def r2(name):
        estimator = strategies.get_strategy(name).build(
            profile.feature_plan,
            REGISTRY.get("ridge"),
            tasks.TaskType.REGRESSION,
            0,
        )
        return model_selection.cross_val_score(
            estimator, frame.drop(columns="y"), frame["y"], cv=4, scoring="r2"
        ).mean()

    assert r2("construct_features") > r2("default") + 0.2


def test_feature_selection_keeps_the_signal_and_drops_noise_columns():
    frame = wide_problem()
    profile = profile_of(frame)
    pipeline = strategies.get_strategy("select_features").build(
        profile.feature_plan, REGISTRY.get("logistic_regression"), BINARY, 0
    )

    pipeline.fit(frame.drop(columns="y"), frame["y"])
    kept = pipeline.named_steps["select"].support_

    assert kept.sum() < 8
    names = list(pipeline.named_steps["prep"].get_feature_names_out())
    kept_names = [n for n, flag in zip(names, kept, strict=True) if flag]
    assert {"s0", "s1"} <= set(kept_names)


def test_selection_scores_as_well_as_the_full_set_with_fewer_features():
    frame = wide_problem()

    full = cross_validated(frame, "default", "logistic_regression")
    chosen = cross_validated(frame, "select_features", "logistic_regression")

    assert chosen >= full - 0.01


def test_applicability_of_the_feature_strategies_is_explained():
    single = pd.DataFrame({"a": np.arange(100.0), "y": [0, 1] * 50})
    profile = profile_of(single)

    assert "fewer than two numeric" in reason(
        "construct_features", profile, "logistic_regression"
    )
    assert "fewer than 6 features" in reason(
        "select_features", profile, "logistic_regression"
    )
    assert (
        reason(
            "select_features", profile_of(wide_problem()), "logistic_regression"
        )
        is None
    )
    assert "baseline" in reason(
        "construct_features", profile_of(ratio_problem()), "dummy"
    )


def gappy_problem(rows=900, seed=7):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=rows)
    frame = pd.DataFrame(
        {
            "a": base,
            "b": base * 2 + rng.normal(0, 0.1, rows),
            "c": rng.normal(size=rows),
        }
    )
    frame["y"] = (frame.b + rng.normal(0, 0.3, rows) > 0).astype(int)
    holes = rng.random(rows) < 0.3
    frame.loc[holes, "b"] = np.nan
    return frame, holes


def test_missing_indicators_append_a_flag_for_features_with_gaps():
    frame = gappy_problem()[0]
    profile = profile_of(frame)
    pipeline = strategies.get_strategy("missing_indicators").build(
        profile.feature_plan, REGISTRY.get("logistic_regression"), BINARY, 0
    )

    pipeline.fit(frame.drop(columns="y"), frame["y"])
    width = (
        pipeline.named_steps["prep"].transform(frame.drop(columns="y")).shape[1]
    )

    assert width == 4


def test_iterative_imputation_recovers_values_better_than_the_median():
    frame, holes = gappy_problem()
    truth = frame.copy()
    rng = np.random.default_rng(7)
    base = rng.normal(size=len(frame))
    truth["b"] = base * 2 + rng.normal(0, 0.1, len(frame))
    profile = profile_of(frame)

    def error(strategy):
        pipeline = strategies.get_strategy(strategy).build(
            profile.feature_plan, REGISTRY.get("logistic_regression"), BINARY, 0
        )
        pipeline.fit(frame.drop(columns="y"), frame["y"])
        names = list(pipeline.named_steps["prep"].get_feature_names_out())
        matrix = pipeline.named_steps["prep"].transform(frame.drop(columns="y"))
        return matrix, names

    matrix, names = error("iterative_imputation")
    filled = matrix[:, names.index("b")][holes]
    partner = matrix[:, names.index("a")][holes]
    assert abs(np.corrcoef(filled, partner)[0, 1]) > 0.9
    default_matrix, default_names = error("default")
    flat = default_matrix[:, default_names.index("b")][holes]
    assert np.std(flat) < 0.01


def test_feature_strategies_for_gaps_are_not_offered_when_the_model_copes():
    frame = gappy_problem()[0]
    profile = profile_of(frame)
    complete = profile_of(frame.drop(columns="y").fillna(0).assign(y=frame.y))

    assert "natively" in reason("missing_indicators", profile, "lightgbm")
    assert reason("missing_indicators", profile, "logistic_regression") is None
    assert "no numeric feature is missing" in reason(
        "iterative_imputation", complete, "logistic_regression"
    )


def test_target_transform_helps_a_linear_model_on_a_long_tailed_target():
    from sklearn import model_selection

    rng = np.random.default_rng(9)
    frame = pd.DataFrame({"x": rng.normal(size=1000)})
    frame["y"] = np.exp(1.2 * frame.x + rng.normal(0, 0.2, 1000))
    spec = tasks.ProjectSpec(data=tasks.DataSpec(uri="m", target="y"))
    profile = profiler.DatasetProfiler(spec).profile(
        frame,
        tabular.build_manifest(frame, "y"),
        tasks.TaskType.REGRESSION,
    )

    def mae(name):
        estimator = strategies.get_strategy(name).build(
            profile.feature_plan,
            REGISTRY.get("ridge"),
            tasks.TaskType.REGRESSION,
            0,
        )
        return -model_selection.cross_val_score(
            estimator,
            frame.drop(columns="y"),
            frame["y"],
            cv=4,
            scoring="neg_median_absolute_error",
        ).mean()

    assert mae("target_transform") < mae("default")
    assert (
        reason("target_transform", profile, "ridge", tasks.TaskType.REGRESSION)
        is None
    )
    assert "regression technique" in reason(
        "target_transform", profile, "ridge", BINARY
    )
