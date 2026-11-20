"""Parameter-estimation declarations, policy rules and configuration."""

import dataclasses
import math

import pytest

from scmle.contracts import errors
from scmle.contracts import records
from scmle.evaluation.envelope import rename
from scmle.evaluation.parameter_estimation import config
from scmle.evaluation.parameter_estimation import declaration
from scmle.evaluation.parameter_estimation import rules

CAUSAL = ("consistency", "conditional_exchangeability", "positivity")
CONTINUOUS = (
    "consistency",
    "conditional_exchangeability",
    "continuous_treatment_positivity",
    "linear_dose_response",
    "normal_conditional_treatment_density",
)
HIERARCHICAL = (
    "random_intercept",
    "cluster_independence",
    "random_effect_exogeneity",
    "constant_residual_variance",
)


def prior(size=2):
    return declaration.Prior(
        (0.0,) * size,
        tuple(
            tuple(1.0 if i == j else 0.0 for j in range(size))
            for i in range(size)
        ),
    )


def estimand(**changes):
    values = {
        "name": "n",
        "outcome": "y",
        "population": "p",
        "likelihood": "gaussian",
        "features": ("a", "b"),
        "assumptions": ("x",),
    }
    values.update(changes)
    return declaration.Estimand(**values)


# declaration


def test_the_declaration_constants_are_fixed():
    assert declaration.DEFAULT_REML_VARIANCE_BOUNDS == (1e-12, 1e12)
    assert declaration.COVARIANCE_CONDITION_THRESHOLD == 1e12
    assert {
        "bayesian_ate",
        "ipw_ate",
        "aipw_ate",
        "continuous_ipw_ate",
    } == declaration.CAUSAL_ESTIMATORS
    assert {"ipw_ate", "aipw_ate"} == declaration.PROPENSITY_ESTIMATORS
    assert declaration.SCALE_FLOOR == 1e-12
    assert declaration.CLIP_TOLERANCE == 1e-12
    assert declaration.PROPENSITY_CLIP_CEILING == 0.5
    assert declaration.DEFAULT_PROPENSITY_CLIP == 0.01
    assert declaration.DEFAULT_CONTINUOUS_WEIGHT_CAP == 20.0
    assert declaration.MATRIX_NDIM == 2
    assert declaration.MIN_ROWS_PER_ARM == 2
    assert declaration.MIN_TREATMENT_LEVELS == 4
    assert declaration.MIN_EFFECTIVE_SAMPLE_SIZE == 2.0
    assert declaration.MIN_VARIANCE_DEGREES == 2.0
    assert declaration.MIN_CLUSTERS == 2
    assert declaration.SMALL_CLUSTER_COUNT == 30
    assert declaration.MIN_FOLDS == 2
    assert declaration.MIN_INFLUENCE_ROWS == 2
    assert declaration.MIN_REPLICATES == 2
    assert declaration.RANDOM_EFFECT_COLUMNS == 2
    assert set(CAUSAL) == declaration.REQUIRED_CAUSAL_ASSUMPTIONS
    assert {
        *CAUSAL,
        "linearity",
    } == declaration.REQUIRED_BAYESIAN_ATE_ASSUMPTIONS
    assert (
        set(CONTINUOUS) == declaration.REQUIRED_CONTINUOUS_TREATMENT_ASSUMPTIONS
    )
    assert set(HIERARCHICAL) == declaration.REQUIRED_HIERARCHICAL_ASSUMPTIONS
    assert {
        *HIERARCHICAL,
        "random_slope",
    } == declaration.REQUIRED_HIERARCHICAL_RANDOM_SLOPE_ASSUMPTIONS
    assert len(declaration.INTERCEPT_LIMITS) == 4


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"name": " "}, "estimand name must not be empty"),
        ({"outcome": ""}, "estimand outcome must not be empty"),
        ({"population": ""}, "estimand population must not be empty"),
        ({"likelihood": " "}, "estimand likelihood must not be empty"),
        ({"features": ()}, "estimand needs at least one feature"),
        ({"features": ("a", "a")}, "feature names must be unique"),
        ({"assumptions": ()}, "estimand needs explicit assumptions"),
    ],
)
def test_an_invalid_estimand_is_rejected(changes, message):
    with pytest.raises(errors.InputError, match=message):
        estimand(**changes)


def test_an_estimand_writes_features():
    record = estimand(description="d").to_dict()
    assert record["features"] == ("a", "b")
    assert record["description"] == "d"
    assert "feature_names" not in record


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (((), ()), "needs a coefficient mean"),
        ((((0.0,),), ((1.0,),)), "mean must be one-dimensional"),
        (((0.0, 0.0), ((1.0,),)), "precision must be square"),
        (((math.nan,), ((1.0,),)), "mean must be finite"),
        (((0.0,), ((math.inf,),)), "precision must be finite"),
        (((0.0, 0.0), ((1.0, 0.5), (0.0, 1.0))), "precision must be symmetric"),
        (((0.0, 0.0), ((1.0, 2.0), (2.0, 1.0))), "positive definite"),
        (((0.0,), ((1.0,),), 0.0), "shape must be positive"),
        (((0.0,), ((1.0,),), 2.0, math.inf), "scale must be positive"),
    ],
)
def test_an_invalid_prior_is_rejected(arguments, message):
    with pytest.raises(errors.InputError, match=message):
        declaration.Prior(*arguments)


def test_a_prior_defaults_and_serialises():
    found = prior(1)
    assert (found.shape, found.scale) == (2.0, 1.0)
    assert found.to_dict() == {
        "mean": (0.0,),
        "precision": ((1.0,),),
        "shape": 2.0,
        "scale": 1.0,
    }


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((" ", 1.0, 0.1, (0.0, 2.0), 0.5), "parameter name must not be empty"),
        (("b", math.nan, 0.1, (0.0, 2.0), 0.5), "estimate must be finite"),
        (
            ("b", 1.0, 0.1, (0.0, math.inf), 0.5),
            "interval_upper must be finite",
        ),
        (
            ("b", 1.0, -0.1, (0.0, 2.0), 0.5),
            "standard_error must not be negative",
        ),
        (("b", 1.0, 0.1, (0.0, 2.0), 1.01), "p_value must lie between 0 and 1"),
        (
            ("b", 1.0, 0.1, (0.0, 2.0), -0.01),
            "p_value must lie between 0 and 1",
        ),
        (
            ("b", 1.0, 0.1, (2.0, 0.0), 0.5),
            "parameter interval must be ordered",
        ),
    ],
)
def test_an_invalid_parameter_estimate_is_rejected(arguments, message):
    with pytest.raises(errors.InputError, match=message):
        declaration.Estimate(*arguments)


def test_a_parameter_estimate_writes_se_and_accepts_bounds():
    found = declaration.Estimate("b", 1.0, 0.0, (1.0, 1.0), 1.0)
    assert found.to_dict() == {
        "name": "b",
        "estimate": 1.0,
        "se": 0.0,
        "interval": (1.0, 1.0),
        "p_value": 1.0,
    }
    assert declaration.Estimate("b", 1.0, 0.1, (0.0, 2.0), 0.0).p_value == 0.0


# rules and config


def test_rules_hold_no_state():
    assert rules.Rules.__slots__ == ()
    assert issubclass(config.Config, rules.Rules)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"estimator": "probit"}, "estimator must be linear, logistic"),
        ({"covariance": "hc1"}, "covariance must be iid, hc3, or clustered"),
        ({"missing": "impute"}, "missing must be reject or complete_case"),
        ({"confidence": 0.0}, "confidence_level must lie strictly"),
        ({"confidence": 1.0}, "confidence_level must lie strictly"),
        (
            {"covariance": "clustered"},
            "clustered covariance needs cluster_name",
        ),
        (
            {"replicates": -1},
            "bootstrap_replicates must not be negative",
        ),
        ({"iterations": 0}, "max_iterations must be positive"),
        ({"tolerance": 0.0}, "estimation tolerances must be positive"),
        (
            {"reference": 0.0},
            "estimation tolerances must be positive",
        ),
        ({"clip": 0.5}, "propensity_clip must lie strictly"),
        ({"clip": 0.0}, "propensity_clip must lie strictly"),
        (
            {"cap": 1.0},
            "continuous_weight_cap must be greater",
        ),
        (
            {"cap": math.nan},
            "continuous_weight_cap must be greater",
        ),
        ({"folds": 0}, "cross_fitting_folds must be a positive"),
        (
            {"folds": True},
            "cross_fitting_folds must be a positive",
        ),
        (
            {"hierarchy": "ml"},
            "hierarchical_method must be moment_gls",
        ),
        (
            {"slope": -1},
            "random_slope_position must be a non-negative",
        ),
        (
            {"slope": False},
            "random_slope_position must be a non-negative",
        ),
        ({"bounds": (1.0,)}, "reml_variance_bounds must contain"),
        (
            {"bounds": (0.0, 1.0)},
            "reml_variance_bounds must contain",
        ),
        (
            {"bounds": (2.0, 1.0)},
            "reml_variance_bounds must contain",
        ),
        (
            {"bounds": (1.0, math.inf)},
            "reml_variance_bounds must contain",
        ),
        ({"treatment": 0}, "treatment_position is supported only"),
        ({"clip": 0.02}, "propensity_clip is supported only"),
        (
            {"causal_assumptions": CAUSAL},
            "causal assumptions are supported only",
        ),
        (
            {"continuous_treatment_assumptions": CONTINUOUS},
            "continuous treatment assumptions",
        ),
        (
            {"cap": 5.0},
            "continuous_weight_cap is supported only",
        ),
        (
            {"folds": 2},
            "cross_fitting_folds is supported only by aipw_ate",
        ),
        (
            {"hierarchical_assumptions": HIERARCHICAL},
            "hierarchical assumptions are supported",
        ),
        ({"prior": prior()}, "a Bayesian prior is supported only"),
        (
            {"slope": 0},
            "random_slope_position is supported only",
        ),
        (
            {"hierarchy": "reml"},
            "hierarchical_method is supported only",
        ),
        (
            {"bounds": (1e-6, 1e6)},
            "reml_variance_bounds is supported only",
        ),
    ],
)
def test_an_invalid_linear_policy_is_rejected(changes, message):
    with pytest.raises(errors.InputError, match=message):
        config.Config(**{"estimator": "linear", **changes})


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            {"estimator": "bayesian_linear"},
            "bayesian_linear estimation requires an explicit prior",
        ),
        (
            {"estimator": "bayesian_linear", "prior": prior()},
            "supports only iid covariance",
        ),
        (
            {
                "estimator": "bayesian_linear",
                "prior": prior(),
                "covariance": "iid",
                "replicates": 5,
            },
            "bootstrap sensitivity is unsupported",
        ),
        (
            {
                "estimator": "bayesian_ate",
                "prior": prior(),
                "covariance": "iid",
            },
            "bayesian_ate requires treatment_position",
        ),
        (
            {
                "estimator": "bayesian_ate",
                "prior": prior(),
                "covariance": "iid",
                "treatment": 0,
                "causal_assumptions": CAUSAL,
            },
            "requires consistency, conditional_exchangeability, positivity, and linearity",
        ),
        (
            {
                "estimator": "bayesian_ate",
                "prior": prior(),
                "covariance": "clustered",
                "cluster": "c",
            },
            "supports only iid covariance",
        ),
        (
            {"estimator": "ipw_ate", "prior": prior(), "covariance": "iid"},
            "a Bayesian prior is not valid for ipw_ate",
        ),
        (
            {"estimator": "ipw_ate"},
            "ipw_ate supports only iid or clustered covariance; hc3",
        ),
        (
            {"estimator": "aipw_ate", "covariance": "iid"},
            "aipw_ate requires treatment_position",
        ),
        (
            {
                "estimator": "ipw_ate",
                "covariance": "iid",
                "treatment": 0,
                "causal_assumptions": (*CAUSAL, "positivity"),
            },
            "ipw_ate requires consistency",
        ),
        (
            {
                "estimator": "continuous_ipw_ate",
                "covariance": "iid",
                "treatment": 0,
                "causal_assumptions": CAUSAL,
            },
            "continuous_ipw_ate requires continuous_treatment_assumptions",
        ),
        (
            {
                "estimator": "continuous_ipw_ate",
                "covariance": "iid",
                "treatment": 0,
                "continuous_treatment_assumptions": CONTINUOUS[:4],
            },
            "continuous_ipw_ate requires consistency",
        ),
        (
            {"estimator": "hierarchical_linear", "prior": prior()},
            "a Bayesian prior is not valid for hierarchical_linear",
        ),
        (
            {"estimator": "hierarchical_linear"},
            "supplies its own covariance model",
        ),
        (
            {"estimator": "hierarchical_linear", "covariance": "iid"},
            "hierarchical_linear requires cluster_name",
        ),
        (
            {
                "estimator": "hierarchical_linear",
                "covariance": "iid",
                "cluster": "c",
                "hierarchical_assumptions": HIERARCHICAL,
                "slope": 0,
            },
            "requires all assumptions",
        ),
    ],
)
def test_an_invalid_estimator_policy_is_rejected(values, message):
    with pytest.raises(errors.InputError, match=message):
        config.Config(**values)


def test_a_bayesian_ate_rejects_clustered_data_with_iid_covariance():
    with pytest.raises(
        errors.InputError, match="bayesian_ate does not support clustered data"
    ):
        config.Config(
            estimator="bayesian_ate",
            prior=prior(),
            covariance="iid",
            treatment=0,
            causal_assumptions=(*CAUSAL, "linearity"),
            cluster="c",
        )


def test_a_linear_policy_writes_only_its_own_fields():
    record = config.Config(estimator="linear").to_dict()
    assert record == {
        "estimator": "linear",
        "covariance": "hc3",
        "confidence": 0.95,
        "missing": "reject",
        "cluster": None,
        "replicates": 0,
        "seed": 0,
        "iterations": 100,
        "tolerance": 1e-08,
        "reference": 1e-06,
    }


def test_a_causal_policy_writes_its_assumptions_and_clip():
    record = config.Config(
        estimator="aipw_ate",
        covariance="iid",
        treatment=1,
        causal_assumptions=CAUSAL,
        folds=3,
        clip=0.05,
    ).to_dict()
    assert record["treatment"] == 1
    assert record["clip"] == 0.05
    assert record["folds"] == 3
    assert record["assumptions"] == {"causal": CAUSAL}
    assert "cap" not in record and "hierarchy" not in record


def test_a_hierarchical_policy_writes_its_method_and_bounds():
    policy = config.Config(
        estimator="hierarchical_linear",
        covariance="iid",
        cluster="site",
        hierarchical_assumptions=(*HIERARCHICAL, "random_slope"),
        slope=0,
        hierarchy="reml",
    )
    record = policy.to_dict()
    assert record["hierarchy"] == "reml"
    assert record["bounds"] == (1e-12, 1e12)
    assert record["slope"] == 0
    assert record["assumptions"] == {
        "hierarchical": (*HIERARCHICAL, "random_slope")
    }
    assert "treatment" not in record


def test_a_bayesian_policy_writes_its_prior():
    record = config.Config(
        estimator="bayesian_linear", prior=prior(), covariance="iid"
    ).to_dict()
    assert record["prior"] == prior().to_dict()


def test_a_continuous_policy_writes_its_cap():
    record = config.Config(
        estimator="continuous_ipw_ate",
        covariance="iid",
        treatment=0,
        continuous_treatment_assumptions=CONTINUOUS,
        cap=8.0,
    ).to_dict()
    assert record["cap"] == 8.0
    assert record["assumptions"] == {"continuous": CONTINUOUS}
    assert "clip" not in record


def test_a_gather_drops_an_empty_assumption_group():
    assert config.gather({"x": 1}) == {"x": 1}
    assert config.gather({"causal_assumptions": (), "x": 1}) == {
        "x": 1,
        "assumptions": {"causal": ()},
    }
    assert config.gather({"causal_assumptions": ("a",)}) == {
        "assumptions": {"causal": ("a",)}
    }


def test_a_config_round_trips_through_its_migration():
    policy = config.Config(
        estimator="ipw_ate",
        covariance="iid",
        treatment=0,
        causal_assumptions=CAUSAL,
    )
    fields = records.downgrade(config.Config.migrations, policy.to_dict())
    assert config.Config(**rename(config.Config, fields)) == policy
    assert dataclasses.replace(policy, seed=3).seed == 3
