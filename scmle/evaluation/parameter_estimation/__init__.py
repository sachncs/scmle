"""Reproducible frequentist parameter estimation with uncertainty evidence.

This package answers a different question from model selection: what is the
estimated value of a declared population parameter, and how uncertain is that
estimate under an explicit likelihood and covariance policy?  It provides
transparent frequentist linear and logistic likelihoods plus a bounded,
conjugate Bayesian Gaussian estimator.  More specialised estimators can
implement the same result contract without changing the artifact or review
surface.

The package keeps every name the single module parameter_estimation.py
had, so callers, tests and pickled paths are unchanged. optimize stays
importable here because a test patches
scmle.evaluation.parameter_estimation.optimize.minimize.
"""

# ruff: noqa: F401 - the names below are re-exported for existing callers.

from scipy import optimize

from scmle.evaluation.envelope import created_temporary
from scmle.evaluation.parameter_estimation.audit import crosscheck
from scmle.evaluation.parameter_estimation.audit import (
    reference_hierarchical_random_slope,
)
from scmle.evaluation.parameter_estimation.audit import (
    reference_hierarchical_random_slope_reml,
)
from scmle.evaluation.parameter_estimation.audit import (
    reference_hierarchical_reml,
)
from scmle.evaluation.parameter_estimation.bayesian import bayes
from scmle.evaluation.parameter_estimation.bayesian import conform
from scmle.evaluation.parameter_estimation.bayesian import dichotomous
from scmle.evaluation.parameter_estimation.bayesian import fit_bayesian_linear
from scmle.evaluation.parameter_estimation.bayesian import (
    reference_bayesian_ate,
)
from scmle.evaluation.parameter_estimation.bayesian import (
    reference_bayesian_linear,
)
from scmle.evaluation.parameter_estimation.causal import dr
from scmle.evaluation.parameter_estimation.causal import fit_ipw_ate
from scmle.evaluation.parameter_estimation.causal import (
    fit_propensity_estimator,
)
from scmle.evaluation.parameter_estimation.config import Config
from scmle.evaluation.parameter_estimation.continuous import dosage
from scmle.evaluation.parameter_estimation.continuous import (
    fit_continuous_estimator,
)
from scmle.evaluation.parameter_estimation.continuous import (
    fit_treatment_density,
)
from scmle.evaluation.parameter_estimation.continuous import (
    reference_continuous_ipw_effect,
)
from scmle.evaluation.parameter_estimation.crossfitting import caveats
from scmle.evaluation.parameter_estimation.crossfitting import (
    fit_aipw_nuisance_models,
)
from scmle.evaluation.parameter_estimation.crossfitting import interleave
from scmle.evaluation.parameter_estimation.crossfitting import (
    validate_cross_fitting_support,
)
from scmle.evaluation.parameter_estimation.declaration import BayesianPrior
from scmle.evaluation.parameter_estimation.declaration import CAUSAL_ESTIMATORS
from scmle.evaluation.parameter_estimation.declaration import CausalAssumption
from scmle.evaluation.parameter_estimation.declaration import CLIP_TOLERANCE
from scmle.evaluation.parameter_estimation.declaration import (
    ContinuousTreatmentAssumption,
)
from scmle.evaluation.parameter_estimation.declaration import (
    COVARIANCE_CONDITION_THRESHOLD,
)
from scmle.evaluation.parameter_estimation.declaration import CovarianceKind
from scmle.evaluation.parameter_estimation.declaration import (
    DEFAULT_CONTINUOUS_WEIGHT_CAP,
)
from scmle.evaluation.parameter_estimation.declaration import (
    DEFAULT_PROPENSITY_CLIP,
)
from scmle.evaluation.parameter_estimation.declaration import (
    DEFAULT_REML_VARIANCE_BOUNDS,
)
from scmle.evaluation.parameter_estimation.declaration import Estimand
from scmle.evaluation.parameter_estimation.declaration import EstimatorKind
from scmle.evaluation.parameter_estimation.declaration import (
    HierarchicalAssumption,
)
from scmle.evaluation.parameter_estimation.declaration import HierarchicalMethod
from scmle.evaluation.parameter_estimation.declaration import MATRIX_NDIM
from scmle.evaluation.parameter_estimation.declaration import MIN_CLUSTERS
from scmle.evaluation.parameter_estimation.declaration import (
    MIN_EFFECTIVE_SAMPLE_SIZE,
)
from scmle.evaluation.parameter_estimation.declaration import MIN_FOLDS
from scmle.evaluation.parameter_estimation.declaration import MIN_INFLUENCE_ROWS
from scmle.evaluation.parameter_estimation.declaration import MIN_REPLICATES
from scmle.evaluation.parameter_estimation.declaration import MIN_ROWS_PER_ARM
from scmle.evaluation.parameter_estimation.declaration import (
    MIN_TREATMENT_LEVELS,
)
from scmle.evaluation.parameter_estimation.declaration import (
    MIN_VARIANCE_DEGREES,
)
from scmle.evaluation.parameter_estimation.declaration import MissingPolicy
from scmle.evaluation.parameter_estimation.declaration import ParameterEstimate
from scmle.evaluation.parameter_estimation.declaration import (
    PROPENSITY_CLIP_CEILING,
)
from scmle.evaluation.parameter_estimation.declaration import (
    PROPENSITY_ESTIMATORS,
)
from scmle.evaluation.parameter_estimation.declaration import (
    RANDOM_EFFECT_COLUMNS,
)
from scmle.evaluation.parameter_estimation.declaration import (
    REQUIRED_BAYESIAN_ATE_ASSUMPTIONS,
)
from scmle.evaluation.parameter_estimation.declaration import (
    REQUIRED_CAUSAL_ASSUMPTIONS,
)
from scmle.evaluation.parameter_estimation.declaration import (
    REQUIRED_CONTINUOUS_TREATMENT_ASSUMPTIONS,
)
from scmle.evaluation.parameter_estimation.declaration import (
    REQUIRED_HIERARCHICAL_ASSUMPTIONS,
)
from scmle.evaluation.parameter_estimation.declaration import (
    REQUIRED_HIERARCHICAL_RANDOM_SLOPE_ASSUMPTIONS,
)
from scmle.evaluation.parameter_estimation.declaration import SCALE_FLOOR
from scmle.evaluation.parameter_estimation.declaration import SHA256_HEX_LENGTH
from scmle.evaluation.parameter_estimation.declaration import (
    SMALL_CLUSTER_COUNT,
)
from scmle.evaluation.parameter_estimation.estimate import compute
from scmle.evaluation.parameter_estimation.estimate import estimate_parameters
from scmle.evaluation.parameter_estimation.hierarchical import (
    fit_hierarchical_estimator,
)
from scmle.evaluation.parameter_estimation.hierarchical import (
    fit_hierarchical_linear,
)
from scmle.evaluation.parameter_estimation.hierarchical import (
    fit_hierarchical_random_slope,
)
from scmle.evaluation.parameter_estimation.hierarchical import (
    hierarchical_reference,
)
from scmle.evaluation.parameter_estimation.inputs import absorb
from scmle.evaluation.parameter_estimation.inputs import admit
from scmle.evaluation.parameter_estimation.inputs import clustering
from scmle.evaluation.parameter_estimation.inputs import condition
from scmle.evaluation.parameter_estimation.inputs import prepare
from scmle.evaluation.parameter_estimation.inputs import raw_input_fingerprint
from scmle.evaluation.parameter_estimation.inputs import screen
from scmle.evaluation.parameter_estimation.inputs import validate_clusters
from scmle.evaluation.parameter_estimation.inputs import validate_matrix
from scmle.evaluation.parameter_estimation.linear import corroborate
from scmle.evaluation.parameter_estimation.linear import covariance_matrix
from scmle.evaluation.parameter_estimation.linear import fit_logistic_estimator
from scmle.evaluation.parameter_estimation.linear import logit
from scmle.evaluation.parameter_estimation.linear import loglik
from scmle.evaluation.parameter_estimation.linear import ols
from scmle.evaluation.parameter_estimation.linear import regress
from scmle.evaluation.parameter_estimation.mixed import gls
from scmle.evaluation.parameter_estimation.mixed import (
    hierarchical_random_slope_diagnostics,
)
from scmle.evaluation.parameter_estimation.mixed import (
    hierarchical_random_slope_gls_system,
)
from scmle.evaluation.parameter_estimation.mixed import precisions
from scmle.evaluation.parameter_estimation.mixed import project
from scmle.evaluation.parameter_estimation.references import reference_aipw_ate
from scmle.evaluation.parameter_estimation.references import (
    reference_cross_fitted_aipw_ate,
)
from scmle.evaluation.parameter_estimation.references import reference_ipw_ate
from scmle.evaluation.parameter_estimation.reml import deviance
from scmle.evaluation.parameter_estimation.reml import fit_hierarchical_reml
from scmle.evaluation.parameter_estimation.reml import optimise
from scmle.evaluation.parameter_estimation.reml import (
    random_slope_covariance_from_reml_point,
)
from scmle.evaluation.parameter_estimation.reml import (
    random_slope_reml_profile_objective,
)
from scmle.evaluation.parameter_estimation.reml import (
    validate_reml_optimizer_result,
)
from scmle.evaluation.parameter_estimation.report import Estimation
from scmle.evaluation.parameter_estimation.report import Fit
from scmle.evaluation.parameter_estimation.resampling import (
    bootstrap_sample_digest,
)
from scmle.evaluation.parameter_estimation.resampling import (
    causal_bootstrap_sensitivity,
)
from scmle.evaluation.parameter_estimation.resampling import clone
from scmle.evaluation.parameter_estimation.resampling import (
    draw_causal_bootstrap_sample,
)
from scmle.evaluation.parameter_estimation.resampling import (
    hierarchical_bootstrap_sensitivity,
)
from scmle.evaluation.parameter_estimation.resampling import rerun
from scmle.evaluation.parameter_estimation.sensitivity import reprior
from scmle.evaluation.parameter_estimation.sensitivity import sensitivity_report
from scmle.evaluation.parameter_estimation.sensitivity import transport
from scmle.evaluation.parameter_estimation.weighting import center
from scmle.evaluation.parameter_estimation.weighting import dispersion
from scmle.evaluation.parameter_estimation.weighting import hajek

__all__ = [
    "BayesianPrior",
    "CausalAssumption",
    "Config",
    "CovarianceKind",
    "Estimand",
    "Estimation",
    "EstimatorKind",
    "HierarchicalAssumption",
    "MissingPolicy",
    "ParameterEstimate",
    "estimate_parameters",
]
