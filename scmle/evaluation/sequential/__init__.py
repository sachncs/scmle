"""Bounded sequential causal estimation for binary treatment histories.

This module estimates a static sustained-treatment contrast from a balanced
long-format panel.  At each period it fits a denominator propensity model
conditioned on treatment history, baseline covariates, and declared
pre-treatment time-varying covariates.  A numerator model conditioned only on
treatment history stabilizes the cumulative inverse-probability weight.

The contract is deliberately bounded: it does not claim support for dynamic
regimes, intermittent missing visits, continuous treatment, g-computation,
or nuisance-model uncertainty propagation.

The package keeps every name the single module sequential.py had, so
callers and tests are unchanged; its classes keep that module path as
`__module__`, which the public API snapshot records in their signatures.
"""

from scmle.evaluation.envelope import require
from scmle.evaluation.sequential.declaration import Config
from scmle.evaluation.sequential.declaration import COVARIANCE_DIMENSION
from scmle.evaluation.sequential.declaration import Estimand
from scmle.evaluation.sequential.declaration import MATRIX_NDIM
from scmle.evaluation.sequential.declaration import MIN_PERIODS
from scmle.evaluation.sequential.declaration import PROPENSITY_CLIP_CEILING
from scmle.evaluation.sequential.declaration import REGIME_COUNT
from scmle.evaluation.sequential.declaration import REQUIRED_ASSUMPTIONS
from scmle.evaluation.sequential.declaration import SHA256_HEX_LENGTH
from scmle.evaluation.sequential.estimator import Estimator
from scmle.evaluation.sequential.panel import chronologize
from scmle.evaluation.sequential.panel import conform
from scmle.evaluation.sequential.panel import nonrepeating
from scmle.evaluation.sequential.panel import numericize
from scmle.evaluation.sequential.panel import order
from scmle.evaluation.sequential.panel import prepare
from scmle.evaluation.sequential.panel import pretreat
from scmle.evaluation.sequential.panel import situate
from scmle.evaluation.sequential.panel import tabulate
from scmle.evaluation.sequential.report import Diagnostics
from scmle.evaluation.sequential.report import Estimate
from scmle.evaluation.sequential.report import Estimation
from scmle.evaluation.sequential.weighting import average
from scmle.evaluation.sequential.weighting import covary
from scmle.evaluation.sequential.weighting import effective_sample_size
from scmle.evaluation.sequential.weighting import imprint
from scmle.evaluation.sequential.weighting import overlap
from scmle.evaluation.sequential.weighting import periodize
from scmle.evaluation.sequential.weighting import referee
from scmle.evaluation.sequential.weighting import regress

__all__ = [
    "COVARIANCE_DIMENSION",
    "MATRIX_NDIM",
    "MIN_PERIODS",
    "PROPENSITY_CLIP_CEILING",
    "REGIME_COUNT",
    "REQUIRED_ASSUMPTIONS",
    "SHA256_HEX_LENGTH",
    "Config",
    "Diagnostics",
    "Estimand",
    "Estimate",
    "Estimation",
    "Estimator",
    "average",
    "chronologize",
    "conform",
    "covary",
    "effective_sample_size",
    "imprint",
    "nonrepeating",
    "numericize",
    "order",
    "overlap",
    "periodize",
    "prepare",
    "pretreat",
    "referee",
    "regress",
    "require",
    "situate",
    "tabulate",
]
