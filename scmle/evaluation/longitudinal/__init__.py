"""Bounded period-specific causal estimation for longitudinal data.

The estimator in this module treats each time period as a cross-sectional
binary-treatment study.  It does not estimate a sequential g-formula,
marginal structural model, or other longitudinal intervention estimand.
Longitudinal structure is used for identity, ordering, and influence-function
covariance only; the propensity model is refit independently in each period.

The package keeps every name the single module longitudinal.py had, so
callers and tests are unchanged; its classes keep that module path as
`__module__`, which the public API snapshot records in their signatures.
"""

from scmle.evaluation.envelope import require
from scmle.evaluation.longitudinal.declaration import Config
from scmle.evaluation.longitudinal.declaration import Estimand
from scmle.evaluation.longitudinal.declaration import MATRIX_NDIM
from scmle.evaluation.longitudinal.declaration import Period
from scmle.evaluation.longitudinal.declaration import PROPENSITY_CLIP_CEILING
from scmle.evaluation.longitudinal.declaration import REQUIRED_ASSUMPTIONS
from scmle.evaluation.longitudinal.declaration import SHA256_HEX_LENGTH
from scmle.evaluation.longitudinal.estimator import Estimator
from scmle.evaluation.longitudinal.panel import chronologize
from scmle.evaluation.longitudinal.panel import conform
from scmle.evaluation.longitudinal.panel import monotone
from scmle.evaluation.longitudinal.panel import numericize
from scmle.evaluation.longitudinal.panel import prepare
from scmle.evaluation.longitudinal.panel import pretreat
from scmle.evaluation.longitudinal.panel import situate
from scmle.evaluation.longitudinal.report import Estimation
from scmle.evaluation.longitudinal.weighting import effective_sample_size
from scmle.evaluation.longitudinal.weighting import imprint
from scmle.evaluation.longitudinal.weighting import periodise
from scmle.evaluation.longitudinal.weighting import periodize
from scmle.evaluation.longitudinal.weighting import referee
from scmle.evaluation.longitudinal.weighting import regress
from scmle.evaluation.longitudinal.weighting import sandwich

__all__ = [
    "MATRIX_NDIM",
    "PROPENSITY_CLIP_CEILING",
    "REQUIRED_ASSUMPTIONS",
    "SHA256_HEX_LENGTH",
    "Config",
    "Estimand",
    "Estimation",
    "Estimator",
    "Period",
    "chronologize",
    "conform",
    "effective_sample_size",
    "imprint",
    "monotone",
    "numericize",
    "periodise",
    "periodize",
    "prepare",
    "pretreat",
    "referee",
    "regress",
    "require",
    "sandwich",
    "situate",
]
