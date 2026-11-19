"""Multiverse stability of an exploration finding (PCS).

A finding is the rank correlation (Spearman's rho) between one feature and the
target on the exploration set `X`. The declared specification grid
(`multiverse`) varies the judgment calls the charter named; this module
implements the ones that act on a feature-target pair:

* `imputation`: `median`, `mean`, `zero` or `complete_case` for missing values
  of the feature;
* `outlier_rule`: `none`, `winsorize` (clip to the 1st and 99th percentile)
  or `iqr_drop` (drop values beyond 1.5 interquartile ranges of the quartiles);
* `inclusion`: `all`, `first_half` or `second_half` of the rows in their
  stored order (time order when the data are sorted by time).

Any other declared choice or option is reported as not applicable and is not
run; it never counts towards stability.

For every specification the finding is re-estimated and perturbed by
bootstrap resamples (whole groups or blocks when `strata` is given), which
gives the PCS-style perturbation interval of Yu and Kumbier (2020, PNAS
117:3920-3929). The survival rule, fixed here and not tuned to results:
a finding is `stable` iff

* the sign of the median effect agrees in at least `s_min` of the
  specifications,
* the median effect is within `band` of the headline (default specification)
  effect, and
* the perturbation interval excludes zero in at least `q_min` of them.

Otherwise it is `fragile`. A joint specification-curve test (Simonsohn,
Simmons and Nelson, 2020, Nature Human Behaviour 4:1208-1214) permutes the
target within strata and recomputes the whole curve, asking whether the curve
is more than chance without treating specifications as independent.

Stability is relative to the declared set. It says nothing about analyses that
were not declared, and a `stable` verdict is not a confirmation: confirmation
comes only from the single test on `C`. The rule's operating characteristics
are in `scmle/certificates/f6_certificates.json`; until a current
certificate passes, a `fragile` verdict is a warning, not a quarantine.
"""

from scmle.assurance.robustness.stability.assessment import adjudication
from scmle.assurance.robustness.stability.assessment import assess
from scmle.assurance.robustness.stability.assessment import fingerprint
from scmle.assurance.robustness.stability.assessment import Stability
from scmle.assurance.robustness.stability.estimation import actionable
from scmle.assurance.robustness.stability.estimation import attribution
from scmle.assurance.robustness.stability.estimation import BOOTSTRAP
from scmle.assurance.robustness.stability.estimation import CONFIDENCE
from scmle.assurance.robustness.stability.estimation import DEFAULT_RULE
from scmle.assurance.robustness.stability.estimation import DEFECT_NULL
from scmle.assurance.robustness.stability.estimation import DEFECT_OUTLIER
from scmle.assurance.robustness.stability.estimation import DEFECT_REVERSAL
from scmle.assurance.robustness.stability.estimation import DEFECT_TYPES
from scmle.assurance.robustness.stability.estimation import DETECTOR_ID
from scmle.assurance.robustness.stability.estimation import estimate
from scmle.assurance.robustness.stability.estimation import FRAGILE
from scmle.assurance.robustness.stability.estimation import IMPLEMENTED
from scmle.assurance.robustness.stability.estimation import IMPUTATION
from scmle.assurance.robustness.stability.estimation import INCLUSION
from scmle.assurance.robustness.stability.estimation import IQR_FENCE
from scmle.assurance.robustness.stability.estimation import MIN_CHOICE_LEVELS
from scmle.assurance.robustness.stability.estimation import MIN_PAIRS
from scmle.assurance.robustness.stability.estimation import MIN_SPECS
from scmle.assurance.robustness.stability.estimation import NOT_EVALUABLE
from scmle.assurance.robustness.stability.estimation import OUTLIER_RULE
from scmle.assurance.robustness.stability.estimation import prepare
from scmle.assurance.robustness.stability.estimation import rankcorr
from scmle.assurance.robustness.stability.estimation import resample
from scmle.assurance.robustness.stability.estimation import SIGNIFICANCE
from scmle.assurance.robustness.stability.estimation import STABLE
from scmle.assurance.robustness.stability.estimation import Thresholds
from scmle.assurance.robustness.stability.estimation import Variant
from scmle.assurance.robustness.stability.estimation import verdict
from scmle.assurance.robustness.stability.estimation import WINSOR_TAILS
from scmle.assurance.robustness.stability.permutation import Curve
from scmle.assurance.robustness.stability.permutation import specify
from scmle.assurance.robustness.stability.permutation import summarize

__all__ = [
    "BOOTSTRAP",
    "CONFIDENCE",
    "DEFAULT_RULE",
    "DEFECT_NULL",
    "DEFECT_OUTLIER",
    "DEFECT_REVERSAL",
    "DEFECT_TYPES",
    "DETECTOR_ID",
    "FRAGILE",
    "IMPLEMENTED",
    "IMPUTATION",
    "INCLUSION",
    "IQR_FENCE",
    "MIN_CHOICE_LEVELS",
    "MIN_PAIRS",
    "MIN_SPECS",
    "NOT_EVALUABLE",
    "OUTLIER_RULE",
    "SIGNIFICANCE",
    "STABLE",
    "WINSOR_TAILS",
    "Curve",
    "Stability",
    "Thresholds",
    "Variant",
    "actionable",
    "adjudication",
    "assess",
    "attribution",
    "estimate",
    "fingerprint",
    "prepare",
    "rankcorr",
    "resample",
    "specify",
    "summarize",
    "verdict",
]
