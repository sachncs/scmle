"""Runs a chartered causal analysis end to end.

Order: select (a function of the charter), diagnose the design and fit frozen
nuisances on `X`, estimate on `C` once, run the pre-registered falsification
battery and the replication, assess sensitivity, and tier the claim. The
function receives `X` and `C` as frames; opening `C` through the custody vault
exactly once is the caller's job (the confirmation function), and nothing in
this module reads `C` before the estimate step or refits anything on it.

Outputs are five JSON-ready documents, named in `DOCUMENTS`. A claim is
`not_identified` (and no estimate is printed) when no strategy is eligible,
when the overlap gate fails with no trimming declared, when trimming leaves
too few rows, or when the regression-discontinuity density check rejects on
`X`.

Single-treated-unit synthetic control has nothing to hold out: it runs on `X`
and its tier is capped at `attested`.

The package keeps every name the single module run.py had, so callers and
tests are unchanged: tests patch `tier.decide` through this package and
replace entries of `ADAPTERS`, the one registry `identify` reads.
"""

# ruff: noqa: F401 - the names below are re-exported for existing callers.

from scmle.causal.identification import tier
from scmle.causal.identification.run.adjustment import Backdoor
from scmle.causal.identification.run.adjustment import Experiment
from scmle.causal.identification.run.analysis import ADAPTERS
from scmle.causal.identification.run.analysis import alphas
from scmle.causal.identification.run.analysis import BATTERY_GATES
from scmle.causal.identification.run.analysis import caveats
from scmle.causal.identification.run.analysis import DESIGN_GATES
from scmle.causal.identification.run.analysis import diagnose
from scmle.causal.identification.run.analysis import fragility
from scmle.causal.identification.run.analysis import identify
from scmle.causal.identification.run.analysis import INTERVAL_GATES
from scmle.causal.identification.run.analysis import NOT_ESTIMATED
from scmle.causal.identification.run.analysis import SENSITIVITY_GATES
from scmle.causal.identification.run.documents import arbitrate
from scmle.causal.identification.run.documents import assemble
from scmle.causal.identification.run.documents import digest
from scmle.causal.identification.run.documents import DOCUMENTS
from scmle.causal.identification.run.documents import LIMITS
from scmle.causal.identification.run.documents import Outcome
from scmle.causal.identification.run.documents import prerequisites
from scmle.causal.identification.run.documents import refuse
from scmle.causal.identification.run.estimator import Est
from scmle.causal.identification.run.estimator import Estimator
from scmle.causal.identification.run.estimator import waldify
from scmle.causal.identification.run.exogeneity import Discontinuity
from scmle.causal.identification.run.exogeneity import Instrument
from scmle.causal.identification.run.panel import Differences
from scmle.causal.identification.run.panel import Synth
from scmle.causal.identification.run.strata import certify
from scmle.causal.identification.run.strata import estimate
from scmle.causal.identification.run.strata import MIN_STRATUM_ROWS
from scmle.causal.identification.run.strata import replication
from scmle.causal.identification.run.strata import stratify

__all__ = [
    "ADAPTERS",
    "BATTERY_GATES",
    "DESIGN_GATES",
    "DOCUMENTS",
    "INTERVAL_GATES",
    "LIMITS",
    "MIN_STRATUM_ROWS",
    "NOT_ESTIMATED",
    "SENSITIVITY_GATES",
    "Backdoor",
    "Differences",
    "Discontinuity",
    "Est",
    "Estimator",
    "Experiment",
    "Instrument",
    "Outcome",
    "Synth",
    "alphas",
    "arbitrate",
    "assemble",
    "caveats",
    "certify",
    "diagnose",
    "digest",
    "estimate",
    "fragility",
    "identify",
    "prerequisites",
    "refuse",
    "replication",
    "stratify",
    "waldify",
]
