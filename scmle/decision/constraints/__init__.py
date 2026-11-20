"""Fairness as explicit constraints on a frozen rule.

Default: no constraint. A `Constraint` in the charter bounds the gap
between the groups of one attribute in the declared criterion for the frozen
decision rule. The constraint is solved on development data (group thresholds
that match the criterion's rate across groups, picking the one that
maximises net benefit); satisfaction is a claim tested once on the final
table: the upper bound of the gap must be at most `max_gap`, a point estimate
below it is not satisfaction. The bound is the largest absolute difference
over group pairs from Newcombe's hybrid score interval (Wilson intervals for
each rate), each pair at level `alpha / pairs`. Rows are taken as
independent. Groups too small for the denominator of the rate are
`not_assessable`, never passing.

When two declared criteria cannot hold together for groups whose base rates
differ (the impossibility results of Kleinberg, Mullainathan and Raghavan,
2016 and Chouldechova, 2017, recalled and not re-read) the set is refused,
naming the pair, using the measured base rates.

The package keeps every name the single module constraints.py had, so callers
and tests are unchanged: `rates` holds the group rates, their intervals and
the feasibility check, `satisfaction` the tests on the final table and `cuts`
the group thresholds fitted on development data.
"""

# The names below are re-exported for existing callers.

from scmle.decision.constraints.cuts import contrast
from scmle.decision.constraints.cuts import equalize
from scmle.decision.constraints.cuts import frontier
from scmle.decision.constraints.cuts import GRID
from scmle.decision.constraints.cuts import Thresholds
from scmle.decision.constraints.cuts import toll
from scmle.decision.constraints.rates import check
from scmle.decision.constraints.rates import count
from scmle.decision.constraints.rates import CRITERION_RATES
from scmle.decision.constraints.rates import INCOMPATIBLE
from scmle.decision.constraints.rates import InfeasibleError
from scmle.decision.constraints.rates import newcombe
from scmle.decision.constraints.rates import prepared
from scmle.decision.constraints.rates import RATE_PARTS
from scmle.decision.constraints.rates import tabulate
from scmle.decision.constraints.rates import wilson
from scmle.decision.constraints.satisfaction import MIN_CLUSTERS
from scmle.decision.constraints.satisfaction import MIN_GROUPS
from scmle.decision.constraints.satisfaction import NOT_ASSESSABLE
from scmle.decision.constraints.satisfaction import NOT_SATISFIED
from scmle.decision.constraints.satisfaction import resample
from scmle.decision.constraints.satisfaction import Satisfaction
from scmle.decision.constraints.satisfaction import SATISFIED
from scmle.decision.constraints.satisfaction import satisfy

__all__ = [
    "CRITERION_RATES",
    "GRID",
    "INCOMPATIBLE",
    "MIN_CLUSTERS",
    "MIN_GROUPS",
    "NOT_ASSESSABLE",
    "NOT_SATISFIED",
    "RATE_PARTS",
    "SATISFIED",
    "InfeasibleError",
    "Satisfaction",
    "Thresholds",
    "check",
    "contrast",
    "count",
    "equalize",
    "frontier",
    "newcombe",
    "prepared",
    "resample",
    "satisfy",
    "tabulate",
    "toll",
    "wilson",
]
