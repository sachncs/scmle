"""Calibration tests, and the claim each structured task can support.

A probability is calibrated when, among the rows given probability `p`, a
share `p` turn out positive. Each task reduces its predictions to such
probabilities (or, for uplift, to effects) and tests them on held-out rows:

* ordinal (`labels`): the cumulative probability `P(level <= k)` at every
  threshold `k`;
* multilabel (`labels`): the probability of every label;
* survival (`hazard`): the event-free probability at each horizon, against
  the Kaplan-Meier estimate within groups of similar predicted risk;
* ranking (`relevance`): the NDCG gain of the scores' order over a random
  order, and the probability of relevance that an increasing logistic
  (Platt) map, learnt on other queries, gives each score;
* uplift (`treatment`): the predicted effect against the treated-minus-
  control difference within groups of similar predicted effect, beside the
  Qini coefficient and the area under the uplift curve with bootstrap
  intervals.

Every test is the grouped chi-square of `verdict`. The predictions must come
from a model that never saw the rows; in-sample predictions make every test
too lenient. A claim holds when no test rejects at `LEVEL` after a
Bonferroni split over its tests, and every interval with a null value lies
above it. A test that cannot run (too few rows, events or arms) never
supports a claim.
"""

from scmle.evaluation.calibration.hazard import horizon
from scmle.evaluation.calibration.hazard import meier
from scmle.evaluation.calibration.hazard import QUARTILES
from scmle.evaluation.calibration.hazard import quartiles
from scmle.evaluation.calibration.hazard import survival
from scmle.evaluation.calibration.labels import multilabel
from scmle.evaluation.calibration.labels import ordinal
from scmle.evaluation.calibration.relevance import chance
from scmle.evaluation.calibration.relevance import crossfit
from scmle.evaluation.calibration.relevance import FOLDS
from scmle.evaluation.calibration.relevance import gain
from scmle.evaluation.calibration.relevance import Platt
from scmle.evaluation.calibration.relevance import ranking
from scmle.evaluation.calibration.treatment import area
from scmle.evaluation.calibration.treatment import ARM
from scmle.evaluation.calibration.treatment import bootstrap
from scmle.evaluation.calibration.treatment import effects
from scmle.evaluation.calibration.treatment import spread
from scmle.evaluation.calibration.treatment import uplift
from scmle.evaluation.calibration.verdict import bernoulli
from scmle.evaluation.calibration.verdict import Check
from scmle.evaluation.calibration.verdict import Claim
from scmle.evaluation.calibration.verdict import CONFIDENCE
from scmle.evaluation.calibration.verdict import deviation
from scmle.evaluation.calibration.verdict import DRAWS
from scmle.evaluation.calibration.verdict import EXPECTED
from scmle.evaluation.calibration.verdict import Group
from scmle.evaluation.calibration.verdict import GROUPS
from scmle.evaluation.calibration.verdict import Interval
from scmle.evaluation.calibration.verdict import LEVEL
from scmle.evaluation.calibration.verdict import partition
from scmle.evaluation.calibration.verdict import probable
from scmle.evaluation.calibration.verdict import STATEMENTS
from scmle.evaluation.calibration.verdict import vector

__all__ = [
    "ARM",
    "CONFIDENCE",
    "DRAWS",
    "EXPECTED",
    "FOLDS",
    "GROUPS",
    "LEVEL",
    "QUARTILES",
    "STATEMENTS",
    "Check",
    "Claim",
    "Group",
    "Interval",
    "Platt",
    "area",
    "bernoulli",
    "bootstrap",
    "chance",
    "crossfit",
    "deviation",
    "effects",
    "gain",
    "horizon",
    "meier",
    "multilabel",
    "ordinal",
    "partition",
    "probable",
    "quartiles",
    "ranking",
    "spread",
    "survival",
    "uplift",
    "vector",
]
