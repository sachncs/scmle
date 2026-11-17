"""Feature synthesis through the as-of view.

A fixed grammar, Deep Feature Synthesis style: ten primitives over the rows of
a related table in declared windows, parent-attribute lookups, and depth-2
nesting (an aggregate of per-row aggregates of a grandchild table). The
grammar, the windows and the cap `B_feat` are declared; the family size is
fixed before exploration and every kept feature is registered in the ledger
as a `synthesized` hypothesis, so the multiplicity the ledger corrects for
is the multiplicity that exists.

This module never touches a table. It receives `View` objects and asks
them for the rows visible at each target row's cutoff; `tests` hold it to
that by checking its imports. Label-free filters (constant, mostly missing,
duplicate of an earlier feature) run first; they use no labels, and nothing
here reads the target.
"""

from scmle.assurance.ingest.synthesize.evaluation import aggregate
from scmle.assurance.ingest.synthesize.evaluation import evaluate
from scmle.assurance.ingest.synthesize.evaluation import fold
from scmle.assurance.ingest.synthesize.evaluation import MIN_SLOPE_ROWS
from scmle.assurance.ingest.synthesize.evaluation import outcomes
from scmle.assurance.ingest.synthesize.evaluation import slope
from scmle.assurance.ingest.synthesize.grammar import caption
from scmle.assurance.ingest.synthesize.grammar import CHILD
from scmle.assurance.ingest.synthesize.grammar import COUNT
from scmle.assurance.ingest.synthesize.grammar import Feature
from scmle.assurance.ingest.synthesize.grammar import LAST
from scmle.assurance.ingest.synthesize.grammar import MAX
from scmle.assurance.ingest.synthesize.grammar import MEAN
from scmle.assurance.ingest.synthesize.grammar import MIN
from scmle.assurance.ingest.synthesize.grammar import Nest
from scmle.assurance.ingest.synthesize.grammar import nested_features
from scmle.assurance.ingest.synthesize.grammar import NESTED_INNER
from scmle.assurance.ingest.synthesize.grammar import NESTED_OUTER
from scmle.assurance.ingest.synthesize.grammar import NUMERIC_ONLY
from scmle.assurance.ingest.synthesize.grammar import NUNIQUE
from scmle.assurance.ingest.synthesize.grammar import PARENT
from scmle.assurance.ingest.synthesize.grammar import PRIMITIVES
from scmle.assurance.ingest.synthesize.grammar import propose
from scmle.assurance.ingest.synthesize.grammar import roll
from scmle.assurance.ingest.synthesize.grammar import sift
from scmle.assurance.ingest.synthesize.grammar import SLOPE
from scmle.assurance.ingest.synthesize.grammar import Source
from scmle.assurance.ingest.synthesize.grammar import STD
from scmle.assurance.ingest.synthesize.grammar import SUM
from scmle.assurance.ingest.synthesize.grammar import TIME_SINCE_LAST
from scmle.assurance.ingest.synthesize.grammar import VALUE_FREE
from scmle.assurance.ingest.synthesize.grammar import WINDOWS_DAYS
from scmle.assurance.ingest.synthesize.selection import describe
from scmle.assurance.ingest.synthesize.selection import enlist
from scmle.assurance.ingest.synthesize.selection import FAMILY
from scmle.assurance.ingest.synthesize.selection import fingerprint
from scmle.assurance.ingest.synthesize.selection import lineage
from scmle.assurance.ingest.synthesize.selection import MultiplicityError
from scmle.assurance.ingest.synthesize.selection import plurality
from scmle.assurance.ingest.synthesize.selection import Selection
from scmle.assurance.ingest.synthesize.selection import STATISTIC
from scmle.assurance.ingest.synthesize.selection import synthesize

__all__ = [
    "CHILD",
    "COUNT",
    "FAMILY",
    "LAST",
    "MAX",
    "MEAN",
    "MIN",
    "MIN_SLOPE_ROWS",
    "NESTED_INNER",
    "NESTED_OUTER",
    "NUMERIC_ONLY",
    "NUNIQUE",
    "PARENT",
    "PRIMITIVES",
    "SLOPE",
    "STATISTIC",
    "STD",
    "SUM",
    "TIME_SINCE_LAST",
    "VALUE_FREE",
    "WINDOWS_DAYS",
    "Feature",
    "MultiplicityError",
    "Nest",
    "Selection",
    "Source",
    "aggregate",
    "caption",
    "describe",
    "enlist",
    "evaluate",
    "fingerprint",
    "fold",
    "lineage",
    "nested_features",
    "outcomes",
    "plurality",
    "propose",
    "roll",
    "sift",
    "slope",
    "synthesize",
]
