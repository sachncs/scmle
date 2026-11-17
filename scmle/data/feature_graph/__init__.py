"""The feature-transformation graph: operations, fitting and execution.

Two kinds of operation exist:

* **Row-wise** operations (`log1p`, `ratio`, `bin`, `datepart`, ...) run
  inside the candidate pipeline. Operations that need statistics (`clip`,
  `bin`, `standardize`) learn them in `fit`, i.e. from the training fold only.
* **Ordered** operations (`lag`, `diff`, `rolling_mean`, ...) need row order.
  They are computed per group in time order, using only the current and past
  rows (never future ones, and never the target), at ingestion and again at
  inference. A saved model carries the recent history it needs (the last few
  rows of every entity), so even a single new record gets filled-in lag and
  rolling features; history rows from the batch itself take precedence.
"""

from scmle.data.feature_graph.history import Temporal
from scmle.data.feature_graph.operations import Absolute
from scmle.data.feature_graph.operations import Bin
from scmle.data.feature_graph.operations import Calendar
from scmle.data.feature_graph.operations import Clip
from scmle.data.feature_graph.operations import Difference
from scmle.data.feature_graph.operations import EPOCH
from scmle.data.feature_graph.operations import Inputs
from scmle.data.feature_graph.operations import Log1p
from scmle.data.feature_graph.operations import MAX_ARITY
from scmle.data.feature_graph.operations import Missingness
from scmle.data.feature_graph.operations import Operation
from scmle.data.feature_graph.operations import Params
from scmle.data.feature_graph.operations import Product
from scmle.data.feature_graph.operations import Ratio
from scmle.data.feature_graph.operations import Sqrt
from scmle.data.feature_graph.operations import Square
from scmle.data.feature_graph.operations import Standardize
from scmle.data.feature_graph.operations import Sum
from scmle.data.feature_graph.registry import lookup
from scmle.data.feature_graph.registry import OPS
from scmle.data.feature_graph.registry import roots
from scmle.data.feature_graph.registry import segregate
from scmle.data.feature_graph.registry import validate_graph
from scmle.data.feature_graph.sequences import Diff
from scmle.data.feature_graph.sequences import ExpandingMean
from scmle.data.feature_graph.sequences import Lag
from scmle.data.feature_graph.sequences import Rolling
from scmle.data.feature_graph.sequences import Sequential
from scmle.data.feature_graph.sequences import Volatility
from scmle.data.feature_graph.transformer import Transformer

__all__ = [
    "EPOCH",
    "MAX_ARITY",
    "OPS",
    "Absolute",
    "Bin",
    "Calendar",
    "Clip",
    "Diff",
    "Difference",
    "ExpandingMean",
    "Inputs",
    "Lag",
    "Log1p",
    "Missingness",
    "Operation",
    "Params",
    "Product",
    "Ratio",
    "Rolling",
    "Sequential",
    "Sqrt",
    "Square",
    "Standardize",
    "Sum",
    "Temporal",
    "Transformer",
    "Volatility",
    "lookup",
    "roots",
    "segregate",
    "validate_graph",
]
