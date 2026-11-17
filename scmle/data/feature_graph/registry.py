"""The operation registry and graph validation."""

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.feature_graph.operations import Absolute
from scmle.data.feature_graph.operations import Bin
from scmle.data.feature_graph.operations import Calendar
from scmle.data.feature_graph.operations import Clip
from scmle.data.feature_graph.operations import Difference
from scmle.data.feature_graph.operations import Log1p
from scmle.data.feature_graph.operations import Missingness
from scmle.data.feature_graph.operations import Operation
from scmle.data.feature_graph.operations import Product
from scmle.data.feature_graph.operations import Ratio
from scmle.data.feature_graph.operations import Sqrt
from scmle.data.feature_graph.operations import Square
from scmle.data.feature_graph.operations import Standardize
from scmle.data.feature_graph.operations import Sum
from scmle.data.feature_graph.sequences import Diff
from scmle.data.feature_graph.sequences import ExpandingMean
from scmle.data.feature_graph.sequences import Lag
from scmle.data.feature_graph.sequences import Rolling
from scmle.data.feature_graph.sequences import Volatility

OPS: dict[str, Operation] = {
    op.name: op
    for op in (
        Log1p(),
        Sqrt(),
        Square(),
        Absolute(),
        Missingness(),
        Sum(),
        Difference(),
        Product(),
        Ratio(),
        Clip(),
        Standardize(),
        Bin(),
        Calendar(),
        Lag(),
        Diff(),
        Rolling(),
        Volatility(),
        ExpandingMean(),
    )
}


def lookup(name: str) -> Operation:
    """Looks up a registered operation.

    Args:
        name: The operation name, as used in `Node.op`.

    Returns:
        The registered operation.

    Raises:
        ConfigError: If the name is unknown.
    """
    if name not in OPS:
        raise errors.ConfigError(
            f"unknown feature operation '{name}'; available: {sorted(OPS)}"
        )
    return OPS[name]


def segregate(
    spec: tasks.Derivation,
) -> tuple[list[tasks.Node], list[tasks.Node]]:
    """Separates ordered (series) nodes from row-wise nodes.

    Args:
        spec: The graph.

    Returns:
        The ordered nodes and the row-wise nodes, each in graph order. An
        unknown operation name propagates `lookup`'s `ConfigError`.
    """
    ordered = [n for n in spec.nodes if lookup(n.op).ordered]
    rowwise = [n for n in spec.nodes if not lookup(n.op).ordered]
    return ordered, rowwise


def validate_graph(
    spec: tasks.Derivation,
    available: set[str],
    data: tasks.Dataset,
) -> None:
    """Validates a graph against the dataset's columns and the data spec.

    Args:
        spec: The graph.
        available: Raw column names in the dataset.
        data: Column roles (target, group, time).

    Raises:
        ConfigError: For unknown operations, bad arity or parameters, missing
            inputs, use of the target, a node name that clashes with a raw
            column, ordered operations without a time column, or ordered
            nodes consuming row-wise ones.
    """
    known = set(available)
    ordered_names: set[str] = set()
    rowwise_names: set[str] = set()
    for node in spec.nodes:
        op = lookup(node.op)
        op.validate(node)
        if node.name in available:
            raise errors.ConfigError(
                f"node '{node.name}' clashes with an existing column"
            )
        if data.target in node.inputs:
            raise errors.ConfigError(
                f"node '{node.name}' uses the target as an input (leakage)"
            )
        missing = [c for c in node.inputs if c not in known]
        if missing:
            raise errors.ConfigError(
                f"node '{node.name}' uses unknown columns {missing}"
            )
        if op.ordered:
            if not data.time_column:
                raise errors.ConfigError(
                    f"node '{node.name}': {node.op} needs data.time_column"
                )
            if set(node.inputs) & rowwise_names:
                raise errors.ConfigError(
                    f"ordered node '{node.name}' cannot consume row-wise nodes"
                )
            ordered_names.add(node.name)
        else:
            rowwise_names.add(node.name)
        known.add(node.name)


def roots(nodes: list[tasks.Node], derived: set[str]) -> list[str]:
    """Lists the raw columns that `nodes` read.

    Args:
        nodes: The nodes to inspect.
        derived: Names of columns produced by the graph, which are skipped.

    Returns:
        The distinct input columns not in `derived`, in first-use order.
    """
    needed: list[str] = []
    for node in nodes:
        for column in node.inputs:
            if column not in derived and column not in needed:
                needed.append(column)
    return needed
