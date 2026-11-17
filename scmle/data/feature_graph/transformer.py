"""Row-wise graph nodes as a scikit-learn transformer."""

from typing import Any
from typing import Self

import pandas
from sklearn import base

from scmle.contracts import tasks
from scmle.data.feature_graph.registry import lookup


class Transformer(base.TransformerMixin, base.BaseEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
    """Row-wise graph nodes as a fitted scikit-learn transformer.

    Statistics for `clip`, `bin` and `standardize` are learned in `fit`
    (the training fold) and stored, so the transform is identical at
    inference. Derived columns are appended to the input frame.

    Attributes:
        nodes: The row-wise nodes, in dependency order.
        states_: Learned state per node name.
    """

    __module__ = "scmle.data.feature_graph"

    def __init__(self, nodes: tuple[tasks.Node, ...] = ()) -> None:
        """Initializes the transformer.

        Args:
            nodes: Row-wise nodes in dependency order.
        """
        self.nodes = nodes

    def fit(self, features: Any, target: Any = None) -> Self:  # noqa: ARG002 -- scikit-learn fit signature
        """Learns operation state by evaluating the graph on training data.

        Args:
            features: The training rows, convertible to a data frame.
            target: Unused.

        Returns:
            The fitted transformer.
        """
        frame = pandas.DataFrame(features).copy()
        self.states_: dict[str, Any] = {}
        for node in self.nodes:
            op = lookup(node.op)
            params = op.resolved(node)
            inputs = [frame[c] for c in node.inputs]
            self.states_[node.name] = op.fit(inputs, params)
            frame[node.name] = op.apply(inputs, params, self.states_[node.name])
        return self

    def transform(self, features: Any) -> pandas.DataFrame:
        """Appends the derived columns using the learned state.

        Args:
            features: The rows, convertible to a data frame.

        Returns:
            A copy of the rows with one column per node.
        """
        frame = pandas.DataFrame(features).copy()
        for node in self.nodes:
            op = lookup(node.op)
            params = op.resolved(node)
            inputs = [frame[c] for c in node.inputs]
            frame[node.name] = op.apply(inputs, params, self.states_[node.name])
        return frame
