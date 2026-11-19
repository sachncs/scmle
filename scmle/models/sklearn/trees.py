"""Classification trees that split the same on every CPU architecture.

scikit-learn's classification criteria rank a candidate split by
`-w_right * i_right - w_left * i_left` (weights `w`, impurities `i`). Two
candidates whose children hold the same class counts, swapped, are an exact
tie, which the split search leaves to the first candidate. The arm64 builds
fuse that expression into one multiply-add, whose single rounding is not
symmetric in the two terms, so the tie becomes a one-ulp win for either
candidate and the tree differs from the one an x86-64 build grows (claim
C6, the `interaction` task).

The squared error of the class indicators (one 0/1 column per class) is the
Gini impurity: `w * sum_k p_k (1 - p_k) = w - sum_k c_k^2 / w` for class
counts `c_k`. Its split proxy, `sum_left^2 / w_left + sum_right^2 / w_right`,
is symmetric, and with integer row weights (bootstrap counts) every term
before the division is an exact integer, so ties are decided by correctly
rounded divisions and one addition, the same on every platform. The classes
here grow those regression trees and store them as stock classification
trees, whose leaves hold the class shares; prediction, persistence and
export see ordinary scikit-learn trees.
"""

import copy
from typing import Any

import numpy
from sklearn import ensemble
from sklearn import tree
from sklearn.utils import class_weight

from scmle.contracts import errors

# Options whose meaning the regression twin cannot reproduce: out-of-bag
# scores would be the indicators' R^2, warm starts would mix the two kinds of
# tree, and per-bootstrap class weights are drawn inside the twin's fit.
REFUSED = ("oob_score", "warm_start")


def indicators(labels: Any) -> tuple[numpy.ndarray, numpy.ndarray]:
    """The classes and one 0/1 column per class.

    Args:
        labels: One label per row.

    Returns:
        The sorted classes and the row-by-class indicator matrix.

    Raises:
        InputError: If the labels are not one column.
    """
    values = numpy.asarray(labels)
    if values.ndim != 1:
        raise errors.InputError(
            f"labels must be one column, not shape {values.shape}"
        )
    classes, codes = numpy.unique(values, return_inverse=True)
    return classes, numpy.eye(len(classes))[codes]


def reshape(grown: Any, count: int) -> Any:
    """A regression tree on `count` indicators as a classification tree.

    Args:
        grown: The fitted low-level tree (`tree_`) of the regression twin.
        count: Number of classes.

    Returns:
        A low-level tree with one output of `count` classes, the same
        nodes, the class shares as leaf values and the Gini impurity (the
        regression impurity is averaged over the `count` outputs).
    """
    state = grown.__getstate__()
    state["values"] = numpy.ascontiguousarray(
        state["values"].transpose(0, 2, 1)
    )
    # A node record has seven padding bytes after `missing_go_to_left`,
    # which pickling writes out. `copy` (and `zeros_like`) leave them as
    # whatever the allocator returned, so the bundle bytes would differ
    # between processes (claim C6). `zeros` clears every byte; the fields
    # are then copied one by one, leaving the padding zero.
    source = state["nodes"]
    nodes = numpy.zeros(source.shape, dtype=source.dtype)
    for name in source.dtype.names:
        nodes[name] = source[name]
    nodes["impurity"] *= count
    state["nodes"] = nodes
    shaped = type(grown)(
        grown.n_features, numpy.array([count], dtype=numpy.intp), 1
    )
    shaped.__setstate__(state)
    return shaped


def adopt(target: Any, grown: Any, classes: numpy.ndarray) -> None:
    """Moves the fitted state of a regression tree onto a classifier.

    Args:
        target: The classifier (a tree or a forest) to fill.
        grown: The fitted regression twin.
        classes: The sorted classes.
    """
    params = grown.get_params()
    vars(target).update(
        {key: value for key, value in vars(grown).items() if key not in params}
    )
    target.classes_ = classes
    target.n_classes_ = len(classes)
    target.n_outputs_ = 1


def classify(
    kind: type, grown: Any, classes: numpy.ndarray, own: dict[str, Any]
) -> Any:
    """One fitted regression member as a stock classification tree.

    Args:
        kind: The stock classification tree class.
        grown: A fitted member of the regression twin.
        classes: The sorted classes.
        own: The forest's parameters, whose criterion and impurity
            thresholds the member reports.

    Returns:
        The member as a fitted `kind`.
    """
    params = {
        key: value
        for key, value in grown.get_params().items()
        if key in kind().get_params()
    }
    for key in ("criterion", "min_impurity_decrease", "ccp_alpha"):
        params[key] = own[key]
    member = kind(**params)
    adopt(member, grown, classes)
    member.tree_ = reshape(grown.tree_, len(classes))
    return member


class Symmetric:
    """Mixin: grows a classifier as a regression twin on class indicators.

    Attributes:
        twin: The regression estimator class grown in its place.
        stock: The scikit-learn classifier class this one stands for.
        member: The stock classification tree class of one forest member,
            or None for a single tree.
    """

    twin: type
    stock: type
    member: type | None = None

    def fit(
        self, features: Any, labels: Any, sample_weight: Any = None
    ) -> "Symmetric":
        """Grows the regression twin and keeps it as classification trees.

        Args:
            features: The feature matrix.
            labels: One label per row.
            sample_weight: Optional row weights.

        Returns:
            The fitted classifier.

        Raises:
            InputError: For an option in `REFUSED` or `balanced_subsample`
                class weights.
        """
        own: dict[str, Any] = self.get_params()  # type: ignore[attr-defined]  # scikit-learn base
        refused = [name for name in REFUSED if own.get(name)]
        if own["class_weight"] == "balanced_subsample":
            refused.append("class_weight='balanced_subsample'")
        if refused:
            raise errors.InputError(
                f"{type(self).__name__} cannot reproduce {refused}"
            )
        classes, targets = indicators(labels)
        weights = sample_weight
        if own["class_weight"] is not None:
            weights = class_weight.compute_sample_weight(
                own["class_weight"], numpy.asarray(labels)
            )
            if sample_weight is not None:
                weights = weights * numpy.asarray(sample_weight)
        params = {
            key: value
            for key, value in own.items()
            if key in self.twin().get_params()
        }
        # The twin's impurity is the Gini impurity over the class count.
        params["criterion"] = "squared_error"
        params["min_impurity_decrease"] = own["min_impurity_decrease"] / len(
            classes
        )
        params["ccp_alpha"] = own["ccp_alpha"] / len(classes)
        grown = self.twin(**params).fit(
            features, targets, sample_weight=weights
        )
        adopt(self, grown, classes)
        if self.member is None:
            self.tree_ = reshape(grown.tree_, len(classes))
        else:
            self.estimator_ = self.member(criterion=own["criterion"])
            self.estimators_ = [
                classify(self.member, part, classes, own)
                for part in grown.estimators_
            ]
        return self


class Tree(Symmetric, tree.DecisionTreeClassifier):  # type: ignore[misc]  # scikit-learn ships no type information
    """A decision tree classifier grown by `Symmetric`."""

    twin = tree.DecisionTreeRegressor
    stock = tree.DecisionTreeClassifier


class Forest(Symmetric, ensemble.RandomForestClassifier):  # type: ignore[misc]  # scikit-learn ships no type information
    """A random forest classifier grown by `Symmetric`."""

    twin = ensemble.RandomForestRegressor
    stock = ensemble.RandomForestClassifier
    member = tree.DecisionTreeClassifier


class Extra(Symmetric, ensemble.ExtraTreesClassifier):  # type: ignore[misc]  # scikit-learn ships no type information
    """An extremely randomized trees classifier grown by `Symmetric`."""

    twin = ensemble.ExtraTreesRegressor
    stock = ensemble.ExtraTreesClassifier
    member = tree.ExtraTreeClassifier


def stock(estimator: Any) -> Any:
    """The stock scikit-learn classifier holding a symmetric fit.

    Converters that look estimators up by their exact type (ONNX export)
    see the scikit-learn class; prediction is unchanged.

    Args:
        estimator: Any estimator.

    Returns:
        A shallow copy of a `Symmetric` estimator as its `stock` class, or
        `estimator` itself otherwise.
    """
    if not isinstance(estimator, Symmetric):
        return estimator
    plain = copy.copy(estimator)
    plain.__class__ = estimator.stock
    return plain
