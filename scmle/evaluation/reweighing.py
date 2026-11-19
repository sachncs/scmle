"""Training-time fairness: reweighing before the model is fitted.

Kamiran and Calders (2012) give each row the weight
`w(g, y) = P(g) P(y) / P(g, y)`, which makes the sensitive group `g` and the
label `y` statistically independent in the weighted training data. Rows of a
group-label pair that is rarer than independence predicts count for more.
The weights are passed to the project as `weight_column`, so every model that
accepts weights learns from the balanced view. Unlike group thresholds, which
adjust decisions afterwards, this changes what the model learns; it needs the
sensitive column only at training time.

A continuous outcome (regression) has no labels to balance, so it is first
cut into equal-count bins and the bins play the part of the labels: group
and outcome level become independent, which balances each group's share of
low, middle and high outcomes. The bin count is Sturges' rule,
`ceil(log2 n) + 1` (Sturges, 1926, JASA 21:65-66), the class count of a
histogram of a near-normal sample of `n` rows, capped at `MAX_BINS` (deciles)
so each group-by-bin cell keeps enough rows for its share to be estimated.
"""

import math

import numpy
import pandas

from scmle.contracts import errors

MAX_BINS = 10


def bins(rows: int) -> int:
    """The number of outcome bins for `rows` rows (Sturges' rule).

    Args:
        rows: Rows of the outcome; at least 1.

    Returns:
        `ceil(log2 rows) + 1`, at most `MAX_BINS`.
    """
    return min(math.ceil(math.log2(max(rows, 1))) + 1, MAX_BINS)


def discretise(values: pandas.Series) -> pandas.Series:
    """Cuts a continuous outcome into equal-count bins.

    Equal values always share a bin, so heavy ties give fewer bins.

    Args:
        values: The outcome of each row.

    Returns:
        The bin number of each row, 0 for the lowest, on the same index.

    Raises:
        InputError: If an outcome is missing.
    """
    if values.isna().any():
        raise errors.InputError("the outcome has missing values")
    cut = pandas.qcut(
        values, bins(len(values)), labels=False, duplicates="drop"
    )
    return pandas.Series(cut, index=values.index).astype(int)


def weigh(groups: pandas.Series, labels: pandas.Series) -> numpy.ndarray:
    """Returns row weights that remove the association of group and label.

    Args:
        groups: Sensitive group of each row.
        labels: Class label of each row.

    Returns:
        One weight per row, averaging 1.

    Raises:
        ValueError: If the inputs differ in length or are empty.
    """
    if len(groups) != len(labels) or len(groups) == 0:
        raise errors.InputError(
            "groups and labels must be the same non-zero length"
        )
    frame = pandas.DataFrame(
        {"group": groups.to_numpy(), "label": labels.to_numpy()}
    )
    total = float(len(frame))
    group_share = frame.group.map(frame.group.value_counts() / total)
    label_share = frame.label.map(frame.label.value_counts() / total)
    joint = frame.groupby(["group", "label"]).group.transform("size") / total
    weights = (group_share * label_share / joint).to_numpy(dtype=float)
    normalised: numpy.ndarray = weights / weights.mean()
    return normalised
