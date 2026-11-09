"""Training-time fairness: reweighing before the model is fitted.

Kamiran and Calders (2012) give each row the weight
`w(g, y) = P(g) P(y) / P(g, y)`, which makes the sensitive group `g` and the
label `y` statistically independent in the weighted training data. Rows of a
group-label pair that is rarer than independence predicts count for more.
The weights are passed to the project as `weight_column`, so every model that
accepts weights learns from the balanced view. Unlike group thresholds, which
adjust decisions afterwards, this changes what the model learns; it needs the
sensitive column only at training time.
"""

import numpy
import pandas


def reweighing_weights(
    groups: pandas.Series, labels: pandas.Series
) -> numpy.ndarray:
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
        raise ValueError("groups and labels must be the same non-zero length")
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
