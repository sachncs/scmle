"""Module-level metric functions that trial processes can import."""

import numpy as np

FALSE_NEGATIVE_COST = 10.0
FALSE_POSITIVE_COST = 1.0


def business_cost(true, predicted):
    """Total cost of mistakes: a missed positive costs ten false alarms."""
    true, predicted = np.asarray(true), np.asarray(predicted)
    missed = ((true == 1) & (predicted == 0)).sum()
    false_alarms = ((true == 0) & (predicted == 1)).sum()
    return float(
        FALSE_NEGATIVE_COST * missed + FALSE_POSITIVE_COST * false_alarms
    ) / len(true)


def mean_absolute_percentage(true, predicted):
    """MAPE guarded against zero targets."""
    true = np.asarray(true, dtype=float)
    return float(
        np.mean(np.abs(true - predicted) / np.maximum(np.abs(true), 1e-9))
    )
