"""Module-level metric functions that trial processes can import."""

import numpy

FALSE_NEGATIVE_COST = 10.0
FALSE_POSITIVE_COST = 1.0


def business_cost(true, predicted):
    """Total cost of mistakes: a missed positive costs ten false alarms."""
    true, predicted = numpy.asarray(true), numpy.asarray(predicted)
    missed = ((true == 1) & (predicted == 0)).sum()
    false_alarms = ((true == 0) & (predicted == 1)).sum()
    return float(
        FALSE_NEGATIVE_COST * missed + FALSE_POSITIVE_COST * false_alarms
    ) / len(true)


def mean_absolute_percentage(true, predicted):
    """MAPE guarded against zero targets."""
    true = numpy.asarray(true, dtype=float)
    return float(
        numpy.mean(
            numpy.abs(true - predicted) / numpy.maximum(numpy.abs(true), 1e-9)
        )
    )
