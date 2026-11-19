"""Edge cases that mutants of the ledger helpers survived."""

import numpy
import pandas
import pytest

from scmle.assurance.ledger import controls
from scmle.assurance.ledger import explore


def test_the_quantile_index_never_goes_below_the_smallest_count():
    assert controls.rank(5, q=0.1) == 0
    assert controls.rank(20) == 19
    assert controls.rank(1) == 0


def test_the_percentile_interval_reaches_the_ends_for_extreme_quantiles():
    assert controls.bracket((1, 2), q=0.99) == (2, 2)
    assert controls.bracket((1, 5), q=0.01) == (1, 1)


def test_missing_labels_do_not_make_a_binary_target_multiclass():
    target = pandas.Series([0.0, 1.0, numpy.nan, 1.0, 0.0])
    assert explore.classify(target, True) == explore.BINARY
    three = pandas.Series([0.0, 1.0, 2.0, numpy.nan])
    assert explore.classify(three, True) == explore.MULTICLASS
    assert explore.classify(target, False) == explore.NUMERIC


def test_a_text_regression_target_is_refused_in_full():
    with pytest.raises(ValueError) as caught:
        explore.vectorize(pandas.Series(["a", "b"]), False)
    assert str(caught.value) == "a regression target must be numeric"
