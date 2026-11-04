"""Edge cases that mutants of the ledger helpers survived."""

import numpy as np
import pandas as pd
import pytest

from scmle.data.ledger import controls
from scmle.data.ledger import explore


def test_the_quantile_index_never_goes_below_the_smallest_count():
    assert controls.quantile_index(5, q=0.1) == 0
    assert controls.quantile_index(20) == 19
    assert controls.quantile_index(1) == 0


def test_the_percentile_interval_reaches_the_ends_for_extreme_quantiles():
    assert controls.percentile_interval((1, 2), q=0.99) == (2, 2)
    assert controls.percentile_interval((1, 5), q=0.01) == (1, 1)


def test_missing_labels_do_not_make_a_binary_target_multiclass():
    target = pd.Series([0.0, 1.0, np.nan, 1.0, 0.0])
    assert explore.target_kind(target, True) == explore.BINARY
    three = pd.Series([0.0, 1.0, 2.0, np.nan])
    assert explore.target_kind(three, True) == explore.MULTICLASS
    assert explore.target_kind(target, False) == explore.NUMERIC


def test_a_text_regression_target_is_refused_in_full():
    with pytest.raises(ValueError) as caught:
        explore.encode_target(pd.Series(["a", "b"]), False)
    assert str(caught.value) == "a regression target must be numeric"
