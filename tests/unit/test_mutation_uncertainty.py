"""The uncertainty block's refusals, word for word."""

import pytest

from scmle.assurance.charter import uncertainty
from scmle.contracts import errors


def test_confirmatory_must_be_a_bool() -> None:
    with pytest.raises(
        errors.InputError, match=r"^confirmatory must be a bool$"
    ):
        uncertainty.Uncertainty(confirmatory=1)


def test_slices_must_be_a_tuple() -> None:
    with pytest.raises(errors.InputError, match=r"^slices must be a tuple$"):
        uncertainty.Uncertainty(slices=["a"])


@pytest.mark.parametrize("slices", [("a", "a"), ("a", " "), ("a", 1)])
def test_slices_are_distinct_non_blank_names(slices) -> None:
    with pytest.raises(
        errors.InputError, match=r"^slices are distinct, non-blank names$"
    ):
        uncertainty.Uncertainty(slices=slices)
