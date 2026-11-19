"""Targeted tests for charter rule validation messages and bounds."""

import pytest

from scmle.assurance.charter import rules


def test_each_rule_error_names_the_offending_field():
    with pytest.raises(ValueError, match=r"ledger must be one of .*e-lond"):
        rules.Procedure(ledger="nope")
    with pytest.raises(
        ValueError, match=r"confirmation must be one of .*bonferroni"
    ):
        rules.Procedure(confirmation="nope")
    with pytest.raises(
        ValueError,
        match=r"failure must be one of \('unconfirmed', 'thresholdout'\)",
    ):
        rules.Procedure(failure="retry")
    with pytest.raises(ValueError, match=r"^stability must be a boolean$"):
        rules.Procedure(stability=1)
    with pytest.raises(
        ValueError, match=r"^agreement must be within \(0, 1\]$"
    ):
        rules.Procedure(agreement=0.0)
    with pytest.raises(
        ValueError, match=r"^exclusion must be within \(0, 1\]$"
    ):
        rules.Procedure(exclusion=1.5)
    with pytest.raises(ValueError, match=r"^band must be positive$"):
        rules.Procedure(band=0.0)


def test_non_finite_values_are_reported_under_their_field_name():
    with pytest.raises(
        ValueError, match=r"^agreement must be a finite number$"
    ):
        rules.Procedure(agreement=float("nan"))
    with pytest.raises(
        ValueError, match=r"^exclusion must be a finite number$"
    ):
        rules.Procedure(exclusion="x")
    with pytest.raises(ValueError, match=r"^band must be a finite number$"):
        rules.Procedure(band=float("inf"))


def test_a_share_of_exactly_one_is_allowed():
    assert rules.Procedure(agreement=1.0, exclusion=1.0).agreement == 1.0
