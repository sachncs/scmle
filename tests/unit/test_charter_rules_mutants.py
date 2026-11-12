"""Targeted tests for charter rule validation messages and bounds."""

import pytest

from scmle.data.charter import rules


def test_each_rule_error_names_the_offending_field():
    with pytest.raises(
        ValueError, match=r"ledger_rule must be one of .*e-lond"
    ):
        rules.Procedure(ledger_rule="nope")
    with pytest.raises(
        ValueError, match=r"confirmation_rule must be one of .*bonferroni"
    ):
        rules.Procedure(confirmation_rule="nope")
    with pytest.raises(
        ValueError, match="failure_policy must be 'unconfirmed': no other"
    ):
        rules.Procedure(failure_policy="retry")
    with pytest.raises(
        ValueError, match=r"^stability_filter must be a boolean$"
    ):
        rules.Procedure(stability_filter=1)
    with pytest.raises(ValueError, match=r"^s_min must be within \(0, 1\]$"):
        rules.Procedure(s_min=0.0)
    with pytest.raises(ValueError, match=r"^q_min must be within \(0, 1\]$"):
        rules.Procedure(q_min=1.5)
    with pytest.raises(ValueError, match=r"^band must be positive$"):
        rules.Procedure(band=0.0)


def test_non_finite_values_are_reported_under_their_field_name():
    with pytest.raises(ValueError, match=r"^s_min must be a finite number$"):
        rules.Procedure(s_min=float("nan"))
    with pytest.raises(ValueError, match=r"^q_min must be a finite number$"):
        rules.Procedure(q_min="x")
    with pytest.raises(ValueError, match=r"^band must be a finite number$"):
        rules.Procedure(band=float("inf"))


def test_a_share_of_exactly_one_is_allowed():
    assert rules.Procedure(s_min=1.0, q_min=1.0).s_min == 1.0
