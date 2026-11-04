"""Pins the run-rule messages, bounds and the layout of a written charter."""

from functools import partial
import json

import pytest

from scmle.data.charter import rules
from scmle.data.charter import sealing
from scmle.data.ledger import online_fdr
from tests.unit.test_charter_contracts import make_charter


def message(call):
    with pytest.raises(ValueError) as caught:
        call()
    return str(caught.value)


def test_rule_choices_are_named_in_full():
    assert message(partial(rules.RunRules, ledger_rule="x")) == (
        f"ledger_rule must be one of {online_fdr.RULE_NAMES}"
    )
    allowed = ("bonferroni", "holm", "bh", "by")
    assert message(partial(rules.RunRules, confirmation_rule="x")) == (
        f"confirmation_rule must be one of {allowed}"
    )
    assert message(partial(rules.RunRules, failure_policy="retry")) == (
        "failure_policy must be 'unconfirmed': no other policy is implemented"
    )
    assert message(partial(rules.RunRules, stability_filter=1)) == (
        "stability_filter must be a boolean"
    )


def test_stability_shares_are_in_the_half_open_unit_interval():
    assert rules.RunRules(s_min=1.0, q_min=1.0).s_min == 1.0
    for name in ("s_min", "q_min"):
        for bad in (0.0, -0.1, 1.01):
            assert message(partial(rules.RunRules, **{name: bad})) == (
                f"{name} must be within (0, 1]"
            )
        assert message(partial(rules.RunRules, **{name: float("nan")})) == (
            f"{name} must be a finite number"
        )


def test_the_band_must_be_positive_and_finite():
    assert message(partial(rules.RunRules, band=0.0)) == "band must be positive"
    assert message(partial(rules.RunRules, band=float("inf"))) == (
        "band must be a finite number"
    )
    assert rules.RunRules(band=1e-9).band == 1e-9


def test_a_written_charter_is_sorted_indented_and_in_a_new_folder(tmp_path):
    path = tmp_path / "a" / "b" / "charter.json"
    written = sealing.write_charter(path, make_charter(), signing_key="k")
    assert written == path
    text = path.read_text(encoding="utf-8")
    record = json.loads(text)
    assert text == json.dumps(record, indent=2, sort_keys=True) + "\n"
    assert text.startswith('{\n  "')
    assert "signature" in text


def test_an_amendment_that_loosens_any_one_of_the_three_is_refused():
    old = make_charter()
    same = make_charter()
    sealing.check_amendment(old, same)
    sealing.check_amendment(old, make_charter(k_max=old.k_max - 1))
    for loosened in (
        make_charter(k_max=old.k_max + 1),
        make_charter(question="Another question?"),
        make_charter(seed=99),
    ):
        with pytest.raises(Exception) as caught:
            sealing.check_amendment(old, loosened)
        assert str(caught.value) == (
            "an amendment may only tighten the budget (smaller alpha_confirm,"
            " K_max or exploration wealth); everything else must be unchanged"
        )
    assert sealing.tighter(5, 5) is True
    assert sealing.tighter(5, 6) is False
