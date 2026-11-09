"""Pins the run-rule messages, bounds and the layout of a written charter."""

from functools import partial
import json

import pytest

from scmle.data.charter import contracts
from scmle.data.charter import rules
from scmle.data.charter import sealing
from scmle.data.ledger import budget
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


def test_an_amendment_may_tighten_alpha_or_wealth_but_not_loosen_either():
    old = make_charter(error_budget=budget.ErrorBudget.create(0.05, 0.1))
    for tighter in (
        budget.ErrorBudget.create(0.04, 0.1),
        budget.ErrorBudget.create(0.05, 0.05),
        budget.ErrorBudget.create(0.04, 0.05),
    ):
        sealing.check_amendment(old, make_charter(error_budget=tighter))
    for looser in (
        budget.ErrorBudget.create(0.06, 0.1),
        budget.ErrorBudget.create(0.05, 0.2),
    ):
        with pytest.raises(Exception, match="may only tighten"):
            sealing.check_amendment(old, make_charter(error_budget=looser))


def test_an_amendment_that_changes_another_budget_field_is_refused():
    old = make_charter(error_budget=budget.ErrorBudget.create(0.05, 0.1))
    other = make_charter(
        error_budget=budget.ErrorBudget.create(
            0.05, 0.1, permutations_b=old.error_budget.permutations_b + 100
        )
    )
    with pytest.raises(Exception, match="may only tighten"):
        sealing.check_amendment(old, other)


def test_read_verified_reports_signed_or_not_with_the_recomputed_hash(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("SCMLE_SIGNING_KEY", raising=False)
    charter = make_charter()
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(charter.to_dict()))
    plain = sealing.read_verified(bare)
    assert plain.charter == charter
    assert plain.charter_hash == charter.charter_hash
    assert plain.signed is False
    signed_path = sealing.write_charter(tmp_path / "s.json", charter, "k")
    signed = sealing.read_verified(signed_path, signing_key="k")
    assert signed.charter == charter and signed.signed is True
    assert signed.charter_hash == charter.charter_hash
    with pytest.raises(Exception) as caught:
        sealing.read_verified(bare, require_signature=True)
    assert str(caught.value) == f"charter {bare} is unsigned"


def test_the_format_errors_name_the_file(tmp_path):
    charter = make_charter()
    path = sealing.write_charter(tmp_path / "c.json", charter, "k")
    record = json.loads(path.read_text())
    record["format"] = 99
    path.write_text(json.dumps(record))
    with pytest.raises(Exception) as caught:
        sealing.read_verified(path, signing_key="k")
    assert str(caught.value).startswith(f"charter {path} has format 99; ")
    record["format"] = 2
    record["charter"]["rules"] = {}
    path.write_text(json.dumps(record))
    with pytest.raises(Exception) as caught:
        sealing.read_verified(path, signing_key="k")
    assert str(caught.value) == (
        f"charter {path} declares format 2 but carries 'rules', "
        f"which need format {sealing.RULES_FORMAT}"
    )
    record["format"] = 2
    del record["charter"]["rules"]
    sealing.check_format(path, {"format": 2})
    sealing.check_format(path, {"format": 2, "charter": {"x": 1}})


def test_a_nan_power_is_named():
    assert message(partial(make_charter, power=float("nan"))) == (
        "power must be a finite number"
    )


def test_a_signed_charter_read_without_a_key_is_unsigned_not_an_error(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("SCMLE_SIGNING_KEY", raising=False)
    charter = make_charter()
    path = sealing.write_charter(tmp_path / "s.json", charter, "k")
    seen = sealing.read_verified(path)
    assert seen.charter == charter and seen.signed is False
    assert seen.charter_hash == charter.charter_hash
    with pytest.raises(Exception) as caught:
        sealing.read_verified(path, require_signature=True)
    assert str(caught.value) == (
        f"charter {path} is signed but no signing key is configured"
    )
    with pytest.raises(Exception) as caught:
        sealing.read_verified(path, signing_key="wrong")
    assert str(caught.value) == f"charter signature does not verify for {path}"


def test_a_wrapper_without_a_signature_reads_as_unsigned(tmp_path, monkeypatch):
    monkeypatch.delenv("SCMLE_SIGNING_KEY", raising=False)
    charter = make_charter()
    path = sealing.write_charter(tmp_path / "u.json", charter)
    assert json.loads(path.read_text())["signature"] is None
    seen = sealing.read_verified(path)
    assert seen.charter == charter and seen.signed is False
    assert seen.charter_hash == charter.charter_hash


def test_read_charter_unwraps_a_signed_file_and_reads_a_bare_one(tmp_path):
    charter = make_charter()
    wrapped = sealing.write_charter(tmp_path / "w.json", charter, "k")
    assert contracts.read_charter(wrapped) == charter
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(charter.to_dict()))
    assert contracts.read_charter(bare) == charter


def test_unreadable_and_invalid_charters_say_what_went_wrong(tmp_path):
    missing = tmp_path / "none.json"
    with pytest.raises(Exception) as caught:
        contracts.load_json(missing)
    assert str(caught.value).startswith(f"cannot read charter {missing}: ")
    assert "No such file" in str(caught.value)
    listing = tmp_path / "list.json"
    listing.write_text("[]")
    with pytest.raises(Exception) as caught:
        contracts.load_json(listing)
    assert str(caught.value) == f"charter {listing} is not a JSON object"
    with pytest.raises(Exception) as caught:
        contracts.charter_from_record({"question": "x"})
    assert str(caught.value).startswith("invalid analysis charter: ")
