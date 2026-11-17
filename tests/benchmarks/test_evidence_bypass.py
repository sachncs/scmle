"""Claim C8 evidence: the automated search for ways past the controls."""

import json
import os

import pytest

from benchmarks.evidence import bypass
from scmle.contracts import errors


def record(name: str, outcome: str, gate: str = "leakage.policy") -> dict:
    return {"name": name, "gate": gate, "outcome": outcome}


def document(outcomes: dict[str, str] | None = None) -> dict:
    """A current document; every attempt blocked unless told otherwise."""
    outcomes = outcomes or {}
    return {
        "code": bypass.digest(),
        "records": [
            record(n, outcomes.get(n, "blocked"), a.gate)
            for n, a in bypass.ATTEMPTS.items()
        ],
    }


def lab(tmp_path) -> bypass.Lab:
    return bypass.Lab(tmp_path)


def test_every_attempt_targets_a_registered_or_listed_control():
    known = set(bypass.registered()) | set(bypass.UNREGISTERED)
    assert {a.gate for a in bypass.ATTEMPTS.values()} <= known
    assert len(bypass.ATTEMPTS) == len(set(bypass.ATTEMPTS))


def test_every_control_has_a_plain_case_among_its_attempts():
    gates = {a.gate for a in bypass.ATTEMPTS.values()}
    plain = {a.gate for a in bypass.ATTEMPTS.values() if a.control}
    assert plain == gates


def test_the_verdict_is_never_a_pass_even_with_every_attempt_blocked():
    text, verdict = bypass.summary(document())
    assert bypass.problems(document()) == []
    assert verdict == "not measured"
    assert "not the red-team exercise" in text
    assert "0 bypasses" in text


def test_a_bypass_is_named_and_fails_the_claim():
    # One bypass disproves "cannot be bypassed"; only people can pass it.
    found = document({"leak_xor": "bypass", "bundle_strip_trusted": "override"})
    text, verdict = bypass.summary(found)
    assert verdict == "fail"
    assert "1 bypasses (leak_xor)" in text
    assert "1 through a documented override" in text


def test_missing_stale_or_broken_attempts_are_problems():
    found = document({"leak_copy": "error"})
    found["records"] = [
        r for r in found["records"] if r["name"] != "budget_small"
    ]
    found["records"].append(record("odd", "blocked", "no.such.gate"))
    issues = bypass.problems(found)
    assert "bypass: attempt budget_small not run" in issues
    assert "bypass: attempt leak_copy did not set up" in issues
    assert "bypass: odd targets unknown gate no.such.gate" in issues
    stale = document() | {"code": "0" * 64}
    assert bypass.problems(stale) == ["bypass: measured with other code"]
    assert bypass.summary(stale)[0].endswith("; stale")


def test_an_empty_document_is_not_measured():
    assert bypass.summary({"records": []}) == (
        "no measurement",
        "not measured",
    )


def test_the_allowed_bypasses_are_zero_without_an_acceptance_row():
    assert bypass.limit() == bypass.BYPASSES == 0


def test_classify_records_each_ending(tmp_path):
    sites = bypass.refusal.registry(bypass.GATES)

    def attempt(prepare, override=""):
        return bypass.Attempt(
            "probe", "leakage.policy", "api", "probe", prepare, override
        )

    def through(place):
        del place
        return lambda: {"ok": True}

    def stopped(place):
        del place

        def act():
            raise errors.BlockedError("no")

        return act

    def broken(place):
        del place
        raise bypass.Broken("setup failed")

    endings = [
        bypass.classify(attempt(through), lab(tmp_path), sites),
        bypass.classify(attempt(through, "warn"), lab(tmp_path), sites),
        bypass.classify(attempt(stopped), lab(tmp_path), sites),
        bypass.classify(attempt(broken), lab(tmp_path), sites),
    ]
    assert [e["outcome"] for e in endings] == [
        "bypass",
        "override",
        "blocked",
        "error",
    ]
    assert endings[0]["detail"] == {"ok": True}
    assert endings[2]["stopped"] == "unregistered"
    assert bypass.tally(endings)["bypasses"] == ["probe"]


def test_a_refused_cli_call_raises(tmp_path):
    with pytest.raises(bypass.Refused):
        bypass.cli(["predict", str(tmp_path / "none"), "x.csv"])


def test_the_environment_is_restored(monkeypatch):
    monkeypatch.setenv("SCMLE_ENV", "kept")
    with bypass.environ(SCMLE_ENV=None, SCMLE_SIGNING_KEY="k"):
        assert "SCMLE_ENV" not in os.environ
    assert os.environ["SCMLE_ENV"] == "kept"
    assert "SCMLE_SIGNING_KEY" not in os.environ


def test_each_leak_form_determines_the_target():
    for form in ("copy", "renamed", "labels", "affine", "xor", "text"):
        frame = bypass.leaked(form)
        rest = frame.drop(columns=["a", "b", "c"])
        assert rest.groupby(list(rest.columns.drop("y"))).y.nunique().max() == 1
    with pytest.raises(errors.InputError):
        bypass.leaked("nothing")


def test_a_misspelled_policy_is_refused():
    result = bypass.execute("leak_policy_case", None)
    assert result["outcome"] == "blocked"
    assert result["stopped"] == "ValidationError"


def test_the_stored_search_is_for_the_current_code():
    stored = json.loads(bypass.PATH.read_text(encoding="utf-8"))
    assert stored["code"] == bypass.digest()
    assert bypass.problems() == []
    assert bypass.summary()[1] in ("fail", "not measured")
