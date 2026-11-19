"""Claim C8 evidence: the automated search for ways past the controls."""

import json
import os
import pickle

import pytest

from benchmarks.evidence import bypass
from scmle import envelope
from scmle.assurance.custody import journal
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


def test_a_broken_attack_is_an_error_not_a_refusal(tmp_path):
    def prepare(place):
        del place

        def act():
            raise bypass.Broken("never reached the control")

        return act

    attempt = bypass.Attempt("probe", "leakage.policy", "api", "p", prepare)
    found = bypass.classify(attempt, lab(tmp_path), {})
    assert found["outcome"] == "error"
    assert found["stopped"] == "Broken"
    assert found["message"] == "attack: never reached the control"


def test_an_unregistered_control_is_named_by_its_raising_site(tmp_path):
    path = tmp_path / "forged.envelope"
    bypass.forge(path, {"product": "Search"}, b"bytes", None)

    def unsigned(place):
        del place

        def act():
            with bypass.environ(SCMLE_SIGNING_KEY=bypass.KEY):
                envelope.read(path)
            return {}

        return act

    def other(place):
        del place

        def act():
            raise errors.ArtifactError("raised elsewhere")

        return act

    gate = "envelope.authenticate"
    named = bypass.classify(
        bypass.Attempt("probe", gate, "api", "p", unsigned), lab(tmp_path), {}
    )
    unnamed = bypass.classify(
        bypass.Attempt("probe", gate, "api", "p", other), lab(tmp_path), {}
    )
    assert (named["outcome"], named["stopped"]) == ("blocked", gate)
    assert named["site"] == "scmle.envelope:read"
    assert (unnamed["outcome"], unnamed["stopped"]) == (
        "blocked",
        "ArtifactError",
    )


def test_a_cli_refusal_is_named_by_its_reported_label(tmp_path):
    prefix = "2026-11-19 08:10:26,258 ERROR scmle.cli [run= request=]: "
    lines = {
        "blocked: unresolved leakage risk": ("blocked", "cli.blocked"),
        "artifact error: bundle signature is invalid": (
            "artifact",
            "cli.artifact",
        ),
        "config error: unknown config sections": ("config", "cli.config"),
        "internal error (a defect; please report it): x": ("", "Refused"),
    }
    for text, (label, control) in lines.items():
        refused = bypass.Refused(prefix + text, 2)
        assert (refused.label, refused.control) == (label, control), text
    assert bypass.Refused("usage: scmle run", 64).control == "Refused"

    def act():
        raise bypass.Refused(prefix + "blocked: no", 2)

    attempt = bypass.Attempt(
        "probe", "leakage.policy", "cli", "p", lambda p: act
    )
    found = bypass.classify(attempt, lab(tmp_path), {})
    assert (found["stopped"], found["status"]) == ("cli.blocked", 2)


def test_an_attempt_stops_at_its_gate_or_a_listed_control():
    def prepare(place):
        return place

    api = bypass.Attempt(
        "probe",
        "budget.minimum_run",
        "api",
        "p",
        prepare,
        also=("ValidationError",),
    )
    cli = bypass.Attempt("probe", "leakage.policy", "cli", "p", prepare)
    artifact = bypass.Attempt(
        "probe", "bundle.authenticate", "cli", "p", prepare
    )
    assert api.stops == {"budget.minimum_run", "ValidationError"}
    assert cli.stops == {"leakage.policy", "cli.blocked"}
    assert artifact.stops == {"bundle.authenticate", "cli.artifact"}


def test_a_refused_cli_call_raises(tmp_path):
    with pytest.raises(bypass.Refused) as caught:
        bypass.cli(["predict", str(tmp_path / "none"), "x.csv"])
    assert caught.value.status != 0


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


def test_witnessed_judges_an_attack_by_its_effect():
    def refuse():
        raise errors.BlockedError("no")

    def finish():
        return {"call": "done"}

    def effect():
        return {"payload": "ran"}

    def none():
        return None

    assert bypass.witnessed(refuse, effect)() == {
        "payload": "ran",
        "refused_after": "no",
    }
    assert bypass.witnessed(finish, effect)() == {
        "payload": "ran",
        "call": "done",
    }
    with pytest.raises(errors.BlockedError):
        bypass.witnessed(refuse, none)()
    with pytest.raises(bypass.Held):
        bypass.witnessed(finish, none)()


def test_a_payload_runs_when_unpickled(tmp_path):
    place = lab(tmp_path)
    data = pickle.dumps(bypass.Payload(place.marker))
    assert bypass.ran(place)() is None
    pickle.loads(data)
    assert bypass.ran(place)() == {"payload": "ran"}


def test_a_forged_envelope_states_its_digest_and_metadata(tmp_path):
    path = tmp_path / "forged.envelope"
    bypass.forge(path, {"product": "Search"}, b"bytes", None)
    assert bypass.metadata(path) == {"product": "Search"}
    assert envelope.read(path, trusted=True) == (
        {"product": "Search"},
        b"bytes",
    )


def test_observed_keeps_the_refusals_of_the_watched_files(tmp_path):
    watched = tmp_path / "watched.envelope"
    other = tmp_path / "other.envelope"
    for path in (watched, other):
        bypass.forge(path, {"product": "Search"}, b"bytes", None)
    with (
        bypass.environ(SCMLE_SIGNING_KEY=bypass.KEY),
        bypass.observed([watched]) as seen,
    ):
        for path in (watched, other):
            with pytest.raises(errors.ArtifactError):
                envelope.read(path)
    assert len(seen) == 1
    assert str(watched) in str(seen[0])
    assert envelope.read(other, trusted=True)[1] == b"bytes"


def test_the_journal_chain_is_recomputed_without_a_key(tmp_path):
    run = tmp_path / "runs" / "run_1"
    run.mkdir(parents=True)
    record = journal.Journal.locate(run)
    record.start("cafe")
    record.witness()
    bypass.chain(run, lambda e: e["kind"] != "opened")
    assert [e["kind"] for e in record.read()] == ["started"]
    bypass.chain(run, extra=bypass.restart(run))
    assert [e["kind"] for e in record.read()] == ["started", "started"]
    assert record.read()[1]["config"] == "cafe"


def test_a_token_from_another_set_is_refused():
    for name in ("vault_cross", "token_cross"):
        prepare = bypass.ATTEMPTS[name].prepare
        with pytest.raises(errors.LeakageError):
            prepare(None)()
