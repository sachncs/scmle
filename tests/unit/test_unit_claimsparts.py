"""The checks of a claims audit, module by module.

The ledger is stood in for by a plain namespace with the attributes each
check reads (`head`, `header`, `confirmation`, `remaining`, `claims`,
`control`), so each branch is driven directly.
"""

import json
import types

import pytest

from scmle.artifacts.claims_audit import attestation
from scmle.artifacts.claims_audit import custody
from scmle.artifacts.claims_audit import files
from scmle.artifacts.claims_audit import registration
from scmle.artifacts.claims_audit import seal
from scmle.artifacts.claims_audit import verdict
from scmle.assurance import coverage_run
from scmle.assurance.ledger import confirm
import scmle.assurance.ledger.ledger
from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.decision import reports


def write(path, body):
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def ledger(**fields):
    defaults = {
        "head": "abcdef0123456789",
        "header": {"rule": {"rule": "holm"}, "charter_hash": "c"},
        "confirmation": None,
        "remaining": None,
        "claims": {},
        "control": None,
    }
    return types.SimpleNamespace(**{**defaults, **fields})


def test_the_file_names_of_a_charter_run():
    assert files.CHARTER_FILE == "analysis_charter.json"
    assert files.CUSTODY_FILE == "custody_log.jsonl"
    assert files.CONTROL_FILE == "negative_control.json"
    assert files.BASELINE_FILE == "baseline_report.json"
    assert files.CAUSAL_FILE == "causal_claims.json"


def test_read_optional_returns_only_json_objects(tmp_path):
    assert files.read(tmp_path / "none.json") is None
    assert files.read(write(tmp_path / "a.json", [1])) is None
    assert files.read(write(tmp_path / "b.json", {"k": 1})) == {"k": 1}
    (tmp_path / "c.json").write_text("{", encoding="utf-8")
    assert files.read(tmp_path / "c.json") is None


def test_migrate_moves_an_old_claim_row_to_the_current_keys():
    row = files.migrate(qualification.Claim, {"claim_id": "h1", "tier": "x"})
    assert row["identifier"] == "h1"
    assert "claim_id" not in row
    check = files.migrate(qualification.Check, {"check_id": "h1:C"})
    assert check["identifier"] == "h1:C"


def test_recall_refuses_a_missing_or_foreign_claims_file(tmp_path):
    with pytest.raises(errors.ArtifactError, match="cannot read"):
        files.recall(tmp_path)
    write(tmp_path / confirm.CLAIMS_FILE, {"other": 1})
    with pytest.raises(errors.ArtifactError, match="not a claims document"):
        files.recall(tmp_path)
    write(tmp_path / confirm.CLAIMS_FILE, [1])
    with pytest.raises(errors.ArtifactError, match="not a claims document"):
        files.recall(tmp_path)


def test_rehydrate_turns_a_ledger_error_into_an_artifact_error(tmp_path):
    with pytest.raises(errors.ArtifactError):
        files.rehydrate(tmp_path)


def test_openings_fail_a_second_open_and_an_unopened_confirmation():
    name = "n"
    twice = custody.openings(name, {"outcome": "x"}, 2)
    assert twice == verdict.Finding(name, verdict.FAIL, "C was opened 2 times")
    confirmed = {"outcome": confirm.CONFIRMED}
    assert custody.openings(name, confirmed, 0) == verdict.Finding(
        name, verdict.FAIL, "outcome is confirmed but C has no open event"
    )
    assert custody.openings(name, confirmed, 1) is None
    assert custody.openings(name, {"outcome": "no_survivors"}, 0) == (
        verdict.Finding(
            name, verdict.PASS, "C not opened (outcome no_survivors)"
        )
    )
    assert custody.openings(name, {"outcome": "no_survivors"}, 1) is None


def frozen(detail):
    return [
        {"event": "selection_frozen", "detail": detail},
        {"event": "open"},
    ]


def test_the_frozen_selection_must_come_once_before_the_open():
    document = {"digests": {"survivors": "s"}}
    entries = frozen({"survivors_sha256": "s"})
    names = [e["event"] for e in entries]
    assert custody.collate(entries, names, document) is None
    reversed_entries = entries[::-1]
    assert custody.collate(
        reversed_entries, [e["event"] for e in reversed_entries], document
    ) == ("no single selection_frozen before the open")
    doubled = [entries[0], *entries]
    assert custody.collate(
        doubled, [e["event"] for e in doubled], document
    ) == ("no single selection_frozen before the open")


def test_the_frozen_hashes_must_match_the_claims():
    entries = frozen({"survivors_sha256": "s", "family_sha256": "f"})
    names = ["selection_frozen", "open"]
    assert custody.collate(entries, names, {"digests": {"survivors": "t"}}) == (
        "frozen survivor hash differs from the claims"
    )
    document = {"digests": {"survivors": "s"}, "second": {"family_sha256": "g"}}
    assert custody.collate(entries, names, document) == (
        "frozen family hash differs from the claims"
    )
    document["second"]["family_sha256"] = "f"
    assert custody.collate(entries, names, document) is None


def test_custody_is_skipped_without_a_log_and_fails_on_a_broken_chain(
    tmp_path,
):
    found = custody.custodial(tmp_path, {}, None)
    assert found.status == verdict.SKIPPED
    assert found.detail == "custody_log.jsonl is not in the run"
    write(tmp_path / files.CUSTODY_FILE, {"event": "open", "set": "C"})
    assert custody.custodial(tmp_path, {}, None).status == verdict.FAIL
    assert custody.audit(tmp_path, None).status == verdict.FAIL
    write(tmp_path / coverage_run.CLAIM_FILE, {"k": 1})
    assert custody.corroborate(tmp_path, None).status == verdict.FAIL


def test_a_custody_log_that_is_not_json_fails_the_check(tmp_path):
    (tmp_path / files.CUSTODY_FILE).write_text("not json\n", encoding="utf-8")
    found = custody.custodial(tmp_path, {}, None)
    assert found.status == verdict.FAIL
    assert found.detail.startswith(f"{files.CUSTODY_FILE} is not JSON lines")
    (tmp_path / files.CUSTODY_FILE).write_bytes(b"\xff\n")
    assert custody.custodial(tmp_path, {}, None).status == verdict.FAIL


def test_tampered_names_each_unregistered_missing_or_changed_file(tmp_path):
    first, second, third, fourth = reports.DECISION_FILES
    write(tmp_path / first, {"a": 1})
    write(tmp_path / second, {"b": 1})
    (tmp_path / fourth).write_text("[1]", encoding="utf-8")
    registered = {
        second: scmle.assurance.ledger.ledger.digest({"b": 2}),
        third: scmle.assurance.ledger.ledger.digest({"c": 1}),
    }
    assert custody.tampered(tmp_path, registered) == [
        f"{first}: not registered",
        f"{second}: differs from the registered hash",
        f"{third}: missing",
        f"{fourth}: unreadable ({fourth} is not a JSON object)",
    ]
    assert custody.tampered(tmp_path / "none", {}) == []


def test_decision_files_without_a_registration(tmp_path):
    assert custody.audit(tmp_path, None) is None
    write(tmp_path / reports.DECISION_FILES[0], {"a": 1})
    found = custody.audit(tmp_path, None)
    assert found == verdict.Finding(
        "decision files: hashes match the custody log",
        verdict.SKIPPED,
        "the run registered no decision file hashes (older run)",
    )


def test_coverage_report_problems(tmp_path):
    path = tmp_path / coverage_run.REPORT_FILE
    assert custody.examine(path, "h") == []
    write(path, {"claim_sha256": "h"})
    assert custody.examine(path, "h") == []
    assert custody.examine(path, None) == []
    assert custody.examine(path, "g") == [
        "coverage_report.json: names another claim"
    ]
    path.write_text("{", encoding="utf-8")
    assert custody.examine(path, "h")[0].startswith(
        "coverage_report.json: unreadable ("
    )


def test_corroborate_without_claim_or_log(tmp_path):
    assert custody.corroborate(tmp_path, None) is None
    write(tmp_path / coverage_run.CLAIM_FILE, {"k": 1})
    found = custody.corroborate(tmp_path, None)
    assert found.status == verdict.SKIPPED
    assert found.detail == "custody_log.jsonl is not in the run"


def test_orphaned_judges_registration_against_evaluation():
    name = "n"
    unevaluated = {"status": confirm.NOT_EVALUATED}
    assert registration.orphaned(name, unevaluated, {"x": 1}).status == (
        verdict.FAIL
    )
    assert registration.orphaned(name, unevaluated, None) == verdict.Finding(
        name, verdict.SKIPPED, "the claim could not be tested on C"
    )
    assert registration.orphaned(name, {"status": "met"}, None) == (
        verdict.Finding(
            name, verdict.FAIL, "the claim is not registered in the ledger"
        )
    )
    assert registration.orphaned(name, {"status": "met"}, {"x": 1}) is None


def test_reconcile_compares_the_status_with_the_test_and_the_seal():
    sealed = ledger(
        confirmation={"claims": {"coverage_nominal": {"status": "met"}}}
    )
    assert (
        registration.reconcile(
            {"status": "met"},
            {"status": "met"},
            sealed,
        )
        == []
    )
    assert registration.reconcile(
        {"status": "met"}, {"status": "missed"}, ledger()
    ) == [
        "the status does not follow from the test",
        "the status differs from the sealed ledger row",
    ]


def test_reconfirm_is_absent_without_a_document(tmp_path):
    assert registration.reconfirm(tmp_path, ledger()) is None


def test_witness_compares_the_control_file_with_the_ledger(tmp_path):
    name = "negative control matches the ledger"
    assert registration.witness(tmp_path, ledger()).status == verdict.SKIPPED
    body = {"control": 1}
    write(tmp_path / files.CONTROL_FILE, body)
    assert registration.witness(tmp_path, ledger()).status == verdict.SKIPPED
    matching = ledger(
        control={"sha256": scmle.assurance.ledger.ledger.digest(body)}
    )
    assert registration.witness(tmp_path, matching) == verdict.Finding(
        name, verdict.PASS, "hash equal"
    )
    other = ledger(control={"sha256": "0"})
    assert registration.witness(tmp_path, other).status == verdict.FAIL
    (tmp_path / files.CONTROL_FILE).write_text("{", encoding="utf-8")
    assert registration.witness(tmp_path, matching) == verdict.Finding(
        name, verdict.FAIL, "negative_control.json is not JSON"
    )


def test_check_registered_lists_every_disagreement(tmp_path):
    attached = {"a": 1}
    write(tmp_path / "attached.json", attached)
    sealed = ledger(
        claims={"h1": {"sha256": "x", "kind": "plain"}},
        confirmation={
            "claims": {"h1": {"sha256": "x"}},
            "attachments": {
                "attached.json": scmle.assurance.ledger.ledger.digest(attached)
            },
        },
    )
    document = {
        "registered_claims": {"h1": {"sha256": "x"}},
        "attachments": {
            "attached.json": scmle.assurance.ledger.ledger.digest(attached)
        },
    }
    assert registration.check(tmp_path, document, sealed) == (
        verdict.Finding(
            "registered claims: files hash to the sealed ledger",
            verdict.PASS,
            "1 claim(s), 1 file(s) bound",
        )
    )
    write(tmp_path / "attached.json", {"a": 2})
    found = registration.check(tmp_path, {"registered_claims": {}}, sealed)
    assert found.detail == (
        "claims file and confirmation row differ; "
        "attached.json: differs from the sealed hash; "
        "attachment hashes differ from the claims file"
    )
    (tmp_path / "attached.json").unlink()
    sealed.claims["h2"] = {"sha256": "y", "kind": "plain"}
    found = registration.check(tmp_path, document, sealed)
    assert found.detail == (
        "h2: registration hash differs; attached.json: missing"
    )


def test_misnamed_checks_the_baseline_registration(tmp_path):
    assert registration.misnamed(tmp_path, ledger()) == []
    write(
        tmp_path / files.BASELINE_FILE,
        {"digests": {"registration": "r"}, "schema_version": 2},
    )
    claims = {
        "b1": {"kind": "baseline", "sha256": "r"},
        "b2": {"kind": "baseline", "sha256": "s"},
        "p1": {"kind": "plain", "sha256": "t"},
    }
    assert registration.misnamed(tmp_path, ledger(claims=claims)) == [
        "b2: report names another registration"
    ]


def test_the_ledger_head_must_be_named():
    found = seal.anchor({"heads": {"ledger": "abcdef0123456789"}}, ledger())
    assert found == verdict.Finding(
        "claims file names the ledger head", verdict.PASS, "head abcdef012345"
    )
    assert seal.anchor({}, ledger()).status == verdict.FAIL


def test_authenticate_is_skipped_without_a_charter(tmp_path):
    found = seal.authenticate(tmp_path, ledger(), None)
    assert found == [
        verdict.Finding(
            "charter verifies and matches the ledger",
            verdict.SKIPPED,
            "analysis_charter.json is not in the run",
        )
    ]


def test_ruled_compares_the_rules_used_with_the_charter():
    rules = types.SimpleNamespace(ledger="holm", confirmation="hochberg")
    assert seal.ruled(None, ledger()).status == verdict.SKIPPED
    assert seal.ruled(rules, ledger()) == verdict.Finding(
        "chartered rules were used", verdict.PASS, "holm and hochberg"
    )
    used = ledger(confirmation={"rule": "hochberg"})
    assert seal.ruled(rules, used).status == verdict.PASS
    other = ledger(confirmation={"rule": "bonferroni"})
    assert seal.ruled(rules, other).detail == (
        "ledger rule 'holm' / confirmation rule 'bonferroni' differ from "
        "the charter's 'holm' / 'hochberg'"
    )
    ledgered = ledger(header={"rule": {"rule": "bh"}})
    assert seal.ruled(rules, ledgered).status == verdict.FAIL


def survivors(*ids):
    return types.SimpleNamespace(sha256="s", hypotheses=ids)


def test_compare_names_each_differing_header_field():
    sealed = {"outcome": "confirmed", "rule": "holm", "alpha_confirm": "1/20"}
    document = {
        "digests": {"survivors": "s"},
        "outcome": "confirmed",
        "rule": "holm",
        "alpha": "1/20",
    }
    assert seal.compare(document, sealed, survivors()) == []
    assert seal.compare({}, sealed, survivors()) == [
        "survivor hash",
        "outcome",
        "rule",
        "alpha_confirm",
    ]


def test_crosscheck_needs_a_sealed_ledger():
    found = seal.crosscheck({}, ledger())
    assert found == verdict.Finding(
        "claims agree with the sealed ledger",
        verdict.FAIL,
        "the ledger is not sealed",
    )


def test_crosscheck_agrees_on_ids_and_the_confirmed_set():
    sealed = ledger(
        confirmation={
            "outcome": "confirmed",
            "rule": "holm",
            "alpha_confirm": "1/20",
            "confirmed": ["h1"],
        },
        remaining=survivors("h1", "h2"),
    )
    document = {
        "digests": {"survivors": "s"},
        "outcome": "confirmed",
        "rule": "holm",
        "alpha": "1/20",
        "claims": [
            {"identifier": "h1", "tier": "confirmed"},
            {"identifier": "h2", "tier": "suggestive"},
        ],
    }
    assert seal.crosscheck(document, sealed) == verdict.Finding(
        "claims agree with the sealed ledger",
        verdict.PASS,
        "1 confirmed of 2 claims",
    )
    document["claims"] = document["claims"][:1]
    document["second"] = {"x": 1}
    assert seal.crosscheck(document, sealed).detail == (
        "differs from the ledger: claim ids differ from the frozen "
        "survivors, second-stage accounting"
    )
    sealed.confirmation["confirmed"] = []
    sealed.confirmation["outcome"] = confirm.NO_SURVIVORS
    empty = {**document, "outcome": confirm.NO_SURVIVORS, "claims": []}
    empty.pop("second")
    assert seal.crosscheck(empty, sealed).status == verdict.PASS


def test_upheld_is_true_only_for_confirmed():
    assert seal.upheld({"tier": "confirmed"}) is True
    assert seal.upheld({"tier": "suggestive"}) is False
    assert seal.upheld({}) is False


def test_rederive_replays_the_adjustment():
    assert seal.rederive({"outcome": "no_survivors"}) == verdict.Finding(
        "confirmation replays from the p-values on C",
        verdict.SKIPPED,
        "outcome is 'no_survivors'",
    )
    document = {
        "outcome": confirm.CONFIRMED,
        "rule": "bonferroni",
        "alpha": "1/20",
        "checks": [
            {"identifier": "h1:C", "p_value": 0.01},
            {"identifier": "h2:C", "p_value": 0.2},
        ],
        "claims": [
            {"identifier": "h1", "adjusted_p_value": 0.02, "tier": "confirmed"},
            {"identifier": "h2", "adjusted_p_value": 0.4, "tier": "suggestive"},
        ],
    }
    assert seal.rederive(document) == verdict.Finding(
        "confirmation replays from the p-values on C",
        verdict.PASS,
        "2 hypotheses, rule bonferroni",
    )
    document["claims"][1]["adjusted_p_value"] = 0.41
    assert seal.rederive(document).detail == "h2: adjusted p differs"
    document["claims"][1]["adjusted_p_value"] = 0.4
    document["claims"][1]["tier"] = "confirmed"
    assert seal.rederive(document).detail == "h2: tier differs"
    document["claims"][0]["adjusted_p_value"] = None
    assert seal.rederive(document).detail == "h1: adjusted p differs"
    assert seal.rederive({"outcome": confirm.CONFIRMED}).detail.startswith(
        "cannot replay: KeyError"
    )


def test_stage_two_record_problems_names_each_inconsistency():
    frozen_set = survivors("h1", "h2")
    record = {
        "family": ["h2"],
        "family_sha256": "e",
        "survivors_sha256": "s",
    }
    assert seal.screen(record, frozen_set, {"h1"}, ["h2"], "e") == []
    wrong = {"family": [], "family_sha256": "x", "survivors_sha256": "y"}
    assert seal.screen(wrong, frozen_set, {"h9"}, ["h2"], "e") == [
        "a withdrawn id is not a survivor",
        "family is not survivors minus withdrawn",
        "family hash",
        "survivor hash",
    ]


def test_tally_checks_the_withdrawal_accounting():
    name = "second-stage accounting is consistent"
    assert seal.tally({}, ledger()) == verdict.Finding(
        name, verdict.SKIPPED, "no withdrawal rule ran"
    )
    assert seal.tally({"second": {}}, ledger()).detail == (
        "the ledger has no frozen survivors"
    )
    frozen_set = survivors("h1", "h2")
    family = ["h2"]
    record = {
        "withdrawn": ["h1"],
        "family": family,
        "family_sha256": scmle.assurance.ledger.ledger.digest(
            {"survivors_sha256": "s", "family": family}
        ),
        "survivors_sha256": "s",
    }
    document = {
        "second": record,
        "checks": [{"identifier": "h2:C"}],
        "claims": [{"identifier": "h1", "tier": "suggestive"}],
    }
    sealed = ledger(remaining=frozen_set)
    assert seal.tally(document, sealed) == verdict.Finding(
        name, verdict.PASS, "1 tested, 1 withdrawn before C opened"
    )
    document["checks"].append({"identifier": "h1:C"})
    document["checks"].append({"identifier": "h3:C"})
    document["claims"][0]["tier"] = "confirmed"
    assert seal.tally(document, sealed).detail == (
        "a withdrawn survivor was tested on C; "
        "a check on C is outside the family; "
        "a withdrawn survivor is confirmed"
    )


def test_vouch_is_skipped_without_claims_or_charter(tmp_path):
    found = attestation.vouch(tmp_path, None)
    assert found == verdict.Finding(
        "causal attestations verify",
        verdict.SKIPPED,
        "no causal claims or charter file",
    )
    write(tmp_path / files.CAUSAL_FILE, {"tier": "unverified"})
    assert attestation.vouch(tmp_path, None).status == verdict.SKIPPED
    write(tmp_path / files.CHARTER_FILE, {"charter": {"broken": True}})
    assert attestation.vouch(tmp_path, None).status == verdict.FAIL


def test_vet_reports_malformed_attestations():
    accepted, problems, unchecked = attestation.vet(
        {"accepted": [{"nonsense": 1}]}, None, {}, None
    )
    assert accepted == {}
    assert len(problems) == 1
    assert problems[0].startswith("malformed attestation: ")
    assert unchecked is False
    assert attestation.vet({}, None, {}, None) == ({}, [], False)
