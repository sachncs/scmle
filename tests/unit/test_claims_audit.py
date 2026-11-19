"""`scmle claims`: list, show and verify the artifacts of a charter run."""

import json

import pytest

from scmle import cli
from scmle.artifacts import claims_audit
from scmle.assurance.charter import sealing
import scmle.assurance.charter.rules
from scmle.assurance.custody import sealed
from scmle.assurance.ledger import confirm
from scmle.assurance.ledger import governed
from scmle.assurance.ledger import ledger
from scmle.contracts import errors
from scmle.contracts import records
from scmle.data.splitting.splitters import lockbox
from tests.unit.test_charter_contracts import make_charter
from tests.unit.test_ledger_governed import build

KEY = "audit-key"


def withdraw_f1(survivors):
    reasons = {
        h.identifier: "fragile on X" for h in survivors if h.spec.column == "f1"
    }
    return confirm.Withdrawal(reasons, {"name": "test", "requested": True})


def make_run(tmp_path, withdraw=True, rules=None):
    features, target = build(rows=500, seed=0, effect=0.7)
    cfeatures, ctarget = build(rows=400, seed=9, effect=0.7)
    log = sealed.Journal(tmp_path / claims_audit.CUSTODY_FILE, KEY)
    source = sealed.Locker(
        "C",
        lockbox.Resident(cfeatures, ctarget, tuple(range(len(ctarget)))),
        log,
    )
    charter = make_charter(rules=rules)
    sealing.inscribe(
        tmp_path / claims_audit.CHARTER_FILE, charter, signing_key=KEY
    )
    options = {}
    if rules is not None:
        options = {
            "rule": rules.ledger,
            "confirmation_rule": confirm.budget.Correction(rules.confirmation),
        }
    return governed.govern(
        charter=charter,
        features=features,
        target=target,
        classification=False,
        source=source,
        run_dir=tmp_path,
        second_stage=withdraw_f1 if withdraw else None,
        **options,
    )


def statuses(result):
    return {f.name: f.status for f in result.findings}


def test_a_consistent_run_verifies_every_step(tmp_path):
    make_run(tmp_path)
    result = claims_audit.verify(tmp_path, KEY)

    assert result.ok
    assert statuses(result).pop("chartered rules were used") == "skipped"
    assert len(result.findings) == 10
    assert result.verified == 8
    assert "signature verified" in next(
        f.detail for f in result.findings if f.name.startswith("charter")
    )


def test_a_signed_run_cannot_pass_the_audit_without_its_key(tmp_path):
    # ERR-3: auditing a signed run without the key used to skip the MAC
    # checks and pass as "hash-checked only"; a signed custody log now
    # refuses to verify without its key, so the audit fails and says why.
    make_run(tmp_path, withdraw=False)
    result = claims_audit.verify(tmp_path)
    assert not result.ok
    assert any(
        "signed and no key was given" in f.detail for f in result.findings
    )
    assert claims_audit.verify(tmp_path, KEY).ok


def test_chartered_rules_are_checked_against_the_ledger(tmp_path):
    rules = scmle.assurance.charter.rules.Procedure(
        ledger="lord++", confirmation="bh"
    )
    make_run(tmp_path, rules=rules)
    result = claims_audit.verify(tmp_path, KEY)
    assert result.ok
    assert statuses(result)["chartered rules were used"] == "pass"

    other = tmp_path / "other"
    other.mkdir()
    make_run(other)
    changed = json.loads((other / "analysis_charter.json").read_text())
    changed["charter"]["rules"] = scmle.assurance.charter.rules.Procedure(
        confirmation="holm"
    ).to_dict()
    changed["format"] = 3
    (other / "analysis_charter.json").write_text(json.dumps(changed))
    assert not claims_audit.verify(other, KEY).ok  # hash no longer matches


def edit_claims(tmp_path, change):
    path = tmp_path / confirm.CLAIMS_FILE
    document = json.loads(path.read_text())
    change(document)
    path.write_text(json.dumps(document))


def test_a_claims_file_written_before_the_key_renames_verifies(tmp_path):
    make_run(tmp_path)

    def downgrade(document):
        document.pop(records.KEY)
        old = records.downgrade(confirm.Confirmation.migrations, document)
        document.clear()
        document.update(old)

    edit_claims(tmp_path, downgrade)
    stored = json.loads((tmp_path / confirm.CLAIMS_FILE).read_text())
    assert "alpha_confirm" in stored and "second" not in stored
    assert claims_audit.verify(tmp_path, KEY).ok


@pytest.mark.parametrize(
    ("change", "step"),
    [
        (
            lambda d: d["claims"][0].update(adjusted_p_value=0.5),
            "confirmation replays from the p-values on C",
        ),
        (
            lambda d: d["claims"][0].update(tier="unverified"),
            "claims agree with the sealed ledger",
        ),
        (
            lambda d: d["digests"].update(survivors="0" * 64),
            "claims agree with the sealed ledger",
        ),
        (
            lambda d: d["heads"].update(ledger="0" * 64),
            "claims file names the ledger head",
        ),
        (
            lambda d: d["second"]["withdrawn"].clear(),
            "claims agree with the sealed ledger",
        ),
    ],
)
def test_an_edited_claims_file_fails_the_matching_step(tmp_path, change, step):
    make_run(tmp_path)
    edit_claims(tmp_path, change)
    result = claims_audit.verify(tmp_path, KEY)
    assert not result.ok
    assert statuses(result)[step] == "fail"


def test_a_withdrawn_survivor_tested_on_c_is_caught(tmp_path):
    make_run(tmp_path)

    def add_check(document):
        withdrawn = next(iter(document["second"]["withdrawn"]))
        document["checks"].append(
            dict(document["checks"][0], identifier=f"{withdrawn}:C")
        )

    edit_claims(tmp_path, add_check)
    assert (
        statuses(claims_audit.verify(tmp_path, KEY))[
            "second-stage accounting is consistent"
        ]
        == "fail"
    )


def test_a_removed_or_edited_ledger_row_fails_the_chain(tmp_path):
    make_run(tmp_path)
    path = tmp_path / ledger.FILE_NAME
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    result = claims_audit.verify(tmp_path, KEY)
    assert not result.ok
    assert statuses(result)["claims file names the ledger head"] == "fail"

    again = tmp_path / "again"
    again.mkdir()
    make_run(again)
    lines = (again / ledger.FILE_NAME).read_text().splitlines()
    index = next(i for i, line in enumerate(lines) if '"f0"' in line)
    lines[index] = lines[index].replace('"f0"', '"fX"')
    (again / ledger.FILE_NAME).write_text("\n".join(lines) + "\n")
    assert not claims_audit.verify(again, KEY).ok


def test_an_edited_custody_log_or_control_fails(tmp_path):
    make_run(tmp_path)
    log = tmp_path / claims_audit.CUSTODY_FILE
    log.write_text(log.read_text().replace('"open"', '"opem"'))
    assert (
        statuses(claims_audit.verify(tmp_path, KEY))[
            "custody log: selection frozen before C opened, C opened once"
        ]
        == "fail"
    )

    other = tmp_path / "other"
    other.mkdir()
    make_run(other)
    control = other / "negative_control.json"
    document = json.loads(control.read_text())
    document["observed"] = document["observed"] + 1
    control.write_text(json.dumps(document))
    assert (
        statuses(claims_audit.verify(other, KEY))[
            "negative control matches the ledger"
        ]
        == "fail"
    )


def test_a_directory_without_a_ledger_fails_cleanly(tmp_path):
    result = claims_audit.verify(tmp_path)
    assert not result.ok
    with pytest.raises(errors.ArtifactError):
        claims_audit.inventory(tmp_path)


def test_list_and_show_report_the_claims(tmp_path):
    make_run(tmp_path)
    rows = claims_audit.inventory(tmp_path)
    assert {r["feature"]: r["tier"] for r in rows} == {
        "f0": "confirmed",
        "f1": "unverified",
    }
    assert [r["withdrawn"] for r in rows if r["feature"] == "f1"] == [True]
    detail = claims_audit.inspect(tmp_path, rows[0]["claim_id"])
    assert detail["hypothesis"]["column"] == "f0"
    assert detail["confirmation_check"]["identifier"].endswith(":C")
    assert detail["exploration"]["decision"] == "reject"
    held = claims_audit.inspect(tmp_path, rows[1]["claim_id"])
    assert held["withdrawn"] == "fragile on X"
    assert held["confirmation_check"] is None
    with pytest.raises(errors.ArtifactError, match="no claim"):
        claims_audit.inspect(tmp_path, "h999999")


def test_the_cli_lists_shows_and_verifies(tmp_path, capsys, monkeypatch):
    make_run(tmp_path)
    monkeypatch.setenv("SCMLE_SIGNING_KEY", KEY)

    assert cli.main(["claims", "list", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert "confirmed" in text and "withdrawn before C" in text
    assert (
        cli.main(
            ["claims", "list", str(tmp_path), "--tier", "confirmed", "--json"]
        )
        == 0
    )
    rows = json.loads(capsys.readouterr().out)
    assert [r["feature"] for r in rows] == ["f0"]

    claim = rows[0]["claim_id"]
    assert cli.main(["claims", "show", str(tmp_path), claim]) == 0
    assert json.loads(capsys.readouterr().out)["claim"]["identifier"] == claim

    assert cli.main(["claims", "verify", str(tmp_path)]) == 0
    assert "8 of 10 steps passed" in capsys.readouterr().out
    assert cli.main(["claims", "verify", str(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True

    edit_claims(tmp_path, lambda d: d["claims"][0].update(tier="unverified"))
    assert cli.main(["claims", "verify", str(tmp_path)]) == 1
    assert "FAILED" in capsys.readouterr().out
    assert cli.main(["claims", "show", str(tmp_path), "h999999"]) == 1


DECISION = "decision files: hashes match the custody log"


def notarize(tmp_path, files):
    """Writes the files and the custody event that registers their hashes."""
    from scmle.decision import reports

    for name, body in files.items():
        (tmp_path / name).write_text(json.dumps(body), encoding="utf-8")
    log = sealed.Journal(tmp_path / claims_audit.CUSTODY_FILE, KEY)
    log.append(
        "final_test",
        "decision_files",
        "scmle.decision",
        {"files": reports.hashsums(tmp_path)},
    )


FOUR = {
    "decision_threshold.json": {"threshold": 0.3},
    "net_benefit.json": {"claim": "useful"},
    "policy_value.json": {"status": "issued"},
    "rollout_design.json": {"plan_hash": "ab"},
}


def test_the_four_decision_files_are_checked_against_the_registration(
    tmp_path,
):
    make_run(tmp_path)
    notarize(tmp_path, FOUR)

    result = claims_audit.verify(tmp_path, KEY)

    assert result.ok
    assert statuses(result)[DECISION] == "pass"


def test_a_tampered_decision_file_fails_with_its_name(tmp_path):
    make_run(tmp_path)
    notarize(tmp_path, FOUR)
    (tmp_path / "net_benefit.json").write_text(
        json.dumps({"claim": "not_useful_in_range"}), encoding="utf-8"
    )

    result = claims_audit.verify(tmp_path, KEY)

    assert not result.ok
    found = next(f for f in result.findings if f.name == DECISION)
    assert found.status == "fail"
    assert "net_benefit.json: differs from the registered hash" in found.detail


def test_a_missing_decision_file_fails_with_its_name(tmp_path):
    make_run(tmp_path)
    notarize(tmp_path, FOUR)
    (tmp_path / "policy_value.json").unlink()

    result = claims_audit.verify(tmp_path, KEY)

    found = next(f for f in result.findings if f.name == DECISION)
    assert found.status == "fail"
    assert "policy_value.json: missing" in found.detail


def test_an_unregistered_absent_file_that_appears_fails(tmp_path):
    make_run(tmp_path)
    files = {k: v for k, v in FOUR.items() if k != "rollout_design.json"}
    notarize(tmp_path, files)
    (tmp_path / "rollout_design.json").write_text("{}", encoding="utf-8")

    found = next(
        f
        for f in claims_audit.verify(tmp_path, KEY).findings
        if f.name == DECISION
    )
    assert found.status == "fail"
    assert "rollout_design.json: not registered" in found.detail


def test_no_decision_files_adds_no_step(tmp_path):
    make_run(tmp_path)
    assert DECISION not in statuses(claims_audit.verify(tmp_path, KEY))


def test_a_malformed_claims_document_fails_the_replay_instead_of_raising():
    document = {
        "outcome": confirm.CONFIRMED,
        "rule": "holm",
        "alpha": "1/20",
        "claims": [{"claim_id": "h1"}],
        "checks": [{"check_id": "h1:C", "p_value": 0.001}],
    }
    assert claims_audit.rederive(document).status == "fail"
    del document["checks"]
    assert claims_audit.rederive(document).status == "fail"


def test_showing_a_claim_the_ledger_does_not_know_is_an_artifact_error(
    tmp_path,
):
    make_run(tmp_path)
    edit_claims(
        tmp_path,
        lambda d: d["claims"].append(
            dict(d["claims"][0], identifier="h424242")
        ),
    )
    with pytest.raises(errors.ArtifactError, match="h424242"):
        claims_audit.inspect(tmp_path, "h424242")


def custody(tmp_path):
    """Writes a one-entry custody log signed with `KEY`."""
    log = sealed.Journal(tmp_path / claims_audit.CUSTODY_FILE, KEY)
    log.append("C", "sealed", "scmle.test", {"rows": 1})


def test_an_empty_key_is_an_input_error_not_a_failed_step(tmp_path):
    # ERR-20: an empty key raised ValueError out of every step before the
    # error types were narrowed; the steps must not turn it into a finding.
    sealing.inscribe(
        tmp_path / claims_audit.CHARTER_FILE, make_charter(), signing_key=KEY
    )
    custody(tmp_path)
    (tmp_path / "coverage_claim.json").write_text("{}", encoding="utf-8")
    with pytest.raises(errors.InputError, match="must not be empty"):
        claims_audit.authenticate(tmp_path, None, "")
    with pytest.raises(errors.InputError, match="must not be empty"):
        claims_audit.custodial(tmp_path, {}, "")
    with pytest.raises(errors.InputError, match="must not be empty"):
        claims_audit.check_decision_files(tmp_path, "")
    with pytest.raises(errors.InputError, match="must not be empty"):
        claims_audit.corroborate(tmp_path, "")


def test_verify_with_an_empty_key_raises(tmp_path):
    make_run(tmp_path)
    with pytest.raises(ValueError, match="must not be empty"):
        claims_audit.verify(tmp_path, "")
