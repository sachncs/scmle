"""`scmle claims`: list, show and verify the artifacts of a charter run."""

import json

import pytest

from scmle import cli
from scmle.artifacts import claims_audit
from scmle.contracts import errors
from scmle.data.charter import rules as rules_module
from scmle.data.charter import sealing
from scmle.data.custody import sealed
from scmle.data.ledger import confirm
from scmle.data.ledger import governed
from scmle.data.ledger import ledger as ledger_module
from scmle.data.splitting import splitters
from tests.unit.test_charter_contracts import make_charter
from tests.unit.test_ledger_governed import build

KEY = "audit-key"


def withdraw_f1(survivors):
    reasons = {
        h.hypothesis_id: "fragile on X"
        for h in survivors
        if h.spec.column == "f1"
    }
    return confirm.Withdrawal(reasons, {"name": "test", "requested": True})


def make_run(tmp_path, withdraw=True, rules=None):
    features, target = build(rows=500, seed=0, effect=0.7)
    cfeatures, ctarget = build(rows=400, seed=9, effect=0.7)
    log = sealed.OpenLog(tmp_path / claims_audit.CUSTODY_FILE, KEY)
    source = sealed.SealedSet(
        "C",
        splitters.ResidentTestSet(
            cfeatures, ctarget, tuple(range(len(ctarget)))
        ),
        log,
    )
    charter = make_charter(rules=rules)
    sealing.write_charter(
        tmp_path / claims_audit.CHARTER_FILE, charter, signing_key=KEY
    )
    options = {}
    if rules is not None:
        options = {
            "rule": rules.ledger_rule,
            "confirmation_rule": confirm.budget_module.MultiplicityRule(
                rules.confirmation_rule
            ),
        }
    return governed.run_governed(
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


def test_without_a_key_the_charter_is_hash_checked_only(tmp_path):
    make_run(tmp_path, withdraw=False)
    result = claims_audit.verify(tmp_path)
    assert result.ok
    assert (
        statuses(result)["second-stage accounting is consistent"] == "skipped"
    )
    assert any("not signed" in f.detail for f in result.findings)


def test_chartered_rules_are_checked_against_the_ledger(tmp_path):
    rules = rules_module.RunRules(ledger_rule="lord++", confirmation_rule="bh")
    make_run(tmp_path, rules=rules)
    result = claims_audit.verify(tmp_path, KEY)
    assert result.ok
    assert statuses(result)["chartered rules were used"] == "pass"

    other = tmp_path / "other"
    other.mkdir()
    make_run(other)
    changed = json.loads((other / "analysis_charter.json").read_text())
    changed["charter"]["rules"] = rules_module.RunRules(
        confirmation_rule="holm"
    ).to_dict()
    changed["format"] = 3
    (other / "analysis_charter.json").write_text(json.dumps(changed))
    assert not claims_audit.verify(other, KEY).ok  # hash no longer matches


def edit_claims(tmp_path, change):
    path = tmp_path / confirm.CLAIMS_FILE
    document = json.loads(path.read_text())
    change(document)
    path.write_text(json.dumps(document))


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
            lambda d: d.update(survivors_sha256="0" * 64),
            "claims agree with the sealed ledger",
        ),
        (
            lambda d: d.update(ledger_head="0" * 64),
            "claims file names the ledger head",
        ),
        (
            lambda d: d["stage_two"]["withdrawn"].clear(),
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
        withdrawn = next(iter(document["stage_two"]["withdrawn"]))
        document["checks"].append({"check_id": f"{withdrawn}:C"})

    edit_claims(tmp_path, add_check)
    assert (
        statuses(claims_audit.verify(tmp_path, KEY))[
            "second-stage accounting is consistent"
        ]
        == "fail"
    )


def test_a_removed_or_edited_ledger_row_fails_the_chain(tmp_path):
    make_run(tmp_path)
    path = tmp_path / ledger_module.FILE_NAME
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    result = claims_audit.verify(tmp_path, KEY)
    assert not result.ok
    assert statuses(result)["claims file names the ledger head"] == "fail"

    again = tmp_path / "again"
    again.mkdir()
    make_run(again)
    lines = (again / ledger_module.FILE_NAME).read_text().splitlines()
    index = next(i for i, line in enumerate(lines) if '"f0"' in line)
    lines[index] = lines[index].replace('"f0"', '"fX"')
    (again / ledger_module.FILE_NAME).write_text("\n".join(lines) + "\n")
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
        claims_audit.list_claims(tmp_path)


def test_list_and_show_report_the_claims(tmp_path):
    make_run(tmp_path)
    rows = claims_audit.list_claims(tmp_path)
    assert {r["feature"]: r["tier"] for r in rows} == {
        "f0": "confirmed",
        "f1": "unverified",
    }
    assert [r["withdrawn"] for r in rows if r["feature"] == "f1"] == [True]
    detail = claims_audit.show_claim(tmp_path, rows[0]["claim_id"])
    assert detail["hypothesis"]["column"] == "f0"
    assert detail["confirmation_check"]["check_id"].endswith(":C")
    assert detail["exploration"]["decision"] == "reject"
    held = claims_audit.show_claim(tmp_path, rows[1]["claim_id"])
    assert held["withdrawn"] == "fragile on X"
    assert held["confirmation_check"] is None
    with pytest.raises(errors.ArtifactError, match="no claim"):
        claims_audit.show_claim(tmp_path, "h999999")


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
    assert json.loads(capsys.readouterr().out)["claim"]["claim_id"] == claim

    assert cli.main(["claims", "verify", str(tmp_path)]) == 0
    assert "8 of 10 steps passed" in capsys.readouterr().out
    assert cli.main(["claims", "verify", str(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True

    edit_claims(tmp_path, lambda d: d["claims"][0].update(tier="unverified"))
    assert cli.main(["claims", "verify", str(tmp_path)]) == 1
    assert "FAILED" in capsys.readouterr().out
    assert cli.main(["claims", "show", str(tmp_path), "h999999"]) == 1


DECISION = "decision files: hashes match the custody log"


def register_decision_files(tmp_path, files):
    """Writes the files and the custody event that registers their hashes."""
    from scmle.core import decision_reports

    for name, body in files.items():
        (tmp_path / name).write_text(json.dumps(body), encoding="utf-8")
    log = sealed.OpenLog(tmp_path / claims_audit.CUSTODY_FILE, KEY)
    log.append(
        "final_test",
        "decision_files",
        "scmle.decision",
        {"files": decision_reports.file_digests(tmp_path)},
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
    register_decision_files(tmp_path, FOUR)

    result = claims_audit.verify(tmp_path, KEY)

    assert result.ok
    assert statuses(result)[DECISION] == "pass"


def test_a_tampered_decision_file_fails_with_its_name(tmp_path):
    make_run(tmp_path)
    register_decision_files(tmp_path, FOUR)
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
    register_decision_files(tmp_path, FOUR)
    (tmp_path / "policy_value.json").unlink()

    result = claims_audit.verify(tmp_path, KEY)

    found = next(f for f in result.findings if f.name == DECISION)
    assert found.status == "fail"
    assert "policy_value.json: missing" in found.detail


def test_an_unregistered_absent_file_that_appears_fails(tmp_path):
    make_run(tmp_path)
    files = {k: v for k, v in FOUR.items() if k != "rollout_design.json"}
    register_decision_files(tmp_path, files)
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
