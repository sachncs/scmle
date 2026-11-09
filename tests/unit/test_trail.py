"""The business trail: build, verify, and each seeded defect it must catch.

The fixtures are copies of the committed Phase 18 trail, which was built from
a real run of the Titanic extract. A seeded defect edits one retained record
in the copy and re-seals it, so the only thing the verifier can object to is
the defect.
"""

import copy
import json
import pathlib
import shutil

import pytest

from scmle import cli
from scmle.contracts import signing
from scmle.trail import chain
from scmle.trail import inputs as inputs_module
from scmle.trail import outcomes
from scmle.trail import records
from scmle.trail import show
from scmle.trail import trace

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "tests" / "fixtures" / "trail_sample"
KEYS = json.loads((EVIDENCE / "keys.json").read_text(encoding="utf-8"))


@pytest.fixture
def folder(tmp_path):
    """A writable copy of the committed trail folder."""
    target = tmp_path / "trail"
    shutil.copytree(EVIDENCE / "trail", target)
    return target


def reseal(folder):
    """Re-derives and re-chains the trail after a seeded edit."""
    return chain.seal(folder, KEYS)


def edit_json(path, change):
    """Applies ``change`` to the JSON document at ``path``."""
    document = json.loads(path.read_text(encoding="utf-8"))
    change(document)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def audit_rows(folder):
    """The registry audit rows."""
    return records.read_lines(folder / "retained/registry/audit.jsonl")


def write_rows(path, rows):
    """Writes JSON lines."""
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def promotion_row(rows):
    """The champion promotion row of the audit log."""
    return next(
        r
        for r in rows
        if r.get("action") == "promote" and r.get("alias") == "champion"
    )


def kinds(report):
    """The set of finding kinds."""
    return {finding.kind for finding in report.findings}


def test_the_committed_trail_verifies_from_its_own_folder():
    report = chain.verify(EVIDENCE / "trail", KEYS)
    assert report.ok, [f.to_dict() for f in report.findings]
    assert report.steps == len(trace.KINDS)


def test_every_step_names_the_records_it_rests_on(folder):
    document = records.read_json(folder / records.TRAIL_FILE)
    assert [s["kind"] for s in document["steps"]] == list(trace.KINDS)
    for step in document["steps"]:
        for relative, held in step["records"].items():
            path = (
                folder / relative
                if relative == records.INPUTS
                else folder / records.RETAINED / relative
            )
            assert records.sha256_file(path) == held


def test_the_joins_hold_across_the_records(folder):
    document = records.read_json(folder / records.TRAIL_FILE)
    facts = {s["kind"]: s["facts"] for s in document["steps"]}
    charter = facts["question"]["charter_hash"]
    assert document["charter_hash"] == charter
    assert facts["deployment"]["charter_hash"] == charter
    content = facts["selection"]["content_id"]
    assert facts["approval"]["registered_content_id"] == content
    assert facts["deployment"]["scope"].startswith(content)


def test_the_reports_are_deterministic(folder):
    first = chain.verify(folder, KEYS)
    second = chain.verify(folder, KEYS)
    assert first == second
    stored = records.read_json(folder / records.EXPLORATORY)
    body = {k: v for k, v in stored.items() if k != "sha256"}
    assert stored["sha256"] == records.digest(body)


def test_a_report_edited_after_sealing_is_not_reproducible(folder):
    edit_json(
        folder / records.EXPERIMENTS,
        lambda d: d.__setitem__("selected_model", "dummy"),
    )
    assert trace.MISMATCH in kinds(chain.verify(folder, KEYS))


def test_a_deleted_record_is_a_missing_link(folder):
    (folder / "retained/deployment/deployment_ledger.jsonl").unlink()
    assert trace.MISSING_LINK in kinds(chain.verify(folder, KEYS))
    assert trace.MISSING_LINK in kinds(reseal(folder))


def test_a_step_removed_and_rechained_is_a_missing_link(folder):
    path = folder / records.TRAIL_FILE
    document = records.read_json(path)
    document["steps"] = [
        s for s in document["steps"] if s["kind"] != "approval"
    ]
    document["steps"] = chain.link(
        tuple(
            {k: v for k, v in s.items() if k not in ("prev", "hash")}
            for s in document["steps"]
        )
    )
    document["head"] = document["steps"][-1]["hash"]
    document["builder_signature"] = chain.signature_for(
        document["head"], document["retained"], KEYS["analyst"]
    )
    records.write_json(path, document)
    report = chain.verify(folder, KEYS)
    assert trace.MISSING_LINK in kinds(report)
    assert any(f.step == "approval" for f in report.findings)


def test_an_edited_step_breaks_the_chain(folder):
    path = folder / records.TRAIL_FILE
    edit_json(path, lambda d: d["steps"][2]["facts"].update(signed=False))
    assert trace.BROKEN_CHAIN in kinds(chain.verify(folder, KEYS))


def test_an_edited_deployment_ledger_row_breaks_its_chain(folder):
    path = folder / "retained/deployment/deployment_ledger.jsonl"
    rows = records.read_lines(path)
    rows[2]["body"]["cases"] = 1
    write_rows(path, rows)
    assert trace.BROKEN_CHAIN in kinds(reseal(folder))


def test_an_edited_claim_ledger_row_breaks_its_chain(folder):
    path = folder / "retained/run/hypothesis_ledger.jsonl"
    rows = records.read_lines(path)
    rows[3]["body"]["source"] = "edited"
    write_rows(path, rows)
    assert trace.BROKEN_CHAIN in kinds(reseal(folder))


def test_a_record_replaced_after_sealing_is_tampered(folder):
    edit_json(
        folder / "retained/run/evaluation.json",
        lambda d: d["test_metrics"].update(roc_auc=0.99),
    )
    assert trace.TAMPERED in kinds(chain.verify(folder, KEYS))


def test_an_unsigned_stakeholder_section_is_an_unsigned_step(folder):
    edit_json(
        folder / records.INPUTS,
        lambda d: d["stakeholder"].pop("signature"),
    )
    report = reseal(folder)
    assert trace.UNSIGNED in kinds(report)
    assert any(f.step == "stakeholder" for f in report.findings)


def test_a_wrong_reviewer_key_is_an_unsigned_step(folder):
    wrong = dict(KEYS, reviewer="not-the-reviewer-key")
    report = chain.verify(folder, wrong)
    assert trace.UNSIGNED in kinds(report)


def test_an_unsigned_charter_is_an_unsigned_step(folder):
    edit_json(
        folder / "retained/run/analysis_charter.json",
        lambda d: d.update(signature=None),
    )
    report = reseal(folder)
    assert trace.UNSIGNED in kinds(report)
    assert any(f.step == "question" for f in report.findings)


def test_a_promotion_without_a_signature_is_an_unsigned_step(folder):
    path = folder / "retained/registry/audit.jsonl"
    rows = audit_rows(folder)
    promotion_row(rows)["approval_record"].pop("signature")
    write_rows(path, rows)
    report = reseal(folder)
    assert trace.UNSIGNED in kinds(report)


def test_a_promotion_with_no_approval_is_an_unapproved_step(folder):
    path = folder / "retained/registry/audit.jsonl"
    rows = [r for r in audit_rows(folder) if r.get("action") != "promote"]
    write_rows(path, rows)
    assert trace.UNAPPROVED in kinds(reseal(folder))


def test_self_approval_is_an_unapproved_step(folder):
    path = folder / "retained/registry/audit.jsonl"
    rows = audit_rows(folder)
    record = promotion_row(rows)["approval_record"]
    record["reviewer"] = "sachin"
    record.pop("signature")
    record["signature"] = signing.sign_payload(record, KEYS["reviewer"])
    write_rows(path, rows)
    report = reseal(folder)
    assert trace.UNAPPROVED in kinds(report)
    assert trace.UNSIGNED not in kinds(report)


def test_an_approval_of_another_bundle_is_not_accepted(folder):
    path = folder / "retained/registry/audit.jsonl"
    rows = audit_rows(folder)
    record = promotion_row(rows)["approval_record"]
    record["content_id"] = "0" * 64
    record.pop("signature")
    record["signature"] = signing.sign_payload(record, KEYS["reviewer"])
    write_rows(path, rows)
    assert trace.UNAPPROVED in kinds(reseal(folder))


def test_outcomes_outside_the_approved_expectation_are_reported(folder):
    def worsen(feed):
        for batch in feed["batches"]:
            batch["label"] = [1] * len(batch["label"])
            batch["probability"] = [0.0] * len(batch["probability"])

    edit_json(folder / "retained/deployment/observed_outcomes.json", worsen)
    report = reseal(folder)
    assert trace.OUTSIDE in kinds(report)
    tracking = records.read_json(folder / records.OUTCOME_TRACKING)
    assert tracking["status"] == outcomes.ABOVE
    assert "not a forecast" in tracking["uncertainty"]


def test_a_feed_that_disagrees_with_the_ledger_is_reported(folder):
    def drop(feed):
        feed["batches"][0]["label"].pop()
        feed["batches"][0]["probability"].pop()
        feed["batches"][0]["case_ids"].pop()

    edit_json(folder / "retained/deployment/observed_outcomes.json", drop)
    assert trace.MISMATCH in kinds(reseal(folder))


def test_a_failed_quality_threshold_is_reported(folder):
    path = folder / records.INPUTS
    record = records.read_json(path)
    record["reviewer"]["thresholds"][0]["value"] = 10**6
    path.write_text(
        json.dumps(inputs_module.sign_inputs(record, KEYS), indent=2) + "\n",
        encoding="utf-8",
    )
    report = reseal(folder)
    assert trace.NOT_MET in kinds(report)
    assert trace.UNSIGNED not in kinds(report)


def test_a_record_changed_after_the_reviewer_signed_is_reported(folder):
    edit_json(
        folder / "retained/run/feasibility.json",
        lambda d: d.update(verdict="adequate"),
    )
    assert trace.TAMPERED in kinds(reseal(folder))


def test_the_charter_hash_must_agree_across_records(folder):
    edit_json(
        folder / "retained/framing/framing_draft.json",
        lambda d: d.update(charter_hash="1" * 64),
    )
    assert trace.IDENTITY in kinds(reseal(folder))


def test_inputs_structure_problems_are_listed():
    record = copy.deepcopy(
        records.read_json(EVIDENCE / "trail" / records.INPUTS)
    )
    assert inputs_module.structure_problems(record) == []
    record["stakeholder"].pop("expectation")
    record["reviewer"]["thresholds"][0]["op"] = "~"
    found = inputs_module.structure_problems(record)
    assert any("expectation" in item for item in found)
    assert any("operator" in item for item in found)


def test_the_interval_and_the_status_follow_the_expectation():
    costs = {"c_00": 0.0, "c_01": 5.0, "c_10": 1.0, "c_11": 0.0}
    assert outcomes.optimal_threshold(costs) == pytest.approx(1 / 6)
    feed = {
        "threshold": 1 / 6,
        "batches": [{"batch": 0, "probability": [0.9] * 40, "label": [1] * 40}],
    }
    expectation = {"low": 0.0, "high": 0.5}
    assert outcomes.track(feed, costs, expectation, 10)["status"] == "within"
    assert (
        outcomes.track(feed, costs, expectation, 100)["status"]
        == "insufficient"
    )
    feed["batches"][0]["probability"] = [0.0] * 40
    assert outcomes.track(feed, costs, expectation, 10)["status"] == "above"
    low_range = {"low": 9.0, "high": 10.0}
    feed["batches"][0]["probability"] = [0.9] * 40
    assert outcomes.track(feed, costs, low_range, 10)["status"] == "below"
    assert outcomes.track(None, costs, expectation, 10)["status"] == "absent"


def test_show_prints_every_step(folder):
    text = show.render(folder)
    for title in trace.TITLES.values():
        assert title in text
    assert "Predict `survived`" in text


def test_cli_verify_and_show(folder, tmp_path, capsys):
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps(KEYS), encoding="utf-8")
    assert cli.main(["trail", "verify", str(folder), "--keys", str(keys)]) == 0
    assert cli.main(["trail", "show", str(folder)]) == 0
    assert "Trail for" in capsys.readouterr().out
    (folder / "retained/deployment/deployment_ledger.jsonl").unlink()
    code = cli.main(["trail", "verify", str(folder), "--keys", str(keys)])
    assert code == cli.EXIT_FAILED
    assert "missing_link" in capsys.readouterr().out


def test_cli_build_reproduces_the_committed_trail(folder, tmp_path):
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps(KEYS), encoding="utf-8")
    base = folder / records.RETAINED
    rebuilt = tmp_path / "rebuilt"
    code = cli.main(
        [
            "trail",
            "build",
            str(rebuilt),
            "--run",
            str(base / "run"),
            "--framing",
            str(base / "framing"),
            "--registry",
            str(base / "registry"),
            "--deployment",
            str(base / "deployment"),
            "--inputs",
            str(folder / records.INPUTS),
            "--keys",
            str(keys),
        ]
    )
    assert code == 0
    assert (rebuilt / records.TRAIL_FILE).read_text() == (
        folder / records.TRAIL_FILE
    ).read_text()


def test_the_committed_evidence_says_what_was_not_run():
    summary = json.loads((EVIDENCE / "summary.json").read_text())
    assert summary["independent_reviewer"] is False
    assert summary["production_traffic"] is False
    assert summary["cloud"] is False
    text = " ".join(summary["not_run"])
    for word in (
        "reviewer who did not build",
        "production traffic",
        "AWS or GCP",
    ):
        assert word in text
