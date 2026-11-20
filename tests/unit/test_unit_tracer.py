"""Behaviour of the trail builder and each module of the trail tracer."""

import json
import pathlib
import shutil

import pytest

from scmle.contracts import signing
from scmle.trail import builder
from scmle.trail import records
from scmle.trail.trace import evidence
from scmle.trail.trace import finding
from scmle.trail.trace import planning
from scmle.trail.trace import reader
from scmle.trail.trace import release
from scmle.trail.trace import tracer

ROOT = pathlib.Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "tests" / "fixtures" / "trail_sample"
KEYS = json.loads((SAMPLE / "keys.json").read_text(encoding="utf-8"))
CHARTER = "run/analysis_charter.json"


@pytest.fixture
def folder(tmp_path):
    target = tmp_path / "trail"
    shutil.copytree(SAMPLE / "trail", target)
    return target


def change(path, edit):
    data = json.loads(path.read_text(encoding="utf-8"))
    edit(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def found(derived):
    return {(f.kind, f.step) for f in derived.findings}


def details(derived):
    return [f.detail for f in derived.findings]


# ---- finding ---------------------------------------------------------------


def test_finding_kinds_are_the_stable_strings():
    assert finding.MISSING_LINK == "missing_link"
    assert finding.BROKEN_CHAIN == "broken_chain"
    assert finding.UNSIGNED == "unsigned_step"
    assert finding.UNAPPROVED == "unapproved_step"
    assert finding.OUTSIDE == "outcome_outside_expectation"
    assert finding.TAMPERED == "record_tampered"
    assert finding.MISMATCH == "fact_mismatch"
    assert finding.NOT_MET == "criterion_not_met"
    assert finding.IDENTITY == "identity_mismatch"


def test_every_step_kind_has_a_title_in_order():
    assert len(finding.KINDS) == 10
    assert finding.KINDS[0] == "question"
    assert finding.KINDS[-1] == "outcomes"
    assert list(finding.TITLES) == list(finding.KINDS)


def test_a_finding_dumps_its_three_fields():
    item = finding.Finding("missing_link", "question", "gone")
    assert item.to_dict() == {
        "kind": "missing_link",
        "step": "question",
        "detail": "gone",
    }


# ---- reader ----------------------------------------------------------------


def test_reader_starts_empty_and_binds_the_retained_folder(tmp_path):
    state = reader.Reader(tmp_path, {"analyst": "k"})
    assert state.base == tmp_path / "retained"
    assert state.keys == {"analyst": "k"}
    assert state.findings == []
    assert state.steps == {}
    assert state.charter == ""
    assert state.inputs == {}


def test_need_records_a_missing_link_for_an_absent_record(tmp_path):
    state = reader.Reader(tmp_path, {})
    assert state.need("run/x.json", "question") is None
    assert state.findings == [
        finding.Finding(
            "missing_link", "question", "retained/run/x.json is absent"
        )
    ]
    (tmp_path / "retained" / "run").mkdir(parents=True)
    (tmp_path / "retained" / "run" / "x.json").write_text("{}")
    assert (
        state.need("run/x.json", "question") == tmp_path / "retained/run/x.json"
    )
    assert len(state.findings) == 1


def test_json_caches_values_and_failures(tmp_path):
    base = tmp_path / "retained"
    base.mkdir()
    (base / "a.json").write_text('{"v": 1}')
    (base / "b.json").write_text("{oops")
    state = reader.Reader(tmp_path, {})
    assert state.json("a.json", "s") == {"v": 1}
    (base / "a.json").write_text('{"v": 2}')
    assert state.json("a.json", "s") == {"v": 1}
    assert state.json("b.json", "s") is None
    assert state.json("b.json", "s") is None
    assert state.findings == [
        finding.Finding(
            "record_tampered", "s", "retained/b.json is not valid JSON"
        )
    ]
    assert state.cache == {"a.json": {"v": 1}, "b.json": None}


def test_rows_reads_json_lines_or_records_tampering(tmp_path):
    base = tmp_path / "retained"
    base.mkdir()
    (base / "good.jsonl").write_text('{"a": 1}\n{"a": 2}\n')
    (base / "bad.jsonl").write_text("{nope\n")
    state = reader.Reader(tmp_path, {})
    assert state.rows("good.jsonl", "s") == [{"a": 1}, {"a": 2}]
    assert state.rows("bad.jsonl", "s") is None
    assert state.rows("none.jsonl", "s") is None
    assert [f.kind for f in state.findings] == [
        "record_tampered",
        "missing_link",
    ]
    assert (
        state.findings[0].detail == "retained/bad.jsonl is not valid JSON lines"
    )


def test_add_hashes_only_the_records_that_exist(tmp_path):
    base = tmp_path / "retained"
    base.mkdir()
    (base / "b.json").write_text("{}")
    (tmp_path / "inputs.json").write_text("{}")
    state = reader.Reader(tmp_path, {})
    state.add("question", ["b.json", "gone.json", "inputs.json"], {"x": 1}, [])
    step = state.steps["question"]
    assert step["kind"] == "question"
    assert step["title"] == finding.TITLES["question"]
    assert list(step["records"]) == ["b.json", "inputs.json"]
    assert step["records"]["b.json"] == records.checksum(base / "b.json")
    assert step["facts"] == {"x": 1}
    assert step["requires"] == []


def test_evaluate_reports_each_criterion_and_unresolvable_paths(tmp_path):
    base = tmp_path / "retained" / "run"
    base.mkdir(parents=True)
    (base / "m.json").write_text('{"rows": 10}')
    state = reader.Reader(tmp_path, {})
    results = state.evaluate(
        [
            {"id": "A", "path": "run/m.json#rows", "op": ">=", "value": 10},
            {"id": "B", "path": "run/m.json#rows", "op": ">", "value": 10},
            {"id": "C", "path": "run/gone.json#rows", "op": "==", "value": 1},
        ],
        "quality",
    )
    assert [(r["id"], r["observed"], r["passed"]) for r in results] == [
        ("A", 10, True),
        ("B", 10, False),
        ("C", None, False),
    ]
    assert results[0]["reason"] == ""
    assert [f.kind for f in state.findings] == ["missing_link"]
    assert "criterion C reads run/gone.json#rows" in state.findings[0].detail


def test_recompute_compares_the_stored_report_with_the_fresh_one(tmp_path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "r.json").write_text('{"sha256": "x"}')
    state = reader.Reader(tmp_path, {})
    state.recompute("exploration", "reports/r.json", {"sha256": "x"}, [])
    assert state.findings == []
    assert state.derived["reports/r.json"] == {"sha256": "x"}
    assert state.steps["exploration"]["facts"] == {
        "report": "reports/r.json",
        "sha256": "x",
    }
    assert state.steps["exploration"]["requires"] == ["question"]
    state.recompute("exploration", "reports/r.json", {"sha256": "y"}, [])
    state.recompute("experiments", "reports/none.json", {}, [])
    assert [(f.kind, f.step) for f in state.findings] == [
        ("fact_mismatch", "exploration"),
        ("missing_link", "experiments"),
    ]


def test_recompute_treats_an_unreadable_report_as_a_mismatch(tmp_path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "r.json").write_text("{bad")
    state = reader.Reader(tmp_path, {})
    state.recompute("exploration", "reports/r.json", {}, [])
    assert state.findings[0].kind == "fact_mismatch"
    assert state.findings[0].detail == (
        "reports/r.json is not what the retained records give"
    )


# ---- tracer ----------------------------------------------------------------


def test_the_committed_trail_derives_without_findings(folder):
    derived = tracer.derive(folder, KEYS)
    assert derived.findings == ()
    assert [s["kind"] for s in derived.steps] == list(finding.KINDS)
    assert all(value is not None for value in derived.reports.values())
    assert sorted(derived.reports) == [
        "experiment_comparison.json",
        "exploratory_report.json",
        "outcome_tracking.json",
    ]


def test_the_tracer_combines_the_three_step_families():
    assert issubclass(tracer.Tracer, planning.Planning)
    assert issubclass(tracer.Tracer, evidence.Evidence)
    assert issubclass(tracer.Tracer, release.Release)


def test_absent_inputs_are_a_missing_link_of_the_stakeholder(folder):
    (folder / "inputs.json").unlink()
    derived = tracer.derive(folder, KEYS)
    assert ("missing_link", "stakeholder") in found(derived)
    assert "inputs.json is absent" in details(derived)


def test_inputs_that_are_not_json_are_tampered(folder):
    (folder / "inputs.json").write_text("{nope", encoding="utf-8")
    derived = tracer.derive(folder, KEYS)
    assert "inputs.json is not JSON" in details(derived)
    assert ("record_tampered", "stakeholder") in found(derived)


# ---- planning --------------------------------------------------------------


def test_the_question_step_reads_the_signed_charter(folder):
    derived = tracer.derive(folder, KEYS)
    facts = derived.steps[0]["facts"]
    charter = json.loads((folder / "retained" / CHARTER).read_text())["charter"]
    assert facts["charter_hash"] == records.digest(charter)
    assert facts["target"] == "survived"
    assert facts["unit"] == "passenger_id"
    assert facts["prediction_moment"] == "boarding_date"
    assert derived.steps[0]["requires"] == []


def test_the_framing_step_reads_costs_and_the_threshold(folder):
    facts = tracer.derive(folder, KEYS).steps[1]["facts"]
    assert facts["costs"] == {
        "c_00": 0.0,
        "c_01": 5.0,
        "c_10": 1.0,
        "c_11": 0.0,
    }
    assert facts["threshold"] == pytest.approx(1 / 6)
    assert facts["framing_status"] == "issued_with_questions"
    assert facts["field_sources"] == ["template"]
    assert facts["llm_filled"] == []
    assert facts["data_checks_passed"] is True
    assert facts["open_questions"] == 0


def test_an_unsigned_charter_is_an_unsigned_question(folder):
    change(folder / "retained" / CHARTER, lambda d: d.pop("signature"))
    derived = tracer.derive(folder, KEYS)
    assert "the charter carries no signature" in details(derived)


def test_a_charter_without_an_analyst_key_is_unsigned(folder):
    keys = {k: v for k, v in KEYS.items() if k != "analyst"}
    derived = tracer.derive(folder, keys)
    assert "no analyst key supplied to check the charter" in details(derived)


def test_a_charter_signed_with_another_key_does_not_verify(folder):
    derived = tracer.derive(folder, {**KEYS, "analyst": "wrong-key"})
    assert "the charter signature does not verify" in details(derived)
    assert ("unsigned_step", "question") in found(derived)


def test_a_charter_hash_that_does_not_match_its_text_is_tampered(folder):
    def edit(data):
        data["charter_hash"] = "0" * 64

    change(folder / "retained" / CHARTER, edit)
    derived = tracer.derive(folder, KEYS)
    assert "the charter hash does not match its text" in details(derived)


def test_a_cost_model_other_than_the_charters_is_an_identity_mismatch(folder):
    def edit(data):
        data["c_01"] = 9.0

    change(folder / "retained" / "framing" / "decision_cost_model.json", edit)
    derived = tracer.derive(folder, KEYS)
    assert ("identity_mismatch", "framing") in found(derived)
    assert "the cost model on file is not the one in the charter" in details(
        derived
    )


def test_a_framing_draft_for_another_charter_is_an_identity_mismatch(folder):
    def edit(data):
        data["charter_hash"] = "other"

    change(folder / "retained" / "framing" / "framing_draft.json", edit)
    derived = tracer.derive(folder, KEYS)
    assert "the framing draft was written for another charter" in details(
        derived
    )


def test_a_stakeholder_section_signed_with_another_key_is_unsigned(folder):
    derived = tracer.derive(folder, {**KEYS, "stakeholder": "other"})
    assert ("unsigned_step", "stakeholder") in found(derived)
    stakeholder = derived.steps[2]
    assert stakeholder["facts"]["signed"] is False
    assert stakeholder["requires"] == ["question", "framing"]


def test_inputs_naming_another_charter_are_an_identity_mismatch(folder):
    def edit(data):
        data["charter_hash"] = "other"

    change(folder / "inputs.json", edit)
    derived = tracer.derive(folder, KEYS)
    assert "inputs.json names another charter hash" in details(derived)


def test_a_failed_acceptance_criterion_is_not_met(folder):
    def edit(data):
        data["stakeholder"]["acceptance"][0]["value"] = 10**9
        data["stakeholder"]["acceptance"][0]["op"] = ">="

    change(folder / "inputs.json", edit)
    derived = tracer.derive(folder, KEYS)
    met = [f for f in derived.findings if f.kind == "criterion_not_met"]
    assert met
    assert met[0].step == "stakeholder"
    assert ">= 1000000000 is not met" in met[0].detail


def test_verify_records_nothing_for_a_good_signature(tmp_path):
    state = planning.Planning(tmp_path, {"analyst": "k"})
    charter = {"q": 1}
    state.verify({"signature": signing.mac(charter, "k")}, charter, "question")
    assert state.findings == []


# ---- evidence --------------------------------------------------------------


def test_evidence_constants_are_the_documented_policy():
    assert evidence.MINIMUM_CASES == 100
    assert evidence.CUT_TOLERANCE == 1e-9


def test_the_evidence_steps_name_their_reports_and_dependencies(folder):
    steps = {s["kind"]: s for s in tracer.derive(folder, KEYS).steps}
    assert steps["exploration"]["facts"]["report"] == (
        "reports/exploratory_report.json"
    )
    assert steps["experiments"]["facts"]["report"] == (
        "reports/experiment_comparison.json"
    )
    assert "run/trials/trial_001.json" in steps["experiments"]["records"]
    assert steps["quality"]["requires"] == [
        "stakeholder",
        "exploration",
        "experiments",
    ]
    assert steps["selection"]["facts"]["selected_model"] == "stacking"
    assert steps["selection"]["facts"]["n_test"] == 217
    outcomes = steps["outcomes"]["facts"]
    assert outcomes["status"] == "within"
    assert outcomes["cases"] == 217
    assert outcomes["approved_range"] == [0.28, 0.52]


def test_every_quality_threshold_passes_on_the_committed_trail(folder):
    steps = {s["kind"]: s for s in tracer.derive(folder, KEYS).steps}
    thresholds = steps["quality"]["facts"]["thresholds"]
    assert thresholds
    assert all(t["passed"] for t in thresholds)
    assert thresholds[0]["observed"] == 1309


def test_a_quality_section_signed_with_another_key_is_unsigned(folder):
    derived = tracer.derive(folder, {**KEYS, "reviewer": "other"})
    assert ("unsigned_step", "quality") in found(derived)


def test_a_reviewed_record_edited_afterwards_is_tampered(folder):
    def edit(data):
        data["row_count"] = data["row_count"] + 1

    change(folder / "retained" / "run" / "dataset_manifest.json", edit)
    derived = tracer.derive(folder, KEYS)
    assert any(
        f.kind == "record_tampered"
        and "changed after the reviewer signed" in f.detail
        for f in derived.findings
    )


def test_an_edited_stored_report_is_a_mismatch(folder):
    change(
        folder / "reports" / "exploratory_report.json", lambda d: d.update(x=1)
    )
    derived = tracer.derive(folder, KEYS)
    assert ("fact_mismatch", "exploration") in found(derived)


def test_outcomes_outside_the_approved_range_are_reported(folder):
    def edit(data):
        data["stakeholder"]["expectation"]["low"] = 0.0
        data["stakeholder"]["expectation"]["high"] = 0.1

    change(folder / "inputs.json", edit)
    derived = tracer.derive(folder, KEYS)
    outside = [
        f for f in derived.findings if f.kind == "outcome_outside_expectation"
    ]
    assert len(outside) == 1
    assert "observed cost per case 0.401" in outside[0].detail
    assert "is above the approved range [0.0, 0.1]" in outside[0].detail


# ---- release ---------------------------------------------------------------


def test_the_release_steps_read_the_registry_and_the_ledger(folder):
    steps = {s["kind"]: s for s in tracer.derive(folder, KEYS).steps}
    approval = steps["approval"]["facts"]
    assert approval["model"] == "titanic_survival"
    assert approval["version"] == 1
    assert approval["bundle_signed"] is True
    assert approval["champion_version"] == 1
    assert approval["approval"]["decision"] == "approved"
    assert steps["approval"]["requires"] == ["selection"]
    deployment = steps["deployment"]["facts"]
    assert deployment["ledger_rows"] == 8
    assert deployment["row_kinds"] == {"observe_summary": 7, "open": 1}
    assert deployment["batches"] == 7
    assert deployment["cases"] == 217
    assert deployment["answered"] == 217
    assert deployment["escalated"] == 0
    assert deployment["halted"] is False
    assert steps["deployment"]["requires"] == ["approval"]


def test_sanction_requires_every_signed_field():
    record = {
        "model": "m",
        "version": 1,
        "content_id": "c" * 64,
        "reviewer": "r",
        "decision": "approved",
        "reason": "ok",
        "evidence": ["e"],
        "approved_at": "2026-01-01T00:00:00+00:00",
        "independent": True,
    }
    made = release.sanction(record)
    assert made.model == "m"
    assert made.evidence == ("e",)
    assert made.signature is None
    record.pop("reason")
    with pytest.raises(KeyError):
        release.sanction(record)


def test_a_missing_registry_is_a_missing_link_of_the_approval(folder):
    (folder / "retained" / "registry" / "registry.json").unlink()
    derived = tracer.derive(folder, KEYS)
    assert ("missing_link", "approval") in found(derived)


def test_an_approval_checked_with_another_key_is_unsigned(folder):
    derived = tracer.derive(folder, {**KEYS, "reviewer": "other"})
    assert [(f.kind, f.step, f.detail) for f in derived.findings] == [
        ("unsigned_step", "quality", "the reviewer signature does not verify"),
        ("unsigned_step", "approval", "the approval signature does not verify"),
        (
            "unapproved_step",
            "approval",
            "champion promotion requires a valid signature",
        ),
    ]


def test_an_edited_ledger_row_breaks_the_deployment_chain(folder):
    path = folder / "retained" / "deployment" / "deployment_ledger.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["seq"] = 99
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    derived = tracer.derive(folder, KEYS)
    assert ("broken_chain", "deployment") in found(derived)


# ---- builder ---------------------------------------------------------------


def test_collect_copies_only_existing_files_with_their_paths(tmp_path):
    source = tmp_path / "in"
    (source / "deep").mkdir(parents=True)
    (source / "deep" / "a.json").write_text("A")
    (source / "dir").mkdir()
    target = tmp_path / "out"
    builder.collect(source, target, ("deep/a.json", "gone.json", "dir"))
    assert (target / "deep" / "a.json").read_text() == "A"
    assert not (target / "gone.json").exists()
    assert not (target / "dir").exists()


def test_build_reproduces_a_verified_trail(folder, tmp_path):
    base = folder / "retained"
    output = tmp_path / "rebuilt"
    result = builder.build(
        output,
        run_dir=base / "run",
        framing_dir=base / "framing",
        registry_dir=base / "registry",
        deployment_dir=base / "deployment",
        inputs_file=folder / "inputs.json",
        keys=KEYS,
    )
    assert result.ok
    assert result.steps == 10
    assert (output / "trail.json").read_text() == (
        folder / "trail.json"
    ).read_text()
    assert (output / "retained" / "run" / "trials" / "trial_001.json").is_file()
    assert (output / "inputs.json").read_text() == (
        folder / "inputs.json"
    ).read_text()
