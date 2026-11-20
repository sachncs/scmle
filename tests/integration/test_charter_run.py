"""A run governed by an analysis charter: custody, artifacts and limits."""

import json
import pathlib

import pytest
import support

from scmle.assurance.charter import sealing
import scmle.assurance.charter.contracts
from scmle.assurance.custody import sealed
import scmle.assurance.custody.manifest
from scmle.contracts import errors
from tests.unit.test_charter_contracts import make_charter

KEY = "integration-key"


def inscribe(tmp_path, **overrides):
    values = {"effect": 0.5, "quota": 3}
    values.update(overrides)
    path = tmp_path / "charter.json"
    sealing.inscribe(path, make_charter(**values), signing_key=KEY)
    return path


def run(frame, tmp_path, charter=None, **kwargs):
    return support.make_project(
        frame,
        "y",
        budget=300,
        directory=tmp_path / "out",
        charter=charter,
        **kwargs,
    ).run()


def events(run_dir):
    lines = (pathlib.Path(run_dir) / "events.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def event(run_dir, kind):
    return next(e for e in events(run_dir) if e["kind"] == kind)


def test_a_charter_run_seals_c_even_in_cooperative_mode(binary_frame, tmp_path):
    # Enforces website/src/pages/docs/threat-model.astro:24.
    result = run(binary_frame, tmp_path, inscribe(tmp_path))

    custody = event(result.directory, "custody.sealed")["data"]
    assert custody["isolation"] == {
        "C": "vault process",
        "final_test": "vault process",
    }
    assert custody["confirmation_rows"] > 0
    assert custody["scope"] == "confirmatory"
    created = event(result.directory, "split.created")["data"]
    assert created["test_set_isolation"] == "separate vault process"


def test_the_charter_artifacts_are_written_and_verifiable(
    binary_frame, tmp_path
):
    result = run(binary_frame, tmp_path, inscribe(tmp_path))
    folder = pathlib.Path(result.directory)

    for name in (
        "analysis_charter.json",
        "split_manifest.json",
        "error_budget.json",
        "feasibility.json",
        "custody_log.jsonl",
    ):
        assert (folder / name).exists(), name
    loaded = scmle.assurance.custody.manifest.restore(
        json.loads((folder / "split_manifest.json").read_text())
    )
    assert len(loaded.sha256) == 64
    assert sealing.read(
        folder / "analysis_charter.json", signing_key=KEY
    ).signed
    sealed.Journal(folder / "custody_log.jsonl")


def test_profiling_sees_exploration_rows_only(binary_frame, tmp_path):
    plain = run(binary_frame, tmp_path / "plain")
    governed = run(binary_frame, tmp_path, inscribe(tmp_path))

    def profiled(result):
        folder = pathlib.Path(result.directory)
        report = json.loads((folder / "curation_report.json").read_text())
        return report["rows"]["profiled"]

    manifest = json.loads(
        (pathlib.Path(governed.directory) / "split_manifest.json").read_text()
    )
    assert profiled(governed) == len(manifest["sets"]["exploration"])
    assert profiled(governed) < profiled(plain)
    assert set(manifest["sets"]["final"]) == set(
        json.loads(
            (
                pathlib.Path(plain.directory) / "validation_manifest.json"
            ).read_text()
        )["test"]
    )


def test_the_final_test_still_opens_once_and_scores(binary_frame, tmp_path):
    result = run(binary_frame, tmp_path, inscribe(tmp_path))
    assert result.evaluation["opened"] is True
    assert result.evaluation["rows"]["test"] > 0


def test_a_run_without_a_charter_is_unchanged(binary_frame, tmp_path):
    result = run(binary_frame, tmp_path)
    folder = pathlib.Path(result.directory)
    assert not (folder / "split_manifest.json").exists()
    assert not (folder / "custody_log.jsonl").exists()
    created = event(result.directory, "split.created")["data"]
    assert created["test_set_isolation"] == "in-process protocol"
    assert all(e["kind"] != "custody.sealed" for e in events(result.directory))


def test_a_tampered_charter_stops_the_run_before_profiling(
    binary_frame, tmp_path
):
    path = inscribe(tmp_path)
    record = json.loads(path.read_text())
    record["charter"]["question"] = "A different question?"
    path.write_text(json.dumps(record))

    with pytest.raises(errors.ArtifactError, match="hash"):
        run(binary_frame, tmp_path, path)
    runs = list((tmp_path / "out").glob("run_*"))
    assert runs and not (runs[0] / "dataset_profile.json").exists()


def test_too_few_rows_run_unconfirmed_instead_of_failing(tmp_path):
    import numpy
    import pandas

    rng = numpy.random.default_rng(1)
    frame = pandas.DataFrame(
        {"a": rng.normal(size=120), "y": (rng.random(120) < 0.4).astype(int)}
    )
    result = run(frame, tmp_path, inscribe(tmp_path))
    custody = event(result.directory, "custody.sealed")["data"]
    assert custody["confirmation_rows"] == 0
    assert custody["scope"] == "exploratory_unconfirmed"
    assert custody["isolation"] == {"final_test": "vault process"}


def test_a_missing_charter_file_is_a_config_error(binary_frame, tmp_path):
    with pytest.raises(errors.ConfigError):
        run(binary_frame, tmp_path, tmp_path / "nope.json")


def test_a_charter_run_writes_the_ledger_control_and_claims(
    binary_frame, tmp_path
):
    result = run(binary_frame, tmp_path, inscribe(tmp_path))
    folder = pathlib.Path(result.directory)
    for name in (
        "hypothesis_ledger.jsonl",
        "negative_control.json",
        "confirmed_claims.json",
    ):
        assert (folder / name).exists(), name
    done = event(result.directory, "ledger.completed")["data"]
    claims = json.loads((folder / "confirmed_claims.json").read_text())
    assert done["ledger_head"] == claims["heads"]["ledger"]
    import scmle.assurance.ledger.ledger

    ledger = scmle.assurance.ledger.ledger.Ledger.load(
        folder / "hypothesis_ledger.jsonl", expected_head=done["ledger_head"]
    )
    # Only numeric features are candidates; the categorical one is skipped.
    assert {h.spec.column for h in ledger.hypotheses.values()} == {"a", "b"}
    assert claims["outcome"] == done["outcome"]
    log = sealed.Journal(folder / "custody_log.jsonl")
    events_c = [e["event"] for e in log.entries if e["set"] == "C"]
    if claims["outcome"] == "confirmed":
        assert events_c.count("open") == 1
        assert "selection_frozen" in events_c
        assert events_c.index("selection_frozen") < events_c.index("open")
    else:
        assert "open" not in events_c


def test_a_run_without_a_charter_has_no_ledger(binary_frame, tmp_path):
    result = run(binary_frame, tmp_path)
    folder = pathlib.Path(result.directory)
    assert not (folder / "hypothesis_ledger.jsonl").exists()
    assert all(e["kind"] != "ledger.completed" for e in events(result.directory))


def leaky_frame(binary_frame):
    frame = binary_frame.copy()
    frame["echo"] = frame["y"] * 1.0
    return frame


def test_a_charter_run_writes_the_admissibility_artifacts(
    binary_frame, tmp_path
):
    # Enforces docs/WHEN_THINGS_GO_WRONG.md:269.
    # Enforces website/src/learn/when-things-go-wrong.html:125.
    result = run(binary_frame, tmp_path, inscribe(tmp_path))
    folder = pathlib.Path(result.directory)

    table = json.loads((folder / "admissibility_table.json").read_text())
    assert table["kind"] == "predictive" and table["mode"] == "permissive"
    assert {d["name"] for d in table["decisions"]} == {"a", "b", "c"}
    assert all(d["admission"] == "admit_flagged" for d in table["decisions"])
    copied = json.loads((folder / "detector_certificates.json").read_text())
    assert copied["certificates"]
    done = event(result.directory, "admissibility.completed")["data"]
    assert done["quarantined"] == {}
    assert set(done["detector_status"]) >= {"f5-sentinel", "f5-missingness"}
    page = (folder / "eda_report.html").read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>") and "<script" not in page
    assert "Stability" in page and "Shift" in page


def test_a_certified_leakage_suspect_is_quarantined_and_ledgered(
    binary_frame, tmp_path
):
    result = run(leaky_frame(binary_frame), tmp_path, inscribe(tmp_path))
    folder = pathlib.Path(result.directory)

    done = event(result.directory, "admissibility.completed")["data"]
    assert done["leakage_gate"] == "gating"
    assert "echo" in done["suspects"] and set(done["quarantined"]) == {"echo"}
    rows = [
        json.loads(line)
        for line in (folder / "hypothesis_ledger.jsonl")
        .read_text()
        .splitlines()
    ]
    diagnostics = [r for r in rows if r["kind"] == "diagnostic"]
    assert diagnostics
    assert {r["body"]["check"]["subset"] for r in diagnostics} == {"X"}
    plan = json.loads((folder / "dataset_profile.json").read_text())
    assert "echo" not in plan["plan"]["numeric"]
    assert "quarantined by admissibility" in plan["plan"]["dropped"]["echo"]


def test_the_curation_report_and_card_list_the_final_feature_plan(
    binary_frame, tmp_path
):
    result = run(leaky_frame(binary_frame), tmp_path, inscribe(tmp_path))
    folder = pathlib.Path(result.directory)

    plan = json.loads((folder / "dataset_profile.json").read_text())
    curation = json.loads((folder / "curation_report.json").read_text())
    card = json.loads((folder / "dataset_card.json").read_text())
    for key in ("numeric", "categorical", "text"):
        assert curation["features"][key] == plan["plan"][key]
        assert card["features"][key] == plan["plan"][key]
    assert "echo" not in curation["features"]["numeric"]
    assert "quarantined by admissibility" in curation["dropped"]["echo"]


def test_a_stale_certificate_no_longer_keeps_a_suspect_field(
    binary_frame, tmp_path, monkeypatch
):
    # ERR-1: a stale certificate used to make the screen descriptive, so the
    # suspect field stayed in the features. Flags now act regardless; the
    # stale certificate is only reported.
    from scmle.assurance.admissibility import screen

    monkeypatch.setattr(screen, "fingerprint", lambda: "0" * 64)
    result = run(leaky_frame(binary_frame), tmp_path, inscribe(tmp_path))

    done = event(result.directory, "admissibility.completed")["data"]
    assert done["leakage_gate"] == "gating"
    assert "echo" in done["suspects"] and "echo" in done["quarantined"]


def test_a_valid_attestation_keeps_a_certified_suspect_and_is_ledgered(
    binary_frame, tmp_path
):
    # Enforces docs/WHEN_THINGS_GO_WRONG.md:318.
    # Enforces website/src/learn/when-things-go-wrong.html:150.
    attestation = {
        "column": "echo",
        "lead_seconds": 60.0,
        "name_or_id": "A. Owner",
        "role": "data owner",
        "statement": "echo is the clinic's own earlier reading",
        "evidence": "ticket DATA-7",
        "date": "2020-01-01",
        "scope": make_charter(effect=0.5, quota=3).digest,
        "expires": "2099-01-01",
    }
    path = tmp_path / "attestations.json"
    path.write_text(
        json.dumps({"format_version": 1, "attestations": [attestation]})
    )
    result = run(
        leaky_frame(binary_frame),
        tmp_path,
        inscribe(tmp_path),
        attestations=path,
    )
    folder = pathlib.Path(result.directory)

    done = event(result.directory, "admissibility.completed")["data"]
    assert "echo" in done["suspects"] and done["quarantined"] == {}
    rows = [
        json.loads(line)
        for line in (folder / "hypothesis_ledger.jsonl")
        .read_text()
        .splitlines()
    ]
    overrides = [
        r["body"]["check"]
        for r in rows
        if r["kind"] == "diagnostic"
        and r["body"]["check"]["identifier"] == "leakage.attestation:echo"
    ]
    assert [c["tier"] for c in overrides] == ["attested"]
    table = json.loads((folder / "admissibility_table.json").read_text())
    echo = next(d for d in table["decisions"] if d["name"] == "echo")
    assert echo["tier"] == "attested" and echo["admission"] == "admit_flagged"
    assert "PROVISIONAL" not in result.summary()


def test_a_malformed_attestation_file_is_a_config_error(binary_frame, tmp_path):
    path = tmp_path / "attestations.json"
    path.write_text(json.dumps({"format_version": 1}))
    project = support.make_project(
        binary_frame,
        "y",
        directory=tmp_path / "out",
        charter=inscribe(tmp_path),
        attestations=path,
    )
    with pytest.raises(errors.ConfigError, match="leakage attestations"):
        project.plan()


def test_a_run_without_a_charter_writes_no_admissibility_artifacts(
    binary_frame, tmp_path
):
    result = run(binary_frame, tmp_path)
    folder = pathlib.Path(result.directory)

    assert not (folder / "admissibility_table.json").exists()
    assert not (folder / "detector_certificates.json").exists()
    assert all(
        e["kind"] != "admissibility.completed" for e in events(result.directory)
    )


RICH_MULTIVERSE = (
    scmle.assurance.charter.contracts.Choice(
        "imputation", ("median", "complete_case"), "median"
    ),
    scmle.assurance.charter.contracts.Choice(
        "outlier_rule", ("none", "winsorize", "iqr_drop"), "none"
    ),
    scmle.assurance.charter.contracts.Choice(
        "inclusion", ("all", "first_half", "second_half"), "all"
    ),
)


def test_a_charter_run_writes_valid_stability_and_shift_reports(
    binary_frame, tmp_path
):
    # Enforces docs/ARTIFACTS.md:43.
    # Enforces website/src/pages/docs/artifacts.astro:27.
    import scmle.assurance.robustness.report

    result = run(binary_frame, tmp_path, inscribe(tmp_path))
    folder = pathlib.Path(result.directory)

    stability = json.loads((folder / "stability_report.json").read_text())
    shifted = json.loads((folder / "shift_report.json").read_text())
    scmle.assurance.robustness.report.conform(stability)
    scmle.assurance.robustness.report.validate(shifted)
    # The default charter declares one implemented choice with one usable
    # option, so no finding can be called stable.
    assert stability["status"] == "not_evaluated"
    assert "specification" in stability["reason"]
    assert shifted["status"] == "unverifiable"
    assert shifted["forward"]["status"] == "skipped"
    done = event(result.directory, "robustness.completed")["data"]
    assert done["shift"] == "unverifiable"
    assert done["stability"] == "not_evaluated"


def test_survivors_are_rerun_over_the_declared_multiverse(
    binary_frame, tmp_path
):
    import scmle.assurance.ledger.ledger

    path = inscribe(tmp_path, multiverse=RICH_MULTIVERSE)
    result = run(binary_frame, tmp_path, path)
    folder = pathlib.Path(result.directory)

    report = json.loads((folder / "stability_report.json").read_text())
    done = event(result.directory, "ledger.completed")["data"]
    ledger = scmle.assurance.ledger.ledger.Ledger.load(
        folder / "hypothesis_ledger.jsonl", expected_head=done["ledger_head"]
    )
    survivors = {h.spec.column for h in ledger.survivors()}
    assert survivors
    assert report["status"] == "evaluated"
    assert {f["name"] for f in report["findings"]} == survivors
    assert report["grid"]["run"] == 12 and report["grid"]["total"] == 18
    assert report["grid"]["subset"] is True
    first = report["findings"][0]
    assert len(first["specs"]) == 12 and first["attribution"]
    assert report["certificate"]["gate"] in {"gating", "descriptive"}
    assert report["confirmation_filter"].startswith("not applied")


def test_a_reference_sample_gives_a_pass_or_fail_never_unverifiable(
    binary_frame, tmp_path
):
    # Enforces docs/ARTIFACTS.md:43.
    # Enforces website/src/pages/docs/artifacts.astro:27.
    reference = tmp_path / "reference.csv"
    binary_frame.drop(columns="y").head(200).to_csv(reference, index=False)
    result = run(
        binary_frame,
        tmp_path,
        inscribe(tmp_path),
        reference=str(reference),
        sensitive=["c"],
    )
    shifted = json.loads(
        (pathlib.Path(result.directory) / "shift_report.json").read_text()
    )
    assert shifted["status"] in {"pass", "fail"}
    assert shifted["c2st"]["identifier"] == "f6-shift-c2st"
    assert shifted["mmd"]["subset"] == "X"
    assert {row["column"] for row in shifted["coverage"]} == {"c"}
    assert shifted["periods"] == {
        "status": "unverifiable",
        "reason": "the data have no time column",
        "column": None,
        "periods": [],
    }


def test_a_timed_charter_run_reports_drift_by_period(binary_frame, tmp_path):
    timed = binary_frame.assign(t=range(len(binary_frame)))
    timed.loc[timed.t >= 300, "b"] += 3.0

    result = run(
        timed,
        tmp_path,
        inscribe(tmp_path),
        timestamp="t",
        validation="temporal",
    )
    shifted = json.loads(
        (pathlib.Path(result.directory) / "shift_report.json").read_text()
    )

    periods = shifted["periods"]
    assert periods["status"] == "fail"
    assert periods["column"] == "t"
    assert periods["periods"][0]["index"] == 0
    assert periods["periods"][0]["start"] == "0"
    assert sum(p["rows"] for p in periods["periods"]) < len(timed)
    assert "b" in periods["periods"][-1]["rejected"]


def test_a_run_without_a_charter_has_no_robustness_reports(
    binary_frame, tmp_path
):
    result = run(binary_frame, tmp_path)
    folder = pathlib.Path(result.directory)
    assert not (folder / "stability_report.json").exists()
    assert not (folder / "shift_report.json").exists()
    assert all(
        e["kind"] != "robustness.completed" for e in events(result.directory)
    )


def test_chartered_rules_drive_the_ledger_and_the_second_stage(
    binary_frame, tmp_path
):
    import scmle.assurance.charter.rules
    import scmle.assurance.ledger.ledger

    rules = scmle.assurance.charter.rules.Procedure(
        ledger="lord++", confirmation="bh", stability=True
    )
    path = inscribe(tmp_path, multiverse=RICH_MULTIVERSE, rules=rules)
    result = run(binary_frame, tmp_path, path)
    folder = pathlib.Path(result.directory)

    done = event(result.directory, "ledger.completed")["data"]
    assert done["rules"] == "chartered"
    assert done["rule"] == "lord++" and done["withdrawn"] is not None
    ledger = scmle.assurance.ledger.ledger.Ledger.load(
        folder / "hypothesis_ledger.jsonl", expected_head=done["ledger_head"]
    )
    assert ledger.confirmation["rule"] == "bh"
    claims = json.loads((folder / "confirmed_claims.json").read_text())
    stage_two = claims["second"]
    assert stage_two["rule"]["requested"] is True
    assert stage_two["family"] == [
        i for i in stage_two["survivors"] if i not in stage_two["withdrawn"]
    ]
    stability = json.loads((folder / "stability_report.json").read_text())
    assert stability["confirmation_filter"] == stage_two["rule"]["note"]
    assert any(
        c.identifier.startswith("f6-stability:") for c in ledger.diagnostics
    )
    robustness = event(result.directory, "robustness.completed")["data"]
    assert robustness["stability_filter"] == stage_two["rule"]["note"]


def test_a_charter_without_rules_is_reported_as_not_chartered(
    binary_frame, tmp_path
):
    result = run(binary_frame, tmp_path, inscribe(tmp_path))
    done = event(result.directory, "ledger.completed")["data"]
    assert done["rules"] == "not chartered" and done["rule"] == "e-lond"
    folder = pathlib.Path(result.directory)
    claims = json.loads((folder / "confirmed_claims.json").read_text())
    assert "second" not in claims


def test_a_specification_that_contradicts_the_chartered_rule_is_refused():
    # Enforces docs/CLI.md:146.
    # Enforces website/src/pages/docs/cli.astro:110.
    from scmle.pipeline import stages

    assert stages.governing("x", "by", None, "by") == "by"
    assert stages.governing("x", "by", "bh", "by") == "bh"
    assert stages.governing("x", "bh", "bh", "by") == "bh"
    with pytest.raises(errors.ConfigError, match="charter fixes"):
        stages.governing("x", "holm", "bh", "by")


def test_the_summary_and_report_put_leaks_and_findings_first(
    binary_frame, tmp_path
):
    # Enforces docs/BEGINNER_FINDINGS.md:28.
    # Enforces website/src/learn/beginner-findings.html:13.
    result = run(leaky_frame(binary_frame), tmp_path, leakage="warn")

    assert "PROVISIONAL: 1 flagged column(s) not attested: echo" in (
        result.summary()
    )
    page = (pathlib.Path(result.directory) / "report.html").read_text()
    assert page.index("<h2>Findings</h2>") < page.index("<h2>Leaderboard</h2>")
    assert page.index("<h2>Leaderboard</h2>") < page.index("<details>")


def test_the_summary_counts_claims_by_tier(binary_frame, tmp_path):
    result = run(binary_frame, tmp_path, inscribe(tmp_path))
    claims = {"claims": [{"tier": "confirmed"}, {"tier": "confirmed"}]}
    (pathlib.Path(result.directory) / "confirmed_claims.json").write_text(
        json.dumps(claims)
    )

    assert result.tally() == {"confirmed": 2}
    assert result.summary().endswith("claims: 2 confirmed")
