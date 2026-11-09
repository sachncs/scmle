"""A causal charter run: the identification stage and its files."""

import dataclasses
import json
import pathlib

import numpy
import pandas
import pytest
import support

from scmle import cli
from scmle.artifacts import claims_audit
from scmle.contracts import errors
import scmle.data.admissibility.dag
from scmle.data.charter import causal
from scmle.data.charter import sealing
from scmle.data.identification import attest
import scmle.data.identification.run
import scmle.data.ledger.ledger
from tests.unit.test_charter_contracts import make_charter

KEY = "identification-key"
ASSUMPTIONS = (
    "unconfoundedness_given:x1,x2",
    "positivity",
    "no_interference",
)


def causal_frame() -> pandas.DataFrame:
    rng = numpy.random.default_rng(0)
    rows = 600
    x1, x2 = rng.normal(size=rows), rng.normal(size=rows)
    t = (rng.uniform(size=rows) < 1 / (1 + numpy.exp(-0.5 * x1))).astype(int)
    y = 2.0 * t + x1 - 0.5 * x2 + rng.normal(size=rows)
    return pandas.DataFrame({"x1": x1, "x2": x2, "t": t, "y": y})


def charter_file(
    tmp_path, assumptions, dag=None, subgroups=(), attestors=(), signed=None
) -> pathlib.Path:
    base = make_charter(min_effect=0.5, k_max=3)
    estimand = dataclasses.replace(
        base.estimand,
        kind="causal",
        causal=causal.CausalSpec(
            treatment="t",
            outcome="y",
            estimand="ATE",
            assumptions=assumptions,
            strategies=("backdoor_aipw",),
            dag=dag,
            subgroups=subgroups,
            attestors=attestors,
            attestations=None if signed is None else str(signed),
        ),
    )
    path = tmp_path / "charter.json"
    sealing.write_charter(
        path, make_charter(estimand=estimand, min_effect=0.5, k_max=3), KEY
    )
    return path


def run(
    tmp_path,
    assumptions,
    dag=None,
    subgroups=(),
    frame=None,
    attestors=(),
    signed=None,
):
    return support.make_project(
        causal_frame() if frame is None else frame,
        "y",
        time_budget=300,
        output_dir=tmp_path / "out",
        charter=charter_file(
            tmp_path, assumptions, dag, subgroups, attestors, signed
        ),
    ).run()


def read(result, name):
    return json.loads((pathlib.Path(result.run_dir) / name).read_text())


def test_an_identified_charter_estimates_on_the_single_opening_of_c(tmp_path):
    result = run(tmp_path, ASSUMPTIONS)
    plan = read(result, "identification_plan.json")
    report = read(result, "identification_report.json")
    claims = read(result, "causal_claims.json")
    assert plan["primary"] == "backdoor_aipw"
    estimate = report["estimate"]
    assert estimate["low"] < 2.0 < estimate["high"]
    assert abs(estimate["point"] - 2.0) < 0.6
    assert claims["status"] == "identified"
    assert claims["tier"] != "confirmed"
    assert "no named attestor" in " ".join(claims["reasons"])
    assert set(claims["hashes"]) == {
        "identification_plan.json",
        "identification_report.json",
        "falsification_report.json",
        "sensitivity_report.json",
    }
    folder = pathlib.Path(result.run_dir)
    log = (folder / "custody_log.jsonl").read_text()
    assert log.count('"event":"open"') + log.count('"event": "open"') == 1
    ledger = scmle.data.ledger.ledger.Ledger.load(
        folder / scmle.data.ledger.ledger.FILE_NAME
    )
    registered = ledger.claims["causal_primary_theta_zero"]
    assert registered["kind"] == "causal_primary"
    sealed = ledger.confirmation["attachments"]
    assert set(sealed) == set(scmle.data.identification.run.DOCUMENTS)
    for name, expected in sealed.items():
        assert scmle.data.ledger.ledger.digest(read(result, name)) == expected
    assert ledger.confirmation["claims"] == {
        "causal_primary_theta_zero": {
            "kind": "causal_primary",
            "sha256": registered["sha256"],
            "status": "identified",
        }
    }


def test_claims_prints_the_identification_status_and_verifies(tmp_path, capsys):
    result = run(tmp_path, ASSUMPTIONS)

    assert cli.main(["claims", "list", str(result.run_dir)]) == 0
    printed = capsys.readouterr().out
    assert "registered causal_primary" in printed
    assert "identification=identified" in printed
    verified = claims_audit.verify(result.run_dir)
    names = {f.name: f.status for f in verified.findings}
    assert names["registered claims: files hash to the sealed ledger"] == "pass"
    assert verified.ok

    folder = pathlib.Path(result.run_dir)
    report = json.loads((folder / "identification_report.json").read_text())
    report["estimate"]["point"] = 0.0
    (folder / "identification_report.json").write_text(json.dumps(report))
    assert not claims_audit.verify(result.run_dir).ok


def test_a_declared_dag_is_loaded_checked_and_lifts_the_adjustment_cap(
    tmp_path,
):
    graph = scmle.data.admissibility.dag.Dag(
        ("t", "x1", "x2", "y"),
        (("t", "y"), ("x1", "t"), ("x1", "y"), ("x2", "y")),
        (),
        "test",
        "none",
    )
    path = tmp_path / "dag.json"
    path.write_text(json.dumps(graph.artifact()))
    result = run(tmp_path, ASSUMPTIONS, dag=str(path))

    report = read(result, "identification_report.json")
    assert report["dag"]["dag_hash"] == graph.digest()
    assert report["dag"]["falsification"]["ran"] is True
    assert report["dag"]["adjustment_members_refused"] == {}
    plan = read(result, "identification_plan.json")
    assert plan["primary"] == "backdoor_aipw"
    table = read(result, "admissibility_table.json")
    assert table["kind"] == "causal" and table["dag_hash"] == graph.digest()
    claims = read(result, "causal_claims.json")
    assert "adjustment set is an unchecked attestation" not in " ".join(
        read(result, "identification_report.json")["notes"]
    )
    assert claims["status"] == "identified"


def test_an_adjustment_member_that_is_a_descendant_makes_backdoor_ineligible(
    tmp_path,
):
    graph = scmle.data.admissibility.dag.Dag(
        ("t", "x1", "x2", "y"),
        (("t", "x2"), ("t", "y"), ("x1", "t"), ("x1", "y")),
        (),
        "test",
        "none",
    )
    path = tmp_path / "dag.json"
    path.write_text(json.dumps(graph.artifact()))
    result = run(tmp_path, ASSUMPTIONS, dag=str(path))

    claims = read(result, "causal_claims.json")
    report = read(result, "identification_report.json")
    assert claims["status"] == "not_identified"
    assert "x2" in report["dag"]["adjustment_members_refused"]


def test_a_dag_that_cannot_be_loaded_is_a_config_error(tmp_path):
    path = tmp_path / "dag.json"
    path.write_text("{}")
    with pytest.raises(errors.ConfigError, match="declared DAG"):
        run(tmp_path, ASSUMPTIONS, dag=str(path))


def test_a_charter_no_strategy_identifies_is_refused_without_an_estimate(
    tmp_path,
):
    result = run(tmp_path, ("positivity",))
    claims = read(result, "causal_claims.json")
    report = read(result, "identification_report.json")
    assert claims["status"] == "not_identified" and claims["tier"] is None
    assert report["estimate"] is None and report["action"] == "refuse_claim"
    assert result.evaluation["test_set_opened"] is True


def test_subgroup_hypotheses_are_registered_and_reported_in_the_causal_files(
    tmp_path,
):
    frame = causal_frame()
    frame["seg"] = numpy.where(frame["x1"] > 0, "hi", "lo")
    plain = run(tmp_path / "plain", ASSUMPTIONS, frame=frame)
    split = run(
        tmp_path / "split",
        ASSUMPTIONS,
        subgroups=(causal.Subgroup("seg", ("hi", "lo")),),
        frame=frame,
    )
    report = read(split, "identification_report.json")["subgroups"]
    assert report["rule"] == "holm" and report["family_size"] == 2
    assert {r["value"] for r in report["hypotheses"]} == {"hi", "lo"}
    assert report["rejected"] == ["seg=hi", "seg=lo"]
    claims = read(split, "causal_claims.json")
    assert claims["subgroups"]["family_size"] == 2
    assert "subgroups" not in read(plain, "causal_claims.json")
    registered = {}
    for name, result in (("plain", plain), ("split", split)):
        folder = pathlib.Path(result.run_dir)
        ledger = scmle.data.ledger.ledger.Ledger.load(
            folder / scmle.data.ledger.ledger.FILE_NAME
        )
        registered[name] = ledger.claims["causal_primary_theta_zero"]["sha256"]
    assert registered["plain"] != registered["split"]
    assert claims_audit.verify(split.run_dir).ok


ATTESTORS = (
    causal.Attestor("Dr. A. Rivera", "study statistician", ASSUMPTIONS[:2]),
    causal.Attestor("J. Okafor", "domain lead", ASSUMPTIONS[2:]),
)


def write_attestations(tmp_path, key=KEY, skip=()):
    """Signed statements bound to the hash of the charter on disk."""
    signed = tmp_path / "attestations.json"
    charter = charter_file(
        tmp_path, ASSUMPTIONS, attestors=ATTESTORS, signed=signed
    )
    scope = sealing.read_verified(charter).charter_hash
    items = []
    for number, assumption in enumerate(ASSUMPTIONS):
        if assumption in skip:
            continue
        item = attest.Attestation(
            assumption,
            ATTESTORS[0 if number < 2 else 1].name,
            "I judge this assumption to hold for the study population",
            "design memo 2026-10",
            "2020-01-01",
            "2099-01-01",
            scope,
        ).signed(key)
        items.append(item.to_dict())
    signed.write_text(json.dumps({"format_version": 1, "attestations": items}))
    return signed


def attested_run(tmp_path, monkeypatch, key=KEY, skip=()):
    monkeypatch.setenv("SCMLE_SIGNING_KEY", KEY)
    signed = write_attestations(tmp_path, key, skip)
    return run(tmp_path, ASSUMPTIONS, attestors=ATTESTORS, signed=signed)


def test_verified_attestations_lift_the_claim_above_unverified(
    tmp_path, monkeypatch
):
    result = attested_run(tmp_path, monkeypatch)
    claims = read(result, "causal_claims.json")
    assert claims["tier"] in ("attested", "confirmed")
    assert "no named attestor" not in " ".join(claims["reasons"])
    assert set(claims["assumptions"].values()) == {
        "Dr. A. Rivera",
        "J. Okafor",
    }
    assert len(claims["attestations"]["accepted"]) == 3
    verified = claims_audit.verify(result.run_dir, KEY)
    names = {f.name: f.status for f in verified.findings}
    assert names["causal attestations verify"] == "pass"
    assert verified.ok


def test_a_forged_or_missing_attestation_leaves_the_claim_unverified(
    tmp_path, monkeypatch
):
    forged = attested_run(tmp_path / "forged", monkeypatch, key="forger")
    claims = read(forged, "causal_claims.json")
    assert claims["tier"] == "unverified"
    assert "no named attestor" in " ".join(claims["reasons"])
    assert "signature does not verify" in " ".join(
        claims["attestations"]["rejected"].values()
    )
    missing = attested_run(
        tmp_path / "missing", monkeypatch, skip=(ASSUMPTIONS[2],)
    )
    claims = read(missing, "causal_claims.json")
    assert claims["tier"] == "unverified"
    assert list(claims["attestations"]["rejected"]) == [ASSUMPTIONS[2]]


def test_verify_fails_when_the_attestations_do_not_check_out(
    tmp_path, monkeypatch
):
    result = attested_run(tmp_path, monkeypatch)
    wrong = claims_audit.verify(result.run_dir, "other-key")
    status = {f.name: f.status for f in wrong.findings}
    assert status["causal attestations verify"] == "fail"
    monkeypatch.delenv("SCMLE_SIGNING_KEY")
    nokey = claims_audit.verify(result.run_dir)
    status = {f.name: f.status for f in nokey.findings}
    assert status["causal attestations verify"] == "skipped"


def test_a_charter_without_attestors_stays_unverified_and_verifies(tmp_path):
    result = run(tmp_path, ASSUMPTIONS)
    assert read(result, "causal_claims.json")["tier"] == "unverified"
    verified = claims_audit.verify(result.run_dir)
    names = {f.name: f.status for f in verified.findings}
    assert names["causal attestations verify"] == "pass"
