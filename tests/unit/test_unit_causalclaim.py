"""The causal estimate registered as a ledger claim."""

import json
import re

import pandas
import pytest

import scmle.assurance.admissibility.dag
from scmle.assurance.charter import causal
from scmle.assurance.ledger import budget
from scmle.assurance.ledger import ledger
from scmle.causal import claim
from scmle.causal.identification import attest
from scmle.causal.identification import run
from scmle.causal.identification import select
from scmle.contracts import errors
from tests.unit.test_identification_run import observational

BUDGET = budget.Alpha.create(0.05, 0.1)
ASSUMPTIONS = ("unconfoundedness_given:a,b", "positivity", "no_interference")
KEY = "attestor-key"
SCOPE = "ab" * 32


def block(**changes):
    values = {
        "treatment": "t",
        "outcome": "y",
        "estimand": "ATE",
        "assumptions": ASSUMPTIONS,
        "strategies": ("backdoor_aipw",),
    }
    values.update(changes)
    return causal.Question(**values)


def graph():
    return scmle.assurance.admissibility.dag.Dag.build(
        source="test",
        edges=(("a", "t"), ("a", "y"), ("b", "t"), ("b", "y"), ("t", "y")),
    )


def test_the_claim_names_are_fixed():
    assert claim.KIND == "causal_primary"
    assert claim.CLAIM_ID == "causal_primary_theta_zero"
    assert claim.CLAIM_FILE == "causal_claims.json"


def test_an_unseal_of_no_path_is_none():
    assert claim.unseal(None) is None
    assert claim.unseal("") is None


def test_an_unseal_loads_a_stored_graph(tmp_path):
    path = tmp_path / "dag.json"
    path.write_text(json.dumps(graph().artifact()), encoding="utf-8")
    loaded = claim.unseal(str(path))
    assert loaded.digest() == graph().digest()


def test_an_unseal_of_a_bad_file_is_a_config_error(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(
        errors.ConfigError, match="cannot load the declared DAG"
    ):
        claim.unseal(str(missing))
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(
        errors.ConfigError, match=re.escape(f"declared DAG {broken}: ")
    ):
        claim.unseal(str(broken))
    tampered = tmp_path / "tampered.json"
    document = graph().artifact()
    document["hash"] = "0" * 64
    tampered.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(errors.ConfigError):
        claim.unseal(str(tampered))


def test_a_registration_hashes_the_frozen_claim():
    spec = block()
    plan = select.select(spec)
    found = claim.registration(spec, plan, graph(), BUDGET, 120)
    body = {
        "hypothesis": "ATE of t on y = 0",
        "causal": spec.to_dict(),
        "strategy": "backdoor_aipw",
        "dag": graph().digest(),
        "alpha_confirm": str(BUDGET.confirmation.alpha_confirm),
        "exploration_rows": 120,
    }
    assert found.identifier == "causal_primary_theta_zero"
    assert found.kind == "causal_primary"
    assert found.statement == (
        "ATE of t on y differs from zero, by backdoor_aipw, under the "
        "attested assumptions"
    )
    assert found.sha256 == ledger.digest(body)
    bare = claim.registration(spec, plan, None, BUDGET, 120)
    assert bare.sha256 == ledger.digest({**body, "dag": None})
    assert claim.registration(spec, plan, None, BUDGET, 121).sha256 != (
        bare.sha256
    )


def test_a_jsonable_document_is_its_json_round_trip():
    stamp = pandas.Timestamp("2026-01-01")
    assert claim.jsonable({"a": (1, 2), "b": stamp}) == {
        "a": [1, 2],
        "b": str(stamp),
    }


def test_a_credit_without_an_attestations_file_is_a_config_error():
    spec = block()
    with pytest.raises(
        errors.ConfigError, match="declares no attestations file"
    ):
        claim.credit(spec, select.select(spec), SCOPE, KEY)
    unplanned = block(assumptions=("positivity",))
    with pytest.raises(
        errors.ConfigError, match="declares no attestations file"
    ):
        claim.credit(unplanned, select.select(unplanned), SCOPE, KEY)


def attested(tmp_path, records):
    path = tmp_path / "attestations.json"
    path.write_text(
        json.dumps({"format_version": 1, "attestations": records}),
        encoding="utf-8",
    )
    return block(
        attestors=(causal.Attestor("Ann", "statistician", ASSUMPTIONS),),
        attestations=str(path),
    )


def signed(assumption):
    return (
        attest.Attestation(
            assumption=assumption,
            attestor="Ann",
            statement="holds",
            evidence="memo",
            date="2020-01-01",
            expires="2099-01-01",
            scope=SCOPE,
        )
        .signed(KEY)
        .to_dict()
    )


def test_a_credit_accepts_the_valid_signed_attestations(tmp_path):
    spec = attested(tmp_path, [signed(a) for a in ASSUMPTIONS])
    applied = claim.credit(spec, select.select(spec), SCOPE, KEY)
    needed = run.prerequisites("backdoor_aipw", spec)
    assert applied.accepted == dict.fromkeys(needed, "Ann")
    wrong = claim.credit(spec, select.select(spec), SCOPE, "other")
    assert wrong.accepted == {}


def test_a_credit_of_a_malformed_file_is_a_config_error(tmp_path):
    spec = attested(tmp_path, [signed(ASSUMPTIONS[0])] * 2)
    with pytest.raises(errors.ConfigError, match="one attestation"):
        claim.credit(spec, select.select(spec), SCOPE, KEY)


def test_a_rig_of_an_unidentified_charter_is_none():
    spec = block(assumptions=("positivity",))
    assert claim.rig(spec, observational(100, 1), "y", BUDGET) is None


def test_a_rig_with_attestors_needs_the_charter_hash(tmp_path):
    spec = attested(tmp_path, [signed(a) for a in ASSUMPTIONS])
    with pytest.raises(
        errors.ConfigError, match="attestors need the charter hash"
    ):
        claim.rig(spec, observational(100, 1), "y", BUDGET)


def test_a_rig_registers_and_runs_the_claim_on_the_released_rows():
    spec = block()
    x = observational(1200, 1)
    docket = claim.rig(spec, x, "y", BUDGET, attestors={})
    (registered,) = docket.registered
    assert registered == claim.registration(
        spec, select.select(spec), None, BUDGET, 1200
    )
    c = observational(1200, 2)
    output = docket.run(c.drop(columns="y"), c["y"])
    expected = run.identify(spec, x, c, BUDGET, attestors={})
    assert output.statuses == {
        "causal_primary_theta_zero": expected.documents["causal_claims.json"][
            "status"
        ]
    }
    assert output.documents == {
        k: claim.jsonable(v) for k, v in expected.documents.items()
    }


def test_a_rig_with_attestors_credits_the_signed_file(tmp_path):
    spec = attested(tmp_path, [signed(a) for a in ASSUMPTIONS])
    x = observational(1200, 1)
    docket = claim.rig(
        spec, x, "y", BUDGET, charter_hash=SCOPE, signing_key=KEY
    )
    c = observational(1200, 2)
    output = docket.run(c.drop(columns="y"), c["y"])
    claims = output.documents["causal_claims.json"]
    assert claims["assumptions"]["positivity"] == "Ann"
    assert "attestations" in claims
