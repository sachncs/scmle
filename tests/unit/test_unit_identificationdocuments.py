"""The causal run's estimate record, base adapter and documents."""

import hashlib
import json

import pandas
import pytest

from scmle.assurance.charter import causal
from scmle.causal.identification import battery
from scmle.causal.identification import select
from scmle.causal.identification import tier
from scmle.causal.identification.run import documents
from scmle.causal.identification.run import estimator
from scmle.contracts import qualification

ASSUMPTIONS = ("unconfoundedness_given:a,b", "positivity", "no_interference")


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


@pytest.mark.parametrize(
    ("low", "high", "conclusive"),
    [
        (0.1, 1.0, True),
        (-1.0, -0.1, True),
        (0.0, 1.0, False),
        (-1.0, 0.0, False),
        (-1.0, 1.0, False),
    ],
)
def test_an_estimate_is_conclusive_only_when_zero_is_strictly_outside(
    low, high, conclusive
):
    assert estimator.Est(0.5, 0.1, low, high, 5).conclusive() is conclusive


def test_an_estimate_defaults_to_a_fresh_empty_detail():
    first = estimator.Est(1.0, 0.1, 0.8, 1.2, 9)
    second = estimator.Est(1.0, 0.1, 0.8, 1.2, 9)
    assert first.detail == {}
    assert first.detail is not second.detail
    assert first.n == 9


def test_a_wald_test_needs_a_finite_positive_standard_error():
    est = estimator.Est(2.0, 0.5, 1.0, 3.0, 10)
    assert estimator.waldify(est, "nco", "negative_outcome") == battery.wald(
        "nco", "negative_outcome", 2.0, 0.5
    )
    for se in (0.0, -0.5, float("nan"), float("inf")):
        bad = estimator.Est(2.0, se, 1.0, 3.0, 10)
        assert estimator.waldify(bad, "n", "k") is None


def test_the_base_estimator_stores_settings_and_has_empty_hooks():
    spec = block()
    frame = pandas.DataFrame({"t": [0.0], "y": [1.0]})
    adapter = estimator.Estimator(spec, frame, 3, "linear", 0.01)
    assert estimator.Estimator.name == ""
    assert (adapter.spec, adapter.x, adapter.seed) == (spec, frame, 3)
    assert adapter.learner == "linear"
    assert adapter.alpha_falsify == 0.01
    assert adapter.design == {}
    assert adapter.refusal is None
    assert adapter.weak is False
    assert adapter.single_unit is False
    assert adapter.notes == []
    assert adapter.prepare() is None
    assert adapter.controls(frame, 0.9) == []
    assert adapter.sensitivity(frame, 0.9) == {
        "holds": None,
        "note": "no sensitivity analysis for this strategy",
    }
    with pytest.raises(NotImplementedError):
        adapter.estimate(frame, 0.9)
    with pytest.raises(NotImplementedError):
        adapter.reckon(frame, 0.9, "y")


def test_the_documents_are_the_five_named_files():
    assert documents.DOCUMENTS == (
        "identification_plan.json",
        "identification_report.json",
        "falsification_report.json",
        "sensitivity_report.json",
        "causal_claims.json",
    )
    assert documents.LIMITS.startswith("confirmed means: under the attested")
    assert documents.LIMITS.endswith("passes undetected")


def test_a_digest_hashes_sorted_compact_json_with_text_fallback():
    stamp = pandas.Timestamp("2026-01-01")
    text = json.dumps(
        {"a": 1, "b": str(stamp)}, sort_keys=True, separators=(",", ":")
    )
    assert documents.digest({"b": stamp, "a": 1}) == (
        hashlib.sha256(text.encode("utf-8")).hexdigest()
    )


def test_an_arbitration_states_the_hypothesis():
    assert documents.arbitrate(block()) == {
        "hypothesis": "ATE of t on y = 0",
        "labelled": "causal, under the attested assumptions",
    }


def test_prerequisites_expand_the_unconfoundedness_word():
    spec = block()
    needed = documents.prerequisites("backdoor_aipw", spec)
    assert "unconfoundedness_given:a,b" in needed
    assert "unconfoundedness_given" not in needed
    assert needed.count("unconfoundedness_given:a,b") == 1
    expected = []
    for word in select.STRATEGIES["backdoor_aipw"].assumptions:
        if word == "unconfoundedness_given":
            expected.append("unconfoundedness_given:a,b")
        else:
            expected.append(word)
    assert needed == tuple(expected)


def test_prerequisites_of_a_strategy_without_unconfoundedness_are_its_own():
    spec = block()
    assert documents.prerequisites("rd", spec) == tuple(
        select.STRATEGIES["rd"].assumptions
    )


def assignment():
    return tier.Assignment(
        qualification.Tier.ATTESTED,
        "identified",
        {"estimated_on_c": True},
        ("why",),
        "downgrade_tier",
    )


def test_an_assembly_writes_the_claim_file_over_the_hashed_documents():
    spec = block()
    plan = select.select(spec)
    out = documents.assemble(
        plan,
        spec,
        {"r": 1},
        {"f": 1},
        {"s": 1},
        assignment(),
        "backdoor_aipw",
        ("positivity", "no_interference"),
        {"positivity": "Ann"},
        ("z", "a"),
    )
    claims = out.documents["causal_claims.json"]
    assert claims == {
        "status": "identified",
        "strategy": "backdoor_aipw",
        "estimand": "ATE",
        "tier": "attested",
        "conditions": {"estimated_on_c": True},
        "reasons": ["why"],
        "action": "downgrade_tier",
        "assumptions": {"positivity": "Ann", "no_interference": ""},
        "certified_gates_used": ["a", "z"],
        "hashes": {
            name: documents.digest(out.documents[name])
            for name in documents.DOCUMENTS[:4]
        },
        "limits": documents.LIMITS,
    }
    assert out.hashes["causal_claims.json"] == documents.digest(claims)
    assert list(out.documents) == list(documents.DOCUMENTS)
    assert out.documents["identification_plan.json"]["causal_spec"] == (
        spec.to_dict()
    )
    assert out.decision == assignment()


def test_an_assembly_copies_subgroup_and_attestation_summaries():
    spec = block()
    plan = select.select(spec)
    subgroups = {
        "rule": "holm",
        "alpha": 0.05,
        "family_size": 3,
        "rejected": ("r=a",),
        "gate": {"status": "descriptive"},
        "hypotheses": [],
    }
    out = documents.assemble(
        plan,
        spec,
        {},
        {},
        {},
        assignment(),
        "",
        (),
        None,
        (),
        subgroups,
        {"ok": 1},
    )
    claims = out.documents["causal_claims.json"]
    assert claims["strategy"] is None
    assert claims["subgroups"] == {
        "rule": "holm",
        "alpha": 0.05,
        "family_size": 3,
        "rejected": ["r=a"],
        "gate": "descriptive",
    }
    assert claims["attestations"] == {"ok": 1}


def test_a_refusal_writes_no_tier_and_records_reason_and_dag():
    spec = block(assumptions=("positivity",))
    plan = select.select(spec)
    out = documents.refuse(plan, spec, "because", {"hash": "h"})
    report = out.documents["identification_report.json"]
    assert report["reason"] == "because"
    assert report["dag"] == {"hash": "h"}
    claims = out.documents["causal_claims.json"]
    assert claims["status"] == "not_identified"
    assert claims["tier"] is None
    assert claims["strategy"] is None
    assert out.documents["falsification_report.json"] == {}
    assert out.documents["sensitivity_report.json"] == {}
    plain = documents.refuse(plan, spec)
    assert "reason" not in plain.documents["identification_report.json"]
    assert "dag" not in plain.documents["identification_report.json"]
    assert plain.documents["identification_report.json"] == dict(
        select.refusal(plan, spec)
    )
