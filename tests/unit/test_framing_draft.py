import json
import socket

import numpy
import pandas
import pytest

from benchmarks.qualification import framing_certify
from scmle.data.charter import contracts
from scmle.data.framing import costs
from scmle.data.framing import draft
from scmle.data.framing import provenance
from scmle.data.framing import refusal

HUMAN = (costs.CostSource.HUMAN,) * 4
MODEL = costs.DecisionCostModel(
    0.0, 5.0, 1.0, 0.0, 0.05, 0.5, HUMAN, costs.CostSource.HUMAN
)


def table(n=400, seed=0):
    rng = numpy.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    return pandas.DataFrame(
        {
            "customer_id": numpy.arange(n),
            "as_of": pandas.date_range("2024-01-01", periods=n, freq="h"),
            "resolved_at": pandas.date_range("2024-03-01", periods=n, freq="h"),
            "region": rng.choice(["n", "s"], n),
            "churn": y,
            "current_rule": (rng.random(n) < 0.3).astype(int),
        }
    )


def request(frame, text="Predict churn for each customer.", **kw):
    if "resolved_at" in frame.columns:
        kw.setdefault("target_time_column", "resolved_at")
    source = draft.SourceDescription(
        "warehouse://crm/accounts",
        tuple(frame.columns),
        primary_key=("customer_id",),
        **kw,
    )
    return draft.Request(text, source)


def test_deterministic_draft_assembles_an_unsigned_charter():
    frame = table()
    result = draft.frame(request(frame), frame, cost_model=MODEL)
    assert result.status == draft.WITH_QUESTIONS
    assert result.charter is not None
    assert result.charter.estimand.target == "churn"
    assert result.charter.estimand.prediction_moment == "as_of"
    assert result.charter.unit == "customer_id"
    assert result.charter.cost_model == MODEL
    assert "deployment_population" in result.open_items
    assert provenance.FieldSource.LLM not in result.fields.values()
    again = draft.frame(request(frame), frame, cost_model=MODEL)
    assert again.charter.charter_hash == result.charter.charter_hash


def test_missing_named_column_refuses_and_lists_it():
    frame = table()
    result = draft.frame(request(frame, "Predict `churn_flag` please."), frame)
    assert result.status == draft.REFUSED
    assert result.refusal["missing_columns"] == ["churn_flag"]
    assert result.charter is None


def test_unfillable_critical_slot_refuses_with_a_closed_question():
    frame = table().drop(columns=["as_of", "resolved_at"])
    result = draft.frame(request(frame), frame)
    assert result.status == draft.REFUSED
    assert result.refusal["unfilled"] == ["prediction_moment"]


def test_moment_after_target_time_is_a_blocking_check():
    frame = table()
    frame["resolved_at"] = frame["as_of"] - pandas.Timedelta(days=1)
    result = draft.frame(
        request(frame, target_time_column="resolved_at"), frame
    )
    assert result.status == draft.REFUSED
    assert result.refusal["failed"] == ["moment_before_target"]


def test_data_checks_flag_population_and_duplicates():
    frame = table()
    slots = {
        "target": "churn",
        "unit": ["region"],
        "population": "region == 'n'",
    }
    checks = {
        c.name: c
        for c in draft.check_draft(frame, slots, request(frame).source)
    }
    assert checks["unit_unique"].ok is False
    assert checks["population_coverage"].ok is True
    everything = draft.check_draft(
        frame, {"population": "customer_id >= 0"}, request(frame).source
    )
    assert everything[0].ok is False
    odd = draft.check_draft(
        frame, {"population": "__import__('os')"}, request(frame).source
    )
    assert odd[0].blocking and not odd[0].ok


def test_causal_wording_sets_the_kind_only():
    frame = table()
    result = draft.frame(
        request(frame, "What is the effect of discounts on churn?"), frame
    )
    assert result.charter.estimand.kind == "causal"


def test_cost_question_is_raised_only_when_it_changes_the_verdict():
    assert framing_certify.omission_asks(10_000, True, False)
    assert not framing_certify.omission_asks(10_000, False, False)
    assert not framing_certify.omission_asks(10_000, True, True)


def test_artifacts_are_written(tmp_path):
    frame = table()
    result = draft.frame(
        request(frame), frame, cost_model=MODEL, baseline_column="current_rule"
    )
    names = draft.write_artifacts(result, tmp_path)
    assert {
        "framing_draft.json",
        "decision_cost_model.json",
        "baseline_report.json",
        "framing_questions.json",
    } <= set(names)
    stored = json.loads((tmp_path / "framing_draft.json").read_text())
    assert stored["charter_hash"] == result.charter.charter_hash
    refused = draft.frame(request(frame, "Predict `nope`."), frame)
    assert "framing_refusal.json" in draft.write_artifacts(
        refused, tmp_path / "r"
    )


def test_link_report_is_descriptive():
    frame = table()
    result = draft.frame(request(frame), frame)
    link = draft.link_report(result.charter, list(frame.columns))
    assert link["fields"]["target"]["status"] == "resolved"
    assert link["gating"] is False
    other = draft.link_report(result.charter, ["x"])
    assert other["fields"]["target"]["status"] == "unresolved"


SAMPLE = {
    "question": "q",
    "target": "churn",
    "prediction_moment": "as_of",
    "unit": ["customer_id"],
    "population": "region == 'n'",
    "deployment_population": "same",
    "summary_measure": "net benefit",
}


def replay(samples, model_id="recorded-model"):
    return refusal.ReplaySampler(model_id, samples)


def test_llm_path_fills_slots_as_llm_and_replay_is_hash_identical(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("network used")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    frame = table()
    plain = request(frame, "Who leaves?")
    first = draft.frame(
        plain,
        frame,
        sampler=replay([SAMPLE] * 10),
        tau=0.1,
        human={"question": "Who leaves?"},
    )
    second = draft.frame(
        plain,
        frame,
        sampler=replay([SAMPLE] * 10),
        tau=0.1,
        human={"question": "Who leaves?"},
    )
    assert first.status != draft.REFUSED
    assert first.fields["target"] is provenance.FieldSource.LLM
    assert first.charter.charter_hash == second.charter.charter_hash
    assert first.draft_hash == second.draft_hash
    assert first.llm["samples_hash"] == second.llm["samples_hash"]


def test_disagreement_or_missing_tau_refuses():
    frame = table()
    mixed = [SAMPLE] * 6 + [{**SAMPLE, "target": "region"}] * 4
    result = draft.frame(request(frame), frame, sampler=replay(mixed), tau=0.1)
    assert result.status == draft.REFUSED
    assert "target" in result.refusal["conflicts"]
    unset = draft.frame(request(frame), frame, sampler=replay([SAMPLE] * 10))
    assert unset.status == draft.REFUSED


def test_llm_proposal_naming_a_missing_column_is_discarded():
    frame = table()
    bad = {**SAMPLE, "prediction_moment": "not_a_column"}
    result = draft.frame(
        request(frame), frame, sampler=replay([bad] * 10), tau=0.1
    )
    assert result.status != draft.REFUSED
    assert result.discarded
    assert provenance.FieldSource.LLM not in result.fields.values()


def test_consensus_scores_population_by_jaccard():
    frame = table()
    samples = [{**SAMPLE, "population": "region == 'n'"}] * 5 + [
        {**SAMPLE, "population": "region == 's'"}
    ] * 5
    agreed = refusal.consensus(frame, samples)
    assert agreed.agreement["population"] < 0.6
    assert "population" in agreed.conflicts


def test_refusal_follows_the_calibrated_tau_and_the_cp_bound():
    assert refusal.refuse(0.0, None) and not refusal.refuse(0.0, 0.0)
    assert refusal.clopper_pearson_upper(0, 59) == pytest.approx(
        1 - 0.05 ** (1 / 59), rel=1e-6
    )


def test_replay_sampler_runs_out():
    with pytest.raises(Exception, match="no recorded"):
        replay([]).sample("p", 0)


def test_schema_gate_rejects_malformed_charter_records():
    found = framing_certify.fuzz_rows(1000)
    assert found["accepted"] == 0
    assert contracts.template()["unit"] == ""
