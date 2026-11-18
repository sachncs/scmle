"""Unit tests of the framing draft submodules, each by its full name."""

import json

import numpy
import pandas
import pytest

from scmle.assurance.framing import costs
from scmle.assurance.framing import provenance
from scmle.assurance.framing.draft import filling
from scmle.assurance.framing.draft import framer
from scmle.assurance.framing.draft import records

HUMAN = (costs.Basis.HUMAN,) * 4
MODEL = costs.Tariff(0.0, 5.0, 1.0, 0.0, 0.05, 0.5, HUMAN, costs.Basis.HUMAN)


def table(n=40):
    rng = numpy.random.default_rng(0)
    return pandas.DataFrame(
        {
            "customer_id": numpy.arange(n),
            "as_of": pandas.date_range("2024-01-01", periods=n, freq="h"),
            "resolved_at": pandas.date_range("2024-03-01", periods=n, freq="h"),
            "region": rng.choice(["n", "s"], n),
            "churn": numpy.arange(n) % 2,
        }
    )


def offer(frame, key=("customer_id",), time="resolved_at"):
    return records.Offer("db://x", tuple(frame.columns), key=key, time=time)


def ask(frame, text="Predict churn for each customer.", **kw):
    return records.Request(text, offer(frame, **kw))


# ---- records ---------------------------------------------------------------


def test_the_draft_constants_have_their_values():
    assert (records.ISSUED, records.WITH_QUESTIONS, records.REFUSED) == (
        "issued",
        "issued_with_questions",
        "refused",
    )
    assert records.MAX_TARGET_MISSING == 0.2
    assert records.MIN_DISTINCT_VALUES == 2
    assert records.MIN_SUBSTRING_CHARS == 4
    assert records.MIN_POPULATION_SHARE == 0.01
    assert records.UNIT_DUPLICATE_TOLERANCE == 0.01
    assert records.CRITICAL_FIELDS == ("target", "prediction_moment", "unit")
    assert records.MAX_MOMENT_VIOLATIONS == 0
    assert records.COST_RANGE == (0.05, 0.5)
    assert records.UNSPECIFIED == "unspecified (open question)"


def test_causal_words_and_backticks_are_found():
    assert records.CAUSAL_WORDS.search("the Effect of price")
    assert records.CAUSAL_WORDS.search("predict churn") is None
    assert records.CAUSAL_WORDS.search("causality") is None
    assert records.BACKTICKED.findall("use `a` and `b c`") == ["a", "b c"]


def test_an_offer_reads_current_and_old_layouts():
    current = records.Offer.from_dict(
        {"uri": "u", "columns": ["a"], "key": ["a"], "time": "t"}
    )
    assert current == records.Offer("u", ("a",), ("a",), "t")
    old = records.Offer.from_dict(
        {
            "uri": 5,
            "columns": ["a"],
            "primary_key": ["a"],
            "target_time_column": "t",
        }
    )
    assert old == records.Offer("5", ("a",), ("a",), "t")
    bare = records.Offer.from_dict({"uri": "u", "columns": []})
    assert (bare.key, bare.time) == ((), None)
    with pytest.raises(KeyError):
        records.Offer.from_dict({"columns": []})


def test_a_probe_serializes_every_field():
    probe = records.Probe("n", ok=True, blocking=False, detail="d")
    assert probe.to_dict() == {
        "name": "n",
        "ok": True,
        "blocking": False,
        "detail": "d",
    }


def test_a_refused_draft_document_and_digest():
    draft = framer.refused("why", {"x": 1}, discarded=("gone",))
    document = draft.to_dict()
    assert document["schema_version"] == 2
    assert document["status"] == "refused"
    assert document["charter"] is None
    assert document["charter_hash"] is None
    assert document["pending"] == []
    assert document["discarded"] == ["gone"]
    assert document["label"].startswith("a drafted charter is a draft")
    anchor = draft.anchor()
    assert anchor["schema_version"] == 1
    assert "open_items" in anchor
    assert "pending" not in anchor
    assert len(draft.sha) == 64
    assert draft.sha == framer.refused("why", {"x": 1}, discarded=("gone",)).sha
    assert draft.sha != framer.refused("other", {"x": 1}).sha


# ---- filling ---------------------------------------------------------------


def test_normalise_lowercases_and_drops_separators():
    assert filling.normalise("Customer_ID-2 x") == "customerid2x"


def test_mentioned_finds_words_and_long_substrings():
    columns = ["churn", "id", "region_code", "abcd"]
    assert filling.mentioned("predict churn by region code", columns) == [
        "churn",
        "region_code",
    ]
    assert filling.mentioned("theid", columns) == []
    assert filling.mentioned("xabcdx", columns) == []
    assert filling.mentioned("x abcd", columns) == ["abcd"]


def test_prefill_reads_an_explicit_target():
    frame = table()
    slots = filling.prefill(ask(frame, "Target: `churn`, by region"), frame)
    assert slots["target"] == "churn"
    assert slots["prediction_moment"] == "as_of"
    assert slots["unit"] == ["customer_id"]
    assert slots["question"] == "Target: `churn`, by region"


def test_prefill_falls_back_to_a_conventional_target_name():
    frame = pandas.DataFrame({"label": [0, 1], "a": [1, 2], "b": [3, 4]})
    request = records.Request(
        "  what of a and b?  ", records.Offer("u", ("label", "a", "b"))
    )
    slots = filling.prefill(request, frame)
    assert slots["target"] == "label"
    assert "unit" not in slots
    assert "prediction_moment" not in slots
    assert slots["question"] == "what of a and b?"


def test_prefill_leaves_the_target_open_when_ambiguous():
    frame = pandas.DataFrame({"a": [1], "b": [2]})
    request = records.Request("a or b", records.Offer("u", ("a", "b")))
    assert "target" not in filling.prefill(request, frame)


def test_prefill_needs_a_single_datetime_column():
    frame = table()
    frame["other"] = frame["as_of"]
    slots = filling.prefill(ask(frame), frame)
    assert "prediction_moment" not in slots


def test_audit_reports_a_missing_target_as_blocking():
    probe = filling.audit(table(), "nope")
    assert probe == records.Probe("target_exists", False, True, "nope missing")


def test_audit_fails_a_sparse_or_constant_target_without_blocking():
    frame = pandas.DataFrame({"y": [0, 1, 0, 1, None], "c": [1] * 5})
    probe = filling.audit(frame, "y")
    assert probe.ok is True
    assert probe.detail == "missing share 0.200; constant=False"
    sparse = pandas.DataFrame({"y": [0, 1, None, None]})
    assert filling.audit(sparse, "y").ok is False
    constant = filling.audit(frame, "c")
    assert constant.ok is False
    assert constant.blocking is False
    assert constant.detail.endswith("constant=True")


def test_order_checks_the_moment_precedes_the_target_time():
    frame = table()
    good = filling.order(frame, "as_of", offer(frame))
    assert good == records.Probe(
        "moment_before_target",
        True,
        True,
        "0 rows have the moment at or after the target time",
    )
    frame.loc[0, "as_of"] = frame.loc[0, "resolved_at"]
    bad = filling.order(frame, "as_of", offer(frame))
    assert bad.ok is False
    assert bad.detail.startswith("1 rows")
    assert filling.order(frame, "nope", offer(frame)).name == "moment_exists"
    assert filling.order(frame, "as_of", offer(frame, time="zz")) is None


def test_vet_checks_unit_uniqueness_and_existence():
    frame = table()
    found = filling.vet(frame, {"unit": "customer_id"}, offer(frame))
    assert found == (
        records.Probe("unit_unique", True, False, "duplicate share 0.0000"),
    )
    dup = pandas.concat([frame, frame.iloc[:1]], ignore_index=True)
    assert (
        filling.vet(dup, {"unit": ["customer_id"]}, offer(dup))[0].ok is False
    )
    missing = filling.vet(frame, {"unit": ["x", "customer_id"]}, offer(frame))
    assert missing[0].name == "unit_exists"
    assert missing[0].blocking is True
    assert missing[0].detail == "['x', 'customer_id'] missing"


def test_vet_checks_population_coverage():
    frame = table()
    some = filling.vet(frame, {"population": "churn == 1"}, offer(frame))
    assert some[0].name == "population_coverage"
    assert some[0].ok is True
    assert some[0].detail == "selects 0.500 of the rows"
    everyone = filling.vet(frame, {"population": "all"}, offer(frame))
    assert everyone[0].ok is False
    assert everyone[0].blocking is False
    garbled = filling.vet(frame, {"population": "nope >"}, offer(frame))
    assert garbled[0].ok is False
    assert garbled[0].blocking is True
    assert garbled[0].detail == "predicate not understood"


def test_vet_orders_checks_and_skips_absent_slots():
    frame = table()
    slots = {
        "target": "churn",
        "prediction_moment": "as_of",
        "unit": ["customer_id"],
    }
    names = [p.name for p in filling.vet(frame, slots, offer(frame))]
    assert names == ["target_complete", "moment_before_target", "unit_unique"]
    assert filling.vet(frame, {}, offer(frame)) == ()


def test_assemble_fills_placeholders_and_lists_open_items():
    frame = table()
    slots = {
        "target": "churn",
        "prediction_moment": "as_of",
        "unit": ["customer_id", "region"],
    }
    charter, items = filling.assemble(slots, ask(frame), None)
    assert charter.unit == "customer_id-region"
    assert charter.question == records.UNSPECIFIED
    assert charter.estimand.kind == "predictive"
    assert charter.estimand.population == "all rows of db://x"
    assert charter.population == "same as db://x"
    assert charter.estimand.summary == records.UNSPECIFIED
    assert items == [
        "population",
        "summary_measure",
        "deployment_population",
        "intercurrent_events",
        "cost_model",
    ]


def test_assemble_with_costs_states_the_net_benefit_range():
    frame = table()
    slots = {
        "target": "churn",
        "prediction_moment": "as_of",
        "unit": "customer_id",
        "question": "q",
        "population": "everyone",
        "deployment_population": "same",
    }
    charter, items = filling.assemble(
        slots, ask(frame, "the effect of price"), MODEL
    )
    assert charter.estimand.kind == "causal"
    assert charter.estimand.summary == "net benefit over thresholds [0.05, 0.5]"
    assert charter.costs == MODEL
    assert items == ["intercurrent_events"]


def test_linkage_resolves_words_to_columns_without_gating():
    frame = table()
    charter = filling.assemble(
        {
            "target": "churn",
            "prediction_moment": "as_of",
            "unit": ["Customer_ID"],
        },
        ask(frame),
        None,
    )[0]
    out = filling.linkage(charter, list(frame.columns))
    assert out["gating"] is False
    assert out["schema_version"] == 1
    assert out["fields"]["target"] == {
        "text": "churn",
        "columns": ["churn"],
        "status": "resolved",
    }
    assert out["fields"]["unit"]["columns"] == ["customer_id"]
    assert filling.linkage(charter, ["zzz"])["fields"]["target"]["status"] == (
        "unresolved"
    )


# ---- framer ----------------------------------------------------------------


def test_refused_keeps_the_reason_checks_and_record():
    probe = records.Probe("p", False, True, "d")
    draft = framer.refused("r", {"a": 1}, [probe], {"m": 1}, ["x"])
    assert draft.status == records.REFUSED
    assert draft.refusal == {"reason": "r", "a": 1}
    assert draft.checks == (probe,)
    assert draft.llm == {"m": 1}
    assert draft.discarded == ("x",)
    assert draft.fields == {}


def test_mismatch_refuses_named_columns_the_source_lacks():
    frame = table()
    out = framer.mismatch(ask(frame, "use `nope` and `two words`"), frame)
    assert out.refusal == {
        "reason": "request names columns the source does not have",
        "missing_columns": ["nope"],
    }


def test_mismatch_refuses_a_sample_lacking_declared_columns():
    frame = table()
    out = framer.mismatch(ask(frame), frame.drop(columns=["region"]))
    assert out.refusal["missing_columns"] == ["region"]
    assert framer.mismatch(ask(frame, "use `churn`"), frame) is None


def test_consult_without_a_sampler_does_nothing():
    slots, sources = {}, {}
    assert framer.consult(
        ask(table()), table(), slots, sources, None, 0.1, 3
    ) == (
        None,
        [],
        None,
    )
    assert slots == sources == {}


def test_screen_reports_blocking_checks_before_unfilled_slots():
    blocked = records.Probe("b", False, True, "")
    soft = records.Probe("s", False, False, "")
    reason, detail = framer.screen({}, [blocked, soft])
    assert reason == "a blocking data-side check failed"
    assert detail == {"failed": ["b"]}
    reason, detail = framer.screen({"target": "y", "unit": ""}, [soft])
    assert reason.startswith("critical slots cannot be filled")
    assert detail["unfilled"] == ["prediction_moment", "unit"]
    assert detail["ask"] == {
        "prediction_moment": "name the column",
        "unit": "name the column",
    }
    full = {"target": "y", "prediction_moment": "t", "unit": "u"}
    assert framer.screen(full, [soft]) is None


def test_attribute_adds_template_sources_for_open_items():
    out = framer.attribute({"a": provenance.Filling.HUMAN}, ["a", "b"])
    assert out == {
        "a": provenance.Filling.HUMAN,
        "b": provenance.Filling.TEMPLATE,
    }


def test_interview_without_baseline_asks_for_its_column():
    items = ["cost_model"]
    profile, asked, robust = framer.interview(
        table(), "churn", None, None, None, items
    )
    assert (profile, asked, robust) == (None, [], [])
    assert items == ["cost_model", "baseline_decision_column"]


def test_frame_issues_a_draft_with_questions():
    frame = table()
    draft = framer.frame(ask(frame), frame, cost_model=MODEL)
    assert draft.status == records.WITH_QUESTIONS
    assert draft.charter.estimand.target == "churn"
    assert draft.charter.estimand.moment == "as_of"
    assert draft.charter.unit == "customer_id"
    assert draft.fields["target"] is provenance.Filling.TEMPLATE
    assert draft.fields["deployment_population"] is provenance.Filling.TEMPLATE
    assert "baseline_decision_column" in draft.open_items


def test_frame_takes_human_slots_over_the_rules():
    frame = table()
    draft = framer.frame(
        ask(frame), frame, cost_model=MODEL, human={"target": "region"}
    )
    assert draft.charter.estimand.target == "region"
    assert draft.fields["target"] is provenance.Filling.HUMAN


def test_frame_refuses_on_a_blocking_check():
    frame = table()
    draft = framer.frame(ask(frame), frame, human={"target": "nope"})
    assert draft.status == records.REFUSED
    assert draft.refusal["failed"] == ["target_exists"]


def test_frame_refuses_a_slot_set_the_charter_rejects():
    frame = table()
    draft = framer.frame(ask(frame), frame, human={"question": "   "})
    assert draft.status == records.REFUSED
    assert draft.refusal["reason"] == "the draft does not form a valid charter"


def test_frame_refuses_a_mismatched_request():
    frame = table()
    assert framer.frame(ask(frame, "`zz`"), frame).status == records.REFUSED


def test_emit_writes_the_documents_that_exist(tmp_path):
    frame = table()
    draft = framer.frame(ask(frame), frame, cost_model=MODEL)
    names = framer.emit(draft, tmp_path / "out")
    assert names == [
        "decision_cost_model.json",
        "framing_draft.json",
        "framing_questions.json",
    ]
    stored = json.loads((tmp_path / "out" / "framing_draft.json").read_text())
    assert stored["status"] == "issued_with_questions"
    refused = framer.refused("r", {})
    assert framer.emit(refused, tmp_path / "r") == [
        "framing_draft.json",
        "framing_questions.json",
        "framing_refusal.json",
    ]
