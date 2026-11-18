"""Behavioural tests of the framing draft records, filling and framer modules."""

import json

import pandas
import pytest

from scmle.assurance.framing import costs
from scmle.assurance.framing import provenance
from scmle.assurance.framing.draft import filling
from scmle.assurance.framing.draft import framer
from scmle.assurance.framing.draft import records

HUMAN = (costs.Basis.HUMAN,) * 4


def tariff():
    return costs.Tariff(0.0, 5.0, 1.0, 0.0, 0.05, 0.5, HUMAN, costs.Basis.HUMAN)


def offer(**changes):
    values = {
        "uri": "data.csv",
        "columns": ("customer", "churned", "asof", "closed_at"),
        "key": ("customer",),
        "time": "closed_at",
    }
    values.update(changes)
    return records.Offer(**values)


def sample(rows=10):
    return pandas.DataFrame(
        {
            "customer": range(rows),
            "churned": [0, 1] * (rows // 2),
            "asof": pandas.date_range("2024-01-01", periods=rows),
            "closed_at": pandas.date_range("2024-02-01", periods=rows),
        }
    )


def request(text="Predict `churned` for each customer"):
    return records.Request(text, offer())


# ---- records ----------------------------------------------------------------


def test_the_record_constants_hold_their_values():
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


def test_the_causal_pattern_matches_whole_words_only():
    assert records.CAUSAL_WORDS.search("the Effect of price")
    assert records.CAUSAL_WORDS.search("what CAUSED it") is None
    assert records.CAUSAL_WORDS.search("because of rain")
    assert records.BACKTICKED.findall("`a` and `b c`") == ["a", "b c"]


def test_an_offer_reads_both_layouts():
    current = records.Offer.from_dict(
        {"uri": 3, "columns": ["a", "b"], "key": ["a"], "time": "b"}
    )
    assert current == records.Offer("3", ("a", "b"), ("a",), "b")
    old = records.Offer.from_dict(
        {
            "uri": "u",
            "columns": ["a"],
            "primary_key": ["a"],
            "target_time_column": "a",
        }
    )
    assert (old.key, old.time) == (("a",), "a")
    bare = records.Offer.from_dict({"uri": "u", "columns": []})
    assert (bare.key, bare.time) == ((), None)
    with pytest.raises(KeyError):
        records.Offer.from_dict({"uri": "u"})


def test_a_probe_serialises_every_field():
    probe = records.Probe("name", ok=True, blocking=False, detail="d")
    assert probe.to_dict() == {
        "name": "name",
        "ok": True,
        "blocking": False,
        "detail": "d",
    }


def test_a_refused_draft_document_has_no_charter():
    draft = framer.refused("why", {"x": 1}, discarded=("a",))
    document = draft.to_dict()
    assert document["schema_version"] == 2
    assert document["status"] == "refused"
    assert document["charter"] is None and document["charter_hash"] is None
    assert document["pending"] == []
    assert document["discarded"] == ["a"]
    assert document["label"].startswith("a drafted charter is a draft")
    assert draft.refusal == {"reason": "why", "x": 1}
    anchor = draft.anchor()
    assert anchor["schema_version"] == 1
    assert "open_items" in anchor and "pending" not in anchor
    assert len(draft.sha) == 64


def test_the_draft_digest_covers_the_charter_anchor():
    draft = framer.frame(request(), sample())
    assert draft.charter is not None
    document = draft.to_dict()
    assert document["charter_hash"] == draft.charter.charter_hash
    assert document["fields"] == dict(sorted(document["fields"].items()))
    other = framer.frame(request("Predict `churned` per customer"), sample())
    assert other.sha != draft.sha


# ---- filling ----------------------------------------------------------------


def test_normalise_lowers_and_removes_separators():
    assert filling.normalise("Churn_Date-2") == "churndate2"
    assert filling.normalise("a b") == "ab"


def test_mentioned_finds_words_and_long_substrings():
    columns = ["churned", "id", "age", "region_code"]
    assert filling.mentioned("predict churned by age", columns) == [
        "churned",
        "age",
    ]
    assert filling.mentioned("the regioncode field", columns) == ["region_code"]
    # "age" is short, so a substring match does not count.
    assert filling.mentioned("the average", columns) == []
    assert filling.mentioned("about churnedness", columns) == ["churned"]


def test_prefill_reads_an_explicit_target_and_the_schema():
    slots = filling.prefill(request("target: churned, please  "), sample())
    assert slots == {
        "target": "churned",
        "prediction_moment": "asof",
        "unit": ["customer"],
        "question": "target: churned, please",
    }


def test_prefill_falls_back_to_a_conventional_target_name():
    source = records.Offer("u", ("a", "label", "b"))
    frame = pandas.DataFrame({"a": [1], "label": [0], "b": [2]})
    slots = filling.prefill(
        records.Request("nothing named here", source), frame
    )
    assert slots == {"target": "label", "question": "nothing named here"}
    two = records.Request("a and b", source)
    assert filling.prefill(two, frame)["target"] == "label"
    both = records.Offer("u", ("y", "label"))
    frame = pandas.DataFrame({"y": [1], "label": [0]})
    assert "target" not in filling.prefill(records.Request("none", both), frame)


def test_prefill_skips_the_moment_when_two_stamps_are_present():
    frame = sample()
    frame["other"] = frame["asof"]
    source = offer(columns=(*offer().columns, "other"))
    slots = filling.prefill(records.Request("churned", source), frame)
    assert "prediction_moment" not in slots


def test_audit_checks_presence_completeness_and_variation():
    frame = pandas.DataFrame({"y": [0, 1, None, 1, 0], "c": [1] * 5})
    assert filling.audit(frame, "z") == records.Probe(
        "target_exists", ok=False, blocking=True, detail="z missing"
    )
    ok = filling.audit(frame, "y")
    assert ok == records.Probe(
        "target_complete",
        ok=True,
        blocking=False,
        detail="missing share 0.200; constant=False",
    )
    assert not filling.audit(frame, "c").ok
    gappy = pandas.DataFrame({"y": [0, 1, None, None, 1]})
    assert not filling.audit(gappy, "y").ok


def test_order_checks_the_moment_precedes_the_target_time():
    frame = sample(4)
    assert filling.order(frame, "nope", offer()).name == "moment_exists"
    good = filling.order(frame, "asof", offer())
    assert good.ok and good.blocking
    assert good.detail == "0 rows have the moment at or after the target time"
    frame.loc[0, "closed_at"] = frame.loc[0, "asof"]
    bad = filling.order(frame, "asof", offer())
    assert not bad.ok
    assert bad.detail.startswith("1 rows")
    assert filling.order(frame, "asof", offer(time=None)) is None


def test_vet_runs_the_checks_in_order_for_present_slots():
    frame = sample()
    slots = {
        "target": "churned",
        "prediction_moment": "asof",
        "unit": "customer",
        "population": "churned == 1",
    }
    checks = filling.vet(frame, slots, offer())
    assert [c.name for c in checks] == [
        "target_complete",
        "moment_before_target",
        "unit_unique",
        "population_coverage",
    ]
    assert checks[2].detail == "duplicate share 0.0000"
    assert checks[3].ok and checks[3].detail == "selects 0.500 of the rows"
    assert filling.vet(frame, {}, offer()) == ()


def test_vet_flags_missing_keys_duplicates_and_bad_populations():
    frame = sample()
    frame["dup"] = 1
    missing = filling.vet(frame, {"unit": ["customer", "nope"]}, offer())
    assert missing[0] == records.Probe(
        "unit_exists",
        ok=False,
        blocking=True,
        detail="['customer', 'nope'] missing",
    )
    dup = filling.vet(frame, {"unit": "dup"}, offer())
    assert dup[0].ok is False and dup[0].blocking is False
    assert dup[0].detail == "duplicate share 0.9000"
    garbled = filling.vet(frame, {"population": "nonsense words"}, offer())
    assert garbled[0].blocking and not garbled[0].ok
    assert garbled[0].detail == "predicate not understood"
    everyone = filling.vet(frame, {"population": "all"}, offer())
    assert not everyone[0].ok and not everyone[0].blocking


def test_assemble_fills_placeholders_and_lists_open_items():
    slots = {
        "target": "churned",
        "prediction_moment": "asof",
        "unit": ["customer", "month"],
    }
    charter, open_items = filling.assemble(
        slots, request("effect of price"), None
    )
    assert charter.question == records.UNSPECIFIED
    assert charter.estimand.kind == "causal"
    assert charter.unit == "customer-month"
    assert charter.estimand.population == "all rows of data.csv"
    assert charter.population == "same as data.csv"
    assert charter.estimand.summary == records.UNSPECIFIED
    assert open_items == [
        "population",
        "summary_measure",
        "deployment_population",
        "intercurrent_events",
        "cost_model",
    ]


def test_assemble_uses_the_costs_for_the_summary():
    slots = {
        "target": "churned",
        "prediction_moment": "asof",
        "unit": "customer",
        "question": "Q",
        "population": "p",
        "deployment_population": "d",
    }
    charter, open_items = filling.assemble(slots, request(), tariff())
    assert charter.estimand.kind == "predictive"
    assert charter.estimand.summary == "net benefit over thresholds [0.05, 0.5]"
    assert charter.costs == tariff()
    assert open_items == ["intercurrent_events"]


def test_linkage_resolves_words_to_columns_descriptively():
    charter = filling.assemble(
        {
            "target": "Churned",
            "prediction_moment": "the as-of",
            "unit": "customer",
        },
        request(),
        None,
    )[0]
    report = filling.linkage(charter, ["churned", "customer", "asof"])
    assert report["gating"] is False
    assert report["schema_version"] == 1
    assert report["fields"]["target"] == {
        "text": "Churned",
        "columns": ["churned"],
        "status": "resolved",
    }
    assert report["fields"]["prediction_moment"]["status"] == "unresolved"
    assert report["fields"]["unit"]["columns"] == ["customer"]


# ---- framer -----------------------------------------------------------------


def test_mismatch_refuses_unknown_named_columns_and_absent_sample_columns():
    stopped = framer.mismatch(
        request("Predict `nope` and `two words`"), sample()
    )
    assert stopped.status == "refused"
    assert stopped.refusal == {
        "reason": "request names columns the source does not have",
        "missing_columns": ["nope"],
    }
    lacking = framer.mismatch(request(), sample().drop(columns="asof"))
    assert lacking.refusal["missing_columns"] == ["asof"]
    assert framer.mismatch(request(), sample()) is None


def test_screen_reports_blocking_failures_before_unfilled_slots():
    bad = records.Probe("x", ok=False, blocking=True, detail="")
    soft = records.Probe("y", ok=False, blocking=False, detail="")
    assert framer.screen({}, [bad, soft]) == (
        "a blocking data-side check failed",
        {"failed": ["x"]},
    )
    reason, detail = framer.screen({"target": "t"}, [soft])
    assert reason.startswith("critical slots cannot be filled")
    assert detail["unfilled"] == ["prediction_moment", "unit"]
    assert detail["ask"] == {
        "prediction_moment": "name the column",
        "unit": "name the column",
    }
    full = {"target": "t", "prediction_moment": "m", "unit": "u"}
    assert framer.screen(full, [soft]) is None


def test_attribute_marks_open_items_without_a_source_as_template():
    sources = {"target": provenance.Filling.HUMAN}
    assert framer.attribute(sources, ["target", "cost_model"]) == {
        "target": provenance.Filling.HUMAN,
        "cost_model": provenance.Filling.TEMPLATE,
    }


def test_frame_issues_a_draft_with_questions_from_the_template():
    draft = framer.frame(request(), sample())
    assert draft.status == "issued_with_questions"
    assert draft.charter.estimand.target == "churned"
    assert draft.fields["target"] is provenance.Filling.TEMPLATE
    assert draft.fields["cost_model"] is provenance.Filling.TEMPLATE
    assert "baseline_decision_column" in draft.open_items
    assert draft.llm is None and draft.refusal is None


def test_frame_lets_a_person_override_and_refuses_a_bad_override():
    human = framer.frame(request(), sample(), human={"target": "customer"})
    assert human.charter.estimand.target == "customer"
    assert human.fields["target"] is provenance.Filling.HUMAN
    refused = framer.frame(request(), sample(), human={"target": "absent"})
    assert refused.status == "refused"
    assert refused.refusal["failed"] == ["target_exists"]


def test_frame_refuses_when_critical_slots_are_unfilled():
    source = offer(key=())
    draft = framer.frame(records.Request("Predict `churned`", source), sample())
    assert draft.refusal["unfilled"] == ["unit"]


class Replay:
    model_id = "replay"

    def __init__(self, proposal):
        self.proposal = proposal

    def sample(self, prompt, index):
        del prompt, index
        return dict(self.proposal)


def test_consult_without_a_sampler_does_nothing():
    assert framer.consult(request(), sample(), {}, {}, None, 0.5, 3) == (
        None,
        [],
        None,
    )


def test_consult_refuses_without_a_calibrated_tau():
    llm, discarded, stopped = framer.consult(
        request(), sample(), {}, {}, Replay({"target": "churned"}), None, 2
    )
    assert stopped.status == "refused"
    assert stopped.refusal["tau"] is None
    assert llm["model_id"] == "replay" and llm["tier"] == "unverified"
    assert discarded == []


def test_consult_fills_only_empty_slots_from_an_agreed_proposal():
    slots = {"target": "churned"}
    sources = {"target": provenance.Filling.TEMPLATE}
    proposal = {"target": "customer", "unit": "customer"}
    llm, discarded, stopped = framer.consult(
        request(), sample(), slots, sources, Replay(proposal), 1.0, 2
    )
    assert stopped is None and discarded == []
    assert slots == {"target": "churned", "unit": "customer"}
    assert sources["unit"] is provenance.Filling.LLM
    assert llm["tau"] == 1.0


def test_consult_discards_a_proposal_that_fails_a_blocking_check():
    slots = {}
    llm, discarded, stopped = framer.consult(
        request(), sample(), slots, {}, Replay({"target": "absent"}), 1.0, 2
    )
    assert discarded == ["llm draft failed a blocking data-side check"]
    assert slots == {} and stopped is None
    assert llm is not None


def test_interview_asks_for_a_baseline_column_when_none_is_named():
    open_items = ["cost_model"]
    profile, asked, robust = framer.interview(
        sample(), "churned", None, None, None, open_items
    )
    assert (profile, asked, robust) == (None, [], [])
    assert open_items == ["cost_model", "baseline_decision_column"]


def test_interview_profiles_the_baseline_with_costs():
    frame = sample()
    frame["policy"] = [1, 0] * 5
    open_items = []
    profile, asked, robust = framer.interview(
        frame, "churned", "policy", tariff(), None, open_items
    )
    assert profile is not None
    assert open_items == []
    assert asked == [] and robust == []


def test_emit_writes_the_draft_and_its_companions(tmp_path):
    issued = framer.frame(request(), sample(), cost_model=tariff())
    names = framer.emit(issued, tmp_path / "out")
    assert names == [
        "decision_cost_model.json",
        "framing_draft.json",
        "framing_questions.json",
    ]
    written = json.loads((tmp_path / "out" / "framing_draft.json").read_text())
    assert written["status"] == issued.status
    questions = json.loads(
        (tmp_path / "out" / "framing_questions.json").read_text()
    )
    assert questions["open_items"] == list(issued.open_items)
    refused = framer.refused("why", {})
    assert framer.emit(refused, tmp_path / "r") == [
        "framing_draft.json",
        "framing_questions.json",
        "framing_refusal.json",
    ]
