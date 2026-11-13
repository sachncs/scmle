"""Pins the drafter: slots, data checks, assembly, refusal wording and artifacts."""

import hashlib
import json
import pathlib

import numpy
import pandas
import pytest

from scmle.contracts import signing
from scmle.data.framing import draft
from scmle.data.framing import provenance
from scmle.data.framing import refusal
from tests.unit.test_framing_draft import MODEL
from tests.unit.test_framing_draft import replay
from tests.unit.test_framing_draft import request
from tests.unit.test_framing_draft import SAMPLE
from tests.unit.test_framing_draft import table


def source(columns=("a", "b"), **options):
    return draft.SourceDescription("uri://x", tuple(columns), **options)


def asked(text, columns, **options):
    return draft.Request(text, source(columns, **options))


def test_the_constants_are_the_documented_ones():
    assert draft.ISSUED == "issued"
    assert draft.WITH_QUESTIONS == "issued_with_questions"
    assert draft.REFUSED == "refused"
    assert draft.MAX_TARGET_MISSING == 0.2
    assert draft.MIN_DISTINCT_VALUES == 2
    assert draft.MIN_SUBSTRING_CHARS == 4
    assert draft.MIN_POPULATION_SHARE == 0.01
    assert draft.UNIT_DUPLICATE_TOLERANCE == 0.01
    assert draft.TARGET_WORDS == ("target", "label", "outcome", "y")
    assert draft.UNSPECIFIED == "unspecified (open question)"
    assert draft.CRITICAL_FIELDS == ("target", "prediction_moment", "unit")
    assert draft.MAX_MOMENT_VIOLATIONS == 0
    assert draft.COST_RANGE == (0.05, 0.5)


def test_the_causal_pattern_matches_whole_phrases_in_any_case():
    for text in (
        "the cause",
        "Causal question",
        "EFFECT OF x",
        "impact of x",
        "because of x",
        "treatment effect",
    ):
        assert draft.CAUSAL_WORDS.search(text), text
    for text in ("causes", "effects", "unrelated", "defect of x"):
        assert not draft.CAUSAL_WORDS.search(text), text
    assert draft.BACKTICKED.findall("a `x` and `y z`") == ["x", "y z"]


def test_a_source_description_reads_a_record_with_defaults():
    plain = draft.SourceDescription.from_dict({"uri": 5, "columns": ["a", "b"]})
    assert plain == draft.SourceDescription("5", ("a", "b"), (), None)
    full = draft.SourceDescription.from_dict(
        {
            "uri": "u",
            "columns": ["a", "b"],
            "primary_key": ["a"],
            "target_time_column": "b",
        }
    )
    assert full.primary_key == ("a",) and full.target_time_column == "b"
    with pytest.raises(KeyError):
        draft.SourceDescription.from_dict({"columns": ["a"]})
    with pytest.raises(KeyError):
        draft.SourceDescription.from_dict({"uri": "u"})


def test_a_data_check_serialises_every_field():
    check = draft.DataCheck("n", ok=True, blocking=False, detail="d")
    assert check.to_dict() == {
        "name": "n",
        "ok": True,
        "blocking": False,
        "detail": "d",
    }


def test_normalise_lowers_and_removes_separators():
    assert draft.normalise("Cust_ID-2 x") == "custid2x"
    assert draft.normalise("a.b") == "ab"


def test_mentioned_matches_words_and_long_names_inside_the_text():
    columns = ["age", "customer_id", "ab", "wxyz", "income"]
    assert draft.mentioned("the Age here", columns) == ["age"]
    assert draft.mentioned("find customer id now", columns) == ["customer_id"]
    assert draft.mentioned("my customerid", columns) == ["customer_id"]
    assert draft.mentioned("xxabxx", columns) == []
    assert draft.mentioned("xxwxyzxx", columns) == []
    assert draft.mentioned("xincomex", columns) == ["income"]
    assert draft.mentioned("ab wxyz", columns) == ["ab", "wxyz"]
    assert draft.mentioned("income age", columns) == ["age", "income"]


def test_a_target_is_taken_from_an_explicit_keyword():
    columns = ["churn", "region", "y"]
    for text in (
        "target: region",
        "Label = `region`",
        "predicting region",
        "predict region",
        "outcome column region",
        "TARGET column: region",
    ):
        slots = draft.deterministic_slots(
            asked(text, columns), pandas.DataFrame()
        )
        assert slots["target"] == "region", text


def test_a_target_falls_back_to_a_single_mention_or_a_conventional_name():
    frame = pandas.DataFrame()
    one = draft.deterministic_slots(
        asked("Who is lost in region?", ["churn", "region"]), frame
    )
    assert one["target"] == "region"
    two = draft.deterministic_slots(
        asked("churn and region", ["churn", "region", "y"]), frame
    )
    assert two["target"] == "y"
    none = draft.deterministic_slots(
        asked("churn and region", ["churn", "region"]), frame
    )
    assert "target" not in none
    twice = draft.deterministic_slots(asked("hello", ["label", "y"]), frame)
    assert "target" not in twice
    unknown = draft.deterministic_slots(
        asked("predict nothing", ["label"]), frame
    )
    assert unknown["target"] == "label"


def test_the_prediction_moment_is_the_only_datetime_column_apart_from_the_end():
    frame = table()
    slots = draft.deterministic_slots(request(frame), frame)
    assert slots["prediction_moment"] == "as_of"
    both = request(frame, target_time_column=None)
    assert "prediction_moment" not in draft.deterministic_slots(both, frame)
    none = draft.deterministic_slots(
        request(frame), frame.drop(columns=["as_of"])
    )
    assert "prediction_moment" not in none
    text = frame.assign(as_of=frame["as_of"].astype(str))
    assert "prediction_moment" not in draft.deterministic_slots(
        request(frame), text
    )


def test_the_unit_is_the_declared_key_and_the_question_is_stripped():
    frame = table()
    slots = draft.deterministic_slots(request(frame, "  Who leaves?  "), frame)
    assert slots["unit"] == ["customer_id"]
    assert slots["question"] == "Who leaves?"
    keyless = draft.Request("q", draft.SourceDescription("u", ("a",)))
    assert "unit" not in draft.deterministic_slots(keyless, pandas.DataFrame())


def checks_of(frame, slots, **options):
    return draft.check_draft(frame, slots, source(frame.columns, **options))


def test_a_missing_target_is_a_blocking_check_with_its_name():
    frame = pandas.DataFrame({"a": [1, 2]})
    (check,) = checks_of(frame, {"target": "zz"})
    assert check == draft.DataCheck(
        "target_exists", ok=False, blocking=True, detail="zz missing"
    )


def test_target_completeness_has_exact_boundaries_and_detail():
    good = pandas.DataFrame({"y": [0, 1, 0, 1, 0, 1, 0, 1, None, None]})
    (check,) = checks_of(good, {"target": "y"})
    assert check == draft.DataCheck(
        "target_complete",
        ok=True,
        blocking=False,
        detail="missing share 0.200; constant=False",
    )
    worse = pandas.DataFrame({"y": [0, 1, 0, 1, 0, 1, 0, None, None, None]})
    (check,) = checks_of(worse, {"target": "y"})
    assert check.ok is False and check.blocking is False
    assert check.detail == "missing share 0.300; constant=False"
    constant = pandas.DataFrame({"y": [1, 1, 1, 1]})
    (check,) = checks_of(constant, {"target": "y"})
    assert (
        check.ok is False
        and check.detail == "missing share 0.000; constant=True"
    )
    two = pandas.DataFrame({"y": [1, 0]})
    assert checks_of(two, {"target": "y"})[0].ok is True
    nulls = pandas.DataFrame({"y": [1.0, None, None, None, None]})
    (check,) = checks_of(nulls, {"target": "y"})
    assert check.ok is False and check.detail.endswith("constant=True")


def test_the_moment_checks_say_what_is_wrong():
    frame = pandas.DataFrame({"a": [1, 2], "e": [3, 4]})
    (missing,) = checks_of(frame, {"prediction_moment": "m"})
    assert missing == draft.DataCheck(
        "moment_exists", ok=False, blocking=True, detail="m missing"
    )
    assert checks_of(frame, {"prediction_moment": "a"}) == ()
    assert (
        checks_of(frame, {"prediction_moment": "a"}, target_time_column="zz")
        == ()
    )


def test_a_moment_equal_to_the_target_time_counts_as_a_violation():
    frame = pandas.DataFrame(
        {
            "m": pandas.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "e": pandas.to_datetime(["2024-01-02", "2024-01-02", "2024-01-04"]),
        }
    )
    (check,) = checks_of(
        frame, {"prediction_moment": "m"}, target_time_column="e"
    )
    assert check == draft.DataCheck(
        "moment_before_target",
        ok=False,
        blocking=True,
        detail="1 rows have the moment at or after the target time",
    )
    clean = frame.assign(e=pandas.to_datetime(["2024-02-01"] * 3))
    (check,) = checks_of(
        clean, {"prediction_moment": "m"}, target_time_column="e"
    )
    assert check.ok is True and check.blocking is True
    assert check.detail == "0 rows have the moment at or after the target time"
    after = frame.assign(
        e=pandas.to_datetime(["2023-01-01", "2024-01-02", "2024-01-04"])
    )
    (check,) = checks_of(
        after, {"prediction_moment": "m"}, target_time_column="e"
    )
    assert check.detail.startswith("2 rows")
    unparsable = frame.assign(e=["x", "y", "z"])
    (check,) = checks_of(
        unparsable, {"prediction_moment": "m"}, target_time_column="e"
    )
    assert check.ok is True


def test_the_unit_checks_name_the_keys_and_the_duplicate_share():
    frame = pandas.DataFrame({"k": list(range(100)), "v": [1] * 100})
    (missing,) = checks_of(frame, {"unit": ["k", "zz"]})
    assert missing == draft.DataCheck(
        "unit_exists", ok=False, blocking=True, detail="['k', 'zz'] missing"
    )
    (text,) = checks_of(frame, {"unit": "zz"})
    assert text.detail == "['zz'] missing"
    (unique,) = checks_of(frame, {"unit": "k"})
    assert unique == draft.DataCheck(
        "unit_unique", ok=True, blocking=False, detail="duplicate share 0.0000"
    )
    one = frame.assign(k=[0, *range(99)])
    (edge,) = checks_of(one, {"unit": ["k"]})
    assert edge.ok is True and edge.detail == "duplicate share 0.0100"
    two = frame.assign(k=[0, 0, *range(98)])
    (over,) = checks_of(two, {"unit": ["k"]})
    assert over.ok is False and over.blocking is False
    assert over.detail == "duplicate share 0.0200"
    assert checks_of(frame, {"unit": ""}) == ()
    assert checks_of(frame, {"unit": []}) == ()


def test_the_population_check_reports_the_selected_share():
    frame = pandas.DataFrame({"k": list(range(100))})
    (edge,) = checks_of(frame, {"population": "k < 1"})
    assert edge == draft.DataCheck(
        "population_coverage",
        ok=True,
        blocking=False,
        detail="selects 0.010 of the rows",
    )
    (tiny,) = checks_of(frame, {"population": "k > 98.5"})
    assert tiny.ok is True and tiny.detail == "selects 0.010 of the rows"
    (none,) = checks_of(frame, {"population": "k > 1000"})
    assert none.ok is False and none.blocking is False
    assert none.detail == "selects 0.000 of the rows"
    (everyone,) = checks_of(frame, {"population": "k >= 0"})
    assert everyone.ok is False and everyone.blocking is False
    assert everyone.detail == "selects 1.000 of the rows"
    (half,) = checks_of(frame, {"population": "k < 50"})
    assert half.ok is True and half.detail == "selects 0.500 of the rows"
    (odd,) = checks_of(frame, {"population": "nonsense"})
    assert odd == draft.DataCheck(
        "population_coverage",
        ok=False,
        blocking=True,
        detail="predicate not understood",
    )
    assert checks_of(frame, {"population": ""}) == ()
    empty = pandas.DataFrame({"k": []})
    (check,) = checks_of(empty, {"population": "k < 1"})
    assert check.ok is False and check.detail == "selects 0.000 of the rows"


def test_the_checks_come_in_order_and_absent_slots_are_skipped():
    frame = table()
    slots = {
        "population": "region == 'n'",
        "unit": ["customer_id"],
        "prediction_moment": "as_of",
        "target": "churn",
    }
    names = [
        c.name
        for c in checks_of(frame, slots, target_time_column="resolved_at")
    ]
    assert names == [
        "target_complete",
        "moment_before_target",
        "unit_unique",
        "population_coverage",
    ]
    assert checks_of(frame, {}) == ()
    assert checks_of(frame, {"question": "q"}) == ()


def slot_set(**changes):
    slots = {
        "unit": ["customer_id"],
        "target": "churn",
        "prediction_moment": "as_of",
        "question": "Who leaves?",
    }
    slots.update(changes)
    return slots


def test_assembly_fills_blanks_with_placeholders_and_lists_them():
    frame = table()
    plain = request(frame, "Who leaves?")
    charter, open_items = draft.assemble(slot_set(), plain, None)
    record = charter.to_dict()
    assert record["question"] == "Who leaves?"
    assert record["estimand"]["kind"] == "predictive"
    assert record["estimand"]["population"] == (
        "all rows of warehouse://crm/accounts"
    )
    assert record["estimand"]["target"] == "churn"
    assert record["estimand"]["prediction_moment"] == "as_of"
    assert record["estimand"]["summary_measure"] == draft.UNSPECIFIED
    assert record["estimand"]["intercurrent_events"] == [
        {"event": "none", "strategy": "not applicable"}
    ]
    assert record["unit"] == "customer_id"
    assert record["deployment_population"] == "same as warehouse://crm/accounts"
    assert record["cost_model"] is None if "cost_model" in record else True
    assert open_items == [
        "population",
        "summary_measure",
        "deployment_population",
        "intercurrent_events",
        "cost_model",
    ]


def test_assembly_uses_given_slots_and_the_cost_model():
    frame = table()
    slots = slot_set(
        population="region == 'n'",
        deployment_population="all of them",
        summary_measure="auc",
        unit=["a", "b"],
        question="",
    )
    charter, open_items = draft.assemble(slots, request(frame, "x"), MODEL)
    record = charter.to_dict()
    assert record["estimand"]["population"] == "region == 'n'"
    assert record["deployment_population"] == "all of them"
    assert record["unit"] == "a-b"
    assert record["question"] == draft.UNSPECIFIED
    assert record["estimand"]["summary_measure"] == (
        "net benefit over thresholds [0.05, 0.5]"
    )
    assert charter.cost_model == MODEL
    assert open_items == ["intercurrent_events"]
    with_text = draft.assemble(
        slot_set(unit="solo", summary_measure="auc"), request(frame, "x"), None
    )[0]
    assert with_text.unit == "solo"
    assert with_text.estimand.summary_measure == "auc"


def test_causal_words_set_the_kind_and_a_missing_required_slot_raises():
    frame = table()
    causal = request(frame, "Because of price, who leaves?")
    assert draft.assemble(slot_set(), causal, None)[0].estimand.kind == "causal"
    for name in ("unit", "target", "prediction_moment"):
        broken = slot_set()
        del broken[name]
        with pytest.raises(KeyError):
            draft.assemble(broken, causal, None)


def test_a_refused_draft_carries_the_reason_the_detail_and_the_context():
    check = draft.DataCheck("c", ok=False, blocking=True, detail="d")
    result = draft.refused(
        "why", {"k": 1}, [check], {"model_id": "m"}, ["gone"]
    )
    assert result == draft.Draft(
        "refused",
        None,
        {},
        (check,),
        (),
        (),
        (),
        None,
        {"reason": "why", "k": 1},
        {"model_id": "m"},
        ("gone",),
    )
    bare = draft.refused("r", {})
    assert bare.checks == () and bare.discarded == () and bare.llm is None
    assert bare.refusal == {"reason": "r"}


def test_column_refusals_name_the_missing_columns():
    frame = table()
    unknown = draft.column_refusal(
        request(frame, "use `nope` and `two words` and `customer_id`"), frame
    )
    assert unknown.refusal == {
        "reason": "request names columns the source does not have",
        "missing_columns": ["nope"],
    }
    assert unknown.status == draft.REFUSED
    lacking = draft.column_refusal(
        request(frame), frame.drop(columns=["region"])
    )
    assert lacking.refusal == {
        "reason": "the data sample lacks declared columns",
        "missing_columns": ["region"],
    }
    assert draft.column_refusal(request(frame, "use `churn`"), frame) is None


def test_without_a_sampler_nothing_is_asked():
    frame = table()
    slots = {}
    sources = {}
    got = draft.consult_sampler(
        request(frame), frame, slots, sources, None, 0.1, 5
    )
    assert got == (None, [], None)
    assert slots == {} and sources == {}


def test_the_llm_record_holds_the_prompt_the_samples_and_the_scores():
    frame = table()
    plain = request(frame, "Who leaves?")
    slots = {}
    sources = {}
    sampler = replay([SAMPLE] * 5, "model-x")
    llm, discarded, stopped = draft.consult_sampler(
        plain, frame, slots, sources, sampler, 0.1, 3
    )
    prompt = refusal.prompt_for("Who leaves?", plain.source.columns)
    samples = [SAMPLE] * 3
    assert stopped is None and discarded == []
    assert llm == {
        "model_id": "model-x",
        "prompt": prompt,
        "samples": samples,
        "samples_hash": refusal.samples_hash("model-x", prompt, samples),
        "ambiguity": 0.0,
        "agreement": {
            "target": 1.0,
            "prediction_moment": 1.0,
            "unit": 1.0,
            "population": 1.0,
        },
        "tau": 0.1,
        "tier": "unverified",
    }
    assert slots["target"] == "churn" and slots["unit"] == ["customer_id"]
    assert set(sources.values()) == {provenance.FieldSource.LLM}
    assert set(slots) == set(SAMPLE)


def test_an_ambiguous_draft_is_refused_with_the_conflicts_and_the_record():
    frame = table()
    mixed = [SAMPLE] * 3 + [{**SAMPLE, "target": "region"}] * 1
    llm, discarded, stopped = draft.consult_sampler(
        request(frame), frame, {}, {}, replay(mixed), 0.1, 4
    )
    assert discarded == []
    assert stopped.status == draft.REFUSED
    assert stopped.refusal == {
        "reason": "ambiguity above tau, or no calibrated tau on file",
        "ambiguity": 0.25,
        "tau": 0.1,
        "conflicts": ["target"],
    }
    assert stopped.llm == llm and stopped.charter is None
    equal = draft.consult_sampler(
        request(frame), frame, {}, {}, replay(mixed), 0.25, 4
    )
    assert equal[2] is None


def test_an_llm_slot_never_overrides_a_template_slot_and_blanks_are_dropped():
    frame = table()
    slots = {"target": "region"}
    sources = {"target": provenance.FieldSource.TEMPLATE}
    blanks = {**SAMPLE, "population": "", "deployment_population": None}
    draft.consult_sampler(
        request(frame), frame, slots, sources, replay([blanks] * 4), 0.1, 4
    )
    assert slots["target"] == "region"
    assert sources["target"] is provenance.FieldSource.TEMPLATE
    assert "population" not in slots and "deployment_population" not in slots
    assert slots["unit"] == ["customer_id"]
    assert sources["unit"] is provenance.FieldSource.LLM


def test_a_proposal_that_fails_a_blocking_check_is_discarded_whole():
    frame = table()
    bad = {**SAMPLE, "unit": ["not_a_column"]}
    slots = {}
    sources = {}
    llm, discarded, stopped = draft.consult_sampler(
        request(frame), frame, slots, sources, replay([bad] * 4), 0.1, 4
    )
    assert stopped is None and llm is not None
    assert discarded == ["llm draft failed a blocking data-side check"]
    assert slots == {} and sources == {}


def test_the_baseline_is_profiled_only_with_a_column_present_and_costs():
    frame = table()
    items = []
    profile, asked_, robust = draft.baseline_and_questions(
        frame, "churn", "current_rule", MODEL, None, items
    )
    assert profile is not None and len(profile["thresholds"]) == 5
    assert profile["thresholds"][0] == 0.05 and profile["thresholds"][-1] == 0.5
    assert asked_ == [] and robust == [] and items == []
    missing = draft.baseline_and_questions(
        frame, "churn", "absent", MODEL, None, items
    )
    assert missing[0] is None and items == []
    no_costs = draft.baseline_and_questions(
        frame, "churn", "current_rule", None, None, items
    )
    assert no_costs[0] is None and items == []


def test_no_baseline_column_is_an_open_item_only_when_none_is_named():
    frame = table()
    items = ["cost_model"]
    profile, asked_, robust = draft.baseline_and_questions(
        frame, "churn", None, MODEL, None, items
    )
    assert profile is None and asked_ == [] and robust == []
    assert items == ["cost_model", "baseline_decision_column"]
    named = ["x"]
    draft.baseline_and_questions(frame, "churn", "absent", None, None, named)
    assert named == ["x"]


def pilot_world(better):
    n = 200
    y = numpy.array([1, 0] * (n // 2))
    base = numpy.zeros(n, dtype=int)
    frame = pandas.DataFrame({"churn": y, "rule": base})
    actions = y.copy() if better else numpy.zeros(n, dtype=int)
    return frame, actions


def test_the_cost_threshold_is_asked_only_when_the_verdict_can_change():
    frame = pandas.DataFrame(
        {
            "churn": [1.0] * 50 + [0.0] * 150,
            "rule": [1] * 20 + [0] * 30 + [0] * 150,
        }
    )
    actions = [1] * 50 + [1] * 30 + [0] * 120
    items = ["cost_model"]
    result = draft.baseline_and_questions(
        frame, "churn", "rule", None, actions, items
    )
    questions, robust = result[1], result[2]
    assert result[0] is None and robust == []
    assert len(questions) == 1 and items == ["cost_model"]
    (question,) = questions
    assert question.slot == "cost_threshold"
    assert question.low == 0.05 and question.high == 0.5
    assert question.form == "a threshold t = c_FP / (c_FP + c_FN) in (0, 1)"
    assert question.verdicts == ("model_better", "baseline_better")
    assert 0.05 < question.tipping_value < 0.5


def test_a_robust_threshold_is_reported_and_closes_the_cost_item():
    frame, actions = pilot_world(better=True)
    items = ["cost_model"]
    result = draft.baseline_and_questions(
        frame, "churn", "rule", None, actions, items
    )
    questions, robust = result[1], result[2]
    assert questions == [] and items == []
    assert robust == [
        {
            "slot": "cost_threshold",
            "low": 0.05,
            "high": 0.5,
            "verdict": "model_better",
        }
    ]


def test_unobserved_outcomes_are_left_out_of_the_threshold_question():
    frame, actions = pilot_world(better=True)
    frame = frame.astype({"churn": float})
    frame.loc[::3, "churn"] = numpy.nan
    items = ["cost_model"]
    result = draft.baseline_and_questions(
        frame, "churn", "rule", None, actions, items
    )
    questions, robust = result[1], result[2]
    assert questions == [] and len(robust) == 1


def test_no_pilot_actions_means_no_threshold_question():
    frame = pilot_world(better=True)[0]
    items = ["cost_model"]
    assert draft.baseline_and_questions(
        frame, "churn", "rule", None, None, items
    ) == (None, [], [])
    assert items == ["cost_model"]


def test_a_full_draft_records_sources_checks_and_the_baseline():
    frame = table()
    result = draft.frame(
        request(frame),
        frame,
        cost_model=MODEL,
        baseline_column="current_rule",
    )
    assert result.status == draft.WITH_QUESTIONS
    assert result.open_items == (
        "population",
        "deployment_population",
        "intercurrent_events",
    )
    assert result.refusal is None and result.llm is None
    assert result.discarded == () and result.questions == ()
    assert result.baseline is not None
    assert [c.name for c in result.checks] == [
        "target_complete",
        "moment_before_target",
        "unit_unique",
    ]
    assert result.fields == {
        "target": provenance.FieldSource.TEMPLATE,
        "prediction_moment": provenance.FieldSource.TEMPLATE,
        "unit": provenance.FieldSource.TEMPLATE,
        "question": provenance.FieldSource.TEMPLATE,
        "population": provenance.FieldSource.TEMPLATE,
        "deployment_population": provenance.FieldSource.TEMPLATE,
        "intercurrent_events": provenance.FieldSource.TEMPLATE,
    }


def test_human_slots_override_every_other_source():
    frame = table()
    result = draft.frame(
        request(frame),
        frame,
        cost_model=MODEL,
        human={"target": "region", "population": "region == 'n'"},
    )
    assert result.charter.estimand.target == "region"
    assert result.charter.estimand.population == "region == 'n'"
    assert result.fields["target"] is provenance.FieldSource.HUMAN
    assert result.fields["population"] is provenance.FieldSource.HUMAN
    assert "population" not in result.open_items
    assert "population_coverage" in [c.name for c in result.checks]


def test_a_failed_blocking_check_refuses_and_keeps_the_checks():
    frame = table()
    frame["resolved_at"] = frame["as_of"]
    result = draft.frame(request(frame), frame)
    assert result.status == draft.REFUSED
    assert result.refusal == {
        "reason": "a blocking data-side check failed",
        "failed": ["moment_before_target"],
    }
    assert [c.name for c in result.checks][-2:] == [
        "moment_before_target",
        "unit_unique",
    ]
    assert result.charter is None and result.fields == {}


def test_a_non_blocking_failure_does_not_refuse():
    frame = table()
    frame["churn"] = 1
    result = draft.frame(request(frame), frame)
    assert result.status == draft.WITH_QUESTIONS
    assert result.checks[0].ok is False


def test_unfilled_critical_slots_are_listed_with_a_question_each():
    frame = pandas.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = draft.frame(asked("hello", ["a", "b"]), frame)
    assert result.refusal == {
        "reason": "critical slots cannot be filled from the request and the "
        "schema",
        "unfilled": ["target", "prediction_moment", "unit"],
        "ask": {
            "target": "name the column",
            "prediction_moment": "name the column",
            "unit": "name the column",
        },
    }
    assert result.checks == ()


def test_a_draft_the_charter_rejects_is_refused_with_the_error():
    frame = table()
    frame[" "] = [0, 1] * 200
    result = draft.frame(request(frame), frame, human={"target": " "})
    assert result.status == draft.REFUSED
    assert result.refusal["reason"] == "the draft does not form a valid charter"
    assert result.refusal["error"]
    assert result.charter is None


def test_the_llm_record_and_discards_reach_the_final_draft():
    frame = table()
    bad = {**SAMPLE, "prediction_moment": "not_a_column"}
    result = draft.frame(
        request(frame),
        frame,
        sampler=replay([bad] * 3),
        tau=0.1,
        k=3,
    )
    assert result.discarded == ("llm draft failed a blocking data-side check",)
    assert result.llm["model_id"] == "recorded-model"
    assert len(result.llm["samples"]) == 3


def test_a_refusal_after_sampling_keeps_the_llm_record():
    frame = table()
    frame["resolved_at"] = frame["as_of"]
    result = draft.frame(
        request(frame), frame, sampler=replay([SAMPLE] * 2), tau=0.1, k=2
    )
    assert result.status == draft.REFUSED
    assert result.llm["tier"] == "unverified"
    result = draft.frame(
        asked("hello", ["a", "b"]),
        pandas.DataFrame({"a": [1], "b": [2]}),
        sampler=replay([{"question": "q"}] * 2),
        tau=1.0,
        k=2,
    )
    assert result.llm is not None and result.llm["tau"] == 1.0
    assert result.refusal["unfilled"]


def test_the_default_number_of_draws_is_the_documented_one():
    frame = table()
    result = draft.frame(
        request(frame), frame, sampler=replay([SAMPLE] * 10), tau=0.1
    )
    assert len(result.llm["samples"]) == refusal.DEFAULT_K


def test_a_draft_serialises_to_the_stored_document():
    frame = table()
    result = draft.frame(request(frame), frame, cost_model=MODEL)
    document = result.to_dict()
    assert document["schema_version"] == 1
    assert document["status"] == draft.WITH_QUESTIONS
    assert document["charter"] == result.charter.to_dict()
    assert document["charter_hash"] == result.charter.charter_hash
    assert document["fields"] == {
        k: v.value for k, v in sorted(result.fields.items())
    }
    assert list(document["fields"]) == sorted(document["fields"])
    assert document["checks"] == [c.to_dict() for c in result.checks]
    assert document["open_items"] == list(result.open_items)
    assert document["llm"] is None and document["discarded"] == []
    assert document["label"] == (
        "a drafted charter is a draft: nothing here shows it is correct; llm "
        "fields are unverified until a person signs"
    )
    assert set(document) == {
        "schema_version",
        "status",
        "charter",
        "charter_hash",
        "fields",
        "checks",
        "open_items",
        "llm",
        "discarded",
        "label",
    }


def test_a_refused_draft_has_no_charter_in_its_document():
    frame = table()
    result = draft.frame(request(frame, "Predict `nope`."), frame)
    document = result.to_dict()
    assert document["charter"] is None and document["charter_hash"] is None
    assert document["status"] == "refused"


def test_the_draft_hash_is_the_sha256_of_the_canonical_document():
    frame = table()
    result = draft.frame(request(frame), frame)
    assert (
        result.draft_hash
        == hashlib.sha256(
            signing.canonical_payload(result.to_dict())
        ).hexdigest()
    )


def test_the_artifacts_hold_the_documents_and_the_directory_is_created(
    tmp_path,
):
    frame = table()
    result = draft.frame(
        request(frame), frame, cost_model=MODEL, baseline_column="current_rule"
    )
    target = tmp_path / "deep" / "er"
    names = draft.write_artifacts(result, target)
    assert names == sorted(
        [
            "framing_draft.json",
            "framing_questions.json",
            "decision_cost_model.json",
            "baseline_report.json",
        ]
    )
    stored = (target / "framing_draft.json").read_text(encoding="utf-8")
    assert stored == refusal.dump(result.to_dict()) + "\n"
    questions = json.loads(
        (target / "framing_questions.json").read_text(encoding="utf-8")
    )
    assert questions == {
        "questions": [],
        "robust": [],
        "open_items": list(result.open_items),
    }
    assert (
        json.loads(
            (target / "decision_cost_model.json").read_text(encoding="utf-8")
        )
        == MODEL.to_dict()
    )
    assert json.loads(
        (target / "baseline_report.json").read_text(encoding="utf-8")
    ) == json.loads(refusal.dump(result.baseline))
    assert not (target / "framing_refusal.json").exists()
    assert (target / "framing_draft.json").read_bytes().endswith(b"}\n")


def test_a_draft_without_costs_or_a_baseline_writes_only_the_basics(tmp_path):
    frame = table()
    result = draft.frame(request(frame), frame)
    assert draft.write_artifacts(result, tmp_path) == [
        "framing_draft.json",
        "framing_questions.json",
    ]


def test_a_refusal_is_written_with_its_reason(tmp_path):
    frame = table()
    result = draft.frame(request(frame, "Predict `nope`."), frame)
    names = draft.write_artifacts(result, tmp_path)
    assert names == [
        "framing_draft.json",
        "framing_questions.json",
        "framing_refusal.json",
    ]
    stored = json.loads((tmp_path / "framing_refusal.json").read_text())
    assert stored["missing_columns"] == ["nope"]


def test_the_link_report_matches_parts_of_each_field_to_columns():
    frame = table()
    result = draft.frame(request(frame), frame)
    columns = list(frame.columns)
    link = draft.link_report(result.charter, columns)
    assert link == {
        "schema_version": 1,
        "fields": {
            "target": {
                "text": "churn",
                "columns": ["churn"],
                "status": "resolved",
            },
            "prediction_moment": {
                "text": "as_of",
                "columns": ["as_of"],
                "status": "resolved",
            },
            "unit": {
                "text": "customer_id",
                "columns": ["customer_id"],
                "status": "resolved",
            },
        },
        "gating": False,
    }
    partial = draft.link_report(result.charter, ["churn", "x"])
    assert partial["fields"]["unit"] == {
        "text": "customer_id",
        "columns": [],
        "status": "unresolved",
    }


def test_the_link_report_splits_on_the_documented_separators():
    frame = table()
    plain = request(frame)
    long = "customer_id, region + as_of & churn;resolved_at"
    charter = draft.assemble(slot_set(unit=long), plain, None)[0]
    link = draft.link_report(charter, list(frame.columns))
    assert link["fields"]["unit"]["columns"] == [
        "customer_id",
        "region",
        "as_of",
        "churn",
        "resolved_at",
    ]
    spaced = draft.assemble(slot_set(unit="region customer_id"), plain, None)[0]
    assert draft.link_report(spaced, list(frame.columns))["fields"]["unit"][
        "columns"
    ] == ["region", "customer_id"]
    dashed = draft.assemble(slot_set(unit="as_of-churn"), plain, None)[0]
    assert draft.link_report(dashed, list(frame.columns))["fields"]["unit"][
        "columns"
    ] == ["as_of", "churn"]


def test_a_charter_target_with_an_unnormalised_name_still_resolves():
    frame = table()
    charter = draft.assemble(slot_set(target="Churn"), request(frame), None)[0]
    link = draft.link_report(charter, list(frame.columns))
    assert link["fields"]["target"]["columns"] == ["churn"]


def test_an_explicit_target_keyword_wins_over_an_ambiguous_mention():
    frame = pandas.DataFrame()
    slots = draft.deterministic_slots(
        asked("Predict churn from region", ["churn", "region"]), frame
    )
    assert slots["target"] == "churn"
    upper = draft.deterministic_slots(
        asked("TARGET churn with region", ["churn", "region"]), frame
    )
    assert upper["target"] == "churn"
    lower = draft.deterministic_slots(
        asked("predicting churn, region matters", ["churn", "region"]), frame
    )
    assert lower["target"] == "churn"


def test_blank_slot_values_are_placeholders_and_open_items():
    frame = table()
    slots = slot_set(
        population="", deployment_population=[], summary_measure=""
    )
    charter, open_items = draft.assemble(slots, request(frame), None)
    assert charter.estimand.population == "all rows of warehouse://crm/accounts"
    assert charter.deployment_population == "same as warehouse://crm/accounts"
    assert charter.estimand.summary_measure == draft.UNSPECIFIED
    assert open_items[:3] == [
        "population",
        "summary_measure",
        "deployment_population",
    ]
    nothing = draft.assemble(
        slot_set(population=None, deployment_population=None),
        request(frame),
        None,
    )[1]
    assert "population" in nothing and "deployment_population" in nothing


class Prompts:
    model_id = "p"

    def __init__(self):
        self.prompts = []

    def sample(self, prompt, index):
        self.prompts.append((prompt, index))
        return SAMPLE


def test_the_sampler_is_given_the_prompt_and_each_draw_index():
    frame = table()
    plain = request(frame, "Who leaves?")
    sampler = Prompts()
    draft.consult_sampler(plain, frame, {}, {}, sampler, 0.1, 3)
    prompt = refusal.prompt_for("Who leaves?", plain.source.columns)
    assert sampler.prompts == [(prompt, 0), (prompt, 1), (prompt, 2)]


def test_non_numeric_outcomes_are_treated_as_unobserved_in_the_profile():
    frame = table()
    frame["churn"] = frame["churn"].astype(object)
    frame.loc[::7, "churn"] = "n/a"
    result = draft.frame(
        request(frame),
        frame,
        cost_model=MODEL,
        baseline_column="current_rule",
    )
    assert result.baseline is not None
    assert result.baseline["selective_labels"]["flagged"] is False


def test_non_numeric_outcomes_do_not_stop_the_threshold_question():
    frame = pilot_world(better=True)[0].astype(object)
    frame.loc[::7, "churn"] = "n/a"
    items = ["cost_model"]
    result = draft.baseline_and_questions(
        frame, "churn", "rule", None, [1, 0] * 100, items
    )
    assert result[0] is None and len(result[1]) + len(result[2]) == 1


def test_a_refusal_after_discarding_keeps_what_was_discarded():
    frame = table()
    bad = {**SAMPLE, "prediction_moment": "not_a_column"}
    discarded = ("llm draft failed a blocking data-side check",)
    keyless = draft.Request(
        "Who leaves?", draft.SourceDescription("u", tuple(frame.columns))
    )
    unfilled = draft.frame(
        keyless, frame, sampler=replay([bad] * 3), tau=0.1, k=3
    )
    assert unfilled.refusal["unfilled"] and unfilled.discarded == discarded
    blocked = draft.frame(
        request(frame),
        frame,
        sampler=replay([bad] * 3),
        tau=0.1,
        k=3,
        human={"unit": ["not_a_key"]},
    )
    assert blocked.refusal["failed"] == ["unit_exists"]
    assert blocked.discarded == discarded
    assert blocked.llm["model_id"] == "recorded-model"


def test_artifacts_are_written_as_utf8_with_unix_newlines(
    tmp_path, monkeypatch
):
    frame = table()
    result = draft.frame(request(frame), frame, cost_model=MODEL)
    seen = []
    real = pathlib.Path.write_text

    def spy(self, data, encoding=None, errors=None, newline=None):
        seen.append((encoding, newline))
        return real(
            self, data, encoding=encoding, errors=errors, newline=newline
        )

    monkeypatch.setattr(pathlib.Path, "write_text", spy)
    draft.write_artifacts(result, tmp_path)
    assert seen and set(seen) == {("utf-8", "\n")}


def test_an_unparsable_moment_is_not_counted_as_a_violation():
    frame = pandas.DataFrame(
        {
            "m": ["garbage", "2024-01-01", "also garbage"],
            "e": pandas.to_datetime(["2024-02-01", "2023-01-01", "2024-02-01"]),
        }
    )
    (check,) = checks_of(
        frame, {"prediction_moment": "m"}, target_time_column="e"
    )
    assert check.detail == "1 rows have the moment at or after the target time"
    assert check.ok is False


def test_a_single_row_sample_selected_whole_is_not_a_subpopulation():
    frame = pandas.DataFrame({"k": [1]})
    (check,) = checks_of(frame, {"population": "k > 0"})
    assert check.ok is False and check.detail == "selects 1.000 of the rows"


def test_a_rejected_charter_refusal_carries_the_error_checks_and_llm_record():
    frame = table()
    frame[" "] = [0, 1] * 200
    bad = {**SAMPLE, "prediction_moment": "not_a_column"}
    plain = request(frame)
    with pytest.raises(Exception) as caught:
        draft.assemble(
            {
                "target": " ",
                "prediction_moment": "as_of",
                "unit": ["customer_id"],
            },
            plain,
            None,
        )
    result = draft.frame(
        plain,
        frame,
        sampler=replay([bad] * 3),
        tau=0.1,
        k=3,
        human={"target": " "},
    )
    assert result.refusal["error"] == str(caught.value) != "None"
    assert result.checks[0].name == "target_complete"
    assert result.llm["model_id"] == "recorded-model"
    assert result.discarded == ("llm draft failed a blocking data-side check",)
