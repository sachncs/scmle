"""Pins the slot sampler, the canonical forms, the consensus and the cut-off."""

import dataclasses
import hashlib
import json

import pandas
import pytest

from scmle.contracts import errors
from scmle.data.framing import refusal


def modal(result):
    return dataclasses.asdict(result)["values"]


def frame():
    return pandas.DataFrame(
        {
            "region": ["n", "s", "n", "e", "s"],
            "age": [10, 20, 30, 40, 50],
        }
    )


class Recorder:
    model_id = "m"

    def __init__(self):
        self.calls = []

    def sample(self, prompt, index):
        self.calls.append((prompt, index))
        return {"target": "t", "junk": 1, "unit": ["a"]}


def test_the_constants_are_the_documented_ones():
    assert refusal.CRITICAL_SLOTS == (
        "target",
        "prediction_moment",
        "unit",
        "population",
    )
    assert refusal.SLOT_NAMES == (
        "question",
        "target",
        "prediction_moment",
        "unit",
        "population",
        "deployment_population",
        "summary_measure",
    )
    assert refusal.DEFAULT_K == 10
    assert refusal.DEFAULT_EPSILON == 0.05


def test_the_prompt_lists_slots_columns_and_request_in_order():
    assert refusal.prompt_for("Who leaves?", ["a", "b"]) == (
        "Fill these charter slots as JSON: question, target, "
        "prediction_moment, unit, population, deployment_population, "
        "summary_measure. Columns: a, b. Request: Who leaves?"
    )


def test_the_replay_sampler_copies_and_serves_by_index():
    original = [{"target": "x"}, {"target": "y"}]
    sampler = refusal.ReplaySampler("model", original)
    original[0]["target"] = "changed"
    original.append({"target": "z"})
    assert sampler.model_id == "model"
    assert sampler.recorded == [{"target": "x"}, {"target": "y"}]
    assert sampler.sample("p", 0) == {"target": "x"}
    assert sampler.sample("p", 1) == {"target": "y"}
    with pytest.raises(errors.ConfigError) as caught:
        sampler.sample("p", 2)
    assert str(caught.value) == "no recorded sample for this draw"


def test_draw_asks_k_times_with_the_index_and_keeps_known_slots():
    sampler = Recorder()
    out = refusal.draw(sampler, "prompt", 3)
    assert out == [{"target": "t", "unit": ["a"]}] * 3
    assert sampler.calls == [("prompt", 0), ("prompt", 1), ("prompt", 2)]
    assert len(refusal.draw(Recorder(), "p")) == 10


def test_the_samples_hash_covers_model_prompt_and_samples():
    body = {"model": "m", "prompt": "p", "samples": [{"a": 1}]}
    expected = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert refusal.samples_hash("m", "p", [{"a": 1}]) == expected
    assert refusal.samples_hash("m2", "p", [{"a": 1}]) != expected
    assert refusal.samples_hash("m", "p2", [{"a": 1}]) != expected
    assert refusal.samples_hash("m", "p", [{"a": 2}]) != expected


def test_population_predicates_select_row_positions_for_each_operator():
    data = frame()
    cases = {
        "age == 30": {2},
        "age != 30": {0, 1, 3, 4},
        "age < 30": {0, 1},
        "age <= 30": {0, 1, 2},
        "age > 30": {3, 4},
        "age >= 30": {2, 3, 4},
        "age > -5": {0, 1, 2, 3, 4},
        "age < 20.5": {0, 1},
        "region == 'n'": {0, 2},
        'region == "s"': {1, 4},
        "region != 'n'": {1, 3, 4},
        "  age>=30  ": {2, 3, 4},
    }
    for text, expected in cases.items():
        assert refusal.population_ids(data, text) == frozenset(expected), text
    assert isinstance(refusal.population_ids(data, "age == 30"), frozenset)


def test_population_predicates_join_with_and_and_reject_everything_else():
    data = frame()
    assert refusal.population_ids(data, "age > 10 and region == 's'") == (
        frozenset({1, 4})
    )
    assert refusal.population_ids(data, "age > 10  AND region == 's'") is None
    assert refusal.population_ids(
        data, "age > 10 and age < 40 and age != 20"
    ) == (frozenset({2}))
    for bad in (
        "age > 10 and",
        "nope == 1",
        "age ~ 1",
        "__import__('os')",
        "age == 1 or age == 2",
        "region > 5",
        "age == 'x' and region == 'n'",
    ):
        assert refusal.population_ids(data, bad) is None, bad


def test_empty_and_all_like_predicates_select_every_row():
    data = frame()
    everything = frozenset(range(5))
    for text in ("", None, "true", "TRUE", " All ", "all", False):
        assert refusal.population_ids(data, text) == everything, text
    assert refusal.population_ids(data.iloc[0:0], "") == frozenset()


def test_a_type_mismatch_in_an_ordering_is_invalid_not_an_error():
    assert refusal.population_ids(frame(), "region < 5") is None
    assert refusal.population_ids(frame(), "region <= 5") is None
    assert refusal.population_ids(frame(), "region >= 5") is None
    assert refusal.population_ids(frame(), "region > 5") is None


def test_canonical_forms_by_slot():
    data = frame()
    assert refusal.canonical(data, "unit", "b") == ("b",)
    assert refusal.canonical(data, "unit", "ab") == ("ab",)
    assert refusal.canonical(data, "unit", ["b", "a"]) == ("a", "b")
    assert refusal.canonical(data, "unit", None) == ()
    assert refusal.canonical(data, "unit", [3, 1]) == ("1", "3")
    assert refusal.canonical(data, "population", "age > 30") == frozenset(
        {3, 4}
    )
    assert refusal.canonical(data, "population", "zzz") is None
    assert refusal.canonical(data, "target", None) is None
    assert refusal.canonical(data, "target", 5) == "5"
    assert refusal.canonical(data, "target", "") == ""


def test_jaccard_is_intersection_over_union():
    assert refusal.jaccard(frozenset({1, 2}), frozenset({2, 3})) == (
        pytest.approx(1 / 3)
    )
    assert refusal.jaccard(frozenset(), frozenset()) == 1.0
    assert refusal.jaccard(frozenset({1}), frozenset()) == 0.0
    assert refusal.jaccard(frozenset({1}), frozenset({1})) == 1.0


def sample(**changes):
    base = {
        "question": "q",
        "target": "t",
        "prediction_moment": "m",
        "unit": ["u"],
        "population": "age > 10",
        "deployment_population": "d",
        "summary_measure": "s",
    }
    base.update(changes)
    return base


def test_consensus_needs_a_sample():
    with pytest.raises(ValueError) as caught:
        refusal.consensus(frame(), [])
    assert str(caught.value) == "at least one sample is needed"


def test_full_agreement_has_no_ambiguity_and_no_conflict():
    result = refusal.consensus(frame(), [sample()] * 4)
    assert result.agreement == {
        "target": 1.0,
        "prediction_moment": 1.0,
        "unit": 1.0,
        "population": 1.0,
    }
    assert result.ambiguity == 0.0
    assert result.conflicts == ()
    assert modal(result) == sample()


def test_modal_values_agreement_and_conflicts_follow_the_counts():
    samples = [
        sample(target="a", question="x"),
        sample(target="b", question="y"),
        sample(target="b", question="y"),
        sample(target="b", question="z", unit=["v"]),
    ]
    result = refusal.consensus(frame(), samples)
    assert modal(result)["target"] == "b"
    assert modal(result)["question"] == "y"
    assert modal(result)["unit"] == ["u"]
    assert result.agreement["target"] == 0.75
    assert result.agreement["unit"] == 0.75
    assert result.agreement["prediction_moment"] == 1.0
    assert result.ambiguity == 0.25
    assert result.conflicts == ("target", "unit")


def test_ties_take_the_first_seen_value_and_its_raw_form():
    samples = [sample(unit="a"), sample(unit=["a"]), sample(unit=["b"])]
    result = refusal.consensus(frame(), samples)
    assert modal(result)["unit"] == "a"
    assert result.agreement["unit"] == pytest.approx(2 / 3)
    ordered = [sample(target="z"), sample(target="y")]
    assert modal(refusal.consensus(frame(), ordered))["target"] == "z"


def test_a_missing_slot_counts_as_one_shared_value():
    samples = [{"target": "t"}, {"target": "t"}]
    result = refusal.consensus(frame(), samples)
    assert modal(result)["question"] is None
    assert result.agreement["target"] == 1.0
    assert result.agreement["prediction_moment"] == 1.0


def test_population_agreement_is_mean_jaccard_scaled_by_valid_share():
    data = frame()
    same = [sample(population="age > 10")] * 2
    assert refusal.consensus(data, same).agreement["population"] == 1.0
    disjoint = [
        sample(population="age < 30"),
        sample(population="age > 30"),
    ]
    assert refusal.consensus(data, disjoint).agreement["population"] == 0.0
    overlap = [
        sample(population="age < 40"),
        sample(population="age > 10"),
    ]
    # {0,1,2} vs {1,2,3,4}: 2 / 5
    assert refusal.consensus(data, overlap).agreement["population"] == (
        pytest.approx(0.4)
    )
    three = [
        sample(population="age > 10"),
        sample(population="age > 10"),
        sample(population="age > 30"),
    ]
    # pairs: 1, 0.5, 0.5 -> mean 2/3
    assert refusal.consensus(data, three).agreement["population"] == (
        pytest.approx(2 / 3)
    )


def test_invalid_population_proposals_lower_the_agreement():
    data = frame()
    one_bad = [
        sample(population="age > 10"),
        sample(population="age > 10"),
        sample(population="zzz"),
        sample(population="age > 10"),
    ]
    result = refusal.consensus(data, one_bad)
    assert result.agreement["population"] == pytest.approx(0.75)
    assert modal(result)["population"] == "age > 10"
    lone = [sample(population="age > 10"), sample(population="zzz")]
    assert refusal.consensus(data, lone).agreement["population"] == 0.5
    leading_bad = [sample(population="zzz"), sample(population="age > 10")]
    assert modal(refusal.consensus(data, leading_bad))["population"] == (
        "age > 10"
    )
    all_bad = [sample(population="zzz")] * 3
    result = refusal.consensus(data, all_bad)
    assert result.agreement["population"] == 0.0
    assert modal(result)["population"] is None
    assert "population" in result.conflicts
    assert result.ambiguity == 1.0


def test_clopper_pearson_upper_edges_and_level():
    assert refusal.clopper_pearson_upper(0, 0) == 1.0
    assert refusal.clopper_pearson_upper(3, 3) == 1.0
    assert refusal.clopper_pearson_upper(4, 3) == 1.0
    assert refusal.clopper_pearson_upper(2, 3) < 1.0
    assert refusal.clopper_pearson_upper(0, 1) == pytest.approx(0.95)
    assert refusal.clopper_pearson_upper(1, 10) == pytest.approx(
        0.3941, abs=1e-4
    )
    assert refusal.clopper_pearson_upper(1, 10, 0.9) == pytest.approx(
        0.3368, abs=1e-4
    )
    assert refusal.clopper_pearson_upper(0, 59) == pytest.approx(
        1 - 0.05 ** (1 / 59)
    )
    assert isinstance(refusal.clopper_pearson_upper(1, 10), float)


def test_the_calibration_record_lists_its_curve_as_lists():
    calibration = refusal.Calibration(0.2, 0.05, 40, ((0.1, 0.5, 0.0, 0.1),))
    assert calibration.to_dict() == {
        "tau": 0.2,
        "epsilon": 0.05,
        "n_calibration": 40,
        "risk_coverage": [[0.1, 0.5, 0.0, 0.1]],
    }


def test_refusal_is_strictly_above_tau_and_always_without_one():
    assert refusal.refuse(0.3, 0.2) is True
    assert refusal.refuse(0.2, 0.2) is False
    assert refusal.refuse(0.1, 0.2) is False
    assert refusal.refuse(0.0, None) is True


def test_dump_is_indented_sorted_json_with_sets_as_lists():
    assert refusal.dump({"b": 1, "a": 2}) == '{\n  "a": 2,\n  "b": 1\n}'
    assert json.loads(refusal.dump({"s": frozenset({1})})) == {"s": [1]}
