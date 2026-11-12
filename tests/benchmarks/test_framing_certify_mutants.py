"""Pins the worlds, parameters and reports of the framing certification.

Nothing here reads the shipped certificate files: a mutated or edited source
changes the hash they record, so the tests build their own stores and
measurement documents.
"""

import json
import math
from pathlib import Path

import numpy
import pytest

from benchmarks.qualification import framing_certify
from scmle.core import randomness
from scmle.data.charter import contracts
from scmle.data.detectors import certificates
from scmle.data.framing import baseline
from scmle.data.framing import costs
from scmle.data.framing import draft
from scmle.data.framing import feasibility


def test_the_size_bound_and_the_power_floor_are_two_standard_errors_off():
    assert framing_certify.size_bound() == pytest.approx(
        0.05 + 2 * math.sqrt(0.05 * 0.95 / 2000)
    )
    assert framing_certify.size_bound(100) == pytest.approx(0.09358898943540674)
    assert framing_certify.size_bound(2000) == pytest.approx(
        0.059746794344808965
    )
    assert framing_certify.power_floor() == pytest.approx(0.7821114561800018)
    assert framing_certify.power_floor(100) == pytest.approx(0.72)
    assert framing_certify.power_floor(400) == pytest.approx(0.76)


def test_a_world_has_clustered_binary_outcomes_and_actions():
    y, a, b, group = framing_certify.world(3, 60, 0.7, 0.2)
    assert group.size == 570
    assert (y.sum(), a.sum(), b.sum()) == (150.0, 186.0, 169.0)
    assert group[:12].tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    assert (y * (group % 7)).sum() == 458.0
    for column in (y, a, b):
        assert set(numpy.unique(column)) <= {0.0, 1.0}
        assert column.dtype == numpy.float64
    sizes = numpy.bincount(group)
    assert len(sizes) == 60
    assert sizes.min() >= 5
    assert sizes.max() <= 15


def test_a_big_world_has_the_stated_cluster_sizes_and_rates():
    y, a, b, group = framing_certify.world(7, 4000, 0.7, 0.2)
    sizes = numpy.bincount(group)
    assert (sizes.min(), sizes.max()) == (5, 15)
    assert group.size / 4000 == pytest.approx(10.0, abs=0.1)
    assert y.mean() == pytest.approx(0.3, abs=0.02)
    assert a[y == 1].mean() == pytest.approx(0.7, abs=0.02)
    assert a[y == 0].mean() == pytest.approx(0.2, abs=0.02)
    assert b[y == 1].mean() == pytest.approx(0.6, abs=0.02)
    assert b[y == 0].mean() == pytest.approx(0.2, abs=0.02)
    cluster_rate = numpy.bincount(group, weights=y) / sizes
    # 0.21 * (rho + (1 - rho) / mean size)
    assert cluster_rate.var() == pytest.approx(0.04, abs=0.005)


def test_a_perfect_model_copies_the_outcome():
    y, a = framing_certify.world(3, 60, 1.0, 0.0)[:2]
    assert (a == y).all()


def test_null_worlds_have_no_gain_at_the_named_end_of_the_range():
    assert framing_certify.null_parameters("null_both") == (0.6, 0.2)
    assert framing_certify.null_parameters("null_hi") == pytest.approx(
        (0.7, 0.2642857142857143)
    )
    assert framing_certify.null_parameters("null_lo") == pytest.approx(
        (0.5, -0.18571428571428567)
    )
    with pytest.raises(ValueError) as caught:
        framing_certify.null_parameters("null_mid")
    assert str(caught.value) == "unknown kind: null_mid"


def test_the_gain_vanishes_where_the_null_says_it_does():
    prevalence = framing_certify.PREVALENCE
    for kind, end in (("null_hi", 0.4), ("null_lo", 0.1)):
        se_a, fp_a = framing_certify.null_parameters(kind)
        weight = costs.odds(end)
        gain = prevalence * (se_a - framing_certify.BASE_SENSITIVITY) - (
            weight
            * (1 - prevalence)
            * (fp_a - framing_certify.BASE_FALSE_POSITIVE)
        )
        assert gain == pytest.approx(0.0, abs=1e-12)


def test_the_claim_p_value_is_the_larger_of_the_two_ends():
    assert framing_certify.claim_p(1, 20, 0.72, 0.14) == pytest.approx(
        0.20925352728035915
    )
    assert framing_certify.claim_p(2, 50, 0.6, 0.2) == pytest.approx(
        0.7921591091930448
    )


def test_size_rows_count_rejections_per_world_and_cluster_count(monkeypatch):
    calls = []

    def fake(seed, clusters, se_a, fp_a):
        calls.append((seed, clusters, se_a, fp_a))
        return framing_certify.ALPHA

    monkeypatch.setattr(framing_certify, "claim_p", fake)
    rows = framing_certify.size_rows(5, 3)
    assert list(rows) == [
        f"{kind}/{g}"
        for kind in ("null_hi", "null_lo", "null_both")
        for g in (20, 50, 200)
    ]
    hi = framing_certify.null_parameters("null_hi")
    assert calls[:3] == [(5 + s, 20, *hi) for s in range(3)]
    assert calls[3:6] == [(5 + s, 50, *hi) for s in range(3)]
    assert calls[-3:] == [
        (5 + s, 200, *framing_certify.null_parameters("null_both"))
        for s in range(3)
    ]
    for key, row in rows.items():
        assert row == {
            "clusters": int(key.split("/")[1]),
            "rejections": 3,
            "runs": 3,
            "rate": 1.0,
            "meets_bound": False,
        }


def test_size_rows_meet_the_bound_when_nothing_is_rejected(monkeypatch):
    monkeypatch.setattr(framing_certify, "claim_p", lambda *a: 0.051)
    rows = framing_certify.size_rows(0, 4)
    assert all(
        row["rejections"] == 0 and row["rate"] == 0.0 and row["meets_bound"]
        for row in rows.values()
    )


def test_size_rows_of_real_worlds_use_the_size_bound():
    rows = framing_certify.size_rows(5, 2)
    assert len(rows) == 9
    for row in rows.values():
        assert row["rate"] == row["rejections"] / 2
        assert row["meets_bound"] == (
            row["rate"] <= framing_certify.size_bound(2)
        )


def cells(ok_by_clusters):
    return {
        f"{kind}/{g}": {"clusters": g, "meets_bound": ok}
        for g, ok in ok_by_clusters.items()
        for kind in ("null_hi", "null_both")
    }


def test_the_minimum_clusters_is_the_smallest_count_from_which_all_pass():
    assert (
        framing_certify.minimum_clusters(cells({20: True, 50: True, 200: True}))
        == 20
    )
    assert (
        framing_certify.minimum_clusters(
            cells({20: False, 50: True, 200: True})
        )
        == 50
    )
    assert (
        framing_certify.minimum_clusters(
            cells({20: True, 50: False, 200: True})
        )
        == 200
    )
    assert (
        framing_certify.minimum_clusters(
            cells({20: True, 50: True, 200: False})
        )
        is None
    )
    mixed = cells({20: True, 50: True, 200: True})
    mixed["null_lo/20"] = {"clusters": 20, "meets_bound": False}
    assert framing_certify.minimum_clusters(mixed) == 50


def test_the_closed_form_power_grows_with_the_number_of_clusters():
    assert framing_certify.closed_form_power(0.72, 0.14, 100) == pytest.approx(
        0.9544588413906322
    )
    assert framing_certify.closed_form_power(0.72, 0.14, 25) == pytest.approx(
        0.508947981156312
    )
    assert framing_certify.closed_form_power(0.6, 0.2, 100) == pytest.approx(
        0.04390049944966737
    )


def test_power_rows_report_counts_floor_and_attainability(monkeypatch):
    seeds = []

    def fake(seed, clusters, se_a, fp_a):
        seeds.append((seed, clusters, se_a, fp_a))
        return framing_certify.ALPHA

    monkeypatch.setattr(framing_certify, "claim_p", fake)
    monkeypatch.setattr(framing_certify, "closed_form_power", lambda *a: 0.9)
    row = framing_certify.power_rows(5, 4)
    assert seeds == [(5 + s, 100, 0.72, 0.14) for s in range(4)]
    assert row == {
        "clusters": 100,
        "closed_form_power": 0.9,
        "attainable": True,
        "empirical_power": 1.0,
        "floor": framing_certify.power_floor(4),
        "meets_floor": True,
        "runs": 4,
    }


def test_power_rows_with_few_rejections_miss_the_floor(monkeypatch):
    monkeypatch.setattr(framing_certify, "claim_p", lambda *a: 0.5)
    monkeypatch.setattr(framing_certify, "closed_form_power", lambda *a: 0.89)
    row = framing_certify.power_rows(0, 10)
    assert row["attainable"] is False
    assert row["empirical_power"] == 0.0
    assert row["meets_floor"] is False
    assert row["runs"] == 10


def test_label_worlds_plant_selection_only_on_b_zero():
    expected = {
        (True, 11): (372, 306, 66),
        (True, 12): (304, 265, 39),
        (True, 13): (325, 312, 13),
        (False, 11): (962, 271, 701),
        (False, 12): (965, 298, 674),
        (False, 13): (1000, 327, 673),
    }
    for (planted, seed), counts in expected.items():
        observed, b = framing_certify.label_world(planted, seed)
        assert observed.dtype == bool
        assert set(numpy.unique(b)) <= {0.0, 1.0}
        got = (int(observed.sum()), int(b.sum()), int(observed[b == 0].sum()))
        assert got == counts
        assert observed.size == framing_certify.LABEL_ROWS
        if planted:
            assert observed[b == 1].all()


def test_clean_label_worlds_lose_at_most_fifteen_percent_at_random():
    shares = [
        framing_certify.label_world(False, seed)[0].mean() for seed in range(60)
    ]
    assert min(shares) >= 0.8
    assert max(shares) == 1.0
    assert sum(share == 1.0 for share in shares) > 15
    assert sum(share < 1.0 for share in shares) > 15


def test_planted_label_worlds_keep_a_small_share_of_b_zero_rows():
    for seed in range(30):
        observed, b = framing_certify.label_world(True, seed)
        assert observed[b == 0].mean() <= 0.35


def test_the_label_certificate_counts_detections_and_false_alarms(
    monkeypatch,
):
    monkeypatch.setattr(framing_certify, "LABEL_RUNS", 4)
    cert, labels = framing_certify.label_certificate(9)
    assert labels == {
        "planted": 4,
        "detected": 4,
        "clean": 4,
        "false_alarms": 0,
        "tolerance": baseline.LABEL_TOLERANCE,
        "alpha": baseline.LABEL_ALPHA,
        "seed_offset": 9,
    }
    assert cert.detector_id == baseline.DETECTOR_ID
    assert cert.code_hash == framing_certify.code_digest()
    assert cert.policy == framing_certify.LABEL_POLICY
    assert cert.seeds == (9,)
    result = cert.outcomes[0].certificate
    assert result.detector_id == baseline.DETECTOR_ID
    assert result.defect_type == baseline.DEFECT_TYPE
    assert (result.planted, result.detected) == (4, 4)
    assert (result.clean_rows, result.false_alarms) == (4, 0)
    assert result.seed_path == randomness.seed_path(
        9, "framing.certify.labels", 0
    )
    assert result.seed_path == (9, "framing.certify.labels", 0)


def test_the_label_certificate_hashes_the_world_description(monkeypatch):
    monkeypatch.setattr(framing_certify, "LABEL_RUNS", 2)
    cert = framing_certify.label_certificate(0)[0]
    description = json.dumps(
        {
            "rows": 1000,
            "tolerance": baseline.LABEL_TOLERANCE,
            "alpha": baseline.LABEL_ALPHA,
            "min_arm_rows": baseline.MIN_ARM_ROWS,
            "keep": [0.0, 0.3],
            "clean_rate": [0.0, 0.15],
        },
        sort_keys=True,
    )
    assert cert.corpus_hash == certificates.corpus_hash(
        [certificates.hash_source(description)]
    )
    assert description.startswith('{"alpha": ')


def test_the_feasibility_verdicts_match_the_oracle_away_from_the_boundary():
    verdicts = {
        0.6: ["downgrade"] * 6,
        1.0: [
            "downgrade",
            "inconclusive",
            "downgrade",
            "feasible",
            "inconclusive",
            "feasible",
        ],
        1.6: ["feasible"] * 6,
    }
    for ratio, expected in verdicts.items():
        got = [framing_certify.feasibility_run(s, ratio) for s in range(6)]
        assert [g[0] for g in got] == expected
        oracle = "feasible" if ratio >= 1.0 else "downgrade"
        assert [g[1] for g in got] == [oracle] * 6


def test_feasibility_rows_count_agreement_per_ratio():
    rows = framing_certify.feasibility_rows(0, 3)
    assert rows == {
        "ratio_0.6": {"agree": 3, "runs": 3},
        "ratio_1.6": {"agree": 3, "runs": 3},
        "agreement": 1.0,
        "meets_floor": True,
        "boundary_inconclusive_share": pytest.approx(1 / 3),
    }


def test_omission_worlds_have_the_stated_rates():
    flips, pilot = framing_certify.omission_frame(3, True)
    assert list(flips.columns) == [
        "customer_id",
        "as_of",
        "churn",
        "current_rule",
    ]
    assert len(flips) == 600
    assert pilot.shape == (600,)
    assert flips.customer_id.tolist() == list(range(600))
    assert str(flips.as_of.iloc[0]) == "2024-01-01 00:00:00"
    assert str(flips.as_of.iloc[1]) == "2024-01-01 01:00:00"
    assert str(flips.as_of.iloc[-1]) == "2024-01-25 23:00:00"
    for column in (flips.churn, flips.current_rule):
        assert column.dtype == numpy.int64
    assert pilot.dtype == numpy.int64
    churn = flips.churn == 1
    assert (
        int(flips.churn.sum()),
        int(flips.current_rule.sum()),
        int(pilot.sum()),
    ) == (181, 166, 342)
    assert (
        int(pilot[churn].sum()),
        int(pilot[~churn].sum()),
        int(flips.current_rule[churn].sum()),
        int(flips.current_rule[~churn].sum()),
    ) == (151, 191, 99, 67)
    robust, pilot = framing_certify.omission_frame(3, False)
    churn = robust.churn == 1
    assert (
        int(robust.churn.sum()),
        int(robust.current_rule.sum()),
        int(pilot.sum()),
    ) == (164, 203, 166)
    assert (
        int(pilot[churn].sum()),
        int(pilot[~churn].sum()),
        int(robust.current_rule[churn].sum()),
        int(robust.current_rule[~churn].sum()),
    ) == (148, 18, 77, 126)
    assert pilot.dtype == numpy.int64
    assert robust.current_rule.dtype == numpy.int64
    assert robust.churn.dtype == numpy.int64


class Stub:
    questions = ()


def test_the_omission_request_is_the_stated_one(monkeypatch):
    seen = {}

    def fake(request, data, **kwargs):
        seen["request"] = request
        seen["data"] = data
        seen["kwargs"] = kwargs
        return Stub()

    monkeypatch.setattr(draft, "frame", fake)
    assert framing_certify.omission_asks(4, True, False) is False
    request = seen["request"]
    assert request.text == "Predict churn for each customer."
    assert request.source == draft.SourceDescription(
        "warehouse://crm/accounts",
        ("customer_id", "as_of", "churn", "current_rule"),
        primary_key=("customer_id",),
    )
    assert list(seen["data"].columns) == list(request.source.columns)
    kwargs = seen["kwargs"]
    assert kwargs["cost_model"] is None
    assert kwargs["baseline_column"] == "current_rule"
    frame, pilot = framing_certify.omission_frame(4, True)
    assert kwargs["pilot_actions"] == pilot.tolist()
    assert set(kwargs) == {"cost_model", "baseline_column", "pilot_actions"}
    assert seen["data"].equals(frame)


def test_a_given_cost_model_is_the_one_in_the_design(monkeypatch):
    seen = {}

    def fake(request, data, **kwargs):
        seen.update(kwargs)
        return Stub()

    monkeypatch.setattr(draft, "frame", fake)
    framing_certify.omission_asks(4, False, True)
    assert seen["cost_model"] == costs.DecisionCostModel(
        0.0,
        5.0,
        1.0,
        0.0,
        0.05,
        0.5,
        (costs.CostSource.HUMAN,) * 4,
        costs.CostSource.HUMAN,
    )


def test_the_cost_question_is_found_by_its_slot(monkeypatch):
    class Question:
        def __init__(self, slot):
            self.slot = slot

    class Result:
        def __init__(self, slots):
            self.questions = [Question(s) for s in slots]

    answers = iter([Result(["x", "cost_threshold"]), Result(["x"])])
    monkeypatch.setattr(draft, "frame", lambda *a, **k: next(answers))
    assert framing_certify.omission_asks(0, True, False) is True
    assert framing_certify.omission_asks(0, True, False) is False


def test_the_drafter_asks_only_when_the_verdict_flips_and_no_costs_are_given():
    for seed in range(3):
        assert framing_certify.omission_asks(seed, True, False) is True
        assert framing_certify.omission_asks(seed, False, False) is False
        assert framing_certify.omission_asks(seed, True, True) is False


def test_omission_rows_count_the_three_cases(monkeypatch):
    calls = []

    def fake(seed, flips, given):
        calls.append((seed, flips, given))
        return flips and not given

    monkeypatch.setattr(framing_certify, "omission_asks", fake)
    rows = framing_certify.omission_rows(10, 3)
    assert calls == (
        [(10 + s, True, False) for s in range(3)]
        + [(10 + s, False, False) for s in range(3)]
        + [(10 + s, True, True) for s in range(3)]
    )
    low = certificates.wilson_interval(3, 3)[0]
    high = certificates.wilson_interval(0, 6)[1]
    assert rows == {
        "runs_each": 3,
        "asked_when_critical": 3,
        "asked_when_robust": 0,
        "asked_when_given": 0,
        "recall_wilson_lower": low,
        "false_question_wilson_upper": high,
        "meets_floor": low >= 0.8 and high <= 0.1,
    }


def test_omission_floors_need_both_recall_and_few_false_questions(
    monkeypatch,
):
    monkeypatch.setattr(
        framing_certify, "omission_asks", lambda seed, flips, given: flips
    )
    rows = framing_certify.omission_rows(0, 200)
    assert rows["asked_when_critical"] == 200
    assert rows["asked_when_robust"] == 0
    assert rows["asked_when_given"] == 200
    assert rows["meets_floor"] is False
    monkeypatch.setattr(
        framing_certify,
        "omission_asks",
        lambda seed, flips, given: flips and not given,
    )
    assert framing_certify.omission_rows(0, 200)["meets_floor"] is True
    monkeypatch.setattr(
        framing_certify, "omission_asks", lambda seed, flips, given: False
    )
    assert framing_certify.omission_rows(0, 200)["meets_floor"] is False


def expected_base(number):
    base = contracts.template()
    base["question"] = "q"
    base["estimand"].update(
        population="p",
        target="y",
        prediction_moment="m",
        summary_measure="s",
        intercurrent_events=[{"event": "none", "strategy": "n/a"}],
    )
    base["unit"] = "u"
    base["deployment_population"] = "d"
    base["seed"] = number
    return base


def test_the_fuzz_tries_the_count_of_records_and_accepts_none(monkeypatch):
    tried = []
    real = contracts.AnalysisCharter.from_dict

    def spy(record):
        tried.append(json.loads(json.dumps(record)))
        return real(record)

    monkeypatch.setattr(contracts.AnalysisCharter, "from_dict", spy)
    assert framing_certify.fuzz_rows(1000) == {
        "malformed": 1000,
        "accepted": 0,
    }
    assert len(tried) == 1000
    assert [r.get("seed") for r in tried if "seed" in r] != []
    extras = classify(tried)
    assert {k: len(v) for k, v in extras.items()} == {
        "extra": 157,
        "deleted": 152,
        "blank": 183,
        "power": 194,
        "kind": 160,
        "kmax": 154,
    }
    for number, record in extras["extra"]:
        names = [k for k in record if k.startswith("extra")]
        assert len(names) == 1
        assert 0 <= int(names[0].removeprefix("extra")) < 1000
        assert record[names[0]] == 1
        del record[names[0]]
        assert record == expected_base(number)
    assert {
        frozenset(expected_base(n)) - frozenset(r) for n, r in extras["deleted"]
    } == {frozenset({k}) for k in expected_base(0)}
    blanked = set()
    for number, record in extras["blank"]:
        blanks = [k for k in ("question", "unit") if record[k] == ""]
        assert len(blanks) == 1
        blanked.add(blanks[0])
        record[blanks[0]] = expected_base(number)[blanks[0]]
        assert record == expected_base(number)
    assert blanked == {"question", "unit"}
    powers = set()
    for number, record in extras["power"]:
        powers.add(record["power"])
        record["power"] = expected_base(number)["power"]
        assert record == expected_base(number)
    assert powers == {0.0, 1.0, -1.0, 2.0}
    for number, record in extras["kind"]:
        assert record["estimand"]["kind"] == "other"
        record["estimand"]["kind"] = expected_base(number)["estimand"]["kind"]
        assert record == expected_base(number)
    kmax = set()
    for number, record in extras["kmax"]:
        kmax.add(record["k_max"])
        assert isinstance(record["k_max"], float)
        record["k_max"] = expected_base(number)["k_max"]
        assert record == expected_base(number)
    assert kmax == {0.0, -3.0}


def classify(records):
    base = contracts.template()
    found = {
        key: []
        for key in ("extra", "deleted", "blank", "power", "kind", "kmax")
    }
    for number, record in enumerate(records):
        entry = (number, record)
        if any(k.startswith("extra") for k in record):
            found["extra"].append(entry)
        elif "estimand" in record and record["estimand"].get("kind") == "other":
            found["kind"].append(entry)
        elif record.get("power", base["power"]) != base["power"]:
            found["power"].append(entry)
        elif record.get("question") == "" or record.get("unit") == "":
            found["blank"].append(entry)
        elif record.get("k_max") in (0.0, -3.0):
            found["kmax"].append(entry)
        else:
            found["deleted"].append(entry)
    return found


def test_the_fuzz_counts_what_it_is_asked_for():
    assert framing_certify.fuzz_rows(0) == {"malformed": 0, "accepted": 0}
    assert framing_certify.fuzz_rows(7) == {"malformed": 7, "accepted": 0}


def test_a_charter_that_is_accepted_is_counted(monkeypatch):
    monkeypatch.setattr(
        contracts.AnalysisCharter, "from_dict", lambda record: record
    )
    assert framing_certify.fuzz_rows(25) == {"malformed": 25, "accepted": 25}


def test_certify_all_passes_the_offset_and_the_run_counts_through(
    monkeypatch,
):
    monkeypatch.setattr(framing_certify, "LABEL_RUNS", 2)
    seen = {}

    def record(name, value):
        def call(*args):
            seen[name] = args
            return value

        return call

    monkeypatch.setattr(
        framing_certify, "size_rows", record("size", {"k": {"clusters": 20}})
    )
    monkeypatch.setattr(
        framing_certify, "minimum_clusters", record("minimum", 50)
    )
    monkeypatch.setattr(
        framing_certify, "power_rows", record("power", {"p": 1})
    )
    monkeypatch.setattr(
        framing_certify, "feasibility_rows", record("feasibility", {"f": 1})
    )
    monkeypatch.setattr(
        framing_certify, "omission_rows", record("omission", {"o": 1})
    )
    monkeypatch.setattr(
        framing_certify, "fuzz_rows", record("fuzz", {"malformed": 1})
    )
    store, document = framing_certify.certify_all()
    assert seen["size"] == (0, framing_certify.RUNS)
    assert seen["power"] == (0, framing_certify.RUNS)
    assert seen["feasibility"] == (0, framing_certify.FEASIBILITY_RUNS)
    assert seen["omission"] == (0, framing_certify.OMISSION_RUNS)
    assert seen["minimum"] == ({"k": {"clusters": 20}},)
    assert seen["fuzz"] == ()
    assert len(store.certificates) == 1
    assert store.certificates[0].detector_id == baseline.DETECTOR_ID
    assert set(document) == {
        "size",
        "size_bound",
        "min_clusters_measured",
        "min_clusters_in_code",
        "power",
        "selective_labels",
        "feasibility",
        "planted_omission",
        "schema_fuzz",
        "development_seed_base",
        "seconds",
    }
    assert document["size_bound"] == framing_certify.size_bound()
    assert document["min_clusters_measured"] == 50
    assert document["min_clusters_in_code"] == baseline.MIN_CLUSTERS
    assert document["development_seed_base"] == 10_000
    assert document["selective_labels"]["seed_offset"] == 0
    assert document["selective_labels"]["planted"] == 2
    assert isinstance(document["seconds"], float)
    assert document["seconds"] >= 0
    store, document = framing_certify.certify_all(7, 11)
    assert seen["size"] == (7, 11)
    assert document["size_bound"] == framing_certify.size_bound(11)
    assert document["selective_labels"]["seed_offset"] == 7


def good_document(**changes):
    document = {
        "min_clusters_measured": baseline.MIN_CLUSTERS,
        "schema_fuzz": {"malformed": 1000, "accepted": 0},
        "power": {"attainable": True, "meets_floor": True},
        "feasibility": {"meets_floor": True},
        "planted_omission": {"meets_floor": True},
    }
    document.update(changes)
    return document


@pytest.fixture
def stored(monkeypatch):
    monkeypatch.setattr(framing_certify, "LABEL_RUNS", 4)
    store = certificates.Store()
    store.add(framing_certify.label_certificate(0)[0])
    return store


def test_an_empty_store_is_missing_its_certificate():
    empty = certificates.Store()
    assert framing_certify.problems(empty, good_document()) == [
        f"{baseline.DETECTOR_ID}: missing"
    ]
    assert framing_certify.problems(empty, None) == [
        f"{baseline.DETECTOR_ID}: missing",
        "framing_measurements.json missing",
    ]


def test_a_certificate_for_other_code_is_stale(stored, monkeypatch):
    monkeypatch.setattr(framing_certify, "code_digest", lambda: "0" * 64)
    assert framing_certify.problems(stored, good_document()) == [
        f"{baseline.DETECTOR_ID}: stale"
    ]


def test_a_current_store_with_a_consistent_document_has_no_problems(stored):
    assert framing_certify.problems(stored, good_document()) == []
    assert framing_certify.problems(stored, None) == [
        "framing_measurements.json missing"
    ]


def test_the_problems_name_a_min_clusters_mismatch_and_an_accepted_record(
    stored,
):
    document = good_document(min_clusters_measured=50)
    assert framing_certify.problems(stored, document) == [
        "baseline.MIN_CLUSTERS differs from the measured minimum (50)"
    ]
    document = good_document(min_clusters_measured=None)
    assert framing_certify.problems(stored, document) == [
        "baseline.MIN_CLUSTERS differs from the measured minimum (None)"
    ]
    document = good_document(schema_fuzz={"malformed": 5, "accepted": 2})
    assert framing_certify.problems(stored, document) == [
        "a malformed charter record was accepted"
    ]
    document = good_document(
        min_clusters_measured=50, schema_fuzz={"malformed": 5, "accepted": 1}
    )
    assert len(framing_certify.problems(stored, document)) == 2


def test_a_method_below_its_floor_stays_descriptive():
    empty = certificates.Store()
    assert framing_certify.descriptive_only(empty, good_document()) == []
    assert framing_certify.descriptive_only(empty, None) == []
    assert framing_certify.descriptive_only(
        empty, good_document(min_clusters_measured=None)
    ) == ["baseline claim (size bound never met)"]
    assert framing_certify.descriptive_only(
        empty, good_document(power={"attainable": True, "meets_floor": False})
    ) == ["baseline claim power"]
    assert (
        framing_certify.descriptive_only(
            empty,
            good_document(power={"attainable": False, "meets_floor": False}),
        )
        == []
    )
    assert (
        framing_certify.descriptive_only(
            empty,
            good_document(power={"attainable": False, "meets_floor": True}),
        )
        == []
    )
    assert framing_certify.descriptive_only(
        empty, good_document(feasibility={"meets_floor": False})
    ) == ["framing feasibility verdict"]
    assert framing_certify.descriptive_only(
        empty, good_document(planted_omission={"meets_floor": False})
    ) == ["framing questions"]
    everything = good_document(
        min_clusters_measured=None,
        power={"attainable": True, "meets_floor": False},
        feasibility={"meets_floor": False},
        planted_omission={"meets_floor": False},
    )
    assert framing_certify.descriptive_only(empty, everything) == [
        "baseline claim (size bound never met)",
        "baseline claim power",
        "framing feasibility verdict",
        "framing questions",
    ]


def test_a_certificate_that_misses_its_policy_is_listed_first(stored):
    assert framing_certify.descriptive_only(
        stored, good_document(feasibility={"meets_floor": False})
    ) == [
        f"{baseline.DETECTOR_ID}/{baseline.DEFECT_TYPE}",
        "framing feasibility verdict",
    ]


def test_the_measurements_are_read_as_utf8_json(tmp_path, monkeypatch):
    path = tmp_path / "framing_measurements.json"
    monkeypatch.setattr(framing_certify, "MEASUREMENTS_PATH", path)
    assert framing_certify.load_document() is None
    path.write_text('{"note": "caf\\u00e9", "n": 3}', encoding="utf-8")
    seen = {}
    original = Path.read_text

    def spy(self, *args, **kwargs):
        seen.update(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy)
    assert framing_certify.load_document() == {"note": "café", "n": 3}
    assert seen == {"encoding": "utf-8"}


def test_check_reports_problems_descriptive_methods_and_the_missing_corpus(
    stored, tmp_path, monkeypatch
):
    store_path = tmp_path / "framing_certificates.json"
    document_path = tmp_path / "framing_measurements.json"
    stored.save(store_path)
    monkeypatch.setattr(framing_certify, "STORE_PATH", store_path)
    monkeypatch.setattr(framing_certify, "MEASUREMENTS_PATH", document_path)
    document = good_document(feasibility={"meets_floor": False})
    document_path.write_text(json.dumps(document), encoding="utf-8")
    assert framing_certify.check() == {
        "ok": True,
        "problems": [],
        "descriptive_only": [
            f"{baseline.DETECTOR_ID}/{baseline.DEFECT_TYPE}",
            "framing feasibility verdict",
        ],
        "gold_corpus": "absent: no LLM-assisted field may gate",
    }
    document_path.unlink()
    report = framing_certify.check()
    assert report["ok"] is False
    assert report["problems"] == ["framing_measurements.json missing"]
    assert report["descriptive_only"] == [
        f"{baseline.DETECTOR_ID}/{baseline.DEFECT_TYPE}"
    ]


def test_the_boundary_share_of_agreeing_runs_meets_the_feasibility_floor(
    monkeypatch,
):
    calls = []

    def fake(seed, ratio):
        calls.append((seed, ratio))
        agreeing = ratio == 0.6 or seed != 3
        return ("feasible", "feasible" if agreeing else "downgrade")

    monkeypatch.setattr(framing_certify, "feasibility_run", fake)
    rows = framing_certify.feasibility_rows(3, 10)
    assert rows["ratio_0.6"] == {"agree": 10, "runs": 10}
    assert rows["ratio_1.6"] == {"agree": 9, "runs": 10}
    assert rows["agreement"] == 0.95
    assert rows["meets_floor"] is True
    assert calls[:10] == [(3 + s, 0.6) for s in range(10)]
    assert calls[10:20] == [(3 + s, 1.6) for s in range(10)]
    assert calls[20:] == [(3 + s, 1.0) for s in range(10)]


def test_half_agreement_misses_the_feasibility_floor(monkeypatch):
    monkeypatch.setattr(
        framing_certify,
        "feasibility_run",
        lambda seed, ratio: ("feasible", "feasible" if seed % 2 else "x"),
    )
    rows = framing_certify.feasibility_rows(0, 10)
    assert rows["agreement"] == 0.5
    assert rows["meets_floor"] is False


def test_a_mismatching_verdict_is_not_an_agreement(monkeypatch):
    monkeypatch.setattr(
        framing_certify,
        "feasibility_run",
        lambda seed, ratio: ("inconclusive", "feasible"),
    )
    rows = framing_certify.feasibility_rows(0, 4)
    assert rows["ratio_0.6"] == {"agree": 0, "runs": 4}
    assert rows["agreement"] == 0.0
    assert rows["meets_floor"] is False
    assert rows["boundary_inconclusive_share"] == 1.0


def test_the_feasibility_run_sizes_the_confirmation_from_the_true_requirement(
    monkeypatch,
):
    seen = []
    real = feasibility.assess

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(feasibility, "assess", spy)
    for ratio in (0.6, 1.0, 1.6):
        framing_certify.feasibility_run(0, ratio)
    assert [call["n_confirmation"] for call in seen] == [404, 673, 1077]
    for call in seen:
        assert call["delta_min"] == 0.04
        assert call["alpha"] == 0.05
        assert call["k_max"] == 1
        assert call["power"] == 0.8


def test_a_rejection_among_many_runs_still_meets_the_size_bound(monkeypatch):
    monkeypatch.setattr(
        framing_certify,
        "claim_p",
        lambda seed, *rest: 0.01 if seed == 0 else 0.5,
    )
    rows = framing_certify.size_rows(0, 100)
    for row in rows.values():
        assert row["rejections"] == 1
        assert row["rate"] == 0.01
        assert row["meets_bound"] is True
    monkeypatch.setattr(
        framing_certify,
        "claim_p",
        lambda seed, *rest: 0.01 if seed < 20 else 0.5,
    )
    assert not any(
        row["meets_bound"] for row in framing_certify.size_rows(0, 100).values()
    )


def test_some_rejections_below_the_floor_do_not_meet_it(monkeypatch):
    monkeypatch.setattr(
        framing_certify,
        "claim_p",
        lambda seed, *rest: 0.01 if seed == 0 else 0.5,
    )
    monkeypatch.setattr(framing_certify, "closed_form_power", lambda *a: 0.9)
    row = framing_certify.power_rows(0, 10)
    assert row["empirical_power"] == 0.1
    assert row["meets_floor"] is False


def test_the_fuzz_default_is_a_thousand_records_with_known_extra_fields(
    monkeypatch,
):
    names = []
    real = contracts.AnalysisCharter.from_dict

    def spy(record):
        names.extend(k for k in record if k.startswith("extra"))
        return real(record)

    monkeypatch.setattr(contracts.AnalysisCharter, "from_dict", spy)
    assert framing_certify.fuzz_rows() == {"malformed": 1000, "accepted": 0}
    assert names[:6] == [
        "extra440",
        "extra799",
        "extra220",
        "extra82",
        "extra42",
        "extra932",
    ]
    assert sum(int(n.removeprefix("extra")) for n in names) == 79549
    assert max(int(n.removeprefix("extra")) for n in names) == 993


def test_the_elapsed_seconds_are_the_difference_rounded_to_a_tenth(
    monkeypatch,
):
    ticks = iter([100.0, 102.34])
    monkeypatch.setattr(framing_certify.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(framing_certify, "LABEL_RUNS", 2)
    for name in (
        "size_rows",
        "power_rows",
        "feasibility_rows",
        "omission_rows",
        "fuzz_rows",
    ):
        monkeypatch.setattr(framing_certify, name, lambda *a, **k: {})
    monkeypatch.setattr(framing_certify, "minimum_clusters", lambda rows: 20)
    assert framing_certify.certify_all()[1]["seconds"] == 2.3
