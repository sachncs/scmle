"""Recorded runs and file handling of the coverage certification harness.

The numbers below were recorded from the harness on development seeds with
small replicate counts; they pin the worlds, the seeds and the pass rules.
"""

import json
import types

import numpy
import pytest

from benchmarks.qualification import uncertainty_certify
from scmle.data.detectors import certificates
from scmle.evaluation import coverage_slices

GOLD = {
    "g1": {
        "design": {"n_cal": 500, "alpha": 0.1, "beta": [451, 50]},
        "mean": 0.8979419869927266,
        "band": [0.8936543670485804, 0.9083416409354517],
        "ks_p": 0.44023393300038316,
        "passed": True,
    },
    "g3": {
        "replicates": 2,
        "cqr_marginal": {
            "mean": 0.889625,
            "se": 0.0018749999999999598,
            "floor": 0.8943750000000001,
            "meets_floor": False,
        },
        "cqr_top_decile": 0.8512500000000001,
        "constant_width_top_decile": 0.71375,
        "constant_width_marginal": 0.891125,
        "passed": False,
    },
    "g4": {
        "replicates": 3,
        "class_conditional": [
            {
                "mean": 0.8908218457270607,
                "se": 0.00194544267206555,
                "floor": 0.8941636719838034,
                "meets_floor": False,
            },
            {
                "mean": 0.8913254284142442,
                "se": 0.0051677932142533715,
                "floor": 0.8844966203572399,
                "meets_floor": True,
            },
            {
                "mean": 0.9037377519473316,
                "se": 0.003973051365521865,
                "floor": 0.8880808459034344,
                "meets_floor": True,
            },
            {
                "mean": 0.9183986197365313,
                "se": 0.020017133292953987,
                "floor": 0.839948600121138,
                "meets_floor": True,
            },
            {
                "mean": 0.8910218809434935,
                "se": 0.0034376777091238886,
                "floor": 0.8896869668726284,
                "meets_floor": True,
            },
        ],
        "pooled_control": [
            {
                "mean": 0.9778708059631706,
                "se": 0.0032422604096598103,
                "floor": 0.8902732187710206,
                "meets_floor": True,
            },
            {
                "mean": 0.9365953472450183,
                "se": 0.005067482309528614,
                "floor": 0.8847975530714142,
                "meets_floor": True,
            },
            {
                "mean": 0.8764740598905664,
                "se": 0.008157422750074032,
                "floor": 0.8755277317497779,
                "meets_floor": True,
            },
            {
                "mean": 0.5824756517984081,
                "se": 0.02679144708990979,
                "floor": 0.8196256587302706,
                "meets_floor": False,
            },
            {"mean": 0.0, "se": 0.0, "floor": 0.9, "meets_floor": False},
        ],
        "passed": False,
    },
    "g5": {
        "replicates": 40,
        "familywise_false_flag": 0.175,
        "false_flag_bound": 0.11892024376045111,
        "recall_by_slice_rows": {
            "30": {"recall": 1.0, "wilson_lower": 0.9123783988027135},
            "50": {"recall": 1.0, "wilson_lower": 0.9123783988027135},
            "100": {"recall": 1.0, "wilson_lower": 0.9123783988027135},
            "200": {"recall": 1.0, "wilson_lower": 0.9123783988027135},
            "400": {"recall": 1.0, "wilson_lower": 0.9123783988027135},
        },
        "min_certified_rows": 30,
        "passed": False,
    },
    "g5b": {
        "replicates": 40,
        "calibration_rows": 500,
        "margin": 0.05,
        "familywise_false_flag": 0.0,
        "false_flag_bound": 0.11892024376045111,
        "detection_by_slice_rows": {
            "noise_3x": {
                "30": {"recall": 1.0, "wilson_lower": 0.9123783988027135},
                "50": {"recall": 1.0, "wilson_lower": 0.9123783988027135},
                "100": {"recall": 1.0, "wilson_lower": 0.9123783988027135},
                "200": {"recall": 1.0, "wilson_lower": 0.9123783988027135},
                "400": {"recall": 1.0, "wilson_lower": 0.9123783988027135},
            },
            "noise_1.5x": {
                "30": {"recall": 0.075, "wilson_lower": 0.025836025774588198},
                "50": {"recall": 0.3, "wilson_lower": 0.18074845229746528},
                "100": {"recall": 0.65, "wilson_lower": 0.49505880837257693},
                "200": {"recall": 0.875, "wilson_lower": 0.738878801611489},
                "400": {"recall": 1.0, "wilson_lower": 0.9123783988027135},
            },
            "noise_1.25x": {
                "30": {"recall": 0.0, "wilson_lower": 0.0},
                "50": {"recall": 0.1, "wilson_lower": 0.03957952868260638},
                "100": {"recall": 0.125, "wilson_lower": 0.05459500250945401},
                "200": {"recall": 0.25, "wilson_lower": 0.14187118639096302},
                "400": {"recall": 0.475, "wilson_lower": 0.329354701126789},
            },
        },
        "min_certified_rows": 30,
        "passed": True,
    },
    "g6": {
        "known_ratio": {
            "replicates": 4,
            "weighted": {
                "mean": 0.9093333333333333,
                "se": 0.016264594585695548,
                "floor": 0.8512062162429134,
                "meets_floor": True,
            },
            "unweighted": 0.7283333333333334,
        },
        "estimated_ratio": {
            "replicates": 3,
            "weighted": {
                "mean": 0.9097777777777778,
                "se": 0.015461221902425722,
                "floor": 0.8536163342927229,
                "meets_floor": True,
            },
            "unweighted": 0.7290000000000001,
            "effective_size_mean": 266.7672738516459,
        },
        "passed": True,
        "estimated_passed": True,
    },
    "g9": {
        "replicates": 4,
        "mean_difference": -4.199040600023751,
        "upper_95": -2.767165145224295,
        "coherence_residual": 3.552713678800501e-15,
        "passed": True,
    },
    "g11i": {
        "replicates": 3,
        "tau": 1.250895466477202,
        "true_concordance": 0.7278883739673585,
        "covariate_censoring_bias": -0.0008949241665746621,
        "marginal_km_bias": -0.015670983837399954,
        "se": [0.0009845733776263968, 0.0006919627944553104],
        "mean_floored_share": 0.005555555555555556,
        "passed": True,
    },
    "g13": {
        "replicates": 5,
        "entities": 30,
        "rows_per_entity": 40,
        "entity_wise": {
            "mean": 0.98065,
            "se": 0.009066973034039543,
            "floor": 0.8727990808978814,
            "meets_floor": True,
        },
        "row_wise": {
            "mean": 0.9595499999999999,
            "se": 0.009284530144277636,
            "floor": 0.8721464095671672,
            "meets_floor": True,
        },
        "passed": True,
    },
    "cv": {
        "knn1_10": [0.939, 0.901],
        "knn1_5": [0.9605, 0.9215],
        "linear_10": [0.901, 0.904],
        "linear_5": [0.9025, 0.8965],
    },
    "class_world": {
        "labels": [2, 2, 1, 0, 0, 0],
        "prob": [
            [0.542081, 0.06309, 0.350911, 0.021782, 0.022136],
            [0.058023, 0.03219, 0.685012, 0.202682, 0.022092],
            [0.217249, 0.750459, 0.007448, 0.013721, 0.011123],
            [0.459075, 0.032839, 0.471581, 0.017991, 0.018514],
            [0.75708, 0.218164, 0.014264, 0.002192, 0.008299],
            [0.794502, 0.162325, 0.029627, 0.003228, 0.010317],
        ],
    },
    "survival_c": {
        "x": [
            [-0.801931, -1.324359],
            [-0.248362, 0.420445],
            [1.136047, 0.109706],
            [-0.552647, -0.78478],
        ],
        "risk": [0.15307, -0.450956, 0.843013, 0.02875],
        "event": [0.135025, 1.351361, 0.603757, 1.086714],
        "time": [0.135025, 1.351361, 0.603757, 1.086714],
        "ind": [1.0, 1.0, 1.0, 1.0],
    },
    "survival_n": {
        "risk": [0.15307, -0.450956, 0.843013, 0.02875],
        "event": [0.135025, 1.351361, 0.603757, 1.086714],
        "time": [0.135025, 1.351361, 0.603757, 1.086714],
        "ind": [1.0, 1.0, 1.0, 1.0],
    },
}


def close(actual, expected, where="result"):
    """Recursive comparison with a float tolerance."""
    if isinstance(expected, dict):
        assert set(actual) == set(expected), where
        for key, value in expected.items():
            close(actual[key], value, f"{where}.{key}")
    elif isinstance(expected, list | tuple):
        assert len(actual) == len(expected), where
        for index, value in enumerate(expected):
            close(actual[index], value, f"{where}[{index}]")
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-7, abs=1e-12), where
    else:
        assert actual == expected, where


def test_the_constants_are_exact():
    assert uncertainty_certify.DEV_SEED_BASE == 10_000
    assert uncertainty_certify.ALPHA == 0.1
    assert uncertainty_certify.LEVEL == 1.0 - 0.1
    assert uncertainty_certify.REPLICATES == 2000
    assert uncertainty_certify.CQR_REPLICATES == 150
    assert uncertainty_certify.ESTIMATED_RATIO_REPLICATES == 200
    assert uncertainty_certify.SURVIVAL_REPLICATES == 100
    assert uncertainty_certify.KS_P_FLOOR == 0.001
    assert uncertainty_certify.TOP_DECILE_FLOOR == 0.87
    assert uncertainty_certify.SLICE_RECALL_FLOOR == 0.80
    assert uncertainty_certify.COHERENCE_TOLERANCE == 1e-8
    assert uncertainty_certify.GATES == (
        "G1",
        "G2",
        "G3",
        "G4",
        "G5",
        "G5b",
        "G6",
        "G9",
        "G11i",
        "G13",
    )
    assert len(uncertainty_certify.NOT_MEASURED) == 8
    assert uncertainty_certify.STORE_PATH.name == (
        "uncertainty_certificates.json"
    )


def test_the_runners_pair_each_gate_with_its_replicates():
    runners = uncertainty_certify.RUNNERS
    assert tuple(runners) == uncertainty_certify.GATES
    assert runners["G1"] == (uncertainty_certify.gate_g1, 2000)
    assert runners["G2"] == (uncertainty_certify.gate_g2, 2000)
    assert runners["G3"] == (uncertainty_certify.gate_g3, 150)
    assert runners["G4"] == (uncertainty_certify.gate_g4, 300)
    assert runners["G5"] == (uncertainty_certify.gate_g5, 2000)
    assert runners["G5b"] == (uncertainty_certify.gate_g5b, 2000)
    assert runners["G6"] == (uncertainty_certify.gate_g6, 2000)
    assert runners["G9"] == (uncertainty_certify.gate_g9, 2000)
    assert runners["G11i"] == (uncertainty_certify.gate_g11i, 100)
    assert runners["G13"] == (uncertainty_certify.gate_g13, 2000)
    assert set(uncertainty_certify.sources()) == set(uncertainty_certify.GATES)
    assert set(uncertainty_certify.current_digests()) == set(
        uncertainty_certify.GATES
    )


def test_wilson_lower_is_the_lower_end_of_the_95_percent_interval():
    expected = certificates.wilson_interval(30, 40, 0.95)[0]
    assert uncertainty_certify.wilson_lower(30, 40) == expected
    assert 0.59 < expected < 0.62


def test_mean_floor_uses_three_standard_errors():
    values = numpy.array([0.8, 0.9, 1.0, 0.9])
    se = float(values.std(ddof=1) / 2)

    result = uncertainty_certify.mean_floor(values)

    assert result == {
        "mean": pytest.approx(0.9),
        "se": pytest.approx(se),
        "floor": pytest.approx(0.9 - 3 * se),
        "meets_floor": True,
    }
    other = uncertainty_certify.mean_floor(values, 1.2)
    assert other["floor"] == pytest.approx(1.2 - 3 * se)
    assert other["meets_floor"] is False


def test_g1_matches_its_recorded_run():
    close(uncertainty_certify.gate_g1(40, 10_000), GOLD["g1"])


def test_g1_defaults_to_the_certification_seeds():
    first = uncertainty_certify.gate_g1(5)
    assert first["mean"] == uncertainty_certify.gate_g1(5, 0)["mean"]
    assert first["mean"] != uncertainty_certify.gate_g1(5, 1)["mean"]


@pytest.mark.parametrize("key", sorted(GOLD["cv"]))
def test_cv_plus_replicates_match_their_recorded_runs(key):
    learner, folds = key.rsplit("_", 1)
    pair = uncertainty_certify.cv_plus_replicate(
        10_000 + 3, int(folds), learner
    )
    close(list(pair), GOLD["cv"][key])


def test_g2_runs_every_learner_and_fold_cell_over_fifty_worlds(monkeypatch):
    calls = []

    def fake(seed, folds, learner):
        calls.append((seed, folds, learner))
        return (0.95 if learner == "knn1" else 0.85, 0.5)

    monkeypatch.setattr(uncertainty_certify, "cv_plus_replicate", fake)

    result = uncertainty_certify.gate_g2(10, 7)

    assert len(calls) == 200
    assert calls[0] == (7, 10, "knn1")
    assert calls[49] == (56, 10, "knn1")
    assert calls[50] == (7, 5, "knn1")
    assert calls[100] == (7, 10, "linear")
    assert calls[150] == (7, 5, "linear")
    assert result["replicates"] == 50
    assert list(result["worlds"]) == [
        "knn1_K10",
        "knn1_K5",
        "linear_K10",
        "linear_K5",
    ]
    knn = result["worlds"]["knn1_K10"]
    assert knn["cv_plus"]["mean"] == pytest.approx(0.95)
    assert knn["cv_plus"]["se"] == pytest.approx(0.0, abs=1e-12)
    assert knn["cv_plus"]["meets_floor"] is True
    assert knn["pooled_refit_control"] == pytest.approx(0.5)
    assert result["worlds"]["linear_K5"]["cv_plus"]["meets_floor"] is False
    assert result["passed"] is False


def test_g2_scales_the_worlds_with_the_replicates(monkeypatch):
    calls = []

    def fake(seed, folds, learner):
        calls.append(seed)
        return (0.95, 0.9)

    monkeypatch.setattr(uncertainty_certify, "cv_plus_replicate", fake)

    result = uncertainty_certify.gate_g2(600)

    assert result["replicates"] == 60
    assert len(calls) == 240
    assert result["passed"] is True


def test_g3_matches_its_recorded_run():
    close(uncertainty_certify.gate_g3(2, 10_000), GOLD["g3"])


def test_class_world_matches_its_recorded_draw():
    labels, prob = uncertainty_certify.class_world(
        numpy.random.default_rng(5), 6
    )
    assert labels.tolist() == GOLD["class_world"]["labels"]
    close(prob.round(6).tolist(), GOLD["class_world"]["prob"])
    assert prob.shape == (6, 5)
    assert prob.sum(axis=1) == pytest.approx(numpy.ones(6))


def test_g4_matches_its_recorded_run():
    close(uncertainty_certify.gate_g4(3, 10_000), GOLD["g4"])


def test_g5_matches_its_recorded_run():
    close(uncertainty_certify.gate_g5(40, 10_000), GOLD["g5"])


def test_g5b_matches_its_recorded_run():
    result = uncertainty_certify.gate_g5b(40, 10_000)
    expected = dict(GOLD["g5b"])
    assert result.pop("method_hash") == coverage_slices.flag_method_hash()
    close(result, expected)


def test_g6_matches_its_recorded_run(monkeypatch):
    monkeypatch.setattr(uncertainty_certify, "ESTIMATED_RATIO_REPLICATES", 3)
    close(uncertainty_certify.gate_g6(4, 10_000), GOLD["g6"])


def test_g9_matches_its_recorded_run():
    close(uncertainty_certify.gate_g9(4, 10_000), GOLD["g9"])


def test_survival_worlds_match_their_recorded_draws():
    x, risk, event, time, indicator = uncertainty_certify.survival_world(
        numpy.random.default_rng(5), 4, True
    )
    expected = GOLD["survival_c"]
    close(x.round(6).tolist(), expected["x"])
    close(risk.round(6).tolist(), expected["risk"])
    close(event.round(6).tolist(), expected["event"])
    close(time.round(6).tolist(), expected["time"])
    assert indicator.tolist() == expected["ind"]

    x, risk, event, time, indicator = uncertainty_certify.survival_world(
        numpy.random.default_rng(5), 4, False
    )
    expected = GOLD["survival_n"]
    assert x.shape == (4, 2)
    close(risk.round(6).tolist(), expected["risk"])
    close(event.round(6).tolist(), expected["event"])
    assert time.tolist() == event.tolist()
    assert indicator.tolist() == [1.0, 1.0, 1.0, 1.0]


def test_g11i_matches_its_recorded_run():
    close(uncertainty_certify.gate_g11i(3, 10_000), GOLD["g11i"])


def test_g13_matches_its_recorded_run():
    close(uncertainty_certify.gate_g13(5, 10_000), GOLD["g13"])


def fake_runners(monkeypatch):
    calls = []

    def runner(label, passed):
        def run(replicates, base):
            calls.append((label, replicates, base))
            return {"passed": passed, "label": label}

        return run

    monkeypatch.setattr(
        uncertainty_certify,
        "RUNNERS",
        {"G1": (runner("one", True), 11), "G2": (runner("two", False), 22)},
    )
    monkeypatch.setattr(
        uncertainty_certify,
        "current_digests",
        lambda: {"G1": "h1", "G2": "h2"},
    )
    return calls


def test_certify_all_files_each_gate_with_its_hash_and_seed_range(
    monkeypatch, capsys
):
    calls = fake_runners(monkeypatch)

    document = uncertainty_certify.certify_all()

    assert calls == [("one", 11, 0), ("two", 22, 0)]
    assert document == {
        "version": 1,
        "dev_seed_base": 10_000,
        "not_measured": list(uncertainty_certify.NOT_MEASURED),
        "gates": {
            "G1": {
                "passed": True,
                "label": "one",
                "code_hash": "h1",
                "seeds": [0, 11],
            },
            "G2": {
                "passed": False,
                "label": "two",
                "code_hash": "h2",
                "seeds": [0, 22],
            },
        },
    }
    assert capsys.readouterr().err == "G1 True\nG2 False\n"


def test_passed_gates_and_descriptive_only_split_the_document():
    document = {
        "gates": {
            "G1": {"passed": True},
            "G2": {"passed": False},
            "G3": {"passed": True},
        }
    }

    assert uncertainty_certify.passed_gates(document) == ["G1", "G3"]
    assert uncertainty_certify.descriptive_only(document) == ["G2"]


def test_a_missing_store_is_reported_and_makes_every_gate_descriptive(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        uncertainty_certify, "STORE_PATH", tmp_path / "absent.json"
    )

    assert uncertainty_certify.problems() == [
        "uncertainty_certificates.json is missing"
    ]
    assert uncertainty_certify.descriptive_only() == list(
        uncertainty_certify.GATES
    )


def test_the_store_is_read_from_its_path_when_no_document_is_given(
    monkeypatch, tmp_path
):
    path = tmp_path / "store.json"
    path.write_text(
        json.dumps({"gates": {"G1": {"passed": False, "code_hash": "x"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(uncertainty_certify, "STORE_PATH", path)

    assert uncertainty_certify.descriptive_only() == ["G1"]
    found = uncertainty_certify.problems()
    assert found[0] == "G1: stale (code changed since measurement)"
    assert found[1] == "G2: no certificate"
    assert len(found) == len(uncertainty_certify.GATES)


def test_problems_accepts_a_current_entry():
    digests = uncertainty_certify.current_digests()
    document = {
        "gates": {name: {"code_hash": digests[name]} for name in digests}
    }

    assert uncertainty_certify.problems(document) == []
    document["gates"]["G9"] = None
    assert uncertainty_certify.problems(document) == ["G9: no certificate"]


def test_write_files_the_document_and_reports_the_passing_gates(
    monkeypatch, tmp_path, capsys
):
    fake_runners(monkeypatch)
    path = tmp_path / "store.json"
    monkeypatch.setattr(uncertainty_certify, "STORE_PATH", path)

    status = uncertainty_certify.main(["--write"])

    assert status == 0
    text = path.read_text(encoding="utf-8")
    assert text.endswith("}\n")
    assert text == (
        json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"
    )
    assert json.loads(text)["gates"]["G1"]["code_hash"] == "h1"
    assert json.loads(capsys.readouterr().out) == {"passed": ["G1"]}


def test_check_succeeds_on_the_shipped_store_and_fails_on_a_bad_one(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(
        uncertainty_certify, "problems", lambda document=None: []
    )
    assert uncertainty_certify.main(["--check"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "ok": True,
        "problems": [],
    }

    monkeypatch.setattr(
        uncertainty_certify, "problems", lambda document=None: ["bad"]
    )
    assert uncertainty_certify.main(["--check"]) == 1
    assert json.loads(capsys.readouterr().out) == {
        "ok": False,
        "problems": ["bad"],
    }


def test_main_needs_exactly_one_mode():
    with pytest.raises(SystemExit):
        uncertainty_certify.main([])
    with pytest.raises(SystemExit):
        uncertainty_certify.main(["--write", "--check"])


def test_mean_floor_boundaries():
    exact = uncertainty_certify.mean_floor(numpy.array([1.0, 1.0]), 1.0)
    assert exact["floor"] == 1.0
    assert exact["meets_floor"] is True
    # between the 3 se and the 4 se floors
    values = numpy.array([0.8, 0.9, 1.0, 0.9])
    assert uncertainty_certify.mean_floor(values, 1.04)["meets_floor"] is False


def test_g1_passes_only_with_a_ks_p_value_above_the_floor(monkeypatch):
    def with_p(pvalue):
        monkeypatch.setattr(
            uncertainty_certify.stats,
            "kstest",
            lambda values, cdf: types.SimpleNamespace(pvalue=pvalue),
        )
        return uncertainty_certify.gate_g1(40, 10_000)["passed"]

    assert with_p(0.0005) is False
    assert with_p(0.001) is False
    assert with_p(0.0011) is True


@pytest.mark.parametrize(
    "gate", ["gate_g3", "gate_g4", "gate_g5", "gate_g5b", "gate_g9"]
)
def test_the_default_seed_offset_is_zero(gate):
    function = getattr(uncertainty_certify, gate)
    runs = 40 if gate in {"gate_g5", "gate_g5b"} else 2
    close(function(runs), function(runs, 0))


def test_the_default_seed_offset_is_zero_for_the_remaining_gates(monkeypatch):
    monkeypatch.setattr(uncertainty_certify, "ESTIMATED_RATIO_REPLICATES", 2)
    for gate in ("gate_g6", "gate_g11i", "gate_g13"):
        function = getattr(uncertainty_certify, gate)
        close(function(2), function(2, 0))
    calls = []
    monkeypatch.setattr(
        uncertainty_certify,
        "cv_plus_replicate",
        lambda seed, folds, learner: calls.append(seed) or (0.9, 0.9),
    )
    uncertainty_certify.gate_g2(10)
    assert calls[0] == 0


def scripted_mean_floor(monkeypatch, meets, floor=0.9):
    sequence = iter(meets)

    def fake(values, nominal=0.9):
        flag = next(sequence)
        return {"mean": 0.9, "se": 0.01, "floor": floor, "meets_floor": flag}

    monkeypatch.setattr(uncertainty_certify, "mean_floor", fake)


def test_g3_passes_on_the_marginal_floor_and_the_decile_window(monkeypatch):
    base = uncertainty_certify.gate_g3(2, 10_000)
    top, constant = base["cqr_top_decile"], base["constant_width_top_decile"]
    assert constant < top

    def passed(meets, floor):
        scripted_mean_floor(monkeypatch, [meets])
        monkeypatch.setattr(uncertainty_certify, "TOP_DECILE_FLOOR", floor)
        return uncertainty_certify.gate_g3(2, 10_000)["passed"]

    assert passed(True, (top + constant) / 2) is True
    assert passed(False, (top + constant) / 2) is False
    assert passed(True, top) is True
    assert passed(True, constant) is False
    assert passed(True, top + 0.01) is False


def test_g4_passes_when_the_class_sets_meet_and_the_pooled_ones_do_not(
    monkeypatch,
):
    def passed(classes, pooled):
        scripted_mean_floor(monkeypatch, [*classes, *pooled])
        return uncertainty_certify.gate_g4(1, 10_000)["passed"]

    good = [True] * 5
    assert passed(good, [True, True, True, False, False]) is True
    assert passed(good, good) is False
    assert passed([True, True, True, True, False], [False] * 5) is False


def scripted_wilson(monkeypatch, first):
    sequence = iter(first)
    monkeypatch.setattr(
        uncertainty_certify,
        "wilson_lower",
        lambda hits, rows: next(sequence, 0.9),
    )


def test_g5_certifies_from_the_smallest_size_at_the_recall_floor(monkeypatch):
    def run(flagged_worlds):
        calls = []

        def fake(covered, labels, nominal=0.9):
            calls.append(len(labels))
            if len(calls) <= 4:
                return {"flagged": [1] if len(calls) <= flagged_worlds else []}
            return {"flagged": ["0"]}

        monkeypatch.setattr(
            uncertainty_certify.coverage, "slice_coverage", fake
        )
        scripted_wilson(monkeypatch, [0.5, 0.5, 0.8, 0.9, 0.9])
        return uncertainty_certify.gate_g5(4, 10_000)

    result = run(1)
    assert result["familywise_false_flag"] == 0.25
    assert result["min_certified_rows"] == 100
    assert result["recall_by_slice_rows"]["100"] == {
        "recall": 1.0,
        "wilson_lower": 0.8,
    }
    assert result["passed"] is True
    assert run(2)["passed"] is False
    assert run(0)["familywise_false_flag"] == 0.0


def test_g5_without_a_certified_size_does_not_pass(monkeypatch):
    monkeypatch.setattr(
        uncertainty_certify.coverage,
        "slice_coverage",
        lambda covered, labels, nominal=0.9: {"flagged": []},
    )
    monkeypatch.setattr(uncertainty_certify, "wilson_lower", lambda h, r: 0.1)

    result = uncertainty_certify.gate_g5(4, 10_000)

    assert result["min_certified_rows"] is None
    assert result["passed"] is False


def test_g5b_certifies_at_three_times_noise_and_counts_false_flags(
    monkeypatch,
):
    def run(flagged_worlds):
        calls = []

        def fake(covered, labels, nominal=0.9):
            calls.append(len(labels))
            if len(calls) <= 4:
                return {"flagged": [1] if len(calls) <= flagged_worlds else []}
            return {"flagged": ["0"]}

        monkeypatch.setattr(uncertainty_certify.coverage, "slice_flag", fake)
        scripted_wilson(monkeypatch, [0.5, 0.5, 0.8, 0.9, 0.9])
        return uncertainty_certify.gate_g5b(4, 10_000)

    result = run(1)
    assert result["familywise_false_flag"] == 0.25
    assert result["min_certified_rows"] == 100
    assert result["passed"] is True
    assert set(result["detection_by_slice_rows"]) == {
        "noise_3x",
        "noise_1.5x",
        "noise_1.25x",
    }
    assert result["calibration_rows"] == 500
    assert result["margin"] == uncertainty_certify.coverage.SLICE_MARGIN
    assert run(2)["passed"] is False


def test_g5b_without_a_certified_size_does_not_pass(monkeypatch):
    monkeypatch.setattr(
        uncertainty_certify.coverage,
        "slice_flag",
        lambda covered, labels, nominal=0.9: {"flagged": []},
    )
    monkeypatch.setattr(uncertainty_certify, "wilson_lower", lambda h, r: 0.1)

    result = uncertainty_certify.gate_g5b(4, 10_000)

    assert result["min_certified_rows"] is None
    assert result["passed"] is False


def test_g6_passes_only_when_the_weighted_floor_holds_and_beats_plain(
    monkeypatch,
):
    monkeypatch.setattr(uncertainty_certify, "ESTIMATED_RATIO_REPLICATES", 2)
    plain = uncertainty_certify.gate_g6(2, 10_000)["known_ratio"]["unweighted"]

    def passed(meets, floor):
        monkeypatch.setattr(
            uncertainty_certify,
            "mean_floor",
            lambda values, nominal=0.9: {
                "mean": 0.9,
                "se": 0.01,
                "floor": floor,
                "meets_floor": meets,
            },
        )
        result = uncertainty_certify.gate_g6(2, 10_000)
        assert result["estimated_passed"] is meets
        return result["passed"]

    assert passed(True, 0.99) is True
    assert passed(False, 0.99) is False
    assert passed(True, plain) is False
    assert passed(True, plain - 0.01) is False


def test_g9_passes_on_a_negative_upper_limit_and_a_small_residual(
    monkeypatch,
):
    real_mint = uncertainty_certify.reconcile.mint

    def residual(value):
        monkeypatch.setattr(
            uncertainty_certify.reconcile,
            "coherence_residual",
            lambda matrix, summing: value,
        )

    residual(0.0)
    assert uncertainty_certify.gate_g9(4, 10_000)["passed"] is True
    residual(1.0)
    assert uncertainty_certify.gate_g9(4, 10_000)["passed"] is False
    residual(1e-8)
    assert uncertainty_certify.gate_g9(4, 10_000)["passed"] is False

    residual(0.0)
    monkeypatch.setattr(
        uncertainty_certify.reconcile,
        "mint",
        lambda base_f, summing, w: real_mint(base_f, summing, w) + 1.9,
    )
    worse = uncertainty_certify.gate_g9(4, 10_000)
    assert 0 < worse["upper_95"] < 1
    assert worse["passed"] is False


def test_g11i_passes_when_only_the_covariate_estimate_is_unbiased(
    monkeypatch,
):
    truth = 0.7
    pattern = numpy.array([-0.01, 0.01, -0.01, 0.01])
    se = float(pattern.std(ddof=1) / 2)

    def passed(cox_bias, marginal_bias):
        calls = []

        def fake(time, event, risk, tau, weights):
            calls.append(1)
            if len(calls) == 1:
                return truth
            index = (len(calls) - 2) // 2
            bias = cox_bias if len(calls) % 2 == 0 else marginal_bias
            return truth + bias + float(pattern[index])

        monkeypatch.setattr(
            uncertainty_certify.ipcw, "truncated_concordance", fake
        )
        return uncertainty_certify.gate_g11i(4, 10_000)["passed"]

    assert passed(0.0, 0.5) is True
    assert passed(0.0, 0.0) is False
    assert passed(0.5, 0.5) is False
    assert passed(3.5 * se, 0.5) is False
    assert passed(0.0, 3.5 * se) is True


def test_problems_needs_a_gates_section():
    found = uncertainty_certify.problems({})
    assert found == [
        f"{name}: no certificate" for name in uncertainty_certify.GATES
    ]


def test_the_help_text_and_the_check_output_are_exact(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        uncertainty_certify.main(["--help"])
    assert "Measured gates for the coverage and breadth methods" in (
        capsys.readouterr().out
    )
    monkeypatch.setattr(
        uncertainty_certify, "problems", lambda document=None: ["bad"]
    )
    uncertainty_certify.main(["--check"])
    assert capsys.readouterr().out == (
        '{\n  "ok": false,\n  "problems": [\n    "bad"\n  ]\n}\n'
    )
