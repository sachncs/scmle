"""Claim C3, reference agreement: product statistics against references."""

import json
import math
import pathlib
import time
import tomllib
from typing import Any

import numpy
import pandas
import pytest

from benchmarks.evidence import statistics
from scmle.assurance.ledger import budget
from scmle.assurance.ledger import multiplicity
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.evaluation import coverage
from scmle.evaluation import metrics
from scmle.evaluation import resources


@pytest.fixture(autouse=True)
def frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins the code fingerprint, so edits elsewhere cannot stale a test."""
    monkeypatch.setattr(statistics, "fingerprint", lambda: "code")


def gates() -> set[str]:
    """Every gate key of `gates.toml`, as its dotted table name."""
    found = tomllib.loads(statistics.GATES.read_text(encoding="utf-8"))
    keys: set[str] = set()

    def walk(table: dict[str, Any], prefix: str) -> None:
        for name, value in table.items():
            if isinstance(value, dict) and "site" in value:
                keys.add(prefix + name)
            elif isinstance(value, dict):
                walk(value, prefix + name + ".")

    walk(found, "")
    return keys


def one(name: str) -> statistics.Statistic:
    return next(s for s in statistics.STATISTICS if s.name == name)


def test_error_is_relative_above_one_and_absolute_below() -> None:
    assert statistics.error(2.0 + 2e-9, 2.0) == pytest.approx(1e-9)
    assert statistics.error(0.5 + 1e-9, 0.5) == pytest.approx(1e-9)
    assert statistics.error(math.nan, math.nan) == 0.0
    assert statistics.error(math.inf, math.inf) == 0.0
    assert statistics.error(math.nan, 0.5) == math.inf
    assert statistics.error(-math.inf, 0.5) == math.inf


def test_every_gate_is_classified_once() -> None:
    deciding = set(statistics.DECIDING)
    predicates = set(statistics.PREDICATES)
    assert deciding.isdisjoint(predicates)
    assert deciding | predicates == gates()


def test_every_deciding_statistic_exists_and_has_two_references() -> None:
    names = {s.name for s in statistics.STATISTICS}
    for chosen in statistics.DECIDING.values():
        for name in chosen:
            assert name in names
            assert len(one(name).references) >= statistics.SECOND


def test_inventory_lists_every_reported_metric() -> None:
    reported = {
        name
        for names in metrics.Repertoire.report_names.values()
        for name in names
    }
    assert statistics.reported() == reported


def test_a_disagreeing_product_fails_and_a_missing_reference_fails() -> None:
    case = statistics.Case("fixed", "simulated", {})
    wrong = statistics.Statistic(
        "wrong",
        lambda: [case],
        lambda case: {"a": 0.5 + 1e-6, "b": 0.25},
        {"scratch": lambda case: {"a": 0.5}},
    )
    found = statistics.compare(wrong, case, 1e-8, deciding=False)
    assert [(c.key, c.within) for c in found] == [("a", False), ("b", False)]
    assert found[0].error == pytest.approx(1e-6)
    assert found[1].error == math.inf


def test_a_deciding_statistic_needs_a_second_reference() -> None:
    case = statistics.Case("fixed", "simulated", {})
    single = statistics.Statistic(
        "single",
        lambda: [case],
        lambda case: {"a": 0.5},
        {"scratch": lambda case: {"a": 0.5}},
    )
    assert statistics.compare(single, case, 1e-8, deciding=False)[0].within
    assert not statistics.compare(single, case, 1e-8, deciding=True)[0].within


@pytest.mark.parametrize(
    "task",
    [
        tasks.Task.BINARY,
        tasks.Task.MULTICLASS,
        tasks.Task.REGRESSION,
        tasks.Task.ORDINAL,
        tasks.Task.MULTILABEL,
        tasks.Task.MULTIOUTPUT,
        tasks.Task.FORECAST,
        tasks.Task.UPLIFT,
        tasks.Task.RANKING,
        tasks.Task.SURVIVAL,
    ],
)
def test_simulated_metrics_agree_with_their_reference(
    task: tasks.Task,
) -> None:
    statistic = one(f"metrics.{task.value}")
    case = statistics.simulate(task, 0)
    found = statistics.compare(statistic, case, 1e-8, deciding=False)
    assert {c.key for c in found} == set(metrics.CATALOG.report_names[task])
    assert [c.key for c in found if not c.within] == []


def test_qwk_disagreement_on_an_absent_level_is_detected() -> None:
    case = statistics.simulate(tasks.Task.ORDINAL, statistics.TIED)
    found = statistics.compare(
        one("metrics.ordinal"), case, 1e-8, deciding=False
    )
    assert [c.key for c in found if not c.within] == ["qwk"]


def test_latency_quantile_is_deciding_for_both_serving_gates() -> None:
    assert statistics.DECIDING["select.serving_limits"] == ("latency",)
    assert statistics.DECIDING["select.pareto_front"] == ("latency",)


def test_latency_p95_agrees_on_tied_and_short_timing_vectors() -> None:
    statistic = one("latency")
    cases = statistics.simulated("latency")
    assert len(cases) >= 4
    for case in cases:
        found = statistics.compare(statistic, case, 1e-8, deciding=True)
        assert [c.key for c in found] == ["p95"]
        assert [c.key for c in found if not c.within] == []


def test_latency_replay_feeds_the_product_the_given_spans() -> None:
    ticks = statistics.ticks([1.0, 2.0, 2.0, 2.0, 9.0])
    assert statistics.spans(ticks) == pytest.approx([1.0, 2.0, 2.0, 2.0, 9.0])
    usage = statistics.timed(statistics.Clock(ticks), 5)
    assert usage.rows_timed == 5
    assert usage.p95_latency_ms == pytest.approx(7.6)
    assert vars(resources)["time"] is time


def test_latency_real_case_records_a_genuine_measurement() -> None:
    real = statistics.recorded()
    assert real.source == "real"
    spans = statistics.spans(real.data["ticks"])
    assert len(spans) == resources.LATENCY_ROWS
    assert all(s > 0 for s in spans)
    found = statistics.compare(one("latency"), real, 1e-8, deciding=True)
    assert [(c.key, c.within) for c in found] == [("p95", True)]


def test_latency_replay_refuses_a_short_tick_stream() -> None:
    with pytest.raises(errors.InputError, match="ran out of ticks"):
        statistics.timed(statistics.Clock([0.0, 0.001]), 2)


def test_kappa_reference_reads_level_values_not_their_positions() -> None:
    truth = numpy.array([0, 0, 2, 2, 3, 3])
    guess = numpy.array([0, 2, 2, 3, 3, 0])
    weights = numpy.subtract.outer(range(4), range(4)) ** 2
    observed = numpy.zeros((4, 4))
    numpy.add.at(observed, (truth, guess), 1)
    expected = numpy.outer(observed.sum(1), observed.sum(0)) / 6
    kappa = 1 - (weights * observed).sum() / (weights * expected).sum()
    assert statistics.kappa(truth, guess) == pytest.approx(kappa, abs=1e-15)


def test_multiplicity_references_match_the_product_on_ties_and_extremes() -> (
    None
):
    values = [1e-300, 0.01, 0.01, 0.04, 0.5, 1.0]
    case = statistics.Case("ties", "simulated", {"p": values})
    for name, deciding in (("multiplicity", False), ("multiplicity.by", True)):
        found = statistics.compare(one(name), case, 1e-8, deciding=deciding)
        assert found
        assert [c.key for c in found if not c.within] == []
    adjusted = multiplicity.adjust(
        values, budget.Correction.BY, statistics.ALPHA
    ).adjusted
    assert statistics.stepup(values, "by")[0] == pytest.approx(adjusted)


def test_clopper_pearson_scratch_bisection_matches_the_product() -> None:
    for covered, rows in [(0, 5), (5, 5), (212, 569), (816, 882)]:
        low, high = coverage.clopper_pearson(covered, rows, 0.9)
        assert statistics.exact(covered, rows, 0.9) == pytest.approx(
            (low, high), abs=1e-11
        )


def test_gate_thresholds_derive_from_their_bounds() -> None:
    for name in ("support", "onnx", "thresholdout"):
        statistic = one(name)
        for case in statistic.cases():
            found = statistics.compare(statistic, case, 1e-8, deciding=True)
            assert [c.key for c in found if not c.within] == []


def test_missingness_test_agrees_on_chi_square_and_fisher_tables() -> None:
    statistic = one("missingness")
    for case in statistics.simulated("missingness"):
        found = statistics.compare(statistic, case, 1e-8, deciding=True)
        assert {c.key for c in found} == {"p_value", "cramers_v"}
        assert [c.key for c in found if not c.within] == []


def test_screen_recount_agrees_on_a_small_simulated_table() -> None:
    statistic = one("screen")
    case = statistics.simulated("screen")[0]
    found = statistics.compare(statistic, case, 1e-8, deciding=True)
    kinds = {c.key.rsplit("|", 1)[1] for c in found}
    assert {"statistic", "p", "adjusted", "significant", "bound"} <= kinds
    assert [c.key for c in found if not c.within] == []


def test_recount_refuses_a_draw_that_is_not_a_permutation() -> None:
    with pytest.raises(errors.InputError, match="not a permutation"):
        statistics.permutation(numpy.array([[0, 0, 1]]))


def document(**changes: Any) -> dict[str, Any]:
    found = {
        "version": 1,
        "claim": "C3",
        "code": "code",
        "tolerance": {"name": statistics.VARIABLE, "value": 1e-8},
        "comparisons": [
            {
                "statistic": "metrics.binary",
                "case": "breast_cancer",
                "source": "real",
                "key": "roc_auc",
                "deciding": False,
                "product": 0.9,
                "references": {"scratch": 0.9},
                "error": 0.0,
                "within": True,
            },
            {
                "statistic": "screen",
                "case": "breast_cancer",
                "source": "real",
                "key": "x|skill|p",
                "deciding": True,
                "product": 0.01,
                "references": {"scratch": 0.01, "sklearn": 0.01},
                "error": 0.0,
                "within": True,
            },
        ],
        "residue": {"reported": [], "deciding": []},
    }
    found.update(changes)
    return found


def test_summary_passes_only_with_nothing_left_unmeasured() -> None:
    assert statistics.summary(document())[1] == "pass"
    assert statistics.summary(document(), "deciding")[1] == "pass"
    left = document(residue={"reported": ["intervals"], "deciding": []})
    assert statistics.summary(left)[1] == "not measured"
    assert statistics.summary(left, "deciding")[1] == "pass"


def test_summary_fails_on_any_disagreement_even_with_residue() -> None:
    bad = document(residue={"reported": ["intervals"], "deciding": ["m"]})
    bad["comparisons"][1]["within"] = False
    bad["comparisons"][1]["error"] = 1e-3
    value, verdict = statistics.summary(bad, "deciding")
    assert verdict == "fail"
    assert "x|skill|p" in value
    assert statistics.summary(bad)[1] == "fail"


def test_summary_refuses_an_unknown_scope() -> None:
    with pytest.raises(errors.InputError, match="unknown scope"):
        statistics.summary(document(), "other")


def test_stale_or_missing_documents_are_not_measured(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(statistics, "PATH", tmp_path / "statistics.json")
    assert statistics.summary()[1] == "not measured"
    assert statistics.problems() == ["statistics: statistics.json is missing"]
    stale = document(code="old")
    assert statistics.summary(stale)[1] == "not measured"
    assert statistics.problems(stale) == [
        "statistics: measured on other code; run --write"
    ]


def test_show_prints_the_comparisons_of_the_chosen_statistics(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert statistics.main(["--show", "--statistic", "support"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert {c["statistic"] for c in shown["comparisons"]} == {"support"}
    assert shown["verdicts"]["deciding"][1] in {"pass", "not measured"}


def test_show_refuses_an_unknown_statistic() -> None:
    with pytest.raises(errors.InputError, match="unknown statistic"):
        statistics.main(["--show", "--statistic", "nothing"])


def test_simulated_real_table_frames_have_no_missing_target() -> None:
    frame = statistics.simulated("screen")[0].data["frame"]
    assert isinstance(frame, pandas.DataFrame)
    assert frame["target"].notna().all()
