"""The helper functions of the pipeline stages, one module at a time.

The stages read a run context; each test stands one in with only the fields
the helper reads, and a governor whose clock is a fixed number of seconds.
"""

import dataclasses
import hashlib
import json
import types

import numpy
import pandas
import pytest

from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import experiments
from scmle.contracts import tasks
from scmle.evaluation import resources
from scmle.pipeline.stages import admission
from scmle.pipeline.stages import cost
from scmle.pipeline.stages import examination
from scmle.pipeline.stages import exploration
from scmle.pipeline.stages import reporting
from scmle.pipeline.stages import scoring
from scmle.pipeline.stages import selecting
from scmle.pipeline.stages import surveying


def trial(name="m", seconds=10.0, folds=5, strategy="default", score=0.5):
    return experiments.Trial(
        f"t-{name}",
        name,
        experiments.Termination.OK,
        "accuracy",
        score,
        0.0,
        [score] * folds,
        seconds,
        params={"alpha": 1},
        strategy=strategy,
    )


class Governor:
    """Leaves `seconds` before the finish; the package reserve is `package`."""

    def __init__(self, seconds, package=0.0):
        self.seconds = seconds
        self.reserve = types.SimpleNamespace(parts={"package": package})

    def left(self, phase):
        return self.seconds

    def overdue(self, seconds, keep=0.0):
        left = self.seconds - keep
        if seconds > left:
            return f"needs {seconds:.1f} s of {left:.1f} s"
        return None


def context(seconds=100.0, folds=5, package=0.0, budget=60):
    recorded = []
    entered = []
    return types.SimpleNamespace(
        regulator=Governor(seconds, package),
        split=types.SimpleNamespace(
            manifest=types.SimpleNamespace(folds=[([0], [1])] * folds)
        ),
        spec=types.SimpleNamespace(
            resources=types.SimpleNamespace(time_budget_seconds=budget)
        ),
        forgo=lambda work, reason: recorded.append((work, reason)),
        enter=lambda *args: entered.append(args),
        recorded=recorded,
        entered=entered,
    )


def test_a_fold_costs_its_share_of_the_trial_with_the_margin():
    assert cost.FIT_MARGIN == 1.5
    assert cost.TAIL_SECONDS == 1.0
    assert cost.foldcost(trial(seconds=10.0, folds=5)) == pytest.approx(3.0)
    assert cost.foldcost(trial(seconds=10.0, folds=0)) == pytest.approx(15.0)


def test_a_refit_scales_a_fold_by_k_over_k_minus_one():
    assert cost.refitcost(context(folds=5), trial()) == pytest.approx(3.75)
    assert cost.refitcost(context(folds=2), trial()) == pytest.approx(6.0)
    assert cost.refitcost(context(folds=1), trial()) == pytest.approx(3.0)
    missing = context()
    missing.split = None
    with pytest.raises(errors.ScmleError, match="'split' is unavailable"):
        cost.refitcost(missing, trial())


def test_affordable_passes_over_late_trials_on_record():
    board = context(seconds=10.0)
    slow, quick = trial("slow", 100.0), trial("quick", 1.0)
    assert cost.affordable(board, [slow, quick]) is quick
    assert board.recorded == [
        ("selecting slow/default", "its refit needs 37.5 s of 10.0 s")
    ]


def test_affordable_takes_the_cheapest_when_none_fits():
    board = context(seconds=1.0)
    slower, slow = trial("slower", 200.0), trial("slow", 100.0)
    assert cost.affordable(board, [slower, slow]) is slow
    assert board.recorded[-1] == (
        "selecting slow/default",
        "no refit fits before the run's end; the cheapest refit is taken",
    )
    assert len(board.recorded) == 3


def test_late_rejects_a_candidate_whose_refit_cannot_end():
    board = context(seconds=10.0)
    assert cost.late(board, trial("quick", 1.0)) is False
    assert board.entered == []
    assert cost.late(board, trial("slow", 100.0)) is True
    assert board.entered == [
        (
            "slow/default",
            experiments.Edict.REJECT,
            "refit needs 37.5 s of 10.0 s",
        )
    ]
    assert board.recorded == [
        ("measuring slow/default", "its refit needs 37.5 s of 10.0 s")
    ]


def test_halt_keeps_the_packaging_tail():
    board = context(seconds=4.0)
    assert cost.halt(board, trial(seconds=10.0), "curve") is None
    reason = cost.halt(board, trial(seconds=20.0), "curve")
    assert reason == "needs 6.0 s of 3.0 s"
    assert board.recorded == [("curve", reason)]
    edge = context(seconds=4.0)
    assert cost.halt(edge, trial(seconds=10.0, folds=5), "x") is None


class Timed:
    def __init__(self, seconds):
        self.seconds = seconds
        self.calls = []

    def predict(self, frame):
        self.calls.append(len(frame))
        return numpy.zeros(len(frame))


def test_meter_times_every_row_when_time_allows(monkeypatch):
    board = context(seconds=1000.0)
    measured = []
    monkeypatch.setattr(
        cost.resources,
        "measure",
        lambda pipeline, features, rows: measured.append(rows) or "usage",
    )
    rows = pandas.DataFrame({"a": range(3)})
    assert cost.meter(board, Timed(0), rows) == "usage"
    assert measured == [resources.LATENCY_ROWS]
    assert board.recorded == []


def test_meter_times_fewer_rows_when_time_is_short(monkeypatch):
    board = context(seconds=10.0, package=4.0)
    measured = []
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(cost.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(
        cost.resources,
        "measure",
        lambda pipeline, features, rows: measured.append(rows) or "usage",
    )
    cost.meter(board, Timed(0), pandas.DataFrame({"a": range(3)}))
    assert measured == [4]
    assert board.recorded == [
        (
            "serving measurement",
            (
                f"timed 4 of {resources.LATENCY_ROWS} rows: one prediction "
                "takes about 1.50 s and 6.0 s remain"
            ),
        )
    ]


def test_meter_times_at_least_one_row(monkeypatch):
    board = context(seconds=0.0)
    measured = []
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(cost.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(
        cost.resources,
        "measure",
        lambda pipeline, features, rows: measured.append(rows),
    )
    cost.meter(board, Timed(0), pandas.DataFrame({"a": [1]}))
    assert measured == [1]
    assert "0.0 s remain" in board.recorded[0][1]


def test_dimensions_count_labelled_rows_features_and_classes():
    frame = pandas.DataFrame(
        {"a": [1, 2, 3, 4], "b": [1, 1, 1, 1], "y": [0, 1, None, 2], "z": 1}
    )
    board = types.SimpleNamespace(
        frame=frame,
        spec=types.SimpleNamespace(
            data=types.SimpleNamespace(target="y", targets=["z"])
        ),
        task=tasks.Task.MULTICLASS,
    )
    assert cost.dimensions(board) == (3, 2, 3)
    board.task = tasks.Task.REGRESSION
    assert cost.dimensions(board) == (3, 2, 0)


def test_the_diagnostics_get_a_share_of_the_budget_capped_by_time_left():
    assert examination.DIAGNOSTIC_BUDGET_SHARE == 0.25
    assert examination.DIAGNOSTIC_MIN_SECONDS == 30.0
    ration = examination.Examination.ration
    assert ration(context(seconds=1000.0, budget=200)) == 50.0
    assert ration(context(seconds=1000.0, budget=60)) == 30.0
    assert ration(context(seconds=40.0, package=15.0, budget=600)) == 25.0
    assert ration(context(seconds=5.0, package=15.0, budget=600)) == 0.0


class Fitted:
    classes_ = numpy.array([0, 1])

    def predict_proba(self, frame):
        return numpy.column_stack([1 - frame.a, frame.a])

    def predict(self, frame):
        return numpy.column_stack([frame.a * 2, frame.a * 3])


def test_reference_scores_are_probabilities_or_first_predictions():
    frame = pandas.DataFrame({"a": [0.25, 0.75]})
    binary = scoring.reference_scores(Fitted(), frame, tasks.Task.BINARY)
    assert binary.name == "score"
    assert binary.tolist() == [0.25, 0.75]
    multi = scoring.reference_scores(Fitted(), frame, tasks.Task.MULTIOUTPUT)
    assert multi.name == "prediction"
    assert multi.tolist() == [0.5, 1.5]
    assert scoring.MATRIX_DIMENSIONS == 2


def test_reference_scores_keep_one_dimensional_predictions():
    class Flat:
        def predict(self, frame):
            return numpy.array([1.0, 2.0])

    frame = pandas.DataFrame({"a": [0.0, 1.0]})
    assert scoring.reference_scores(
        Flat(), frame, tasks.Task.REGRESSION
    ).tolist() == [1.0, 2.0]


def test_pin_digests_the_frozen_selection():
    chosen = trial("ridge")
    frozen = {
        "trial": "t-ridge",
        "model": "ridge",
        "strategy": "default",
        "params": {"alpha": 1},
        "decision": {"threshold": 0.4},
    }
    text = json.dumps(frozen, sort_keys=True, default=str)
    assert scoring.pin(chosen, {"threshold": 0.4}) == (
        hashlib.sha256(text.encode("utf-8")).hexdigest()
    )
    assert scoring.pin(chosen, None) != scoring.pin(chosen, {"threshold": 0.4})


def test_coverage_on_confirmation_needs_a_confirmatory_block():
    def charter(block):
        return types.SimpleNamespace(uncertainty=block)

    confirmatory = types.SimpleNamespace(confirmatory=True)
    assert (
        scoring.coverage_on_confirmation(charter(None), tasks.Task.REGRESSION)
        is False
    )
    assert (
        scoring.coverage_on_confirmation(
            charter(types.SimpleNamespace(confirmatory=False)),
            tasks.Task.REGRESSION,
        )
        is False
    )
    assert (
        scoring.coverage_on_confirmation(
            charter(confirmatory), tasks.Task.REGRESSION
        )
        is True
    )


def test_without_a_charter_there_are_no_slices_or_decision_layer():
    board = types.SimpleNamespace(spec=types.SimpleNamespace(charter_path=None))
    assert scoring.slicers(board) == ()
    assert scoring.decisional(board) is None
    board.spec.charter_path = ""
    assert scoring.slicers(board) == ()


def test_freeze_needs_a_decision_layer():
    charter = types.SimpleNamespace(decision=None, costs=object())
    with pytest.raises(errors.ScmleError, match="no decision layer"):
        scoring.freeze(None, None, None, charter, 0, 0)


def test_no_threshold_is_chosen_outside_binary_tasks():
    board = types.SimpleNamespace(
        spec=types.SimpleNamespace(
            decision=types.SimpleNamespace(threshold_objective="f1"),
            charter_path=None,
        ),
        task=tasks.Task.REGRESSION,
    )
    fitted = object()
    assert scoring.choose_decision_threshold(board, fitted) == (fitted, None)
    board.task = tasks.Task.BINARY
    board.spec.decision.threshold_objective = "none"
    assert scoring.choose_decision_threshold(board, fitted) == (fitted, None)


def test_the_threshold_search_is_skipped_on_record_when_late():
    board = context(seconds=1.0, package=0.5)
    board.spec = types.SimpleNamespace(
        decision=types.SimpleNamespace(threshold_objective="f1"),
        charter_path=None,
    )
    board.task = tasks.Task.BINARY
    board.selected = trial(seconds=10.0)
    fitted = object()
    assert scoring.choose_decision_threshold(board, fitted) == (fitted, None)
    assert board.recorded == [("decision threshold", "needs 15.0 s of 0.5 s")]


def test_appraise_needs_a_chartered_rule():
    board = types.SimpleNamespace(spec=types.SimpleNamespace(charter_path=None))
    assert scoring.appraise(board, None, None, None, {"f11": {}}) is None


def test_governing_prefers_the_charter_and_refuses_a_contradiction():
    assert exploration.governing("rule", "holm", None, "holm") == "holm"
    assert exploration.governing("rule", "holm", "bh", "holm") == "bh"
    assert exploration.governing("rule", "bh", "bh", "holm") == "bh"
    with pytest.raises(errors.ConfigError) as caught:
        exploration.governing("rule", "bonferroni", "bh", "holm")
    assert str(caught.value) == (
        "the run specification sets rule='bonferroni' but the charter fixes "
        "'bh'; change the specification or re-sign the charter"
    )


def plan():
    return data.Blueprint(
        ("a", "b"),
        ("c", "d"),
        text=("t",),
        monotonic={"a": 1, "b": -1},
        high_cardinality=("d",),
        dropped={"old": "constant"},
    )


def test_prune_removes_quarantined_fields_and_records_why():
    pruned = admission.prune(plan(), {"a": "leaks", "d": "proxy", "zz": "x"})
    assert pruned.numeric == ("b",)
    assert pruned.categorical == ("c",)
    assert pruned.text == ("t",)
    assert pruned.monotonic == {"b": -1}
    assert pruned.high_cardinality == ()
    assert pruned.dropped == {
        "old": "constant",
        "a": "quarantined by admissibility: leaks",
        "d": "quarantined by admissibility: proxy",
    }
    assert admission.prune(plan(), {}) == plan()


def test_register_writes_only_what_the_qualification_holds():
    written = {}
    board = types.SimpleNamespace(
        write_json=lambda path, body: written.update({path: body})
    )
    table = types.SimpleNamespace(to_dict=lambda: {"t": 1})
    store = types.SimpleNamespace(to_dict=lambda: {"s": 1})
    found = types.SimpleNamespace(table=table, f5_store=store)
    assert admission.register(board, found) == [
        "admissibility_table.json",
        "detector_certificates.json",
    ]
    assert written == {
        "admissibility_table.json": {"t": 1},
        "detector_certificates.json": {"s": 1},
    }
    empty = types.SimpleNamespace(table=None, f5_store=None)
    assert admission.register(board, empty) == []


def test_select_adds_the_rows_set_aside_as_unseen(monkeypatch):
    seen_calls = []
    monkeypatch.setattr(
        admission.scmle.assurance.framing.baseline,
        "selective_labels",
        lambda seen, decisions: seen_calls.append(
            (seen.tolist(), decisions.tolist())
        ),
    )
    view = types.SimpleNamespace(features=pandas.DataFrame({"b": [1, 0]}))
    board = types.SimpleNamespace(unlabeled=pandas.DataFrame({"b": [1.0]}))
    admission.select(board, view, "b")
    assert seen_calls[-1] == ([True, True, False], [1.0, 0.0, 1.0])
    board.unlabeled = pandas.DataFrame({"other": [1.0]})
    admission.select(board, view, "b")
    assert seen_calls[-1] == ([True, True], [1.0, 0.0])
    board.unlabeled = None
    admission.select(board, view, "b")
    assert seen_calls[-1] == ([True, True], [1.0, 0.0])


def test_attested_loads_nothing_unless_named():
    assert (
        admission.attested(types.SimpleNamespace(leakage_attestations=None))
        == ()
    )


def test_attested_explains_an_invalid_file(tmp_path):
    (tmp_path / "a.json").write_text("{", encoding="utf-8")
    spec = types.SimpleNamespace(leakage_attestations=str(tmp_path / "a.json"))
    with pytest.raises(
        errors.ConfigError, match="invalid leakage attestations"
    ):
        admission.attested(spec)


@dataclasses.dataclass(frozen=True)
class Profile:
    feature_plan: object

    def to_dict(self):
        return {"plan": self.feature_plan.numeric}


def test_amend_replaces_the_profile_without_a_curation():
    written = {}
    board = types.SimpleNamespace(
        profile=Profile(None),
        curation=None,
        write_json=lambda path, body: written.update({path: body}),
    )
    assert admission.amend(board, plan()) == []
    assert board.profile == Profile(plan())
    assert written == {"dataset_profile.json": {"plan": ("a", "b")}}
    board.profile = None
    with pytest.raises(errors.ScmleError, match="'profile' is unavailable"):
        admission.amend(board, plan())


def usage(latency, size):
    return types.SimpleNamespace(p95_latency_ms=latency, model_mb=size)


def candidate(name, score, latency, size, broken=()):
    return selecting.Candidate(
        trial(name, score=score), object(), usage(latency, size), list(broken)
    )


def test_the_frontier_marks_dominated_candidates_and_the_choice():
    best = candidate("best", 0.9, 10.0, 5.0)
    worse = candidate("worse", 0.8, 20.0, 6.0, ["latency"])
    tied = candidate("tied", 0.9, 10.0, 5.0)
    quick = candidate("quick", 0.7, 1.0, 5.0)
    rows = selecting.frontier([best, worse, tied, quick], best, 1)
    assert [r["dominated"] for r in rows] == [False, True, False, False]
    assert [r["chosen"] for r in rows] == [True, False, False, False]
    assert rows[1] == {
        "model": "worse",
        "strategy": "default",
        "score": 0.8,
        "p95_latency_ms": 20.0,
        "model_mb": 6.0,
        "violations": ["latency"],
        "dominated": True,
        "chosen": False,
    }


def test_the_frontier_respects_a_lower_is_better_metric():
    low = candidate("low", 0.1, 10.0, 5.0)
    high = candidate("high", 0.2, 10.0, 5.0)
    assert [
        r["dominated"] for r in selecting.frontier([low, high], low, -1)
    ] == [False, True]
    assert [
        r["dominated"] for r in selecting.frontier([low, high], low, 1)
    ] == [True, False]
    smaller = candidate("small", 0.1, 10.0, 4.0)
    assert [
        r["dominated"] for r in selecting.frontier([low, smaller], low, -1)
    ] == [True, False]


def test_reporting_reads_a_declared_column_as_floats():
    frame = pandas.DataFrame({"w": [1, 2]})
    assert reporting.Reporting.column(frame, "w").tolist() == [1.0, 2.0]
    assert reporting.Reporting.column(frame, "w").dtype == float
    assert reporting.Reporting.column(frame, None) is None
    assert reporting.Reporting.column(frame, "") is None
    assert reporting.Reporting.column(frame, "missing") is None


def test_graft_leaves_a_profile_without_a_graph_alone():
    profile = types.SimpleNamespace(feature_plan=plan())
    spec = types.SimpleNamespace(feature_graph=types.SimpleNamespace(nodes=()))
    assert surveying.graft(spec, profile) is profile
