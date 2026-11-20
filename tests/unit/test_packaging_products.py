"""The frozen products of the packaging stage and of the parts it inherits.

Each part returns what it made (`Uncertainty`, `Diagnostics`, `Verdict`),
and `Packaging` returns one `Package`, whose `apply` holds it on the
run's board by type. The board and its services are fakes.
"""

import dataclasses
import types

import pytest

from scmle.assurance import coverage_confirm
from scmle.contracts import lineage
from scmle.contracts import tasks
from scmle.pipeline.stages import assessment
from scmle.pipeline.stages import bracketing
from scmle.pipeline.stages import diagnosis
from scmle.pipeline.stages import examination
from scmle.pipeline.stages import packaging
from scmle.pipeline.stages import profiling
from scmle.pipeline.stages import releasing
from scmle.pipeline.stages import selecting


class Journal:
    """A fake run journal (or run store) that keeps what is written."""

    def __init__(self):
        self.files = {}

    def write(self, name, document):
        self.files[name] = document

    def write_json(self, name, document):
        self.files[name] = document


def blank(held=None, **fields):
    """A fake run's board holding stand-ins for products, by product type."""
    made = dict(held or {})
    return types.SimpleNamespace(
        journal=Journal(), made=made, get=made.__getitem__, **fields
    )


def undeclared():
    """A fake board whose spec declares no success criteria."""
    partition = types.SimpleNamespace(
        task=tasks.Task.BINARY,
        metric=types.SimpleNamespace(name="roc_auc", greater_is_better=True),
    )
    return blank(
        {profiling.Partition: partition},
        services=types.SimpleNamespace(
            spec=types.SimpleNamespace(
                problem=types.SimpleNamespace(success_criteria={})
            )
        ),
    )


def graph():
    made = lineage.Graph()
    made.add_node(lineage.Node("raw-source", lineage.Kind.RAW_SOURCE, "0" * 64))
    return made


def test_a_verdict_is_returned_without_touching_the_evaluation():
    store = Journal()
    evaluation = {"test_metrics": {"roc_auc": 0.99}}
    verdict = packaging.Packaging().judge(undeclared(), store, evaluation)
    assert evaluation == {"test_metrics": {"roc_auc": 0.99}}
    assert verdict.policy is None
    assert verdict.outcome["status"] == "policy_not_declared"
    assert verdict.evidence == {"records": [], "ready": False}
    assert list(verdict.entries()) == [
        "release_policy",
        "release_evaluation",
        "release_evidence",
    ]
    assert list(store.files) == ["release_evaluation.json"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        verdict.policy = {}  # type: ignore[misc]


def test_recording_a_verdict_adds_its_entries_in_order():
    evaluation = {"test_metrics": {}}
    packaging.Packaging().release(undeclared(), Journal(), evaluation)
    assert list(evaluation) == [
        "test_metrics",
        "release_policy",
        "release_evaluation",
        "release_evidence",
    ]


def test_diagnostics_gathers_the_four_reports_in_record_order(monkeypatch):
    for method, name in (
        ("chart", "learning_curve"),
        ("revalidate", "nested_validation"),
        ("decompose", "bias_variance"),
    ):
        monkeypatch.setattr(
            diagnosis.Diagnosis,
            method,
            lambda self, context, selected, name=name: {"part": name},
        )
    monkeypatch.setattr(
        diagnosis.Diagnosis, "sensitivity", lambda self, context, selected: None
    )
    found = packaging.Packaging().probe(blank(), object())
    assert found == diagnosis.Diagnostics(
        {"part": "learning_curve"},
        {"part": "nested_validation"},
        {"part": "bias_variance"},
        None,
    )
    assert found.entries() == {
        "learning_curve": {"part": "learning_curve"},
        "nested_cv": {"part": "nested_validation"},
        "bias_variance": {"part": "bias_variance"},
        "sensitivity": None,
    }


def test_uncertainty_without_test_rows_measures_nothing():
    bundle = types.SimpleNamespace(
        conformal=None, quantile=None, coverage_report=None
    )
    exam = types.SimpleNamespace(rows=None)
    found = packaging.Packaging().gauge(blank({assessment.Exam: exam}), bundle)
    assert found == bracketing.Uncertainty(None, None)
    assert found.entries() == {}
    found.equip(bundle)
    assert bundle.coverage_report is None


def test_uncertainty_equips_the_bundle_with_its_report():
    bundle = types.SimpleNamespace(coverage_report=None)
    made = bracketing.Uncertainty(
        None, None, [{"level": 0.9, "coverage": 0.88}], {"methods": {}}
    )
    made.equip(bundle)
    assert bundle.coverage_report == {"methods": {}}
    assert made.entries() == {
        "quantile_coverage": [{"level": 0.9, "coverage": 0.88}]
    }


def test_settling_opens_the_pending_ledger_without_assigning_it(monkeypatch):
    concluded = []
    monkeypatch.setattr(
        profiling.Profiling,
        "conclude",
        lambda self, context, pending, subject, runner: concluded.append(
            (pending, subject, runner)
        ),
    )
    pending = types.SimpleNamespace(subject="charter")
    board = blank(
        {
            selecting.Choice: types.SimpleNamespace(ledger=pending),
            profiling.Partition: types.SimpleNamespace(task=tasks.Task.BINARY),
            examination.Decision: examination.Decision(),
        }
    )
    packaging.Packaging().settle(board, object())
    assert concluded == [(pending, "charter", None)]
    assert list(board.journal.files) == [coverage_confirm.DOCUMENT]


def test_settling_without_a_pending_ledger_does_nothing():
    choice = types.SimpleNamespace(ledger=None)
    board = blank({selecting.Choice: choice})
    packaging.Packaging().settle(board, object())
    assert board.journal.files == {}


def test_a_package_is_held_by_type():
    made = packaging.Package(
        bundle=None,
        result=None,
        graph=graph(),
        evaluation={"run_id": "r"},
    )
    board = types.SimpleNamespace(made={})
    made.apply(board)
    assert board.made == {packaging.Package: made}
    assert made.summary() == made.graph.digest()
    with pytest.raises(dataclasses.FrozenInstanceError):
        made.bundle = None  # type: ignore[misc]


def test_the_package_digest_ignores_everything_but_the_graph():
    first = packaging.Package(None, None, graph(), {"a": 1})
    second = packaging.Package(None, None, graph(), {"b": 2})
    assert first.save(None) == second.save(None)  # type: ignore[arg-type]


def test_each_part_names_its_product():
    assert releasing.Verdict.__dataclass_params__.frozen
    assert diagnosis.Diagnostics.__dataclass_params__.frozen
    assert bracketing.Uncertainty.__dataclass_params__.frozen
    assert packaging.Packaging.makes is packaging.Package
