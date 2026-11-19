"""The frozen products of the packaging stage and of the parts it inherits.

Each part returns what it made (`Uncertainty`, `Diagnostics`, `Verdict`)
instead of setting run-context fields, and `PackageStage` returns one
`Package`, whose `apply` sets the context fields the readers that have not
converted yet still use. The context and its services are fakes.
"""

import dataclasses
import types

import pytest

from scmle.assurance import coverage_confirm
from scmle.contracts import lineage
from scmle.contracts import tasks
from scmle.pipeline.stages import bracketing
from scmle.pipeline.stages import diagnosis
from scmle.pipeline.stages import packaging
from scmle.pipeline.stages import profiling
from scmle.pipeline.stages import releasing


class Journal:
    """A fake run journal that keeps what is written."""

    def __init__(self):
        self.files = {}

    def write_json(self, name, document):
        self.files[name] = document


def blank(**fields):
    """A fake run context with the fields a part reads."""
    return types.SimpleNamespace(journal=Journal(), **fields)


def undeclared():
    """A fake context whose spec declares no success criteria."""
    return blank(
        task=tasks.Task.BINARY,
        metric=types.SimpleNamespace(name="roc_auc", greater_is_better=True),
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
    verdict = packaging.PackageStage().judge(undeclared(), store, evaluation)
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
    packaging.PackageStage().release_evaluation(
        undeclared(), Journal(), evaluation
    )
    assert list(evaluation) == [
        "test_metrics",
        "release_policy",
        "release_evaluation",
        "release_evidence",
    ]


def test_diagnostics_gathers_the_four_reports_in_record_order(monkeypatch):
    for name in ("learning_curve", "nested_validation", "bias_variance"):
        monkeypatch.setattr(
            diagnosis.Diagnosis,
            name,
            lambda self, context, selected, name=name: {"part": name},
        )
    monkeypatch.setattr(
        diagnosis.Diagnosis, "sensitivity", lambda self, context, selected: None
    )
    found = packaging.PackageStage().probe(blank(), object())
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
    found = packaging.PackageStage().gauge(blank(test_rows=None), bundle)
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
        profiling.ProfileStage,
        "conclude",
        lambda self, context, pending, subject, runner: concluded.append(
            (pending, subject, runner)
        ),
    )
    pending = types.SimpleNamespace(subject="charter")
    context = blank(
        pending_ledger=pending,
        task=tasks.Task.BINARY,
        coverage_claim_sha256=None,
    )
    packaging.PackageStage().settle(context, object())
    assert concluded == [(pending, "charter", None)]
    assert context.pending_ledger is pending
    assert list(context.journal.files) == [coverage_confirm.DOCUMENT]


def test_settling_without_a_pending_ledger_does_nothing():
    context = blank(pending_ledger=None)
    packaging.PackageStage().settle(context, object())
    assert context.journal.files == {}


def test_a_package_applies_every_field_the_old_readers_use():
    made = packaging.Package(
        bundle=None,
        result=None,
        evidence_graph=graph(),
        evaluation={"run_id": "r"},
    )
    board = types.SimpleNamespace()
    made.apply(board)
    assert board.evidence_graph is made.evidence_graph
    assert board.evaluation == {"run_id": "r"}
    assert board.bundle is None and board.result is None
    assert made.summary() == made.evidence_graph.digest()
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
    assert packaging.PackageStage.makes is packaging.Package
