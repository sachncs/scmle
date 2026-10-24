"""Tests for events, errors, plans and model-adapter contract."""

import json

import pytest

from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import events
from scmle.contracts import experiments
from scmle.contracts import models
from scmle.contracts import tasks


def test_event_log_writes_jsonl(tmp_path):
    log = events.EventLog(tmp_path / "events.jsonl")
    log.emit(events.Event("r1", "state.CREATED", "start"))
    log.emit(events.Event("r1", "state.VALIDATING"))
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert [json.loads(line)["kind"] for line in lines] == list(log.kinds())
    assert json.loads(lines[0])["reason"] == "start"


def test_errors_carry_category_and_details():
    error = errors.SchemaError("bad", {"missing": ["a"]})
    assert isinstance(error, errors.DataError)
    assert error.category == "schema"
    assert error.details == {"missing": ["a"]}
    assert errors.BlockedError("x").details == {}


def test_feature_plan_helpers():
    plan = data.FeaturePlan(("a",), ("b", "c"), {"d": "constant"})
    assert plan.features == ("a", "b", "c")
    assert plan.columns_for("categorical") == ("b", "c")
    assert plan.columns_for("datetime") == ()
    assert plan.to_dict()["dropped"] == {"d": "constant"}


def test_plan_summary_and_dict():
    decision = experiments.Decision(
        "dummy", experiments.DecisionAction.INCLUDE, "baseline"
    )
    plan = experiments.ExperimentPlan(
        "p1", "regression", "rmse", "random", ("dummy",), 60, (decision,)
    )
    assert "[include] dummy: baseline" in plan.summary()
    assert plan.to_dict()["decisions"][0]["action"] == "include"


def test_trial_result_serializes_status():
    trial = experiments.TrialResult(
        "t1", "dummy", experiments.TrialStatus.FAILED, error="boom"
    )
    assert trial.to_dict()["status"] == "failed"


def test_adapter_rejection_reasons():
    class Probe(models.ModelAdapter):
        name = "probe"
        family = "x"
        capabilities = models.Capabilities(
            frozenset({tasks.TaskType.BINARY}), max_rows=10
        )
        requires = ("definitely_not_installed_pkg",)

        def build(self, task, seed):
            raise NotImplementedError

    adapter = Probe()
    assert "does not support" in adapter.rejection_reason(
        tasks.TaskType.REGRESSION
    )
    assert "missing dependency" in adapter.rejection_reason(
        tasks.TaskType.BINARY
    )
    assert not adapter.is_available()


def test_abstract_adapter_cannot_be_instantiated():
    with pytest.raises(TypeError):
        models.ModelAdapter()
