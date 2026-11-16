"""The handlers that used to hide an error now say what happened (task 2.1).

Each test fails on the code before the fix: dropped resamples were not
counted, an unreadable report was misreported, and a fingerprint fallback
was indistinguishable from the default scheme.
"""

import types

import numpy
import pandas

from scmle.artifacts import claims_audit
from scmle.contracts import tasks
from scmle.monitoring import abtest
from scmle.monitoring import events


def arm(seed):
    rng = numpy.random.default_rng(seed)
    outcomes = rng.normal(size=200)
    return abtest.Arm(outcomes + rng.normal(scale=0.5, size=200), outcomes)


def failing(monkeypatch, share):
    """Makes `share` of the resampled scorings raise like a missing class."""
    real = abtest.score_arm
    calls = {"n": 0}

    def score(chosen, rows, metric, task):
        calls["n"] += 1
        full = len(rows) == len(chosen.outcomes) and calls["n"] <= 2
        if not full and (calls["n"] % 100) < share * 100:
            raise ValueError("a resample missing a class")
        return real(chosen, rows, metric, task)

    monkeypatch.setattr(abtest, "score_arm", score)


def test_every_dropped_resample_is_counted(monkeypatch):
    failing(monkeypatch, 0.0)
    clean = abtest.compare_arms(
        tasks.Task.REGRESSION, "rmse", arm(1), arm(2), bootstraps=100
    )
    assert clean.dropped == 0


def test_too_few_kept_resamples_give_no_interval_and_no_decision(monkeypatch):
    failing(monkeypatch, 0.9)
    thin = abtest.compare_arms(
        tasks.Task.REGRESSION,
        "rmse",
        arm(1),
        arm(2),
        bootstraps=100,
        min_rows=10,
    )
    assert thin.dropped > 50
    assert numpy.isnan(thin.interval[0]) and numpy.isnan(thin.interval[1])
    assert thin.decision == "inconclusive"
    assert thin.to_dict()["dropped"] == thin.dropped


def test_an_unreadable_coverage_report_is_named_not_misreported(tmp_path):
    report = tmp_path / "coverage_report.json"
    report.write_text("{not json", encoding="utf-8")
    problems = claims_audit.coverage_report_problems(report, None)
    assert len(problems) == 1 and "unreadable" in problems[0]
    problems = claims_audit.coverage_report_problems(report, "a" * 64)
    assert "unreadable" in problems[0]


class Bundle:
    """The parts of a bundle `log_predictions` reads."""

    schema = types.SimpleNamespace(task="regression")
    info = types.SimpleNamespace(model_version="v1")

    def prepare_frame(self, data):
        return data

    def predict(self, data):
        return numpy.zeros(len(data))


def fingerprints(log, monkeypatch):
    """Captures the input fingerprint of every observability event."""
    seen = []

    def capture(bundle, request_id, started, outcome, **fields):
        seen.append(fields.get("input_fingerprint"))

    monkeypatch.setattr(log, "log_observability", capture)
    return seen


def test_a_fallback_fingerprint_names_its_scheme(tmp_path, monkeypatch):
    frame = pandas.DataFrame({"a": [1.0, 2.0]})
    log = events.PredictionLog(tmp_path / "log.jsonl")
    seen = fingerprints(log, monkeypatch)
    log.log_predictions(Bundle(), frame, ["r1", "r2"])
    assert seen and not seen[0].startswith("json:")

    def refuse(*args, **kwargs):
        raise TypeError("unhashable cell")

    monkeypatch.setattr(pandas.util, "hash_pandas_object", refuse)
    seen.clear()
    log.log_predictions(Bundle(), frame, ["r3", "r4"])
    assert seen and all(f.startswith("json:") for f in seen)
