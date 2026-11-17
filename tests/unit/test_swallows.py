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
    real = abtest.tally
    calls = {"n": 0}

    def score(chosen, rows, metric, task):
        calls["n"] += 1
        full = len(rows) == len(chosen.outcomes) and calls["n"] <= 2
        if not full and (calls["n"] % 100) < share * 100:
            raise ValueError("a resample missing a class")
        return real(chosen, rows, metric, task)

    monkeypatch.setattr(abtest, "tally", score)


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
    """The parts of a bundle `chronicle` reads."""

    schema = types.SimpleNamespace(task="regression")
    info = types.SimpleNamespace(model_version="v1")

    def prepare(self, data):
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
    log.chronicle(Bundle(), frame, ["r1", "r2"])
    assert seen and not seen[0].startswith("json:")

    def refuse(*args, **kwargs):
        raise TypeError("unhashable cell")

    monkeypatch.setattr(pandas.util, "hash_pandas_object", refuse)
    seen.clear()
    log.chronicle(Bundle(), frame, ["r3", "r4"])
    assert seen and all(f.startswith("json:") for f in seen)


def test_an_unreadable_source_file_is_not_hashed_as_no_file(
    tmp_path, monkeypatch
):
    # ERR-14: an unreadable file hashed to "", the value for "not a file".
    from scmle.contracts import errors
    from scmle.data.readers import tabular

    source = tmp_path / "data.csv"
    source.write_text("a\n1\n", encoding="utf-8")
    assert len(tabular.raw_source_hash(source)) == 64

    def refuse(self, *args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(type(source), "open", refuse)
    import pytest

    with pytest.raises(errors.DataError, match="cannot read"):
        tabular.raw_source_hash(source)
    assert tabular.raw_source_hash("memory://frame") == ""


def test_an_undefined_metric_says_why_it_is_nan(caplog):
    # ERR-14: a metric that could not be computed became NaN with no reason.
    from sklearn.linear_model import LogisticRegression

    from scmle.evaluation import metrics

    model = LogisticRegression().fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])
    features = pandas.DataFrame({"x": [0.0, 1.0, 2.0]})
    target = pandas.Series([0, 0, 0])
    with caplog.at_level("WARNING", logger=metrics.__name__):
        scores = metrics.CATALOG.evaluate(
            model, features, target, tasks.Task.BINARY
        )
    undefined = [name for name, value in scores.items() if numpy.isnan(value)]
    assert undefined
    assert any("is undefined here" in r.getMessage() for r in caplog.records)


def test_champion_counts_every_dropped_resample(monkeypatch):
    # ERR-13: champion resamples that failed were dropped uncounted, and when
    # every one failed the percentile of nothing raised IndexError.
    from scmle.evaluation import metrics
    from scmle.monitoring import champion

    calls = {"n": 0}

    def score(model, features, target, positive):
        calls["n"] += 1
        if calls["n"] > 2:
            raise ValueError("a resample missing a class")
        return 0.5

    metric = types.SimpleNamespace(
        name="rmse", greater_is_better=False, score=score
    )
    monkeypatch.setattr(metrics.CATALOG, "resolve", lambda name, task: metric)
    model = types.SimpleNamespace(
        schema=types.SimpleNamespace(
            task="regression", group_column=None, time_column=None
        ),
        info=types.SimpleNamespace(metric="rmse"),
        pipeline=None,
        prepare=lambda frame: frame,
    )
    rows = champion.MIN_ROWS + 10
    frame = pandas.DataFrame({"x": numpy.arange(rows, dtype=float)})
    outcome = champion.Referee(model, model, bootstraps=40).compare(
        frame, pandas.Series(numpy.zeros(rows))
    )
    assert outcome.dropped == 40
    assert numpy.isnan(outcome.interval[0])
    assert outcome.decision == "keep"


def test_a_failed_attestation_check_keeps_its_reason(monkeypatch, caplog):
    # ERR-4: the reason a release attestation could not be checked was
    # discarded; the gate blocked with no word on why.
    from scmle.evaluation import policies
    from scmle.evaluation import release
    from scmle.pipeline import stages

    policy = types.SimpleNamespace(metrics=(), to_dict=dict)
    index = types.SimpleNamespace(
        sign=lambda key: None, verify=lambda key: False, to_dict=dict
    )
    monkeypatch.setattr(policies, "derive", lambda *args: policy)
    monkeypatch.setattr(
        stages.PackageStage, "evidence_index", lambda *args: index
    )
    verdict = types.SimpleNamespace(to_dict=dict)
    monkeypatch.setattr(release, "assess", lambda *args: verdict)
    monkeypatch.setattr(release, "faults", lambda outcome, plan: outcome)
    monkeypatch.setattr(release, "certify", lambda outcome, *args: outcome)
    monkeypatch.setattr(release, "countersign", lambda outcome, *args: outcome)
    context = types.SimpleNamespace(
        task=tasks.Task.REGRESSION,
        metric=types.SimpleNamespace(name="rmse", greater_is_better=False),
        spec=types.SimpleNamespace(
            problem=types.SimpleNamespace(success_criteria={"rmse": 1.0})
        ),
        diagnostics={},
    )
    store = types.SimpleNamespace(write_json=lambda name, document: None)
    evaluation = {"test_metrics": {}}
    with caplog.at_level("WARNING", logger=stages.__name__):
        stages.PackageStage().release_evaluation(context, store, evaluation)
    assert any(
        "release attestation" in r.getMessage()
        and "release_attestation" in r.getMessage()
        for r in caplog.records
    )


def test_a_malformed_policy_amendment_is_refused_not_applied(tmp_path):
    # ERR-23: an amendment that was not a mapping was applied by
    # `dict.update` (or failed with a raw error) instead of being refused.
    import pytest

    from scmle.contracts import errors
    from scmle.governance import deployment_ledger
    from scmle.governance import loop

    policy = loop.Preregistration(
        reference_risk=0.1,
        tolerance=0.05,
        r_star=0.6,
        initial_tau=0.5,
        min_matured=5,
        outcome_window=0,
        scope="m:s",
    )
    charter = types.SimpleNamespace(charter_hash="a" * 64)
    opened = loop.Loop.open(tmp_path / "deploy", charter, policy)
    opened.ledger.commit(
        deployment_ledger.OBSERVE_SUMMARY,
        {"policy_amendment": [["min_matured", 7]]},
    )
    with pytest.raises(errors.ArtifactError, match="policy amendment"):
        loop.Loop.resume(tmp_path / "deploy")
    with pytest.raises(errors.ArtifactError, match="policy amendment"):
        loop.Loop.open(tmp_path / "deploy", charter, policy)
