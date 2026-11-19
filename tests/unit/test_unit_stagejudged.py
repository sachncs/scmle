"""The search, selection and final evaluation return frozen products.

Each stage runs on fake services (no run directory, a governor with a fixed
number of seconds left) and returns its product without setting the context;
the product's `apply` sets the context fields the later, unconverted stages
still read.
"""

import dataclasses
import json
import threading
import types

import numpy
import pandas
import pytest
from sklearn import linear_model

from scmle.contracts import experiments
from scmle.contracts import tasks
import scmle.decision.threshold
from scmle.models import thresholds
from scmle.pipeline import ensemble_search
from scmle.pipeline import journal
from scmle.pipeline import products
from scmle.pipeline import strategy_search
from scmle.pipeline import trials
from scmle.pipeline import tuning
from scmle.pipeline.stages import assessment
from scmle.pipeline.stages import examination
from scmle.pipeline.stages import profiling
from scmle.pipeline.stages import selecting
from scmle.pipeline.stages import training


def trial(name, score=0.5, seconds=1.0):
    return experiments.Trial(
        f"t-{name}",
        name,
        experiments.Termination.OK,
        "accuracy",
        score,
        0.0,
        [score] * 5,
        seconds,
        params={},
        strategy="default",
    )


class Governor:
    """Leaves `seconds` before the finish and has its reserve provisioned."""

    def __init__(self, seconds=1000.0):
        self.seconds = seconds
        self.reserve = types.SimpleNamespace(shape=(1, 1), parts={})

    def overdue(self, seconds, keep=0.0):
        left = self.seconds - keep
        return f"needs {seconds} s" if seconds > left else None

    def left(self, phase):
        return self.seconds


def fake(ranked=(), pareto=False, unlabeled="none"):
    """Fake services: the spec's sections the stages read, and no store."""
    spec = types.SimpleNamespace(
        selection=types.SimpleNamespace(
            max_model_mb=None,
            max_p95_latency_ms=None,
            latency_safety_factor=1.0,
            pareto=pareto,
            max_candidates_checked=3,
            pareto_tolerance=0.0,
        ),
        data=types.SimpleNamespace(
            baseline_column=None,
            baseline_subject=None,
            unlabeled=unlabeled,
        ),
        validation=types.SimpleNamespace(seed=0),
    )
    return types.SimpleNamespace(
        spec=spec,
        governor=Governor(),
        policy=types.SimpleNamespace(
            description="best mean", rank=lambda *args: list(ranked)
        ),
        registry=None,
        check=lambda: None,
        elapsed=lambda: 0.0,
    )


def board(given, **fields):
    """A context stand-in: the services, an unpersisted journal, products."""
    return types.SimpleNamespace(
        services=given,
        journal=journal.Journal(given.spec, None, None, None),
        **fields,
    )


class Runner:
    """Runs a chunk of candidates instantly, each as an OK trial."""

    workers = 2

    def __init__(self, context):
        self.context = context

    def run_many(self, jobs, first):
        return [
            dataclasses.replace(trial(job.model_name), trial_id=job.trial_id)
            for job in jobs
        ]


class Idle:
    """A search after the candidates that adds nothing."""

    def __init__(self, context, runner):
        self.context = context

    def search(self, done):
        return None

    def tune(self, done):
        return None


@pytest.fixture
def searched(monkeypatch):
    monkeypatch.setattr(trials, "Runner", Runner)
    monkeypatch.setattr(strategy_search, "Strategist", Idle)
    monkeypatch.setattr(tuning, "Tuner", Idle)
    monkeypatch.setattr(ensemble_search, "Stacker", Idle)
    context = board(
        fake(), plan=types.SimpleNamespace(candidates=["a", "b", "c"])
    )
    return context, training.TrainStage().execute(context)


def test_the_search_is_a_copy_of_the_journal_after_training(searched):
    context, made = searched
    assert isinstance(made, training.Search)
    assert [t.trial_id for t in made.trials] == [
        "trial_001",
        "trial_002",
        "trial_003",
    ]
    assert made.trials == context.journal.trials
    assert made.trials is not context.journal.trials
    assert made.trial_counter == context.journal.trial_counter == 3
    assert made.outcomes == [] and made.skipped == []
    context.journal.forgo("tuning", "late")
    assert made.skipped == []


def test_the_search_is_frozen(searched):
    with pytest.raises(dataclasses.FrozenInstanceError):
        searched[1].trial_counter = 0


def test_the_selection_returns_its_choice_and_sets_nothing():
    ranked = [trial("best", 0.9), trial("next", 0.8)]
    context = board(
        fake(ranked),
        split=types.SimpleNamespace(
            manifest=types.SimpleNamespace(folds=[([0], [1])] * 5)
        ),
        metric=types.SimpleNamespace(greater_is_better=True),
        refits={},
        pending_ledger=None,
    )
    before = dict(vars(context))
    made = selecting.SelectStage().execute(context)
    assert made == selecting.Choice(ranked[0], {}, None, None, [], [])
    assert vars(context) == before
    made.apply(context)
    assert context.selected is ranked[0]
    assert context.pareto_front is None


@dataclasses.dataclass(frozen=True)
class Escrow:
    holds_for_coverage: bool
    subject: object = None


@pytest.mark.parametrize("coverage", [True, False])
def test_a_pending_confirmation_waits_or_concludes(monkeypatch, coverage):
    concluded = []
    monkeypatch.setattr(
        profiling.ProfileStage,
        "conclude",
        lambda self, context, pending, subject: concluded.append(pending),
    )
    pending = Escrow(coverage, subject="stale")
    context = board(
        fake([trial("best")]),
        split=types.SimpleNamespace(
            manifest=types.SimpleNamespace(folds=[([0], [1])] * 5)
        ),
        metric=types.SimpleNamespace(greater_is_better=True),
        refits={},
        pending_ledger=pending,
    )
    made = selecting.SelectStage().execute(context)
    if coverage:
        assert made.pending_ledger == Escrow(True, None)
        assert concluded == []
    else:
        assert made.pending_ledger is None
        assert concluded == [pending]
    assert context.pending_ledger is pending


@pytest.mark.parametrize(
    ("part", "name"),
    [
        (examination.Pseudolabels({"used": True}), "semi_supervised"),
        (examination.Remedy({"applied": False}), "fairness_mitigation"),
        (
            examination.Decision({"threshold": 0.4}, None, "abc"),
            "coverage_claim_sha256",
        ),
    ],
)
def test_each_part_fills_its_context_field_and_reads_it_back(part, name):
    context = types.SimpleNamespace()
    part.apply(context)
    assert list(vars(context)) == [name]
    captured = type(part).capture(context)
    assert isinstance(captured, products.Derived)
    if isinstance(part, examination.Decision):
        assert captured == examination.Decision(claim="abc")
    else:
        assert captured == part


def test_the_readings_fill_the_reference_scores_and_diagnostics():
    scores = pandas.Series([0.2, 0.8])
    made = examination.Readings(scores, {"calibration": {"ece": 0.1}})
    context = types.SimpleNamespace()
    made.apply(context)
    assert sorted(vars(context)) == ["diagnostics", "reference_scores"]
    assert context.reference_scores is scores
    again = examination.Readings.capture(context)
    assert again.scores is scores
    assert again.diagnostics == made.diagnostics
    assert examination.Readings().diagnostics == {}


def test_the_exam_applies_its_readings():
    readings = examination.Readings(None, {"fairness": {}})
    context = types.SimpleNamespace()
    exam(readings=readings).apply(context)
    assert context.diagnostics == {"fairness": {}}
    assert context.reference_scores is None
    assert assessment.Exam.capture(context).readings == readings


def test_a_part_is_digested_by_its_fields(tmp_path):
    path = tmp_path / "part.product"
    first = examination.Remedy({"applied": True}).save(path)
    assert first == examination.Remedy({"applied": True}).save(path)
    assert first != examination.Remedy({"applied": False}).save(path)
    assert not path.exists()


def exam(**parts):
    features = pandas.DataFrame({"a": [1.0, 2.0]})
    return assessment.Exam(
        {"metrics": {"accuracy": 1.0}},
        "fitted",
        "served",
        (features, pandas.Series([0, 1])),
        **parts,
    )


def test_the_exam_applies_its_fields_and_each_part():
    made = exam(
        pseudolabels=examination.Pseudolabels({"used": False}),
        decision=examination.Decision(claim="abc"),
    )
    context = types.SimpleNamespace()
    made.apply(context)
    assert context.final_pipeline == "fitted"
    assert context.shipped == "served"
    assert context.semi_supervised == {"used": False}
    assert context.fairness_mitigation is None
    assert context.coverage_claim_sha256 == "abc"
    assert not hasattr(context, "pseudolabels")
    again = assessment.Exam.capture(context)
    assert again.pseudolabels == made.pseudolabels
    assert again.decision == examination.Decision(claim="abc")


def test_the_exam_digest_is_the_evaluation_and_the_opening(tmp_path):
    expected = products.digest(
        json.dumps(
            {"evaluation": {"metrics": {"accuracy": 1.0}}, "opened": True},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    path = tmp_path / "exam.product"
    assert exam().save(path) == expected
    assert exam(remedy=examination.Remedy({"applied": True})).save(path) == (
        expected
    )


class Segmented:
    """A group-threshold model that predicts one class for group `x`."""

    classes_ = numpy.array([0, 1])
    assigner = types.SimpleNamespace(
        assign=lambda column: pandas.Series(column.to_numpy())
    )

    def __init__(self, flip):
        self.flip = flip

    def predict(self, features):
        return numpy.where(features["g"] == "x", 1 - self.flip, self.flip)


def test_corroborating_returns_a_new_remedy_with_the_released_rows():
    record = {
        "applied": True,
        "column": "g",
        "criterion": "demographic_parity",
    }
    remedy = examination.Remedy(record)
    features = pandas.DataFrame({"g": ["x", "x", "y", "y"]})
    target = pandas.Series([1, 0, 1, 0])
    stage = assessment.FinalEvaluationStage()
    made = stage.corroborate(
        remedy, Segmented(0), Segmented(1), features, target
    )
    assert remedy.record == record
    assert list(made.record) == [*record, "locked_rows"]
    assert made.record["locked_rows"]["rows"] == len(features)
    skipped = examination.Remedy({"applied": False})
    assert stage.corroborate(skipped, None, None, features, target) is skipped
    assert stage.corroborate(
        examination.Remedy(), None, None, features, target
    ) == (examination.Remedy())


def test_no_pseudolabels_are_judged_unless_asked():
    context = board(fake(unlabeled="none"), unlabeled=pandas.DataFrame())
    stage = assessment.FinalEvaluationStage()
    made = stage.pseudolabel(context, trial("m"), "fitted")
    assert made == ("fitted", examination.Pseudolabels())


def development(rows=200):
    rng = numpy.random.default_rng(0)
    features = pandas.DataFrame({"a": rng.normal(size=rows)})
    target = pandas.Series(
        (features.a + rng.normal(scale=1.0, size=rows) > 0.5).astype(int)
    )
    positions = numpy.arange(rows)
    folds = [
        (positions[positions % 4 != k], positions[positions % 4 == k])
        for k in range(4)
    ]
    return features, target, folds


@pytest.mark.parametrize("objective", ["f1", "cost"])
def test_the_threshold_is_chosen_on_development_rows_only(objective):
    features, target, folds = development()
    fitted = linear_model.LogisticRegression().fit(features, target)
    settings = tasks.Threshold(
        threshold_objective=objective,
        false_positive_cost=1.0,
        false_negative_cost=4.0,
    )
    built = []

    def build():
        built.append(1)
        return linear_model.LogisticRegression()

    wrapped, description = scmle.decision.threshold.choose(
        fitted, build, (features, target), folds, settings, None, 0
    )
    assert len(built) == len(folds)
    probabilities, positions = thresholds.heldout(
        linear_model.LogisticRegression, features, target, folds
    )
    kind = thresholds.OBJECTIVES[objective]
    measure = kind(1.0, 4.0) if objective == "cost" else kind()
    cut, score = thresholds.argmax(
        probabilities, (target.iloc[positions] == 1).to_numpy(), measure
    )
    assert description == {
        "objective": objective,
        "threshold": cut,
        "validation_score": score,
        "validation_rows": len(features),
    }
    assert isinstance(wrapped, thresholds.Classifier)


def test_the_journal_copy_holds_the_lock():
    given = fake()
    record = journal.Journal(given.spec, None, None, None)
    held = threading.Event()
    release = threading.Event()

    def hold():
        with record.lock:
            held.set()
            release.wait(5)

    worker = threading.Thread(target=hold)
    worker.start()
    held.wait(5)
    copied = []
    reader = threading.Thread(
        target=lambda: copied.append(training.Search.copy(record))
    )
    reader.start()
    reader.join(0.2)
    assert copied == []
    release.set()
    worker.join()
    reader.join(5)
    assert copied[0].trial_counter == 0
