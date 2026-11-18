"""The run context, its journal, the planning rules and the search steps."""

import json
import types

import numpy
import pandas
import pytest

import scmle.api.project
import scmle.artifacts.store
from scmle.contracts import errors
from scmle.contracts import events
from scmle.contracts import experiments
from scmle.contracts import tasks
from scmle.pipeline import ensemble_search
from scmle.pipeline import hyperband
from scmle.pipeline import journal
from scmle.pipeline import rules
import scmle.pipeline.context


def config(budget=1800, **data):
    return tasks.Config(
        data=tasks.Dataset(uri="x.csv", target="y", **data),
        resources=tasks.Resources(time_budget_seconds=budget),
    )


def trial(
    trial_id,
    model="m",
    score=0.5,
    status=experiments.Termination.OK,
    strategy="default",
):
    return experiments.Trial(
        trial_id,
        model,
        status,
        "accuracy",
        score,
        0.0,
        [score],
        1.0,
        params={"id": trial_id},
        strategy=strategy,
    )


class Board(journal.Journal):
    """A journal with a real store, event log and a recording tracker."""

    def __init__(self, root=None):
        self.spec = config()
        self.trials = []
        self.outcomes = []
        self.trial_counter = 0
        self.skipped = []
        self.tracked = []
        self.revised = []
        self.tracker = types.SimpleNamespace(
            transcribe=lambda run, outcome: self.tracked.append(
                (run, outcome.trial_id)
            ),
            revise=lambda run, **fields: self.revised.append((run, fields)),
        )
        if root is None:
            self.store = None
            self.log = None
        else:
            root.mkdir()
            self.store = scmle.artifacts.store.Workspace("r1", root)
            self.log = events.Log(root / "events.jsonl")


def kinds(root):
    lines = (root / "events.jsonl").read_text("utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_require_returns_a_value_and_names_a_missing_one():
    assert scmle.pipeline.context.require(0, "zero") == 0
    with pytest.raises(errors.ScmleError) as caught:
        scmle.pipeline.context.require(None, "split")
    assert (
        str(caught.value) == "'split' is unavailable: its stage has not run yet"
    )


def test_allocate_hands_out_padded_sequential_identifiers():
    board = Board()
    assert board.allocate(2) == ["trial_001", "trial_002"]
    assert board.allocate(1) == ["trial_003"]
    assert board.allocate(0) == []
    assert board.trial_counter == 3


def test_a_preview_journal_writes_nothing(tmp_path):
    board = Board()
    board.write_json("a.json", {"a": 1})
    board.write_text("a.txt", "x")
    board.emit("x")
    board.track(state="done")
    board.transcribe(trial("t1"))
    assert board.tracked == []
    assert board.revised == []
    assert [t.trial_id for t in board.trials] == ["t1"]


def test_forgo_records_and_announces_the_skipped_work(tmp_path):
    board = Board(tmp_path / "run")
    board.forgo("conformal", "needs 9 s")
    assert board.skipped == [{"work": "conformal", "reason": "needs 9 s"}]
    event = kinds(tmp_path / "run")[-1]
    assert event["kind"] == "budget.skipped"
    assert event["reason"] == "conformal skipped or reduced: needs 9 s"
    assert event["data"] == {"work": "conformal", "reason": "needs 9 s"}


def test_transcribe_persists_announces_and_tracks_a_trial(tmp_path):
    board = Board(tmp_path / "run")
    outcome = trial("trial_001", model="ridge")
    board.transcribe(outcome)
    stored = json.loads(
        (tmp_path / "run" / "trials" / "trial_001.json").read_text("utf-8")
    )
    assert stored == json.loads(json.dumps(outcome.to_dict()))
    event = kinds(tmp_path / "run")[-1]
    assert event["kind"] == "trial.ok"
    assert event["reason"] == "ridge"
    assert event["artifacts"] == ["trials/trial_001.json"]
    assert board.tracked == [("r1", "trial_001")]
    with pytest.raises(errors.ScmleError, match="duplicate trial id trial_001"):
        board.transcribe(trial("trial_001"))
    assert len(board.trials) == 1


def test_write_and_track_reach_the_store(tmp_path):
    board = Board(tmp_path / "run")
    board.write_json("a/b.json", {"k": 1})
    board.write_text("c.txt", "hello")
    board.track(state="done")
    assert json.loads((tmp_path / "run" / "a" / "b.json").read_text()) == {
        "k": 1
    }
    assert (tmp_path / "run" / "c.txt").read_text() == "hello"
    assert board.revised == [("r1", {"state": "done"})]


def test_enter_records_a_decision():
    board = Board()
    board.enter("smote", experiments.Edict.REJECT, "no gain")
    assert board.outcomes == [
        experiments.Decision("smote", experiments.Edict.REJECT, "no gain")
    ]


@pytest.fixture(scope="module")
def context():
    frame = pandas.DataFrame({"a": numpy.arange(40.0), "y": [0, 1] * 20})
    made = scmle.api.project.Project(frame, "y").orchestrator.new_context()
    yield made
    made.close()


def test_a_preview_context_has_no_store_and_a_budget_deadline(context):
    assert context.directory == ""
    assert context.deadline == context.started_at + 1800
    with pytest.raises(errors.ScmleError, match="'store' is unavailable"):
        assert context.workspace is None
    assert context.regulator is context.regulator


def test_cancellation_raises_with_the_run_directory(context):
    context.check_cancelled()
    context.cancel_event.set()
    try:
        with pytest.raises(errors.CancelledError) as caught:
            context.check_cancelled()
        assert caught.value.details == {"run_dir": ""}
    finally:
        context.cancel_event.clear()


def test_observe_needs_the_capability_observation(context):
    saved = context.capability_observation
    context.capability_observation = None
    try:
        with pytest.raises(errors.ScmleError, match="capability_observation"):
            context.observe()
    finally:
        context.capability_observation = saved


def situation(spec=None, rows=10, width=2, wide=()):
    plan = types.SimpleNamespace(
        features=tuple(f"f{i}" for i in range(width)), high_cardinality=wide
    )
    profile = types.SimpleNamespace(
        manifest=types.SimpleNamespace(rows=rows), feature_plan=plan
    )
    return rules.Situation(spec or config(), profile, tasks.Task.BINARY)


def adapter(name, family="linear", **capabilities):
    defaults = {
        "memory_factor": 1.0,
        "cost": 1,
        "incremental": False,
        "stream_only": False,
        "serializable": True,
        "native_categorical": False,
    }
    return types.SimpleNamespace(
        name=name,
        family=family,
        capabilities=types.SimpleNamespace(**{**defaults, **capabilities}),
        nominals=lambda columns: columns,
    )


def test_memory_rejects_a_model_over_the_limit():
    rows = 1024**3 // 8
    context = situation(rows=rows, width=1)
    decisions = []
    small = adapter("small", memory_factor=8.0)
    large = adapter("large", memory_factor=8.5)
    kept = rules.Memory().apply(context, [small, large], decisions)
    assert kept == [small]
    assert decisions == [
        experiments.Decision(
            "large",
            experiments.Edict.REJECT,
            "estimated training memory 8.50 GB exceeds the 8 GB limit",
        )
    ]


def test_memory_counts_at_least_one_feature():
    context = situation(rows=1024**3 // 8, width=0)
    assert (
        rules.Memory().apply(context, [adapter("a", memory_factor=9.0)], [])
        == []
    )


def test_budget_keeps_everything_on_a_generous_budget():
    candidates = [adapter("slow", cost=5)]
    decisions = []
    assert rules.Budget().apply(situation(), candidates, decisions) == (
        candidates
    )
    assert decisions == []


def test_budget_keeps_fast_models_and_the_named_exceptions():
    context = situation(config(budget=60))
    cheap = adapter("cheap", cost=2)
    edge = adapter("edge", cost=rules.EXPENSIVE_COST)
    boosted = adapter("hist_gradient_boosting", cost=3)
    decisions = []
    kept = rules.Budget().apply(context, [cheap, edge, boosted], decisions)
    assert kept == [boosted, cheap]
    assert decisions[0].technique == "reduce_search_complexity"
    assert decisions[0].action is experiments.Edict.INCLUDE
    assert decisions[0].reason == (
        "time budget < 300s: tune at most 1 model(s) with 3 config(s) each "
        "and try strategies on at most 1 model(s)"
    )
    assert decisions[1] == experiments.Decision(
        "edge", experiments.Edict.REJECT, "time budget < 300s: fast models only"
    )
    assert len(decisions) == 2


def test_budget_runs_the_boosted_learner_right_after_the_baseline():
    # On a tight budget the trial window can close before the last
    # candidate; the strongest learner must not be the one skipped.
    context = situation(config(budget=60))
    dummy = adapter("dummy", family="baseline")
    tree = adapter("decision_tree")
    near = adapter("knn", cost=2)
    boosted = adapter("hist_gradient_boosting", cost=3)
    kept = rules.Budget().apply(context, [tree, dummy, near, boosted], [])
    assert kept == [dummy, boosted, tree, near]


def test_streaming_rules_depend_on_the_training_mode():
    batch = adapter("batch")
    online = adapter("online", incremental=True)
    only = adapter("only", stream_only=True, incremental=True)
    dummy = adapter("dummy", family="baseline")
    decisions = []
    assert rules.Streaming().apply(
        situation(), [batch, online, only, dummy], decisions
    ) == [batch, online, dummy]
    assert [d.technique for d in decisions] == ["only"]
    assert decisions[0].reason.startswith("built for data larger than memory")
    streamed = situation(config(training="stream", memory_budget_gb=1.0))
    decisions = []
    assert rules.Streaming().apply(
        streamed, [batch, online, only, dummy], decisions
    ) == [online, only, dummy]
    assert decisions == [
        experiments.Decision(
            "batch",
            experiments.Edict.REJECT,
            "cannot learn incrementally, so it cannot train on a stream "
            "larger than memory",
        )
    ]


def test_persistence_rejects_models_that_cannot_be_saved():
    saved = adapter("saved")
    lost = adapter("lost", serializable=False)
    decisions = []
    assert rules.Persistence().apply(situation(), [lost, saved], decisions) == [
        saved
    ]
    assert decisions == [
        experiments.Decision(
            "lost",
            experiments.Edict.REJECT,
            "model cannot be serialized into a bundle",
        )
    ]


def test_cardinality_leaves_narrow_data_alone():
    candidates = [adapter("a"), adapter("b", native_categorical=True)]
    decisions = []
    assert rules.Cardinality().apply(situation(), candidates, decisions) == (
        candidates
    )
    assert decisions == []


def test_cardinality_searches_native_models_first():
    base = adapter("dummy", family="baseline", native_categorical=True)
    plain = adapter("plain")
    native = adapter("native", native_categorical=True)
    decisions = []
    kept = rules.Cardinality().apply(
        situation(wide=("city", "zip")), [plain, native, base], decisions
    )
    assert kept == [base, native, plain]
    assert decisions[0] == experiments.Decision(
        "native_categorical_boosting",
        experiments.Edict.INCLUDE,
        "2 high-cardinality categorical feature(s) (city, zip); "
        "searched first: native",
    )
    assert decisions[1] == experiments.Decision(
        "regularized_target_encoding",
        experiments.Edict.INCLUDE,
        "evaluated as the 'target_encoding' strategy on the original "
        "validation folds",
    )


def test_cardinality_defers_without_a_native_model():
    picky = adapter("picky", native_categorical=True)
    picky.nominals = lambda columns: []
    decisions = []
    kept = rules.Cardinality().apply(
        situation(wide=("city",)), [picky], decisions
    )
    assert kept == [picky]
    assert decisions[0] == experiments.Decision(
        "native_categorical_boosting",
        experiments.Edict.DEFER,
        "no compatible model handles categoricals natively",
    )


def test_exclusion_drops_named_models_and_families():
    spec = config().model_copy(
        update={
            "models": tasks.Candidates(
                exclude_models=["gone"], exclude_families=["tree"]
            )
        }
    )
    named = adapter("gone", family="tree")
    tree = adapter("forest", family="tree")
    kept_one = adapter("ridge")
    decisions = []
    assert rules.Exclusion().apply(
        situation(spec), [named, tree, kept_one], decisions
    ) == [kept_one]
    assert decisions == [
        experiments.Decision(
            "gone",
            experiments.Edict.REJECT,
            "excluded by models.exclude_models",
        ),
        experiments.Decision(
            "forest", experiments.Edict.REJECT, "family 'tree' is excluded"
        ),
    ]


class Optimizer:
    def __init__(self):
        self.count = 0
        self.told = []

    def ask(self):
        self.count += 1
        return {"n": self.count}

    def tell(self, params, score):
        self.told.append((params["n"], score))


class Runner:
    """Scores a configuration by its number; a rung is the fold count."""

    def __init__(self, folds=9, failing=()):
        self.folds = list(range(folds))
        self.failing = set(failing)
        self.calls = []

    def run_many(self, jobs):
        self.calls.append([(job.trial_id, job.fold_count) for job in jobs])
        made = []
        for job in jobs:
            outcome = trial(
                job.trial_id,
                job.model_name,
                float(job.params["n"]),
                experiments.Termination.FAILED
                if job.params["n"] in self.failing
                else experiments.Termination.OK,
            )
            outcome.params = job.params
            made.append(outcome)
        return made


class Tuner(hyperband.Hyperband):
    def __init__(self, folds=9, eta=3, configs=6, failing=(), skip=None):
        self.runner = Runner(folds, failing)
        self.search = types.SimpleNamespace(eta=eta, configs_per_model=configs)
        self.metric = types.SimpleNamespace(greater_is_better=True)
        self.recorded = []
        self.context = types.SimpleNamespace(
            allocate=self.allocate,
            transcribe=self.recorded.append,
            regulator=types.SimpleNamespace(skip_reason=lambda first: skip),
            forgo=lambda work, reason: self.recorded.append((work, reason)),
            check_cancelled=lambda: None,
        )
        self.counter = 0

    def allocate(self, count):
        start = self.counter
        self.counter += count
        return [f"t{start + i + 1}" for i in range(count)]


def test_the_ladder_starts_each_bracket_one_rung_later():
    assert Tuner(folds=9).ladder() == [[1, 3, None], [3, None], [None]]
    assert Tuner(folds=27).ladder() == [
        [1, 3, 9, None],
        [3, 9, None],
        [9, None],
        [None],
    ]
    assert Tuner(folds=2, eta=2).ladder() == [[1, None], [None]]
    assert Tuner(folds=1).ladder() == [[None]]


def test_a_bracket_keeps_the_best_third_at_each_rung():
    tuner = Tuner()
    optimizer = Optimizer()
    tuner.ascend("m", "default", optimizer, 6, [1, 3, None])
    assert tuner.runner.calls == [
        [("t1", 1), ("t2", 1), ("t3", 1), ("t4", 1), ("t5", 1), ("t6", 1)],
        [("t6", 3), ("t5", 3)],
        [("t6", None)],
    ]
    pruned = [
        t for t in tuner.recorded if t.status is experiments.Termination.PRUNED
    ]
    assert [t.trial_id for t in pruned] == ["t1", "t2", "t3", "t4", "t5"]
    assert pruned[0].error == "pruned before the 3 rung"
    assert pruned[-1].error == "pruned before the full rung"
    assert tuner.recorded[-1].trial_id == "t6"
    assert tuner.recorded[-1].status is experiments.Termination.OK
    assert optimizer.told[-1] == (6, 6.0)
    assert len(optimizer.told) == 6


def test_a_failed_survivor_is_recorded_without_telling_the_optimizer():
    tuner = Tuner(failing={1})
    optimizer = Optimizer()
    tuner.ascend("m", "default", optimizer, 1, [None])
    assert [t.status for t in tuner.recorded] == [
        experiments.Termination.FAILED
    ]
    assert optimizer.told == []


def test_lower_is_better_keeps_the_smallest_scores():
    tuner = Tuner()
    tuner.metric.greater_is_better = False
    tuner.ascend("m", "default", Optimizer(), 3, [1, None])
    assert tuner.runner.calls[1] == [("t1", None)]


def test_hyperband_shares_the_configurations_between_brackets():
    tuner = Tuner(configs=6)
    tuner.hyperband("m", "default", Optimizer())
    shape = [(len(call), call[0][1]) for call in tuner.runner.calls]
    assert shape == [(3, 1), (1, 3), (1, None), (2, 3), (1, None), (1, None)]


def test_hyperband_stops_and_records_when_told_to_skip():
    tuner = Tuner(skip="no time")
    tuner.hyperband("m", "default", Optimizer())
    assert tuner.recorded == [("tuning m: 6 configurations", "no time")]
    assert tuner.runner.calls == []


def test_hyperband_brackets_get_at_least_two_configurations():
    tuner = Tuner(configs=3)
    tuner.hyperband("m", "default", Optimizer())
    assert [len(c) for c in tuner.runner.calls if c and c[0][1] == 1] == [2]
    assert hyperband.MIN_BATCH == 2
    assert hyperband.MIN_FOLDS == 2


class Stacking(ensemble_search.Stacker):
    def __init__(self, size=3, skip=None, greater=True, objections=None):
        families = {"dummy": "baseline"}
        objections = objections or {}
        self.entered = []
        self.recorded = []
        self.objected = []
        self.context = types.SimpleNamespace(
            task="the task",
            registry=types.SimpleNamespace(
                get=lambda name: types.SimpleNamespace(
                    family=families.get(name, "x"),
                    objection=lambda task: (
                        self.objected.append((name, task))
                        or objections.get(name)
                    ),
                )
            ),
            spec=types.SimpleNamespace(
                ensemble=types.SimpleNamespace(size=size)
            ),
            regulator=types.SimpleNamespace(skip_reason=lambda first: skip),
            enter=lambda *args: self.entered.append(args),
            allocate=lambda count: ["e1"],
            transcribe=self.recorded.append,
            check_cancelled=lambda: None,
        )
        self.metric = types.SimpleNamespace(
            greater_is_better=greater, name="accuracy"
        )
        self.runner = types.SimpleNamespace(
            run=lambda trial_id, combiner, params: trial(
                trial_id, combiner, 0.9
            )
        )


def test_members_are_the_best_default_trial_of_each_model():
    stacker = Stacking()
    evaluated = [
        trial("1", "a", 0.6),
        trial("2", "a", 0.8),
        trial("3", "b", 0.7),
        trial("4", "c", 0.99, strategy="smote"),
        trial("5", "dummy", 0.95),
        trial("6", "stacking", 0.97),
        trial("7", "d", 0.98, status=experiments.Termination.FAILED),
        trial("8", "e", 0.1),
    ]
    assert [t.trial_id for t in stacker.members(evaluated, 3)] == [
        "2",
        "3",
        "8",
    ]
    assert [t.trial_id for t in stacker.members(evaluated, 2)] == ["2", "3"]
    stacker.metric.greater_is_better = False
    assert [t.trial_id for t in stacker.members(evaluated, 3)] == [
        "8",
        "1",
        "3",
    ]


def test_too_few_members_rejects_both_combiners():
    stacker = Stacking()
    stacker.search([trial("1", "a", 0.6)])
    assert stacker.entered == [
        (
            "stacking",
            experiments.Edict.REJECT,
            "fewer than two eligible distinct models",
        ),
        (
            "blending",
            experiments.Edict.REJECT,
            "fewer than two eligible distinct models",
        ),
    ]


def test_search_does_nothing_for_a_small_size_or_a_skip():
    for stacker in (Stacking(size=1), Stacking(skip="late")):
        stacker.search([trial("1", "a"), trial("2", "b")])
        assert stacker.entered == []
        assert stacker.recorded == []


def test_search_evaluates_and_judges_each_combiner():
    stacker = Stacking()
    stacker.search([trial("1", "a", 0.8), trial("2", "b", 0.7)])
    assert [t.model_name for t in stacker.recorded] == ["stacking", "blending"]
    assert stacker.entered[0] == (
        "stacking",
        experiments.Edict.INCLUDE,
        (
            "stacking of a, b: cv accuracy 0.9000 vs best member 0.8000 "
            "(+0.1000 in the metric's favour)"
        ),
    )
    assert ensemble_search.COMBINERS == ("stacking", "blending")


def test_search_rejects_a_combiner_its_adapter_objects_to_without_running_it():
    stacker = Stacking(objections={"stacking": "needs probabilities"})
    stacker.search([trial("1", "a", 0.8), trial("2", "b", 0.7)])
    assert stacker.objected == [
        ("stacking", "the task"),
        ("blending", "the task"),
    ]
    assert stacker.entered[0] == (
        "stacking",
        experiments.Edict.REJECT,
        "needs probabilities",
    )
    assert [t.model_name for t in stacker.recorded] == ["blending"]


def test_search_without_a_task_raises_before_any_combiner_runs():
    stacker = Stacking()
    stacker.context.task = None
    with pytest.raises(errors.ScmleError, match="'task' is unavailable"):
        stacker.search([trial("1", "a", 0.8), trial("2", "b", 0.7)])
    assert stacker.recorded == []


def test_judge_rejects_a_tie_a_loss_and_a_failure():
    stacker = Stacking()
    leader = trial("1", "a", 0.9)
    stacker.judge("stacking", leader, trial("e", "stacking", 0.9), ["a"])
    assert stacker.entered[-1][1] is experiments.Edict.REJECT
    assert stacker.entered[-1][2].endswith("(+0.0000 in the metric's favour)")
    failed = trial("e", "stacking", 0.9, status=experiments.Termination.FAILED)
    failed.error = "boom"
    stacker.judge("stacking", leader, failed, ["a"])
    assert stacker.entered[-1] == (
        "stacking",
        experiments.Edict.REJECT,
        "trial failed: boom",
    )
    stacker.metric.greater_is_better = False
    stacker.judge("blending", leader, trial("e", "blending", 0.5), ["a"])
    assert stacker.entered[-1][1] is experiments.Edict.INCLUDE
    assert "(+0.4000 in the metric's favour)" in stacker.entered[-1][2]
