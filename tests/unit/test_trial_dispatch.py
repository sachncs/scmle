"""Tests that every trial batch gets its ceiling when it is dispatched."""

from types import SimpleNamespace

from scmle.contracts import experiments
from scmle.contracts import tasks
from scmle.pipeline import trials
from scmle.runtime import limits


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class Backend:
    """A one-worker backend whose every item takes ten seconds."""

    workers = 1

    def __init__(self, clock):
        self.clock = clock
        self.ceilings = []

    def map(self, function, items, on_failure=None):
        outputs = []
        for item in items:
            self.ceilings.append(item.ceiling_seconds)
            self.clock.now += 10.0
            outputs.append(
                experiments.Trial(
                    item.job.trial_id,
                    item.job.model_name,
                    experiments.Termination.OK,
                    "accuracy",
                    0.5,
                    0.0,
                    [0.5],
                )
            )
        return outputs


def runner(monkeypatch, budget, reserve):
    clock = Clock()
    monkeypatch.setattr(limits.time, "monotonic", clock)
    governor = limits.Governor(
        tasks.Resources(time_budget_seconds=budget), started_at=clock.now
    )
    governor.reserve.parts = {"refit": reserve}
    backend = Backend(clock)
    context = SimpleNamespace(
        regulator=governor,
        backend=backend,
        checkpoints={},
        check_cancelled=lambda: None,
    )
    built = trials.Runner.__new__(trials.Runner)
    built.context = context
    built.folds = [([0], [1])]
    built.metric = SimpleNamespace(name="accuracy")
    monkeypatch.setattr(
        trials.Runner,
        "payload",
        lambda self, job: SimpleNamespace(
            job=job, ceiling_seconds=governor.deadline()
        ),
    )
    return built, backend


def test_each_job_gets_the_window_left_when_it_is_dispatched(monkeypatch):
    built, backend = runner(monkeypatch, budget=45, reserve=10)
    jobs = [
        trials.Job(f"t{index}", "dummy", {}, None, "smote")
        for index in range(5)
    ]
    results = built.run_many(jobs)
    assert backend.ceilings == [35.0, 25.0, 15.0, 5.0]
    assert [r.status for r in results][:4] == [experiments.Termination.OK] * 4
    skipped = results[4]
    assert skipped.status is experiments.Termination.SKIPPED
    assert "trial window" in skipped.error
    assert skipped.strategy == "smote" and skipped.trial_id == "t4"


def test_a_strategy_search_the_window_cannot_hold_is_skipped_on_record():
    from scmle.pipeline import strategy_search

    governor = limits.Governor(
        tasks.Resources(time_budget_seconds=10), started_at=0.0
    )
    recorded = []
    context = SimpleNamespace(
        spec=SimpleNamespace(
            effective=lambda: SimpleNamespace(names=["smote"], top_models=1)
        ),
        registry=SimpleNamespace(adapters={}),
        regulator=governor,
        metric=SimpleNamespace(greater_is_better=True),
        task=tasks.Task.BINARY,
        profile=object(),
        forgo=lambda work, reason: recorded.append((work, reason)),
    )
    leader = experiments.Trial(
        "t1", "knn", experiments.Termination.OK, "accuracy", 0.9, 0.0, [0.9]
    )
    strategy_search.Strategist(context, None).search([leader])
    assert recorded and recorded[0][0] == "strategy search for knn"
    assert "trial window" in recorded[0][1]
