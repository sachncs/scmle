"""The data-strategy search: its leaders, skips, jobs and verdicts."""

from types import SimpleNamespace

import pytest

from scmle.contracts import experiments
from scmle.contracts import tasks
from scmle.pipeline import strategy_search
from scmle.preparation import strategies


def trial(model, score, **changes) -> experiments.Trial:
    """A screened trial of `model` with mean score `score`."""
    return experiments.Trial(
        changes.pop("trial_id", f"{model}-trial"),
        model,
        changes.pop("status", experiments.Termination.OK),
        "accuracy",
        score,
        0.0,
        [score],
        **changes,
    )


class Registry:
    """Adapters by name, each with a family."""

    def __init__(self, families: dict[str, str]) -> None:
        self.adapters = {
            name: SimpleNamespace(name=name, family=family)
            for name, family in families.items()
        }

    def get(self, name: str) -> SimpleNamespace:
        return self.adapters[name]


class Context:
    """Records every verdict, skip, allocation and transcription.

    It is its own services and journal, so it answers both.
    """

    def __init__(self, names, top=2, better=True, reasons=None) -> None:
        self.spec = SimpleNamespace(
            effective=lambda: SimpleNamespace(names=names, top_models=top)
        )
        self.registry = Registry(
            {"dummy": "baseline", "a": "linear", "b": "tree", "c": "tree"}
        )
        pending = list(reasons or [])
        self.governor = SimpleNamespace(
            skip_reason=lambda first: pending.pop(0) if pending else None
        )
        self.metric = SimpleNamespace(name="accuracy", greater_is_better=better)
        self.task = tasks.Task.BINARY
        self.profile = SimpleNamespace(rows=10)
        self.entries: list[tuple[str, str, str]] = []
        self.forgone: list[tuple[str, str]] = []
        self.counts: list[int] = []
        self.transcribed: list[experiments.Trial] = []
        self.services = self
        self.journal = self

    def enter(self, technique: str, edict: str, reason: str) -> None:
        self.entries.append((technique, edict, reason))

    def forgo(self, work: str, reason: str) -> None:
        self.forgone.append((work, reason))

    def allocate(self, count: int) -> list[str]:
        self.counts.append(count)
        return [f"s{len(self.counts)}-{index}" for index in range(count)]

    def transcribe(self, outcome: experiments.Trial) -> None:
        self.transcribed.append(outcome)

    def check(self) -> None:
        return None


class Runner:
    """Scores each strategy job from a table; records the jobs."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.jobs: list = []

    def run_many(self, jobs) -> list[experiments.Trial]:
        self.jobs.extend(jobs)
        return [
            trial(
                job.model_name,
                self.scores[job.strategy],
                trial_id=job.trial_id,
                strategy=job.strategy,
            )
            for job in jobs
        ]


class Strategy:
    """Applicable unless named in `refused`; records what it was asked."""

    def __init__(self, name: str, refused: dict[str, str], seen: list) -> None:
        self.name = name
        self.refused = refused
        self.seen = seen

    def applicability(self, profile, task, adapter) -> str | None:
        self.seen.append((self.name, profile, task, adapter))
        return self.refused.get(self.name)


@pytest.fixture(name="seen")
def lookup(monkeypatch) -> list:
    """Strategy `x` is not applicable; every other one is."""
    seen: list = []
    monkeypatch.setattr(
        strategies,
        "lookup",
        lambda name: Strategy(name, {"x": "no minority class"}, seen),
    )
    return seen


SCREENED = [
    trial("dummy", 0.99),
    trial("a", 0.8),
    trial("b", 0.7),
    trial("c", 0.9),
    trial("b", 0.95, trial_id="tuned", params={"depth": 3}),
    trial("b", 0.97, trial_id="weighted", strategy="weights"),
    trial("a", 0.98, trial_id="broken", status=experiments.Termination.FAILED),
]


def test_the_best_default_non_baseline_trials_lead(seen) -> None:
    context = Context(["w"])
    runner = Runner({"w": 0.5})
    strategy_search.Strategist(context, runner).search(SCREENED)
    assert [job.model_name for job in runner.jobs] == ["c", "a"]


def test_a_lower_is_better_metric_leads_with_the_lowest(seen) -> None:
    context = Context(["w"], better=False)
    runner = Runner({"w": 0.5})
    strategy_search.Strategist(context, runner).search(SCREENED)
    assert [job.model_name for job in runner.jobs] == ["b", "a"]


def test_no_strategies_or_no_leaders_search_nothing(seen) -> None:
    for context in (Context([]), Context(["w"], top=0)):
        runner = Runner({"w": 0.5})
        strategy_search.Strategist(context, runner).search(SCREENED)
        assert runner.jobs == [] and context.entries == []


def test_a_skipped_leader_is_forgone_and_the_next_still_runs(seen) -> None:
    context = Context(["w"], reasons=["trial window closed"])
    runner = Runner({"w": 0.5})
    strategy_search.Strategist(context, runner).search(SCREENED)
    assert context.forgone == [("strategy search for c", "trial window closed")]
    assert [job.model_name for job in runner.jobs] == ["a"]


def test_an_inapplicable_strategy_is_rejected_and_not_run(seen) -> None:
    context = Context(["w", "x", "y"], top=1)
    runner = Runner({"w": 0.95, "y": 0.85})
    strategy_search.Strategist(context, runner).search(SCREENED)
    adapter = context.services.registry.adapters["c"]
    assert seen == [
        (name, context.profile, tasks.Task.BINARY, adapter)
        for name in ("w", "x", "y")
    ]
    assert context.counts == [2]
    assert [
        (job.trial_id, job.model_name, dict(job.params), job.strategy)
        for job in runner.jobs
    ] == [("s1-0", "c", {}, "w"), ("s1-1", "c", {}, "y")]
    assert [t.trial_id for t in context.transcribed] == ["s1-0", "s1-1"]
    assert context.entries == [
        ("x@c", experiments.Edict.REJECT, "not applicable: no minority class"),
        (
            "w@c",
            experiments.Edict.INCLUDE,
            (
                "cv accuracy 0.9500 vs default 0.9000 "
                "(+0.0500 in the metric's favour)"
            ),
        ),
        (
            "y@c",
            experiments.Edict.REJECT,
            (
                "cv accuracy 0.8500 vs default 0.9000 "
                "(-0.0500 in the metric's favour)"
            ),
        ),
    ]


def test_a_lower_score_is_a_gain_when_lower_is_better(seen) -> None:
    context = Context(["w"], better=False)
    strategist = strategy_search.Strategist(context, Runner({}))
    strategist.judge(trial("c", 0.4), trial("c", 0.3, strategy="w"))
    strategist.judge(trial("c", 0.4), trial("c", 0.5, strategy="v"))
    assert context.entries == [
        (
            "w@c",
            experiments.Edict.INCLUDE,
            (
                "cv accuracy 0.3000 vs default 0.4000 "
                "(+0.1000 in the metric's favour)"
            ),
        ),
        (
            "v@c",
            experiments.Edict.REJECT,
            (
                "cv accuracy 0.5000 vs default 0.4000 "
                "(-0.1000 in the metric's favour)"
            ),
        ),
    ]


def test_a_failed_strategy_trial_is_rejected_with_its_error(seen) -> None:
    context = Context(["w"])
    failed = trial(
        "c",
        0.0,
        strategy="w",
        status=experiments.Termination.FAILED,
        error="boom",
    )
    strategy_search.Strategist(context, Runner({})).judge(
        trial("c", 0.9), failed
    )
    assert context.entries == [
        ("w@c", experiments.Edict.REJECT, "trial failed: boom")
    ]
