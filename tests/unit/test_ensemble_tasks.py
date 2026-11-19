"""Ensembles are only tried on tasks their combiners support.

OpenML 40691 (wine quality, declared ordinal) failed at packaging with
"this model does not provide probabilities": the ensemble search stacked
the ordinal members as a plain regression, the stack beat its best member
in cross-validation, and the bundle of an ordinal model must give class
probabilities.
"""

import types

from scmle.contracts import experiments
from scmle.contracts import tasks
from scmle.models import ensembles
from scmle.models import registry
from scmle.pipeline import ensemble_search


def trial(trial_id, model, score):
    return experiments.Trial(
        trial_id,
        model,
        experiments.Termination.OK,
        "m",
        score,
        0.0,
        [score],
        1.0,
        params={},
        strategy=experiments.DEFAULT_STRATEGY,
    )


class Search(ensemble_search.Stacker):
    def __init__(self, task):
        self.entered = []
        self.ran = []
        self.task = task
        self.board = types.SimpleNamespace(
            services=types.SimpleNamespace(
                registry=registry.Catalog.stock(include_user=False),
                spec=types.SimpleNamespace(
                    ensemble=types.SimpleNamespace(size=3)
                ),
                governor=types.SimpleNamespace(skip_reason=lambda first: None),
                check=lambda: None,
            ),
            journal=types.SimpleNamespace(
                enter=lambda *args: self.entered.append(args),
                allocate=lambda count: ["e1"],
                transcribe=lambda outcome: None,
            ),
        )
        self.metric = types.SimpleNamespace(greater_is_better=True, name="m")
        self.runner = types.SimpleNamespace(run=self.run)

    def run(self, trial_id, combiner, params):
        self.ran.append(combiner)
        return trial(trial_id, combiner, 0.0)


def test_ordinal_members_are_never_stacked_or_blended():
    search = Search(tasks.Task.ORDINAL)
    search.search(
        [
            trial("1", "ordinal_ridge", 0.8),
            trial("2", "ordinal_random_forest", 0.7),
        ]
    )
    assert search.ran == []
    assert search.entered == [
        ("stacking", experiments.Edict.REJECT, "does not support ordinal"),
        ("blending", experiments.Edict.REJECT, "does not support ordinal"),
    ]


def test_supported_tasks_still_run_both_combiners():
    search = Search(tasks.Task.REGRESSION)
    search.search([trial("1", "ridge", 0.8), trial("2", "random_forest", 0.7)])
    assert search.ran == ["stacking", "blending"]


def test_combiners_support_no_task_without_probabilities_of_their_own():
    for task in (tasks.Task.ORDINAL, tasks.Task.SURVIVAL, tasks.Task.RANKING):
        assert task not in ensembles.Stacking.capabilities.tasks
        assert task not in ensembles.Blending.capabilities.tasks
