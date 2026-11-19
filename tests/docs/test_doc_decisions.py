"""Guarantees of docs/DECISIONS.md that no other test enforced."""

import types

from scmle.contracts import tasks
from scmle.pipeline import stages


class Store:
    """Keeps the JSON files a stage writes, by name."""

    def __init__(self):
        self.files = {}

    def write_json(self, name, document):
        self.files[name] = document


def test_no_success_criteria_blocks_the_release_and_invents_none():
    """Enforces docs/DECISIONS.md:221.

    Also enforces website/src/learn/decisions.html:82.
    """
    context = types.SimpleNamespace(
        task=tasks.Task.BINARY,
        metric=types.SimpleNamespace(name="roc_auc", greater_is_better=True),
        spec=types.SimpleNamespace(
            problem=types.SimpleNamespace(success_criteria={})
        ),
    )
    store = Store()
    evaluation = {"test_metrics": {"roc_auc": 0.99}}
    stages.PackageStage().release_evaluation(context, store, evaluation)
    record = evaluation["release_evaluation"]
    assert record["status"] == "policy_not_declared"
    assert record["blocked"] is True and record["passed"] is False
    assert "not invented" in record["blockers"][0]
    assert evaluation["release_policy"] is None
    assert list(store.files) == ["release_evaluation.json"]
