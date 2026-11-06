"""Coverage violations of a served bundle reach the loop and gate nothing."""

import types

import numpy as np
import pandas as pd
import pytest
from test_loop import make_policy
from test_loop import open_loop

from scmle.contracts import errors
from scmle.governance import uncertainty_feed


class StubBundle:
    """Sets of one label; `wide` rows get two."""

    novelty = None

    def predict_set(self, features, coverage):
        return [["a", "b"] if w else ["a"] for w in features["wide"]]


def served(rows=40, wide_share=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"wide": rng.random(rows) < wide_share})


def run(feed, batches, label, rows=40):
    violations = 0
    for index in range(batches):
        ids = [f"b{index}-{i}" for i in range(rows)]
        feed.observe(served(rows), np.zeros(rows), ["approve"] * rows, ids)
        violations += feed.mature(
            dict.fromkeys(ids, label), dict.fromkeys(ids, 0.0)
        )
        feed.loop.step()
    return violations


def test_violations_are_counted_ledgered_and_escalate_without_retraining(
    tmp_path,
):
    loop = open_loop(
        tmp_path, make_policy(coverage_target=0.9, alarm_gates_retrain=False)
    )
    feed = uncertainty_feed.UncertaintyFeed(loop, StubBundle(), 0.9)

    violations = run(feed, 30, "z")

    assert violations == 30 * 40
    rows = [
        r["body"]
        for r in loop.ledger.rows
        if r["body"].get("source") == uncertainty_feed.SOURCE
    ]
    assert rows[0]["violations"] == 40 and rows[0]["gates_retrain"] is False
    actions = {e.question["action"] for e in loop.pending_escalations()}
    assert "coverage_below_nominal" in actions
    kinds = {r["kind"] for r in loop.ledger.rows}
    assert "retrain" not in kinds
    assert loop.policy.alarm_gates_retrain is False


def test_covered_cases_raise_no_coverage_escalation(tmp_path):
    loop = open_loop(
        tmp_path, make_policy(coverage_target=0.9, alarm_gates_retrain=False)
    )
    feed = uncertainty_feed.UncertaintyFeed(loop, StubBundle(), 0.9)

    assert run(feed, 30, "a") == 0

    actions = {e.question["action"] for e in loop.pending_escalations()}
    assert "coverage_below_nominal" not in actions


def test_set_size_and_novelty_go_to_the_loop(tmp_path):
    loop = open_loop(tmp_path, make_policy(alarm_gates_retrain=False))
    bundle = StubBundle()
    bundle.novelty = types.SimpleNamespace(
        score=lambda f: pd.DataFrame(
            {"out_of_distribution": [True] * len(f)}, index=f.index
        )
    )
    feed = uncertainty_feed.UncertaintyFeed(loop, bundle, 0.9)
    ids = [f"x{i}" for i in range(10)]

    feed.observe(served(10, 1.0), np.zeros(10), ["approve"] * 10, ids)

    row = [r for r in loop.ledger.rows if r["kind"] == "observe_summary"][-1]
    assert row["body"]["cases"] == 10
    assert loop.pending_batches[-1]["novel"] == [True] * 10
    assert set(feed.sets["x0"]) == {"a", "b"}


def test_labels_of_unserved_cases_and_bad_nominals_are_refused(tmp_path):
    loop = open_loop(tmp_path, make_policy(alarm_gates_retrain=False))
    feed = uncertainty_feed.UncertaintyFeed(loop, StubBundle(), 0.9)
    with pytest.raises(errors.ScmleError, match="did not serve"):
        feed.mature({"ghost": "a"}, {})
    with pytest.raises(ValueError):
        uncertainty_feed.UncertaintyFeed(loop, StubBundle(), 1.0)
    with pytest.raises(ValueError):
        feed.observe(served(3), np.zeros(3), ["approve"] * 3, ["a"])
