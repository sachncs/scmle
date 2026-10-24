"""Tests for the protocol's development / hold-out split."""

import json

from benchmarks.sota import split


def test_the_committed_split_matches_its_recorded_seed():
    committed = json.loads(split.OUTPUT.read_text(encoding="utf-8"))

    assert committed == split.document(split.load_tasks())


def test_every_task_is_in_exactly_one_set():
    tasks = split.load_tasks()
    sets = split.assign(tasks)

    everything = {int(task["task_id"]) for task in tasks}
    assert set(sets["development"]) | set(sets["holdout"]) == everything
    assert not set(sets["development"]) & set(sets["holdout"])


def test_each_problem_type_is_represented_in_both_sets():
    tasks = split.load_tasks()
    types = {int(task["task_id"]): task["problem_type"] for task in tasks}
    sets = split.assign(tasks)

    for name in ("development", "holdout"):
        assert {types[i] for i in sets[name]} == set(types.values())


def test_the_assignment_depends_on_the_seed_but_not_on_row_order():
    tasks = split.load_tasks()

    reordered = split.assign(list(reversed(tasks)))
    other = split.assign(tasks, seed="another-seed")

    assert reordered == split.assign(tasks)
    assert other != split.assign(tasks)


def test_check_mode_fails_when_the_file_disagrees(tmp_path, monkeypatch):
    stale = tmp_path / "split.json"
    stale.write_text(json.dumps({"seed": "old"}), encoding="utf-8")
    monkeypatch.setattr(split, "OUTPUT", stale)

    assert split.main(["--check"]) == 1
