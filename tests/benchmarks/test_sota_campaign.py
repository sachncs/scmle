"""Tests for the campaign driver."""

import json
import os
import pathlib

import pytest

from benchmarks.sota import campaign
from benchmarks.sota import split


def test_the_holdout_is_refused_without_an_explicit_opt_in():
    with pytest.raises(PermissionError, match="sealed"):
        campaign.phase_tasks("holdout", open_holdout=False)


def test_the_development_phase_lists_exactly_the_development_tasks():
    document = json.loads(split.OUTPUT.read_text(encoding="utf-8"))

    assert campaign.phase_tasks("development", False) == document["development"]


def test_each_system_gets_its_own_runner_and_all_budgets(tmp_path):
    scmle = campaign.command("scmle", 7, (60, 600), tmp_path, "py", False)
    base = campaign.command("flaml", 7, (60, 600), tmp_path, "py", False)

    assert "benchmarks.sota.scmle_runner" in scmle
    assert "benchmarks.sota.baselines" in base and "--lite" in base
    for run in (scmle, base):
        start = run.index("--budgets") + 1
        assert run[start : start + 2] == ["60", "600"]
        assert str(
            tmp_path / f"{run[run.index('--output') + 1].split('/')[-1]}"
        )
        assert "--open-holdout" not in run


def test_the_holdout_flag_is_passed_on_only_when_given(tmp_path):
    run = campaign.command("scmle", 7, (60,), tmp_path, "py", True)

    assert "--open-holdout" in run


def test_a_live_campaign_blocks_a_second_one(tmp_path):
    lock = tmp_path / "campaign.lock"
    lock.write_text(str(os.getpid()), encoding="utf-8")

    with pytest.raises(RuntimeError, match="already running"):
        campaign.acquire(lock)


def test_a_stale_lock_is_replaced(tmp_path):
    lock = tmp_path / "campaign.lock"
    lock.write_text("999999999", encoding="utf-8")

    campaign.acquire(lock)

    assert pathlib.Path(lock).read_text(encoding="utf-8") == str(os.getpid())
