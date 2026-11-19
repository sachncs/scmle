"""Guarantees of docs/BEGINNER_FINDINGS.md that no other test enforced."""

import logging
import types

from scmle.contracts import tasks
from scmle.pipeline import stages


def test_a_warn_run_logs_its_unsettled_suspects_while_profiling(caplog):
    """Enforces docs/BEGINNER_FINDINGS.md:28 (the warning line).

    Also enforces website/src/learn/beginner-findings.html:13.
    """
    spec = tasks.Config(
        data=tasks.Dataset(uri="m", target="survived"),
        leakage_policy="warn",
    )
    profile = types.SimpleNamespace(
        untested=False,
        admonition="",
        blockers=("boat", "body"),
        leakage={"suspects": ["body", "boat"], "probe": {}},
    )
    with caplog.at_level(logging.WARNING, logger=stages.__name__):
        stages.ProfileStage().enforce_leakage_policy(spec, profile)
    lines = [r.getMessage() for r in caplog.records]
    assert any(
        "leakage suspect(s) boat, body" in line and "provisional" in line
        for line in lines
    ), lines
