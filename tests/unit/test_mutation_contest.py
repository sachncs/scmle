"""Exact evidence fed to challengers and exact records of their decisions."""

import datetime
import types

import numpy
import pytest

from scmle.governance import deployment_ledger
from scmle.governance.loop import contest
from scmle.governance.loop import cycle
from scmle.governance.loop import terms

HUMAN_KEY = "human-key"
SCOPE = "model-v1:stream-a"


def open_loop(tmp_path, **changes):
    values = {
        "reference": 0.10,
        "tolerance": 0.05,
        "scope": SCOPE,
        "tau": 0.5,
        "matured": 3,
        "target": 0.6,
        "maturity": 0,
        "timeout": 2,
    }
    values.update(changes)
    loop = cycle.Loop.open(
        tmp_path / "deploy",
        types.SimpleNamespace(charter_hash="a" * 64),
        terms.Preregistration(**values),
        verification_keys={deployment_ledger.thumbprint(HUMAN_KEY): HUMAN_KEY},
        system_keys=["system-key"],
        today=lambda: datetime.date(2026, 11, 4),
    )
    assert isinstance(loop, contest.Contest)
    return loop


class Recorder:
    """A process that keeps what it was fed."""

    def __init__(self):
        self.seen = []

    def ingest(self, value):
        self.seen.append(value)


def feed(loop, served, other):
    ids = [f"c{i}" for i in range(len(served))]
    record = {
        "propensity": None,
        "audit": None,
        "challenger_losses": {"c1": dict(zip(ids, other, strict=True))},
    }
    everything = numpy.ones(len(ids), dtype=bool)
    loop.challenge(record, ids, numpy.array(served), everything, everything)


@pytest.mark.parametrize(
    ("served", "other", "expected"),
    [
        ([0.0, 0.0, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2], -0.1),
        ([0.4, 0.2, 0.2, 0.2], [0.0, 0.0, 0.2, 0.2], 0.15),
    ],
)
def test_challenge_feeds_the_mean_paired_difference(
    tmp_path, served, other, expected
):
    loop = open_loop(tmp_path)
    loop.enroll("c1")
    process = Recorder()
    loop.active["c1"]["process"] = process

    feed(loop, served, other)

    assert process.seen == [pytest.approx(expected)]
    assert loop.active["c1"]["batches"] == 1


def test_drain_records_the_mechanism_of_its_decision(tmp_path):
    loop = open_loop(tmp_path)
    loop.enroll("c1")
    loop.active["c1"]["process"] = types.SimpleNamespace(e_value=1e6)

    loop.drain()

    assert loop.ledger.decided["c1"]["mechanism"] == loop.mechanism.value


def test_a_rollback_proposal_ends_the_watch_and_escalates_a_second(tmp_path):
    loop = open_loop(tmp_path, step=0.05)
    crossed = types.SimpleNamespace(crossed=True, e_value=40.0)
    loop.batch = 3
    loop.watch = {"previous": "old", "process": crossed}

    assert loop.propose() is None
    assert loop.watch is None
    loop.watch = {"previous": "old", "process": crossed}
    second = loop.propose()

    assert second.question == {
        "action": "two_consecutive_rollbacks",
        "detail": {"batch": 2},
        "scope": SCOPE,
    }
