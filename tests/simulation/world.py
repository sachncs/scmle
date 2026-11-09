"""A seeded deployment world for the controller.

A `World` owns a deployment directory and an inbox. `deliver(i)` writes what
the outside world hands the controller at tick `i`: the served batch `i` and
the labels that arrive at tick `i`. The truth comes from the stream simulator
that certified the loop (`benchmarks/qualification/streams.py`).
"""

import datetime
import pathlib
import types

import numpy
import pandas

from benchmarks.qualification import streams
from scmle.governance import deployment_ledger
from scmle.governance import loop
from scmle.operations import controller as controller_module

PERSON_KEY = "person-key"
SYSTEM_KEY = "system-key"
SCOPE = "model-v1:stream-a"
TODAY = datetime.date(2026, 11, 4)
CHARTER = types.SimpleNamespace(charter_hash="a" * 64)
KEYS = {deployment_ledger.key_fingerprint(PERSON_KEY): PERSON_KEY}
ALLOWED_KINDS = frozenset(
    {
        deployment_ledger.OPEN,
        deployment_ledger.OBSERVE_SUMMARY,
        deployment_ledger.ALARM,
        deployment_ledger.RETRAIN,
        deployment_ledger.CHALLENGER_RESULT,
        deployment_ledger.PROMOTION_PROPOSAL,
        deployment_ledger.ROLLBACK_PROPOSAL,
        deployment_ledger.ESCALATION,
        deployment_ledger.HALT,
        deployment_ledger.INCIDENT,
        deployment_ledger.ENVELOPE,
        deployment_ledger.ENVELOPE_USED,
        deployment_ledger.ENVELOPE_REVOKED,
        deployment_ledger.CONTROLLER,
    }
)


def make_policy(**changes):
    """A loop policy that lets an alarm gate retraining."""
    values = {
        "reference_risk": 0.10,
        "tolerance": 0.05,
        "scope": SCOPE,
        "initial_tau": 0.5,
        "min_matured": 5,
        "r_star": 0.6,
        "alarm_gates_retrain": True,
        "outcome_window": 0,
    }
    values.update(changes)
    return loop.LoopPolicy(**values)


def make_stream(**changes):
    """A stream with no label window unless the caller asks for one."""
    return streams.simulate(streams.StreamSpec(**{"window": 0, **changes}))


def envelope(limit=1, **changes):
    """A signed envelope for the world's scope."""
    fields = {
        "envelope_id": "e1",
        "limits": {"train_challenger": limit},
        "statement": "train a challenger when the alarm gates retraining",
        "role": "owner",
        "name_or_id": "P. Owner",
        "date": "2026-11-01",
        "scope": SCOPE,
        "expires": "2026-12-01",
    }
    fields.update(changes)
    return deployment_ledger.Envelope.create(PERSON_KEY, **fields)


class World:
    """A deployment, an inbox and the stream that feeds it."""

    def __init__(self, root, stream, policy=None):
        self.root = pathlib.Path(root)
        self.stream = stream
        self.deploy = self.root / "deploy"
        self.inbox = self.root / "inbox"
        (self.inbox / "batches").mkdir(parents=True)
        (self.inbox / "labels").mkdir(parents=True)
        loop.Loop.open(
            self.deploy,
            CHARTER,
            policy or make_policy(),
            verification_keys=KEYS,
            system_keys=[SYSTEM_KEY],
            today=lambda: TODAY,
        )

    def controller(self, **kwargs):
        """A fresh controller, as a new process would open it."""
        return controller_module.Controller(
            self.deploy,
            verification_keys=KEYS,
            system_keys=(SYSTEM_KEY,),
            today=lambda: TODAY,
            **kwargs,
        )

    def deliver(self, index, labels=True, challenger=None, corrupt=False):
        """Writes batch `index` and the labels that arrive at tick `index`.

        Args:
            index: The tick.
            labels: Whether this tick's labels arrive (False models an outage).
            challenger: Optional ``(name, losses)`` where ``losses`` is a
                matrix like the stream's; its losses are written paired.
            corrupt: Write the batch without its prediction column.
        """
        stream = self.stream
        size = stream.loss.shape[1]
        batch = pandas.DataFrame(
            {
                "case_id": [f"b{index}-{i}" for i in range(size)],
                "propensity": stream.pi[index],
                "audit": stream.audit[index],
                "prediction": numpy.zeros(size),
                "decision": ["approve"] * size,
            }
        )
        if corrupt:
            batch = batch.drop(columns=["prediction"])
        batch.to_csv(
            self.inbox / "batches" / f"b{index:04d}.csv",
            index=False,
            lineterminator="\n",
        )
        if not labels:
            return
        rows = []
        for earlier in range(index + 1):
            hit = (stream.arrival[earlier] == index) & stream.selected[earlier]
            for position in numpy.flatnonzero(hit):
                row = {
                    "case_id": f"b{earlier}-{position}",
                    "loss": stream.loss[earlier, position],
                }
                if challenger is not None:
                    name, matrix = challenger
                    row[f"challenger:{name}"] = matrix[earlier, position]
                rows.append(row)
        if rows:
            pandas.DataFrame(rows).to_csv(
                self.inbox / "labels" / f"l{index:04d}.csv",
                index=False,
                lineterminator="\n",
            )


def projection(ledger):
    """The ledger without tick rows: what happened, not how often it ran."""
    return [
        (row["kind"], row["body"])
        for row in ledger.rows
        if not (
            row["kind"] == deployment_ledger.CONTROLLER
            and row["body"]["phase"] == deployment_ledger.TICK
        )
    ]


def kinds(ledger, kind):
    """The rows of one kind."""
    return [row for row in ledger.rows if row["kind"] == kind]
