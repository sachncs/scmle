"""A loop, a stream of cases and a release world for continuous-evaluation."""

import datetime
import types

import numpy as np
import pandas as pd

from scmle.governance import deployment_ledger as dl
from scmle.governance import loop as loop_module
from scmle.governance import release_state as rs
from scmle.operations import continuous
from scmle.operations import signals
from tests import support_release as sr
from tests import support_runtime as rt

HUMAN_KEY = "human-key"
SYSTEM_KEY = "system-key"
TODAY = datetime.date(2026, 11, 4)
CHARTER = types.SimpleNamespace(charter_hash="b" * 64)
SCOPE = "model-v1:stream-a"
MODEL = "churn"


def make_loop(tmp_path, **changes):
    values = {
        "reference_risk": 0.10,
        "tolerance": 0.10,
        "scope": SCOPE,
        "initial_tau": 0.5,
        "min_matured": 5,
        "r_star": 0.6,
        "outcome_window": 0,
        "alarm_gates_retrain": False,
        "challenger_budget": 100,
        "reply_timeout": 10_000,
        "delta_promo": 0.0,
    }
    values.update(changes)
    return loop_module.Loop.open(
        tmp_path / "deploy",
        CHARTER,
        loop_module.LoopPolicy(**values),
        verification_keys={dl.key_fingerprint(HUMAN_KEY): HUMAN_KEY},
        system_keys=[SYSTEM_KEY],
        today=lambda: TODAY,
    )


def reply_to(escalation, decision="approve"):
    return dl.Attestation.create(
        HUMAN_KEY,
        escalation_id=escalation.id,
        question_hash=escalation.question_hash,
        decision=decision,
        statement="reviewed the evidence",
        evidence="ticket-9",
        role="owner",
        name_or_id="P. Owner",
        date="2026-11-01",
        scope=SCOPE,
        expires="2026-12-01",
    )


class Stream:
    """Batches of cases whose predictions are wrong with a given rate."""

    def __init__(self, loop, evaluation, seed=0, size=100):
        self.loop = loop
        self.evaluation = evaluation
        self.rng = np.random.default_rng(seed)
        self.size = size
        self.index = 0

    def batch(self, champion_error, challengers=None):
        challengers = challengers or {}
        ids = [f"b{self.index}-{i}" for i in range(self.size)]
        truth = self.rng.integers(0, 2, self.size)

        def predict(error):
            wrong = self.rng.random(self.size) < error
            return np.where(wrong, 1 - truth, truth)

        frame = pd.DataFrame({"case_id": ids})
        self.loop.observe(frame, np.zeros(self.size), ["approve"] * self.size)
        made = dict(zip(ids, predict(champion_error), strict=True))
        theirs = {
            name: dict(zip(ids, predict(error), strict=True))
            for name, error in challengers.items()
        }
        self.evaluation.feed_labels(
            made, dict(zip(ids, truth, strict=True)), theirs or None
        )
        self.index += 1
        return self.evaluation.advance()

    def run(self, count, champion_error, challengers=None):
        for ignored in range(count):
            self.batch(champion_error, challengers)


def loss_signals(value, samples=200):
    return [
        signals.Signal(
            "windowed_loss",
            signals.Category.MODEL_PERFORMANCE,
            signals.Status.BREACHED if value > 0.2 else signals.Status.OK,
            value,
            0.2,
            "max",
            samples,
        )
    ]


def kinds(loop, kind):
    return [r for r in loop.ledger.rows if r["kind"] == kind]


def evaluation(tmp_path, loop, **kwargs):
    policy = kwargs.pop("policy", continuous.RetrainPolicy())
    return continuous.ContinuousEvaluation(
        loop, tmp_path / "ops", policy, clock=lambda: 1000.0, **kwargs
    )


# ---- the controlled degradation ----------------------------------------


def degraded(tmp_path, **kwargs):
    loop = make_loop(tmp_path)
    called = []

    def retrain():
        called.append(1)
        return "challenger-1"

    ev = evaluation(tmp_path, loop, retrain=retrain, **kwargs)
    stream = Stream(loop, ev, seed=3)
    stream.run(15, 0.05)
    assert not kinds(loop, dl.ALARM)
    stream.run(25, 0.40)
    return loop, ev, stream, called


def winning_challenger(tmp_path):
    loop, ev, stream, ignored = degraded(tmp_path)
    decision = ev.review(loss_signals(0.4))
    target = next(
        e for e in loop.pending_escalations() if e.id == decision.escalation_id
    )
    loop.reply(target.id, reply_to(target))
    ev.run_approved_retrain(target.id)
    stream.run(60, 0.40, challengers={"challenger-1": 0.05})
    return loop, ev


def release_world(tmp_path):
    store = rt.make_registry(tmp_path / "registry")
    release = rs.ReleaseRegistry(store, None, rt.NO_WAIT)
    release.approve(MODEL, 1, rt.approval_for(1), sr.KEY)
    release.approve(MODEL, 2, rt.approval_for(2), sr.KEY)
    plane = rt.MemoryPlane()
    return release, plane, store


def candidate_for(store, version=2, **changes):
    values = {
        "model": store.resolve(MODEL, version),
        "approval": rt.approval_for(version),
    }
    values.update(changes)
    return sr.candidate("staging", **values)
