"""The controller against seeded worlds, and killed at every checkpoint.

These are measurements of simulated streams, not of real traffic. Properties,
fixed before the first run:

* it takes no action that needs a person: no attestation is spent, no
  deployment is recorded, and no ledger row is of a kind outside the loop's
  own and the controller's;
* it never exceeds an envelope: trainings equal the uses on the ledger and
  never exceed the limit;
* the ledger it leaves verifies from disk;
* killed at any checkpoint and started again, it ends where an uninterrupted
  run ends;
* with labels missing or an input corrupted it carries on and asks a person;
  it does not halt the loop or act.
"""

import numpy
import pytest

from scmle.governance import deployment_ledger
import scmle.operations.controller
import tests.simulation.world


class CrashError(Exception):
    """Raised by a checkpoint to stand for a killed process."""


class Trainer:
    """A retrain hook that counts its calls."""

    def __init__(self, name="challenger-1"):
        self.name = name
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.name


def run(world, ticks, controller, **deliveries):
    """Delivers and ticks `ticks` times; returns the last result."""
    result = None
    for index in range(ticks):
        world.deliver(index, **deliveries.get("always", {}))
        result = controller.tick(world.inbox)
    return result


def check_properties(world, trainer=None, limit=0):
    ledger = deployment_ledger.DeploymentLedger.load(
        world.deploy / deployment_ledger.FILE_NAME
    )  # on disk
    assert {
        row["kind"] for row in ledger.rows
    } <= tests.simulation.world.ALLOWED_KINDS
    assert not ledger.attestations and not ledger.used  # no person needed
    uses = tests.simulation.world.kinds(ledger, deployment_ledger.ENVELOPE_USED)
    assert len(uses) <= limit  # within the envelope
    if trainer is not None:
        assert trainer.calls == len(uses)  # trainings equal uses
    assert world.controller().loop.verify_ledger()  # the ledger verifies
    return ledger


def abrupt(**changes):
    return tests.simulation.world.make_stream(
        **{
            "kind": "abrupt",
            "batches": 80,
            "change": 30,
            "delta": 0.25,
            "seed": 4,
            **changes,
        }
    )


# ----------------------------------------------------------------- worlds


def test_a_stable_world_raises_nothing(tmp_path):
    world = tests.simulation.world.World(
        tmp_path, tests.simulation.world.make_stream(batches=40, seed=3)
    )
    trainer = Trainer()
    controller = world.controller(retrain=trainer)
    controller.envelopes.grant(tests.simulation.world.envelope(limit=2))
    run(world, 40, controller)
    ledger = check_properties(world, trainer, limit=2)
    assert not tests.simulation.world.kinds(ledger, deployment_ledger.ALARM)
    assert trainer.calls == 0
    assert not controller.loop.pending_escalations()


def test_an_abrupt_rise_trains_one_challenger_inside_the_envelope(tmp_path):
    world = tests.simulation.world.World(tmp_path, abrupt())
    trainer = Trainer()
    controller = world.controller(retrain=trainer)
    controller.envelopes.grant(tests.simulation.world.envelope(limit=1))
    run(world, 80, controller)
    ledger = check_properties(world, trainer, limit=1)
    assert trainer.calls == 1
    assert tests.simulation.world.kinds(ledger, deployment_ledger.RETRAIN)[0][
        "body"
    ]["challenger"] == ("challenger-1")
    assert controller.envelopes.remaining("e1") == {"train_challenger": 0}
    alarms = [
        r
        for r in tests.simulation.world.kinds(ledger, deployment_ledger.ALARM)
        if r["body"]["rule"] == "L1"
    ]
    assert alarms and alarms[0]["body"]["batch"] >= 30


def test_a_gradual_rise_is_handled_the_same_way(tmp_path):
    world = tests.simulation.world.World(
        tmp_path, abrupt(kind="gradual", ramp=20, delta=0.3, seed=5)
    )
    trainer = Trainer()
    controller = world.controller(retrain=trainer)
    controller.envelopes.grant(tests.simulation.world.envelope(limit=1))
    run(world, 80, controller)
    check_properties(world, trainer, limit=1)
    assert trainer.calls == 1


def test_without_an_envelope_the_alarm_trains_nothing_and_asks(tmp_path):
    world = tests.simulation.world.World(tmp_path, abrupt())
    trainer = Trainer()
    controller = world.controller(retrain=trainer)
    run(world, 80, controller)
    check_properties(world, trainer, limit=0)
    assert trainer.calls == 0
    actions = {
        e.question["action"] for e in controller.loop.pending_escalations()
    }
    assert "retrain_not_authorized" in actions


def test_an_uncertified_alarm_never_reaches_the_envelope(tmp_path):
    world = tests.simulation.world.World(
        tmp_path,
        abrupt(),
        policy=tests.simulation.world.make_policy(alarm_gates_retrain=False),
    )
    trainer = Trainer()
    controller = world.controller(retrain=trainer)
    controller.envelopes.grant(tests.simulation.world.envelope(limit=3))
    run(world, 80, controller)
    check_properties(world, trainer, limit=3)
    assert trainer.calls == 0
    assert controller.envelopes.remaining("e1") == {"train_challenger": 3}
    actions = {
        e.question["action"] for e in controller.loop.pending_escalations()
    }
    assert "l1_alarm_investigate" in actions


def test_a_failing_training_is_escalated_and_the_loop_goes_on(tmp_path):
    world = tests.simulation.world.World(tmp_path, abrupt())

    def broken():
        raise RuntimeError("out of memory")

    controller = world.controller(retrain=broken)
    controller.envelopes.grant(tests.simulation.world.envelope(limit=1))
    result = run(world, 80, controller)
    assert result.status != "halted"
    actions = {
        e.question["action"] for e in controller.loop.pending_escalations()
    }
    assert "retrain_failed" in actions
    check_properties(world, None, limit=1)


def test_an_envelope_without_a_trainer_asks_instead(tmp_path):
    world = tests.simulation.world.World(tmp_path, abrupt())
    controller = world.controller()
    controller.envelopes.grant(tests.simulation.world.envelope(limit=1))
    run(world, 80, controller)
    actions = {
        e.question["action"] for e in controller.loop.pending_escalations()
    }
    assert "retrain_not_configured" in actions


def test_delayed_labels_alarm_only_after_they_arrive(tmp_path):
    stream = abrupt(window=3, batches=100, seed=6)
    world = tests.simulation.world.World(
        tmp_path,
        stream,
        policy=tests.simulation.world.make_policy(outcome_window=3),
    )
    trainer = Trainer()
    controller = world.controller(retrain=trainer)
    controller.envelopes.grant(tests.simulation.world.envelope(limit=1))
    run(world, 85, controller)
    ledger = check_properties(world, trainer, limit=1)
    alarms = [
        r
        for r in tests.simulation.world.kinds(ledger, deployment_ledger.ALARM)
        if r["body"]["rule"] == "L1"
    ]
    assert alarms and alarms[0]["body"]["batch"] >= 30


def test_a_label_outage_carries_on_and_asks(tmp_path):
    world = tests.simulation.world.World(tmp_path, abrupt())
    trainer = Trainer()
    controller = world.controller(
        retrain=trainer,
        policy=scmle.operations.controller.ControllerPolicy(
            max_unlabelled_ticks=3
        ),
    )
    controller.envelopes.grant(tests.simulation.world.envelope(limit=1))
    for index in range(20):
        world.deliver(index)
        controller.tick(world.inbox)
    for index in range(20, 30):
        world.deliver(index, labels=False)
        result = controller.tick(world.inbox)
    assert result.status == "insufficient_labels"
    assert not controller.loop.halted
    assert trainer.calls == 0
    actions = {
        e.question["action"] for e in controller.loop.pending_escalations()
    }
    assert "controller_labels_not_arriving" in actions
    check_properties(world, trainer, limit=1)


def test_a_corrupted_batch_is_set_aside_and_the_rest_applied(tmp_path):
    world = tests.simulation.world.World(
        tmp_path, tests.simulation.world.make_stream(batches=20, size=20)
    )
    controller = world.controller()
    for index in range(10):
        world.deliver(index, corrupt=index == 4)
        controller.tick(world.inbox)
    assert controller.loop.batch == 9
    assert len(controller.loop.ledger.rejected) == 1
    assert not controller.loop.halted
    actions = [
        e.question["action"] for e in controller.loop.pending_escalations()
    ]
    assert actions == ["controller_input_rejected"]
    check_properties(world, None, limit=0)


def challenger_losses(stream, kind):
    """Paired losses of a challenger that is better or worse than the champion."""
    rng = numpy.random.default_rng(11)
    if kind == "better":
        return numpy.where(
            rng.random(stream.loss.shape) < 0.8, 0.0, stream.loss
        )
    return numpy.maximum(
        stream.loss, (rng.random(stream.loss.shape) < 0.3) * 1.0
    )


@pytest.mark.parametrize("kind", ["better", "poisoned"])
def test_a_challenger_may_earn_a_proposal_but_never_a_promotion(tmp_path, kind):
    stream = abrupt(batches=120, size=100, change=20, delta=0.3, seed=8)
    world = tests.simulation.world.World(tmp_path, stream)
    trainer = Trainer(name="candidate")
    controller = world.controller(retrain=trainer)
    controller.envelopes.grant(tests.simulation.world.envelope(limit=1))
    matrix = challenger_losses(stream, kind)
    for index in range(120):
        world.deliver(index, challenger=("candidate", matrix))
        controller.tick(world.inbox)
    ledger = check_properties(world, trainer, limit=1)
    proposals = tests.simulation.world.kinds(
        ledger, deployment_ledger.PROMOTION_PROPOSAL
    )
    if kind == "poisoned":
        assert not proposals
    else:
        assert proposals
    assert not ledger.attestations and not ledger.used
    assert not tests.simulation.world.kinds(
        ledger, deployment_ledger.DEPLOYMENT
    )


# ------------------------------------------------------ kill and start again


def run_ticks(world, ticks, **kwargs):
    controller = world.controller(**kwargs)
    for index in range(ticks):
        world.deliver(index)
        controller.tick(world.inbox)
    return controller


def counted_checkpoints(tmp_path, ticks):
    seen = []
    world = tests.simulation.world.World(
        tmp_path / "count",
        tests.simulation.world.make_stream(batches=ticks, size=20),
    )
    run_ticks(world, ticks, checkpoint=seen.append)
    return seen, tests.simulation.world.projection(
        world.controller().loop.ledger
    )


def test_killed_at_any_checkpoint_it_ends_where_an_uninterrupted_run_ends(
    tmp_path,
):
    ticks = 4
    seen, expected = counted_checkpoints(tmp_path, ticks)
    # per tick: batch intent, loop, applied; labels the same; then the step
    assert len(seen) == 7 * ticks
    for kill in range(len(seen)):
        world = tests.simulation.world.World(
            tmp_path / f"kill-{kill}",
            tests.simulation.world.make_stream(batches=ticks, size=20),
        )
        count = [0]

        def checkpoint(label, kill=kill, count=count):
            if count[0] == kill:
                count[0] += 1
                raise CrashError(label)
            count[0] += 1

        interrupted_at = None
        controller = world.controller(checkpoint=checkpoint)
        for index in range(ticks):
            world.deliver(index)
            try:
                controller.tick(world.inbox)
            except CrashError:
                interrupted_at = index
                controller = world.controller()  # a new process
                controller.tick(world.inbox)
        assert interrupted_at is not None, kill
        assert (
            tests.simulation.world.projection(controller.loop.ledger)
            == expected
        ), (
            kill,
            seen[kill],
        )
        assert controller.loop.batch == ticks
        deployment_ledger.DeploymentLedger.load(
            world.deploy / deployment_ledger.FILE_NAME
        )
