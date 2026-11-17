"""Guarantees of docs/AUTONOMY.md that no other test enforced."""

import pytest

from scmle.governance import deployment_ledger
from scmle.governance import envelope
from scmle.governance import never_list
import scmle.operations.controller
import tests.simulation.world


class CrashError(Exception):
    """Stands for a process killed at a checkpoint."""


@pytest.fixture
def world(tmp_path):
    return tests.simulation.world.World(
        tmp_path, tests.simulation.world.make_stream(batches=20, size=20)
    )


def pending(controller):
    """The actions of the escalations that wait for a person."""
    return [e.question["action"] for e in controller.loop.pending_escalations()]


@pytest.mark.parametrize("action", list(never_list.Forbidden))
def test_no_never_list_action_can_be_put_in_an_envelope(action):
    """Enforces docs/AUTONOMY.md:52 and docs/AUTONOMY.md:60."""
    with pytest.raises(deployment_ledger.MisuseError, match="cannot be put"):
        tests.simulation.world.envelope(limits={action.value: 1})
    assert tests.simulation.world.envelope().limits == {"train_challenger": 1}


def test_an_envelope_never_authorizes_moving_the_champion(world):
    """Enforces docs/AUTONOMY.md:14 (neither does an envelope)."""
    controller = world.controller()
    controller.envelopes.grant(tests.simulation.world.envelope(limit=5))
    with pytest.raises(envelope.RefusedError, match="never list"):
        controller.envelopes.authorize(
            never_list.Forbidden.MOVE_CHAMPION, tests.simulation.world.SCOPE
        )
    assert not tests.simulation.world.kinds(
        controller.loop.ledger, deployment_ledger.ENVELOPE_USED
    )


def test_a_use_is_on_the_ledger_before_the_training_runs(world):
    """Enforces docs/AUTONOMY.md:72 (chained before the action)."""
    seen = []

    def trainer():
        ledger = deployment_ledger.DeploymentLedger.load(
            world.deploy / deployment_ledger.FILE_NAME
        )
        seen.append(
            len(
                tests.simulation.world.kinds(
                    ledger, deployment_ledger.ENVELOPE_USED
                )
            )
        )
        return "challenger-1"

    controller = world.controller(retrain=trainer)
    controller.envelopes.grant(tests.simulation.world.envelope(limit=1))
    assert controller.train() == "challenger-1"
    assert seen == [1]


def test_the_ledger_refuses_a_use_the_envelope_does_not_allow(world):
    """Enforces docs/AUTONOMY.md:72 (the ledger itself refuses)."""
    controller = world.controller()
    ledger = controller.loop.ledger
    controller.envelopes.grant(tests.simulation.world.envelope(limit=1))
    controller.envelopes.grant(
        tests.simulation.world.envelope(envelope_id="e2")
    )
    controller.envelopes.revoke("e2", "retired")

    def use(**changes):
        body = {
            "envelope_id": "e1",
            "action": "train_challenger",
            "scope": tests.simulation.world.SCOPE,
            "date": tests.simulation.world.TODAY.isoformat(),
            "detail": {},
            **changes,
        }
        ledger.commit(deployment_ledger.ENVELOPE_USED, body)

    for changes, message in (
        ({"envelope_id": "nobody"}, "unknown"),
        ({"envelope_id": "e2"}, "revoked"),
        ({"scope": "model-v9:stream-z"}, "scope"),
        ({"action": "shadow_deploy"}, "does not cover"),
    ):
        with pytest.raises(deployment_ledger.MisuseError, match=message):
            use(**changes)
    use()
    with pytest.raises(deployment_ledger.MisuseError, match="spent"):
        use()


def test_a_controller_halt_opens_an_escalation(world):
    """Enforces docs/AUTONOMY.md:53 (a halt by the controller)."""
    world.deliver(0)

    def crash(label):
        raise CrashError(label)

    with pytest.raises(CrashError):
        world.controller(checkpoint=crash).tick(world.inbox)
    batch = world.inbox / "batches" / "b0000.csv"
    batch.write_bytes(batch.read_bytes() + b"\n")
    controller = world.controller()
    result = controller.tick(world.inbox)
    assert result.status == scmle.operations.controller.HALTED
    assert "controller_input_changed" in pending(controller)


def test_a_rejected_file_and_silence_each_open_an_escalation(world):
    """Enforces docs/AUTONOMY.md:53 (a rejected file, silence)."""
    world.deliver(0, corrupt=True)
    limits = scmle.operations.controller.Limits(max_idle_ticks=2)
    controller = world.controller(policy=limits)
    controller.tick(world.inbox)
    assert pending(controller) == ["controller_input_rejected"]
    for ignored in range(3):
        controller.tick(world.inbox)
    assert "controller_no_new_data" in pending(controller)
