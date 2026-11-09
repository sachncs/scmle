"""The controller applies each inbox file once and keeps its limits."""

import json

import pytest

from scmle import cli
from scmle.governance import deployment_ledger
from scmle.operations import controller as controller_module
from tests.simulation import world as worlds


@pytest.fixture
def world(tmp_path):
    return worlds.World(tmp_path, worlds.make_stream(batches=20, size=20))


def test_a_file_is_applied_once_whatever_it_is_called(world):
    world.deliver(0)
    controller = world.controller()
    first = controller.tick(world.inbox)
    assert (first.observed, first.matured) == (1, 20)
    again = world.controller().tick(world.inbox)
    assert (again.observed, again.matured) == (0, 0)
    copy = world.inbox / "batches" / "renamed.csv"
    copy.write_bytes((world.inbox / "batches" / "b0000.csv").read_bytes())
    assert world.controller().tick(world.inbox).observed == 0
    assert world.controller().loop.batch == 1


def test_applying_is_journalled_as_intent_then_applied(world):
    world.deliver(0)
    controller = world.controller()
    controller.tick(world.inbox)
    phases = [
        (row["body"]["phase"], row["body"].get("what"))
        for row in controller.loop.ledger.rows
        if row["kind"] == deployment_ledger.CONTROLLER
    ]
    assert phases == [
        ("intent", "batch"),
        ("applied", "batch"),
        ("intent", "labels"),
        ("applied", "labels"),
        ("tick", None),
    ]


def test_a_malformed_file_is_rejected_once_and_escalated(world):
    world.deliver(0, corrupt=True)
    world.deliver(1)
    controller = world.controller()
    result = controller.tick(world.inbox)
    assert result.rejected == ("b0000.csv",)
    assert result.observed == 1
    assert not controller.loop.halted
    questions = [
        e.question["action"] for e in controller.loop.pending_escalations()
    ]
    assert questions.count("controller_input_rejected") == 1
    again = world.controller().tick(world.inbox)
    assert again.rejected == () and again.observed == 0
    assert len(world.controller().loop.ledger.rejected) == 1


def test_an_unreadable_labels_file_is_rejected(world):
    world.deliver(0)
    (world.inbox / "labels" / "l0000.csv").write_text("no,header\n")
    (world.inbox / "labels" / "l0001.csv").write_text("")
    result = world.controller().tick(world.inbox)
    assert sorted(result.rejected) == ["l0000.csv", "l0001.csv"]
    assert result.observed == 1


def test_no_new_data_opens_one_escalation(world):
    policy = controller_module.ControllerPolicy(max_idle_ticks=2)
    controller = world.controller(policy=policy)
    controller.tick(world.inbox)
    assert not controller.loop.pending_escalations()
    controller.tick(world.inbox)
    controller.tick(world.inbox)
    actions = [
        e.question["action"] for e in controller.loop.pending_escalations()
    ]
    assert actions == ["controller_no_new_data"]


def test_labels_that_never_arrive_open_one_escalation(world):
    policy = controller_module.ControllerPolicy(max_unlabelled_ticks=2)
    controller = world.controller(policy=policy)
    for index in range(4):
        world.deliver(index, labels=False)
        controller.tick(world.inbox)
    actions = [
        e.question["action"] for e in controller.loop.pending_escalations()
    ]
    assert actions == ["controller_labels_not_arriving"]


def test_a_halted_loop_is_left_alone(world):
    world.deliver(0)
    controller = world.controller()
    controller.loop.enter_halt("operator_halt", {"reason": "test"})
    controller.loop.save_state()
    result = world.controller().tick(world.inbox)
    assert result.status == controller_module.HALTED
    assert result.observed == 0
    assert world.controller().loop.batch == 0


def test_a_state_that_trails_the_ledger_halts_instead_of_guessing(world):
    world.deliver(0)
    controller = world.controller()
    controller.tick(world.inbox)
    controller.loop.ledger.commit(
        deployment_ledger.CONTROLLER, {"phase": deployment_ledger.TICK}
    )
    world.deliver(1)
    result = world.controller().tick(world.inbox)
    assert result.status == controller_module.HALTED
    assert result.observed == 0
    assert len(world.controller().loop.ledger.consumed) == 2


def test_the_policy_needs_positive_limits():
    with pytest.raises(ValueError):
        controller_module.ControllerPolicy(max_idle_ticks=0)
    with pytest.raises(ValueError):
        controller_module.ControllerPolicy(max_unlabelled_ticks=0)


def test_the_tick_report_is_json_ready(world):
    world.deliver(0)
    report = world.controller().tick(world.inbox).to_dict()
    assert json.loads(json.dumps(report)) == report
    assert report["status"] in {"ok", "insufficient_labels"}


# -------------------------------------------------------------------- CLI


def run(*arguments):
    return cli.main([str(a) for a in arguments])


def test_the_operate_command_ticks_grants_revokes_and_halts(
    world, tmp_path, capsys
):
    world.deliver(0)
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps(worlds.KEYS))
    assert (
        run(
            "operate",
            "tick",
            world.deploy,
            "--inbox",
            world.inbox,
            "--keys-file",
            keys,
        )
        == 0
    )
    tick = json.loads(capsys.readouterr().out)
    assert tick["observed"] == 1

    envelope = tmp_path / "envelope.json"
    envelope.write_text(json.dumps(worlds.envelope(limit=2).to_dict()))
    key = tmp_path / "person.key"
    key.write_text(worlds.PERSON_KEY)
    assert (
        run(
            "operate",
            "grant",
            world.deploy,
            "--envelope",
            envelope,
            "--key-file",
            key,
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {
        "e1": {"train_challenger": 2}
    }
    stranger = tmp_path / "stranger.key"
    stranger.write_text("stranger")
    assert (
        run(
            "operate",
            "grant",
            world.deploy,
            "--envelope",
            envelope,
            "--key-file",
            stranger,
        )
        != 0
    )

    assert run("operate", "status", world.deploy) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["applied"] == 2 and status["halted"] is False
    assert status["envelopes"] == {"e1": {"train_challenger": 2}}

    assert run("operate", "revoke", world.deploy, "e1", "--reason", "done") == 0
    assert json.loads(capsys.readouterr().out) == {"e1": {}}
    assert (
        run("operate", "revoke", world.deploy, "e1", "--reason", "again") != 0
    )

    assert run("operate", "halt", world.deploy, "--reason", "drill") == 0
    assert json.loads(capsys.readouterr().out)["halted"] is True


def test_a_missing_deployment_is_a_clean_error(tmp_path):
    assert run("operate", "status", tmp_path / "nothing") != 0
