"""Guarantees of docs/WHAT_SCMLE_CANNOT_DO.md that no other test enforced."""

from scmle import cli


def actions(parser):
    """Returns the sub-commands of a parser.

    Args:
        parser: An argument parser.

    Returns:
        Sub-command name to its parser; empty when it has none.
    """
    # argparse has no public way to list sub-commands.
    for action in vars(parser)["_actions"]:
        if isinstance(getattr(action, "choices", None), dict):
            return action.choices
    return {}


def options(parser):
    """Returns the destinations of a parser's own arguments.

    Args:
        parser: An argument parser.

    Returns:
        The set of argument destinations, without `help`.
    """
    return {a.dest for a in vars(parser)["_actions"] if a.dest != "help"}


def test_the_command_line_only_records_replies_and_never_lifts_a_halt():
    """Enforces docs/WHAT_SCMLE_CANNOT_DO.md:173.

    A new `loop` or `operate` action fails this test, so that whoever adds a
    command to sign a reply or release a halt also corrects the page.
    """
    commands = actions(cli.build_parser())
    loop = actions(commands["loop"])
    operate = actions(commands["operate"])

    assert set(loop) == {
        "init",
        "mature",
        "observe",
        "pending",
        "reply",
        "step",
        "verify",
    }
    assert set(operate) == {"grant", "halt", "revoke", "status", "tick"}
    assert options(loop["reply"]) == {
        "deploy_dir",
        "escalation_id",
        "attestation",
        "key_file",
    }
    assert options(operate["halt"]) == {"deploy_dir", "reason"}
