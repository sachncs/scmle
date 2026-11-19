"""Each CLI outcome has its own exit code and an honest report (ERR-21).

Before, any exception other than `ScmleError` escaped as a raw traceback
with exit 1, the code of a handled failure; `LeakageError` exited 1 rather
than as blocked; argparse's usage exit (2) collided with "blocked"; and no
failure could show its traceback or details.
"""

import logging

import pytest

from scmle import cli
from scmle.contracts import errors

COMMAND = next(c for c in cli.COMMANDS if c.name == "catalog")


def failing(monkeypatch, error):
    """Makes the `catalog` command, which needs no arguments, raise `error`."""

    def execute(args):
        raise error

    monkeypatch.setattr(COMMAND, "execute", execute)


def run(argv, caplog):
    with caplog.at_level(logging.DEBUG, logger=cli.__name__):
        return cli.main(argv)


def test_the_exit_codes_are_distinct():
    codes = [
        cli.EXIT_OK,
        cli.EXIT_FAILED,
        cli.EXIT_BLOCKED,
        cli.EXIT_USAGE,
        cli.EXIT_INTERNAL,
    ]
    assert len(set(codes)) == len(codes)


def test_a_bad_command_line_exits_with_the_usage_code(capsys):
    with pytest.raises(SystemExit) as raised:
        cli.main(["no-such-command"])
    assert raised.value.code == cli.EXIT_USAGE
    assert "error:" in capsys.readouterr().err


def test_a_leak_exits_as_blocked(monkeypatch, caplog):
    # Enforces docs/CLI.md:15.
    # Enforces docs/WHEN_THINGS_GO_WRONG.md:22.
    # Enforces website/src/learn/when-things-go-wrong.html:11.
    failing(monkeypatch, errors.LeakageError("test set opened twice"))
    assert run([COMMAND.name], caplog) == cli.EXIT_BLOCKED
    assert "blocked: test set opened twice" in caplog.text


def test_a_defect_exits_as_internal_with_its_traceback(monkeypatch, caplog):
    # Enforces docs/CLI.md:17.
    failing(monkeypatch, RuntimeError("unexpected state"))
    assert run([COMMAND.name], caplog) == cli.EXIT_INTERNAL
    assert "internal error" in caplog.text
    assert "Traceback" in caplog.text and "unexpected state" in caplog.text


def test_debug_shows_causes_details_and_traceback(monkeypatch, caplog):
    try:
        try:
            raise KeyError("missing column")
        except KeyError as cause:
            raise errors.DataError("cannot read", {"path": "x.csv"}) from cause
    except errors.DataError as error:
        caught = error
    failing(monkeypatch, caught)
    assert run(["--debug", COMMAND.name], caplog) == cli.EXIT_FAILED
    assert "KeyError: 'missing column'" in caplog.text
    assert "x.csv" in caplog.text and "Traceback" in caplog.text


def test_without_debug_a_handled_failure_is_one_line(monkeypatch, caplog):
    # Enforces docs/WHEN_THINGS_GO_WRONG.md:21.
    # Enforces website/src/learn/when-things-go-wrong.html:10.
    failing(monkeypatch, errors.DataError("cannot read", {"path": "x.csv"}))
    assert run([COMMAND.name], caplog) == cli.EXIT_FAILED
    assert "Traceback" not in caplog.text
    assert "data error: cannot read" in caplog.text


@pytest.mark.parametrize(
    "error",
    [
        errors.InputError("bad column"),
        errors.KindError("bad column"),
        errors.OrderError("bad column"),
    ],
)
def test_an_input_error_exits_as_failed(monkeypatch, caplog, error):
    # Enforces docs/CLI.md:14: bad data is a handled failure, not a defect.
    # Before ERR-20 the same raise was a bare ValueError and exited 70.
    failing(monkeypatch, error)
    assert run([COMMAND.name], caplog) == cli.EXIT_FAILED
    assert "input error: bad column" in caplog.text
    assert "internal error" not in caplog.text


def test_a_plain_value_error_still_exits_as_internal(monkeypatch, caplog):
    # Enforces docs/CLI.md:17: a bare ValueError marks a defect (ERR-20).
    failing(monkeypatch, ValueError("unexpected value"))
    assert run([COMMAND.name], caplog) == cli.EXIT_INTERNAL
    assert "internal error" in caplog.text
