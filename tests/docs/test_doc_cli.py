"""Guarantees of docs/CLI.md that no other test enforced."""

import logging
import os
import pathlib

from scmle import cli
from scmle.deploy import parity

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "deploy" / "local" / "golden"


def options(parser):
    """Yields every option string of a parser and of all its sub-commands.

    Args:
        parser: The argument parser to walk.

    Yields:
        Each option string, such as `--key-file`.
    """
    for action in vars(parser)["_actions"]:
        yield from action.option_strings
        for child in getattr(action, "choices", None) or {}:
            if isinstance(action.choices, dict):
                yield from options(action.choices[child])


def test_deploy_commands_take_keys_only_as_files():
    """Enforces docs/CLI.md:163.

    Also enforces website/src/pages/docs/cli.astro:96.
    """
    deploy = next(
        action.choices["deploy"]
        for action in vars(cli.build_parser())["_actions"]
        if isinstance(getattr(action, "choices", None), dict)
    )
    named = {option for option in options(deploy) if "key" in option}
    assert "--key-file" in named
    assert all(option.endswith("-file") for option in named), named


def briefly(adapter, *unused):
    """Stands in for `runtime.hold`: starts the server, then stops it."""
    adapter.start()
    adapter.stop()


def test_serve_needs_authentication_beyond_loopback(
    tmp_path, monkeypatch, caplog
):
    """Enforces docs/CLI.md:190.

    Also enforces website/src/pages/docs/cli.astro:100.
    """
    for name in list(os.environ):
        if name.startswith("SCMLE_AUTH_"):
            monkeypatch.delenv(name)
    monkeypatch.delenv("SCMLE_ENV", raising=False)
    monkeypatch.setattr(cli.delivery.runtime, "hold", briefly)
    folder = tmp_path / "bundle"
    parity.handwrite(parity.goldens(GOLDEN).spec).save(folder)
    command = ["serve", str(folder), "--trusted", "--port", "0"]
    with caplog.at_level(logging.INFO, logger=cli.__name__):
        exposed = cli.main([*command, "--host", "0.0.0.0"])  # nosec B104
        assert exposed == cli.EXIT_FAILED
        assert "without authentication" in caplog.text
        assert "serving" not in caplog.text
        assert cli.main([*command, "--host", "127.0.0.1"]) == cli.EXIT_OK
        assert "serving" in caplog.text
