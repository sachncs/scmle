"""Signing, storing and verifying the charter; the `scmle charter` CLI."""

import json

import pytest

from scmle import cli
from scmle.contracts import errors
from scmle.data.charter import contracts
from scmle.data.charter import sealing
from tests.unit.test_charter_contracts import make_charter

KEY = "unit-test-key"


def test_signed_charter_verifies(tmp_path):
    path = tmp_path / "analysis_charter.json"
    sealing.write_charter(path, make_charter(), signing_key=KEY)
    loaded = sealing.read_verified(path, signing_key=KEY)
    assert loaded.charter == make_charter()
    assert loaded.signed is True


def test_unsigned_charter_is_allowed_but_reported_unsigned(tmp_path):
    path = tmp_path / "analysis_charter.json"
    sealing.write_charter(path, make_charter())
    loaded = sealing.read_verified(path)
    assert loaded.signed is False
    assert loaded.charter_hash == make_charter().charter_hash


def test_wrong_key_fails_verification(tmp_path):
    path = tmp_path / "analysis_charter.json"
    sealing.write_charter(path, make_charter(), signing_key=KEY)
    with pytest.raises(errors.ArtifactError, match="signature"):
        sealing.read_verified(path, signing_key="another-key")


def test_a_signed_charter_needs_a_key_to_be_trusted(tmp_path):
    path = tmp_path / "analysis_charter.json"
    sealing.write_charter(path, make_charter(), signing_key=KEY)
    with pytest.raises(errors.ArtifactError, match="key"):
        sealing.read_verified(path, require_signature=True)


@pytest.mark.parametrize("offset", [0, 40, 120, 300])
def test_tampering_with_any_byte_fails_verification(tmp_path, offset):
    path = tmp_path / "analysis_charter.json"
    sealing.write_charter(path, make_charter(), signing_key=KEY)
    raw = bytearray(path.read_bytes())
    index = min(offset, len(raw) - 1)
    # Flip one character to a different printable one.
    raw[index] = ord("x") if raw[index] != ord("x") else ord("y")
    path.write_bytes(bytes(raw))
    with pytest.raises((errors.ArtifactError, errors.ConfigError)):
        sealing.read_verified(path, signing_key=KEY)


def test_editing_a_field_without_resigning_fails(tmp_path):
    path = tmp_path / "analysis_charter.json"
    sealing.write_charter(path, make_charter(), signing_key=KEY)
    record = json.loads(path.read_text())
    record["charter"]["question"] = "A different question?"
    path.write_text(json.dumps(record))
    with pytest.raises(errors.ArtifactError, match="hash"):
        sealing.read_verified(path, signing_key=KEY)


def test_an_amendment_may_only_tighten_the_budget():
    base = make_charter()
    tighter = make_charter(k_max=5)
    looser = make_charter(k_max=20)
    sealing.check_amendment(base, tighter)
    with pytest.raises(errors.ConfigError, match="tighten"):
        sealing.check_amendment(base, looser)
    other_question = make_charter(question="Something else entirely?")
    with pytest.raises(errors.ConfigError, match="tighten"):
        sealing.check_amendment(base, other_question)


def test_cli_init_writes_a_template_that_fails_check(tmp_path):
    path = tmp_path / "charter.json"
    assert cli.main(["charter", "init", str(path)]) == cli.EXIT_OK
    assert path.exists()
    assert cli.main(["charter", "check", str(path)]) == cli.EXIT_FAILED


def test_cli_check_and_sign_a_valid_charter(tmp_path, monkeypatch):
    path = tmp_path / "charter.json"
    path.write_text(json.dumps(make_charter().to_dict()))
    monkeypatch.setenv("SCMLE_SIGNING_KEY", KEY)
    assert cli.main(["charter", "check", str(path)]) == cli.EXIT_OK
    signed = tmp_path / "signed.json"
    assert (
        cli.main(["charter", "sign", str(path), "--output", str(signed)])
        == cli.EXIT_OK
    )
    assert sealing.read_verified(signed, signing_key=KEY).signed is True
    assert (
        cli.main(["charter", "check", str(signed), "--rows", "400"])
        == cli.EXIT_OK
    )


def test_cli_sign_without_a_key_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("SCMLE_SIGNING_KEY", raising=False)
    path = tmp_path / "charter.json"
    path.write_text(json.dumps(make_charter().to_dict()))
    assert cli.main(["charter", "sign", str(path)]) == cli.EXIT_FAILED


def test_charter_is_a_plain_dict_when_unsigned(tmp_path):
    path = tmp_path / "analysis_charter.json"
    sealing.write_charter(path, make_charter())
    record = json.loads(path.read_text())
    assert record["charter_hash"] == make_charter().charter_hash
    assert "signature" not in record or record["signature"] is None
    assert (
        contracts.AnalysisCharter.from_dict(record["charter"]) == make_charter()
    )


def test_run_and_plan_accept_a_charter_option():
    for command in ("run", "plan"):
        args = cli.build_parser().parse_args(
            [command, "data.csv", "--target", "y", "--charter", "c.json"]
        )
        assert args.charter == "c.json"
    default = cli.build_parser().parse_args(
        ["run", "data.csv", "--target", "y"]
    )
    assert default.charter is None
