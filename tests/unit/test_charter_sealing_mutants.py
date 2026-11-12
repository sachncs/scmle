"""Pins the format boundaries and the write call of the charter envelope."""

from pathlib import Path

import pytest

from scmle.contracts import errors
from scmle.data.charter import contracts
from scmle.data.charter import sealing
from tests.unit.test_charter_contracts import make_charter


def accepted(record):
    sealing.check_format("c.json", record)


def refused(record):
    with pytest.raises(errors.ConfigError) as caught:
        sealing.check_format("c.json", record)
    return str(caught.value)


def test_the_current_format_is_readable_and_the_next_one_is_not():
    accepted({"format": contracts.FORMAT_VERSION})
    message = refused({"format": contracts.FORMAT_VERSION + 1})
    assert message == (
        f"charter c.json has format {contracts.FORMAT_VERSION + 1}; this "
        f"version reads formats up to {contracts.FORMAT_VERSION}"
    )


def test_a_block_is_accepted_in_the_format_that_introduced_it():
    accepted(
        {
            "format": sealing.CAUSAL_FORMAT,
            "charter": {"estimand": {"causal": {"x": 1}}},
        }
    )
    accepted(
        {
            "format": sealing.CONFIRMATION_COVERAGE_FORMAT,
            "charter": {"uncertainty": {"coverage_on_confirmation": {"x": 1}}},
        }
    )
    accepted(
        {
            "format": sealing.SUBGROUP_FORMAT,
            "charter": {
                "estimand": {"causal": {"subgroups": [], "attestors": []}}
            },
        }
    )
    for name in ("entity", "reconcile"):
        accepted(
            {
                "format": sealing.INGEST_ENTITY_FORMAT,
                "charter": {"ingest": {name: {"x": 1}}},
            }
        )


def test_a_block_is_refused_one_format_before_it_was_introduced():
    assert "needs format 4" in refused(
        {"format": 3, "charter": {"estimand": {"causal": {"x": 1}}}}
    )
    assert "needs format 11" in refused(
        {
            "format": 10,
            "charter": {"uncertainty": {"coverage_on_confirmation": {"x": 1}}},
        }
    )
    assert "need format 9" in refused(
        {"format": 8, "charter": {"estimand": {"causal": {"subgroups": []}}}}
    )
    for name in ("entity", "reconcile"):
        message = refused({"format": 7, "charter": {"ingest": {name: {}}}})
        assert message == (
            f"charter c.json declares format 7 but its ingest block carries "
            f"{name!r}, which needs format 8"
        )


def test_each_later_causal_field_is_named_in_the_refusal():
    for name in ("subgroups", "attestors", "attestations"):
        record = {"format": 8, "charter": {"estimand": {"causal": {name: []}}}}
        assert refused(record) == (
            "charter c.json declares format 8 but its causal block carries "
            f"{[name]}, which need format 9"
        )
    record = {
        "format": 8,
        "charter": {
            "estimand": {"causal": {"attestations": 1, "subgroups": 1}}
        },
    }
    assert "['subgroups', 'attestations']" in refused(record)


def test_the_write_call_names_the_encoding_and_the_newline(
    tmp_path, monkeypatch
):
    seen = {}
    original = Path.write_text

    def spy(self, data, *args, **kwargs):
        seen.update(kwargs)
        seen["args"] = args
        return original(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", spy)
    sealing.write_charter(tmp_path / "c.json", make_charter(), "k")
    assert seen == {"args": (), "encoding": "utf-8", "newline": "\n"}
