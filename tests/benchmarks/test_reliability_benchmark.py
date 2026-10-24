"""Tests for the reliability benchmark runner."""

import json

from benchmarks import reliability


def test_every_check_states_its_claim():
    for name, check in reliability.CHECKS.items():
        assert check.__doc__, name


def test_a_crashing_check_is_a_failed_record_not_an_exception(monkeypatch):
    def broken(directory):
        raise RuntimeError("boom")

    monkeypatch.setitem(reliability.CHECKS, "determinism", broken)

    record = reliability.run_check("determinism")

    assert not record.passed
    assert "boom" in record.error


def test_the_runner_writes_results_and_gates_on_failure(tmp_path, monkeypatch):
    output = tmp_path / "out.json"
    arguments = ["--only", "bundle_roundtrip", "--output", str(output)]

    status = reliability.main(arguments)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert [r["name"] for r in payload["results"]] == ["bundle_roundtrip"]
    assert payload["results"][0]["passed"] is True

    monkeypatch.setitem(
        reliability.CHECKS,
        "bundle_roundtrip",
        lambda directory: reliability.CheckRecord("bundle_roundtrip", ""),
    )
    assert reliability.main(arguments) == 1
