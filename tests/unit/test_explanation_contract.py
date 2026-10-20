"""Tests for the portable explanation evidence contract."""

import datetime
import json

import pydantic
import pytest

from scml.contracts import explanations


def build_evidence(**overrides: object) -> explanations.ExplanationEvidence:
    """Builds a valid evidence record for contract tests."""
    record: dict[str, object] = {
        "model_fingerprint": "model-123",
        "bundle_digest": "sha256:bundle-123",
        "input_fingerprint": "input-123",
        "dataset_snapshot": {
            "identifier": "customer-risk",
            "version": "2026-10-20",
            "digest": "sha256:data-123",
        },
        "feature_graph": {"version": 1, "fingerprint": "graph-123"},
        "method": {
            "name": "permutation_importance",
            "parameters": {"repeats": 5, "seed": 7},
        },
        "output": {"features": [{"name": "age", "importance": 0.4}]},
        "timestamp": datetime.datetime(
            2026, 10, 20, 10, 30, tzinfo=datetime.UTC
        ),
        "software_version": "0.1.0b1",
        "uncertainty": {"method": "bootstrap", "confidence": 0.95},
        "stability": {"method": "seed_repeat", "score": 0.98},
        "limitations": ("Associational attribution only.",),
        "causal_warning": "This explanation does not identify causal effects.",
        "privacy_decision": {
            "decision": "allowed",
            "reason": "No direct identifiers are present.",
        },
        "evidence_links": (
            {"relation": "evaluation", "uri": "artifact://run/evaluation.json"},
        ),
    }
    record.update(overrides)
    return explanations.ExplanationEvidence.model_validate(record)


def test_evidence_contains_every_phase_four_binding():
    evidence = build_evidence()

    record = evidence.to_dict()

    assert record["model_fingerprint"] == "model-123"
    assert record["bundle_digest"] == "sha256:bundle-123"
    assert record["input_fingerprint"] == "input-123"
    assert record["dataset_snapshot"]["digest"] == "sha256:data-123"
    assert record["feature_graph"]["fingerprint"] == "graph-123"
    assert record["method"]["parameters"] == {"repeats": 5, "seed": 7}
    assert record["output"]["features"][0]["importance"] == 0.4
    assert record["timestamp"] == "2026-10-20T10:30:00Z"
    assert record["software_version"] == "0.1.0b1"
    assert record["uncertainty"]["confidence"] == 0.95
    assert record["stability"]["score"] == 0.98
    assert record["limitations"] == ["Associational attribution only."]
    assert record["causal_warning"].startswith("This explanation")
    assert record["privacy_decision"]["decision"] == "allowed"
    assert record["evidence_links"][0]["uri"].startswith("artifact:")


def test_canonical_json_is_deterministic_and_html_safe():
    first = build_evidence(output={"text": "<script>&"})
    second = build_evidence(output={"text": "<script>&"})

    serialized = first.canonical_json()

    assert serialized == second.canonical_json()
    assert "<script>" not in serialized
    assert "&" not in serialized
    assert "\\u003cscript\\u003e\\u0026" in serialized
    assert json.loads(serialized)["output"] == {"text": "<script>&"}


def test_timestamp_is_normalized_to_utc():
    evidence = build_evidence(
        timestamp=datetime.datetime(
            2026,
            10,
            20,
            12,
            30,
            tzinfo=datetime.timezone(datetime.timedelta(hours=2)),
        )
    )

    assert evidence.timestamp == datetime.datetime(
        2026, 10, 20, 10, 30, tzinfo=datetime.UTC
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_fingerprint", ""),
        ("bundle_digest", "   "),
        ("input_fingerprint", ""),
        ("software_version", ""),
        ("causal_warning", ""),
        ("limitations", ()),
        ("evidence_links", ()),
        ("uncertainty", {}),
        ("stability", {}),
    ],
)
def test_required_fields_cannot_be_silent_or_empty(field: str, value: object):
    with pytest.raises((pydantic.ValidationError, ValueError)):
        build_evidence(**{field: value})


def test_naive_timestamp_is_rejected():
    with pytest.raises(pydantic.ValidationError, match="timezone"):
        build_evidence(timestamp=datetime.datetime(2026, 10, 20, 10, 30))


@pytest.mark.parametrize(
    "uri", ["javascript:alert(1)", "data:text/plain,secret"]
)
def test_unsafe_evidence_links_are_rejected(uri: str):
    with pytest.raises(pydantic.ValidationError, match="scheme"):
        build_evidence(evidence_links=({"relation": "source", "uri": uri},))


def test_non_json_output_is_rejected():
    with pytest.raises(pydantic.ValidationError, match="JSON-native"):
        build_evidence(output={"importance": object()})


def test_non_finite_measurement_is_rejected():
    with pytest.raises(pydantic.ValidationError, match="non-finite"):
        build_evidence(uncertainty={"confidence": float("nan")})


def test_unknown_fields_are_rejected():
    with pytest.raises(pydantic.ValidationError, match="extra"):
        build_evidence(unexpected="not part of the contract")
