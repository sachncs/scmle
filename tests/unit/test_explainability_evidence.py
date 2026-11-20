"""Tests for polymorphic explanation evidence construction."""

import datetime
from typing import Any
from typing import cast
from typing import TypedDict

import pydantic
import pytest

from scmle.contracts import explanations
from scmle.evaluation import diagnostics
from scmle.explainability import suite


class CustomExplanation(diagnostics.Diagnostic):
    """Example diagnostic proving the builder is polymorphic."""

    name = "custom_explanation"

    def compute(self, inputs: diagnostics.Input) -> dict[str, Any]:
        """Returns a representative explanation result."""
        del inputs
        return {"features": {"signal": {"importance": 0.8}}}


class UnnamedExplanation(diagnostics.Diagnostic):
    """Diagnostic without a method identity for rejection testing."""

    def compute(self, inputs: diagnostics.Input) -> dict[str, Any]:
        """Returns an empty result."""
        del inputs
        return {}


class EvidenceInputs(TypedDict):
    """Typed keyword arguments for the evidence builder."""

    model_fingerprint: str
    bundle_digest: str
    input_fingerprint: str
    dataset_snapshot: explanations.Snapshot
    feature_graph: explanations.Lineage
    method_parameters: dict[str, Any]
    timestamp: datetime.datetime
    software_version: str
    uncertainty: dict[str, Any]
    stability: dict[str, Any]
    limitations: tuple[str, ...]
    causal_warning: str
    privacy_decision: explanations.Clearance
    evidence_links: tuple[explanations.Link, ...]


def build_inputs() -> EvidenceInputs:
    """Returns explicit evidence inputs shared by builder tests."""
    return {
        "model_fingerprint": "model-123",
        "bundle_digest": "sha256:bundle-123",
        "input_fingerprint": "input-123",
        "dataset_snapshot": explanations.Snapshot(
            identifier="customer-risk",
            version="2026-10-20",
            digest="sha256:data-123",
        ),
        "feature_graph": explanations.Lineage(
            version=1,
            fingerprint="graph-123",
        ),
        "method_parameters": {"repeats": 5, "seed": 7},
        "timestamp": datetime.datetime(
            2026, 10, 20, 10, 30, tzinfo=datetime.UTC
        ),
        "software_version": "0.1.0b1",
        "uncertainty": {"method": "bootstrap", "confidence": 0.95},
        "stability": {"method": "seed_repeat", "score": 0.98},
        "limitations": ("Associational attribution only.",),
        "causal_warning": "This explanation does not identify causal effects.",
        "privacy_decision": explanations.Clearance(
            decision="allowed",
            reason="No direct identifiers are present.",
        ),
        "evidence_links": (
            explanations.Link(
                relation="evaluation",
                uri="artifact://run/evaluation.json",
            ),
        ),
    }


def substantiate(
    result: dict[str, object] | None = None,
    **overrides: Any,
) -> explanations.Evidence:
    """Builds evidence using the test diagnostic and explicit identities."""
    values = cast("EvidenceInputs", {**build_inputs(), **overrides})
    return suite.substantiate(
        CustomExplanation(),
        result
        if result is not None
        else {"features": {"signal": {"importance": 0.8}}},
        **values,
    )


def test_builder_accepts_any_diagnostic_subclass_and_binds_all_evidence():
    evidence = substantiate()

    assert evidence.method.name == "custom_explanation"
    assert evidence.method.parameters == {"repeats": 5, "seed": 7}
    assert evidence.output["features"] == {"signal": {"importance": 0.8}}
    assert evidence.model_fingerprint == "model-123"
    assert evidence.bundle_digest == "sha256:bundle-123"
    assert evidence.input_fingerprint == "input-123"
    assert evidence.snapshot.digest == "sha256:data-123"
    assert evidence.lineage.fingerprint == "graph-123"
    assert evidence.uncertainty["confidence"] == 0.95
    assert evidence.stability["score"] == 0.98
    assert evidence.limitations == ("Associational attribution only.",)
    assert evidence.clearance.decision == "allowed"
    assert evidence.links[0].uri == "artifact://run/evaluation.json"


def test_builder_output_has_deterministic_canonical_json():
    first = substantiate(
        result={"b": {"z": 2, "a": 1}, "a": 0},
        method_parameters={"z": 2, "a": 1},
        uncertainty={"z": 2, "a": 1},
        stability={"z": 2, "a": 1},
    )
    second = substantiate(
        result={"a": 0, "b": {"a": 1, "z": 2}},
        method_parameters={"a": 1, "z": 2},
        uncertainty={"a": 1, "z": 2},
        stability={"a": 1, "z": 2},
    )

    assert first.canonize() == second.canonize()


def test_builder_rejects_missing_top_level_identity():
    with pytest.raises(pydantic.ValidationError, match="model_fingerprint"):
        substantiate(model_fingerprint="")
    with pytest.raises(pydantic.ValidationError, match="bundle_digest"):
        substantiate(bundle_digest="")
    with pytest.raises(pydantic.ValidationError, match="input_fingerprint"):
        substantiate(input_fingerprint="")
    with pytest.raises(pydantic.ValidationError, match="snapshot"):
        substantiate(dataset_snapshot=None)
    with pytest.raises(pydantic.ValidationError, match="lineage"):
        substantiate(feature_graph=None)


def test_builder_does_not_invent_identity_from_diagnostic_output():
    with pytest.raises(pydantic.ValidationError, match="model_fingerprint"):
        substantiate(
            result={"model_fingerprint": "untrusted-result-value"},
            model_fingerprint="",
        )


def test_builder_rejects_unnamed_diagnostics():
    with pytest.raises(ValueError, match=r"diagnostic\.name"):
        suite.substantiate(
            UnnamedExplanation(),
            {"value": 1},
            **build_inputs(),
        )


def test_default_suite_includes_bounded_interaction_evidence():
    diagnostics_in_suite = suite.panoply()

    assert tuple(diagnostic.name for diagnostic in diagnostics_in_suite) == (
        "permutation_importance",
        "partial_dependence",
        "pairwise_interaction",
    )
    assert all(
        isinstance(diagnostic, diagnostics.Diagnostic)
        for diagnostic in diagnostics_in_suite
    )
