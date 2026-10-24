"""Tests for safe human-readable explanation evidence rendering."""

import datetime

from scmle.artifacts import explanation_report
from scmle.contracts import explanations


def evidence(
    *, decision: explanations.PrivacyStatus = "allowed", output: object
) -> explanations.ExplanationEvidence:
    """Builds a compact explanation record for renderer tests."""
    return explanations.ExplanationEvidence(
        model_fingerprint="model-123",
        bundle_digest="sha256:bundle-123",
        input_fingerprint="input-123",
        dataset_snapshot=explanations.DatasetSnapshot(
            identifier="demo",
            version="v1",
            digest="sha256:data-123",
        ),
        feature_graph=explanations.FeatureGraphReference(
            version=1, fingerprint="graph-123"
        ),
        method=explanations.ExplanationMethod(
            name="permutation_importance", parameters={"repeats": 5}
        ),
        output=output,
        timestamp=datetime.datetime(2026, 10, 21, tzinfo=datetime.UTC),
        software_version="0.1.0b1",
        uncertainty={"status": "not_estimated"},
        stability={"status": "not_estimated"},
        limitations=("Associational evidence is not causal evidence.",),
        causal_warning="This explanation is associational.",
        privacy_decision=explanations.PrivacyDecision(
            decision=decision,
            reason="The output was reviewed for disclosure.",
        ),
        evidence_links=(
            explanations.EvidenceLink(
                relation="diagnostic-report",
                uri="diagnostics/permutation_importance.json",
            ),
        ),
    )


def test_renderer_escapes_output_and_keeps_links_non_navigable():
    rendered = explanation_report.ExplanationHtmlReport().render(
        evidence(output={"text": "<script>alert('&')</script>"})
    )

    assert "<script>alert('&')</script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "&#x27;&amp;&#x27;" in rendered
    assert "<a href=" not in rendered
    assert "diagnostics/permutation_importance.json" in rendered


def test_renderer_withholds_denied_output():
    rendered = explanation_report.ExplanationHtmlReport().render(
        evidence(decision="denied", output={"secret": "do-not-render"})
    )

    assert "do-not-render" not in rendered
    assert "&quot;status&quot;: &quot;withheld&quot;" in rendered
    assert "Privacy decision: denied" in rendered
