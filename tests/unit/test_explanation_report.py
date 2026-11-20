"""Tests for safe human-readable explanation evidence rendering."""

import datetime

from scmle.artifacts import explanation_report
from scmle.contracts import explanations


def evidence(
    *, decision: explanations.PrivacyStatus = "allowed", output: object
) -> explanations.Evidence:
    """Builds a compact explanation record for renderer tests."""
    return explanations.Evidence(
        model_fingerprint="model-123",
        bundle_digest="sha256:bundle-123",
        input_fingerprint="input-123",
        snapshot=explanations.Snapshot(
            identifier="demo",
            version="v1",
            digest="sha256:data-123",
        ),
        lineage=explanations.Lineage(version=1, fingerprint="graph-123"),
        method=explanations.Method(
            name="permutation_importance", parameters={"repeats": 5}
        ),
        output=output,
        timestamp=datetime.datetime(2026, 10, 21, tzinfo=datetime.UTC),
        software="0.1.0b1",
        uncertainty={"status": "not_estimated"},
        stability={"status": "not_estimated"},
        limitations=("Associational evidence is not causal evidence.",),
        caveat="This explanation is associational.",
        clearance=explanations.Clearance(
            decision=decision,
            reason="The output was reviewed for disclosure.",
        ),
        links=(
            explanations.Link(
                relation="diagnostic-report",
                uri="diagnostics/permutation_importance.json",
            ),
        ),
    )


def test_renderer_escapes_output_and_keeps_links_non_navigable():
    rendered = explanation_report.Renderer().render(
        evidence(output={"text": "<script>alert('&')</script>"})
    )

    assert "<script>alert('&')</script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "&#x27;&amp;&#x27;" in rendered
    assert "<a href=" not in rendered
    assert "diagnostics/permutation_importance.json" in rendered


def test_renderer_withholds_denied_output():
    rendered = explanation_report.Renderer().render(
        evidence(decision="denied", output={"secret": "do-not-render"})
    )

    assert "do-not-render" not in rendered
    assert "&quot;status&quot;: &quot;withheld&quot;" in rendered
    assert "Privacy decision: denied" in rendered
