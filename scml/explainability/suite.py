"""Explanation diagnostics and portable evidence construction."""

from collections.abc import Mapping
from collections.abc import Sequence
import datetime
from typing import Any

from scml.contracts import explanations
from scml.evaluation import diagnostics
from scml.explainability import dependence
from scml.explainability import importance
from scml.explainability import interaction


class ExplanationEvidenceBuilder:
    """Builds portable evidence for any explanation diagnostic.

    The builder accepts the polymorphic ``Diagnostic`` interface, so every
    explanation implementation can produce the same reviewable evidence
    record without knowing about model bundles, dataset storage, or feature
    graph implementations. All identity and interpretation fields are
    explicit inputs; the builder never derives them from an estimator or
    diagnostic output.
    """

    def build(
        self,
        diagnostic: diagnostics.Diagnostic,
        result: Mapping[str, Any],
        *,
        model_fingerprint: str,
        bundle_digest: str,
        input_fingerprint: str,
        dataset_snapshot: explanations.DatasetSnapshot,
        feature_graph: explanations.FeatureGraphReference,
        method_parameters: Mapping[str, Any],
        timestamp: datetime.datetime,
        software_version: str,
        uncertainty: Mapping[str, Any],
        stability: Mapping[str, Any],
        limitations: Sequence[str],
        causal_warning: str,
        privacy_decision: explanations.PrivacyDecision,
        evidence_links: Sequence[explanations.EvidenceLink],
    ) -> explanations.ExplanationEvidence:
        """Builds deterministic explanation evidence from a diagnostic result.

        Args:
            diagnostic: Explanation diagnostic that produced ``result``.
            result: JSON-compatible diagnostic output.
            model_fingerprint: Explicit identity of the explained model.
            bundle_digest: Explicit digest of the model bundle.
            input_fingerprint: Explicit identity of the explained input.
            dataset_snapshot: Explicit dataset identity and version.
            feature_graph: Explicit preprocessing graph identity.
            method_parameters: Resolved parameters used by the diagnostic.
            timestamp: A timezone-aware explanation creation time.
            software_version: Version of the producing software.
            uncertainty: Uncertainty evidence for the explanation.
            stability: Stability evidence for the explanation.
            limitations: Known limitations that travel with the result.
            causal_warning: Causal interpretation boundary.
            privacy_decision: Disclosure and redaction decision.
            evidence_links: References to independently reviewable evidence.

        Returns:
            A validated, portable explanation evidence record.

        Raises:
            ValueError: If the diagnostic does not declare a usable name.
            pydantic.ValidationError: If any explicit identity or evidence
                field violates the portable explanation contract.
        """
        method_name = getattr(diagnostic, "name", None)
        if not isinstance(method_name, str) or not method_name.strip():
            raise ValueError("diagnostic.name must be a non-empty string")
        if isinstance(limitations, str):
            raise TypeError("limitations must be a sequence of strings")
        if isinstance(evidence_links, str):
            raise TypeError(
                "evidence_links must be a sequence of EvidenceLink values"
            )

        return explanations.ExplanationEvidence(
            model_fingerprint=model_fingerprint,
            bundle_digest=bundle_digest,
            input_fingerprint=input_fingerprint,
            dataset_snapshot=dataset_snapshot,
            feature_graph=feature_graph,
            method=explanations.ExplanationMethod(
                name=method_name,
                parameters=dict(method_parameters),
            ),
            output=dict(result),
            timestamp=timestamp,
            software_version=software_version,
            uncertainty=dict(uncertainty),
            stability=dict(stability),
            limitations=tuple(limitations),
            causal_warning=causal_warning,
            privacy_decision=privacy_decision,
            evidence_links=tuple(evidence_links),
        )


def default_suite() -> tuple[diagnostics.Diagnostic, ...]:
    """Returns the standard explanation diagnostics."""
    return (
        importance.PermutationImportance(),
        dependence.PartialDependence(),
        interaction.PairwiseInteraction(),
    )
