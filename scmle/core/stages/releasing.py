"""Release evaluation against explicit success criteria."""

import datetime
import logging
from typing import Any

import scmle.artifacts.bundle
from scmle.contracts import signing
import scmle.contracts.evaluation
import scmle.contracts.evidence
import scmle.core.context
from scmle.core.stages import interface
from scmle.evaluation import policies
from scmle.evaluation import release
from scmle.governance import attestation

LOGGER = logging.getLogger(__name__)


class Releasing(interface.Stage):
    """The success criteria evaluated against the release evidence.

    Part of `PackageStage`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    def evidence_index(
        self,
        context: scmle.core.context.Context,
        store: Any,
        policy: scmle.contracts.evaluation.Criteria,
    ) -> scmle.contracts.evidence.Index:
        """Collects accepted evidence for artifacts already written.

        Acceptance is deliberately narrow: an automated record is accepted
        only when the referenced artifact exists and its digest is captured.
        The evidence limitation records that an independent human review is
        not represented by this automated run.

        Args:
            context: The run context, supplying the manifest, task and
                configuration hash.
            store: The run store holding the artifacts.
            policy: The evaluation policy, naming the evidence references that
                a release requires.

        Returns:
            An index with one accepted record for each required reference whose
            artifact exists; missing artifacts are omitted.
        """
        manifest = scmle.core.context.require(context.manifest, "manifest")
        raw_inputs = {
            "snapshot_id": manifest.snapshot_id or manifest.data_hash,
            "config_hash": context.spec.config_hash(),
        }
        records: list[scmle.contracts.evidence.Record] = []
        for reference in release.citations(policy):
            path = store.path(reference)
            if not path.exists():
                continue
            records.append(
                scmle.contracts.evidence.Record(
                    criterion_id=reference,
                    claim=(
                        f"artifact exists and matches run output: {reference}"
                    ),
                    scope={
                        "task": scmle.core.context.require(
                            context.task, "task"
                        ).value,
                        "modality": manifest.modality,
                    },
                    raw_inputs=raw_inputs,
                    command_or_test=(
                        "scmle automated release evidence collector"
                    ),
                    artifact={
                        "path": reference,
                        "sha256": scmle.artifacts.bundle.shasum(path),
                    },
                    result=scmle.contracts.evidence.Outcome.PASS,
                    reviewer="scmle.automation",
                    observed_at=datetime.datetime.now(datetime.UTC).isoformat(),
                    limitations=(
                        (
                            "automated artifact-presence and digest check; "
                            "independent human review is not represented"
                        ),
                    ),
                    decision=scmle.contracts.evidence.Disposition.ACCEPTED,
                )
            )
        return scmle.contracts.evidence.Index(records)

    def release_evaluation(
        self,
        context: scmle.core.context.Context,
        store: Any,
        evaluation: dict[str, Any],
    ) -> None:
        """Evaluates explicit success criteria against release evidence.

        Writes the release policy, evidence and evaluation files into the run
        store and records the results in `evaluation`. When no success criteria
        were declared, the evaluation is blocked with status
        `policy_not_declared` and only `release_evaluation.json` is written;
        thresholds are never invented from observed metrics. Otherwise the
        outcome is also gated by the reliability plan, the release attestation
        and the evidence index, which is signed when a signing key is
        configured.

        Args:
            context: The run context, holding the task, metric and diagnostics.
            store: The run store receiving the files.
            evaluation: The evaluation document, which must already hold the
                test metrics; the release policy, evaluation and evidence
                entries are added to it.
        """
        task = scmle.core.context.require(context.task, "task")
        metric = scmle.core.context.require(context.metric, "metric")
        policy = policies.derive(
            task,
            metric.name,
            metric.greater_is_better,
            context.spec.problem.success_criteria,
        )
        if policy is None:
            record = {
                "status": "policy_not_declared",
                "passed": False,
                "blocked": True,
                "blocking_reasons": [
                    (
                        "no explicit success criteria were declared; release "
                        "thresholds were not invented from observed metrics"
                    ),
                ],
                "findings": [],
            }
            evaluation["release_policy"] = None
            evaluation["release_evaluation"] = record
            evaluation["release_evidence"] = {"records": [], "ready": False}
            store.write_json("release_evaluation.json", record)
            return

        evidence_index = self.evidence_index(context, store, policy)
        signing_key = signing.keying(None)
        if signing_key is not None:
            evidence_index.sign(signing_key)
        declared_metrics = {metric.name for metric in policy.metrics}
        release_metrics = {
            name: value
            for name, value in evaluation["test_metrics"].items()
            if name in declared_metrics
        }
        outcome = release.assess(
            policy,
            release_metrics,
            context.diagnostics,
            {"fairness": context.diagnostics.get("fairness", {})},
            evidence_index,
        )
        outcome = release.faults(outcome, evaluation.get("reliability_plan"))
        verification = False
        try:
            release_attestation = attestation.Attestation.from_dict(
                evaluation["release_attestation"]
            )
            verification = release_attestation.verify(
                evaluation["provenance"],
                evaluation["sbom"],
                signing_key=signing.keying(None),
            )
        except (KeyError, TypeError, ValueError) as error:
            # handled: verification=False fed to attestation gate, which
            # blocks; the reason is logged, not discarded (ERR-4)
            LOGGER.warning(
                "release attestation could not be checked (%s: %s); the "
                "release is blocked",
                type(error).__name__,
                error,
            )
            verification = False
        outcome = release.certify(
            outcome, evaluation.get("release_attestation"), verification
        )
        evidence_verified = signing_key is not None and evidence_index.verify(
            signing_key
        )
        outcome = release.countersign(
            outcome, evidence_index, evidence_verified
        )
        evaluation["release_policy"] = policy.to_dict()
        evaluation["release_evaluation"] = outcome.to_dict()
        evaluation["release_evidence"] = evidence_index.to_dict()
        store.write_json("evaluation_policy.json", policy.to_dict())
        store.write_json("release_evidence.json", evidence_index.to_dict())
        store.write_json("release_evaluation.json", outcome.to_dict())
