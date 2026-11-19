"""Release evaluation against explicit success criteria."""

import dataclasses
import logging
from typing import Any

import scmle.artifacts.bundle
from scmle.contracts import records
from scmle.contracts import signing
import scmle.contracts.evaluation
import scmle.contracts.evidence
from scmle.evaluation import policies
from scmle.evaluation import release
from scmle.governance import attestation
from scmle.pipeline import engine
from scmle.pipeline.stages import examination
from scmle.pipeline.stages import ingestion
from scmle.pipeline.stages import interface
from scmle.pipeline.stages import profiling

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Verdict:
    """The release verdict: the policy, its outcome and the evidence.

    Attributes:
        policy: The derived evaluation policy, or None when no success
            criteria were declared.
        outcome: The release evaluation record.
        evidence: The evidence index record.
    """

    policy: records.Layout | None
    outcome: records.Layout
    evidence: records.Layout

    def entries(self) -> records.Layout:
        """The evaluation record's release entries, in record order.

        Returns:
            The release policy, evaluation and evidence entries.
        """
        return {
            "release_policy": self.policy,
            "release_evaluation": self.outcome,
            "release_evidence": self.evidence,
        }


class Releasing(interface.Stage):
    """The success criteria evaluated against the release evidence.

    Part of `PackageStage`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    def evidence_index(
        self,
        board: engine.Board,
        store: Any,
        policy: scmle.contracts.evaluation.Criteria,
    ) -> scmle.contracts.evidence.Index:
        """Collects accepted evidence for artifacts already written.

        Acceptance is deliberately narrow: an automated record is accepted
        only when the referenced artifact exists and its digest is captured.
        The evidence limitation records that an independent human review is
        not represented by this automated run.

        Args:
            board: The run's board, supplying the manifest, task and
                configuration hash.
            store: The run store holding the artifacts.
            policy: The evaluation policy, naming the evidence references that
                a release requires.

        Returns:
            An index with one accepted record for each required reference whose
            artifact exists; missing artifacts are omitted.
        """
        manifest = board.get(ingestion.Intake).manifest
        raw_inputs = {
            "snapshot_id": manifest.snapshot or manifest.digests["data"],
            "config_hash": board.services.spec.config_hash(),
        }
        records: list[scmle.contracts.evidence.Record] = []
        for reference in release.citations(policy):
            path = store.path(reference)
            if not path.exists():
                continue
            records.append(
                scmle.contracts.evidence.Record(
                    criterion=reference,
                    claim=(
                        f"artifact exists and matches run output: {reference}"
                    ),
                    scope={
                        "task": board.get(profiling.Partition).task.value,
                        "modality": manifest.modality,
                    },
                    inputs=raw_inputs,
                    command=("scmle automated release evidence collector"),
                    artifact={
                        "path": reference,
                        "sha256": scmle.artifacts.bundle.shasum(path),
                    },
                    result=scmle.contracts.evidence.Outcome.PASS,
                    reviewer="scmle.automation",
                    observed=board.services.clock.now().isoformat(),
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
        board: engine.Board,
        store: Any,
        evaluation: records.Layout,
    ) -> None:
        """Records the release verdict (`judge`) in the evaluation document.

        Args:
            board: The run's board, holding the task, metric and diagnostics.
            store: The run store receiving the files.
            evaluation: The evaluation document, which must already hold the
                test metrics; the release policy, evaluation and evidence
                entries are added to it.
        """
        evaluation.update(self.judge(board, store, evaluation).entries())

    def judge(
        self,
        board: engine.Board,
        store: Any,
        evaluation: records.Layout,
    ) -> Verdict:
        """Evaluates explicit success criteria against release evidence.

        Writes the release policy, evidence and evaluation files into the run
        store. When no success criteria were declared, the verdict is blocked
        with status `policy_not_declared` and only `release_evaluation.json`
        is written; thresholds are never invented from observed metrics.
        Otherwise the outcome is also gated by the reliability plan, the
        release attestation and the evidence index, which is signed when a
        signing key is configured.

        Args:
            board: The run's board, holding the task, metric and diagnostics.
            store: The run store receiving the files.
            evaluation: The evaluation document, read only; it must already
                hold the test metrics.

        Returns:
            The verdict.
        """
        partition = board.get(profiling.Partition)
        task = partition.task
        metric = partition.metric
        policy = policies.derive(
            task,
            metric.name,
            metric.greater_is_better,
            board.services.spec.problem.success_criteria,
        )
        if policy is None:
            return self.abstain(store)

        evidence_index = self.evidence_index(board, store, policy)
        signing_key = signing.keying(None)
        if signing_key is not None:
            evidence_index.sign(signing_key)
        declared_metrics = {metric.name for metric in policy.metrics}
        release_metrics = {
            name: value
            for name, value in evaluation["test_metrics"].items()
            if name in declared_metrics
        }
        readings = board.get(examination.Readings)
        outcome = release.assess(
            policy,
            release_metrics,
            readings.diagnostics,
            {"fairness": readings.diagnostics.get("fairness", {})},
            evidence_index,
        )
        outcome = release.faults(outcome, evaluation.get("reliability_plan"))
        verification = self.vouch(evaluation)
        outcome = release.certify(
            outcome, evaluation.get("release_attestation"), verification
        )
        evidence_verified = signing_key is not None and evidence_index.verify(
            signing_key
        )
        outcome = release.countersign(
            outcome, evidence_index, evidence_verified
        )
        verdict = Verdict(
            policy.model_dump(mode="json"),
            outcome.to_dict(),
            evidence_index.to_dict(),
        )
        store.write_json(
            "evaluation_policy.json", policy.model_dump(mode="json")
        )
        store.write_json("release_evidence.json", evidence_index.to_dict())
        store.write_json("release_evaluation.json", outcome.to_dict())
        return verdict

    def abstain(self, store: Any) -> Verdict:
        """A release blocked because no success criteria were declared.

        Args:
            store: The run store receiving `release_evaluation.json`.

        Returns:
            The blocked verdict, with no policy and no evidence.
        """
        record = {
            "status": "policy_not_declared",
            "passed": False,
            "blocked": True,
            "blockers": [
                (
                    "no explicit success criteria were declared; release "
                    "thresholds were not invented from observed metrics"
                ),
            ],
            "findings": [],
        }
        store.write_json("release_evaluation.json", record)
        return Verdict(None, record, {"records": [], "ready": False})

    def vouch(self, evaluation: records.Layout) -> bool:
        """Verifies the release attestation against the provenance and SBOM.

        Args:
            evaluation: The evaluation document holding the attestation,
                provenance and SBOM.

        Returns:
            Whether the attestation verifies; False when it cannot be read.
        """
        try:
            release_attestation = attestation.Attestation.from_dict(
                evaluation["release_attestation"]
            )
            return release_attestation.verify(
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
            return False
