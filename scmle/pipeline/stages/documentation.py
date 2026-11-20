"""Explanation evidence, provenance and the SBOM of a package."""

from typing import Any

import scmle
from scmle.artifacts import explanation_report
import scmle.artifacts.bundle
from scmle.contracts import explanations
from scmle.contracts import reliability
import scmle.explainability.suite
from scmle.governance import sbom
from scmle.pipeline import engine
from scmle.pipeline.stages import examination
from scmle.pipeline.stages import ingestion
from scmle.pipeline.stages import profiling
from scmle.pipeline.stages import releasing


class Documentation(releasing.Releasing):
    """Explanation evidence, provenance, SBOM and fault-injection plan.

    Part of `Packaging`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    def explain(
        self,
        board: engine.Board,
        store: Any,
    ) -> dict[str, dict[str, Any]]:
        """Persists portable evidence for explanations produced by the run.

        Each record is written as JSON, with an HTML rendering, under
        `explanations/`.

        Args:
            board: The run's board, holding the diagnostics, manifest and
                split.
            store: The run store; the evidence is written under `explanations/`.

        Returns:
            A mapping from diagnostic name to its evidence record, for each
            explanation diagnostic that ran without error and has a report.
        """
        manifest = board.get(ingestion.Intake).manifest
        model_manifest = store.path(
            f"model/{scmle.artifacts.bundle.MANIFEST_FILE}"
        )
        bundle_digest = scmle.artifacts.bundle.shasum(model_manifest)
        snapshot = explanations.Snapshot(
            identifier=manifest.identifier,
            version=manifest.snapshot or manifest.digests["data"],
            digest=manifest.digests["data"],
        )
        feature_graph = explanations.Lineage(
            version=board.services.spec.feature_graph.version,
            fingerprint=board.services.spec.feature_graph.fingerprint(),
        )
        diagnostics_by_name = {
            diagnostic.name: diagnostic
            for diagnostic in scmle.explainability.suite.panoply()
        }
        evidence: dict[str, dict[str, Any]] = {}
        for name, diagnostic in diagnostics_by_name.items():
            result = board.get(examination.Readings).diagnostics.get(name)
            if (
                result is None
                or "error" in result
                or result.get("available") is False
            ):
                continue
            report_path = store.path(f"diagnostics/{name}.json")
            if not report_path.exists():
                continue
            record = self.substantiate(
                board,
                diagnostic,
                (report_path, bundle_digest),
                snapshot,
                feature_graph,
            )
            evidence[name] = record.model_dump(mode="json")
            store.write_json(f"explanations/{name}.json", evidence[name])
            store.write_text(
                f"explanations/{name}.html",
                explanation_report.Renderer().render(
                    record, title=f"{name} explanation"
                ),
            )
        return evidence

    def substantiate(
        self,
        board: engine.Board,
        diagnostic: Any,
        report: tuple[Any, str],
        snapshot: explanations.Snapshot,
        feature_graph: explanations.Lineage,
    ) -> Any:
        """Builds the evidence record of one explanation diagnostic.

        Args:
            board: The run's board, holding the diagnostics and split.
            diagnostic: The explanation diagnostic.
            report: The path of its report and the digest of the bundle
                manifest.
            snapshot: The dataset snapshot explained.
            feature_graph: The feature graph's version and fingerprint.

        Returns:
            The evidence record.
        """
        split = board.get(profiling.Partition).split
        result = board.get(examination.Readings).diagnostics[diagnostic.name]
        report_path, bundle_digest = report
        limitations = [
            "associational evidence is not causal evidence",
            "stability and fidelity require separate validation",
        ]
        if result.get("truncated"):
            limitations.append("feature coverage was truncated by budget")
        return scmle.explainability.suite.substantiate(
            diagnostic,
            result,
            model_fingerprint=bundle_digest,
            bundle_digest=bundle_digest,
            input_fingerprint=split.manifest.fingerprint,
            dataset_snapshot=snapshot,
            feature_graph=feature_graph,
            method_parameters={"seed": board.services.spec.validation.seed},
            timestamp=board.services.clock.now(),
            software_version=scmle.__version__,
            uncertainty={
                "status": "not_estimated",
                "reason": "explanation evidence does not estimate "
                "prediction uncertainty",
            },
            stability={
                "status": "not_estimated",
                "reason": "repeatability must be evaluated separately",
            },
            limitations=tuple(limitations),
            causal_warning=(
                "This explanation describes model associations; it does "
                "not establish that changing a feature causes an outcome."
            ),
            privacy_decision=explanations.Clearance(
                decision="allowed",
                reason=(
                    "diagnostic output contains aggregate feature evidence"
                ),
            ),
            evidence_links=(
                explanations.Link(
                    relation="diagnostic-report",
                    uri=f"diagnostics/{diagnostic.name}.json",
                    digest=scmle.artifacts.bundle.shasum(report_path),
                ),
            ),
        )

    def notarise(
        self,
        board: engine.Board,
        store: Any,
        bundle: scmle.artifacts.bundle.Bundle,
    ) -> dict[str, Any]:
        """Builds a content-linked provenance record for the packaged run.

        The record is also written to `provenance.json`.

        Args:
            board: The run's board, supplying the manifest and configuration.
            store: The run store, supplying the run id and model manifest.
            bundle: The packaged model bundle, supplying the model and
                environment information.

        Returns:
            The provenance record: run id, software version, configuration
            hash, dataset, model manifest digest, environment and signature
            presence.
        """
        manifest = board.get(ingestion.Intake).manifest
        model_manifest = store.path(
            f"model/{scmle.artifacts.bundle.MANIFEST_FILE}"
        )
        record = {
            "schema_version": 1,
            "run_id": store.run_id,
            "software_version": scmle.__version__,
            "config_hash": board.services.spec.config_hash(),
            "dataset": {
                "snapshot_id": manifest.snapshot or manifest.digests["data"],
                "dataset_id": manifest.identifier,
                "data_hash": manifest.digests["data"],
                "raw_source_hash": manifest.lineage.digests.get("raw-source"),
            },
            "model": {
                "manifest_path": (
                    f"model/{scmle.artifacts.bundle.MANIFEST_FILE}"
                ),
                "manifest_sha256": scmle.artifacts.bundle.shasum(
                    model_manifest
                ),
                "model_name": bundle.info.model_name,
                "model_version": bundle.info.model_version,
            },
            "environment": {
                "python": bundle.info.python,
                "platform": bundle.info.platform,
                "libraries": bundle.info.libraries,
                "lock": bundle.info.environment_lock(),
            },
            "signature": {
                "path": f"model/{scmle.artifacts.bundle.SIGNATURE_FILE}",
                "present": store.path(
                    f"model/{scmle.artifacts.bundle.SIGNATURE_FILE}"
                ).exists(),
            },
        }
        store.write_json("provenance.json", record)
        return record

    def inventory(
        self,
        store: Any,
        bundle: scmle.artifacts.bundle.Bundle,
    ) -> dict[str, Any]:
        """Builds and persists the packaged environment's SBOM.

        Args:
            store: The run store receiving `sbom.json`.
            bundle: The packaged model bundle, supplying the environment.

        Returns:
            The software bill of materials.
        """
        record = sbom.build(
            bundle.info.environment,
            run_id=store.run_id,
            software_version=scmle.__version__,
        )
        store.write_json("sbom.json", record)
        return record

    def persist(
        self,
        board: engine.Board,
        store: Any,
    ) -> dict[str, Any]:
        """Persists the production fault-injection plan, marked as not run.

        Args:
            board: The run's board; not consulted.
            store: The run store receiving `fault_injection_plan.json`.

        Returns:
            The fault-injection plan of the production beta, marked as not
            executed with the release blocked, since the fault plan declares
            required evidence and does not claim that the scenarios ran.
        """
        del board
        record = reliability.Registry.prerelease().to_dict()
        record["execution_status"] = "planned"
        record["release_blocked"] = True
        record["limitation"] = (
            "The plan declares required fault evidence; it does not claim "
            "that the scenarios have executed."
        )
        store.write_json("fault_injection_plan.json", record)
        return record
