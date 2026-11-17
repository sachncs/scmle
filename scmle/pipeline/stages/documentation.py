"""Explanation evidence, provenance and the SBOM of a package."""

import datetime
from typing import Any

import scmle
from scmle.artifacts import explanation_report
import scmle.artifacts.bundle
from scmle.contracts import explanations
from scmle.contracts import reliability
import scmle.explainability.suite
from scmle.governance import sbom
import scmle.pipeline.context
from scmle.pipeline.stages import releasing


class Documentation(releasing.Releasing):
    """Explanation evidence, provenance, SBOM and fault-injection plan.

    Part of `PackageStage`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    def explanation_evidence(
        self,
        context: scmle.pipeline.context.Context,
        store: Any,
    ) -> dict[str, dict[str, Any]]:
        """Persists portable evidence for explanations produced by the run.

        Each record is written as JSON, with an HTML rendering, under
        `explanations/`.

        Args:
            context: The run context, holding the diagnostics, manifest and
                split.
            store: The run store; the evidence is written under `explanations/`.

        Returns:
            A mapping from diagnostic name to its evidence record, for each
            explanation diagnostic that ran without error and has a report.
        """
        manifest = scmle.pipeline.context.require(context.manifest, "manifest")
        split = scmle.pipeline.context.require(context.split, "split")
        model_manifest = store.path(
            f"model/{scmle.artifacts.bundle.MANIFEST_FILE}"
        )
        bundle_digest = scmle.artifacts.bundle.shasum(model_manifest)
        snapshot = explanations.Snapshot(
            identifier=manifest.dataset_id,
            version=manifest.snapshot_id or manifest.data_hash,
            digest=manifest.data_hash,
        )
        feature_graph = explanations.Lineage(
            version=context.spec.feature_graph.version,
            fingerprint=context.spec.feature_graph.fingerprint(),
        )
        diagnostics_by_name = {
            diagnostic.name: diagnostic
            for diagnostic in scmle.explainability.suite.panoply()
        }
        evidence: dict[str, dict[str, Any]] = {}
        for name, diagnostic in diagnostics_by_name.items():
            result = context.diagnostics.get(name)
            if (
                result is None
                or "error" in result
                or result.get("available") is False
            ):
                continue
            report_path = store.path(f"diagnostics/{name}.json")
            if not report_path.exists():
                continue
            limitations = [
                "associational evidence is not causal evidence",
                "stability and fidelity require separate validation",
            ]
            if result.get("truncated"):
                limitations.append("feature coverage was truncated by budget")
            record = scmle.explainability.suite.substantiate(
                diagnostic,
                result,
                model_fingerprint=bundle_digest,
                bundle_digest=bundle_digest,
                input_fingerprint=split.manifest.fingerprint,
                dataset_snapshot=snapshot,
                feature_graph=feature_graph,
                method_parameters={"seed": context.spec.validation.seed},
                timestamp=datetime.datetime.now(datetime.UTC),
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
                        uri=f"diagnostics/{name}.json",
                        digest=scmle.artifacts.bundle.shasum(report_path),
                    ),
                ),
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

    def notarise(
        self,
        context: scmle.pipeline.context.Context,
        store: Any,
        bundle: scmle.artifacts.bundle.Bundle,
    ) -> dict[str, Any]:
        """Builds a content-linked provenance record for the packaged run.

        The record is also written to `provenance.json`.

        Args:
            context: The run context, supplying the manifest and configuration.
            store: The run store, supplying the run id and model manifest.
            bundle: The packaged model bundle, supplying the model and
                environment information.

        Returns:
            The provenance record: run id, software version, configuration
            hash, dataset, model manifest digest, environment and signature
            presence.
        """
        manifest = scmle.pipeline.context.require(context.manifest, "manifest")
        model_manifest = store.path(
            f"model/{scmle.artifacts.bundle.MANIFEST_FILE}"
        )
        record = {
            "schema_version": 1,
            "run_id": store.run_id,
            "software_version": scmle.__version__,
            "config_hash": context.spec.config_hash(),
            "dataset": {
                "snapshot_id": manifest.snapshot_id or manifest.data_hash,
                "dataset_id": manifest.dataset_id,
                "data_hash": manifest.data_hash,
                "raw_source_hash": manifest.lineage.object_hashes.get(
                    "raw-source"
                ),
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
        context: scmle.pipeline.context.Context,
        store: Any,
    ) -> dict[str, Any]:
        """Persists the production fault-injection plan, marked as not run.

        Args:
            context: The run context; not consulted.
            store: The run store receiving `fault_injection_plan.json`.

        Returns:
            The fault-injection plan of the production beta, marked as not
            executed with the release blocked, since the fault plan declares
            required evidence and does not claim that the scenarios ran.
        """
        del context
        record = reliability.Registry.prerelease().to_dict()
        record["execution_status"] = "planned"
        record["release_blocked"] = True
        record["limitation"] = (
            "The plan declares required fault evidence; it does not claim "
            "that the scenarios have executed."
        )
        store.write_json("fault_injection_plan.json", record)
        return record
