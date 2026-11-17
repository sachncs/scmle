"""The content-linked graph of a completed package."""

import hashlib
from typing import Any

import scmle.artifacts.bundle
from scmle.contracts import lineage
import scmle.core.context
from scmle.core.stages import bracketing


class Tracing(bracketing.Bracketing):
    """The content-linked graph and the artifact checksums.

    Part of `PackageStage`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    def trace(
        self,
        context: scmle.core.context.Context,
        store: Any,
    ) -> lineage.Graph:
        """Builds the content-linked graph for one completed package.

        The graph is validated and kept on the context.

        Args:
            context: The run context, supplying the manifest, split and trial.
            store: The run store holding the artifacts that are hashed; the
                curation report, pipeline, evaluation and bundle manifest
                must exist.

        Returns:
            The graph from the raw source through the snapshot, curation,
            optional feature graph, split, model and evaluation to the bundle,
            with an explanations node when explanation files exist.
        """
        manifest = scmle.core.context.require(context.manifest, "manifest")
        split = scmle.core.context.require(context.split, "split")
        selected = scmle.core.context.require(context.selected, "selected")
        graph = lineage.Graph()

        raw_digest = manifest.lineage.object_hashes.get(
            "raw-source", manifest.data_hash
        )
        graph.add_node(
            lineage.Node(
                "raw-source",
                lineage.Kind.RAW_SOURCE,
                raw_digest,
                {"uri": manifest.uri},
            )
        )
        graph.add_node(
            lineage.Node(
                "snapshot",
                lineage.Kind.SNAPSHOT,
                manifest.snapshot_id or manifest.data_hash,
                {"dataset_id": manifest.dataset_id},
            )
        )
        graph.connect(lineage.Edge("raw-source", "snapshot", "materializes"))
        graph.add_node(
            lineage.Node(
                "curation",
                lineage.Kind.CURATION,
                scmle.artifacts.bundle.shasum(
                    store.path("curation_report.json")
                ),
                {"artifact": "curation_report.json"},
            )
        )
        graph.connect(lineage.Edge("snapshot", "curation", "curated"))

        if context.spec.feature_graph.nodes:
            graph.add_node(
                lineage.Node(
                    "feature-graph",
                    lineage.Kind.FEATURE_GRAPH,
                    context.spec.feature_graph.fingerprint(),
                    {"version": str(context.spec.feature_graph.version)},
                )
            )
            graph.connect(
                lineage.Edge("curation", "feature-graph", "transforms")
            )
            split_parent = "feature-graph"
        else:
            split_parent = "curation"

        graph.add_node(
            lineage.Node(
                "split",
                lineage.Kind.SPLIT,
                split.manifest.fingerprint,
                {"artifact": "validation_manifest.json"},
            )
        )
        graph.connect(lineage.Edge(split_parent, "split", "partitions"))

        pipeline_path = store.path("model/pipeline.joblib")
        model_manifest_path = store.path(
            f"model/{scmle.artifacts.bundle.MANIFEST_FILE}"
        )
        graph.add_node(
            lineage.Node(
                "model",
                lineage.Kind.MODEL,
                scmle.artifacts.bundle.shasum(pipeline_path),
                {"model_family": selected.model_name},
            )
        )
        graph.connect(lineage.Edge("split", "model", "trains"))
        graph.add_node(
            lineage.Node(
                "evaluation",
                lineage.Kind.EVALUATION,
                scmle.artifacts.bundle.shasum(store.path("evaluation.json")),
                {"artifact": "evaluation.json"},
            )
        )
        graph.connect(lineage.Edge("model", "evaluation", "evaluates"))

        explanation_paths = sorted(
            store.path("explanations").glob("*.json")
            if store.path("explanations").exists()
            else ()
        )
        if explanation_paths:
            explanation_identity = hashlib.sha256(
                "\n".join(
                    f"{path.name}:{scmle.artifacts.bundle.shasum(path)}"
                    for path in explanation_paths
                ).encode("utf-8")
            ).hexdigest()
            graph.add_node(
                lineage.Node(
                    "explanations",
                    lineage.Kind.EXPLANATION,
                    explanation_identity,
                    {"artifact_count": str(len(explanation_paths))},
                )
            )
            graph.connect(lineage.Edge("model", "explanations", "explains"))
            graph.connect(
                lineage.Edge("evaluation", "explanations", "contextualizes")
            )

        graph.add_node(
            lineage.Node(
                "bundle",
                lineage.Kind.BUNDLE,
                scmle.artifacts.bundle.shasum(model_manifest_path),
                {"artifact": "model/" + scmle.artifacts.bundle.MANIFEST_FILE},
            )
        )
        graph.connect(lineage.Edge("model", "bundle", "packages"))
        graph.connect(lineage.Edge("evaluation", "bundle", "gates"))
        if explanation_paths:
            graph.connect(lineage.Edge("explanations", "bundle", "includes"))
        graph.validate()
        context.evidence_graph = graph
        return graph

    def lineage(self, store: Any) -> dict[str, str]:
        """Returns checksums of the artifacts a run produced, for lineage.

        Args:
            store: The run store whose artifacts are hashed.

        Returns:
            A mapping from artifact path to its SHA-256 checksum, for the
            known artifacts that exist.
        """
        names = (
            "config.json",
            "dataset_manifest.json",
            "dataset_card.json",
            "model_card.json",
            "source_registry.json",
            "fault_injection_plan.json",
            "provenance.json",
            "sbom.json",
            "release_attestation.json",
            "release_evidence.json",
            "parameter_estimation.json",
            "evidence_graph.json",
            "evaluation.json",
            f"model/{scmle.artifacts.bundle.MANIFEST_FILE}",
            f"model/{scmle.artifacts.bundle.MODEL_FILE}",
            "explanations/permutation_importance.html",
            "explanations/partial_dependence.html",
            "explanations/pairwise_interaction.html",
        )
        return {
            name: scmle.artifacts.bundle.shasum(store.path(name))
            for name in names
            if store.path(name).exists()
        }
