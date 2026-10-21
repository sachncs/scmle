"""End-to-end evidence tests for the non-tabular baseline lifecycle."""

import hashlib
import io
import json
from pathlib import Path
import wave

import numpy as np
import pytest

from scml.contracts import data
from scml.contracts import modalities
from scml.contracts import performance
from scml.data.modalities import audio
from scml.data.modalities import document
from scml.data.modalities import geospatial
from scml.data.modalities import graph
from scml.data.modalities import image
from scml.data.modalities import video
from scml.models import modality_lifecycle


def document_adapter() -> modality_lifecycle.DocumentAdapter:
    """Build a labelled document fixture with independent document IDs."""
    samples = tuple(
        document.DocumentSample(
            document_id=f"document-{index}",
            content=("alpha signal " if index % 2 else "beta signal ")
            * (index + 2),
            label="alpha" if index % 2 else "beta",
        )
        for index in range(12)
    )
    return modality_lifecycle.DocumentAdapter(document.DocumentDataset(samples))


def image_bytes(value: int) -> bytes:
    """Encode one deterministic RGB image."""
    pillow_image = pytest.importorskip("PIL.Image")
    canvas = pillow_image.new("RGB", (4, 4), (value, 255 - value, value // 2))
    output = io.BytesIO()
    canvas.save(output, format="PNG")
    return output.getvalue()


def image_adapter() -> modality_lifecycle.ImageAdapter:
    """Build a labelled image fixture with distinct pixel values."""
    samples = tuple(
        image.ImageSample.from_bytes(
            image_bytes(20 + index * 15),
            source=f"image-{index}.png",
            label="alpha" if index % 2 else "beta",
            group_id=f"group-{index}",
        )
        for index in range(12)
    )
    return modality_lifecycle.ImageAdapter(image.ImageDataset(samples))


def wav_bytes(amplitude: int) -> bytes:
    """Encode a short mono PCM fixture."""
    samples = np.asarray([amplitude, -amplitude] * 40, dtype="<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as opened:
        opened.setnchannels(1)
        opened.setsampwidth(2)
        opened.setframerate(8_000)
        opened.writeframes(samples.tobytes())
    return output.getvalue()


def audio_adapter() -> modality_lifecycle.AudioAdapter:
    """Build a labelled WAV fixture with varying signal statistics."""
    samples = tuple(
        audio.AudioSample.from_bytes(
            wav_bytes(500 + index * 300),
            source=f"audio-{index}.wav",
            label="alpha" if index % 2 else "beta",
            group_id=f"speaker-{index}",
        )
        for index in range(12)
    )
    return modality_lifecycle.AudioAdapter(audio.AudioDataset(samples))


def video_adapter() -> modality_lifecycle.VideoAdapter:
    """Build a labelled probed-video fixture."""
    samples = tuple(
        video.VideoSample.from_bytes(
            bytes([index + 1]) * 16,
            video.VideoMetadata(
                "fixture",
                16 + index,
                16,
                8 + index,
                8.0,
                "fixture-codec",
            ),
            source=f"video-{index}.fixture",
            label="alpha" if index % 2 else "beta",
            group_id=f"camera-{index}",
        )
        for index in range(12)
    )
    return modality_lifecycle.VideoAdapter(video.VideoDataset(samples))


def decoded_video_adapter() -> modality_lifecycle.VideoAdapter:
    """Build a video baseline with an explicit bounded frame decoder."""
    samples = tuple(
        video.VideoSample.from_bytes(
            bytes([index + 1]) * 16,
            video.VideoMetadata(
                "fixture",
                16 + index,
                16,
                8 + index,
                8.0,
                "fixture-codec",
            ),
            source=f"decoded-video-{index}.fixture",
            label="alpha" if index % 2 else "beta",
            group_id=f"camera-{index}",
        )
        for index in range(12)
    )

    def decoder(payload: bytes, frame_index: int) -> np.ndarray:
        return np.full((2, 2, 3), payload[0] + frame_index, dtype=np.uint8)

    return modality_lifecycle.VideoAdapter(
        video.VideoDataset(samples), frame_decoder=decoder, max_frames=4
    )


def graph_adapter() -> modality_lifecycle.GraphAdapter:
    """Build an inductive labelled graph fixture."""
    nodes = tuple(
        graph.GraphNode(
            node_id=f"node-{index}",
            features=(float(index), float(index % 3)),
            label="alpha" if index % 2 else "beta",
        )
        for index in range(12)
    )
    edges = tuple(
        graph.GraphEdge(f"node-{index}", f"node-{index + 1}")
        for index in range(11)
    )
    dataset = graph.GraphData(
        nodes=nodes,
        edges=edges,
        feature_names=("position", "remainder"),
    )
    return modality_lifecycle.GraphAdapter(dataset)


def geospatial_adapter() -> modality_lifecycle.GeospatialAdapter:
    """Build spatially separated labelled point features."""
    features = tuple(
        geospatial.GeoFeature(
            feature_id=f"feature-{index}",
            geometry={"type": "Point", "coordinates": [float(index), 0.0]},
            label="alpha" if index % 2 else "beta",
        )
        for index in range(12)
    )
    dataset = geospatial.GeospatialDataset(
        features=features,
        crs="EPSG:4326",
        source="memory://fixture.geojson",
    )
    return modality_lifecycle.GeospatialAdapter(dataset, block_size=1.0)


def raster_adapter() -> modality_lifecycle.RasterAdapter:
    """Build a labelled raster metadata fixture with a sidecar."""
    samples = []
    labels = {}
    for index in range(12):
        source = f"memory://raster-{index}"
        payload = bytes([index + 1]) * 32
        samples.append(
            geospatial.RasterSample(
                source=source,
                encoded_bytes=payload,
                file_hash=hashlib.sha256(payload).hexdigest(),
                width=32,
                height=32,
                bands=3,
                dtype="uint8",
                crs="EPSG:3857",
                transform=(1.0, 0.0, float(index), 0.0, -1.0, 0.0),
            )
        )
        labels[source] = "alpha" if index % 2 else "beta"
    return modality_lifecycle.RasterAdapter(samples, labels)


@pytest.mark.parametrize(
    "factory",
    [
        document_adapter,
        image_adapter,
        audio_adapter,
        video_adapter,
        graph_adapter,
        geospatial_adapter,
        raster_adapter,
    ],
    ids=[
        "document",
        "image",
        "audio",
        "video",
        "graph",
        "geospatial",
        "raster",
    ],
)
def test_every_non_tabular_adapter_completes_all_fifteen_stages(
    factory, tmp_path: Path
):
    """Every adapter produces signed package, inference, and recovery evidence."""
    adapter = factory()
    report_path = tmp_path / "modality_lifecycle.json"
    report = modality_lifecycle.ModalityLifecycle().run(
        adapter,
        tmp_path / "package",
        signing_key="test-signing-key",
        seed=17,
        report_path=report_path,
    )

    stages = tuple(evidence.stage for evidence in report.stage_evidence)
    assert stages == modalities.REQUIRED_LIFECYCLE_STAGES
    assert all(evidence.evidenced for evidence in report.stage_evidence)
    evidence_by_stage = {
        evidence.stage: evidence for evidence in report.stage_evidence
    }
    assert evidence_by_stage[modalities.LifecycleStage.INGESTION].evidence == {
        "dataset_hash": report.dataset_hash,
        "sample_count": len(adapter.samples),
    }
    assert isinstance(
        evidence_by_stage[
            modalities.LifecycleStage.MODEL_EXPLAINABILITY
        ].evidence,
        dict,
    )
    model_quality = evidence_by_stage[
        modalities.LifecycleStage.MODEL_EXPLAINABILITY
    ].evidence["quality"]
    assert model_quality["passed"] is True
    assert model_quality["fidelity_score"] == pytest.approx(1.0)
    assert model_quality["stability_score"] >= 0.9
    perturbation = evidence_by_stage[
        modalities.LifecycleStage.MODEL_EXPLAINABILITY
    ].evidence["perturbation_stability"]
    assert perturbation["feature_space_only"] is True
    assert perturbation["report_count"] > 1
    repeated = evidence_by_stage[
        modalities.LifecycleStage.MODEL_EXPLAINABILITY
    ].evidence["repeated_perturbation"]
    assert repeated["evaluated_sample_count"] == min(
        repeated["sample_count"],
        modality_lifecycle.MAX_REPEATED_EXPLANATION_SAMPLES,
    )
    assert repeated["score"] >= 0.9
    assert repeated["feature_space_only"] is True
    data_explanation = evidence_by_stage[
        modalities.LifecycleStage.DATA_EXPLAINABILITY
    ].evidence
    assert data_explanation["partition_counts"]["train"] > 0
    assert data_explanation["partition_counts"]["test"] > 0
    sample_lineage = data_explanation["sample_lineage"]
    assert len(sample_lineage) == len(adapter.samples)
    assert {item["sample_id"] for item in sample_lineage} == {
        adapter.sample_id(sample) for sample in adapter.samples
    }
    assert {item["partition"] for item in sample_lineage} == {
        "train",
        "validation",
        "test",
    }
    assert all(
        isinstance(item["sample_id"], str) and item["sample_id"]
        for item in sample_lineage
    )
    assert (
        data_explanation["label_distribution"]["sample_count"]
        == (data_explanation["partition_counts"]["train"])
    )
    assert all(
        statistics["nonfinite_count"] == 0
        for statistics in data_explanation["feature_statistics"].values()
    )
    assert report.package.signature
    dataset_card = report.documentation["dataset_card"]
    assert (
        dataset_card["dataset_hash"] == dataset_card["curation"]["dataset_hash"]
    )
    assert dataset_card["privacy"]["status"] == "not_assessed"
    assert dataset_card["provenance"]["status"] == (
        "adapter_source_metadata_not_declared"
    )
    assert report.inference == {
        "package_verified": True,
        "prediction_count": report.evaluation["sample_count"],
        "predictions_match": True,
    }
    performance_evidence = report.performance
    assert [
        observation["operation"]
        for observation in performance_evidence["observations"]
    ] == [operation.value for operation in performance.PerformanceOperation]
    assert all(
        observation["sample_count"] > 0
        for observation in performance_evidence["observations"]
    )
    assert all(
        observation["peak_memory_gb"] > 0
        for observation in performance_evidence["observations"]
    )
    assert performance_evidence["budget"] is None
    assert performance_evidence["result"] is None
    assert report.recovery["tampered_manifest_rejected"] is True
    declaration = modality_lifecycle.ModalityLifecycle().declaration(report)
    assert declaration.status is modalities.ModalityStatus.EXPERIMENTAL
    assert declaration.complete
    assert json.loads(report_path.read_text()) == report.to_dict()


def test_package_signature_rejects_a_wrong_key(tmp_path: Path):
    """A signed baseline package cannot be loaded with another key."""
    adapter = document_adapter()
    report = modality_lifecycle.ModalityLifecycle().run(
        adapter,
        tmp_path / "package",
        signing_key="right-key",
        report_path=tmp_path / "report.json",
    )

    with pytest.raises(ValueError, match="signature"):
        modality_lifecycle.load_package(report.package.path, "wrong-key")


def test_modality_predictor_verifies_package_and_predicts_new_samples(
    tmp_path: Path,
):
    """Signed modality packages expose checked library inference."""
    adapter = image_adapter()
    report = modality_lifecycle.ModalityLifecycle().run(
        adapter,
        tmp_path / "predictor-package",
        signing_key="test-signing-key",
    )

    predictor = modality_lifecycle.ModalityPredictor.from_package(
        adapter,
        report.package.path,
        "test-signing-key",
    )
    predictions = predictor.predict(adapter.samples[:2])
    probabilities = predictor.predict_proba(adapter.samples[:2])

    assert predictions.shape == (2,)
    assert probabilities.shape == (2, 2)
    assert predictor.health() == {
        "package_verified": True,
        "modality": "image",
        "training_dataset_hash": adapter.dataset_hash,
        "feature_names": list(predictor.feature_names),
        "model_sha256": report.package.model_sha256,
    }
    with pytest.raises(ValueError, match="samples must not be empty"):
        predictor.predict(())


def test_modality_predictor_rejects_a_different_adapter_modality(
    tmp_path: Path,
):
    """A signed package cannot be paired with a different modality."""
    image = image_adapter()
    report = modality_lifecycle.ModalityLifecycle().run(
        image,
        tmp_path / "predictor-package",
        signing_key="test-signing-key",
    )

    with pytest.raises(ValueError, match="package modality"):
        modality_lifecycle.ModalityPredictor.from_package(
            document_adapter(),
            report.package.path,
            "test-signing-key",
        )


def test_image_lifecycle_emits_region_occlusion_evidence(tmp_path: Path):
    """The image adapter exposes bounded transformed-pixel occlusion."""
    report = modality_lifecycle.ModalityLifecycle().run(
        image_adapter(),
        tmp_path / "image-package",
        signing_key="test-signing-key",
    )

    evidence = {item.stage: item.evidence for item in report.stage_evidence}[
        modalities.LifecycleStage.MODEL_EXPLAINABILITY
    ]
    local = evidence["modality_specific"]
    assert local["status"] == "available"
    assert local["method"] == "transformed_pixel_region_occlusion"
    assert local["baseline"] == "zero-valued transformed RGB pixels"
    assert len(local["terms"]) == 10
    assert local["feature_space_only"] is True
    raw = local["raw_input_occlusion"]
    assert raw["method"] == "raw_pixel_region_occlusion"
    assert raw["source_dimensions"] == [4, 4]
    assert len(raw["terms"]) == 10
    assert raw["feature_space_only"] is False


def test_document_lifecycle_emits_span_provenance_evidence(tmp_path: Path):
    """The document adapter preserves page and section source context."""
    report = modality_lifecycle.ModalityLifecycle().run(
        document_adapter(),
        tmp_path / "document-package",
        signing_key="test-signing-key",
    )

    evidence = {item.stage: item.evidence for item in report.stage_evidence}[
        modalities.LifecycleStage.MODEL_EXPLAINABILITY
    ]
    local = evidence["modality_specific"]
    assert local["status"] == "available"
    assert local["method"] == "document_span_provenance"
    assert local["content_hash"]
    assert local["spans"]
    assert {span["kind"] for span in local["spans"]} == {"page"}
    raw = local["raw_input_occlusion"]
    assert raw["method"] == "raw_document_span_occlusion"
    assert len(raw["spans"]) == len(local["spans"])
    assert raw["spans"][0]["masked_content_hash"]
    assert raw["feature_space_only"] is False


def test_graph_lifecycle_emits_one_hop_context_evidence(tmp_path: Path):
    """The graph adapter exposes bounded topology context explicitly."""
    nodes = tuple(
        graph.GraphNode(
            node_id=f"node-{index}",
            features=(float(index), float(index + 1)),
            label="alpha" if index % 2 else "beta",
        )
        for index in range(12)
    )
    edges = tuple(
        graph.GraphEdge(
            source=f"node-{index}",
            target=f"node-{index + 1}",
            relation="next",
        )
        for index in range(11)
    )
    adapter = modality_lifecycle.GraphAdapter(
        graph.GraphData(
            nodes=nodes,
            edges=edges,
            feature_names=("first", "second"),
        )
    )
    report = modality_lifecycle.ModalityLifecycle().run(
        adapter,
        tmp_path / "graph-package",
        signing_key="test-signing-key",
    )

    evidence = {item.stage: item.evidence for item in report.stage_evidence}[
        modalities.LifecycleStage.MODEL_EXPLAINABILITY
    ]
    local = evidence["modality_specific"]
    assert local["status"] == "available"
    assert (
        local["method"] == "one_hop_graph_context_and_node_feature_contribution"
    )
    assert local["neighbors"]
    assert local["edges"]
    assert local["model_uses_topology"] is False


def test_geospatial_lifecycle_emits_crs_bound_explanation(tmp_path: Path):
    """The geospatial adapter preserves CRS and geometry evidence."""
    features = tuple(
        geospatial.GeoFeature(
            feature_id=f"feature-{index}",
            geometry={
                "type": "Point",
                "coordinates": [float(index), float(index % 3)],
            },
            label="alpha" if index % 2 else "beta",
        )
        for index in range(12)
    )
    adapter = modality_lifecycle.GeospatialAdapter(
        geospatial.GeospatialDataset(features=features, crs="EPSG:4326"),
        block_size=1.0,
    )
    report = modality_lifecycle.ModalityLifecycle().run(
        adapter,
        tmp_path / "geospatial-package",
        signing_key="test-signing-key",
    )

    evidence = {item.stage: item.evidence for item in report.stage_evidence}[
        modalities.LifecycleStage.MODEL_EXPLAINABILITY
    ]
    local = evidence["modality_specific"]
    assert local["status"] == "available"
    assert local["method"] == "crs_bound_geometry_and_feature_contribution"
    assert local["crs"] == "EPSG:4326"
    assert local["spatial_units_preserved"] is True
    assert local["feature_terms"]
    raw = local["raw_input_occlusion"]
    assert raw["method"] == "raw_geospatial_coordinate_axis_occlusion"
    assert raw["crs"] == "EPSG:4326"
    assert {item["axis"] for item in raw["axes"]} == {"x", "y"}
    assert all(item["masked_geometry_hash"] for item in raw["axes"])
    assert raw["feature_space_only"] is False


def test_audio_lifecycle_emits_time_local_explanation(tmp_path):
    """The audio adapter exposes bounded frame-window score evidence."""
    report = modality_lifecycle.ModalityLifecycle().run(
        audio_adapter(),
        tmp_path / "audio-package",
        signing_key="test-signing-key",
    )

    evidence = {item.stage: item.evidence for item in report.stage_evidence}[
        modalities.LifecycleStage.MODEL_EXPLAINABILITY
    ]
    local = evidence["modality_specific"]
    assert local["status"] == "available"
    assert local["method"] == "time_local_wav_window_score"
    assert local["window_count"] == len(local["windows"])
    assert local["window_count"] > 1
    assert local["feature_space_only"] is True
    raw = local["raw_input_occlusion"]
    assert raw["method"] == "raw_wav_waveform_window_occlusion"
    assert raw["frame_count"] == 80
    assert len(raw["windows"]) == local["window_count"]
    assert raw["feature_space_only"] is False


def test_lifecycle_persists_supplied_modality_manifest(tmp_path: Path):
    """Lifecycle documentation carries caller-owned governance evidence."""
    adapter = document_adapter()
    manifest = data.ModalityManifest(
        dataset_id="document-dataset",
        snapshot_id="document-snapshot",
        modality="document",
        task="binary_classification",
        sample_count=len(adapter.samples),
        content_hash=adapter.dataset_hash,
        source=data.SourceMetadata(
            owner="records-team",
            acquisition_method="approved export",
            license="internal-use",
        ),
        privacy=data.PrivacyMetadata(
            classification="restricted",
            contains_pii=True,
            consent_basis="employment-contract",
            retention_period="90d",
            deletion_policy="case-closure",
        ),
    )

    report = modality_lifecycle.ModalityLifecycle().run(
        adapter,
        tmp_path / "manifest-package",
        signing_key="test-signing-key",
        manifest=manifest,
    )

    card = report.documentation["dataset_card"]
    assert card["manifest"] == manifest.to_dict()
    assert card["provenance"]["status"] == "declared"
    assert card["privacy"]["status"] == "declared"
    assert card["manifest"]["privacy"]["contains_pii"] is True


def test_video_lifecycle_emits_temporal_frame_evidence(tmp_path: Path):
    """A supplied decoder enables bounded frame-local score evidence."""
    report = modality_lifecycle.ModalityLifecycle().run(
        decoded_video_adapter(),
        tmp_path / "video-package",
        signing_key="test-signing-key",
    )

    evidence = {item.stage: item.evidence for item in report.stage_evidence}[
        modalities.LifecycleStage.MODEL_EXPLAINABILITY
    ]
    local = evidence["modality_specific"]
    assert local["status"] == "available"
    assert local["method"] == "temporal_frame_statistics_occlusion"
    assert local["feature_space_only"] is True
    assert len(local["terms"]) == 4
    assert len(local["sampled_frame_indices"]) == 4
    raw = local["raw_input_occlusion"]
    assert raw["method"] == "raw_video_frame_occlusion"
    assert len(raw["windows"]) == 4
    assert raw["feature_space_only"] is False


def test_lifecycle_enforces_declared_performance_budget(tmp_path: Path):
    """A lifecycle budget is emitted as evidence and fails closed."""
    adapter = document_adapter()
    budget = performance.PerformanceBudget(
        dict.fromkeys(performance.PerformanceOperation, 30.0),
        max_peak_memory_gb=1024.0,
    )
    report = modality_lifecycle.ModalityLifecycle().run(
        adapter,
        tmp_path / "passing-package",
        signing_key="test-signing-key",
        performance_budget=budget,
    )

    assert report.performance["budget"] == budget.to_dict()
    assert report.performance["result"]["passed"] is True

    with pytest.raises(ValueError, match="performance budget failed"):
        modality_lifecycle.ModalityLifecycle().run(
            adapter,
            tmp_path / "failing-package",
            signing_key="test-signing-key",
            performance_budget=performance.PerformanceBudget(
                {
                    performance.PerformanceOperation.INGEST: 1e-12,
                }
            ),
        )


def test_lifecycle_compares_against_a_repeated_performance_baseline(
    tmp_path: Path,
):
    """Lifecycle evidence can enforce a repeated resource baseline."""
    adapter = document_adapter()
    reference_reports = [
        modality_lifecycle.ModalityLifecycle().run(
            adapter,
            tmp_path / f"reference-package-{index}",
            signing_key="test-signing-key",
            seed=index,
        )
        for index in range(2)
    ]
    reference_runs = tuple(
        performance.PerformanceRun.from_dict(
            {
                "run_id": f"reference-{index}",
                "observations": report.performance["observations"],
            }
        )
        for index, report in enumerate(reference_reports)
    )
    baseline = performance.PerformanceBaseline(
        reference_runs, max_relative_regression=10.0
    )

    report = modality_lifecycle.ModalityLifecycle().run(
        adapter,
        tmp_path / "current-package",
        signing_key="test-signing-key",
        seed=2,
        performance_baseline=baseline,
    )

    assert report.performance["comparison"]["passed"] is True
    assert report.performance["comparison"]["baseline_run_count"] == 2


def test_blocking_geospatial_curation_stops_before_packaging(tmp_path: Path):
    """Missing CRS evidence blocks before a model or package is created."""
    features = tuple(
        geospatial.GeoFeature(
            feature_id=f"feature-{index}",
            geometry={"type": "Point", "coordinates": [float(index), 0.0]},
            label="alpha" if index % 2 else "beta",
        )
        for index in range(8)
    )
    adapter = modality_lifecycle.GeospatialAdapter(
        geospatial.GeospatialDataset(features=features, crs=""),
        block_size=1.0,
    )

    with pytest.raises(ValueError, match="blocking curation"):
        modality_lifecycle.ModalityLifecycle().run(
            adapter,
            tmp_path / "package",
            signing_key="test-signing-key",
        )
    assert not (tmp_path / "package").exists()


def test_graph_without_declared_features_stops_before_modelling(tmp_path: Path):
    """A topology-only graph cannot silently inherit a feature policy."""
    nodes = tuple(
        graph.GraphNode(
            node_id=f"node-{index}",
            label="alpha" if index % 2 else "beta",
        )
        for index in range(8)
    )
    adapter = modality_lifecycle.GraphAdapter(graph.GraphData(nodes=nodes))

    with pytest.raises(ValueError, match="declared node features"):
        modality_lifecycle.ModalityLifecycle().run(
            adapter,
            tmp_path / "package",
            signing_key="test-signing-key",
        )
    assert not (tmp_path / "package").exists()
