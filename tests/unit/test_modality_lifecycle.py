"""End-to-end evidence tests for the non-tabular baseline lifecycle."""

import hashlib
import io
import json
from pathlib import Path
import wave

import numpy as np
from PIL import Image
import pytest

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
    canvas = Image.new("RGB", (4, 4), (value, 255 - value, value // 2))
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
                    performance.PerformanceOperation.INGEST: 0.000001,
                }
            ),
        )


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
