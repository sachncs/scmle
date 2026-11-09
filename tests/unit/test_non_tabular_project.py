"""Focused tests for the bounded non-tabular modelling project."""

import hashlib
import io
from pathlib import Path
import wave

import numpy
import pytest

import scmle.api.non_tabular
from scmle.data.modalities import audio
from scmle.data.modalities import document
from scmle.data.modalities import geospatial
from scmle.data.modalities import graph
from scmle.data.modalities import image
from scmle.data.modalities import video
from scmle.models import modality_lifecycle


def text_adapter() -> modality_lifecycle.TextPreparer:
    """Build a small deterministic labelled text dataset."""
    samples = tuple(
        document.Sample(
            document_id=f"document-{index}",
            content=("alpha signal " if index % 2 else "beta signal ")
            * (index + 2),
            label="alpha" if index % 2 else "beta",
        )
        for index in range(12)
    )
    return modality_lifecycle.TextPreparer(document.Dataset(samples))


def metadata_video_adapter() -> modality_lifecycle.VideoPreparer:
    """Build a video adapter that uses its declared metadata representation."""
    samples = tuple(
        video.Sample.from_bytes(
            bytes([index + 1]) * 16,
            video.Metadata(
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
    return modality_lifecycle.VideoPreparer(video.Dataset(samples))


def image_adapter() -> modality_lifecycle.ImagePreparer:
    """Build one decoded image representation."""
    image_library = pytest.importorskip("PIL.Image")
    output = io.BytesIO()
    image_library.new("RGB", (4, 4), (32, 64, 96)).save(output, format="PNG")
    sample = image.Sample.from_bytes(
        output.getvalue(), source="image.png", label="alpha", group_id="one"
    )
    return modality_lifecycle.ImagePreparer(image.Dataset((sample,)))


def audio_adapter() -> modality_lifecycle.AudioPreparer:
    """Build one WAV signal representation."""
    output = io.BytesIO()
    with wave.open(output, "wb") as opened:
        opened.setnchannels(1)
        opened.setsampwidth(2)
        opened.setframerate(8_000)
        opened.writeframes(
            numpy.asarray([200, -200] * 8, dtype="<i2").tobytes()
        )
    sample = audio.Sample.from_bytes(
        output.getvalue(), source="audio.wav", label="alpha", group_id="one"
    )
    return modality_lifecycle.AudioPreparer(audio.Dataset((sample,)))


def graph_adapter() -> modality_lifecycle.GraphPreparer:
    """Build one declared node-feature representation."""
    dataset = graph.Dataset(
        nodes=(graph.Node("node-1", (1.0, 2.0), "alpha"),),
        edges=(),
        feature_names=("x", "y"),
    )
    return modality_lifecycle.GraphPreparer(dataset)


def geospatial_adapter() -> modality_lifecycle.GeospatialPreparer:
    """Build one CRS-bound geometry representation."""
    dataset = geospatial.Dataset(
        features=(
            geospatial.GeoFeature(
                feature_id="feature-1",
                geometry={"type": "Point", "coordinates": [1.0, 2.0]},
                label="alpha",
            ),
        ),
        crs="EPSG:4326",
        source="memory://fixture.geojson",
    )
    return modality_lifecycle.GeospatialPreparer(dataset)


def raster_adapter() -> modality_lifecycle.RasterPreparer:
    """Build one metadata-only raster representation."""
    payload = b"raster"
    sample = geospatial.RasterSample(
        source="memory://raster-1",
        encoded_bytes=payload,
        file_hash=hashlib.sha256(payload).hexdigest(),
        width=2,
        height=2,
        bands=1,
        dtype="uint8",
        crs="EPSG:3857",
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
    )
    return modality_lifecycle.RasterPreparer(
        (sample,), {sample.source: "alpha"}
    )


def test_project_requires_explicit_experimental_opt_in(tmp_path: Path):
    """Experimental non-tabular paths never become an implicit default."""
    with pytest.raises(
        scmle.api.non_tabular.UnsupportedNonTabularModelError,
        match="experimental_opt_in=True",
    ):
        scmle.api.non_tabular.Modeler(
            text_adapter(), "test-key", output_dir=tmp_path
        )


def test_project_runs_and_reloads_verified_bounded_result(tmp_path: Path):
    """A run returns a verified package tied to its representation contract."""
    adapter = text_adapter()
    project = scmle.api.non_tabular.Modeler(
        adapter,
        "test-key",
        experimental_opt_in=True,
        output_dir=tmp_path / "run",
    )

    result = project.run()

    assert result.capability.representation == (
        "bounded_character_and_token_counts"
    )
    assert result.report.documentation["status"] == "experimental"
    assert result.health()["package_verified"] is True
    assert result.predict(list(adapter.samples[:2])).shape == (2,)
    assert result.predict_proba(list(adapter.samples[:2])).shape == (2, 2)

    loaded = project.load_result()
    assert loaded.report == result.report
    assert loaded.predictor.health()["lifecycle_report_verified"] is True


def test_project_rejects_custom_adapter_subclasses(tmp_path: Path):
    """Caller-defined model semantics remain on the broader extension path."""

    class CustomTextAdapter(modality_lifecycle.TextPreparer):
        """Adapter subclass intentionally outside the exact allowlist."""

    adapter = CustomTextAdapter(text_adapter().dataset)
    with pytest.raises(
        scmle.api.non_tabular.UnsupportedNonTabularModelError,
        match="exact built-in Preparer types",
    ):
        scmle.api.non_tabular.Modeler(
            adapter,
            "test-key",
            experimental_opt_in=True,
            output_dir=tmp_path,
        )


def test_capability_reports_existing_metadata_representation():
    """Metadata-only video is explicit and does not imply visual modelling."""
    capability = scmle.api.non_tabular.Modeler.capability_for_adapter(
        metadata_video_adapter()
    )

    assert capability.modality == "video"
    assert capability.representation == "bounded_container_metadata"
    assert "does not claim visual modelling" in capability.limitations[0]


@pytest.mark.parametrize(
    "factory,representation",
    [
        (image_adapter, "decoded_resized_rgb_pixels"),
        (audio_adapter, "bounded_decoded_wav_signal_statistics"),
        (graph_adapter, "declared_inductive_node_features"),
        (geospatial_adapter, "declared_crs_geometry_features"),
        (raster_adapter, "bounded_raster_metadata"),
    ],
)
def test_capability_allowlist_uses_existing_feature_representations(
    factory, representation: str
):
    """Every admitted built-in branch names the representation it consumes."""
    capability = scmle.api.non_tabular.Modeler.capability_for_adapter(factory())

    assert capability.representation == representation
    assert capability.status == "experimental"


def test_capability_rejects_non_modelling_curation_adapter():
    """Curation-only extension adapters cannot cross the modelling boundary."""
    with pytest.raises(TypeError, match="Preparer"):
        scmle.api.non_tabular.Modeler.capability_for_adapter(  # type: ignore[arg-type]
            object()
        )


def test_project_rejects_mismatched_observed_representation(tmp_path: Path):
    """Monitoring cannot silently compare incompatible feature contracts."""
    with pytest.raises(
        scmle.api.non_tabular.UnsupportedNonTabularModelError,
        match="same bounded capability",
    ):
        scmle.api.non_tabular.Modeler(
            text_adapter(),
            "test-key",
            experimental_opt_in=True,
            output_dir=tmp_path,
            observed_adapter=metadata_video_adapter(),
        )


def test_public_capability_record_is_json_native():
    """The representation claim can travel with release evidence."""
    record = scmle.api.non_tabular.Modeler.capability_for_adapter(
        text_adapter()
    ).to_dict()

    assert record["status"] == "experimental"
    assert record["model_family"] == "regularized_linear_classifier"
    assert isinstance(record["limitations"], list)
