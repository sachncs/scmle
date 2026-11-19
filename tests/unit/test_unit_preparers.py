"""The modality preparers, their decision scores and the package checks.

Each preparer is driven directly: its identity, partitions, features,
monitoring and the explanation evidence of a classifier fitted on its own
features. The fixtures are small, labelled and deterministic.
"""

import hashlib
import io
import json
import pathlib
import wave

import numpy
import pytest
from sklearn.linear_model import LogisticRegression

from scmle.contracts import errors
from scmle.contracts import signing
from scmle.data import modalities
from scmle.pipeline.modality_lifecycle import constants
from scmle.pipeline.modality_lifecycle import scoring
from scmle.pipeline.modality_lifecycle import verification
from scmle.pipeline.modality_lifecycle.preparers import audio
from scmle.pipeline.modality_lifecycle.preparers import document
from scmle.pipeline.modality_lifecycle.preparers import geospatial
from scmle.pipeline.modality_lifecycle.preparers import graph
from scmle.pipeline.modality_lifecycle.preparers import image
from scmle.pipeline.modality_lifecycle.preparers import raster
from scmle.pipeline.modality_lifecycle.preparers import video


def label(index):
    return "alpha" if index % 2 else "beta"


def documents(kind=document.DocumentPreparer):
    samples = tuple(
        modalities.document.Sample(
            document_id=f"document-{index}",
            content=("alpha signal " if index % 2 else "beta signal ")
            * (index + 2),
            label=label(index),
        )
        for index in range(12)
    )
    return kind(modalities.document.Dataset(samples))


def png(value):
    pillow = pytest.importorskip("PIL.Image")
    canvas = pillow.new("RGB", (4, 4), (value, 255 - value, value // 2))
    output = io.BytesIO()
    canvas.save(output, format="PNG")
    return output.getvalue()


def images():
    samples = tuple(
        modalities.image.Sample.from_bytes(
            png(20 + index * 15),
            source=f"image-{index}.png",
            label=label(index),
            group_id=f"group-{index}",
        )
        for index in range(12)
    )
    return image.ImagePreparer(modalities.image.Dataset(samples))


def wav(amplitude, frames=40):
    values = numpy.asarray([amplitude, -amplitude] * frames, dtype="<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as opened:
        opened.setnchannels(1)
        opened.setsampwidth(2)
        opened.setframerate(8_000)
        opened.writeframes(values.tobytes())
    return output.getvalue()


def sounds():
    samples = tuple(
        modalities.audio.Sample.from_bytes(
            wav(500 + index * 300),
            source=f"audio-{index}.wav",
            label=label(index),
            group_id=f"speaker-{index}",
        )
        for index in range(12)
    )
    return audio.AudioPreparer(modalities.audio.Dataset(samples))


def clips(decoded=False):
    samples = tuple(
        modalities.video.Sample.from_bytes(
            bytes([index + 1]) * 16,
            modalities.video.Metadata(
                "fixture", 16 + index, 16, 8 + index, 8.0, "fixture-codec"
            ),
            source=f"video-{index}.fixture",
            label=label(index),
            group_id=f"camera-{index}",
        )
        for index in range(12)
    )
    if not decoded:
        return video.Footage(modalities.video.Dataset(samples))

    def decoder(payload, frame_index):
        return numpy.full(
            (2, 2, 3), payload[0] + frame_index, dtype=numpy.uint8
        )

    return video.Footage(
        modalities.video.Dataset(samples), frame_decoder=decoder, max_frames=4
    )


def nodes():
    made = tuple(
        modalities.graph.Node(
            node_id=f"node-{index}",
            features=(float(index), float(index % 3)),
            label=label(index),
        )
        for index in range(12)
    )
    edges = tuple(
        modalities.graph.Edge(f"node-{index}", f"node-{index + 1}")
        for index in range(11)
    )
    return graph.GraphPreparer(
        modalities.graph.Dataset(
            nodes=made, edges=edges, feature_names=("position", "remainder")
        )
    )


def placemarks():
    features = tuple(
        modalities.geospatial.Placemark(
            feature_id=f"feature-{index}",
            geometry={"type": "Point", "coordinates": [float(index), 0.0]},
            label=label(index),
        )
        for index in range(12)
    )
    dataset = modalities.geospatial.Dataset(
        features=features, crs="EPSG:4326", source="memory://fixture.geojson"
    )
    return geospatial.GeospatialPreparer(dataset, block_size=1.0)


def tiles():
    samples = []
    labels = {}
    for index in range(12):
        source = f"memory://raster-{index}"
        payload = bytes([index + 1]) * 32
        samples.append(
            modalities.geospatial.Tile(
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
        labels[source] = label(index)
    return raster.Imagery(samples, labels)


BUILDERS = {
    "document": documents,
    "text": lambda: documents(document.Plaintext),
    "image": images,
    "audio": sounds,
    "video": clips,
    "graph": nodes,
    "geospatial": placemarks,
    "raster": tiles,
}


def fitted(preparer):
    batch = preparer.features(preparer.samples)
    labels = [preparer.label(sample) for sample in preparer.samples]
    model = LogisticRegression(max_iter=500).fit(batch.values, labels)
    return model, batch


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_a_preparer_names_one_row_per_sample(name):
    preparer = BUILDERS[name]()
    batch = preparer.features(preparer.samples)
    assert batch.values.shape == (
        len(preparer.samples),
        len(batch.feature_names),
    )
    assert batch.sample_ids == tuple(
        preparer.sample_id(sample) for sample in preparer.samples
    )
    assert len(set(batch.sample_ids)) == len(preparer.samples)
    assert [preparer.label(s) for s in preparer.samples] == [
        label(index) for index in range(12)
    ]
    assert numpy.isfinite(batch.values).all()


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_a_preparer_splits_every_sample_once(name):
    preparer = BUILDERS[name]()
    parts = preparer.split(7)
    together = [*parts.train, *parts.validation, *parts.test]
    ids = sorted(preparer.sample_id(sample) for sample in together)
    assert ids == sorted(preparer.sample_id(s) for s in preparer.samples)
    assert parts.train
    assert parts.test
    again = preparer.split(7)
    assert [preparer.sample_id(s) for s in again.train] == [
        preparer.sample_id(s) for s in parts.train
    ]


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_a_preparer_has_a_stable_identity_and_states_its_limits(name):
    preparer = BUILDERS[name]()
    assert preparer.digest == BUILDERS[name]().digest
    assert containers_hashlike(preparer.digest)
    remarks = preparer.remarks()
    assert remarks
    assert all(isinstance(line, str) and line for line in remarks)
    assert isinstance(preparer.monitor(BUILDERS[name]()), dict)


def containers_hashlike(value):
    return len(value) == constants.SHA256_LENGTH and all(
        c in "0123456789abcdef" for c in value
    )


@pytest.mark.parametrize(
    "name", ["document", "image", "audio", "graph", "geospatial", "raster"]
)
def test_a_preparer_refuses_to_monitor_another_modality(name):
    other = "video" if name != "video" else "image"
    with pytest.raises(errors.KindError, match="observed adapter must be"):
        BUILDERS[name]().monitor(BUILDERS[other]())


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_a_preparer_explains_one_sample_with_bounded_terms(name):
    preparer = BUILDERS[name]()
    model, batch = fitted(preparer)
    coefficients = numpy.asarray(model.coef_[0])
    evidence = preparer.elucidate(
        preparer.samples[1], model, batch.feature_names, coefficients
    )
    assert isinstance(evidence, dict)
    assert evidence["status"] in {"available", "not_available"}
    assert isinstance(evidence.get("limitations", []), list)
    terms = evidence.get("terms", [])
    assert len(terms) <= 10
    magnitudes = [abs(term[1]) for term in terms]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_image_explanations_occlude_sixteen_regions():
    preparer = images()
    model, batch = fitted(preparer)
    evidence = preparer.elucidate(preparer.samples[1], model, (), None)
    assert evidence["method"] == "transformed_pixel_region_occlusion"
    assert evidence["feature_space_only"] is True
    assert len(evidence["terms"]) == 10
    assert all(name.startswith("region[") for name, weight in evidence["terms"])
    assert len(batch.feature_names) == 768
    assert batch.feature_names[:2] == ("pixel[0,0,0]", "pixel[0,0,1]")
    with pytest.raises(errors.KindError, match="sample must be an Sample"):
        preparer.elucidate("not an image", model, (), None)


def test_the_image_sample_id_joins_source_and_hash():
    preparer = images()
    sample = preparer.samples[0]
    # A bare file name is a local path, so its source is its file URI.
    source = pathlib.Path("image-0.png").resolve().as_uri()
    assert preparer.sample_id(sample) == f"{source}:{sample.content_hash}"


def test_the_raster_headers_name_four_statistics_per_band():
    assert raster.Imagery.headers(2) == (
        "pixel_band_0_mean",
        "pixel_band_0_std",
        "pixel_band_0_min",
        "pixel_band_0_max",
        "pixel_band_1_mean",
        "pixel_band_1_std",
        "pixel_band_1_min",
        "pixel_band_1_max",
    )
    with pytest.raises(errors.InputError, match="band_count must be positive"):
        raster.Imagery.headers(0)


def test_the_raster_summary_is_mean_std_min_max_per_band():
    pixels = numpy.array([[[0, 10], [2, 10]], [[4, 10], [6, 10]]], float)
    assert raster.summary(pixels, 2) == (
        3.0,
        numpy.sqrt(5.0),
        0.0,
        6.0,
        10.0,
        0.0,
        10.0,
        10.0,
    )
    single = numpy.array([[1.0, 3.0]])
    assert raster.summary(single, 1) == (2.0, 1.0, 1.0, 3.0)
    with pytest.raises(
        modalities.geospatial.GeometryError, match="must declare one band"
    ):
        raster.summary(single, 2)
    with pytest.raises(
        modalities.geospatial.GeometryError, match="do not match"
    ):
        raster.summary(pixels, 3)


def test_video_groups_fall_back_to_the_source():
    samples = clips().samples[:2]
    assert video.groups(samples) == ("camera-0", "camera-1")
    loose = modalities.video.Sample.from_bytes(
        b"\x01" * 16,
        modalities.video.Metadata("fixture", 16, 16, 8, 8.0, "fixture-codec"),
        source="z.fixture",
    )
    assert video.groups((loose, samples[0])) == ("camera-0", loose.source)
    assert loose.group_id is None


def test_video_occlusion_zeroes_five_statistics_per_frame():
    sample = clips().samples[0]
    model = LogisticRegression().fit(
        numpy.array([[0.0] * 10, [1.0] * 10]), [0, 1]
    )
    values = numpy.arange(10, dtype=float)
    full, terms = video.occlusion(sample, model, values, [0, 4])
    assert full == pytest.approx(
        float(model.decision_function(values[None])[0])
    )
    first = values.copy()
    first[:5] = 0.0
    second = values.copy()
    second[5:] = 0.0
    deltas = {
        0: float(model.decision_function(first[None])[0]) - full,
        4: float(model.decision_function(second[None])[0]) - full,
    }
    expected = sorted(deltas, key=lambda key: (-abs(deltas[key]), key))
    assert [t["frame_index"] for t in terms] == expected
    assert {t["frame_index"]: t["score_delta"] for t in terms} == (
        pytest.approx(deltas)
    )
    assert {t["frame_index"]: t["time_seconds"] for t in terms} == {
        0: 0.0,
        4: 0.5,
    }


def test_video_frames_need_a_decoder():
    # Enforces website/src/pages/docs/data.astro:277.
    preparer = clips()
    with pytest.raises(errors.OrderError, match="explicit frame decoder"):
        video.frames(preparer, preparer.samples[0], None, [0], 0.0)


def test_decoded_video_frames_report_each_blanked_frame():
    preparer = clips(decoded=True)
    model = fitted(preparer)[0]
    windows = video.frames(preparer, preparer.samples[0], model, [0, 2], 0.0)
    assert [w["frame_index"] for w in windows] == [0, 2]
    assert [w["time_seconds"] for w in windows] == [0.0, 0.25]


def test_graph_knockouts_zero_each_feature_in_turn():
    preparer = nodes()
    model = fitted(preparer)[0]
    sample = preparer.samples[5]
    anchored = scoring.anchor(model, preparer.features((sample,)).values)
    found = graph.knockouts(
        preparer, sample, model, ("position", "remainder"), anchored
    )
    assert [f["feature"] for f in found] == ["position", "remainder"]
    assert [f["feature_index"] for f in found] == [0, 1]
    masked = modalities.graph.Node(
        node_id=sample.node_id,
        features=(0.0, *sample.features[1:]),
        label=sample.label,
    )
    expected = anchored[1] - scoring.score(
        model, preparer.features((masked,)).values, anchored[0]
    )
    assert found[0]["score_delta_from_full"] == pytest.approx(expected)
    assert found[1]["score_delta_from_full"] == pytest.approx(
        anchored[1]
        - scoring.score(
            model,
            numpy.array([[sample.features[0], 0.0]]),
            anchored[0],
        )
    )
    only = graph.knockouts(preparer, sample, model, ("position",), anchored)
    assert len(only) == 1


def test_geospatial_axes_mask_x_then_y_at_the_midpoint():
    preparer = placemarks()
    model = fitted(preparer)[0]
    sample = preparer.samples[3]
    anchored = scoring.anchor(model, preparer.features((sample,)).values)
    found = geospatial.axes(preparer, sample, model, anchored)
    assert [a["axis"] for a in found] == ["x", "y"]
    assert [a["baseline_value"] for a in found] == [3.0, 0.0]


def test_audio_mute_silences_only_the_window():
    sample = sounds().samples[0]
    with wave.open(io.BytesIO(sample.encoded_bytes), "rb") as reader:
        parameters = reader.getparams()
        frames = bytearray(reader.readframes(reader.getnframes()))
    masked = audio.mute(sample, parameters, frames, (2, 5))
    with wave.open(io.BytesIO(masked.encoded_bytes), "rb") as reader:
        values = numpy.frombuffer(reader.readframes(80), dtype="<i2")
    original = numpy.frombuffer(bytes(frames), dtype="<i2")
    assert values[2:5].tolist() == [0, 0, 0]
    assert values[:2].tolist() == original[:2].tolist()
    assert values[5:].tolist() == original[5:].tolist()
    assert (masked.source, masked.label, masked.group_id) == (
        sample.source,
        sample.label,
        sample.group_id,
    )
    assert frames == bytearray(original.tobytes())


def test_audio_timeline_scores_at_most_eight_windows():
    preparer = sounds()
    model = fitted(preparer)[0]
    sample = preparer.samples[1]
    row = preparer.features((sample,)).values[0]
    width, full, windows = audio.timeline(sample, model, row)
    assert width == 10
    assert len(windows) == 8
    assert windows[0]["start_frame"] == 0
    assert windows[1]["start_frame"] == 10
    assert windows[1]["start_seconds"] == 10 / 8000
    for window in windows:
        assert window["score_delta_from_full"] == pytest.approx(
            window["score"] - full
        )


class Binary:
    classes_ = numpy.array(["a", "b"])

    def decision_function(self, features):
        return numpy.asarray(features)[:, 0] * 2.0

    def predict(self, features):
        return numpy.where(self.decision_function(features) > 0, "b", "a")


class Three:
    classes_ = numpy.array(["a", "b", "c"])

    def decision_function(self, features):
        row = numpy.asarray(features)
        return numpy.column_stack([row[:, 0], row[:, 1], row[:, 2]])

    def predict(self, features):
        return self.classes_[numpy.argmax(self.decision_function(features), 1)]


def test_anchor_tracks_the_predicted_class():
    assert scoring.anchor(Binary(), numpy.array([[1.5]])) == (0, 3.0)
    assert scoring.anchor(Three(), numpy.array([[0.1, 0.9, 0.2]])) == (1, 0.9)
    held = numpy.array([[7.0, 1.0, 0.0]])
    assert scoring.anchor(Three(), numpy.array([[0.0, 1.0, 0.0]]), held) == (
        1,
        1.0,
    )
    assert scoring.anchor(
        Binary(), numpy.array([[1.0]]), numpy.array([5.0])
    ) == (
        0,
        5.0,
    )


def test_score_reads_the_tracked_class():
    assert scoring.score(Binary(), numpy.array([[2.0]]), 4) == 4.0
    assert scoring.score(Three(), numpy.array([[1.0, 2.0, 3.0]]), 2) == 3.0


def test_rank_keeps_the_ten_largest_changes_ties_by_name():
    terms = [(f"t{i:02d}", float(i % 7) - 3) for i in range(14)]
    ranked = scoring.rank(terms)
    assert len(ranked) == 10
    assert ranked[:4] == [
        ("t00", -3.0),
        ("t06", 3.0),
        ("t07", -3.0),
        ("t13", 3.0),
    ]
    assert scoring.rank([("b", 1.0), ("a", -1.0)]) == [("a", -1.0), ("b", 1.0)]


def test_file_digest_is_the_sha256_of_the_file(tmp_path):
    path = tmp_path / "f"
    path.write_bytes(b"abc" * 1000)
    assert verification.file_digest(path) == (
        hashlib.sha256(b"abc" * 1000).hexdigest()
    )


def test_feature_names_must_be_distinct_non_blank_strings():
    assert verification.require_feature_names(["a", "b"]) == ["a", "b"]
    for bad in ([], ("a",), ["a", "a"], ["a", " "], ["a", 1], None):
        with pytest.raises(
            errors.InputError, match="feature_names are invalid"
        ):
            verification.require_feature_names(bad)


KEY = "k" * 32


def manifest(**changes):
    unsigned = {
        "schema_version": constants.PACKAGE_SCHEMA_VERSION,
        "modality": "image",
        "dataset_hash": "d",
        "feature_names": ["a", "b"],
        "model_sha256": "0" * 64,
        "evaluation": {},
    }
    unsigned.update(changes)
    return {
        **unsigned,
        "signature": {
            "algorithm": signing.SIGNATURE_ALGORITHM,
            "value": signing.mac(unsigned, KEY),
        },
    }


def test_a_signed_manifest_yields_its_features_and_digest():
    assert verification.verify_package_manifest(manifest(), KEY) == (
        ["a", "b"],
        "0" * 64,
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": 2}, "unsupported modality package schema"),
        ({"schema_version": True}, "unsupported modality package schema"),
        ({"modality": " "}, "modality is invalid"),
        ({"dataset_hash": ""}, "dataset_hash is invalid"),
        ({"model_sha256": "0" * 63}, "model_sha256 is invalid"),
        ({"model_sha256": "A" * 64}, "model_sha256 is invalid"),
        ({"evaluation": []}, "evaluation is invalid"),
    ],
)
def test_a_malformed_manifest_names_the_field(change, message):
    with pytest.raises(errors.InputError, match=message):
        verification.verify_package_manifest(manifest(**change), KEY)


def test_the_manifest_signature_must_verify():
    with pytest.raises(errors.InputError, match="must be an object"):
        verification.verify_package_manifest([], KEY)
    forged = manifest()
    forged["dataset_hash"] = "e"
    with pytest.raises(errors.InputError, match="does not verify"):
        verification.verify_package_manifest(forged, KEY)
    with pytest.raises(errors.InputError, match="does not verify"):
        verification.verify_package_manifest(manifest(), "other" * 8)
    unsigned = manifest()
    unsigned["signature"] = {"algorithm": "md5", "value": "x"}
    with pytest.raises(errors.InputError, match="missing or unsupported"):
        verification.verify_package_manifest(unsigned, KEY)
    unsigned["signature"] = {"algorithm": signing.SIGNATURE_ALGORITHM}
    with pytest.raises(errors.InputError, match="signature is empty"):
        verification.verify_package_manifest(unsigned, KEY)


def test_vet_checks_the_classifier_against_the_schema():
    model = LogisticRegression().fit([[0.0, 1.0], [1.0, 0.0]], [0, 1])
    verification.vet(model, 2)
    with pytest.raises(errors.KindError, match="expected classifier"):
        verification.vet(object(), 2)
    with pytest.raises(errors.InputError, match="do not match"):
        verification.vet(model, 3)
    model.coef_ = model.coef_ * numpy.nan
    with pytest.raises(errors.InputError, match="non-finite"):
        verification.vet(model, 2)
    single = LogisticRegression().fit([[0.0, 1.0], [1.0, 0.0]], [0, 1])
    single.classes_ = numpy.array([0])
    with pytest.raises(errors.InputError, match="fewer than two classes"):
        verification.vet(single, 2)


def test_load_package_refuses_an_incomplete_package(tmp_path):
    with pytest.raises(errors.InputError, match="incomplete"):
        verification.load_package(tmp_path, KEY)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest()))
    with pytest.raises(errors.InputError, match="incomplete"):
        verification.load_package(tmp_path, KEY)
