"""The pixel image model: records, estimator, explanations, packages."""

import hashlib
import json

import numpy
import pytest
from sklearn.linear_model import LogisticRegression

from scmle.contracts import errors
import scmle.contracts.records
from scmle.data.modalities import image
from scmle.models.images import archive
from scmle.models.images import classifier
from scmle.models.images import estimator
from scmle.models.images import monitor
from scmle.models.images import records


class Report:
    """A stand-in curation report."""

    def __init__(self, means, duplicates=0, count=4, blocking=()):
        self.blockers = tuple(
            type("Finding", (), {"identifier": name})() for name in blocking
        )
        self.profile = {"duplicates": duplicates, "pixel": {"mean": means}}
        self.samples = count
        self.dataset_hash = f"hash-{means}-{duplicates}"


class Picture:
    """A stand-in image sample carrying its flattened pixels."""

    def __init__(self, pixels, label=None):
        self.pixels = numpy.asarray(pixels, dtype=float)
        self.label = label
        self.content_hash = hashlib.sha256(self.pixels.tobytes()).hexdigest()


class Pictures:
    """A stand-in image dataset that flattens to its samples' pixels."""

    def __init__(self, samples, report=None):
        self.samples = tuple(samples)
        self.report = report or Report([0.0, 0.0, 0.0], count=len(self.samples))

    def curate(self):
        return self.report

    def flatten(self, target_size, mode, normalize):
        del target_size, mode, normalize
        if not self.samples:
            return numpy.zeros((0, 12))
        return numpy.vstack([s.pixels for s in self.samples])


def tile(red, blue):
    """Four RGB pixels in a 2x2 image, normalised."""
    return [red, 0.0, blue] * 4


TRAINING = Pictures(
    (
        Picture(tile(1.0, 0.0), "red"),
        Picture(tile(0.86, 0.04), "red"),
        Picture(tile(0.0, 1.0), "blue"),
        Picture(tile(0.04, 0.86), "blue"),
    )
)
HELD = Pictures(
    (Picture(tile(0.98, 0.08), "red"), Picture(tile(0.08, 0.98), "blue"))
)


def config(**changes):
    values = {"size": (2, 2)}
    values.update(changes)
    return records.Config(**values)


@pytest.fixture
def wrapped(monkeypatch):
    monkeypatch.setattr(image, "Dataset", Pictures)


@pytest.fixture(scope="module")
def fitted():
    return classifier.Classifier(config()).fit(TRAINING)


# records


def test_the_record_constants_are_fixed():
    assert records.PATCH_DIMENSION_COUNT == 2
    assert records.MIN_CLASS_COUNT == 2
    assert records.BINARY_CLASS_COUNT == 2


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"size": (0, 4)},
            "target_size must contain two positive integers",
        ),
        ({"mode": "CMYK"}, "mode must be one of"),
        ({"regularization": 0.0}, "regularization must be positive"),
        ({"max_iter": 0}, "max_iter must be positive"),
    ],
)
def test_an_invalid_config_is_rejected(changes, message):
    with pytest.raises(errors.InputError, match=message):
        config(**changes)


def test_a_config_serialises_its_size_as_a_list():
    assert records.Config().to_dict() == {
        "size": [16, 16],
        "mode": "RGB",
        "normalize": True,
        "regularization": 1.0,
        "max_iter": 500,
        "random_state": 0,
    }


def test_the_evidence_records_serialise_to_their_layouts():
    assert (
        records.Evaluation(2, 1.0, 1.0, 0.1, ("b", "r")).to_dict()["samples"]
        == 2
    )
    assert records.Explanation(
        "m", "p", "x", (("t", 2.0),), ("l",)
    ).to_dict() == {
        "digests": {"model": "m", "payload": "p"},
        "method": "x",
        "terms": [["t", 2.0]],
        "limitations": ["l"],
    }
    assert records.Package("p", "d").to_dict() == {
        "path": "p",
        "digest": "d",
        "schema_version": 1,
    }


@pytest.mark.parametrize(
    ("pixels", "duplicates", "shifted"),
    [(0.25, 0.25, False), (0.26, 0.0, True), (0.0, 0.26, True)],
)
def test_a_drift_is_shifted_when_a_signal_passes_the_threshold(
    pixels, duplicates, shifted
):
    drift = records.Drift("r", "o", pixels, duplicates, 0.25)
    assert drift.shifted is shifted
    assert drift.to_dict() == {
        "digests": {"reference": "r", "observed": "o"},
        "pixels": pixels,
        "duplicates": duplicates,
        "threshold": 0.25,
        "shifted": shifted,
    }


# estimator


def test_an_unfitted_estimator_refuses_to_predict():
    model = estimator.Estimator()
    assert model.config == records.Config()
    assert model.names == ()
    assert model.fitted is False
    with pytest.raises(
        errors.OrderError, match="image classifier has not been fitted"
    ):
        model.predict(HELD)


def test_feature_names_follow_row_column_channel_order():
    model = estimator.Estimator(config(size=(2, 1), mode="RGB"))
    assert model.label() == (
        "pixel[0,0,0]",
        "pixel[0,0,1]",
        "pixel[0,0,2]",
        "pixel[0,1,0]",
        "pixel[0,1,1]",
        "pixel[0,1,2]",
    )
    grey = estimator.Estimator(config(size=(1, 2), mode="L"))
    assert grey.label() == ("pixel[0,0,0]", "pixel[1,0,0]")


def test_a_fit_matches_a_logistic_regression_on_flattened_pixels(fitted):
    features = TRAINING.flatten((4, 4), "RGB", True)
    model = LogisticRegression(
        C=1.0, max_iter=500, random_state=0, solver="lbfgs"
    )
    model.fit(features, [s.label for s in TRAINING.samples])
    numpy.testing.assert_allclose(fitted.estimator.coef_, model.coef_)
    assert len(fitted.names) == 12
    assert fitted.classes == ("blue", "red")
    assert fitted.predict(HELD) == ("red", "blue")
    numpy.testing.assert_allclose(
        numpy.asarray(fitted.predict_proba(HELD)),
        model.predict_proba(HELD.flatten((4, 4), "RGB", True)),
    )
    assert fitted.predict(Pictures(())) == ()
    assert fitted.predict_proba(Pictures(())) == ()


def test_a_model_digest_hashes_the_version_one_config(fitted):
    state = {
        "config": scmle.contracts.records.downgrade(
            records.Config.migrations, config().to_dict()
        ),
        "feature_names": list(fitted.names),
        "classes": fitted.estimator.classes_.tolist(),
        "coef": fitted.estimator.coef_.tolist(),
        "intercept": fitted.estimator.intercept_.tolist(),
    }
    encoded = json.dumps(
        state, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()
    assert fitted.digest == hashlib.sha256(encoded).hexdigest()
    assert "target_size" in state["config"]


def test_a_fit_rejects_unlabelled_or_single_class_images():
    model = estimator.Estimator(config())
    unlabelled = Pictures((*TRAINING.samples, Picture(tile(0.5, 0.5))))
    with pytest.raises(
        errors.InputError, match="every training image must have a label"
    ):
        model.fit(unlabelled)
    with pytest.raises(errors.InputError, match="must contain two classes"):
        model.fit(Pictures(TRAINING.samples[:2]))
    assert model.fitted is False


def test_a_fit_rejects_duplicates_blocking_findings_and_empty_pixels():
    model = estimator.Estimator(config())
    copies = Pictures(TRAINING.samples, Report([0.0], duplicates=1))
    with pytest.raises(
        errors.InputError,
        match="training images contain duplicate content; deduplicate before fitting",
    ):
        model.fit(copies)
    conflict = Pictures(
        TRAINING.samples,
        Report(
            [0.0], blocking=("risk.image-conflicting-duplicate-labels", "other")
        ),
    )
    with pytest.raises(errors.InputError) as caught:
        model.fit(conflict)
    assert str(caught.value) == (
        "image curation has unresolved blocking findings (duplicate content has "
        "conflicting labels): risk.image-conflicting-duplicate-labels, other"
    )
    review = Pictures(TRAINING.samples, Report([0.0], blocking=("risk.other",)))
    with pytest.raises(errors.InputError) as caught:
        model.fit(review)
    assert str(caught.value) == (
        "image curation has unresolved blocking findings (curation policy requires review): risk.other"
    )
    blank = Pictures((Picture([], "red"), Picture([], "blue")))
    with pytest.raises(
        errors.InputError, match="training images contain no usable pixels"
    ):
        model.fit(blank)
    assert model.fitted is False


def test_an_unfitted_estimator_refuses_its_properties():
    model = estimator.Estimator(config())
    with pytest.raises(
        errors.OrderError, match="image classifier has not been fitted"
    ):
        assert model.classes is None
    with pytest.raises(
        errors.OrderError, match="image classifier has not been fitted"
    ):
        assert model.digest is None
    with pytest.raises(
        errors.OrderError, match="image classifier has not been fitted"
    ):
        model.transform(HELD)


def test_a_fit_keeps_the_curation_report(fitted):
    assert fitted.report is TRAINING.report
    assert fitted.fitted is True


def test_an_evaluation_scores_held_out_images(fitted):
    found = fitted.evaluate(HELD)
    assert found.samples == 2
    assert found.accuracy == 1.0
    assert found.macro_f1 == 1.0
    assert found.labels == ("blue", "red")
    p = numpy.asarray(fitted.predict_proba(HELD))
    assert found.log_loss == pytest.approx(
        -numpy.mean(numpy.log([p[0, 1], p[1, 0]]))
    )
    with pytest.raises(
        errors.InputError, match="evaluation images must not be empty"
    ):
        fitted.evaluate(Pictures(()))
    with pytest.raises(
        errors.InputError, match="every evaluation image must have a label"
    ):
        fitted.evaluate(Pictures((Picture(tile(0.5, 0.5)),)))


def test_an_evaluation_counts_a_wrong_prediction(fitted):
    swapped = Pictures(
        (Picture(tile(0.98, 0.08), "blue"), Picture(tile(0.08, 0.98), "blue"))
    )
    found = fitted.evaluate(swapped)
    assert found.accuracy == 0.5
    assert found.macro_f1 == pytest.approx(1 / 3)


# classifier


def test_an_overview_ranks_mean_absolute_pixel_weights(fitted):
    found = fitted.overview(top_k=4)
    strengths = numpy.abs(fitted.estimator.coef_).mean(axis=0)
    expected = sorted(
        zip(fitted.names, strengths, strict=True),
        key=lambda i: (-float(i[1]), i[0]),
    )[:4]
    assert found.terms == tuple((n, float(v)) for n, v in expected)
    assert found.method == "linear_global_pixel_weight"
    assert found.input_hash is None
    with pytest.raises(errors.InputError, match="top_k must be positive"):
        fitted.overview(top_k=0)


def test_a_spotlight_signs_pixels_toward_the_predicted_class(fitted, wrapped):
    target = HELD.samples[1]
    found = fitted.spotlight(target, top_k=5)
    features = fitted.transform(Pictures((target,)))[0]
    coefficients = -fitted.estimator.coef_[0]
    expected = sorted(
        zip(fitted.names, features * coefficients, strict=True),
        key=lambda i: (-abs(float(i[1])), i[0]),
    )[:5]
    assert found.terms == tuple((n, float(v)) for n, v in expected)
    assert found.input_hash == target.content_hash
    red = fitted.spotlight(HELD.samples[0], top_k=1)
    assert red.terms[0][1] > 0
    with pytest.raises(errors.InputError, match="top_k must be positive"):
        fitted.spotlight(target, top_k=0)


def test_a_confidence_is_the_signed_decision(fitted):
    blue = fitted.transform(Pictures((HELD.samples[1],)))
    red = fitted.transform(Pictures((HELD.samples[0],)))
    assert fitted.confidence(blue) == pytest.approx(
        -float(fitted.estimator.decision_function(blue)[0])
    )
    assert fitted.confidence(red) == pytest.approx(
        float(fitted.estimator.decision_function(red)[0])
    )


def test_an_occlusion_zeroes_each_patch_and_reports_the_drop(fitted, wrapped):
    target = HELD.samples[0]
    found = fitted.occlude(target, top_k=10, patch_size=(1, 2))
    names = sorted(name for name, ignored in found.terms)
    assert names == ["region[0:2,0:1]", "region[0:2,1:2]"]
    features = fitted.transform(Pictures((target,)))
    base = fitted.confidence(features)
    left = features.copy()
    left[
        0,
        [
            (r * 2 + c) * 3 + ch
            for r in range(2)
            for c in range(1)
            for ch in range(3)
        ],
    ] = 0.0
    assert dict(found.terms)["region[0:2,0:1]"] == pytest.approx(
        base - fitted.confidence(left)
    )
    assert found.method == "occlusion_pixel_region"
    assert found.input_hash == target.content_hash
    assert found.digest == fitted.digest
    whole = fitted.occlude(target, top_k=10, patch_size=(3, 3))
    assert [n for n, ignored in whole.terms] == ["region[0:2,0:2]"]
    assert whole.terms[0][1] == pytest.approx(
        base - fitted.confidence(numpy.zeros_like(features))
    )
    single = fitted.occlude(target, top_k=10, patch_size=(1, 1))
    assert sorted(n for n, ignored in single.terms) == [
        "region[0:1,0:1]",
        "region[0:1,1:2]",
        "region[1:2,0:1]",
        "region[1:2,1:2]",
    ]
    magnitudes = [abs(v) for ignored, v in single.terms]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert len(fitted.occlude(target, top_k=1, patch_size=(1, 1)).terms) == 1
    with pytest.raises(errors.InputError, match="top_k must be positive"):
        fitted.occlude(target, top_k=0)


@pytest.mark.parametrize("patch", [(0, 2), (2,), (2, 2, 2)])
def test_an_occlusion_rejects_bad_patches(fitted, patch):
    with pytest.raises(
        errors.InputError, match="patch_size must contain positive dimensions"
    ):
        fitted.occlude(HELD.samples[0], patch_size=patch)


# archive


def test_a_package_round_trips_with_trust_and_writes_version_two(
    tmp_path, fitted
):
    target = tmp_path / "model"
    package = fitted.save(target)
    assert package.digest == archive.digest(target / "model.joblib")
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["schema_version"] == archive.MANIFEST == 2
    loaded = classifier.Classifier.load(target, trusted=True)
    assert loaded.digest == fitted.digest
    path = target / "manifest.json"
    for version in (1, 2):
        path.write_text(json.dumps({**manifest, "schema_version": version}))
        classifier.Classifier.load(target, trusted=True)
    for version in (0, 3):
        path.write_text(json.dumps({**manifest, "schema_version": version}))
        with pytest.raises(
            errors.InputError, match="unsupported image model package version"
        ):
            classifier.Classifier.load(target, trusted=True)
    path.write_text(json.dumps({**manifest, "model_path": "none"}))
    with pytest.raises(errors.InputError, match="has no model payload"):
        classifier.Classifier.load(target, trusted=True)


def test_a_package_rejects_a_changed_model_digest(tmp_path, fitted):
    target = tmp_path / "model"
    fitted.save(target)
    path = target / "manifest.json"
    manifest = json.loads(path.read_text())
    path.write_text(json.dumps({**manifest, "model_digest": "0" * 64}))
    with pytest.raises(
        errors.InputError, match="digest does not match manifest"
    ):
        classifier.Classifier.load(target, trusted=True)


# monitor


def test_a_monitor_needs_a_threshold_in_the_unit_interval():
    reference = TRAINING.curate()
    with pytest.raises(
        errors.InputError, match=r"threshold must be in \(0, 1\]"
    ):
        monitor.Monitor(reference, 0.0)
    assert monitor.Monitor(reference).threshold == 0.25


def test_a_monitor_measures_mean_pixel_distance():
    reference = Report([100.0, 50.0, 0.0], count=4)
    watcher = monitor.Monitor(reference, 0.1)
    same = watcher.observe(
        Pictures(TRAINING.samples, Report([100.0, 50.0, 0.0]))
    )
    assert same.pixels == 0.0
    assert same.duplicates == 0.0
    assert same.shifted is False
    observed = Report([151.0, 50.0, 0.0], duplicates=1, count=4)
    drift = watcher.observe(Pictures(TRAINING.samples, observed))
    assert drift.pixels == pytest.approx(51.0 / 3 / 255)
    assert drift.duplicates == 0.25
    assert drift.shifted is True
    assert drift.threshold == 0.1
    assert drift.reference_hash == reference.dataset_hash
    assert drift.observed_hash == observed.dataset_hash


def test_a_monitor_rates_an_empty_observation_without_dividing_by_zero():
    watcher = monitor.Monitor(Report([0.0]), 1.0)
    found = watcher.observe(Pictures((), Report([0.0], duplicates=0, count=0)))
    assert found.duplicates == 0.0
    assert found.shifted is False


@pytest.mark.parametrize("threshold", [-0.1, 0.0, 1.0001])
def test_a_monitor_rejects_a_threshold_outside_the_unit_interval(threshold):
    with pytest.raises(
        errors.InputError, match=r"threshold must be in \(0, 1\]"
    ):
        monitor.Monitor(Report([0.0]), threshold)
