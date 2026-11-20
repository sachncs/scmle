"""The TF-IDF document model: records, estimator, explanations, packages."""

import hashlib
import json
import math

import numpy
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from scmle.contracts import errors
from scmle.data.modalities.document import Dataset
from scmle.data.modalities.document import Sample
from scmle.models.documents import archive
from scmle.models.documents import classifier
from scmle.models.documents import estimator
from scmle.models.documents import monitor
from scmle.models.documents import records

TRAINING = (
    Sample("train-1", "red apple orchard fruit", label="fruit"),
    Sample("train-2", "red orange orchard harvest", label="fruit"),
    Sample("train-3", "blue ocean water sailing", label="water"),
    Sample("train-4", "blue lake water boat", label="water"),
)
HELD = (
    Sample("test-1", "red fruit harvest", label="fruit"),
    Sample("test-2", "blue water sailing", label="water"),
)
THREE = (
    *TRAINING,
    Sample("train-5", "green grass meadow field", label="land"),
    Sample("train-6", "green hill field meadow", label="land"),
)


def config():
    return records.Config(max_features=32, ngram_range=(1, 1))


@pytest.fixture(scope="module")
def fitted():
    return classifier.Classifier(config()).fit(TRAINING)


def reference(samples=TRAINING):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=32,
        ngram_range=(1, 1),
        min_df=1,
        token_pattern=r"(?u)\b\w\w+\b",
    )
    matrix = vectorizer.fit_transform([s.content for s in samples])
    model = LogisticRegression(
        C=1.0, max_iter=500, random_state=0, solver="lbfgs"
    )
    model.fit(matrix, [s.label for s in samples])
    return vectorizer, model


# records


def test_the_record_constants_are_fixed():
    assert records.NGRAM_BOUND_COUNT == 2
    assert records.MIN_CLASS_COUNT == 2
    assert records.BINARY_CLASS_COUNT == 2


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"max_features": 0}, "max_features must be positive"),
        (
            {"ngram_range": (2, 1)},
            "ngram_range must contain positive ordered bounds",
        ),
        (
            {"ngram_range": (0, 1)},
            "ngram_range must contain positive ordered bounds",
        ),
        (
            {"ngram_range": (1, 2, 3)},
            "ngram_range must contain positive ordered bounds",
        ),
        ({"min_df": 0}, "min_df must be positive"),
        ({"regularization": 0.0}, "regularization must be positive"),
        ({"max_iter": 0}, "max_iter must be positive"),
    ],
)
def test_an_invalid_config_is_rejected(changes, message):
    with pytest.raises(errors.InputError, match=message):
        records.Config(**changes)


def test_a_config_serialises_with_a_list_range():
    assert records.Config().to_dict() == {
        "max_features": 10_000,
        "ngram_range": [1, 2],
        "min_df": 1,
        "regularization": 1.0,
        "max_iter": 500,
        "random_state": 0,
    }
    assert records.Config(ngram_range=(2, 2)).ngram_range == (2, 2)


def test_the_evidence_records_serialise_to_their_layouts():
    assert records.Evaluation(3, 0.5, 0.4, 0.7, ("a", "b")).to_dict() == {
        "samples": 3,
        "accuracy": 0.5,
        "macro_f1": 0.4,
        "log_loss": 0.7,
        "labels": ("a", "b"),
    }
    explanation = records.Explanation("m", None, "x", (("t", 1.0),), ("l",))
    assert explanation.to_dict() == {
        "digests": {"model": "m", "payload": None},
        "method": "x",
        "terms": [["t", 1.0]],
        "limitations": ["l"],
        "citations": [],
    }
    assert records.Package("p", "d").to_dict() == {
        "path": "p",
        "digest": "d",
        "schema_version": 1,
    }


@pytest.mark.parametrize(
    ("values", "shifted"),
    [
        ((0.2, 0.2, 0.2), False),
        ((0.21, 0.0, 0.0), True),
        ((0.0, -0.21, 0.0), True),
        ((0.0, 0.0, 0.21), True),
        ((0.2, -0.2, 0.2), False),
    ],
)
def test_a_drift_is_shifted_when_any_signal_passes_the_threshold(
    values, shifted
):
    drift = records.Drift("r", "o", *values, 0.2)
    assert drift.shifted is shifted
    record = drift.to_dict()
    assert record["shifted"] is shifted
    assert record["digests"] == {"reference": "r", "observed": "o"}
    assert record["language"] == values[0]
    assert record["characters"] == values[1]
    assert record["duplicates"] == values[2]


# estimator


def test_an_unfitted_estimator_refuses_to_predict():
    model = estimator.Estimator()
    assert model.config == records.Config()
    assert model.fitted is False
    assert model.report is None
    with pytest.raises(
        errors.OrderError, match="document classifier has not been fitted"
    ):
        model.predict(HELD)
    with pytest.raises(errors.OrderError):
        ignored = model.digest


def test_a_fit_matches_tfidf_and_logistic_regression(fitted):
    vectorizer, model = reference()
    assert fitted.fitted is True
    assert fitted.vectorizer.vocabulary_ == vectorizer.vocabulary_
    numpy.testing.assert_allclose(fitted.estimator.coef_, model.coef_)
    assert fitted.classes == ("fruit", "water")
    probabilities = fitted.predict_proba(HELD)
    expected = model.predict_proba(
        vectorizer.transform([s.content for s in HELD])
    )
    numpy.testing.assert_allclose(numpy.asarray(probabilities), expected)
    assert fitted.predict(HELD) == ("fruit", "water")
    assert fitted.predict(()) == ()
    assert fitted.predict_proba(()) == ()
    assert fitted.report is not None


def test_a_model_digest_hashes_the_fitted_state(fitted):
    vectorizer, model = reference()
    state = {
        "config": config().to_dict(),
        "vocabulary": sorted(
            (str(t), int(i)) for t, i in vectorizer.vocabulary_.items()
        ),
        "classes": model.classes_.tolist(),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
    }
    encoded = json.dumps(
        state, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()
    assert fitted.digest == hashlib.sha256(encoded).hexdigest()


def test_a_fit_rejects_empty_unlabelled_single_class_or_tokenless_input():
    model = estimator.Estimator(config())
    with pytest.raises(
        errors.InputError, match="training_samples must not be empty"
    ):
        model.fit(())
    with pytest.raises(
        errors.InputError, match="every training document must have a label"
    ):
        model.fit((*TRAINING, Sample("x", "plain words here")))
    with pytest.raises(errors.InputError, match="must contain two classes"):
        model.fit(TRAINING[:2])
    with pytest.raises(ValueError, match=r"vocabulary|no usable tokens"):
        model.fit(
            (Sample("a", "a b", label="x"), Sample("b", "c d", label="y"))
        )
    assert model.fitted is False


def test_an_evaluation_scores_held_out_documents(fitted):
    found = fitted.evaluate(HELD)
    assert found.samples == 2
    assert found.accuracy == 1.0
    assert found.macro_f1 == 1.0
    probabilities = numpy.asarray(fitted.predict_proba(HELD))
    assert found.log_loss == pytest.approx(
        -numpy.mean(numpy.log([probabilities[0, 0], probabilities[1, 1]]))
    )
    assert found.labels == ("fruit", "water")
    with pytest.raises(
        errors.InputError, match="evaluation samples must not be empty"
    ):
        fitted.evaluate(())
    with pytest.raises(
        errors.InputError, match="every evaluation document must have a label"
    ):
        fitted.evaluate((Sample("u", "red apple"),))


# classifier


def test_an_overview_ranks_mean_absolute_coefficients(fitted):
    found = fitted.overview(top_k=3)
    terms = fitted.vectorizer.get_feature_names_out()
    strengths = numpy.abs(fitted.estimator.coef_).mean(axis=0)
    expected = sorted(
        ((str(t), float(s)) for t, s in zip(terms, strengths, strict=True)),
        key=lambda i: (-i[1], i[0]),
    )[:3]
    assert found.terms == tuple(expected)
    assert found.input_hash is None
    assert found.method == "linear_global_term_weight"
    assert found.digest == fitted.digest
    assert len(found.limitations) == 2
    with pytest.raises(errors.InputError, match="top_k must be positive"):
        fitted.overview(top_k=0)


def test_a_spotlight_signs_contributions_toward_the_predicted_class(fitted):
    sample = HELD[0]
    found = fitted.spotlight(sample, top_k=10)
    row = fitted.transform((sample,)).toarray()[0]
    coefficients = -fitted.estimator.coef_[0]
    terms = fitted.vectorizer.get_feature_names_out()
    expected = sorted(
        (
            (str(terms[i]), float(row[i] * coefficients[i]))
            for i in numpy.flatnonzero(row)
        ),
        key=lambda i: (-abs(i[1]), i[0]),
    )
    assert found.terms == tuple(expected)
    assert all(value > 0 for ignored, value in found.terms)
    assert found.input_hash == sample.content_hash
    assert found.method == "linear_local_term_contribution"
    water = fitted.spotlight(HELD[1], top_k=1)
    assert len(water.terms) == 1
    assert water.terms[0][1] > 0
    with pytest.raises(errors.InputError, match="top_k must be positive"):
        fitted.spotlight(sample, top_k=0)


def test_a_confidence_is_the_signed_decision_of_the_predicted_class(fitted):
    fruit = fitted.transform((HELD[0],))
    water = fitted.transform((HELD[1],))
    assert fitted.confidence(fruit) == pytest.approx(
        -float(fitted.estimator.decision_function(fruit)[0])
    )
    assert fitted.confidence(water) == pytest.approx(
        float(fitted.estimator.decision_function(water)[0])
    )
    assert fitted.confidence(fruit) > 0 and fitted.confidence(water) > 0


def test_an_ablation_reports_score_drops_per_removed_term(fitted):
    sample = HELD[0]
    found = fitted.ablate(sample, top_k=2, max_terms=5)
    base = fitted.confidence(fitted.transform((sample,)))
    for term, delta in found.terms:
        content = sample.content.replace(term, " ")
        assert delta == pytest.approx(
            base - fitted.confidence(fitted.vectorizer.transform([content]))
        )
    assert len(found.terms) == 2
    assert found.method == "term_ablation"
    assert len(found.limitations) == 3
    one = fitted.ablate(sample, top_k=5, max_terms=1)
    assert len(one.terms) == 1
    with pytest.raises(errors.InputError, match="max_terms must be positive"):
        fitted.ablate(sample, max_terms=0)


def test_an_ablation_that_flips_the_class_is_scored_on_the_original_class(
    fitted,
):
    """MM-13: removing `orchard` flips the prediction to water."""
    sample = Sample("q", "orchard ocean")
    matrix = fitted.transform((sample,))
    assert fitted.estimator.predict(matrix)[0] == "fruit"
    removed = fitted.vectorizer.transform(["ocean"])
    assert fitted.estimator.predict(removed)[0] == "water"
    index = list(fitted.estimator.classes_).index("fruit")
    base = fitted.confidence(matrix)

    found = dict(fitted.ablate(sample, top_k=2).terms)

    assert found["orchard"] == pytest.approx(
        base - fitted.confidence(removed, index)
    )
    assert found["orchard"] > base > 0


def test_a_multiclass_model_uses_the_class_row():
    model = classifier.Classifier(config()).fit(THREE)
    sample = Sample("q", "green meadow")
    found = model.spotlight(sample)
    row = model.transform((sample,)).toarray()[0]
    index = list(model.classes).index("land")
    terms = model.vectorizer.get_feature_names_out()
    expected = {
        str(terms[i]): float(row[i] * model.estimator.coef_[index][i])
        for i in numpy.flatnonzero(row)
    }
    assert dict(found.terms) == pytest.approx(expected)
    matrix = model.transform((sample,))
    assert model.confidence(matrix) == pytest.approx(
        float(model.estimator.decision_function(matrix)[0, index])
    )
    ablated = model.ablate(sample, top_k=2)
    assert all(math.isfinite(v) for ignored, v in ablated.terms)


# archive


def test_a_package_round_trips_with_trust(tmp_path, fitted):
    target = tmp_path / "model"
    package = fitted.save(target)
    assert package.path == str(target)
    assert package.digest == archive.digest(target / "model.joblib")
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["model_path"] == "model.joblib"
    assert manifest["model_digest"] == fitted.digest
    loaded = classifier.Classifier.load(target, trusted=True)
    assert loaded.digest == fitted.digest
    with pytest.raises(FileExistsError, match="model package already exists"):
        fitted.save(target)


def test_a_package_load_rejects_missing_and_bad_manifests(tmp_path, fitted):
    with pytest.raises(FileNotFoundError):
        classifier.Classifier.load(tmp_path / "absent", trusted=True)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(errors.InputError, match="has no manifest"):
        classifier.Classifier.load(empty, trusted=True)
    target = tmp_path / "model"
    fitted.save(target)
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_path.write_text(json.dumps({**manifest, "schema_version": 2}))
    with pytest.raises(
        errors.InputError, match="unsupported document model package version"
    ):
        classifier.Classifier.load(target, trusted=True)
    manifest_path.write_text(
        json.dumps({**manifest, "model_path": "missing.joblib"})
    )
    with pytest.raises(errors.InputError, match="has no model payload"):
        classifier.Classifier.load(target, trusted=True)


def test_a_file_digest_is_the_sha256_of_the_bytes(tmp_path):
    path = tmp_path / "f"
    path.write_bytes(b"abc")
    assert archive.digest(path) == hashlib.sha256(b"abc").hexdigest()


# monitor


def test_a_monitor_needs_a_threshold_in_the_unit_interval():
    reference = Dataset(TRAINING).curate()
    for threshold in (0.0, 1.5):
        with pytest.raises(
            errors.InputError, match=r"threshold must be in \(0, 1\]"
        ):
            monitor.Monitor(reference, threshold)
    assert monitor.Monitor(reference, 1.0).threshold == 1.0
    assert monitor.Monitor(reference).threshold == 0.25


def test_a_monitor_of_the_same_corpus_reports_no_shift():
    reference = Dataset(TRAINING).curate()
    drift = monitor.Monitor(reference).observe(Dataset(TRAINING))
    assert drift.language == 0.0
    assert drift.characters == 0.0
    assert drift.duplicates == 0.0
    assert drift.reference_hash == reference.dataset_hash
    assert drift.shifted is False


def test_a_monitor_measures_length_and_duplicate_shift():
    reference = Dataset(TRAINING).curate()
    watcher = monitor.Monitor(reference, 0.2)
    long = tuple(
        Sample(f"l{i}", s.content * 3, label=s.label)
        for i, s in enumerate(TRAINING)
    )
    drift = watcher.observe(Dataset(long))
    base = watcher.measure(reference)
    assert base == pytest.approx(sum(len(s.content) for s in TRAINING) / 4)
    assert drift.characters == pytest.approx(
        (watcher.measure(Dataset(long).curate()) - base) / base
    )
    assert drift.shifted is True
    copies = (
        TRAINING[0],
        Sample("dup", TRAINING[0].content, label="fruit"),
        *TRAINING[1:3],
    )
    duplicated = watcher.observe(Dataset(copies))
    assert duplicated.duplicates == pytest.approx(2 / 4)
