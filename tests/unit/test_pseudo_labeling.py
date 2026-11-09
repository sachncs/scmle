"""Self-training labels only confident rows and is judged on held-out rows."""

import numpy
import pandas
from sklearn import linear_model
from sklearn import metrics

from scmle.sampling import pseudo_labeling


def blobs(rows, seed):
    rng = numpy.random.default_rng(seed)
    labels = rng.integers(0, 2, rows)
    features = pandas.DataFrame(
        {
            "a": rng.normal(labels * 3.0, 1.0),
            "b": rng.normal(0, 1.0, rows),
        }
    )
    return features, pandas.Series(labels)


def test_only_confident_rows_receive_a_label():
    features, labels = blobs(300, 0)
    fitted = linear_model.LogisticRegression().fit(features, labels)
    unlabeled = blobs(400, 1)[0]

    rows, given = pseudo_labeling.confident_rows(fitted, unlabeled, 0.95)

    assert 0 < len(rows) < len(unlabeled)
    assert (fitted.predict_proba(rows).max(axis=1) >= 0.95).all()
    assert list(given.index) == list(rows.index)


def test_too_few_confident_rows_leave_the_model_unchanged():
    features, labels = blobs(300, 0)
    fitted = linear_model.LogisticRegression().fit(features, labels)
    unlabeled = blobs(5, 1)[0]

    model, added = pseudo_labeling.augmented_fit(
        fitted, features, labels, unlabeled, 0.9
    )

    assert added == 0 and model is fitted


def test_judging_reports_the_gain_over_the_plain_fit():
    features, labels = blobs(120, 0)
    unlabeled = blobs(800, 1)[0]
    folds = [(list(range(0, 90)), list(range(90, 120)))]

    outcome = pseudo_labeling.judge(
        linear_model.LogisticRegression(),
        features,
        labels,
        folds,
        unlabeled,
        metrics.get_scorer("roc_auc"),
        0.9,
    )

    assert outcome is not None
    assert outcome.folds == 1 and outcome.pseudo_rows >= 20
    assert outcome.accepted == (outcome.gain > 0)


def test_time_running_out_abandons_the_judgement():
    features, labels = blobs(120, 0)
    outcome = pseudo_labeling.judge(
        linear_model.LogisticRegression(),
        features,
        labels,
        [(list(range(90)), list(range(90, 120)))],
        features,
        metrics.get_scorer("roc_auc"),
        0.9,
        lambda: True,
    )

    assert outcome is None
