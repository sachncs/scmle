"""Feature construction and sparse selection: defaults and fallbacks."""

import numpy
import pandas
from sklearn import linear_model

from scmle.preparation import selection


def test_a_constructor_without_a_target_builds_nothing() -> None:
    features = pandas.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    constructor = selection.Constructor(("a", "b")).fit(features, None)
    assert constructor.pairs_ == []
    assert constructor.transform(features).equals(features)


def test_the_selector_is_seeded_with_zero_by_default() -> None:
    assert selection.Selector().random_state == 0


def test_the_selector_seeds_its_classifier(monkeypatch) -> None:
    seen = []
    original = linear_model.LogisticRegression

    def record(**settings):
        seen.append(settings["random_state"])
        return original(**settings)

    monkeypatch.setattr(selection.linear_model, "LogisticRegression", record)
    rng = numpy.random.default_rng(0)
    matrix = rng.normal(size=(60, 3))
    target = (matrix[:, 0] > 0).astype(int)
    selection.Selector(random_state=4).fit(matrix, target)
    assert seen == [4]


def test_a_fully_penalised_classifier_keeps_its_strongest_column() -> None:
    rng = numpy.random.default_rng(1)
    matrix = rng.normal(size=(80, 3))
    target = numpy.where(matrix[:, 2] > 0, "yes", "no")
    chooser = selection.Selector(strength=1e-9).fit(matrix, target)
    assert chooser.support_.tolist() == [False, False, True]
    assert chooser.n_features_in_ == 3
