"""Counterfactuals change only what they may and only as much as needed."""

from hypothesis import given
from hypothesis import settings
from hypothesis import strategies
import numpy
import pandas
import pytest

import scmle.artifacts.bundle
from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.curation import preprocessing
from scmle.models import registry

PLAN = data.FeaturePlan(("age", "income"), ("city",))
REGISTRY = registry.Catalog.with_builtins()


def people(rows=800, seed=0):
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame(
        {
            "age": rng.integers(20, 70, rows).astype(float),
            "income": rng.normal(50, 15, rows),
            "city": rng.choice(["north", "south", "east"], rows),
        }
    )
    score = (frame.income - 50) / 15 + (frame.city == "north") * 1.0
    frame["approved"] = numpy.where(
        score + rng.normal(0, 0.3, rows) > 0.4, "yes", "no"
    )
    return frame


@pytest.fixture(scope="module")
def classifier():
    frame = people()
    features = frame.drop(columns="approved")
    fitted = preprocessing.build_pipeline(
        PLAN,
        REGISTRY.get("logistic_regression"),
        tasks.Task.BINARY,
        0,
    ).fit(features, frame.approved)
    bundle = scmle.artifacts.bundle.Bundle.create(
        fitted,
        PLAN,
        features,
        "approved",
        "binary_classification",
        "roc_auc",
        "lr",
    )
    return bundle, features


def denied(bundle, features):
    predictions = bundle.predict(features)
    return features[predictions == "no"].iloc[0].to_dict()


def test_a_denied_applicant_gets_a_small_change_that_flips_the_decision(
    classifier,
):
    bundle, features = classifier
    row = denied(bundle, features)

    found = bundle.counterfactual(
        row, "yes", ["income", "city"], features, max_changes=2
    )

    assert found.found and found.prediction == "yes"
    assert 1 <= found.n_changed <= 2
    assert found.distance > 0
    assert bundle.predict(pandas.DataFrame([found.row]))[0] == "yes"


def test_every_change_in_the_answer_is_needed(classifier):
    bundle, features = classifier
    row = denied(bundle, features)

    found = bundle.counterfactual(
        row, "yes", ["age", "income", "city"], features, max_changes=3
    )

    for column in found.changed:
        reverted = {**found.row, column: row[column]}
        assert bundle.predict(pandas.DataFrame([reverted]))[0] != "yes"


def test_an_irrelevant_column_cannot_reach_the_outcome(classifier):
    bundle, features = classifier
    row = denied(bundle, features)

    found = bundle.counterfactual(row, "yes", ["age"], features)

    assert not found.found
    assert found.row["income"] == row["income"]


@settings(max_examples=25, deadline=None)
@given(
    index=strategies.integers(min_value=0, max_value=799),
    chosen=strategies.lists(
        strategies.sampled_from(["age", "income", "city"]),
        unique=True,
        min_size=1,
    ),
)
def test_columns_that_are_not_actionable_never_change(
    classifier, index, chosen
):
    bundle, features = classifier
    row = features.iloc[index].to_dict()

    found = bundle.counterfactual(row, "yes", chosen, features)

    fixed = set(row) - set(chosen)
    assert all(found.row[c] == row[c] for c in fixed)
    assert set(found.changed) <= set(chosen)


def test_unknown_columns_and_outcomes_are_refused(classifier):
    bundle, features = classifier
    row = denied(bundle, features)

    with pytest.raises(errors.ArtifactError, match="cannot be changed"):
        bundle.counterfactual(row, "yes", ["nope"], features)
    with pytest.raises(errors.ArtifactError, match="not one of"):
        bundle.counterfactual(row, "maybe", ["income"], features)


def test_regression_aims_for_a_range():
    frame = people()
    features = frame.drop(columns="approved")
    target = features.income * 2 + (features.city == "north") * 10
    fitted = preprocessing.build_pipeline(
        PLAN,
        REGISTRY.get("ridge")
        if "ridge" in REGISTRY.adapters
        else REGISTRY.get("elastic_net"),
        tasks.Task.REGRESSION,
        0,
    ).fit(features, target)
    bundle = scmle.artifacts.bundle.Bundle.create(
        fitted, PLAN, features, "y", "regression", "rmse", "linear"
    )
    row = features.iloc[0].to_dict()
    wanted = (float(target.max()) - 5, float(target.max()) + 5)

    found = bundle.counterfactual(row, wanted, ["income", "city"], features)

    assert found.found
    assert wanted[0] <= found.prediction <= wanted[1]
