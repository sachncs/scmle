"""Ordered labels are modelled, scored and predicted as ordered labels."""

import numpy as np
import pandas as pd
import pytest
import support

from mml.contracts import errors

ORDER = ["low", "medium", "high", "extreme"]


def frame(rows=1200, seed=0):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {"a": rng.normal(size=rows), "b": rng.normal(size=rows)}
    )
    latent = 1.5 * data.a - data.b + rng.logistic(size=rows)
    cuts = np.quantile(latent, [0.25, 0.5, 0.75])
    data["level"] = np.array(ORDER)[np.searchsorted(cuts, latent)]
    return data


def run(tmp_path, data, **kwargs):
    kwargs.setdefault("ordinal_order", ORDER)
    return support.make_project(
        data,
        "level",
        task="ordinal",
        time_budget=300,
        output_dir=tmp_path,
        **kwargs,
    ).run()


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return run(tmp_path_factory.mktemp("ordinal"), frame())


def test_the_model_beats_the_median_baseline_on_agreement(result):
    assert result.evaluation["test_metrics"]["qwk"] > 0.4
    baseline = next(t for t in result.trials if t.model_name == "dummy")
    assert baseline.score_mean < 0.1


def test_predictions_are_the_users_labels_in_the_stated_order(result):
    rows = pd.DataFrame({"a": [-3.0, 3.0], "b": [0.0, 0.0]})

    low, high = result.bundle.predict(rows)

    assert low in ORDER and high in ORDER
    assert ORDER.index(low) < ORDER.index(high)
    assert list(result.bundle.classes) == ORDER


def test_probabilities_follow_the_label_order(result):
    probabilities = result.bundle.predict_proba(
        pd.DataFrame({"a": [-3.0, 3.0], "b": [0.0, 0.0]})
    )

    assert probabilities.shape == (2, 4)
    assert probabilities[0].argmax() < probabilities[1].argmax()


def test_an_order_that_omits_a_present_label_is_refused(tmp_path):
    with pytest.raises(errors.BlockedError, match="omits"):
        run(tmp_path, frame(), ordinal_order=["low", "medium", "high"])


def test_an_order_that_lists_an_absent_label_is_refused(tmp_path):
    with pytest.raises(errors.BlockedError, match="no rows"):
        run(tmp_path, frame(), ordinal_order=[*ORDER, "never"])


def test_ordinal_needs_an_order_and_other_tasks_forbid_one(tmp_path):
    with pytest.raises(ValueError, match="ordinal_order"):
        support.make_project(
            frame(), "level", task="ordinal", output_dir=tmp_path
        )
    with pytest.raises(ValueError, match="ordinal_order"):
        support.make_project(
            frame(),
            "level",
            task="multiclass_classification",
            ordinal_order=ORDER,
            output_dir=tmp_path,
        )


def test_ordinal_modelling_misses_by_fewer_levels_than_plain_classes(
    result, tmp_path
):
    plain = support.make_project(
        frame(),
        "level",
        task="multiclass_classification",
        time_budget=300,
        output_dir=tmp_path,
    ).run()
    fresh = frame(3000, seed=9)
    truth = fresh.level.map(ORDER.index).to_numpy()
    features = fresh[["a", "b"]]

    ordinal_guess = pd.Series(result.bundle.predict(features)).map(ORDER.index)
    plain_guess = pd.Series(plain.bundle.predict(features)).map(ORDER.index)

    ordinal_error = np.abs(truth - ordinal_guess.to_numpy()).mean()
    plain_error = np.abs(truth - plain_guess.to_numpy()).mean()
    assert ordinal_error < plain_error
