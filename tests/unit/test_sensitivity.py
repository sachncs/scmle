"""Training, evaluation and performance sensitivity."""

import numpy as np
import pandas as pd
import pytest
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import tree

from scmle.evaluation import sensitivity


def data(rows=400, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {"a": rng.normal(size=rows), "b": rng.normal(size=rows)}
    )
    target = pd.Series(
        (frame.a + 0.5 * frame.b + rng.normal(0, 0.8, rows) > 0).astype(int)
    )
    return frame, target


def auc(model, features, target):
    return metrics.roc_auc_score(target, model.predict_proba(features)[:, 1])


def test_a_deterministic_model_has_no_seed_sensitivity_but_does_resample():
    features, target = data()

    by_seed, by_resample = sensitivity.training_sensitivity(
        linear_model.LogisticRegression, features, target, auc, repeats=6
    )

    assert by_seed.std == pytest.approx(0.0, abs=1e-12)
    assert by_resample.std > 0.0


def test_a_random_model_moves_with_its_seed():
    features, target = data()

    by_seed = sensitivity.training_sensitivity(
        lambda: ensemble.RandomForestClassifier(n_estimators=10),
        features,
        target,
        auc,
        repeats=6,
    )[0]

    assert by_seed.std > 0.0


def test_the_split_alone_moves_the_score():
    features, target = data(200)

    spread = sensitivity.evaluation_sensitivity(
        linear_model.LogisticRegression, features, target, auc, repeats=8
    )

    assert spread.range > 0.01 and len(spread.scores) == 8


def test_noise_lowers_performance_and_the_slope_says_so():
    features, target = data(600)
    model = linear_model.LogisticRegression().fit(features, target)

    curve, slope = sensitivity.performance_sensitivity(
        model, features, target, auc
    )

    assert curve[0.0] > curve[1.0] and slope < 0


def test_the_report_names_the_dominant_source_and_serializes():
    features, target = data(300)

    report = sensitivity.analyze(
        lambda: tree.DecisionTreeClassifier(max_depth=4),
        features,
        target,
        auc,
        repeats=6,
    )

    assert report.dominant() in {"seed", "resample", "split"}
    assert set(report.to_dict()) == {
        "seed",
        "resample",
        "split",
        "noise",
        "decay",
    }


def test_too_few_rows_are_refused():
    features, target = data(20)
    with pytest.raises(ValueError, match="rows"):
        sensitivity.analyze(
            linear_model.LogisticRegression, features, target, auc
        )


def test_noise_on_a_single_row_column_leaves_it_finite():
    frame = pd.DataFrame({"a": [1.0]})
    out = sensitivity.noisy(frame, 0.5, np.random.default_rng(0))
    assert np.isfinite(out["a"]).all()
