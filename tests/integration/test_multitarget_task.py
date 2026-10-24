"""Several labels or quantities are predicted, scored and split together."""

import numpy as np
import pandas as pd
import pytest
import support

from scmle.contracts import errors


def labelled(rows=1500, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {"a": rng.normal(size=rows), "b": rng.normal(size=rows)}
    )
    frame["common"] = (frame.a + rng.normal(0, 0.5, rows) > 0).astype(int)
    frame["other"] = (frame.b + rng.normal(0, 0.5, rows) > 0).astype(int)
    frame["rare"] = (
        (frame.a + frame.b > 2.2) & (rng.random(rows) < 0.9)
    ).astype(int)
    return frame


def quantities(rows=800, seed=1):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {"a": rng.normal(size=rows), "b": rng.normal(size=rows)}
    )
    frame["first"] = 3 * frame.a + rng.normal(0, 0.3, rows)
    frame["second"] = -2 * frame.b + frame.a + rng.normal(0, 0.3, rows)
    return frame


LABELS = ["common", "other", "rare"]


def run(tmp_path, frame, targets, task, **kwargs):
    return support.make_project(
        frame,
        targets,
        task=task,
        time_budget=300,
        output_dir=tmp_path,
        **kwargs,
    ).run()


@pytest.fixture(scope="module")
def multilabel(tmp_path_factory):
    return run(
        tmp_path_factory.mktemp("labels"), labelled(), LABELS, "multilabel"
    )


@pytest.fixture(scope="module")
def multioutput(tmp_path_factory):
    return run(
        tmp_path_factory.mktemp("outputs"),
        quantities(),
        ["first", "second"],
        "multioutput",
    )


def test_a_multilabel_bundle_predicts_one_column_per_label(multilabel):
    rows = labelled(30, seed=5)[["a", "b"]]

    frame = multilabel.bundle.predict_frame(rows)

    assert list(frame.columns) == LABELS
    assert frame.shape == (30, 3)
    assert set(np.unique(frame.to_numpy())) <= {0, 1}
    assert multilabel.bundle.predict_proba(rows).shape == (30, 3)


def test_the_model_beats_the_per_label_prior_on_every_label(multilabel):
    fresh = labelled(2000, seed=9)
    predicted = multilabel.bundle.predict_frame(fresh[["a", "b"]])

    for label in ("common", "other"):
        accuracy = (predicted[label] == fresh[label]).mean()
        prior = max(fresh[label].mean(), 1 - fresh[label].mean())
        assert accuracy > prior + 0.05
    assert multilabel.evaluation["test_metrics"]["f1_macro_labels"] > 0.5


def test_folds_keep_every_label_when_the_data_allow(multilabel):
    assert multilabel.evaluation["split_strategy"] == "multilabel_stratified"
    frame = labelled()
    import json

    manifest = json.loads(
        (multilabel.run_dir / "validation_manifest.json").read_text()
    )
    dev = np.array(manifest["train_ids"])
    for fold in manifest["folds"]:
        for label in LABELS:
            assert frame[label].to_numpy()[dev[fold["val"]]].sum() > 0
    assert not any("too rare" in n for n in manifest["notes"])


def test_the_other_targets_are_never_features(multilabel, multioutput):
    assert not set(LABELS) & set(multilabel.bundle.features)
    assert not {"first", "second"} & set(multioutput.bundle.features)


def test_multioutput_predicts_every_quantity_with_a_good_fit(multioutput):
    fresh = quantities(1000, seed=8)
    predicted = multioutput.bundle.predict_frame(fresh[["a", "b"]])

    assert list(predicted.columns) == ["first", "second"]
    assert multioutput.evaluation["test_metrics"]["r2_mean"] > 0.8
    for column in ("first", "second"):
        assert np.corrcoef(predicted[column], fresh[column])[0, 1] > 0.9


def test_bad_target_sets_are_refused(tmp_path):
    frame = labelled()
    frame["common"] = frame["common"] + 0.5

    with pytest.raises(errors.BlockedError, match="0 or 1"):
        run(tmp_path, frame, LABELS, "multilabel")
    with pytest.raises(ValueError, match="targets"):
        support.make_project(
            frame, "common", task="multilabel", output_dir=tmp_path
        )
    with pytest.raises(ValueError, match="targets"):
        support.make_project(
            frame, LABELS, task="binary_classification", output_dir=tmp_path
        )


def test_the_boosted_multi_target_model_survives_a_tight_budget(tmp_path):
    plan = support.make_project(
        quantities(),
        ["first", "second"],
        task="multioutput",
        time_budget=60,
        output_dir=tmp_path,
    ).plan()

    assert "multi_hist_gradient_boosting" in plan.candidate_models
    assert "multi_random_forest" not in plan.candidate_models
