"""Reweighing removes the group-label association and narrows the model gap."""

import json

import numpy
import pandas
import pytest

import scmle
from scmle.contracts import tasks
from scmle.evaluation import reweighing


def biased(rows=1600, seed=0):
    rng = numpy.random.default_rng(seed)
    group = rng.choice(["a", "b"], rows, p=[0.6, 0.4])
    signal = rng.normal(size=rows)
    # Group b receives fewer positives for the same signal.
    label = signal - 0.9 * (group == "b") + rng.normal(0, 0.5, rows) > 0
    return pandas.DataFrame(
        {"group": group, "signal": signal, "y": label.astype(int)}
    )


def test_weights_make_group_and_label_independent():
    frame = biased()
    weights = reweighing.reweighing_weights(frame.group, frame.y)
    weighted = frame.assign(w=weights)

    rates = weighted.groupby("group").apply(
        lambda part: numpy.average(part.y, weights=part.w), include_groups=False
    )

    assert rates.max() - rates.min() < 1e-9
    assert weights.mean() == pytest.approx(1.0)
    raw = frame.groupby("group").y.mean()
    assert raw.max() - raw.min() > 0.2


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError, match="same non-zero length"):
        reweighing.reweighing_weights(
            pandas.Series(["a"]), pandas.Series([1, 0])
        )


def test_a_project_trained_on_the_weights_selects_more_evenly():
    frame = biased()
    frame["w"] = reweighing.reweighing_weights(frame.group, frame.y)

    def gap(weight_column):
        result = scmle.MLProject(
            frame,
            "y",
            metric="roc_auc",
            time_budget=60,
            weight_column=weight_column,
            sensitive_columns=["group"],
        ).run()
        scored = result.predict(frame.drop(columns="y"))
        rates = pandas.Series(scored).groupby(frame.group.to_numpy()).mean()
        return float(rates.max() - rates.min())

    assert gap("w") < gap(None)


def test_the_fairness_option_reweights_development_rows_automatically():
    frame = biased()

    def run(mitigation):
        result = scmle.MLProject(
            frame,
            "y",
            metric="roc_auc",
            time_budget=60,
            sensitive_columns=["group"],
            fairness=tasks.FairnessSpec(mitigation=mitigation),
        ).run()
        scored = result.predict(frame.drop(columns="y"))
        rates = pandas.Series(scored).groupby(frame.group.to_numpy()).mean()
        return result, float(rates.max() - rates.min())

    reweighed, narrow = run("reweighing")
    unchanged, wide = run("none")

    assert narrow < wide
    log = json.loads(
        (reweighed.run_dir / "diagnostics/decision_log.json").read_text()
    )
    notes = [
        o["reason"]
        for o in log["outcomes"]
        if o["technique"] == "fairness:group"
    ]
    assert notes and "reweighted" in notes[0]
    assert unchanged.bundle.predict(frame.drop(columns="y")).shape[0] == 1600


def test_reweighing_cannot_be_combined_with_a_weight_column():
    with pytest.raises(ValueError, match="weight_column"):
        scmle.MLProject(
            biased().assign(w=1.0),
            "y",
            weight_column="w",
            sensitive_columns=["group"],
            fairness=tasks.FairnessSpec(mitigation="reweighing"),
        )
