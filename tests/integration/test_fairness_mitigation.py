"""Group thresholds close an outcome gap, or are dropped with the evidence."""

import json

import numpy as np
import pandas as pd
import pytest
import support

from scml.contracts import tasks


def biased(rows=3000, seed=0, shift=1.0):
    rng = np.random.default_rng(seed)
    group = rng.choice(["a", "b"], rows)
    skill = rng.normal(size=rows)
    score = skill + np.where(group == "b", -shift, 0.0)
    data = pd.DataFrame(
        {
            "group": group,
            "skill": skill + np.where(group == "b", -shift, 0.0),
            "noise": rng.normal(size=rows),
        }
    )
    data["y"] = (score + rng.normal(0, 0.8, rows) > 0).astype(int)
    return data


def run(tmp_path, data, **kwargs):
    return support.make_project(
        data,
        "y",
        time_budget=300,
        output_dir=tmp_path,
        sensitive_columns=["group"],
        models=tasks.ModelSpec(exclude_families=["deep_learning"]),
        **kwargs,
    ).run()


SPEC = tasks.FairnessSpec(
    mitigation="group_thresholds",
    criterion="demographic_parity",
    trigger_gap=0.05,
    max_accuracy_cost=0.2,
)


@pytest.fixture(scope="module")
def mitigated(tmp_path_factory):
    return run(tmp_path_factory.mktemp("fair"), biased(), fairness=SPEC)


def test_the_gap_on_the_released_rows_shrinks_and_the_cost_is_reported(
    mitigated,
):
    record = mitigated.evaluation["fairness_mitigation"]

    assert record["applied"]
    locked = record["locked_rows"]
    assert locked["gap_after"] < locked["gap_before"]
    assert {"accuracy_before", "accuracy_after", "rows"} <= set(locked)


def test_the_decision_is_logged_with_its_evidence(mitigated):
    log = json.loads(
        (mitigated.run_dir / "diagnostics/decision_log.json").read_text()
    )

    entries = [o for o in log["outcomes"] if o["technique"] == "fairness:group"]

    assert entries and entries[0]["action"] == "include"
    assert "gap" in entries[0]["reason"]


def test_thresholds_apply_only_when_the_group_is_supplied(tmp_path):
    result = run(tmp_path, biased(), fairness=SPEC, excluded_columns=["group"])
    rows = biased(1500, seed=5).drop(columns="y")

    with_group = result.bundle.predict(rows)
    without = result.bundle.predict(rows.drop(columns="group"))

    assert "group" in result.bundle.schema.optional_inputs
    assert "group" not in result.bundle.features
    assert (with_group != without).any()

    def gap(predictions):
        rates = [
            (predictions[(rows.group == g).to_numpy()] == 1).mean()
            for g in ("a", "b")
        ]
        return abs(rates[0] - rates[1])

    assert gap(with_group) < gap(without)


def test_a_small_gap_is_left_alone_and_the_reason_recorded(tmp_path):
    result = run(tmp_path, biased(shift=0.0), fairness=SPEC)

    record = result.evaluation["fairness_mitigation"]

    assert not record["applied"]
    assert "within" in record["reason"]
    assert "locked_rows" not in record


def test_mitigation_needs_a_sensitive_column(tmp_path):
    with pytest.raises(ValueError, match="sensitive_columns"):
        support.make_project(
            biased(200),
            "y",
            output_dir=tmp_path,
            fairness=SPEC,
        )


def test_nothing_is_changed_unless_asked(tmp_path):
    result = run(tmp_path, biased())

    assert result.evaluation["fairness_mitigation"] is None
