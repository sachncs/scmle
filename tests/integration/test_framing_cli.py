"""Tests for `scmle frame`."""

import json

import numpy as np
import pandas as pd

from scmle import cli
from scmle.data.charter import sealing

MODEL = {
    "c_00": 0.0,
    "c_01": 5.0,
    "c_10": 1.0,
    "c_11": 0.0,
    "t_lo": 0.05,
    "t_hi": 0.5,
    "sources": ["human"] * 4,
    "range_source": "human",
}


def inputs(tmp_path, columns=None):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "customer_id": np.arange(300),
            "as_of": pd.date_range("2024-01-01", periods=300, freq="h"),
            "churn": (rng.random(300) < 0.3).astype(int),
        }
    )
    frame.to_csv(tmp_path / "x.csv", index=False)
    (tmp_path / "request.txt").write_text("Predict churn for each customer.")
    (tmp_path / "source.json").write_text(
        json.dumps(
            {
                "uri": "warehouse://crm/accounts",
                "columns": columns or list(frame.columns),
                "primary_key": ["customer_id"],
            }
        )
    )
    (tmp_path / "cost.json").write_text(json.dumps(MODEL))


def run(tmp_path, *extra):
    return cli.main(
        [
            "frame",
            str(tmp_path / "request.txt"),
            "--source",
            str(tmp_path / "source.json"),
            "--data",
            str(tmp_path / "x.csv"),
            "--parse-dates",
            "as_of",
            "--output",
            str(tmp_path / "out"),
            *extra,
        ]
    )


def test_frame_writes_artifacts_and_an_unsigned_charter(tmp_path):
    inputs(tmp_path)
    assert run(tmp_path, "--cost-model", str(tmp_path / "cost.json")) == 0
    out = tmp_path / "out"
    assert {p.name for p in out.iterdir()} >= {
        "framing_draft.json",
        "framing_questions.json",
        "decision_cost_model.json",
        "charter.json",
    }
    loaded = sealing.read_verified(out / "charter.json")
    assert not loaded.signed
    assert loaded.charter.cost_model is not None
    stored = json.loads((out / "framing_draft.json").read_text())
    assert stored["charter_hash"] == loaded.charter_hash
    assert stored["status"] == "issued_with_questions"


def test_frame_refuses_a_request_naming_a_missing_column(tmp_path):
    inputs(tmp_path)
    (tmp_path / "request.txt").write_text("Predict `churn_flag`.")
    assert run(tmp_path) == 2
    refusal = json.loads(
        (tmp_path / "out" / "framing_refusal.json").read_text()
    )
    assert refusal["missing_columns"] == ["churn_flag"]
    assert not (tmp_path / "out" / "charter.json").exists()


def test_frame_replay_without_tau_is_refused(tmp_path):
    inputs(tmp_path)
    sample = {
        "target": "churn",
        "prediction_moment": "as_of",
        "unit": ["customer_id"],
    }
    (tmp_path / "replay.json").write_text(
        json.dumps({"model_id": "m", "samples": [sample] * 10})
    )
    assert run(tmp_path, "--replay", str(tmp_path / "replay.json")) == 2
