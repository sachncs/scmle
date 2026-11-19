"""Tests for `scmle frame`."""

import json

import numpy
import pandas

from scmle import cli
from scmle.assurance.charter import sealing

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
    rng = numpy.random.default_rng(0)
    frame = pandas.DataFrame(
        {
            "customer_id": numpy.arange(300),
            "as_of": pandas.date_range("2024-01-01", periods=300, freq="h"),
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
    # Enforces docs/CLI.md:32.
    # Enforces docs/ARTIFACTS.md:69.
    # Enforces website/src/pages/docs/artifacts.astro:46.
    # Enforces website/src/pages/docs/causal-decision.astro:44.
    # Enforces website/src/pages/docs/cli.astro:17.
    inputs(tmp_path)
    assert run(tmp_path, "--cost-model", str(tmp_path / "cost.json")) == 0
    out = tmp_path / "out"
    assert {p.name for p in out.iterdir()} >= {
        "framing_draft.json",
        "framing_questions.json",
        "decision_cost_model.json",
        "charter.json",
    }
    loaded = sealing.read(out / "charter.json")
    assert not loaded.signed
    assert loaded.charter.costs is not None
    stored = json.loads((out / "framing_draft.json").read_text())
    assert stored["charter_hash"] == loaded.digest
    assert stored["status"] == "issued_with_questions"


def test_frame_refuses_a_request_naming_a_missing_column(tmp_path):
    # Enforces docs/CLI.md:32.
    # Enforces docs/ARTIFACTS.md:69.
    # Enforces docs/WHEN_THINGS_GO_WRONG.md:22.
    # Enforces website/src/learn/when-things-go-wrong.html:11.
    # Enforces website/src/pages/docs/artifacts.astro:46.
    # Enforces website/src/pages/docs/causal-decision.astro:33.
    # Enforces website/src/pages/docs/causal-decision.astro:40.
    # Enforces website/src/pages/docs/cli.astro:17.
    inputs(tmp_path)
    (tmp_path / "request.txt").write_text("Predict `churn_flag`.")
    assert run(tmp_path) == 2
    refusal = json.loads(
        (tmp_path / "out" / "framing_refusal.json").read_text()
    )
    assert refusal["missing_columns"] == ["churn_flag"]
    assert not (tmp_path / "out" / "charter.json").exists()


def test_frame_replay_without_tau_is_refused(tmp_path):
    # Enforces docs/ARTIFACTS.md:82.
    # Enforces docs/WHAT_SCMLE_CANNOT_DO.md:266.
    # Enforces docs/WHEN_THINGS_GO_WRONG.md:356.
    # Enforces website/src/learn/what-scmle-cannot-do.html:70-71.
    # Enforces website/src/learn/when-things-go-wrong.html:162.
    # Enforces website/src/pages/docs/artifacts.astro:119.
    # Enforces website/src/pages/docs/causal-decision.astro:46.
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
