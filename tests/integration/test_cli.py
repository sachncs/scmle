"""Tests for the command-line interface."""

import logging

import pandas as pd

from mml import cli


def test_plan_run_predict_roundtrip(binary_frame, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    data = tmp_path / "d.csv"
    binary_frame.to_csv(data, index=False)
    common = [
        str(data),
        "--target",
        "y",
        "--output-dir",
        str(tmp_path / "runs"),
    ]
    assert cli.main(["plan", *common, "--time-budget", "120"]) == 0
    assert "Candidates:" in caplog.text
    caplog.clear()
    assert not (tmp_path / "runs").exists()

    assert cli.main(["run", *common, "--time-budget", "120"]) == 0
    assert "selected" in caplog.text and "artifacts:" in caplog.text
    caplog.clear()
    bundle_dir = (
        next(d for d in (tmp_path / "runs").iterdir() if d.is_dir()) / "model"
    )

    features = binary_frame.drop(columns="y")
    features.to_csv(tmp_path / "new.csv", index=False)
    output = tmp_path / "pred.csv"
    code = cli.main(
        [
            "predict",
            "--trusted",
            str(bundle_dir),
            str(tmp_path / "new.csv"),
            "--output",
            str(output),
        ]
    )
    refused = cli.main(
        [
            "predict",
            str(bundle_dir),
            str(tmp_path / "new.csv"),
            "--output",
            str(output),
        ]
    )
    assert refused == cli.EXIT_FAILED
    assert "unauthenticated" in caplog.text
    predictions = pd.read_csv(output)
    assert code == 0 and len(predictions) == len(binary_frame)
    assert {"prediction", "proba_0", "proba_1"} <= set(predictions.columns)


def test_blocked_run_returns_exit_code_two(binary_frame, tmp_path, caplog):
    broken = binary_frame.assign(y=range(len(binary_frame)))
    broken["y"] = broken["y"] % 5
    path = tmp_path / "d.csv"
    broken.to_csv(path, index=False)
    code = cli.main(
        ["run", str(path), "--target", "y", "--output-dir", str(tmp_path / "r")]
    )
    assert code == cli.EXIT_BLOCKED
    assert "blocked:" in caplog.text


def test_missing_file_returns_failure_code(tmp_path, caplog):
    code = cli.main(["plan", str(tmp_path / "nope.csv"), "--target", "y"])
    assert code == cli.EXIT_FAILED
    assert "data error" in caplog.text


def test_commands_are_polymorphic():
    names = [command.name for command in cli.COMMANDS]
    assert names == [
        "plan",
        "run",
        "resume",
        "predict",
        "serve",
        "export-onnx",
    ]
    assert all(isinstance(c, cli.Command) for c in cli.COMMANDS)


def test_predict_with_explicit_renames(binary_frame, tmp_path):
    data = tmp_path / "d.csv"
    binary_frame.to_csv(data, index=False)
    code = cli.main(
        [
            "run",
            str(data),
            "--target",
            "y",
            "--output-dir",
            str(tmp_path / "runs"),
            "--top-models",
            "0",
        ]
    )
    assert code == 0
    bundle_dir = (
        next(d for d in (tmp_path / "runs").iterdir() if d.is_dir()) / "model"
    )
    renamed = binary_frame.drop(columns="y").rename(
        columns={"a": "A_value", "b": "B"}
    )
    renamed.to_csv(tmp_path / "new.csv", index=False)
    output = tmp_path / "pred.csv"
    base = [
        "predict",
        "--trusted",
        str(bundle_dir),
        str(tmp_path / "new.csv"),
        "--output",
        str(output),
    ]
    assert cli.main(base) == cli.EXIT_FAILED  # exact names required by default
    code = cli.main([*base, "--rename", "A_value=a", "--case-insensitive"])
    assert code == 0 and len(pd.read_csv(output)) == len(binary_frame)


def test_config_file_and_flags_reach_the_full_specification(
    binary_frame, tmp_path
):
    import json

    data = tmp_path / "d.csv"
    binary_frame.to_csv(data, index=False)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "selection": {"tie_breaker": "lower_variance"},
                "ensemble": {"size": 0},
                "strategies": {"top_models": 0},
                "search": {"top_models": 0},
            }
        )
    )
    args = [
        "run",
        str(data),
        "--target",
        "y",
        "--output-dir",
        str(tmp_path / "runs"),
        "--time-budget",
        "300",
        "--config",
        str(config),
        "--folds",
        "3",
        "--enforcement",
        "cooperative",
        "--leakage-policy",
        "block",
    ]
    assert cli.main(args) == 0
    saved = json.loads(
        next((tmp_path / "runs").glob("run_*"))
        .joinpath("config.json")
        .read_text()
    )
    assert saved["selection"]["tie_breaker"] == "lower_variance"
    assert saved["validation"]["folds"] == 3
    assert saved["resources"]["enforcement"] == "cooperative"
    assert saved["leakage_policy"] == "block"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"mystery": {}}))
    assert cli.main([*args[:-8], "--config", str(bad)]) != 0
