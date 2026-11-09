"""Tests for the command-line interface."""

import hashlib
import io
import json
import logging

import pandas

from scmle import cli
from scmle.data.catalog import DatasetCatalog


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
    predictions = pandas.read_csv(output)
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
        "catalog",
        "predict",
        "batch",
        "serve",
        "export-onnx",
        "package",
        "ingest",
        "charter",
        "claims",
        "frame",
        "loop",
        "operate",
        "deploy",
        "trail",
    ]
    assert all(isinstance(c, cli.Command) for c in cli.COMMANDS)


def test_catalog_command_catalogs_tabular_data_and_writes_verified_artifact(
    tmp_path, capsys
):
    """The command emits the public catalog contract and persists it."""
    data = tmp_path / "events.csv"
    pandas.DataFrame(
        {"value": [1, 2, 2], "label": ["keep", "drop", "drop"]}
    ).to_csv(data, index=False)
    output = tmp_path / "catalog.json"

    code = cli.main(
        [
            "catalog",
            str(data),
            "--label-column",
            "label",
            "--source-uri",
            "s3://bucket/events.csv?version=7",
            "--output",
            str(output),
        ]
    )

    assert code == cli.EXIT_OK
    emitted = json.loads(capsys.readouterr().out)
    restored = DatasetCatalog.read_verified(output)
    assert emitted == restored.to_dict()
    assert restored.shape == (3, 2)
    assert restored.source.uri == "s3://bucket/events.csv?version=7"
    assert restored.label.field == "label"
    assert restored.label.status == "observed"


def test_catalog_command_does_not_infer_a_label_column(tmp_path, capsys):
    """A table without an explicit label stays unsupervised evidence."""
    data = tmp_path / "events.csv"
    pandas.DataFrame({"value": [1, 2], "label": ["keep", "drop"]}).to_csv(
        data, index=False
    )

    assert cli.main(["catalog", str(data)]) == cli.EXIT_OK

    emitted = json.loads(capsys.readouterr().out)
    assert emitted["label"]["status"] == "not_declared"
    assert emitted["label"]["field"] is None


def test_catalog_command_catalogs_raw_bytes_without_decoding(tmp_path, capsys):
    """An explicit raw-byte input remains opaque and receives direct identity."""
    payload = b"not a tabular format"
    raw_path = tmp_path / "payload.unknown"
    raw_path.write_bytes(payload)

    assert cli.main(["catalog", "--raw-bytes", str(raw_path)]) == cli.EXIT_OK

    emitted = json.loads(capsys.readouterr().out)
    assert emitted["data_kind"] == "bytes"
    assert emitted["content_identity"] == (
        f"sha256:{hashlib.sha256(payload).hexdigest()}"
    )
    assert (
        "modality_specific_decoding_was_not_attempted" in emitted["limitations"]
    )


def test_catalog_command_reads_raw_bytes_from_standard_input(
    monkeypatch, capsys
):
    """The '-' input gives a pipe-friendly raw-byte catalog path."""
    payload = b"streamed bytes"
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(payload)))

    assert cli.main(["catalog", "-"]) == cli.EXIT_OK

    emitted = json.loads(capsys.readouterr().out)
    assert emitted["source"]["uri"] == "memory://stdin"
    assert emitted["sample_count"] == 1


def test_catalog_command_rejects_unknown_tabular_suffix(tmp_path, caplog):
    """Unknown formats fail through the existing data-error exit convention."""
    path = tmp_path / "events.unknown"
    path.write_bytes(b"opaque")

    assert cli.main(["catalog", str(path)]) == cli.EXIT_FAILED
    assert "data error" in caplog.text


def test_predict_with_explicit_renames(binary_frame, tmp_path, monkeypatch):
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
    writes = []
    real_write = cli.durability.write_atomically

    def record_write(path, content):
        writes.append((path, content))
        return real_write(path, content)

    monkeypatch.setattr(cli.durability, "write_atomically", record_write)
    code = cli.main([*base, "--rename", "A_value=a", "--case-insensitive"])
    assert code == 0 and len(pandas.read_csv(output)) == len(binary_frame)
    assert writes and writes[0][0] == output


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
