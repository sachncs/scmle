"""A file larger than the memory budget runs the whole lifecycle."""

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import support

ROWS = 120_000


@pytest.fixture(scope="module")
def big_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("big") / "big.parquet"
    writer = None
    for part in range(4):
        rng = np.random.default_rng(part)
        size = ROWS // 4
        x = rng.normal(size=(size, 5))
        y = (x[:, 0] + 0.7 * x[:, 1] + rng.normal(0, 0.5, size) > 0).astype(int)
        table = pa.table({**{f"x{i}": x[:, i] for i in range(5)}, "y": y})
        writer = writer or pq.ParquetWriter(path, table.schema)
        writer.write_table(table)
    writer.close()
    return path


def run(tmp_path, path, **kwargs):
    return support.make_project(
        str(path),
        "y",
        time_budget=300,
        output_dir=tmp_path,
        memory_budget_gb=0.002,
        **kwargs,
    ).run()


@pytest.fixture(scope="module")
def sampled(tmp_path_factory, big_file):
    return run(tmp_path_factory.mktemp("sampled"), big_file)


def test_every_locked_row_is_scored_by_streaming_the_file(sampled):
    streamed = sampled.evaluation["streamed_locked"]

    assert streamed["of_total"] == ROWS
    assert streamed["rows"] == pytest.approx(0.2 * ROWS, rel=0.05)
    assert streamed["metrics"]["roc_auc"] > 0.9
    in_memory = sampled.evaluation["n_dev"] + sampled.evaluation["n_test"]
    assert in_memory < ROWS / 3


def test_the_profile_says_the_file_was_streamed(sampled):
    assert any(
        "streamed" in finding for finding in sampled.profile.quality_findings
    )
    assert sampled.evaluation["split_strategy"] == "preset"


def test_the_split_side_column_is_never_a_feature(sampled):
    assert "split_side" not in sampled.bundle.features


def test_stream_training_keeps_only_models_that_learn_in_batches(
    tmp_path, big_file
):
    result = run(tmp_path, big_file, training="stream")

    names = {t.model_name for t in result.trials}
    assert names <= {"dummy", "sgd_logistic"}
    log = json.loads(
        (result.run_dir / "diagnostics/decision_log.json").read_text()
    )
    reasons = [d["reason"] for d in log["plan"] if d["action"] == "reject"]
    assert any("cannot learn incrementally" in r for r in reasons)
    assert result.evaluation["streamed_locked"]["metrics"]["roc_auc"] > 0.85
    events = [
        json.loads(line)
        for line in (result.run_dir / "events.jsonl").read_text().splitlines()
    ]
    streamed = [e for e in events if e["kind"] == "training.streamed"]
    assert streamed and streamed[0]["data"]["rows"] > 0.7 * ROWS


def test_out_of_core_refuses_options_that_need_the_whole_table(
    tmp_path, big_file
):
    with pytest.raises(ValueError, match="random or stratified"):
        support.make_project(
            str(big_file),
            "y",
            memory_budget_gb=0.002,
            time_column="x0",
            output_dir=tmp_path,
        )
    with pytest.raises(ValueError, match="memory_budget_gb"):
        support.make_project(
            str(big_file), "y", training="stream", output_dir=tmp_path
        )


def test_a_small_file_is_read_normally_even_with_a_budget(tmp_path):
    frame = pd.DataFrame(
        {"a": np.random.default_rng(1).normal(size=600)}
    ).assign(y=lambda f: (f.a > 0).astype(int))
    path = tmp_path / "small.parquet"
    frame.to_parquet(path)

    result = support.make_project(
        str(path),
        "y",
        time_budget=300,
        output_dir=tmp_path / "runs",
        memory_budget_gb=5.0,
    ).run()

    assert result.evaluation["streamed_locked"] is None
    assert result.evaluation["split_strategy"] != "preset"


def test_streaming_models_are_not_candidates_for_ordinary_runs(tmp_path):
    frame = pd.DataFrame(
        {"a": np.random.default_rng(3).normal(size=400)}
    ).assign(y=lambda f: (f.a > 0).astype(int))

    plan = support.make_project(
        frame, "y", time_budget=300, output_dir=tmp_path
    ).plan()

    assert "sgd_logistic" not in plan.candidate_models
    reasons = [
        d.reason for d in plan.decisions if d.technique == "sgd_logistic"
    ]
    assert reasons and "data larger than memory" in reasons[0]
