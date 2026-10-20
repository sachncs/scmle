"""Benchmark resource and failure evidence tests."""

from pathlib import Path

import pandas as pd
import pytest

from benchmarks import datasets
from benchmarks import run_benchmarks


class BrokenDataset(datasets.BenchmarkDataset):
    """Dataset fixture whose source fails before any seed can complete."""

    def fetch(self) -> pd.DataFrame:
        """Raise a deterministic source failure."""
        raise OSError("fixture source unavailable")


def test_failed_benchmark_records_resource_identity_and_failure_rate(
    tmp_path: Path,
):
    """A failed benchmark still emits reviewable resource evidence."""
    record = run_benchmarks.run_one(
        BrokenDataset("broken", "y", "classification", "roc_auc"),
        budget=1,
        workers=1,
        root=tmp_path,
        seeds=3,
    )

    assert record.status == "error"
    assert record.completed_seeds == 0
    assert record.failure_rate == 1.0
    assert record.peak_memory_gb > 0
    assert record.machine
    assert record.software


def test_benchmark_rejects_empty_seed_repetition(tmp_path: Path):
    """A benchmark cannot claim a zero-repetition evidence set."""
    with pytest.raises(ValueError, match="seeds"):
        run_benchmarks.run_one(
            BrokenDataset("broken", "y", "classification", "roc_auc"),
            budget=1,
            workers=1,
            root=tmp_path,
            seeds=0,
        )
