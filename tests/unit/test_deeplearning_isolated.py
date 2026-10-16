"""Runs the PyTorch tests in a fresh interpreter.

PyTorch bundles its own OpenMP runtime, and loading it into a process that
also uses XGBoost or LightGBM crashes, so these tests never share a process
with the rest of the suite.
"""

import pathlib
import subprocess
import sys

import pytest

DIRECTORY = pathlib.Path(__file__).resolve().parents[1] / "deeplearning"


def test_pytorch_families_pass_in_their_own_process():
    pytest.importorskip("torch")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            str(DIRECTORY),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout[-2000:]
