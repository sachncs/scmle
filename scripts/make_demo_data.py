#!/usr/bin/env python3
"""Write the explicit datasets of the local demo (fixed seeds, no downloads).

    train.csv              the concept the good model learns
    bad_release_train.csv  the same rows with the label relationship reversed;
                           a model trained on it is the demo's bad release
    fresh.csv              new labelled rows from the original concept
    drifted.csv            new labelled rows with the reversed concept

The files are committed so the demo reads them rather than generating them;
this script exists so anyone can see exactly how they were made and
`--check` can prove the committed files still match it.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

DEFAULT_DIRECTORY = Path("deploy/local/data")
PLANS = ["basic", "pro", "team"]


def make_frame(rows: int, seed: int, reverse: bool = False) -> pd.DataFrame:
    """Customers with a churn label that depends on their features.

    Args:
        rows: Number of customers.
        seed: Seed of the random generator.
        reverse: Whether to reverse the label relationship.

    Returns:
        A frame with age, balance, tenure in months, plan and the 0/1
        `churned` label.
    """
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "age": rng.integers(18, 80, rows).astype(float),
            "balance": rng.gamma(2.0, 800.0, rows).round(2),
            "tenure_months": rng.integers(1, 120, rows).astype(float),
            "plan": rng.choice(PLANS, rows),
        }
    )
    signal = (
        0.04 * (frame["age"] - 45)
        - 0.0006 * (frame["balance"] - 1600)
        - 0.02 * (frame["tenure_months"] - 60)
        + (frame["plan"] == "basic") * 0.8
    )
    signal = -signal if reverse else signal
    noise = rng.normal(scale=0.5, size=rows)
    frame["churned"] = (signal + noise > 0).astype(int)
    return frame


def datasets() -> dict[str, pd.DataFrame]:
    """The four demo tables by file name.

    Returns:
        Mapping of file name to table.
    """
    return {
        "train.csv": make_frame(800, 101),
        "bad_release_train.csv": make_frame(800, 101, reverse=True),
        "fresh.csv": make_frame(400, 202),
        "drifted.csv": make_frame(400, 303, reverse=True),
    }


def render(frame: pd.DataFrame) -> str:
    """The exact CSV text of a table.

    Args:
        frame: The table.

    Returns:
        CSV text without the index and with Unix line endings.
    """
    return frame.to_csv(index=False, lineterminator="\n")


def main(arguments: list[str] | None = None) -> int:
    """Write the files, or with --check compare them.

    Args:
        arguments: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        0 after writing or when the committed files match; 1 when `--check`
        finds a file missing or different.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--check", action="store_true")
    options = parser.parse_args(arguments)
    stale = []
    for name, frame in datasets().items():
        path = options.directory / name
        text = render(frame)
        if options.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(name)
        else:
            path.write_text(text, encoding="utf-8", newline="\n")
            print(f"wrote {path}")
    if stale:
        print(f"stale: {', '.join(stale)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
