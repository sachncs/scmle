"""Freezes cleanlab's confident-learning output as a differential oracle.

Run offline in an environment that has `cleanlab` (not a project
dependency):

    python -m benchmarks.qualification.make_f5_oracles \
        tests/fixtures/f5_oracles.json

It draws seeded noisy probabilities, asks `cleanlab.filter.find_label_issues`
with `filter_by="confident_learning"` for the issue mask, and stores inputs,
outputs and tool versions. The tests compare `scmle.data.detectors.labels`
with the frozen mask; agreement shows we implemented the reference, not that
the reference is right.
"""

import json
from pathlib import Path
import sys

import numpy as np

CASES = ((0, 300, 3), (1, 500, 2), (2, 400, 4))


def build() -> dict[str, object]:
    """Generates the fixture document (imports cleanlab lazily)."""
    import cleanlab
    from cleanlab.filter import find_label_issues

    cases = []
    for seed, rows, classes in CASES:
        rng = np.random.default_rng(seed)
        truth = rng.integers(0, classes, rows)
        logits = rng.normal(size=(rows, classes)) + 2.5 * np.eye(classes)[truth]
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        flip = rng.random(rows) < 0.1
        given = np.where(flip, (truth + 1) % classes, truth)
        issues = find_label_issues(given, probs, filter_by="confident_learning")
        cases.append(
            {
                "seed": seed,
                "labels": given.tolist(),
                "probabilities": probs.tolist(),
                "issues": issues.tolist(),
            }
        )
    return {
        "tool": "cleanlab",
        "version": cleanlab.__version__,
        "numpy": np.__version__,
        "cases": cases,
    }


def main(arguments: list[str]) -> int:
    """Writes the fixture to the path in `arguments[0]`."""
    Path(arguments[0]).write_text(
        json.dumps(build(), indent=1) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
