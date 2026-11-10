"""Peak memory of a plain run grows by a bounded amount per input row.

A run holds several copies of the table and builds float matrices from it, so
the peak grows with the rows. The slope is measured between two sizes in fresh
processes (the interpreter and library imports are the same in both and
cancel), with `resource.getrusage` reporting each process's peak resident set.
The bound is stated as a multiple of the table's own in-memory size so that it
does not depend on the machine.

Measured on macOS (arm64, Python 3.14t, pandas 3.0, scikit-learn 1.9) with a
30-column table of 242 bytes per row, the slope is about 4.2 KB per row (this
test's 4,000 and 20,000 rows), which is 17 times the table. About half of that
is the system allocator keeping freed blocks; the same run with
`MallocLargeCache=0` peaks at 7 times the table. The ceiling leaves room for allocator and platform differences, so it
catches a regression that adds several more copies of the table, not one.
"""

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).parents[2]
SMALL_ROWS = 4000
LARGE_ROWS = 20000
CEILING_TABLE_MULTIPLE = 22
KILOBYTE = 1024
PROBE = """
import json, resource, sys, tempfile
import numpy, pandas
import scmle.api.project
from scmle.contracts import tasks
from scmle.models import registry

rows = int(sys.argv[1])
rng = numpy.random.default_rng(0)
data = {f"n{i}": rng.normal(size=rows) for i in range(26)}
data["i0"] = rng.integers(0, 100, rows)
data["c0"] = rng.choice(["a", "b", "c", "d"], rows)
data["c1"] = rng.choice(["u", "v", "w"], rows)
frame = pandas.DataFrame(data)
signal = frame.n0 - frame.n1 + 0.5 * frame.n2 * frame.n3 + 0.02 * frame.i0
frame["y"] = (signal + rng.normal(size=rows) > 0.5).astype(int)
kept = {"dummy", "logistic_regression", "decision_tree"}
with tempfile.TemporaryDirectory() as directory:
    scmle.api.project.Project(
        frame,
        "y",
        output_dir=directory,
        folds=3,
        search=tasks.Search(top_models=0),
        strategies=tasks.Strategies(top_models=0, names=[]),
        ensemble=tasks.Stacking(size=0),
        models=tasks.Candidates(
            exclude_models=sorted(registry.builtin_names() - kept)
        ),
        enforcement="cooperative",
        time_budget=600,
        seed=7,
        metric="roc_auc",
    ).run()
print(json.dumps({
    "peak": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    "table": int(frame.memory_usage(deep=True).sum()),
}))
"""


def probe(rows):
    """Runs the probe in a fresh process.

    Returns:
        The process's peak resident set in bytes and the table's size in bytes.
    """
    done = subprocess.run(
        [sys.executable, "-c", PROBE, str(rows)],
        capture_output=True,
        check=True,
        cwd=ROOT,
        text=True,
        timeout=300,
    )
    reported = json.loads(done.stdout.strip().splitlines()[-1])
    unit = 1 if sys.platform == "darwin" else KILOBYTE
    return reported["peak"] * unit, reported["table"]


def test_peak_memory_grows_by_a_bounded_amount_per_row():
    small_peak, _ = probe(SMALL_ROWS)
    large_peak, large_table = probe(LARGE_ROWS)
    growth_per_row = (large_peak - small_peak) / (LARGE_ROWS - SMALL_ROWS)
    table_per_row = large_table / LARGE_ROWS
    assert growth_per_row > 0
    assert growth_per_row <= CEILING_TABLE_MULTIPLE * table_per_row, (
        f"peak memory grows {growth_per_row:.0f} bytes per row, "
        f"{growth_per_row / table_per_row:.1f} times the table's "
        f"{table_per_row:.0f} bytes per row"
    )
