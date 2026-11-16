"""Profiles every pipeline stage on this machine (plan stage A).

Wraps the `execute` method of each pipeline stage to record its wall time
and the growth of the process's peak resident memory, runs seeded synthetic
tables of several sizes through the fast configuration the tests use (no
search, strategies or ensembles), and writes the results beside the other
baselines. The C13 targets are re-aimed at where the time actually goes.

    python -m benchmarks.profile --rows 2000 20000 --output plan/baseline
"""

import argparse
import datetime
import json
import pathlib
import platform
import resource
import sys
import tempfile
import time
from typing import Any

import numpy
import pandas

import scmle.api.project
from scmle.contracts import tasks
import scmle.core.stages

STAGES = (
    scmle.core.stages.IngestStage,
    scmle.core.stages.ProfileStage,
    scmle.core.stages.PlanStage,
    scmle.core.stages.TrainStage,
    scmle.core.stages.SelectStage,
    scmle.core.stages.FinalEvaluationStage,
    scmle.core.stages.PackageStage,
)
# macOS reports peak resident memory in bytes, Linux in kilobytes.
SCALE = 1 if sys.platform == "darwin" else 1024
# Per-stage times of the current run, filled by the wrappers `instrument`
# installs once and cleared by `measure` before each run.
TIMES: dict[str, dict[str, float]] = {}


def peak() -> int:
    """Peak resident memory of this process so far, in bytes."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * SCALE


def table(rows: int, seed: int = 7) -> pandas.DataFrame:
    """A seeded table with ten numeric, two categorical and one text column."""
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame(
        {f"x{i}": rng.normal(size=rows) for i in range(10)}
    )
    frame["c0"] = rng.choice(list("abcdef"), rows)
    frame["c1"] = rng.choice(list("pqrs"), rows)
    signal = frame["x0"] + 0.5 * frame["x1"] - 0.3 * frame["x2"]
    frame["y"] = (signal + rng.normal(scale=0.7, size=rows) > 0).astype(int)
    return frame


def instrument(found: dict[str, dict[str, float]]) -> None:
    """Wraps each stage so a run records its time and memory into `found`."""
    for stage in STAGES:
        original = stage.execute

        def timed(
            self: Any,
            context: Any,
            original: Any = original,
            name: str = stage.__name__,
        ) -> None:
            before, started = peak(), time.perf_counter()
            try:
                original(self, context)
            finally:
                found[name] = {
                    "seconds": round(time.perf_counter() - started, 3),
                    "peak_growth_mb": round((peak() - before) / 2**20, 1),
                }

        setattr(stage, "execute", timed)  # noqa: B010 -- instrumenting a class method


def measure(rows: int) -> dict[str, object]:
    """Profiles one run of `rows` rows.

    The stages must be instrumented once, before the first run, with
    `instrument(TIMES)`; wrapping them again would nest the wrappers.
    """
    TIMES.clear()
    frame = table(rows)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as folder:
        scmle.api.project.Project(
            frame,
            "y",
            output_dir=folder,
            search=tasks.Search(top_models=0),
            strategies=tasks.Strategies(top_models=0),
            ensemble=tasks.Stacking(size=0),
            models=tasks.Candidates(
                exclude_families=["bagging", "deep_learning", "foundation"]
            ),
            enforcement="cooperative",
        ).run()
    return {
        "rows": rows,
        "data_mb": round(frame.memory_usage(deep=True).sum() / 2**20, 2),
        "total_seconds": round(time.perf_counter() - started, 3),
        "peak_mb": round(peak() / 2**20, 1),
        "stages": dict(TIMES),
    }


def main(arguments: list[str] | None = None) -> int:
    """Profiles each size and writes `stages_<date>.json`.

    Args:
        arguments: Command-line arguments; `sys.argv` when None.

    Returns:
        0.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[2000])
    parser.add_argument("--output", type=pathlib.Path, default=None)
    options = parser.parse_args(arguments)
    instrument(TIMES)
    record = {
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
        },
        "runs": [measure(rows) for rows in options.rows],
    }
    text = json.dumps(record, indent=2)
    if options.output is None:
        print(text)
    else:
        day = datetime.datetime.now(tz=datetime.UTC).date().isoformat()
        path = options.output / f"stages_{day}.json"
        path.write_text(text + "\n", encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
