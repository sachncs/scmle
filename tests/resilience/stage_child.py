"""Child process for the stage-boundary kill test.

    python stage_child.py OUTPUT INDEX PHASE

runs the standard tabular workflow into OUTPUT and kills itself with SIGKILL
(no cleanup, no `finally`, like a lost machine) before or after the stage
record with position INDEX is written. With INDEX -1 it runs to the end.
"""

import pathlib
import sys

import failpoints
import numpy
import pandas

from scmle.contracts import tasks
from scmle.pipeline import archive
import scmle.pipeline.orchestrator

ROWS = 300
BUDGET = 600


def frame():
    """A seeded binary table with a learnable target."""
    rng = numpy.random.default_rng(5)
    table = pandas.DataFrame(
        {
            "a": rng.normal(size=ROWS),
            "b": rng.normal(size=ROWS),
            "c": rng.choice(list("xyz"), ROWS),
        }
    )
    signal = table.a + 0.5 * table.b + rng.normal(scale=0.5, size=ROWS)
    table["y"] = (signal > 0).astype(int)
    return table


def spec():
    """A fast configuration: no search, strategies, ensembles or PyTorch."""
    return tasks.Config(
        data=tasks.Dataset(uri="memory", target="y"),
        resources=tasks.Resources(
            time_budget_seconds=BUDGET, enforcement="cooperative"
        ),
        search=tasks.Search(top_models=0),
        strategies=tasks.Strategies(top_models=0),
        ensemble=tasks.Stacking(size=0),
        models=tasks.Candidates(
            exclude_families=["bagging", "deep_learning", "foundation"]
        ),
    )


def arm(index, phase):
    """Kills the process at one stage record boundary."""
    real = archive.Directory.record

    def record(self, position, name, progress, seconds):
        if position == index and phase == "before":
            failpoints.kill_process()
        body = real(self, position, name, progress, seconds)
        if position == index and phase == "after":
            failpoints.kill_process()
        return body

    archive.Directory.record = record


def main():
    output, index, phase = (
        pathlib.Path(sys.argv[1]),
        int(sys.argv[2]),
        sys.argv[3],
    )
    if index >= 0:
        arm(index, phase)
    orchestrator = scmle.pipeline.orchestrator.Orchestrator(
        spec(), frame=frame()
    )
    orchestrator.run(output)


if __name__ == "__main__":
    main()
