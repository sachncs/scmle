"""Rows of `C` left in this process's memory after sealing.

A canary column holds a distinct value in every row. After custody is applied
and the caller's own copies are dropped, no DataFrame, Series, Index or
container that the garbage collector tracks may hold a canary of a sealed
row. The caller's copy is the documented limit (docs/SECURITY.md); the test
drops it so it measures what the orchestrator itself retains.

Limit of the check: it sees objects the collector tracks and the arrays
their blocks point to. Raw buffers copied into other C extensions, the
operating-system page cache and swap are outside what Python can inspect.
"""

import gc

import numpy as np
import pandas as pd

from scmle.contracts import tasks
from scmle.data.custody import pipeline
from scmle.data.custody import sealed
from scmle.data.splitting import splitters
from tests.unit.test_charter_contracts import make_charter

CANARY = "canary"
BASE = 7_000_000.0
STEP = 1.0009765625


def canary_values(rows):
    return BASE + np.arange(rows) * STEP


def custody(rows=2000, process=True):
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=rows),
            CANARY: canary_values(rows),
            "y": (rng.random(rows) < 0.3).astype(int),
        }
    )
    spec = tasks.ProjectSpec(
        data=tasks.DataSpec(uri="m", target="y"),
        validation=tasks.ValidationSpec(strategy="stratified", seed=3),
    )
    task = tasks.TaskType.BINARY
    split = splitters.SplitManager(spec, task).split(frame)
    log = sealed.OpenLog()
    vault = sealed.CustodyVault(log, process=process)
    outcome = pipeline.apply_custody(
        charter=make_charter(min_effect=0.3, k_max=5),
        split=split,
        frame=frame,
        spec=spec,
        task=task,
        data_hash="8" * 64,
        vault=vault,
    )
    rows_c = outcome.manifest.confirmation_ids
    rows_f = outcome.manifest.final_test_ids
    return (
        outcome.view,
        vault,
        set(canary_values(rows)[list(rows_c)].tolist()),
        set(canary_values(rows)[list(rows_f)].tolist()),
    )


def holders(values):
    """Tracked objects that hold any of `values`, by type name."""
    found = []
    for item in gc.get_objects():
        try:
            if isinstance(item, pd.DataFrame) and CANARY in item.columns:
                column = item[CANARY].to_numpy()
            elif isinstance(item, pd.Series):
                column = item.to_numpy()
            elif isinstance(item, np.ndarray):
                column = item
            else:
                continue
            if column.dtype.kind == "f" and values.intersection(
                column.tolist()
            ):
                found.append(type(item).__name__)
        except Exception:
            continue
    return found


def test_the_scan_finds_a_canary_that_is_still_held():
    frame = pd.DataFrame({CANARY: canary_values(50)})
    assert "DataFrame" in holders({float(frame[CANARY].iloc[3])})
    del frame
    gc.collect()


def test_no_tracked_object_holds_a_canary_of_c_after_sealing():
    view, vault, in_c, in_f = custody()
    gc.collect()
    try:
        assert in_c and in_f
        assert holders(in_c) == []
        assert holders(in_f) == []
        assert in_c.isdisjoint(set(view.features[CANARY].tolist()))
    finally:
        vault.close()


def test_the_resident_protocol_does_hold_the_rows_so_the_scan_is_not_vacuous():
    _, vault, in_c, _ = custody(process=False)
    gc.collect()
    try:
        assert holders(in_c) != []
    finally:
        vault.close()
