"""Access attempts: code that receives only the exploration view.

The build fails if anything reachable from the `ExplorationView` (attributes,
containers, DataFrame values, pickles, the open log) gives access to the rows
of `C`, or to the object that can release them.
"""

import json
import pickle
import typing

import numpy as np
import pandas as pd

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.custody import pipeline
from scmle.data.custody import sealed
from scmle.data.splitting import splitters
from scmle.governance import vault as vault_module
from tests.unit.test_charter_contracts import make_charter

BINARY = tasks.TaskType.BINARY
SENTINEL = "sentinel"


class Built(typing.NamedTuple):
    frame: pd.DataFrame
    outcome: pipeline.CustodyOutcome
    vault: sealed.CustodyVault
    log: sealed.OpenLog


def build():
    rng = np.random.default_rng(11)
    rows = 2000
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=rows),
            SENTINEL: np.arange(rows, dtype=float) + 0.123456,
            "y": (rng.random(rows) < 0.3).astype(int),
        }
    )
    spec = tasks.ProjectSpec(
        data=tasks.DataSpec(uri="m", target="y"),
        validation=tasks.ValidationSpec(strategy="stratified", seed=3),
    )
    split = splitters.SplitManager(spec, BINARY).split(frame)
    log = sealed.OpenLog()
    vault = sealed.CustodyVault(log, process=True)
    outcome = pipeline.apply_custody(
        charter=make_charter(min_effect=0.3, k_max=5),
        split=split,
        frame=frame,
        spec=spec,
        task=BINARY,
        data_hash="8" * 64,
        vault=vault,
    )
    return Built(frame, outcome, vault, log)


def reachable(root, limit=200_000):
    """Every object reachable from `root` through attributes and containers."""
    seen: dict[int, object] = {}
    stack = [root]
    while stack and len(seen) < limit:
        item = stack.pop()
        if id(item) in seen:
            continue
        seen[id(item)] = item
        if isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, list | tuple | set | frozenset):
            stack.extend(item)
        elif isinstance(item, pd.DataFrame | pd.Series):
            continue
        else:
            stack.extend(getattr(item, "__dict__", {}).values())
            for cls in type(item).__mro__:
                for name in getattr(cls, "__slots__", ()):
                    if hasattr(item, name):
                        stack.append(getattr(item, name))
    return list(seen.values())


def test_the_view_reaches_no_sealed_object_or_token():
    built = build()
    outcome, vault = built.outcome, built.vault
    tokens = {s.issue_token() for s in vault.sets.values()}
    objects = reachable(outcome.view)
    assert not any(
        isinstance(
            o,
            sealed.SealedSet
            | sealed.CustodyVault
            | sealed.OpenLog
            | vault_module.SealedTestSet
            | splitters.LockedTestSet,
        )
        for o in objects
    )
    assert not any(isinstance(o, str) and o in tokens for o in objects)
    vault.close()


def test_the_view_holds_no_row_of_the_confirmation_set():
    built = build()
    frame, outcome, vault = built.frame, built.outcome, built.vault
    held_rows = set(frame[SENTINEL].iloc[list(vault.sets["C"].ids)].tolist())
    view_values = set(outcome.view.features[SENTINEL].tolist())
    assert held_rows and view_values.isdisjoint(held_rows)
    assert set(outcome.view.ids).isdisjoint(vault.sets["C"].ids)
    vault.close()


def test_pickling_the_view_leaks_neither_rows_nor_tokens():
    built = build()
    frame, outcome, vault = built.frame, built.outcome, built.vault
    blob = pickle.dumps(outcome.view)
    for token in (s.issue_token() for s in vault.sets.values()):
        assert token.encode() not in blob
    restored = pickle.loads(blob)  # nosec B301 - our own bytes
    held = set(frame[SENTINEL].iloc[list(vault.sets["C"].ids)].tolist())
    assert set(restored.features[SENTINEL].tolist()).isdisjoint(held)
    vault.close()


def test_the_open_log_and_manifest_hold_ids_but_never_values():
    frame, outcome, vault, log = build()
    held = vault.sets["C"]
    held.open(held.issue_token(), actor="confirmation")
    text = json.dumps(log.entries) + json.dumps(outcome.manifest.to_dict())
    for value in frame[SENTINEL].iloc[list(held.ids)].head(50):
        assert repr(float(value)) not in text
    vault.close()


def test_the_orchestrator_process_keeps_only_the_ids():
    built = build()
    outcome, vault = built.outcome, built.vault
    held = vault.sets["C"]
    assert isinstance(held.locked, vault_module.SealedTestSet)
    assert not hasattr(held.locked, "features")
    assert not hasattr(held.locked, "target")
    assert held.ids == outcome.manifest.confirmation_ids
    vault.close()


def test_a_second_open_is_refused_and_both_attempts_are_logged():
    built = build()
    vault, log = built.vault, built.log
    held = vault.sets["C"]
    token = held.issue_token()
    held.open(token, actor="confirmation")
    try:
        held.open(token, actor="exploration")
    except errors.LeakageError:
        pass
    else:
        raise AssertionError("a sealed set opened twice")
    events = [(e["event"], e["actor"]) for e in log.entries]
    assert ("open", "confirmation") in events
    assert ("denied", "exploration") in events
    log.verify()
    vault.close()
