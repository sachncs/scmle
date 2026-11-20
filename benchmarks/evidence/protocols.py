"""Claim C7, protocols: the lease and journal protocols, model-checked.

The criterion (plan section 3, C7): the lease and journal protocols are
model-checked, with no counterexample. Each `Model` here is an executable,
explicit-state model of one protocol as the code implements it, and
`explore` checks it exhaustively: a breadth-first search over every
interleaving of its actors, with a crash allowed before any step, inside a
bounded scope. Every invariant that fails gets its shortest counterexample.

* `Lease` models the workflow executor's run lease with fencing tokens
  (`scmle/workflow/executor/leases.py`) over the object store's conditional
  writes (`scmle/storage/objectstore.py`): executors acquire (create, or
  take over an expired lease by compare-and-swap), write a run record
  fenced by their token, renew from a heartbeat thread and relinquish;
  any of them may stall or crash while the clock runs.
* `Custody` and `Erasure` model the two line journals that are appended,
  synced and read back on recovery: the custody journal
  (`scmle/assurance/custody/journal.py`) and the erasure journal
  (`scmle/api/erasure/receipts.py`). A client appends records in order,
  crashes at any step, recovers and resumes as the journal's real caller
  does: the erasure skips every step the journal shows; the orchestrator
  decides every gate again.
  A `kill` crash keeps every byte written (one `write` call is never torn
  by a process kill); a `power` crash keeps the synced bytes and any
  prefix of the rest, so a line can be cut.

Every model step is one store or file operation, and `Link` ties each to
the code it models by a text the code must contain, so a model goes stale
(`links` fail) when the code moves. `Model.replay` drives the real
implementation through a path of the model, one store or file operation
at a time (threads gated per operation for the lease), and compares the
real state with the model's after every step: each counterexample, and
`SAMPLES` paths spread over the search (a deepest one among them), is
replayed. A model whose replay diverges is unfaithful and decides
nothing.

The verdict is "pass" only when every invariant of every model holds,
every link resolves and every replay conforms; "fail" when a faithful
model has a counterexample; "not measured" when a replay diverges or a
link is broken.

    python -m benchmarks.evidence.protocols --show [--model NAME ...]
    python -m benchmarks.evidence.protocols --write
    python -m benchmarks.evidence.protocols --check
"""

import abc
import argparse
import collections
from collections.abc import Callable
from collections.abc import Hashable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
import dataclasses
import functools
import hashlib
import importlib
import inspect
import json
import pathlib
import sys
import tempfile
import threading
from typing import Any

from scmle.api.erasure import receipts
from scmle.assurance.custody import journal
from scmle.contracts import errors
from scmle.storage import objectstore
from scmle.workflow import definition
from scmle.workflow import executor

ROOT = pathlib.Path(__file__).resolve().parents[2]
PATH = pathlib.Path(__file__).with_suffix(".json")
# The code the models stand for; a change to any of it makes the stored
# result stale.
MODELED = (
    "scmle/workflow/executor/leases.py",
    "scmle/workflow/executor/records.py",
    "scmle/storage/objectstore.py",
    "scmle/assurance/custody/journal.py",
    "scmle/api/erasure/receipts.py",
    "scmle/api/erasure/erasures.py",
    "scmle/pipeline/engine.py",
    "scmle/pipeline/orchestrator.py",
    "scmle/durability.py",
)
# A resource bound, not a pass threshold: a scope whose state space grows
# past it is refused (`BoundError`) instead of being sampled, because an
# exhaustive check that stopped early would claim what it never saw.
STATES_MAX = 3_000_000
# Paths replayed against the real code besides the counterexamples. If the
# model and the code disagreed on a share p of the reachable paths, all 20
# replays would agree with probability (1 - p) ** 20, under 0.05 once p
# reaches 0.14 (the rule of three, 3 / 20); a smaller disagreement can go
# unseen.
SAMPLES = 20
# Seconds a replay waits for the real code to reach its next store call.
WAIT_SECONDS = 10.0
# Each actor's program and the clock are bounded (the small-scope
# hypothesis: protocol bugs show in small instances, Jackson 2006). The
# lease scope gives two executors two drives each of one write, one
# renewal, one crash, a lease of two ticks and a clock that runs out at
# three, so a lease can expire, be taken over, be released and be taken
# afresh within it (474,902 states with distinct owners; a clock to four,
# or two writes a drive, passes `STATES_MAX`); the journal scope gives
# three records and two crashes, so a crash can hit a re-append after a
# crash.
LEASE_TICKS = 2
HORIZON = 3
ROUNDS = 2
WRITES = 1
BEATS = 1
CRASHES = 1
RECORDS = 3
FAULTS = 2


class BoundError(errors.ScmleError):
    """A model's reachable states exceed `STATES_MAX`."""

    category = "model_bound"


class Crash(BaseException):
    """Ends a replayed executor's thread, as a process kill would.

    A `BaseException`, so no handler in the real code catches it; its
    store calls after it raise it again and touch nothing.
    """


@dataclasses.dataclass(frozen=True)
class Link:
    """Ties one model action to the code it models.

    Attributes:
        action: The model's action name.
        target: `module:qualified.name` of the function.
        marker: Text the function's source must contain.
    """

    action: str
    target: str
    marker: str

    def resolve(self) -> bool:
        """Whether the function exists and still contains the marker.

        Returns:
            True when it does.
        """
        module, name = self.target.split(":")
        try:
            found = functools.reduce(
                getattr, name.split("."), importlib.import_module(module)
            )
            source = inspect.getsource(found)
        except (ImportError, AttributeError, OSError, TypeError):
            # handled: an unresolvable link is reported as broken
            return False
        return self.marker in source


@dataclasses.dataclass(frozen=True)
class Path:
    """A path through a model's states.

    Attributes:
        labels: The action taken at each step.
        states: The states, the initial one first; one more than labels.
    """

    labels: tuple[str, ...]
    states: tuple[Any, ...]


@dataclasses.dataclass(frozen=True)
class Verdict:
    """What exploring a model found.

    Attributes:
        states: Distinct states reached.
        transitions: Transitions taken.
        depth: Length of the longest shortest path.
        broken: The shortest counterexample of each invariant that fails.
        samples: Paths to `SAMPLES` states spread evenly over the search
            order, the last a deepest state.
    """

    states: int
    transitions: int
    depth: int
    broken: Mapping[str, Path]
    samples: tuple[Path, ...]


class Model(abc.ABC):
    """A protocol as a transition system with invariants.

    Attributes:
        name: Names the model and its scope.
        links: The code each action models.
    """

    name: str
    links: tuple[Link, ...]

    @abc.abstractmethod
    def start(self) -> Iterable[Hashable]:
        """The initial states."""

    @abc.abstractmethod
    def moves(self, state: Any) -> Iterator[tuple[str, Hashable]]:
        """Every step from a state, with its label."""

    @abc.abstractmethod
    def invariants(self) -> Mapping[str, Callable[[Any], bool]]:
        """Each safety property, as a predicate every state must meet."""

    @abc.abstractmethod
    def replay(self, path: Path, folder: pathlib.Path) -> list[str]:
        """Drives the real implementation through a path.

        Args:
            path: A path of this model.
            folder: An empty scratch folder.

        Returns:
            Each step where the real state differs from the model's;
            empty when the real code followed the path.
        """

    @abc.abstractmethod
    def scope(self) -> dict[str, Any]:
        """The bounds explored, for the record."""


def trace(parents: Mapping[Any, tuple[Any, str]], state: Any) -> Path:
    """The path BFS took to a state.

    Args:
        parents: Each reached state's predecessor and label (None for an
            initial state).
        state: The state.

    Returns:
        The path from an initial state.
    """
    labels: list[str] = []
    states = [state]
    while parents[state][0] is not None:
        state, label = parents[state]
        labels.append(label)
        states.append(state)
    return Path(tuple(reversed(labels)), tuple(reversed(states)))


def explore(model: Model, limit: int = STATES_MAX) -> Verdict:
    """Checks every invariant in every reachable state, breadth first.

    Args:
        model: The model.
        limit: Most states to reach.

    Returns:
        The verdict; each broken invariant with a shortest counterexample.

    Raises:
        BoundError: If more than `limit` states are reachable.
    """
    rules = model.invariants()
    parents: dict[Any, tuple[Any, str]] = {}
    depths: dict[Any, int] = {}
    broken: dict[str, Path] = {}
    queue: collections.deque[Any] = collections.deque()
    transitions = 0

    def visit(state: Any, parent: Any, label: str, depth: int) -> None:
        if len(parents) >= limit:
            raise BoundError(
                f"{model.name}: more than {limit} states; shrink the scope",
                {"model": model.name, "limit": limit},
            )
        parents[state] = (parent, label)
        depths[state] = depth
        queue.append(state)
        for name, rule in rules.items():
            if name not in broken and not rule(state):
                broken[name] = trace(parents, state)

    for state in model.start():
        if state not in parents:
            visit(state, None, "", 0)
    while queue:
        state = queue.popleft()
        for label, after in model.moves(state):
            transitions += 1
            if after not in parents:
                visit(after, state, label, depths[state] + 1)
    order = list(parents)
    stride = max(len(order) // SAMPLES, 1)
    picked = order[::-stride][:SAMPLES]
    return Verdict(
        states=len(parents),
        transitions=transitions,
        depth=max(depths.values(), default=0),
        broken={name: broken[name] for name in rules if name in broken},
        samples=tuple(trace(parents, state) for state in picked),
    )


# --- The lease ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Lock:
    """The stored lease.

    Attributes:
        owner: The owner name written into it.
        fence: Its fencing token.
        expires: The tick it lapses at.
        holder: Which executor wrote it (a ghost: the code cannot see it).
    """

    owner: str
    fence: int
    expires: int
    holder: int

    @property
    def version(self) -> tuple[Any, ...]:
        """The store's version: the content.

        `LocalObjectStore` versions an object by its SHA-256, so equal
        content is the same version.
        """
        return (self.owner, self.fence, self.expires)


@dataclasses.dataclass(frozen=True, slots=True)
class Record:
    """The stored run record the lease fences.

    Attributes:
        fence: The token it was written under.
        writer: Which executor wrote it.
        number: Which write of the writer's drive it is.
    """

    fence: int
    writer: int
    number: int

    @property
    def version(self) -> tuple[Any, ...]:
        """The store's version: the content."""
        return (self.fence, self.writer, self.number)


@dataclasses.dataclass(frozen=True, slots=True)
class Worker:
    """One executor: its drive's program counter and its local state.

    Attributes:
        pc: The next store call of its main thread (`Lease.ACTIONS`), or
            `done` or `crashed`.
        fence: Its `fences` entry for the run, or None.
        grant: The position of its lease among every grant (a ghost), or
            -1 without one.
        seen: What its last read kept: the lock's fence and version, or the
            record's version (None when absent).
        stamp: The clock its main thread read last (the lease body is
            built before the store call).
        tries: Attempts left at taking the lease.
        rounds: Drives left.
        writes: Writes left in this drive.
        beats: Renewals left on the heartbeat thread.
        beat: The heartbeat's read lock (its version) and clock, while a
            renewal is between its check and its write.
    """

    pc: str
    fence: int | None
    grant: int
    seen: tuple[Any, ...] | None
    stamp: int
    tries: int
    rounds: int
    writes: int
    beats: int
    beat: tuple[Lock, int] | None


@dataclasses.dataclass(frozen=True, slots=True)
class Field:
    """The lease model's state.

    Attributes:
        clock: The tick.
        lock: The stored lease, or None.
        record: The stored run record, or None.
        workers: The executors.
        crashes: Crashes left.
        grants: Every fence granted, in order (ghost).
        peak: The highest fence a stored record ever carried (ghost).
        latest: The latest grant (position in `grants`) a record was
            written under (ghost).
        stale: Whether a record was written under an earlier grant after
            one under a later grant (ghost).
        refused: Whether a valid holder was refused a write (ghost).
    """

    clock: int
    lock: Lock | None
    record: Record | None
    workers: tuple[Worker, ...]
    crashes: int
    grants: tuple[int, ...]
    peak: int
    latest: int
    stale: bool
    refused: bool


LEASES = "scmle.workflow.executor.leases:Leases"
LOOP = ("fetch", "hold", "put")
WORKFLOW = definition.Workflow(
    "protocol",
    "1",
    (definition.Step("work", definition.Kind.TRAIN),),
)
RUN = "run"


def rest(context: executor.Context) -> dict[str, Any]:
    """The replayed workflow's handler; a replay never runs a step."""
    del context
    return {}


def letter(index: int) -> str:
    """An executor's name in labels: a, b, ..."""
    return chr(ord("a") + index)


class Lease(Model):
    """The run lease with fencing tokens, and the run record it fences.

    Attributes:
        owners: The owner name of each executor.
        guards: The checks the code makes; removing one makes a mutant
            (`version`: the record write's compare-and-swap; `stored`: the
            stored-fence check; `hold`: the lease check before writing).
    """

    GUARDS = frozenset({"version", "stored", "hold"})
    # The store call each action makes: operation and object.
    ACTIONS = {
        "create": ("put", "lock"),
        "read": ("get", "lock"),
        "takeover": ("put", "lock"),
        "fetch": ("get", "record"),
        "hold": ("get", "lock"),
        "put": ("put", "record"),
        "release": ("get", "lock"),
        "drop": ("delete", "lock"),
        "beat": ("get", "lock"),
        "renew": ("put", "lock"),
    }
    links = (
        Link(
            "create",
            f"{LEASES}.acquire",
            "self.store.put(key, self.lease(run_id, 1), if_absent=True)",
        ),
        Link("read", f"{LEASES}.acquire", 'if held["expires"] > self.clock():'),
        Link("takeover", f"{LEASES}.acquire", "if_version=meta.version"),
        Link("fetch", f"{LEASES}.write", "if stored > fence:"),
        Link("hold", f"{LEASES}.hold", 'or held.get("fence") != fence'),
        Link("put", f"{LEASES}.write", "self.store.put(key, body, if_version="),
        Link("release", f"{LEASES}.relinquish", "meta = self.hold(run_id)[1]"),
        Link("drop", f"{LEASES}.relinquish", "self.store.delete("),
        Link("beat", f"{LEASES}.renew", "held, meta = self.hold(run_id)"),
        Link("renew", f"{LEASES}.renew", "if_version=meta.version"),
        Link(
            "expiry",
            f"{LEASES}.lease",
            "max(self.clock() + self.tenure, floor)",
        ),
    )

    def __init__(
        self,
        owners: tuple[str, ...],
        guards: frozenset[str] = GUARDS,
        ticks: int = LEASE_TICKS,
        horizon: int = HORIZON,
        rounds: int = ROUNDS,
        writes: int = WRITES,
        beats: int = BEATS,
        crashes: int = CRASHES,
    ) -> None:
        """Sets the scope.

        Args:
            owners: The owner name of each executor; the code's default
                gives every executor the name `executor`.
            guards: The checks modelled (`GUARDS` for the code).
            ticks: The lease's validity.
            horizon: The last tick.
            rounds: Drives per executor.
            writes: Record writes per drive.
            beats: Renewals per executor.
            crashes: Crashes in a run.
        """
        self.owners = owners
        self.guards = guards
        self.ticks = ticks
        self.horizon = horizon
        self.rounds = rounds
        self.writes = writes
        self.beats = beats
        self.crashes = crashes
        shared = "shared" if len(set(owners)) < len(owners) else "distinct"
        mutant = sorted(self.GUARDS - guards)
        self.name = f"lease, {shared} owners" + (
            f", without {', '.join(mutant)}" if mutant else ""
        )

    def scope(self) -> dict[str, Any]:
        """The bounds explored."""
        return {
            "owners": list(self.owners),
            "guards": sorted(self.guards),
            "ticks": self.ticks,
            "horizon": self.horizon,
            "rounds": self.rounds,
            "writes": self.writes,
            "beats": self.beats,
            "crashes": self.crashes,
        }

    def start(self) -> Iterable[Field]:
        """Every executor about to take the lease; nothing stored."""
        worker = Worker(
            "create",
            None,
            -1,
            None,
            0,
            executor.LEASE_ATTEMPTS,
            self.rounds - 1,
            self.writes,
            self.beats,
            None,
        )
        yield Field(
            clock=0,
            lock=None,
            record=None,
            workers=(worker,) * len(self.owners),
            crashes=self.crashes,
            grants=(),
            peak=0,
            latest=-1,
            stale=False,
            refused=False,
        )

    def invariants(self) -> Mapping[str, Callable[[Field], bool]]:
        """The safety properties.

        `unique`: no fence is granted twice (fencing tokens rise).
        `fenced`: a record written under a fence is never overwritten under
        a lower one. `exclusive`: once the record was written under a
        lease, no lease granted before it writes the record again (two
        holders never act on it interleaved). `admitted`: an executor
        holding the valid lease is never refused a write.
        """
        return {
            "unique": lambda s: len(set(s.grants)) == len(s.grants),
            "fenced": lambda s: s.record is None or s.record.fence == s.peak,
            "exclusive": lambda s: not s.stale,
            "admitted": lambda s: not s.refused,
        }

    def moves(self, state: Field) -> Iterator[tuple[str, Field]]:
        """A tick, a crash, a main-thread or a heartbeat step."""
        if state.clock < self.horizon:
            yield "tick", dataclasses.replace(state, clock=state.clock + 1)
        for index, worker in enumerate(state.workers):
            name = letter(index)
            if worker.pc in ("done", "crashed"):
                continue
            if state.crashes:
                yield f"{name} crash", self.crash(state, index)
            if worker.beat is not None:
                yield f"{name} renew", self.renew(state, index)
            elif worker.pc in LOOP and worker.beats and worker.fence:
                yield f"{name} beat", self.check(state, index)
            if worker.pc != "release" or worker.beat is None:
                yield f"{name} {worker.pc}", self.step(state, index)

    def update(self, state: Field, index: int, **changes: Any) -> Field:
        """The state with one executor changed.

        Args:
            state: The state.
            index: The executor.
            **changes: Its new fields.

        Returns:
            The new state.
        """
        workers = list(state.workers)
        workers[index] = dataclasses.replace(workers[index], **changes)
        return dataclasses.replace(state, workers=tuple(workers))

    def crash(self, state: Field, index: int) -> Field:
        """The executor dies; its lease stays until it lapses."""
        changed = self.update(
            state,
            index,
            pc="crashed",
            fence=None,
            grant=-1,
            seen=None,
            beat=None,
        )
        return dataclasses.replace(changed, crashes=state.crashes - 1)

    def ours(self, state: Field, index: int) -> bool:
        """`Leases.hold`: the stored lease has our owner name and fence."""
        lock = state.lock
        worker = state.workers[index]
        return (
            lock is not None
            and lock.owner == self.owners[index]
            and lock.fence == worker.fence
        )

    def end(self, state: Field, index: int) -> Field:
        """A drive ends (busy, evicted or released): the fence is forgotten."""
        worker = state.workers[index]
        return self.update(
            state,
            index,
            pc="create" if worker.rounds else "done",
            fence=None,
            grant=-1,
            seen=None,
            stamp=state.clock,
            tries=executor.LEASE_ATTEMPTS,
            rounds=max(worker.rounds - 1, 0),
            writes=self.writes,
        )

    def grant(self, state: Field, index: int, fence: int) -> Field:
        """Stores our lease under a fence and enters the write loop."""
        worker = state.workers[index]
        lock = Lock(self.owners[index], fence, worker.stamp + self.ticks, index)
        changed = dataclasses.replace(
            state, lock=lock, grants=(*state.grants, fence)
        )
        return self.update(
            changed,
            index,
            pc="fetch" if self.writes else "release",
            fence=fence,
            grant=len(state.grants),
            seen=None,
            stamp=state.clock,
        )

    def step(self, state: Field, index: int) -> Field:
        """One store call of an executor's main thread, by its action."""
        action = getattr(self, state.workers[index].pc)
        changed: Field = action(state, index)
        return changed

    def create(self, state: Field, index: int) -> Field:
        """`Leases.acquire`: create the lease if absent, at fence 1."""
        if state.lock is None:
            return self.grant(state, index, 1)
        return self.update(state, index, pc="read")

    def read(self, state: Field, index: int) -> Field:
        """`Leases.acquire`: read the lease; busy unless it lapsed."""
        worker = state.workers[index]
        lock = state.lock
        if lock is None:
            if worker.tries <= 1:
                return self.end(state, index)
            return self.update(
                state,
                index,
                pc="create",
                tries=worker.tries - 1,
                stamp=state.clock,
            )
        if lock.expires > state.clock:
            return self.end(state, index)
        seen = (lock.fence, lock.version)
        return self.update(
            state, index, pc="takeover", seen=seen, stamp=state.clock
        )

    def takeover(self, state: Field, index: int) -> Field:
        """`Leases.acquire`: replace the lapsed lease by compare-and-swap."""
        lock = state.lock
        seen = state.workers[index].seen or (0, None)
        if lock is not None and lock.version == seen[1]:
            return self.grant(state, index, seen[0] + 1)
        return self.end(state, index)

    def fetch(self, state: Field, index: int) -> Field:
        """`Leases.write`: read the record; refuse a newer fence."""
        worker = state.workers[index]
        record = state.record
        fence = worker.fence or 0
        if record and record.fence > fence and "stored" in self.guards:
            lock = state.lock
            valid = (
                lock is not None
                and lock.holder == index
                and lock.fence == fence
                and lock.expires > state.clock
            )
            state = dataclasses.replace(state, refused=state.refused or valid)
            return self.update(state, index, pc="release")
        seen = record.version if record else None
        return self.update(state, index, pc="hold", seen=seen)

    def hold(self, state: Field, index: int) -> Field:
        """`Leases.write` calls `Leases.hold`: is the lease still ours?"""
        if self.ours(state, index) or "hold" not in self.guards:
            return self.update(state, index, pc="put")
        return self.update(state, index, pc="release")

    def put(self, state: Field, index: int) -> Field:
        """`Leases.write`: write the record if it is the version read."""
        worker = state.workers[index]
        record = state.record
        matches = (record.version if record else None) == worker.seen
        if not matches and "version" in self.guards:
            return self.update(state, index, pc="release")
        fence = worker.fence or 0
        changed = dataclasses.replace(
            state,
            record=Record(fence, index, self.writes - worker.writes),
            peak=max(state.peak, fence),
            latest=max(state.latest, worker.grant),
            stale=state.stale or worker.grant < state.latest,
        )
        left = worker.writes - 1
        return self.update(
            changed,
            index,
            pc="fetch" if left else "release",
            writes=left,
            stamp=state.clock,
        )

    def release(self, state: Field, index: int) -> Field:
        """`Leases.relinquish`: check the lease is still ours."""
        lock = state.lock
        if not self.ours(state, index) or lock is None:
            return self.end(state, index)
        return self.update(state, index, pc="drop", seen=lock.version)

    def drop(self, state: Field, index: int) -> Field:
        """`Leases.relinquish`: delete the lease if it is the version read."""
        lock = state.lock
        if lock is not None and lock.version == state.workers[index].seen:
            state = dataclasses.replace(state, lock=None)
        return self.end(state, index)

    def check(self, state: Field, index: int) -> Field:
        """The heartbeat's `hold` before a renewal."""
        lock = state.lock
        if not self.ours(state, index) or lock is None:
            return self.update(state, index, beats=0)
        return self.update(state, index, beat=(lock, state.clock))

    def renew(self, state: Field, index: int) -> Field:
        """The heartbeat's compare-and-swap that extends the lease."""
        worker = state.workers[index]
        lock = state.lock
        if worker.beat is None:
            return state
        held, stamp = worker.beat
        if lock is None or lock.version != held.version:
            return self.update(state, index, beats=0, beat=None)
        renewed = Lock(
            held.owner,
            held.fence,
            max(stamp + self.ticks, held.expires),
            index,
        )
        changed = dataclasses.replace(state, lock=renewed)
        return self.update(changed, index, beats=worker.beats - 1, beat=None)

    def replay(self, path: Path, folder: pathlib.Path) -> list[str]:
        """Drives real executors through the path, one store call a step.

        Each executor runs its drives on a thread (`Leases.acquire`, then
        `Leases.write`, then `Leases.relinquish`) and each renewal on a
        thread started at its `beat` step (`Leases.renew`; the
        heartbeat starts once the lease is taken), over a `Gate` that lets
        one store call through per step of the path. A tick advances
        their shared clock; a crash ends the executor's threads with no
        store call.

        Args:
            path: A path of this model.
            folder: An empty scratch folder.

        Returns:
            The steps where the real store, clock or fences differ from
            the model's, or where the real code made another store call.
        """
        gate = Gate(objectstore.LocalObjectStore(folder / "state"))
        moment = [0]
        executors = [
            executor.Executor(
                WORKFLOW,
                gate,
                {"work": rest},
                clock=lambda: float(moment[0]),
                lease_seconds=float(self.ticks),
                owner=owner,
            )
            for owner in self.owners
        ]
        for index, real in enumerate(executors):
            name = letter(index)
            gate.spawn(name, functools.partial(self.drive, real, index))
        beats: dict[str, str] = {}
        found: list[str] = []
        try:
            gate.settle()
            found += self.compare(gate, executors, path.states[0], 0)
            steps = zip(path.labels, path.states[1:], strict=True)
            for number, (label, state) in enumerate(steps, 1):
                name, action = [*label.split(" "), ""][:2]
                if label == "tick":
                    moment[0] += 1
                elif action == "crash":
                    gate.crash(name, beats.get(name, name))
                else:
                    if action == "beat":
                        # A heartbeat starts inside `Leases.pulse`, after
                        # the lease is taken; it reads `fences` then.
                        beats[name] = f"{name} beat {number}"
                        real = executors[ord(name) - ord("a")]
                        gate.spawn(
                            beats[name], functools.partial(self.pulse, real)
                        )
                        gate.settle()
                    channel = (
                        beats[name] if action in ("beat", "renew") else name
                    )
                    made = gate.grant(channel)
                    if made != self.ACTIONS[action]:
                        found.append(
                            f"step {number} ({label}): the code made {made}"
                        )
                        break
                found += self.compare(gate, executors, state, number)
                if found:
                    break
        finally:
            found += gate.close()
        return found

    def drive(self, real: executor.Executor, index: int) -> None:
        """An executor's drives, as `Executor.drive` makes them."""
        key = real.base(RUN) + "steps/work.json"
        for ignored in range(self.rounds):
            try:
                real.acquire(RUN)
            except executor.BusyError:
                continue
            try:
                for written in range(self.writes):
                    real.write(key, {"by": index, "n": written})
            except executor.EvictionError:
                # handled: the drive stops; the lease is given up below
                pass
            finally:
                real.relinquish(RUN)

    def pulse(self, real: executor.Executor) -> None:
        """One renewal of an executor's heartbeat (`Leases.beat`).

        A failed renewal ends the heartbeat, as in `Leases.beat`.
        """
        try:
            real.renew(RUN)
        except executor.EvictionError:
            # handled: the heartbeat ends, as `Leases.beat` ends it
            return

    def compare(
        self,
        gate: "Gate",
        executors: list[executor.Executor],
        state: Field,
        number: int,
    ) -> list[str]:
        """Where the real store and executors differ from a model state."""
        real = gate.inner
        base = executors[0].base(RUN)
        found = []
        lock = stored(real, base + "lock")
        pairs: dict[str, tuple[Any, Any]] = {
            "lock": (
                lock and [lock["owner"], lock["fence"], lock["expires"]],
                state.lock
                and [state.lock.owner, state.lock.fence, state.lock.expires],
            ),
        }
        record = stored(real, base + "steps/work.json")
        pairs["record"] = (
            record and [record["fence"], record["by"]],
            state.record and [state.record.fence, state.record.writer],
        )
        for name, (have, want) in pairs.items():
            if have != want:
                found.append(f"step {number}: {name} {have}, model {want}")
        for index, worker in enumerate(state.workers):
            fence = executors[index].fences.get(RUN)
            if worker.pc != "crashed" and fence != worker.fence:
                found.append(
                    f"step {number}: executor {index} fence {fence}, "
                    f"model {worker.fence}"
                )
        return found


def stored(store: objectstore.Depot, key: str) -> dict[str, Any] | None:
    """A stored JSON object, or None when absent."""
    try:
        data = store.get(key)[0]
    except objectstore.AbsentError:
        # handled: absent is part of the compared state
        return None
    found: dict[str, Any] = json.loads(data)
    return found


class Gate(objectstore.Depot):
    """A store that lets one call through per turn, for a replay.

    Each replayed thread is a channel; a call waits until the replay
    grants its channel a turn. A crashed channel's calls raise `Crash`.

    Attributes:
        inner: The real store.
        condition: Guards the fields below.
        names: Each thread's channel.
        waiting: The call each waiting channel is about to make.
        granted: Channels allowed one call.
        finished: Channels whose thread ended.
        crashed: Channels crashed.
        failures: Unexpected errors the threads raised.
        threads: The threads.
    """

    def __init__(self, inner: objectstore.Depot) -> None:
        """Wraps a store.

        Args:
            inner: The real store.
        """
        super().__init__(inner.location)
        self.inner = inner
        self.condition = threading.Condition()
        self.names: dict[int, str] = {}
        self.waiting: dict[str, tuple[str, str]] = {}
        self.granted: set[str] = set()
        self.finished: set[str] = set()
        self.crashed: set[str] = set()
        self.failures: list[str] = []
        self.threads: list[threading.Thread] = []

    def spawn(self, name: str, program: Callable[[], None]) -> None:
        """Starts a channel's thread.

        Args:
            name: The channel.
            program: What the thread runs.
        """

        def run() -> None:
            with self.condition:
                self.names[threading.get_ident()] = name
            try:
                program()
            except Crash:
                # handled: a crash ends the thread with no store call
                pass
            except Exception as error:  # noqa: BLE001 -- reported as divergence
                self.failures.append(f"{name}: {error!r}")
            finally:
                with self.condition:
                    self.finished.add(name)
                    self.condition.notify_all()

        thread = threading.Thread(target=run, daemon=True)
        self.threads.append(thread)
        thread.start()

    def settled(self) -> bool:
        """Whether every thread waits at a call or has ended."""
        names = set(self.waiting) | self.finished
        return len(names) == len(self.threads) and not self.granted

    def settle(self) -> None:
        """Waits until every thread waits at a call or has ended.

        Raises:
            ReliabilityError: If that does not happen in `WAIT_SECONDS`.
        """
        with self.condition:
            if not self.condition.wait_for(self.settled, WAIT_SECONDS):
                raise errors.ReliabilityError(
                    "a replayed thread never reached a store call"
                )

    def grant(self, name: str) -> tuple[str, str] | None:
        """Lets a channel make its next call and waits for the next pause.

        Args:
            name: The channel.

        Returns:
            The call it made, or None when it had none to make.
        """
        with self.condition:
            made = self.waiting.get(name)
            if made is None:
                return None
            self.granted.add(name)
            self.condition.notify_all()
        self.settle()
        return made

    def crash(self, *names: str) -> None:
        """Ends channels as a kill would: their next call raises `Crash`.

        Args:
            *names: The channels.
        """
        with self.condition:
            self.crashed.update(names)
            self.condition.notify_all()
        self.settle()

    def close(self) -> list[str]:
        """Crashes every channel and joins the threads.

        Returns:
            The unexpected errors the threads raised.
        """
        with self.condition:
            names = set(self.names.values())
        self.crash(*names)
        for thread in self.threads:
            thread.join(WAIT_SECONDS)
        return list(self.failures)

    def enter(self, operation: str, key: str) -> None:
        """Waits for the calling channel's turn.

        Args:
            operation: The call: put, get or delete.
            key: The object key.

        Raises:
            Crash: If the channel was crashed.
        """
        kind = "lock" if key.endswith("/lock") else "record"
        with self.condition:
            name = self.names[threading.get_ident()]
            self.waiting[name] = (operation, kind)
            self.condition.notify_all()
            self.condition.wait_for(
                lambda: name in self.granted or name in self.crashed
            )
            del self.waiting[name]
            self.granted.discard(name)
            if name in self.crashed:
                raise Crash(name)

    def put(
        self,
        key: str,
        data: bytes,
        *,
        if_absent: bool = False,
        if_version: str | None = None,
    ) -> objectstore.ObjectMeta:
        """Stores an object on the channel's turn."""
        self.enter("put", key)
        return self.inner.put(
            key, data, if_absent=if_absent, if_version=if_version
        )

    def get(self, key: str) -> tuple[bytes, objectstore.ObjectMeta]:
        """Reads an object on the channel's turn."""
        self.enter("get", key)
        return self.inner.get(key)

    def head(self, key: str) -> objectstore.ObjectMeta:
        """Reads metadata on the channel's turn."""
        self.enter("head", key)
        return self.inner.head(key)

    def list_objects(self, prefix: str = "") -> list[objectstore.ObjectMeta]:
        """Lists objects on the channel's turn."""
        self.enter("list", prefix)
        return self.inner.list_objects(prefix)

    def delete(self, key: str, *, if_version: str | None = None) -> None:
        """Removes an object on the channel's turn."""
        self.enter("delete", key)
        self.inner.delete(key, if_version=if_version)


# --- The journals ------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Shelf:
    """A journal model's state: the file and its one client.

    Attributes:
        tokens: The file: `("line", k)` a whole line of record k, `("part",
            k)` a line cut short, `("end", 0)` a lone newline.
        synced: How many tokens are on stable storage.
        exists: Whether the file exists.
        pc: The client's next step (`Journal.ACTIONS`), `done` or
            `halted`.
        next: The record being appended.
        skip: The records the client's caller skips since it last
            recovered (`Journal.resume`).
        acked: Records whose append returned (ghost).
        pending: The tokens the append will write.
        crashes: Crashes left.
        lost: Whether recovery missed an acknowledged record, or could not
            read the journal (ghost).
    """

    tokens: tuple[tuple[str, int], ...]
    synced: int
    exists: bool
    pc: str
    next: int
    skip: frozenset[int]
    acked: int
    pending: tuple[tuple[str, int], ...]
    crashes: int
    lost: bool


def lines(tokens: Iterable[tuple[str, int]]) -> list[int | None]:
    """The file's lines: a record number, or None for an unreadable line.

    Args:
        tokens: The file.

    Returns:
        One entry per line, a trailing unterminated line included.
    """
    found: list[int | None] = []
    cut = False
    for kind, number in tokens:
        if kind == "part":
            cut = True
            continue
        found.append(number if kind == "line" and not cut else None)
        cut = False
    if cut:
        found.append(None)
    return found


def twice(tokens: Iterable[tuple[str, int]]) -> list[int]:
    """The records the file holds as more than one whole line.

    Args:
        tokens: The file.

    Returns:
        Their numbers, ascending.
    """
    counts = collections.Counter(n for n in lines(tokens) if n is not None)
    return sorted(n for n, count in counts.items() if count > 1)


class Journal(Model):
    """An append-only line journal, its recovery and a crashing client.

    The client is the journal's real caller: it recovers, then appends
    each record its resume rule (`resume`) does not skip, in order.

    Attributes:
        crash: `kill` or `power`.
        records: Records the client appends.
        crashes: Crashes in a run.
        distinct: Whether records are keyed: the reader returns each record
            once and the caller records each once, so a record appended
            twice is a defect (`single`). False for a log whose caller
            appends a record again by design.
    """

    # The client's steps, in order; `recover` also applies `resume`.
    ACTIONS = ("recover", "prepare", "write", "sync", "ack")
    CRASHES = ("kill", "power")
    kind = "journal"
    distinct = True

    def __init__(
        self, crash: str, records: int = RECORDS, crashes: int = FAULTS
    ) -> None:
        """Sets the scope.

        Args:
            crash: `kill` keeps every written byte; `power` keeps the
                synced bytes and any prefix of the rest.
            records: Records appended.
            crashes: Crashes in a run.

        Raises:
            InputError: If the crash kind is unknown.
        """
        if crash not in self.CRASHES:
            raise errors.InputError(f"crash is one of {self.CRASHES}: {crash}")
        self.crash = crash
        self.records = records
        self.crashes = crashes
        self.name = f"{self.kind} journal, {crash} crashes"

    def scope(self) -> dict[str, Any]:
        """The bounds explored."""
        return {
            "crash": self.crash,
            "records": self.records,
            "crashes": self.crashes,
        }

    @abc.abstractmethod
    def scan(self, tokens: tuple[tuple[str, int], ...]) -> list[int] | None:
        """What recovery reads: the records in order, or None if it fails."""

    @abc.abstractmethod
    def resume(self, found: list[int]) -> frozenset[int]:
        """The records the caller skips after recovery read `found`."""

    @abc.abstractmethod
    def emit(
        self, state: Shelf, number: int
    ) -> tuple[tuple[str, int], ...] | None:
        """The tokens an append writes, or None if it fails first."""

    @abc.abstractmethod
    def open(self, folder: pathlib.Path) -> Any:
        """The real journal in a folder."""

    @abc.abstractmethod
    def append(self, real: Any, number: int) -> None:
        """Appends record `number` to the real journal."""

    @abc.abstractmethod
    def recover(self, real: Any) -> list[int]:
        """Reads the real journal back, record numbers in order."""

    @abc.abstractmethod
    def file(self, real: Any) -> pathlib.Path:
        """The real journal's file."""

    def start(self) -> Iterable[Shelf]:
        """A client about to recover an absent journal."""
        yield Shelf(
            tokens=(),
            synced=0,
            exists=False,
            pc="recover",
            next=0,
            skip=frozenset(),
            acked=0,
            pending=(),
            crashes=self.crashes,
            lost=False,
        )

    def invariants(self) -> Mapping[str, Callable[[Shelf], bool]]:
        """The safety properties.

        `kept`: recovery reads the journal and shows every acknowledged
        record. `single`, for a `distinct` journal: no record is in the
        file twice as a whole line (the caller never appends a record it
        recovered).
        """
        rules: dict[str, Callable[[Shelf], bool]] = {
            "kept": lambda s: not s.lost,
        }
        if self.distinct:
            rules["single"] = lambda s: twice(s.tokens) == []
        return rules

    def moves(self, state: Shelf) -> Iterator[tuple[str, Shelf]]:
        """The client's next step, or a crash."""
        if state.pc == "halted":
            return
        if state.crashes:
            yield from self.fail(state)
        if state.pc != "done":
            yield state.pc, self.step(state)

    def fail(self, state: Shelf) -> Iterator[tuple[str, Shelf]]:
        """Every crash: what survives of the unsynced tokens."""
        kept = state.tokens[: state.synced]
        loose = state.tokens[state.synced :]
        survivors = [state.tokens]
        if self.crash == "power":
            survivors = [kept + loose[:cut] for cut in range(len(loose) + 1)]
            survivors += [
                kept + loose[:cut] + (("part", number),)
                for cut, (kind, number) in enumerate(loose)
                if kind == "line"
            ]
        for tokens in survivors:
            cut = len(tokens) > state.synced and tokens[-1][0] == "part"
            yield (
                f"crash, {len(tokens)} of {len(state.tokens)} tokens kept"
                + (", the last cut" if cut else ""),
                dataclasses.replace(
                    state,
                    tokens=tokens,
                    synced=len(tokens),
                    pc="recover",
                    pending=(),
                    crashes=state.crashes - 1,
                ),
            )

    def advance(self, state: Shelf, first: int) -> Shelf:
        """Moves to the first record from `first` the caller does not skip."""
        following = next(
            (n for n in range(first, self.records) if n not in state.skip),
            self.records,
        )
        return dataclasses.replace(
            state,
            pc="prepare" if following < self.records else "done",
            next=following,
        )

    def restore(self, state: Shelf) -> Shelf:
        """Recovery: read the journal, check it, resume by the caller."""
        found = self.scan(state.tokens)
        if found is None:
            return dataclasses.replace(state, pc="halted", lost=True)
        changed = dataclasses.replace(
            state,
            skip=self.resume(found),
            lost=state.lost or any(n not in found for n in range(state.acked)),
        )
        return self.advance(changed, 0)

    def prepare(self, state: Shelf) -> Shelf:
        """The append's work before it writes."""
        pending = self.emit(state, state.next)
        if pending is None:
            return dataclasses.replace(state, pc="halted", lost=True)
        return dataclasses.replace(state, pc="write", pending=pending)

    def step(self, state: Shelf) -> Shelf:
        """One step of the client."""
        if state.pc == "recover":
            return self.restore(state)
        if state.pc == "prepare":
            return self.prepare(state)
        if state.pc == "write":
            tokens = state.tokens + state.pending
            return dataclasses.replace(
                state, pc="sync", tokens=tokens, exists=True, pending=()
            )
        if state.pc == "sync":
            return dataclasses.replace(
                state, pc="ack", synced=len(state.tokens)
            )
        acked = dataclasses.replace(
            state, acked=max(state.acked, state.next + 1)
        )
        return self.advance(acked, state.next + 1)

    def replay(self, path: Path, folder: pathlib.Path) -> list[str]:
        """Drives the real journal through the path.

        `write` is the real append (it also syncs; the model's `sync` and
        `ack` then change nothing on disk). A crash rewrites the file to
        the bytes the model says survive. At `recover` and `prepare` the
        real reader's outcome is compared with the model's (as keys for a
        `distinct` journal, whose reader returns each record once), and
        after every step the file's bytes with the model's tokens, so a
        record written twice shows there.

        Args:
            path: A path of this model.
            folder: An empty scratch folder.

        Returns:
            The steps where the real journal differs from the model.
        """
        real = self.open(folder)
        target = self.file(real)
        pieces: list[bytes] = []
        found: list[str] = []
        steps = zip(path.labels, path.states[:-1], path.states[1:], strict=True)
        for number, (label, before, after) in enumerate(steps, 1):
            if label in ("recover", "prepare"):
                found += self.confront(real, before, label, number)
            elif label == "write":
                old = target.read_bytes() if target.exists() else b""
                self.append(real, before.next)
                new = target.read_bytes()[len(old) :]
                if before.pending[0][0] == "end":
                    pieces += [new[:1], new[1:]]
                else:
                    pieces.append(new)
            elif label.startswith("crash"):
                pieces = [
                    piece[: len(piece) // 2] if token != was else piece
                    for token, was, piece in zip(
                        after.tokens, before.tokens, pieces, strict=False
                    )
                ]
                if target.exists():
                    target.write_bytes(b"".join(pieces))
            have = target.read_bytes() if target.exists() else b""
            if have != b"".join(pieces) or len(pieces) != len(after.tokens):
                found.append(f"step {number} ({label}): file differs")
            if found:
                break
        return found

    def confront(
        self, real: Any, state: Shelf, label: str, number: int
    ) -> list[str]:
        """Compares the real reader with the model at a read."""
        try:
            have: list[int] | None = self.recover(real)
        except (ValueError, errors.ScmleError):
            # handled: an unreadable journal is part of the compared state
            have = None
        want = self.scan(state.tokens)
        if self.distinct and want is not None and have is not None:
            want = list(dict.fromkeys(want))
            have = list(dict.fromkeys(have))
        if have != want:
            return [f"step {number} ({label}): read {have}, model {want}"]
        return []


ENGINE = "scmle.pipeline.engine:Pipeline.advance"
ORCHESTRATOR = "scmle.pipeline.orchestrator:Orchestrator.consult"
ERASURE = "scmle.api.erasure.erasures:Erasure.execute"


class Custody(Journal):
    """The custody journal: a hash chain read whole on every append.

    Its caller is the orchestrator's gate review: a resumed run consults
    every gate again, restored stages included (`Pipeline.advance`), and
    `consult` records each verdict with `Journal.decide`
    without looking at the journal, so a resume appends every gate's
    verdict again. The journal is a log of verdicts, read latest first
    (`Journal.rejected`), so a repeated verdict is by design and `single`
    does not apply.
    """

    kind = "custody"
    distinct = False
    links = (
        Link(
            "recover",
            "scmle.assurance.custody.journal:Journal.read",
            "entry: Entry = json.loads(line)",
        ),
        Link("recover", ENGINE, "review(stage.state, board)"),
        Link("recover", ORCHESTRATOR, "record.decide("),
        Link(
            "prepare",
            "scmle.assurance.custody.journal:Journal.append",
            "entries = self.read()",
        ),
        Link(
            "write",
            "scmle.assurance.custody.journal:Journal.append",
            'durability.append_text(self.path, line + "\\n")',
        ),
        Link("write", "scmle.durability:append_text", 'target.open("a"'),
        Link(
            "sync",
            "scmle.assurance.custody.journal:Journal.append",
            "durability.sync(self.path)",
        ),
        Link("ack", ENGINE, "for index, stage in enumerate(stages):"),
    )

    def resume(self, found: list[int]) -> frozenset[int]:
        """The orchestrator skips no gate: it decides every one again."""
        del found
        return frozenset()

    def scan(self, tokens: tuple[tuple[str, int], ...]) -> list[int] | None:
        """`Journal.read`: every line must parse, or reading fails."""
        found = lines(tokens)
        if None in found:
            return None
        return [number for number in found if number is not None]

    def emit(
        self, state: Shelf, number: int
    ) -> tuple[tuple[str, int], ...] | None:
        """`Journal.append` reads the journal first, then writes a line."""
        if self.scan(state.tokens) is None:
            return None
        return (("line", number),)

    def open(self, folder: pathlib.Path) -> journal.Journal:
        """A custody journal for run `run`."""
        return journal.Journal(folder / "custody" / f"{RUN}.jsonl", RUN)

    def append(self, real: journal.Journal, number: int) -> None:
        """Appends a checkpoint verdict naming the record."""
        real.decide(
            f"g{number}", approved=True, reviewer="r", reason=f"{number}"
        )

    def recover(self, real: journal.Journal) -> list[int]:
        """The records `Journal.read` returns."""
        return [int(str(entry["gate"])[1:]) for entry in real.read()]

    def file(self, real: journal.Journal) -> pathlib.Path:
        """The journal's file."""
        return real.path


class Erasure(Journal):
    """The erasure journal: a torn last line is skipped and fenced off.

    Its caller is `Erasure.execute`, which reads the finished steps once
    and performs, in order, every step they do not show.
    """

    kind = "erasure"
    links = (
        Link(
            "recover",
            "scmle.api.erasure.receipts:Journal.done",
            "except json.JSONDecodeError:",
        ),
        Link("recover", ERASURE, "done = ledger.done()"),
        Link("recover", ERASURE, "if index not in done:"),
        Link(
            "prepare",
            "scmle.api.erasure.receipts:Journal.finish",
            'torn = path.is_file() and not path.read_bytes().endswith(b"\\n")',
        ),
        Link(
            "write",
            "scmle.api.erasure.receipts:Journal.finish",
            'durability.append_text(path, ("\\n" if torn else "") + line',
        ),
        Link("write", "scmle.durability:append_text", 'target.open("a"'),
        Link(
            "sync",
            "scmle.api.erasure.receipts:Journal.finish",
            "durability.sync(path)",
        ),
        Link("ack", ERASURE, "ledger.finish(index, done[index])"),
    )

    def resume(self, found: list[int]) -> frozenset[int]:
        """`Erasure.execute` skips every step the journal shows."""
        return frozenset(found)

    def scan(self, tokens: tuple[tuple[str, int], ...]) -> list[int] | None:
        """`Journal.done`: unreadable lines are skipped."""
        return [number for number in lines(tokens) if number is not None]

    def emit(
        self, state: Shelf, number: int
    ) -> tuple[tuple[str, int], ...] | None:
        """`Journal.finish` ends a torn last line before its own."""
        tokens = state.tokens
        torn = state.exists and (not tokens or tokens[-1][0] == "part")
        line = ("line", number)
        return (("end", 0), line) if torn else (line,)

    def open(self, folder: pathlib.Path) -> receipts.Journal:
        """An erasure journal in a fresh folder."""
        return receipts.Journal(folder / "erasure")

    def append(self, real: receipts.Journal, number: int) -> None:
        """Records step `number` as finished."""
        real.finish(number, {"record": number})

    def recover(self, real: receipts.Journal) -> list[int]:
        """The steps `Journal.done` returns, in file order."""
        return list(real.done())

    def file(self, real: receipts.Journal) -> pathlib.Path:
        """The journal's file."""
        return real.folder / receipts.JOURNAL


def models() -> tuple[Model, ...]:
    """Every model the claim checks, at its full scope.

    The lease is checked with distinct owner names and with the shared
    default name (`Executor(owner="executor")`), which is how the code's
    own callers build executors; the journals under both crash kinds.
    """
    return (
        Lease(("a", "b")),
        Lease(("executor", "executor")),
        Custody("kill"),
        Custody("power"),
        Erasure("kill"),
        Erasure("power"),
    )


def replay(model: Model, path: Path) -> list[str]:
    """Replays a path against the real code in a fresh scratch folder.

    Args:
        model: The model.
        path: One of its paths.

    Returns:
        Where the real code diverged; empty when it followed the path.
    """
    with tempfile.TemporaryDirectory(prefix="protocols-") as scratch:
        return model.replay(path, pathlib.Path(scratch))


def judge(model: Model, limit: int = STATES_MAX) -> dict[str, Any]:
    """Explores a model and replays its counterexamples and samples.

    Args:
        model: The model.
        limit: Most states to reach.

    Returns:
        The model's record: scope, state counts, each invariant's outcome
        (with the counterexample's labels and its replay), the sample
        replays and each link's resolution.
    """
    verdict = explore(model, limit)
    outcomes: dict[str, dict[str, Any]] = {}
    for name in model.invariants():
        path = verdict.broken.get(name)
        if path is None:
            outcomes[name] = {"holds": True}
            continue
        outcomes[name] = {
            "holds": False,
            "counterexample": list(path.labels),
            "diverged": replay(model, path),
        }
    diverged = [
        found for path in verdict.samples if (found := replay(model, path))
    ]
    return {
        "model": model.name,
        "scope": model.scope(),
        "states": verdict.states,
        "transitions": verdict.transitions,
        "depth": verdict.depth,
        "invariants": outcomes,
        "samples": {
            "paths": len(verdict.samples),
            "steps": sum(len(path.labels) for path in verdict.samples),
            "diverged": diverged,
        },
        "links": {
            f"{link.action} -> {link.target}": link.resolve()
            for link in model.links
        },
    }


def decide(records: list[dict[str, Any]]) -> str:
    """The claim's verdict over every model's record.

    Args:
        records: The models' records from `judge`.

    Returns:
        "not measured" when a replay diverged or a link is broken (the
        model is not the code), else "fail" when an invariant fails, else
        "pass".
    """
    faithful = all(
        not record["samples"]["diverged"]
        and all(record["links"].values())
        and not any(
            outcome.get("diverged") for outcome in record["invariants"].values()
        )
        for record in records
    )
    if not faithful:
        return "not measured"
    holds = all(
        outcome["holds"]
        for record in records
        for outcome in record["invariants"].values()
    )
    return "pass" if holds else "fail"


def digest() -> str:
    """Hash of the modelled code and of the models.

    Returns:
        SHA-256 over the `MODELED` files and this module.
    """
    found = hashlib.sha256()
    for name in MODELED:
        found.update(name.encode())
        found.update(hashlib.sha256((ROOT / name).read_bytes()).digest())
    found.update(pathlib.Path(__file__).read_bytes())
    return found.hexdigest()


NOTES = [
    (
        "the preregistration names TLA+; the models are Python transition "
        "systems checked by an explicit-state breadth-first search, with "
        "each step replayed against the real code"
    ),
    (
        "the scope is bounded (two executors, two drives of one write, one "
        "renewal, one crash, a two-tick lease, three ticks; three records "
        "and two crashes); a bug needing more is not excluded"
    ),
    (
        "the lease model's drive keeps the heartbeat over every record "
        "write; `Executor.drive` writes run.json once before and once "
        "after the heartbeat runs"
    ),
    (
        "store versions are content, as `LocalObjectStore` makes them (a "
        "SHA-256); a backend with generation ids lets fewer conditional "
        "writes through, so its behaviours are a subset"
    ),
    (
        "one global clock; clock skew between executors is not modelled "
        "(CON-20's renewal floor is)"
    ),
    (
        "a power crash keeps any prefix of the unsynced bytes and the "
        "file's directory entry; reordered or zero-filled blocks and a "
        "lost directory entry are not modelled"
    ),
    (
        "duplicate custody verdicts are outside the model: a resumed run "
        "decides every gate again by design, and the journal is a log "
        "read latest first, so `single` is checked on the erasure journal "
        "only"
    ),
    (
        "the journal models stop at the journal; the pipeline's events "
        "log and trial checkpoints, and the ledger, are not modelled"
    ),
]


def measure(chosen: tuple[Model, ...] | None = None) -> dict[str, Any]:
    """Checks the models and decides the clause.

    Args:
        chosen: The models to check; None checks `models()`.

    Returns:
        The document: the code hash, every model's record and the verdict
        (only over every model).
    """
    every = models()
    picked = every if chosen is None else chosen
    records = [judge(model) for model in picked]
    complete = {m.name for m in picked} == {m.name for m in every}
    return {
        "version": 1,
        "claim": "C7",
        "code": digest(),
        "complete": complete,
        "models": records,
        "verdict": decide(records) if complete else "not measured",
        "not_measured": NOTES,
    }


def load(document: dict[str, Any] | None) -> dict[str, Any] | None:
    """The given document, or the stored one (None when nothing is stored).

    Args:
        document: A document, or None to read `PATH`.

    Returns:
        The document, or None.
    """
    if document is not None:
        return document
    if not PATH.exists():
        return None
    loaded: dict[str, Any] = json.loads(PATH.read_text(encoding="utf-8"))
    return loaded


def problems(document: dict[str, Any] | None = None) -> list[str]:
    """Why the document cannot decide the protocols clause.

    Args:
        document: The document; read from `PATH` when None.

    Returns:
        Messages for a missing file, another code hash or a partial
        measurement; empty when it decides.
    """
    found = load(document)
    if found is None:
        return [f"protocols: {PATH.name} is missing"]
    issues = []
    if found.get("code") != digest():
        issues.append("protocols: measured on other code")
    if not found.get("complete"):
        issues.append("protocols: not every model was checked")
    return issues


def summary(document: dict[str, Any] | None = None) -> tuple[str, str]:
    """The scorecard's value and verdict for the protocols clause.

    Args:
        document: The document; read from `PATH` when None.

    Returns:
        Each model's failing invariants (or "all hold"), and the verdict;
        "not measured" when missing, stale or partial.
    """
    found = load(document)
    if found is None:
        return "no measurement", "not measured"
    issues = problems(found)
    if issues:
        return "; ".join(issues), "not measured"
    parts = []
    for record in found["models"]:
        failing = [
            name
            for name, outcome in record["invariants"].items()
            if not outcome["holds"]
        ]
        told = ", ".join(failing) + " fail" if failing else "all hold"
        parts.append(f"{record['model']}: {told}")
    return "; ".join(parts), str(found["verdict"])


def main(arguments: list[str] | None = None) -> int:
    """Measures (`--show`, `--write`) or checks (`--check`) the clause.

    Args:
        arguments: Command-line arguments; None reads `sys.argv`.

    Returns:
        0 after `--show` or `--write`; for `--check`, 0 when `problems` is
        empty and 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--show", action="store_true")
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--model", action="append", default=[])
    options = parser.parse_args(arguments)
    if options.check:
        issues = problems()
        print(json.dumps({"ok": not issues, "problems": issues}, indent=2))
        return 0 if not issues else 1
    if options.write and options.model:
        parser.error("--write checks every model; --model is for --show")
    known = {model.name: model for model in models()}
    unknown = sorted(set(options.model) - set(known))
    if unknown:
        parser.error(f"unknown models {unknown}; known: {sorted(known)}")
    chosen = tuple(known[name] for name in options.model) or None
    document = measure(chosen)
    text = json.dumps(document, indent=2, sort_keys=True)
    if options.write:
        PATH.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
