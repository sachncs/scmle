"""Claim C7 in part: kill -9 anywhere durable resumes to the same result.

Plan section 5, claim C7 "Operates without loss or stall", its clause "kill
-9 at every stage boundary, journal append and lease operation resumes to
an identical result with no lost or duplicated record" (pre-registered in
`docs/preregistration/c7.md`, "Kill and resume").

Four subjects, each a real use of the product run in child processes:

* `Engine`: the pipeline engine, `Project(...).run()` on a generated
  binary table, resumed with `Project.resume`. Its kill points are every
  stage record (`archive.Directory.record`, a stage boundary), the
  completion marker (`Directory.finish`), every event appended to the run's
  event log (`events.Log.emit`), every trial checkpoint
  (`journal.Journal.transcribe`) and every custody journal entry
  (`custody.journal.Journal.append`).
* `Executor`: the workflow executor (`workflow.executor.Executor`) running
  a four-step workflow on a local object store, continued by starting the
  same trigger again after its lease expired. Its kill points are every
  lease write and release (a store `put` or `delete` of the run's `lock`,
  a lease operation) and every fenced record write (a store `put` of a
  `.json` record, the executor's journal).
* `Batch`: the batch scorer (`Batcher.score_file`) scoring a file chunk by
  chunk with the run's model, continued by scoring it again. Its kill
  points are every chunk appended to the partial output, every progress
  journal write, the swap of the finished file and its manifest.
* `Erasure`: an erasure (`erasure.Forgetting(...).execute`) forgetting the
  run's dataset from a folder holding the run and the batch predictions,
  continued by executing it again. Its kill points are the plan, every
  step appended to the erasure journal and the signed receipt.

For each subject an uninterrupted reference run is traced: a fault hook
installed by this module in the child (it wraps the product function in
that process only; no product file is changed) counts every call of every
kill point, before and after it. Every traced (operation, ordinal, moment)
is then one trial, not sampled: a fresh run whose hook sends SIGKILL to its
whole process group exactly there, then a resume in a new process, then a
byte-for-byte comparison of the final outputs with the reference.

What "final outputs" are, per subject: the files of the run directory
(`Engine`), the leaderboard read into canonical JSON, each kept stage
product unsealed into canonical JSON without its timing attributes
(`TIMINGS`), and each stage record with its digests replaced by whether
they hold: its own `digest` over its body, its `product` over the product
file beside it, and each of its `needs` against the product the needed
stage recorded; every object of the store as canonical JSON without the
wall-clock fields `CLOCKED` (`Executor`), whose fencing tokens are checked
on their own: a record the resume wrote carries one more than the lease
the killed run left (1 when it left none), a record it left untouched
keeps the reference's (CON-1); the predictions and their manifest
(`Batch`); and every file left under the erasure's folder, its receipt
included (`Erasure`). `CONTROLS` more uninterrupted runs (the controls)
are compared with the reference first. Where an output differs between
the reference and any control (timings, wall-clock stamps the product
takes outside `events.stamp`, the run's absolute path, disk and memory
readings, and the digests over them), the comparison is narrowed field by
field, never file by file: for an output that parses as JSON or JSON
lines, only the exact fields (JSON pointers) that differ in some control
are masked, they must still be present, and every other field is compared
exactly; any other output that differs (an HTML page, a binary file) is
volatile and compared for presence only. The masked fields and the
volatile outputs are listed in the document. An output a subject names
`essential` (the packaged model, the stage records and products, the
leaderboard, the trial records, the predictions, the receipt) must not be
volatile and may have masked only timing fields (a pointer whose last
token ends in one of `TIMINGS`), or the subject fails. Children freeze
`events.stamp` at `STAMP`.

No lost or duplicated record: lost records show as missing or changed
outputs. A duplicated record is one durable at the kill and written again
by the resume: a trial checkpoint whose `trial.*` event is appended again
(`Engine`), a step whose success record existed and whose handler ran again
(`Executor`), an erasure step journaled twice (`Erasure`); scored rows
written twice change the predictions' bytes (`Batch`).

A point's status is `identical`; `differs` (outputs changed, missing or
extra); `duplicated`; `refused` (the resume failed); `unreached` (the run
never reached the point: the trace was not reproducible); `failed` (the run
failed before the point); `stalled` (a child outlived `CHILD_SECONDS`).
Verdict: "pass" only when every point of every subject is `identical`,
every subject has points, and no essential output is volatile (the policy
is all-or-nothing, so no sample-size derivation applies: every point is
run); otherwise "fail". Stale or missing measurements are "not measured".

NOT measured: a kill inside a single `write` system call (a torn write;
the product's atomic writes and synced appends are trusted), machine crash
or power loss (the page cache survives SIGKILL), the lease heartbeat
thread's renewals (they share `renew_lease` with the run loop's, which are
measured), other stores than the local object store, and the modality
lifecycle's checkpoints.

POSIX only: kill -9 of a process group.

    python -m benchmarks.evidence.crashes --write
    python -m benchmarks.evidence.crashes --check
    python -m benchmarks.evidence.crashes --measure OUT --subjects executor
"""

import abc
import argparse
import collections
from concurrent import futures
import contextlib
import dataclasses
import enum
import functools
import hashlib
import importlib
import io
import json
import math
import numbers
import os
from pathlib import Path
import pickle  # nosec B403 -- for UnpicklingError only
import shutil
import signal
import subprocess  # nosec B404
import sys
import tempfile
import threading
import time
from typing import Any
from typing import ClassVar

from benchmarks import evidence
from benchmarks import reliability
from scmle.contracts import errors
from scmle.contracts import events

PATH = Path(__file__).with_suffix(".json")
ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "scmle"
MODULE = "benchmarks.evidence.crashes"
# The product packages the four subjects run, from the call to the final
# output. A change to any file in them can move a kill point or change an
# output, so the stored measurements go stale with it.
PRODUCT = (
    "api",
    "artifacts",
    "assurance",
    "contracts",
    "data",
    "evaluation",
    "explainability",
    "models",
    "pipeline",
    "preparation",
    "runtime",
    "serving",
    "storage",
    "workflow",
)
# Product modules outside `PRODUCT` the subjects run.
FILES = ("durability.py",)

# What kind of kill point an operation is; the clause names the first three.
BOUNDARY = "stage boundary"
APPEND = "journal append"
LEASE = "lease operation"
OUTPUT = "output write"
KINDS = (BOUNDARY, APPEND, LEASE, OUTPUT)
BEFORE = "before"
AFTER = "after"
MOMENTS = (BEFORE, AFTER)
STATUSES = (
    "identical",
    "differs",
    "duplicated",
    "refused",
    "unreached",
    "failed",
    "stalled",
)

# Seed and size of the generated training table: the practice project of
# `budget.py` (600 rows, a quick fit), so a trial takes seconds.
SEED = 0
ROWS = 600
# Ample, so the wall-clock budget, which depends on machine load, never
# decides which trials run: a resumed run must redo exactly the same work.
BUDGET_SECONDS = 600
# Rows scored and their chunking: six chunks, so the batch journal is
# written six times and a resume can start after any of them.
SCORE_ROWS = 600
SCORE_SEED = 1
CHUNK_ROWS = 100
# A frozen wall clock for the executor's children. The resume's clock is
# past the killed run's lease, as a process started after it expired sees.
EPOCH = 1_700_000_000.0
# Longer than a whole executor run (well under a second), so the heartbeat
# thread never renews during it and the traced call sequence is the same in
# every run; the run loop renews the lease on every iteration regardless.
LEASE_SECONDS = 60.0
# Executor record fields that carry the wall clock; every other field is
# compared byte for byte, and the fencing token (`FENCE`) by its own rule.
CLOCKED = ("started_at", "ended_at", "expires")
FENCE = "fence"
# Key-name endings of a measured time or a wall-clock stamp: the only
# fields an essential output may have masked, and the attributes left out
# of a stage product's canonical form (the product's own rule,
# `products.TIMING`, drops only `seconds`).
TIMINGS = ("seconds", "elapsed", "timestamp", "generated", "captured")
# Uninterrupted runs compared with the reference to find what varies
# between runs. A field that differs from the reference in a share q of
# runs agrees in all k controls with probability (1 - q) ** k; k = 3 keeps
# that under 0.125 for a field that varies in half the runs or more, and
# under 0.34 for one varying in a third. A missed field shows as a
# `differs` point, so too few controls fail the verdict, never pass it.
CONTROLS = 3
# The ISO stamp every child's `events.stamp` returns.
STAMP = "2026-01-01T00:00:00+00:00"
SIGNING_KEY = "crashes-evidence-key"
REASON = "kill -9 evidence"
TRIGGER = "crash-trigger"
# A child is stalled past this; the slowest (a pipeline resume) takes
# under 30 s on an unloaded laptop.
CHILD_SECONDS = 600.0
# Children at once. Each is capped at one math-library thread, so half the
# cores keeps them from contending; the trials are independent.
WORKERS = max(1, (os.cpu_count() or 2) // 2)
# Characters of a failed child's error output kept in the document, from
# its end: the exception line and the frames that raised it.
TAIL = 1500
# How long a killed child waits for its own signal, which is asynchronous.
KILL_WAIT = 60.0
# What a masked field reads as when two outputs are compared.
MASK = "<volatile>"


@dataclasses.dataclass(frozen=True)
class Point:
    """One place a run is killed.

    Attributes:
        label: The operation, as its subject names it.
        ordinal: Which call of the operation, from 1.
        moment: `before` the call starts or `after` it returned.
    """

    label: str
    ordinal: int
    moment: str

    @property
    def text(self) -> str:
        """The point as `label:ordinal:moment`."""
        return f"{self.label}:{self.ordinal}:{self.moment}"

    @classmethod
    def parse(cls, text: str) -> "Point":
        """Reads a point written as `label:ordinal:moment`.

        Args:
            text: The point's text.

        Returns:
            The point.

        Raises:
            InputError: If the text is not of that form.
        """
        parts = text.strip().split(":")
        if (
            len(parts) != len(dataclasses.fields(cls))
            or not parts[1].isdigit()
            or int(parts[1]) < 1
            or parts[2] not in MOMENTS
        ):
            raise errors.InputError(f"not a kill point: {text!r}")
        return cls(parts[0], int(parts[1]), parts[2])


@dataclasses.dataclass(frozen=True)
class Operation:
    """A product call a run can be killed at.

    Attributes:
        label: Its name in kill points.
        kind: One of `KINDS`.
        owner: `module:Class` for a method, or `module` for a function.
        attribute: The method or function name.
        suffix: The call's key or path (its first argument) must end with
            this; empty for every call.
    """

    label: str
    kind: str
    owner: str
    attribute: str
    suffix: str = ""

    def target(self) -> Any:
        """The class or module the operation is an attribute of.

        Returns:
            The imported class or module.
        """
        module, _, name = self.owner.partition(":")
        found: Any = importlib.import_module(module)
        return getattr(found, name) if name else found

    def matches(self, subject: Any) -> bool:
        """Whether a call on `subject` is this operation.

        Args:
            subject: The call's first argument (a key or path), or an
                empty string for a call without one.

        Returns:
            True when it ends with `suffix`.
        """
        return str(subject).endswith(self.suffix)


class Hook:
    """Counts kill-point calls in a child, traces them and kills at one.

    Attributes:
        trace: The file each visited point is appended to, or None.
        armed: The point to kill the process group at, or None.
        counts: Calls of each label so far.
        lock: Serialises visits from several threads.
    """

    def __init__(self, trace: Path | None, armed: Point | None) -> None:
        """Arms the hook.

        Args:
            trace: Where visited points go, or None.
            armed: Where to kill, or None.
        """
        self.trace = trace
        self.armed = armed
        self.counts: collections.Counter[str] = collections.Counter()
        self.lock = threading.RLock()

    def install(self, operations: tuple[Operation, ...]) -> None:
        """Wraps every operation's product function in this process.

        Operations on the same attribute share one wrapper, which takes the
        first whose suffix matches the call.

        Args:
            operations: The subject's operations.
        """
        grouped: dict[tuple[str, str], list[Operation]] = {}
        for operation in operations:
            key = (operation.owner, operation.attribute)
            grouped.setdefault(key, []).append(operation)
        for (owner, attribute), choices in grouped.items():
            target = choices[0].target()
            method = ":" in owner
            original = getattr(target, attribute)
            setattr(
                target,
                attribute,
                self.wrap(original, tuple(choices), method=method),
            )

    def wrap(
        self,
        original: Any,
        choices: tuple[Operation, ...],
        *,
        method: bool,
    ) -> Any:
        """A function that visits the matching point around `original`.

        Args:
            original: The product function.
            choices: The operations on it.
            method: Whether the first argument is `self`.

        Returns:
            The wrapper.
        """

        @functools.wraps(original)
        def hooked(*arguments: Any, **options: Any) -> Any:
            index = 1 if method else 0
            subject = arguments[index] if len(arguments) > index else ""
            found = next((c for c in choices if c.matches(subject)), None)
            if found is None:
                return original(*arguments, **options)
            ordinal = self.visit(found.label, BEFORE)
            result = original(*arguments, **options)
            self.visit(found.label, AFTER, ordinal)
            return result

        return hooked

    def visit(self, label: str, moment: str, ordinal: int = 0) -> int:
        """Records one moment of a call and kills there if armed.

        Args:
            label: The operation.
            moment: `before` or `after`.
            ordinal: The call's ordinal for `after`; `before` counts it.

        Returns:
            The call's ordinal.
        """
        with self.lock:
            if moment == BEFORE:
                self.counts[label] += 1
                ordinal = self.counts[label]
            point = Point(label, ordinal, moment)
            if self.trace is not None:
                with self.trace.open("a", encoding="utf-8") as handle:
                    handle.write(point.text + "\n")
            if point == self.armed:
                os.killpg(os.getpgid(0), signal.SIGKILL)
                time.sleep(KILL_WAIT)
        return ordinal


def clip(text: str) -> str:
    """The end of a child's error output, its first line kept.

    Args:
        text: The error output.

    Returns:
        The first line, then the last `TAIL` characters when it is longer.
    """
    if len(text) <= TAIL:
        return text
    return text.splitlines()[0] + "\n...\n" + text[-TAIL:]


def frozen() -> str:
    """The ISO stamp every child's `events.stamp` returns.

    Returns:
        `STAMP`.
    """
    return STAMP


def digest(content: bytes) -> str:
    """SHA-256 of some bytes.

    Args:
        content: The bytes.

    Returns:
        The hex digest.
    """
    return hashlib.sha256(content).hexdigest()


def tree(folder: Path, prefix: str) -> dict[str, bytes]:
    """The bytes of every file under a folder.

    Args:
        folder: The folder; missing means no files.
        prefix: Put before each relative path.

    Returns:
        `prefix + relative POSIX path` to the file's bytes, sorted.
    """
    if not folder.is_dir():
        return {}
    return {
        prefix + path.relative_to(folder).as_posix(): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def parse(content: bytes) -> Any:
    """An output's bytes as JSON, or as JSON lines (a list).

    Args:
        content: The bytes.

    Returns:
        The parsed value.

    Raises:
        ValueError: If the bytes are neither JSON nor JSON lines.
    """
    try:
        return json.loads(content)
    except ValueError:
        lines = content.splitlines()
        return [json.loads(line) for line in lines if line.strip()]


def drift(first: Any, second: Any, here: tuple[Any, ...] = ()) -> list[Any]:
    """The paths at which two parsed outputs differ.

    A key on one side only, a list whose length changed and a leaf whose
    value changed are each one path; equal subtrees add none.

    Args:
        first: One value.
        second: The other.
        here: The path to both.

    Returns:
        The paths (tuples of keys and list indices), in document order.
    """
    if isinstance(first, dict) and isinstance(second, dict):
        found: list[Any] = []
        for key in sorted(set(first) | set(second)):
            if key not in first or key not in second:
                found.append((*here, key))
            else:
                found += drift(first[key], second[key], (*here, key))
        return found
    if (
        isinstance(first, list)
        and isinstance(second, list)
        and len(first) == len(second)
    ):
        return [
            path
            for index, pair in enumerate(zip(first, second, strict=True))
            for path in drift(*pair, (*here, index))
        ]
    return [] if first == second else [here]


def pointer(path: tuple[Any, ...]) -> str:
    """A path as a JSON pointer (RFC 6901).

    Args:
        path: Keys and list indices.

    Returns:
        The pointer; empty for the whole document.
    """
    return "".join(
        "/" + str(token).replace("~", "~0").replace("/", "~1") for token in path
    )


def cover(value: Any, paths: set[Any], here: tuple[Any, ...] = ()) -> Any:
    """A parsed output with each masked path's value replaced by `MASK`.

    A masked path the value lacks stays absent, so it still compares.

    Args:
        value: The parsed output.
        paths: The paths to mask.
        here: The path to `value`.

    Returns:
        The masked copy.
    """
    if here in paths:
        return MASK
    if isinstance(value, dict):
        return {k: cover(v, paths, (*here, k)) for k, v in value.items()}
    if isinstance(value, list):
        return [cover(v, paths, (*here, i)) for i, v in enumerate(value)]
    return value


def canonical(content: bytes, paths: list[Any]) -> str:
    """An output as sorted JSON with its masked fields replaced.

    Args:
        content: The output's bytes.
        paths: Its masked paths.

    Returns:
        The canonical text.

    Raises:
        ValueError: If the bytes do not parse.
    """
    return json.dumps(cover(parse(content), set(paths)), sort_keys=True)


def volatility(
    reference: dict[str, bytes], *controls: dict[str, bytes]
) -> tuple[list[str], dict[str, list[Any]]]:
    """What differs between uninterrupted runs, as narrowly as it can.

    The union over the controls: an output volatile against any control is
    volatile, and the paths of each other output are those that differ
    against any control, in the order first found.

    Args:
        reference: The reference's outputs.
        *controls: Each control's outputs.

    Returns:
        The outputs compared for presence only (they do not parse, or
        differ as a whole), sorted, and for each other differing output
        the paths to mask.
    """
    volatile: set[str] = set()
    fields: dict[str, dict[Any, None]] = {}
    for control in controls:
        for name in sorted(set(reference) & set(control)):
            if reference[name] == control[name]:
                continue
            try:
                found = drift(parse(reference[name]), parse(control[name]))
            except ValueError:
                volatile.add(name)
                continue
            if () in found:
                volatile.add(name)
            else:
                fields.setdefault(name, {}).update(dict.fromkeys(found))
    return sorted(volatile), {
        name: list(paths)
        for name, paths in sorted(fields.items())
        if name not in volatile
    }


def timing(field: str) -> bool:
    """Whether a masked field is a measured time or a wall-clock stamp.

    Args:
        field: A JSON pointer.

    Returns:
        True when its last token ends in one of `TIMINGS`.
    """
    return field.rsplit("/", 1)[-1].endswith(TIMINGS)


def members(value: Any) -> Any:
    """What an object holds, as a mapping or a list where it can.

    Args:
        value: The object.

    Returns:
        A table's split form, a set sorted, a tuple or array a list, a
        dataclass or an object with attributes a mapping with its `type`,
        and anything else unchanged.
    """
    import pandas

    if isinstance(value, pandas.DataFrame):
        value = value.to_dict(orient="split")
    elif isinstance(value, set | frozenset):
        value = sorted(value, key=repr)
    elif isinstance(value, tuple | pandas.Series | pandas.Index):
        value = list(value)
    elif hasattr(value, "tolist"):
        value = value.tolist()
    elif dataclasses.is_dataclass(value):
        names = [f.name for f in dataclasses.fields(value)]
        value = {
            "type": type(value).__qualname__,
            **{name: getattr(value, name) for name in names},
        }
    elif hasattr(value, "__dict__") and not isinstance(value, dict | list):
        value = {"type": type(value).__qualname__, **vars(value)}
    return value


def plain(value: Any, seen: frozenset[int] = frozenset()) -> Any:
    """An unpickled value as JSON-ready data, its timing attributes dropped.

    Enums become their value, numbers Python numbers (a float that is not
    finite its text), other objects what they hold (`members`), with
    mapping keys and attributes ending in one of `TIMINGS` left out;
    what holds nothing JSON can carry is its `repr`.

    Args:
        value: The value.
        seen: Containers on the path to it, so a cycle ends.

    Returns:
        The data.
    """
    if isinstance(value, enum.Enum):
        value = value.value
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        value = int(value)
    elif isinstance(value, numbers.Real):
        value = float(value)
        if not math.isfinite(value):
            value = repr(value)
    if value is None or isinstance(value, int | float | str):
        return value
    if id(value) in seen:
        return "<cycle>"
    held = members(value)
    seen = seen | {id(value)}
    if isinstance(held, dict):
        return {
            str(k): plain(v, seen)
            for k, v in held.items()
            if not str(k).endswith(TIMINGS)
        }
    if isinstance(held, list):
        return [plain(v, seen) for v in held]
    return plain(held, seen) if held is not value else repr(value)


def table(path: Path, content: bytes) -> bytes:
    """A Parquet output as canonical JSON, one object per row.

    Args:
        path: The file.
        content: Its bytes, returned when it does not read.

    Returns:
        The rows as JSON, or `content`.
    """
    import pandas

    try:
        frame = pandas.read_parquet(path)
    except (OSError, ValueError):
        # handled: unreadable bytes compare as they are
        return content
    return frame.to_json(orient="records").encode()


def product(path: Path, content: bytes) -> bytes:
    """A kept stage product as canonical JSON, timing attributes dropped.

    Unseals it as `products.Kept.load` does, vouching for it (this
    measurement's own children wrote it), and unpickles its fields.

    Args:
        path: The product envelope.
        content: Its bytes, returned when it does not unseal.

    Returns:
        `{"meta": ..., "fields": ...}` as sorted JSON (`plain`), or
        `content`.
    """
    import joblib

    from scmle import envelope

    try:
        meta, payload = envelope.read(path, trusted=True)
        fields = joblib.load(io.BytesIO(payload))  # nosec B301 -- own child's
    except (
        OSError,
        EOFError,
        ValueError,
        pickle.UnpicklingError,
        errors.InputError,
        errors.ArtifactError,
    ):
        # handled: unreadable bytes compare as they are
        return content
    document = {"meta": plain(dict(meta)), "fields": plain(fields)}
    return json.dumps(document, sort_keys=True).encode()


def attest(path: Path, content: bytes) -> bytes:
    """A stage record with each digest replaced by whether it holds.

    Its `digest` by whether it hashes the rest of the record
    (`archive.digest`); its `product`, when a kept product lies beside it,
    by whether it hashes that product's payload; and each of its `needs`
    by whether it equals the `product` the needed stage's record names.
    A digest covers timings, so it cannot equal another run's, but what it
    covers is compared field by field and that it holds is compared here.

    Args:
        path: The stage record.
        content: Its bytes, returned when it does not parse.

    Returns:
        The record as sorted JSON, or `content`.
    """
    from scmle import envelope
    from scmle.pipeline import archive
    from scmle.pipeline import products

    try:
        body = json.loads(content)
        recorded = {
            other["stage"]: other.get("product")
            for other in (
                json.loads(record.read_bytes())
                for record in sorted(path.parent.glob("*.json"))
            )
        }
    except (OSError, ValueError, KeyError, TypeError):
        # handled: unreadable bytes compare as they are
        return content
    stated = body.pop("digest", None)
    body["digest"] = {"holds": archive.digest(body) == stated}
    kept = path.with_suffix(archive.SUFFIX)
    if kept.is_file():
        try:
            payload = envelope.read(kept, trusted=True)[1]
        except (OSError, errors.InputError, errors.ArtifactError):
            # handled: an unreadable product does not hold
            payload = b""
        body["product"] = {
            "holds": products.digest(payload) == body.get("product")
        }
    needs = body.get("needs")
    if isinstance(needs, dict):
        body["needs"] = {
            stage: {"holds": stage in recorded and recorded[stage] == value}
            for stage, value in needs.items()
        }
    return json.dumps(body, sort_keys=True).encode()


class Subject(abc.ABC):
    """A real use of the product that is killed and resumed.

    Attributes:
        name: The subject's key.
        operations: Its kill points.
        essential: Output name prefixes that must not be volatile.
        needs: Subjects whose reference runs it reads.
        shared: Whether every run must use the same folder (its outputs
            carry their absolute paths), so its runs go one at a time.
        sources: Reference folders of the subjects it needs, by name.
    """

    name: ClassVar[str]
    operations: ClassVar[tuple[Operation, ...]]
    essential: ClassVar[tuple[str, ...]]
    needs: ClassVar[tuple[str, ...]] = ()
    shared: ClassVar[bool] = False

    def __init__(self) -> None:
        """Starts with no sources."""
        self.sources: dict[str, Path] = {}

    def place(self, scratch: Path, tag: str) -> Path:
        """The folder of one run.

        Args:
            scratch: The measurement's scratch folder.
            tag: The run's name, unique within the subject.

        Returns:
            `scratch/<name>/<tag>`.
        """
        return scratch / self.name / tag

    def prepare(self, folder: Path) -> None:
        """Lays out a run's inputs in a fresh folder.

        Args:
            folder: The run's folder; replaced if it exists.
        """
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)

    @abc.abstractmethod
    def start(self, folder: Path) -> None:
        """Runs the subject from the start (in a child).

        Args:
            folder: The run's folder.
        """

    @abc.abstractmethod
    def resume(self, folder: Path) -> None:
        """Continues an interrupted run as a user would (in a child).

        Args:
            folder: The run's folder.
        """

    @abc.abstractmethod
    def contents(self, folder: Path) -> dict[str, bytes]:
        """The final outputs, by name.

        Args:
            folder: The run's folder.

        Returns:
            Output name to its bytes.
        """

    def outputs(self, folder: Path) -> dict[str, str]:
        """The final outputs, by name, as digests of their bytes.

        Args:
            folder: The run's folder.

        Returns:
            Output name to SHA-256.
        """
        return {
            name: digest(content)
            for name, content in self.contents(folder).items()
        }

    def ledger(self, folder: Path) -> collections.Counter[str]:
        """The durable records and how often each was written.

        Args:
            folder: The run's folder.

        Returns:
            Record to times written; empty when the outputs' bytes already
            show a duplicate.
        """
        del folder
        return collections.Counter()

    def state(self, folder: Path) -> dict[str, Any]:
        """What the outputs leave out but `audit` checks, as it stands now.

        Args:
            folder: The run's folder.

        Returns:
            Empty: most subjects compare everything in their outputs.
        """
        del folder
        return {}

    def audit(
        self,
        final: dict[str, Any],
        kept: dict[str, Any],
        reference: dict[str, Any],
    ) -> list[str]:
        """What a resume got wrong outside the compared outputs.

        Args:
            final: `state` after the resume.
            kept: `state` the killed run left.
            reference: `state` of the reference run.

        Returns:
            One entry per fault; empty for most subjects.
        """
        del final, kept, reference
        return []


class Engine(Subject):
    """The pipeline engine: a tabular run, resumed with `Project.resume`."""

    name = "engine"
    operations = (
        Operation(
            "stage", BOUNDARY, "scmle.pipeline.archive:Directory", "record"
        ),
        Operation(
            "marker", BOUNDARY, "scmle.pipeline.archive:Directory", "finish"
        ),
        Operation("event", APPEND, "scmle.contracts.events:Log", "emit"),
        Operation(
            "trial", APPEND, "scmle.pipeline.journal:Journal", "transcribe"
        ),
        Operation(
            "custody",
            APPEND,
            "scmle.assurance.custody.journal:Journal",
            "append",
        ),
    )
    essential = (
        "run/model/",
        "run/stages/",
        "run/leaderboard.parquet",
        "run/trials/",
    )

    def frame(self) -> Any:
        """The training table.

        Returns:
            The generated binary table.
        """
        return reliability.binary_data(ROWS, SEED)

    def run(self, folder: Path) -> Path | None:
        """The run directory, once the run has made one.

        Args:
            folder: The run's folder.

        Returns:
            The only `runs/run_*` directory, or None.
        """
        found = sorted((folder / "runs").glob("run_*"))
        return found[-1] if found else None

    def start(self, folder: Path) -> None:
        """Trains the quick project on the generated table.

        Args:
            folder: The run's folder.
        """
        reliability.fast_project(
            self.frame(),
            "y",
            folder / "runs",
            time_budget=BUDGET_SECONDS,
            seed=SEED,
        ).run()

    def resume(self, folder: Path) -> None:
        """Resumes the run; starts again if it made no directory yet.

        A run whose completion marker exists has nothing left to resume.

        Args:
            folder: The run's folder.
        """
        from scmle.api import project
        from scmle.pipeline import archive

        run = self.run(folder)
        if run is None:
            self.start(folder)
        elif not (run / archive.MARKER).is_file():
            project.Project.resume(run, data=self.frame())

    def contents(self, folder: Path) -> dict[str, bytes]:
        """Every file of the run directory, its records made comparable.

        The leaderboard is read into canonical JSON, each kept product
        unsealed into canonical JSON (`product`) and each stage record's
        digests replaced by whether they hold (`attest`), so the fields of
        all three compare one by one.

        Args:
            folder: The run's folder.

        Returns:
            `run/<path>` to its bytes; empty without a run directory.
        """
        from scmle.pipeline import archive

        run = self.run(folder)
        if run is None:
            return {}
        found = tree(run, "run/")
        stages = run / archive.FOLDER
        for name in found:
            path = run / name.removeprefix("run/")
            if path.suffix == ".parquet":
                found[name] = table(path, found[name])
            elif path.suffix == archive.SUFFIX:
                found[name] = product(path, found[name])
            elif path.parent == stages and path.suffix == ".json":
                found[name] = attest(path, found[name])
        return found

    def ledger(self, folder: Path) -> collections.Counter[str]:
        """Each trial checkpoint's `trial.*` events in the event log.

        Args:
            folder: The run's folder.

        Returns:
            Trial checkpoint path to the events announcing it.
        """
        run = self.run(folder)
        found: collections.Counter[str] = collections.Counter()
        log = None if run is None else run / "events.jsonl"
        if log is None or not log.is_file():
            return found
        for line in log.read_text(encoding="utf-8").splitlines():
            with contextlib.suppress(ValueError):
                event = json.loads(line)
                if str(event.get("kind", "")).startswith("trial."):
                    found.update(event.get("artifacts", [])[:1])
        return found


class Handler:
    """A workflow step's handler: logs its call and adds up its upstream.

    Attributes:
        step: The step's name.
        calls: The file each call is appended to, synced.
    """

    def __init__(self, step: str, calls: Path) -> None:
        """Names the handler's step and call log.

        Args:
            step: The step's name.
            calls: The call log.
        """
        self.step = step
        self.calls = calls

    def __call__(self, context: Any) -> dict[str, Any]:
        """Logs the call, then returns the step's deterministic outputs.

        Args:
            context: The executor's step context.

        Returns:
            The step's name and one more than its upstream's totals.
        """
        with self.calls.open("a", encoding="utf-8") as handle:
            handle.write(self.step + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        total = 1 + sum(int(up["total"]) for up in context.upstream.values())
        return {"step": self.step, "total": total}


class Executor(Subject):
    """The workflow executor: a four-step run under a lease."""

    name = "executor"
    STORE = "scmle.storage.objectstore:LocalObjectStore"
    operations = (
        Operation("lease", LEASE, STORE, "put", "/lock"),
        Operation("release", LEASE, STORE, "delete", "/lock"),
        Operation("record", APPEND, STORE, "put", ".json"),
    )
    essential = ("workflows/crashes/runs/",)
    STEPS = ("ingest", "train", "evaluate", "score")

    def runner(self, folder: Path, clock: float) -> Any:
        """The executor over the run's store, its clock frozen.

        Args:
            folder: The run's folder.
            clock: The epoch seconds its clock returns.

        Returns:
            The executor.
        """
        from scmle.storage import objectstore
        from scmle.workflow import definition
        from scmle.workflow import executor

        kinds = (
            definition.StepKind.INGEST,
            definition.StepKind.TRAIN,
            definition.StepKind.EVALUATE,
            definition.StepKind.BATCH_PREDICT,
        )
        steps = tuple(
            definition.Step(
                name, kind, (self.STEPS[index - 1],) if index else ()
            )
            for index, (name, kind) in enumerate(
                zip(self.STEPS, kinds, strict=True)
            )
        )
        flow = definition.Workflow("crashes", "1", steps, max_parallel=1)
        calls = folder / "calls.txt"
        return executor.Executor(
            flow,
            objectstore.LocalObjectStore(folder / "store"),
            {step: Handler(step, calls) for step in self.STEPS},
            clock=lambda: clock,
            lease_seconds=LEASE_SECONDS,
        )

    def start(self, folder: Path) -> None:
        """Starts the trigger's run at the frozen clock.

        Args:
            folder: The run's folder.
        """
        self.runner(folder, EPOCH).start(TRIGGER)

    def resume(self, folder: Path) -> None:
        """Starts the same trigger once the killed run's lease expired.

        Args:
            folder: The run's folder.
        """
        self.runner(folder, EPOCH + LEASE_SECONDS + 1).start(TRIGGER)

    def records(self, folder: Path) -> dict[str, dict[str, Any]]:
        """Every object of the store, parsed.

        Args:
            folder: The run's folder.

        Returns:
            Key to record, sorted by key.
        """
        from scmle.storage import objectstore

        root = folder / "store"
        if not root.is_dir():
            return {}
        store = objectstore.LocalObjectStore(root)
        return {
            meta.key: json.loads(store.get(meta.key)[0])
            for meta in store.list_objects()
        }

    def contents(self, folder: Path) -> dict[str, bytes]:
        """Every record as canonical JSON without `CLOCKED` and `FENCE`.

        The fencing token is checked on its own (`audit`).

        Args:
            folder: The run's folder.

        Returns:
            Key to the record's canonical bytes.
        """
        left = {*CLOCKED, FENCE}
        return {
            key: json.dumps(
                {k: v for k, v in record.items() if k not in left},
                sort_keys=True,
            ).encode()
            for key, record in self.records(folder).items()
        }

    def state(self, folder: Path) -> dict[str, Any]:
        """Every record of the store, whole.

        Args:
            folder: The run's folder.

        Returns:
            Key to record.
        """
        return self.records(folder)

    def audit(
        self,
        final: dict[str, Any],
        kept: dict[str, Any],
        reference: dict[str, Any],
    ) -> list[str]:
        """Records whose fencing token breaks the lease protocol (CON-1).

        A record the resume wrote (new, or changed since the kill) must
        carry the resume's lease: one more than the fence of the `lock`
        the killed run left, or 1 when it left none. A record it left
        untouched must keep the reference's fence.

        Args:
            final: The records after the resume.
            kept: The records the killed run left.
            reference: The reference's records.

        Returns:
            `fence KEY` per offending record, sorted by key.
        """
        issues = []
        for key, record in sorted(final.items()):
            head, runs, tail = key.partition("/runs/")
            lock = kept.get(f"{head}{runs}{tail.split('/')[0]}/lock")
            if kept.get(key) != record:
                expected = 1 if lock is None else int(lock.get(FENCE, 0)) + 1
            elif key in reference:
                expected = reference[key].get(FENCE)
            else:
                continue
            if record.get(FENCE) != expected:
                issues.append(f"fence {key}")
        return issues

    def ledger(self, folder: Path) -> collections.Counter[str]:
        """Handler calls of the steps whose success is recorded.

        Args:
            folder: The run's folder.

        Returns:
            Step name to its handler's calls, for succeeded steps only.
        """
        done = {
            record.get("step")
            for key, record in self.records(folder).items()
            if "/steps/" in key and record.get("state") == "succeeded"
        }
        calls = folder / "calls.txt"
        lines = (
            calls.read_text(encoding="utf-8").split() if calls.is_file() else []
        )
        return collections.Counter(line for line in lines if line in done)


class Batch(Subject):
    """The batch scorer: a file scored chunk by chunk with the run's model."""

    name = "batch"
    SERVER = "scmle.serving.server.batch"
    operations = (
        Operation("chunk", APPEND, f"{SERVER}:Batcher", "append"),
        Operation(
            "journal",
            APPEND,
            "scmle.durability",
            "write_atomically",
            ".progress.json",
        ),
        Operation(
            "manifest",
            OUTPUT,
            "scmle.durability",
            "write_atomically",
            ".manifest.json",
        ),
        Operation("swap", OUTPUT, "scmle.durability", "swap"),
    )
    essential = ("scores/predictions.csv",)
    needs = ("engine",)

    def prepare(self, folder: Path) -> None:
        """Copies the reference run's model and writes the rows to score.

        Args:
            folder: The run's folder.

        Raises:
            ArtifactError: If the engine's reference run has no model.
        """
        super().prepare(folder)
        run = Engine().run(self.sources["engine"])
        if run is None or not (run / "model").is_dir():
            raise errors.ArtifactError("the engine reference has no model")
        shutil.copytree(run / "model", folder / "model")
        rows = reliability.binary_data(SCORE_ROWS, SCORE_SEED)
        rows.drop(columns="y").to_csv(folder / "score.csv", index=False)

    def score(self, folder: Path) -> None:
        """Scores the rows, continuing an interrupted scoring.

        Args:
            folder: The run's folder.
        """
        from scmle.artifacts import bundle
        from scmle.serving import server

        model = bundle.Bundle.load(folder / "model", trusted=True)
        server.Batcher(model).score_file(
            folder / "score.csv",
            folder / "scores" / "predictions.csv",
            CHUNK_ROWS,
        )

    def start(self, folder: Path) -> None:
        """Scores the rows.

        Args:
            folder: The run's folder.
        """
        self.score(folder)

    def resume(self, folder: Path) -> None:
        """Scores the rows again, as the CLI's rerun does.

        Args:
            folder: The run's folder.
        """
        self.score(folder)

    def contents(self, folder: Path) -> dict[str, bytes]:
        """Every file the scoring left: predictions and manifest.

        Args:
            folder: The run's folder.

        Returns:
            `scores/<file>` to its bytes.
        """
        return tree(folder / "scores", "scores/")


class Erasure(Subject):
    """An erasure forgetting the run's dataset everywhere it is held."""

    name = "erasure"
    RECEIPTS = "scmle.api.erasure.receipts"
    operations = (
        Operation("plan", APPEND, f"{RECEIPTS}:Journal", "record"),
        Operation("step", APPEND, f"{RECEIPTS}:Journal", "finish"),
        Operation("receipt", OUTPUT, f"{RECEIPTS}:Receipt", "write"),
    )
    essential = ("receipts/",)
    needs = ("engine", "batch")
    shared = True

    def place(self, scratch: Path, tag: str) -> Path:
        """The one folder every erasure run uses.

        The plan and the receipt name absolute paths and the erasure's id
        is a digest of them, so runs compare only in the same folder.

        Args:
            scratch: The measurement's scratch folder.
            tag: Unused: every run shares the folder.

        Returns:
            `scratch/erasure/site`.
        """
        del tag
        return scratch / self.name / "site"

    def prepare(self, folder: Path) -> None:
        """Lays out the run, its predictions and the dataset's digests.

        Args:
            folder: The run's folder.

        Raises:
            ArtifactError: If the engine's reference run has no dataset
                manifest.
        """
        from scmle.api.erasure import holdings

        super().prepare(folder)
        shutil.copytree(self.sources["engine"] / "runs", folder / "runs")
        shutil.copytree(self.sources["batch"] / "scores", folder / "scores")
        run = Engine().run(folder)
        manifest = None if run is None else run / holdings.DATASET
        if manifest is None or not manifest.is_file():
            raise errors.ArtifactError("the engine reference has no dataset")
        found = sorted(holdings.identities(manifest))
        (folder / "digests.json").write_text(
            json.dumps(found), encoding="utf-8"
        )

    def execute(self, folder: Path) -> None:
        """Forgets the dataset under the runs and the predictions.

        Args:
            folder: The run's folder.
        """
        from scmle.api import erasure

        digests = json.loads(
            (folder / "digests.json").read_text(encoding="utf-8")
        )
        erasure.Forgetting(
            [folder / "runs", folder / "scores"],
            erasure.Dataset(digests),
            folder / "receipts",
            REASON,
        ).execute(SIGNING_KEY)

    def start(self, folder: Path) -> None:
        """Executes the erasure.

        Args:
            folder: The run's folder.
        """
        self.execute(folder)

    def resume(self, folder: Path) -> None:
        """Executes the same erasure again, which resumes it.

        Args:
            folder: The run's folder.
        """
        self.execute(folder)

    def contents(self, folder: Path) -> dict[str, bytes]:
        """Every file left in the folder, the receipt and journal included.

        Args:
            folder: The run's folder.

        Returns:
            Relative path to its bytes.
        """
        return tree(folder, "")

    def ledger(self, folder: Path) -> collections.Counter[str]:
        """How often each step was journaled as finished.

        Args:
            folder: The run's folder.

        Returns:
            `<erasure>/<step index>` to its journal lines.
        """
        found: collections.Counter[str] = collections.Counter()
        for path in sorted((folder / "receipts").glob("*/journal.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                with contextlib.suppress(ValueError, KeyError, TypeError):
                    found[f"{path.parent.name}/{json.loads(line)['step']}"] += 1
        return found


SUBJECTS: dict[str, type[Subject]] = {
    kind.name: kind for kind in (Engine, Executor, Batch, Erasure)
}


def child(
    name: str,
    role: str,
    folder: Path,
    armed: Point | None,
    trace: Path | None,
) -> int:
    """Runs one subject's role inside a child process.

    Args:
        name: The subject.
        role: `start` or `resume`.
        folder: The run's folder.
        armed: Where to kill this process group, or None.
        trace: Where to trace the kill points, or None.

    Returns:
        0 once the role finished.

    Raises:
        InputError: For an unknown subject or role.
    """
    if name not in SUBJECTS or role not in {"start", "resume"}:
        raise errors.InputError(f"unknown child {name} {role}")
    subject = SUBJECTS[name]()
    events.stamp = frozen
    if armed is not None or trace is not None:
        Hook(trace, armed).install(subject.operations)
    getattr(subject, role)(folder)
    return 0


@dataclasses.dataclass(frozen=True)
class Exit:
    """How a child ended.

    Attributes:
        code: Its return code; `-SIGKILL` when the hook killed it.
        error: The tail of its error output.
        stalled: Whether it outlived `CHILD_SECONDS` and was killed.
        seconds: Its wall time.
    """

    code: int
    error: str
    stalled: bool
    seconds: float

    @property
    def killed(self) -> bool:
        """Whether SIGKILL ended it (not a stall)."""
        return self.code == -signal.SIGKILL and not self.stalled


def spawn(
    name: str,
    role: str,
    folder: Path,
    armed: Point | None = None,
    trace: Path | None = None,
) -> Exit:
    """Runs one role of a subject in its own process group.

    Args:
        name: The subject.
        role: `start` or `resume`.
        folder: The run's folder.
        armed: Where to kill it, or None.
        trace: Where to trace, or None.

    Returns:
        How the child ended.
    """
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), environment.get("PYTHONPATH", "")]
    )
    environment["OMP_NUM_THREADS"] = "1"
    command = [sys.executable, "-m", MODULE, "--child", name, role]
    command += [str(folder)]
    if armed is not None:
        command += ["--kill", armed.text]
    if trace is not None:
        command += ["--trace", str(trace)]
    started = time.monotonic()
    process = subprocess.Popen(  # nosec B603 -- this module, fixed arguments
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    stalled = False
    try:
        error = process.communicate(timeout=CHILD_SECONDS)[1]
    except subprocess.TimeoutExpired:
        stalled = True
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        error = process.communicate()[1]
    return Exit(
        process.returncode,
        clip(error or ""),
        stalled,
        round(time.monotonic() - started, 3),
    )


def points(trace: Path) -> list[Point]:
    """The kill points a traced run visited, in order, each once.

    Args:
        trace: The trace file.

    Returns:
        The points.
    """
    if not trace.is_file():
        return []
    found = [
        Point.parse(line)
        for line in trace.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return list(dict.fromkeys(found))


def same(first: bytes, second: bytes, paths: list[Any] | None) -> bool:
    """Whether two versions of an output agree outside their masked fields.

    Args:
        first: One version's bytes.
        second: The other's.
        paths: The output's masked paths; None to compare the bytes.

    Returns:
        True when the bytes are equal, or both parse and agree once masked.
    """
    if first == second:
        return True
    if paths is None:
        return False
    try:
        return canonical(first, paths) == canonical(second, paths)
    except ValueError:
        return False


def differences(
    reference: dict[str, bytes],
    found: dict[str, bytes],
    volatile: list[str],
    fields: dict[str, list[Any]],
) -> list[str]:
    """How a resumed run's outputs differ from the reference's.

    Args:
        reference: The reference's outputs.
        found: The resumed run's outputs.
        volatile: Outputs compared only for presence.
        fields: Outputs compared with these paths masked.

    Returns:
        `missing NAME`, `extra NAME` and `changed NAME`, sorted by name.
    """
    names = sorted(set(reference) | set(found))
    issues = []
    for name in names:
        if name not in found:
            issues.append(f"missing {name}")
        elif name not in reference:
            issues.append(f"extra {name}")
        elif name not in volatile and not same(
            reference[name], found[name], fields.get(name)
        ):
            issues.append(f"changed {name}")
    return issues


@dataclasses.dataclass
class Baseline:
    """A subject's uninterrupted runs.

    Attributes:
        folder: The reference run's folder.
        outputs: The reference's outputs, by name.
        volatile: Outputs that differ as a whole between two uninterrupted
            runs, compared for presence only.
        fields: For the other outputs that differ, the paths that do.
        points: The kill points the reference visited.
        seconds: Wall time of the reference run.
        error: Why a baseline run failed, empty when every run finished.
        state: The reference's `Subject.state`.
    """

    folder: Path
    outputs: dict[str, bytes]
    volatile: list[str]
    fields: dict[str, list[Any]]
    points: list[Point]
    seconds: float
    error: str = ""
    state: dict[str, Any] = dataclasses.field(default_factory=dict)


def baseline(subject: Subject, scratch: Path) -> Baseline:
    """Runs the traced reference and `CONTROLS` controls.

    The controls run beside the reference, or after it for a shared
    subject, whose runs all use one folder.

    Args:
        subject: The subject.
        scratch: The measurement's scratch folder.

    Returns:
        The reference's outputs, kill points and volatile outputs.
    """
    trace = scratch / subject.name / "reference.trace"
    folder = subject.place(scratch, "reference")
    subject.prepare(folder)
    trace.unlink(missing_ok=True)

    def control(index: int) -> tuple[Exit, dict[str, bytes]]:
        place = subject.place(scratch, f"control-{index}")
        subject.prepare(place)
        ended = spawn(subject.name, "start", place)
        return ended, subject.contents(place)

    indices = range(1, CONTROLS + 1)
    with futures.ThreadPoolExecutor(max_workers=CONTROLS) as pool:
        # a shared subject's runs use one folder, so its controls wait
        ran = (
            [] if subject.shared else [pool.submit(control, i) for i in indices]
        )
        first = spawn(subject.name, "start", folder, trace=trace)
        reference = subject.contents(folder)
        state = subject.state(folder)
        kept = folder
        if subject.shared:
            kept = scratch / subject.name / "kept"
            shutil.rmtree(kept, ignore_errors=True)
            shutil.copytree(folder, kept, symlinks=True)
            controls = [control(i) for i in indices]
        else:
            controls = [job.result() for job in ran]
    ends = [first, *(ended for ended, _ in controls)]
    failures = [e.error or f"exit {e.code}" for e in ends if e.code]
    volatile, fields = volatility(reference, *(found for _, found in controls))
    return Baseline(
        kept,
        reference,
        volatile,
        fields,
        points(trace) if not failures else [],
        first.seconds,
        clip("; ".join(failures)),
        state,
    )


def trial(
    subject: Subject, point: Point, scratch: Path, base: Baseline
) -> dict[str, Any]:
    """Kills one run at `point`, resumes it and compares its outputs.

    Args:
        subject: The subject.
        point: Where to kill.
        scratch: The measurement's scratch folder.
        base: The subject's baseline.

    Returns:
        The point, its kind, its status, the differing outputs, the
        duplicated records, the error of a failed child and the seconds.
    """
    tag = f"{point.label}-{point.ordinal}-{point.moment}"
    folder = subject.place(scratch, tag)
    subject.prepare(folder)
    kinds = {o.label: o.kind for o in subject.operations}
    found: dict[str, Any] = {
        "point": point.text,
        "kind": kinds.get(point.label, ""),
        "differing": [],
        "duplicates": [],
        "error": "",
    }
    first = spawn(subject.name, "start", folder, armed=point)
    found["seconds"] = first.seconds
    if not first.killed:
        status = "stalled" if first.stalled else "failed"
        found.update(
            status="unreached" if first.code == 0 else status,
            error=first.error,
        )
        return found
    before = subject.ledger(folder)
    kept = subject.state(folder)
    second = spawn(subject.name, "resume", folder)
    found["seconds"] = round(first.seconds + second.seconds, 3)
    if second.code != 0:
        status = "stalled" if second.stalled else "refused"
        found.update(status=status, error=second.error)
        return found
    after = subject.ledger(folder)
    found["differing"] = differences(
        base.outputs, subject.contents(folder), base.volatile, base.fields
    ) + subject.audit(subject.state(folder), kept, base.state)
    found["duplicates"] = sorted(
        record for record, count in before.items() if after[record] > count
    )
    found["status"] = (
        "differs"
        if found["differing"]
        else "duplicated"
        if found["duplicates"]
        else "identical"
    )
    if not subject.shared:
        shutil.rmtree(folder, ignore_errors=True)
    return found


def series(
    subject: Subject, chosen: list[Point], scratch: Path, base: Baseline
) -> list[dict[str, Any]]:
    """Runs a subject's trials one after another.

    Args:
        subject: The subject.
        chosen: Its kill points.
        scratch: The measurement's scratch folder.
        base: Its baseline.

    Returns:
        One result per point, in order.
    """
    return [trial(subject, point, scratch, base) for point in chosen]


def settings() -> dict[str, Any]:
    """The fixed design of the measurement.

    Returns:
        Every constant that changes what is measured, and each subject's
        kill-point operations.
    """
    return {
        "seed": SEED,
        "rows": ROWS,
        "budget_seconds": BUDGET_SECONDS,
        "score_rows": SCORE_ROWS,
        "score_seed": SCORE_SEED,
        "chunk_rows": CHUNK_ROWS,
        "lease_seconds": LEASE_SECONDS,
        "clocked": list(CLOCKED),
        "fence": FENCE,
        "timings": list(TIMINGS),
        "controls": CONTROLS,
        "stamp": STAMP,
        "child_seconds": CHILD_SECONDS,
        "operations": {
            name: [dataclasses.asdict(o) for o in kind.operations]
            for name, kind in SUBJECTS.items()
        },
    }


def current() -> str:
    """The fingerprint of the measured code.

    Returns:
        SHA-256 over this module, the product subpackages in `PRODUCT` and
        the modules in `FILES`.
    """
    combined = hashlib.sha256()
    combined.update(Path(__file__).read_bytes())
    combined.update(evidence.source(PRODUCT, PACKAGE).encode())
    for name in FILES:
        combined.update((PACKAGE / name).read_bytes())
    return combined.hexdigest()


def closure(names: list[str]) -> list[str]:
    """The subjects to run: the chosen ones and those they need, in order.

    Args:
        names: The chosen subjects.

    Returns:
        Subject names in `SUBJECTS` order.

    Raises:
        InputError: For an unknown subject.
    """
    unknown = sorted(set(names) - set(SUBJECTS))
    if unknown:
        raise errors.InputError(f"unknown subjects {unknown}")
    wanted = set(names)
    for name in reversed(list(SUBJECTS)):
        if name in wanted:
            wanted |= set(SUBJECTS[name].needs)
    return [name for name in SUBJECTS if name in wanted]


def measure(
    scratch: Path, names: list[str] | None = None, workers: int = WORKERS
) -> dict[str, Any]:
    """Runs every subject's baseline, then every kill point's trial.

    Baselines run by levels (a subject after those it needs) on their own
    threads; a chosen subject's trials are queued on `workers` children as
    soon as its baseline is done, so they overlap the later baselines. A
    shared subject's trials run as one series.

    Args:
        scratch: An empty scratch folder.
        names: The subjects to measure; None for all.
        workers: Trials at once.

    Returns:
        The JSON-ready document.
    """
    chosen = list(SUBJECTS) if names is None else names
    order = closure(chosen)
    subjects = {name: SUBJECTS[name]() for name in order}
    bases: dict[str, Baseline] = {}
    jobs: dict[str, list[futures.Future[list[dict[str, Any]]]]] = {}
    with (
        futures.ThreadPoolExecutor(max_workers=len(order)) as leads,
        futures.ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        remaining = list(order)
        while remaining:
            level = [
                n
                for n in remaining
                if all(need in bases for need in subjects[n].needs)
            ]
            for name in level:
                subjects[name].sources = {
                    need: bases[need].folder for need in subjects[name].needs
                }
            ran = {
                n: leads.submit(baseline, subjects[n], scratch) for n in level
            }
            for name, job in ran.items():
                bases[name] = base = job.result()
                if name not in chosen:
                    continue
                shared = subjects[name].shared
                groups = [base.points] if shared else [[p] for p in base.points]
                jobs[name] = [
                    pool.submit(series, subjects[name], group, scratch, base)
                    for group in groups
                    if group
                ]
            remaining = [n for n in remaining if n not in bases]
        results = {
            name: [row for job in queued for row in job.result()]
            for name, queued in jobs.items()
        }
    return {
        "settings": settings(),
        "code": current(),
        "subjects": {
            name: {
                "essential": list(subjects[name].essential),
                "volatile": bases[name].volatile,
                "fields": {
                    output: [pointer(path) for path in paths]
                    for output, paths in bases[name].fields.items()
                },
                "outputs": len(bases[name].outputs),
                "reference_seconds": bases[name].seconds,
                "error": bases[name].error,
                "points": results[name],
            }
            for name in chosen
        },
    }


def read(document: dict[str, Any] | None) -> dict[str, Any]:
    """The measurements document, read from `PATH` when not given.

    Args:
        document: The document, or None.

    Returns:
        The document.

    Raises:
        InputError: If it is missing, unreadable or lacks its subjects.
    """
    if document is None:
        try:
            document = json.loads(PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise errors.InputError(f"no measurements: {error}") from error
    if not isinstance(document, dict) or not isinstance(
        document.get("subjects"), dict
    ):
        raise errors.InputError("the document has no subjects")
    return document


def problems(document: dict[str, Any] | None = None) -> list[str]:
    """Why the stored measurements are not for the current code and design.

    Args:
        document: The document; read from `PATH` when None.

    Returns:
        Messages for a missing or malformed document, other code, another
        design or a subject not measured; empty when current.
    """
    try:
        found = read(document)
    except errors.InputError as error:
        return [f"crashes: {error}"]
    issues = []
    if found.get("code") != current():
        issues.append("crashes: measurements are for other code")
    if found.get("settings") != settings():
        issues.append("crashes: measurements are for another design")
    missing = sorted(set(SUBJECTS) - set(found["subjects"]))
    if missing:
        issues.append(f"crashes: subjects not measured: {missing}")
    return issues


def faults(name: str, record: dict[str, Any]) -> list[str]:
    """What fails one subject's part of the clause.

    Args:
        name: The subject.
        record: Its record in the document.

    Returns:
        A message per failing point, per essential output found volatile,
        per field of an essential output masked that is not a timing
        (`timing`), for a failed baseline and for a subject with no point.
    """
    issues = []
    if record.get("error"):
        issues.append(f"{name}: baseline failed: {record['error'][:200]}")
    if not record["points"]:
        issues.append(f"{name}: no kill point")
    essential = tuple(record["essential"])
    issues += [
        f"{name}: essential output {v} is volatile"
        for v in record["volatile"]
        if v.startswith(essential)
    ]
    issues += [
        f"{name}: essential output {output} masks {field}"
        for output, fields in record.get("fields", {}).items()
        if output.startswith(essential)
        for field in fields
        if not timing(field)
    ]
    for row in record["points"]:
        if row["status"] != "identical":
            detail = row["differing"][:3] or row["duplicates"][:3]
            text = "; ".join(detail) or row["error"][-200:]
            issues.append(f"{name} {row['point']}: {row['status']} {text}")
    return issues


def tally(found: dict[str, Any]) -> dict[str, collections.Counter[str]]:
    """Points by kind and by status, over every subject.

    Args:
        found: The document.

    Returns:
        `kinds` and `statuses`, each a counter.
    """
    rows = [r for s in found["subjects"].values() for r in s["points"]]
    return {
        "kinds": collections.Counter(r["kind"] for r in rows),
        "statuses": collections.Counter(r["status"] for r in rows),
    }


def summary(document: dict[str, Any] | None = None) -> tuple[str, str]:
    """The scorecard's "Value now" text and verdict for this clause of C7.

    Args:
        document: The document; read from `PATH` when None.

    Returns:
        The value text and "pass", "fail" or "not measured".
    """
    issues = problems(document)
    if issues:
        return "; ".join(issues), "not measured"
    found = read(document)
    counts = tally(found)
    total = sum(counts["statuses"].values())
    kinds = ", ".join(f"{counts['kinds'][k]} {k}" for k in KINDS)
    statuses = ", ".join(
        f"{counts['statuses'][s]} {s}"
        for s in STATUSES
        if counts["statuses"][s]
    )
    records = found["subjects"].values()
    volatile = sum(len(s["volatile"]) for s in records)
    masked = [p for s in records for p in s.get("fields", {}).values()]
    text = (
        f"kill -9 at {total} points over {len(found['subjects'])} subjects "
        f"({kinds}): {statuses or 'none'}; {volatile} volatile outputs "
        f"compared for presence only, {sum(map(len, masked))} volatile "
        f"fields masked in {len(masked)} outputs"
    )
    failing = [
        issue
        for name, record in found["subjects"].items()
        for issue in faults(name, record)
    ]
    if failing:
        return text + "; " + "; ".join(failing[:5]), "fail"
    return text, "pass"


def main(arguments: list[str] | None = None) -> int:
    """Writes (`--write`), measures (`--measure`) or checks (`--check`).

    `--child SUBJECT ROLE FOLDER [--kill POINT] [--trace FILE]` is the
    child process `spawn` starts.

    Args:
        arguments: Command-line arguments; None reads `sys.argv`.

    Returns:
        0 for `--write`, `--measure` and `--child`; for `--check`, 0 when
        the stored measurements are current and 1 otherwise.

    Raises:
        InputError: For `--subjects` with `--write` or a malformed child.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--measure", type=Path, metavar="OUT")
    mode.add_argument("--child", nargs=3, metavar=("SUBJECT", "ROLE", "DIR"))
    parser.add_argument("--subjects", nargs="+", choices=list(SUBJECTS))
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--kill", type=Point.parse)
    parser.add_argument("--trace", type=Path)
    options = parser.parse_args(arguments)
    if options.child:
        name, role, folder = options.child
        return child(name, role, Path(folder), options.kill, options.trace)
    if options.check:
        issues = problems()
        print(json.dumps({"ok": not issues, "problems": issues}, indent=2))
        return 0 if not issues else 1
    if options.write and options.subjects:
        raise errors.InputError("--write measures every subject")
    if options.workers < 1:
        raise errors.InputError("--workers must be positive")
    target = PATH if options.write else options.measure
    with tempfile.TemporaryDirectory(prefix="scmle-crashes-") as scratch:
        document = measure(Path(scratch), options.subjects, options.workers)
    target.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    value, verdict = summary(document) if options.write else ("", "")
    if not options.write:
        counts = tally(document)
        value = json.dumps({k: dict(v) for k, v in counts.items()})
    print(json.dumps({"value": value, "verdict": verdict}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
