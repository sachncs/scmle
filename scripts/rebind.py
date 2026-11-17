"""Re-points stored code hashes after a proven rename-only move.

A module move (`scripts/move.py`, register phase 3, DES-13) changes the
hashes that certificates and evidence records store for the code they
measured: a whole-module hash covers the rewritten import lines, and a
qualified-name digest covers the module path. The measured code is the same
apart from those paths, so its verdicts still hold; re-measuring every
family at every move would take hours each time.

`record` runs every certificate and evidence `--check` with `hashlib`
instrumented and writes, per check, the hex digests in the order they were
computed. Run it before the move; every check must pass. `apply` runs after
the move: it rewrites the module paths the stores name (the mutation
evidence keys files by path), records again, and pairs each old digest with
the new one computed at the same point. The same checks over code that
differs only in module paths compute the same sequence.

`apply` rewrites the stores only when every condition holds:

1. the "before" recording passed every check, so each stored value was
   current;
2. `move.py <step> --verify` passes, so the change is rename-only;
3. both recordings have the same shape (the same checks, each with the same
   number of digests);
4. a stored digest maps to one new value (truncated prefixes included);
5. after the rewrite every check passes again.

Each re-pointed value is appended to `benchmarks/qualification/rebinds.jsonl`
with the step, file, old and new value, so the custody of the evidence can
be audited. A wrong pairing can only make a check fail, never pass: the
check recomputes the digest of the code it names and compares. A rebind
grades nothing; the phase-close regeneration re-measures every family. The
guarantees record scans docs and tests as text and is regenerated with
`guarantees --write` instead.

    uv run python scripts/rebind.py record before.json
    uv run python scripts/move.py runtime
    uv run python scripts/rebind.py apply before.json --step runtime
"""

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import runpy
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
# `scripts/` is not a package, so the codemod is loaded from its file.
SPEC = importlib.util.spec_from_file_location(
    "move", pathlib.Path(__file__).with_name("move.py")
)
assert SPEC
assert SPEC.loader
move = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(move)
LOG = "benchmarks/qualification/rebinds.jsonl"
CERTIFY = (
    "uncertainty",
    "subgroup",
    "framing",
    "loop",
    "f10",
    "f11",
    "f4",
    "f6",
    "ingest",
    "screen",
)
EVIDENCE = (
    "coverage",
    "reproducibility",
    "budget",
    "refusal",
    "faults",
    "leakage",
    "quality",
    "reading",
    "custody",
    "mutation",
    "bypass",
)
CHECKS = [
    *(f"benchmarks.qualification.{name}_certify" for name in CERTIFY),
    "benchmarks.qualification",
    *(f"benchmarks.evidence.{name}" for name in EVIDENCE),
]
STORES = ("scmle/certificates/", "benchmarks/")
# The shortest stored hex prefix re-pointed: 8 hex digits give 2**32 values,
# so a chance match against the few thousand digests a recording holds has
# probability below 1e-6.
PREFIX = 8
HEX = re.compile(rf'"([0-9a-f]{{{PREFIX},}})"')
CONSTRUCTORS = (
    "md5",
    "sha1",
    "sha224",
    "sha256",
    "sha384",
    "sha512",
    "blake2b",
    "blake2s",
    "sha3_256",
    "new",
    "file_digest",
)


class Recorder:
    """A hash object that records every hex digest it produces."""

    def __init__(self, inner: object, seen: list[str]) -> None:
        """Wraps a hash object.

        Args:
            inner: The `hashlib` object.
            seen: The digests recorded so far, appended to.
        """
        self.inner = inner
        self.seen = seen

    def __getattr__(self, name: str) -> object:
        """Delegates every other attribute to the wrapped object."""
        return getattr(self.inner, name)

    def hexdigest(self, *arguments: int) -> str:
        """The wrapped digest, recorded."""
        value = self.inner.hexdigest(*arguments)  # type: ignore[attr-defined]
        self.seen.append(value)
        return value

    def digest(self, *arguments: int) -> bytes:
        """The wrapped digest, recorded as hex."""
        value = self.inner.digest(*arguments)  # type: ignore[attr-defined]
        self.seen.append(value.hex())
        return value

    def copy(self) -> "Recorder":
        """A recording copy."""
        return Recorder(self.inner.copy(), self.seen)  # type: ignore[attr-defined]


def instrument(seen: list[str]) -> None:
    """Makes every `hashlib` constructor return a recording hash object."""
    for name in CONSTRUCTORS:
        original = getattr(hashlib, name)

        def made(
            *arguments: object, original: object = original, **options: object
        ) -> Recorder:
            return Recorder(original(*arguments, **options), seen)  # type: ignore[operator]

        setattr(hashlib, name, made)


def trace(module: str, output: pathlib.Path) -> int:
    """Runs one check with `hashlib` instrumented and saves the digests.

    Returns:
        The check's exit status.
    """
    seen: list[str] = []
    instrument(seen)
    sys.path.insert(0, str(pathlib.Path.cwd()))
    sys.argv = [
        module,
        "--check",
        *(["all"] if module.endswith("qualification") else []),
    ]
    status = 0
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    except SystemExit as stop:
        code = stop.code
        status = code if isinstance(code, int) else int(bool(code))
    output.write_text(json.dumps({"status": status, "digests": seen}), "utf-8")
    return status


def record(
    output: pathlib.Path, root: pathlib.Path
) -> dict[str, dict[str, object]]:
    """Runs every check in its own process and collects the digests.

    Returns:
        Check module to its exit status and digests, in run order.
    """
    found = {}
    scratch = output.with_suffix(".part")
    environment = {**os.environ, "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1"}
    for module in CHECKS:
        subprocess.run(
            [sys.executable, __file__, "trace", module, str(scratch)],
            cwd=root,
            env=environment,
            capture_output=True,
            check=False,
        )
        found[module] = json.loads(scratch.read_text("utf-8"))
        scratch.unlink()
    output.write_text(json.dumps(found, indent=1) + "\n", "utf-8")
    return found


def failures(recording: dict[str, dict[str, object]]) -> list[str]:
    """The checks of a recording that did not pass."""
    return [module for module, run in recording.items() if run["status"]]


def pairing(
    before: dict[str, dict[str, object]], after: dict[str, dict[str, object]]
) -> dict[str, str]:
    """Old digest to new, from two recordings of the same checks.

    Raises:
        SystemExit: When the recordings differ in shape, or one old digest
            pairs with two new values.
    """
    if list(before) != list(after):
        raise SystemExit("the recordings ran different checks")
    found: dict[str, str] = {}
    for module in before:
        old = list(before[module]["digests"])  # type: ignore[call-overload]
        new = list(after[module]["digests"])  # type: ignore[call-overload]
        if len(old) != len(new):
            raise SystemExit(
                f"{module}: {len(old)} digests before, {len(new)} after"
            )
        for first, second in zip(old, new, strict=True):
            if first == second:
                continue
            if found.setdefault(first, second) != second:
                raise SystemExit(f"{module}: {first} pairs with two values")
    return found


def lookup(value: str, pairs: dict[str, str]) -> str | None:
    """The new value of a stored digest or digest prefix, if it has one.

    Raises:
        SystemExit: When a prefix matches old digests with different news.
    """
    if value in pairs:
        return pairs[value]
    width = len(value)
    news = {pairs[old][:width] for old in pairs if old.startswith(value)}
    if len(news) > 1:
        raise SystemExit(f"{value} is a prefix of several re-pointed digests")
    return news.pop() if news else None


def stores(root: pathlib.Path) -> list[str]:
    """The stored records a rebind may rewrite."""
    return [
        name
        for name in move.listing(root)
        if name.startswith(STORES) and name.endswith(".json")
    ]


def relocate(root: pathlib.Path, moves: dict[str, str]) -> None:
    """Rewrites the module paths the stores name.

    The mutation evidence keys files by path.
    """
    for name in stores(root):
        path = root / name
        text = path.read_text("utf-8")
        result = move.translate(text, moves, prose=False)
        if result != text:
            path.write_text(result, "utf-8")


def repoint(
    root: pathlib.Path, pairs: dict[str, str], step: str
) -> list[dict[str, str]]:
    """Rewrites every stored digest the pairing re-points.

    Returns:
        One entry per re-pointed value.
    """
    entries = []
    for name in stores(root):
        path = root / name
        text = path.read_text("utf-8")

        def swap(match: re.Match[str], name: str = name) -> str:
            new = lookup(match[1], pairs)
            if new is None:
                return match[0]
            entries.append(
                {"step": step, "file": name, "old": match[1], "new": new}
            )
            return f'"{new}"'

        result = HEX.sub(swap, text)
        if result != text:
            path.write_text(result, "utf-8")
    return entries


def log(root: pathlib.Path, entries: list[dict[str, str]]) -> None:
    """Appends the re-pointed values to the custody log."""
    commit = move.git(root, "rev-parse", "--short", "HEAD").strip()
    stamp = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    with (root / LOG).open("a", encoding="utf-8") as handle:
        for entry in entries:
            row = {"base": commit, "at": stamp, **entry}
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def apply(
    root: pathlib.Path,
    before: dict[str, dict[str, object]],
    step: str,
    moves: dict[str, str],
    hand: set[str],
) -> int:
    """Re-points the stores when every condition in the module doc holds.

    Args:
        root: The checkout, with the step applied and not committed.
        before: The recording made before the step.
        step: The step's name in the map.
        moves: The step's moves, old dotted path to new.
        hand: Files changed by hand, excused from the rename proof.

    Returns:
        0 when the stores were re-pointed and every check passes.

    Raises:
        SystemExit: When a check failed before the step, or the step is not
            rename-only.
    """
    if failures(before):
        raise SystemExit(f"stale before the move: {failures(before)}")
    if move.verify(root, moves, hand):
        raise SystemExit(f"move.py {step} --verify fails")
    relocate(root, moves)
    with tempfile.TemporaryDirectory() as folder:
        scratch = pathlib.Path(folder) / "digests.json"
        after = record(scratch, root)
        entries = repoint(root, pairing(before, after), step)
        found = record(scratch, root)
    if failures(found):
        print(f"still failing after the rebind: {failures(found)}")
        return 1
    log(root, entries)
    print(f"{step}: {len(entries)} stored values re-pointed")
    return 0


def main(arguments: list[str] | None = None) -> int:
    """Records digests or re-points stores.

    Returns:
        0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    traced = commands.add_parser("trace")
    traced.add_argument("module")
    traced.add_argument("output", type=pathlib.Path)
    recorded = commands.add_parser("record")
    recorded.add_argument("output", type=pathlib.Path)
    recorded.add_argument("--allow-failures", action="store_true")
    applied = commands.add_parser("apply")
    applied.add_argument("before", type=pathlib.Path)
    applied.add_argument("--step", required=True)
    applied.add_argument("--hand", action="append", default=[])
    options = parser.parse_args(arguments)
    if options.command == "trace":
        return trace(options.module, options.output)
    if options.command == "record":
        found = record(options.output, options.root)
        failed = failures(found)
        print(f"{len(found)} checks recorded; failing: {failed or 'none'}")
        return int(bool(failed) and not options.allow_failures)
    steps, _ = move.read(move.MAP)
    return apply(
        options.root,
        json.loads(options.before.read_text("utf-8")),
        options.step,
        steps[options.step],
        set(options.hand),
    )


if __name__ == "__main__":
    sys.exit(main())
