"""Claim C10: the unit tests kill at least the floor share of mutants.

`mutation_score_min` (acceptance.toml) is a `Floor` on the share of
mutants the tests kill. This module measures it with mutmut 3 on a sample
of the package drawn before anything is run, by a rule fixed here:

1. Population: every `scmle/**/*.py`.
2. Frame: the population less every module that cannot produce a
   decision, removed before the draw and listed in the document with the
   reason (`REASONS`), in this order: an entry point (`__main__.py`); no
   mutants (mutmut finds none, as in a package `__init__.py` that only
   re-exports); untested (no `tests/unit/test_*.py` imports it, step 6).
   An untested module is a finding, stored under `untested`, not hidden.
3. Strata: the top-level package (`data`, `models`, ...); single-file
   modules at the top of `scmle/` form the stratum `root`.
4. Modules: `least` of them, the smallest number of trials the floor rule
   can pass on at all (`Floor.minimum(alpha)`), so the pooled count rests
   on at least as many independent clusters as the rule needs units. They
   are allocated to strata in proportion to stratum size (largest
   remainder) and drawn from the frame without replacement with
   `default_rng(SEED)`.
5. Mutants: mutmut's mutants of each module (it mutates function and
   method bodies), of which `MARGIN * ceil(planned / modules)` are drawn
   with `default_rng([SEED, position])`. `planned` is the exact sample the
   floor needs for `power` when the true score is `ASSUMED`; `MARGIN`
   covers modules with fewer mutants and the correlation of mutants inside
   one module.
6. Kill set: the files `tests/unit/test_*.py` that import the module by
   name, and only those.
7. Run: in a copy of the repository, one mutmut run per module, at most
   `WORKERS` workers, a CPU limit per mutant of
   `2 * MULTIPLIER * (baseline seconds + CONSTANT)`, a wall limit per
   module of `CAP` seconds and `BUDGET` seconds in all.

The score is killed over decided mutants. A timeout, a crash, a suspicious
result or `no tests` is not a kill; no surviving mutant is removed as
equivalent, so the score is a lower estimate. The row passes when the
one-sided Clopper-Pearson lower bound of the pooled score at `alpha` is at
least the floor, on at least `planned` mutants from at least `least`
modules.

NOT measured: tests outside `tests/unit`, modules not drawn, modules out
of the frame (untested ones are listed), module-level code (mutmut mutates
function bodies only), equivalent-mutant review.

    python -m benchmarks.evidence.mutation --write
    python -m benchmarks.evidence.mutation --check
"""

import argparse
import ast
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

from mutmut.mutation import file_mutation
import numpy

from benchmarks import acceptance
from scmle.contracts import errors
from scmle.data.detectors import certificates
import scmle.evaluation.coverage

PATH = Path(__file__).with_suffix(".json")
ROOT = Path(__file__).resolve().parents[2]
VARIABLE = "mutation_score_min"
# Seed of the draw, fixed before the first run: the claim's number.
SEED = 10
# True score assumed for the planned sample: midway between the floor 0.85
# and a perfect score. `Floor(0.85).plan(0.90, 0.05, 0.80)` = 281 mutants.
ASSUMED = 0.90
# Mutants drawn per module, as a multiple of the even share of `planned`.
MARGIN = 2
WORKERS = 2
# mutmut's timeout settings: the CPU limit per mutant is
# 2 * MULTIPLIER * (seconds of the tests that cover it + CONSTANT).
MULTIPLIER = 2.0
CONSTANT = 5.0
# Wall seconds per module and in all (the run is capped near 45 minutes).
CAP = 420
BUDGET = 45 * 60
# Directories never copied into the work tree.
SKIP = (
    ".git",
    ".venv",
    "mutants",
    "runs",
    "website",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".hypothesis",
    ".grimp_cache",
    ".import_linter_cache",
    ".coverage",
)
THREADS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
STATUSES = ("killed", "survived", "no tests", "timeout", "crash")
# Why a module is out of the frame, in the order the rules are applied.
REASONS = ("entry point", "no mutants", "untested")
NOT_MEASURED = (
    "tests outside tests/unit",
    "modules not drawn",
    "modules out of the frame (untested ones are listed)",
    "module-level code (mutmut mutates function bodies only)",
    "equivalent-mutant review (survivors are all counted)",
)
# Lists the mutants of the configured module without running a test.
GENERATE = """
import json, os, pathlib
import mutmut.__main__ as main
from mutmut.utils import file_utils
os.makedirs("mutants", exist_ok=True)
file_utils.copy_src_dir()
file_utils.copy_also_copy_files()
file_utils.setup_source_paths()
main.create_mutants(1)
"""


def population(root: Path = ROOT) -> list[str]:
    """Every module of the package.

    Args:
        root: Repository root.

    Returns:
        Paths relative to `root`, POSIX style, sorted.
    """
    found = (root / "scmle").rglob("*.py")
    return sorted(
        p.relative_to(root).as_posix()
        for p in found
        if "__pycache__" not in p.parts
    )


def mutants(path: str, root: Path = ROOT) -> int:
    """How many mutants mutmut makes of a module, without running any.

    Args:
        path: Module path.
        root: Repository root.

    Returns:
        The number of mutant names mutmut's own generator gives it.
    """
    code = (root / path).read_text(encoding="utf-8")
    return len(file_mutation.mutate_file_contents(path, code).mutant_names)


def reason(path: str, root: Path = ROOT) -> str | None:
    """Why a module cannot produce a decision.

    Args:
        path: Module path.
        root: Repository root.

    Returns:
        The first of `REASONS` that applies, or None when it can.
    """
    if Path(path).name == "__main__.py":
        return "entry point"
    if not mutants(path, root):
        return "no mutants"
    if not suite(path, root):
        return "untested"
    return None


def frame(
    paths: list[str], root: Path = ROOT
) -> tuple[list[str], dict[str, list[str]]]:
    """The sampling frame: modules that can produce a decision.

    Args:
        paths: The population, sorted.
        root: Repository root.

    Returns:
        The frame, sorted, and the excluded paths under each of `REASONS`.
    """
    kept = []
    dropped: dict[str, list[str]] = {r: [] for r in REASONS}
    for path in paths:
        found = reason(path, root)
        if found is None:
            kept.append(path)
        else:
            dropped[found].append(path)
    return kept, dropped


def stratum(path: str) -> str:
    """The stratum of a module.

    Args:
        path: Module path such as `scmle/data/ledger/budget.py`.

    Returns:
        The top-level package, or `root` for a file directly in `scmle/`.
    """
    parts = path.split("/")
    return parts[1] if len(parts) > len(("scmle", "file.py")) else "root"


def rule() -> acceptance.Floor:
    """The decision rule of the mutation row.

    Returns:
        The `Floor` of `mutation_score_min`.

    Raises:
        InputError: If acceptance.toml gives the row another rule.
    """
    found = acceptance.load()[VARIABLE].rule
    if not isinstance(found, acceptance.Floor):
        raise errors.InputError(f"{VARIABLE} must have the floor rule")
    return found


@functools.cache
def required() -> tuple[int, int]:
    """Mutants the rule needs, from acceptance.toml.

    Returns:
        The smallest count that can pass at all, and the planned count for
        `power` when the true score is `ASSUMED`.
    """
    variables = acceptance.load()
    alpha, power = variables["alpha"].value, variables["power"].value
    floor = rule()
    return floor.minimum(alpha), floor.plan(ASSUMED, alpha, power)


def design() -> tuple[int, int]:
    """Modules to draw and mutants to draw per module.

    Returns:
        `least` modules and `MARGIN * ceil(planned / least)` mutants each.
    """
    least, planned = required()
    return least, MARGIN * math.ceil(planned / least)


def allocate(sizes: dict[str, int], count: int) -> dict[str, int]:
    """Splits a sample over strata in proportion to their sizes.

    Args:
        sizes: Modules per stratum.
        count: Modules to draw in all.

    Returns:
        Modules to draw per stratum, by largest remainder; ties go to the
        stratum that sorts first.
    """
    total = sum(sizes.values())
    shares = {s: count * n / total for s, n in sizes.items()}
    taken = {s: math.floor(v) for s, v in shares.items()}
    order = sorted(shares, key=lambda s: (taken[s] - shares[s], s))
    for name in order[: count - sum(taken.values())]:
        taken[name] += 1
    return taken


def draw(paths: list[str], count: int) -> list[str]:
    """The stratified random sample of modules.

    Args:
        paths: The frame, sorted.
        count: Modules to draw.

    Returns:
        The drawn paths, sorted.
    """
    strata: dict[str, list[str]] = {}
    for path in paths:
        strata.setdefault(stratum(path), []).append(path)
    taken = allocate({s: len(m) for s, m in strata.items()}, count)
    rng = numpy.random.default_rng(SEED)
    chosen = []
    for name in sorted(strata):
        if taken[name]:
            picks = rng.choice(len(strata[name]), taken[name], replace=False)
            chosen += [strata[name][i] for i in sorted(picks)]
    return sorted(chosen)


def dotted(path: str) -> str:
    """The import name of a module path.

    Args:
        path: Module path such as `scmle/core/randomness.py`.

    Returns:
        The dotted name, such as `scmle.core.randomness`; a package's
        `__init__.py` takes the package's name.
    """
    name = path.removesuffix(".py").removesuffix("/__init__")
    return name.replace("/", ".")


def imports(source: str) -> set[str]:
    """Every module name a test file can reach by an import statement.

    Args:
        source: Python source.

    Returns:
        The names in `import a.b` and, for `from a import b`, both `a`
        and `a.b`.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif (
            isinstance(node, ast.ImportFrom) and node.module and not node.level
        ):
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


def suite(path: str, root: Path = ROOT) -> list[str]:
    """The module's own unit tests: the kill set.

    Args:
        path: Module path.
        root: Repository root.

    Returns:
        Paths of `tests/unit/test_*.py` files that import the module, sorted.
    """
    name = dotted(path)
    return [t for t, names in catalog(root) if name in names]


@functools.cache
def catalog(root: Path) -> tuple[tuple[str, frozenset[str]], ...]:
    """The import names of every unit test file, read once per root.

    Args:
        root: Repository root.

    Returns:
        Pairs of a `tests/unit/test_*.py` path and its import names,
        sorted by path.
    """
    tests = sorted((root / "tests" / "unit").glob("test_*.py"))
    return tuple(
        (
            t.relative_to(root).as_posix(),
            frozenset(imports(t.read_text(encoding="utf-8"))),
        )
        for t in tests
    )


def pick(names: list[str], position: int, count: int) -> list[str]:
    """The mutants drawn from one module.

    Args:
        names: All of the module's mutant names.
        position: The module's place in the sorted sample.
        count: Mutants to draw.

    Returns:
        At most `count` names, drawn with `default_rng([SEED, position])`,
        sorted.
    """
    ordered = sorted(names)
    if len(ordered) <= count:
        return ordered
    rng = numpy.random.default_rng([SEED, position])
    picks = rng.choice(len(ordered), count, replace=False)
    return sorted(ordered[i] for i in picks)


def classify(code: int | None) -> str | None:
    """The status of a mutmut exit code.

    Args:
        code: The exit code mutmut stored for a mutant.

    Returns:
        One of `STATUSES`, or None for a mutant not run.
    """
    if code is None:
        return None
    if code in {1, 3}:
        return "killed"
    if code == 0:
        return "survived"
    if code in {5, 33}:
        return "no tests"
    if code in {24, -24, 36, 152, 255}:
        return "timeout"
    return "crash"


def fingerprint(paths: list[str], root: Path = ROOT) -> str:
    """Hash of the files a module's result depends on.

    Args:
        paths: The module and its kill set.
        root: Repository root.

    Returns:
        SHA-256 over each path and its bytes; a missing file hashes as
        empty.
    """
    digest = hashlib.sha256()
    for path in paths:
        file = root / path
        digest.update(path.encode("utf-8"))
        digest.update(file.read_bytes() if file.exists() else b"")
    return digest.hexdigest()


def snapshot(target: Path) -> None:
    """Copies the repository into a work tree.

    Args:
        target: An empty directory to fill.
    """
    for entry in sorted(ROOT.iterdir()):
        if entry.name in SKIP:
            continue
        if entry.is_dir():
            shutil.copytree(
                entry,
                target / entry.name,
                ignore=shutil.ignore_patterns(*SKIP),
                symlinks=True,
            )
        else:
            shutil.copy2(entry, target / entry.name)


def configure(work: Path, path: str, tests: list[str]) -> None:
    """Writes the mutmut configuration for one module.

    Args:
        work: The work tree.
        path: Module to mutate.
        tests: Its kill set; empty means no test runs.
    """
    shutil.rmtree(work / "mutants", ignore_errors=True)
    copied = sorted(
        e.name + ("/" if e.is_dir() else "")
        for e in work.iterdir()
        if e.name not in {"mutants", "setup.cfg"}
    )
    lines = [
        "[mutmut]",
        f"source_paths =\n    {path}",
        "also_copy =\n" + "\n".join(f"    {c}" for c in copied),
        "mutate_only_covered_lines = false",
        f"timeout_multiplier = {MULTIPLIER}",
        f"timeout_constant = {CONSTANT}",
        "use_git_change_detection = false",
        "pytest_add_cli_args =\n    -p\n    no:cacheprovider",
    ]
    if tests:
        lines.append(
            "pytest_add_cli_args_test_selection =\n"
            + "\n".join(f"    {t}" for t in tests)
        )
    (work / "setup.cfg").write_text("\n".join(lines) + "\n", encoding="utf-8")


def environment() -> dict[str, str]:
    """The environment of every mutmut process.

    Returns:
        The current environment with one thread per numeric library.
    """
    found = dict(os.environ)
    for name in THREADS:
        found[name] = "1"
    return found


def call(work: Path, command: list[str], limit: float) -> tuple[int, str]:
    """Runs a command in the work tree, killing it at the wall limit.

    Args:
        work: The work tree.
        command: Arguments after the interpreter.
        limit: Wall seconds.

    Returns:
        The exit code (-1 when killed at the limit) and the output's tail.
    """
    with subprocess.Popen(
        [sys.executable, *command],
        cwd=work,
        env=environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            output, _ = process.communicate(timeout=max(limit, 1.0))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
            return -1, output[-2000:]
    return process.returncode, output[-2000:]


def codes(work: Path, path: str) -> dict[str, int | None]:
    """The exit codes mutmut stored for a module's mutants.

    Args:
        work: The work tree.
        path: The module.

    Returns:
        Mutant name to exit code (None when not run); empty when mutmut
        stored nothing.
    """
    meta = work / "mutants" / f"{path}.meta"
    if not meta.exists():
        return {}
    stored = json.loads(meta.read_text(encoding="utf-8"))
    found: dict[str, int | None] = stored.get("exit_code_by_key", {})
    return found


def idle(output: str) -> bool:
    """Whether mutmut stopped because no selected test reaches the module.

    mutmut refuses to run when the kill set covers no mutated function, or
    when every selected test is skipped (it then cannot force a failure).
    Both leave every mutant alive, so they count as `no tests`.

    Args:
        output: The tail of mutmut's output.

    Returns:
        True for either refusal.
    """
    if "could not find any test case for any mutant" in output:
        return True
    skipped = re.search(r"(\d+) skipped in", output)
    return (
        "Unable to force test failures" in output
        and skipped is not None
        and " passed" not in output
        and " failed" not in output
    )


def examine(
    work: Path, path: str, position: int, limit: float
) -> dict[str, Any]:
    """Generates, draws and runs one module's mutants.

    Args:
        work: The work tree.
        path: The module.
        position: Its place in the sorted sample.
        limit: Wall seconds left for this module.

    Returns:
        The module's kill set, drawn mutants, counts per status, the
        mutants not run, a note on anything that went wrong, its file
        fingerprint and the seconds taken.
    """
    started = time.monotonic()
    tests = suite(path, work)
    configure(work, path, tests)
    count = design()[1]
    status, output = call(work, ["-c", GENERATE], limit)
    names = pick(list(codes(work, path)), position, count)
    note = None if status == 0 else f"generation failed: {output[-500:]}"
    outcome: dict[str, int | None] = dict.fromkeys(names)
    if names and note is None:
        if tests:
            left = limit - (time.monotonic() - started)
            command = ["-m", "mutmut", "run", "--max-children", str(WORKERS)]
            status, output = call(work, [*command, *names], left)
            stored = codes(work, path)
            outcome = {n: stored.get(n) for n in names}
            if status == -1:
                note = "stopped at the wall limit"
            elif idle(output):
                outcome = dict.fromkeys(names, 33)
                note = "the kill set ran no test that reaches the module"
            elif status != 0:
                note = f"mutmut exited {status}: {output[-500:]}"
        else:
            outcome = dict.fromkeys(names, 33)
    counts = dict.fromkeys(STATUSES, 0)
    for code in outcome.values():
        found = classify(code)
        if found is not None:
            counts[found] += 1
    return {
        "module": path,
        "stratum": stratum(path),
        "tests": tests,
        "mutants": len(codes(work, path)),
        "drawn": names,
        "counts": counts,
        "decided": sum(counts.values()),
        "not_run": sum(c is None for c in outcome.values()),
        "survivors": sorted(n for n, c in outcome.items() if c == 0),
        "note": note,
        "files": fingerprint([path, *tests], work),
        "seconds": round(time.monotonic() - started, 1),
    }


def bound(hits: int, trials: int) -> float | None:
    """One-sided Clopper-Pearson lower bound at acceptance's `alpha`.

    Args:
        hits: Mutants killed.
        trials: Mutants decided.

    Returns:
        The lower bound, or None with no trials.
    """
    if trials == 0:
        return None
    alpha = acceptance.load()["alpha"].value
    return float(
        scmle.evaluation.coverage.clopper_pearson(hits, trials, 1 - 2 * alpha)[
            0
        ]
    )


def pool(modules: list[dict[str, Any]]) -> dict[str, Any]:
    """Pools the modules' counts.

    Args:
        modules: What `examine` returned for each module.

    Returns:
        Counts per status, killed, decided, the score and its lower bound.
    """
    counts = {s: sum(m["counts"][s] for m in modules) for s in STATUSES}
    decided = sum(counts.values())
    return {
        "counts": counts,
        "killed": counts["killed"],
        "decided": decided,
        "not_run": sum(m["not_run"] for m in modules),
        "score": counts["killed"] / decided if decided else None,
        "lower": bound(counts["killed"], decided),
    }


def digest() -> str:
    """Hash of the code that draws, runs and counts.

    Returns:
        The `code_hash` of this module's measuring functions.
    """
    return certificates.code_hash(
        population,
        mutants,
        reason,
        frame,
        stratum,
        required,
        design,
        allocate,
        draw,
        dotted,
        imports,
        suite,
        catalog,
        pick,
        classify,
        configure,
        idle,
        examine,
        pool,
    )


def measure() -> dict[str, Any]:
    """Draws the sample, runs it and builds the stored document.

    Returns:
        The JSON-ready document: the design, each module, the pooled
        score, the threshold it was decided with, what was not measured,
        the code hash and the duration.
    """
    started = time.monotonic()
    variables = acceptance.load()
    least, planned = required()
    count, each = design()
    paths = population()
    kept, dropped = frame(paths)
    sample = draw(kept, count)
    modules = []
    with tempfile.TemporaryDirectory(prefix="mutation") as folder:
        work = Path(folder)
        snapshot(work)
        for position, path in enumerate(sample):
            left = BUDGET - (time.monotonic() - started)
            if left <= 0:
                modules.append({"module": path, "note": "budget spent"})
                continue
            modules.append(examine(work, path, position, min(CAP, left)))
            print(json.dumps({k: modules[-1][k] for k in ("module", "counts")}))
    run = [m for m in modules if "counts" in m]
    return {
        "claim": "C10",
        "variable": VARIABLE,
        "floor": variables[VARIABLE].value,
        "alpha": variables["alpha"].value,
        "power": variables["power"].value,
        "assumed": ASSUMED,
        "minimum_mutants": least,
        "planned_mutants": planned,
        "design": {
            "population": len(paths),
            "frame": len(kept),
            "excluded": {r: len(dropped[r]) for r in REASONS},
            "rule": "out of the frame before the draw: an entry point "
            "(__main__.py), no mutmut mutants, or untested (no "
            "tests/unit/test_*.py imports it)",
            "strata": {
                s: sum(stratum(p) == s for p in kept)
                for s in sorted({stratum(p) for p in kept})
            },
            "modules": count,
            "per_module": each,
            "seed": SEED,
            "draw": "largest-remainder allocation by top-level package, "
            "default_rng(SEED).choice per stratum in sorted order",
            "mutants": "default_rng([SEED, position]).choice of mutmut names",
            "kill_set": "tests/unit/test_*.py files importing the module",
            "workers": WORKERS,
            "timeout": f"2 * {MULTIPLIER} * (baseline + {CONSTANT}) CPU s",
            "cap_seconds": CAP,
            "budget_seconds": BUDGET,
        },
        "excluded": {r: dropped[r] for r in REASONS if r != "untested"},
        "untested": dropped["untested"],
        "sample": sample,
        "modules": modules,
        "pooled": pool(run),
        "not_measured": list(NOT_MEASURED),
        "code": digest(),
        "seconds": round(time.monotonic() - started, 1),
    }


def read(document: dict[str, Any] | None) -> dict[str, Any] | None:
    """The given document, or the stored one.

    Args:
        document: A document, or None to read `PATH`.

    Returns:
        The document, or None when nothing is stored.
    """
    if document is not None:
        return document
    if not PATH.exists():
        return None
    loaded: dict[str, Any] = json.loads(PATH.read_text(encoding="utf-8"))
    return loaded


def stale(
    document: dict[str, Any] | None = None, root: Path = ROOT
) -> list[str]:
    """Why the stored document is not for the current code (`--check`).

    Args:
        document: The document; read from `PATH` when None.
        root: Repository root whose files are compared with the measured
            ones.

    Returns:
        Messages for a missing file, code that changed, a threshold other
        than acceptance.toml's, or a module or its tests changed since the
        run; empty when the stored result is current. A sample too small
        to decide is not staleness: `problems` adds it and `summary`
        reports it as not measured.
    """
    found = read(document)
    if found is None:
        return [f"{PATH.name} is missing"]
    issues = []
    if found.get("code") != digest():
        issues.append("mutation: measured with other code")
    floor = acceptance.load()[VARIABLE].value
    if found.get("floor") != floor:
        issues.append(f"mutation: floor {found.get('floor')} is not {floor}")
    run = [m for m in found.get("modules", []) if "counts" in m]
    for module in run:
        paths = [module["module"], *module["tests"]]
        if fingerprint(paths, root) != module["files"]:
            issues.append(f"mutation: {module['module']} changed since run")
    return issues


def problems(
    document: dict[str, Any] | None = None, root: Path = ROOT
) -> list[str]:
    """Why the document cannot decide the row.

    Args:
        document: The document; read from `PATH` when None.
        root: Repository root whose files are compared with the measured
            ones.

    Returns:
        The `stale` messages, plus fewer modules decided than the rule's
        minimum or fewer decided mutants than planned; empty when the
        document decides.
    """
    found = read(document)
    issues = stale(found, root)
    if found is None:
        return issues
    least, planned = required()
    run = [m for m in found.get("modules", []) if "counts" in m]
    used = [m for m in run if m["decided"]]
    if len(used) < least:
        issues.append(
            f"mutation: {len(used)} modules decided; none can pass below "
            f"{least}"
        )
    decided = sum(m["decided"] for m in run)
    if decided < planned:
        issues.append(
            f"mutation: {decided} mutants decided, fewer than the "
            f"{planned} planned"
        )
    return issues


def summary(
    document: dict[str, Any] | None = None, root: Path = ROOT
) -> tuple[str, str]:
    """The scorecard's value and verdict for the row.

    Args:
        document: The document; read from `PATH` when None.
        root: Repository root, passed to `problems`.

    Returns:
        The value text (killed over decided, the score, its lower bound
        and the modules behind it) and `pass` or `fail` by the floor rule
        on the pooled count; `not measured` when nothing is stored or
        `problems` finds any.
    """
    found = read(document)
    if found is None or not found.get("modules"):
        return "no measurement", "not measured"
    run = [m for m in found["modules"] if "counts" in m]
    pooled = pool(run)
    if not pooled["decided"]:
        return "no mutant decided", "not measured"
    text = (
        f"{pooled['killed']}/{pooled['decided']} mutants killed "
        f"({pooled['score']:.3f}), lower bound {pooled['lower']:.4f}, "
        f"{len(run)} sampled modules"
    )
    if problems(found, root):
        return text, "not measured"
    alpha = acceptance.load()["alpha"].value
    passed = rule().passes(pooled["killed"], pooled["decided"], alpha)
    return text, "pass" if passed else "fail"


def main(arguments: list[str] | None = None) -> int:
    """Writes (`--write`) or checks (`--check`) the stored measurement.

    Args:
        arguments: Command-line arguments; None reads `sys.argv`.

    Returns:
        0 after `--write`; for `--check`, 0 when `stale` is empty (the
        stored result is for the current code) and 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    options = parser.parse_args(arguments)
    if options.write:
        document = measure()
        PATH.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary(document)))
        return 0
    issues = stale()
    print(json.dumps({"ok": not issues, "problems": issues}, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
