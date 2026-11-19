"""Claim C9: a secure, private, trustworthy supply chain (three clauses).

Pre-registered in docs/preregistration/c9.md. Measured here:

* `deserialisation`, "no unauthenticated deserialisation path": every call
  site in `scmle/` that unpickles, found statically
  (`scmle.security.deserialisation`), must read authenticated bytes, and
  each must refuse an unsigned and a wrongly signed tripwire pickle in its
  own attack test (`tests/unit/test_deserialisation_paths.py`, run here),
  whose positive control proves the same tripwire, signed with the right
  key, does reach that site's deserialiser. A `roundtrip` site reads no
  input, so it has one attack (it must open no file), not two. Pass when no
  site is unauthenticated and every site's attacks pass.
* `artifacts`, "no raw rows, categories or secrets in artifacts": real
  runs (`Project.run`) on a PII-planted corpus, two tables per task (binary,
  regression, multiclass) with planted names, identifiers, free text that
  carries a secret token, rare category strings, and a signing key, are
  scanned file by file (`scmle.security.leaks`). Tables are split by a
  seeded hash, one per task to `validation` (decides) and one to
  `development` (reported); the validation tables are sealed by digest.
  Fail on any validation leak. A validation run that crashed wrote less
  than a real run does, so any crash makes the clause `not measured`
  whatever the scan found. A clean result is `not measured`, never
  `pass`: the pre-registration asks for values planted by someone other
  than the builder, and these are planted by this module.
* `release`, "SLSA provenance, signed releases, tag protection, no known
  high vulnerability": `.github/workflows/release.yml` is read for SLSA
  provenance from `slsa-github-generator` (at a full release tag, the one
  form it accepts, whose commit is checked live), Sigstore signing pinned
  by commit, and a pip-audit gate before upload. The file is parsed, so a
  comment never satisfies a check: steps are found by their `run` and
  `uses` keys and their order is checked. Live, the repository's
  rulesets must protect release tags from deletion and update, and
  `pip-audit` over the locked dependencies, judged by
  `scmle.deploy.vuln_gate` (fail closed: pip-audit gives no severity, so
  any finding blocks), must find nothing. Pass when every check passes.

    python -m benchmarks.evidence.supply --write
    python -m benchmarks.evidence.supply --check
    python -m benchmarks.evidence.supply --protected OWNER/REPO --tag v1.2.3
"""

import abc
import argparse
import collections
from collections.abc import Callable
import contextlib
import fnmatch
import hashlib
import itertools
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Any
import xml.etree.ElementTree

import numpy
import pandas
import yaml

from scmle.contracts import errors
from scmle.deploy import vuln_gate
from scmle.security import deserialisation
from scmle.security import leaks

PATH = pathlib.Path(__file__).with_suffix(".json")
ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "scmle"
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
ATTACKS = ROOT / "tests" / "unit" / "test_deserialisation_paths.py"
POLICY = ROOT / "deploy" / "vuln-policy.json"
SEED = 20261119
TASKS = ("binary", "regression", "multiclass")
VALIDATION = "validation"
DEVELOPMENT = "development"
ROWS = 300
# Seconds per run. A 300-row run spends up to ~15 s in preparation on a
# loaded machine, the budget reserves 8.5 s for the final refit, diagnostics
# and packaging, and one honest trial takes a few seconds: about 25 s, the
# smallest budget the run itself named when 20 s was refused. Twice that
# for headroom on a slower or busier machine.
BUDGET = 60
# A run still refused names the smallest budget that fits
# (`smallest_budget_seconds`); it is retried once at this multiple of it.
HEADROOM = 2
# Each signed site must refuse an unsigned and a wrongly signed tripwire; a
# round trip reads no input and has one attack, that it opens no file.
ATTEMPTS = {"roundtrip": 1}
SIGNED = 2
GENERATOR = "slsa-framework/slsa-github-generator"
# The one action not pinned by commit: the generator's reusable workflow at
# an exact release tag, the only ref its builder-fetch.sh accepts.
WORKFLOW_REF = re.compile(
    r"slsa-framework/slsa-github-generator/\.github/workflows/"
    r"generator_generic_slsa3\.yml@v\d+\.\d+\.\d+"
)
PROVENANCE = re.compile(
    r"^\s*uses:\s*slsa-framework/slsa-github-generator/\.github/workflows/"
    r"generator_generic_slsa3\.yml@(?P<tag>v\d+\.\d+\.\d+)\s*#\s*"
    r"(?P<commit>[0-9a-f]{40})\s*$",
    re.MULTILINE,
)
SIGNER = re.compile(r"sigstore/gh-action-sigstore-python@[0-9a-f]{40}")
UPLOADER = re.compile(r"pypa/gh-action-pypi-publish@[0-9a-f]{40}")
PINNED = re.compile(r"[^\s@]+@[0-9a-f]{40}")
SUBJECTS = "needs.publish.outputs.hashes"
SYLLABLES = (
    "vor", "qua", "thel", "zin", "bra", "mox", "lune", "dra", "kel", "fis",
    "ter", "quo", "ry", "sa", "van", "dil", "orn", "pex", "wyl", "gash",
)  # fmt: skip
REGIONS = ("north", "south", "east", "west")
COLOURS = ("red", "green", "blue")
Shell = Callable[[list[str]], str]


def shell(command: list[str]) -> str:
    """Runs a command and returns its standard output.

    Args:
        command: The program and its arguments.

    Returns:
        Standard output.

    Raises:
        CalledProcessError: If the command fails.
    """
    return subprocess.run(
        command, check=True, capture_output=True, text=True, cwd=ROOT
    ).stdout


def word(rng: numpy.random.Generator, parts: int) -> str:
    """A capitalised made-up word of `parts` syllables.

    Args:
        rng: The random source.
        parts: Syllables.

    Returns:
        The word.
    """
    return "".join(rng.choice(SYLLABLES, parts)).capitalize()


def unique(make: Callable[[], str], count: int) -> list[str]:
    """`count` distinct values from a generator.

    Args:
        make: Draws one value.
        count: How many.

    Returns:
        The values, in draw order.
    """
    found: dict[str, None] = {}
    while len(found) < count:
        found[make()] = None
    return list(found)


def table(name: str, task: str, seed: int = SEED) -> dict[str, Any]:
    """One PII-planted table and its planted secrets.

    Args:
        name: The table's name, which seeds it.
        task: `binary`, `regression` or `multiclass`.
        seed: The corpus seed.

    Returns:
        `frame` (with target `y`), `secrets` (the free-text tokens and the
        signing key) and `key` (the signing key the run uses).
    """
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    rng = numpy.random.default_rng(int.from_bytes(digest[:8], "big"))
    names = unique(lambda: f"{word(rng, 3)} {word(rng, 3)}", ROWS)
    tokens = [f"sk_live_{rng.bytes(12).hex()}" for row in range(ROWS)]
    rare = unique(lambda: word(rng, 4), 2)
    region = list(rng.choice(REGIONS, ROWS - 3))
    region[:0] = [rare[0], rare[1], rare[1]]
    frame = pandas.DataFrame(
        {
            "a": rng.normal(size=ROWS),
            "b": rng.normal(size=ROWS),
            "age": rng.integers(18, 90, ROWS),
            "name": names,
            "ident": [f"CUST-{n:09d}" for n in rng.permutation(10**6)[:ROWS]],
            "note": [
                f"{n} asked us to call back; access token {t}"
                for n, t in zip(names, tokens, strict=True)
            ],
            "colour": rng.choice(COLOURS, ROWS),
            "region": region,
        }
    )
    noise = rng.normal(size=ROWS)
    signal = frame.a + 0.5 * frame.b + noise
    if task == "binary":
        frame["y"] = (signal > 0).astype(int)
    elif task == "regression":
        frame["y"] = signal
    else:
        frame["y"] = numpy.digitize(signal, [-0.5, 0.5])
    key = rng.bytes(24).hex()
    return {"frame": frame, "secrets": [*tokens, key], "key": key}


def splits(seed: int = SEED) -> dict[str, str]:
    """Each corpus table's split: per task, the first by hash validates.

    Args:
        seed: The corpus seed.

    Returns:
        Table name to `validation` or `development`.
    """
    placed = {}
    for task in TASKS:
        names = sorted(
            (f"{task}-{i}" for i in range(2)),
            key=lambda n: hashlib.sha256(f"{seed}:{n}".encode()).hexdigest(),
        )
        placed[names[0]] = VALIDATION
        placed[names[1]] = DEVELOPMENT
    return placed


@contextlib.contextmanager
def environment(values: dict[str, str]) -> Any:
    """Sets environment variables for the duration of a block.

    Args:
        values: Variables to set.

    Yields:
        Nothing.
    """
    saved = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def train(
    frame: pandas.DataFrame,
    folder: pathlib.Path,
    key: str,
    budget: int = BUDGET,
) -> str:
    """One real, fast run; returns the error it raised, or "".

    A run refused because its budget cannot fit the minimum run is retried
    once, at `HEADROOM` times the smallest budget the refusal names.

    Args:
        frame: The table; its target is `y`.
        folder: The run store.
        key: The signing key the run signs its bundle with.
        budget: Seconds for the run.

    Returns:
        The error type and message, or "" when the run completed.
    """
    try:
        attempt = run(frame, folder, key, budget)
    except errors.BlockedError as error:
        smallest = error.details.get("smallest_budget_seconds")
        if not isinstance(smallest, int | float) or smallest <= budget:
            return f"{type(error).__name__}: {error}"[:300]
        return train(frame, folder, key, math.ceil(HEADROOM * smallest))
    return attempt


def run(
    frame: pandas.DataFrame, folder: pathlib.Path, key: str, budget: int
) -> str:
    """One real run at a given budget.

    Args:
        frame: The table; its target is `y`.
        folder: The run store.
        key: The signing key the run signs its bundle with.
        budget: Seconds for the run.

    Returns:
        The error type and message, or "" when the run completed.

    Raises:
        BlockedError: If the budget cannot fit the minimum run.
    """
    import scmle.api.project
    from scmle.contracts import tasks

    with environment({"SCMLE_SIGNING_KEY": key}):
        try:
            scmle.api.project.Project(
                frame,
                "y",
                output_dir=folder,
                time_budget=budget,
                search=tasks.Search(top_models=0),
                strategies=tasks.Strategies(top_models=0),
                ensemble=tasks.Stacking(size=0),
                models=tasks.Candidates(
                    exclude_families=["bagging", "deep_learning", "foundation"]
                ),
                enforcement="cooperative",
            ).run()
        except errors.BlockedError:
            raise
        except Exception as error:  # noqa: BLE001 - a failed run still wrote artifacts
            return f"{type(error).__name__}: {error}"[:300]
    return ""


def tally(report: leaks.Report) -> list[dict[str, Any]]:
    """Leaks grouped by file, kind, column and form, run folders unnamed.

    Args:
        report: A scan.

    Returns:
        One record per group with its count, sorted.
    """
    counted = collections.Counter(
        (re.sub(r"run_\d{8}_\d{3}", "run", f.path), f.kind, f.column, f.form)
        for f in report.leaks()
    )
    return [
        {"path": p, "kind": k, "column": c, "form": f, "count": n}
        for (p, k, c, f), n in sorted(counted.items())
    ]


class Clause(abc.ABC):
    """One clause of C9: how it is measured, read and judged.

    Attributes:
        name: The clause's key in the stored document.
        criterion: The clause as the scorecard words it.
    """

    name = ""
    criterion = ""

    @abc.abstractmethod
    def measure(self) -> dict[str, Any]:
        """Measures the clause.

        Returns:
            The JSON-ready record.
        """

    @abc.abstractmethod
    def judge(self, record: dict[str, Any]) -> tuple[str, str]:
        """The value text and verdict of a stored record.

        Args:
            record: The clause's stored record.

        Returns:
            The text and `pass`, `fail` or `not measured`.
        """


class Deserialisation(Clause):
    """No unauthenticated deserialisation path."""

    name = "deserialisation"
    criterion = "no unauthenticated deserialisation path"

    def measure(self) -> dict[str, Any]:
        """Scans the package and runs the attack per site.

        Returns:
            The sites, and passed and failed attacks per site.
        """
        sites = deserialisation.scan(PACKAGE)
        with tempfile.TemporaryDirectory() as folder:
            report = pathlib.Path(folder) / "attacks.xml"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    str(ATTACKS),
                    f"--junitxml={report}",
                ],
                check=False,
                capture_output=True,
                cwd=ROOT,
                env={**os.environ, "OMP_NUM_THREADS": "1"},
            )
            attacks = outcomes(report.read_text(encoding="utf-8"))
        return {
            "sites": [
                {
                    "key": s.key,
                    "line": s.line,
                    "sink": s.sink,
                    "verdict": s.verdict,
                    "reason": s.reason,
                }
                for s in sites
            ],
            "attacks": {
                s.key: attacks.get(s.key, {"passed": 0, "failed": 0})
                for s in sites
            },
        }

    def judge(self, record: dict[str, Any]) -> tuple[str, str]:
        """Counts unauthenticated sites and failed or missing attacks.

        Args:
            record: The clause's stored record.

        Returns:
            The text and verdict.
        """
        sites = record["sites"]
        verdicts = collections.Counter(s["verdict"] for s in sites)
        unsafe = [s["key"] for s in sites if s["verdict"] == "unauthenticated"]
        needed = {s["key"]: ATTEMPTS.get(s["verdict"], SIGNED) for s in sites}
        untested = [
            key
            for key, tried in record["attacks"].items()
            if tried["failed"] or tried["passed"] < needed.get(key, SIGNED)
        ]
        text = (
            f"{len(sites)} sites: "
            + ", ".join(f"{verdicts[v]} {v}" for v in deserialisation.VERDICTS)
            + f"; attacks refused at {len(sites) - len(untested)} of "
            f"{len(sites)} sites"
        )
        if unsafe or untested:
            return text + f" (open: {', '.join(unsafe + untested)})", "fail"
        return text, "pass"


def outcomes(junit: str) -> dict[str, dict[str, int]]:
    """Passed and failed attack tests per site, from a JUnit report.

    An attack counts as passed only when the site's positive control (the
    same tripwire signed with the right key reaches the deserialiser) also
    passed; a failed or missing control fails every attack on that site,
    since a refusal that comes before the sink proves nothing.

    Args:
        junit: The JUnit XML text.

    Returns:
        Site key to its passed and failed counts.
    """
    found: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"passed": 0, "failed": 0}
    )
    attack = re.compile(
        r"test_unauthenticated_bytes_never_run"
        r"\[(.+)-(unsigned|wrong-key|no-input)\]"
    )
    control = re.compile(r"test_attack_reaches_the_sink\[(.+)\]")
    attacks: list[tuple[str, str, bool]] = []
    reached: set[str] = set()
    for case in xml.etree.ElementTree.fromstring(junit).iter("testcase"):
        name = case.get("name", "")
        failed = any(
            case.find(tag) is not None
            for tag in ("failure", "error", "skipped")
        )
        match = attack.fullmatch(name)
        if match is not None:
            attacks.append((match.group(1), match.group(2), failed))
        match = control.fullmatch(name)
        if match is not None and not failed:
            reached.add(match.group(1))
    for site, kind, failed in attacks:
        # a site with no input has no sink to reach: its attack is that it
        # opens no file at all
        controlled = kind == "no-input" or site in reached
        passed = controlled and not failed
        found[site]["passed" if passed else "failed"] += 1
    return dict(found)


class Artifacts(Clause):
    """No raw rows, categories or secrets in artifacts."""

    name = "artifacts"
    criterion = "no raw rows, categories or secrets in artifacts"

    def measure(self) -> dict[str, Any]:
        """Runs every corpus table for real and scans what it wrote.

        Returns:
            The seal of the validation tables and one record per table.
        """
        placed = splits()
        sealed = hashlib.sha256()
        records = []
        with tempfile.TemporaryDirectory() as folder:
            for name, split in sorted(placed.items()):
                made = table(name, name.split("-")[0])
                frame = made["frame"]
                if split == VALIDATION:
                    sealed.update(frame.to_csv(index=False).encode("utf-8"))
                store = pathlib.Path(folder) / name
                error = train(frame, store, made["key"])
                schemas = sorted(store.rglob("model/schema.json"))
                levels = leaks.declared(schemas[0]) if schemas else {}
                report = leaks.scan(
                    store, leaks.probes(frame, levels, made["secrets"])
                )
                summary = report.to_dict()
                records.append(
                    {
                        "table": name,
                        "split": split,
                        "error": error,
                        "files": report.files,
                        "probes": report.probes,
                        "unsearchable": report.unsearchable,
                        "leaks": report.counts(),
                        "cells": summary["cells"],
                        "found": tally(report),
                    }
                )
        return {"seal": sealed.hexdigest(), "tables": records}

    def judge(self, record: dict[str, Any]) -> tuple[str, str]:
        """Counts validation leaks by kind and the files that hold them.

        Args:
            record: The clause's stored record.

        Returns:
            The text and verdict: `fail` on any validation leak, else `not
            measured` (the planted values are the builder's own).
        """
        chosen = [t for t in record["tables"] if t["split"] == VALIDATION]
        totals: collections.Counter[str] = collections.Counter()
        files: set[str] = set()
        for each in chosen:
            totals.update(each["leaks"])
            files.update(f["path"] for f in each["found"])
        text = (
            f"{len(chosen)} validation runs scanned: "
            + ", ".join(f"{totals[k]} {k}" for k in leaks.LEAKS)
            + " leaks"
        )
        if files:
            text += f" in {', '.join(sorted(files))}"
        crashed = [f"{t['table']}: {t['error']}" for t in chosen if t["error"]]
        if crashed:
            text += f"; {len(crashed)} runs crashed: {'; '.join(crashed)}"
        if sum(totals.values()):
            return text, "fail"
        if crashed:
            return text, "not measured"
        return text + "; values planted by the builder", "not measured"


class Release(Clause):
    """SLSA provenance, signed releases, tag protection, no high CVE.

    Attributes:
        run: Runs a command and returns its output (live checks).
        repository: The GitHub repository, `owner/name`.
    """

    name = "release"
    criterion = (
        "SLSA provenance, signed releases, tag protection, no known high "
        "vulnerability"
    )

    def __init__(self, run: Shell = shell, repository: str = "") -> None:
        """Keeps the command runner and the repository.

        Args:
            run: Runs a command and returns its output.
            repository: `owner/name`; empty reads the `origin` remote.
        """
        self.run = run
        self.repository = repository

    def measure(self) -> dict[str, Any]:
        """Reads the release workflow and checks the live controls.

        Returns:
            One record per check with `ok` and `detail`.
        """
        text = WORKFLOW.read_text(encoding="utf-8")
        checks = workflow(text)
        repository = self.repository or origin(self.run)
        checks["generator"] = attempt(lambda: tagged(text, self.run))
        checks["protection"] = attempt(
            lambda: protected(repository, "v0.0.0", self.run)
        )
        checks["vulnerabilities"] = attempt(lambda: audited(self.run))
        return {"repository": repository, "checks": checks}

    def judge(self, record: dict[str, Any]) -> tuple[str, str]:
        """Names the checks that failed.

        Args:
            record: The clause's stored record.

        Returns:
            The text and verdict: `pass` only when every check passed.
        """
        checks = record["checks"]
        failed = sorted(k for k, v in checks.items() if not v["ok"])
        text = f"{len(checks) - len(failed)} of {len(checks)} checks pass"
        if failed:
            details = "; ".join(f"{k}: {checks[k]['detail']}" for k in failed)
            return f"{text} ({details})", "fail"
        return text, "pass"


def attempt(check: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Runs a live check; a check that cannot run fails, saying why.

    Args:
        check: Returns `{"ok": ..., "detail": ...}`.

    Returns:
        Its result, or a failure naming the error.
    """
    try:
        return check()
    except (
        OSError,
        ValueError,
        KeyError,
        subprocess.CalledProcessError,
    ) as error:
        detail = getattr(error, "stderr", "") or str(error)
        return {"ok": False, "detail": f"could not check: {detail}"[:300]}
    except vuln_gate.GateError as error:
        return {"ok": False, "detail": f"gate refused the report: {error}"}


def active(step: Any) -> bool:
    """Whether a step always runs and fails its job when it fails.

    Args:
        step: A parsed step.

    Returns:
        False for a non-mapping, or a step with an `if` condition or
        `continue-on-error`, either of which can skip or excuse it.
    """
    return (
        isinstance(step, dict)
        and "if" not in step
        and not step.get("continue-on-error", False)
    )


def ran(steps: list[Any], command: re.Pattern[str]) -> tuple[int, int] | None:
    """Where a command first runs among a job's steps.

    Only a line that starts with the command counts: a shell comment, an
    `echo` that names it, or a step that may be skipped does not.

    Args:
        steps: The job's parsed steps.
        command: Matched at the start of each stripped `run` line.

    Returns:
        The step and line index, or None when it never runs.
    """
    for index, step in enumerate(steps):
        script = step.get("run") if active(step) else None
        if not isinstance(script, str):
            continue
        for line, text in enumerate(script.splitlines()):
            if command.match(text.strip()):
                return index, line
    return None


def used(steps: list[Any], action: re.Pattern[str]) -> tuple[int, int] | None:
    """Where an action is first used among a job's steps.

    Args:
        steps: The job's parsed steps.
        action: Must match the whole `uses` value.

    Returns:
        The step index (and line 0), or None when it is never used.
    """
    for index, step in enumerate(steps):
        target = step.get("uses") if active(step) else None
        if isinstance(target, str) and action.fullmatch(target):
            return index, 0
    return None


def ordered(*positions: tuple[int, int] | None) -> bool:
    """Whether every position was found, each strictly after the last.

    Args:
        *positions: Step and line indices, or None.

    Returns:
        True when none is None and they increase.
    """
    found = [p for p in positions if p is not None]
    return len(found) == len(positions) and all(
        a < b for a, b in itertools.pairwise(found)
    )


def references(document: dict[str, Any]) -> list[str]:
    """Every action and reusable workflow a parsed workflow uses.

    Args:
        document: The parsed workflow.

    Returns:
        Each job's `uses` and each step's `uses`, in file order.
    """
    found = []
    for job in field(document, "jobs").values():
        if not isinstance(job, dict):
            continue
        steps = job.get("steps")
        chosen = (
            [s for s in steps if isinstance(s, dict)]
            if isinstance(steps, list)
            else []
        )
        found += [job.get("uses"), *(s.get("uses") for s in chosen)]
    return [t for t in found if isinstance(t, str)]


def field(parent: dict[str, Any], key: str) -> dict[str, Any]:
    """A mapping inside a parsed workflow, empty when it is anything else.

    Args:
        parent: The enclosing mapping.
        key: The field.

    Returns:
        The field's mapping, or an empty one.
    """
    value = parent.get(key)
    return value if isinstance(value, dict) else {}


def workflow(text: str) -> dict[str, dict[str, Any]]:
    """The static checks of the release workflow, read from its parse.

    Comments are not part of the parse, so none can satisfy a check.

    Args:
        text: `release.yml`.

    Returns:
        `provenance`, `signing`, `audit` and `pinning`, each with `ok` and
        `detail`.
    """
    details = {
        "provenance": "generator_generic_slsa3 at a release tag whose commit "
        "is recorded, after publish, subjects from the built distributions, "
        "attached to the release",
        "signing": "Sigstore, pinned by commit, signs every distribution "
        "before the GitHub release is created",
        "audit": "pip-audit and the vulnerability gate run before upload",
        "pinning": "every other action pinned by commit",
    }
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        problem = f"release.yml does not parse: {error}"[:300]
        return {k: {"ok": False, "detail": problem} for k in details}
    document = field({"document": document}, "document")
    jobs = field(document, "jobs")
    publish = field(jobs, "publish")
    steps = publish.get("steps")
    steps = steps if isinstance(steps, list) else []
    pip = ran(steps, re.compile(r"(?:uv\s+run\s+)?pip-audit\s"))
    gate = ran(
        steps,
        re.compile(r"(?:uv\s+run\s+)?python\s+-m\s+scmle\.deploy\.vuln_gate\s"),
    )
    upload = used(steps, UPLOADER)
    sign = used(steps, SIGNER)
    release = ran(steps, re.compile(r"gh\s+release\s+create\s"))
    chain = field(jobs, "provenance")
    target = chain.get("uses")
    given = field(chain, "with")
    needs = chain.get("needs")
    recorded = PROVENANCE.search(text)
    loose = [
        t
        for t in references(document)
        if not t.startswith("./")
        and not PINNED.fullmatch(t)
        and not WORKFLOW_REF.fullmatch(t)
    ]
    checks = {
        "provenance": isinstance(target, str)
        and WORKFLOW_REF.fullmatch(target) is not None
        and recorded is not None
        and target.endswith(f"@{recorded.group('tag')}")
        and given.get("upload-assets") is True
        and SUBJECTS in str(given.get("base64-subjects", ""))
        and "publish" in ([needs] if isinstance(needs, str) else needs or []),
        "signing": ordered(sign, release),
        "audit": ordered(pip, gate, upload),
        "pinning": not loose,
    }
    if loose:
        details["pinning"] += f"; loose: {', '.join(loose)}"
    return {k: {"ok": bool(v), "detail": details[k]} for k, v in checks.items()}


def tagged(text: str, run: Shell) -> dict[str, Any]:
    """Whether the generator's tag still names the commit recorded beside it.

    Args:
        text: `release.yml`.
        run: Runs a command.

    Returns:
        `ok` when the tag resolves to the recorded commit.
    """
    found = PROVENANCE.search(text)
    if found is None:
        return {"ok": False, "detail": "no generator reference"}
    tag, commit = found.group("tag"), found.group("commit")
    resolved = run(
        ["gh", "api", f"repos/{GENERATOR}/commits/{tag}", "--jq", ".sha"]
    ).strip()
    return {
        "ok": resolved == commit,
        "detail": f"{tag} is {resolved}, recorded {commit}",
    }


def origin(run: Shell) -> str:
    """The GitHub repository of the `origin` remote.

    Args:
        run: Runs a command.

    Returns:
        `owner/name`, or "" when the remote is not on GitHub.
    """
    try:
        url = run(["git", "remote", "get-url", "origin"]).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""
    found = re.search(r"github\.com[:/](.+?)(?:\.git)?$", url)
    return found.group(1) if found else ""


def protected(repository: str, tag: str, run: Shell) -> dict[str, Any]:
    """Whether an active ruleset stops a release tag being deleted or moved.

    Args:
        repository: `owner/name`.
        tag: A release tag name, such as `v1.2.3`.
        run: Runs a command.

    Returns:
        `ok` when some active tag ruleset that covers `refs/tags/<tag>`
        has both a `deletion` and an `update` rule.

    Raises:
        ValueError: If no repository is given.
    """
    if not repository:
        raise ValueError("no GitHub repository to check")
    ref = f"refs/tags/{tag}"
    listed = json.loads(
        run(
            [
                "gh",
                "api",
                f"repos/{repository}/rulesets?targets=tag&includes_parents=true",
            ]
        )
    )
    for summary in listed:
        ruleset = json.loads(
            run(["gh", "api", f"repos/{repository}/rulesets/{summary['id']}"])
        )
        names = ruleset.get("conditions", {}).get("ref_name", {})
        covered = any(
            p == "~ALL" or fnmatch.fnmatch(ref, p)
            for p in names.get("include", [])
        ) and not any(fnmatch.fnmatch(ref, p) for p in names.get("exclude", []))
        kinds = {rule.get("type") for rule in ruleset.get("rules", [])}
        if (
            ruleset.get("target") == "tag"
            and ruleset.get("enforcement") == "active"
            and covered
            and {"deletion", "update"} <= kinds
        ):
            return {"ok": True, "detail": f"ruleset {ruleset.get('name')}"}
    return {
        "ok": False,
        "detail": f"no active tag ruleset forbids deleting or moving {ref} "
        f"({len(listed)} tag rulesets)",
    }


def audited(run: Shell) -> dict[str, Any]:
    """pip-audit over every locked dependency, judged by the release gate.

    Args:
        run: Runs a command.

    Returns:
        `ok` when the gate passes; the detail names the blocking ids.
    """
    with tempfile.TemporaryDirectory() as folder:
        locked = pathlib.Path(folder) / "requirements.txt"
        found = pathlib.Path(folder) / "audit.json"
        run(
            [
                "uv",
                "export",
                "--frozen",
                "--all-extras",
                "--no-emit-project",
                "--output-file",
                str(locked),
            ]
        )
        with contextlib.suppress(subprocess.CalledProcessError):
            # pip-audit exits 1 on a finding; the gate reads the report
            run(
                [
                    "uv",
                    "run",
                    "pip-audit",
                    "--requirement",
                    str(locked),
                    "--no-deps",
                    "--disable-pip",
                    "--format",
                    "json",
                    "--output",
                    str(found),
                ]
            )
        report = json.loads(found.read_text(encoding="utf-8"))
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    verdict = vuln_gate.evaluate(report, policy)
    blocking = sorted(f["id"] for f in verdict["blocking"])
    return {
        "ok": bool(verdict["passed"]),
        "detail": f"{verdict['packages_scanned']} packages scanned, "
        f"{len(blocking)} blocking"
        + (f" ({', '.join(blocking)})" if blocking else ""),
    }


CLAUSES: tuple[Clause, ...] = (Deserialisation(), Artifacts(), Release())


def digest() -> str:
    """Hash of everything the measurement reads.

    Returns:
        SHA-256 over this module, the package, the release workflow, the
        attack tests, the lock file and the vulnerability policy.
    """
    hasher = hashlib.sha256()
    paths = [pathlib.Path(__file__), *sorted(PACKAGE.rglob("*.py"))]
    paths += [WORKFLOW, ATTACKS, ROOT / "uv.lock", POLICY]
    for path in paths:
        hasher.update(path.relative_to(ROOT).as_posix().encode())
        hasher.update(hashlib.sha256(path.read_bytes()).digest())
    return hasher.hexdigest()


def measure(chosen: tuple[Clause, ...] = CLAUSES) -> dict[str, Any]:
    """Measures every clause.

    Args:
        chosen: The clauses.

    Returns:
        The document: the code digest and a record per clause.
    """
    return {"code": digest(), **{c.name: c.measure() for c in chosen}}


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


def problems(document: dict[str, Any] | None = None) -> list[str]:
    """Why the stored measurement is not current.

    Args:
        document: The document; read from `PATH` when None.

    Returns:
        Messages for a missing file, other code or a missing clause.
    """
    found = read(document)
    if found is None:
        return [f"{PATH.name} is missing"]
    issues = []
    if found.get("code") != digest():
        issues.append("supply: measured with other code")
    issues.extend(
        f"supply: clause {c.name} not measured"
        for c in CLAUSES
        if c.name not in found
    )
    return issues


def summary(
    document: dict[str, Any] | None = None, clause: str = ""
) -> tuple[str, str]:
    """The scorecard's value and verdict, for one clause or all three.

    Args:
        document: The document; read from `PATH` when None.
        clause: A clause name; "" joins every clause, worst verdict first.

    Returns:
        The value text and verdict; `not measured` when nothing is stored
        or the measurement is stale.
    """
    found = read(document)
    if found is None:
        return "no measurement", "not measured"
    chosen = [c for c in CLAUSES if not clause or c.name == clause]
    judged = [c.judge(found[c.name]) for c in chosen if c.name in found]
    if len(judged) < len(chosen):
        return "not every clause measured", "not measured"
    text = "; ".join(t for t, v in judged)
    if problems(found):
        return text + "; stale", "not measured"
    verdicts = {v for t, v in judged}
    for verdict in ("fail", "not measured"):
        if verdict in verdicts:
            return text, verdict
    return text, "pass"


def main(arguments: list[str] | None = None) -> int:
    """Writes, checks, or gates a release on tag protection.

    Args:
        arguments: Command-line arguments; None reads `sys.argv`.

    Returns:
        0 after `--write`; for `--check`, 0 when current; for
        `--protected`, 0 when the release tag is protected; otherwise 1.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--protected", metavar="OWNER/REPO")
    parser.add_argument("--tag", default="v0.0.0")
    options = parser.parse_args(arguments)
    if options.protected:
        verdict = attempt(
            lambda: protected(options.protected, options.tag, shell)
        )
        print(json.dumps(verdict, indent=2))
        return 0 if verdict["ok"] else 1
    if options.write:
        document = measure()
        PATH.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary(document)))
        return 0
    issues = problems()
    print(json.dumps({"ok": not issues, "problems": issues}, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
