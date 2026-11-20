"""Claim C8: an automated search for ways past the product's controls.

C8 says controls cannot be bypassed, decided by zero bypasses in a red-team
exercise run by people (stage D). This module is NOT that exercise. It is an
automated search, run by the product's own developers, that tries scripted
attacks through the public API (`scmle.api.project.Project`), the CLI
(`scmle ...` in a subprocess), the plugin registry and the files and
objects a run leaves behind: no monkeypatching, no private calls. It
is evidence for the row, and the row's verdict stays `not measured` however
the search comes out; a successful bypass is a product defect to report.

Each attempt targets one control while the condition it guards holds:

* `leakage.policy`: a copy of the target under `leakage_policy="block"`,
  renamed, relabelled, rescaled, split into two columns (XOR), as
  text or as a date; a misspelled policy; a CLI `--config` that tries to
  switch the policy; resuming a refused run, as is and with the policy
  edited in `config.json`.
* `budget.minimum_run`: budgets below the minimum run, as zero, negative,
  boolean, fractional, NaN or infinite values, through the CLI and by
  resuming.
* `data.unlabeled`: missing target values as NaN, None, `pandas.NA` and
  empty CSV cells.
* `policy.checkpoints`: a run rejected by a checkpoint, then resumed.
* `split.resident.open` and `confirm.single_use` (the exam sets): resuming a
  completed run, after deleting its evaluation and completion marker and
  after editing its configuration; rerunning a charter, into the same
  output folder, into another and after deleting its `.confirmations`;
  resuming a confirmed charter run.
* `charter.verify` (not in gates.toml; `sealing.read`): a signed
  charter edited, with and without its hash recomputed, with and without the
  key.
* `bundle.authenticate` (not in gates.toml; `Bundle.load`): unsigned and
  tampered bundles through `predict`, `package` and `serve`, with `--trusted`,
  `SCMLE_ENV`, an empty key and a missing key file; pickle payloads as the
  model (signed, stripped, in production) and as an unlisted member.
* `envelope.authenticate` (not in gates.toml; `scmle.envelope.read`): a
  deep-learning checkpoint carrying a pickle payload, forged, unsigned, in
  production and trusted without a key; a resumed run's signed stage
  products swapped for unsigned payloads.
* custody journal edits (`split.resident.journal`, `split.resident.reopen`,
  `policy.checkpoints`): a completed or rejected run's journal edited,
  truncated, rechained, restarted with a `started` entry, given a forged
  approval, deleted, or bypassed by copying the run or resuming it through
  a symlink.
* custody tokens (`split.resident.bind`, `split.resident.open`,
  `vault.bind`, `vault.open`): reuse, another set's token, rebinding, a
  missing or other binding and a reset `opened` flag, in process and in
  the vault.
* plugin stages (`split.resident.open`): a stage enrolled under kind
  `stages` that opens the locked set before selection.

A pickle payload or a plugin is judged by its effect (`witnessed`): a
payload that ran, or rows that were read, got through even when the call was
refused afterwards. An attempt that got through when it was added names the
product code at fault (`Attempt.defect`); tests/redteam turns every attempt
into a regression test, a strict xfail while its defect stands.

An attempt ends in one of four ways:

* `blocked`: the product refused (an exception, or a CLI exit other than 0),
  or the call finished without the attack's effect (`Held`);
* `override`: it got through only by a documented override, named on the
  attempt (`--trusted` on an unsigned bundle, `leakage_policy="warn"`);
* `bypass`: it got through with no documented override, so the control
  failed; each is a product defect;
* `error`: the attempt's own setup failed, so it decides nothing.

    python -m benchmarks.evidence.bypass --write
    python -m benchmarks.evidence.bypass --check
"""

import argparse
from collections.abc import Callable
from collections.abc import Mapping
from concurrent import futures
import contextlib
import dataclasses
import hashlib
import json
import multiprocessing
import os
import pathlib
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy
import pandas

from benchmarks import acceptance
from benchmarks.evidence import refusal
from scmle.artifacts import results
from scmle.assurance.detectors import certificates
from scmle.contracts import errors

PATH = pathlib.Path(__file__).with_suffix(".json")
ROOT = pathlib.Path(__file__).resolve().parents[2]
GATES = ROOT / "gates.toml"
VARIABLE = "bypasses_max"
# The claim's own criterion, "zero bypasses", used while acceptance.toml has
# no `bypasses_max` row.
BYPASSES = 0
SCOPE = (
    "automated search by the developers through the public API and CLI; "
    "not the red-team exercise by people that C8 requires (stage D)"
)
# Controls attempted that gates.toml does not register (they raise
# ArtifactError, not BlockedError or LeakageError).
UNREGISTERED = {
    "bundle.authenticate": "scmle.artifacts.bundle:Bundle.load",
    "envelope.authenticate": "scmle.envelope:read",
    "charter.verify": "scmle.assurance.charter.sealing:read",
}
OUTCOMES = ("blocked", "override", "bypass", "error")
# Policy: training budget of a run that should train, in seconds.
BUDGET = 60
# Policy: seconds before a CLI call is killed; `serve` that is still up
# then counts as served.
TIMEOUT = 120
WORKERS = 2
ROWS = 400
KEY = "bypass-search-key"
# Fast public options, as in tests/support.py: no search, strategies,
# ensembles or slow families; trials run in process.
FAST = (
    "--leaders",
    "0",
    "--exclude-families",
    "bagging",
    "deep_learning",
    "foundation",
    "--enforcement",
    "cooperative",
)
CHARTER = {
    "question": "Which customers churn within 30 days?",
    "estimand": {
        "kind": "predictive",
        "population": "active subscribers in 2024",
        "target": "churned within 30 days of the prediction moment",
        "prediction_moment": "first day of each billing month",
        "intercurrent_events": [
            {"event": "account closed", "strategy": "treat as outcome"}
        ],
        "summary_measure": "ROC AUC",
    },
    "unit": "customer-month",
    "deployment_population": "stated marginals: all active subscribers",
    "gating_checks": ["leakage.skill"],
    "decision_map": [
        {
            "check_id": "leakage.skill",
            "trigger": "lower bound above floor at the budgeted level",
            "action": "quarantine_column",
            "certificate_id": "cert.leakage.skill",
        }
    ],
    "sets": {"confirmation_fraction": 0.25, "embargo_rows": 0},
    "min_effect": 0.5,
    "k_max": 3,
    "power": 0.8,
    "error_budget": {
        "confirmation": {"alpha_confirm": "1/20"},
        "exploration": {"initial": "1/10", "wealth": "1/10"},
        "permutations_b": 20,
        "detector_tolerances": [],
    },
    "multiverse": [
        {
            "name": "imputation",
            "options": ["median", "mice"],
            "default": "median",
        }
    ],
    "seed": 7,
}


# What `scmle.cli.report` writes before a refusal's message: the logger,
# its context and the failure's label (`blocked`, or `<category> error`).
LABEL = re.compile(r"scmle\.cli \[[^\]]*\]: ([a-z]+)(?: error)?: ")


class Refused(Exception):  # noqa: N818 -- a refusal, not a fault
    """A CLI call that exited with a status other than 0.

    The control that refused it ran in another process, so only the label
    the CLI reported names it: `blocked` (a registered gate's
    `BlockedError` or `LeakageError`) or an error category such as
    `artifact` or `config`.

    Attributes:
        status: The exit status.
        label: The reported label; empty when the CLI reported none (a
            usage error or a defect's traceback).
    """

    def __init__(self, line: str, status: int) -> None:
        """Initializes the refusal.

        Args:
            line: The last line the CLI wrote to standard error.
            status: The exit status.
        """
        super().__init__(line[-400:])
        self.status = status
        found = LABEL.search(line)
        self.label = found.group(1) if found else ""

    @property
    def control(self) -> str:
        """The `stopped` name of the refusal: `cli.<label>`, else `Refused`."""
        return f"cli.{self.label}" if self.label else type(self).__name__


class Broken(Exception):  # noqa: N818 -- the attempt's own fault
    """An attempt's setup did not reach the state it attacks."""


class Held(Exception):  # noqa: N818 -- a control that held, not a fault
    """The call finished, but the attack's effect never happened.

    A pickle payload that was never unpickled, or a plugin that never saw
    the locked rows: the call itself may succeed while the control holds.
    """


class Payload:
    """A pickle that touches a marker file when it is unpickled.

    Unpickling it is running attacker code; the marker is the proof.

    Attributes:
        marker: The file it creates.
    """

    def __init__(self, marker: pathlib.Path) -> None:
        """Initializes the payload.

        Args:
            marker: The file to create when unpickled.
        """
        self.marker = marker

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Pickles as a call to `pathlib.Path.touch` on the marker.

        Returns:
            The callable and its arguments.
        """
        return (pathlib.Path.touch, (self.marker,))


@dataclasses.dataclass
class Lab:
    """Where one attempt works.

    Attributes:
        folder: The attempt's own empty folder.
        bundle: A signed model bundle the attempt may copy, or None.
    """

    folder: pathlib.Path
    bundle: pathlib.Path | None = None

    @property
    def marker(self) -> pathlib.Path:
        """The file a `Payload` creates when it runs."""
        return self.folder / "payload-ran"


@dataclasses.dataclass(frozen=True)
class Attempt:
    """One scripted attack on one control.

    Attributes:
        name: Unique name.
        gate: The gate attacked: a gates.toml name or an `UNREGISTERED` key.
        surface: `api` or `cli`.
        vector: What the attack does.
        prepare: Sets the attack up in a `Lab` and returns the one call the
            control should stop; the call returns a detail mapping when it
            gets through.
        override: The documented override the attack relies on, if any; a
            call that gets through with one is `override`, not `bypass`.
        control: True for the plain case every other attempt varies.
        defect: For an attack that got through when it was added, the
            product code that let it through; empty for one that was
            stopped. Its regression test (tests/redteam) is a strict xfail
            until the defect is fixed.
        also: Other controls documented to stop the attack before `gate`
            (defence in depth): gate names, or the `stopped` names of
            refusals outside the registry, such as `ValidationError` for
            the configuration schema.
    """

    name: str
    gate: str
    surface: str
    vector: str
    prepare: Callable[[Lab], Callable[[], dict[str, Any]]]
    override: str = ""
    control: bool = False
    defect: str = ""
    also: tuple[str, ...] = ()

    @property
    def stops(self) -> frozenset[str]:
        """The `stopped` names that show the attempt's control held.

        Its gate and `also`; through the CLI, whose refusal is reported
        from another process, the label the gate's error carries there:
        `cli.blocked` for a registered gate, `cli.artifact` for one listed
        in `UNREGISTERED`.
        """
        names = {self.gate, *self.also}
        if self.surface == "cli":
            label = "artifact" if self.gate in UNREGISTERED else "blocked"
            names.add(f"cli.{label}")
        return frozenset(names)


def table(rows: int = ROWS, seed: int = 0) -> pandas.DataFrame:
    """A clean binary task: three features and a logistic label `y`.

    Args:
        rows: Rows to draw.
        seed: Random seed.

    Returns:
        The table.
    """
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame(
        {name: rng.normal(size=rows) for name in ("a", "b", "c")}
    )
    noise = rng.normal(size=rows)
    frame["y"] = (frame.a + 0.5 * frame.b + noise > 0).astype(int)
    return frame


def options() -> dict[str, Any]:
    """Fast public `Project` options (see `FAST`).

    Returns:
        Keyword arguments for `Project`.
    """
    from scmle.contracts import tasks

    return {
        "search": tasks.Search(top_models=0),
        "strategies": tasks.Strategies(top_models=0),
        "ensemble": tasks.Stacking(size=0),
        "models": tasks.Candidates(
            exclude_families=["bagging", "deep_learning", "foundation"]
        ),
        "enforcement": "cooperative",
    }


def project(lab: Lab, frame: pandas.DataFrame, **settings: Any) -> Any:
    """A fast project writing into the lab.

    Args:
        lab: The attempt's lab.
        frame: The table; its target is `y`.
        **settings: Further `Project` arguments; they win over the defaults.

    Returns:
        The project.
    """
    import scmle.api.project

    chosen = {"budget": BUDGET, **options(), **settings}
    return scmle.api.project.Project(
        frame, "y", directory=lab.folder / "runs", **chosen
    )


def trained(result: Any) -> dict[str, Any]:
    """What a run that got through produced.

    Args:
        result: The completed run.

    Returns:
        Its selected model, its test metrics (a leak that was used shows as
        a near-perfect score) and its run directory.
    """
    path = pathlib.Path(result.directory) / "evaluation.json"
    evaluation = results.evaluation(
        json.loads(path.read_text(encoding="utf-8"))
    )
    return {
        "model": result.model,
        "test": evaluation.get("metrics"),
        "run": pathlib.Path(result.directory).name,
    }


def refused(call: Callable[[], Any]) -> pathlib.Path:
    """Runs a setup step that must be refused and returns its run folder.

    Args:
        call: The setup run.

    Returns:
        The refused run's directory.

    Raises:
        Broken: If the step was not refused, or left no run directory.
    """
    try:
        call()
    except (errors.BlockedError, errors.LeakageError) as error:
        folder = error.details.get("run_dir")
        if folder:
            return pathlib.Path(folder)
        raise Broken("the refusal names no run directory") from error
    raise Broken("setup expected a refusal and the run went through")


def cli(
    arguments: list[str],
    environment: dict[str, str] | None = None,
    folder: pathlib.Path | None = None,
) -> dict[str, Any]:
    """Runs the `scmle` CLI in a fresh process.

    Every `SCMLE_` variable of this process is dropped first, so only
    `environment` sets them.

    Args:
        arguments: Arguments after `scmle`.
        environment: `SCMLE_` variables to set.
        folder: Working directory.

    Returns:
        The exit status and the last line of standard error. A call still
        running after `TIMEOUT` (a server that came up) returns status None.

    Raises:
        Refused: If the CLI exits with a status other than 0.
    """
    variables = {
        k: v for k, v in os.environ.items() if not k.startswith("SCMLE_")
    }
    variables.update(environment or {})
    variables["OMP_NUM_THREADS"] = "1"
    command = [sys.executable, "-m", "scmle.cli", *arguments]
    try:
        done = subprocess.run(
            command,
            cwd=folder,
            env=variables,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": None, "note": f"still running after {TIMEOUT} s"}
    lines = done.stderr.strip().splitlines() or [""]
    if done.returncode != 0:
        raise Refused(lines[-1], done.returncode)
    return {"status": 0, "note": lines[-1][-200:]}


@contextlib.contextmanager
def environ(**values: str | None) -> Any:
    """Sets or clears environment variables for the duration of a block.

    Args:
        **values: Variable to value; None clears it.

    Yields:
        Nothing.
    """
    saved = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def leaked(form: str) -> pandas.DataFrame:
    """The clean table plus a leak of the target in the given form.

    Args:
        form: One of `copy`, `renamed`, `labels`, `affine`, `xor`, `text`
            or `date`; each determines the target exactly, so a model that
            uses it scores perfectly on the test rows.

    Returns:
        The table with the leak.

    Raises:
        InputError: For an unknown form.
    """
    frame = table()
    rng = numpy.random.default_rng(1)
    target = frame.y
    if form == "copy":
        frame["leak"] = target
    elif form == "renamed":
        frame["tenure_months"] = target
    elif form == "labels":
        frame["status"] = target.map({0: "active", 1: "churned"})
    elif form == "affine":
        frame["score"] = 3 * target + 7 + rng.normal(0, 0.01, len(frame))
    elif form == "xor":
        mask = rng.integers(0, 2, len(frame))
        frame["p"] = mask
        frame["q"] = mask ^ target
    elif form == "text":
        words = target.map({0: "renewed the plan", 1: "closed the account"})
        frame["note"] = [
            f"ticket {i}: customer {w}" for i, w in enumerate(words)
        ]
    elif form == "date":
        days = rng.integers(0, 300, len(frame)) + 365 * target
        start = pandas.Timestamp("2020-01-01")
        frame["closed_on"] = start + pandas.to_timedelta(days, unit="D")
    else:
        raise errors.InputError(f"unknown leak form {form!r}")
    return frame


def leak(form: str) -> Callable[[Lab], Callable[[], dict[str, Any]]]:
    """An attempt that trains on a leak under the block policy.

    Args:
        form: The form of the leak (see `leaked`).

    Returns:
        The attempt's `prepare`.
    """

    def prepare(lab: Lab) -> Callable[[], dict[str, Any]]:
        frame = leaked(form)
        return lambda: trained(project(lab, frame, leakage="block").run())

    return prepare


def misspelled(lab: Lab) -> Callable[[], dict[str, Any]]:
    """A leak under `leakage_policy="BLOCK"`, hoping it reads as warn."""
    frame = leaked("copy")
    return lambda: trained(project(lab, frame, leakage="BLOCK").run())


def written(lab: Lab, frame: pandas.DataFrame) -> pathlib.Path:
    """Writes the table as CSV in the lab.

    Args:
        lab: The attempt's lab.
        frame: The table.

    Returns:
        The CSV path.
    """
    path = lab.folder / "data.csv"
    frame.to_csv(path, index=False)
    return path


def command(lab: Lab, data: pathlib.Path, *extra: str) -> list[str]:
    """Arguments of a fast `scmle run` on the lab's CSV.

    Args:
        lab: The attempt's lab.
        data: The CSV.
        *extra: Further arguments.

    Returns:
        The arguments after `scmle`.
    """
    return [
        "run",
        str(data),
        "--target",
        "y",
        "--output-dir",
        str(lab.folder / "runs"),
        *FAST,
        *extra,
    ]


def clileak(lab: Lab) -> Callable[[], dict[str, Any]]:
    """A renamed leak through `scmle run --leakage block`."""
    data = written(lab, leaked("renamed"))
    arguments = command(lab, data, "--leakage", "block")
    return lambda: cli(arguments, folder=lab.folder)


def overlay(lab: Lab) -> Callable[[], dict[str, Any]]:
    """A `--config` file that tries to set the policy back to warn."""
    data = written(lab, leaked("renamed"))
    config = lab.folder / "config.json"
    config.write_text(json.dumps({"leakage": "warn"}), "utf-8")
    arguments = command(
        lab, data, "--leakage", "block", "--config", str(config)
    )
    return lambda: cli(arguments, folder=lab.folder)


def rehash(run: pathlib.Path) -> None:
    """Makes a run folder's saved files agree with an edited config.json.

    Recomputes the configuration hash the parameter-estimation declaration
    carries, its digest and its lock, as anyone editing the folder can.

    Args:
        run: The run directory.
    """
    from scmle.contracts import tasks

    config = tasks.Config.model_validate_json(
        (run / "config.json").read_text(encoding="utf-8")
    )
    path = run / "parameter_estimation_declaration.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record.pop("digest", None)
    record["project_config_hash"] = config.config_hash()
    digest = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps({**record, "digest": digest}), "utf-8")
    lock = run / "parameter_estimation_declaration.lock"
    lock.write_text(f"{digest}\n", encoding="utf-8")


def edit(run: pathlib.Path, change: Callable[[dict[str, Any]], None]) -> None:
    """Edits a run's config.json and rehashes what depends on it.

    Args:
        run: The run directory.
        change: Changes the parsed configuration in place.
    """
    path = run / "config.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    change(record)
    path.write_text(json.dumps(record), encoding="utf-8")
    rehash(run)


def resume(
    run: pathlib.Path,
    frame: pandas.DataFrame,
    checkpoints: tuple[Any, ...] = (),
) -> dict[str, Any]:
    """Resumes a run through `Project.resume`.

    Args:
        run: The run directory.
        frame: The original in-memory table.
        checkpoints: Reviewers consulted on resume, as for `Project`.

    Returns:
        What the resumed run produced.
    """
    import scmle.api.project

    return trained(
        scmle.api.project.Project.resume(
            run, data=frame, checkpoints=checkpoints
        )
    )


def leakresume(lab: Lab) -> Callable[[], dict[str, Any]]:
    """Resumes a run the leakage policy refused, unchanged."""
    frame = leaked("copy")
    run = refused(lambda: project(lab, frame, leakage="block").run())
    return lambda: resume(run, frame)


def leakedit(lab: Lab) -> Callable[[], dict[str, Any]]:
    """Resumes a refused leak run after setting the policy to warn."""
    frame = leaked("copy")
    run = refused(lambda: project(lab, frame, leakage="block").run())
    edit(run, lambda c: c.update(leakage="warn"))
    return lambda: resume(run, frame)


def budget(value: Any) -> Callable[[Lab], Callable[[], dict[str, Any]]]:
    """An attempt that runs with a budget below the minimum run.

    Args:
        value: The `time_budget` given.

    Returns:
        The attempt's `prepare`.
    """

    def prepare(lab: Lab) -> Callable[[], dict[str, Any]]:
        frame = table()
        return lambda: trained(project(lab, frame, budget=value).run())

    return prepare


def clibudget(lab: Lab) -> Callable[[], dict[str, Any]]:
    """A two-second budget through `scmle run --budget 2`."""
    data = written(lab, table())
    arguments = command(lab, data, "--budget", "2")
    return lambda: cli(arguments, folder=lab.folder)


def budgetresume(lab: Lab) -> Callable[[], dict[str, Any]]:
    """Resumes a run the budget refused, unchanged."""
    frame = table()
    run = refused(lambda: project(lab, frame, budget=1).run())
    return lambda: resume(run, frame)


def unlabeled(kind: str) -> Callable[[Lab], Callable[[], dict[str, Any]]]:
    """An attempt that trains with missing labels under `unlabeled=block`.

    Args:
        kind: `nan` (float NaN), `none` (None in an object column) or `na`
            (`pandas.NA` in a nullable integer column).

    Returns:
        The attempt's `prepare`.
    """

    def prepare(lab: Lab) -> Callable[[], dict[str, Any]]:
        frame = table()
        if kind == "none":
            frame["y"] = frame.y.astype(object)
            frame.loc[:19, "y"] = None
        elif kind == "na":
            frame["y"] = frame.y.astype("Int64")
            frame.loc[:19, "y"] = pandas.NA
        else:
            frame["y"] = frame.y.astype(float)
            frame.loc[:19, "y"] = numpy.nan
        return lambda: trained(project(lab, frame).run())

    return prepare


def cliunlabeled(lab: Lab) -> Callable[[], dict[str, Any]]:
    """Empty target cells in a CSV through `scmle run`."""
    frame = table().astype({"y": object})
    frame.loc[:19, "y"] = ""
    data = written(lab, frame)
    return lambda: cli(command(lab, data), folder=lab.folder)


def deny(review: Any) -> bool:
    """Rejects a checkpoint review.

    Args:
        review: What the checkpoint is shown; ignored.

    Returns:
        False.
    """
    del review
    return False


def grant(review: Any) -> bool:
    """Approves a checkpoint review.

    Args:
        review: What the checkpoint is shown; ignored.

    Returns:
        True.
    """
    del review
    return True


def rejecter() -> Any:
    """A checkpoint that rejects every review.

    Returns:
        The checkpoint.
    """
    from scmle.pipeline import policies

    return policies.Callback(deny)


def checkpoint(lab: Lab) -> Callable[[], dict[str, Any]]:
    """A run whose checkpoint rejects the plan."""
    frame = table()
    reviewers = (rejecter(),)
    return lambda: trained(project(lab, frame, checkpoints=reviewers).run())


def checkresume(lab: Lab) -> Callable[[], dict[str, Any]]:
    """Resumes a run a checkpoint rejected (resume takes no checkpoints)."""
    frame = table()
    reviewers = (rejecter(),)
    run = refused(lambda: project(lab, frame, checkpoints=reviewers).run())
    return lambda: resume(run, frame)


def finished(lab: Lab, frame: pandas.DataFrame, **settings: Any) -> Any:
    """A completed fast run.

    Args:
        lab: The attempt's lab.
        frame: The table.
        **settings: Further `Project` arguments.

    Returns:
        The run directory and its evaluation.
    """
    run = pathlib.Path(project(lab, frame, **settings).run().directory)
    evaluation = results.evaluation(
        json.loads((run / "evaluation.json").read_text("utf-8"))
    )
    return run, evaluation


def reopened(
    run: pathlib.Path, frame: pandas.DataFrame, before: dict[str, Any]
) -> dict[str, Any]:
    """Resumes a completed run and reports the second test opening.

    Args:
        run: The run directory.
        frame: The table.
        before: The first evaluation.

    Returns:
        The model and test metrics of both openings.
    """
    result = resume(run, frame)
    after = results.evaluation(
        json.loads((run / "evaluation.json").read_text("utf-8"))
    )
    return {
        **result,
        "first": [before["selected"]["model"], before["metrics"]],
        "second": [after["selected"]["model"], after["metrics"]],
    }


def examdone(lab: Lab) -> Callable[[], dict[str, Any]]:
    """Resumes a completed run as it is."""
    frame = table()
    run, _ = finished(lab, frame)
    return lambda: resume(run, frame)


def examagain(lab: Lab) -> Callable[[], dict[str, Any]]:
    """Deletes a completed run's evaluation and marker and resumes it."""
    frame = table()
    run, before = finished(lab, frame)
    (run / "evaluation.json").unlink()
    (run / "complete.json").unlink()
    return lambda: reopened(run, frame, before)


def examedit(lab: Lab) -> Callable[[], dict[str, Any]]:
    """Excludes the winner in config.json and resumes the completed run.

    The saved plan, evaluation and completion marker are deleted and the
    declaration rehashed, so a different model is tested on the same locked
    rows.
    """
    frame = table()
    run, before = finished(lab, frame)
    winner = before["selected"]["model"]
    edit(run, lambda c: c["models"]["exclude"]["models"].append(winner))
    (run / "evaluation.json").unlink()
    (run / "complete.json").unlink()
    (run / "execution_plan.json").unlink(missing_ok=True)
    return lambda: reopened(run, frame, before)


def charter(lab: Lab, signed: bool = True) -> pathlib.Path:
    """Writes the charter and, when asked, signs it with the CLI.

    Args:
        lab: The attempt's lab.
        signed: Whether to sign it with `KEY`.

    Returns:
        The charter file.
    """
    bare = lab.folder / "charter.json"
    bare.write_text(json.dumps(CHARTER), encoding="utf-8")
    if not signed:
        return bare
    key = lab.folder / "key.txt"
    key.write_text(KEY, encoding="utf-8")
    path = lab.folder / "signed.json"
    cli(
        [
            "charter",
            "sign",
            str(bare),
            "--output",
            str(path),
            "--key-file",
            str(key),
        ]
    )
    return path


def field(document: dict[str, Any], *keys: str) -> Any:
    """Reads a stored record key under its current name or an older one.

    Stage B renamed stored record keys (docs/RECORD_KEYS.md); a dotted key
    names a nested object. The first key present wins.

    Args:
        document: The parsed record.
        *keys: Dotted keys, the current name first.

    Returns:
        The value at the first key present.

    Raises:
        KeyError: If none of the keys is present.
    """
    for key in keys:
        value: Any = document
        try:
            for part in key.split("."):
                value = value[part]
        except (KeyError, TypeError):
            continue
        return value
    raise KeyError(keys[-1])


def confirmation(run: pathlib.Path) -> list[int]:
    """The confirmation rows of a charter run.

    Args:
        run: The run directory.

    Returns:
        The row ids of its confirmation set.
    """
    manifest = json.loads((run / "split_manifest.json").read_text("utf-8"))
    return list(field(manifest, "sets.confirmation", "confirmation_ids"))


def rerun(change: str) -> Callable[[Lab], Callable[[], dict[str, Any]]]:
    """An attempt that runs the same signed charter on the same data again.

    Args:
        change: `same` (into the same output folder), `elsewhere` (into
            another output folder) or `forgotten` (into the same folder
            after deleting its record of spent confirmation sets,
            `.confirmations`).

    Returns:
        The attempt's `prepare`; the call reports whether the second run
        tested the same confirmation rows.
    """

    def prepare(lab: Lab) -> Callable[[], dict[str, Any]]:
        frame = table(500)
        path = charter(lab)
        with environ(SCMLE_SIGNING_KEY=KEY):
            first = project(lab, frame, charter=path).run()
        rows = confirmation(pathlib.Path(first.directory))
        second = lab
        if change == "elsewhere":
            second = Lab(lab.folder / "elsewhere", lab.bundle)
            second.folder.mkdir()
        elif change == "forgotten":
            shutil.rmtree(lab.folder / "runs" / ".confirmations")
        elif change != "same":
            raise Broken(f"unknown rerun change {change!r}")

        def act() -> dict[str, Any]:
            with environ(SCMLE_SIGNING_KEY=KEY):
                result = project(second, frame, charter=path).run()
            run = pathlib.Path(result.directory)
            claims = json.loads(
                (run / "confirmed_claims.json").read_text("utf-8")
            )
            return {
                **trained(result),
                "same_rows": confirmation(run) == rows,
                "outcome": claims.get("outcome"),
            }

        return act

    return prepare


def charterresume(lab: Lab) -> Callable[[], dict[str, Any]]:
    """Deletes a confirmed charter run's evaluation and marker, resumes it."""
    frame = table(500)
    path = charter(lab)
    with environ(SCMLE_SIGNING_KEY=KEY):
        run = pathlib.Path(project(lab, frame, charter=path).run().directory)
    (run / "evaluation.json").unlink()
    (run / "complete.json").unlink()

    def act() -> dict[str, Any]:
        with environ(SCMLE_SIGNING_KEY=KEY):
            return resume(run, frame)

    return act


def hashed(lab: Lab, record: dict[str, Any]) -> str:
    """The charter hash the CLI reports for a bare charter.

    Args:
        lab: The attempt's lab.
        record: The bare charter.

    Returns:
        The 64-character hash `scmle charter check` logs.

    Raises:
        Broken: If the CLI reports no hash.
    """
    path = lab.folder / "edited.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    note = cli(["charter", "check", str(path)])["note"]
    found = re.search(r"\b[0-9a-f]{64}\b", note)
    if found is None:
        raise Broken(f"no charter hash in {note!r}")
    return found.group(0)


def tampered(lab: Lab, recompute: bool) -> pathlib.Path:
    """A signed charter whose confirmation alpha was relaxed afterwards.

    Args:
        lab: The attempt's lab.
        recompute: Whether to recompute the stored hash with the CLI.

    Returns:
        The edited charter file; its signature is the original one.
    """
    path = charter(lab)
    record = json.loads(path.read_text(encoding="utf-8"))
    budget = field(record["charter"], "budget", "error_budget")
    budget["confirmation"]["alpha_confirm"] = "1/2"
    if recompute:
        stored = hashed(lab, record["charter"])
        if "charter_hash" in record:
            record["charter_hash"] = stored
        else:
            record["digests"]["charter"] = stored
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def charteredit(
    recompute: bool, key: str | None
) -> Callable[[Lab], Callable[[], dict[str, Any]]]:
    """An attempt that runs an edited signed charter.

    Args:
        recompute: Whether the stored hash is recomputed after the edit.
        key: The signing key in the environment, or None for none.

    Returns:
        The attempt's `prepare`.
    """

    def prepare(lab: Lab) -> Callable[[], dict[str, Any]]:
        frame = table(500)
        path = tampered(lab, recompute)

        def act() -> dict[str, Any]:
            with environ(SCMLE_SIGNING_KEY=key):
                return trained(project(lab, frame, charter=path).run())

        return act

    return prepare


def sha(path: pathlib.Path) -> str:
    """SHA-256 of a file.

    Args:
        path: The file.

    Returns:
        Its hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bundle(lab: Lab, change: str) -> pathlib.Path:
    """A copy of the signed bundle, changed as asked.

    Args:
        lab: The attempt's lab; `lab.bundle` is the signed original.
        change: `none`; `bytes` (a byte appended to novelty.joblib);
            `rehash` (also its manifest checksum updated); `strip`
            (`rehash` and signature.json deleted); `algorithm` (the
            signature's algorithm set to `none`); `payload` (the model
            replaced by a `Payload` and its checksum updated);
            `payload_strip` (`payload` and signature.json deleted);
            `payload_unlisted` (a `Payload` added as calibrator.joblib,
            which the signed manifest does not list).

    Returns:
        The copy.

    Raises:
        Broken: When the lab has no bundle.
    """
    if lab.bundle is None:
        raise Broken("no signed bundle was built")
    target = lab.folder / "bundle"
    shutil.copytree(lab.bundle, target)
    member = target / "novelty.joblib"
    manifest = target / "model_manifest.json"
    signature = target / "signature.json"
    if change in {"bytes", "rehash", "strip"}:
        member.write_bytes(member.read_bytes() + b"\0")
    if change in {"rehash", "strip"}:
        record = json.loads(manifest.read_text(encoding="utf-8"))
        record["files"][member.name] = sha(member)
        manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if change == "strip":
        signature.unlink()
    if change == "algorithm":
        record = json.loads(signature.read_text(encoding="utf-8"))
        record["algorithm"] = "none"
        signature.write_text(json.dumps(record), encoding="utf-8")
    if change in {"payload", "payload_strip"}:
        pipeline = target / "pipeline.joblib"
        pipeline.write_bytes(pickle.dumps(Payload(lab.marker)))
        record = json.loads(manifest.read_text(encoding="utf-8"))
        record["files"][pipeline.name] = sha(pipeline)
        manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if change == "payload_strip":
        signature.unlink()
    if change == "payload_unlisted":
        unlisted = target / "calibrator.joblib"
        unlisted.write_bytes(pickle.dumps(Payload(lab.marker)))
    return target


def load(
    change: str,
    environment: dict[str, str],
    trusted: bool = False,
    verb: str = "predict",
) -> Callable[[Lab], Callable[[], dict[str, Any]]]:
    """An attempt that loads a bundle through the CLI.

    Args:
        change: How the bundle is changed (see `bundle`).
        environment: `SCMLE_` variables for the call.
        trusted: Whether to pass `--trusted`.
        verb: `predict`, `package` or `serve`.

    Returns:
        The attempt's `prepare`.
    """

    def prepare(lab: Lab) -> Callable[[], dict[str, Any]]:
        target = bundle(lab, change)
        arguments = [verb, str(target)]
        if verb == "predict":
            data = written(lab, table().drop(columns="y"))
            arguments += [str(data), "--output", str(lab.folder / "out.csv")]
        elif verb == "package":
            arguments += ["--output", str(lab.folder / "service")]
        else:
            arguments += ["--port", "0"]
        if trusted:
            arguments.append("--trusted")

        def call() -> dict[str, Any]:
            return cli(arguments, environment, lab.folder)

        if change.startswith("payload"):
            return witnessed(call, ran(lab))
        return call

    return prepare


def witnessed(
    call: Callable[[], dict[str, Any]],
    evidence: Callable[[], dict[str, Any] | None],
) -> Callable[[], dict[str, Any]]:
    """An attack judged by its effect, not by how the call ended.

    A pickle payload that ran before the load was refused, or a plugin that
    read the locked rows before the run was refused, got through: the
    refusal came too late.

    Args:
        call: The attacked call.
        evidence: What the attack achieved, or None when it achieved
            nothing.

    Returns:
        The attempt's call: it returns the evidence (with the call's detail
        or refusal), re-raises the refusal when there is no evidence, and
        raises `Held` when the call finished without the effect.
    """

    def act() -> dict[str, Any]:
        try:
            detail = call()
        except Exception as error:
            found = evidence()
            if found is None:
                raise
            return {**found, "refused_after": str(error)[:200]}
        found = evidence()
        if found is None:
            raise Held("the call finished and the attack had no effect")
        return {**found, **detail}

    return act


def ran(lab: Lab) -> Callable[[], dict[str, Any] | None]:
    """Whether the lab's `Payload` ran.

    Args:
        lab: The attempt's lab.

    Returns:
        A check returning `{"payload": "ran"}` once the marker exists, else
        None.
    """
    return lambda: {"payload": "ran"} if lab.marker.exists() else None


def wipe(run: pathlib.Path) -> None:
    """Deletes a completed run's evaluation and completion marker.

    Args:
        run: The run directory.
    """
    (run / "evaluation.json").unlink()
    (run / "complete.json").unlink()


def ledger(run: pathlib.Path) -> pathlib.Path:
    """The custody journal of a run directory.

    Args:
        run: The run directory.

    Returns:
        The journal file, beside the run folder.
    """
    from scmle.assurance.custody import journal

    return journal.Journal.locate(run).path


def entries(run: pathlib.Path) -> list[dict[str, Any]]:
    """The lines of a run's custody journal, parsed.

    Args:
        run: The run directory.

    Returns:
        The entries, oldest first.
    """
    text = ledger(run).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def chain(
    run: pathlib.Path,
    keep: Callable[[dict[str, Any]], bool] | None = None,
    extra: tuple[dict[str, Any], ...] = (),
) -> None:
    """Rewrites a run's custody journal with its hash chain recomputed.

    The chain is plain SHA-256 of canonical JSON with no key, so anyone who
    can write the file can recompute it; no secret is used here.

    Args:
        run: The run directory.
        keep: Which existing entries stay; all of them when None.
        extra: Entries appended after them (kind and fields; the run, time
            and chain are filled in).
    """
    from scmle.assurance.custody import journal

    found = entries(run)
    stamp = found[-1]["time"]
    previous = journal.GENESIS
    lines = []
    for entry in (
        *(e for e in found if keep is None or keep(e)),
        *({"run": run.name, "time": stamp, **e} for e in extra),
    ):
        body = {
            k: v for k, v in entry.items() if k not in {"digest", "previous"}
        }
        body["previous"] = previous
        text = json.dumps(body, sort_keys=True, separators=(",", ":"))
        previous = hashlib.sha256(text.encode("utf-8")).hexdigest()
        lines.append(json.dumps({**body, "digest": previous}))
    ledger(run).write_text("\n".join(lines) + "\n", encoding="utf-8")


def restart(run: pathlib.Path) -> tuple[dict[str, Any], ...]:
    """A fresh `started` entry, as a run id reused after deletion writes.

    Args:
        run: The run directory.

    Returns:
        The entry to append, with the configuration hash the run started
        with.
    """
    first = entries(run)[0]
    return ({"kind": "started", "config": first.get("config", "")},)


def examjournal(change: str) -> Callable[[Lab], Callable[[], dict[str, Any]]]:
    """An attempt that reopens a completed run's test set past its journal.

    Args:
        change: What is done to the custody record after the evaluation and
            completion marker are deleted: `restart` (a chained `started`
            entry appended), `rechain` (the `opened` entry removed and the
            chain recomputed), `truncate` (the `opened` line, the last one,
            removed and nothing recomputed), `edit` (the first entry's
            configuration hash changed, the chain left broken), `delete`
            (the journal deleted), `moved` (the run copied to a new name),
            `symlink` (the run resumed through a link in
            another folder) or `winner` (the winner excluded in config.json,
            the declaration rehashed, then `restart`).

    Returns:
        The attempt's `prepare`.
    """

    def prepare(lab: Lab) -> Callable[[], dict[str, Any]]:
        frame = table()
        run, before = finished(lab, frame)
        wipe(run)
        if change == "restart":
            chain(run, extra=restart(run))
        elif change == "rechain":
            chain(run, lambda e: e["kind"] != "opened")
        elif change == "truncate":
            path = ledger(run)
            kept = [
                line
                for line in path.read_text(encoding="utf-8").splitlines()
                if json.loads(line)["kind"] != "opened"
            ]
            path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        elif change == "edit":
            found = entries(run)
            found[0]["config"] = "0" * 16
            lines = [json.dumps(e) for e in found]
            ledger(run).write_text("\n".join(lines) + "\n", encoding="utf-8")
        elif change == "delete":
            ledger(run).unlink()
        elif change == "moved":
            copy = run.with_name(f"{run.name}_copy")
            shutil.copytree(run, copy)
            run = copy
        elif change == "symlink":
            elsewhere = lab.folder / "elsewhere"
            elsewhere.mkdir()
            link = elsewhere / run.name
            link.symlink_to(run, target_is_directory=True)
            run = link
        elif change == "winner":
            winner = before["selected"]["model"]
            edit(run, lambda c: c["models"]["exclude"]["models"].append(winner))
            (run / "execution_plan.json").unlink(missing_ok=True)
            chain(run, extra=restart(run))
        else:
            raise Broken(f"unknown journal change {change!r}")
        return lambda: reopened(run, frame, before)

    return prepare


def checkjournal(change: str) -> Callable[[Lab], Callable[[], dict[str, Any]]]:
    """An attempt that lifts a checkpoint's rejection in the journal.

    Args:
        change: `approve` (a chained approval by `attacker` appended for
            every rejected gate), `rechain` (the verdicts removed and the
            chain recomputed) or `restart` (a chained `started` entry
            appended).

    Returns:
        The attempt's `prepare`.
    """

    def prepare(lab: Lab) -> Callable[[], dict[str, Any]]:
        frame = table()
        reviewers = (rejecter(),)
        run = refused(lambda: project(lab, frame, checkpoints=reviewers).run())
        if change == "approve":
            approvals = tuple(
                {
                    "kind": "decided",
                    "gate": e["gate"],
                    "approved": True,
                    "reviewer": "attacker",
                    "reason": "",
                }
                for e in entries(run)
                if e["kind"] == "decided" and not e["approved"]
            )
            chain(run, extra=approvals)
        elif change == "rechain":
            chain(run, lambda e: e["kind"] != "decided")
        elif change == "restart":
            chain(run, extra=restart(run))
        else:
            raise Broken(f"unknown journal change {change!r}")
        return lambda: resume(run, frame)

    return prepare


def forge(
    path: pathlib.Path, meta: dict[str, Any], payload: bytes, mac: str | None
) -> None:
    """Writes an envelope (`scmle.envelope`) with a chosen MAC, no key used.

    Args:
        path: The file.
        meta: The header's metadata.
        payload: The pickled bytes; their digest is stated correctly.
        mac: A forged hex MAC, or None for an unsigned envelope.
    """
    import scmle.envelope

    header = {
        "meta": meta,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "mac": mac,
    }
    line = json.dumps(header, sort_keys=True).encode("utf-8") + b"\n"
    path.write_bytes(scmle.envelope.MAGIC + line + payload)


def metadata(path: pathlib.Path) -> dict[str, Any]:
    """The metadata of an envelope file.

    Args:
        path: The envelope.

    Returns:
        Its header's `meta`, read without authenticating it.
    """
    import scmle.envelope

    data = path.read_bytes()
    start = len(scmle.envelope.MAGIC)
    header = json.loads(data[start : data.index(b"\n", start)])
    found: dict[str, Any] = header["meta"]
    return found


def restored(
    mac: str | None, environment: Mapping[str, str | None], trusted: bool
) -> Callable[[Lab], Callable[[], dict[str, Any]]]:
    """An attempt that restores a checkpoint carrying a pickle payload.

    Args:
        mac: The envelope's MAC (see `envelope`).
        environment: Variables set for the call; None clears one.
        trusted: Whether `restore` is called with `trusted=True`.

    Returns:
        The attempt's `prepare`; the attack got through when the payload
        ran.
    """

    def prepare(lab: Lab) -> Callable[[], dict[str, Any]]:
        from scmle.models.deeplearning import estimators

        path = lab.folder / "checkpoint.envelope"
        version = estimators.NetworkEstimator.CHECKPOINT_SCHEMA_VERSION
        payload = pickle.dumps({"estimator": Payload(lab.marker)})
        forge(path, {"schema_version": version}, payload, mac)

        def call() -> dict[str, Any]:
            with environ(**environment):
                estimators.TorchClassifier.restore(path, trusted=trusted)
            return {}

        return witnessed(call, ran(lab))

    return prepare


def products(lab: Lab) -> Callable[[], dict[str, Any]]:
    """Swaps a refused run's signed stage products for unsigned payloads.

    The run, signed with the key, is rejected at selection, so its plan,
    search and choice are kept as signed envelopes; each is replaced by an
    unsigned envelope of a `Payload`, with the same metadata, and the run
    resumed with the key set and a checkpoint that approves selection, so
    the rejection does not stop the resume before the products are read.

    Raises:
        Broken: From the call, when the resume was stopped by anything but
            `envelope.authenticate` before the payload ran, or finished
            without reading a stage product.
    """
    from scmle.pipeline import policies
    from scmle.runtime import state

    frame = table()
    gates = frozenset({state.Progress.SELECTING})
    reviewers = (policies.Callback(deny, gates),)
    approvers = (policies.Callback(grant, gates),)
    with environ(SCMLE_SIGNING_KEY=KEY):
        run = refused(lambda: project(lab, frame, checkpoints=reviewers).run())
    kept = sorted((run / "stages").glob("*.product"))
    if not kept:
        raise Broken("the refused run kept no stage product")
    payload = pickle.dumps(Payload(lab.marker))
    for path in kept:
        forge(path, metadata(path), payload, None)
    gate = "envelope.authenticate"
    site = {UNREGISTERED[gate]: gate}

    def call() -> dict[str, Any]:
        with environ(SCMLE_SIGNING_KEY=KEY), observed(kept) as seen:
            try:
                detail = resume(run, frame, approvers)
            except Exception as error:
                if (
                    lab.marker.exists()
                    or refusal.locate(error, site)[0] == gate
                ):
                    raise
                raise Broken(
                    f"the resume was stopped before it read a stage product: "
                    f"{type(error).__name__}: {str(error)[:200]}"
                ) from error
        if lab.marker.exists():
            return detail
        if not seen:
            raise Broken("the resume never read a swapped stage product")
        # The loader handles the refusal by running the stage again, so the
        # resume finishes; the refusal it handled is the control's record.
        raise seen[0]

    return witnessed(call, ran(lab))


@contextlib.contextmanager
def observed(paths: list[pathlib.Path]) -> Any:
    """Records the refusals `scmle.envelope.read` raises for some files.

    Every call goes through to the real `read` unchanged; the refusals it
    raises for `paths` are kept, so an attack whose refusal the caller
    handles still shows which control refused it.

    Args:
        paths: The files to watch.

    Yields:
        The `ArtifactError`s raised for them, in order.
    """
    import scmle.envelope

    original = scmle.envelope.read
    watched = {path.resolve() for path in paths}
    seen: list[errors.ArtifactError] = []

    def read(path: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return original(path, *args, **kwargs)
        except errors.ArtifactError as error:
            if pathlib.Path(path).resolve() in watched:
                seen.append(error)
            raise

    scmle.envelope.read = read
    try:
        yield seen
    finally:
        scmle.envelope.read = original


def stripped(lab: Lab) -> Callable[[], dict[str, Any]]:
    """Relaxes a signed charter, recomputes its hash, drops its signature."""
    frame = table(500)
    path = tampered(lab, recompute=True)
    record = json.loads(path.read_text(encoding="utf-8"))
    record.pop("signature", None)
    path.write_text(json.dumps(record), encoding="utf-8")

    def act() -> dict[str, Any]:
        with environ(SCMLE_SIGNING_KEY=KEY):
            return trained(project(lab, frame, charter=path).run())

    return act


def lockbox(sealed: bool) -> Any:
    """A locked test set of 40 rows, in this process or in a vault.

    Args:
        sealed: Whether the rows live in a vault process.

    Returns:
        The lockbox.
    """
    frame = table(40)
    features = frame.drop(columns="y")
    if sealed:
        from scmle.assurance.custody import vault

        return vault.SealedTestSet(features, frame.y, frame.index)
    import scmle.data.splitting.splitters.lockbox

    return scmle.data.splitting.splitters.lockbox.Resident(
        features, frame.y, tuple(frame.index)
    )


TOKENS = (
    "reuse",
    "reset",
    "cross",
    "rebind",
    "unbound",
    "mismatch",
    "unbinding",
)


def token(
    change: str, sealed: bool
) -> Callable[[Lab], Callable[[], dict[str, Any]]]:
    """An attempt on a locked test set's one-time release.

    Args:
        change: `reuse` (open twice with the token), `reset` (open, set
            `opened` back to False, open again), `cross` (open with another
            set's token), `rebind` (open, then bind again), `unbound` (open
            before any binding), `mismatch` (open with another binding) or
            `unbinding` (open with no binding after one was set).
        sealed: Whether the rows live in a vault process.

    Returns:
        The attempt's `prepare`; the call reports the rows it obtained.
    """

    def prepare(lab: Lab) -> Callable[[], dict[str, Any]]:
        del lab
        if change not in TOKENS:
            raise Broken(f"unknown token change {change!r}")
        box = lockbox(sealed)
        other = lockbox(sealed) if change == "cross" else box
        if change != "unbound":
            box.bind("frozen")
        if other is not box:
            other.bind("frozen")
        if change in {"reuse", "reset", "rebind"}:
            box.open(box.mint(), "frozen")
        if change == "reset":
            box.opened = False
        attacks: dict[str, Callable[[], Any]] = {
            "reuse": lambda: box.open(box.mint(), "frozen"),
            "reset": lambda: box.open(box.mint(), "frozen"),
            "cross": lambda: box.open(other.mint(), "frozen"),
            "rebind": lambda: box.bind("another"),
            "unbound": lambda: box.open(box.mint(), "frozen"),
            "mismatch": lambda: box.open(box.mint(), "another"),
            "unbinding": lambda: box.open(box.mint(), None),
        }

        def act() -> dict[str, Any]:
            try:
                rows = attacks[change]()
            finally:
                box.close()
                other.close()
            if rows is None:
                return {"bound": "another"}
            return {"rows": len(rows[1])}

        return act

    return prepare


def plugin(lab: Lab) -> Callable[[], dict[str, Any]]:
    """Enrols a plugin stage that opens the locked test set while planning.

    The stage runs right after planning, before any model is trained or
    selected; it binds the locked set to its own digest and asks for the
    rows. The plugin is withdrawn when the run ends.
    """
    from scmle.pipeline.stages import interface
    from scmle.pipeline.stages import planning
    from scmle.pipeline.stages import profiling
    from scmle.runtime import plugins
    import scmle.runtime.state

    seen: dict[str, int] = {}

    class Peek(interface.Stage):
        """Reads the locked rows before the selection exists."""

        state = scmle.runtime.state.Progress.PLANNING
        requires = (planning.Planning,)

        def describe(self, board: Any) -> str:
            del board
            return "peek at the locked test rows"

        def execute(self, board: Any) -> None:
            locked = board.get(profiling.Partition).split.test
            locked.bind("peek")
            seen["rows"] = len(locked.open(locked.mint(), "peek")[1])

    frame = table()

    def call() -> dict[str, Any]:
        plugins.REGISTRY.enrol(plugins.Kind.STAGES, "peek", Peek)
        try:
            return trained(project(lab, frame).run())
        finally:
            plugins.REGISTRY.withdraw(plugins.Kind.STAGES, "peek")

    return witnessed(call, lambda: {"peeked": seen["rows"]} if seen else None)


SIGNED = {"SCMLE_SIGNING_KEY": KEY}
# Where a confirmation set can be spent twice: the register of spent sets
# is plain files in the output folder.
REGISTER = (
    "scmle/assurance/ledger/usage.py: Register keeps the spent "
    "confirmation sets as unsigned files in the output folder "
    "(`.confirmations`), so another output folder, or deleting the folder, "
    "forgets them (the module's stated limit)"
)
PLUGIN = (
    "scmle/pipeline/stages/defaults.py: sequence runs enrolled stages "
    "with the board, whose profiling.Partition holds the raw locked test set "
    "(split.test); only assessment.py Assessment.execute "
    "wraps it in journal.Custodian, so an earlier bind and open is neither "
    "refused nor journaled"
)
# Where the custody journal lets an edit through: its hash chain is keyless
# and unanchored (`digest`, `Journal.read`), and `Journal.current` reads
# only the entries after the latest `started` one, which `Journal.admit`
# (split.resident.reopen) and `Journal.rejected` (policy.checkpoints) trust.
JOURNAL = (
    "scmle/assurance/custody/journal.py: digest/Journal.read (an unkeyed "
    "SHA-256 chain with no anchored head: anyone can recompute it, and "
    "removing its last lines leaves a valid chain) and Journal.current (a "
    "later `started` entry hides the opening and the rejections before it)"
)
# Where a signed envelope runs with no key to check it: the module says
# envelopes follow the bundle rule, and `Bundle.load` refuses this case
# (pickle_bundle_trusted_nokey is blocked).
ENVELOPE = (
    "scmle/envelope.py: read accepts a signed envelope with no key when "
    "trusted=True, while Bundle.load refuses the same case; the module "
    "docstring claims both follow one rule"
)
TRUSTED = "--trusted vouches for an unsigned bundle (documented)"
# Defence in depth (`Attempt.also`): the controls that stop an attack before
# the gate it aims at, each the first check the attacked value meets.
# The configuration schema refuses the value outright (pydantic).
SCHEMA = ("ValidationError",)
# `scmle.runtime.config.overlay` takes only the listed sections, so a
# `--config` file cannot carry a leakage policy at all.
OVERLAY = ("cli.config",)
# `signing.keying` refuses an empty key or an unreadable key file before
# any bundle is read.
KEYING = ("cli.config",)
# `archive.Directory.read` refuses an edited stage record before the gate's
# own check is reached.
RECORDS = ("split.resident.records",)
# `Journal.admit` refuses to resume a run whose test set was opened.
REOPEN = ("split.resident.reopen",)
# `Project.resume` refuses a run that wrote its completion marker.
COMPLETED = ("ConfigError",)
# The loader resolves each member against the signed manifest, so an
# unlisted file is never read and the load finishes without it (`Held`).
UNLISTED = ("Held",)
ATTEMPTS = {
    a.name: a
    for a in (
        Attempt(
            "leak_copy",
            "leakage.policy",
            "api",
            "column `leak` equal to the target",
            leak("copy"),
            control=True,
        ),
        Attempt(
            "leak_renamed",
            "leakage.policy",
            "api",
            "target copy named `tenure_months`",
            leak("renamed"),
        ),
        Attempt(
            "leak_labels",
            "leakage.policy",
            "api",
            "target as the strings active / churned",
            leak("labels"),
        ),
        Attempt(
            "leak_affine",
            "leakage.policy",
            "api",
            "3 * target + 7 plus tiny noise",
            leak("affine"),
        ),
        Attempt(
            "leak_xor",
            "leakage.policy",
            "api",
            "target split into two columns, p XOR q (each column alone "
            "independent of it)",
            leak("xor"),
        ),
        Attempt(
            "leak_text",
            "leakage.policy",
            "api",
            "target written into a free-text note per row",
            leak("text"),
        ),
        Attempt(
            "leak_date",
            "leakage.policy",
            "api",
            "target encoded as the year of a date column",
            leak("date"),
        ),
        Attempt(
            "leak_policy_case",
            "leakage.policy",
            "api",
            "leakage_policy='BLOCK' (wrong case)",
            misspelled,
            also=SCHEMA,
        ),
        Attempt(
            "leak_cli",
            "leakage.policy",
            "cli",
            "renamed copy through scmle run --leakage block",
            clileak,
        ),
        Attempt(
            "leak_cli_config",
            "leakage.policy",
            "cli",
            "--config file setting leakage_policy to warn beside "
            "--leakage block",
            overlay,
            also=OVERLAY,
        ),
        Attempt(
            "leak_resume",
            "leakage.policy",
            "api",
            "resume the refused run unchanged",
            leakresume,
        ),
        Attempt(
            "leak_resume_edited",
            "leakage.policy",
            "api",
            "edit config.json to leakage_policy warn, rehash the "
            "declaration, resume",
            leakedit,
            override="leakage_policy='warn' (documented override)",
            also=RECORDS,
        ),
        Attempt(
            "budget_small",
            "budget.minimum_run",
            "api",
            "time_budget=1",
            budget(1),
            control=True,
        ),
        Attempt(
            "budget_zero",
            "budget.minimum_run",
            "api",
            "time_budget=0",
            budget(0),
            also=SCHEMA,
        ),
        Attempt(
            "budget_negative",
            "budget.minimum_run",
            "api",
            "time_budget=-5",
            budget(-5),
            also=SCHEMA,
        ),
        Attempt(
            "budget_bool",
            "budget.minimum_run",
            "api",
            "time_budget=True",
            budget(value=True),
        ),
        Attempt(
            "budget_fraction",
            "budget.minimum_run",
            "api",
            "time_budget=1.9",
            budget(1.9),
            also=SCHEMA,
        ),
        Attempt(
            "budget_nan",
            "budget.minimum_run",
            "api",
            "time_budget=nan (every comparison with it is false)",
            budget(float("nan")),
            also=SCHEMA,
        ),
        Attempt(
            "budget_infinite",
            "budget.minimum_run",
            "api",
            "time_budget=inf",
            budget(float("inf")),
            also=SCHEMA,
        ),
        Attempt(
            "budget_cli",
            "budget.minimum_run",
            "cli",
            "scmle run --budget 2",
            clibudget,
        ),
        Attempt(
            "budget_resume",
            "budget.minimum_run",
            "api",
            "resume the refused run unchanged",
            budgetresume,
        ),
        Attempt(
            "unlabeled_nan",
            "data.unlabeled",
            "api",
            "20 target values NaN",
            unlabeled("nan"),
            control=True,
        ),
        Attempt(
            "unlabeled_none",
            "data.unlabeled",
            "api",
            "20 target values None in an object column",
            unlabeled("none"),
        ),
        Attempt(
            "unlabeled_na",
            "data.unlabeled",
            "api",
            "20 target values pandas.NA in a nullable integer column",
            unlabeled("na"),
        ),
        Attempt(
            "unlabeled_cli",
            "data.unlabeled",
            "cli",
            "20 empty target cells in the CSV",
            cliunlabeled,
        ),
        Attempt(
            "checkpoint_reject",
            "policy.checkpoints",
            "api",
            "a checkpoint rejects the plan",
            checkpoint,
            control=True,
        ),
        Attempt(
            "checkpoint_resume",
            "policy.checkpoints",
            "api",
            "resume the rejected run (Project.resume takes no checkpoints)",
            checkresume,
        ),
        Attempt(
            "exam_resume_done",
            "split.resident.open",
            "api",
            "resume a completed run",
            examdone,
            control=True,
            also=COMPLETED,
        ),
        Attempt(
            "exam_resume_deleted",
            "split.resident.open",
            "api",
            "delete evaluation.json and complete.json of a completed run "
            "and resume",
            examagain,
            also=REOPEN,
        ),
        Attempt(
            "exam_resume_edited",
            "split.resident.open",
            "api",
            "exclude the winner in config.json, rehash the "
            "declaration, delete execution_plan.json, "
            "evaluation.json and complete.json, resume",
            examedit,
            also=REOPEN,
        ),
        Attempt(
            "charter_rerun",
            "confirm.single_use",
            "api",
            "run the same signed charter on the same data again",
            rerun("same"),
        ),
        Attempt(
            "charter_rerun_elsewhere",
            "confirm.single_use",
            "api",
            "run the same signed charter on the same data again, into "
            "another output folder",
            rerun("elsewhere"),
            defect=REGISTER,
        ),
        Attempt(
            "charter_rerun_forgotten",
            "confirm.single_use",
            "api",
            "delete the output folder's `.confirmations` record, then run "
            "the same signed charter on the same data again",
            rerun("forgotten"),
            defect=REGISTER,
        ),
        Attempt(
            "charter_resume",
            "confirm.single_use",
            "api",
            "delete evaluation.json and complete.json of a confirmed "
            "charter run and resume",
            charterresume,
            control=True,
            also=REOPEN,
        ),
        Attempt(
            "charter_edited",
            "charter.verify",
            "api",
            "relax alpha_confirm in a signed charter",
            charteredit(recompute=False, key=KEY),
            control=True,
        ),
        Attempt(
            "charter_rehashed",
            "charter.verify",
            "api",
            "relax alpha_confirm and recompute the charter hash, key set",
            charteredit(recompute=True, key=KEY),
        ),
        Attempt(
            "charter_rehashed_nokey",
            "charter.verify",
            "api",
            "relax alpha_confirm and recompute the charter hash, no key",
            charteredit(recompute=True, key=None),
            override="a signed charter read without a key is treated "
            "as unsigned (documented in read_verified)",
        ),
        Attempt(
            "bundle_unsigned",
            "bundle.authenticate",
            "cli",
            "predict with a signed bundle and no key",
            load("none", {}),
            control=True,
        ),
        Attempt(
            "bundle_bytes",
            "bundle.authenticate",
            "cli",
            "a byte appended to novelty.joblib, key set",
            load("bytes", SIGNED),
        ),
        Attempt(
            "bundle_rehash",
            "bundle.authenticate",
            "cli",
            "member changed and its manifest checksum updated, key set",
            load("rehash", SIGNED),
        ),
        Attempt(
            "bundle_rehash_trusted",
            "bundle.authenticate",
            "cli",
            "as bundle_rehash with --trusted, key set",
            load("rehash", SIGNED, trusted=True),
        ),
        Attempt(
            "bundle_rehash_trusted_nokey",
            "bundle.authenticate",
            "cli",
            "as bundle_rehash with --trusted and no key: the invalid "
            "signature.json is never checked",
            load("rehash", {}, trusted=True),
        ),
        Attempt(
            "bundle_strip",
            "bundle.authenticate",
            "cli",
            "member changed, checksum updated, signature.json deleted, key set",
            load("strip", SIGNED),
        ),
        Attempt(
            "bundle_strip_trusted",
            "bundle.authenticate",
            "cli",
            "as bundle_strip with --trusted, key set",
            load("strip", SIGNED, trusted=True),
            override=TRUSTED,
        ),
        Attempt(
            "bundle_production",
            "bundle.authenticate",
            "cli",
            "stripped bundle, --trusted, SCMLE_ENV=' Production '",
            load("strip", {"SCMLE_ENV": " Production "}, trusted=True),
        ),
        Attempt(
            "bundle_prod_alias",
            "bundle.authenticate",
            "cli",
            "stripped bundle, --trusted, SCMLE_ENV=prod",
            load("strip", {"SCMLE_ENV": "prod"}, trusted=True),
            override=TRUSTED + "; only SCMLE_ENV=production refuses it",
        ),
        Attempt(
            "bundle_trusted_variable",
            "bundle.authenticate",
            "cli",
            "stripped bundle, no --trusted, SCMLE_TRUSTED=1 (the packaged "
            "image's variable) in the environment",
            load("strip", {"SCMLE_TRUSTED": "1"}),
        ),
        Attempt(
            "bundle_algorithm",
            "bundle.authenticate",
            "cli",
            "signature algorithm set to none, key set",
            load("algorithm", SIGNED),
        ),
        Attempt(
            "bundle_empty_key",
            "bundle.authenticate",
            "cli",
            "SCMLE_SIGNING_KEY='' with --trusted",
            load("none", {"SCMLE_SIGNING_KEY": ""}, trusted=True),
            also=KEYING,
        ),
        Attempt(
            "bundle_key_file",
            "bundle.authenticate",
            "cli",
            "SCMLE_SIGNING_KEY_FILE names a missing file, --trusted",
            load(
                "none", {"SCMLE_SIGNING_KEY_FILE": "/nonexistent"}, trusted=True
            ),
            also=KEYING,
        ),
        Attempt(
            "package_rehash_trusted_nokey",
            "bundle.authenticate",
            "cli",
            "scmle package of bundle_rehash with --trusted and no key",
            load("rehash", {}, trusted=True, verb="package"),
        ),
        Attempt(
            "package_production",
            "bundle.authenticate",
            "cli",
            "scmle package --trusted with SCMLE_ENV=production",
            load(
                "strip",
                {"SCMLE_ENV": "production"},
                trusted=True,
                verb="package",
            ),
        ),
        Attempt(
            "serve_production",
            "bundle.authenticate",
            "cli",
            "scmle serve --trusted with SCMLE_ENV=production",
            load(
                "strip", {"SCMLE_ENV": "production"}, trusted=True, verb="serve"
            ),
        ),
        Attempt(
            "journal_edited",
            "split.resident.journal",
            "api",
            "delete a completed run's evaluation and marker, change the "
            "configuration hash in the first line of its custody journal, "
            "resume",
            examjournal("edit"),
            control=True,
        ),
        Attempt(
            "journal_truncated",
            "split.resident.journal",
            "api",
            "delete a completed run's evaluation and marker, remove the "
            "last line (`opened`) of its custody journal, recompute "
            "nothing, resume",
            examjournal("truncate"),
            defect=JOURNAL,
        ),
        Attempt(
            "journal_rechained",
            "split.resident.journal",
            "api",
            "remove the `opened` entry and recompute the journal's hash "
            "chain (plain SHA-256, no key), resume",
            examjournal("rechain"),
            defect=JOURNAL,
        ),
        Attempt(
            "journal_restarted",
            "split.resident.reopen",
            "api",
            "delete a completed run's evaluation and marker, append a "
            "chained `started` entry to its custody journal, resume",
            examjournal("restart"),
            defect=JOURNAL,
        ),
        Attempt(
            "journal_restarted_winner",
            "split.resident.reopen",
            "api",
            "as exam_resume_edited (winner excluded, declaration rehashed) "
            "plus a chained `started` entry: another model on the same rows",
            examjournal("winner"),
            also=RECORDS,
        ),
        Attempt(
            "journal_deleted",
            "split.resident.reopen",
            "api",
            "delete a completed run's evaluation, marker and custody "
            "journal, resume",
            examjournal("delete"),
            control=True,
        ),
        Attempt(
            "journal_moved",
            "split.resident.reopen",
            "api",
            "copy a completed run, without evaluation and marker, to a new "
            "name beside it and resume the copy",
            examjournal("moved"),
        ),
        Attempt(
            "journal_symlink",
            "split.resident.reopen",
            "api",
            "resume a completed run, without evaluation and marker, through "
            "a symlink in another folder (the journal is looked up there)",
            examjournal("symlink"),
        ),
        Attempt(
            "checkpoint_approved",
            "policy.checkpoints",
            "api",
            "append a chained approval by `attacker` for the rejected gate "
            "to the custody journal, resume without checkpoints",
            checkjournal("approve"),
            defect=JOURNAL,
        ),
        Attempt(
            "checkpoint_rechained",
            "policy.checkpoints",
            "api",
            "remove the rejection from the custody journal, recompute the "
            "chain, resume without checkpoints",
            checkjournal("rechain"),
            defect=JOURNAL,
        ),
        Attempt(
            "checkpoint_restarted",
            "policy.checkpoints",
            "api",
            "append a chained `started` entry to the custody journal of a "
            "rejected run, resume without checkpoints",
            checkjournal("restart"),
            defect=JOURNAL,
        ),
        Attempt(
            "pickle_bundle",
            "bundle.authenticate",
            "cli",
            "model replaced by a pickle payload, manifest checksum updated, "
            "key set; got through if the payload ran",
            load("payload", SIGNED),
        ),
        Attempt(
            "pickle_bundle_trusted_nokey",
            "bundle.authenticate",
            "cli",
            "as pickle_bundle with --trusted and no key",
            load("payload", {}, trusted=True),
        ),
        Attempt(
            "pickle_bundle_stripped",
            "bundle.authenticate",
            "cli",
            "as pickle_bundle with signature.json deleted, key set, no "
            "--trusted",
            load("payload_strip", SIGNED),
        ),
        Attempt(
            "pickle_bundle_production",
            "bundle.authenticate",
            "cli",
            "as pickle_bundle_stripped with --trusted and SCMLE_ENV=production",
            load("payload_strip", {"SCMLE_ENV": "production"}, trusted=True),
        ),
        Attempt(
            "pickle_bundle_unlisted",
            "bundle.authenticate",
            "cli",
            "a pickle payload added as calibrator.joblib, which the signed "
            "manifest does not list, key set",
            load("payload_unlisted", SIGNED),
            also=UNLISTED,
        ),
        Attempt(
            "pickle_checkpoint",
            "envelope.authenticate",
            "api",
            "deep-learning checkpoint with a pickle payload and a forged "
            "MAC, key set (TorchClassifier.restore)",
            restored("0" * 64, SIGNED, trusted=False),
            control=True,
        ),
        Attempt(
            "pickle_checkpoint_unsigned",
            "envelope.authenticate",
            "api",
            "as pickle_checkpoint with the MAC removed, key set",
            restored(None, SIGNED, trusted=False),
        ),
        Attempt(
            "pickle_checkpoint_production",
            "envelope.authenticate",
            "api",
            "unsigned payload checkpoint, trusted=True, SCMLE_ENV=production",
            restored(
                None,
                {"SCMLE_ENV": "production", "SCMLE_SIGNING_KEY": None},
                trusted=True,
            ),
        ),
        Attempt(
            "pickle_checkpoint_trusted_nokey",
            "envelope.authenticate",
            "api",
            "payload checkpoint with a forged MAC, trusted=True, no key",
            restored("0" * 64, {"SCMLE_SIGNING_KEY": None}, trusted=True),
            defect=ENVELOPE,
        ),
        Attempt(
            "pickle_products",
            "envelope.authenticate",
            "api",
            "the signed stage products of a run rejected at selection "
            "swapped for unsigned pickle payloads, resumed with the key",
            products,
        ),
        Attempt(
            "charter_stripped",
            "charter.verify",
            "api",
            "relax alpha_confirm, recompute the charter hash and delete the "
            "signature, key set",
            stripped,
            override="a charter without a signature runs as unsigned "
            "(documented in read_verified)",
        ),
        Attempt(
            "token_rebind",
            "split.resident.bind",
            "object",
            "bind the in-process test set again after its release",
            token("rebind", sealed=False),
            control=True,
        ),
        Attempt(
            "token_reuse",
            "split.resident.open",
            "object",
            "open the in-process test set a second time with its token",
            token("reuse", sealed=False),
        ),
        Attempt(
            "token_cross",
            "split.resident.open",
            "object",
            "open one in-process test set with another set's token",
            token("cross", sealed=False),
        ),
        Attempt(
            "token_unbound",
            "split.resident.open",
            "object",
            "open the in-process test set before it is bound",
            token("unbound", sealed=False),
        ),
        Attempt(
            "token_mismatch",
            "split.resident.open",
            "object",
            "open the in-process test set with another binding",
            token("mismatch", sealed=False),
        ),
        Attempt(
            "token_unbinding",
            "split.resident.open",
            "object",
            "open the bound in-process test set with no binding",
            token("unbinding", sealed=False),
        ),
        Attempt(
            "token_reset",
            "split.resident.open",
            "object",
            "open the in-process test set, set `opened` back to False, "
            "open again",
            token("reset", sealed=False),
            override="the in-process set is a protocol, not a boundary "
            "(documented in lockbox.Resident; the vault is the boundary)",
        ),
        Attempt(
            "vault_rebind",
            "vault.bind",
            "object",
            "bind the vault test set again after its release",
            token("rebind", sealed=True),
            control=True,
        ),
        Attempt(
            "vault_reuse",
            "vault.open",
            "object",
            "open the vault test set a second time with its token",
            token("reuse", sealed=True),
            control=True,
        ),
        Attempt(
            "vault_reset",
            "vault.open",
            "object",
            "open the vault test set, set `opened` back to False, open again",
            token("reset", sealed=True),
        ),
        Attempt(
            "vault_cross",
            "vault.open",
            "object",
            "open one vault test set with another vault's token",
            token("cross", sealed=True),
        ),
        Attempt(
            "vault_mismatch",
            "vault.open",
            "object",
            "open the vault test set with another binding",
            token("mismatch", sealed=True),
        ),
        Attempt(
            "plugin_peek",
            "split.resident.open",
            "api",
            "a plugin stage enrolled under kind `stages` binds and opens the "
            "locked test set right after planning, before any selection",
            plugin,
            defect=PLUGIN,
        ),
    )
}


def fixture(folder: pathlib.Path) -> pathlib.Path:
    """Trains one model with the signing key set; its bundle is signed.

    Args:
        folder: Where the run is written.

    Returns:
        The signed bundle directory.
    """
    lab = Lab(folder)
    with environ(SCMLE_SIGNING_KEY=KEY):
        run = pathlib.Path(project(lab, table()).run().directory)
    return run / "model"


def classify(
    attempt: Attempt, lab: Lab, sites: dict[str, str]
) -> dict[str, Any]:
    """Sets an attempt up, makes the attack and records how it ended.

    Args:
        attempt: The attempt.
        lab: Its lab.
        sites: Registered raising site to gate (`refusal.registry`).

    Returns:
        The attempt's record: its fields, the outcome, what stopped it and
        its message, or the detail of what got through, and the seconds
        taken. What stopped it is the registered gate, the `UNREGISTERED`
        control whose site raised it, a CLI refusal's `Refused.control`,
        else the error type. An attack that raised `Broken` never reached
        its control, so it is an `error`, not a refusal.
    """
    listed = {site: gate for gate, site in UNREGISTERED.items()}
    started = time.monotonic()
    record: dict[str, Any] = {
        "name": attempt.name,
        "gate": attempt.gate,
        "registered": attempt.gate not in UNREGISTERED,
        "surface": attempt.surface,
        "vector": attempt.vector,
        "override": attempt.override,
        "control": attempt.control,
    }
    try:
        act = attempt.prepare(lab)
    except Exception as error:  # noqa: BLE001 -- recorded as `error`
        record.update(
            outcome="error",
            stopped=type(error).__name__,
            message=f"setup: {str(error)[:400]}",
        )
    else:
        try:
            detail = act()
        except (errors.BlockedError, errors.LeakageError) as error:
            gate, site = refusal.locate(error, sites)
            record.update(
                outcome="blocked",
                stopped=gate,
                site=site,
                message=str(error)[:400],
            )
        except Broken as error:
            record.update(
                outcome="error",
                stopped=type(error).__name__,
                message=f"attack: {str(error)[:400]}",
            )
        except Refused as error:
            record.update(
                outcome="blocked",
                stopped=error.control,
                status=error.status,
                message=str(error),
            )
        except Exception as error:  # noqa: BLE001 -- every refusal counts
            gate, site = refusal.locate(error, listed)
            named = gate != "unregistered"
            record.update(
                outcome="blocked",
                stopped=gate if named else type(error).__name__,
                message=str(error)[:400],
            )
            if named:
                record["site"] = site
        else:
            record.update(
                outcome="override" if attempt.override else "bypass",
                detail=detail,
            )
    record["seconds"] = round(time.monotonic() - started, 1)
    return record


def execute(name: str, bundle: str | None) -> dict[str, Any]:
    """Runs one attempt in a fresh folder (in a worker process).

    Args:
        name: The attempt's name.
        bundle: The signed bundle path, or None.

    Returns:
        The attempt's record.
    """
    with tempfile.TemporaryDirectory() as folder:
        lab = Lab(
            pathlib.Path(folder), pathlib.Path(bundle) if bundle else None
        )
        return classify(ATTEMPTS[name], lab, refusal.registry(GATES))


def quiet() -> None:
    """Quiets a worker: one thread, warnings and INFO logs off."""
    import logging
    import warnings

    os.environ["OMP_NUM_THREADS"] = "1"
    logging.disable(logging.INFO)
    warnings.simplefilter("ignore")


def registered() -> list[str]:
    """Every gate name in gates.toml.

    Returns:
        The dotted gate names, sorted.
    """
    return sorted(set(refusal.registry(GATES).values()))


def limit() -> int:
    """Bypasses the claim allows: acceptance.toml's, else the claim's zero.

    Returns:
        The largest number of bypasses that would meet the criterion.
    """
    found = acceptance.load().get(VARIABLE)
    return BYPASSES if found is None else int(found.value)


def tally(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts the outcomes, overall and per gate.

    Args:
        records: One record per attempt.

    Returns:
        The count per outcome, the count per gate and outcome, and the
        names of the bypasses.
    """
    gates: dict[str, dict[str, int]] = {}
    for record in records:
        counts = gates.setdefault(record["gate"], dict.fromkeys(OUTCOMES, 0))
        counts[record["outcome"]] += 1
    return {
        "attempts": len(records),
        **{o: sum(r["outcome"] == o for r in records) for o in OUTCOMES},
        "gates": dict(sorted(gates.items())),
        "bypasses": [r["name"] for r in records if r["outcome"] == "bypass"],
    }


def digest() -> str:
    """Hash of this module, which decides what is attempted and counted.

    Returns:
        The combined source hash.
    """
    return certificates.fingerprint(sys.modules[__name__])


def measure(workers: int = WORKERS) -> dict[str, Any]:
    """Builds the signed bundle, runs every attempt and the document.

    Each attempt runs in its own worker process (one task per process), so
    no state carries between attempts.

    Args:
        workers: Worker processes.

    Returns:
        The JSON-ready document: scope, records, counts, the gates attempted
        and not attempted, the allowed bypasses, code hash and duration.
    """
    started = time.monotonic()
    context = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory() as folder:
        with futures.ProcessPoolExecutor(
            1, mp_context=context, initializer=quiet
        ) as pool:
            signed = pool.submit(fixture, pathlib.Path(folder)).result()
        with futures.ProcessPoolExecutor(
            workers,
            mp_context=context,
            initializer=quiet,
            max_tasks_per_child=1,
        ) as pool:
            jobs = {
                name: pool.submit(execute, name, str(signed))
                for name in ATTEMPTS
            }
            records = [jobs[name].result() for name in ATTEMPTS]
    targeted = sorted({a.gate for a in ATTEMPTS.values()})
    return {
        "claim": "C8",
        "scope": SCOPE,
        "variable": VARIABLE,
        "allowed": limit(),
        "records": records,
        "counts": tally(records),
        "targeted": targeted,
        "untargeted": [g for g in registered() if g not in targeted],
        "unregistered": UNREGISTERED,
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


def problems(document: dict[str, Any] | None = None) -> list[str]:
    """Why the stored search is not current.

    The search can never decide the row (see `summary`); these are the
    reasons it is not even a current search.

    Args:
        document: The document; read from `PATH` when None.

    Returns:
        Messages for a missing file, other code, an attempt not run, an
        attempt whose setup failed, or a gate in the records that is
        neither registered nor listed in `UNREGISTERED`; empty when current.
    """
    found = read(document)
    if found is None:
        return [f"{PATH.name} is missing"]
    issues = []
    if found.get("code") != digest():
        issues.append("bypass: measured with other code")
    records = {r.get("name"): r for r in found.get("records", [])}
    known = set(registered()) | set(UNREGISTERED)
    for name in ATTEMPTS:
        if name not in records:
            issues.append(f"bypass: attempt {name} not run")
        elif records[name].get("outcome") == "error":
            issues.append(f"bypass: attempt {name} did not set up")
    issues.extend(
        f"bypass: {r.get('name')} targets unknown gate {r.get('gate')}"
        for r in records.values()
        if r.get("gate") not in known
    )
    return issues


def summary(document: dict[str, Any] | None = None) -> tuple[str, str]:
    """The scorecard's value and verdict for the row.

    Args:
        document: The document; read from `PATH` when None.

    Returns:
        The value text (attempts, gates, blocked, overrides and the
        bypasses by name, plainly labelled as an automated search) and
        the verdict: `fail` when a current search found a bypass (one
        bypass disproves "cannot be bypassed"), otherwise `not measured`,
        because only a red-team exercise by people can pass C8.
    """
    found = read(document)
    if found is None or not found.get("records"):
        return "no measurement", "not measured"
    counts = tally(found["records"])
    text = (
        f"automated search, not the red-team exercise: "
        f"{counts['attempts']} attempts on {len(counts['gates'])} controls; "
        f"{counts['blocked']} blocked, {counts['override']} through a "
        f"documented override, {counts['bypass']} bypasses"
    )
    if counts["bypasses"]:
        text += " (" + ", ".join(counts["bypasses"]) + ")"
    if counts["error"]:
        text += f"; {counts['error']} did not set up"
    if problems(found):
        return text + "; stale", "not measured"
    return text, "fail" if counts["bypass"] else "not measured"


def main(arguments: list[str] | None = None) -> int:
    """Writes (`--write`) or checks (`--check`) the stored search.

    Args:
        arguments: Command-line arguments; None reads `sys.argv`.

    Returns:
        0 after `--write`; for `--check`, 0 when `problems` is empty and 1
        otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parsed = parser.parse_args(arguments)
    if parsed.write:
        document = measure(parsed.workers)
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
