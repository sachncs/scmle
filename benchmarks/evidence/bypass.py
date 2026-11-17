"""Claim C8: an automated search for ways past the product's controls.

C8 says controls cannot be bypassed, decided by zero bypasses in a red-team
exercise run by people (stage D). This module is NOT that exercise. It is an
automated search, run by the product's own developers, that tries scripted
attacks through the public API (`scmle.api.project.Project`) and the CLI
(`scmle ...` in a subprocess) only: no monkeypatching, no private calls. It
is evidence for the row, and the row's verdict stays `not measured` however
the search comes out; a successful bypass is a product defect to report.

Each attempt targets one control while the condition it guards holds:

* `leakage.policy`: a copy of the target under `leakage_policy="block"`,
  renamed, relabelled, rescaled, split into two columns (XOR), as
  text or as a date; a misspelled policy; a CLI `--config` that tries to
  switch the policy; resuming a refused run, as is and with the policy
  edited in `config.json`.
* `budget.minimum_run`: budgets below the minimum run, as zero, negative,
  boolean or fractional values, through the CLI and by resuming.
* `data.unlabeled`: missing target values as NaN, None and empty CSV cells.
* `policy.checkpoints`: a run rejected by a checkpoint, then resumed.
* `split.resident.open` and `confirm.single_use` (the exam sets): resuming a
  completed run, after deleting its evaluation and after editing its
  configuration; rerunning a charter; resuming a confirmed charter run.
* `charter.verify` (not in gates.toml; `sealing.read_verified`): a signed
  charter edited, with and without its hash recomputed, with and without the
  key.
* `bundle.authenticate` (not in gates.toml; `Bundle.load`): unsigned and
  tampered bundles through `predict`, `package` and `serve`, with `--trusted`,
  `SCMLE_ENV`, an empty key and a missing key file.

An attempt ends in one of four ways:

* `blocked`: the product refused (an exception, or a CLI exit other than 0);
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
from concurrent import futures
import contextlib
import dataclasses
import hashlib
import json
import multiprocessing
import os
import pathlib
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
from scmle.contracts import errors
from scmle.data.detectors import certificates

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
    "charter.verify": "scmle.assurance.charter.sealing:read_verified",
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
    "--top-models",
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


class Refused(Exception):  # noqa: N818 -- a refusal, not a fault
    """A CLI call that exited with a status other than 0."""


class Broken(Exception):  # noqa: N818 -- the attempt's own fault
    """An attempt's setup did not reach the state it attacks."""


@dataclasses.dataclass
class Lab:
    """Where one attempt works.

    Attributes:
        folder: The attempt's own empty folder.
        bundle: A signed model bundle the attempt may copy, or None.
    """

    folder: pathlib.Path
    bundle: pathlib.Path | None = None


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
    """

    name: str
    gate: str
    surface: str
    vector: str
    prepare: Callable[[Lab], Callable[[], dict[str, Any]]]
    override: str = ""
    control: bool = False


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

    chosen = {"time_budget": BUDGET, **options(), **settings}
    return scmle.api.project.Project(
        frame, "y", output_dir=lab.folder / "runs", **chosen
    )


def trained(result: Any) -> dict[str, Any]:
    """What a run that got through produced.

    Args:
        result: The completed run.

    Returns:
        Its selected model, its test metrics (a leak that was used shows as
        a near-perfect score) and its run directory.
    """
    path = pathlib.Path(result.run_dir) / "evaluation.json"
    evaluation = json.loads(path.read_text(encoding="utf-8"))
    return {
        "model": result.selected_model,
        "test": evaluation.get("test_metrics"),
        "run": pathlib.Path(result.run_dir).name,
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
        raise Refused(lines[-1][-400:])
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
        return lambda: trained(
            project(lab, frame, leakage_policy="block").run()
        )

    return prepare


def misspelled(lab: Lab) -> Callable[[], dict[str, Any]]:
    """A leak under `leakage_policy="BLOCK"`, hoping it reads as warn."""
    frame = leaked("copy")
    return lambda: trained(project(lab, frame, leakage_policy="BLOCK").run())


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
    """A renamed leak through `scmle run --leakage-policy block`."""
    data = written(lab, leaked("renamed"))
    arguments = command(lab, data, "--leakage-policy", "block")
    return lambda: cli(arguments, folder=lab.folder)


def overlay(lab: Lab) -> Callable[[], dict[str, Any]]:
    """A `--config` file that tries to set the policy back to warn."""
    data = written(lab, leaked("renamed"))
    config = lab.folder / "config.json"
    config.write_text(json.dumps({"leakage_policy": "warn"}), "utf-8")
    arguments = command(
        lab, data, "--leakage-policy", "block", "--config", str(config)
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


def resume(run: pathlib.Path, frame: pandas.DataFrame) -> dict[str, Any]:
    """Resumes a run through `Project.resume`.

    Args:
        run: The run directory.
        frame: The original in-memory table.

    Returns:
        What the resumed run produced.
    """
    import scmle.api.project

    return trained(scmle.api.project.Project.resume(run, data=frame))


def leakresume(lab: Lab) -> Callable[[], dict[str, Any]]:
    """Resumes a run the leakage policy refused, unchanged."""
    frame = leaked("copy")
    run = refused(lambda: project(lab, frame, leakage_policy="block").run())
    return lambda: resume(run, frame)


def leakedit(lab: Lab) -> Callable[[], dict[str, Any]]:
    """Resumes a refused leak run after setting the policy to warn."""
    frame = leaked("copy")
    run = refused(lambda: project(lab, frame, leakage_policy="block").run())
    edit(run, lambda c: c.update(leakage_policy="warn"))
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
        return lambda: trained(project(lab, frame, time_budget=value).run())

    return prepare


def clibudget(lab: Lab) -> Callable[[], dict[str, Any]]:
    """A two-second budget through `scmle run --time-budget 2`."""
    data = written(lab, table())
    arguments = command(lab, data, "--time-budget", "2")
    return lambda: cli(arguments, folder=lab.folder)


def budgetresume(lab: Lab) -> Callable[[], dict[str, Any]]:
    """Resumes a run the budget refused, unchanged."""
    frame = table()
    run = refused(lambda: project(lab, frame, time_budget=1).run())
    return lambda: resume(run, frame)


def unlabeled(kind: str) -> Callable[[Lab], Callable[[], dict[str, Any]]]:
    """An attempt that trains with missing labels under `unlabeled=block`.

    Args:
        kind: `nan` (float NaN) or `none` (None in an object column).

    Returns:
        The attempt's `prepare`.
    """

    def prepare(lab: Lab) -> Callable[[], dict[str, Any]]:
        frame = table()
        if kind == "none":
            frame["y"] = frame.y.astype(object)
            frame.loc[:19, "y"] = None
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


def rejecter() -> Any:
    """A checkpoint that rejects every review.

    Returns:
        The checkpoint.
    """
    from scmle.core import policies

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
    run = pathlib.Path(project(lab, frame, **settings).run().run_dir)
    evaluation = json.loads((run / "evaluation.json").read_text("utf-8"))
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
    after = json.loads((run / "evaluation.json").read_text("utf-8"))
    return {
        **result,
        "first": [before["selected_model"], before["test_metrics"]],
        "second": [after["selected_model"], after["test_metrics"]],
    }


def examdone(lab: Lab) -> Callable[[], dict[str, Any]]:
    """Resumes a completed run as it is."""
    frame = table()
    run, _ = finished(lab, frame)
    return lambda: resume(run, frame)


def examagain(lab: Lab) -> Callable[[], dict[str, Any]]:
    """Deletes a completed run's evaluation.json and resumes it."""
    frame = table()
    run, before = finished(lab, frame)
    (run / "evaluation.json").unlink()
    return lambda: reopened(run, frame, before)


def examedit(lab: Lab) -> Callable[[], dict[str, Any]]:
    """Excludes the winner in config.json and resumes the completed run.

    The saved plan and evaluation are deleted and the declaration rehashed,
    so a different model is tested on the same locked rows.
    """
    frame = table()
    run, before = finished(lab, frame)
    winner = before["selected_model"]
    edit(run, lambda c: c["models"]["exclude_models"].append(winner))
    (run / "evaluation.json").unlink()
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


def confirmation(run: pathlib.Path) -> list[int]:
    """The confirmation rows of a charter run.

    Args:
        run: The run directory.

    Returns:
        The row ids of its confirmation set.
    """
    manifest = json.loads((run / "split_manifest.json").read_text("utf-8"))
    return list(manifest["confirmation_ids"])


def rerun(lab: Lab) -> Callable[[], dict[str, Any]]:
    """Runs the same signed charter on the same data a second time."""
    frame = table(500)
    path = charter(lab)

    def act() -> dict[str, Any]:
        with environ(SCMLE_SIGNING_KEY=KEY):
            result = project(lab, frame, charter=path).run()
        run = pathlib.Path(result.run_dir)
        claims = json.loads((run / "confirmed_claims.json").read_text("utf-8"))
        return {
            **trained(result),
            "same_rows": confirmation(run) == rows,
            "outcome": claims.get("outcome"),
        }

    with environ(SCMLE_SIGNING_KEY=KEY):
        first = pathlib.Path(project(lab, frame, charter=path).run().run_dir)
    rows = confirmation(first)
    return act


def charterresume(lab: Lab) -> Callable[[], dict[str, Any]]:
    """Deletes a confirmed charter run's evaluation.json and resumes it."""
    frame = table(500)
    path = charter(lab)
    with environ(SCMLE_SIGNING_KEY=KEY):
        run = pathlib.Path(project(lab, frame, charter=path).run().run_dir)
    (run / "evaluation.json").unlink()

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
    confirm = record["charter"]["error_budget"]["confirmation"]
    confirm["alpha_confirm"] = "1/2"
    if recompute:
        record["charter_hash"] = hashed(lab, record["charter"])
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
            signature's algorithm set to `none`).

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
        return lambda: cli(arguments, environment, lab.folder)

    return prepare


SIGNED = {"SCMLE_SIGNING_KEY": KEY}
TRUSTED = "--trusted vouches for an unsigned bundle (documented)"
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
        ),
        Attempt(
            "leak_cli",
            "leakage.policy",
            "cli",
            "renamed copy through scmle run --leakage-policy block",
            clileak,
        ),
        Attempt(
            "leak_cli_config",
            "leakage.policy",
            "cli",
            "--config file setting leakage_policy to warn beside "
            "--leakage-policy block",
            overlay,
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
        ),
        Attempt(
            "budget_negative",
            "budget.minimum_run",
            "api",
            "time_budget=-5",
            budget(-5),
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
        ),
        Attempt(
            "budget_cli",
            "budget.minimum_run",
            "cli",
            "scmle run --time-budget 2",
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
        ),
        Attempt(
            "exam_resume_deleted",
            "split.resident.open",
            "api",
            "delete evaluation.json of a completed run and resume",
            examagain,
        ),
        Attempt(
            "exam_resume_edited",
            "split.resident.open",
            "api",
            "exclude the winner in config.json, rehash the "
            "declaration, delete execution_plan.json and "
            "evaluation.json, resume",
            examedit,
        ),
        Attempt(
            "charter_rerun",
            "confirm.single_use",
            "api",
            "run the same signed charter on the same data again",
            rerun,
        ),
        Attempt(
            "charter_resume",
            "confirm.single_use",
            "api",
            "delete evaluation.json of a confirmed charter run and resume",
            charterresume,
            control=True,
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
        ),
        Attempt(
            "bundle_key_file",
            "bundle.authenticate",
            "cli",
            "SCMLE_SIGNING_KEY_FILE names a missing file, --trusted",
            load(
                "none", {"SCMLE_SIGNING_KEY_FILE": "/nonexistent"}, trusted=True
            ),
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
        run = pathlib.Path(project(lab, table()).run().run_dir)
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
        The attempt's record: its fields, the outcome, what stopped it (the
        registered gate, else the error type) and its message, or the
        detail of what got through, and the seconds taken.
    """
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
        except Exception as error:  # noqa: BLE001 -- every refusal counts
            record.update(
                outcome="blocked",
                stopped=type(error).__name__,
                message=str(error)[:400],
            )
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
    return certificates.code_hash(sys.modules[__name__])


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
