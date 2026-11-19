"""Claim C10, change cost: what one new extension touches outside its package.

The criterion (plan section 5): adding a model family, task type, data kind,
metric or reader touches at most `change_cost_files_max` files outside its
own new package, done as a scripted exercise. Each `Exercise` here is that
exercise for one extension point, done in a fresh copy of the repository:

1. The extension is written as its own installed distribution (a module and
   a `.dist-info` with `entry_points.txt`) in a site folder outside the
   copy, advertised through the plugin registry's entry-point group
   (`scmle.runtime.plugins.GROUPS`). That folder is the extension's package;
   nothing in it counts.
2. Its probe, an end-to-end run through `scmle.api.project.Project` that
   uses the extension by name, runs with the copy unchanged.
3. The exercise's `Edit`s, the smallest changes outside the package that
   the probe was found to need, are applied and the probe runs again.
4. Each edited file is reverted on its own and the probe runs once more, so
   every counted file is shown to be needed (a file the probe passes
   without is a measurement error, and fails the point).
5. The test files relevant to the point run against the edited copy.

A point passes when its probe passes with the edits, every edited file is
needed, its tests pass and it edits at most `change_cost_files_max` files.
The claim's verdict is "pass" only when every point passes. The edits are
written against the current source; one that no longer applies is refused
(`InputError`) and the exercise must be redone, never skipped.

    python -m benchmarks.evidence.extensibility --show [--point NAME ...]
        [--root CHECKOUT]
    python -m benchmarks.evidence.extensibility --write
    python -m benchmarks.evidence.extensibility --check
"""

import argparse
from collections.abc import Mapping
import dataclasses
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from benchmarks import acceptance
from benchmarks import evidence
from scmle.contracts import errors

ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "scmle"
PATH = pathlib.Path(__file__).with_suffix(".json")
VARIABLE = "change_cost_files_max"
# The subpackages an extension point reaches; a change to any of them can
# change what an exercise needs, so the stored result goes stale with them.
MEASURED = (
    "api",
    "contracts",
    "data",
    "evaluation",
    "models",
    "pipeline",
    "preparation",
    "runtime",
)
# Folders whose untracked files are copied too: work not yet committed is
# measured as it stands.
WORKING = ("scmle", "tests", "benchmarks")
PROBE_SECONDS = 900
TESTS_SECONDS = 3600
TAIL = 2000


@dataclasses.dataclass(frozen=True)
class Edit:
    """One change to a file outside the extension's package.

    Attributes:
        path: The file, relative to the repository root.
        old: Text that must occur exactly once in it.
        new: Its replacement.
    """

    path: str
    old: str
    new: str

    def apply(self, root: pathlib.Path) -> None:
        """Makes the change in a copy of the repository.

        Args:
            root: The copy.

        Raises:
            InputError: If the file is missing or `old` does not occur
                exactly once, so the exercise no longer fits the code.
        """
        target = root / self.path
        if not target.is_file():
            raise errors.InputError(f"{self.path} is missing; redo the edit")
        text = target.read_text(encoding="utf-8")
        count = text.count(self.old)
        if count != 1:
            raise errors.InputError(
                f"{self.path}: the edit's text occurs {count} times, not "
                "once; the code moved, so the exercise must be redone"
            )
        target.write_text(text.replace(self.old, self.new), encoding="utf-8")


@dataclasses.dataclass(frozen=True)
class Exercise:
    """A minimal extension of one point, and what it needed.

    Attributes:
        point: The extension point.
        name: The distribution the extension is installed as.
        sources: The extension's module files, by file name.
        entries: Its `entry_points.txt`.
        edits: The changes it needs outside its package.
        probe: A script whose exit status says whether the extension works.
        tests: Test files relevant to the point.
        reason: Why each edited file is needed, by path.
    """

    point: str
    name: str
    sources: Mapping[str, str]
    entries: str
    edits: tuple[Edit, ...]
    probe: str
    tests: tuple[str, ...]
    reason: Mapping[str, str]

    @property
    def files(self) -> tuple[str, ...]:
        """The distinct files the edits touch, sorted."""
        return tuple(sorted({edit.path for edit in self.edits}))

    def install(self, site: pathlib.Path) -> pathlib.Path:
        """Writes the extension as an installed distribution.

        Args:
            site: The folder put on the path, outside the repository copy.

        Returns:
            The folder holding the extension: its package.
        """
        home = site / self.name
        info = home / f"{self.name}-0.1.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {self.name}\nVersion: 0.1\n",
            encoding="utf-8",
        )
        (info / "entry_points.txt").write_text(self.entries, encoding="utf-8")
        for file, text in self.sources.items():
            (home / file).write_text(text, encoding="utf-8")
        return home


@dataclasses.dataclass(frozen=True)
class Attempt:
    """One run of a probe or of a test set.

    Attributes:
        passed: Whether it exited with status 0.
        tail: The end of its output, for the record.
    """

    passed: bool
    tail: str


def files(root: pathlib.Path) -> list[str]:
    """The repository's files a copy needs.

    Args:
        root: The repository.

    Returns:
        Every tracked file and every untracked, unignored file under
        `WORKING`, as paths relative to `root`, sorted.

    Raises:
        InputError: If `root` is not a git checkout.
    """
    found: set[str] = set()
    for extra in ([], ["--others", "--exclude-standard", "--", *WORKING]):
        listed = subprocess.run(
            ["git", "ls-files", "-z", *extra],
            cwd=root,
            capture_output=True,
            check=False,
        )
        if listed.returncode != 0:
            raise errors.InputError(f"{root} is not a git checkout")
        found.update(
            name for name in listed.stdout.decode().split("\0") if name
        )
    return sorted(name for name in found if (root / name).is_file())


def clone(root: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
    """Copies the repository's files into a fresh folder.

    Args:
        root: The repository.
        target: An empty folder.

    Returns:
        `target`.
    """
    for name in files(root):
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, destination)
    return target


def launch(
    command: list[str], repo: pathlib.Path, site: pathlib.Path, seconds: int
) -> Attempt:
    """Runs a command against a repository copy and an extension site.

    Args:
        command: The command.
        repo: The copy; it is the working folder and leads the path.
        site: The extension's package folder, next on the path.
        seconds: The time limit.

    Returns:
        Whether it passed, with the end of its output. Running out of time
        is a failure.
    """
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(repo), str(site)])
    environment["OMP_NUM_THREADS"] = "1"
    try:
        done = subprocess.run(
            command,
            cwd=repo,
            env=environment,
            capture_output=True,
            text=True,
            timeout=seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return Attempt(passed=False, tail=f"timed out after {seconds} s")
    output = (done.stdout + done.stderr)[-TAIL:]
    return Attempt(done.returncode == 0, output)


def probe(
    exercise: Exercise, repo: pathlib.Path, site: pathlib.Path
) -> Attempt:
    """Runs the exercise's probe.

    Args:
        exercise: The exercise.
        repo: The repository copy.
        site: The extension's package folder.

    Returns:
        The attempt.
    """
    script = site / "probe.py"
    script.write_text(PRELUDE + exercise.probe, encoding="utf-8")
    command = [sys.executable, str(script)]
    return launch(command, repo, site, PROBE_SECONDS)


def check(
    exercise: Exercise, repo: pathlib.Path, site: pathlib.Path
) -> Attempt:
    """Runs the test files relevant to the point.

    Args:
        exercise: The exercise.
        repo: The repository copy.
        site: The extension's package folder.

    Returns:
        The attempt; with no test file, a failure, because nothing showed
        the change safe.
    """
    if not exercise.tests:
        return Attempt(
            passed=False, tail="no test file is listed for the point"
        )
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    return launch([*command, *exercise.tests], repo, site, TESTS_SECONDS)


def attempt(exercise: Exercise, root: pathlib.Path) -> dict[str, Any]:
    """Does one exercise in a fresh copy of the repository.

    Args:
        exercise: The exercise.
        root: The repository to copy.

    Returns:
        The point's record: the files edited outside the package, whether
        the probe passed without and with the edits, whether each file is
        needed (the probe fails without it), the tests' outcome and why
        each file is needed.
    """
    with tempfile.TemporaryDirectory(prefix="extensibility-") as scratch:
        repo = clone(root, pathlib.Path(scratch) / "repo")
        site = exercise.install(pathlib.Path(scratch) / "site")
        alone = probe(exercise, repo, site)
        originals = {
            name: (repo / name).read_text(encoding="utf-8")
            for name in exercise.files
        }
        for edit in exercise.edits:
            edit.apply(repo)
        works = probe(exercise, repo, site)
        needed = {}
        for name in exercise.files:
            if len(exercise.files) == 1:
                needed[name] = not alone.passed
                continue
            edited = (repo / name).read_text(encoding="utf-8")
            (repo / name).write_text(originals[name], encoding="utf-8")
            needed[name] = not probe(exercise, repo, site).passed
            (repo / name).write_text(edited, encoding="utf-8")
        tests = check(exercise, repo, site)
    return {
        "point": exercise.point,
        "files": list(exercise.files),
        "alone": alone.passed,
        "works": works.passed,
        "needed": needed,
        "tests": {"passed": tests.passed, "tail": tests.tail},
        "probe": works.tail,
        "reason": dict(exercise.reason),
    }


def decide(record: dict[str, Any], limit: int) -> tuple[str, list[str]]:
    """The verdict of one point and what failed it.

    Args:
        record: A point's record from `attempt`.
        limit: `change_cost_files_max`.

    Returns:
        "pass" or "fail", and one reason per failed condition.
    """
    reasons = []
    if not record["works"]:
        reasons.append("the extension does not work even with the edits")
    spare = sorted(name for name, need in record["needed"].items() if not need)
    if spare:
        reasons.append(f"edits not shown needed: {spare}")
    if not record["tests"]["passed"]:
        reasons.append("its relevant tests fail")
    if len(record["files"]) > limit:
        reasons.append(
            f"{len(record['files'])} files outside the package > {limit}"
        )
    return ("fail" if reasons else "pass"), reasons


def digest() -> str:
    """Hash of the measured code and of the exercises.

    Returns:
        SHA-256 over `MEASURED` (`evidence.source`) and this module.
    """
    found = hashlib.sha256(evidence.source(MEASURED, PACKAGE).encode())
    found.update(pathlib.Path(__file__).read_bytes())
    return found.hexdigest()


def measure(
    chosen: tuple[Exercise, ...], root: pathlib.Path = ROOT
) -> dict[str, Any]:
    """Does the exercises and decides each point.

    Args:
        chosen: The exercises to do.
        root: The repository to copy.

    Returns:
        The document: the code hash, the limit, every point's record with
        its verdict and reasons, and the claim's verdict over all points.
    """
    limit = int(acceptance.load()[VARIABLE].value)
    points = []
    for exercise in chosen:
        record = attempt(exercise, root)
        record["verdict"], record["reasons"] = decide(record, limit)
        points.append(record)
    complete = {e.point for e in chosen} == {e.point for e in EXERCISES}
    verdict = "pass" if all(p["verdict"] == "pass" for p in points) else "fail"
    return {
        "version": 1,
        "claim": "C10",
        "code": digest(),
        "limit": limit,
        "complete": complete,
        "points": points,
        "verdict": verdict if complete else "not measured",
        "not_measured": NOTES,
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
    """Why the document cannot decide the change-cost row.

    Args:
        document: The document; read from `PATH` when None.

    Returns:
        Messages for a missing file, another code hash or limit, or a
        measurement of only some points; empty when it decides.
    """
    found = read(document)
    if found is None:
        return [f"extensibility: {PATH.name} is missing"]
    issues = []
    if found.get("code") != digest():
        issues.append("extensibility: measured on other code")
    if found.get("limit") != int(acceptance.load()[VARIABLE].value):
        issues.append(f"extensibility: measured against another {VARIABLE}")
    if not found.get("complete"):
        issues.append("extensibility: not every extension point was measured")
    return issues


def summary(document: dict[str, Any] | None = None) -> tuple[str, str]:
    """The scorecard's value and verdict for the change-cost criterion.

    Args:
        document: The document; read from `PATH` when None.

    Returns:
        The files each point edits outside its package, and "pass",
        "fail" or "not measured" (missing, stale or partial).
    """
    found = read(document)
    if found is None:
        return "no measurement", "not measured"
    issues = problems(found)
    if issues:
        return "; ".join(issues), "not measured"
    value = "; ".join(
        f"{p['point']}: {len(p['files'])} files ({p['verdict']})"
        for p in found["points"]
    )
    return value, str(found["verdict"])


PRELUDE = '''"""Probe: the extension works end to end, used by name."""

import dataclasses
import os
import pathlib
import tempfile

import numpy
import pandas

import scmle
from scmle.api import project


def rows(seed=0, size=300):
    """Two numeric features and a binary label that depends on the first."""
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame(
        {"a": rng.normal(size=size), "b": rng.normal(size=size)}
    )
    frame["y"] = (frame.a + rng.normal(scale=0.5, size=size) > 0).astype(int)
    return frame


def fit(folder, plan, model, **declared):
    """Plans a project, then runs it with only `model` as the candidate."""
    made = project.Project(
        output_dir=pathlib.Path(folder) / "runs", time_budget=60, **declared
    )
    planned = made.plan()
    plan(planned)
    run = made.run(dataclasses.replace(planned, candidates=(model,)))
    assert run.selected_model == model, run.selected_model
    return run


assert scmle.__file__.startswith(os.getcwd()), scmle.__file__
'''

MODEL = '''"""A new model family: nearest centroid, outside the guarantee."""

from sklearn import neighbors

from scmle.contracts import models
from scmle.models.sklearn import classical


class Centroid(classical.Learner):
    """Nearest-centroid classification."""

    name = "centroid"
    family = "centroid"
    capability_family = "custom"
    capabilities = models.Capabilities(classical.CLASSIFICATION, cost=1)
    classifier = neighbors.NearestCentroid


adapter = Centroid()
'''

MODELPROBE = """

def main():
    with tempfile.TemporaryDirectory() as folder:
        fit(
            folder,
            lambda plan: None,
            "centroid",
            data=rows(),
            target="y",
        )


if __name__ == "__main__":
    main()
"""

METRIC = '''"""A new metric: Youden's J."""

from sklearn import metrics

from scmle.contracts import tasks
import scmle.contracts.metrics


def youden(truth, predicted):
    """Sensitivity plus specificity minus one."""
    return 2.0 * metrics.balanced_accuracy_score(truth, predicted) - 1.0


metric = scmle.contracts.metrics.Formula(
    "youden", "", True, frozenset({tasks.Task.BINARY}), youden, "predict"
)
'''

METRICPROBE = """

def check(plan):
    assert plan.metric == "youden", plan.metric


def main():
    with tempfile.TemporaryDirectory() as folder:
        run = fit(
            folder,
            check,
            "logistic_regression",
            data=rows(),
            target="y",
            metric="youden",
        )
        assert "youden" in run.evaluation["metrics"], run.evaluation


if __name__ == "__main__":
    main()
"""

READER = '''"""A new reader: pipe-separated values."""

import pathlib

import pandas

from scmle.data.readers import tabular


class Pipes(tabular.TableReader):
    """Text tables whose fields are separated by `|`."""

    suffixes = (".psv",)

    def load(self, path: pathlib.Path) -> pandas.DataFrame:
        """Reads the table."""
        return pandas.read_csv(path, sep="|")


reader = Pipes()
'''

READERPROBE = """

def main():
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "rows.psv"
        rows().to_csv(path, sep="|", index=False)
        fit(
            folder,
            lambda plan: None,
            "logistic_regression",
            data=str(path),
            target="y",
        )


if __name__ == "__main__":
    main()
"""


def lines(name: str) -> str:
    """The source of a data kind of labelled text lines.

    Args:
        name: The modality it serves.

    Returns:
        A module that defines the kind as `kind`, declaring library
        inference of the beta classification families on its modality.
    """
    return f'''"""A new data kind: one `label<TAB>text` record per line."""

import pandas

from scmle.contracts import capabilities
from scmle.contracts import tasks
from scmle.pipeline import kinds
from scmle.pipeline import roster


class Lines(kinds.Tabular):
    """Labelled lines of text, profiled and featurised as one text column."""

    name = "{name}"

    def read(self, spec):
        """Reads `label<TAB>text` lines into a `label` and a `text` column."""
        frame = pandas.read_csv(
            spec.data.uri, sep="\\t", names=["label", "text"], dtype=str
        )
        return frame, None

    def declare(self):
        """Library inference of the beta classification families."""
        return tuple(
            capabilities.Capability(
                key=capabilities.Coordinate(
                    modality=capabilities.Modality(self.name),
                    task=task,
                    model_family=family,
                    evaluation=capabilities.Evaluation.STANDARD,
                    uncertainty=capabilities.Uncertainty.NONE,
                    fairness=fairness,
                    explainability=capabilities.Explainability.NONE,
                    inference=capabilities.Inference.LIBRARY,
                ),
                status=capabilities.Status.SUPPORTED,
                required_extras=("core",),
                resources=capabilities.Resources(memory_gb=1.0),
                failure_modes=("a line without a tab is one text field",),
                evidence_links=("probe.py",),
                limitations=("labelled lines only",),
            )
            for task, families in kinds.TABULAR_BETA_TASK_FAMILIES[:2]
            for family in families
            for fairness in (
                capabilities.Fairness.NONE,
                capabilities.Fairness.MITIGATION,
            )
        )


kind = Lines()
'''


def reading(name: str) -> str:
    """The probe of a data kind of labelled text lines.

    Args:
        name: The modality the kind serves.

    Returns:
        A probe that trains on a file of such lines declared as `name`.
    """
    return f"""

WORDS = ("good fine great happy calm", "bad poor sad angry awful")


def main():
    rng = numpy.random.default_rng(0)
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "lines.txt"
        records = []
        for index in range(300):
            label = int(rng.integers(0, 2))
            words = rng.choice(WORDS[label].split(), size=5).tolist()
            words += rng.choice(WORDS[1 - label].split(), size=1).tolist()
            text = " ".join(words)
            records.append(f"{{'yes' if label else 'no'}}\\t{{text}} {{index}}")
        path.write_text("\\n".join(records) + "\\n")
        fit(
            folder,
            lambda plan: None,
            "logistic_regression",
            data=str(path),
            target="label",
            modality="{name}",
        )


if __name__ == "__main__":
    main()
"""


TASK = '''"""A new task type: counts, with a Poisson model and its deviance."""

from sklearn import linear_model
from sklearn import metrics

from scmle.contracts import models
from scmle.contracts import tasks
import scmle.contracts.metrics
from scmle.models.sklearn import classical

COUNT = tasks.Task("count")


class Poisson(classical.Learner):
    """Poisson regression for counts."""

    name = "poisson"
    family = "linear"
    capabilities = models.Capabilities(frozenset({COUNT}), cost=1)
    regressor = linear_model.PoissonRegressor


def deviance(truth, predicted):
    """Mean Poisson deviance; lower is better."""
    return metrics.mean_poisson_deviance(truth, predicted.clip(1e-9))


adapter = Poisson()
metric = scmle.contracts.metrics.Formula(
    "poisson_deviance", "", False, frozenset({COUNT}), deviance, "predict"
)
'''

TASKPROBE = """

def check(plan):
    assert plan.task == "count", plan.task
    assert plan.metric == "poisson_deviance", plan.metric


def main():
    rng = numpy.random.default_rng(0)
    frame = rows()
    frame["y"] = rng.poisson(numpy.exp(0.5 + 0.8 * frame.a))
    with tempfile.TemporaryDirectory() as folder:
        run = fit(
            folder,
            check,
            "poisson",
            data=frame,
            target="y",
            task="count",
        )
        assert "poisson_deviance" in run.evaluation["metrics"]


if __name__ == "__main__":
    main()
"""

# The edits each exercise was found to need, smallest first: the probe was
# run, the first failure fixed at its source, and the probe run again.
DISCOVER = (
    "            shelf = plugins.REGISTRY.discover(plugins.Kind.METRICS)\n"
    "            for found in shelf.items:\n"
    "                if found.name not in self.metrics:\n"
    "                    self.register(found)\n"
)
METRICS = "scmle/evaluation/metrics.py"
RESOLVE = """        name = name or self.default_metric[task]
        if name == "roc_auc" and task is tasks.Task.MULTICLASS:
            name = "roc_auc_ovr"
"""
TABULAR = "scmle/data/readers/tabular.py"
PLUGINS = "scmle/runtime/plugins.py"
KINDS = "scmle/pipeline/kinds/__init__.py"
ROSTER = "scmle/pipeline/roster.py"
CONTRACTS = "scmle/contracts/tasks/kinds.py"
AXES = "scmle/contracts/capabilities/axes.py"
CALIBRATION = "scmle/pipeline/stages/calibration.py"

KINDEDITS = (
    Edit(
        PLUGINS,
        '    TRANSFORMERS = "transformers"\n',
        '    TRANSFORMERS = "transformers"\n    KINDS = "kinds"\n',
    ),
    Edit(
        PLUGINS,
        '    Kind.TRANSFORMERS: "scmle.transformers",\n',
        '    Kind.TRANSFORMERS: "scmle.transformers",\n'
        '    Kind.KINDS: "scmle.kinds",\n',
    ),
    Edit(
        KINDS,
        "from scmle.preparation import preprocessing\n",
        "from scmle.preparation import preprocessing\n"
        "from scmle.runtime import plugins\n",
    ),
    Edit(
        KINDS,
        "    found = KINDS.get(data.modality)\n",
        "    found = known().get(data.modality)\n",
    ),
    Edit(
        KINDS,
        "def pick(data: tasks.Dataset) -> Kind:",
        '''def known() -> dict[str, Kind]:
    """Every kind: the built-in ones and those plugins advertise.

    Returns:
        Kind by modality name; a plugin never replaces a built-in kind.
    """
    found = plugins.REGISTRY.discover(plugins.Kind.KINDS).items
    return {**{item.name: item for item in found}, **KINDS}


def pick(data: tasks.Dataset) -> Kind:''',
    ),
    Edit(
        ROSTER,
        "        record for kind in kinds.KINDS.values() for record in "
        "kind.declare()\n",
        "        record for kind in kinds.known().values() for record in "
        "kind.declare()\n",
    ),
)
KINDREASONS = {
    PLUGINS: "the registry has no kind (entry-point group) for data kinds",
    KINDS: "`pick` reads only the built-in `KINDS` mapping",
    ROSTER: (
        "the capability registry enrols the built-in `KINDS` alone, not the "
        "kinds plugins advertise"
    ),
}

EXERCISES = (
    Exercise(
        point="model family",
        name="scmle_ext_model",
        sources={"extmodel.py": MODEL},
        entries="[scmle.models]\ncentroid = extmodel:adapter\n",
        edits=(),
        probe=MODELPROBE,
        tests=(
            "tests/unit/test_plugins.py",
            "tests/unit/test_registry.py",
            "tests/unit/test_catalog.py",
            "tests/unit/test_capability_use.py",
            "tests/integration/test_plugin_distribution.py",
            "tests/integration/test_register_estimator.py",
            "tests/unit/test_conventions.py",
        ),
        reason={},
    ),
    Exercise(
        point="metric",
        name="scmle_ext_metric",
        sources={"extmetric.py": METRIC},
        entries="[scmle.metrics]\nyouden = extmetric:metric\n",
        edits=(
            Edit(
                METRICS,
                RESOLVE,
                RESOLVE + "        if name not in self.metrics:\n" + DISCOVER,
            ),
        ),
        probe=METRICPROBE,
        tests=(
            "tests/unit/test_evaluation.py",
            "tests/unit/test_plugins.py",
            "tests/unit/test_survival_metrics.py",
            "tests/unit/test_unit_multitargetmetrics.py",
            "tests/integration/test_custom_metric.py",
            "tests/unit/test_conventions.py",
        ),
        reason={
            METRICS: (
                "the metric catalog never reads the `scmle.metrics` group; "
                "only `register_metric` in the same process adds a metric"
            ),
        },
    ),
    Exercise(
        point="reader",
        name="scmle_ext_reader",
        sources={"extreader.py": READER},
        entries="[scmle.readers]\npipes = extreader:reader\n",
        edits=(
            Edit(
                TABULAR,
                "            Feather(),\n        )\n",
                "            Feather(),\n"
                "            *plugins.REGISTRY.discover("
                "plugins.Kind.READERS).items,\n        )\n",
            ),
            Edit(
                TABULAR,
                "from scmle.data.readers import stream\n",
                "from scmle.data.readers import stream\n"
                "from scmle.runtime import plugins\n",
            ),
        ),
        probe=READERPROBE,
        tests=(
            "tests/unit/test_readers.py",
            "tests/unit/test_plugins.py",
            "tests/unit/test_incremental_readers.py",
            "tests/integration/test_interchange.py",
            "tests/unit/test_conventions.py",
        ),
        reason={
            TABULAR: (
                "`Loader` reads a fixed tuple of readers; the `scmle.readers` "
                "group is declared but never read"
            ),
        },
    ),
    Exercise(
        point="data kind (declared modality)",
        name="scmle_ext_kind",
        sources={"extkind.py": lines("text")},
        entries="[scmle.kinds]\ntext = extkind:kind\n",
        edits=KINDEDITS,
        probe=reading("text"),
        tests=(
            "tests/unit/test_kinds.py",
            "tests/unit/test_plugins.py",
            "tests/unit/test_capability_registry.py",
            "tests/unit/test_unit_capabilities.py",
            "tests/integration/test_capability_gate.py",
            "tests/unit/test_conventions.py",
        ),
        reason=KINDREASONS,
    ),
    Exercise(
        point="data kind (new modality)",
        name="scmle_ext_lines",
        sources={"extkind.py": lines("lines")},
        entries="[scmle.kinds]\nlines = extkind:kind\n",
        edits=(
            *KINDEDITS,
            Edit(
                CONTRACTS, '    "raster",\n]', '    "raster",\n    "lines",\n]'
            ),
            Edit(
                AXES,
                '    RASTER = "raster"\n',
                '    RASTER = "raster"\n    LINES = "lines"\n',
            ),
        ),
        probe=reading("lines"),
        tests=(
            "tests/unit/test_kinds.py",
            "tests/unit/test_plugins.py",
            "tests/unit/test_capability_registry.py",
            "tests/unit/test_unit_capabilities.py",
            "tests/unit/test_tasks.py",
            "tests/integration/test_capability_gate.py",
            "tests/unit/test_conventions.py",
        ),
        reason={
            **KINDREASONS,
            CONTRACTS: "`DatasetModality` is a closed literal of names",
            AXES: "the capability `Modality` is a closed enumeration",
        },
    ),
    Exercise(
        point="task type",
        name="scmle_ext_task",
        sources={"exttask.py": TASK},
        entries=(
            "[scmle.models]\npoisson = exttask:adapter\n"
            "[scmle.metrics]\npoisson_deviance = exttask:metric\n"
        ),
        edits=(
            Edit(
                CONTRACTS,
                '    UPLIFT = "uplift"\n',
                '    UPLIFT = "uplift"\n    COUNT = "count"\n',
            ),
            Edit(
                CONTRACTS,
                '    "uplift",\n    "binary_classification",',
                '    "uplift",\n    "count",\n    "binary_classification",',
            ),
            Edit(
                METRICS,
                RESOLVE,
                "        if name not in self.metrics or task not in "
                "self.default_metric:\n"
                + DISCOVER
                + "        name = name or self.default_metric.get(task) or "
                "next(\n"
                "            iter(self.report_names.get(task, ())), None\n"
                "        )\n" + RESOLVE.split("\n", 1)[1],
            ),
            Edit(
                METRICS,
                "                names = report_names[task]\n",
                "                names = report_names.get(task, ())\n",
            ),
            Edit(
                METRICS,
                "        names = self.report_names[task]\n",
                "        names = self.report_names.get(task, ())\n",
            ),
            Edit(
                KINDS,
                '    (tasks.Task.UPLIFT, ("baseline", "linear", "causal")),'
                "\n)\n",
                '    (tasks.Task.UPLIFT, ("baseline", "linear", "causal")),\n'
                '    (tasks.Task.COUNT, ("linear",)),\n)\n',
            ),
            Edit(
                KINDS,
                "    tasks.Task.UPLIFT: capabilities.Evaluation.UPLIFT,\n}",
                "    tasks.Task.UPLIFT: capabilities.Evaluation.UPLIFT,\n"
                "    tasks.Task.COUNT: capabilities.Evaluation.STANDARD,\n}",
            ),
            Edit(
                CALIBRATION,
                "    tasks.Task.UPLIFT,\n)\n",
                "    tasks.Task.UPLIFT,\n    tasks.Task.COUNT,\n)\n",
            ),
        ),
        probe=TASKPROBE,
        tests=(
            "tests/unit/test_tasks.py",
            "tests/unit/test_unit_tasks.py",
            "tests/unit/test_task_resolution.py",
            "tests/unit/test_evaluation.py",
            "tests/unit/test_capability_registry.py",
            "tests/integration/test_capability_gate.py",
            "tests/integration/test_calibration_reports.py",
            "tests/unit/test_conventions.py",
        ),
        reason={
            CONTRACTS: "`Task` and `TaskRequest` are closed enumerations",
            METRICS: (
                "the default and reported metrics are tables keyed by every "
                "task, and plugin metrics are never discovered"
            ),
            KINDS: (
                "the tabular kind's capability families and evaluation of a "
                "task are tables keyed by every task"
            ),
            CALIBRATION: (
                "conformal calibration treats every task outside a fixed "
                "list as classification or regression"
            ),
        },
    ),
)

NOTES = [
    (
        "streamed reads (`memory_budget_gb`) and erasure keep their own "
        "format tables (`data/readers/streaming/feed.py`, "
        "`api/erasure/steps.py`); the reader exercise covers in-memory "
        "reads only"
    ),
    (
        "serving and the CLI are not probed; the probes train and package "
        "through `Project`"
    ),
    (
        "the exercise was scripted by the code's author, not by someone "
        "new to it, as the plan's evidence column asks"
    ),
]


def main(arguments: list[str] | None = None) -> int:
    """Measures (`--show`, `--write`) or checks (`--check`) the criterion.

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
    parser.add_argument("--point", action="append", default=[])
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    options = parser.parse_args(arguments)
    if options.check:
        issues = problems()
        print(json.dumps({"ok": not issues, "problems": issues}, indent=2))
        return 0 if not issues else 1
    if options.write and options.point:
        parser.error("--write measures every point; --point is for --show")
    known = {exercise.point: exercise for exercise in EXERCISES}
    unknown = sorted(set(options.point) - set(known))
    if unknown:
        parser.error(f"unknown points {unknown}; known: {sorted(known)}")
    chosen = tuple(known[p] for p in options.point) or EXERCISES
    document = measure(chosen, options.root)
    text = json.dumps(document, indent=2, sort_keys=True)
    if options.write:
        PATH.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
