"""Runs an `Project` as the workload of a job, unchanged.

The project specification is a JSON mapping::

    {
        "data": "data/train.csv",  # a key in the store
        "target": "churn",
        "options": {"time_budget": 600},
    }  # keyword arguments of Project

The data object is copied from the store to local scratch space, the project
runs there, and its run directory is mirrored to ``runs/<id>/workdir/`` in the
store: periodically while it runs (`WorkdirSync`) and once more when it ends,
however it ends. After a restart on a machine with an empty disk the
workdir is restored and `Project.resume` reuses the completed trials. A
hard kill therefore loses at most the trials finished since the last
periodic sync (default 30 seconds). The pipeline itself is not modified:
this module only moves files and calls the public API.
"""

from collections.abc import Callable
from collections.abc import Mapping
import hashlib
import pathlib
import shutil
import tempfile
import threading
from typing import Any

import scmle.api.project
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.store import objectstore
from scmle.runners import local

SYNC_SECONDS = 30.0
SECTIONS = {
    "search": tasks.Search,
    "strategies": tasks.Strategies,
    "models": tasks.Candidates,
    "ensemble": tasks.Stacking,
    "feature_graph": tasks.FeatureGraph,
    "selection": tasks.Selection,
    "decision": tasks.Threshold,
    "fairness": tasks.Parity,
    "problem": tasks.Problem,
}
CONFIG_NAMES = ("config.json", "spec.json")


class WorkdirSync:
    """Mirrors a local directory to ``<prefix>`` in a store, and back.

    Attributes:
        store: Where the mirror lives.
        prefix: Key prefix of the mirror.
        directory: The local directory.
        uploaded: SHA-256 of each file as last uploaded or restored, by relative
            path.
        stop: Set to end the periodic push.
        thread: The periodic push thread, once started.
    """

    def __init__(
        self,
        store: objectstore.ObjectStore,
        prefix: str,
        directory: pathlib.Path,
    ) -> None:
        """Initializes the mirror.

        Args:
            store: Where the mirror lives.
            prefix: Key prefix (with a trailing slash).
            directory: The local directory.
        """
        self.store = store
        self.prefix = prefix
        self.directory = directory
        self.uploaded: dict[str, str] = {}
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def push(self) -> int:
        """Uploads new and changed files.

        Files that cannot be read are skipped.

        Returns:
            The number of files uploaded.
        """
        sent = 0
        for path in sorted(self.directory.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(self.directory).as_posix()
            try:
                data = path.read_bytes()
            except OSError:
                continue
            digest = hashlib.sha256(data).hexdigest()
            if self.uploaded.get(relative) == digest:
                continue
            self.store.put(self.prefix + relative, data)
            self.uploaded[relative] = digest
            sent += 1
        return sent

    def restore(self) -> int:
        """Downloads the mirror into the directory.

        Returns:
            The number of files downloaded.
        """
        count = 0
        for meta in self.store.list_objects(self.prefix):
            relative = meta.key.removeprefix(self.prefix)
            target = self.directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            data, ignored = self.store.get(meta.key)
            target.write_bytes(data)
            self.uploaded[relative] = meta.sha256
            count += 1
        return count

    def start(self, interval: float) -> None:
        """Pushes every `interval` seconds on a daemon thread.

        A push that fails with a store error is skipped and tried again at the
        next interval.

        Args:
            interval: Seconds between pushes.
        """

        def loop() -> None:
            """Pushes until `stop` is set."""
            while not self.stop.wait(interval):
                try:
                    self.push()
                except objectstore.StoreError:
                    continue

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def close(self) -> None:
        """Stops the thread and does a final push."""
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
        self.push()


def saved_run(directory: pathlib.Path) -> pathlib.Path | None:
    """Finds the run directory inside a restored workdir.

    Args:
        directory: The restored workdir.

    Returns:
        The directory holding the first `config.json` or `spec.json` found, or
        None if there is none.
    """
    for config in sorted(directory.rglob("*.json")):
        if config.name in CONFIG_NAMES:
            return config.parent
    return None


def project_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Builds keyword arguments for `Project` from a JSON mapping.

    The nested sections (`search`, `models`, ...) arrive as mappings and are
    turned into their specification types; everything else passes through.

    Args:
        options: The `options` of the project specification.

    Returns:
        The keyword arguments.
    """
    converted = dict(options)
    for name, kind in SECTIONS.items():
        if isinstance(converted.get(name), Mapping):
            converted[name] = kind(**converted[name])
    return converted


def run_project(
    context: local.JobContext,
    sync_seconds: float = SYNC_SECONDS,
    on_start: Callable[[Any], None] | None = None,
) -> None:
    """Runs (or resumes) the job's project and records its result.

    The scratch directory has the same path on every attempt, because a saved
    run refers to its data by path. If the run is cancelled the job is reported
    interrupted, so the supervisor retries and the retry resumes. If a previous
    attempt finished the project but not the job, the saved run cannot be
    resumed ("already completed") and the project runs again from scratch into a
    new run directory.

    Args:
        context: The job context; its spec's `project` is the mapping above.
        sync_seconds: Seconds between mirrors of the run directory.
        on_start: Called with the `Project` just before it runs (a hook for
            cancellation).

    Raises:
        ConfigError: If the project mapping lacks `data` or `target`.
        JobInterruptedError: If the project was cancelled.
    """
    spec = context.spec.project
    if "data" not in spec or "target" not in spec:
        raise errors.ConfigError("the project needs 'data' and 'target'")
    root = pathlib.Path(tempfile.gettempdir()) / f"scmle-job-{context.run_id}"
    shutil.rmtree(root, ignore_errors=True)
    try:
        data_path = root / "data" / pathlib.Path(str(spec["data"])).name
        data_path.parent.mkdir(parents=True)
        content, ignored = context.store.get(str(spec["data"]))
        data_path.write_bytes(content)
        workdir = root / "runs"
        workdir.mkdir()
        sync = WorkdirSync(
            context.store, f"runs/{context.run_id}/workdir/", workdir
        )
        restored = sync.restore()
        sync.start(sync_seconds)
        try:
            result, resumed = execute(
                spec, data_path, workdir, restored > 0, on_start
            )
        except errors.CancelledError as error:
            raise local.JobInterruptedError("project cancelled") from error
        finally:
            sync.close()
        context.step(1, {"run": result.run_id})
    finally:
        shutil.rmtree(root, ignore_errors=True)
    context.finish(summarize(result, resumed))


def execute(
    spec: Mapping[str, Any],
    data_path: pathlib.Path,
    workdir: pathlib.Path,
    restored: bool,
    on_start: Callable[[Any], None] | None,
) -> tuple[Any, bool]:
    """Resumes a restored run or starts a new one.

    Args:
        spec: The project specification.
        data_path: Local copy of the training data.
        workdir: Directory for the run, possibly restored from the store.
        restored: Whether the workdir was restored from earlier attempts.
        on_start: Called with a newly started `Project` before it runs.

    Returns:
        The run result and whether it was resumed.

    Raises:
        ConfigError: If resuming the saved run fails for any reason other than
            that the run had already completed (in which case a new run is
            started).
    """
    previous = saved_run(workdir) if restored else None
    if previous is not None:
        try:
            return scmle.api.project.Project.resume(previous), True
        except errors.ConfigError as error:
            if "already completed" not in str(error):
                raise
    started = scmle.api.project.Project(
        data_path,
        spec["target"],
        output_dir=workdir,
        **project_options(spec.get("options", {})),
    )
    if on_start is not None:
        on_start(started)
    return started.run(), False


def summarize(result: Any, resumed: bool) -> dict[str, Any]:
    """Summarizes a finished project as the JSON result to record.

    Args:
        result: The project's run result.
        resumed: Whether the run was resumed.

    Returns:
        A dict with the run id, state, whether it was resumed, the selected
        model, the metric, its cross-validation score and test metrics, each
        trial's model and status, and the result summary.
    """
    evaluation = result.evaluation
    return {
        "run_id": result.run_id,
        "state": result.state,
        "resumed": resumed,
        "selected_model": result.selected_model,
        "metric": evaluation["metric"],
        "selected_cv_score": evaluation["selected_cv_score"],
        "test_metrics": evaluation["test_metrics"],
        "trials": [
            {
                "model": trial.model_name,
                "status": getattr(trial.status, "value", str(trial.status)),
            }
            for trial in result.trials
        ],
        "summary": result.summary(),
    }
