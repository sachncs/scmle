"""Probes of the environment a bundle is built in."""

import importlib
import importlib.metadata
import pathlib
import shutil
import subprocess  # nosec B404

import threadpoolctl

import scmle

# Policy: `git rev-parse` answers in milliseconds; a hung git (a lock, a
# network filesystem) must not stall packaging, so give up after this long.
GIT_SECONDS = 5.0
# A full SHA-1 commit id; anything else is not a commit.
COMMIT_LENGTH = 40
TRACKED_LIBRARIES = (
    "numpy",
    "scipy",
    "joblib",
    "lightgbm",
    "xgboost",
    "catboost",
    "imblearn",
)


def inventory() -> dict[str, str]:
    """Lists every installed distribution and its version (for the lock).

    Returns:
        A mapping from normalized distribution name (lower case, with `_`
        replaced by `-`) to version, sorted by name.
    """
    found = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if name:
            found[name.lower().replace("_", "-")] = distribution.version
    return dict(sorted(found.items()))


def revision() -> str:
    """The git commit of the scmle source that is running, if known.

    Returns:
        The 40-character commit of the repository holding the scmle
        package, or an empty string when scmle runs from an installed wheel
        (its version then names the code) or git is unavailable.
    """
    root = pathlib.Path(scmle.__file__).resolve().parent.parent
    executable = shutil.which("git")
    if executable is None or not (root / ".git").exists():
        return ""
    try:
        done = subprocess.run(  # noqa: S603 -- fixed arguments  # nosec B603
            [executable, "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=GIT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        # handled: provenance is best effort; an empty commit is recorded
        return ""
    commit = done.stdout.strip()
    return (
        commit if done.returncode == 0 and len(commit) == COMMIT_LENGTH else ""
    )


def pools() -> tuple[dict[str, int], str]:
    """The math-library thread pools loaded now and the BLAS in use.

    Returns:
        The thread count per loaded library (`openmp`, `openblas`, `mkl`,
        ...) and the BLAS implementation's name, empty if none is loaded.
    """
    info = threadpoolctl.threadpool_info()
    threads = {
        str(pool["internal_api"]): int(pool["num_threads"]) for pool in info
    }
    blas = next(
        (
            str(pool["internal_api"])
            for pool in info
            if pool.get("user_api") == "blas"
        ),
        "",
    )
    return dict(sorted(threads.items())), blas


def detect() -> dict[str, str]:
    """Reports the versions of the tracked libraries that are installed.

    Returns:
        A mapping from library name in `TRACKED_LIBRARIES` to its version
        (`unknown` if it has none); libraries that cannot be imported are left
        out.
    """
    found = {}
    for name in TRACKED_LIBRARIES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            # handled: unimportable library left out of mapping, as docstring
            # states
            continue
        found[name] = str(getattr(module, "__version__", "unknown"))
    return found
