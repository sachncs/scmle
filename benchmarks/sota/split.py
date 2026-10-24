"""Deterministic development / hold-out split of the benchmark tasks.

The hold-out tasks stay sealed until the one evaluation the protocol allows,
so the assignment must be reproducible from a recorded seed and independent of
any result. Tasks are grouped by problem type, ordered by a hash of the seed
and the OpenML task id, and the first share of each group becomes the
development set.

Usage:
    python -m benchmarks.sota.split [--check]
"""

import argparse
import csv
import hashlib
import json
import logging
import math
import pathlib
import sys

LOGGER = logging.getLogger(__name__)

HERE = pathlib.Path(__file__).parent
TASKS = HERE / "tasks_tabarena_v0pt1.csv"
OUTPUT = HERE / "split.json"
SEED = "scmle-sota-protocol-v1"
DEVELOPMENT_SHARE = 0.6


def load_tasks(path: pathlib.Path = TASKS) -> list[dict[str, str]]:
    """Reads the pinned task table."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def rank(seed: str, task_id: str) -> str:
    """The sort key that decides a task's place within its group."""
    return hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()


def assign(
    tasks: list[dict[str, str]],
    seed: str = SEED,
    share: float = DEVELOPMENT_SHARE,
) -> dict[str, list[int]]:
    """Splits task ids into `development` and `holdout`, per problem type."""
    groups: dict[str, list[str]] = {}
    for task in tasks:
        groups.setdefault(task["problem_type"], []).append(task["task_id"])
    development: list[int] = []
    holdout: list[int] = []
    for _, ids in sorted(groups.items()):
        ordered = sorted(ids, key=lambda task_id: rank(seed, task_id))
        cut = math.ceil(len(ordered) * share)
        development.extend(int(task_id) for task_id in ordered[:cut])
        holdout.extend(int(task_id) for task_id in ordered[cut:])
    return {"development": sorted(development), "holdout": sorted(holdout)}


def document(tasks: list[dict[str, str]]) -> dict[str, object]:
    """The committed split, with the rule that produced it."""
    return {
        "seed": SEED,
        "rule": (
            "group by problem_type; order by sha256(seed:task_id); the first "
            f"ceil({DEVELOPMENT_SHARE} * n) of each group is development"
        ),
        "task_table": TASKS.name,
        **assign(tasks),
    }


def main(argv: list[str] | None = None) -> int:
    """Writes `split.json`, or with `--check` verifies the committed file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    expected = document(load_tasks())
    if args.check:
        found = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if found != expected:
            LOGGER.error("split.json does not match the recorded seed")
            return 1
        LOGGER.info("split.json matches the recorded seed")
        return 0
    OUTPUT.write_text(
        json.dumps(expected, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    LOGGER.info("wrote %s", OUTPUT.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
