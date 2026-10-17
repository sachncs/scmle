"""Fails when benchmark results fall below the recorded baseline.

Usage:
    python -m benchmarks.run_benchmarks --offline --output results.json
    python -m benchmarks.check_regression results.json

Each baseline entry sets a bound on the locked-test score. Scores are
compared in the metric's own direction, so a lower-is-better metric such as
log loss gets an upper bound.
"""

import argparse
import json
import logging
import pathlib
import sys

from benchmarks import intervals

BASELINE = pathlib.Path(__file__).with_name("baseline.json")
LOWER_IS_BETTER = {"log_loss", "rmse", "mae"}


def violations(results: list[dict], baseline: dict[str, dict]) -> list[str]:
    """Lists every benchmark that regressed or is missing.

    Args:
        results: The `results` list written by `run_benchmarks`.
        baseline: Mapping of dataset name to `{"metric", "bound"}`.

    Returns:
        Human-readable violations; empty when everything holds.
    """
    by_name = {record["name"]: record for record in results}
    problems = []
    for name, expected in baseline.items():
        record = by_name.get(name)
        if record is None:
            problems.append(f"{name}: no result recorded")
            continue
        if record["status"] != "ok":
            problems.append(f"{name}: run failed")
            continue
        if record["metric"] != expected["metric"]:
            was, now = expected["metric"], record["metric"]
            problems.append(f"{name}: metric changed {was} -> {now}")
            continue
        score, bound = record["test_score"], expected["bound"]
        if record.get("seeds", 1) > 1:
            # Repeated runs: a breach needs the whole interval beyond the
            # bound, so noise alone cannot fail the gate.
            scores = record["test_scores"]
            low, high = intervals.mean_interval(scores)[1:]
            score = low if expected["metric"] in LOWER_IS_BETTER else high
        worse = (
            score > bound
            if expected["metric"] in LOWER_IS_BETTER
            else score < bound
        )
        if worse:
            metric = expected["metric"]
            problems.append(
                f"{name}: {metric} {score:.4f} breaches bound {bound}"
            )
    return problems


LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Checks a results file against the baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results")
    parser.add_argument("--baseline", default=str(BASELINE))
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", stream=sys.stdout
    )
    results = json.loads(pathlib.Path(args.results).read_text())["results"]
    baseline = json.loads(pathlib.Path(args.baseline).read_text())
    problems = violations(results, baseline)
    for problem in problems:
        LOGGER.error("REGRESSION %s", problem)
    LOGGER.info(
        "checked %d benchmarks, %d regression(s)", len(baseline), len(problems)
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
