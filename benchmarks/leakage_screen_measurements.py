"""Measures what the leakage screen flags on shipped tables.

For each table the script records which columns the certified screen
(`scmle.data.admissibility.screen`, settings `leakage.CERTIFIED`, Benjamini-
Yekutieli level 0.05) calls suspects, and compares them with a ground truth
that is decided by provenance, not by the data:

* scikit-learn tables (breast cancer, wine, iris, digits): every column is a
  legitimate measurement, so every suspect is a false flag.
* Titanic: `boat` and `body` are recorded after the outcome (leaks); the
  other columns are known at boarding.
* Planted leaks: five leak types planted among real breast-cancer columns,
  several seeds each; the planted column is the only leak.

It also copies the certified false-alarm and detection rates from
`benchmarks/qualification/screen_measurements.json`, so the documentation
quotes one committed file.

    python -m benchmarks.leakage_screen_measurements --write
    python -m benchmarks.leakage_screen_measurements --check
"""

import argparse
from collections.abc import Sequence
import dataclasses
import json
from pathlib import Path
import sys
from typing import Any

import pandas
from sklearn import datasets

from benchmarks.qualification import f4_certify
from benchmarks.qualification import screen_certify
from scmle.data.admissibility import leakage
from scmle.data.admissibility import screen

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).with_name("leakage_screen_measurements.json")
CERTIFIED = Path(__file__).with_name("qualification") / (
    "screen_measurements.json"
)
TITANIC = ROOT / "tests" / "fixtures" / "titanic_extract.csv"
LEAKED_TITANIC = ("boat", "body")
PLANT_SEEDS = tuple(range(10))
PLANT_COLUMNS = 9
TARGET = "target"


def sklearn_table(loader: Any) -> pandas.DataFrame:
    """A shipped scikit-learn table with its target column.

    Args:
        loader: A `sklearn.datasets.load_*` function accepting `as_frame`.

    Returns:
        The frame (features and target) with a fresh 0..n-1 index.
    """
    return loader(as_frame=True).frame.reset_index(drop=True)


def screened(frame: pandas.DataFrame, target: str) -> screen.Screen:
    """Runs the certified screen with the shipped certificates.

    Args:
        frame: Table to screen, including the target column.
        target: Name of the target column.

    Returns:
        The screen report, computed with `leakage.CERTIFIED` at run seed 0
        for a classification task.
    """
    cfg = dataclasses.replace(leakage.CERTIFIED, run_seed=0)
    return screen.screen(
        frame, target, "classification", cfg, store=screen.shipped_store()
    )


def flagged_names(report: screen.Screen) -> list[str]:
    """Distinct suspect columns, in column order of first appearance.

    Args:
        report: Screen report to read the suspects from.

    Returns:
        Column names, each once, in the order the suspects are listed.
    """
    names: list[str] = []
    for finding in report.suspects:
        if str(finding.feature) not in names:
            names.append(str(finding.feature))
    return names


def blocking_names(report: screen.Screen) -> list[str]:
    """Distinct columns that would stop a run under the block policy.

    Args:
        report: Screen report to read the blocking findings from.

    Returns:
        Column names, each once, in the order the findings are listed.
    """
    names: list[str] = []
    for finding in report.blocking:
        if str(finding.feature) not in names:
            names.append(str(finding.feature))
    return names


def measure_table(
    name: str, frame: pandas.DataFrame, target: str, leaks: Sequence[str]
) -> dict[str, Any]:
    """Screens one table and scores the flags against the provenance.

    Args:
        name: Label of the table in the output.
        frame: The table, including the target column.
        target: Name of the target column.
        leaks: Feature columns known to be recorded after the outcome.

    Returns:
        Row, class and column counts, the screen's family and gate values,
        the flagged and would-block columns, the false flags (flagged
        legitimate columns), the leaks found, and the false-flag rate (None
        when there is no legitimate column).
    """
    report = screened(frame, target)
    features = [c for c in frame.columns if c != target]
    legitimate = [c for c in features if c not in leaks]
    flagged = flagged_names(report)
    false_flags = [c for c in flagged if c in legitimate]
    found = [c for c in flagged if c in leaks]
    return {
        "table": name,
        "rows": len(frame),
        "classes": int(frame[target].nunique()),
        "columns": len(features),
        "family": report.family,
        "gating": sorted({str(v.value) for v in report.gates.values()}),
        "true_leaks": list(leaks),
        "flagged": flagged,
        "would_block": blocking_names(report),
        "false_flags": false_flags,
        "leaks_found": found,
        "legitimate_columns": len(legitimate),
        "false_flag_rate": (
            len(false_flags) / len(legitimate) if legitimate else None
        ),
    }


def titanic_tables() -> list[dict[str, Any]]:
    """Titanic with and without the columns recorded after the outcome.

    Returns:
        Two measurements: the full table, where `boat` and `body` are the
        true leaks, and the table without them, which has none.
    """
    frame = pandas.read_csv(TITANIC)
    clean = frame.drop(columns=list(LEAKED_TITANIC))
    return [
        measure_table("titanic", frame, "survived", LEAKED_TITANIC),
        measure_table("titanic without boat and body", clean, "survived", ()),
    ]


def planted() -> dict[str, Any]:
    """Plants each leak type among real breast-cancer columns.

    Returns:
        For every defect type, the number of tables screened, how many had
        the planted column flagged, and how many real columns were screened
        and flagged, over the seeds in `PLANT_SEEDS`.
    """
    frame = f4_certify.breast_cancer()
    rows: dict[str, Any] = {}
    for defect in screen_certify.DEFECTS:
        found = 0
        false_flags = 0
        real_total = 0
        for seed in PLANT_SEEDS:
            names = screen_certify.real_columns(frame, PLANT_COLUMNS, seed)
            plant = screen_certify.plant_column(frame, defect, seed)
            table = frame[[*names, f4_certify.TARGET]].assign(**{defect: plant})
            report = screened(table, f4_certify.TARGET)
            flagged = flagged_names(report)
            found += int(defect in flagged)
            false_flags += len([c for c in flagged if c in names])
            real_total += len(names)
        rows[defect] = {
            "tables": len(PLANT_SEEDS),
            "detected": found,
            "real_columns_screened": real_total,
            "real_columns_flagged": false_flags,
        }
    return rows


def certified_rates() -> dict[str, Any]:
    """The certified false-alarm and detection rates, from the repository.

    Returns:
        The values copied from `screen_measurements.json`: the level, null
        replicates and false alarms with their Wilson interval, plant
        replicates per type, detections with their intervals, and the
        certified ceilings.
    """
    record = json.loads(CERTIFIED.read_text(encoding="utf-8"))
    return {
        "source": "benchmarks/qualification/screen_measurements.json",
        "alpha": record["alpha"],
        "null_replicates": record["null_replicates"],
        "null_worlds_with_a_suspect": record["suspect_false_alarms"],
        "null_false_alarm_wilson_95": record["suspect_false_alarm_wilson_95"],
        "plant_replicates_per_type": record["plant_replicates_per_type"],
        "detected": {k: v["detected"] for k, v in record["recall"].items()},
        "detection_wilson_95": {
            k: v["wilson_95"] for k, v in record["recall"].items()
        },
        "skill_ceiling": record["settings"]["skill_ceiling"],
        "association_ceiling": record["settings"]["association_ceiling"],
    }


def build() -> dict[str, Any]:
    """Runs every measurement.

    Returns:
        The record written to the output file: the screen settings, the
        certified rates, the per-table measurements and the planted-leak
        results.
    """
    tables = [
        measure_table(
            "breast_cancer",
            sklearn_table(datasets.load_breast_cancer),
            "target",
            (),
        ),
        measure_table("wine", sklearn_table(datasets.load_wine), "target", ()),
        measure_table("iris", sklearn_table(datasets.load_iris), "target", ()),
        measure_table(
            "digits", sklearn_table(datasets.load_digits), "target", ()
        ),
        *titanic_tables(),
    ]
    return {
        "settings": {
            "skill_ceiling": leakage.CERTIFIED.skill_ceiling,
            "association_ceiling": leakage.CERTIFIED.association_ceiling,
            "alpha": leakage.CERTIFIED.alpha,
            "permutations": leakage.CERTIFIED.permutations,
            "run_seed": 0,
        },
        "certified": certified_rates(),
        "tables": tables,
        "planted": planted(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Writes or checks the measurements file.

    Args:
        argv: Command-line arguments (`--write` or `--check`); `None` reads
            `sys.argv`.

    Returns:
        0 after a write, or after a check that finds the file equal to a
        fresh measurement; 1 when the check finds a difference.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    record = build()
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.write:
        OUTPUT.write_text(text, encoding="utf-8")
        return 0
    same = json.loads(OUTPUT.read_text(encoding="utf-8")) == json.loads(text)
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
