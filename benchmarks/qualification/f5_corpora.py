"""F5 detectors measured on more corpora than the one they are certified on.

The shipped certificates (`detector_certificates.json`) plant defects in the
breast-cancer table only (see `f5.py`). This module repeats the same plants
and the same detectors, with the same pre-registered floors (recall lower
Wilson bound at least 0.80, false-alarm upper bound at most 0.05, 95
percent), on every other table that ships with scikit-learn and so needs no
download: wine (13 continuous features, 3 classes), iris (4 continuous
features, 3 classes, 150 rows) and digits (64 integer pixel counts, 10
classes; the always-zero pixel columns are dropped, because the duplicate
planter cannot jitter a column without spread). The target of each is the
class label.

Plants are the ones of `f5.py`, with columns taken from the front of each
table (the first two for sentinels, the first three for outliers, the first
six for missingness) and class-conditional label noise at 0.20 for the first
class and 0.05 for the others. Nothing here was tuned: the plan, the seeds
and the floors were written down before the first run.

What this does and does not show. Each corpus is judged by rows, as in the
shipped certificate, so a corpus verdict inherits the row-independence
caveat. The unit of replication across corpora is the table: a defect type
is reported as passing on k of the n tables, and with n this small that
count carries almost no statistical weight (three of three tables passing
has a Wilson lower bound near 0.44). The result is evidence about
robustness to the kind of table, not a certificate; it never gates a run and
is not read by `--check`. The corpus hashes are in the output, and a
detector's code hash is recorded so the file goes stale with the code.

Result of the first run (2026-11-04, 5 seeds, row-level Wilson bounds):
duplicates (exact and near) and outlier triage meet the floors on all three
tables; label noise meets them on digits only (recall lower bound 0.51 and
0.35 on wine, 0.67 and 0.61 on iris, against 0.80); the sentinel detector
flags the clean rows of every table (29 of 150 on iris, 34 of 178 on wine, all
1797 on digits, against a ceiling of 5 percent), so its certificate on the
breast-cancer table does not carry over to tables with natural repeated or
zero values.

    python -m benchmarks.qualification.f5_corpora --write
    python -m benchmarks.qualification.f5_corpora --check
"""

import argparse
from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn import datasets

from benchmarks.qualification import certify
from benchmarks.qualification import f5
from benchmarks.qualification import inject
from benchmarks.qualification import standins
from scmle.data.detectors import certificates
from scmle.data.detectors import registry

OUTPUT_PATH = Path(__file__).with_name("f5_corpora_measurements.json")
TARGET = standins.TARGET
CORPORA: dict[str, Callable[..., Any]] = {
    "wine": datasets.load_wine,
    "iris": datasets.load_iris,
    "digits": datasets.load_digits,
}
SENTINEL_COUNT = 2
OUTLIER_COUNT = 3
FIRST_CLASS_RATE = 0.20
OTHER_CLASS_RATE = 0.05


def corpus_frame(name: str) -> pd.DataFrame:
    """The bundled table `name` with its class label in `TARGET`."""
    frame = CORPORA[name](as_frame=True).frame
    frame = frame.rename(columns={"target": TARGET})
    constant = [c for c in frame.columns if frame[c].nunique() < 2]
    return frame.drop(columns=constant).reset_index(drop=True)


def feature_names(frame: pd.DataFrame) -> list[str]:
    """Columns of the table other than the label, in order."""
    return [name for name in frame.columns if name != TARGET]


def class_rates(frame: pd.DataFrame) -> dict[int, float]:
    """Class-conditional flip rates: 0.20 for the first class, else 0.05."""
    classes = sorted(frame[TARGET].unique())
    return {
        int(c): FIRST_CLASS_RATE if i == 0 else OTHER_CLASS_RATE
        for i, c in enumerate(classes)
    }


def plans(frame: pd.DataFrame) -> dict[str, list[certify.DefectPlan]]:
    """The plans of `f5.row_plans`, with columns picked from this table."""
    names = feature_names(frame)
    sentinel_columns = names[:SENTINEL_COUNT]
    outlier_columns = names[:OUTLIER_COUNT]
    rates = class_rates(frame)
    return {
        "f5-sentinel": [
            certify.DefectPlan(
                "sentinel",
                lambda f, s: inject.inject_sentinels(
                    f, sentinel_columns, 0.10, s
                ),
            )
        ],
        "f5-duplicates": [
            certify.DefectPlan(
                "duplicate_exact",
                lambda f, s: inject.inject_duplicates(f, 0.10, s),
            ),
            certify.DefectPlan(
                "duplicate_near",
                lambda f, s: inject.inject_duplicates(
                    f, 0.10, s, near=True, scale=1e-3
                ),
            ),
        ],
        "f5-label-noise": [
            f5.renamed(
                certify.DefectPlan(
                    "label_flip",
                    lambda f, s: inject.flip_labels(f, TARGET, 0.10, s),
                ),
                "label_flip_symmetric",
            ),
            f5.renamed(
                certify.DefectPlan(
                    "label_flip",
                    lambda f, s: inject.flip_labels(
                        f,
                        TARGET,
                        0.0,
                        s,
                        mode="class_conditional",
                        class_rates=rates,
                    ),
                ),
                "label_flip_class_conditional",
            ),
        ],
        "f5-outlier-triage": [
            f5.renamed(
                certify.DefectPlan(
                    "outlier",
                    lambda f, s: inject.inject_outliers(
                        f, outlier_columns, 0.10, s, magnitude=8.0
                    ),
                ),
                "outlier_error",
            )
        ],
    }


def detectors(
    frame: pd.DataFrame,
) -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    """The F5 row detectors with outlier provenance from this table."""
    original = f5.OUTLIER_COLUMNS
    f5.OUTLIER_COLUMNS = feature_names(frame)[:OUTLIER_COUNT]
    try:
        return f5.row_detectors(frame)
    finally:
        f5.OUTLIER_COLUMNS = original


def measure_corpus(name: str) -> dict[str, Any]:
    """Outcomes of every F5 detector on one table."""
    frame = corpus_frame(name)
    table_plans = plans(frame)
    found = detectors(frame)
    rows: list[dict[str, Any]] = []
    for detector_id, detector in found.items():
        cert = certify.certify_detector(
            detector_id,
            detector,
            registry.SPECS[detector_id].code_hash(),
            frame,
            table_plans[detector_id],
            f5.ROW_SEEDS,
            f5.POLICY,
        )
        for outcome in cert.outcomes:
            item = outcome.certificate
            rows.append(
                {
                    "detector": detector_id,
                    "defect_type": item.defect_type,
                    "planted": item.planted,
                    "detected": item.detected,
                    "recall_lower": round(outcome.recall_interval[0], 4),
                    "clean_rows": item.clean_rows,
                    "false_alarms": item.false_alarms,
                    "false_alarm_upper": round(
                        outcome.false_alarm_interval[1], 4
                    ),
                    "passes_floor": outcome.passed,
                }
            )
    return {
        "rows": len(frame),
        "features": len(feature_names(frame)),
        "corpus_sha256": certify.frame_digest(frame),
        "outcomes": rows,
    }


def tally(tables: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Per defect type, on how many of the tables the floor was met."""
    keys: dict[tuple[str, str], int] = {}
    for table in tables.values():
        for row in table["outcomes"]:
            key = (row["detector"], row["defect_type"])
            keys[key] = keys.get(key, 0) + int(row["passes_floor"])
    count = len(tables)
    return [
        {
            "detector": detector,
            "defect_type": defect,
            "tables_passing": passed,
            "tables": count,
            "table_level_wilson_lower": round(
                certificates.wilson_interval(passed, count, 0.95)[0], 4
            ),
        }
        for (detector, defect), passed in sorted(keys.items())
    ]


def measure_all() -> dict[str, Any]:
    """Every corpus and the per-defect tally."""
    tables = {name: measure_corpus(name) for name in CORPORA}
    return {
        "policy": {
            "recall_floor": f5.POLICY.recall_floor,
            "false_alarm_ceiling": f5.POLICY.false_alarm_ceiling,
            "confidence": f5.POLICY.confidence,
        },
        "seeds": list(f5.ROW_SEEDS),
        "code_hashes": {
            key: spec.code_hash() for key, spec in registry.SPECS.items()
        },
        "tables": tables,
        "tally": tally(tables),
    }


def stale() -> list[str]:
    """Detectors whose code changed since the measurement file was written."""
    document = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    return [
        key
        for key, digest in document["code_hashes"].items()
        if key not in registry.SPECS
        or registry.SPECS[key].code_hash() != digest
    ]


def main(arguments: list[str] | None = None) -> int:
    """Measures (`--write`) or checks the file against the code (`--check`)."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    options = parser.parse_args(arguments)
    if options.check:
        found = stale()
        print(json.dumps({"stale": found}))
        return 1 if found else 0
    document = measure_all()
    OUTPUT_PATH.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(document["tally"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
