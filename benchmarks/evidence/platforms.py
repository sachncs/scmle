"""Claim C6 across platforms: merged per-platform reproducibility records.

`benchmarks.evidence.reproducibility --write` measures one platform and keeps
the other platforms' records made by the same code. The workflow
`.github/workflows/reproducibility.yml` runs it on Linux, Windows and macOS
runners, each in its own job, and this module merges the documents those
jobs upload into one reproducibility document and compares every pair of
platforms.

What is measured: for every pair of platforms and every task both measured,
the largest absolute and relative gap between their first-run locked-test
metrics (`reproducibility.differ`), against `cross_platform_tol`
(`acceptance.toml`), together with each platform's own byte-identity and
trace verdict from `reproducibility.summary`.

Verdict: "fail" when any platform fails on its own or any pair of platforms
differs beyond the tolerance; "pass" only with at least `MINIMUM` platforms
(the fewest that can be compared) whose metrics all agree; otherwise "not
measured". Documents for other code or another tolerance are refused.

    python -m benchmarks.evidence.platforms --write [RECORD ...]
        [--into MERGED] [--gate]
    python -m benchmarks.evidence.platforms --check
"""

import argparse
import json
import pathlib
import sys
from typing import Any

from benchmarks.evidence import reproducibility
from scmle.contracts import errors

PATH = pathlib.Path(__file__).with_suffix(".json")
# platforms needed before two can be compared; the rule's minimum sample
MINIMUM = reproducibility.PAIR


def load(paths: list[pathlib.Path]) -> list[dict[str, Any]]:
    """Reads reproducibility documents.

    Args:
        paths: JSON files written by `reproducibility --write`.

    Returns:
        The parsed documents, in the given order.

    Raises:
        InputError: If a file is missing or holds no platform records.
    """
    documents = []
    for path in paths:
        if not path.is_file():
            raise errors.InputError(f"{path} is missing")
        found = json.loads(path.read_text(encoding="utf-8"))
        documents.append(reproducibility.read(found))
    return documents


def merge(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Merges per-platform reproducibility documents into one.

    Every document must be for the same code and tolerance. A platform found
    in several documents must carry the same record in each (a runner keeps
    the committed records of other platforms, so copies are expected).

    Args:
        documents: Documents written by `reproducibility --write`.

    Returns:
        A reproducibility document holding every platform's record, sorted
        by platform text; the other fields come from the first document.

    Raises:
        InputError: If no document is given, the code hashes or tolerances
            differ, or one platform has two different records.
    """
    if not documents:
        raise errors.InputError("no reproducibility document to merge")
    first = documents[0]
    records: dict[str, Any] = {}
    for found in documents:
        if found.get("code") != first.get("code"):
            raise errors.InputError("documents are for different code")
        if found.get("tolerance") != first.get("tolerance"):
            raise errors.InputError("documents use different tolerances")
        for name, record in found["platforms"].items():
            if name in records and records[name] != record:
                raise errors.InputError(f"two different records for {name}")
            records[name] = record
    merged = dict(first)
    merged["platforms"] = dict(sorted(records.items()))
    return merged


def pairs(records: dict[str, Any]) -> list[dict[str, Any]]:
    """Metric gaps between every pair of platforms.

    Args:
        records: Platform text to reproducibility record.

    Returns:
        One entry per pair and task both measured: the two platforms, the
        task and its `absolute` and `relative` gap, sorted by platform and
        task.
    """
    found = []
    names = sorted(records)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            one, two = records[left]["tasks"], records[right]["tasks"]
            for kind in sorted(set(one) & set(two)):
                gap = reproducibility.differ(
                    one[kind]["metrics"], two[kind]["metrics"]
                )
                found.append(
                    {"left": left, "right": right, "task": kind, **gap}
                )
    return found


def measure(paths: list[pathlib.Path]) -> dict[str, Any]:
    """Merges the given records and compares their platforms.

    Args:
        paths: JSON files written by `reproducibility --write`.

    Returns:
        The document written to `PATH`: the code hash, tolerance, minimum
        platforms, the sources, every pair's gaps, the largest relative gap
        and the merged reproducibility document.
    """
    merged = merge(load(paths))
    gaps = pairs(merged["platforms"])
    return {
        "version": 1,
        "claim": "C6",
        "code": merged.get("code"),
        "tolerance": merged.get("tolerance"),
        "minimum": MINIMUM,
        "sources": [path.as_posix() for path in paths],
        "pairs": gaps,
        "spread": max((g["relative"] for g in gaps), default=0.0),
        "merged": merged,
        "not_measured": [
            "platforms whose runner job did not upload a record",
            "Python versions other than each runner's",
        ],
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


def stale(found: dict[str, Any]) -> list[str]:
    """Why the merged records are not for the current code and tolerance.

    Args:
        found: A document written to `PATH`.

    Returns:
        `reproducibility.problems` of the merged document, under this
        module's name; empty when current.
    """
    merged = found.get("merged")
    if not isinstance(merged, dict):
        return ["platforms: the document has no merged records"]
    return [
        message.replace("reproducibility:", "platforms:", 1)
        for message in reproducibility.problems(merged)
    ]


def problems(document: dict[str, Any] | None = None) -> list[str]:
    """Why the document cannot decide the cross-platform row.

    Args:
        document: The document; read from `PATH` when None.

    Returns:
        Messages for a missing file, records for other code or another
        tolerance (`stale`) and fewer than `MINIMUM` platforms; empty when
        the document decides.
    """
    found = read(document)
    if found is None:
        return [f"platforms: {PATH.name} is missing"]
    issues = stale(found)
    if issues:
        return issues
    count = len(found["merged"]["platforms"])
    if count < MINIMUM:
        issues.append(
            f"platforms: {count} platform recorded; none can pass below "
            f"{MINIMUM}"
        )
    return issues


def summary(document: dict[str, Any] | None = None) -> tuple[str, str]:
    """The scorecard's value and verdict for cross-platform reproducibility.

    Args:
        document: The document; read from `PATH` when None.

    Returns:
        The value text and "pass", "fail" or "not measured". A platform
        failing on its own is "fail" even below `MINIMUM` platforms; stale or
        missing records are "not measured".
    """
    found = read(document)
    if found is None or not isinstance(found.get("merged"), dict):
        return "no measurement", "not measured"
    issues = stale(found)
    if issues:
        return "; ".join(issues), "not measured"
    merged = found["merged"]
    value, verdict = reproducibility.summary(merged)
    limit = float(merged["tolerance"]["value"])
    beyond = [g for g in found["pairs"] if g["relative"] > limit]
    for gap in beyond:
        value += (
            f"; {gap['task']} differs between {gap['left']} and "
            f"{gap['right']} by {gap['relative']:.3g}"
        )
    if verdict == "fail" or beyond:
        return value, "fail"
    if len(merged["platforms"]) < MINIMUM:
        return value, "not measured"
    return value, verdict


def main(arguments: list[str] | None = None) -> int:
    """Writes (`--write`) or checks (`--check`) the merged measurement.

    Args:
        arguments: Command-line arguments; None reads `sys.argv`.

    Returns:
        0 after `--write`, or with `--gate` 0 only when the verdict is
        "pass"; for `--check`, 0 when `problems` is empty and 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("records", nargs="*", type=pathlib.Path)
    parser.add_argument("--into", type=pathlib.Path)
    parser.add_argument("--gate", action="store_true")
    options = parser.parse_args(arguments)
    if options.write:
        document = measure(options.records or [reproducibility.PATH])
        PATH.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if options.into is not None:
            merged = json.dumps(document["merged"], indent=2, sort_keys=True)
            options.into.write_text(merged + "\n", encoding="utf-8")
        value, verdict = summary(document)
        print(json.dumps({"value": value, "verdict": verdict}, indent=2))
        return 1 if options.gate and verdict != "pass" else 0
    issues = problems()
    print(json.dumps({"ok": not issues, "problems": issues}, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
