#!/usr/bin/env python3
"""Reject unexplained pytest skips and all xfail outcomes."""

import argparse
from collections import Counter
import json
from pathlib import Path
import xml.etree.ElementTree as ElementTree


def parse_skips(path: Path) -> tuple[tuple[str, str], ...]:
    """Return stable test identifiers and reasons from a JUnit report."""
    root = ElementTree.parse(path).getroot()
    records: list[tuple[str, str]] = []
    for case in root.iter("testcase"):
        skipped = case.find("skipped")
        if skipped is None:
            continue
        classname = case.attrib.get("classname", "")
        name = case.attrib.get("name", "")
        test_id = f"{classname}::{name}" if classname else name
        message = skipped.attrib.get("message", "").strip()
        reason = message.removeprefix("Skipped:").strip()
        records.append((test_id, reason))
    return tuple(records)


def parse_xfails(path: Path) -> tuple[tuple[str, str], ...]:
    """Return test identifiers and reasons for xfail or xpass outcomes."""
    root = ElementTree.parse(path).getroot()
    records: list[tuple[str, str]] = []
    for case in root.iter("testcase"):
        classname = case.attrib.get("classname", "")
        name = case.attrib.get("name", "")
        test_id = f"{classname}::{name}" if classname else name
        for outcome in case:
            outcome_type = outcome.attrib.get("type", "").lower()
            outcome_text = " ".join(
                part.strip() for part in (outcome_type, outcome.text or "")
            ).lower()
            if "xfail" not in outcome_text and "xpass" not in outcome_text:
                continue
            message = outcome.attrib.get("message", "").strip()
            reason = message or (outcome.text or "").strip()
            records.append((test_id, reason))
    return tuple(records)


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse the JUnit report path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Return zero only when skips are explained and no xfail is collected."""
    options = parse_arguments(arguments)
    records = parse_skips(options.report)
    xfails = parse_xfails(options.report)
    unexplained = tuple(test_id for test_id, reason in records if not reason)
    reason_counts = Counter(record[1] for record in records if record[1])
    result = {
        "report": str(options.report),
        "skipped": len(records),
        "unexplained": list(unexplained),
        "xfails": [
            {"test_id": test_id, "reason": reason} for test_id, reason in xfails
        ],
        "reasons": dict(sorted(reason_counts.items())),
    }
    print(json.dumps(result, sort_keys=True))
    return 1 if unexplained or xfails else 0


if __name__ == "__main__":
    raise SystemExit(main())
