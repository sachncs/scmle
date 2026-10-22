#!/usr/bin/env python3
"""Reject unexplained or unapproved pytest skips and all xfail outcomes."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ElementTree

DEFAULT_MANIFEST = Path(__file__).with_name("pytest_skip_manifest.json")


def load_skip_rules(
    path: Path = DEFAULT_MANIFEST,
) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Load the approved skip taxonomy from the checked-in manifest."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    categories = manifest.get("approved_categories")
    rules = manifest.get("rules")
    if not isinstance(categories, list) or not all(
        isinstance(category, str) for category in categories
    ):
        raise ValueError("skip manifest must define approved_categories")
    if not isinstance(rules, list):
        raise ValueError("skip manifest must define rules")

    compiled: list[tuple[str, re.Pattern[str]]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError("skip manifest rules must be objects")
        category = rule.get("category")
        pattern = rule.get("pattern")
        if category not in categories or not isinstance(pattern, str):
            raise ValueError(f"invalid skip manifest rule: {rule!r}")
        compiled.append((category, re.compile(pattern)))
    return tuple(compiled)


def classify_reason(
    reason: str, rules: tuple[tuple[str, re.Pattern[str]], ...]
) -> str | None:
    """Return the approved category for a reason, if one is registered."""
    for category, pattern in rules:
        if pattern.fullmatch(reason):
            return category
    return None


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
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Return zero only when skips are approved and no xfail is collected."""
    options = parse_arguments(arguments)
    records = parse_skips(options.report)
    rules = load_skip_rules(options.manifest)
    xfails = parse_xfails(options.report)
    unexplained = tuple(test_id for test_id, reason in records if not reason)
    classified = tuple(
        (test_id, reason, classify_reason(reason, rules))
        for test_id, reason in records
    )
    unapproved = tuple(
        {"test_id": test_id, "reason": reason}
        for test_id, reason, category in classified
        if reason and category is None
    )
    category_counts = Counter(
        category for _, _, category in classified if category is not None
    )
    reason_counts = Counter(record[1] for record in records if record[1])
    result = {
        "report": str(options.report),
        "manifest": str(options.manifest),
        "skipped": len(records),
        "skips": [
            {"category": category, "reason": reason, "test_id": test_id}
            for test_id, reason, category in classified
        ],
        "unexplained": list(unexplained),
        "unapproved": list(unapproved),
        "categories": dict(sorted(category_counts.items())),
        "xfails": [
            {"test_id": test_id, "reason": reason} for test_id, reason in xfails
        ],
        "reasons": dict(sorted(reason_counts.items())),
    }
    print(json.dumps(result, sort_keys=True))
    return 1 if unexplained or unapproved or xfails else 0


if __name__ == "__main__":
    raise SystemExit(main())
