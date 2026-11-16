"""Writes the scorecard: every claim, its target, its evidence and verdict.

Plan section 5: one generated file, `plan/scorecard.md`, shows every claim
(C1 to C14) with the criterion taken from `acceptance.toml`, the value
measured now, where that value comes from, and `pass`, `fail` or `not
measured`. Nothing in it is typed by hand: a row with no evidence yet says
so rather than guessing.

    python -m benchmarks.scorecard            # print the scorecard
    python -m benchmarks.scorecard --write    # write plan/scorecard.md
"""

import argparse
import dataclasses
import json
import pathlib
import sys

from benchmarks import acceptance
from benchmarks import registry

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGET = ROOT / "plan" / "scorecard.md"
STRUCTURE = ROOT / "scripts" / "structure.json"
SCREEN = ROOT / "benchmarks" / "qualification" / "screen_measurements.json"
UNMEASURED = "not measured"
VARIABLES = acceptance.load()


@dataclasses.dataclass(frozen=True)
class Row:
    """One claim on the scorecard.

    Attributes:
        claim: The claim, for example `C2 Leakage caught`.
        criterion: What passing means, with its variables' values.
        value: What is measured now.
        evidence: Where the value comes from.
        verdict: `pass`, `fail` or `not measured`.
    """

    claim: str
    criterion: str
    value: str
    evidence: str
    verdict: str


def value(name: str) -> str:
    """A threshold's value from `acceptance.toml`, as short text."""
    return f"{VARIABLES[name].value:g}"


def gates() -> Row:
    """C1: every gate registered and the code agreeing with the registry."""
    found = registry.load()
    problems = registry.compare(found, registry.scan())
    return Row(
        "C1 No silent wrong outcome",
        "every gate registered with pass, block and override paths; every "
        "injected failure blocks or is a recorded override (share "
        f"{value('gate_fault_block_min')})",
        f"{len(found)} gates registered, {len(problems)} disagreements; "
        "fault-injection suite not written",
        "gates.toml, benchmarks/registry.py",
        "fail" if problems else UNMEASURED,
    )


def leakage() -> Row:
    """C2: the screen's recall and false blocks, so far on planted leaks."""
    alpha = VARIABLES["alpha"].value
    recall = VARIABLES["leak_recall_min"].rule
    blocks = VARIABLES["false_block_max"].rule
    measured = json.loads(SCREEN.read_text(encoding="utf-8"))
    parts = []
    for kind, data in measured.items():
        worst = min(v["detected"] for v in data["recall"].values())
        parts.append(
            f"{kind}: {data['suspect_false_alarms']}/"
            f"{data['null_replicates']} false alarms, worst type "
            f"{worst}/{data['plant_replicates_per_type']}"
        )
    minimum = (
        ""
        if recall is None or blocks is None
        else f" (at least {recall.minimum(alpha)} leaks and "
        f"{blocks.minimum(alpha)} clean datasets)"
    )
    return Row(
        "C2 Leakage caught, clean data not blocked",
        f"recall lower bound >= {value('leak_recall_min')} on real leaks, "
        f"false-block upper bound <= {value('false_block_max')} on clean real "
        f"datasets{minimum}",
        "planted leaks only: " + "; ".join(parts) + "; real leak corpus not "
        "built",
        "benchmarks/qualification/screen_measurements.json",
        UNMEASURED,
    )


def structure() -> Row:
    """C10: size budgets and the naming rule, from the structure report."""
    counts = json.loads(STRUCTURE.read_text(encoding="utf-8"))
    over = sum(
        counts[k] for k in ("modules_over", "classes_over", "functions_over")
    )
    return Row(
        "C10 Can keep changing safely",
        f"no module, class or function over {value('module_lines_max')}/"
        f"{value('class_lines_max')}/{value('function_lines_max')} lines; "
        "layer contract with no exceptions; mutation score >= "
        f"{value('mutation_score_min')}",
        f"{counts['modules_over']} modules, {counts['classes_over']} classes "
        f"and {counts['functions_over']} functions over; "
        f"{counts['compound_functions']} multi-word function and "
        f"{counts['compound_classes']} class names; {counts['bare_raises']} "
        "bare raises",
        "scripts/structure.json",
        "fail" if over else "pass",
    )


def pending(claim: str, name: str, criterion: str) -> Row:
    """A claim whose evidence does not exist yet."""
    return Row(
        claim,
        criterion.format(value=value(name)),
        UNMEASURED,
        f"acceptance.toml: {name}",
        UNMEASURED,
    )


def rows() -> list[Row]:
    """Every claim, in the plan's order."""
    return [
        gates(),
        leakage(),
        pending(
            "C3 Statistics correct and uncertainty honest",
            "interval_coverage_min",
            "interval coverage lower bound >= {value}",
        ),
        pending(
            "C4 At least a competent practitioner's quality",
            "benchmark_median_gap_max",
            "median normalised gap <= {value} per data kind",
        ),
        pending(
            "C5 Data read and joined correctly",
            "reference_rel_tol",
            "awkward files correct or refused; point-in-time features equal a "
            "naive reference (tolerance {value})",
        ),
        pending(
            "C6 Reproducible and traceable",
            "cross_platform_tol",
            "byte-identical bundle per platform; metrics within {value}",
        ),
        pending(
            "C7 Operates without loss or stall",
            "budget_compliance_min",
            "share of runs within budget >= {value}; kill and resume pass",
        ),
        pending(
            "C8 Controls cannot be bypassed",
            "alpha",
            "zero bypasses in a red-team exercise",
        ),
        pending(
            "C9 Secure, private, trustworthy supply chain",
            "openssf_score_min",
            "OpenSSF Scorecard >= {value}",
        ),
        structure(),
        pending(
            "C11 A non-specialist uses it correctly",
            "user_task_minutes_max",
            "five non-specialists finish in <= {value} minutes",
        ),
        pending(
            "C12 Docs tell the truth",
            "alpha",
            "every documented guarantee linked to an enforcing test",
        ),
        pending(
            "C13 Every stage on steroids",
            "speedup_min",
            "practitioner speed-up lower bound >= {value}",
        ),
        pending(
            "C14 Refuses only when it should",
            "refusal_rate_max",
            "refusal-rate upper bound <= {value}",
        ),
    ]


def render(found: list[Row]) -> str:
    """The scorecard as Markdown."""
    lines = [
        "# Scorecard",
        "",
        (
            "Generated by `python -m benchmarks.scorecard --write` from the "
            "evidence named in each row; do not edit by hand."
        ),
        "",
        "| Claim | Criterion | Value now | Evidence | Verdict |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {r.claim} | {r.criterion} | {r.value} | {r.evidence} | "
        f"{r.verdict} |"
        for r in found
    ]
    return "\n".join(lines) + "\n"


def main(arguments: list[str] | None = None) -> int:
    """Prints the scorecard, or writes it with `--write`.

    Args:
        arguments: Command-line arguments; `sys.argv` when None.

    Returns:
        0.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    options = parser.parse_args(arguments)
    text = render(rows())
    if options.write:
        TARGET.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
