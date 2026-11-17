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
import types

from benchmarks import acceptance
from benchmarks.evidence import budget
from benchmarks.evidence import bypass
from benchmarks.evidence import coverage
from benchmarks.evidence import faults
from benchmarks.evidence import guarantees
from benchmarks.evidence import leakage
from benchmarks.evidence import mutation
from benchmarks.evidence import platforms
from benchmarks.evidence import quality
from benchmarks.evidence import reading
from benchmarks.evidence import refusal

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGET = ROOT / "plan" / "scorecard.md"
STRUCTURE = ROOT / "scripts" / "structure.json"
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


def structure() -> Row:
    """C10: size budgets and the naming rule, from the structure report."""
    counts = json.loads(STRUCTURE.read_text(encoding="utf-8"))
    over = sum(
        counts[k] for k in ("modules_over", "classes_over", "functions_over")
    )
    score, verdict = mutation.summary()
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
        f"bare raises; mutation: {score}",
        "scripts/structure.json, benchmarks/evidence/mutation.json",
        "fail" if over or verdict == "fail" else verdict,
    )


def measured(
    claim: str, name: str, criterion: str, module: types.ModuleType
) -> Row:
    """A claim decided by an evidence module's stored results.

    Every module under `benchmarks.evidence` has the same `summary()`,
    returning the value text and the verdict, so one row serves them all.

    Args:
        claim: The claim, for example `C3 Statistics correct`.
        name: The `acceptance.toml` variable the criterion quotes.
        criterion: What passing means, with `{value}` for the variable.
        module: The evidence module that measured the claim.

    Returns:
        The row, citing the module's stored results.
    """
    text, verdict = module.summary()
    stem = module.__name__.rsplit(".", 1)[-1]
    return Row(
        claim,
        criterion.format(value=value(name)),
        text,
        f"benchmarks/evidence/{stem}.json",
        verdict,
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
        measured(
            "C1 No silent wrong outcome",
            "gate_fault_block_min",
            "every gate registered with pass, block and override paths; every "
            "injected failure blocks or is a recorded override (share {value})",
            faults,
        ),
        measured(
            "C2 Leakage caught, clean data not blocked",
            "leak_recall_min",
            "recall lower bound >= {value} on real leaks; false-block upper "
            f"bound <= {value('false_block_max')} on clean real datasets",
            leakage,
        ),
        measured(
            "C3 Statistics correct and uncertainty honest",
            "interval_coverage_min",
            "interval coverage lower bound >= {value}",
            coverage,
        ),
        measured(
            "C4 At least a competent practitioner's quality",
            "benchmark_median_gap_max",
            "median normalised gap <= {value} per data kind",
            quality,
        ),
        measured(
            "C5 Data read and joined correctly",
            "reference_rel_tol",
            "awkward files correct or refused; point-in-time features equal a "
            "naive reference (tolerance {value})",
            reading,
        ),
        measured(
            "C6 Reproducible and traceable",
            "cross_platform_tol",
            "byte-identical bundle per platform; metrics within {value}",
            platforms,
        ),
        measured(
            "C7 Operates without loss or stall",
            "budget_compliance_min",
            "share of runs within budget >= {value}; kill and resume pass",
            budget,
        ),
        measured(
            "C8 Controls cannot be bypassed",
            "alpha",
            "zero bypasses in a red-team exercise",
            bypass,
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
        measured(
            "C12 Docs tell the truth",
            "alpha",
            "every documented guarantee linked to an enforcing test",
            guarantees,
        ),
        pending(
            "C13 Every stage on steroids",
            "speedup_min",
            "practitioner speed-up lower bound >= {value}",
        ),
        measured(
            "C14 Refuses only when it should",
            "refusal_rate_max",
            "refusal-rate upper bound <= {value}",
            refusal,
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
