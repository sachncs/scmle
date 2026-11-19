"""Writes the scorecard: every claim, its target, its evidence and verdict.

Plan section 5: one generated file, `plan/scorecard.md`, shows every claim
(C1 to C14) with every clause of its criterion (plan section 3, values
from `acceptance.toml`), the value measured now, where that value comes
from, and `pass`, `fail` or `not measured`. A claim passes only when every
clause passes; a failed clause makes it `fail`, and otherwise an
unmeasured clause makes it `not measured`. Nothing in it is typed by hand:
a clause with no evidence yet says so rather than guessing.

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
from benchmarks.evidence import snippets
from scmle.contracts import errors

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGET = ROOT / "plan" / "scorecard.md"
STRUCTURE = ROOT / "scripts" / "structure.json"
UNMEASURED = "not measured"
VARIABLES = acceptance.load()


@dataclasses.dataclass(frozen=True)
class Clause:
    """One clause of a claim's criterion (plan section 3).

    Attributes:
        criterion: What the clause requires, with its variables' values.
        value: What is measured now, or `not measured`.
        evidence: Where the value comes from.
        verdict: `pass`, `fail` or `not measured`.
    """

    criterion: str
    value: str
    evidence: str
    verdict: str


@dataclasses.dataclass(frozen=True)
class Row:
    """One claim on the scorecard: every clause of its criterion.

    A claim passes only when every clause passes. A failed clause makes it
    `fail`; otherwise an unmeasured clause makes it `not measured`. One
    measured clause never stands for the others.

    Attributes:
        claim: The claim, for example `C2 Leakage caught`.
        clauses: The clauses of its criterion, in the plan's order.
    """

    claim: str
    clauses: tuple[Clause, ...]

    @property
    def criterion(self) -> str:
        """Every clause's criterion, in order."""
        return "; ".join(c.criterion for c in self.clauses)

    @property
    def value(self) -> str:
        """The measured values, then the clauses not measured."""
        found = [c.value for c in self.clauses if c.value != UNMEASURED]
        missing = [c.criterion for c in self.clauses if c.value == UNMEASURED]
        if missing:
            found.append(f"{UNMEASURED}: " + "; ".join(missing))
        return "; ".join(found)

    @property
    def evidence(self) -> str:
        """Where the measured values come from."""
        named = [c.evidence for c in self.clauses if c.value != UNMEASURED]
        return ", ".join(dict.fromkeys(named)) or "none yet"

    @property
    def verdict(self) -> str:
        """`fail` if any clause fails, else `pass` only if all pass."""
        verdicts = {c.verdict for c in self.clauses}
        if "fail" in verdicts:
            return "fail"
        if verdicts == {"pass"}:
            return "pass"
        return UNMEASURED


def value(name: str) -> str:
    """A threshold's value from `acceptance.toml`, as short text."""
    return f"{VARIABLES[name].value:g}"


def measured(criterion: str, module: types.ModuleType) -> Clause:
    """A clause decided by an evidence module's stored results.

    Every module under `benchmarks.evidence` has the same `summary()`,
    returning the value text and the verdict, so one clause serves them
    all.

    Args:
        criterion: What the clause requires.
        module: The evidence module that measured it.

    Returns:
        The clause, citing the module's stored results; `not measured`
        when those results cannot be read.
    """
    stem = module.__name__.rsplit(".", 1)[-1]
    evidence = f"benchmarks/evidence/{stem}.json"
    try:
        text, verdict = module.summary()
    except (errors.InputError, KeyError, TypeError, ValueError) as error:
        text = f"{stem}.json cannot be read ({error})"
        return Clause(criterion, text, evidence, UNMEASURED)
    return Clause(criterion, text, evidence, verdict)


def pending(criterion: str) -> Clause:
    """A clause with no evidence yet."""
    return Clause(criterion, UNMEASURED, "", UNMEASURED)


def structure() -> list[Clause]:
    """C10's clauses: size budgets and mutation measured, the rest not."""
    counts = json.loads(STRUCTURE.read_text(encoding="utf-8"))
    over = sum(
        counts[k] for k in ("modules_over", "classes_over", "functions_over")
    )
    sizes = Clause(
        f"no module, class or function over {value('module_lines_max')}/"
        f"{value('class_lines_max')}/{value('function_lines_max')} lines",
        f"{counts['modules_over']} modules, {counts['classes_over']} classes "
        f"and {counts['functions_over']} functions over; "
        f"{counts['compound_functions']} multi-word function and "
        f"{counts['compound_classes']} class names; {counts['bare_raises']} "
        "bare raises",
        "scripts/structure.json",
        "fail" if over else "pass",
    )
    return [
        pending(
            "a new model family, task type, data kind, metric or reader "
            f"touches <= {value('change_cost_files_max')} files outside its "
            "package"
        ),
        pending("import-layer contract with zero exceptions"),
        sizes,
        measured(
            f"mutation score >= {value('mutation_score_min')} per gated module",
            mutation,
        ),
        pending("no test that cannot fail"),
        pending(f"pull-request checks <= {value('ci_minutes_max')} minutes"),
    ]


def rows() -> list[Row]:
    """Every claim with every clause of its criterion, in the plan's order."""
    claims: list[tuple[str, list[Clause]]] = [
        (
            "C1 No silent wrong outcome",
            [
                measured(
                    "every gate registered with pass, block and override "
                    "paths; every injected failure blocks or is a recorded "
                    f"override (share {value('gate_fault_block_min')})",
                    faults,
                ),
                pending("an unregistered gate fails CI"),
                pending("every certificate read by the code it justifies"),
            ],
        ),
        (
            "C2 Leakage caught, clean data not blocked",
            [
                measured(
                    f"recall lower bound >= {value('leak_recall_min')} on real "
                    "leaks, entity-level leaks included; false-block upper "
                    f"bound <= {value('false_block_max')} on clean real "
                    "datasets",
                    leakage,
                ),
            ],
        ),
        (
            "C3 Statistics correct and uncertainty honest",
            [
                pending(
                    "every reported statistic within "
                    f"{value('reference_rel_tol')} of a reference"
                ),
                pending(
                    "gate-deciding statistics match a second implementation"
                ),
                measured(
                    "interval coverage lower bound >= "
                    f"{value('interval_coverage_min')}",
                    coverage,
                ),
                pending(
                    f"on >= {value('null_tasks_min')} null tasks, skill-claim "
                    f"upper bound <= {value('null_skill_max')}"
                ),
                pending(
                    "every interval and test nominal by simulation-based "
                    "calibration"
                ),
                pending(
                    "confirmation data, test set and exploration budget "
                    "spent once per analysis"
                ),
            ],
        ),
        (
            "C4 At least a competent practitioner's quality",
            [
                measured(
                    "median normalised gap <= "
                    f"{value('benchmark_median_gap_max')} and bootstrap "
                    f"upper bound <= {value('benchmark_gap_upper_max')} per "
                    "data kind",
                    quality,
                ),
                pending(
                    "practitioner study: not worse by more than "
                    f"{value('practitioner_noninferiority_margin')}"
                ),
                pending("no leaking submission from scmle"),
            ],
        ),
        (
            "C5 Data read and joined correctly",
            [
                measured(
                    "awkward files correct or refused; point-in-time features "
                    "equal a naive reference (tolerance "
                    f"{value('reference_rel_tol')}) on generated and real "
                    "relational data",
                    reading,
                ),
                pending(
                    "no declared entity and no tied timestamp on both sides "
                    "of any split; duplicate rows that cross are reported, "
                    "not counted, unless they belong to one declared entity"
                ),
            ],
        ),
        (
            "C6 Reproducible and traceable",
            [
                measured(
                    "byte-identical bundle per platform; metrics across "
                    f"platforms within {value('cross_platform_tol')}",
                    platforms,
                ),
                pending(
                    "every bundle traces to data hash, configuration, "
                    "commit and environment"
                ),
                pending("rebuild on a clean runner"),
            ],
        ),
        (
            "C7 Operates without loss or stall",
            [
                measured(
                    "share of runs within budget >= "
                    f"{value('budget_compliance_min')}; kill and resume pass",
                    budget,
                ),
                pending(
                    "service objectives under load and a 24-hour soak with "
                    "flat memory"
                ),
                pending("a hung request cannot wedge a replica"),
                pending("zero reset connections on SIGTERM"),
                pending(
                    "kill -9 at every stage boundary, journal append and "
                    "lease operation resumes identically"
                ),
                pending("lease and journal protocols model-checked"),
            ],
        ),
        (
            "C8 Controls cannot be bypassed",
            [
                measured("zero bypasses in a red-team exercise", bypass),
                pending("a regression test per attempt"),
            ],
        ),
        (
            "C9 Secure, private, trustworthy supply chain",
            [
                pending("no unauthenticated deserialisation path"),
                pending("no raw rows, categories or secrets in artifacts"),
                pending("a forget operation"),
                pending(f"OpenSSF Scorecard >= {value('openssf_score_min')}"),
                pending(
                    "SLSA provenance, signed releases, tag protection, no "
                    "known high vulnerability"
                ),
            ],
        ),
        ("C10 Can keep changing safely", structure()),
        (
            "C11 A non-specialist uses it correctly",
            [
                pending(
                    f"{value('non_specialists')} non-specialists finish in <= "
                    f"{value('user_task_minutes_max')} minutes"
                ),
                pending(
                    f">= {value('comprehension_min')} correct on reading "
                    "the result"
                ),
            ],
        ),
        (
            "C12 Docs tell the truth",
            [
                measured(
                    "every guarantee in docs and site linked to an enforcing "
                    "test",
                    guarantees,
                ),
                measured(
                    "every snippet and command runs in a fresh process",
                    snippets,
                ),
                pending("a planted false statement fails CI"),
            ],
        ),
        (
            "C13 Every stage on steroids",
            [
                pending("every stage's speed and memory ratio within target"),
                pending(
                    f"practitioner speed-up lower bound >= "
                    f"{value('speedup_min')}"
                ),
            ],
        ),
        (
            "C14 Refuses only when it should",
            [
                measured(
                    "refusal-rate upper bound <= "
                    f"{value('refusal_rate_max')}; justified share lower bound "
                    f">= {value('refusal_precision_min')}",
                    refusal,
                ),
            ],
        ),
    ]
    return [Row(claim, tuple(clauses)) for claim, clauses in claims]


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
