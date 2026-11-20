"""Evidence-backed release evaluation for declared task policies.

This package evaluates release evidence; it does not calculate metrics or
diagnostics. A caller supplies the outputs produced by those algorithms and
the evaluator applies the thresholds and evidence references declared by an
``Criteria``. Unknown inputs and incomplete evidence fail closed.

The package keeps every name the single module release.py had, so callers
and tests (which patch `assess`, `faults`, `certify` and `countersign` here)
are unchanged.
"""

from scmle.evaluation.release.assessment import adjudge
from scmle.evaluation.release.assessment import assess
from scmle.evaluation.release.assessment import audit
from scmle.evaluation.release.assessment import citations
from scmle.evaluation.release.assessment import disparities
from scmle.evaluation.release.assessment import explain
from scmle.evaluation.release.assessment import grade
from scmle.evaluation.release.assessment import judge
from scmle.evaluation.release.assessment import referenced
from scmle.evaluation.release.assessment import ruling
from scmle.evaluation.release.assessment import unreadable
from scmle.evaluation.release.finding import exhaustive
from scmle.evaluation.release.finding import Finding
from scmle.evaluation.release.finding import Kind
from scmle.evaluation.release.finding import parse
from scmle.evaluation.release.finding import quantify
from scmle.evaluation.release.finding import Readiness
from scmle.evaluation.release.finding import tallies
from scmle.evaluation.release.finding import Tally
from scmle.evaluation.release.guard import certify
from scmle.evaluation.release.guard import countersign
from scmle.evaluation.release.guard import faults
from scmle.evaluation.release.verdict import Scorecard
from scmle.evaluation.release.verdict import Verdict

__all__ = [
    "Finding",
    "Kind",
    "Readiness",
    "Scorecard",
    "Tally",
    "Verdict",
    "adjudge",
    "assess",
    "audit",
    "certify",
    "citations",
    "countersign",
    "disparities",
    "exhaustive",
    "explain",
    "faults",
    "grade",
    "judge",
    "parse",
    "quantify",
    "referenced",
    "ruling",
    "tallies",
    "unreadable",
]
