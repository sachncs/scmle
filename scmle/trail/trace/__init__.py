"""Derives the steps of a trail from the retained records alone.

`derive` reads ``retained/`` (and ``inputs.json``) of a trail folder and
returns the ten steps, each with the records it rests on, the facts read
from them, and every problem found. Building a trail and verifying one both
call it, so a fact is never trusted from the file that states it.

`finding` holds the problem and step kinds, `reader` the record access a
`Tracer` shares, `planning`, `evidence` and `release` its steps, and
`tracer` the `Tracer` and `derive`.
"""

from scmle.trail.trace.evidence import CUT_TOLERANCE
from scmle.trail.trace.evidence import MINIMUM_CASES
from scmle.trail.trace.finding import BROKEN_CHAIN
from scmle.trail.trace.finding import Finding
from scmle.trail.trace.finding import IDENTITY
from scmle.trail.trace.finding import KINDS
from scmle.trail.trace.finding import MISMATCH
from scmle.trail.trace.finding import MISSING_LINK
from scmle.trail.trace.finding import NOT_MET
from scmle.trail.trace.finding import OUTSIDE
from scmle.trail.trace.finding import TAMPERED
from scmle.trail.trace.finding import TITLES
from scmle.trail.trace.finding import UNAPPROVED
from scmle.trail.trace.finding import UNSIGNED
from scmle.trail.trace.tracer import derive
from scmle.trail.trace.tracer import Derived
from scmle.trail.trace.tracer import Tracer

__all__ = [
    "BROKEN_CHAIN",
    "CUT_TOLERANCE",
    "IDENTITY",
    "KINDS",
    "MINIMUM_CASES",
    "MISMATCH",
    "MISSING_LINK",
    "NOT_MET",
    "OUTSIDE",
    "TAMPERED",
    "TITLES",
    "UNAPPROVED",
    "UNSIGNED",
    "Derived",
    "Finding",
    "Tracer",
    "derive",
]
