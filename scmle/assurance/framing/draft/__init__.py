"""Draft charter, data-side checks and the framing result.

`frame` turns a plain-language request and a source description into a draft
`AnalysisCharter` (unsigned), reads only `X` rows, and records where every
field came from. The deterministic drafter fills slots by rules; an optional
LLM slot-filler may fill the rest, behind the refusal rule in `refusal`. Every
LLM output must round-trip `AnalysisCharter.from_dict` and pass the data-side
checks, or it is discarded and the deterministic result is used. Nothing here
claims a drafted charter is correct.
"""

from scmle.assurance.framing.draft.filling import assemble
from scmle.assurance.framing.draft.filling import linkage
from scmle.assurance.framing.draft.filling import mentioned
from scmle.assurance.framing.draft.filling import normalise
from scmle.assurance.framing.draft.filling import prefill
from scmle.assurance.framing.draft.filling import vet
from scmle.assurance.framing.draft.framer import consult
from scmle.assurance.framing.draft.framer import emit
from scmle.assurance.framing.draft.framer import frame
from scmle.assurance.framing.draft.framer import interview
from scmle.assurance.framing.draft.framer import mismatch
from scmle.assurance.framing.draft.framer import refused
from scmle.assurance.framing.draft.records import BACKTICKED
from scmle.assurance.framing.draft.records import CAUSAL_WORDS
from scmle.assurance.framing.draft.records import COST_RANGE
from scmle.assurance.framing.draft.records import CRITICAL_FIELDS
from scmle.assurance.framing.draft.records import Draft
from scmle.assurance.framing.draft.records import ISSUED
from scmle.assurance.framing.draft.records import MAX_MOMENT_VIOLATIONS
from scmle.assurance.framing.draft.records import MAX_TARGET_MISSING
from scmle.assurance.framing.draft.records import MIN_DISTINCT_VALUES
from scmle.assurance.framing.draft.records import MIN_POPULATION_SHARE
from scmle.assurance.framing.draft.records import MIN_SUBSTRING_CHARS
from scmle.assurance.framing.draft.records import Offer
from scmle.assurance.framing.draft.records import Probe
from scmle.assurance.framing.draft.records import REFUSED
from scmle.assurance.framing.draft.records import Request
from scmle.assurance.framing.draft.records import TARGET_WORDS
from scmle.assurance.framing.draft.records import UNIT_DUPLICATE_TOLERANCE
from scmle.assurance.framing.draft.records import UNSPECIFIED
from scmle.assurance.framing.draft.records import WITH_QUESTIONS

__all__ = [
    "BACKTICKED",
    "CAUSAL_WORDS",
    "COST_RANGE",
    "CRITICAL_FIELDS",
    "ISSUED",
    "MAX_MOMENT_VIOLATIONS",
    "MAX_TARGET_MISSING",
    "MIN_DISTINCT_VALUES",
    "MIN_POPULATION_SHARE",
    "MIN_SUBSTRING_CHARS",
    "REFUSED",
    "TARGET_WORDS",
    "UNIT_DUPLICATE_TOLERANCE",
    "UNSPECIFIED",
    "WITH_QUESTIONS",
    "Draft",
    "Offer",
    "Probe",
    "Request",
    "assemble",
    "consult",
    "emit",
    "frame",
    "interview",
    "linkage",
    "mentioned",
    "mismatch",
    "normalise",
    "prefill",
    "refused",
    "vet",
]
