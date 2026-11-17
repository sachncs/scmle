"""Contracts of the analysis charter.

The charter fixes, before any data is profiled, what is being asked, how a
claim will be judged and what error the run may spend. Its estimand follows
the attributes of the ICH E9 addendum: population, target variable, intercurrent
events and their handling strategy, and a summary measure; a prediction task
adds the prediction moment. The class is named `AnalysisEstimand` because
`Estimand` already exists for parameter estimation.

The charter hash is the SHA-256 of its canonical JSON, so it is identical
across runs and platforms and any one-byte edit changes it.
"""

from scmle.assurance.charter.contracts.parts import AnalysisEstimand
from scmle.assurance.charter.contracts.parts import Choice
from scmle.assurance.charter.contracts.parts import Edict
from scmle.assurance.charter.contracts.parts import ESTIMAND_KINDS
from scmle.assurance.charter.contracts.parts import Event
from scmle.assurance.charter.contracts.parts import FORMAT_VERSION
from scmle.assurance.charter.contracts.parts import KNOWN_ENTITY_FUTURE
from scmle.assurance.charter.contracts.parts import MAX_CONFIRMATION_FRACTION
from scmle.assurance.charter.contracts.parts import NEW_ENTITY
from scmle.assurance.charter.contracts.parts import OLDEST_READABLE_FORMAT
from scmle.assurance.charter.contracts.parts import positive
from scmle.assurance.charter.contracts.parts import SetSizes
from scmle.assurance.charter.contracts.parts import texts
from scmle.assurance.charter.contracts.parts import Trigger
from scmle.assurance.charter.contracts.parts import UNITS_OF_PREDICTION
from scmle.assurance.charter.contracts.specification import AnalysisCharter
from scmle.assurance.charter.contracts.specification import deserialize
from scmle.assurance.charter.contracts.specification import parse
from scmle.assurance.charter.contracts.specification import peek
from scmle.assurance.charter.contracts.specification import template

__all__ = [
    "ESTIMAND_KINDS",
    "FORMAT_VERSION",
    "KNOWN_ENTITY_FUTURE",
    "MAX_CONFIRMATION_FRACTION",
    "NEW_ENTITY",
    "OLDEST_READABLE_FORMAT",
    "UNITS_OF_PREDICTION",
    "AnalysisCharter",
    "AnalysisEstimand",
    "Choice",
    "Edict",
    "Event",
    "SetSizes",
    "Trigger",
    "deserialize",
    "parse",
    "peek",
    "positive",
    "template",
    "texts",
]
