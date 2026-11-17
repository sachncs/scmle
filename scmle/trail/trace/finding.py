"""The kinds of problem a trail can have, its step kinds and `Finding`."""

import dataclasses

MISSING_LINK = "missing_link"
BROKEN_CHAIN = "broken_chain"
UNSIGNED = "unsigned_step"
UNAPPROVED = "unapproved_step"
OUTSIDE = "outcome_outside_expectation"
TAMPERED = "record_tampered"
MISMATCH = "fact_mismatch"
NOT_MET = "criterion_not_met"
IDENTITY = "identity_mismatch"
KINDS = (
    "question",
    "framing",
    "stakeholder",
    "exploration",
    "experiments",
    "quality",
    "selection",
    "approval",
    "deployment",
    "outcomes",
)
TITLES = {
    "question": "The business question and the signed charter",
    "framing": "Problem framing and the decision-cost model",
    "stakeholder": "KPIs, acceptance criteria and the approved expectation",
    "exploration": "Exploratory report on the exploration rows",
    "experiments": "Experiment comparison",
    "quality": "Domain quality thresholds with reviewer evidence",
    "selection": "Model selection, claims and the bundle identity",
    "approval": "Deployment approval and release state",
    "deployment": "Production decisions in the deployment ledger",
    "outcomes": "Observed outcomes against the approved expectation",
}


@dataclasses.dataclass(frozen=True, slots=True)
class Finding:
    """One problem the trail has.

    Attributes:
        kind: One of the module constants (``missing_link`` and so on).
        step: The step kind it belongs to.
        detail: One sentence.
    """

    kind: str
    step: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        """Builds a JSON-ready record.

        Returns:
            A dict with `kind`, `step` and `detail`.
        """
        return dataclasses.asdict(self)
