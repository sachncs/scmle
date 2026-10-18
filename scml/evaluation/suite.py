"""The complete post-selection diagnostic suite."""

from scml.evaluation import diagnostics
from scml.evaluation import fairness
from scml.evaluation import robustness
from scml.evaluation import shift
from scml.evaluation import slices
from scml.explainability import suite as explanation_suite


def full_suite() -> tuple[diagnostics.Diagnostic, ...]:
    """Returns every diagnostic run after the final evaluation.

    Performance reports, subgroup scores, stress tests, distribution shift
    between development and test rows, and model explanations.
    """
    return (
        *diagnostics.default_suite(),
        slices.SliceReport(),
        fairness.FairnessReport(),
        robustness.RobustnessReport(),
        shift.AdversarialValidation(),
        *explanation_suite.default_suite(),
    )
