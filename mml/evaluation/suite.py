"""The complete post-selection diagnostic suite."""

from mml.evaluation import diagnostics
from mml.evaluation import robustness
from mml.evaluation import shift
from mml.evaluation import slices
from mml.explainability import suite as explanation_suite


def full_suite() -> tuple[diagnostics.Diagnostic, ...]:
    """Returns every diagnostic run after the final evaluation.

    Performance reports, subgroup scores, stress tests, distribution shift
    between development and test rows, and model explanations.
    """
    return (
        *diagnostics.default_suite(),
        slices.SliceReport(),
        robustness.RobustnessReport(),
        shift.AdversarialValidation(),
        *explanation_suite.default_suite(),
    )
