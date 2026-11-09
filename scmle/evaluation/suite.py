"""The complete post-selection diagnostic suite."""

from scmle.evaluation import diagnostics
from scmle.evaluation import fairness
from scmle.evaluation import robustness
from scmle.evaluation import shift
from scmle.evaluation import slices
from scmle.explainability import suite


def full_suite() -> tuple[diagnostics.Diagnostic, ...]:
    """Returns every diagnostic run after the final evaluation.

    Performance reports, subgroup scores, stress tests, distribution shift
    between development and test rows, and model explanations.

    Returns:
        The default diagnostics, then slice, fairness, robustness and
        adversarial-validation diagnostics, then the explanation
        diagnostics.
    """
    return (
        *diagnostics.default_suite(),
        slices.Slices(),
        fairness.Fairness(),
        robustness.Robustness(),
        shift.AdversarialValidation(),
        *suite.default_suite(),
    )
