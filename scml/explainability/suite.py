"""The explanation diagnostics run after final evaluation."""

from scml.evaluation import diagnostics
from scml.explainability import dependence
from scml.explainability import importance


def default_suite() -> tuple[diagnostics.Diagnostic, ...]:
    """Returns the standard explanation diagnostics."""
    return (
        importance.PermutationImportance(),
        dependence.PartialDependence(),
    )
