"""The explanation diagnostics run after final evaluation."""

from mml.evaluation import diagnostics
from mml.explainability import dependence
from mml.explainability import importance


def default_suite() -> tuple[diagnostics.Diagnostic, ...]:
    """Returns the standard explanation diagnostics."""
    return (
        importance.PermutationImportance(),
        dependence.PartialDependence(),
    )
