"""Model-agnostic global feature importance."""

from typing import Any

import numpy as np
from sklearn import inspection

from mml.evaluation import diagnostics

IMPORTANCE_REPEATS = 5


class PermutationImportance(diagnostics.Diagnostic):
    """Model-agnostic feature importance on the evaluation data."""

    name = "permutation_importance"

    def compute(self, inputs: diagnostics.DiagnosticInput) -> dict[str, Any]:
        """Measures the metric drop when each feature is shuffled."""
        columns = list(inputs.feature_plan.input_columns)
        scorer = inputs.metric.scorer(inputs.positive_label)
        result = inspection.permutation_importance(
            inputs.estimator,
            inputs.features[columns],
            inputs.target,
            scoring=scorer,
            n_repeats=IMPORTANCE_REPEATS,
            random_state=inputs.seed,
        )
        ranked = np.argsort(-result.importances_mean)
        return {
            "metric": inputs.metric.name,
            "features": [
                {
                    "feature": columns[i],
                    "importance": float(result.importances_mean[i]),
                    "std": float(result.importances_std[i]),
                }
                for i in ranked
            ],
        }
