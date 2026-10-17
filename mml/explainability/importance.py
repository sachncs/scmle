"""Model-agnostic global feature importance."""

from typing import Any

import numpy as np

from mml.evaluation import diagnostics

IMPORTANCE_REPEATS = 5
IMPORTANCE_ROWS = 1000


class PermutationImportance(diagnostics.Diagnostic):
    """Model-agnostic feature importance on the evaluation data."""

    name = "permutation_importance"

    def compute(self, inputs: diagnostics.DiagnosticInput) -> dict[str, Any]:
        """Measures the metric drop when each feature is shuffled.

        The cost grows with the number of features, so rows are capped, the
        features are visited in a random order and the loop stops when the
        diagnostic's time allowance is spent. The report then covers a random
        subset of the features and says so.
        """
        columns = list(inputs.feature_plan.input_columns)
        rng = np.random.default_rng(inputs.seed)
        rows = rng.permutation(len(inputs.features))[:IMPORTANCE_ROWS]
        features = inputs.features[columns].iloc[rows].reset_index(drop=True)
        target = inputs.target.iloc[rows].reset_index(drop=True)
        scorer = inputs.metric.scorer(inputs.positive_label)
        baseline = scorer(inputs.estimator, features, target)
        measured = []
        for index in rng.permutation(len(columns)):
            if inputs.out_of_time():
                break
            drops = []
            for repeat in range(IMPORTANCE_REPEATS):
                shuffler = np.random.default_rng([inputs.seed, index, repeat])
                shuffled = features.copy()
                shuffled[columns[index]] = shuffler.permutation(
                    features[columns[index]].to_numpy()
                )
                drops.append(
                    baseline - scorer(inputs.estimator, shuffled, target)
                )
            measured.append((columns[index], drops))
        ranked = sorted(measured, key=lambda item: -float(np.mean(item[1])))
        return {
            "metric": inputs.metric.name,
            "truncated": len(measured) < len(columns),
            "features_measured": len(measured),
            "features_total": len(columns),
            "features": [
                {
                    "feature": name,
                    "importance": float(np.mean(drops)),
                    "std": float(np.std(drops)),
                }
                for name, drops in ranked
            ],
        }
