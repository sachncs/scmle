"""Bounded pairwise interaction explanations."""

from typing import Any

import numpy as np
import pandas as pd

from scmle.evaluation import diagnostics

MAX_INTERACTION_FEATURES = 10
MAX_INTERACTION_ROWS = 200


class PairwiseInteraction(diagnostics.Diagnostic):
    """Measure model-output interaction between pairs of numeric features.

    The score is the mean absolute second-order difference after replacing
    each feature with its observed value or a training-independent median from
    the evaluation rows. It describes model behaviour in feature space; it is
    not a causal interaction effect.
    """

    name = "pairwise_interaction"
    applies_to = diagnostics.CLASSIFICATION | diagnostics.REGRESSION_ONLY

    def compute(self, inputs: diagnostics.DiagnosticInput) -> dict[str, Any]:
        """Returns deterministic interaction strengths for bounded pairs.

        Only the first `MAX_INTERACTION_FEATURES` numeric features and the
        first `MAX_INTERACTION_ROWS` rows are used. Pairs are visited in
        declaration order until the diagnostic's time allowance is spent.

        Args:
            inputs: The fitted candidate, evaluation data, feature plan and
                time allowance.

        Returns:
            The compared pairs (`features`) ordered by decreasing
            `mean_absolute_interaction`, then by feature names; the numeric
            features considered; the number of rows used; whether the pair
            loop was cut short (`truncated`); and the stated limitations.
            When fewer than two numeric features or no rows are available,
            `features` is empty and the limitations say so.
        """
        columns = list(inputs.feature_plan.input_columns)
        numeric = [
            column
            for column in inputs.feature_plan.numeric
            if column in columns
        ][:MAX_INTERACTION_FEATURES]
        sample = inputs.features[columns].head(MAX_INTERACTION_ROWS).copy()
        if len(numeric) < 2 or sample.empty:
            return {
                "features": [],
                "features_considered": numeric,
                "rows": len(sample),
                "truncated": False,
                "limitations": [
                    "At least two numeric features are required.",
                    "The result is feature-space and associational.",
                ],
            }
        medians = {
            column: float(
                pd.to_numeric(sample[column], errors="coerce").median()
            )
            for column in numeric
        }
        terms: list[dict[str, object]] = []
        truncated = False
        for left_index, left in enumerate(numeric[:-1]):
            for right in numeric[left_index + 1 :]:
                if inputs.out_of_time():
                    truncated = True
                    break
                both = sample.copy()
                left_only = sample.copy()
                right_only = sample.copy()
                neither = sample.copy()
                left_only.loc[:, right] = medians[right]
                right_only.loc[:, left] = medians[left]
                neither.loc[:, left] = medians[left]
                neither.loc[:, right] = medians[right]
                interaction = (
                    self.score(inputs, both)
                    - self.score(inputs, left_only)
                    - self.score(inputs, right_only)
                    + self.score(inputs, neither)
                )
                terms.append(
                    {
                        "features": [left, right],
                        "mean_absolute_interaction": float(
                            np.mean(np.abs(interaction))
                        ),
                    }
                )
            if truncated:
                break

        def sort_key(item: dict[str, object]) -> tuple[float, tuple[str, ...]]:
            """Order interaction terms by strength and feature names.

            Args:
                item: One interaction term with its feature pair and mean
                    absolute interaction.

            Returns:
                The negated strength followed by the feature names, so that
                ascending sorting puts the strongest term first.

            Raises:
                TypeError: If the strength is not a number or the features
                    are not a list.
            """
            score = item["mean_absolute_interaction"]
            features = item["features"]
            if not isinstance(score, (int, float)) or not isinstance(
                features, list
            ):
                raise TypeError("interaction term has an invalid shape")
            return (-float(score), tuple(str(value) for value in features))

        terms.sort(key=sort_key)
        return {
            "features": terms,
            "features_considered": numeric,
            "rows": len(sample),
            "truncated": truncated,
            "limitations": [
                "Only the first ten declared numeric features are compared.",
                "The statistic describes model output, not causal interaction.",
                "Categorical and raw-modality interactions are not measured.",
            ],
        }

    @staticmethod
    def score(
        inputs: diagnostics.DiagnosticInput, frame: pd.DataFrame
    ) -> np.ndarray:
        """Returns one comparable model-output score per row.

        Args:
            inputs: The diagnostic inputs holding the task and estimator.
            frame: Rows to score.

        Returns:
            The regression prediction, the positive-class probability for
            binary tasks, or the largest class probability otherwise.
        """
        if inputs.task in diagnostics.REGRESSION_ONLY:
            return np.asarray(inputs.estimator.predict(frame), dtype=float)
        probabilities = np.asarray(inputs.estimator.predict_proba(frame))
        if inputs.task in diagnostics.BINARY_ONLY:
            return probabilities[:, 1]
        return probabilities.max(axis=1)
