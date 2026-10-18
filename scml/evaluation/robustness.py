"""Stress tests: how much does the score fall when the input degrades?"""

import abc
from typing import Any

import numpy as np
import pandas as pd

from scml.evaluation import diagnostics

UNSEEN_TOKEN = "__unseen_category__"  # nosec B105 - a data value, not a password
FRAGILE_DROP = 0.10


class Stressor(abc.ABC):
    """One way of degrading the evaluation features.

    Attributes:
        name: Report key.
    """

    name: str

    @abc.abstractmethod
    def corrupt(
        self,
        features: pd.DataFrame,
        inputs: diagnostics.DiagnosticInput,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Returns a degraded copy of the features.

        Args:
            features: The evaluation features (not modified).
            inputs: The diagnostic inputs, for the feature plan.
            rng: Source of randomness.
        """


class MissingValues(Stressor):
    """Blanks a share of the cells."""

    def __init__(self, rate: float) -> None:
        """Stores the share of cells to blank."""
        self.rate = rate
        self.name = f"missing_{int(rate * 100)}pct"

    def corrupt(
        self,
        features: pd.DataFrame,
        inputs: diagnostics.DiagnosticInput,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Sets random cells of the input columns to missing."""
        damaged = features.copy()
        for column in inputs.feature_plan.input_columns:
            mask = rng.random(len(damaged)) < self.rate
            damaged[column] = (
                damaged[column].astype(object).where(~mask, np.nan)
            )
            if column in inputs.feature_plan.numeric:
                damaged[column] = pd.to_numeric(damaged[column])
        return damaged


class GaussianNoise(Stressor):
    """Adds noise proportional to each numeric column's spread."""

    def __init__(self, scale: float) -> None:
        """Stores the noise level in standard deviations."""
        self.scale = scale
        self.name = f"noise_{self.scale:g}sd"

    def corrupt(
        self,
        features: pd.DataFrame,
        inputs: diagnostics.DiagnosticInput,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Perturbs the numeric columns."""
        damaged = features.copy()
        for column in inputs.feature_plan.numeric:
            spread = float(pd.to_numeric(damaged[column]).std() or 0.0)
            damaged[column] = pd.to_numeric(damaged[column]) + rng.normal(
                0.0, self.scale * spread, len(damaged)
            )
        return damaged


class UnseenCategories(Stressor):
    """Replaces a share of categorical values with a level never trained on."""

    def __init__(self, rate: float) -> None:
        """Stores the share of values to replace."""
        self.rate = rate
        self.name = f"unseen_categories_{int(rate * 100)}pct"

    def corrupt(
        self,
        features: pd.DataFrame,
        inputs: diagnostics.DiagnosticInput,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Overwrites random categorical cells with an unseen level."""
        damaged = features.copy()
        for column in inputs.feature_plan.categorical:
            mask = rng.random(len(damaged)) < self.rate
            damaged[column] = (
                damaged[column].astype(object).where(~mask, UNSEEN_TOKEN)
            )
        return damaged


def default_stressors() -> tuple[Stressor, ...]:
    """Returns the standard stress scenarios."""
    return (
        MissingValues(0.10),
        MissingValues(0.30),
        GaussianNoise(0.10),
        GaussianNoise(0.50),
        UnseenCategories(0.20),
    )


class RobustnessReport(diagnostics.Diagnostic):
    """Score under degraded input, relative to the clean score."""

    name = "robustness"

    def __init__(self, stressors: tuple[Stressor, ...] | None = None) -> None:
        """Chooses the scenarios (the standard ones by default)."""
        self.stressors = stressors or default_stressors()

    def compute(self, inputs: diagnostics.DiagnosticInput) -> dict[str, Any]:
        """Scores the model on each degraded copy of the evaluation data.

        Args:
            inputs: The fitted candidate and evaluation data.

        Returns:
            The clean score and, per scenario, the score and the loss in the
            metric's own units (positive means worse), plus the scenarios
            where the relative loss is large.
        """
        metric = inputs.metric
        clean = metric.score(
            inputs.estimator,
            inputs.features,
            inputs.target,
            inputs.positive_label,
        )
        rng = np.random.default_rng(inputs.seed)
        scenarios = {}
        for stressor in self.stressors:
            stressed = stressor.corrupt(inputs.features, inputs, rng)
            score = metric.score(
                inputs.estimator,
                stressed,
                inputs.target,
                inputs.positive_label,
            )
            loss = (
                (clean - score) if metric.greater_is_better else (score - clean)
            )
            scenarios[stressor.name] = {
                "score": float(score),
                "loss": float(loss),
                "relative_loss": float(loss / abs(clean)) if clean else 0.0,
            }
        fragile = sorted(
            name
            for name, record in scenarios.items()
            if record["relative_loss"] > FRAGILE_DROP
        )
        return {
            "metric": metric.name,
            "clean_score": float(clean),
            "scenarios": scenarios,
            "fragile_under": fragile,
        }
