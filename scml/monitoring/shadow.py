"""Shadow evaluation: a challenger scores live traffic without being served.

Every request is answered by the champion. The challenger scores the same rows
silently, so before any outcome is known the two can already be compared on
how often they disagree. When outcomes arrive, the stored rows settle the
question with the same bootstrap comparison used for retraining.
"""

import dataclasses

import numpy as np
import pandas as pd

from scml.artifacts import bundle as bundle_module
from scml.contracts import errors
from scml.monitoring import champion


@dataclasses.dataclass(frozen=True)
class Agreement:
    """How the two models behaved on the same traffic.

    Attributes:
        rows: Requests observed.
        disagreement_rate: Share of rows where predictions differ (values
            further than a tenth of the champion's spread for regression).
        mean_probability_gap: Mean absolute difference in the positive-class
            probability (binary classification), else None.
    """

    rows: int
    disagreement_rate: float
    mean_probability_gap: float | None


class ShadowEvaluator:
    """Runs a challenger beside a champion on live rows."""

    def __init__(
        self,
        champion_bundle: bundle_module.ModelBundle,
        challenger_bundle: bundle_module.ModelBundle,
    ) -> None:
        """Binds the pair.

        Raises:
            ConfigError: If the models solve different tasks.
        """
        if champion_bundle.schema.task != challenger_bundle.schema.task:
            raise errors.ConfigError("models were trained for different tasks")
        self.champion = champion_bundle
        self.challenger = challenger_bundle
        self.rows: list[pd.DataFrame] = []
        self.champion_output: list[np.ndarray] = []
        self.challenger_output: list[np.ndarray] = []
        self.gaps: list[np.ndarray] = []

    def observe(self, data: pd.DataFrame) -> np.ndarray:
        """Answers with the champion; the challenger scores silently.

        Returns:
            The champion's predictions (the only thing a caller sees).
        """
        served = np.asarray(self.champion.predict(data))
        shadowed = np.asarray(self.challenger.predict(data))
        self.rows.append(data.reset_index(drop=True))
        self.champion_output.append(served)
        self.challenger_output.append(shadowed)
        if self.champion.schema.task == "binary_classification":
            self.gaps.append(
                np.abs(
                    self.champion.predict_proba(data)[:, 1]
                    - self.challenger.predict_proba(data)[:, 1]
                )
            )
        return served

    def agreement(self) -> Agreement:
        """Disagreement between the two on everything observed so far."""
        if not self.rows:
            raise errors.ConfigError("nothing has been observed yet")
        served = np.concatenate(self.champion_output)
        shadowed = np.concatenate(self.challenger_output)
        if self.champion.schema.task == "regression":
            tolerance = 0.1 * float(np.std(served)) or 1e-9
            differ = np.abs(served.astype(float) - shadowed.astype(float))
            rate = float(np.mean(differ > tolerance))
        else:
            rate = float(np.mean(served != shadowed))
        gap = float(np.concatenate(self.gaps).mean()) if self.gaps else None
        return Agreement(len(served), rate, gap)

    def settle(
        self, outcomes: pd.Series, metric_name: str | None = None
    ) -> champion.Comparison:
        """Compares the models on the observed rows once outcomes are known.

        Args:
            outcomes: True outcomes aligned to the rows in observation order.
            metric_name: Metric; defaults to the champion's.
        """
        data = pd.concat(self.rows, ignore_index=True)
        return champion.ChampionChallenger(
            self.champion, self.challenger
        ).compare(data, outcomes.reset_index(drop=True), metric_name)
