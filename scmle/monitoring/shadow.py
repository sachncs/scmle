"""Shadow evaluation: a challenger scores live traffic without being served.

Every request is answered by the champion. The challenger scores the same rows
silently, so before any outcome is known the two can already be compared on
how often they disagree. When outcomes arrive, the stored rows settle the
question with the same bootstrap comparison used for retraining.
"""

import dataclasses
import threading

import numpy
import pandas

from scmle.artifacts import bundle
from scmle.contracts import errors
from scmle.monitoring import champion


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
    """Runs a challenger beside a champion on live rows.

    Attributes:
        champion: The model that answers requests.
        challenger: The model that scores silently.
        rows: Observed request frames, in order.
        champion_output: The champion's predictions per frame.
        challenger_output: The challenger's predictions per frame.
        gaps: Absolute positive-class probability differences per frame (binary
            classification only).
        lock: Keeps the parallel lists growing together.
    """

    def __init__(
        self,
        champion_bundle: bundle.Bundle,
        challenger_bundle: bundle.Bundle,
    ) -> None:
        """Binds the pair.

        Args:
            champion_bundle: The model that answers requests.
            challenger_bundle: The model that scores silently.

        Raises:
            ConfigError: If the models solve different tasks.
        """
        if champion_bundle.schema.task != challenger_bundle.schema.task:
            raise errors.ConfigError("models were trained for different tasks")
        self.champion = champion_bundle
        self.challenger = challenger_bundle
        self.rows: list[pandas.DataFrame] = []
        self.champion_output: list[numpy.ndarray] = []
        self.challenger_output: list[numpy.ndarray] = []
        self.gaps: list[numpy.ndarray] = []
        self.lock = threading.Lock()

    def observe(self, data: pandas.DataFrame) -> numpy.ndarray:
        """Answers with the champion; the challenger scores silently.

        Args:
            data: The request rows.

        Returns:
            The champion's predictions (the only thing a caller sees).
        """
        served = numpy.asarray(self.champion.predict(data))
        shadowed = numpy.asarray(self.challenger.predict(data))
        gap = (
            numpy.abs(
                self.champion.predict_proba(data)[:, 1]
                - self.challenger.predict_proba(data)[:, 1]
            )
            if self.champion.schema.task == "binary_classification"
            else None
        )
        with self.lock:  # The parallel lists must grow together.
            self.rows.append(data.reset_index(drop=True))
            self.champion_output.append(served)
            self.challenger_output.append(shadowed)
            if gap is not None:
                self.gaps.append(gap)
        return served

    def agreement(self) -> Agreement:
        """Measures disagreement between the two on everything observed so far.

        Returns:
            The number of rows, the share on which the predictions differ, and
            the mean positive-class probability gap (None unless the task is
            binary classification).

        Raises:
            ConfigError: If nothing has been observed yet.
        """
        with self.lock:
            champion_output = list(self.champion_output)
            challenger_output = list(self.challenger_output)
            all_gaps = list(self.gaps)
        if not champion_output:
            raise errors.ConfigError("nothing has been observed yet")
        served = numpy.concatenate(champion_output)
        shadowed = numpy.concatenate(challenger_output)
        if self.champion.schema.task == "regression":
            tolerance = 0.1 * float(numpy.std(served)) or 1e-9
            differ = numpy.abs(served.astype(float) - shadowed.astype(float))
            rate = float(numpy.mean(differ > tolerance))
        else:
            rate = float(numpy.mean(served != shadowed))
        gap = float(numpy.concatenate(all_gaps).mean()) if all_gaps else None
        return Agreement(len(served), rate, gap)

    def settle(
        self, outcomes: pandas.Series, metric_name: str | None = None
    ) -> champion.Comparison:
        """Compares the models on the observed rows once outcomes are known.

        Args:
            outcomes: True outcomes aligned to the rows in observation order.
            metric_name: Metric; defaults to the champion's.

        Returns:
            The champion-challenger comparison over all observed rows. Too few
            labeled rows raise `ConfigError` (from the comparison).
        """
        with self.lock:
            rows = list(self.rows)
        data = pandas.concat(rows, ignore_index=True)
        return champion.ChampionChallenger(
            self.champion, self.challenger
        ).compare(data, outcomes.reset_index(drop=True), metric_name)
