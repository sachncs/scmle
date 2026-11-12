"""Champion-challenger comparison with a paired bootstrap."""

import dataclasses
import itertools
import operator

import numpy
import pandas

import scmle.artifacts.bundle
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.evaluation import diagnostics
from scmle.evaluation import metrics

BOOTSTRAPS = 300
MIN_ROWS = 30


@dataclasses.dataclass(frozen=True)
class Comparison:
    """Outcome of comparing two models on the same labeled rows.

    Attributes:
        metric: Metric used.
        rows: Rows compared.
        champion_score: Champion's score.
        challenger_score: Challenger's score.
        advantage: Challenger minus champion in the metric's favour
            (positive means the challenger is better).
        interval: 95% bootstrap interval of the advantage.
        probability_better: Share of resamples where the challenger won.
        decision: `promote` or `keep`.
    """

    metric: str
    rows: int
    champion_score: float
    challenger_score: float
    advantage: float
    interval: tuple[float, float]
    probability_better: float
    decision: str


class ChampionChallenger:
    """Decides whether a challenger should replace the champion.

    Attributes:
        champion: Model currently serving.
        challenger: Candidate replacement.
        bootstraps: Number of bootstrap resamples.
        seed: Random seed.
    """

    def __init__(
        self,
        champion: scmle.artifacts.bundle.Bundle,
        challenger: scmle.artifacts.bundle.Bundle,
        bootstraps: int = BOOTSTRAPS,
        seed: int = 0,
    ) -> None:
        """Initializes the comparison.

        Args:
            champion: Model currently serving.
            challenger: Candidate replacement.
            bootstraps: Number of bootstrap resamples.
            seed: Random seed.

        Raises:
            ConfigError: If the models solve different tasks.
        """
        if champion.schema.task != challenger.schema.task:
            raise errors.ConfigError("models were trained for different tasks")
        self.champion = champion
        self.challenger = challenger
        self.bootstraps = bootstraps
        self.seed = seed

    def compare(
        self,
        data: pandas.DataFrame,
        outcomes: pandas.Series,
        metric_name: str | None = None,
    ) -> Comparison:
        """Compares both models on rows whose outcome is known.

        The challenger is promoted only if the lower end of the 95% interval of
        its advantage is above zero, so noise cannot unseat the champion. The
        bootstrap resamples whole groups for clustered data and contiguous time
        blocks for temporal data, so the interval reflects the independent units
        of observation.

        Args:
            data: Rows served (must satisfy both models' schemas).
            outcomes: Observed outcomes aligned to `data`.
            metric_name: Metric; defaults to the champion's.

        Returns:
            The comparison.

        Raises:
            ConfigError: With too few labeled rows to compare.
        """
        task = tasks.Task(self.champion.schema.task)
        metric = metrics.CATALOG.resolve(
            metric_name or self.champion.info.metric, task
        )
        known = outcomes.reset_index(drop=True).notna().to_numpy()
        if known.sum() < MIN_ROWS:
            raise errors.ConfigError(
                f"need at least {MIN_ROWS} labeled rows to compare models"
            )
        frame = data.reset_index(drop=True)[known].reset_index(drop=True)
        target = outcomes.reset_index(drop=True)[known].reset_index(drop=True)
        group_name = self.champion.schema.group_column
        time_name = self.champion.schema.time_column
        groups = (
            frame[group_name]
            if group_name and group_name in frame.columns
            else None
        )
        temporal = bool(
            groups is None and time_name and time_name in frame.columns
        )
        if temporal:
            order = numpy.argsort(
                pandas.to_datetime(
                    frame[time_name], errors="coerce"
                ).to_numpy(),
                kind="stable",
            )
            frame = frame.iloc[order].reset_index(drop=True)
            target = target.iloc[order].reset_index(drop=True)
        champion_view = self.champion.prepare_frame(frame)
        challenger_view = self.challenger.prepare_frame(frame)
        positive = metrics.positive_label(target, task)
        sign = 1.0 if metric.greater_is_better else -1.0

        def score(
            bundle: scmle.artifacts.bundle.Bundle,
            view: pandas.DataFrame,
            rows: numpy.ndarray,
        ) -> float:
            """Scores a model on a subset of the rows.

            Args:
                bundle: The model.
                view: The model's input frame.
                rows: Positions of the rows to score.

            Returns:
                The metric value on those rows.
            """
            return metric.score(
                bundle.pipeline,
                view.iloc[rows].reset_index(drop=True),
                target.iloc[rows].reset_index(drop=True),
                positive,
            )

        everything = numpy.arange(len(target))
        champion_score = score(self.champion, champion_view, everything)
        challenger_score = score(self.challenger, challenger_view, everything)
        rng = numpy.random.default_rng(self.seed)
        advantages = []
        draws = list(
            map(
                operator.call,
                itertools.repeat(
                    lambda: diagnostics.resample_rows(
                        len(target), groups, temporal, rng
                    ),
                    self.bootstraps,
                ),
            )
        )
        for rows in draws:
            try:
                difference = score(
                    self.challenger, challenger_view, rows
                ) - score(self.champion, champion_view, rows)
            except ValueError:  # A resample missing a class.
                continue
            advantages.append(sign * difference)
        samples = numpy.asarray(advantages)
        low, high = numpy.percentile(samples, [2.5, 97.5])
        return Comparison(
            metric=metric.name,
            rows=len(target),
            champion_score=champion_score,
            challenger_score=challenger_score,
            advantage=sign * (challenger_score - champion_score),
            interval=(float(low), float(high)),
            probability_better=float((samples > 0).mean()),
            decision="promote" if low > 0 else "keep",
        )
