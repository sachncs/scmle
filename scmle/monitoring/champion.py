"""Champion-challenger comparison with a paired bootstrap."""

from collections.abc import Callable
import dataclasses
import functools
import itertools
import math
import operator
from typing import Any

import numpy
import pandas

import scmle.artifacts.bundle
from scmle.contracts import errors
from scmle.contracts import tasks
import scmle.contracts.metrics
from scmle.evaluation import bootstrap
from scmle.evaluation import metrics

BOOTSTRAPS = 300
MIN_ROWS = 30
# Share of the planned resamples that must be scorable for an interval.
# Policy: below half, the percentiles come from a selected set of resamples
# (those that kept every class), so nothing is promoted on them.
KEPT = 0.5


@dataclasses.dataclass(frozen=True)
class Comparison:
    """Outcome of comparing two models on the same labeled rows.

    Attributes:
        metric: Metric used.
        rows: Rows compared.
        champion: Champion's score.
        challenger: Challenger's score.
        advantage: Challenger minus champion in the metric's favour
            (positive means the challenger is better).
        interval: 95% bootstrap interval of the advantage.
        better: Share of resamples where the challenger won.
        decision: `promote` or `keep`.
        dropped: Resamples that could not be scored (a resample missing a
            class) and are not in the interval.
    """

    metric: str
    rows: int
    champion: float
    challenger: float
    advantage: float
    interval: tuple[float, float]
    better: float
    decision: str
    dropped: int = 0


def rate(
    metric: scmle.contracts.metrics.Metric,
    target: pandas.Series,
    positive: Any,
    bundle: scmle.artifacts.bundle.Bundle,
    view: pandas.DataFrame,
    rows: numpy.ndarray,
) -> float:
    """Scores a model on a subset of the rows.

    Args:
        metric: The metric.
        target: The outcomes of all rows.
        positive: The positive label, if the task has one.
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


class Referee:
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
        frame, target, groups, temporal = self.arrange(data, outcomes)
        champion_view = self.champion.prepare(frame)
        challenger_view = self.challenger.prepare(frame)
        sign = 1.0 if metric.maximize else -1.0
        score = functools.partial(
            rate, metric, target, metrics.positive_label(target, task)
        )
        everything = numpy.arange(len(target))
        champion_score = score(self.champion, champion_view, everything)
        challenger_score = score(self.challenger, challenger_view, everything)
        samples = self.resample(
            score,
            (champion_view, challenger_view),
            (len(target), groups, temporal),
            sign,
        )
        scant = len(samples) < self.bootstraps * KEPT
        low, high = (
            (math.nan, math.nan)
            if scant
            else numpy.percentile(samples, [2.5, 97.5])
        )
        return Comparison(
            metric=metric.name,
            rows=len(target),
            champion=champion_score,
            challenger=challenger_score,
            advantage=sign * (challenger_score - champion_score),
            interval=(float(low), float(high)),
            better=(math.nan if scant else float((samples > 0).mean())),
            decision="promote" if not scant and low > 0 else "keep",
            dropped=self.bootstraps - len(samples),
        )

    def arrange(
        self, data: pandas.DataFrame, outcomes: pandas.Series
    ) -> tuple[pandas.DataFrame, pandas.Series, pandas.Series | None, bool]:
        """Keeps the labelled rows and orders temporal data by time.

        Args:
            data: Rows served.
            outcomes: Observed outcomes aligned to `data`.

        Returns:
            The labelled rows and their outcomes, the group of each row for
            clustered data (else None), and whether the rows are a time
            series (then sorted by the champion's time column).

        Raises:
            ConfigError: With too few labeled rows to compare.
        """
        known = outcomes.reset_index(drop=True).notna().to_numpy()
        if known.sum() < MIN_ROWS:
            raise errors.ConfigError(
                f"need at least {MIN_ROWS} labeled rows to compare models"
            )
        frame = data.reset_index(drop=True)[known].reset_index(drop=True)
        target = outcomes.reset_index(drop=True)[known].reset_index(drop=True)
        group_name = self.champion.schema.group
        time_name = self.champion.schema.time
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
        return frame, target, groups, temporal

    def resample(
        self,
        score: Callable[
            [scmle.artifacts.bundle.Bundle, pandas.DataFrame, numpy.ndarray],
            float,
        ],
        views: tuple[pandas.DataFrame, pandas.DataFrame],
        design: tuple[int, pandas.Series | None, bool],
        sign: float,
    ) -> numpy.ndarray:
        """Bootstraps the challenger's advantage over the champion.

        Args:
            score: Scores a model on its input frame at the given rows.
            views: The champion's and the challenger's input frames.
            design: The number of rows, the group of each row for clustered
                data (else None) and whether to resample contiguous time
                blocks.
            sign: 1 if larger scores are better, else -1.

        Returns:
            The oriented advantage on each resample that could be scored (a
            resample missing a class cannot, and is left out).
        """
        champion_view, challenger_view = views
        size, groups, temporal = design
        rng = numpy.random.default_rng(self.seed)
        advantages = []
        draws = list(
            map(
                operator.call,
                itertools.repeat(
                    lambda: bootstrap.redraw(size, groups, temporal, rng),
                    self.bootstraps,
                ),
            )
        )
        for rows in draws:
            try:
                difference = score(
                    self.challenger, challenger_view, rows
                ) - score(self.champion, champion_view, rows)
            except (
                ValueError
            ):  # handled: counted in `dropped`; too many keep the champion
                continue
            advantages.append(sign * difference)
        return numpy.asarray(advantages)
