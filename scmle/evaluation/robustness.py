"""Stress tests: how much does the score fall when the input degrades?"""

import abc
from typing import Any

import numpy
import pandas

from scmle.evaluation import diagnostics

UNSEEN_TOKEN = "__unseen_category__"  # nosec B105  # noqa: S105 -- a data value, not a password
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
        features: pandas.DataFrame,
        inputs: diagnostics.Input,
        rng: numpy.random.Generator,
    ) -> pandas.DataFrame:
        """Returns a degraded copy of the features.

        Args:
            features: The evaluation features (not modified).
            inputs: The diagnostic inputs, for the feature plan.
            rng: Source of randomness.

        Returns:
            The degraded copy of `features`.
        """


class Blanking(Stressor):
    """Blanks a share of the cells."""

    def __init__(self, rate: float) -> None:
        """Stores the share of cells to blank.

        Args:
            rate: Probability that a cell of an input column is blanked. The
                stressor is named `missing_<N>pct`, with `N` the rate in
                percent, rounded down.
        """
        self.rate = rate
        self.name = f"missing_{int(rate * 100)}pct"

    def corrupt(
        self,
        features: pandas.DataFrame,
        inputs: diagnostics.Input,
        rng: numpy.random.Generator,
    ) -> pandas.DataFrame:
        """Sets random cells of the input columns to missing.

        Args:
            features: The evaluation features (not modified).
            inputs: The diagnostic inputs, for the feature plan.
            rng: Source of randomness.

        Returns:
            A copy in which each cell of each input column is missing with
            probability `rate`.
        """
        damaged = features.copy()
        for column in inputs.plan.sourced:
            mask = rng.random(len(damaged)) < self.rate
            damaged[column] = (
                damaged[column].astype(object).where(~mask, numpy.nan)
            )
            if column in inputs.plan.numeric:
                damaged[column] = pandas.to_numeric(damaged[column])
        return damaged


class Noise(Stressor):
    """Adds noise proportional to each numeric column's spread."""

    def __init__(self, scale: float) -> None:
        """Stores the noise level in standard deviations.

        Args:
            scale: Standard deviation of the noise, in units of each
                column's own standard deviation.
        """
        self.scale = scale
        self.name = f"noise_{self.scale:g}sd"

    def corrupt(
        self,
        features: pandas.DataFrame,
        inputs: diagnostics.Input,
        rng: numpy.random.Generator,
    ) -> pandas.DataFrame:
        """Perturbs the numeric columns.

        Args:
            features: The evaluation features (not modified).
            inputs: The diagnostic inputs, for the feature plan.
            rng: Source of randomness.

        Returns:
            A copy in which every numeric input column has normal noise
            added, with standard deviation `scale` times the column's own
            (no change when that is 0). Numeric features the pipeline
            derives itself (calendar parts, say) are not inputs.
        """
        damaged = features.copy()
        sourced = inputs.plan.sourced
        for column in (c for c in inputs.plan.numeric if c in sourced):
            spread = float(pandas.to_numeric(damaged[column]).std())
            if not numpy.isfinite(spread):
                spread = 0.0
            damaged[column] = pandas.to_numeric(damaged[column]) + rng.normal(
                0.0, self.scale * spread, len(damaged)
            )
        return damaged


class Novelty(Stressor):
    """Replaces a share of categorical values with a level never trained on."""

    def __init__(self, rate: float) -> None:
        """Stores the share of values to replace.

        Args:
            rate: Probability that a categorical value is replaced.
        """
        self.rate = rate
        self.name = f"unseen_categories_{int(rate * 100)}pct"

    def corrupt(
        self,
        features: pandas.DataFrame,
        inputs: diagnostics.Input,
        rng: numpy.random.Generator,
    ) -> pandas.DataFrame:
        """Overwrites random categorical cells with an unseen level.

        Args:
            features: The evaluation features (not modified).
            inputs: The diagnostic inputs, for the feature plan.
            rng: Source of randomness.

        Returns:
            A copy in which each categorical cell is replaced by
            `UNSEEN_TOKEN` with probability `rate`.
        """
        damaged = features.copy()
        for column in inputs.plan.categorical:
            mask = rng.random(len(damaged)) < self.rate
            damaged[column] = (
                damaged[column].astype(object).where(~mask, UNSEEN_TOKEN)
            )
        return damaged


class Robustness(diagnostics.Diagnostic):
    """Score under degraded input, relative to the clean score.

    Cost model (31 runs, see `diagnostics.MARGIN`): 0.003 s plus 7.0
    predictions plus 13 microseconds per row for the five standard
    scenarios; the smallest useful report is the clean score and one
    scenario.
    """

    name = "robustness"
    fixed = 0.003
    passes = 7.0
    rowcost = 1.3e-5
    # The smallest useful report is the clean score and one scenario: a
    # third of the full estimate (two of six scores, rounded up).
    units = 3

    def __init__(self, stressors: tuple[Stressor, ...] | None = None) -> None:
        """Chooses the scenarios (the standard ones by default).

        Args:
            stressors: The scenarios to run; None or an empty tuple selects
                the standard scenarios (10% and 30% missing values, noise of
                0.1 and 0.5 standard deviations, and 20% unseen categories).
        """
        self.stressors = stressors or (
            Blanking(0.10),
            Blanking(0.30),
            Noise(0.10),
            Noise(0.50),
            Novelty(0.20),
        )

    def compute(self, inputs: diagnostics.Input) -> dict[str, Any]:
        """Scores the model on each degraded copy of the evaluation data.

        Args:
            inputs: The fitted candidate and evaluation data.

        Returns:
            A mapping with `metric` (name), `clean_score`, `scenarios` and
            `fragile_under`. `scenarios` maps each stressor name to its
            `score`, `loss` in the metric's own units (positive means worse)
            and `relative_loss` (loss over the absolute clean score; 0 when
            the clean score is 0). `fragile_under` lists, sorted, the
            scenarios whose relative loss exceeds `FRAGILE_DROP`, and
            `skipped_for_time` the scenarios not run because the longest
            score so far would end past the deadline (then `truncated` and
            a `reason` are added). When even the clean score does not fit,
            the report is `skipped`.
        """
        metric = inputs.metric
        if not inputs.affords(diagnostics.MARGIN * (inputs.pace or 0.0)):
            return diagnostics.skipped(
                "the deadline leaves no time for the clean score"
            )
        began = diagnostics.clock()
        clean = metric.score(
            inputs.estimator,
            inputs.features,
            inputs.target,
            inputs.positive,
        )
        longest = diagnostics.clock() - began
        rng = numpy.random.default_rng(inputs.seed)
        scenarios = {}
        late = []
        for stressor in self.stressors:
            if not inputs.affords(diagnostics.MARGIN * longest):
                late.append(stressor.name)
                continue
            began = diagnostics.clock()
            stressed = stressor.corrupt(inputs.features, inputs, rng)
            score = metric.score(
                inputs.estimator,
                stressed,
                inputs.target,
                inputs.positive,
            )
            loss = (clean - score) if metric.maximize else (score - clean)
            scenarios[stressor.name] = {
                "score": float(score),
                "loss": float(loss),
                "relative_loss": float(loss / abs(clean)) if clean else 0.0,
            }
            longest = max(longest, diagnostics.clock() - began)
        fragile = sorted(
            name
            for name, record in scenarios.items()
            if record["relative_loss"] > FRAGILE_DROP
        )
        report = {
            "metric": metric.name,
            "clean_score": float(clean),
            "scenarios": scenarios,
            "fragile_under": fragile,
            "skipped_for_time": late,
        }
        if late:
            report["truncated"] = True
            report["reason"] = (
                f"{len(late)} scenarios not run: each takes up to "
                f"{longest:.3f} s and the deadline came first"
            )
        return report
