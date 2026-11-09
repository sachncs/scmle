"""Time-ordered data: train on the past, test on the future, add lags.

Lag features come from a covariate; lagging the target is refused as leakage.
"""

import logging

import numpy
import pandas

import scmle.api.project
from scmle.contracts import tasks

LOGGER = logging.getLogger(__name__)


def make_dataset(days: int = 500) -> pandas.DataFrame:
    """Builds daily demand with weekly seasonality and a temperature effect."""
    rng = numpy.random.default_rng(0)
    day = pandas.date_range("2025-01-01", periods=days, freq="D")
    steps = numpy.arange(days)
    temperature = 15 + 10 * numpy.sin(2 * numpy.pi * steps / 365)
    temperature = temperature + rng.normal(0, 1, days)
    demand = (
        100
        + 2 * temperature
        + 15 * numpy.sin(2 * numpy.pi * steps / 7)
        + rng.normal(0, 3, days)
    )
    return pandas.DataFrame(
        {"day": day, "store": "A", "temperature": temperature, "demand": demand}
    )


def main() -> None:
    """Runs with temporal validation and a declarative feature graph."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    graph = tasks.FeatureGraph(
        nodes=[
            tasks.FeatureNode(
                name="temperature_prev_week",
                op="lag",
                inputs=["temperature"],
                params={"k": 7},
            ),
            tasks.FeatureNode(
                name="weekday",
                op="datepart",
                inputs=["day"],
                params={"part": "dayofweek"},
            ),
        ]
    )
    project = scmle.api.project.Project(
        make_dataset(),
        "demand",
        metric="rmse",
        time_column="day",
        group_column="store",
        feature_graph=graph,
        time_budget=60,
    )
    LOGGER.info(project.plan().summary())
    result = project.run()
    LOGGER.info(result.summary())


if __name__ == "__main__":
    main()
