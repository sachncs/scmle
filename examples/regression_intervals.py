"""Regression with calibrated prediction intervals, checked on fresh data."""

import logging

import numpy
import pandas

import scmle
from scmle.contracts import tasks

LOGGER = logging.getLogger(__name__)


def make_dataset(rows: int, seed: int) -> pandas.DataFrame:
    """Builds a linear problem with unit-variance noise."""
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame(
        {"area": rng.uniform(30, 200, rows), "age": rng.uniform(0, 60, rows)}
    )
    frame["price"] = 2.5 * frame.area - 1.2 * frame.age + rng.normal(0, 8, rows)
    return frame


def main() -> None:
    """Trains, then measures how often a 90% interval holds the truth."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Intervals need spare time budget after the search; skip tuning so a
    # short budget still leaves room (a skipped calibration is logged as a
    # `conformal.skipped` event in the run's events.jsonl).
    result = scmle.MLProject(
        make_dataset(800, 0),
        "price",
        metric="rmse",
        time_budget=120,
        search=tasks.SearchSpec(top_models=0),
        strategies=tasks.StrategySpec(top_models=0),
    ).run()
    LOGGER.info(result.summary())

    fresh = make_dataset(1000, 1)
    intervals = result.bundle.predict_interval(
        fresh.drop(columns="price"), coverage=0.9
    )
    inside = (fresh.price >= intervals["lower"]) & (
        fresh.price <= intervals["upper"]
    )
    LOGGER.info("stated coverage 0.90, observed %.3f", inside.mean())


if __name__ == "__main__":
    main()
