"""Watch a trained model: drift on shifted data and delayed-label scoring."""

import logging

import numpy as np
import pandas as pd

import scmle
from scmle.monitoring import monitor

LOGGER = logging.getLogger(__name__)


def make_dataset(rows: int, shift: float = 0.0, seed: int = 2) -> pd.DataFrame:
    """Builds data whose `load` feature can be shifted to simulate drift."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "load": rng.normal(10 + shift, 2, rows),
            "temperature": rng.normal(20, 5, rows),
        }
    )
    frame["fault"] = (
        0.3 * frame.load - 0.1 * frame.temperature + rng.normal(size=rows) > 1
    ).astype(int)
    return frame


def main() -> None:
    """Trains once, then assesses a healthy and a drifted batch."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = scmle.MLProject(
        make_dataset(800), "fault", metric="roc_auc", time_budget=60
    ).run()
    watcher = monitor.ModelMonitor(result.bundle)

    for label, batch in (
        ("same distribution", make_dataset(300, seed=3)),
        ("load shifted by +6", make_dataset(300, shift=6.0, seed=4)),
    ):
        # Outcomes often arrive late; leave the rest missing (NaN).
        outcomes = batch["fault"].where(batch.index < 100)
        report = watcher.assess(batch.drop(columns="fault"), outcomes)
        LOGGER.info(
            "%s -> recommendation: %s", label, report.recommendation.action
        )


if __name__ == "__main__":
    main()
