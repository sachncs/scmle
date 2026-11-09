"""Gate a run on a human decision before the locked test set is opened."""

import logging

import numpy
import pandas

import scmle
from scmle.contracts import errors
from scmle.core import policies
from scmle.core import state

LOGGER = logging.getLogger(__name__)


def make_dataset(rows: int = 500) -> pandas.DataFrame:
    """Builds a small binary-classification dataset."""
    rng = numpy.random.default_rng(3)
    frame = pandas.DataFrame(
        {"x": rng.normal(size=rows), "z": rng.normal(size=rows)}
    )
    frame["y"] = (frame.x + 0.5 * frame.z + rng.normal(size=rows) > 0).astype(
        int
    )
    return frame


def reviewer(review: policies.GateReview) -> bool:
    """Approves the plan, rejects the selection (a stand-in for a person)."""
    LOGGER.info("review at %s: %s", review.gate, review.summary)
    return review.gate == state.RunState.PLANNING


def main() -> None:
    """Shows a rejected selection leaving the run BLOCKED, test set locked."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    project = scmle.MLProject(
        make_dataset(),
        "y",
        time_budget=60,
        checkpoints=(policies.CallbackCheckpoint(reviewer),),
    )
    try:
        project.run()
    except errors.ScmleError as error:
        LOGGER.info("run stopped: %s", type(error).__name__)


if __name__ == "__main__":
    main()
