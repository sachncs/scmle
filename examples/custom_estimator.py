"""Plug your own scikit-learn estimator into the search.

The class must be importable at module level so trial worker processes can
load it; `register_estimator` is the whole integration. Any estimator with
`fit` / `predict` (and `predict_proba` for probabilities) works, including
your own subclass.
"""

import logging

import numpy
import pandas
from sklearn import naive_bayes

import scmle

LOGGER = logging.getLogger(__name__)


def make_dataset(rows: int = 500) -> pandas.DataFrame:
    """Builds a small binary-classification dataset."""
    rng = numpy.random.default_rng(5)
    frame = pandas.DataFrame(
        {"a": rng.normal(size=rows), "b": rng.normal(size=rows)}
    )
    noise = rng.normal(scale=0.5, size=rows)
    frame["y"] = (frame.a - frame.b + noise > 0).astype(int)
    return frame


def main() -> None:
    """Registers the estimator and finds it in the leaderboard."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    scmle.register_estimator(
        "gaussian_nb",
        classifier=naive_bayes.GaussianNB,
        cost=1,  # budgets under 300 s only search cheap models
        replace=True,
    )
    result = scmle.MLProject(
        make_dataset(), "y", metric="roc_auc", time_budget=60
    ).run()
    board = result.leaderboard()
    LOGGER.info("\n%s", board[board.model_name == "gaussian_nb"])


if __name__ == "__main__":
    main()
