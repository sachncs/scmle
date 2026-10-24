"""Save a signed model bundle, then load and score it in a fresh step.

Bundles are pickle-based, so loading is refused unless the HMAC signature
verifies. This example shows the produce-then-consume path and what a wrong
key does.
"""

import logging
import pathlib
import tempfile

import numpy as np
import pandas as pd

import scmle
from scmle.contracts import errors

LOGGER = logging.getLogger(__name__)

SIGNING_KEY = "example-only-key"  # in real use: $SCMLE_SIGNING_KEY


def make_dataset(rows: int = 600) -> pd.DataFrame:
    """Builds a small binary-classification dataset."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {
            "age": rng.integers(18, 80, rows),
            "income": rng.normal(50, 15, rows),
        }
    )
    score = 0.04 * frame.age - 0.05 * frame.income
    frame["approved"] = (score + rng.normal(size=rows) > 0).astype(int)
    return frame


def main() -> None:
    """Trains, signs, reloads and scores; then shows a rejected key."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = make_dataset()
    result = scmle.MLProject(
        data, "approved", metric="roc_auc", time_budget=60
    ).run()
    with tempfile.TemporaryDirectory() as directory:
        bundle_directory = pathlib.Path(directory) / "approval"
        result.save(bundle_directory, signing_key=SIGNING_KEY)

        features = data.drop(columns="approved").head(5)
        LOGGER.info(
            "verified load: %s",
            scmle.predict(bundle_directory, features, signing_key=SIGNING_KEY),
        )
        try:
            scmle.load_model(bundle_directory, signing_key="wrong-key")
        except errors.ScmleError as error:
            LOGGER.info("wrong key refused: %s", type(error).__name__)


if __name__ == "__main__":
    main()
