"""Descriptive exchangeability check of calibration and released scores."""

import numpy as np
import pandas as pd

from scmle.evaluation import coverage_confirm
from scmle.evaluation import exchangeability


def test_identical_distributions_are_not_flagged_at_the_stated_rate():
    rng = np.random.default_rng(0)
    flags = [
        exchangeability.compare(rng.normal(size=300), rng.normal(size=300))[
            "flag"
        ]
        for trial in range(400)
    ]
    # two tests, Bonferroni at 0.05: the rate is at most 0.05 plus noise
    assert np.mean(flags) <= 0.08


def test_a_shifted_or_wider_released_score_is_flagged():
    rng = np.random.default_rng(1)
    calibration = np.abs(rng.normal(size=400))
    shifted = exchangeability.compare(
        calibration, np.abs(rng.normal(size=300)) + 0.7
    )
    wider = exchangeability.compare(
        calibration, np.abs(rng.normal(size=300)) * 2.5
    )
    assert shifted["flag"] and wider["flag"]
    assert shifted["status"] == "described"


def test_the_result_names_its_tests_and_carries_no_gate():
    rng = np.random.default_rng(2)
    found = exchangeability.compare(rng.normal(size=50), rng.normal(size=60))
    assert set(found) >= {
        "ks_p_value",
        "mann_whitney_p_value",
        "alpha",
        "flag",
        "calibration_rows",
        "released_rows",
        "status",
        "note",
    }
    assert found["calibration_rows"] == 50 and found["released_rows"] == 60
    assert found["status"] == "described"


def test_too_few_rows_are_not_tested():
    found = exchangeability.compare([0.1, 0.2], [0.3, 0.4, 0.5])
    assert found["flag"] is None and found["status"] == "not_tested"


class Model:
    def predict(self, frame):
        return frame["x"].to_numpy(dtype=float)


class Calibrator:
    kind = "regression"

    def __init__(self, scores):
        self.scores = tuple(scores)


class Band:
    quantile = None

    def __init__(self, scores):
        self.conformal = Calibrator(scores)

    def predict_interval(self, features, coverage, method):
        centre = features["x"].to_numpy(dtype=float)
        return pd.DataFrame(
            {
                "prediction": centre,
                "lower": centre - 1.0,
                "upper": centre + 1.0,
            }
        )


def test_the_confirmation_document_describes_exchangeability_without_a_status():
    rng = np.random.default_rng(3)
    x = pd.DataFrame({"x": rng.normal(size=400)})
    y = x["x"] + rng.uniform(-1.0, 1.0, 400)
    calibration = np.abs(rng.uniform(-1.0, 1.0, 400))
    runner = coverage_confirm.build_runner(
        Band(calibration), "regression", "c" * 64
    )
    document = runner.run(x, y).documents[coverage_confirm.DOCUMENT]
    described = document["described_on_the_same_rows"]["exchangeability"]
    assert described["status"] == "described"
    assert described["flag"] is False
    assert document["status"] == "confirmed"
