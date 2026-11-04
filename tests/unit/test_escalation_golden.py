"""Snapshots of selective prediction on fixed inputs (mutation pins).

See `tests/unit/golden.py` and `tests/unit/MUTANTS_loop.md`. Regenerate with
SCMLE_WRITE_GOLDEN=1 only after a reviewed change of behaviour.
"""

from pathlib import Path

import numpy as np
import pytest

from benchmarks.qualification import streams
from scmle.governance import escalation
from tests.unit import golden

FIXTURE = Path(__file__).parents[1] / "fixtures" / "escalation_golden.json"


def case_threshold():
    data = streams.draw_calibration(1500, 0.05, seed=3)
    weights = np.linspace(0.5, 2.0, 1500)
    small = streams.draw_calibration(40, 0.05, seed=4)
    return {
        "plain": escalation.select_threshold(
            data["score"], data["loss"], 0.05, 0.1
        ),
        "weighted": escalation.select_threshold(
            data["score"], data["loss"], 0.05, 0.1, weights
        ),
        "grid": escalation.select_threshold(
            data["score"],
            data["loss"],
            0.1,
            0.05,
            grid=[0.9, 0.7, 0.5, 0.3],
            minimum=5,
        ),
        "small": escalation.select_threshold(
            small["score"], small["loss"], 0.05, 0.1
        ),
        "p": [escalation.binomial_p_value(e, 50, 0.1) for e in (0, 3, 10)],
        "curve": escalation.risk_coverage(
            data["score"], data["loss"], points=11
        ),
        "curve_w": escalation.risk_coverage(
            data["score"], data["loss"], weights, 11
        ),
    }


def case_scores_and_routes():
    rng = np.random.default_rng(1)
    score = rng.random(40)
    return {
        "score": escalation.set_size_score(
            [1, 2, 3, 0, 4, 1], [False, False, True, False, False, True]
        ),
        "route": [
            escalation.route(score, 0.5, 0.2),
            escalation.route(score, 0.5, 0.9),
            escalation.route(score, 0.5, 0.2, True),
            escalation.route(score, 0.5, 0.5, False, score < 0.8),
        ],
        "chow": [
            escalation.chow_check(
                1 - score * 0.5, score, 0.5, review, 10.0, tol
            )
            for review, tol in ((1.0, 0.1), (4.0, 0.5))
        ],
    }


CASES = {
    "threshold": case_threshold,
    "scores_and_routes": case_scores_and_routes,
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_selective_prediction_reproduces_its_recorded_outputs(name):
    expected, actual = golden.check(FIXTURE, name, CASES[name])
    assert actual == expected
