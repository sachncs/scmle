"""Snapshots of conformal, coverage, scores, IPCW and MinT (mutation pins).

See `tests/unit/golden.py` and `tests/unit/MUTANTS_evaluation.md`.
Regenerate with SCMLE_WRITE_GOLDEN=1 only after a reviewed change.
"""

from pathlib import Path

import numpy
import pytest

from scmle.evaluation import conformal
from scmle.evaluation import coverage
from scmle.evaluation import ipcw
from scmle.evaluation import proper_scores
from scmle.evaluation import reconcile
from tests.unit import golden

FIXTURE = Path(__file__).parents[1] / "fixtures" / "evaluation_golden.json"


def case_conformal():
    rng = numpy.random.default_rng(0)
    predicted = rng.normal(size=300)
    observed = predicted + rng.normal(size=300)
    regression = conformal.ConformalCalibrator.for_regression(
        observed, predicted
    )
    classes = ["a", "b", "c"]
    labels = rng.choice(classes, 300)
    logits = rng.normal(size=(300, 3))
    probabilities = numpy.exp(logits) / numpy.exp(logits).sum(
        axis=1, keepdims=True
    )
    classification = conformal.ConformalCalibrator.for_classification(
        labels, probabilities, classes
    )
    new = numpy.linspace(-1, 1, 5)
    return {
        "finite": [
            conformal.finite_sample_quantile(observed, c)
            for c in (0.5, 0.9, 0.999)
        ],
        "weighted": conformal.weighted_quantile(
            numpy.arange(1.0, 9.0),
            numpy.array([1, 2, 1, 2, 1, 2, 1, 2.0]),
            1.5,
            0.7,
        ),
        "margins": conformal.weighted_margins(
            numpy.arange(1.0, 9.0),
            numpy.array([1, 2, 1, 2, 1, 2, 1, 2.0]),
            numpy.array([0.5, 1.5, 3.0]),
            0.7,
        ),
        "cv_plus": conformal.cv_plus_bounds(
            numpy.arange(12.0).reshape(3, 4),
            numpy.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1]),
            numpy.arange(1.0, 11.0),
            0.8,
        ),
        "one": conformal.one_row_per_entity(
            ["a", "b", "a", "c", "b"], numpy.random.default_rng(1)
        ),
        "margin": [regression.margin(c) for c in (0.5, 0.9)],
        "interval": regression.interval(new, 0.8),
        "weighted_interval": regression.weighted_interval(
            new, numpy.ones(300), numpy.array([1, 2, 1, 2, 1.0]), 0.8
        ),
        "class_margins": classification.class_margins(0.8),
        "thin": classification.thin_classes(0.99),
        "sets": classification.sets(probabilities[:30], 0.8),
        "sets_mondrian": classification.sets(probabilities[:30], 0.8, "class"),
        "record": regression.to_dict() | {"scores": []},
    }


def case_coverage():
    rng = numpy.random.default_rng(2)
    covered = rng.random(400) < 0.85
    widths = rng.random(400) + 1
    labels = rng.choice(["a", "b", "c"], 400)
    return {
        "cp": [coverage.clopper_pearson(c, 100, 0.9) for c in (0, 57, 100)],
        "holm": coverage.holm_rejections([0.001, 0.02, 0.04, 0.5], 0.05),
        "under": [
            coverage.undercoverage_p_value(c, 100, 0.9) for c in (70, 85, 95)
        ],
        "summary": coverage.coverage_summary(covered, widths, 0.9),
        "slices": coverage.slice_coverage(covered, labels, 0.9),
        "classes": coverage.class_coverage(covered, labels, 0.9),
        "tier": [
            coverage.coverage_tier(
                coverage.coverage_summary(covered, widths, 0.85), flag, clean
            )
            for flag in (False, True)
            for clean in (False, True)
        ],
        "report": coverage.coverage_report(
            covered, widths, 0.9, list(labels), list(labels)
        ),
    }


def case_scores():
    rng = numpy.random.default_rng(3)
    truth = rng.normal(size=50)
    median = truth + rng.normal(scale=0.5, size=50)
    low = median - 1
    high = median + 1
    history = numpy.cumsum(rng.normal(size=60))
    levels = [0.1, 0.5, 0.9]
    return {
        "pinball": [
            proper_scores.pinball_loss(truth, median, q) for q in levels
        ],
        "interval": proper_scores.interval_score(truth, low, high, 0.2),
        "wis": proper_scores.weighted_interval_score(
            truth, median, {0.2: (low, high), 0.5: (low - 1, high + 1)}
        ),
        "crps": proper_scores.crps_from_quantiles(
            truth,
            numpy.column_stack([median - 1, median, median + 1.5]),
            levels,
        ),
        "scale": [proper_scores.insample_scale(history, s) for s in (1, 4)],
        "mase": [proper_scores.mase(truth, median, history, s) for s in (1, 4)],
    }


def case_ipcw():
    rng = numpy.random.default_rng(4)
    n = 200
    x = rng.normal(size=(n, 2))
    time = rng.exponential(1.0, n) * numpy.exp(-0.3 * x[:, 0])
    censor = rng.exponential(1.5, n)
    event = (time <= censor).astype(float)
    observed = numpy.minimum(time, censor)
    risk = x[:, 0]
    marginal = ipcw.censoring_weights(observed, event)
    covariate = ipcw.censoring_weights(observed, event, x, floor=0.1)
    return {
        "marginal": marginal.weights[:30],
        "floored": [marginal.floor, marginal.n_floored],
        "covariate": covariate.weights[:30],
        "flags": [covariate.covariate_dependent, marginal.covariate_dependent],
        "cindex": [
            ipcw.truncated_concordance(observed, event, risk, tau, w)
            for tau in (0.5, 1.5)
            for w in (marginal, covariate)
        ],
    }


def case_reconcile():
    rng = numpy.random.default_rng(5)
    summing = numpy.array(
        [[1, 1, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float
    )
    residuals = rng.normal(size=(40, 5)) @ rng.normal(size=(5, 5)) * 0.3
    base = rng.normal(size=(4, 5))
    covariance = reconcile.shrunk_covariance(residuals)
    return {
        "bottom_up": reconcile.bottom_up(base[:, 2:], summing),
        "covariance": covariance,
        "mint": reconcile.mint(base, summing, covariance),
        "mint_eye": reconcile.mint(base, summing, numpy.eye(5)),
        "residual": [
            reconcile.coherence_residual(base, summing),
            reconcile.coherence_residual(
                reconcile.mint(base, summing, covariance), summing
            ),
        ],
        "short": reconcile.shrunk_covariance(residuals[:3]),
    }


CASES = {
    "conformal": case_conformal,
    "coverage": case_coverage,
    "scores": case_scores,
    "ipcw": case_ipcw,
    "reconcile": case_reconcile,
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_evaluation_helpers_reproduce_their_recorded_outputs(name):
    expected, actual = golden.check(FIXTURE, name, CASES[name])
    assert actual == expected
