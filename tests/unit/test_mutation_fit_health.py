"""The fit-health verdict at each of its boundaries, and its notes."""

import math
import types

from scmle.evaluation import fit_health

EDGE = fit_health.UNDERFIT_RELATIVE_GAIN * fit_health.TINY


def test_a_validation_gain_of_exactly_tiny_has_infinite_noise() -> None:
    found = fit_health.assess([1.0, 1.0], [fit_health.TINY] * 2, 0.0)
    assert found.noise == math.inf


def test_a_validation_gain_on_the_underfit_edge_is_underfit() -> None:
    found = fit_health.assess([EDGE, EDGE], [EDGE, EDGE], 0.0)
    assert found.verdict == "underfit"


def test_the_underfit_edge_scales_with_the_baseline_not_against_it() -> None:
    found = fit_health.assess([0.0, 0.0], [1.0, 1.0], 0.0)
    assert found.verdict == "healthy"


def test_keeping_exactly_three_quarters_of_the_gain_is_healthy() -> None:
    found = fit_health.assess([4.0, 4.0], [3.0, 3.0], 0.0)
    assert found.retained == 0.75
    assert found.verdict == "healthy"


def test_the_overfit_note_names_the_share_kept() -> None:
    found = fit_health.assess([4.0, 4.0], [2.0, 2.0], 0.0)
    assert found.verdict == "overfit"
    assert found.note == (
        "only 50% of the training gain survives on validation; try stronger "
        "regularisation, fewer features, early stopping or more data"
    )


def test_the_unstable_note_word_for_word() -> None:
    found = fit_health.assess([1.0, 1.0], [0.5, 1.5], 0.0)
    assert found.verdict == "unstable"
    assert found.note == (
        "the fold-to-fold spread is large relative to the gain, so the "
        "ranking against other candidates is unreliable"
    )


def test_sanity_keeps_the_scale_of_a_larger_is_better_metric() -> None:
    baseline = types.SimpleNamespace(
        model_name="dummy", strategy="default", fold_scores=[0.0]
    )
    trial = types.SimpleNamespace(train_scores=[2.0], fold_scores=[2.0])
    found = fit_health.sanity(trial, [baseline], greater_is_better=True)
    assert found.train_gain == 2.0
    assert found.validation_gain == 2.0
