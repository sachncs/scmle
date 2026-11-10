"""The cross-validation interval of a winner picked among many candidates.

The winner's mean cross-validation score is flattered by the choice: it won
because it scored best. Each simulation fits `candidates` ordinary least
squares models on independent pure-noise designs, picks the one with the
lowest mean squared error and asks about its expected squared error, which is
known exactly (`1 + p / (m - p - 1)` for a model trained on `m` rows). Every
candidate has that same expected error, so how far the winner's mean falls
below it is the selection optimism, measured here, not assumed.
"""

import numpy
import support_statistics

from scmle.evaluation import confidence

ROWS, FOLDS, FEATURES = 100, 5, 5
TRAINING = ROWS - ROWS // FOLDS
TRUTH = 1.0 + FEATURES / (TRAINING - FEATURES - 1)
NOMINAL = 0.95


def winner_scores(rng, candidates):
    """Fold scores of `candidates` noise models on one shared target."""
    target = rng.normal(size=ROWS)
    order = rng.permutation(ROWS)
    designs = [
        rng.normal(size=(ROWS, FEATURES)) for ignored in range(candidates)
    ]
    scores = [[] for ignored in designs]
    for held in numpy.array_split(order, FOLDS):
        train = numpy.setdiff1d(order, held)
        for position, design in enumerate(designs):
            beta = numpy.linalg.lstsq(design[train], target[train], rcond=None)[
                0
            ]
            residual = target[held] - design[held] @ beta
            scores[position].append(float(numpy.mean(residual**2)))
    best = int(numpy.argmin([numpy.mean(s) for s in scores]))
    return scores[best], scores, len(held)


def test_the_estimated_optimism_follows_the_true_optimism():
    rng = numpy.random.default_rng(5)
    runs = 600
    rows = []
    for candidates in (1, 3, 30):
        true, estimated = [], []
        for ignored in range(runs):
            chosen, every, held = winner_scores(rng, candidates)
            report = confidence.cv_interval(
                chosen, TRAINING, held, "random", every, False
            )
            true.append(TRUTH - report["mean"])
            estimated.append(report["selection_optimism"])
        rows.append((numpy.mean(true), numpy.mean(estimated)))
        print(f"MEASURED optimism with {candidates} candidates: {rows[-1]}")
    assert rows[0][1] == 0.0
    assert rows[0][1] < rows[1][1] < rows[2][1]
    # the estimate leans a little high (conservative) but is the right size
    ratio = rows[2][1] / rows[2][0]
    assert 0.8 <= ratio <= 1.8, ratio


def test_moving_the_interval_by_the_optimism_helps_and_never_hurts_much():
    rng = numpy.random.default_rng(6)
    runs = 800
    gained = lost = covered = 0
    for ignored in range(runs):
        chosen, every, held = winner_scores(rng, 30)
        plain = confidence.cv_interval(chosen, TRAINING, held, "random")
        moved = confidence.cv_interval(
            chosen, TRAINING, held, "random", every, False
        )
        before = plain["interval"][0] <= TRUTH <= plain["interval"][1]
        after = moved["interval"][0] <= TRUTH <= moved["interval"][1]
        gained += after and not before
        lost += before and not after
        covered += after
    print(f"MEASURED gained {gained}, lost {lost} of {runs}")
    support_statistics.assert_rate_at_least(
        covered / runs, NOMINAL, runs, "moved interval, 30 candidates"
    )
    assert gained > lost
