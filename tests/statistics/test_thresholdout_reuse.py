"""Thresholdout keeps the reuse error rate within its bound under adaptivity.

The experiment of Dwork et al. (2015): labels independent of every feature,
so every query's population mean is exactly 0.5. The analyst asks the
agreement of each feature with the label, keeps the features whose own
training correlation and holdout answer agree in sign, and asks the
accuracy of the majority vote of the strongest ones, for growing sizes:
each query depends on the earlier answers. An answer is an error when it
misses 0.5 by more than the derived tolerance; the error rate's
Clopper-Pearson upper bound must stay below the confidence, on more than
the 59 answers the plan's rule needs to pass at all.
"""

import numpy
import pandas
from scipy import stats

from scmle.assurance.ledger.confirm import thresholdout

ROWS = 20000
FEATURES = 100
SIZES = (5, 10, 20, 40, 80)
QUERIES = FEATURES + len(SIZES)
BUDGET = 20
SEEDS = 12


def coins(rng):
    features = pandas.DataFrame(
        rng.choice(numpy.array([-1, 1], dtype=numpy.int8), (ROWS, FEATURES)),
        columns=[f"f{i}" for i in range(FEATURES)],
    )
    return features, pandas.Series(rng.choice([-1, 1], size=ROWS))


def agreement(column):
    return lambda f, y: (1 + f[column].to_numpy() * y.to_numpy()) / 2


def vote(columns, signs):
    def accuracy(f, y):
        guess = numpy.sign(f[columns].to_numpy() @ signs + 0.5)
        return (guess == y.to_numpy()).astype(float)

    return accuracy


def analyst(guard, train):
    features, target = train
    answers = []
    fit = {}
    for column in features.columns:
        fit[column] = agreement(column)(features, target).mean() - 0.5
        if guard.left == 0:
            return answers
        answers.append(guard.query(agreement(column)).value - 0.5)
    kept = [
        c for c, a in zip(features.columns, answers, strict=True)
        if numpy.sign(a) == numpy.sign(fit[c])
    ]  # fmt: skip
    ranked = sorted(kept, key=lambda c: -abs(fit[c]))
    for size in SIZES:
        if guard.left == 0 or len(ranked) < size:
            break
        chosen = ranked[:size]
        signs = numpy.sign([fit[c] for c in chosen])
        answers.append(guard.query(vote(chosen, signs)).value - 0.5)
    return answers


def test_the_reuse_error_rate_stays_within_the_bound():
    terms = thresholdout.Terms(ROWS, queries=QUERIES, budget=BUDGET)
    errors = 0
    answered = 0
    spent = 0
    for seed in range(SEEDS):
        rng = numpy.random.default_rng(seed)
        train, holdout = coins(rng), coins(rng)
        guard = thresholdout.Thresholdout(train, holdout, terms, seed=seed)
        gaps = numpy.abs(analyst(guard, train))
        errors += int((gaps > terms.tolerance).sum())
        answered += len(gaps)
        spent += guard.spent
    upper = stats.binomtest(errors, answered).proportion_ci(
        1 - 2 * terms.confidence
    )
    assert answered > 59 * SEEDS
    assert errors == 0
    assert spent > SEEDS
    assert upper.high <= terms.confidence
