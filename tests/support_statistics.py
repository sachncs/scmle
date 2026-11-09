"""Shared checks for the statistical-rigor tests.

Every simulation test compares an observed rate (coverage, rejection rate,
false discovery proportion) with a known target. The tolerance is a count of
exact binomial standard errors, `sqrt(p (1 - p) / n)` at the target `p`, so
it shrinks as the number of simulated datasets grows. The limit is 3.5
standard errors, which a correct procedure exceeds about once in 2,000
checks.
"""

import math

Z_LIMIT = 3.5


def standard_error(rate, runs):
    """Exact binomial standard error of a rate estimated from `runs`."""
    return math.sqrt(rate * (1.0 - rate) / runs)


def show(label, observed, target, runs):
    """Prints the measured rate so a run with `-s` lists every number."""
    print(f"MEASURED {label}: {observed:.4f} vs target {target:.4f} ({runs})")


def assert_rate_near(observed, target, runs, label=""):
    """Asserts a rate is within `Z_LIMIT` standard errors of the target."""
    show(label, observed, target, runs)
    limit = Z_LIMIT * standard_error(target, runs)
    assert abs(observed - target) <= limit, (
        f"{label}: observed {observed:.4f}, target {target:.4f}, "
        f"limit +/- {limit:.4f} ({runs} runs)"
    )


def assert_rate_at_least(observed, target, runs, label=""):
    """Asserts a rate is not below the target by more than `Z_LIMIT` SE."""
    show(label, observed, target, runs)
    limit = Z_LIMIT * standard_error(target, runs)
    assert observed >= target - limit, (
        f"{label}: observed {observed:.4f} below target {target:.4f} "
        f"by more than {limit:.4f} ({runs} runs)"
    )


def assert_rate_at_most(observed, target, runs, label=""):
    """Asserts a rate is not above the target by more than `Z_LIMIT` SE."""
    show(label, observed, target, runs)
    limit = Z_LIMIT * standard_error(target, runs)
    assert observed <= target + limit, (
        f"{label}: observed {observed:.4f} above target {target:.4f} "
        f"by more than {limit:.4f} ({runs} runs)"
    )
