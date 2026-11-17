"""Mechanisms, identifiers, limits and cost rates of the leakage screen.

The budget arithmetic (`allowance`, `share`, `cost`) and the certificate
key of a target (`family`) are computed from these constants alone.
"""

from collections.abc import Callable
from collections.abc import Mapping
import re

import numpy

import scmle.certificates

DETECTOR_ID = "f4-leakage-screen"


SKILL = "skill"


MISSING = "missingness"


PURITY = "record_purity"


ECHO = "echo"


MECHANISM_DEFECTS: Mapping[str, tuple[str, ...]] = {
    SKILL: ("leak_copy", "leak_noisy_copy", "leak_threshold"),
    MISSING: ("leak_missing_for_class",),
    PURITY: ("leak_rare_record",),
    ECHO: (),
}


# A word of free text: a run of letters or digits.
WORD = re.compile(r"[^\W_]+")


# Columns paired for the joint skill test: the smallest split of a target
# over two columns that hides it from each alone is a parity of two binary
# columns (p XOR q), so pairs are formed of two-level columns. Pairs are
# screened only while there are no more of them than candidate columns, so
# they at most double the skill hypotheses (and the fits `measure` counts);
# beyond that the notes say the pairs were not screened.
PAIR_LEVELS = 2


STORE_PATH = scmle.certificates.DIRECTORY / "screen_certificates.json"


KNOTS = 10


MAX_ROWS = 2000


BINARY_CLASSES = 2


MIN_CLASSES = 2


# Each kind of target has its own certificate, measured on a real table of
# that kind (`benchmarks.qualification.screen_certify`). Multiclass targets
# are certified up to the number of classes measured; more classes stay
# descriptive until measured.
DETECTORS: Mapping[str, str] = {
    "binary": DETECTOR_ID,
    "multiclass": f"{DETECTOR_ID}-multiclass",
    "regression": f"{DETECTOR_ID}-regression",
}


MAX_CERTIFIED_CLASSES = 10


# Cross-fitted fits needed by the permutation test: work = permutations x
# skill columns x classes (one-vs-rest when there are more than two). In a
# run (`screen_table`) the screen fits the declared time budget by a
# deterministic cap, never by a clock: a scan is predicted to take
# SCAN_SECONDS plus FIT_SECONDS of its task per row per fit (`cost`), and
# the real scan and the planted probe together may take at most
# SCREEN_SHARE of the budget (`share`): the real scan first, the probe from
# what is left. A fit's cost grows with its rows, so a fixed rate of fits
# per second let a 2000-row screen take 0.63 B instead of 0.15 B.
# Measured 2026-11-17 with OMP_NUM_THREADS=1, two runs at a time, on 11
# evidence tables (every real scan and planted probe of `screen_table`
# timed against `measure`): no scan's fixed work exceeded 0.94 s
# (SCAN_SECONDS = 1.0), and the seconds above that per row per fit were at
# most 6.9e-8 for classification (2000 rows x 13,743 fits in 2.91 s; the
# median per fit is 3.1e-8), so FIT_SECONDS is 8.0e-8 with about 15 percent
# headroom, and at most 1.04e-8 for regression even with no fixed work
# (1.2e-8). One thread is the slowest case, so a larger thread cap only
# finishes sooner. The cap is arithmetic on rows, work and budget, so the
# same table gets the same flags on every machine.
# `allowance` and FITS_PER_SECOND are the older fits-per-second cap, kept
# because the certified `scan` still applies it when `screen` is called
# directly with a budget; `screen_table` no longer passes one.
SCREEN_SHARE = 0.15


FITS_PER_SECOND = 25_000


SCAN_SECONDS = 1.0


FIT_SECONDS: Mapping[str, float] = {
    "classification": 8.0e-8,
    "regression": 1.2e-8,
}


# Most hypotheses one column can raise: skill, record purity, missingness,
# echo, and one pair's skill on average (pairs are capped at the number of
# candidate columns).
HYPOTHESES = 5


BATCH = 200


TIME_BLOCKS = 20


RIDGE = 1e-3


TOLERANCE = 1e-12


WORDS = {
    SKILL: "cross-fitted skill",
    MISSING: "association of being recorded with the target (Cramer's V)",
    PURITY: "share of one target class among the recorded values",
    ECHO: "the column read as the target itself (agreement, or R^2 unfitted)",
}


Statistic = Callable[[numpy.ndarray], numpy.ndarray]


def family(task: str, n_classes: int) -> str | None:
    """Names the kind of target, as the certificates are keyed.

    Args:
        task: `classification` or `regression`.
        n_classes: Number of classes of the target.

    Returns:
        `binary`, `multiclass` (up to `MAX_CERTIFIED_CLASSES` classes) or
        `regression`; None for a kind no certificate covers.
    """
    if task == "regression":
        return "regression"
    if n_classes == BINARY_CLASSES:
        return "binary"
    if BINARY_CLASSES < n_classes <= MAX_CERTIFIED_CLASSES:
        return "multiclass"
    return None


def allowance(time_budget_seconds: float | None) -> int | None:
    """Computes the most cross-fitted fits the screen may use in a budget.

    Deterministic: the budget alone decides, not the machine's speed.

    Args:
        time_budget_seconds: The declared budget, or None for no cap beyond
            `MAX_PERMUTATIONS` (the certification paths).

    Returns:
        The cap, or None for no cap.
    """
    if time_budget_seconds is None:
        return None
    return int(time_budget_seconds * SCREEN_SHARE * FITS_PER_SECOND)


def share(time_budget_seconds: float | None) -> float | None:
    """Computes the seconds the real scan and the probe may take together.

    Args:
        time_budget_seconds: The declared budget, or None for no cap.

    Returns:
        `SCREEN_SHARE` of the budget, or None for no cap.
    """
    if time_budget_seconds is None:
        return None
    return time_budget_seconds * SCREEN_SHARE


def cost(task: str, rows: int, work: int) -> float:
    """Predicts an upper bound on the seconds of one scan.

    Args:
        task: `classification` or `regression`.
        rows: Rows the scan sees (after `subsample`).
        work: Cross-fitted fits: permutations x skill columns x classes
            (one-vs-rest above two classes).

    Returns:
        `SCAN_SECONDS + FIT_SECONDS[task] x rows x work`.
    """
    return SCAN_SECONDS + FIT_SECONDS[task] * rows * work
