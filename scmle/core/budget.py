"""Time budget planning: the post-training reserve and the smallest run.

Before training, the post-training work (final refit, diagnostics,
packaging) is estimated from measured cost models and held back as a
`Reserve`, so trials run in a window that closes that many seconds before
the run's end (`limits.Governor.until`). A run whose minimum (preparation,
one cheapest trial and the reserve) cannot fit is refused before any work
(`Reserve.admit`).
"""

import dataclasses

from scmle.contracts import errors

# Process start of a trial: 0.93 s for the first one (the forkserver boots
# and preloads numpy, pandas and sklearn), 0.07 s for later ones (macOS
# arm64, 2026-11-17, `scratchpad/train/killtime.py`). A trial window
# shorter than this cannot run a trial.
START_SECONDS = 1.0
# Cost models from the overrun diagnosis (`scratchpad/overrun.md`, 7 OpenML
# tasks at a 60 s budget, OMP_NUM_THREADS=1, two runs at a time). n is the
# row count, p the feature columns, C' the classes when more than two, else
# 1. Each constant is the fitted coefficient of its model. Preparation has
# its own estimate (`scmle.data.profiling.profiler.estimate`).
# The first trial of the run on the cheapest model: 1.86 s on 52,035 rows,
# 0.6-1.4 s on 702-2,949 rows; START_SECONDS plus 2e-5 s per row covers
# all seven.
CHEAPEST_RATE = 2e-5
# metric_uncertainty bootstrap: R = min(2000, max(200, 4e6 / t)) resamples
# of the t = 0.2 n test rows; classification takes R (5.7e-4 + 7.6e-6 p) +
# 1.9e-6 R t seconds (within 15 %), regression 0.4 ms per resample.
TEST_SHARE = 0.2
RESAMPLE_WORK = 4e6
RESAMPLES_MIN = 200
RESAMPLES_MAX = 2000
RESAMPLE_SECONDS = 5.7e-4
RESAMPLE_COLUMN = 7.6e-6
RESAMPLE_ROW = 1.9e-6
RESAMPLE_REGRESSION = 4e-4
# Calibration, slices, robustness and adversarial validation together took
# 0.8-1.7 s on the seven tasks; held at the largest rounded up.
OTHER_DIAGNOSTIC_SECONDS = 2.0
# Permutation importance and pairwise interaction stop at their deadline,
# so they get a share of the budget rather than a cost: they took 1.7-6.6 s
# (3-11 % of 60 s) on six of the seven tasks, MIC's 22.9 s is reduced.
EXPLAIN_SHARE = 0.1
# Final refit, decision threshold and serving measurement took 1.0-3.9 s,
# at most 6.5 % of 60 s; the share is about 1.5 times that. Selection
# falls back to a cheaper candidate whose refit fits (stages.SelectStage).
REFIT_SHARE = 0.1
# Packaging (learning curve, conformal fold fits) took 0.3-0.6 s when no
# fold fit ran late (4.7 and 6.3 s when one did); each fit now checks the
# clock first, so the share only has to cover the first fold fits.
PACKAGE_SHARE = 0.05
BINARY_CLASSES = 2


def factor(classes: int) -> int:
    """Returns C', the class factor of the cost models.

    Args:
        classes: Number of target classes (0 for regression).

    Returns:
        The class count when above two, else 1.
    """
    return classes if classes > BINARY_CLASSES else 1


def preparation(rows: int, columns: int, classes: int, budget: float) -> float:
    """Estimates the seconds ingestion and profiling take.

    Args:
        rows: Training rows.
        columns: Feature columns.
        classes: Target classes (0 for regression).
        budget: The time budget, which caps the capped profiling steps.

    Returns:
        The profiler's upper-bound estimate.
    """
    from scmle.data.profiling import profiler  # noqa: PLC0415 -- cycle

    return profiler.estimate(rows, columns, classes, budget)


def cheapest(rows: int) -> float:
    """Estimates the seconds of the run's first trial on the cheapest model.

    Args:
        rows: Training rows.

    Returns:
        The process start plus a per-row cost.
    """
    return START_SECONDS + CHEAPEST_RATE * rows


def bootstrap(tested: int, columns: int, classes: int) -> float:
    """Estimates the seconds of the metric-uncertainty bootstrap.

    Args:
        tested: Test rows.
        columns: Feature columns.
        classes: Target classes (0 for regression).

    Returns:
        The fitted cost of the resamples the diagnostic draws.
    """
    draws = min(
        RESAMPLES_MAX, max(RESAMPLES_MIN, RESAMPLE_WORK / max(tested, 1))
    )
    if not classes:
        return draws * RESAMPLE_REGRESSION
    return (
        draws * (RESAMPLE_SECONDS + RESAMPLE_COLUMN * columns)
        + RESAMPLE_ROW * draws * tested
    )


def reservation(
    rows: int, columns: int, classes: int, budget: float
) -> dict[str, float]:
    """Estimates the post-training work held back from the trial window.

    Args:
        rows: Training rows.
        columns: Feature columns.
        classes: Target classes (0 for regression).
        budget: The time budget in seconds.

    Returns:
        Seconds for `refit` (final refit, threshold and measurement),
        `diagnostics` (bootstrap, the other diagnostics and an explanation
        share) and `package` (learning curve and conformal fold fits).
    """
    tested = round(TEST_SHARE * rows)
    return {
        "refit": REFIT_SHARE * budget,
        "diagnostics": bootstrap(tested, columns, classes)
        + OTHER_DIAGNOSTIC_SECONDS
        + EXPLAIN_SHARE * budget,
        "package": PACKAGE_SHARE * budget,
    }


@dataclasses.dataclass
class Reserve:
    """Seconds of a time budget held back for post-training work.

    Attributes:
        budget: The time budget in seconds.
        parts: Seconds held back by part (`refit`, `diagnostics`,
            `package`); empty until `provision`.
        shape: Rows, feature columns and classes the reserve was estimated
            for, or None before `provision`.
    """

    budget: float
    parts: dict[str, float] = dataclasses.field(default_factory=dict)
    shape: tuple[int, int, int] | None = None

    @property
    def total(self) -> float:
        """Seconds held back from the trial window for post-training work."""
        return sum(self.parts.values())

    def provision(
        self, rows: int, columns: int, classes: int
    ) -> dict[str, float]:
        """Estimates and holds the post-training reserve for a table.

        Args:
            rows: Training rows.
            columns: Feature columns.
            classes: Target classes (0 for regression).

        Returns:
            The reserve by part (see `reservation`).
        """
        self.shape = (rows, columns, classes)
        self.parts = reservation(rows, columns, classes, self.budget)
        return dict(self.parts)

    def minimum(self, prepared: float | None = None) -> float:
        """Estimates the seconds of the smallest complete run.

        Args:
            prepared: Seconds preparation actually took, or None to use the
                estimate.

        Returns:
            Preparation plus one cheapest trial plus the reserve.

        Raises:
            InputError: If `provision` has not run.
        """
        rows, columns, classes = self.measured()
        if prepared is None:
            prepared = preparation(rows, columns, classes, self.budget)
        return prepared + cheapest(rows) + self.total

    def smallest(self, prepared: float | None = None) -> int:
        """Estimates the smallest budget the minimum run fits in.

        Parts of preparation and of the reserve are shares of the budget, so
        the budget is searched: the smallest whole number of seconds whose
        minimum run fits in it.

        Args:
            prepared: Seconds preparation actually took, or None to use the
                estimate for each budget tried.

        Returns:
            The budget in whole seconds.
        """
        rows, columns, classes = self.measured()

        def fits(budget: int) -> bool:
            done = (
                preparation(rows, columns, classes, budget)
                if prepared is None
                else prepared
            )
            reserve = sum(reservation(rows, columns, classes, budget).values())
            return done + cheapest(rows) + reserve <= budget

        high = 1
        while not fits(high):
            high *= 2
        low = high // 2
        while high - low > 1:
            middle = (low + high) // 2
            if fits(middle):
                high = middle
            else:
                low = middle
        return high

    def measured(self) -> tuple[int, int, int]:
        """Returns the shape the reserve was estimated for.

        Returns:
            Rows, feature columns and classes.

        Raises:
            InputError: If `provision` has not run.
        """
        if self.shape is None:
            raise errors.InputError("provision the reserve before using it")
        return self.shape

    def admit(self, prepared: float | None = None) -> None:
        """Refuses a run whose minimum cannot fit in the budget.

        Args:
            prepared: Seconds preparation actually took, or None to use the
                estimate.

        Raises:
            BlockedError: If preparation, one cheapest trial and the reserve
                exceed the budget; the message and details name the
                estimate and the smallest budget that would fit.
        """
        estimate = self.minimum(prepared)
        if estimate <= self.budget:
            return
        smallest = self.smallest(prepared)
        raise errors.BlockedError(
            f"the {self.budget} s time budget cannot fit the minimum run "
            f"(preparation, one cheapest trial and {self.total:.1f} s "
            f"reserved for the final refit, diagnostics and packaging), "
            f"estimated at {estimate:.1f} s; the smallest budget that fits "
            f"is about {smallest} s",
            {
                "estimate_seconds": round(estimate, 2),
                "smallest_budget_seconds": smallest,
                "reserve_seconds": dict(self.parts),
            },
        )
