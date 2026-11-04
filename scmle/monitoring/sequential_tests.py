"""Anytime-valid monitors: betting tests, confidence sequences, e-processes.

The fixed-sample rules in `monitor.py` and `champion.py` are applied again at
every batch, so their false-alarm and false-promotion rates grow with the
number of looks and nothing bounds them. The monitors here are nonnegative
supermartingales under their null, so by Ville's inequality
``P(sup_t K_t >= 1 / alpha) <= alpha`` over all time: checking at every batch
costs nothing in validity.

Construction (betting for bounded means, after Waudby-Smith and Ramdas,
arXiv:2010.09686; the paper's formal coverage statement has not been read by
the author of this module, the validity used here is only the martingale
argument below). For observations ``x_t`` in ``[0, 1]`` and a null "the mean is
at most ``m0``" the capital

    K_t = prod_{i <= t} (1 + lambda_i * (x_i - m0)),  0 <= lambda_i < 1 / m0

has ``E[K_t | past] <= K_{t-1}`` whenever ``lambda_i`` is a function of the
past only, and stays positive. The mirror null "the mean is at least ``m0``"
uses ``1 - lambda_i * (x_i - m0)`` with ``lambda_i < 1 / (1 - m0)``. The bet
size is the predictable plug-in of the paper,
``lambda_t = sqrt(2 log(1 / alpha) / (s2_{t-1} * t * log(1 + t)))`` with a
running variance ``s2``, capped at ``cap`` times the positivity bound.

Unit of observation. Losses are bounded, not necessarily independent inside a
batch (cases of one group share a cause). The monitors therefore take one
number per batch (the batch mean, in ``[0, 1]``) as the observation, which is
valid whenever the batch means are conditionally mean-bounded given the past.
Per-case streams are supported by passing one value per call.

Not covered: observations that are not bounded, or whose conditional mean
drifts under the null (a null that is a composite of moving means).
"""

from collections.abc import Mapping
from fractions import Fraction
import math
from typing import Any

import numpy as np

from scmle.data.ledger import online_fdr

GREATER = "greater"
LESS = "less"
DIRECTIONS = (GREATER, LESS)
DEFAULT_CAP = 0.5
GRID_POINTS = 199


def validate_unit(value: Any, name: str) -> np.ndarray:
    """The values as floats, which must lie in ``[0, 1]``."""
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array < 0) or np.any(array > 1):
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return array


class BettingTest:
    """Anytime-valid test of one or many means by betting.

    With ``direction="greater"`` the null is "mean <= null_mean" and the test
    crosses when the mean is evidenced to be larger; with ``"less"`` the null
    is "mean >= null_mean". `null_mean` may be an array (one test per entry,
    all fed the same observation), which is how the confidence sequence is
    built.
    """

    def __init__(
        self,
        null_mean: Any,
        direction: str,
        alpha: float,
        cap: float = DEFAULT_CAP,
    ) -> None:
        """Creates a test at level `alpha`.

        Args:
            null_mean: Mean under the null, strictly inside ``(0, 1)``.
            direction: `greater` or `less`.
            alpha: Level, in ``(0, 1)``.
            cap: Share of the positivity bound a bet may use, in ``(0, 1)``.

        Raises:
            ValueError: On an invalid argument.
        """
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        if not 0.0 < cap < 1.0:
            raise ValueError("cap must be strictly between 0 and 1")
        mean = np.asarray(null_mean, dtype=float)
        if np.any(mean <= 0.0) or np.any(mean >= 1.0):
            raise ValueError("null_mean must be strictly between 0 and 1")
        self.null_mean = mean
        self.direction = direction
        self.alpha = alpha
        self.cap = cap
        self.sign = 1.0 if direction == GREATER else -1.0
        room = mean if direction == GREATER else 1.0 - mean
        self.bound = cap / room
        self.count = 0
        self.total: Any = np.zeros(())
        self.deviation: Any = np.zeros(())
        self.log_capital = np.zeros_like(mean)
        self.peak = np.zeros_like(mean)

    @property
    def threshold(self) -> float:
        """``log(1 / alpha)``, the log capital at which the test crosses."""
        return math.log(1.0 / self.alpha)

    def bet(self) -> np.ndarray:
        """The predictable bet size for the next observation."""
        step = self.count + 1
        variance = (0.25 + self.deviation) / step
        raw = np.sqrt(
            2.0 * self.threshold / (variance * step * math.log(1.0 + step))
        )
        return np.minimum(raw, self.bound)

    def update(self, observation: Any) -> np.ndarray:
        """Feeds one observation in ``[0, 1]``; returns the log capital."""
        value = validate_unit(observation, "observation")
        size = self.bet()
        factor = 1.0 + self.sign * size * (value - self.null_mean)
        self.log_capital = self.log_capital + np.log(factor)
        self.peak = np.maximum(self.peak, self.log_capital)
        self.count += 1
        self.total = self.total + value
        previous = (0.5 + self.total) / (self.count + 1)
        self.deviation = self.deviation + (value - previous) ** 2
        return self.log_capital

    @property
    def crossed(self) -> Any:
        """Whether the capital reached ``1 / alpha`` now."""
        return self.log_capital >= self.threshold

    @property
    def ever_crossed(self) -> Any:
        """Whether the capital ever reached ``1 / alpha``."""
        return self.peak >= self.threshold

    @property
    def e_value(self) -> np.ndarray:
        """The capital now: an e-value for the null at any stopping time."""
        return np.exp(self.log_capital)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready state (scalar tests only)."""
        if self.null_mean.ndim != 0:
            raise ValueError("only a single test can be serialized")
        return {
            "null_mean": float(self.null_mean),
            "direction": self.direction,
            "alpha": self.alpha,
            "cap": self.cap,
            "count": self.count,
            "total": float(self.total),
            "deviation": float(self.deviation),
            "log_capital": float(self.log_capital),
            "peak": float(self.peak),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "BettingTest":
        """Rebuilds a test from `to_dict`."""
        test = cls(
            record["null_mean"],
            record["direction"],
            record["alpha"],
            record["cap"],
        )
        test.count = int(record["count"])
        test.total = np.asarray(float(record["total"]))
        test.deviation = np.asarray(float(record["deviation"]))
        test.log_capital = np.asarray(float(record["log_capital"]))
        test.peak = np.asarray(float(record["peak"]))
        return test


class ConfidenceSequence:
    """Time-uniform confidence sequence for a bounded mean, on a grid.

    Two one-sided betting tests per grid point, each at ``alpha / 2``; a grid
    point is out when either crosses. Limits are the running intersection, so
    they only tighten, and are rounded outward to the grid (so the interval
    is never narrower than the exact one on the grid).
    """

    def __init__(self, alpha: float, points: int = GRID_POINTS) -> None:
        """Creates a sequence with `points` grid values in ``(0, 1)``."""
        if points < 3:
            raise ValueError("points must be at least 3")
        self.alpha = alpha
        self.grid = (np.arange(points) + 1.0) / (points + 1.0)
        self.upper_tests = BettingTest(self.grid, GREATER, alpha / 2.0)
        self.lower_tests = BettingTest(self.grid, LESS, alpha / 2.0)
        self.alive = np.ones(points, dtype=bool)

    def update(self, observation: Any) -> tuple[float, float]:
        """Feeds one observation; returns the interval ``(low, high)``."""
        self.upper_tests.update(observation)
        self.lower_tests.update(observation)
        self.alive &= ~(self.upper_tests.crossed | self.lower_tests.crossed)
        return self.interval

    @property
    def interval(self) -> tuple[float, float]:
        """Current limits, rounded outward to the grid and clipped to [0, 1]."""
        if not self.alive.any():
            return (1.0, 0.0)
        where = np.flatnonzero(self.alive)
        step = 1.0 / (len(self.grid) + 1.0)
        low = max(0.0, float(self.grid[where[0]]) - step)
        high = min(1.0, float(self.grid[where[-1]]) + step)
        return (low, high)


class RiskAlarm:
    """L1: alarm when the mean loss is evidenced to exceed ``r_ref + delta``.

    Observations are mean losses in ``[0, scale]`` (``scale`` is the
    charter-declared bound, or ``1 / epsilon`` for propensity-weighted losses)
    and are divided by `scale` before betting.
    """

    def __init__(
        self,
        reference_risk: float,
        tolerance: float,
        alpha: float,
        scale: float = 1.0,
    ) -> None:
        """Creates the alarm.

        Args:
            reference_risk: Reference risk of the bundle, in loss units.
            tolerance: Charter tolerance `Delta`, in loss units.
            alpha: Level `alpha_alarm`: ``P(any false alarm, all time) <=
                alpha`` if the batch means have conditional mean at most
                `reference_risk + tolerance`.
            scale: Upper bound of an observation.

        Raises:
            ValueError: If the null risk is not inside ``(0, scale)``.
        """
        null = (reference_risk + tolerance) / scale
        if not 0.0 < null < 1.0:
            raise ValueError("reference_risk + tolerance must be in (0, scale)")
        self.scale = scale
        self.test = BettingTest(null, GREATER, alpha)

    def update(self, mean_loss: float) -> bool:
        """Feeds the mean loss of one matured batch; True when alarming."""
        self.test.update(min(max(mean_loss / self.scale, 0.0), 1.0))
        return bool(self.test.crossed)


class PairedEProcess:
    """L2 and L3: e-process on paired loss differences.

    The first argument of `update` is the model in service, the second the
    one it is compared with, and the difference is first minus second (loss
    of the serving model above the other's). For promotion the serving model
    is the champion and the null is "the challenger is not better by at
    least `margin`". For rollback (the reverse process) the serving model is
    the new champion, the other is the previous one, and the null is "the new
    champion is not worse by more than `margin`", which is the same bet on
    ``(difference + 1) / 2`` with the roles read that way.
    """

    def __init__(self, margin: float, alpha: float) -> None:
        """Creates the process; `alpha` is the level of one proposal."""
        if not -1.0 < margin < 1.0:
            raise ValueError("margin must be strictly between -1 and 1")
        self.margin = margin
        self.test = BettingTest((margin + 1.0) / 2.0, GREATER, alpha)

    def update(self, champion_loss: Any, other_loss: Any) -> bool:
        """Feeds the mean paired losses of one matured batch.

        Args:
            champion_loss: Per-case losses of the model in service (for
                rollback: the new champion), in ``[0, 1]``.
            other_loss: Per-case losses of the challenger (for rollback: the
                previous champion), on the same cases.

        Returns:
            Whether the process is at or above ``1 / alpha``.

        Raises:
            ValueError: On mismatched or out-of-range losses.
        """
        first = validate_unit(champion_loss, "champion_loss")
        second = validate_unit(other_loss, "other_loss")
        if first.shape != second.shape or first.size == 0:
            raise ValueError("paired losses must be non-empty and aligned")
        gap = float(np.mean(first - second))
        self.test.update((gap + 1.0) / 2.0)
        return bool(self.test.crossed)

    @property
    def e_value(self) -> float:
        """The process value now, an e-value at any stopping time."""
        return float(self.test.e_value)

    @property
    def crossed(self) -> bool:
        """Whether the process is at or above ``1 / alpha``."""
        return bool(self.test.crossed)


class PromotionSequence:
    """e-LOND across challengers, decided in registration order.

    e-LOND controls the false-discovery rate over the sequence under
    arbitrary dependence if every true null has an e-value with
    ``E[e] <= 1`` and the hypotheses are indexed in advance. A challenger is
    registered before it is scored; decisions are taken in registration order
    (a later challenger waits for the earlier ones), so the order never
    depends on the evidence.
    """

    def __init__(self, alpha: Fraction | float | str) -> None:
        """Creates the sequence at FDR level `alpha`."""
        config = online_fdr.RuleConfig(
            rule=online_fdr.ELOND, alpha=Fraction(alpha).limit_denominator()
        )
        self.config = config
        self.rule = config.build()
        self.state = self.rule.initial()
        self.names: list[str] = []
        self.decisions: dict[str, online_fdr.Step] = {}

    def register(self, name: str) -> int:
        """Registers a challenger before it is scored; returns its index."""
        if name in self.names:
            raise ValueError(f"challenger {name!r} is already registered")
        self.names.append(name)
        return len(self.names)

    def decide(self, name: str, e_value: float) -> online_fdr.Step:
        """Decides the next undecided challenger with its final e-value.

        Raises:
            ValueError: If `name` is unregistered, decided, or not next.
        """
        if name not in self.names:
            raise ValueError(f"challenger {name!r} was not registered")
        if name in self.decisions:
            raise ValueError(f"challenger {name!r} is already decided")
        pending = [n for n in self.names if n not in self.decisions]
        if pending[0] != name:
            raise ValueError(
                f"decisions follow registration order; {pending[0]!r} is next"
            )
        step = self.rule.step(self.state, online_fdr.Evidence(e=e_value))
        self.state = step.state
        self.decisions[name] = step
        return step

    @property
    def wealth(self) -> Fraction:
        """Remaining alpha-wealth of the rule."""
        return self.rule.wealth(self.state)
