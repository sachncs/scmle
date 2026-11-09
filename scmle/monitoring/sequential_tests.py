"""Anytime-valid monitors: betting tests, confidence sequences, e-processes.

The fixed-sample rules in `monitor.py` and `champion.py` are applied again at
every batch, so their false-alarm and false-promotion rates grow with the
number of looks and nothing bounds them. The monitors here are nonnegative
supermartingales under their null, so by Ville's inequality
``P(sup_t K_t >= 1 / alpha) <= alpha`` over all time: checking at every batch
costs nothing in validity.

Construction (betting for bounded means, after Waudby-Smith and Ramdas,
arXiv:2010.09686, whose Sections 3 and 4, Theorems 1 to 3 and Proposition 2
were read in full on 2026-11-05). For observations ``x_t`` in ``[0, 1]`` and a
null "the mean is at most ``m0``" the capital

    K_t = prod_{i <= t} (1 + lambda_i * (x_i - m0)),  0 <= lambda_i < 1 / m0

has ``E[K_t | past] <= K_{t-1}`` whenever ``lambda_i`` is a function of the
past only, and stays positive (the paper's capital process, eq. 23, with a
``(-1/(1-m), 1/m)``-valued predictable sequence; Proposition 2 shows it is a
test martingale when the conditional mean is exactly ``m``). The mirror null
"the mean is at least ``m0``" uses ``1 - lambda_i * (x_i - m0)`` with
``lambda_i < 1 / (1 - m0)``. Ville's inequality for a test supermartingale is
quoted in the paper (stated right after Theorem 1):
``P(exists t: K_t >= 1/alpha) <= alpha``.

The bet size is ``lambda_t = sqrt(2 log(1 / alpha) / (s2_{t-1} * t * log(1 +
t)))`` with the running variance ``s2_t = (1/4 + sum (x_i - mu_i)^2) / (t +
1)`` and ``mu_i = (1/2 + sum_{j <= i} x_j) / (i + 1)``, capped at ``cap`` times
the positivity bound. What the paper supports and what it does not: that
formula for ``lambda`` is the paper's recommended plug-in for the
*empirical-Bernstein* confidence sequence (Theorem 2, eq. 15), where it has
``log(2 / alpha)`` for a two-sided interval and a cap ``c`` in ``(0, 1)``
(default 1/2 or 3/4); the hedged capital sequence (Theorem 3, caps ``c/m`` and
``c/(1-m)``) is valid for any predictable bet. Using the Theorem 2 plug-in as
the bet of a one-sided capital with ``log(1 / alpha)`` is this module's
adaptation, not a result of the paper: validity here rests only on the bet
being predictable and below the positivity bound. The paper's coverage
statement is for conditionally mean-``mu`` observations; the one-sided
supermartingale argument above covers a conditional mean at most ``m0``.

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
    """Checks that values are finite and lie in ``[0, 1]``.

    Args:
        value: A number or array of numbers.
        name: Name used in the error message.

    Returns:
        The values as a float array.

    Raises:
        ValueError: If a value is not finite or is outside ``[0, 1]``.
    """
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
        """The value ``log(1 / alpha)``, where the test crosses."""
        return math.log(1.0 / self.alpha)

    def bet(self) -> np.ndarray:
        """Computes the predictable bet size for the next observation.

        Returns:
            The bet, from the running variance of the observations so far and
            capped at the positivity bound; one value per null mean.
        """
        step = self.count + 1
        variance = (0.25 + self.deviation) / step
        raw = np.sqrt(
            2.0 * self.threshold / (variance * step * math.log(1.0 + step))
        )
        return np.minimum(raw, self.bound)

    def update(self, observation: Any) -> np.ndarray:
        """Feeds one observation in ``[0, 1]``.

        Args:
            observation: The observation. A value that is not finite or is
                outside ``[0, 1]`` raises `ValueError`.

        Returns:
            The log capital after the observation.
        """
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
        """Returns the JSON-ready state (scalar tests only).

        Returns:
            A dict with the settings, the observation count and sums, the log
            capital and its peak.

        Raises:
            ValueError: If the test has more than one null mean.
        """
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
        """Rebuilds a test from `to_dict`.

        Args:
            record: A dict produced by `to_dict`.

        Returns:
            The test, in the state that was saved.
        """
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


class MixtureTest:
    """A betting test restarted at fixed times and mixed over its starts.

    A single `BettingTest` multiplies capital from the first observation, so
    after a long quiet stretch the capital is far below one and a later change
    must first climb back; the delay of the alarm grows with the quiet time.
    This test runs one `BettingTest` from each of a fixed set of start times
    (the first observation and then every `stride` observations) and sums
    their capitals with weights ``w_j = 1 / ((j + 1) (j + 2))``, which add up
    to one. Each term is a nonnegative supermartingale under the null that
    equals its weight until its start, so the sum is one as well, and
    ``P(sup_t M_t >= 1 / alpha) <= alpha`` still holds at every stopping time.
    Only the newest `terms` starts are kept; dropping terms can only lower the
    sum, so the guarantee is unchanged (a little conservative). The price is
    the log of the weight of the start that matters, which grows only like
    ``2 log(t / stride)``, and a delay of up to `stride` observations before
    a restart that begins at a change.
    """

    def __init__(
        self,
        null_mean: Any,
        direction: str,
        alpha: float,
        stride: int = 25,
        terms: int = 64,
        cap: float = DEFAULT_CAP,
    ) -> None:
        """Creates the test; arguments as for `BettingTest`.

        Args:
            null_mean: Mean under the null, strictly inside ``(0, 1)``.
            direction: `greater` or `less`.
            alpha: Level, in ``(0, 1)``.
            stride: Observations between starts.
            terms: Newest starts kept.
            cap: Share of the positivity bound a bet may use.

        Raises:
            ValueError: If `stride` or `terms` is not positive.
        """
        if stride < 1 or terms < 1:
            raise ValueError("stride and terms must be positive")
        self.null_mean = np.asarray(null_mean, dtype=float)
        self.direction = direction
        self.alpha = alpha
        self.stride = stride
        self.terms = terms
        self.cap = cap
        self.count = 0
        self.started = 0
        self.inner: list[BettingTest] = []
        self.log_weights: list[float] = []
        self.peak = np.zeros_like(self.null_mean)
        self.log_capital = np.zeros_like(self.null_mean)
        self.start_term()

    def start_term(self) -> None:
        """Adds a term that starts with the next observation.

        The oldest term is dropped when more than `terms` are kept.
        """
        self.inner.append(
            BettingTest(self.null_mean, self.direction, self.alpha, self.cap)
        )
        self.log_weights.append(
            -math.log((self.started + 1) * (self.started + 2))
        )
        self.started += 1
        if len(self.inner) > self.terms:
            self.inner.pop(0)
            self.log_weights.pop(0)

    @property
    def threshold(self) -> float:
        """The value ``log(1 / alpha)``."""
        return math.log(1.0 / self.alpha)

    def update(self, observation: Any) -> np.ndarray:
        """Feeds one observation in ``[0, 1]``.

        A new term starts every `stride` observations.

        Args:
            observation: The observation.

        Returns:
            The log of the weighted sum of the terms' capitals.
        """
        if self.count > 0 and self.count % self.stride == 0:
            self.start_term()
        for test in self.inner:
            test.update(observation)
        self.count += 1
        stacked = np.stack(
            [
                w + t.log_capital
                for w, t in zip(self.log_weights, self.inner, strict=True)
            ]
        )
        top = stacked.max(axis=0)
        self.log_capital = top + np.log(np.exp(stacked - top).sum(axis=0))
        self.peak = np.maximum(self.peak, self.log_capital)
        return self.log_capital

    @property
    def crossed(self) -> Any:
        """Whether the mixture reached ``1 / alpha`` now."""
        return self.log_capital >= self.threshold

    @property
    def ever_crossed(self) -> Any:
        """Whether the mixture ever reached ``1 / alpha``."""
        return self.peak >= self.threshold

    @property
    def e_value(self) -> np.ndarray:
        """The mixture now: an e-value at any stopping time."""
        return np.exp(self.log_capital)

    def to_dict(self) -> dict[str, Any]:
        """Returns the JSON-ready state (scalar tests only).

        Returns:
            A dict with the settings, counts, term weights, the terms' states,
            the log capital and its peak.

        Raises:
            ValueError: If the test has more than one null mean.
        """
        if self.null_mean.ndim != 0:
            raise ValueError("only a single test can be serialized")
        return {
            "null_mean": float(self.null_mean),
            "direction": self.direction,
            "alpha": self.alpha,
            "stride": self.stride,
            "terms": self.terms,
            "cap": self.cap,
            "count": self.count,
            "started": self.started,
            "log_weights": list(self.log_weights),
            "inner": [t.to_dict() for t in self.inner],
            "log_capital": float(self.log_capital),
            "peak": float(self.peak),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "MixtureTest":
        """Rebuilds a test from `to_dict`.

        Args:
            record: A dict produced by `to_dict`.

        Returns:
            The test, in the state that was saved.
        """
        test = cls(
            record["null_mean"],
            record["direction"],
            record["alpha"],
            record["stride"],
            record["terms"],
            record["cap"],
        )
        test.count = int(record["count"])
        test.started = int(record["started"])
        test.log_weights = [float(w) for w in record["log_weights"]]
        test.inner = [BettingTest.from_dict(t) for t in record["inner"]]
        test.log_capital = np.asarray(float(record["log_capital"]))
        test.peak = np.asarray(float(record["peak"]))
        return test


class ConfidenceSequence:
    """Time-uniform confidence sequence for a bounded mean, on a grid.

    Two one-sided betting tests per grid point, each at ``alpha / 2``; a grid
    point is out when either crosses. Limits are the running intersection, so
    they only tighten, and are rounded outward to the grid (so the interval is
    never narrower than the exact one on the grid).

    Attributes:
        alpha: Level of the sequence.
        grid: The grid of candidate means.
        upper_tests: Tests of the null "mean at most the grid value".
        lower_tests: Tests of the null "mean at least the grid value".
        alive: Grid points not yet excluded.
    """

    def __init__(self, alpha: float, points: int = GRID_POINTS) -> None:
        """Creates a sequence with `points` grid values in ``(0, 1)``.

        Args:
            alpha: Level of the sequence.
            points: Number of grid values.

        Raises:
            ValueError: If `points` is below 3.
        """
        if points < 3:
            raise ValueError("points must be at least 3")
        self.alpha = alpha
        self.grid = (np.arange(points) + 1.0) / (points + 1.0)
        self.upper_tests = BettingTest(self.grid, GREATER, alpha / 2.0)
        self.lower_tests = BettingTest(self.grid, LESS, alpha / 2.0)
        self.alive = np.ones(points, dtype=bool)

    def update(self, observation: Any) -> tuple[float, float]:
        """Feeds one observation.

        Args:
            observation: The observation, in ``[0, 1]``.

        Returns:
            The interval ``(low, high)``.
        """
        self.upper_tests.update(observation)
        self.lower_tests.update(observation)
        self.alive &= ~(self.upper_tests.crossed | self.lower_tests.crossed)
        return self.interval

    @property
    def interval(self) -> tuple[float, float]:
        """The current limits, rounded outward to the grid, within [0, 1].

        ``(1.0, 0.0)`` means every grid point has been excluded.
        """
        if not self.alive.any():
            return (1.0, 0.0)
        where = np.flatnonzero(self.alive)
        step = 1.0 / (len(self.grid) + 1.0)
        low = max(0.0, float(self.grid[where[0]]) - step)
        high = min(1.0, float(self.grid[where[-1]]) + step)
        return (low, high)


class RiskAlarm:
    """Alarm when the mean loss is evidenced to exceed ``r_ref + delta``.

    Observations are mean losses in ``[0, scale]`` (``scale`` is the
    charter-declared bound, or ``1 / epsilon`` for propensity-weighted losses)
    and are divided by `scale` before betting.

    Attributes:
        scale: Upper bound of an observation.
        test: The mixture test that decides the alarm.
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
        self.test = MixtureTest(null, GREATER, alpha)

    def to_dict(self) -> dict[str, Any]:
        """Returns the JSON-ready state.

        Returns:
            A dict with the scale and the test's state.
        """
        return {"scale": self.scale, "test": self.test.to_dict()}

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "RiskAlarm":
        """Rebuilds an alarm from `to_dict`.

        Args:
            record: A dict produced by `to_dict`.

        Returns:
            The alarm, in the state that was saved.
        """
        state = record["test"]
        alarm = cls(
            state["null_mean"] * record["scale"],
            0.0,
            state["alpha"],
            record["scale"],
        )
        alarm.test = MixtureTest.from_dict(state)
        return alarm

    def update(self, mean_loss: float) -> bool:
        """Feeds the mean loss of one matured batch.

        Args:
            mean_loss: Mean loss of the batch, in loss units; it is divided by
                the scale and clipped to ``[0, 1]``.

        Returns:
            True when the alarm is raised (capital at or above ``1 / alpha``).
        """
        self.test.update(min(max(mean_loss / self.scale, 0.0), 1.0))
        return bool(self.test.crossed)


class PairedEProcess:
    """E-process on paired loss differences, for promotion and rollback.

    The first argument of `update` is the model in service, the second the one
    it is compared with, and the difference is first minus second (loss of the
    serving model above the other's). For promotion the serving model is the
    champion and the null is "the challenger is not better by at least
    `margin`". For rollback (the reverse process) the serving model is the new
    champion, the other is the previous one, and the null is "the new champion
    is not worse by more than `margin`", which is the same bet on ``(difference
    / scale + 1) / 2`` with the roles read that way.

    With propensity-weighted losses a batch difference lies in ``[-scale,
    scale]`` where ``scale = bound / epsilon``; pass it as `scale` and feed
    `update_gap`.

    Attributes:
        margin: The margin of the null, in loss units.
        scale: Largest absolute batch difference.
        test: The mixture test that carries the process.
    """

    def __init__(self, margin: float, alpha: float, scale: float = 1.0) -> None:
        """Creates the process; `alpha` is the level of one proposal.

        Args:
            margin: Margin of the null, strictly within ``(-scale, scale)``.
            alpha: Level of one proposal.
            scale: Largest absolute batch difference.

        Raises:
            ValueError: If `scale` is not positive or `margin` is not strictly
                within ``(-scale, scale)``.
        """
        if scale <= 0.0 or not -scale < margin < scale:
            raise ValueError("margin must lie strictly within (-scale, scale)")
        self.margin = margin
        self.scale = scale
        self.test = MixtureTest((margin / scale + 1.0) / 2.0, GREATER, alpha)

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
        return self.update_gap(float(np.mean(first - second)))

    def update_gap(self, gap: float) -> bool:
        """Feeds a batch's mean loss difference, in loss units.

        Args:
            gap: Mean of first minus second, within ``[-scale, scale]``.

        Returns:
            Whether the process is at or above ``1 / alpha``.

        Raises:
            ValueError: If `gap` is not finite or lies outside ``[-scale,
                scale]``.
        """
        if not np.isfinite(gap) or abs(gap) > self.scale + 1e-12:
            raise ValueError("gap must be finite and within [-scale, scale]")
        unit = min(max((gap / self.scale + 1.0) / 2.0, 0.0), 1.0)
        self.test.update(unit)
        return bool(self.test.crossed)

    @property
    def e_value(self) -> float:
        """The process value now, an e-value at any stopping time."""
        return float(self.test.e_value)

    @property
    def crossed(self) -> bool:
        """Whether the process is at or above ``1 / alpha``."""
        return bool(self.test.crossed)

    def to_dict(self) -> dict[str, Any]:
        """Returns the JSON-ready state.

        Returns:
            A dict with the margin, the scale and the test's state.
        """
        return {
            "margin": self.margin,
            "scale": self.scale,
            "test": self.test.to_dict(),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "PairedEProcess":
        """Rebuilds a process from `to_dict`.

        Args:
            record: A dict produced by `to_dict`.

        Returns:
            The process, in the state that was saved.
        """
        state = record["test"]
        process = cls(record["margin"], state["alpha"], record["scale"])
        process.test = MixtureTest.from_dict(state)
        return process


class PromotionSequence:
    """E-LOND across challengers, decided in registration order.

    e-LOND controls the false-discovery rate over the sequence under arbitrary
    dependence if every true null has an e-value with ``E[e] <= 1`` and the
    hypotheses are indexed in advance. A challenger is registered before it is
    scored; decisions are taken in registration order (a later challenger waits
    for the earlier ones), so the order never depends on the evidence.

    Attributes:
        config: The online-FDR rule configuration.
        rule: The e-LOND rule.
        state: The rule's current state.
        names: Registered challengers, in registration order.
        decisions: The step taken for each decided challenger.
    """

    def __init__(self, alpha: Fraction | float | str) -> None:
        """Creates the sequence at FDR level `alpha`.

        Args:
            alpha: The false-discovery rate level.
        """
        config = online_fdr.RuleConfig(
            rule=online_fdr.ELOND, alpha=Fraction(alpha).limit_denominator()
        )
        self.config = config
        self.rule = config.build()
        self.state = self.rule.initial()
        self.names: list[str] = []
        self.decisions: dict[str, online_fdr.Step] = {}

    def register(self, name: str) -> int:
        """Registers a challenger before it is scored.

        Args:
            name: Name of the challenger.

        Returns:
            Its index in the registration order, from 1.

        Raises:
            ValueError: If the challenger is already registered.
        """
        if name in self.names:
            raise ValueError(f"challenger {name!r} is already registered")
        self.names.append(name)
        return len(self.names)

    def decide(self, name: str, e_value: float) -> online_fdr.Step:
        """Decides the next undecided challenger with its final e-value.

        Args:
            name: Name of the challenger.
            e_value: Its final e-value.

        Returns:
            The rule's step, which carries the decision.

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
        """The remaining alpha-wealth of the rule."""
        return self.rule.wealth(self.state)
