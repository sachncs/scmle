"""Empirical checks of the claim ledger.

What is measured, on worlds with known truth:

* Empirical FDR of the exploration rules on independent and on dependent
  hypotheses, against ``alpha + 2 * sqrt(alpha * (1 - alpha) / R)``.
  e-LOND is held to the bound in every world. LORD++ and SAFFRON are held to
  it only where their independence assumption holds: their guarantee does
  not cover dependence and a failure there is not a bug.
* Power on a planted effect against a closed-form oracle for the Spearman
  test (Fieller, Hartley and Pearson variance ``1.06 / (n - 3)`` on the
  Fisher scale) evaluated at the e-LOND rejection threshold.
* Uniformity of the p-values on null features (KS test).
* The negative control: it must be exceeded on planted worlds, rarely
  exceeded on null worlds, and, on grouped and slowly drifting worlds, a
  global permutation must be anti-conservative while the permutation within
  groups or blocks is not.

The p-values here come from a vectorized Spearman test (`RankTester`) that is
checked against `scmle.data.ledger.bivariate.spearman` before use, so the
speed does not change what is measured.
"""

from collections.abc import Callable
import dataclasses
import math
from typing import Any

import numpy
import pandas
from scipy import stats
from sklearn.datasets import load_diabetes

from scmle.core import randomness
from scmle.data.ledger import bivariate
from scmle.data.ledger import controls
from scmle.data.ledger import online_fdr

ALPHA = 0.05
FRACTION_ALPHA = online_fdr.exact_alpha(ALPHA, "alpha")
EFFECT_RHO = 0.30
POWER_FLOOR = 0.80
KS_DRAWS = 5000
KS_FLOOR = 1e-3
CONTROL_B = 20
WorldDraw = tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]


@dataclasses.dataclass(frozen=True)
class Finding:
    """One named check and what it found.

    Attributes:
        name: What the check asserts.
        ok: Whether it held.
        detail: The measured values, as text.
        numbers: The same values in machine-readable form.
    """

    name: str
    ok: bool
    detail: str
    numbers: dict[str, Any]


class RankTester:
    """Spearman p-values of many columns against one target, vectorized.

    Valid for columns without missing values; ties are ranked by average
    rank as in `scipy.stats.rankdata`, so it equals `bivariate.spearman` for
    such columns.

    Attributes:
        n: Number of rows.
        centered: Mean-centred average ranks of the columns, shape
            (rows, columns).
        norm: Euclidean norm of each centred rank column.
    """

    def __init__(self, features: numpy.ndarray) -> None:
        """Ranks every column once.

        Args:
            features: Matrix of shape (rows, columns) without missing values.
        """
        self.n = features.shape[0]
        ranks = numpy.apply_along_axis(stats.rankdata, 0, features)
        self.centered = ranks - ranks.mean(axis=0)
        self.norm = numpy.sqrt((self.centered**2).sum(axis=0))

    def pvalues(self, target: numpy.ndarray) -> numpy.ndarray:
        """Two-sided p-value of each column against `target`.

        Args:
            target: Outcome values, one per row.

        Returns:
            An array with one p-value per column (Student t on the Spearman
            correlation with `n - 2` degrees of freedom).
        """
        ry = stats.rankdata(target)
        ry = ry - ry.mean()
        rho = (self.centered.T @ ry) / (self.norm * math.sqrt((ry**2).sum()))
        rho = numpy.clip(rho, -1.0, 1.0)
        t_value = rho * numpy.sqrt(
            (self.n - 2) / numpy.maximum(1e-300, 1 - rho**2)
        )
        return numpy.asarray(2.0 * stats.t.sf(numpy.abs(t_value), self.n - 2))


def rejections(
    config: online_fdr.RuleConfig, p: numpy.ndarray
) -> numpy.ndarray:
    """Rejection flags of the online rule over p-values, in order.

    Args:
        config: The online rule and its level.
        p: P-values in the order they are tested.

    Returns:
        A boolean array, True where the rule rejects that hypothesis.
    """
    rule = config.build()
    state = rule.initial()
    out = numpy.zeros(len(p), dtype=bool)
    for index, value in enumerate(p):
        step = rule.step(state, online_fdr.Evidence(p=float(value)))
        out[index] = step.rejected
        state = step.state
    return out


def band(alpha: float, runs: int) -> float:
    """``alpha + 2 * sqrt(alpha * (1 - alpha) / runs)``.

    Args:
        alpha: Nominal level.
        runs: Number of Monte Carlo runs.

    Returns:
        The upper edge of the acceptance band.
    """
    return alpha + 2.0 * math.sqrt(alpha * (1.0 - alpha) / runs)


def equicorrelated(
    rng: numpy.random.Generator, rows: int, columns: int, rho: float
) -> numpy.ndarray:
    """Gaussian columns with pairwise correlation `rho` (a common factor).

    Args:
        rng: Random generator.
        rows: Number of rows.
        columns: Number of columns.
        rho: Pairwise correlation, in [0, 1].

    Returns:
        An array of shape (rows, columns).
    """
    shared = rng.normal(size=(rows, 1))
    own = rng.normal(size=(rows, columns))
    return numpy.asarray(math.sqrt(rho) * shared + math.sqrt(1 - rho) * own)


def check_tester(seed: int) -> Finding:
    """The vectorized tester equals the library Spearman test.

    Args:
        seed: Seed of the random data.

    Returns:
        A finding that holds when the largest relative difference of the
        p-values is below 1e-9.
    """
    rng = numpy.random.default_rng(seed)
    x = rng.normal(size=(120, 6))
    y = rng.normal(size=120) + x[:, 0]
    fast = RankTester(x).pvalues(y)
    slow = numpy.array([bivariate.spearman(x[:, j], y).p for j in range(6)])
    gap = float(numpy.max(numpy.abs(fast - slow) / numpy.maximum(slow, 1e-300)))
    return Finding(
        "vectorized Spearman equals bivariate.spearman",
        gap < 1e-9,
        f"largest relative difference {gap:.2e}",
        {"max_relative_difference": gap},
    )


def fdp(flags: numpy.ndarray, planted: numpy.ndarray) -> float:
    """False-discovery proportion; zero when nothing is rejected.

    Args:
        flags: Rejection flags.
        planted: True where the hypothesis has a planted effect.

    Returns:
        The share of rejected hypotheses without a planted effect.
    """
    total = int(flags.sum())
    return 0.0 if total == 0 else float((flags & ~planted).sum() / total)


@dataclasses.dataclass(frozen=True)
class FdrWorld:
    """A family of hypotheses with known truth.

    Attributes:
        name: Label for the report.
        independent: Whether the test statistics are independent under the
            nulls (LORD++ and SAFFRON are held to the bound only then).
        make: ``make(run) -> (tester, target, planted)``.
    """

    name: str
    independent: bool
    make: Callable[[int], tuple[RankTester, numpy.ndarray, numpy.ndarray]]


def iid_null(seed: int, rows: int = 200, columns: int = 20) -> FdrWorld:
    """Independent features and an independent outcome: the global null.

    Args:
        seed: Run seed.
        rows: Rows per world.
        columns: Hypotheses (features) per world.

    Returns:
        The world; every hypothesis is a true null.
    """

    def make(run: int) -> tuple[RankTester, numpy.ndarray, numpy.ndarray]:
        """Draws world number `run`.

        Args:
            run: Replicate number.

        Returns:
            The tester over the features, the outcome and an all-False
            planted mask.
        """
        rng = randomness.component_rng(seed, "ledger_checks.iid_null", run)
        x = rng.normal(size=(rows, columns))
        return RankTester(x), rng.normal(size=rows), numpy.zeros(columns, bool)

    return FdrWorld("independent global null", True, make)


def diabetes_null(seed: int) -> FdrWorld:
    """Real correlated features (diabetes) with a permuted outcome.

    Args:
        seed: Run seed.

    Returns:
        The world; every hypothesis is a true null.
    """
    data = load_diabetes(as_frame=True)
    features = data.data.to_numpy(dtype=float)
    outcome = data.target.to_numpy(dtype=float)
    tester = RankTester(features)
    planted = numpy.zeros(features.shape[1], bool)

    def make(run: int) -> tuple[RankTester, numpy.ndarray, numpy.ndarray]:
        """Draws world number `run`.

        Args:
            run: Replicate number.

        Returns:
            The fixed tester, a permutation of the outcome and an all-False
            planted mask.
        """
        rng = randomness.component_rng(seed, "ledger_checks.diabetes", run)
        return tester, rng.permutation(outcome), planted

    return FdrWorld(
        "correlated global null (diabetes, permuted outcome)", False, make
    )


def mixed_world(
    seed: int,
    *,
    correlated: bool,
    rows: int = 300,
    nulls: int = 30,
    signals: int = 5,
    effect: float = 0.45,
) -> FdrWorld:
    """Planted effects among nulls; nulls correlated or independent.

    The signal columns are interleaved with the null columns by a fixed
    permutation, so the order of the hypotheses does not favour the rules.

    Args:
        seed: Run seed.
        correlated: Whether the null columns share a common factor
            (pairwise correlation 0.5).
        rows: Rows per world.
        nulls: Number of null columns.
        signals: Number of signal columns.
        effect: Coefficient of the outcome on the sum of the signal columns.

    Returns:
        The world; its planted mask marks the signal columns.
    """
    columns = nulls + signals

    def make(run: int) -> tuple[RankTester, numpy.ndarray, numpy.ndarray]:
        """Draws world number `run`.

        Args:
            run: Replicate number.

        Returns:
            The tester over the shuffled columns, the outcome and the planted
            mask.
        """
        rng = randomness.component_rng(seed, "ledger_checks.mixed", run)
        if correlated:
            null_block = equicorrelated(rng, rows, nulls, 0.5)
        else:
            null_block = rng.normal(size=(rows, nulls))
        signal_block = rng.normal(size=(rows, signals))
        x = numpy.concatenate([null_block, signal_block], axis=1)
        order = randomness.component_rng(seed, "ledger_checks.order", 0)
        position = order.permutation(columns)
        x = x[:, position]
        planted = (numpy.arange(columns) >= nulls)[position]
        y = effect * signal_block.sum(axis=1) + rng.normal(size=rows)
        return RankTester(x), y, planted

    label = "correlated nulls" if correlated else "independent nulls"
    return FdrWorld(f"planted effects, {label}", not correlated, make)


def empirical_fdr(
    config: online_fdr.RuleConfig, world: FdrWorld, runs: int
) -> tuple[float, float, float]:
    """Mean false-discovery proportion, discovery rate and mean rejections.

    Args:
        config: The online rule and its level.
        world: The family of hypotheses.
        runs: Number of worlds drawn.

    Returns:
        The mean false-discovery proportion, the share of worlds with at
        least one rejection and the mean number of rejections.
    """
    proportions = []
    discovered = 0
    total = 0
    for run in range(runs):
        tester, target, planted = world.make(run)
        flags = rejections(config, tester.pvalues(target))
        proportions.append(fdp(flags, planted))
        discovered += int(flags.any())
        total += int(flags.sum())
    return float(numpy.mean(proportions)), discovered / runs, total / runs


def check_fdr(runs: int, seed: int) -> list[Finding]:
    """Empirical FDR of each rule in each world against the binomial band.

    The e-LOND rule is held to the band in every world; the others only in
    worlds with independent nulls.

    Args:
        runs: Number of worlds per rule and world.
        seed: Run seed.

    Returns:
        One finding per rule and world.
    """
    worlds = [
        iid_null(seed),
        diabetes_null(seed),
        mixed_world(seed, correlated=False),
        mixed_world(seed, correlated=True),
    ]
    limit = band(ALPHA, runs)
    findings = []
    for rule in online_fdr.RULE_NAMES:
        config = online_fdr.RuleConfig(rule, FRACTION_ALPHA)
        for world in worlds:
            mean, _any_rate, rejected = empirical_fdr(config, world, runs)
            held = rule == online_fdr.ELOND or world.independent
            ok = mean <= limit if held else True
            status = (
                "enforced"
                if held
                else "reported only: independence assumption does not hold"
            )
            findings.append(
                Finding(
                    f"FDR {rule} on {world.name}",
                    ok,
                    f"FDR {mean:.4f} (bound {limit:.4f}, {status}), "
                    f"mean rejections {rejected:.2f}, runs {runs}",
                    {
                        "rule": rule,
                        "world": world.name,
                        "fdr": mean,
                        "bound": limit,
                        "enforced": held,
                        "mean_rejections": rejected,
                        "runs": runs,
                    },
                )
            )
    return findings


def spearman_power_oracle(rho: float, rows: int, threshold: float) -> float:
    """Closed-form power of the two-sided Spearman test at level `threshold`.

    Fisher transform with the Fieller-Hartley-Pearson variance
    ``1.06 / (n - 3)``; an approximation, stated as such.

    Args:
        rho: True correlation.
        rows: Sample size.
        threshold: Two-sided level of the test.

    Returns:
        The approximate probability of rejecting.
    """
    sd = math.sqrt(1.06 / (rows - 3))
    shift = math.atanh(rho) / sd
    z = float(stats.norm.isf(threshold / 2.0))
    return float(stats.norm.sf(z - shift) + stats.norm.cdf(-z - shift))


def elond_threshold(
    level: float, kappa: float = online_fdr.DEFAULT_KAPPA
) -> float:
    """p-value below which the calibrated e-value reaches ``1 / level``.

    Args:
        level: Level at which the hypothesis is tested.
        kappa: Calibrator parameter.

    Returns:
        The p-value threshold ``(kappa * level) ** (1 / (1 - kappa))``.
    """
    return float((kappa * level) ** (1.0 / (1.0 - kappa)))


def check_power(runs: int, seed: int) -> list[Finding]:
    """Power on one planted effect, against the closed-form oracle.

    Args:
        runs: Number of worlds.
        seed: Run seed.

    Returns:
        One finding; it holds when the empirical power is close to the
        oracle and reaches `POWER_FLOOR` whenever the oracle does.
    """
    rows, columns = 300, 20
    beta = EFFECT_RHO / math.sqrt(1.0 - EFFECT_RHO**2)
    hits = 0
    for run in range(runs):
        rng = randomness.component_rng(seed, "ledger_checks.power", run)
        x = rng.normal(size=(rows, columns))
        y = beta * x[:, 0] + rng.normal(size=rows)
        config = online_fdr.RuleConfig(online_fdr.ELOND, FRACTION_ALPHA)
        flags = rejections(config, RankTester(x).pvalues(y))
        hits += int(flags[0])
    # The planted column is tested first: level alpha * gamma_1 = alpha / 2.
    threshold = elond_threshold(ALPHA / 2.0)
    oracle = spearman_power_oracle(EFFECT_RHO, rows, threshold)
    power = hits / runs
    se = math.sqrt(max(power * (1 - power), 1e-9) / runs)
    close = abs(power - oracle) <= 3 * se + 0.03
    attainable = oracle >= POWER_FLOOR
    ok = (power >= POWER_FLOOR or not attainable) and close
    return [
        Finding(
            "power on a planted effect",
            ok,
            f"empirical {power:.3f} vs closed form {oracle:.3f} at "
            f"p <= {threshold:.2e} (rho {EFFECT_RHO}, n {rows}); "
            f"floor {POWER_FLOOR} applies when the oracle reaches it: "
            f"{attainable}",
            {
                "power": power,
                "oracle": oracle,
                "threshold": threshold,
                "rho": EFFECT_RHO,
                "rows": rows,
                "runs": runs,
            },
        )
    ]


def check_uniformity(seed: int) -> list[Finding]:
    """KS test of null p-values against Uniform(0, 1).

    Args:
        seed: Run seed.

    Returns:
        One finding; it holds when the KS p-value exceeds `KS_FLOOR`.
    """
    rng = randomness.component_rng(seed, "ledger_checks.uniform", 0)
    draws = [
        bivariate.spearman(rng.normal(size=60), rng.normal(size=60)).p
        for _ in range(KS_DRAWS)
    ]
    result = stats.kstest(draws, "uniform")
    share = float(numpy.mean(numpy.array(draws) <= 0.05))
    return [
        Finding(
            "null p-values are uniform",
            bool(result.pvalue > KS_FLOOR),
            f"KS p {result.pvalue:.3f} over {KS_DRAWS} draws "
            f"(floor {KS_FLOOR}); P(p <= 0.05) = {share:.4f}",
            {"ks_p": float(result.pvalue), "share_below_005": share},
        )
    ]


def grouped_world(
    rng: numpy.random.Generator, groups: int, per_group: int, columns: int
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Group effects in features and outcome, no relation between them.

    Args:
        rng: Random generator.
        groups: Number of groups.
        per_group: Rows per group.
        columns: Number of feature columns.

    Returns:
        The features, the outcome and the group label of each row.
    """
    labels = numpy.repeat(numpy.arange(groups), per_group)
    rows = groups * per_group
    x = rng.normal(size=(groups, columns))[labels] + 0.5 * rng.normal(
        size=(rows, columns)
    )
    y = rng.normal(size=groups)[labels] + 0.5 * rng.normal(size=rows)
    return x, y, labels


def drift_world(
    rng: numpy.random.Generator, rows: int, columns: int, blocks: int = 20
) -> WorldDraw:
    """Independent random walks: a spurious regression between slow drifts.

    The labels cut the rows into contiguous blocks of `rows // blocks` rows
    (at least 2). The block must be short against the timescale of the
    drift: with 10 blocks of 30 rows the within-block permutation exceeded
    the control in about 7.7% of worlds (above the 5% target); with 20
    blocks, about 3.7%.

    Args:
        rng: Random generator.
        rows: Number of rows.
        columns: Number of feature walks.
        blocks: Target number of blocks; the block size is `rows // blocks`,
            at least 2.

    Returns:
        The feature walks, the outcome walk and the block label of each row.
    """
    x = numpy.cumsum(rng.normal(size=(rows, columns)), axis=0)
    y = numpy.cumsum(rng.normal(size=rows))
    labels = controls.block_labels(
        numpy.arange(rows, dtype=float), max(2, rows // blocks)
    )
    return x, y, labels


def control_exceed_rate(
    make: Callable[
        [numpy.random.Generator],
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray],
    ],
    scope: str,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    """Share of worlds whose observed count exceeds the control percentile.

    Args:
        make: Draws a world's features, outcome and strata labels.
        scope: `global` permutes over all rows; any other value permutes
            within the strata.
        repeats: Number of worlds.
        seed: Run seed.

    Returns:
        (exceed rate, mean observed discoveries).
    """
    config = online_fdr.RuleConfig(online_fdr.ELOND, FRACTION_ALPHA)
    exceeded = 0
    observed_total = 0
    for repeat in range(repeats):
        rng = randomness.component_rng(seed, f"ledger_checks.{scope}", repeat)
        x, y, labels = make(rng)
        tester = RankTester(x)

        def count(
            frame: pandas.DataFrame,
            shuffled: pandas.Series,
            tester: RankTester = tester,
        ) -> int:
            """Discoveries of the online rule on a shuffled outcome.

            Args:
                frame: The features (unused; the tester holds them).
                shuffled: The permuted outcome.
                tester: The rank tester of this world.

            Returns:
                The number of rejections.
            """
            p = tester.pvalues(shuffled.to_numpy())
            return int(rejections(config, p).sum())

        observed = int(rejections(config, tester.pvalues(y)).sum())
        observed_total += observed
        strata = None if scope == "global" else labels
        result = controls.permuted_target_control(
            count,
            pandas.DataFrame(x),
            pandas.Series(y),
            observed=observed,
            b=CONTROL_B,
            seed=seed + repeat,
            strata=strata,
        )
        exceeded += int(result.exceeds)
    return exceeded / repeats, observed_total / repeats


def check_controls(repeats: int, seed: int) -> list[Finding]:
    """Negative-control behaviour on null, planted, grouped, drifting data.

    Args:
        repeats: Number of worlds per case.
        seed: Run seed.

    Returns:
        Findings for the null, planted, grouped and drifting worlds.
    """
    limit = band(ALPHA, repeats)

    def null(rng: numpy.random.Generator) -> WorldDraw:
        """A global-null world.

        Args:
            rng: Random generator.

        Returns:
            150 rows of 15 features, an independent outcome and one stratum.
        """
        x = rng.normal(size=(150, 15))
        return x, rng.normal(size=150), numpy.zeros(150, int)

    def planted(rng: numpy.random.Generator) -> WorldDraw:
        """A world with four planted effects.

        Args:
            rng: Random generator.

        Returns:
            200 rows of 15 features, an outcome driven by the first four and one
            stratum.
        """
        x = rng.normal(size=(200, 15))
        y = x[:, :4].sum(axis=1) * 0.6 + rng.normal(size=200)
        return x, y, numpy.zeros(200, int)

    def grouped(rng: numpy.random.Generator) -> WorldDraw:
        """A grouped world with no feature-outcome relation.

        Args:
            rng: Random generator.

        Returns:
            The output of `grouped_world` with 20 groups of 15 rows.
        """
        return grouped_world(rng, groups=20, per_group=15, columns=15)

    findings = []
    null_rate, _ = control_exceed_rate(null, "global", repeats, seed)
    findings.append(
        Finding(
            "control is rarely exceeded on null worlds",
            null_rate <= limit,
            f"exceeded in {null_rate:.3f} of {repeats} (bound {limit:.3f})",
            {"rate": null_rate, "bound": limit, "repeats": repeats},
        )
    )
    planted_rate, planted_mean = control_exceed_rate(
        planted, "global", repeats, seed
    )
    findings.append(
        Finding(
            "control is exceeded on every planted world",
            planted_rate == 1.0,
            f"exceeded in {planted_rate:.3f} of {repeats}, mean "
            f"{planted_mean:.1f} discoveries",
            {"rate": planted_rate, "repeats": repeats},
        )
    )
    for label, make in (("grouped", grouped), ("drifting", drift_world_make)):
        within, obs = control_exceed_rate(make, "strata", repeats, seed)
        global_rate, _ = control_exceed_rate(make, "global", repeats, seed)
        extra: dict[str, Any] = {}
        if label == "drifting":
            coarse, _ = control_exceed_rate(
                coarse_drift_world_make, "strata", repeats, seed
            )
            extra["coarse_10_blocks_rate"] = coarse
        findings.append(
            Finding(
                f"{label} world: within-strata permutation is not "
                "anti-conservative, a global one is",
                within <= limit and global_rate > limit,
                f"strata permutation exceeded {within:.3f}, global "
                f"permutation {global_rate:.3f} of {repeats} worlds "
                f"(bound {limit:.3f}); mean observed {obs:.1f}",
                {
                    "strata_rate": within,
                    "global_rate": global_rate,
                    "bound": limit,
                    "repeats": repeats,
                    "mean_observed": obs,
                    **extra,
                },
            )
        )
    return findings


def drift_world_make(rng: numpy.random.Generator) -> WorldDraw:
    """Random-walk world with 15 independent walks, 20 blocks.

    Args:
        rng: Random generator.

    Returns:
        The output of `drift_world` with 300 rows.
    """
    return drift_world(rng, 300, 15, blocks=20)


def coarse_drift_world_make(rng: numpy.random.Generator) -> WorldDraw:
    """The same world cut into only 10 blocks (reported, not enforced).

    Args:
        rng: Random generator.

    Returns:
        The output of `drift_world` with 300 rows and 10 blocks.
    """
    return drift_world(rng, 300, 15, blocks=10)


def run_all(runs: int, seed: int, repeats: int) -> list[Finding]:
    """Every ledger check.

    Args:
        runs: Number of worlds for the FDR and power checks.
        seed: Run seed.
        repeats: Number of worlds for the control checks.

    Returns:
        The findings of all checks.
    """
    findings = [check_tester(seed)]
    findings += check_fdr(runs, seed)
    findings += check_power(runs, seed)
    findings += check_uniformity(seed)
    findings += check_controls(repeats, seed)
    return findings


def smoke_findings(seed: int) -> list[Finding]:
    """A cheap subset for the smoke gate (fewer runs, wider bands).

    Args:
        seed: Run seed.

    Returns:
        The tester, uniformity, FDR and power findings, with 200 runs for
        the last two.
    """
    findings = [check_tester(seed), *check_uniformity(seed)]
    findings += check_fdr(200, seed)
    findings += check_power(200, seed)
    return findings


def report(findings: list[Finding]) -> dict[str, Any]:
    """JSON-ready report of a list of findings.

    Args:
        findings: The findings to report.

    Returns:
        The overall `ok` and each finding as a dict.
    """
    return {
        "ok": all(f.ok for f in findings),
        "findings": [dataclasses.asdict(f) for f in findings],
    }
