"""The hypotheses the screen tests, and the bound of each significant one.

Every column raises skill, echo, missingness and record-purity hypotheses
as its values allow; together they form one Benjamini-Yekutieli family.
"""

from collections.abc import Hashable
from collections.abc import Sequence
import dataclasses
import math
from typing import Any

import numpy
import pandas

from scmle.assurance.admissibility import leakage
from scmle.assurance.admissibility.screen.association import cramer
from scmle.assurance.admissibility.screen.association import dominance
from scmle.assurance.admissibility.screen.association import purity_statistic
from scmle.assurance.admissibility.screen.association import resample
from scmle.assurance.admissibility.screen.fitting import basis
from scmle.assurance.admissibility.screen.fitting import pair
from scmle.assurance.admissibility.screen.fitting import pairs
from scmle.assurance.admissibility.screen.fitting import Problem
from scmle.assurance.admissibility.screen.fitting import skill_bound
from scmle.assurance.admissibility.screen.fitting import skill_statistic
from scmle.assurance.admissibility.screen.settings import ECHO
from scmle.assurance.admissibility.screen.settings import MIN_CLASSES
from scmle.assurance.admissibility.screen.settings import MISSING
from scmle.assurance.admissibility.screen.settings import PURITY
from scmle.assurance.admissibility.screen.settings import SKILL
from scmle.assurance.admissibility.screen.settings import Statistic
from scmle.runtime import randomness


@dataclasses.dataclass(slots=True)
class Pending:
    """One hypothesis while the shared permutations are drawn.

    Attributes:
        feature: Column name.
        mechanism: `skill`, `missingness` or `record_purity`.
        observed: Observed statistic.
        statistic: Function of a batch of permuted labels, or None when the
            p-value is analytic.
        uses_binned: Whether the statistic reads the binned outcome.
        p_value: Analytic p-value, when there is no statistic.
        exceed: Permutations with a statistic at least the observed one.
        drawn: Permutations drawn for it.
        active: Whether it still draws permutations.
        data: What the effect bound reads: the column's basis for a skill
            hypothesis, the column as the target (`forecast`) for an echo;
            None otherwise.
    """

    feature: str
    mechanism: str
    observed: float
    statistic: Statistic | None
    uses_binned: bool
    p_value: float | None = None
    exceed: int = 0
    drawn: int = 0
    active: bool = True
    data: numpy.ndarray | None = None


def spell(values: pandas.Series) -> pandas.Series:
    """Writes values as text, a whole number without its decimals.

    Args:
        values: A column or the target.

    Returns:
        One string per value, so the label 1 and the value 1.0 read alike.
    """

    def write(value: Any) -> str:
        if (
            isinstance(value, float | numpy.floating)
            and float(value).is_integer()
        ):
            return str(int(value))
        return str(value)

    return values.map(write)


def forecast(
    series: pandas.Series,
    outcome: pandas.Series,
    problem: Problem,
    ceiling: float,
) -> numpy.ndarray | None:
    """Reads a column as the target itself, when it could echo it.

    A field recorded after the outcome often is the outcome, written again
    with errors: the label with some entries garbled, or the number plus
    noise. Read as a prediction without any fitting, it agrees with the
    target almost everywhere, which a legitimate predictor, however strong,
    does not: it is on its own scale. The column is returned only when it
    can reach `ceiling`, decided on its marginal and the target's alone, so
    the choice does not depend on how the two are paired and the
    permutation p-value stays exact.

    For classes, agreement is at most `sum_k min(x_k, y_k) / n`, `x_k` and
    `y_k` counting the rows where the column and the target hold class `k`,
    so that must exceed `ceiling`; a column spelling fewer than two classes
    agrees equally under every pairing and has no test. For a number, the
    unfitted
    R^2 is `1 - mean((y - x)^2) / var(y)`, and `mean((y - x)^2)` is at least
    both `(mean(y) - mean(x))^2` and `(sd(y) - sd(x))^2`; with
    `slack = sqrt(1 - ceiling)` the means and the standard deviations must
    each lie within `slack * sd(y)` of the target's.

    Args:
        series: The column.
        outcome: The target column.
        problem: The encoded target.
        ceiling: The echo ceiling.

    Returns:
        For classes, the class code each value spells (-1 for no class);
        for a number with no missing values, the values; None when the
        column cannot echo the target above `ceiling`.
    """
    if problem.task == "classification":
        labels = pandas.Series(pandas.factorize(outcome, sort=True)[1])
        lookup = {text: code for code, text in enumerate(spell(labels))}
        codes = spell(series).map(lookup).fillna(-1).to_numpy("int64")
        spelled = numpy.bincount(codes[codes >= 0], minlength=len(labels))
        held = numpy.bincount(problem.y, minlength=len(labels))
        reach = float(numpy.minimum(spelled, held).sum()) / len(codes)
        if numpy.count_nonzero(spelled) < MIN_CLASSES or reach <= ceiling:
            return None
        return codes
    numeric = pandas.api.types.is_numeric_dtype(
        series
    ) and not pandas.api.types.is_bool_dtype(series)
    if not numeric or series.isna().any():
        return None
    values = series.to_numpy(dtype="float64")
    spread = float(problem.y.std())
    slack = math.sqrt(1.0 - ceiling) * spread
    if abs(float(values.mean() - problem.y.mean())) > slack:
        return None
    if abs(float(values.std()) - spread) > slack:
        return None
    return values


def reflect(problem: Problem, data: numpy.ndarray) -> Statistic:
    """Builds the echo statistic of a column read as the target.

    Args:
        problem: The encoded target.
        data: The column as the target, from `forecast`.

    Returns:
        A function mapping an array of shape (B, n) of target versions to
        a length-B array: the share of rows where the column spells the
        class, or `1 - sum((y - x)^2) / sum((y - mean(y))^2)` for a number.
    """

    def compute(labels: numpy.ndarray) -> numpy.ndarray:
        if problem.task == "classification":
            share: numpy.ndarray = (labels == data[None, :]).mean(axis=1)
            return share
        squares = ((labels - data[None, :]) ** 2).sum(axis=1)
        efficiency: numpy.ndarray = 1.0 - squares / problem.total_squares
        return efficiency

    return compute


def hedge(
    problem: Problem,
    data: numpy.ndarray,
    cfg: leakage.Config,
    groups: numpy.ndarray | None,
    rng: numpy.random.Generator,
) -> float:
    """One-sided lower confidence bound of the echo statistic by bootstrap.

    Rows are resampled, or whole groups when `groups` is given; a draw with
    a constant numeric target is skipped, and with none left the bound is
    -inf.

    Args:
        problem: The encoded target.
        data: The column as the target, from `forecast`.
        cfg: Settings (`bootstrap`, `confidence`).
        groups: Group label per row, or None.
        rng: Random generator.

    Returns:
        The `1 - confidence` quantile of the bootstrap values, or -inf.
    """
    members = (
        [numpy.flatnonzero(groups == g) for g in pandas.unique(groups)]
        if groups is not None
        else None
    )
    n = len(problem.y)
    values = []
    for draw in range(cfg.bootstrap):
        del draw
        if members is not None:
            picks = rng.integers(0, len(members), size=len(members))
            index = numpy.concatenate([members[i] for i in picks])
        else:
            index = rng.integers(0, n, size=n)
        y = problem.y[index]
        if problem.task == "classification":
            values.append(float((y == data[index]).mean()))
            continue
        total = float(((y - y.mean()) ** 2).sum())
        if total > 0.0:
            values.append(1.0 - float(((y - data[index]) ** 2).sum()) / total)
    if not values:
        return -math.inf
    return float(numpy.quantile(values, 1.0 - cfg.confidence))


def gauge(name: str, phi: numpy.ndarray, problem: Problem) -> list[Pending]:
    """Builds the skill hypothesis of one design, when it can carry any.

    Args:
        name: Column or pair name.
        phi: The design.
        problem: The encoded target.

    Returns:
        The hypothesis with its observed skill, or nothing for a design of
        one column.
    """
    if phi.shape[1] <= 1:
        return []
    statistic = skill_statistic(problem, phi)
    observed = float(statistic(problem.y[None, :])[0])
    return [
        Pending(name, SKILL, observed, statistic, uses_binned=False, data=phi)
    ]


def propose(
    frame: pandas.DataFrame,
    target: str,
    names: Sequence[Hashable],
    problem: Problem,
    cfg: leakage.Config,
    class_codes: numpy.ndarray,
    totals: numpy.ndarray,
    *,
    structured: bool,
) -> tuple[list[Pending], tuple[str, ...]]:
    """Forms every hypothesis of the family, so the scan and `measure` agree.

    Per column: skill, record purity and missingness (`hypothesize`) and
    echo (`forecast`); then the joint skill of each pair of two-level
    columns, while there are no more pairs than candidate columns.

    Args:
        frame: The chosen rows, including the target.
        target: Target column.
        names: Candidate columns.
        problem: The encoded target with its folds.
        cfg: Settings.
        class_codes: Class code of the binned outcome per row.
        totals: Rows per class.
        structured: Whether rows are grouped or in time blocks.

    Returns:
        The hypotheses with their observed statistics, and a note when the
        pairs were too many to screen.
    """
    ceiling = caps(cfg)[ECHO]
    pending: list[Pending] = []
    for name in names:
        series = frame[name]
        pending.extend(gauge(str(name), basis(series), problem))
        pending.extend(
            hypothesize(
                str(name),
                series,
                frame[target],
                problem.task,
                cfg,
                class_codes,
                totals,
                structured=structured,
            )
        )
        data = forecast(series, frame[target], problem, ceiling)
        if data is not None:
            statistic = reflect(problem, data)
            observed = float(statistic(problem.y[None, :])[0])
            pending.append(
                Pending(
                    str(name),
                    ECHO,
                    observed,
                    statistic,
                    uses_binned=False,
                    data=data,
                )
            )
    found = pairs(frame, names)
    if len(found) > len(names):
        note = (
            f"{len(found)} pairs of two-level columns, more than the "
            f"{len(names)} candidate columns, were not screened jointly: a "
            "target split over two columns may remain undetected"
        )
        return pending, (note,)
    for first, second in found:
        joint = frame[first].astype(str) + " & " + frame[second].astype(str)
        pending.extend(
            gauge(pair(str(first), str(second)), basis(joint), problem)
        )
    return pending, ()


def tested(
    pending: Sequence[Pending], names: Sequence[Hashable]
) -> tuple[str, ...]:
    """Lists the candidate columns that have a hypothesis, in order.

    A pair's members have hypotheses of their own (a two-level column has a
    skill test), so pair names are left out.

    Args:
        pending: The hypotheses of the family.
        names: Candidate columns.

    Returns:
        The column names, each once.
    """
    listed = {str(n) for n in names}
    return tuple(
        dict.fromkeys(h.feature for h in pending if h.feature in listed)
    )


def caps(cfg: leakage.Config) -> dict[str, float]:
    """Looks up the pre-registered ceiling of each mechanism's effect size.

    Args:
        cfg: Settings holding the ceilings.

    Returns:
        A dict from mechanism to ceiling; missingness and record purity
        share the association ceiling.
    """
    # Echo is an association of the column with the target read without any
    # fit (agreement, or unfitted R^2), so it shares the association
    # ceiling: no new threshold.
    return {
        SKILL: cfg.skill_ceiling,
        MISSING: cfg.association_ceiling,
        PURITY: cfg.association_ceiling,
        ECHO: cfg.association_ceiling,
    }


def hypothesize(
    name: str,
    series: pandas.Series,
    outcome: pandas.Series,
    task: str,
    cfg: leakage.Config,
    class_codes: numpy.ndarray,
    totals: numpy.ndarray,
    *,
    structured: bool,
) -> list[Pending]:
    """Builds the record-purity and missingness hypotheses of one column.

    Args:
        name: Column name.
        series: The column.
        outcome: The target column.
        task: `classification` or `regression`.
        cfg: Settings.
        class_codes: Class code of the binned outcome per row.
        totals: Rows per class.
        structured: Whether rows are grouped or in time blocks, which
            replaces the analytic missingness test by a permutation one.

    Returns:
        No hypotheses when the column is all missing, never missing, or
        the outcome has a single class; otherwise the purity hypothesis
        followed by the missingness hypothesis, when one can be formed.
    """
    missing = series.isna().to_numpy()
    if missing.all() or not missing.any() or len(totals) < MIN_CLASSES:
        return []
    recorded = ~missing
    pure = purity_statistic(recorded)
    found = [
        Pending(
            name,
            PURITY,
            float(pure(class_codes[None, :])[0]),
            pure,
            uses_binned=True,
        )
    ]
    if structured:
        stat = cramer(missing, totals)
        observed = float(stat(class_codes[None, :])[0])
        found.append(Pending(name, MISSING, observed, stat, uses_binned=True))
        return found
    test = leakage.missingness_test(series, outcome, task, cfg)
    if test is not None:
        found.append(
            Pending(
                name,
                MISSING,
                test.cramers_v,
                None,
                uses_binned=True,
                p_value=test.p_value,
                active=False,
            )
        )
    return found


def floor(
    h: Pending,
    problem: Problem,
    frame: pandas.DataFrame,
    class_codes: numpy.ndarray,
    totals: numpy.ndarray,
    cfg: leakage.Config,
    groups: numpy.ndarray | None,
) -> float:
    """Bounds the effect of one significant hypothesis from below.

    Args:
        h: The significant hypothesis; a skill or echo hypothesis carries
            its `data`.
        problem: The encoded outcome with its folds.
        frame: The chosen rows.
        class_codes: Class code of the binned outcome per row.
        totals: Rows per class.
        cfg: Settings.
        groups: Group label per row, or None.

    Returns:
        The lower confidence bound of the statistic the mechanism measures.
    """
    path = f"leakage.screen.{h.mechanism}.{h.feature}"
    rng = randomness.component_rng(cfg.run_seed, path, 0)
    if h.mechanism == SKILL and h.data is not None:
        return skill_bound(problem, h.data, cfg, groups, rng)
    if h.mechanism == ECHO and h.data is not None:
        return hedge(problem, h.data, cfg, groups, rng)
    if h.mechanism == PURITY:
        recorded = frame[h.feature].notna().to_numpy()
        return dominance(recorded, class_codes, cfg, groups)
    missing = frame[h.feature].isna().to_numpy()
    return resample(missing, class_codes, len(totals), cfg, groups, rng)
