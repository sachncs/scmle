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
from scmle.assurance.admissibility.screen.fitting import aucs
from scmle.assurance.admissibility.screen.fitting import basis
from scmle.assurance.admissibility.screen.fitting import line
from scmle.assurance.admissibility.screen.fitting import oof_fit
from scmle.assurance.admissibility.screen.fitting import pair
from scmle.assurance.admissibility.screen.fitting import pairs
from scmle.assurance.admissibility.screen.fitting import predict
from scmle.assurance.admissibility.screen.fitting import Problem
from scmle.assurance.admissibility.screen.fitting import skill_bound
from scmle.assurance.admissibility.screen.fitting import skill_statistic
from scmle.assurance.admissibility.screen.settings import BINARY_CLASSES
from scmle.assurance.admissibility.screen.settings import DUPLICATION
from scmle.assurance.admissibility.screen.settings import ECHO
from scmle.assurance.admissibility.screen.settings import ENCODING
from scmle.assurance.admissibility.screen.settings import MIN_CLASSES
from scmle.assurance.admissibility.screen.settings import MISSING
from scmle.assurance.admissibility.screen.settings import NATURAL
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
        columns: The columns a skill hypothesis reads: one, or the two of
            a pair; empty otherwise.
        bound: The lower confidence bound of an analytic hypothesis whose
            bound comes with its test (`records`); None otherwise.
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
    columns: tuple[str, ...] = ()
    bound: float | None = None


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


def gauge(
    name: str,
    phi: numpy.ndarray,
    problem: Problem,
    columns: tuple[str, ...] = (),
) -> list[Pending]:
    """Builds the skill hypothesis of one design, when it can carry any.

    Args:
        name: Column or pair name.
        phi: The design.
        problem: The encoded target.
        columns: The columns the design reads; `(name,)` when empty.

    Returns:
        The hypothesis with its observed skill, or nothing for a design of
        one column.
    """
    if phi.shape[1] <= 1:
        return []
    statistic = skill_statistic(problem, phi)
    observed = float(statistic(problem.y[None, :])[0])
    return [
        Pending(
            name,
            SKILL,
            observed,
            statistic,
            uses_binned=False,
            data=phi,
            columns=columns or (name,),
        )
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
        both = (str(first), str(second))
        pending.extend(gauge(pair(*both), basis(joint), problem, both))
    return pending, ()


def sift(pending: Sequence[Pending], cfg: leakage.Config) -> int:
    """Stops, before any permutation, each hypothesis that cannot be a suspect.

    A suspect needs a lower confidence bound above its mechanism's ceiling,
    and the bound (a low bootstrap quantile) lies below the observed
    statistic, so a hypothesis whose observed statistic is at or below the
    ceiling draws no permutations: it keeps no permutation count, and the
    step-up rule reads its p-value as 1. Raising p-values to 1 keeps every
    p-value valid (`p' >= p` is super-uniform under its null), so the
    Benjamini-Yekutieli control holds under arbitrary dependence, and the
    step-up rule can only reject fewer of the others. Permutations are then
    spent only where a suspect is possible, which keeps a wide table with
    no strong column inside the time budget instead of declining it.

    Args:
        pending: The hypotheses of the family; changed in place.
        cfg: Settings holding the ceilings.

    Returns:
        How many hypotheses with a permutation statistic stay active.
    """
    ceilings = caps(cfg)
    left = 0
    for h in pending:
        if h.statistic is None:
            continue
        h.active = h.observed > ceilings[h.mechanism]
        left += h.active
    return left


def rival(
    h: Pending,
    pending: Sequence[Pending],
    problem: Problem,
    apart: Sequence[Hashable] = (),
) -> list[numpy.ndarray] | None:
    """Predictions of the other columns, which a skill suspect must outrun.

    A column above the skill ceiling is a suspect only when it also outruns
    the rest of its table (`report.conclude` bounds the paired difference,
    `margin`). The rest is the better, by observed skill, of a cross-fitted
    linear stack of the other columns' single-column smoothers (each
    column's out-of-fold prediction from its skill design, one per class,
    with the same folds) and the best other column alone: a model of the
    rest can always pick one of its members, and on a small table the
    stack's weights cost more than they gain (iris: the stack of the three
    other columns scores about 0.975, petal length alone about 0.99).
    Stacking keeps one weight per column, so a wide table is not judged by a
    fit with more terms than rows. A legitimate measurement of the case has
    company: real tables whose single columns pass the ceiling (breast
    cancer radius, perimeter and area, mushroom odor, cell size and shape
    uniformity, iris petal width and length) are predicted about as well by
    their other columns. A leak reads the outcome, which the measurements of
    the case together do not match. A tie counts, so leaks that each predict
    the target perfectly do not hide one another. Requiring this only
    removes suspects, so the false-alarm control of the significant list is
    kept. Not guaranteed: an imperfect leak written into two columns is
    matched by its twin (the echo still sees a copy on the target's scale).

    Args:
        h: A skill hypothesis.
        pending: The family.
        problem: The encoded outcome with its folds.
        apart: Columns never counted in another column's rest (the
            probe's plants, each judged against the real columns only).

    Returns:
        One array of out-of-fold predictions per scored outcome, as
        `predict` returns; None when no other column has a skill design.
    """
    left = {str(c) for c in apart} | set(h.columns)
    others = [
        (p.observed, p.data)
        for p in pending
        if p.mechanism == SKILL
        and p.data is not None
        and len(p.columns) == 1
        and p.columns[0] not in left
    ]
    if not others:
        return None
    if problem.task == "regression":
        targets = [problem.y.astype("float64")]
    else:
        wanted = (
            [1] if problem.classes == BINARY_CLASSES else range(problem.classes)
        )
        targets = [(problem.y == k).astype("float64") for k in wanted]
    tests = [split[1] for split in problem.splits]
    stacked = []
    scores = []
    for outcome in targets:
        row = outcome[None, :]
        stack = numpy.column_stack(
            [oof_fit(row, part[1], problem.splits)[0] for part in others]
        )
        fitted = oof_fit(row, stack, problem.splits)
        stacked.append(fitted[0])
        if problem.task == "regression":
            squares = float(((outcome - fitted[0]) ** 2).sum())
            scores.append(1.0 - squares / problem.total_squares)
        else:
            scores.append(float(aucs(fitted, row > 0, tests)[0]))
    skill, design = max(others, key=lambda part: part[0])
    if skill > float(numpy.mean(scores)):
        return predict(problem, design)
    return stacked


def reading(
    h: Pending, phi: numpy.ndarray, frame: pandas.DataFrame, problem: Problem
) -> list[numpy.ndarray]:
    """Predictions of a skill suspect, which its rest must not match.

    The better, by observed skill, of the column's smoother and its linear
    reading (`line`), as the rest is the better of a stack and its best
    column (`rival`): a model of one ordered column can always rank by the
    value, and the smoother's ten knots cost a nearly perfect column part of
    its edge. A noisy class code (the binary threshold plant) scores about
    0.993 AUC through its smoother and 0.997 by its rank, which is what
    separates it from a stack of the nine other breast cancer columns
    (about 0.99): read through the smoother, 12 of 200 certificate worlds
    tied with the stack. Both readings are cross-fitted on the same folds,
    so neither reads its own held-out rows. A pair, or a column with few
    values, has the smoother only.

    Args:
        h: A skill hypothesis.
        phi: Its smoother's design (`h.data`).
        frame: The chosen rows.
        problem: The encoded outcome with its folds.

    Returns:
        One array of out-of-fold predictions per scored outcome, as
        `predict` returns.
    """
    smooth = predict(problem, phi)
    named = {str(c): c for c in frame.columns}
    design = None
    if len(h.columns) == 1 and h.columns[0] in named:
        design = line(frame[named[h.columns[0]]])
    if design is None:
        return smooth
    score = float(skill_statistic(problem, design)(problem.y[None, :])[0])
    return predict(problem, design) if score > h.observed else smooth


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
        share the association ceiling, and the record mechanisms have the
        honest limit `NATURAL`.
    """
    # Echo is an association of the column with the target read without any
    # fit (agreement, or unfitted R^2), so it shares the association
    # ceiling: no new threshold.
    return {
        SKILL: cfg.skill_ceiling,
        MISSING: cfg.association_ceiling,
        PURITY: cfg.association_ceiling,
        ECHO: cfg.association_ceiling,
        ENCODING: NATURAL,
        DUPLICATION: NATURAL,
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
    if h.bound is not None:
        return h.bound
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
