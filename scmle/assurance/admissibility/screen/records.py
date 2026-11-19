"""Leaks that live in the records rather than in one column's skill.

Two hypotheses join the screen's Benjamini-Yekutieli family, each with an
analytic p-value and a bound against a ceiling no honest table can pass
(`settings.NATURAL`):

* `encodings`: a column constant within groups of rows that holds the
  group's mean of the target computed on those rows (a target encoding fitted
  on all rows). Its single-column skill is modest (about 0.75 AUC over five
  rows a level), but it predicts its group's mean more precisely than that
  mean's own sampling noise allows a column that never read the outcomes.
* `duplication`: whole rows copied, so copies straddle the split. Measured
  against the table's own natural duplicate rate: a sample from any
  population holds no more exact pairs than its singletons and triples allow.

A third check reads the user's declaration rather than the outcome, and
stays outside the family: `dated`, a date column holding values after the
declared prediction time (`Dataset.cutoff`), such as a closing date that
falls later for one outcome. Without the declaration a closing date cannot
be told from a legitimate trend; with it, a value after the cutoff was not
known when the prediction is made.
"""

from collections.abc import Hashable
from collections.abc import Sequence
import math

import numpy
import pandas
from scipy import stats

from scmle.assurance.admissibility import leakage
from scmle.assurance.admissibility.screen.fitting import Problem
from scmle.assurance.admissibility.screen.hypotheses import Pending
from scmle.assurance.admissibility.screen.report import Finding
from scmle.assurance.admissibility.screen.settings import DATED
from scmle.assurance.admissibility.screen.settings import DUPLICATION
from scmle.assurance.admissibility.screen.settings import ENCODING
from scmle.assurance.admissibility.screen.settings import LATE
from scmle.assurance.admissibility.screen.settings import ROWS
from scmle.assurance.detectors import certificates
from scmle.contracts import errors

# Rows a group needs for its variance: the precision test compares each
# group's mean with that mean's sampling noise, estimated within the group.
PAIR = 2


# For a mixture of Poisson counts, `a_k = k! E[f_k]` is a moment sequence,
# so `a_2^2 <= a_1 a_3` (Cauchy-Schwarz): `E[f_2]^2 <= 1.5 E[f_1] E[f_3]`.
# A fixed number of rows (`n`) makes the counts binomial, which multiplies
# the bound by `(n - 1) / (n - 2)`.
SCHWARZ = 1.5


def outcomes(problem: Problem) -> list[numpy.ndarray]:
    """The outcomes a target mean can encode.

    Args:
        problem: The encoded target.

    Returns:
        The target for a regression; the indicator of each class otherwise
        (both classes of a binary target, since an encoding may count
        either).
    """
    if problem.task == "regression":
        return [problem.y.astype("float64")]
    return [(problem.y == k).astype("float64") for k in range(problem.classes)]


def keys(
    frame: pandas.DataFrame, names: Sequence[Hashable]
) -> list[numpy.ndarray]:
    """Sort keys the groups of a target mean are sought along.

    Read from the columns alone, never the target: each candidate column's
    sorted codes, and the row order (rows of one entity are often adjacent).

    Args:
        frame: The chosen rows.
        names: Candidate columns.

    Returns:
        One integer array per candidate column, then the row order.
    """
    found = []
    for name in names:
        series = frame[name]
        try:
            codes = pandas.factorize(series, sort=True)[0]
        except TypeError:
            # handled: mixed types sort as text
            codes = pandas.factorize(series.astype(str), sort=True)[0]
        found.append(codes.astype("int64"))
    found.append(numpy.arange(len(frame), dtype="int64"))
    return found


def runs(
    values: numpy.ndarray, orders: Sequence[numpy.ndarray]
) -> numpy.ndarray | None:
    """Groups the rows into runs of one value along the most telling order.

    An encoding of bins of a number (or of a category) is constant along
    that column's order, so its runs there are a small share of the
    distinct (key, value) pairs, even where neighbouring bins share a value
    and the value's own levels merge them; along an order unrelated to the
    value a neighbour differs with chance `1 - sum p_v^2` (`p_v` the share
    of level `v`), the share expected. The order with the smallest share
    below that is used; rows tied on the key are ordered by the value, so a
    bin cut through tied keys stays whole. The choice reads no outcome, so
    the test on the groups stays valid.

    Args:
        values: The column, as numbers.
        orders: Sort keys from `keys`, without the column's own.

    Returns:
        A group label per row; None when no order is below the expected
        share.
    """
    codes, levels = pandas.factorize(values)
    shares = numpy.bincount(codes, minlength=len(levels)) / len(values)
    best = 1.0 - float((shares**2).sum())
    labels = None
    for key in orders:
        order = numpy.lexsort((values, key))
        held = values[order]
        changed = numpy.r_[True, held[1:] != held[:-1]]
        moved = numpy.r_[True, key[order][1:] != key[order][:-1]]
        share = changed.sum() / (changed | moved).sum()
        if share < best:
            best = share
            labels = numpy.empty(len(values), dtype="int64")
            labels[order] = numpy.cumsum(changed) - 1
    return labels


def precision(
    labels: numpy.ndarray, outcome: numpy.ndarray, values: numpy.ndarray
) -> tuple[float, int, float] | None:
    """Compares a column with its groups' means and their sampling noise.

    With `ybar_g` the mean of the outcome over group `g` of `n_g` rows and
    `v_g` the column's value there, `S = sum n_g (ybar_g - v_g)^2` has
    expectation at least `sum sigma_g^2` for any column that did not read
    these outcomes (the mean's bias adds to its noise), and so does
    `V = sum s_g^2`, the groups' unbiased variances. `R = S / V` is about
    `F(G, nu)` at the honest limit (an oracle of each group's true mean),
    `nu` by Satterthwaite; an encoding computed on the rows has `R` near the
    share of its group left out of these rows.

    Args:
        labels: Group label per row.
        outcome: The outcome per row.
        values: The column per row.

    Returns:
        `R`, the groups of at least `PAIR` rows `G`, and `nu`; None with
        fewer than two such groups or no variance within them.
    """
    sizes = numpy.bincount(labels).astype("float64")
    sums = numpy.bincount(labels, outcome)
    squares = numpy.bincount(labels, outcome * outcome)
    level = numpy.bincount(labels, values) / sizes
    used = sizes >= PAIR
    if used.sum() < PAIR:
        return None
    size = sizes[used]
    mean = sums[used] / size
    spread = numpy.maximum((squares[used] - size * mean * mean) / (size - 1), 0)
    noise = float(spread.sum())
    if noise <= 0.0:
        return None
    gap = float((size * (mean - level[used]) ** 2).sum())
    freedom = noise**2 / float((spread**2 / (size - 1)).sum())
    return gap / noise, int(used.sum()), freedom


def encodings(
    frame: pandas.DataFrame,
    names: Sequence[Hashable],
    problem: Problem,
    cfg: leakage.Config,
) -> list[Pending]:
    """Tests every numeric column on the target's scale as a target mean.

    A column is a candidate when it has no missing values and lies within
    the outcomes' range (a mean of the outcome cannot leave it). Its groups
    come from `groupings`; the p-value is the smallest lower tail of
    `F(G, nu)` over the groupings and outcomes, times their number `K`
    (Bonferroni, `judge`). The effect is the precision `1 / R`, and its
    bound `F^-1((1 - confidence) / K) / R`; the ceiling is 1 (`NATURAL`).

    Not guaranteed: the variance within a group also holds the spread of its
    rows' true means, so an honest column that is a calibrated rate over
    large mixed groups (a rate from other data) can look precise; the
    certificates measure that on real tables.

    Args:
        frame: The chosen rows.
        names: Candidate columns.
        problem: The encoded target.
        cfg: Settings (`confidence`).

    Returns:
        One hypothesis per candidate column that has groups to test.
    """
    targets = outcomes(problem)
    low = min(float(t.min()) for t in targets)
    high = max(float(t.max()) for t in targets)
    orders = keys(frame, names)
    found = []
    for index, name in enumerate(names):
        series = frame[name]
        numeric = pandas.api.types.is_numeric_dtype(series)
        if not numeric or series.isna().any():
            continue
        values = series.to_numpy(dtype="float64")
        if values.min() < low or values.max() > high:
            continue
        found.extend(
            judge(
                str(name),
                values,
                groupings(values, orders, index),
                targets,
                cfg,
            )
        )
    return found


def groupings(
    values: numpy.ndarray, orders: Sequence[numpy.ndarray], index: int
) -> list[numpy.ndarray]:
    """The groupings one column is tested on, read from the columns alone.

    Args:
        values: The column, as numbers.
        orders: Sort keys from `keys`.
        index: The column's own key, left out of `runs`.

    Returns:
        The value's own levels (an encoding of a category no other column
        holds), then its runs along the most telling order, when one is.
    """
    found = [pandas.factorize(values)[0]]
    labels = runs(values, [*orders[:index], *orders[index + 1 :]])
    return found if labels is None else [*found, labels]


def judge(
    name: str,
    values: numpy.ndarray,
    groups: Sequence[numpy.ndarray],
    targets: Sequence[numpy.ndarray],
    cfg: leakage.Config,
) -> list[Pending]:
    """The target-mean hypothesis of one column, Bonferroni over its tests.

    Args:
        name: Column name.
        values: The column, as numbers.
        groups: Groupings from `groupings`.
        targets: Outcomes from `outcomes`.
        cfg: Settings (`confidence`).

    Returns:
        One hypothesis, or none when no grouping has two groups with spread.
    """
    tests = [precision(g, t, values) for g in groups for t in targets]
    valid = [t for t in tests if t is not None]
    if not valid:
        return []
    chances = [float(stats.f.cdf(*t)) for t in valid]
    ratio, count, freedom = valid[int(numpy.argmin(chances))]
    share = len(valid)
    p = min(1.0, share * min(chances))
    quantile = float(stats.f.ppf((1 - cfg.confidence) / share, count, freedom))
    return [
        Pending(
            name,
            ENCODING,
            math.inf if ratio == 0.0 else 1.0 / ratio,
            None,
            uses_binned=False,
            p_value=p,
            active=False,
            bound=math.inf if ratio == 0.0 else quantile / ratio,
        )
    ]


def excess(
    singles: int, pairs: int, triples: int, rows: int
) -> tuple[float, float]:
    """Tests the pairs against the most a sample from a population holds.

    The counts are taken as Poisson with means `l1, l2, l3`, constrained
    by `l2^2 <= c l1 l3` (`c` from `SCHWARZ` and `rows`). On the boundary
    the likelihood is largest at `l1 = f1 + d`, `l2 = f2 - 2 d`,
    `l3 = f3 + d`, `d` the smaller root of
    `(4 - c) d^2 - (4 f2 + c (f1 + f3)) d + f2^2 - c f1 f3 = 0`; the
    likelihood ratio is chi-bar-squared, half a `chi^2_1` (one inequality).

    Args:
        singles: Records seen once, `f1`.
        pairs: Records seen exactly twice, `f2`.
        triples: Records seen three times, `f3`.
        rows: Rows counted.

    Returns:
        The p-value, and the pairs over their constrained mean (1 when the
        counts keep the bound).
    """
    c = SCHWARZ * (rows - 1) / (rows - 2)
    if pairs * pairs <= c * singles * triples:
        return 1.0, 1.0
    a = 4.0 - c
    b = 4.0 * pairs + c * (singles + triples)
    shift = (
        b - math.sqrt(b * b - 4 * a * (pairs**2 - c * singles * triples))
    ) / (2 * a)
    means = (singles + shift, pairs - 2 * shift, triples + shift)
    counts = (singles, pairs, triples)
    deviance = 2.0 * sum(
        f * math.log(f / m) for f, m in zip(counts, means, strict=True) if f > 0
    )
    p = 0.5 * float(stats.chi2.sf(max(deviance, 0.0), 1))
    return p, pairs / means[1]


def duplication(
    frame: pandas.DataFrame,
    names: Sequence[Hashable],
    target: str,
    cfg: leakage.Config,
) -> Pending | None:
    """Tests whether whole rows were copied, against the table's own rate.

    Rows drawn from a population repeat naturally, however many duplicates
    that makes, as a mixture of Poisson counts; copying a share of rows
    makes far more exact pairs than its singletons and triples allow
    (`excess`). Every row given is counted, not a subsample, since the test
    needs no fit. The bound is the lower `confidence` limit of the pairs
    over the upper limits of `sqrt(c l1 l3)` (exact Poisson limits).

    Args:
        frame: Every row to screen, with the target.
        names: Candidate columns.
        target: Target column.
        cfg: Settings (`confidence`).

    Returns:
        The hypothesis, named `ROWS`; None with fewer than three rows.
    """
    rows = frame[[*names, target]]
    if len(rows) <= PAIR:
        return None
    try:
        hashed = pandas.util.hash_pandas_object(rows, index=False)
    except TypeError:
        # handled: unhashable cells are counted by their text
        hashed = pandas.util.hash_pandas_object(rows.astype(str), index=False)
    counts = numpy.bincount(hashed.value_counts().to_numpy(), minlength=4)
    singles, pairs, triples = (int(counts[k]) for k in (1, 2, 3))
    p, ratio = excess(singles, pairs, triples, len(rows))
    c = SCHWARZ * (len(rows) - 1) / (len(rows) - 2)
    level = 1.0 - cfg.confidence
    least = stats.chi2.ppf(level, 2 * pairs) / 2 if pairs else 0.0
    most = [
        stats.chi2.ppf(cfg.confidence, 2 * f + 2) / 2
        for f in (singles, triples)
    ]
    bound = float(least) / math.sqrt(c * most[0] * most[1])
    return Pending(
        ROWS,
        DUPLICATION,
        ratio,
        None,
        uses_binned=False,
        p_value=p,
        active=False,
        bound=bound,
    )


def dated(
    frame: pandas.DataFrame,
    names: Sequence[Hashable],
    cutoff: str | None,
) -> list[Finding]:
    """Finds the date columns holding values after the prediction time.

    A record of the past holds no value after the moment it is read, so a
    date column with any value after `cutoff` holds something not known when
    the prediction is made: a closing date, a settlement, a later visit. The
    share of such values is the statistic, against the ceiling `LATE` (zero).
    There is no test and no p-value: the declaration decides, so a finding
    is significant, complete and blocking. A column of dates scheduled in
    advance (a due date) is late by this reading and must be given as a lead
    time (days from the cutoff) instead. Only date columns are read: a
    number that measures something known later (a call's duration) is not
    a date.

    Args:
        frame: Every row to screen.
        names: Candidate columns.
        cutoff: The declared prediction time (any text `pandas.Timestamp`
            reads), or None when none was declared.

    Returns:
        One finding per late column, in column order; none without a
        cutoff.

    Raises:
        InputError: If `cutoff` is not a moment.
    """
    if cutoff is None:
        return []
    try:
        moment = pandas.Timestamp(cutoff)
    except ValueError as error:
        raise errors.InputError(
            f"cutoff {cutoff!r} is not a date or time: {error}"
        ) from error
    if pandas.isna(moment):
        raise errors.InputError(f"cutoff {cutoff!r} is not a date or time")
    found = []
    for name in names:
        series = frame[name]
        if not pandas.api.types.is_datetime64_any_dtype(series):
            continue
        zone = series.dt.tz
        if zone is not None:
            stamp = (
                moment.tz_localize(zone)
                if moment.tz is None
                else moment.tz_convert(zone)
            )
        else:
            stamp = moment if moment.tz is None else moment.tz_localize(None)
        share = float((series > stamp).mean())
        if share <= LATE:
            continue
        found.append(
            Finding(
                str(name),
                DATED,
                share,
                p_value=0.0,
                adjusted_p=0.0,
                significant=True,
                lower_bound=share,
                suspect=True,
                complete=True,
                gate=certificates.Authority.GATING,
            )
        )
    return found
