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

Two checks read the user's declarations rather than the outcome, and stay
outside the family: `dated`, a date column holding values after the
declared prediction time (`Dataset.cutoff`), such as a closing date that
falls later for one outcome; and `declared`, a column declared known only
after the outcome (`Dataset.known`), such as a call's duration. Without the
declaration neither can be told from a legitimate predictor; with it, the
value was not known when the prediction is made.
"""

from collections.abc import Hashable
from collections.abc import Mapping
from collections.abc import Sequence
import math

import numpy
import pandas
from scipy import special
from scipy import stats

from scmle.assurance.admissibility import leakage
from scmle.assurance.admissibility.screen.fitting import Problem
from scmle.assurance.admissibility.screen.hypotheses import Pending
from scmle.assurance.admissibility.screen.report import Finding
from scmle.assurance.admissibility.screen.settings import DATED
from scmle.assurance.admissibility.screen.settings import DECLARED
from scmle.assurance.admissibility.screen.settings import DUPLICATION
from scmle.assurance.admissibility.screen.settings import ENCODING
from scmle.assurance.admissibility.screen.settings import LATE
from scmle.assurance.admissibility.screen.settings import MOMENTS
from scmle.assurance.admissibility.screen.settings import OUTCOME
from scmle.assurance.admissibility.screen.settings import ROWS
from scmle.assurance.detectors import certificates
from scmle.contracts import errors
from scmle.data.validation import dates

# Rows a group needs for its variance: the precision test compares each
# group's mean with that mean's sampling noise, estimated within the group.
PAIR = 2


# For a mixture of Poisson counts, `a_k = k! E[f_k]` is a moment sequence,
# so `a_2^2 <= a_1 a_3` (Cauchy-Schwarz): `E[f_2]^2 <= 1.5 E[f_1] E[f_3]`.
# A fixed number of rows (`n`) makes the counts binomial, which multiplies
# the bound by `(n - 1) / (n - 2)`.
SCHWARZ = 1.5


# Orders sorted together by `runs`: one array of this many orders by the
# rows (at 2,000 rows, 256 orders hold 4 MB per int64 array), so the sorts
# run in numpy rather than one Python call per order, in bounded memory.
SORTS = 256


# Most rows whose codes (each below the row count) fit 16 bits, which numpy
# sorts stably by radix rather than by comparison.
RADIX = 2**16


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
    values: numpy.ndarray, orders: Sequence[numpy.ndarray] | numpy.ndarray
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
        orders: Sort keys from `keys` (a list, or stacked one per row),
            without the column's own.

    Returns:
        A group label per row; None when no order is below the expected
        share.
    """
    codes, levels = pandas.factorize(values)
    shares = numpy.bincount(codes, minlength=len(levels)) / len(values)
    best = 1.0 - float((shares**2).sum())
    # Sorting the rows by value, then stably by the key, orders them as
    # `lexsort((values, key))` does; codes below 2**16 sort by radix.
    ranks = pandas.factorize(values, sort=True)[0]
    first = numpy.argsort(ranks, kind="stable")
    ranked = ranks[first].astype(narrow(len(values)))
    labels = None
    for start in range(0, len(orders), SORTS):
        keys = numpy.asarray(orders[start : start + SORTS])
        keys = keys.astype(narrow(len(values)), copy=False)[:, first]
        then = numpy.argsort(keys, axis=1, kind="stable")
        held = ranked[then]
        changed = held[:, 1:] != held[:, :-1]
        moved = numpy.take_along_axis(keys, then, axis=1)
        moved = moved[:, 1:] != moved[:, :-1]
        found = (numpy.count_nonzero(changed, axis=1) + 1) / (
            numpy.count_nonzero(changed | moved, axis=1) + 1
        )
        index = int(numpy.argmin(found))
        if found[index] < best:
            best = float(found[index])
            steps = numpy.concatenate(([True], changed[index]))
            labels = numpy.empty(len(values), dtype="int64")
            labels[first[then[index]]] = numpy.cumsum(steps) - 1
    return labels


def narrow(rows: int) -> str:
    """The integer type that holds codes below `rows` and sorts fastest.

    Args:
        rows: Rows, an upper bound on every code.

    Returns:
        `uint16` up to `RADIX` rows (sorted by radix), `int64` beyond.
    """
    return "uint16" if rows <= RADIX else "int64"


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

    A column is a candidate when it is numeric and its recorded values lie
    within the outcomes' range (a mean of the outcome cannot leave it); a
    column with gaps is tested on the rows where it is recorded, since an
    encoding that leaves a level blank (a missing source value) is still a
    target mean where it is filled, and a column that never read the
    outcomes stays honest on any rows chosen without them. Its groups
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
    chosen = candidates(frame, names, problem)
    orders = numpy.empty((0, len(frame)), narrow(len(frame)))
    if chosen:
        orders = numpy.stack(keys(frame, names)).astype(narrow(len(frame)))
    found = []
    for index in chosen:
        values = numbers(frame[names[index]])
        held = ~numpy.isnan(values)
        rows = orders if held.all() else orders[:, held]
        found.extend(
            judge(
                str(names[index]),
                values[held],
                groupings(values[held], rows, index),
                [t[held] for t in targets],
                cfg,
            )
        )
    return found


def numbers(series: pandas.Series) -> numpy.ndarray:
    """A numeric column as floats, a missing value as NaN.

    Args:
        series: A numeric column (nullable integers included).

    Returns:
        One float per row.
    """
    return series.to_numpy(dtype="float64", na_value=numpy.nan)


def candidates(
    frame: pandas.DataFrame, names: Sequence[Hashable], problem: Problem
) -> list[int]:
    """The columns the target-mean test reads, without reading the outcome.

    A mean of the outcome cannot leave the outcomes' range, so a candidate
    is a numeric column whose recorded values (at least `PAIR`) lie inside
    it.

    Args:
        frame: The chosen rows.
        names: Candidate columns.
        problem: The encoded target.

    Returns:
        The positions in `names` of the candidate columns, in order.
    """
    targets = outcomes(problem)
    low = min(float(t.min()) for t in targets)
    high = max(float(t.max()) for t in targets)
    found = []
    for index, name in enumerate(names):
        series = frame[name]
        if not pandas.api.types.is_numeric_dtype(series):
            continue
        values = numbers(series)
        values = values[~numpy.isnan(values)]
        if len(values) < PAIR:
            continue
        if values.min() >= low and values.max() <= high:
            found.append(index)
    return found


def groupings(
    values: numpy.ndarray,
    orders: Sequence[numpy.ndarray] | numpy.ndarray,
    index: int,
) -> list[numpy.ndarray]:
    """The groupings one column is tested on, read from the columns alone.

    Args:
        values: The column, as numbers.
        orders: Sort keys from `keys` (a list, or stacked one per row).
        index: The column's own key, left out of `runs`.

    Returns:
        The value's own levels (an encoding of a category no other column
        holds), then its runs along the most telling order, when one is.
    """
    found = [pandas.factorize(values)[0]]
    labels = runs(values, numpy.delete(numpy.asarray(orders), index, axis=0))
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
            binned=False,
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
    by `l2^2 <= c l1 l3` (`c` from `SCHWARZ` and `rows`). The p-value is
    the exact conditional one (`exact`). The effect is the pairs over their
    mean where the likelihood is largest on the boundary: `l1 = f1 + d`,
    `l2 = f2 - 2 d`, `l3 = f3 + d`, `d` the smaller root of
    `(4 - c) d^2 - (4 f2 + c (f1 + f3)) d + f2^2 - c f1 f3 = 0`.

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
    return exact(singles, pairs, triples, c), pairs / (pairs - 2 * shift)


def exact(singles: int, pairs: int, triples: int, c: float) -> float:
    """The exact p-value of the pairs given the singles and the triples.

    The constraint `l2^2 <= c l1 l3` is linear in the Poisson natural
    parameters (`2 log l2 - log l1 - log l3 <= log c`), so conditioning on
    `2 f1 + f2` and `2 f3 + f2` removes the nuisance means and leaves the
    pairs a law with monotone likelihood ratio in that contrast: the tail
    on the boundary is the uniformly most powerful unbiased test, of exact
    size at every count. The half `chi^2_1` of the likelihood ratio it
    replaces rejected 1.9 percent of boundary counts at 0.01 when about
    four triples were expected (`tests/unit/test_duplication_exact.py`).

    Args:
        singles: Records seen once, `f1`.
        pairs: Records seen exactly twice, `f2`.
        triples: Records seen three times, `f3`.
        c: The boundary constant (`SCHWARZ` with the rows' correction).

    Returns:
        The probability of at least `pairs` pairs on the boundary, given
        both sums.
    """
    upper, lower = 2 * singles + pairs, 2 * triples + pairs
    j = numpy.arange(pairs % 2, min(upper, lower) + 1, 2)
    weights = (
        0.5 * j * math.log(c)
        - special.gammaln((upper - j) / 2 + 1)
        - special.gammaln(j + 1)
        - special.gammaln((lower - j) / 2 + 1)
    )
    tail = special.logsumexp(weights[j >= pairs])
    return math.exp(tail - special.logsumexp(weights))


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

    Not tested: a table whose columns' space holds no more records than its
    rows (`enumerated`). A record there is a cell of the space, so a copy on
    both sides of a split is what serving looks like, and a model can learn
    no more from a copied record than from the cell it repeats; the
    sampling bound does not apply to rows that may be laid out, not drawn.

    Returns:
        The hypothesis, named `ROWS`; None with fewer than three rows or
        for a table whose space is no larger than its rows.
    """
    rows = frame[[*names, target]]
    if len(rows) <= PAIR or enumerated(frame, names):
        return None
    hashed = fingerprint(rows)
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
        binned=False,
        p_value=p,
        active=False,
        bound=bound,
    )


def fingerprint(rows: pandas.DataFrame) -> pandas.Series:
    """Hashes each row, so equal rows share a hash.

    Args:
        rows: The rows.

    Returns:
        One hash per row; unhashable cells are hashed by their text.
    """
    try:
        return pandas.util.hash_pandas_object(rows, index=False)
    except TypeError:
        # handled: unhashable cells are counted by their text
        return pandas.util.hash_pandas_object(rows.astype(str), index=False)


def enumerated(frame: pandas.DataFrame, names: Sequence[Hashable]) -> bool:
    """Whether the columns' space holds no more records than the table.

    The product of each column's distinct values is the most distinct
    records the columns can spell. When it is at most the rows, the table
    can lay out its whole space (a designed experiment, or a benchmark's
    truth table with a training sample appended), and a record is a cell of
    that space, not an individual: every input the model will meet is one
    the rows can hold, and a copied row teaches nothing a repeat of its cell
    does not. One column always spans its own values, so at least `PAIR`
    columns must vary; a column of distinct numbers beside another varying
    column spells more records than rows, so a sampled table never is one.

    Args:
        frame: Every row to screen.
        names: Candidate columns.

    Returns:
        True when at least `PAIR` columns vary and the product of their
        distinct values (missing counted as a value) is at most the rows.
    """
    columns = frame[list(names)]
    space, varied = 1, 0
    for name in columns:
        levels = int(fingerprint(columns[[name]]).nunique())
        space *= levels
        varied += levels > 1
        if space > len(columns):
            return False
    return varied >= PAIR


def declared(
    names: Sequence[Hashable], known: Mapping[str, str] | None
) -> list[Finding]:
    """Finds the candidate columns declared known only after the outcome.

    The user's declaration of when a column becomes known (`Dataset.known`)
    decides, as `dated` reads a declared prediction time: a column known
    only once the outcome is cannot be read when the prediction is made, so
    its finding is significant, complete and blocking, with no test and no
    p-value. A column known by the time of the prediction, or not declared,
    raises nothing.

    Args:
        names: Candidate columns.
        known: Column to `prediction` or `outcome` (`MOMENTS`), or None.

    Returns:
        One finding per candidate column declared `outcome`, in column
        order.

    Raises:
        InputError: If a declared moment is not one of `MOMENTS`.
    """
    told = dict(known or {})
    wrong = sorted(k for k, v in told.items() if v not in MOMENTS)
    if wrong:
        raise errors.InputError(
            f"known: columns {wrong} must be declared one of {list(MOMENTS)}"
        )
    return [
        Finding(
            str(name),
            DECLARED,
            1.0,
            p_value=0.0,
            adjusted_p=0.0,
            significant=True,
            lower=1.0,
            suspect=True,
            complete=True,
            gate=certificates.Authority.GATING,
        )
        for name in names
        if told.get(str(name)) == OUTCOME
    ]


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
    time (days from the cutoff) instead. Only date columns are read, a
    datetime column or text the data reader reads as dates (ISO 8601 of one
    layout, `scmle.data.validation.dates.iso`): a number that measures
    something known later (a call's duration) is not a date.

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
            if not dates.iso(series):
                continue
            series = pandas.to_datetime(series)
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
                lower=share,
                suspect=True,
                complete=True,
                gate=certificates.Authority.GATING,
            )
        )
    return found
