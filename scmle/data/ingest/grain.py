"""Grain and key inference (F9 section 2).

A level-wise search over column sets of size at most `L` (default 3) for
approximate keys, with the tolerance measure `g3(X)`: the smallest fraction of
rows to delete so that `X` is unique, `(N - D) / N` where `D` is the number of
distinct fully non-null combinations of `X`. Counts come from grouped
queries run in the source, so a key statement is exact for the snapshot read.

Limits, stated again in every report: a key found here is a key of this
snapshot, not a design decision; on a small table every high-cardinality
column looks like a key (`small_n`); ties between equally good keys are not
broken, they are listed as questions. The unit verdict compares the charter
unit with the table only through column names and uniqueness, and says
`unknown` when the names do not settle it.
"""

from collections.abc import Sequence
import dataclasses
import itertools
import re
from typing import Any

from scmle.data.ingest import profile

LEVELS = 3
SMALL_N = 1000
MATCH = "match"
FINER = "finer"
COARSER = "coarser"
UNKNOWN = "unknown"
VERDICTS = (MATCH, FINER, COARSER, UNKNOWN)
ID_WORDS = frozenset({"id", "key", "code", "no", "number", "uuid", "pk"})
STOP_WORDS = frozenset(
    {"the", "a", "an", "of", "each", "one", "row", "level", "unit", "record"}
)
JOINERS = frozenset({"per", "by", "and", "x", "within"})
SPLIT = re.compile(r"[^a-z0-9]+")
CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def stem(word: str) -> str:
    """Strips a plural `s`; nothing cleverer, so matching stays auditable."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def name_tokens(name: str) -> tuple[str, ...]:
    """Lower-case stemmed words of an identifier (snake or camel case)."""
    spaced = CAMEL.sub("_", name).lower()
    return tuple(stem(w) for w in SPLIT.split(spaced) if w)


def unit_components(unit: str) -> tuple[str, ...]:
    """The grain words of a unit text, e.g. `customer-month` gives two."""
    words = [w for w in SPLIT.split(unit.lower()) if w]
    kept = [
        stem(w)
        for w in words
        if w not in STOP_WORDS and w not in JOINERS and not w.isdigit()
    ]
    return tuple(dict.fromkeys(kept))


@dataclasses.dataclass(frozen=True, slots=True)
class KeyCandidate:
    """An approximate key and its measures.

    Attributes:
        columns: Column names in table order.
        g3: Fraction of rows to delete for uniqueness, in [0, 1].
        null_rate: Share of rows with a null in any column.
        distinct: Distinct non-null combinations.
        rows: Rows in the table.
        small_n: True when the table is small enough for accidental keys.
    """

    columns: tuple[str, ...]
    g3: float
    null_rate: float
    distinct: int
    rows: int
    small_n: bool

    def rank(self) -> tuple[Any, ...]:
        """Sort key: fewer columns, then lower g3, then fewer nulls."""
        return (len(self.columns), self.g3, self.null_rate, self.columns)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return {**dataclasses.asdict(self), "columns": list(self.columns)}


@dataclasses.dataclass(slots=True)
class KeyScan:
    """Result of the lattice search over one table.

    Attributes:
        table: Table name.
        rows: Row count.
        keys: Minimal approximate keys, ranked.
        measured: g3 and null rate of every set that was queried or derived.
        levels_done: Largest lattice level that finished.
        partial: True when the budget stopped the search.
        eps_key: Largest g3 accepted as a key.
        eps_null: Largest null rate accepted in a key.
    """

    table: str
    rows: int
    keys: list[KeyCandidate]
    measured: dict[tuple[str, ...], tuple[float, float]]
    levels_done: int
    partial: bool
    eps_key: float
    eps_null: float

    def ties(self) -> list[list[tuple[str, ...]]]:
        """Groups of keys equal in size, g3 and null rate (to be asked)."""
        groups: dict[tuple[Any, ...], list[tuple[str, ...]]] = {}
        for key in self.keys:
            groups.setdefault(key.rank()[:3], []).append(key.columns)
        return [g for g in groups.values() if len(g) > 1]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return {
            "table": self.table,
            "rows": self.rows,
            "keys": [k.to_dict() for k in self.keys],
            "ties": [[list(c) for c in g] for g in self.ties()],
            "levels_done": self.levels_done,
            "partial": self.partial,
            "eps_key": self.eps_key,
            "eps_null": self.eps_null,
        }


def is_key(g3: float, null_rate: float, scan: KeyScan) -> bool:
    """Whether the measures pass the key tolerances of the scan."""
    return g3 <= scan.eps_key and null_rate <= scan.eps_null


def measure(
    source: profile.SqlSource,
    table: str,
    columns: Sequence[str],
    rows: int,
) -> tuple[float, float]:
    """`(g3, null_rate)` of a column set by two grouped queries."""
    names = profile.quote_all(columns)
    present = " AND ".join(f"{profile.quote(c)} IS NOT NULL" for c in columns)
    absent = " OR ".join(f"{profile.quote(c)} IS NULL" for c in columns)
    quoted = profile.quote(table)
    distinct = source.scalar(
        profile.statement(
            "SELECT COUNT(*) FROM (SELECT DISTINCT {n} FROM {t} WHERE {w})",
            n=names,
            t=quoted,
            w=present,
        )
    )
    nulls = source.scalar(
        profile.statement(
            "SELECT COUNT(*) FROM {t} WHERE {w}", t=quoted, w=absent
        )
    )
    if rows == 0:
        return 1.0, 0.0
    return (rows - int(distinct or 0)) / rows, int(nulls or 0) / rows


def bound_ok(
    table: profile.TableProfile, columns: Sequence[str], eps_key: float
) -> bool:
    """Label-free test: could `columns` reach `g3 <= eps_key` at all?

    The distinct count of a set is at most the product of its columns'
    distinct counts and at most the smallest non-null count. When either
    bound is below `N (1 - eps_key)` the set cannot be a key and no query
    is spent on it.
    """
    need = table.rows * (1.0 - eps_key)
    capacity = 1
    for name in columns:
        capacity *= max(table.column(name).distinct, 1)
        if capacity >= need:
            break
    smallest = min(table.column(name).non_null for name in columns)
    return capacity >= need and smallest >= need


def infer_keys(
    source: profile.SqlSource,
    table: profile.TableProfile,
    eps_key: float = 0.0,
    eps_null: float = 0.0,
    max_size: int = LEVELS,
    stage: str = "grain",
) -> KeyScan:
    """Searches the lattice for minimal approximate keys.

    Args:
        source: Read-only source.
        table: Profile of the table (from `profile_table`).
        eps_key: Largest `g3` accepted as a key.
        eps_null: Largest null rate accepted in a key.
        max_size: Largest column set size, `L`.
        stage: Budget stage name.

    Returns:
        The scan. Level 1 needs no query (the profile has the counts); each
        later level is admitted to the budget as a whole before it runs, so a
        stop is always at a level boundary.
    """
    for name, value in (("eps_key", eps_key), ("eps_null", eps_null)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if max_size < 1:
        raise ValueError("max_size must be at least 1")
    scan = KeyScan(table.name, table.rows, [], {}, 0, False, eps_key, eps_null)
    n = table.rows
    usable = [
        c.name for c in table.columns if c.distinct > 1 and c.non_null > 0
    ]
    nonkey: set[tuple[str, ...]] = set()
    for name in usable:
        column = table.column(name)
        g3 = (n - column.distinct) / n if n else 1.0
        nulls = (n - column.non_null) / n if n else 0.0
        scan.measured[(name,)] = (g3, nulls)
        if is_key(g3, nulls, scan):
            scan.keys.append(
                KeyCandidate(
                    (name,), g3, nulls, column.distinct, n, n < SMALL_N
                )
            )
        elif nulls <= eps_null:
            nonkey.add((name,))
    scan.levels_done = 1
    for size in range(2, max_size + 1):
        order = {name: k for k, name in enumerate(usable)}
        candidates = sorted(
            {
                tuple(sorted((*base, extra), key=order.__getitem__))
                for base in nonkey
                for extra in usable
                if extra not in base
            },
            key=lambda cols: [order[c] for c in cols],
        )
        candidates = [
            cols
            for cols in candidates
            if all(
                sub in nonkey for sub in itertools.combinations(cols, size - 1)
            )
        ]
        runnable = [c for c in candidates if bound_ok(table, c, eps_key)]
        budget = source.meter
        if (
            runnable
            and budget is not None
            and not budget.admit(
                stage,
                f"level {size}",
                n * 2 * len(runnable),
                2 * len(runnable),
            )
        ):
            scan.partial = True
            break
        found: set[tuple[str, ...]] = set()
        for cols in candidates:
            if cols in runnable:
                g3, nulls = measure(source, table.name, cols, n)
                scan.measured[cols] = (g3, nulls)
                if is_key(g3, nulls, scan):
                    distinct = n - round(g3 * n)
                    scan.keys.append(
                        KeyCandidate(cols, g3, nulls, distinct, n, n < SMALL_N)
                    )
                    continue
                if nulls > eps_null:
                    continue
            found.add(cols)
        nonkey = found
        scan.levels_done = size
        if not nonkey:
            break
    scan.keys.sort(key=KeyCandidate.rank)
    return scan


@dataclasses.dataclass(frozen=True, slots=True)
class UnitVerdict:
    """Comparison of the charter unit with the table's grain.

    Attributes:
        verdict: `match`, `finer`, `coarser` or `unknown`.
        unit: The unit text.
        unit_columns: Columns the unit text was matched to.
        reason: Why, in one sentence.
        questions: Things a person must settle, if any.
    """

    verdict: str
    unit: str
    unit_columns: tuple[str, ...]
    reason: str
    questions: tuple[str, ...] = ()

    @property
    def blocks(self) -> bool:
        """Whether the run must stop or ask: anything but `match` or `finer`."""
        return self.verdict in (COARSER, UNKNOWN)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return {
            **dataclasses.asdict(self),
            "unit_columns": list(self.unit_columns),
            "questions": list(self.questions),
            "blocks": self.blocks,
        }


def match_columns(
    unit: str, table: profile.TableProfile
) -> tuple[tuple[str, ...], list[str]]:
    """Columns for each unit word, and the questions left open."""
    questions = []
    chosen: list[str] = []
    components = unit_components(unit)
    if not components:
        return (), ["the unit text has no grain words"]
    for word in components:
        hits = [c.name for c in table.columns if word in name_tokens(c.name)]
        if len(hits) > 1:
            ids = [h for h in hits if ID_WORDS & set(name_tokens(h))]
            hits = ids if ids else hits
        if len(hits) != 1:
            kind = "no column" if not hits else f"columns {hits}"
            questions.append(f"which column is {word!r}? found {kind}")
            continue
        chosen.append(hits[0])
    if questions:
        return (), questions
    order = [c.name for c in table.columns]
    return tuple(sorted(set(chosen), key=order.index)), []


def unit_verdict(
    source: profile.SqlSource,
    table: profile.TableProfile,
    unit: str,
    scan: KeyScan,
) -> UnitVerdict:
    """Compares the unit text with the table's measured grain.

    Rules, in order. Unit words that do not each match one column give
    `unknown`. If a strict subset of the unit columns is already a key, the
    rows are coarser than the unit: `coarser`. Else if the unit columns are a
    key: `match`. Else if some key was found: `finer` (aggregation to the unit
    is needed). Else `unknown`. A table that is `coarser` is never reported as
    `match`, whatever the tolerances: subsets are checked first.
    """
    columns, questions = match_columns(unit, table)
    if not columns:
        return UnitVerdict(
            UNKNOWN,
            unit,
            (),
            "unit words do not map to columns",
            tuple(questions),
        )

    def key_like(cols: tuple[str, ...]) -> bool | None:
        if cols not in scan.measured:
            if not bound_ok(table, cols, scan.eps_key):
                return False
            budget = source.meter
            if budget is not None and not budget.admit(
                "unit", ",".join(cols), table.rows * 2, 2
            ):
                return None
            scan.measured[cols] = measure(source, table.name, cols, table.rows)
        g3, nulls = scan.measured[cols]
        return is_key(g3, nulls, scan)

    for size in range(1, len(columns)):
        for subset in itertools.combinations(columns, size):
            state = key_like(subset)
            if state is None:
                return UnitVerdict(
                    UNKNOWN,
                    unit,
                    columns,
                    "budget stopped the unit check",
                    ("raise the ingest budget",),
                )
            if state:
                return UnitVerdict(
                    COARSER,
                    unit,
                    columns,
                    f"{list(subset)} is already a key, so rows are coarser "
                    "than the unit",
                )
    state = key_like(columns)
    if state is None:
        return UnitVerdict(
            UNKNOWN,
            unit,
            columns,
            "budget stopped the unit check",
            ("raise the ingest budget",),
        )
    if state:
        return UnitVerdict(MATCH, unit, columns, "unit columns are a key")
    if scan.keys:
        return UnitVerdict(
            FINER,
            unit,
            columns,
            f"unit columns repeat; the table is keyed by "
            f"{list(scan.keys[0].columns)}, so aggregation is needed",
        )
    return UnitVerdict(
        UNKNOWN,
        unit,
        columns,
        "no key found at the declared tolerance",
        ("which columns identify a row?",),
    )
