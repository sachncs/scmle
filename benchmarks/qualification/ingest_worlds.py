"""Planted worlds for the F9 ingestion checks.

Each generator builds a small source with a known truth: a mutation applied
mid-read, a join that duplicates or drops rows, late-arriving rows, a schema
with a known grain, surrogate-key tables that include each other by chance,
duplicate people under typos. The truth is set by construction, so these
worlds measure mechanisms. They say nothing about how often real warehouses
look like them, and they are not an external referent.

Seeds. Development seeds start at `DEV_SEED_BASE`; certification uses seeds
from 0. A generator draws only from `randomness.component_rng(seed, name, k)`.
"""

from collections.abc import Callable
import dataclasses
import itertools
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from scmle.core import randomness
from scmle.data.ingest import profile
from scmle.data.readers import connectors

DEV_SEED_BASE = 10_000
URI = "sqlite://ingest/world"


@dataclasses.dataclass
class Database:
    """An in-memory SQLite database that read connections can share."""

    keeper: sqlite3.Connection
    address: str

    def factory(self) -> sqlite3.Connection:
        """A fresh connection to the same database."""
        return sqlite3.connect(self.address, uri=True)

    def source(self) -> profile.SqlSource:
        """A read-only `SqlSource` over the database."""
        return profile.SqlSource(self.factory)

    def execute(
        self, sql: str, rows: list[tuple[Any, ...]] | None = None
    ) -> None:
        """Runs one write on the keeper connection and commits."""
        if rows is None:
            self.keeper.execute(sql)
        else:
            self.keeper.executemany(sql, rows)
        self.keeper.commit()

    def frame(self, sql: str) -> pd.DataFrame:
        """Result of a query as a frame."""
        return pd.read_sql_query(sql, self.keeper)


COUNTER = itertools.count()


def new_database(tag: str) -> Database:
    """A fresh named shared-cache in-memory database."""
    name = f"{tag}{next(COUNTER)}"
    address = f"file:{name}?mode=memory&cache=shared"
    keeper = sqlite3.connect(address, uri=True)
    return Database(keeper, address)


def load(db: Database, table: str, frame: pd.DataFrame) -> None:
    """Writes a frame as a table."""
    frame.to_sql(table, db.keeper, index=False, if_exists="replace")
    db.keeper.commit()


# Torn reads ---------------------------------------------------------------

TORN_KINDS = ("update_behind", "update_ahead", "insert", "delete")


class MutatingConnector(connectors.DbApiConnector):
    """A connector that runs one mutation just before a chosen batch read."""

    def __init__(
        self,
        db: Database,
        query: str,
        mutate: Callable[[], None] | None,
        at_call: int,
    ) -> None:
        """Binds the mutation and the batch call that triggers it."""
        super().__init__(URI, query, db.factory)
        self.mutate = mutate
        self.at_call = at_call
        self.calls = 0

    def read_rows(
        self,
        checkpoint: connectors.SourceCheckpoint | None = None,
        limit: int = 1_000,
    ) -> connectors.RowBatch:
        """Reads a batch, mutating the source first on the chosen call."""
        self.calls += 1
        if self.mutate is not None and self.calls == self.at_call:
            self.mutate()
            self.mutate = None
        return super().read_rows(checkpoint, limit)


def torn_world(
    seed: int, kind: str | None
) -> tuple[MutatingConnector, Database]:
    """A table read in batches of 5, mutated before batch 2 if `kind` is set."""
    rng = randomness.component_rng(seed, "ingest.world.torn", 0)
    rows = int(rng.integers(20, 80))
    db = new_database("torn")
    load(
        db,
        "t",
        pd.DataFrame({"id": np.arange(rows), "v": rng.integers(0, 1000, rows)}),
    )
    pick = int(rng.integers(0, rows // 2 if kind == "update_behind" else rows))
    if kind == "update_ahead":
        pick = int(rng.integers(rows // 2, rows))
    mutations: dict[str, Callable[[], None]] = {
        "update_behind": lambda: db.execute(
            f"UPDATE t SET v = v + 1 WHERE id = {pick}"
        ),
        "update_ahead": lambda: db.execute(
            f"UPDATE t SET v = v + 1 WHERE id = {pick}"
        ),
        "insert": lambda: db.execute(
            "INSERT INTO t VALUES (?, ?)", [(rows + 1, 7)]
        ),
        "delete": lambda: db.execute(f"DELETE FROM t WHERE id = {pick}"),
    }
    mutate = None if kind is None else mutations[kind]
    connector = MutatingConnector(
        db, "SELECT id, v FROM t ORDER BY id", mutate, at_call=2
    )
    return connector, db


# Reconciliation -----------------------------------------------------------

EXTRACTS = ("clean", "fanout", "dropped_rows", "torn_read")


@dataclasses.dataclass
class Extract:
    """An assembled table, its source, and the joins that produced it."""

    db: Database
    assembled: pd.DataFrame
    left_rows: int
    kind: str


def reconcile_world(seed: int, kind: str) -> Extract:
    """Builds orders and customers, then extracts them correctly or not."""
    rng = randomness.component_rng(seed, f"ingest.world.reconcile.{kind}", 0)
    customers = int(rng.integers(30, 80))
    orders = int(rng.integers(120, 400))
    db = new_database("rec")
    load(
        db,
        "customers",
        pd.DataFrame(
            {
                "id": np.arange(customers),
                "region": rng.integers(0, 5, customers),
            }
        ),
    )
    owner = rng.integers(0, customers + 12, orders)  # some orders are orphans
    load(
        db,
        "orders",
        pd.DataFrame(
            {
                "id": np.arange(orders),
                "customer_id": owner,
                "amount": rng.integers(1, 500, orders).astype(float),
            }
        ),
    )
    tag_rows = [
        (int(c), f"t{k}")
        for c in range(customers)
        for k in range(int(rng.integers(1, 4)))
    ]
    db.execute("CREATE TABLE tags (customer_id INTEGER, tag TEXT)")
    db.execute("INSERT INTO tags VALUES (?, ?)", tag_rows)
    base = "SELECT o.id, o.customer_id, o.amount FROM orders o"
    if kind == "fanout":
        sql = base + " JOIN tags g ON g.customer_id = o.customer_id"
    elif kind == "dropped_rows":
        sql = base + " JOIN customers c ON c.id = o.customer_id"
    else:
        sql = base + " LEFT JOIN customers c ON c.id = o.customer_id"
    assembled = db.frame(sql)
    if kind == "torn_read":
        how = int(rng.integers(0, 3))
        if how == 0:
            db.execute(
                "INSERT INTO orders VALUES (?, ?, ?)",
                [(orders + k, 0, 10.0) for k in range(int(rng.integers(1, 6)))],
            )
        elif how == 1:
            db.execute(
                f"DELETE FROM orders WHERE id < {int(rng.integers(1, 6))}"
            )
        else:
            db.execute("UPDATE orders SET amount = amount + 1 WHERE id = 0")
    return Extract(db, assembled, orders, kind)


# Late arrivals ------------------------------------------------------------


@dataclasses.dataclass
class LateWorld:
    """Targets, a child table with recorded times, and the leak counts."""

    targets: pd.DataFrame
    child: pd.DataFrame
    lag_days: float


def late_world(seed: int) -> LateWorld:
    """Events whose recording time lags, with late arrivals and backfills."""
    rng = randomness.component_rng(seed, "ingest.world.late", 0)
    entities = int(rng.integers(5, 15))
    targets = pd.DataFrame(
        {
            "cid": np.arange(entities),
            "moment": rng.uniform(40, 90, entities).round(1),
        }
    )
    rows = int(rng.integers(100, 300))
    event = rng.uniform(0, 100, rows).round(1)
    delay = rng.exponential(1.0, rows)
    late = rng.random(rows) < 0.2
    delay[late] += rng.uniform(5, 60, int(late.sum()))
    backfill = rng.random(rows) < 0.05
    event[backfill] = rng.uniform(0, 30, int(backfill.sum())).round(1)
    delay[backfill] = rng.uniform(40, 120, int(backfill.sum()))
    child = pd.DataFrame(
        {
            "cid": rng.integers(0, entities, rows),
            "event_time": event,
            "recorded": (event + delay).round(1),
            "amount": rng.normal(size=rows),
        }
    )
    return LateWorld(targets, child, float(rng.choice([0.0, 1.0, 2.5])))


# Grain --------------------------------------------------------------------

UNITS = ("customer", "customer-month")
SHAPES = ("customers", "customer_months", "orders")


def expected_verdict(unit: str, shape: str) -> str:
    """Verdict that follows from the construction of the table."""
    if unit == "customer":
        return "match" if shape == "customers" else "finer"
    if shape == "customers":
        return "coarser"
    return "match" if shape == "customer_months" else "finer"


def grain_world(seed: int, unit: str, shape: str) -> tuple[Database, str]:
    """A table whose grain is `shape`, with decoy columns, for a unit text."""
    rng = randomness.component_rng(seed, f"ingest.world.grain.{shape}", 0)
    customers = int(rng.integers(60, 200))
    months = int(rng.integers(3, 8))
    db = new_database("grain")
    if shape == "customers":
        frame = pd.DataFrame(
            {
                "customer_id": rng.permutation(customers),
                "segment": rng.integers(0, 4, customers),
                "signup_month": rng.integers(0, months, customers),
                "spend": rng.normal(size=customers),
            }
        )
    elif shape == "customer_months":
        pairs = [(c, m) for c in range(customers) for m in range(months)]
        frame = pd.DataFrame(pairs, columns=["customer_id", "month"])
        frame["spend"] = rng.normal(size=len(frame))
    else:
        count = customers * months * 3
        frame = pd.DataFrame(
            {
                "order_id": rng.permutation(count),
                "customer_id": rng.integers(0, customers, count),
                "month": rng.integers(0, months, count),
                "spend": rng.normal(size=count),
            }
        )
    load(db, "t", frame)
    return db, "t"


# Joins on surrogate keys --------------------------------------------------


@dataclasses.dataclass
class JoinWorld:
    """Tables, and the foreign keys that are real."""

    db: Database
    tables: list[str]
    truth: set[tuple[str, str, str]]


def surrogate_world(seed: int) -> JoinWorld:
    """Tables with integer keys `1..n`, real FKs, decoys and twin parents."""
    rng = randomness.component_rng(seed, "ingest.world.surrogate", 0)
    db = new_database("sur")
    names = ["customer", "product", "store", "region", "account", "user"]
    sizes = {name: int(rng.integers(40, 200)) for name in names}
    truth: set[tuple[str, str, str]] = set()
    tables = []
    for name in names:
        frame = pd.DataFrame({"id": np.arange(1, sizes[name] + 1)})
        frame["rank"] = rng.integers(1, 30, len(frame))
        frame["level"] = rng.integers(1, 10, len(frame))
        if name in ("account", "user"):
            frame["label"] = [f"l{k}" for k in range(len(frame))]
        load(db, name, frame)
        tables.append(name)
    fact = int(rng.integers(150, 400))
    frame = pd.DataFrame({"id": np.arange(1, fact + 1)})
    for parent in ("customer", "product", "store"):
        if rng.random() < 0.8:
            frame[f"{parent}_id"] = rng.integers(1, sizes[parent] + 1, fact)
            truth.add(("fact", f"{parent}_id", parent))
    frame["user_id"] = rng.integers(
        1, min(sizes["user"], sizes["account"]) + 1, fact
    )
    truth.add(("fact", "user_id", "user"))
    frame["owner_id"] = rng.integers(
        1, min(sizes["user"], sizes["account"]) + 1, fact
    )
    frame["qty"] = rng.integers(1, 20, fact)
    frame["score"] = rng.integers(1, 9, fact)
    load(db, "fact", frame)
    tables.append("fact")
    return JoinWorld(db, tables, truth)


# Entity resolution --------------------------------------------------------

FIRST = [
    "james",
    "mary",
    "john",
    "patricia",
    "robert",
    "jennifer",
    "michael",
    "linda",
    "william",
    "elizabeth",
    "david",
    "barbara",
    "richard",
    "susan",
    "joseph",
    "jessica",
    "thomas",
    "sarah",
    "charles",
    "karen",
    "daniel",
    "nancy",
    "matthew",
    "lisa",
    "anthony",
    "betty",
    "mark",
    "margaret",
    "donald",
    "sandra",
]
LAST = [
    "smith",
    "johnson",
    "williams",
    "brown",
    "jones",
    "garcia",
    "miller",
    "davis",
    "rodriguez",
    "martinez",
    "hernandez",
    "lopez",
    "gonzalez",
    "wilson",
    "anderson",
    "thomas",
    "taylor",
    "moore",
    "jackson",
    "martin",
    "lee",
    "perez",
    "thompson",
    "white",
    "harris",
    "sanchez",
    "clark",
    "ramirez",
]


def typo(text: str, rng: np.random.Generator, rate: float) -> str:
    """Drops, swaps or doubles one character with probability `rate`."""
    if len(text) < 4 or rng.random() >= rate:
        return text
    k = int(rng.integers(1, len(text) - 1))
    choice = int(rng.integers(0, 3))
    if choice == 0:
        return text[:k] + text[k + 1 :]
    if choice == 1:
        return text[:k] + text[k + 1] + text[k] + text[k + 2 :]
    return text[:k] + text[k] + text[k:]


def er_world(seed: int, entities: int = 300) -> pd.DataFrame:
    """People with one or two records, corrupted by typos and gaps.

    Synthetic and Febrl-like in spirit only: the corruption is ours, so the
    numbers do not transfer to a real registry.
    """
    rng = randomness.component_rng(seed, "ingest.world.er", 0)
    rows = []
    for entity in range(entities):
        first = str(rng.choice(FIRST)) + str(rng.choice(FIRST))[:2]
        last = str(rng.choice(LAST)) + str(rng.choice(LAST))[:3]
        year, month, day = (
            rng.integers(1950, 2005),
            rng.integers(1, 13),
            rng.integers(1, 29),
        )
        dob = f"{year}-{month:02d}-{day:02d}"
        addr = f"{rng.integers(1, 999)} {rng.choice(LAST)} street"
        for copy in range(1 + int(rng.random() < 0.5)):
            del copy
            rows.append(
                (
                    entity,
                    typo(first, rng, 0.3),
                    typo(last, rng, 0.3),
                    dob if rng.random() > 0.1 else "",
                    typo(addr, rng, 0.3),
                )
            )
    return pd.DataFrame(
        rows, columns=["entity", "first", "last", "dob", "addr"]
    )
