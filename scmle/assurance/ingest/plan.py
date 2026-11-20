"""`Plan`: runs the ingestion stages in order and writes their artifacts.

Stages: `profile` (catalog), `infer` (keys and the unit verdict),
`link` (link graph), each under one budget. Artifacts, written
atomically into the run directory: `ingest_catalog.json`, `grain_report.json`,
`join_graph.json`, `ingest_budget.json`. Entity resolution, the as-of views
and synthesis take frames and views and are called from code
(`entities.resolve`, `synthesize.synthesize`); `persist` records
their artifacts.
"""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
import json
import pathlib
import sqlite3
from typing import Any

from scmle import durability
from scmle.assurance.ingest import grain
from scmle.assurance.ingest import joins
from scmle.assurance.ingest import profile
import scmle.assurance.ingest.budget
from scmle.contracts import errors

CATALOG = "ingest_catalog.json"
GRAIN = "grain_report.json"
JOINS = "join_graph.json"
BUDGET = "ingest_budget.json"


@dataclasses.dataclass(frozen=True, slots=True)
class Config:
    """Declared tolerances and scope of one ingestion.

    Attributes:
        tables: Tables to profile, in order.
        unit: Charter unit text, or None to skip the unit verdict.
        target: Table the unit is compared with.
        violation: Largest `g3` accepted as a key.
        null: Largest null rate accepted in a key.
        orphans: Largest share of child values without a parent for a
            candidate link.
        width: Largest key size searched, `L`.
        declared: Foreign keys the schema declares.
    """

    tables: tuple[str, ...]
    unit: str | None = None
    target: str | None = None
    violation: float = 0.0
    null: float = 0.0
    orphans: float = 0.0
    width: int = grain.LEVELS
    declared: tuple[joins.Reference, ...] = ()

    def __post_init__(self) -> None:
        """Validates scope.

        Raises:
            ValueError: If `tables` is empty, or a unit is given and
                `target` is not one of `tables`.
        """
        if not self.tables:
            raise errors.InputError("tables must not be empty")
        if self.unit is not None and self.target not in self.tables:
            raise errors.InputError("target_table must be one of tables")


def persist(
    run_dir: pathlib.Path, name: str, document: dict[str, Any]
) -> pathlib.Path:
    """Writes a JSON artifact atomically.

    Args:
        run_dir: Run directory; created if missing.
        name: File name of the artifact.
        document: The content, written as sorted, indented JSON.

    Returns:
        The path of the written file.
    """
    path = run_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    durability.write(
        path, (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    )
    return path


class Plan:
    """One ingestion over a read-only source, under a budget."""

    def __init__(
        self,
        run_dir: pathlib.Path,
        connection_factory: Callable[[], Any],
        settings: Config,
        budget: scmle.assurance.ingest.budget.Budget | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Opens the ingestion; nothing is read yet.

        Args:
            run_dir: Directory the artifacts are written to.
            connection_factory: Callable returning a DB-API connection to
                the source.
            settings: Tolerances and scope.
            budget: Limits for the run; unlimited when None.
            clock: Wall-time source for the budget meter; the meter's
                default clock when None.
        """
        self.directory = pathlib.Path(run_dir)
        self.settings = settings
        limits = budget or scmle.assurance.ingest.budget.Budget()
        self.meter = (
            scmle.assurance.ingest.budget.Meter(limits)
            if clock is None
            else scmle.assurance.ingest.budget.Meter(limits, clock)
        )
        self.source = profile.Sql(connection_factory, self.meter)
        self.catalog: profile.Catalog | None = None
        self.scans: dict[str, grain.Scan] = {}
        self.verdict: grain.Verdict | None = None
        self.graph: joins.Joins | None = None

    def profile(self) -> profile.Catalog:
        """Profiles the tables and writes `ingest_catalog.json`.

        Returns:
            The catalog of the profiled tables.
        """
        self.catalog = profile.inventory(self.source, self.settings.tables)
        persist(self.directory, CATALOG, self.catalog.to_dict())
        return self.catalog

    def infer(self) -> dict[str, grain.Scan]:
        """Finds keys per table, judges the unit, writes `grain_report.json`.

        Profiles first if that has not been done.

        Returns:
            The key scan of each profiled table, by table name.

        Raises:
            ScmleError: Only as a guard that cannot trigger: a unit is set,
                `target_table` is None and yet `None` is among the scanned
                table names.
        """
        catalog = self.catalog or self.profile()
        self.scans = {}
        for table in catalog.tables:
            self.scans[table.name] = grain.keysearch(
                self.source,
                table,
                self.settings.violation,
                self.settings.null,
                self.settings.width,
            )
        document: dict[str, Any] = {
            "tables": {n: s.to_dict() for n, s in self.scans.items()},
            "partial": catalog.partial
            or any(s.partial for s in self.scans.values()),
            "verdict": None,
        }
        target = self.settings.target
        if self.settings.unit is not None and target in self.scans:
            self.verdict = grain.judge(
                self.source,
                catalog.table(target),
                self.settings.unit,
                self.scans[target],
            )
            document["verdict"] = self.verdict.to_dict()
        elif self.settings.unit is not None:
            self.verdict = grain.Verdict(
                grain.UNKNOWN,
                self.settings.unit,
                (),
                "target table was not profiled (budget or scope)",
                ("raise the ingest budget",),
            )
            document["verdict"] = self.verdict.to_dict()
        persist(self.directory, GRAIN, document)
        return self.scans

    def link(self) -> joins.Joins:
        """Measures links, writes `join_graph.json`.

        Profiles and infers grain first if that has not been done.

        Returns:
            The join graph over the tables that have a key scan.
        """
        catalog = self.catalog or self.profile()
        if not self.scans:
            self.infer()
        keyed = profile.Catalog(
            tuple(t for t in catalog.tables if t.name in self.scans),
            catalog.partial,
        )
        self.graph = joins.discover(
            self.source,
            keyed,
            self.scans,
            self.settings.declared,
            self.settings.orphans,
        )
        persist(self.directory, JOINS, self.graph.to_dict())
        return self.graph

    def execute(self) -> dict[str, Any]:
        """Runs every stage and writes `ingest_budget.json`.

        Returns:
            The summary from `finish`.
        """
        self.profile()
        self.infer()
        self.link()
        return self.finish()

    def finish(self) -> dict[str, Any]:
        """Writes the budget artifact and summarizes the run.

        Returns:
            A mapping with `partial`, the number of source queries, the unit
            verdict (None if none), the number of accepted links and the
            open questions.
        """
        document = self.meter.to_dict()
        document["source_queries"] = self.source.queries
        persist(self.directory, BUDGET, document)
        return {
            "partial": self.meter.partial,
            "source_queries": self.source.queries,
            "verdict": None if self.verdict is None else self.verdict.verdict,
            "accepted_links": 0
            if self.graph is None
            else len(self.graph.accepted()),
            "questions": [] if self.graph is None else self.graph.questions(),
        }


def opener(
    path: str | pathlib.Path,
) -> Callable[[], sqlite3.Connection]:
    """Builds a factory of read-only connections to a SQLite file.

    Args:
        path: The SQLite file.

    Returns:
        A callable that opens a new read-only connection on each call.
    """
    address = f"file:{pathlib.Path(path).resolve()}?mode=ro"

    def connect() -> sqlite3.Connection:
        return sqlite3.connect(address, uri=True)

    return connect


def declarations(
    items: Sequence[dict[str, Any]],
) -> tuple[joins.Reference, ...]:
    """Builds declared keys from `[{child, child_columns, parent, ...}]`.

    Args:
        items: Mappings with `child`, `child_columns`, `parent` and
            `parent_columns`.

    Returns:
        One declared key per mapping, in order.
    """
    return tuple(
        joins.Reference(
            str(i["child"]),
            tuple(i["child_columns"]),
            str(i["parent"]),
            tuple(i["parent_columns"]),
        )
        for i in items
    )
