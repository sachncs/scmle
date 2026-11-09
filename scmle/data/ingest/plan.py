"""`IngestPlan`: runs the ingestion stages in order and writes their artifacts.

Stages: `profile` (catalog), `infer_grain` (keys and the unit verdict),
`discover_joins` (link graph), each under one budget. Artifacts, written
atomically into the run directory: `ingest_catalog.json`, `grain_report.json`,
`join_graph.json`, `ingest_budget.json`. Entity resolution, the as-of views
and synthesis take frames and views and are called from code
(`entities.resolve`, `synthesize.synthesize`); `write_artifact` records
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
from scmle.contracts import errors
from scmle.data.ingest import grain
from scmle.data.ingest import joins
from scmle.data.ingest import profile
import scmle.data.ingest.budget

CATALOG = "ingest_catalog.json"
GRAIN = "grain_report.json"
JOINS = "join_graph.json"
BUDGET = "ingest_budget.json"


@dataclasses.dataclass(frozen=True, slots=True)
class IngestSettings:
    """Declared tolerances and scope of one ingestion.

    Attributes:
        tables: Tables to profile, in order.
        unit: Charter unit text, or None to skip the unit verdict.
        target_table: Table the unit is compared with.
        eps_key: Largest `g3` accepted as a key.
        eps_null: Largest null rate accepted in a key.
        eps_fk: Largest share of child values without a parent for a
            candidate link.
        max_key_size: Largest key size searched, `L`.
        declared: Foreign keys the schema declares.
    """

    tables: tuple[str, ...]
    unit: str | None = None
    target_table: str | None = None
    eps_key: float = 0.0
    eps_null: float = 0.0
    eps_fk: float = 0.0
    max_key_size: int = grain.LEVELS
    declared: tuple[joins.DeclaredKey, ...] = ()

    def __post_init__(self) -> None:
        """Validates scope.

        Raises:
            ValueError: If `tables` is empty, or a unit is given and
                `target_table` is not one of `tables`.
        """
        if not self.tables:
            raise ValueError("tables must not be empty")
        if self.unit is not None and self.target_table not in self.tables:
            raise ValueError("target_table must be one of tables")


def write_artifact(
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
    durability.write_atomically(
        path, (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    )
    return path


class IngestPlan:
    """One ingestion over a read-only source, under a budget."""

    def __init__(
        self,
        run_dir: pathlib.Path,
        connection_factory: Callable[[], Any],
        settings: IngestSettings,
        budget: scmle.data.ingest.budget.IngestBudget | None = None,
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
        self.run_dir = pathlib.Path(run_dir)
        self.settings = settings
        limits = budget or scmle.data.ingest.budget.IngestBudget()
        self.meter = (
            scmle.data.ingest.budget.BudgetMeter(limits)
            if clock is None
            else scmle.data.ingest.budget.BudgetMeter(limits, clock)
        )
        self.source = profile.SqlSource(connection_factory, self.meter)
        self.catalog: profile.Catalog | None = None
        self.scans: dict[str, grain.KeyScan] = {}
        self.verdict: grain.UnitVerdict | None = None
        self.graph: joins.JoinGraph | None = None

    def run_profile(self) -> profile.Catalog:
        """Profiles the tables and writes `ingest_catalog.json`.

        Returns:
            The catalog of the profiled tables.
        """
        self.catalog = profile.profile_catalog(
            self.source, self.settings.tables
        )
        write_artifact(self.run_dir, CATALOG, self.catalog.to_dict())
        return self.catalog

    def infer_grain(self) -> dict[str, grain.KeyScan]:
        """Finds keys per table, judges the unit, writes `grain_report.json`.

        Profiles first if that has not been done.

        Returns:
            The key scan of each profiled table, by table name.

        Raises:
            ScmleError: Only as a guard that cannot trigger: a unit is set,
                `target_table` is None and yet `None` is among the scanned
                table names.
        """
        catalog = self.catalog or self.run_profile()
        self.scans = {}
        for table in catalog.tables:
            self.scans[table.name] = grain.infer_keys(
                self.source,
                table,
                self.settings.eps_key,
                self.settings.eps_null,
                self.settings.max_key_size,
            )
        document: dict[str, Any] = {
            "tables": {n: s.to_dict() for n, s in self.scans.items()},
            "partial": catalog.partial
            or any(s.partial for s in self.scans.values()),
            "verdict": None,
        }
        target = self.settings.target_table
        if self.settings.unit is not None and target in self.scans:
            if target is None:
                raise errors.ScmleError("target table is not set")
            self.verdict = grain.unit_verdict(
                self.source,
                catalog.table(target),
                self.settings.unit,
                self.scans[target],
            )
            document["verdict"] = self.verdict.to_dict()
        elif self.settings.unit is not None:
            self.verdict = grain.UnitVerdict(
                grain.UNKNOWN,
                self.settings.unit,
                (),
                "target table was not profiled (budget or scope)",
                ("raise the ingest budget",),
            )
            document["verdict"] = self.verdict.to_dict()
        write_artifact(self.run_dir, GRAIN, document)
        return self.scans

    def discover_joins(self) -> joins.JoinGraph:
        """Measures links, writes `join_graph.json`.

        Profiles and infers grain first if that has not been done.

        Returns:
            The join graph over the tables that have a key scan.
        """
        catalog = self.catalog or self.run_profile()
        if not self.scans:
            self.infer_grain()
        keyed = profile.Catalog(
            tuple(t for t in catalog.tables if t.name in self.scans),
            catalog.partial,
        )
        self.graph = joins.discover(
            self.source,
            keyed,
            self.scans,
            self.settings.declared,
            self.settings.eps_fk,
        )
        write_artifact(self.run_dir, JOINS, self.graph.to_dict())
        return self.graph

    def run_all(self) -> dict[str, Any]:
        """Runs every stage and writes `ingest_budget.json`.

        Returns:
            The summary from `finish`.
        """
        self.run_profile()
        self.infer_grain()
        self.discover_joins()
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
        write_artifact(self.run_dir, BUDGET, document)
        return {
            "partial": self.meter.partial,
            "source_queries": self.source.queries,
            "verdict": None if self.verdict is None else self.verdict.verdict,
            "accepted_links": 0
            if self.graph is None
            else len(self.graph.accepted()),
            "questions": [] if self.graph is None else self.graph.questions(),
        }


def sqlite_factory(
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


def declared_from_json(
    items: Sequence[dict[str, Any]],
) -> tuple[joins.DeclaredKey, ...]:
    """Builds declared keys from `[{child, child_columns, parent, ...}]`.

    Args:
        items: Mappings with `child`, `child_columns`, `parent` and
            `parent_columns`.

    Returns:
        One declared key per mapping, in order.
    """
    return tuple(
        joins.DeclaredKey(
            str(i["child"]),
            tuple(i["child_columns"]),
            str(i["parent"]),
            tuple(i["parent_columns"]),
        )
        for i in items
    )
