"""The ingest plan stage of a charter run.

A charter with an `ingest` block and a data spec that names `ingest_plan` read
their features from a database through the as-of views instead of from the
table alone. The stage runs before the data are split, so nothing it reads
depends on the confirmation set:

1. profile the tables, infer the grain and discover the links under the
   charter's budget (`IngestPlan`); a unit verdict of `coarser` or `unknown`
   stops the run with the questions a person must settle;
2. read each table once through a consistent (snapshot-checked) read and wrap
   it in an `AsOfView` with its declared time rule;
3. synthesize the feature family for the run table's rows, at the cutoff
   `moment - lag`, capped at `B_feat`; the declared family size is fixed here;
4. return the feature columns, the manifest and the provenance table, which
   the caller joins to the table and hands to the admissibility stage.

The plan file (JSON) names the database and the views; the charter fixes the
budget, windows, lag and tolerances, so they are hashed and signed:

    {"database": "warehouse.sqlite",
     "key": "customer_id", "moment": "moment",       # columns of the run table
     "tables": ["customers", "orders"],
     "unit": "customer", "target_table": "customers",
     "declared": [{"child": ..., "child_columns": [...], "parent": ...,
                   "parent_columns": [...]}],
     "views": [{"name": "orders", "table": "orders", "order_by": "order_id",
                "key": "customer_id", "target_key": "customer_id",
                "role": "child", "event_time": "created", "permissive": false,
                "rule": {"mode": "transaction", "known_column": "recorded",
                         "delta_rec_days": 0.0,
                         "attestation": {"name_or_id": ..., ...}}}],
     "nests": [{"mid": 0, "table": "events", "order_by": "order_id, created",
                "key": "order_id", "mid_id": "order_id", ...same view keys}]}

Entity resolution (charter format 8) runs on the run table itself:
`resolve_entities` compares the declared fields, groups records linked above
the lower threshold, and the caller makes the result the split's group column,
so duplicates of one real-world entity under different keys never straddle
development and test. With a `reconcile` block the extracted views are
compared with the totals it declares and with the source database itself,
which is queried for the row count, the declared column sums and the key
checksum of each declared view (`declared.reconcile_source`, built on
`reconcile.reconcile_extract`), before anything is synthesized; a mismatch
stops the run and names each quantity (`declared.require`).
"""

import dataclasses
import json
from pathlib import Path
import re
import tempfile
from typing import Any

import pandas

from scmle.contracts import errors
from scmle.data.admissibility import provenance
from scmle.data.charter import ingest
from scmle.data.ingest import asof_view
from scmle.data.ingest import budget
from scmle.data.ingest import declared
from scmle.data.ingest import entities
from scmle.data.ingest import plan as ingest_plan
from scmle.data.ingest import profile
from scmle.data.ingest import synthesize
from scmle.data.readers import connectors

MANIFEST = "feature_manifest.json"
PROVENANCE = "ingest_field_provenance.json"
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
ORDER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(, ?[A-Za-z_][A-Za-z0-9_]*)*$")
BATCH_ROWS = 1000


@dataclasses.dataclass(frozen=True)
class IngestResult:
    """What the stage produced.

    Attributes:
        features: One column per kept feature, indexed like the run table.
        declared: The declared family size (kept features).
        records: Field provenance of the kept features, for the
            admissibility tests.
        documents: File name to JSON-ready content (the ingest plan's own
            files are written by `IngestPlan`).
        summary: The ingest plan's summary (whether the budget cut the run
            short, source queries, unit verdict, accepted links, open
            questions) plus `declared_family_size` and the number of
            `proposed` features, for the run event.
    """

    features: pandas.DataFrame
    declared: int
    records: tuple[provenance.FieldProvenance, ...]
    documents: dict[str, dict[str, Any]]
    summary: dict[str, Any]


def load_plan(path: str) -> dict[str, Any]:
    """Reads the ingest plan file.

    Args:
        path: Path of the JSON ingest plan.

    Returns:
        The plan document, which holds at least the `database`, `key`,
        `moment`, `tables` and `views` entries.

    Raises:
        ConfigError: If the file cannot be read or parsed, is not a JSON
            object, or lacks a required key.
    """
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise errors.ConfigError(
            f"cannot read ingest plan {path}: {error}"
        ) from error
    needed = ("database", "key", "moment", "tables", "views")
    if not isinstance(document, dict) or any(k not in document for k in needed):
        raise errors.ConfigError(f"ingest plan {path} needs keys {needed}")
    return document


def identifier(value: Any, what: str) -> str:
    """A bare SQL identifier.

    Args:
        value: The candidate name.
        what: What the name is, used in the error message.

    Returns:
        `value` unchanged.

    Raises:
        ConfigError: If `value` is not a string matching the identifier
            pattern.
    """
    if not isinstance(value, str) or not IDENTIFIER.match(value):
        raise errors.ConfigError(f"ingest plan {what} {value!r} is not a name")
    return value


def rule_of(
    record: dict[str, Any], spec: ingest.IngestSpec
) -> asof_view.TimeRule:
    """The time rule a view declares.

    Args:
        record: The view's rule entry: its mode, optional known-time column,
            optional recording delay and optional attestation.
        spec: The charter's ingest settings, whose recording delay applies
            when the entry gives none.

    Returns:
        The time rule, with no delay for a non-versioned view.
    """
    note = record.get("attestation")
    delay = record.get("delta_rec_days", spec.delta_rec)
    return asof_view.TimeRule(
        record["mode"],
        record.get("known_column"),
        None if record["mode"] == asof_view.NON_VERSIONED else float(delay),
        None if note is None else provenance.Attestation(**note),
    )


def read_frame(entry: dict[str, Any], factory: Any) -> pandas.DataFrame:
    """One table, read once with a snapshot check.

    A read that sees the table change on every attempt fails with the
    snapshot check's own `TornReadError`.

    Args:
        entry: The plan entry naming the `table` and its `order_by` clause.
        factory: Opens a database connection.

    Returns:
        The table's rows as read in one consistent snapshot.

    Raises:
        ConfigError: If the table name is not an identifier or `order_by` is
            not a valid ordering.
    """
    table = identifier(entry["table"], "table")
    order = str(entry["order_by"])
    if not ORDER.match(order):
        raise errors.ConfigError(f"ingest plan order_by {order!r} is invalid")
    query = f"SELECT * FROM {table} ORDER BY {order}"  # nosec B608
    connector = connectors.DbApiConnector("sqlite://ingest", query, factory)
    return profile.consistent_read(connector, BATCH_ROWS).frame


def read_view(
    entry: dict[str, Any],
    factory: Any,
    spec: ingest.IngestSpec,
    frame: pandas.DataFrame | None = None,
) -> asof_view.AsOfView:
    """One table as an as-of view; `frame` is the extract if already read.

    Args:
        entry: The plan entry for the view: its table, key, time rule, event
            time column, name and whether it is permissive.
        factory: Opens a database connection.
        spec: The charter's ingest settings.
        frame: The table already extracted, or None to read it.

    Returns:
        The as-of view over the table.
    """
    table = identifier(entry["table"], "table")
    if frame is None:
        frame = read_frame(entry, factory)
    return asof_view.AsOfView(
        frame,
        entry["key"],
        rule_of(entry["rule"], spec),
        entry.get("event_time"),
        entry.get("name", table),
        bool(entry.get("permissive", False)),
    )


def run(
    plan_document: dict[str, Any],
    spec: ingest.IngestSpec,
    frame: pandas.DataFrame,
    run_dir: Path | None,
) -> IngestResult:
    """Runs the ingest plan, then synthesizes features for the rows of `frame`.

    Args:
        plan_document: The loaded ingest plan.
        spec: The charter's ingest settings.
        frame: The target rows; it must hold the ingest plan's key and moment
            columns.
        run_dir: Directory the ingest plan writes its records into, or None
            to use a temporary one that is discarded.

    Returns:
        The synthesized feature frame with its declared family size,
        provenance fields, documents to write and a summary.

    Raises:
        BlockedError: If the unit verdict is `coarser` or `unknown`, or the
            extracted views do not reproduce the declared or source-queried
            totals of the charter's `reconcile` block.
        ConfigError: If `frame` lacks the key or moment column the ingest plan
            names, or a table or ordering it names is not valid.
    """
    key, moment = plan_document["key"], plan_document["moment"]
    missing = [c for c in (key, moment) if c not in frame.columns]
    if missing:
        raise errors.ConfigError(f"ingest plan names columns {missing} absent")
    factory = ingest_plan.sqlite_factory(plan_document["database"])
    settings = ingest_plan.IngestSettings(
        tables=tuple(plan_document["tables"]),
        unit=plan_document.get("unit"),
        target_table=plan_document.get("target_table"),
        eps_key=spec.eps_key,
        eps_null=spec.eps_null,
        eps_fk=spec.eps_fk,
        declared=ingest_plan.declared_from_json(
            plan_document.get("declared", [])
        ),
    )
    limits = budget.IngestBudget(
        spec.max_queries,
        spec.max_rows_scanned,
        spec.max_seconds,
        spec.feature_budget,
    )
    with tempfile.TemporaryDirectory() as scratch:
        folder = run_dir if run_dir is not None else Path(scratch)
        plan = ingest_plan.IngestPlan(folder, factory, settings, limits)
        summary = plan.run_all()
    verdict = plan.verdict
    if verdict is not None and verdict.blocks:
        raise errors.BlockedError(
            f"the charter unit {verdict.unit!r} does not match the grain of "
            f"the target table ({verdict.verdict}): {verdict.reason}",
            {"questions": list(verdict.questions)},
        )
    entries = [*plan_document["views"], *plan_document.get("nests", [])]
    extracts = [read_frame(entry, factory) for entry in entries]
    report = None
    if spec.reconcile is not None:
        frames_by_name = {
            str(entry.get("name", entry["table"])): extract
            for entry, extract in zip(entries, extracts, strict=True)
        }
        tables_by_name = {
            str(entry.get("name", entry["table"])): identifier(
                entry["table"], "table"
            )
            for entry in entries
        }
        report = declared.reconcile(
            frames_by_name,
            spec.reconcile,
            spec.delta_rec,
        )
        report["source_checks"] = declared.reconcile_source(
            frames_by_name,
            tables_by_name,
            spec.reconcile,
            factory,
            spec.delta_rec,
        )
        report["source_passed"] = all(
            c["passed"] for c in report["source_checks"]
        )
        declared.require(report)
    count = len(plan_document["views"])
    views = [
        read_view(v, factory, spec, extract)
        for v, extract in zip(
            plan_document["views"], extracts[:count], strict=True
        )
    ]
    sources = [
        synthesize.Source(
            view,
            entry.get("target_key", entry["key"]),
            entry.get("role", synthesize.CHILD),
        )
        for view, entry in zip(views, plan_document["views"], strict=True)
    ]
    leaves = [
        read_view(n, factory, spec, extract)
        for n, extract in zip(
            plan_document.get("nests", []), extracts[count:], strict=True
        )
    ]
    nests = [
        synthesize.Nest(sources[n["mid"]], leaf, n["mid_id"])
        for leaf, n in zip(leaves, plan_document.get("nests", []), strict=True)
    ]
    targets = frame[[key, moment]].reset_index(drop=True)
    found = synthesize.synthesize(
        targets,
        moment,
        sources,
        nests,
        lag_days=spec.lag,
        windows=spec.windows,
        max_features=spec.feature_budget,
        strict=spec.strict,
    )
    table = synthesize.feature_provenance(found, spec.lag, [*views, *leaves])
    documents = {
        MANIFEST: found.to_dict(),
        PROVENANCE: table.to_dict(),
    }
    if report is not None:
        documents[declared.FILE] = report
    return IngestResult(
        found.frame,
        found.declared_family_size,
        table.fields,
        documents,
        {
            **summary,
            "declared_family_size": found.declared_family_size,
            "proposed": len(found.manifest),
        },
    )


def resolve_entities(
    frame: pandas.DataFrame, spec: ingest.EntitySpec
) -> tuple[pandas.Series, dict[str, Any]]:
    """Group id per row of the run table, from the charter's entity block.

    Without a clerical sample the result is `unverified`: the links serve only
    to keep records of one entity together (link-then-group errs toward
    merging), and no attribute value is ever merged.

    Args:
        frame: The run table.
        spec: Fields and blocking passes the charter fixed.

    Returns:
        The group ids indexed like `frame`, and the `entity_resolution.json`
        document.

    Raises:
        ConfigError: If a declared field or pass column is absent.
    """
    needed = [f.name for f in spec.fields] + [p.field for p in spec.passes]
    missing = sorted({c for c in needed if c not in frame.columns})
    if missing:
        raise errors.ConfigError(
            f"the charter's entity block names columns {missing} that the "
            "run table lacks"
        )
    records = frame.reset_index(drop=True)
    found = entities.resolve(
        records,
        [entities.Field(f.name, f.fuzzy) for f in spec.fields],
        [entities.Pass(p.field, p.kind, p.size) for p in spec.passes],
    )
    groups = pandas.Series(found.groups, index=frame.index, name=spec.column)
    return groups, {**found.to_dict(), "group_column": spec.column}


def check_multiplicity(
    declared: int,
    synthesized: tuple[str, ...],
    registered_columns: set[str],
    excluded: set[str],
) -> None:
    """Fails the run when the ledger did not register the declared family.

    The declared size must equal the number of synthesized columns, every
    synthesized column that was not excluded must be a registered hypothesis
    of the ledger, and no excluded one may be registered.

    Args:
        declared: The declared family size.
        synthesized: Names of the synthesized columns that reached the table.
        registered_columns: Columns registered as hypotheses of the ledger.
        excluded: Synthesized columns left out of the family (quarantined or
            excluded).

    Raises:
        BlockedError: If the declared size differs from the number of
            synthesized columns, or the synthesized columns registered in the
            ledger are not exactly the non-excluded ones.
    """
    if declared != len(synthesized):
        raise errors.BlockedError(
            f"declared family size {declared} but {len(synthesized)} "
            "synthesized columns reached the table"
        )
    expected = {c for c in synthesized if c not in excluded}
    registered = {c for c in synthesized if c in registered_columns}
    if registered != expected:
        raise errors.BlockedError(
            f"the ledger registered {len(registered)} synthesized "
            f"hypotheses but {len(expected)} features were declared and kept",
            {"missing": sorted(expected - registered)},
        )
