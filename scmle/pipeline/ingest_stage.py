"""The ingest plan stage of a charter run.

A charter with an `ingest` block and a data spec that names `ingest_plan` read
their features from a database through the as-of views instead of from the
table alone. The stage runs before the data are split, so nothing it reads
depends on the confirmation set:

1. profile the tables, infer the grain and discover the links under the
   charter's budget (`Plan`); a unit verdict of `coarser` or `unknown`
   stops the run with the questions a person must settle;
2. read each table once through a consistent (snapshot-checked) read and wrap
   it in an `View` with its declared time rule;
3. synthesize the feature family for the run table's rows, at the cutoff
   `moment - lag`, capped at `B_feat`; the declared family size is fixed here;
4. compute every kept feature again with the relational engine
   (`scmle.data.relational.engine`) from the same extracts: a feature that
   differs beyond `AGREEMENT` stops the run, and the summary's `engine`
   entry names each feature the engine has no primitive or rule for
   (`std`, `nunique`, `time_since_last`, depth-2 nests, a `non_versioned`
   view with an event time);
5. return the feature columns, the manifest and the provenance table, which
   the caller joins to the table and hands to the admissibility stage.

The plan file (JSON) names the table store and the views; the charter fixes
the budget, windows, lag and tolerances, so they are hashed and signed:

    {"database": "warehouse.sqlite",     # or "store": {"kind", "path"}
     "key": "customer_id", "moment": "moment",       # columns of the run table
     "tables": ["customers", "orders"],
     "unit": "customer", "target_table": "customers",
     "declared": [{"child": ..., "child_columns": [...], "parent": ...,
                   "parent_columns": [...]}],
     "views": [{"name": "orders", "table": "orders", "order_by": "order_id",
                "key": "customer_id", "target_key": "customer_id",
                "role": "child", "event_time": "created", "permissive": false,
                "cardinality": "one_to_many",
                "rule": {"mode": "transaction", "known_column": "recorded",
                         "delta_rec_days": 0.0,
                         "attestation": {"name_or_id": ..., ...}}}],
     "nests": [{"mid": 0, "table": "events", "order_by": "order_id, created",
                "key": "order_id", "mid_id": "order_id", ...same view keys}]}

The tables live in a SQLite database (`database`, or a `store` of kind
`sqlite`), a Parquet folder (`parquet`) or a CSV set (`csv`); see
`scmle.data.relational.stores`. A folder is mirrored once into a SQLite file
in a scratch directory for the run, its row counts checked, and every read,
profile and source query below runs on that mirror; the source queries of a
`reconcile` block then check the extract against the mirror, not the files.
A view may declare its `cardinality`: `one_to_many` is the `child` role,
`many_to_one` the `parent` role, and a `many_to_one` table whose rows repeat
a (key, event time, recorded time) value stops the run before anything is
synthesized (`scmle.data.relational.schema`).

Entity resolution (charter format 8) runs on the run table itself:
`resolve` compares the declared fields, groups records linked above
the lower threshold, and the caller makes the result the split's group column,
so duplicates of one real-world entity under different keys never straddle
development and test. With a `reconcile` block the extracted views are
compared with the totals it declares and with the source database itself,
which is queried for the row count, the declared column sums and the key
checksum of each declared view (`declared.survey`, built on
`reconcile.crosscheck`), before anything is synthesized; a mismatch
stops the run and names each quantity (`declared.require`).
"""

import dataclasses
import json
from pathlib import Path
import re
import tempfile
from typing import Any

import pandas

from scmle.assurance.admissibility import provenance
from scmle.assurance.charter import ingest
from scmle.assurance.ingest import asof_view
from scmle.assurance.ingest import budget
from scmle.assurance.ingest import declared
from scmle.assurance.ingest import entities
from scmle.assurance.ingest import profile
from scmle.assurance.ingest import synthesize
import scmle.assurance.ingest.plan
from scmle.contracts import errors
from scmle.contracts import records
from scmle.data.readers import connectors
from scmle.data.relational import engine
from scmle.data.relational import schema
from scmle.data.relational import stores

MANIFEST = "feature_manifest.json"
PROVENANCE = "ingest_field_provenance.json"
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
ORDER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(, ?[A-Za-z_][A-Za-z0-9_]*)*$")
BATCH_ROWS = 1000
# The view role each declared cardinality stands for.
ROLES = {schema.MANY: synthesize.CHILD, schema.ONE: synthesize.PARENT}
# The engine primitive that computes each synthesizer primitive; `std`,
# `nunique` and `time_since_last` have none. A lookup is `last` over every
# visible row, which is the synthesizer's definition of it.
PRIMITIVES = {
    synthesize.COUNT: engine.COUNT,
    synthesize.SUM: engine.SUM,
    synthesize.MEAN: engine.MEAN,
    synthesize.MIN: engine.MIN,
    synthesize.MAX: engine.MAX,
    synthesize.LAST: engine.LAST,
    synthesize.SLOPE: engine.TREND,
}
# Largest relative (and absolute) difference allowed between a synthesized
# feature and the engine's. Both sum the same float64 values in different
# orders, which agree to about n * 2.2e-16 relative for n rows per target
# (1e-12 at a million rows); a slope divides two such sums, which at most
# squares the condition of its time spread. 1e-8 keeps four orders of
# headroom above that, and is the reference tolerance (`reference_rel_tol`)
# the C3 evidence holds statistics to.
AGREEMENT = 1e-8


@dataclasses.dataclass(frozen=True)
class Output:
    """What the stage produced.

    Attributes:
        features: One column per kept feature, indexed like the run table.
        declared: The declared family size (kept features).
        records: Field provenance of the kept features, for the
            admissibility tests.
        documents: File name to JSON-ready content (the ingest plan's own
            files are written by `Plan`).
        summary: The ingest plan's summary (whether the budget cut the run
            short, source queries, unit verdict, accepted links, open
            questions) plus `declared_family_size` and the number of
            `proposed` features, for the run event.
    """

    features: pandas.DataFrame
    declared: int
    records: tuple[provenance.Pedigree, ...]
    documents: dict[str, dict[str, Any]]
    summary: dict[str, Any]


def recipe(path: str) -> dict[str, Any]:
    """Reads the ingest plan file.

    Args:
        path: Path of the JSON ingest plan.

    Returns:
        The plan document, which holds at least the `key`, `moment`,
        `tables` and `views` entries and a `database` or a `store`.

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
    needed = ("key", "moment", "tables", "views")
    if not isinstance(document, dict) or any(k not in document for k in needed):
        raise errors.ConfigError(f"ingest plan {path} needs keys {needed}")
    if ("database" in document) == ("store" in document):
        raise errors.ConfigError(
            f"ingest plan {path} needs one of `database` and `store`"
        )
    return document


def locate(plan_document: records.Layout) -> stores.Store:
    """The table store a plan names.

    Args:
        plan_document: The loaded ingest plan: a `database` path, or a
            `store` object with a `path` and an optional `kind` (one of
            `stores.KINDS`; recognised from the path when absent).

    Returns:
        The store.

    Raises:
        ConfigError: If the plan names no store, or a malformed or absent
            one.
    """
    if "database" in plan_document:
        return stores.Database(plan_document["database"])
    entry = plan_document.get("store")
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise errors.ConfigError("ingest plan store needs a `path`")
    return stores.locate(entry["path"], entry.get("kind"))


def role(entry: records.Layout) -> str:
    """The view role an entry states or its cardinality implies.

    Args:
        entry: A plan view entry.

    Returns:
        `child` or `parent`; `child` when the entry states neither.

    Raises:
        ConfigError: If the cardinality is unknown or contradicts the role.
    """
    cardinality = entry.get("cardinality")
    if cardinality is None:
        return str(entry.get("role", synthesize.CHILD))
    if cardinality not in ROLES:
        raise errors.ConfigError(
            f"ingest plan cardinality {cardinality!r} is not one of "
            f"{sorted(ROLES)}"
        )
    if entry.get("role", ROLES[cardinality]) != ROLES[cardinality]:
        raise errors.ConfigError(
            f"ingest plan view {entry.get('name', entry['table'])!r} is "
            f"{cardinality} but has role {entry['role']!r}"
        )
    return ROLES[cardinality]


def hold(
    entries: list[records.Layout], extracts: list[pandas.DataFrame]
) -> None:
    """Holds each extract to the cardinality its entry declares.

    Args:
        entries: The plan's view entries.
        extracts: The table extracted for each.

    Raises:
        ConfigError: If an entry's cardinality is malformed.
        SchemaError: If a declared column is absent, or a `many_to_one`
            table repeats a (key, event time, recorded time) value.
    """
    for entry, extract in zip(entries, extracts, strict=True):
        if "cardinality" not in entry:
            continue
        role(entry)
        name = str(entry.get("name", entry["table"]))
        rule = entry.get("rule", {})
        timed = rule.get("mode") != asof_view.NON_VERSIONED
        layout = schema.Schema(
            (
                schema.Table(
                    name,
                    entry.get("event_time") if timed else None,
                    rule.get("known_column") if timed else None,
                ),
            ),
            (
                schema.Link(
                    name,
                    entry["key"],
                    str(entry.get("target_key", entry["key"])),
                    str(entry["cardinality"]),
                ),
            ),
        )
        layout.check({name: extract})


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


def decode(record: dict[str, Any], spec: ingest.Ingestion) -> asof_view.Timing:
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
    delay = record.get("delta_rec_days", spec.reconciliation)
    return asof_view.Timing(
        record["mode"],
        record.get("known_column"),
        None if record["mode"] == asof_view.NON_VERSIONED else float(delay),
        None if note is None else provenance.Attestation.from_dict(note),
    )


def read(entry: dict[str, Any], factory: Any) -> pandas.DataFrame:
    """One table, read once with a snapshot check.

    A read that sees the table change on every attempt fails with the
    snapshot check's own `TornError`.

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
    query = f"SELECT * FROM {table} ORDER BY {order}"  # noqa: S608 -- identifier and ORDER-matched clause only # nosec B608
    connector = connectors.DbApiConnector("sqlite://ingest", query, factory)
    return profile.untorn(connector, BATCH_ROWS).frame


def load(
    entry: dict[str, Any],
    factory: Any,
    spec: ingest.Ingestion,
    frame: pandas.DataFrame | None = None,
) -> asof_view.View:
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
        frame = read(entry, factory)
    return asof_view.View(
        frame,
        entry["key"],
        decode(entry["rule"], spec),
        entry.get("event_time"),
        entry.get("name", table),
        bool(entry.get("permissive", False)),
    )


def configure(
    plan_document: records.Layout, spec: ingest.Ingestion
) -> tuple[scmle.assurance.ingest.plan.Config, budget.Budget]:
    """The ingest plan's settings and query budget.

    Args:
        plan_document: The loaded ingest plan.
        spec: The charter's ingest settings.

    Returns:
        The plan settings and the query, scan, time and feature budget.
    """
    settings = scmle.assurance.ingest.plan.Config(
        tables=tuple(plan_document["tables"]),
        unit=plan_document.get("unit"),
        target=plan_document.get("target_table"),
        violation=spec.tolerances.key,
        null=spec.tolerances.null,
        orphans=spec.tolerances.link,
        declared=scmle.assurance.ingest.plan.declarations(
            plan_document.get("declared", [])
        ),
    )
    limits = budget.Budget(
        spec.queries,
        spec.scanned,
        spec.seconds,
        spec.features,
    )
    return settings, limits


def reconcile(
    entries: list[records.Layout],
    extracts: list[pandas.DataFrame],
    spec: ingest.Ingestion,
    expectation: ingest.Expectation,
    factory: Any,
) -> records.Layout:
    """Checks the extracts against the charter's declared and source totals.

    Args:
        entries: The plan's view and nest entries.
        extracts: The table extracted for each entry.
        spec: The charter's ingest settings.
        expectation: Their `reconcile` block.
        factory: Opens a database connection.

    Returns:
        The reconciliation report, with the source-queried checks.
    """
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
        expectation,
        spec.reconciliation,
    )
    report["source_checks"] = declared.survey(
        frames_by_name,
        tables_by_name,
        expectation,
        factory,
        spec.reconciliation,
    )
    report["source_passed"] = all(c["passed"] for c in report["source_checks"])
    return report


def assemble(
    plan_document: records.Layout,
    factory: Any,
    spec: ingest.Ingestion,
    extracts: list[pandas.DataFrame],
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    """Builds the as-of views, sources, leaves and nests of the plan.

    Args:
        plan_document: The loaded ingest plan.
        factory: Opens a database connection.
        spec: The charter's ingest settings.
        extracts: The table extracted for each view, then each nest.

    Returns:
        The views, their synthesis sources, the nest leaves and the nests.
    """
    count = len(plan_document["views"])
    views = [
        load(v, factory, spec, extract)
        for v, extract in zip(
            plan_document["views"], extracts[:count], strict=True
        )
    ]
    sources = [
        synthesize.Source(
            view,
            entry.get("target_key", entry["key"]),
            role(entry),
        )
        for view, entry in zip(views, plan_document["views"], strict=True)
    ]
    leaves = [
        load(n, factory, spec, extract)
        for n, extract in zip(
            plan_document.get("nests", []), extracts[count:], strict=True
        )
    ]
    nests = [
        synthesize.Nest(sources[n["mid"]], leaf, n["mid_id"])
        for leaf, n in zip(leaves, plan_document.get("nests", []), strict=True)
    ]
    return views, sources, leaves, nests


def timing(view: asof_view.View) -> schema.Table | None:
    """The engine's declaration of a view's table and time rule.

    Args:
        view: The as-of view.

    Returns:
        The table, or None when the engine has no rule like the view's: a
        `non_versioned` view with an event time (the synthesizer windows and
        orders its rows by that time but shows them at every cutoff, while
        the engine's static table has no times) or a versioned view without
        a recording delay.
    """
    rule = view.rule
    if rule.mode == asof_view.NON_VERSIONED:
        return None if view.time else schema.Table(view.name)
    if rule.delay is None:
        return None
    known = rule.recorded if rule.mode == asof_view.TRANSACTION else None
    return schema.Table(view.name, view.time, known, rule.delay)


def reason(
    feature: synthesize.Feature, tables: list[schema.Table | None]
) -> str | None:
    """Why the engine cannot recompute a synthesized feature.

    Args:
        feature: The synthesized feature.
        tables: The engine's declaration of each source, from `timing`.

    Returns:
        The reason, or None when the engine computes the same feature.
    """
    if feature.kind == "nested":
        return "the engine has no depth-2 nest"
    if feature.primitive not in PRIMITIVES:
        return f"the engine has no {feature.primitive}"
    if tables[feature.source] is None:
        return "the engine has no time rule like its view's"
    return None


def confirm(
    found: synthesize.Selection,
    targets: pandas.DataFrame,
    moment: str,
    sources: list[synthesize.Source],
    nests: list[synthesize.Nest],
    extracts: list[pandas.DataFrame],
    spec: ingest.Ingestion,
) -> dict[str, Any]:
    """Recomputes the kept features with the relational engine.

    Every kept feature the engine can compute (`reason`) is computed again
    by `engine.Engine.features` from the same extracts, at the same cutoffs,
    and must equal the synthesizer's within `AGREEMENT`.

    Args:
        found: The synthesis result.
        targets: The target rows the features are for.
        moment: Their moment column.
        sources: The synthesis sources, one per plan view.
        nests: The depth-2 routes.
        extracts: The table extracted for each source.
        spec: The charter's ingest settings.

    Returns:
        The number of features checked, each one not checked with the
        reason, and the tolerance.

    Raises:
        DataError: If a feature differs from the engine's.
        InputError: If a target row has no moment.
    """
    tables = [timing(s.view) for s in sources]
    kept = synthesize.propose(sources, nests, spec.windows)
    kept = [f for f in kept if f.name in found.frame.columns]
    unchecked = {f.name: r for f in kept if (r := reason(f, tables))}
    checked = [f for f in kept if f.name not in unchecked]
    aggregates = [
        engine.Aggregate(
            sources[f.source].view.name,
            PRIMITIVES[f.primitive],
            f.value,
            f.window,
        )
        for f in checked
    ]
    named = zip(tables, sources, extracts, strict=True)
    known = [(t, s, x) for t, s, x in named if t is not None]
    layout = schema.Schema(
        tuple(t for t, s, x in known),
        tuple(schema.Link(t.name, s.view.key, s.anchor) for t, s, x in known),
    )
    built = engine.Engine(layout, {t.name: x for t, s, x in known})
    computed = built.features(targets, moment, aggregates, spec.lag)
    differ = [
        f.name
        for f, a in zip(checked, aggregates, strict=True)
        if not engine.agree(found.frame[f.name], computed[a.name], AGREEMENT)
    ]
    if differ:
        raise errors.DataError(
            f"{len(differ)} synthesized features differ from the relational "
            f"engine's beyond {AGREEMENT:g}: {differ}"
        )
    return {
        "checked": len(checked),
        "unchecked": unchecked,
        "tolerance": AGREEMENT,
    }


def deliver(
    found: Any, table: Any, report: records.Layout | None, summary: Any
) -> Output:
    """Packs the synthesized features with the documents that describe them.

    Args:
        found: The synthesis result.
        table: Its field provenance.
        report: The reconciliation report, or None.
        summary: The ingest plan's summary.

    Returns:
        The ingest output.
    """
    documents = {
        MANIFEST: found.to_dict(),
        PROVENANCE: table.to_dict(),
    }
    if report is not None:
        documents[declared.FILE] = report
    return Output(
        found.frame,
        found.family,
        table.fields,
        documents,
        {
            **summary,
            "declared_family_size": found.family,
            "proposed": len(found.manifest),
        },
    )


def run(
    plan_document: records.Layout,
    spec: ingest.Ingestion,
    frame: pandas.DataFrame,
    run_dir: Path | None,
) -> Output:
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
            names, the store it names is absent or malformed, or a table,
            ordering or cardinality it names is not valid.
        SchemaError: If a view declared `many_to_one` repeats a version.
        DataError: If a synthesized feature differs from the relational
            engine's.
        InputError: If a row of `frame` has no moment.
    """
    key, moment = plan_document["key"], plan_document["moment"]
    missing = [c for c in (key, moment) if c not in frame.columns]
    if missing:
        raise errors.ConfigError(f"ingest plan names columns {missing} absent")
    store = locate(plan_document)
    with tempfile.TemporaryDirectory() as mirror:
        return synthesis(
            plan_document, spec, frame, store.factory(mirror), run_dir
        )


def synthesis(
    plan_document: records.Layout,
    spec: ingest.Ingestion,
    frame: pandas.DataFrame,
    factory: Any,
    run_dir: Path | None,
) -> Output:
    """Plans, extracts, checks and synthesizes over one table store.

    Args:
        plan_document: The loaded ingest plan.
        spec: The charter's ingest settings.
        frame: The target rows, holding the plan's key and moment columns.
        factory: Opens a read-only connection to the store (or its mirror).
        run_dir: Directory the ingest plan writes its records into, or None.

    Returns:
        The stage output, as `run` describes it.
    """
    key, moment = plan_document["key"], plan_document["moment"]
    settings, limits = configure(plan_document, spec)
    with tempfile.TemporaryDirectory() as scratch:
        folder = run_dir if run_dir is not None else Path(scratch)
        plan = scmle.assurance.ingest.plan.Plan(
            folder, factory, settings, limits
        )
        summary = plan.execute()
    verdict = plan.verdict
    if verdict is not None and verdict.blocks:
        raise errors.BlockedError(
            f"the charter unit {verdict.unit!r} does not match the grain of "
            f"the target table ({verdict.verdict}): {verdict.reason}",
            {"questions": list(verdict.questions)},
        )
    entries = [*plan_document["views"], *plan_document.get("nests", [])]
    extracts = [read(entry, factory) for entry in entries]
    hold(entries, extracts)
    report = None
    if spec.reconcile is not None:
        report = reconcile(entries, extracts, spec, spec.reconcile, factory)
        declared.require(report)
    views, sources, leaves, nests = assemble(
        plan_document, factory, spec, extracts
    )
    targets = frame[[key, moment]].reset_index(drop=True)
    found = synthesize.synthesize(
        targets,
        moment,
        sources,
        nests,
        lag_days=spec.lag,
        windows=spec.windows,
        max_features=spec.features,
        strict=spec.strict,
    )
    count = len(plan_document["views"])
    checks = confirm(
        found, targets, moment, sources, nests, extracts[:count], spec
    )
    table = synthesize.lineage(found, spec.lag, [*views, *leaves])
    return deliver(found, table, report, {**summary, "engine": checks})


def resolve(
    frame: pandas.DataFrame, spec: ingest.Entity
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
