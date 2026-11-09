# Governed data

How data is read, checked and processed before training and inference
(`scmle.data.store`). What is not covered is listed at the end.

## Object stores

`ObjectStore` is bytes under keys: `put` (atomic replace; `if_absent=True`
creates only), `get`, `head`, `list_objects`, `delete`, `exists`. Metadata
always carries a SHA-256 computed by this code, so a changed object raises
`CorruptObjectError`. The package ships the interface and one backend:

| Backend | Class | Client |
| --- | --- | --- |
| Directory | `LocalObjectStore(root)` | none |

Any other backend is written by the user against the same `ObjectStore`
interface; no object-store SDK is a dependency. Keys are validated before
they reach a backend (no `..`, empty segment or leading slash).
`RetryingStore` retries `TransientStoreError` (throttling, timeouts, 5xx)
under a `RetryPolicy`.

`scmle.data.store.conformance.run_conformance(store)` is the one conformance
suite, to be run against any backend you write: round trip, binary and empty and large objects, overwrite, missing
keys, `head` against `get`, conditional create, a race of eight conditional
creates (exactly one wins), sorted prefix listing, pagination, idempotent
delete, key rules and prefix isolation. The tests run it on the directory
backend.

## Snapshots

`SnapshotReader.read(key, contract, parents)` returns the frame and a
`SnapshotRecord` and appends the record to a `ReadLog` (optionally a JSON-lines
file): URI, content hash, size, modification time, backend version, declared
schema version, observed schema digest and whether it matches, row count,
age against the contract's `max_age_seconds` (`fresh`, `stale`, `unknown`)
and the ids of the snapshots it derives from. `snapshot_id` hashes URI,
content and schema, so the same bytes give the same id. `ReadLog.trace(id)`
walks the lineage. `connector_reader(store)` plugs a store into the ingestion
`ObjectStoreConnector`.

## Quality gates

`QualityGate(rules, store).run(frame, record, contract, stage)` for stage
`train` or `inference`. Each rule declares its policy in advance:

| Policy | Effect |
| --- | --- |
| `fail` | `QualityGateError` with the report; nothing continues |
| `quarantine` | row-level rules remove the rows and write them with the report under `quarantine/<snapshot id>/`; dataset-level rules (`SchemaRule`, `FreshnessRule`, `MinRowsRule`, `NullFractionRule`) hold the whole snapshot: a marker is written, the run stops, and the same snapshot is refused until `release` |
| `warn` | recorded in the report, data unchanged |

Rules: `SchemaRule`, `FreshnessRule`, `MinRowsRule`, `NullFractionRule`,
`NoNullsRule`, `UniqueKeyRule`, `RangeRule`, `AllowedValuesRule`, each
limited to the stages it names. Every rule is evaluated before any action, and
the report is written to `quality/` whatever the outcome.

## Incremental processing

`IncrementalProcessor` processes new objects under an input prefix in sorted
key order, once each: dedup of repeated row keys and repeated uploads, a
checkpoint with its own digest written after each object, transient errors
retried under a `RetryPolicy` (reproducible backoff, no jitter), the output
written before the checkpoint, a lock object so two runs cannot share a
checkpoint.

Delivery semantics: **at-least-once for the work, idempotent for the output.**
A crash between writing an output and committing the checkpoint makes the next
run process that object again and rewrite the same bytes under the same key;
the final outputs and checkpoint equal those of an uninterrupted run (tested).
Exactly-once is not claimed: a transform with side effects beyond its output
must be idempotent. A crashed run leaves its lock; `break_lock` is a person's
decision. The set of seen row keys lives in one JSON object, which is fine for
millions of keys and not beyond.

## Not done here

Only the directory backend is tested. A backend you supply is yours to
qualify with the conformance suite; differences in a remote service
(consistency, error codes, conditional-write support) are not covered here.
