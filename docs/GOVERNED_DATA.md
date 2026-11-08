# Governed data

How data is read, checked and processed before training and inference
(`scmle.data.store`). Local half of F14 item 7; what needs a cloud account is
listed at the end.

## Object stores

`ObjectStore` is bytes under keys: `put` (atomic replace; `if_absent=True`
creates only), `get`, `head`, `list_objects`, `delete`, `exists`. Metadata
always carries a SHA-256 computed by this code; the S3 and GCS backends keep
it as user metadata and verify it on read, so a changed object raises
`CorruptObjectError`. Backends:

| Backend | Class | Client |
| --- | --- | --- |
| Directory | `LocalObjectStore(root)` | none |
| S3, MinIO | `S3ObjectStore(client, bucket, prefix)` or `s3_store(...)` | a boto3 S3 client, or anything with `put_object`, `get_object`, `head_object`, `list_objects_v2`, `delete_object` |
| GCS | `GcsObjectStore(client, bucket, prefix)` or `gcs_store(...)` | a `google.cloud.storage.Client`, or anything with `bucket()`, and on it `blob()`, `get_blob()`, `list_blobs()` |

Neither SDK is a dependency: the factories import `boto3` or
`google.cloud.storage` only when called and explain how to install them; a
caller may pass a client it built (credentials, endpoint, proxies). Keys are
validated before they reach a backend (no `..`, empty segment or leading
slash). `RetryingStore` retries `TransientStoreError` (throttling, timeouts,
5xx) under a `RetryPolicy`.

`scmle.data.store.conformance.run_conformance(store)` is the one conformance
suite: round trip, binary and empty and large objects, overwrite, missing
keys, `head` against `get`, conditional create, a race of eight conditional
creates (exactly one wins), sorted prefix listing, pagination, idempotent
delete, key rules and prefix isolation. The tests run it on the directory
backend and on the S3 and GCS backends over in-memory clients;
`scripts/run_object_store_conformance.py` runs it against MinIO, the GCS
emulator or a bucket.

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

MinIO and the GCS emulator were not started, and no real bucket was touched:
the S3 and GCS code ran only against in-memory clients written to the same
protocol, so differences in a real service (consistency, error codes,
conditional-write support in older S3-compatible servers) are untested. One
warehouse per cloud, and a dataset ingested, validated, trained on and traced
in a real account, need the cloud accounts.
