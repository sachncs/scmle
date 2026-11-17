"""Governed data: snapshots, quality gates, incremental runs.

`snapshot` records identity, schema version, freshness and lineage with
every read from an object store (`scmle.storage.objectstore`), `quality` the
gates before training and inference, and `processing` deterministic
incremental processing with retries, deduplication and checkpoint recovery.
"""
