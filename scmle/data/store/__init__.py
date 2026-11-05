"""Governed data: object stores, snapshots, quality gates, incremental runs.

`objectstore` holds the interface and the local backend, `cloud` the S3 and
GCS backends (over injected clients), `snapshot` records identity, schema
version, freshness and lineage with every read, `quality` the gates before
training and inference, and `processing` deterministic incremental
processing with retries, deduplication and checkpoint recovery.
"""
