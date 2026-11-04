# Phase 15: Security, data sources, portable training, orchestration

Feature: [F14](../product/features/F14-cloud-portability-and-production-operation.md)
items 4, 6, 7 and 8. Depends on [Phase 14](phase-14-portable-core.md).
Work packages run in parallel; each is verified locally against emulators
before any cloud run.

## Work packages

- **WP-15.1 Security (item 4).** Token verification (OIDC or JWT) with key
  rotation, authorization scopes, a written tenant-isolation policy, rate and
  request-size limits, safe error bodies, and audit rows for model
  activation, access and administration (reuse the hash-chained ledger).
  Tests for invalid, revoked and expired credentials, expired keys and
  tampered artifacts against a mock identity provider. No unauthenticated
  backend is reachable in any shipped configuration.
- **WP-15.2 Governed data (item 7).** An object-store interface with local,
  S3 and GCS backends, tested against MinIO and a GCS emulator through one
  shared conformance suite. Snapshot identity, schema version, freshness and
  lineage recorded with every read (extend the Phase 9 connectors). Quality
  gates before training and before inference with a declared failure policy
  (fail, quarantine, warn). Deterministic incremental processing with retries,
  deduplication and checkpoint recovery. Choose one warehouse per cloud from
  actual use cases (open question, see the README).
- **WP-15.3 Portable training (item 6).** A `Runner` interface with local,
  AWS container-job and GCP job implementations running one project spec
  without changes to the pipeline. Checkpoints, run manifests, trial results,
  metrics and evidence go to durable storage; an interrupted job resumes;
  retries are controlled; a cost budget stops a runaway job; training and
  serving use separate roles. Specialised compute (GPU, distributed) only
  behind a declared need.
- **WP-15.4 Orchestration (item 8).** One workflow definition for scheduled
  ingestion, training, evaluation, monitoring and batch prediction, with
  explicit dependencies, idempotency keys, durable run identifiers, timeouts,
  backoff, concurrency control, cost budgets, and a human-approval step that
  separates automatic from approved actions. A local executor runs the same
  definition for tests. Targets: Step Functions on AWS, Cloud Workflows with
  Scheduler on GCP.

## Gate

Local: the conformance suite passes on all three storage backends; duplicate
invocation, partial failure, missed schedule and interrupted job each behave
as declared on the local executor; every security test above passes. Cloud
(per provider): the same project spec completes with retained evidence and
resumes after an injected interruption, and its result differs from the local
reference by no more than the tolerance declared before the run; a dataset in
the cloud is ingested, validated, trained on and traced to its source
snapshot; stale or invalid input fails by the declared policy.

## Evidence

`plan/evidence/phase-15/`: conformance reports, interruption and duplicate
run logs, the reproducibility-tolerance report, lineage trace.

## Not claimed

Exactly-once processing. Delivery semantics are stated per component and
proved by the recovery tests, not assumed.
