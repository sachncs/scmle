# Phase 15: Security, data sources, portable training, orchestration

Feature: [F14](../product/features/F14-cloud-portability-and-production-operation.md)
items 4, 6, 7 and 8. Depends on [Phase 14](phase-14-portable-core.md).

The code, tests and local evidence for the four work packages are built and
verified without a cloud account: `scmle.security` and
`docs/TENANT_ISOLATION.md`; `scmle.data.store` and `docs/GOVERNED_DATA.md`;
`scmle.runners` and `docs/PORTABLE_TRAINING.md`; `scmle.workflow` and
`docs/ORCHESTRATION.md`. Evidence: `plan/evidence/phase-15/` (regenerate with
`scripts/make_phase15_evidence.py`). The cloud adapters ran only against
scripted or in-memory stand-ins of the documented service shapes. What remains
needs an AWS account, a GCP account, a spending cap and a named approver, or a
container daemon.

## Open

- **WP-15.1 Security, in each cloud.** TLS ingress, workload identity and
  managed keys on AWS and GCP. A real identity provider (OIDC discovery and
  fetching a JWKS over the network; the key set is a file today). Revocation
  distribution to every replica, and a pen-test style review of the gateway by
  someone other than the author.
- **WP-15.2 Governed data, against real services.** Run
  `scripts/run_object_store_conformance.py` against MinIO, the GCS emulator
  and a real S3 bucket and a real GCS bucket, and fix what differs (error
  codes, conditional-write support, consistency). Choose one warehouse per
  cloud from actual use cases (open question, see the README) and connect it
  through the Phase 9 connectors. Ingest a controlled cloud dataset, validate
  it, train on it and trace the model to its source snapshot; show that stale
  or invalid input fails by the declared policy there.
- **WP-15.3 Portable training, in each cloud.** Build and push the image;
  run the same project spec through `AwsContainerJobRunner` and
  `GcpJobRunner` against real ECS and Cloud Run, with the separate training
  and serving identities in IAM; inject an interruption and show the resume;
  check the budget stop against the provider's billing data (the in-code cost
  is an estimate); exercise `RunV2Client` and `ecs_client`, which no test has
  reached. Produce the reproducibility-tolerance report against the local
  reference with tolerances declared before the run.
- **WP-15.4 Orchestration, on the cloud engines.** Deploy the emitted Step
  Functions and Cloud Workflows definitions with EventBridge Scheduler and
  Cloud Scheduler; confirm the services accept them (the validators check
  documented structure only); wire the approval callback to a notification
  channel; enforce concurrency and cost budgets in the infrastructure (the
  definitions return them as `controls` and enforce nothing); run duplicate,
  partial-failure, missed-schedule and interruption scenarios on each engine
  and keep the logs.
- **Container path.** The compose services now take the demo identity
  settings and the test runner mints a demo token (`scmle.deploy.demo`); none
  of it ran in a container (see Phase 14).

## Gate (still open)

Cloud, per provider: the same project spec completes with retained evidence
and resumes after an injected interruption, and its result differs from the
local reference by no more than the tolerance declared before the run; a
dataset in the cloud is ingested, validated, trained on and traced to its
source snapshot; stale or invalid input fails by the declared policy. The
local gate (conformance suite on the three backends, duplicate invocation,
partial failure, missed schedule and interrupted job on the local executor,
the security tests) passes; the evidence says which backends ran over
in-memory clients.

## Not claimed

Exactly-once processing. Delivery semantics are stated per component and
proved by the recovery tests, not assumed: incremental processing is
at-least-once for the work and idempotent for the output. No cloud result of
any kind, no MinIO or emulator result, no container result. Cost enforcement
is an estimate from declared rates, not billing. The emitted workflow
definitions were never accepted by a service.
