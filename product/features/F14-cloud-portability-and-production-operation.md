# F14: Cloud portability and production operation

Vocabulary and principles: [`../../plan/00-canon.md`](../../plan/00-canon.md).
Built in phases [14](../../plan/phase-14-portable-core.md),
[15](../../plan/phase-15-security-data-training-orchestration.md),
[16](../../plan/phase-16-infrastructure-and-release.md) and
[17](../../plan/phase-17-operate-and-qualify.md).

Why this file exists: SCMLE trains, signs, packages and serves a model, and
it keeps a ledger of what it decided. None of that is yet tied to a running
service in a real cloud: the generated Dockerfile is unpinned, the registry
describes intent rather than the deployed revision, no infrastructure is
under version control, and the failure drills are simulated. A model that
passes every statistical gate but cannot be deployed, secured, observed and
recovered is not a product.

Principle: one ML lifecycle, portable execution, provider adapters for the
infrastructure. Local, AWS and GCP run the same bundle, API and contracts.

## Purpose

Take an approved, signed bundle from a commit to a running, authenticated,
observed and recoverable service on local, AWS and GCP. Bind what the
registry says is deployed to what the cloud says is running. Connect
monitoring, retraining and rollback to real operational signals.

## What is verified in the repo today

- Items 1, 2 and 5 are built and verified locally: a typed deployment
  contract (`scmle/deploy/contract.py`), a local environment (`make demo`,
  `compose.yaml`), a hardened generated service (pinned-base placeholder,
  locked install, non-root, SBOM, signed provenance, verify-before-load,
  `SCMLE_TRUSTED` refused in production, vulnerability gate) and a versioned
  serving contract with `/livez`, `/readyz`, limits, batch and golden-record
  parity. Evidence: `plan/evidence/phase-14/`. The container path was not run.
- The local halves of items 4, 6, 7 and 8 are built and verified without a
  cloud account: token verification, scopes, tenant policy, limits and audit
  (`scmle.security`); object stores, snapshots, quality gates and incremental
  processing (`scmle.data.store`); the `Runner` interface with local, AWS and
  GCP adapters (`scmle.runners`); one workflow definition, a local executor
  and emitted Step Functions and Cloud Workflows definitions
  (`scmle.workflow`). Evidence: `plan/evidence/phase-15/`. The cloud adapters
  ran only against scripted or in-memory stand-ins.
- `scmle/governance/` holds `registry.py`, `lifecycle.py`,
  `deployment_ledger.py`, `keyring.py`, `sbom.py`, `local_faults.py` and the
  closed loop. None reads a cloud control plane.
- `scmle/data/readers/connectors.py` accepts `s3`, `gs` and `az` schemes with
  snapshot digests; it has not run against real buckets or warehouses.
- There is no Terraform or OpenTofu and no CI job that deploys an image.

## Items (backlog numbers 1 to 13)

Each item is done only when its code, tests, environment-specific
integration tests and retained evidence meet its acceptance. "Local" means
verifiable without a cloud account; "cloud" needs credentials, a budget cap
and a named approver.

| # | Item | Build and verify locally | Needs a cloud account |
|---|---|---|---|
| 1 | Reproducible local environment | Built; open: run the container path on a machine with Docker, arm64 | none |
| 2 | Immutable, secure artifacts | Built; open: build and scan a real image, attest its digest, measure two-build reproducibility | Prediction parity and vulnerability gate on AWS and GCP |
| 3 | Infrastructure as code | Terraform or OpenTofu modules for dev, staging and production per provider; `validate`, lint, policy checks and `plan` in CI | `apply`, test, `destroy`, recorded cost |
| 4 | Authentication, authorization, security | Built locally (`scmle.security`, `docs/TENANT_ISOLATION.md`); open: OIDC discovery and a network JWKS, revocation distribution, an independent review | TLS ingress, workload identity and managed keys in each cloud |
| 5 | Online and batch inference | Built (`docs/SERVICE_SLO.md`) | Restart, load and scaling tests on each cloud |
| 6 | Portable training | Built locally (`scmle.runners`, `docs/PORTABLE_TRAINING.md`); open: exercise the AWS and GCP adapters and their SDK factories against real services | Same spec completing on all three with a reproducibility-tolerance report |
| 7 | Governed data sources | Built locally (`scmle.data.store`, `docs/GOVERNED_DATA.md`); open: the conformance suite against MinIO and the GCS emulator | One warehouse per cloud, ingest to train to trace on a controlled dataset |
| 8 | Orchestration | Built locally (`scmle.workflow`, `docs/ORCHESTRATION.md`); open: have the services accept the emitted definitions | Step Functions and Cloud Workflows with Scheduler; duplicate, partial-failure, missed-schedule and interruption runs |
| 9 | CI/CD release gates | Image build and scan, evidence chain binding commit, model digest, image digest and approver; refusal on any failed gate; workflow linting | Federated short-lived credentials, staging then production, live tests |
| 10 | Registry bound to the runtime | Reconcile command comparing registry state with a runtime report; states candidate, challenger, approved, production, retired; promotion and rollback that change registry state only after the runtime confirms | Reading the real revision from ECS and Cloud Run, traffic split, previous-revision rollback |
| 11 | Observability and incidents | Structured logs, trace ids, version attribution, metric exporters, alert rules, pager and ticket webhooks, a classifier separating infrastructure, data-quality and model faults, cost by service and version | CloudWatch and Cloud Monitoring; injected failure and controlled drift |
| 12 | Continuous evaluation | Delayed labels and business outcomes into the F13 loop, retrain recommendation policy, champion and challenger comparison, approval before replacement, post-promotion metric check | Controlled degradation scenario in staging |
| 13 | Failure recovery and DR | Fault catalogue and runner (process kill, bad release, lost storage access, corrupted bundle, schema break, job interruption, secret rotation); RTO and RPO per stateful component | Independent passes on AWS and GCP staging with reviewer-approved evidence |

## Method and measurement

- Deployment contract: one typed specification (image digest, bundle digest,
  configuration, secret references, resources, health) that every provider
  adapter must satisfy. Adapters differ only in infrastructure calls.
- Prediction parity: a fixed set of reference records with expected outputs
  and a declared tolerance per output type; the same records pass on every
  environment.
- Reproducibility across environments: differences in training results are
  reported against tolerances declared before the run; a miss is published,
  not hidden.
- RTO and RPO: measured per stateful component (registry, ledger, object
  store, secrets), never assumed.
- Any gate that misses its pre-registered threshold stays descriptive and is
  not retuned (canon P1).

## Acceptance (machine-checkable)

1. A fresh checkout on a supported machine runs the local demo end to end,
   including a rollback drill, in CI on a clean runner.
2. The same approved bundle passes signature verification, image build,
   vulnerability gate and prediction parity on local, AWS and GCP.
3. A new environment is provisioned, deployed, tested and destroyed from
   version-controlled definitions on each provider, with the resulting
   infrastructure and cost recorded.
4. A permitted release proceeds from commit to staging to production with a
   complete evidence chain; a failed approval or validation gate refuses
   promotion and changes nothing.
5. After a successful and after a failed release, registry state and live
   cloud state reconcile.
6. Staging on each provider passes the declared recovery scenarios within the
   documented RTO and RPO.

## Release gate

"Production-qualified on AWS and GCP" may be claimed only when items 1 to 13
pass their acceptance on both providers. A container that starts, an HTTP
request that succeeds, a registered model or an infrastructure manifest is
not production qualification. Every release identifies the application,
model, data and configuration versions deployed, the cloud environment
tested, the evidence that passed and the responsible approver.

## External referent

The cloud runtime itself (revision, digest, traffic split, metrics), an
independent scanner and attestation verifier for the image, and a reviewer
other than the author who signs the staging evidence.

## Not claimed

Items that need an account are not marked done on local evidence. Differences
between providers in latency, cost and scaling are reported, not equalised.
