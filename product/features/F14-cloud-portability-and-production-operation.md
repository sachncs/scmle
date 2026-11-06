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
  closed loop. `release_state.py` (Phase 16) binds the registry to a runtime
  through a control-plane interface; its ECS and Cloud Run adapters have read
  only recorded and scripted answers.
- `scmle/data/readers/connectors.py` accepts `s3`, `gs` and `az` schemes with
  snapshot digests; it has not run against real buckets or warehouses.
- The local halves of items 3, 9 and 10 are built and verified without a
  cloud account: Terraform modules for AWS and GCP with dev, staging and
  production environments, a Python policy check over the files and a
  destroy-orphan checker (`deploy/terraform/`, `scmle.deploy.iac_policy`,
  `scmle.deploy.orphans`, `docs/INFRASTRUCTURE.md`); manual, guarded release
  workflows, live checks, a promotion gate and a signed deployment record
  (`.github/workflows/cloud-release.yml`, `scmle.deploy.gate`,
  `scmle.deploy.record`); ECS and Cloud Run adapters, release states,
  confirmed promotion, rollback and `scmle deploy reconcile`
  (`scmle.governance.release_state`, `docs/CLOUD_RELEASE.md`). Evidence:
  `plan/evidence/phase-16/`. OpenTofu ran `fmt` and `validate` only (no
  `plan`), the workflows never ran, and every cloud answer is a recorded or
  scripted shape.
- The local halves of items 11, 12 and 13 and the production-claim checker are
  built and verified without a cloud account (`scmle.operations`,
  `docs/OPERATIONS.md`, `docs/RELEASING.md`): metrics with Prometheus text,
  trace-correlated logs, windowed signals, CloudWatch and Cloud Monitoring
  payloads, signed pager and ticket webhooks, a classifier that reports mixed
  causes and cost by version; labels and outcomes into the F13 loop, a
  retraining policy that recommends by default through the escalation path,
  gated replacement and a post-promotion check with intervals; a fault
  catalogue with eleven local scenarios and measured RTO and RPO. Evidence:
  `plan/evidence/phase-17/local/`. Nothing ran on a cloud, the exporters
  were checked against shapes written from the API references, the webhooks
  went to recording transports, and the production claim has not been made.

## Items (backlog numbers 1 to 13)

Each item is done only when its code, tests, environment-specific
integration tests and retained evidence meet its acceptance. "Local" means
verifiable without a cloud account; "cloud" needs credentials, a budget cap
and a named approver.

| # | Item | Build and verify locally | Needs a cloud account |
|---|---|---|---|
| 1 | Reproducible local environment | Built; open: run the container path on a machine with Docker, arm64 | none |
| 2 | Immutable, secure artifacts | Built; open: build and scan a real image, attest its digest, measure two-build reproducibility | Prediction parity and vulnerability gate on AWS and GCP |
| 3 | Infrastructure as code | Built locally (`deploy/terraform/`, `docs/INFRASTRUCTURE.md`); `tofu fmt`, `validate` pass and the lock files are committed; open: `plan` with credentials | `apply`, test, `destroy`, recorded cost |
| 4 | Authentication, authorization, security | Built locally (`scmle.security`, `docs/TENANT_ISOLATION.md`); OIDC discovery, a network-fetched JWKS and a pulled revocation list are built (`scmle.security.remote`, tested against a local server); open: an independent review, a real identity provider | TLS ingress, workload identity and managed keys in each cloud |
| 5 | Online and batch inference | Built (`docs/SERVICE_SLO.md`) | Restart, load and scaling tests on each cloud |
| 6 | Portable training | Built locally (`scmle.runners`, `docs/PORTABLE_TRAINING.md`); open: exercise the AWS and GCP adapters and their SDK factories against real services | Same spec completing on all three with a reproducibility-tolerance report |
| 7 | Governed data sources | Built locally (`scmle.data.store`, `docs/GOVERNED_DATA.md`); open: the conformance suite against MinIO and the GCS emulator | One warehouse per cloud, ingest to train to trace on a controlled dataset |
| 8 | Orchestration | Built locally (`scmle.workflow`, `docs/ORCHESTRATION.md`); open: have the services accept the emitted definitions | Step Functions and Cloud Workflows with Scheduler; duplicate, partial-failure, missed-schedule and interruption runs |
| 9 | CI/CD release gates | Built locally (`cloud-release.yml`, `scmle.deploy.gate`, `docs/CLOUD_RELEASE.md`); open: run the workflows, build and scan a real image | Federated short-lived credentials, staging then production, live tests |
| 10 | Registry bound to the runtime | Built locally (`scmle.governance.release_state`, `scmle deploy reconcile`); open: staged traffic on ECS | Reading the real revision from ECS and Cloud Run, traffic split, previous-revision rollback |
| 11 | Observability and incidents | Built locally (`scmle.operations`, `docs/OPERATIONS.md`); the signals are wired into the generated service; open: tune the classifier on real incidents, price against a bill | CloudWatch and Cloud Monitoring; a real pager; injected failure and controlled drift |
| 12 | Continuous evaluation | Built locally (`scmle.operations.continuous`); a holdback with a sample-ratio guard is built; open: the F13 alarm still does not gate retraining | Controlled degradation scenario in staging |
| 13 | Failure recovery and DR | Built locally (`scmle.operations.recovery`, `faults`); open: machine crash and power loss, objectives revisited with the owner | Independent passes on AWS and GCP staging with reviewer-approved evidence |

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
