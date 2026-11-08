# Stage 4: Cloud and production

Names follow [`00-canon.md`](00-canon.md). Item classes and reasons:
[`COMPLETION_AUDIT.md`](COMPLETION_AUDIT.md), "Stage 4". Feature F14 (items 1
to 13) and F15 in [`../product/features.md`](../product/features.md). Depends on
stage 1. Items 4.1 to 4.7 need a container daemon and an image scanner; the
rest of the class C items need an AWS account, a GCP account, a spending cap and
a named approver.

## Goal

One lifecycle, one signed bundle, running on local, AWS and GCP: a hardened
image, a secured service, infrastructure under version control, a release
pipeline that refuses what has not passed, observation, continuous evaluation
and recovery, with every claim backed by retained evidence on the real
provider.

## Starting point

The local half of all of this is built and verified without a cloud account
(deployment contract, hardened artifact, serving contract, `scmle.security`,
`scmle.data.store`, `scmle.runners`, `scmle.workflow`, `deploy/terraform`,
`scmle.deploy.{iac_policy,orphans,gate,record}`, `cloud-release.yml`,
`scmle.governance.release_state`, `scmle.operations`,
`scripts/check_production_claim.py`, `scmle.trail`). Every cloud answer in the
tests is a recorded or scripted shape of the documented API, not a capture.
Nothing ran on AWS or GCP; `release.yml`, `cloud-release.yml` and
`cloud-deploy.yml` have never run on GitHub. Local evidence is in
`plan/evidence/phase-14/` to `phase-18/` (do not edit; regenerate with the
`scripts/make_phase*_evidence.py` scripts). The Docker daemon is not running on
the development machine; `trivy`, `skopeo` and MinIO are not installed.

## Work packages

### 4.A Container and artifact (F14 items 1, 2, 5)

| Items | Scope | Closing criterion |
|---|---|---|
| 4.1 to 4.3 | Build the image and start compose (`compose.yaml`, `deploy/Dockerfile`, the generated `build.sh`, `make demo-compose`) on a machine with Docker; fix what breaks and keep the transcript. Measure whether two builds on the same pinned inputs give the same digest. Build and run the demo with `PLATFORM=linux/arm64` | a transcript per step; image reproducibility is claimed only to the extent measured |
| 4.4 to 4.6 | `image-attestation.json` bound to a real digest; asymmetric or keyless signing and a verifier other than this code; a scan of the built image with an independent scanner (Trivy) in CI; CI builds the image from pinned inputs and verifies its own signature | CI on a clean runner builds, scans and verifies, then the demo runs in containers |
| 4.7 | Restart, load and scaling tests on each cloud | measured against the declared service levels in `docs/SERVICE_SLO.md` |

### 4.B Security, data, training, orchestration (F14 items 4, 6, 7, 8)

| Items | Scope | Closing criterion |
|---|---|---|
| 4.8 | TLS ingress, workload identity and managed keys on AWS and GCP; a real identity provider behind `scmle.security.remote` (the key set and revocation documents are not signed, so their integrity is that of the TLS connection) | the security tests pass against the real provider |
| 4.9, 4.10 | Run `scripts/run_object_store_conformance.py` against MinIO, the GCS emulator, a real S3 bucket and a real GCS bucket and fix what differs. Choose one warehouse per cloud from actual use cases and connect it through the ingestion connectors. Ingest a controlled cloud dataset, validate it, train on it and trace the model to its source snapshot; show that stale or invalid input fails by the declared policy there | conformance reports per backend; a retained trace |
| 4.11 | Run the same project spec through `AwsContainerJobRunner` and `GcpJobRunner` against real ECS and Cloud Run, with separate training and serving identities; inject an interruption and show the resume; check the budget stop against billing data (the in-code cost is an estimate); exercise `RunV2Client` and `ecs_client`; produce the reproducibility-tolerance report against the local reference with tolerances declared before the run | per provider the spec completes with retained evidence, resumes after an interruption, and differs from the local reference by no more than the declared tolerance |
| 4.12, 4.13 | Deploy the emitted Step Functions and Cloud Workflows definitions with EventBridge Scheduler and Cloud Scheduler; confirm the services accept them (the validators check documented structure only); wire the approval callback to a notification channel; enforce concurrency and cost budgets in the infrastructure; run duplicate, partial-failure, missed-schedule and interruption scenarios on each engine | logs kept; delivery semantics stated per component (at-least-once for the work, idempotent for the output), not exactly-once |

### 4.C Infrastructure and release (F14 items 3, 9, 10)

| Items | Scope | Closing criterion |
|---|---|---|
| 4.14, 4.15 | `tofu plan` per environment with that environment's credentials; then per provider in a sandbox account with a spending cap: apply a new environment, deploy, test, destroy, run `scripts/check_destroy_orphans.py` on real listings, and record infrastructure and cost against the bill | per provider a new environment is provisioned, deployed, tested and destroyed from version control |
| 4.16 | Create the protected `staging` and `production` environments with reviewers other than the author, the OIDC trust in each cloud, and the variables and secrets in `docs/CLOUD_RELEASE.md`; run `cloud-release.yml` once to success and once to a refused promotion; fetch the model and its approval from a store instead of a committed release folder; scan a real image and keep the report; wire staged traffic into the production job; a reviewer other than the author signs the staging evidence; an asymmetric or keyless deployment record | a permitted release runs commit to staging to production with its full evidence chain; a rejected promotion changes nothing |
| 4.17 | Run the adapters against real ECS and Cloud Run responses and fix what differs from the documented shapes (`DescribeServices` carries no digest; Cloud Run image digests; condition names); exercise Cloud Run traffic splitting; decide how ECS gets staged traffic (weighted target groups); show a rollback to the previous revision on each cloud; run `scmle deploy reconcile` after one successful and one failed release | registry and live state reconcile after both |
| 4.28 (D) | Close or accept the known gaps: customer-managed keys for Artifact Registry and Secret Manager, a custom domain and certificate for the AWS gateway, API Gateway access-log permissions, WAF or Cloud Armor | the owner's decision on each, recorded with cost |

### 4.D Operate, evaluate, recover (F14 items 11, 12, 13)

| Items | Scope | Closing criterion |
|---|---|---|
| 4.18, 4.19 | Write the signals the generated service already emits (`SCMLE_SIGNALS_FILE`) to CloudWatch and Cloud Monitoring from the staging service, with a real pager and ticket endpoint; inject a service failure and a controlled drift event in each provider's staging and keep the alert and telemetry captures; an operator acknowledges and resolves them. Price the cost report with real rates against a bill; tune the incident classifier on real incidents | correctly classified alerts, retained telemetry and an auditable operator response on each provider |
| 4.20, 4.29 | Controlled degradation against a deployed model with labels that arrive after a real delay and a real business outcome; keep the proposal, the person's reply, the challenger's evidence and the replacement. The owner decides per product which business metrics and margins `post_promotion_check` watches. The holdback exists (`holdback_assignment`, `holdback_check`); a deployment routes held-back units to the previous model | the expected retraining proposal or escalation; a new model is deployed only after its tests and approvals pass |
| 4.21, 4.22, 4.25, 4.30 | Run every catalogued fault in each provider's staging and record RTO and RPO per stateful component against `recovery.OBJECTIVES`: process and task kill, unhealthy task with rolling replacement, revoked workload identity, denied bucket access, managed-dependency outage, corrupted object, rollback to a real managed revision, job interruption, secret rotation. Add a machine crash or power loss and a zone or region loss. Locally, add a torn-write simulation of the ledger and state files (4.25). Revisit the objectives with the owner (4.30) | each provider passes the declared scenarios within the documented RTO and RPO |

### 4.E The production claim (F14 item 13 and the final gate)

| Item | Scope | Closing criterion |
|---|---|---|
| 4.23 | Run all P0 acceptance on both providers in staging; write `acceptance.json`, `deployment.json`, `recovery.json`, `observability.json` and `continuous-evaluation.json` under `plan/evidence/phase-17/{aws,gcp}/`; a reviewer other than the author signs `signoff.json` with a key the pipeline does not hold; run `scripts/check_production_claim.py` | the checker exits 0. Only then may the sentence in `docs/RELEASING.md` be used and its status line changed |

"Production-qualified on AWS and GCP" is claimed only when F14 items 1 to 13
pass their acceptance on both providers with the exact application, model, data
and configuration versions, the cloud environment tested, the evidence and the
responsible approver recorded. A container that starts, a successful request, a
registered model or an infrastructure manifest is not that claim.

### 4.F Business trail and qualified breadth (F15)

| Items | Scope | Closing criterion |
|---|---|---|
| 4.26, 4.27 | Local, buildable now: `scmle.trail` record types, signature, verify and show for written explanations of data-quality findings, performance trade-offs and deployment decisions, and for point-in-time and source-data steps on a synthetic source | `scmle trail verify` detects a missing or unsigned explanation; the committed trail still passes `scripts/trace_model.py` |
| 4.31 (D) | Breadth slices, one at a time, never by adding unsupported algorithms to the production registry: clustering, association, dimensionality reduction, forecasting, ranking, documents and NLP, image/audio/video, geospatial and graph, pretrained and multimodal models, GPU or distributed training for a justified workload, streaming. The owner declares the requirement and fixes parameters in the slice's file under `benchmarks/breadth/`; the slice is built; its four benchmarks (robustness, drift, explanation, monitoring) run against a public or domain benchmark with an independent replication (stage 5); it completes ingestion, evaluation, packaging, cloud deployment, monitoring and recovery on both providers. Streaming also needs its delivery semantics stated and proved over the whole pipeline. Starts only after 4.23 | a slice is production-supported only after all of those; no slice is qualified today |
| 5.20 to 5.22 | Independent trace of a deployed model and the reasoning in the trail (class B, in stage 5) | see stage 5 |

## Exit gate

Per provider: a new environment is provisioned, deployed, tested and destroyed
from version control with infrastructure and cost recorded; a permitted release
runs from commit to staging to production with its full evidence chain; a
rejected promotion changes nothing; registry and live state reconcile after one
success and one failed release; an injected failure and a drift event produce
classified alerts and an auditable operator response; the declared recovery
scenarios pass within the documented RTO and RPO; `check_production_claim.py`
exits 0.

## Not claimed

Anything that ran on a cloud before it has. Exactly-once processing. Cost
enforcement is an estimate from declared rates, not billing. Anything outside
the declared scenarios: a passing recovery run measures those scenarios, not
availability in general. Differences between providers in latency, cost and
scaling are reported, not equalised.
