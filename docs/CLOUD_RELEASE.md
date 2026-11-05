# Releasing a model to a cloud

A release is a signed, independently approved model bundle in a pinned image,
deployed to staging, tested live, then deployed to production, with a signed
record at each step. The pieces are code you can run without an account; the
workflows that connect them to a cloud have not been run.

## The evidence chain

1. **Approve the model.** `ReleaseRegistry.approve` (`scmle.governance.release_state`)
   accepts a version only if its bundle is signed and its `PromotionApproval`
   is an approved, independent, validly signed review of that exact bundle.
2. **Build and scan.** `cloud-release.yml` packages the bundle with
   `scmle package`, resolves the base image to a digest, builds an OCI archive,
   scans it with Trivy and applies `deploy/vuln-policy.json`
   (`scripts/vuln_gate.py`). The image digest is the digest in the OCI index.
3. **Admit.** `scmle deploy gate admit MANIFEST --key-file K` (`scmle.deploy.gate`)
   refuses unless the model is signed and approved, the scan passed for this
   image, the approver is not the requester and, for production, a signed
   staging record for the same commit, model and image shows all four live
   checks passed.
4. **Deploy by digest.** The workflow pushes the archive with `skopeo copy
   --preserve-digests`, refuses if the registry digest differs from the built
   one, then deploys that digest (ECS: new task definition and service update;
   Cloud Run: `gcloud run deploy`).
5. **Read the platform.** `scmle deploy observe` turns the platform's own
   answers into an `Observation`: revision, image digest, readiness, traffic.
6. **Live tests.** `scmle deploy live` checks health (`/livez`, `/readyz`,
   `/health`), schema (`/schema`), inference (the golden records within the
   cross-environment tolerance, and the answering `artifact_digest` equal to the
   released model) and security (anonymous and forged callers refused, no
   traceback, a challenge header, HTTPS, a read-only token cannot predict).
7. **Certify.** `scmle deploy gate certify` repeats the admission checks and
   additionally requires every live check present, passed and bound to this
   environment, commit, image and model, and the platform observation to show
   this image ready. Only then does it sign a `DeploymentRecord`: source
   commit, model digest, container digest, configuration digest, environment,
   provider, requester, approver, the digest of each check's exchange and the
   revision. `scmle deploy verify-record` checks the signature.

A refusal prints every reason, exits 2 and writes no record.

## Workflows

`cloud-release.yml` starts by hand only (`workflow_dispatch`). Every job
checks the `guard` job, which passes only if the repository variable
`SCMLE_CLOUD_RELEASE` is `true`; without it nothing runs. `cloud-deploy.yml`
is the reusable job for one environment; it declares
`environment: ${{ inputs.environment }}`, so the GitHub environment's required
reviewers are the independent approval, and it holds `id-token: write` only
there. Credentials are the run's OIDC token exchanged for a short-lived role
(`aws-actions/configure-aws-credentials`) or workload identity
(`google-github-actions/auth`); `scripts/verify_hosted_controls.py` fails if a
workflow mentions a long-lived key.

Set up before the first run (none of it exists in this repository):

* Environments `staging` and `production`, each with required reviewers who
  may not be the person starting the run, and "prevent self-review".
* Repository variable `SCMLE_CLOUD_RELEASE=true`; secret `SCMLE_SIGNING_KEY`.
* Per environment, variables: AWS `SCMLE_AWS_ROLE_ARN`, `SCMLE_AWS_REGION`,
  `SCMLE_ECS_CLUSTER`, `SCMLE_ECS_SERVICE`, `SCMLE_ECR_REPOSITORY`; GCP
  `SCMLE_GCP_PROVIDER`, `SCMLE_GCP_SERVICE_ACCOUNT`, `SCMLE_GCP_PROJECT`,
  `SCMLE_GCP_REGION`, `SCMLE_CLOUD_RUN_SERVICE`, `SCMLE_GCP_REPOSITORY`;
  both `SCMLE_SERVICE_URL`. Secrets `SCMLE_LIVE_TOKEN` and
  `SCMLE_LIVE_READONLY_TOKEN` (tokens your identity provider issued for the
  tests).
* The roles and service accounts the Terraform in [`INFRASTRUCTURE.md`](INFRASTRUCTURE.md)
  creates, applied by an operator with their own credentials.
* A release folder with `registry/` and `approval.json` for the model. No job
  fetches a model from a cloud store yet.

Production is deployed with the same image digest staging passed. The
workflow does not shift production traffic in steps; the staged allocation
below is library code not yet wired into it.

## Registry bound to the runtime

`ReleaseRegistry` keeps, beside `registry.json`, a `release_state.json` with
one state per version and what each environment runs.

| State | Meaning |
| --- | --- |
| `candidate` | registered, not judged |
| `challenger` | under comparison (set by hand) |
| `approved` | signed bundle and an independent signed approval (`approve`) |
| `production` | the runtime confirmed it serves all traffic (`promote`, `rollback`) |
| `retired` | superseded, rolled back or withdrawn |

`promote(model, version, plane, environment, image_digest, steps, health)` asks
a `ControlPlane` (`EcsControlPlane`, `CloudRunControlPlane`) to deploy and to
move traffic through `steps`, and reads the platform back after every request
(`control.confirm`). The registry state, the champion alias (production only)
and the ledger change only after the last read shows the revision, its image
digest, readiness and the full share. A read that disagrees, a platform that
does not settle, a health callback that stops the rollout or a registry that
refuses the alias returns `done=False`, restores the previous revision and
leaves `registry.json` and `release_state.json` as they were; the refusal is
recorded as a `deployment` row in the ledger. `rollback` restores the previous
known-good revision and changes the registry only after the platform confirms
it. ECS rolling updates cannot split traffic, so the ECS plane refuses a
staged rollout; Cloud Run can.

`deployed_version` answers "what does the registry say is deployed" only when
`reconcile` finds no disagreement, so an alias moved around the runtime is
never read as deployed.

```bash
scmle deploy observe --provider gcp --input service.json --revisions revisions.json --output observed.json
scmle deploy reconcile REGISTRY --model churn --environment production --observation observed.json [--ledger deployment_ledger.jsonl]
```

`reconcile` exits 1 and lists findings: `no_registry_record`,
`revision_mismatch`, `digest_mismatch`, `traffic_split`, `state_mismatch`,
`not_ready`, `alias_not_deployed`, `ledger_missing`, `ledger_disagrees`. It
repairs nothing.

## What the tests used

The ECS and Cloud Run answers are recorded JSON shaped by the documented
`DescribeServices`, `DescribeTasks`, `DescribeTaskDefinition` and Cloud Run
Admin API v2 responses (`tests/fixtures/cloud/`), changed by scripted
platforms (`tests/support_runtime.py`) that can lag, never settle or run
another image. The service responses behind the live checks are assembled
from [`SERVICE_SLO.md`](SERVICE_SLO.md). None was captured from AWS, Google
Cloud, API Gateway or a load balancer. `DescribeServices` carries no image
digest; the adapter reads it from `DescribeTasks` (or the pinned task
definition), which the workflow fetches. That the real services answer as the
documentation says, and that the workflows run, is not shown.

## Not claimed

A release to a real cloud, a protected environment, federated credentials, a
scan of a real image, a live test of a deployed service, registry and live
state reconciled on a cloud, multi-region availability, cost or latency
equivalence between providers.
