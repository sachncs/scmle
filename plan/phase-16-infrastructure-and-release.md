# Phase 16: Infrastructure as code and release

Feature: [F14](../product/features/F14-cloud-portability-and-production-operation.md)
items 3, 9 and 10. Depends on phases [14](phase-14-portable-core.md) and
[15](phase-15-security-data-training-orchestration.md).

The code, tests and local evidence for the three work packages are built and
verified without a cloud account: the Terraform under `deploy/terraform/` with
its policy check and destroy-orphan checker (`docs/INFRASTRUCTURE.md`); the
release workflows, live checks, promotion gate and signed deployment record,
and the registry bound to the runtime through ECS and Cloud Run adapters with
a reconcile command (`docs/CLOUD_RELEASE.md`). Evidence:
`plan/evidence/phase-16/` (regenerate with `scripts/make_phase16_evidence.py`).
Every cloud answer in the tests is a recorded or scripted shape of the
documented API, not a capture. The Terraform was never run through
`terraform`; the workflows never ran on GitHub. What remains needs an AWS
account, a GCP account, a spending cap and a named approver, or a machine with
Terraform and Docker.

## Open

- **WP-16.1 Infrastructure, with the tool and in each account.** Install
  Terraform or OpenTofu and run `scripts/terraform_checks.sh`: the files were
  written by hand and `fmt -check` and `validate` have never run, so expect
  fixes; commit the provider lock files. Run `plan` for each environment with
  that environment's credentials. Then, per provider, in a sandbox account with
  a spending cap: apply a new environment, deploy, test, destroy, run
  `scripts/check_destroy_orphans.py` on real listings and record the
  infrastructure and the cost against the bill. Known gaps to close or accept:
  customer-managed keys for Artifact Registry and Secret Manager, a custom
  domain and certificate for the AWS gateway, API Gateway access-log
  permissions, WAF or Cloud Armor.
- **WP-16.2 The workflows, on GitHub and in each cloud.** Create the protected
  `staging` and `production` environments with reviewers other than the
  author, the OIDC trust in each cloud, the environment variables and secrets
  listed in `docs/CLOUD_RELEASE.md`, and run `cloud-release.yml` once to
  success and once to a refused promotion. First runs will find what the
  untested shell does wrong (the `skopeo` push and digest check, the ECS task
  definition rewrite, the Cloud Run REST reads, the approver lookup). Fetch the
  model and its approval from a store instead of a committed release folder,
  run the scan on a real image and keep the report, wire staged traffic into
  the production job, and have a reviewer other than the author sign the
  staging evidence. The deployment record is HMAC-signed; an asymmetric or
  keyless signature and a verifier other than this code are open.
- **WP-16.3 The runtime, for real.** Run the adapters against real ECS and
  Cloud Run responses and fix what differs from the documented shapes
  (`DescribeServices` carries no digest; Cloud Run image digests; condition
  names). Exercise traffic splitting on Cloud Run, decide how ECS gets staged
  traffic (weighted target groups), and show a rollback to the previous
  revision on each cloud. Run `scmle deploy reconcile` after one successful
  and one failed release on each provider.

## Gate (still open)

Per provider: a new environment is provisioned, deployed, tested and
destroyed from version control with infrastructure and cost recorded; a
permitted release runs commit to staging to production with its full evidence
chain; a rejected promotion changes nothing; registry and live state reconcile
after one success and one failed release. The local half of the gate (policy
and seeded violations clean, the pipeline logic against recorded responses,
reconcile detecting a seeded disagreement, promotion refusing on each seeded
failed gate) passes; see the evidence summary.

## Not claimed

Anything that ran on a cloud, Terraform validity, workflow behavior on GitHub,
multi-region availability and any cost or latency equivalence between
providers.
