# Phase 16: Infrastructure as code and release

Feature: [F14](../product/features/F14-cloud-portability-and-production-operation.md)
items 3, 9 and 10. Depends on phases [14](phase-14-portable-core.md) and
[15](phase-15-security-data-training-orchestration.md) (security, storage).
The cloud half needs sandbox accounts, a spending cap and a named approver.

## Work packages

- **WP-16.1 Infrastructure as code (item 3).** Terraform or OpenTofu under
  `deploy/` with shared modules and separate dev, staging and production
  configurations per provider. AWS: ECR, ECS Fargate, authenticated ingress,
  S3, IAM, Secrets Manager, CloudWatch. GCP: Artifact Registry, Cloud Run,
  authenticated ingress, Cloud Storage, IAM, Secret Manager, Cloud
  Monitoring. Resource limits, autoscaling, networking, encrypted storage and
  budgets are declared. CI runs `validate`, lint, a policy scanner and `plan`
  on every change. Environments update and destroy without orphans (a check
  lists resources left after `destroy`).
- **WP-16.2 CI/CD (item 9).** Extend `.github/workflows` to build and scan
  the image, authenticate to each cloud with short-lived federated
  credentials, deploy to staging before production, run live schema,
  inference, security and health tests against staging, and write a
  deployment record binding source commit, model digest, container digest,
  configuration and approver. Promotion is refused when evidence is missing,
  a gate fails, or the model artifact is unsigned or unapproved. Controlled
  rollouts and independent approvals use protected environments.
- **WP-16.3 Registry bound to the runtime (item 10).** Adapters read the
  running revision and image digest from ECS and Cloud Run. A reconcile
  command compares `registry.py` and `deployment_ledger.py` with that report
  and flags disagreement. States candidate, challenger, approved,
  production and retired are distinct. Promotion and staged traffic
  allocation call the control plane, and the registry changes only after the
  runtime confirms. Rollback restores the previous known-good revision. A
  registry alias can never claim a model is deployed when the runtime
  disagrees.

## Gate

Local: `validate`, lint, policy and `plan` clean; the pipeline logic passes
against recorded cloud responses; reconcile detects a seeded disagreement;
promotion refuses on each seeded failed gate. Cloud (per provider): a new
environment is provisioned, deployed, tested and destroyed from version
control with infrastructure and cost recorded; a permitted release runs
commit to staging to production with its full evidence chain; a rejected
promotion changes nothing; registry and live state reconcile after one
success and one failed release.

## Evidence

`plan/evidence/phase-16/`: plans, apply and destroy logs, cost record,
deployment records, reconcile reports.

## Not claimed

Multi-region availability and any cost or latency equivalence between
providers.
