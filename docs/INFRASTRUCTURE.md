# Infrastructure as code

Terraform (or OpenTofu) under [`deploy/terraform/`](../deploy/terraform) declares
the same service on AWS and on GCP. This page says what is declared, what is
checked without an account, and what is not checked at all.

Nothing in this repository has been applied to an account. `terraform
validate`, `fmt` and `plan` have not been run by the author (the tool was not
installed); only the Python policy check and its tests have run.

## Layout

```text
deploy/terraform/
  modules/aws_environment/   one shared module per provider
  modules/gcp_environment/
  aws/environments/{dev,staging,production}/
  gcp/environments/{dev,staging,production}/
```

Each environment directory holds a `main.tf` that calls the shared module with
its own sizing, budget and protection settings, a `variables.tf` for the
values that only exist at deploy time (image digest, issuer, repository), and
`outputs.tf`. The state location is not in the files: `backend "s3" {}` and
`backend "gcs" {}` are empty and are filled with `-backend-config` at `init`.
No `.terraform.lock.hcl` is committed because none could be generated; create
and review one with the first `init`.

| | AWS | GCP |
| --- | --- | --- |
| Images | ECR, immutable tags, scan on push, KMS | Artifact Registry, immutable tags |
| Compute | ECS Fargate service, circuit breaker with rollback, CPU target-tracking autoscaling | Cloud Run v2, min/max instances, concurrency, resource limits |
| Authenticated ingress | API Gateway HTTP API, JWT authorizer on every route, VPC link to an internal load balancer | Cloud Run with IAM invoker only (no `allUsers`); the service verifies its bearer token again |
| Network | VPC, private subnets for tasks, NAT, S3 gateway endpoint, no security group open to the internet | VPC, private-Google-access subnet with flow logs, direct VPC egress |
| Storage | S3: KMS, versioning, public access blocked, TLS-only | Cloud Storage: customer key, versioning, public access prevented |
| Secrets | Secrets Manager containers (KMS); values set out of band | Secret Manager containers; values set out of band |
| Identities | Execution, `serve`, `train` and `deploy` roles; `deploy` trusts only this repository and environment through GitHub OIDC | `serve`, `train`, `deploy` service accounts; resource-level grants; workload identity federation limited to this repository and environment |
| Observability | CloudWatch log groups (KMS, retention), alarms, SNS | Cloud Monitoring alert policies, e-mail channels |
| Budget | `aws_budgets_budget` with actual and forecast notifications | `google_billing_budget` with thresholds |

Every taggable resource carries `scmle-environment=<env>`; the destroy-orphan
checker finds leftovers by it.

Known gaps: the GCP module uses Google-managed encryption for Artifact Registry
and Secret Manager (a customer key only on Cloud Storage); the AWS API Gateway
default endpoint is used (no custom domain or certificate); no WAF or Cloud
Armor; no multi-region; Cloud Run containers cannot be made read-only through
this resource; the API Gateway access-log resource policy is not declared.

## Checks that need no account

```bash
python scripts/check_terraform_policy.py deploy/terraform   # always runs
scripts/terraform_checks.sh deploy/terraform                # fmt + validate if the tool exists
```

`check_terraform_policy.py` (`scmle.deploy.iac_policy`, reader in
`scmle.deploy.hcl`) parses the files and refuses:

* a public ingress CIDR, a gateway route without an authorizer, a load
  balancer that is public without authenticating, tasks with public addresses,
  `allUsers` or `allAuthenticatedUsers`, a disabled invoker check;
* storage, registries, log groups, secrets, topics or keys without
  encryption, buckets without a public-access block, mutable image tags, keys
  without rotation;
* wildcard IAM actions, wildcard principals (except in a Deny), wildcard
  resources (except `ecr:GetAuthorizationToken` and the two task-definition
  calls, which cannot be scoped; key policies are exempt), administrator or
  primitive roles, project-wide grants, federation trusts that do not match
  one subject exactly, missing deploy/train/serve identities, a `serve` role
  that can write or a `deploy` role that can touch data, secrets or keys;
* a missing budget or notification, a non-positive amount per environment;
* an image that is not a digest, or a digest variable that does not validate;
* credential-shaped text, literal secret attributes, secret-named defaults,
  secret values in an environment list, state, key and variable files, a
  backend with settings in the file;
* missing resource limits, autoscaling or private networking, fewer than two
  alerts, a production environment that runs fewer than two instances or can
  be destroyed.

It reads text. It does not evaluate expressions, resolve modules or know the
provider's schema, so it cannot tell that a file is valid Terraform and it
cannot see what a variable will hold. A pass means the files declare what the
policy requires, nothing more. The reader supports the subset the repository
uses (one attribute per line, no heredocs) and raises on anything else.

`terraform_checks.sh` runs `fmt -check` and, per module and environment,
`init -backend=false` and `validate`, only if `terraform` or `tofu` is on the
path; otherwise it prints that it skipped. The `infrastructure` job in
`ci.yml` installs Terraform with a pinned action, so `fmt` and `validate` run
there. `plan` needs provider credentials and is not part of CI; run it per
environment with the credentials of that environment.

## Destroy without orphans

After `terraform destroy` for an environment, save the listings and check them:

```bash
terraform -chdir=deploy/terraform/aws/environments/dev state list > state.txt
aws resourcegroupstaggingapi get-resources --tag-filters Key=scmle-environment,Values=dev > listing.json
python scripts/check_destroy_orphans.py --provider aws --environment dev --listing listing.json --state state.txt

gcloud asset search-all-resources --scope=projects/PROJECT --query='labels.scmle-environment=dev' --format=json > listing.json
python scripts/check_destroy_orphans.py --provider gcp --environment dev --listing listing.json --state state.txt
```

Exit 0 means nothing remains in the state or in the listing. Resources the
provider removes later by design (a KMS key in its deletion window, a secret
scheduled for deletion, a GCP resource in `DELETE_REQUESTED`) are listed as
pending, not as orphans. Dev and staging set `destroyable = true` so buckets
and registries do not block the destroy; production keeps it `false`. A
workload identity pool deleted on GCP cannot reuse its ID for 30 days.

## Not claimed

Applying these modules, a clean destroy on a real account, recorded cost,
multi-region availability, and any cost or latency equivalence between the two
providers.
