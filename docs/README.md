# Documentation

Pick the page that matches what you are trying to do.

## New here? Read these first

You need no data-science background. Start with the first page and follow it
in order.

| Need | Page |
| --- | --- |
| Build, check, ship, monitor and roll back a first model, step by step | [`START_HERE.md`](START_HERE.md) |
| Every word you will meet, with an example and why it matters | [`GLOSSARY.md`](GLOSSARY.md) |
| The questions to answer before you run (target, metric, grouping, time, costs, what not to use) | [`DECISIONS.md`](DECISIONS.md) |
| How to read the report, the tiers and the claims; good versus suspicious results | [`READING_RESULTS.md`](READING_RESULTS.md) |
| Every refusal, gate and alarm: what it means and what to do | [`WHEN_THINGS_GO_WRONG.md`](WHEN_THINGS_GO_WRONG.md) |
| What SCMLE does not do or has not proven | [`WHAT_SCMLE_CANNOT_DO.md`](WHAT_SCMLE_CANNOT_DO.md) |
| What went wrong or was confusing when the beginner path was run | [`BEGINNER_FINDINGS.md`](BEGINNER_FINDINGS.md) |

## Use it

| Need | Page |
| --- | --- |
| Learn by example, end to end | [`TUTORIAL.md`](TUTORIAL.md) and [`../examples/`](../examples/README.md) |
| Run it from a shell | [`CLI.md`](CLI.md) |
| Know what a run leaves behind | [`ARTIFACTS.md`](ARTIFACTS.md) |
| See which parts of the API are stable | [`API_STABILITY.md`](API_STABILITY.md) |
| Choose extras and see dependency policy | [`DEPENDENCIES.md`](DEPENDENCIES.md) |

## Understand it

| Need | Page |
| --- | --- |
| How the system is built and extended | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| How evidence and feedback drive changes | [`OPERATING_LOOP.md`](OPERATING_LOOP.md) |
| Which cited methods were checked against their papers | [`LITERATURE_CHECK.md`](LITERATURE_CHECK.md) |
| How well it works, and how we check | [`BENCHMARKS.md`](BENCHMARKS.md) and [`../benchmarks/`](../benchmarks/README.md) |

## Operate and release it

| Need | Page |
| --- | --- |
| Threat model and controls | [`SECURITY.md`](SECURITY.md) (reporting: [`../SECURITY.md`](../SECURITY.md)) |
| Run the whole lifecycle locally, no cloud account | [`LOCAL.md`](LOCAL.md) |
| What the service promises: endpoints, limits, objectives | [`SERVICE_SLO.md`](SERVICE_SLO.md) |
| Who may call the service, tenant isolation, audit | [`TENANT_ISOLATION.md`](TENANT_ISOLATION.md) |
| One workflow for ingest, train, evaluate, monitor, batch predict; approval, schedule | [`ORCHESTRATION.md`](ORCHESTRATION.md) |
| Run one job spec locally, on AWS or on GCP; roles, budget, resume | [`PORTABLE_TRAINING.md`](PORTABLE_TRAINING.md) |
| Object stores, snapshots, quality gates, incremental processing | [`GOVERNED_DATA.md`](GOVERNED_DATA.md) |
| Terraform for AWS and GCP, the policy check, destroy without orphans | [`INFRASTRUCTURE.md`](INFRASTRUCTURE.md) |
| Release a model to a cloud: gate, live tests, record, promote, roll back, reconcile | [`CLOUD_RELEASE.md`](CLOUD_RELEASE.md) |
| Operate a deployed model: telemetry, incidents, retraining, recovery, the production claim | [`OPERATIONS.md`](OPERATIONS.md) |
| Prove recovery works before release | [`OPERATIONS_REHEARSAL.md`](OPERATIONS_REHEARSAL.md) |
| Cut a release | [`RELEASING.md`](RELEASING.md) and [`../scripts/`](../scripts/README.md) |

The hosted site, [sachncs.github.io/scmle](https://sachncs.github.io/scmle/),
is built from `website/` and covers the same ground in a browsable form.
