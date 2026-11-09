# Portable training

One job spec and a `Runner` interface (`scmle.runners`). The package ships the
local runner; a runner for any other place is written by the user against the
same interface.

## The spec and the run

`Job` names a durable `run_id` (the idempotency key), the project
specification (opaque JSON), the training identity, resources, a timeout, a
`Retries`, an optional `CostBudget` and a non-secret environment. A GPU
needs a stated `gpu_reason`: specialised compute only behind a declared need.
`Runner.run(spec)` is the same on every runner:

* the spec is stored at `runs/<id>/spec.json`; the same id with a different
  spec is `RunConflictError`; a finished id returns its recorded manifest and
  starts nothing;
* `runs/<id>/manifest.json` is rewritten atomically at every transition
  (attempts with state, seconds, cost and reason; checkpoint digests; result
  digest; evidence list; final state);
* an attempt that ends **interrupted** (preemption, spot reclaim, SIGTERM) or
  **failed with a retryable cause** is retried after the policy's backoff, up
  to its attempt limit, and the retry resumes from the run's checkpoints;
  other failures, timeouts and budget stops are final;
* the **cost budget** is the declared rate times resources times elapsed time,
  accumulated across attempts; past the ceiling the job is cancelled and the
  run ends `budget_stopped`. It is an estimate from declared rates, not
  billing data;
* final states: `succeeded`, `failed`, `budget_stopped`, `timed_out`,
  `retries_exhausted`.

Workloads save progress with `context.step(n, state)` and read it back with
`context.resume()`. Each checkpoint carries its own digest; a damaged newest
one is skipped and the one before it is used.

## Roles

Training and serving are different identities and different permissions
(`scmle.runners.roles`). Training reads `data/`, `runs/`, `artifacts/` and
writes `runs/`, `artifacts/`, `evidence/`; serving reads `bundles/`,
`artifacts/` and writes `audit/`. `check_separation` refuses overlapping
write prefixes, a training role that can write `bundles/` and a serving role
that can write `runs/`. Workloads receive a `RoleStore` that enforces the
training role. A job spec naming the serving identity is refused by every
runner. Releasing a bundle is a third act, with an approval, outside both.

## Runners

| Runner | Runs the attempt as | Identity | Interruption |
| --- | --- | --- | --- |
| `Runner` | the workload, in this process | the training role of `RoleStore` | injected, or `JobInterruptedError` |

A runner for a container platform starts `python -m scmle.runners.entrypoint`
in the image, which opens the `file://` store named by `SCMLE_STORE_URI`, loads
the stored spec, refuses it if its digest is not the one the runner started,
and runs the workload (`SCMLE_WORKLOAD`, default
`scmle.runners.project:run_project`). `run_project` runs an unchanged
`Project`: it copies the data from the store, mirrors the run directory to
`runs/<id>/workdir/` every 30 seconds and at the end, and after a restart on an
empty disk restores it and calls `Project.resume`. A hard kill loses at most
the trials since the last mirror. `reproducibility.compare` reports each
metric of a result against tolerances declared before the runs; a miss is
published, not hidden.

## Not done here

Only the local runner ships and is tested. No container image was built, no
job was interrupted on a remote platform, and no cost was measured against a
bill. The reproducibility-tolerance report against a second place does not
exist.
