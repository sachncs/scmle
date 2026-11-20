# Portable training

One job spec and a `Runner` interface (`scmle.runners`). The package ships
three runners on this machine (inline, child process, container); a runner
for any other place is written by the user against the same interface.

## The spec and the run

`Job` names a durable `run_id` (the idempotency key), the project
specification (opaque JSON), the training identity, resources, a timeout, a
`Retries`, an optional `Budget` and a non-secret environment. A GPU
needs a stated `gpu_reason`: specialised compute only behind a declared need.
`Runner.run(spec)` is the same on every runner:

* the spec is stored at `runs/<id>/spec.json`; the same id with a different
  spec is `ConflictError`; a finished id returns its recorded manifest and
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
`artifacts/` and writes `audit/`.
`separate` refuses overlapping write prefixes, a training role that
can write `bundles/` and a serving role that can write `runs/`.
Workloads receive a `Store` that enforces the training role.
`Runner.run` refuses a job spec naming the serving identity before any
attempt starts. Releasing a bundle is a third act, with an approval, outside
both.

## Runners

| Runner | Runs the attempt as | Identity | Interruption |
| --- | --- | --- | --- |
| `Runner` | the workload, in this process | the training role of `Store` | injected, or `PreemptedError` |
| `Process` | `python -m scmle.runners.entrypoint`, a child process | the training role, in the child | SIGTERM, then SIGKILL after `grace` |
| `Container` | the same entrypoint in an image, through the Docker CLI | the training role, as uid 10001 on a read-only root | `docker stop`: SIGTERM, then SIGKILL |

`Process` and `Container` read an attempt's end from the entrypoint's exit
status (`base.Exit`: 0 succeeded, 1 failed, 75 transient, 143 interrupted by
SIGTERM, 137 killed) and keep its output at `runs/<id>/logs/<n>.txt`;
`Container` bind-mounts the store, drops every capability and applies the
job's resources as limits, and tells an out-of-memory kill from a
preemption. `python -m scmle.runners.local --runner process|container`
runs a stored spec from the command line.
A runner for a container platform starts `python -m scmle.runners.entrypoint`
in the image, which opens the `file://` store named by `SCMLE_STORE_URI`
(confined to `tenants/<tenant>/` when `SCMLE_TENANT` is set), loads
the stored spec, refuses it if its digest is not the one the runner started,
and runs the workload (`SCMLE_WORKLOAD`, default
`scmle.runners.project:project`). `project` runs an unchanged
`Project`: it copies the data from the store, mirrors the run directory to
`runs/<id>/workdir/` every 30 seconds and at the end, and after a restart on an
empty disk restores it and calls `Project.resume`. A hard kill loses at most
the trials since the last mirror. `reproducibility.compare` reports each
metric of a result against tolerances declared before the runs; a miss is
published, not hidden.

## The image

`Dockerfile` at the repository root builds the image that trains: the
released wheel on `deploy/requirements.lock`, every file installed with
`--require-hashes`, on a base pinned by the digest of its
multi-architecture index (`scmle.deploy.base_image.PINNED`; the placeholder
digest and a bare tag are refused), run as uid 10001.
`python -m scmle.deploy.packaging image --wheel dist/<wheel> --tag <tag>`
builds it and checks that `scmle --help` runs as non-root. The release
workflow's `image` job builds it, runs a tiny job through `Container` and
the runner and image tests against it, and `publish` waits for that job.

## Not done here

The process runner is tested on every platform. The container tests need a
built image named by `SCMLE_IMAGE` and are skipped without one; they run in
the `image` job. On a macOS developer machine Docker Desktop could not
bind-mount pytest's temporary directory, so they have not passed there. No job was
interrupted on a remote platform, and no cost was measured against a bill.
The reproducibility-tolerance report against a second place does not exist.
