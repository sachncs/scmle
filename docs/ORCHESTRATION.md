# Orchestration

One workflow definition for ingestion, training, evaluation, monitoring and
batch prediction (`scmle.workflow`), a local executor that runs it with durable
state, and Step Functions and Cloud Workflows definitions emitted from it as
data. Local half of F14 item 8; what needs a cloud account is listed at the
end.

## The definition

A `Workflow` is named, versioned `Step`s with explicit `depends_on`. A step
has a kind (`ingest`, `train`, `evaluate`, `monitor`, `batch_predict`,
`approval`), a timeout, a `RetryPolicy` (attempts, first delay, factor,
ceiling), an optional cost budget and an optional concurrency group. The
workflow adds an optional `Schedule`, `max_concurrent_runs`, a run cost budget,
group limits and `max_parallel`. Dependency cycles, unknown dependencies and
duplicate names are refused when the object is built.

**Automatic versus approved.** A step with `effect="approved"` may only
exist behind an `approval` step it depends on; the definition refuses
otherwise. `standard_workflow()` is ingest, train, evaluate, then monitor and
an approval in parallel, then batch prediction (approved).

## The local executor

State lives in an `ObjectStore` under `workflows/<name>/runs/<run id>/`.

| Concern | Behaviour |
| --- | --- |
| Durable run id | derived from workflow name, version and the caller's trigger key (a schedule slot, a request id); the same trigger is the same run |
| Idempotency key | hash of run id, step, parameters and upstream outputs; a step already succeeded with that key is not run again and the key is passed to the handler for external deduplication |
| Duplicate invocation | `start` with a finished trigger returns the record and runs nothing; a second executor on a run in flight gets `RunInFlightError` (lease) |
| Partial failure | the failing step is retried under its policy; dependents are `skipped`; independent steps finish; `resume` re-runs only what has not succeeded (an explicit act; `start` never retries a failed run) |
| Interrupted job | `StepInterruptedError` (or any `TransientError`) is retried with the same key and the declared backoff; `runner_handler` runs training through a `Runner`, whose own checkpoints make the retry a resume |
| Crashed executor | its lease expires (`lease_seconds`); another executor takes the run over and re-runs the step that was in flight |
| Timeouts | cooperative: handlers call `context.check()`; cloud engines enforce theirs |
| Cost | `context.spend` raises past the step's or the run's budget; a run over budget stops its remaining steps (`budget_stopped`) |
| Concurrency | `max_concurrent_runs` by leases, `max_parallel` and per-group limits for steps |
| Approval | the approval step waits (`waiting_approval`); a decision is an HMAC-signed record from a listed approver who is not the requester; a rejection or an expiry ends the run and the approved step never runs |

`Scheduler` turns the schedule into slots and applies the declared catch-up
policy to missed ones: `run_all` (bounded by `max_catch_up`), `latest_only`
or `skip` (only a slot still within a tenth of an interval). Run ids come from
the slot, so a scheduler that lost its state cannot repeat a finished slot. The
first tick does not backfill. A slot that cannot start because the workflow is
busy is `deferred` and stays due.

## Emitted definitions

`emit.to_step_functions` and `emit.to_cloud_workflows` return
`{"definition", "schedule", "controls"}`. Independent steps of a layer become a
`Parallel` state or a `parallel` block, retries and timeouts are native
fields, approval is a callback wait (`waitForTaskToken`,
`events.await_callback`) bounded by its expiry with a rejection going to a
failure, and the execution name is the idempotency key. The validators
(`validate_step_functions`, `validate_cloud_workflows`) check structure only:
required fields, state types, that every target exists in its scope, that
every state is reachable, retry and timeout values, parallel branches, unique
step names. Concurrency limits, cost budgets and groups cannot be written in
either language; they come back under `controls` for the infrastructure
(scheduler concurrency, billing alarms) to enforce.

## Not done here

Neither emitted document was deployed or accepted by Step Functions or Cloud
Workflows; the validators encode the documented structure and may miss rules
the services apply. No EventBridge or Cloud Scheduler job exists. No
duplicate, partial-failure, missed-schedule or interruption scenario ran on a
cloud engine, only on the local executor. The approval callback is not wired
to a notification channel or a UI, and cost budgets are not enforced by the
cloud definitions.
