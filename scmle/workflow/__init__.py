"""One workflow definition and a local executor.

`definition` is the single type (ingest, train, evaluate, monitor,
batch-predict and approval steps with dependencies, retries, timeouts,
budgets and concurrency groups); `executor` runs it locally with durable run
ids, idempotency keys and a human-approval gate; `schedule` decides which
runs are due, including after a missed schedule; `handlers` holds the step
handlers.
"""
