"""One workflow definition, a local executor, and cloud definitions as data.

`definition` is the single type (ingest, train, evaluate, monitor,
batch-predict and approval steps with dependencies, retries, timeouts,
budgets and concurrency groups); `executor` runs it locally with durable run
ids, idempotency keys and a human-approval gate; `schedule` decides which
runs are due, including after a missed schedule; `emit` renders AWS Step
Functions and GCP Cloud Workflows definitions and validates their structure.
"""
