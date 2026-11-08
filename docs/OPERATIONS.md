# Operating a deployed model

What `scmle.operations` does once a model is serving: measure it, tell a
fault from a drift from a worse model, tell people, keep evaluating, replace
it only through the gate, recover from failures, and decide whether the
production claim may be made. Everything here ran locally; nothing ran on AWS
or GCP. Evidence: `plan/evidence/phase-17/local/`. Feature:
F14 items 11 to 13.

| Module | What it is |
| --- | --- |
| `metrics` | Counters, gauges and histograms with a Prometheus text exposition; request, latency, error, availability and resource series per model version. |
| `logs` | JSON lines with `trace_id`, `model_version` and `artifact_digest`; W3C `traceparent`; credential-like fields are replaced. |
| `signals` | Windowed monitors over requests, feature rows and delayed labels, with configurable window, minimum samples, sampling rate and thresholds. |
| `classify` | Infrastructure, data quality, model performance, mixed or indeterminate. |
| `exporters` | CloudWatch (embedded metric format, `PutMetricData`) and Cloud Monitoring (`timeSeries.create`) payloads, checked against recorded shapes. |
| `notify` | Pager and ticket webhooks: HMAC-signed, retried, idempotent, injected transport. |
| `respond` | Classify, notify, retain telemetry, audit the operator's actions. |
| `cost` | Cost by service and deployed version from declared rates. |
| `continuous` | Labels and outcomes into the F13 loop, the retraining policy, history, gated replacement, the post-promotion check. |
| `recovery`, `faults`, `spool` | The fault catalogue, the objectives, the local scenarios, the telemetry spool. |
| `claim` | The production claim and the checker that guards it. |

## Telemetry

`ServiceMetrics` counts requests by route and status and observes latency.
*Availability* is the share of requests that were not 5xx or 429; a 4xx the
caller caused is available. Latency quantiles are estimated from histogram
buckets, as Prometheus does, so they are exact only at bucket edges.
`SignalMonitor` looks at a sliding window and reports, per signal, `ok`,
`breached` or `insufficient_data`. A window with too few samples is never
reported healthy. Feature rows are sampled at `sample_rate`; requests and
outcomes are never sampled. Drift is the noise-adjusted PSI of
`scmle.monitoring.reference`, so the thresholds are the ones the batch
monitor already uses (`MonitoringThresholds`).

## Classifying a fault

| Verdict | When |
| --- | --- |
| `infrastructure` | Only server errors, availability or latency are breached. |
| `data_quality` | Only rejected inputs, missing-rate increase, unseen categories or drift are breached. |
| `model_performance` | Only the loss over arrived labels is above the reference plus tolerance. |
| `mixed` | Breaches of more than one kind. |
| `indeterminate` | No signal had enough data. |

`mixed` is deliberate. A data fault can cause a performance drop and an
outage can bias the labels that arrive, so the classifier does not rank them;
it lists every breach and starts the responder at infrastructure, then data.
Caveats travel with the verdict: drift before any label arrived "is not harm",
and a performance drop with the data signals unjudged "does not exclude a data
cause". Critical severity pages and files a ticket; warning files a ticket.

## Notifications

Each webhook body is signed `sha256=HMAC(secret, timestamp + "." + body)` in
`X-Scmle-Signature`, with `X-Scmle-Timestamp` and an idempotency key that is
the same on every retry. A receiver calls `notify.verify_signature`. 5xx, 429
and transport errors retry with exponential backoff; other 4xx are final. A
failed page is returned and written to the audit row, never dropped.

## Continuous evaluation

Delayed labels become 0/1 losses and business outcomes become
`min(outcome / cap, 1)` losses (the cap is declared by the metric's owner);
both go to `Loop.mature`, for the serving model and for each challenger on the
same cases. Raw labels and outcomes are kept in `retained_outcomes.jsonl`.

The loop's own alarm still does **not** gate retraining. `alarm_gates_retrain`
stays False because the loop's detection-delay gates (H2, H10) missed their
floors; the alarm records a row and opens an `l1_alarm_investigate`
escalation, and trains nothing. Retraining comes from `RetrainPolicy`:

* `recommend` (the default) opens a `retrain_recommendation` escalation with
  `no_retrain` as the proposed default. A challenger is trained only after a
  person's signed, allowing reply (`run_approved_retrain`).
* `trigger` also trains a challenger at once and registers it with the loop.
  A challenger serves nothing until it wins.

A primary reason is required (loss above reference plus tolerance with enough
outcomes, a loop alarm, or age). Drift supports a reason and is never enough
alone. A proposal is held back while an infrastructure or data-quality fault is
open, and during a cooldown.

A challenger is judged by the loop's e-process and e-LOND (online false
discovery control over challengers); a win is a `promotion_proposal` row.
`ContinuousEvaluation.replace` then refuses unless the loop proposed the
challenger, the promotion gate admits the build (signed bundle, independent
approval, passed scan, approver other than the requester, a signed staging
record for production), an offline comparison, if given, says promote, and the
release registry confirms the runtime. Dataset versions (content digest, with
a preserved copy), training runs, proposals, replacements and checks are in a
hash-chained `training_history.jsonl`.

`post_promotion_check` compares chosen operational and business metrics
before and after with bootstrap intervals, split over the metrics so they hold
together. A metric is `improved` or `worse` only when its whole interval is
beyond the declared margin. The answer is `helped`, `harmed` or
`inconclusive`, and it always says that two periods cannot separate the model
from seasonality or other changes. The committed example is `inconclusive`.

## Recovery

Objectives are declared before a run in `recovery.OBJECTIVES` and are not
tuned to a result:

| Component | RTO | RPO (acknowledged units lost) | Scope |
| --- | --- | --- | --- |
| service | 30 s | 0 answers that differ | restart to ready |
| ledger | 10 s | 0 rows | reload and verify after the writer is killed |
| registry_state | 10 s | 0 writes | atomic state-file writes killed at random |
| object_store | 30 s | 0 objects | the local emulator, access restored to a verified bundle serving |
| secrets | 60 s | none (held outside) | detection of a credential mismatch to ready |
| telemetry_spool | 30 s | 0 samples | destination back to spool drained (duplicates possible) |
| job_state | 60 s | 0 completed steps | killed run continued, lease wait included |

`faults.run_all()` runs eleven faults against the real local service (the
generated `serve.py` as a subprocess), the local object-store and metrics
emulators, a real ledger writer and a real workflow executor, and records for
each component the measured RTO and RPO and how they were measured. In
`bad_release` the revisions are local processes behind a control plane that
implements the ECS and Cloud Run adapter interface, so the release registry's
promote, confirm, restore and rollback logic runs for real against them.

Not run: the rollback to a real managed revision, a zone or region loss, a
container orchestrator's health-driven replacement, IAM or bucket-policy
failures, a managed-service outage, secret-manager rotation, and on any
machine a crash, power loss, full disk or network partition. The ledger and
state files are shown to survive a killed process; no `fsync` guarantee
against power loss is claimed.

## The production claim

[`RELEASING.md`](RELEASING.md) holds the fixed sentence and says it has not
been made. `scripts/check_production_claim.py` refuses it unless both
providers' signed staging records are complete and a reviewer other than the
author has signed them.

## Not claimed

Anything on a cloud. The local scenarios measure these processes, emulators
and files under the stated methods; they are not availability, not a cloud's
behaviour, and not a recovery objective for a provider. The exporters were
checked against shapes written from the API references, not against a
service. The controlled degradation is a simulated stream with labels that
arrive at once.
