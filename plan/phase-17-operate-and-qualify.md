# Phase 17: Operate, evaluate continuously, recover; the production claim

Feature: [F14](../product/features/F14-cloud-portability-and-production-operation.md)
items 11, 12 and 13, then the final release gate. Depends on
[Phase 16](phase-16-infrastructure-and-release.md) and on the built Phase 13
closed loop (`scmle/governance/loop.py`).

The code, tests and local evidence for the four work packages are built and
verified without a cloud account (`scmle.operations`, `docs/OPERATIONS.md`,
the fixed claim in `docs/RELEASING.md`). Evidence:
`plan/evidence/phase-17/local/` (regenerate with
`scripts/make_phase17_evidence.py`). The local run measured a local service,
emulators, a ledger writer and a workflow executor; exporters were checked
against shapes written from the API references; webhooks went to recording
transports; the replacement ran on a scripted control plane. **Nothing ran on
AWS or GCP and the production claim has not been made.** What remains needs an
AWS account, a GCP account, a staging environment per provider and a reviewer
other than the author.

## Open

- **WP-17.1 Observability, on each provider.** Write the metrics, logs and
  signals to CloudWatch and Cloud Monitoring from the staging service, with a
  real pager and ticket endpoint, and fix what differs from the recorded
  payload shapes. Inject a service failure and a controlled drift event in
  each provider's staging, keep the alert and telemetry captures, and have an
  operator acknowledge and resolve them. Close the known gaps: the signals
  are wired into the generated service (`HttpServing(monitor=...)`,
  `runtime.run` writes JSON lines to `SCMLE_SIGNALS_FILE`) but nothing reads them from a cloud; the
  cost report needs the provider's real rates and a comparison with the bill;
  the incident classifier's rules are untuned and have met no real incident.
- **WP-17.2 Continuous evaluation, in staging.** Run the controlled
  degradation against a deployed model with labels that arrive after a real
  delay and a real business outcome, and keep the proposal, the person's
  reply, the challenger's evidence and the replacement. The F13 alarm still
  does not gate retraining (`alarm_gates_retrain` stays False) until its
  detection-delay certificate passes on a re-measurement; until then
  retraining proposals go through the escalation path. Decide per product
  which business metrics and margins `post_promotion_check` watches (the
  owner's decision). The holdback is built (`holdback_assignment`,
  `holdback_check` with a sample-ratio guard); a deployment has to route the
  held-back units to the previous model.
- **WP-17.3 Recovery, on each provider.** Run every catalogued fault in each
  provider's staging and record RTO and RPO per stateful component against
  the objectives in `recovery.OBJECTIVES`: process and task kill, an unhealthy
  task with the platform's rolling replacement, revoked workload identity,
  denied bucket access, a managed-dependency outage, a corrupted object, a
  rollback to a real managed revision (`rollback_to_managed_revision`), job
  interruption in Step Functions and Cloud Workflows, and secret rotation in
  the provider's secret manager. Add the two the local run cannot see: a
  machine crash or power loss (the ledger and state files are shown only to
  survive a killed process) and a zone or region loss. Revisit the objectives
  with the owner; they were declared for a local run.
- **WP-17.4 The claim.** Run all P0 acceptance on both providers in staging,
  write `acceptance.json`, `deployment.json`, `recovery.json`,
  `observability.json` and `continuous-evaluation.json` under
  `plan/evidence/phase-17/{aws,gcp}/`, have a reviewer other than the author
  sign `signoff.json` with a key the pipeline does not hold, and run
  `scripts/check_production_claim.py`. Only then may the sentence in
  `docs/RELEASING.md` be used, and the status line there be changed.

## Gate (still open)

An injected service failure and a controlled drift event produce correctly
classified alerts, retained telemetry and an auditable operator response on
each provider. A controlled degradation produces the expected retraining
proposal or escalation, and a new model is deployed only after its tests and
approvals pass. Each provider passes the declared recovery scenarios within the
documented RTO and RPO. The local half of the gate passes: a killed local
service is classified as infrastructure and a shifted feed as data quality
(and, once labels get worse, as mixed), the recommendation reaches the
escalation path and trains a challenger only after a person's reply, the
gate refuses each seeded failure, and every local fault recovered within its
objective; see the evidence summary.

## Evidence

`plan/evidence/phase-17/local/`: what ran locally and what did not.
`plan/evidence/phase-17/{aws,gcp}/`: alert and telemetry captures, operator
records, recovery timings and the reviewer sign-off. They are empty, and the
claim checker refuses them.

## Not claimed

Anything outside the declared scenarios. A passing recovery run is a
measurement of those scenarios, not of availability in general. Nothing local
is a cloud result, and the production claim has not been made.
