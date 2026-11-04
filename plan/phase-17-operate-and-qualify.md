# Phase 17: Operate, evaluate continuously, recover; the production claim

Feature: [F14](../product/features/F14-cloud-portability-and-production-operation.md)
items 11, 12 and 13, then the final release gate. Depends on
[Phase 16](phase-16-infrastructure-and-release.md) and on the built Phase 13
closed loop (`scmle/governance/loop.py`).

## Work packages

- **WP-17.1 Observability and incidents (item 11).** Request counts, latency
  distributions, error rates, resource use and availability; structured logs
  with trace correlation and model-version attribution; data drift,
  prediction drift, feature quality and delayed-label performance with
  configurable windows, sampling and thresholds; exporters for CloudWatch
  and Cloud Monitoring; pager and ticket webhooks; a classifier that
  separates infrastructure faults, data-quality issues and model-performance
  degradation; cost reporting by service and deployed version.
- **WP-17.2 Continuous evaluation (item 12).** Feed delayed labels and
  observed business outcomes to the F13 loop. Retraining is triggered or
  recommended by an explicit policy; datasets, evidence and training history
  are preserved; a challenger is compared with the champion; replacing the
  production model needs the declared approval; after promotion the selected
  operational and business metrics are checked to see whether the model
  helped. Note that the F13 alarm does not yet gate retraining because its
  detection-delay gates missed their floors; this work package must use the
  certified path (escalate) until that is re-measured.
- **WP-17.3 Recovery (item 13).** A fault catalogue and runner covering
  process kill, unhealthy container and rolling restart, invalid
  credentials, lost object-store access, dependency outage, corrupted
  bundle, incompatible schema, bad release, job interruption with persistent
  state recovery, real rollback to a managed revision, audit-log continuity
  and secret-rotation recovery. Define and measure RTO and RPO for each
  stateful component. Record real environment failures, not only simulated
  fault artifacts.
- **WP-17.4 The production claim.** Run all P0 acceptance on both providers in
  staging, store the evidence, and obtain a reviewer-approved sign-off. The
  claim text is fixed in `docs/RELEASING.md` before the run.

## Gate

An injected service failure and a controlled drift event produce correctly
classified alerts, retained telemetry and an auditable operator response. A
controlled degradation produces the expected retraining proposal or
escalation, and a new model is deployed only after its tests and approvals
pass. Each provider passes the declared recovery scenarios within the
documented RTO and RPO.

## Evidence

`plan/evidence/phase-17/{aws,gcp}/`: alert and telemetry captures, operator
records, recovery timings, reviewer sign-off.

## Not claimed

Anything outside the declared scenarios. A passing recovery run is a
measurement of those scenarios, not of availability in general.
