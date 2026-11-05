# Product: from a qualification instrument to an autonomous data scientist

Canonical names and principles: [`../plan/00-canon.md`](../plan/00-canon.md).

## Where the product stands

Built: a leakage-safe ML lifecycle with a data-stage qualification instrument
(signed charter, custody-controlled sets, hash-chained hypothesis ledger with
online FDR and one-time confirmation, admissibility and provenance, certified
detectors, robustness and shift checks). It runs unattended on a framed,
prepared table and degrades visibly when a person's facts are missing.

Not built: the parts of a data scientist's job around that. This folder
defines only those.

## What it is not

- Not a proof of correctness. Guarantees are probabilistic and conditional on
  recorded assumptions.
- Not a replacement for a data scientist today. The measured evidence is its
  own benchmarks, which are not a head-to-head with people (see F7).
- Not adversary-proof custody on a single machine.

## Features still to build

| ID | Feature | One line |
|---|---|---|
| R1 | [Instrument residuals](features/R1-instrument-residuals.md) | Open items of the built instrument: uncertified detectors, unwired checks, custody gaps, missing evidence. |
| F7 | [Verification harness](features/F7-verification-harness.md) | Predictive-validity backtest, red team, replication, TabArena campaign, mutation floor. |
| F8 | [Problem framing](features/F8-problem-framing.md) | Plain-language request to a draft charter, decision-cost model and baseline, asking only what the data cannot answer. |
| F9 | [Raw-data ingestion](features/F9-raw-data-ingestion.md) | Source systems to a point-in-time-correct table: grain, keys, joins, entity resolution, feature synthesis. |
| F10 | [Causal identification](features/F10-causal-identification.md) | Estimand-driven strategies, diagnostics, sensitivity, and refusal when not identified. |
| F11 | [Decision layer](features/F11-decision-layer.md) | Cost-optimal thresholds, off-policy evaluation, A/B design, anytime-valid promote and rollback. |
| F12 | [Uncertainty and breadth](features/F12-uncertainty-and-breadth.md) | Coverage-guaranteed intervals and sets; forecasting, ranking, survival and other problem types. |
| F13 | [Closed loop and escalation](features/F13-closed-loop-and-escalation.md) | Drift to retrain to gated promotion, label bias, and a measured rule for when to ask a human. |
| F14 | [Cloud portability and production operation](features/F14-cloud-portability-and-production-operation.md) | One lifecycle on local, AWS and GCP: hardened artifacts, security, serving, infrastructure, release gates, observability, recovery. |
| F15 | [Operational lifecycle and qualified breadth](features/F15-operational-lifecycle-and-qualified-breadth.md) | Traceable business trail from question to outcome; further task families and modalities, one qualified slice at a time. |

## Production standard

The product may claim production-qualified deployment on AWS and GCP only
when F14 items 1 to 13 have passed their acceptance on both providers, with
the exact application, model, data and configuration versions, the cloud
environment tested, the evidence and the responsible approver recorded. A
running container, a successful request, a registered model or an
infrastructure manifest is not that claim.

## Autonomy standard

The product may claim autonomy for a task only at the strength a prospective,
pre-registered head-to-head against practitioners on unseen raw problems
supports (F7). Until then each feature states its own measured operating
characteristic and nothing more.

## Evidence tiers

`confirmed` > `attested` > `unverified`. Shown on every claim (`scmle claims list`);
the run summary and `report.html` do not show them yet. The causal and
decision readings are in the canon.
