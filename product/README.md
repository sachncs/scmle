# Product: from a qualification instrument to an autonomous data scientist

Canonical names and principles: [`../plan/00-canon.md`](../plan/00-canon.md).
What is left to build: [`features.md`](features.md). When, in what order and
with what exit gates: [`../plan/README.md`](../plan/README.md).

## Where the product stands

Built: a leakage-safe ML lifecycle with a data-stage qualification instrument
(signed charter, custody-controlled sets, hash-chained hypothesis ledger with
online FDR and one-time confirmation, admissibility and provenance, certified
detectors, robustness and shift checks), and around it problem framing with a
decision-cost model, raw-data ingestion with point-in-time features, causal
identification, a decision and rollout layer, interval guarantees, a closed
production loop with escalation, and a local portable service with a release
gate. It runs unattended on a framed, prepared table and degrades visibly when
a person's facts are missing.

Not built: unattended operation of the production loop, the container path on
a real daemon, and validation by people and data outside this project. Nothing
is published. `features.md` defines only these and the open items of the parts
above.

## What it is not

- Not a proof of correctness. Guarantees are probabilistic and conditional on
  recorded assumptions.
- Not a replacement for a data scientist today. The measured evidence is its
  own benchmarks, which are not a head-to-head with people (F7).
- Not adversary-proof custody on a single machine.

The per-item register of everything still open, with a class for each (A
buildable here, B outside people or data, C container daemon, real deployment
or approver, D owner
decision), is [`../plan/COMPLETION_AUDIT.md`](../plan/COMPLETION_AUDIT.md).

## Production standard

The product makes no production-qualified deployment claim. A running
container, a successful request or a registered model is not that claim.

## Autonomy standard

The product may claim autonomy for a task only at the strength a prospective,
pre-registered head-to-head against practitioners on unseen raw problems
supports (F7). Autonomous operation is claimed only inside the declared
envelope and the simulated worlds until a real staging deployment passes (F16),
and every promotion needs a person's signed approval. Until then each feature
states its own measured operating characteristic and nothing more.

## Evidence tiers

`confirmed` > `attested` > `unverified`. Shown on every claim (`scmle claims list`);
the run summary ends with the count per tier. The causal and decision readings
are in the canon.
