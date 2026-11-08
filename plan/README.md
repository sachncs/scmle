# Plan: what is left to build

Product definition: [`../product/README.md`](../product/README.md). Features
still to build: [`../product/features.md`](../product/features.md). Shared names
and principles: [`00-canon.md`](00-canon.md). Every open item, with a class:
[`COMPLETION_AUDIT.md`](COMPLETION_AUDIT.md).

The data-stage instrument, problem framing, ingestion, causal identification,
the decision layer, interval guarantees, the closed loop and the local half of
the cloud work are built. This plan covers only what is missing: trustworthy
evidence, unattended operation, the remaining methods, evidence from a real
cloud, and the external validation no one inside this project can supply.

## State on 2026-11-08 (`master` at `9356b89`)

* `ci.yml` is green on this head: 23 of 23 jobs (run 42471349728), including
  Windows 3.11 and 3.14, which failed on `f453091` because text artifacts were
  written with CRLF. Windows jobs take 24 and 27 minutes.
* The scheduled `full-suite.yml` passes its test step (the oracle table now
  holds on Linux 3.14) but its mutation step has failed since 4 Nov because
  mutmut instruments the hashed detector sources. A skip is committed; one more
  run confirms it.
* Nothing is published (PyPI returns 404); `release.yml`, `cloud-release.yml`
  and `cloud-deploy.yml` have never run.
* Nothing has run on AWS or GCP, and the production claim has not been made.
* The envelope, the controller and `scmle operate` exist and are tested
  against simulated worlds only; the F13 alarm does not gate retraining, so the
  controller recommends instead of training, and every promotion needs a
  person.
* 155 open items: 53 class A, 52 class B, 26 class C, 24 class D.

## Stages

| Stage | File | What it delivers | Depends on | Open items |
|---|---|---|---|---|
| 1 | [1-trustworthy-evidence.md](1-trustworthy-evidence.md) | Green cross-platform CI, one run read job by job, a quickstart that runs as written | none | 3 |
| 2 | [2-autonomous-operation.md](2-autonomous-operation.md) | Alarm-gated retraining, a staging run, and the controller's remaining pieces (the envelope, controller, `scmle operate`, simulation and `docs/AUTONOMY.md` are built); every promotion stays human-approved | 1 | 5 |
| 3 | [3-methods-and-instrument.md](3-methods-and-instrument.md) | The class A methods and instrument items, each with a pre-registered design and certificate | 1 | 43 |
| 4 | [4-cloud-and-production.md](4-cloud-and-production.md) | The container path, security, training, orchestration, infrastructure, release gates, observation, recovery and the production claim on AWS and GCP; trail and breadth | 1; 2.5 needs 4.20 | 31 |
| 5 | [5-external-validation-and-release.md](5-external-validation-and-release.md) | Corpora, backtest, red team, replication, R references, gold corpora, owner decisions, the release | 1; the sign-offs rest on 3 | 73 |

```
1 --+--> 2 -----------------------------+
    +--> 3 ---------------------------+ |
    +--> 4 ---------------------------+-+--> 5 (release last)
```

Stages 2, 3 and 4 are independent of each other once stage 1 is done, except
that item 2.5 (autonomy on a real deployment) needs 4.20. Stage 5 starts early,
because it needs people outside the project, and finishes last.

## Raw material and referents

Start from source systems, event logs with true write times, schema and ETL
history, the business process the model must beat, and past incidents, not from
cleaned benchmark tables. Where those are unavailable the affected claims are
tiered `unverified` (P4). What shows a result is right, other than the author or
the user: labelled corpora, reference implementations, randomized experiment
outcomes, a prospective head-to-head against practitioners, a red-team analyst,
independent replication and a domain owner's recorded sign-off. See
[`validation-referents.md`](validation-referents.md).

## Definition of done

A stage is done only when its exit gate passes, every detector or method that
gates has a certificate on file, and its external referent has agreed. The
release is done only when the stage 5 go/no-go checklist is signed by someone
other than the author. No claim of autonomy is made beyond what a prospective
head-to-head against practitioners supports, no claim of autonomous operation is
made beyond the simulated worlds and the declared envelope until item 2.5 passes
on a real deployment, and "production-qualified on AWS and GCP" is claimed only
when F14 items 1 to 13 pass on both providers (item 4.23). A failed, cancelled
or unread CI job is never a pass.
