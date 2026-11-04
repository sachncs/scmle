# Plan: what is left to build

Product definition: [`../product/README.md`](../product/README.md).
Shared names and principles: [`00-canon.md`](00-canon.md).

The data-stage qualification instrument (charter, custody, ledger and
confirmation, admissibility, detectors, robustness) is built and its phase
documents were removed. This plan covers only what is missing between that
instrument and a system that can do the work of a data scientist without one,
plus the open items of the instrument itself and the external gates no one
inside this project can pass.

## Why these phases

An assessment of the product against a data scientist's job found the
mechanical middle (split, search, check, sign, monitor) covered and the rest
missing: framing the problem, raw data, causal questions, the decision and
rollout layer, interval guarantees, the production loop and escalation. Each
phase below closes one of those gaps under the same rule as the built
instrument (canon P1): nothing gates a run until its operating characteristic
is measured and an external referent agrees.

## Raw material and referents

Start from source systems, event logs with true write times, schema and ETL
history, the business process the model must beat, and past incidents, not
from cleaned benchmark tables. Where those are unavailable the affected claims
are tiered `unverified` (P4). What shows a result is right, other than the
author or the user: labelled corpora, reference implementations, randomized
experiment outcomes, a prospective head-to-head against practitioners, a
red-team analyst, independent replication and a domain owner's recorded
sign-off. See [`validation-referents.md`](validation-referents.md).

## Phases

| Phase | File | Feature | Depends on |
|---|---|---|---|
| 7 | [phase-7-instrument-residuals.md](phase-7-instrument-residuals.md) | R1: open items of the built instrument (in progress: leakage screen wired into charter runs, attestations built; A3-A6, B, C, D, E open) | none |
| 8 | [phase-8-problem-framing.md](phase-8-problem-framing.md) | F8: framing and the decision-cost model (partly built: cost model, baseline claim, drafter, questions; gold corpus, LLM gate and run wiring open) | 7 |
| 9 | [phase-9-raw-data-ingestion.md](phase-9-raw-data-ingestion.md) | F9: raw data, grain, joins, point-in-time features (library built; corpus gates and run wiring open) | 7 |
| 10 | [phase-10-causal-identification.md](phase-10-causal-identification.md) | F10: causal identification | 8, 9 |
| 11 | [phase-11-decision-layer.md](phase-11-decision-layer.md) | F11: decisions, OPE, A/B, rollout | 8, 10 |
| 12 | [phase-12-uncertainty-and-breadth.md](phase-12-uncertainty-and-breadth.md) | F12: coverage guarantees, problem types (partly built: library methods and simulation gates; run wiring and outside-data gates open) | 7 |
| 13 | [phase-13-closed-loop-and-escalation.md](phase-13-closed-loop-and-escalation.md) | F13: production loop, escalation | 11, 12 |
| 14 | [phase-14-portable-core.md](phase-14-portable-core.md) | F14 items 1, 2, 5: local environment, hardened artifact, serving contract | none |
| 15 | [phase-15-security-data-training-orchestration.md](phase-15-security-data-training-orchestration.md) | F14 items 4, 6, 7, 8: security, governed data, portable training, orchestration | 14 |
| 16 | [phase-16-infrastructure-and-release.md](phase-16-infrastructure-and-release.md) | F14 items 3, 9, 10: infrastructure as code, CI/CD gates, registry bound to the runtime | 14, 15 |
| 17 | [phase-17-operate-and-qualify.md](phase-17-operate-and-qualify.md) | F14 items 11, 12, 13: observability, continuous evaluation, recovery; the production claim | 16, 13 |
| 18 | [phase-18-lifecycle-evidence-and-breadth.md](phase-18-lifecycle-evidence-and-breadth.md) | F15: business trail, qualified breadth | 17 (item 14 may start earlier) |
| 6 | [phase-6-external-validation-and-release.md](phase-6-external-validation-and-release.md) | F7: backtest, red team, replication, release | overlaps all |

```
7 --+--> 8 --+--> 10 --> 11 --+
    |        |               +--> 13
    +--> 9 --+               |
    +--> 12 -----------------+
                    6 (external gates, throughout; release last)
```

Cloud track (independent of 8 to 13 except where noted): 14 → 15 → 16 → 17 →
18. Phases 14 and 15 are verifiable without a cloud account; 16 and 17 need
sandbox accounts, a spending cap and a named approver, and are not marked
done on local evidence. "Production-qualified on AWS and GCP" is claimed only
when F14 items 1 to 13 pass on both providers.

Parallel lanes: after 7, phases 8, 9 and 12 run concurrently. Phase 6's gates
need people outside the project, so they start early and finish last.

## Definition of done

A phase is done only when its exit gate passes, every detector or method that
gates has a certificate on file, and its external referent has agreed. The
release is done only when the phase 6 go/no-go checklist is signed by someone
other than the author. No claim of autonomy is made beyond what the
prospective head-to-head in F7 supports.
