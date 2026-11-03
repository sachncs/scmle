# Plan: building the qualification instrument

Product definition: [`../product/README.md`](../product/README.md).
Shared names and principles: [`00-canon.md`](00-canon.md).

## Why this version is deeper, not bigger

The earlier plan had about 40 checks and 10 implementation steps. This one
keeps seven features and gives each a gate only when it earns one (canon P1):
it changes a decision, has a measured operating characteristic, and has an
external reference. The rest is descriptive EDA.

## What a top practitioner would add (and this plan now includes)

1. Every check bound to an action in a decision map, so no check exists
   without a consequence (F1).
2. Detector certificates: measured false-alarm rate and recall on planted
   defects, with intervals (F5, F7).
3. A test of whether verdicts predict reality: do "block" datasets
   generalize worse out of time than "pass" ones (F7, phase 6).
4. A clinical-trial style estimand and tipping-point sensitivity for
   missing data (F1, F5).
5. An independent red team (phase 6).

## Rawest material to start from

Not the extract. Source-system change and event logs with true write times,
schema and ETL code with their git history, raw bytes, per-annotator labels,
serving-time feature logs, and data-incident tickets. In this repo: the git
history behind each threshold constant, the TabArena task list, and
`benchmarks/results*.json`. Where these are unavailable the affected claims are
tiered `unverified` (canon P4).

## Methods and tools

Online FDR (e-LOND, LORD++, SAFFRON), sample splitting and Thresholdout,
conditional permutation tests, Little's MCAR test with tipping-point analysis,
confident learning with prediction-powered inference, C2ST and kernel MMD,
specification curves and PCS, simulation-based calibration, Hypothesis and
mutmut for the instrument itself. Detail lives in each feature file.

## What shows it is right, other than the author or the user

Labelled corpora of known defects, differential tests against reference
implementations, forward-in-time and prospective outcomes, an independent
replication through `benchmarks/sota/PROTOCOL.md`, a red-team analyst, a
domain owner's recorded sign-off, and mutation testing. See
[`validation-referents.md`](validation-referents.md).

## Phases and parallel lanes

```
Phase 0  foundations
   |
   +--> Phase 1  verification harness (F7 skeleton)  ----+
   |                                                     v
   +--> Phase 2  charter + custody (F1 || F2) --+--> Phase 4  F4 || F5
                                                |         |
                                                +--> Phase 3  F3
                                                          |
                                      Phase 3 + Phase 4 --+--> Phase 5  F6
                                                                  |
                                                                  v
                                                          Phase 6  external validation + release
```

| Phase | File | Parallel lanes |
|---|---|---|
| 0 | [phase-0-foundations.md](phase-0-foundations.md) | types, budget algebra, determinism |
| 1 | [phase-1-verification-harness.md](phase-1-verification-harness.md) | injector, simulator, oracles, corpus manifest |
| 2 | [phase-2-charter-and-custody.md](phase-2-charter-and-custody.md) | F1 and F2 |
| 3 | [phase-3-ledger-and-confirmation.md](phase-3-ledger-and-confirmation.md) | ledger, FDR rules, controls, unlock |
| 4 | [phase-4-detectors-and-admissibility.md](phase-4-detectors-and-admissibility.md) | F4 and F5 |
| 5 | [phase-5-robustness-and-shift.md](phase-5-robustness-and-shift.md) | stability and shift |
| 6 | [phase-6-external-validation-and-release.md](phase-6-external-validation-and-release.md) | backtest, red team, replication, pilot |

Phases 1 and 2 run concurrently after phase 0; phases 3 and 4 run
concurrently after phases 1 and 2.

## Definition of done

A release is done only when every phase exit gate passes, every gating
detector has a certificate on file, and the phase 6 go/no-go checklist is
signed by someone other than the author.
