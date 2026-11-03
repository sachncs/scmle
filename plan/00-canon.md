# Canon: principles, vocabulary and identifiers

Every file in `plan/` and `product/` uses these names. Change them here first.

## The product in one paragraph

A **qualification instrument** for the data stage of SCMLE. From a signed
analysis charter it runs unattended: it fixes the question and the estimand,
splits the data into custody-controlled sets, explores only the exploration
set under a hypothesis ledger, certifies its own detectors against planted
defects, and issues claims whose error rate is guaranteed only by a single
confirmation on untouched data. Whatever the charter cannot supply is reported
as `unverified`, never silently assumed.

## Principles

- **P1 Gate discipline.** A check may gate a run only if (a) the charter's
  decision map binds it to a concrete action, (b) its operating
  characteristic (false-alarm rate, recall) has been measured and is on file
  as a certificate, and (c) it has an external reference (reference
  implementation, closed form or labelled corpus). All other output is
  descriptive EDA: shown, never gating.
- **P2 Where the guarantee comes from.** The final claim guarantee
  (`FDR <= alpha_confirm`) comes only from testing the exploration survivors
  once on the confirmation set. The exploration alpha-wealth is a triage
  budget that limits how many hypotheses the confirmation set must pay for.
  The two do not add.
- **P3 Who may see what.** Label-free, deterministic structural checks (key
  uniqueness across splits, dtype and range contracts) may read all rows.
  Anything that uses the target, or whose result changes a modelling decision,
  reads the exploration set only.
- **P4 Evidence tiers.** Every claim carries one of `confirmed` (passed fresh
  data), `attested` (a named human supplied the provenance) or `unverified`
  (a default was used).
- **P5 Autonomy with honest degradation.** The default path needs no human.
  Missing attestations lower the tier; they never stall the run or fake a pass.
- **P6 Stated limits.** Custody is tamper-evident and protects against
  in-code or accidental access; it is not adversary-proof on one machine.
  Statistics cannot prove a feature postdates nothing; as-of facts are
  attestations or log-derived, and tiered as such.

## Vocabulary

| Term | Meaning |
|---|---|
| Charter | Signed, hashed `AnalysisCharter`: estimand, unit, population, decision map, set sizes, error budget, multiverse declaration. |
| Estimand | Population, target variable, prediction moment, intercurrent events, summary measure (ICH E9(R1) style). Types: `predictive`, `causal`. |
| Decision map | Table binding each gating check to the action it triggers. |
| Sets | `X` exploration, `C` confirmation (opened once), `F` forward-in-time holdout (temporal data only). The existing locked test set is the model's final-evaluation set: it is `F` for temporal data and has manifest role `final_test` otherwise; `C` is a separate carve-out so confirmed claims never shape the model that `C` would score (see F2). Today rows move to a separate vault process whenever `enforcement="hard"` (the default, `scmle/contracts/tasks.py:363`; `scmle/core/stages.py:629-635`); explicit `cooperative` mode falls back to an in-process protocol. F2 decouples custody from that flag, extends it to `C` and `F`, and handles the caller-held-copy gap before "sealed" is claimed for them. |
| Check | One statistical test or estimator with null, statistic, p-value or interval, effect size, n, decision. |
| Claim | A finding that survived exploration and is tested on `C`. |
| Ledger | Append-only record of every hypothesis spent, with alpha-wealth accounting. |
| Certificate | Measured operating characteristics of a detector (size, recall, intervals) by defect type. |
| Tier | `confirmed`, `attested`, `unverified`. |
| Error budget | `alpha_confirm`, exploration wealth `W0`, detector tolerances, permutation count `B`. |

## Features (product/features/)

| ID | File |
|---|---|
| F1 | `F1-charter-and-decision-map.md` |
| F2 | `F2-custody-and-splits.md` |
| F3 | `F3-claim-ledger-and-confirmation.md` |
| F4 | `F4-admissibility-and-provenance.md` (includes the leakage tests) |
| F5 | `F5-detectors-with-certificates.md` (missingness, labels, duplicates, sentinels, outliers) |
| F6 | `F6-robustness-and-shift.md` |
| F7 | `F7-verification-harness.md` |

## Phases (plan/)

| Phase | File | Features |
|---|---|---|
| 0 | `phase-0-foundations.md` | shared types, error-budget algebra, determinism |
| 1 | `phase-1-verification-harness.md` | F7 skeleton first |
| 2 | `phase-2-charter-and-custody.md` | F1, F2 |
| 3 | `phase-3-ledger-and-confirmation.md` | F3 |
| 4 | `phase-4-detectors-and-admissibility.md` | F4, F5 |
| 5 | `phase-5-robustness-and-shift.md` | F6 |
| 6 | `phase-6-external-validation-and-release.md` | F7 complete |

Dependencies: 0 -> {1, 2} in parallel -> {3 needs 2; 4 needs 1 and 2} in
parallel -> 5 -> 6.

## Proposed package map (under `scmle/`)

`data/charter/`, `data/custody/`, `data/ledger/`, `data/admissibility/`,
`data/detectors/`, `data/robustness/`, and `benchmarks/qualification/` for the
verification harness. The existing `data/profiling/` checks migrate into
`detectors/` and `ledger/` as they gain certificates. The existing
`vault.SealedTestSet` (see `core/stages.py`) is extended, not replaced.

## Conventions for these documents

- State what is verified in the repo versus proposed. Cite `path:line` for
  verified claims.
- Every feature file ends with acceptance criteria that a machine can check
  and an "external referent" that is neither the author nor the user.
- Prefer one rigorous method over a survey of five. Say what was cut and why.
