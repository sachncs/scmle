# Canon: principles, vocabulary and identifiers

Every file in `plan/` and `product/` uses these names. Change them here first.

## The product in one paragraph

SCMLE is a machine-learning lifecycle with a built **qualification
instrument** for the data stage: from a signed analysis charter it fixes the
question and the estimand, splits the data into custody-controlled sets,
explores only the exploration set under a hypothesis ledger, certifies its own
detectors against planted defects, and issues claims whose error rate is
guaranteed only by a single confirmation on untouched data. What is still
missing is everything around that instrument that a data scientist does by
hand: framing the problem, getting from raw sources to a table, causal
identification, the decision and rollout layer, interval guarantees across
problem types, the production loop, and knowing when to ask a person. Those
are the features and phases left in these folders. Whatever a charter or a
person cannot supply is reported as `unverified`, never silently assumed.

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
| Sets | `X` exploration, `C` confirmation (opened once), `F` forward-in-time holdout (temporal data only). The model's final-evaluation set is `F` for temporal data and has manifest role `final_test` otherwise; `C` is a separate carve-out so confirmed claims never shape the model that `C` would score. In a charter run `C` and the final test set are held in a separate vault process regardless of `enforcement`; the residual custody gaps are listed in R1. |
| Check | One statistical test or estimator with null, statistic, p-value or interval, effect size, n, decision. |
| Claim | A finding that survived exploration and is tested on `C`. |
| Ledger | Append-only record of every hypothesis spent, with alpha-wealth accounting. |
| Certificate | Measured operating characteristics of a detector (size, recall, intervals) by defect type. |
| Tier | `confirmed`, `attested`, `unverified`. |
| Error budget | `alpha_confirm`, exploration wealth `W0`, detector tolerances, permutation count `B`. |
| Decision-cost model | Chartered costs or utilities of each outcome and of the current business process; the source of metrics, thresholds and the baseline (F8, F11). |
| Identification strategy | A causal estimator plus the assumptions that make it valid, chosen as a function of what the charter declares (F10). |
| Decision budget | Exact-fraction budget for sequential promote, hold and rollback decisions, beside the error budget (F11). |
| Escalation | A measured rule for when the system stops and asks a named human; the reply is recorded as an `attested` row (F13). |

## Tier readings added for the new features

- Causal claims (built; `scmle/data/identification/tier.py`): `confirmed`
  only when all five hold: (1) the estimate and its interval were computed on
  `C`, opened once; (2) every design diagnostic passed at a certified
  threshold; (3) the pre-registered falsification battery ran and passed with
  at least one negative control declared; (4) the estimate replicated in at
  least two strata with one sign and no heterogeneity; (5) the sensitivity
  gate holds at the declared strength. `attested` means every assumption has a
  named attestor and the design passed but a condition above is missing (no
  negative control, one treated unit, a weak instrument, an uncertified gate,
  an unchecked adjustment set); `unverified` means an attestor is missing, a
  design diagnostic failed or a falsification test rejected. `confirmed` does
  not say the assumptions are true; the held-out set alone never confirms a
  causal effect.
- Decision claims: tiered as F11 ("Tiers for decision claims") specifies; an
  estimate that rests on non-randomized data inherits the F10 tier of the
  effect.
- LLM-drafted fields (F8) are `unverified` until a named human signs.

## Already built (names only, for reference)

Shared types and the exact error budget, the verification harness with
detector certificates, the charter and custody, the hash-chained ledger with
online FDR and one-time confirmation, admissibility and provenance, five
certified detector families, and robustness and shift checks. Packages:
`scmle/data/{charter,custody,ledger,admissibility,detectors,robustness}/`,
`scmle/contracts/qualification.py`, `scmle/core/randomness.py`,
`benchmarks/qualification/`. Their specifications were removed from these
folders once implemented; `git log` and `CHANGELOG.md` hold the record and
`product/features/R1-instrument-residuals.md` lists what is still open in them.
The IDs of the built features are kept for cross-references: F1 charter, F2
custody and splits, F3 claim ledger and confirmation, F4 admissibility and
provenance, F5 detectors with certificates, F6 robustness and shift.

## Features (product/features/)

| ID | File |
|---|---|
| R1 | `R1-instrument-residuals.md` (open items of the built instrument) |
| F7 | `F7-verification-harness.md` (backtest, red team, replication, campaign) |
| F8 | `F8-problem-framing.md` |
| F9 | `F9-raw-data-ingestion.md` |
| F10 | `F10-causal-identification.md` |
| F11 | `F11-decision-layer.md` |
| F12 | `F12-uncertainty-and-breadth.md` |
| F13 | `F13-closed-loop-and-escalation.md` |
| F14 | `F14-cloud-portability-and-production-operation.md` (local, AWS and GCP: artifacts, security, serving, infrastructure, release, operation, recovery) |
| F15 | `F15-operational-lifecycle-and-qualified-breadth.md` (business trail, further task families and modalities) |

## Phases (plan/)

| Phase | File | Features |
|---|---|---|
| 6 | `phase-6-external-validation-and-release.md` | F7 (external gates, release) |
| 7 | `phase-7-instrument-residuals.md` | R1 |
| 8 | `phase-8-problem-framing.md` | F8 |
| 9 | `phase-9-raw-data-ingestion.md` | F9 |
| 10 | `phase-10-causal-identification.md` | F10 |
| 11 | `phase-11-decision-layer.md` | F11 |
| 12 | `phase-12-uncertainty-and-breadth.md` | F12 |
| 13 | `phase-13-closed-loop-and-escalation.md` | F13 |
| 14 | `phase-14-portable-core.md` | F14 items 1, 2, 5 |
| 15 | `phase-15-security-data-training-orchestration.md` | F14 items 4, 6, 7, 8 |
| 16 | `phase-16-infrastructure-and-release.md` | F14 items 3, 9, 10 |
| 17 | `phase-17-operate-and-qualify.md` | F14 items 11, 12, 13 |
| 18 | `phase-18-lifecycle-evidence-and-breadth.md` | F15 |

Cloud track: 14 first (no cloud account needed), then 15, 16, 17; 18 after
17. Cloud acceptance needs accounts, a budget cap and an approver.

Dependencies: 7 first (closes the built instrument). 8 and 9 in parallel after
7. 10 needs 9 (point-in-time data) and 8 (the charter's causal fields). 11
needs 8 and 10. 12 is independent after 7. 13 needs 11 and 12. 6 runs last and
overlaps everything it can: its release gates need people outside the
project.

## Conventions for these documents

- State what is verified in the repo versus proposed. Cite `path:line` for
  verified claims.
- Every feature file ends with acceptance criteria that a machine can check
  and an "external referent" that is neither the author nor the user.
- Prefer one rigorous method per need rather than a survey; say what was cut
  and why.
- Documents describe only what is missing. When something is built, delete its
  text from here and record it in `CHANGELOG.md`.

