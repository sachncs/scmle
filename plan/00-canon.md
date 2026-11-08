# Canon: principles, vocabulary and identifiers

Every file in `plan/` and `product/` uses these names. Change them here first.

## The product in one paragraph

SCMLE is a machine-learning lifecycle with a built **qualification
instrument** for the data stage: from a signed analysis charter it fixes the
question and the estimand, splits the data into custody-controlled sets,
explores only the exploration set under a hypothesis ledger, certifies its own
detectors against planted defects, and issues claims whose error rate is
guaranteed only by a single confirmation on untouched data. Around that
instrument the lifecycle is built end to end: problem framing and the
decision-cost model, raw-data ingestion with point-in-time features, causal
identification, the decision and rollout layer, interval guarantees, the
closed production loop with escalation, and a local portable service with a
release gate. What is missing is unattended operation (a controller that runs
the loop inside a human-signed envelope), evidence from a real cloud, and
validation by people and data outside this project. Those are the five stages
in this folder. Whatever a charter or a person cannot supply is reported as
`unverified`, never silently assumed.

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
| Sets | `X` exploration, `C` confirmation (opened once), `F` forward-in-time holdout (temporal data only). The model's final-evaluation set is `F` for temporal data and has manifest role `final_test` otherwise; `C` is a separate carve-out so confirmed claims never shape the model that `C` would score. In a charter run `C` and the final test set are held in a separate vault process regardless of `enforcement`; the residual custody gaps are stated in `docs/SECURITY.md`. |
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

Feature IDs of the built instrument, kept for cross-reference: F1 charter,
F2 custody and splits, F3 claim ledger and confirmation, F4 admissibility and
provenance, F5 detectors with certificates, F6 robustness and shift. Their
code is `scmle/data/{charter,custody,ledger,admissibility,detectors,robustness}/`
and `benchmarks/qualification/`.

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
- Decision claims: `confirmed` when a frozen rule was tested once on
  untouched data (or an e-process crossed its level with the sample-ratio check
  passing) and the logs are randomized with recorded probabilities; `attested`
  when costs, margins or propensities came from a named person; `unverified`
  for defaults, a failed calibration with the grid fallback, failed support or
  effective-sample gates, and non-randomized logs without an eligible
  causal strategy. An estimate that rests on non-randomized data inherits the
  causal tier of the effect. Costs are never verified by the instrument.
- LLM-drafted fields (F8) are `unverified` until a named human signs.

## Stages

| Stage | File | Contents |
|---|---|---|
| 1 | `1-trustworthy-evidence.md` | Green cross-platform CI and a quickstart that runs as written |
| 2 | `2-autonomous-operation.md` | Envelope, alarm-gated retraining, controller, simulation, docs; promotion stays human-approved |
| 3 | `3-methods-and-instrument.md` | Class A open items of the instrument and the framing, ingestion, causal, decision, uncertainty and loop features |
| 4 | `4-cloud-and-production.md` | AWS and GCP evidence, the production claim, trail and breadth |
| 5 | `5-external-validation-and-release.md` | Red team, replication, gold corpora, owner decisions, the release |

Feature IDs F7 to F16 and R1 are used in `../product/features.md`; the item
register is `COMPLETION_AUDIT.md`. Dependencies and the definition of done are
in `README.md`.

## Conventions for these documents

- State what is verified in the repo versus proposed. Cite a file path (and a
  symbol or test name) for verified claims.
- Every feature entry ends with acceptance criteria that a machine can check
  and an "external referent" that is neither the author nor the user.
- Prefer one rigorous method per need rather than a survey; say what was cut
  and why.
- Documents describe only what is missing. When something is built, delete its
  text from here and record it in `CHANGELOG.md`.

