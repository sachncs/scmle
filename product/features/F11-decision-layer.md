# F11: Decision layer

Vocabulary and principles: [`../../plan/00-canon.md`](../../plan/00-canon.md).
Plan: [Phase 11](../../plan/phase-11-decision-layer.md).

Status: the statistical core is built (`scmle/decision/`, the charter's
`decision` block, `DecisionBudget`, the `choose_decision_threshold` and
`FinalEvaluationStage` wiring, certificates in
`benchmarks/qualification/f11_certificates.json`). `git log` and `CHANGELOG.md`
hold the record. This file lists only what is open. Statements about code
carry `path:line` as read in the repo; anything marked *unverified* was
recalled, not checked against a source.

## Open items

### Not connected to the rest of the system

| Item | State |
|---|---|
| `LifecycleManager` bootstrap paths | `scmle.governance.rollout_hook.RolloutHook` now runs `Rollout` and `RollbackMonitor` through `LifecycleManager` and the deployment ledger with the `DecisionBudget`, but `LifecycleManager.check_rollback` still decides by the bootstrap interval and `review` by the fixed-sample comparison; those two paths are not yet descriptive. Independent approval is unchanged. |
| `DecisionSpec` migration | The run spec's `threshold_objective`, `false_positive_cost` and `false_negative_cost` (`scmle/contracts/tasks.py:493-500`) remain the source when no charter exists; a charter with a decision block overrides them. They are not removed or deprecated. |
| Ledger rows | A charter run still writes only the rollout plan and a one-row `rollout_decisions.jsonl` (the registration): no experiment runs inside a charter run. When `RolloutHook` runs an experiment, each batch of units is written to the deployment ledger as a `rollout_assignment` row (unit, arm, the probability that arm had) and the e-process decisions as `rollout_decision` rows. |
| Existing fairness code | `evaluation/fairness.py` constants (0.10 gap, 0.80 ratio), `mitigation.py` (`trigger_gap`, `min_reduction`, `max_accuracy_cost`) and `reweighing.py` still gate or report as before; they are not clients of `decision.constraints`. Reweighing as a mechanism is accepted in the charter but only `group_thresholds` is solved. |
| Policy classes | Only a one-parameter threshold on an estimated effect is fitted (`ope.fit_threshold_policy`). Trees of depth two and the complexity penalty beyond a per-treated-row cost are not built. |

### Statistical gaps

- **Net benefit band.** The simultaneous band missed its pre-registered
  coverage floor for independent rows at `n = 2000` (0.9375, Wilson 0.926 to
  0.947, floor 0.93), so it is descriptive and nothing gates on it. It is also
  conservative: at `n = 2000` it called a rule with a score separation of 1.0
  useful in none of 300 worlds. Fixing either needs a new pre-registration,
  not a retune.
- **Anytime-valid engine.** The mixture e-process is the author's
  construction; the guarantee is Ville's inequality applied to a mixture of
  martingales, and the finite-sample error was measured. The Waudby-Smith and
  Ramdas (arXiv:2010.09686), Howard et al. (arXiv:1810.08240) and Ramdas et al.
  (arXiv:2210.01948) papers were cited from their abstracts, not read; no
  `confseq` comparison was made. It is slow relative to the best single bet:
  the expected sample number at twice the minimum effect is 1.15 times a
  null-shifted fixed-horizon size for a binary metric and 1.68 times for a
  bounded one (centred pseudo-outcomes; the uncentred form was 6.9 times in a
  development look). The centring is the author's addition.
- **Overlapping experiments.** Levels follow the discount-only rule. Whether
  the e-LOND boost is valid when experiments overlap was not checked
  (Xu and Ramdas, arXiv:2311.06412); the boost is not used.
- **Non-stationarity.** The bounds cover the running average of the effect;
  the drift worlds measured false promotion and false rollback under a
  drifting effect, not detection under a changing one.
- **Missing outcomes.** No rule for outcomes missing differentially across
  arms; the `missingness_not_handled` label of the plan is not implemented.
- **Fairness bounds** treat rows as independent (Newcombe intervals); no
  cluster-aware satisfaction test. The infeasibility check uses three
  incompatible pairs recalled from the impossibility literature (*unverified*).
- **OPE** is certified on simulated binary-action bandits and on one real
  randomized data set (Hillstrom) whose fitted policy treats 98 percent of rows,
  so the real-data check barely tests a policy difference. Empirical-Bernstein
  lower bounds are not implemented.

### Referents not obtained

- Criteo uplift (CC BY-NC-SA 4.0, non-commercial; about 300 MB): not
  downloaded.
- Open Bandit Dataset: not downloaded.
- Hillstrom: downloaded from the publisher's page, SHA-256
  `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`, 64,000
  rows; no licence statement found. Not committed. The gates read it from the
  path in `SCMLE_HILLSTROM` and report `not_run` without it. No corpus manifest
  entry exists in `benchmarks/qualification/corpora.json`.
- R `dcurves` or `rmda` decision-curve output, Open Bandit Pipeline OPE output,
  `confseq` sequences: none installed or compared. The net benefit identity is
  checked against an independent loop implementation only.
- Python 3.11 and 3.14 bit identity of the ledger replay and the report hash:
  one interpreter measured.
- Outside-party reproduction of A6, B1 and B7: has not happened; the phase is
  not closed without it.

## Tiers and failure modes (kept)

Decision claims are tiered as in the canon: `confirmed` when a frozen rule was
tested once on untouched data (or an e-process crossed its level with the
sample-ratio check passing) and the logs are randomized with recorded
probabilities; `attested` when costs, margins or propensities came from a named
person; `unverified` for defaults, a failed calibration with the grid
fallback, failed support or ESS gates, and non-randomized logs without an
eligible F10 strategy. Costs are never verified by the instrument.

## Acceptance still to check by machine

- The Hillstrom and simulated gates are in the certificate file; a gate that
  could not run (data file absent) is `not_run` and descriptive.
- When the outside reproduction exists, record it in the certificate file and
  here.
