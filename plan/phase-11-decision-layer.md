# Phase 11: Decision layer (F11)

Names follow [`00-canon.md`](00-canon.md). Feature definition:
[`../product/features/F11-decision-layer.md`](../product/features/F11-decision-layer.md).
Needs Phase 2 (charter, custody), Phase 3 (budget, e-value rules, ledger) and
Phase 5 (shift). Policies that act on a treatment effect from non-randomized
logs need Phase 10; the offline and online statistics here do not.

## Goal

From a charter that declares a decision model, the instrument chooses the
operating threshold or policy, shows with an interval that it beats the
declared baselines over the declared range of cost ratios, tests declared
fairness constraints with an upper bound on the gap, designs the rollout, and
makes promote, hold and rollback calls whose error rates hold under
continuous monitoring and are accounted for in an exact budget. Every gate
has a measured operating characteristic and an external referent (P1).

## Entry criteria

- Phase 3 gate passes: ledger replay bit-identical; `Evidence(e=...)`
  accepted by e-LOND; `ErrorBudget` fractions exact.
- Final evaluation set opens once through custody (Phase 2).
- Decision recorded in the canon: the tier reading for decision claims (F11,
  "Tiers for decision claims") and the `DecisionBudget` name.
- Licence check done for each external dataset before download (Criteo's
  listed licence is non-commercial; if the project's use does not fit, the
  A6 and B7 referents are replaced and the plan says so).

## Verified starting points

- Cost objective and grid threshold: `scmle/models/thresholds.py:80-135`;
  spec fields `scmle/contracts/tasks.py:493-500`; call site
  `scmle/core/stages.py:1472-1527`; final stage `:1529`.
- Charter has no cost or decision model: `scmle/data/charter/contracts.py:148-162`,
  `:280-312`.
- Fairness reporting and mitigation: `scmle/evaluation/fairness.py:22-23`,
  `scmle/evaluation/mitigation.py:110`, `scmle/contracts/tasks.py:503-529`,
  `scmle/evaluation/reweighing.py:17`.
- Fixed-sample A/B and champion rules: `scmle/monitoring/abtest.py:99-166`,
  `scmle/monitoring/champion.py:165-170`; rollback rule
  `scmle/governance/lifecycle.py:328-406` (`:365`); approval-gated promotion
  `:193-213`.
- E-value plumbing: `scmle/data/ledger/online_fdr.py:174` (`Evidence`),
  `:238` (`calibrate`), `:356` (`ELond`), `:138` (`Discount`).
- Budget: `scmle/data/ledger/budget.py:66-102`, `:137`, `:245` (`spend`),
  `:277` (`split_budget`).
- Power closed forms: `scmle/data/charter/feasibility.py:79-135`.
- Existing IPW and ESS code (causal panels): `scmle/evaluation/sequential.py:500`,
  `:952`.
- No net benefit, no OPE, no sequential or anytime-valid test, no decision
  budget anywhere in `scmle/` (searched on the date of writing).

## Work packages

**WP-11.1 Charter block, decision budget** (one agent)
- `DecisionModel`, `FairnessConstraint`, `RolloutPlan` in
  `scmle/data/charter/contracts.py` with a format bump; `DecisionBudget`
  (exact `Fraction`s, own spend, `BudgetExhaustedError`) in
  `scmle/data/ledger/budget.py`; migration of the cost fields from
  `DecisionSpec`.
- Done when: charter hash changes with every new field; B5 budget algebra
  passes; a charter without the blocks behaves exactly as before (existing
  tests unchanged).

**WP-11.2 Thresholding and calibration gate** (one agent; depends on 11.1)
- `scmle/decision/threshold.py`: `t*` rule, calibration check with measured
  false-alarm rate, recalibration on development data only, grid fallback,
  cluster-bootstrap interval, cost-ratio robustness range. Wire into
  `choose_decision_threshold` (`stages.py:1472`).
- Done when: A1 passes against the closed form.

**WP-11.3 Net benefit and decision curves** (one agent; depends on 11.1;
parallel with 11.2)
- `netbenefit.py`: NB, baselines, simultaneous bootstrap band over
  `ratio_range`, the `useful` / `not_useful_in_range` claim, one-shot
  evaluation in `FinalEvaluationStage` (`stages.py:1529`).
- Done when: A2, A3, A4.

**WP-11.4 Off-policy evaluation** (one agent; depends on 11.1)
- `ope.py`: IPW, SNIPS, DR with influence-function intervals, declared
  clipping, support and ESS gates (reuse `sequential.effective_sample_size`),
  constrained policy classes and penalised DR policy fit on development data.
  Observational logs are accepted only through an eligible F10 strategy;
  otherwise the estimate is labelled `unverified`.
- Done when: A5 and A6.

**WP-11.5 Fairness constraints** (one agent; depends on 11.2, 11.3)
- `constraints.py`: constrained maximisation of net benefit (group
  thresholds, reweighing), upper-bound satisfaction test, frontier and price
  of the constraint with intervals, infeasibility refusal from measured base
  rates, small-group `not_assessable`. Existing constants in `fairness.py`
  and `mitigation.py` stop gating.
- Done when: A7 and A8.

**WP-11.6 Anytime-valid engine** (one agent; independent of 11.2-11.5; starts
after 11.1)
- `anytime.py`: bounded-mean e-process / confidence sequence for the
  randomization pseudo-observation, paired-difference variant, ledger
  `Evidence(e=...)` adapter. Read the cited papers (Waudby-Smith and Ramdas;
  Howard et al.) and the e-LOND paper's treatment of overlapping tests before
  coding; replace *unverified* tags in F11 with checked ones or delete the
  claims.
- Done when: B1 (including the measured peeking rate of the current
  `compare_arms`), B3.

**WP-11.7 Rollout design and decision rule** (one agent; depends on 11.1, 11.6)
- `design.py` (sample size from power, ESN by simulation, salted-hash
  assignment, sample-ratio check), `rollout.py` (promote / hold / rollback /
  `hold_invalid`, intersection-union over guardrails, per-experiment levels
  from `DecisionBudget`, ledger rows). Hook into `LifecycleManager`
  (`lifecycle.py:102`); bootstrap rules become descriptive; approval
  (`lifecycle.py:193`) unchanged.
- Done when: B2, B4, B5, B6, B9.

**WP-11.8 Benchmarks, certificates and referents** (two agents)
- 11.8a Corpus manifests with hash and licence for Criteo uplift, Hillstrom,
  and (candidate) Open Bandit Dataset; offline-only after download.
- 11.8b `benchmarks/qualification/f11_generators.py` (simulated bandits with
  analytic value, binormal groups, drift worlds), `f11_certify.py`,
  `f11_certificates.json`, reference-output fixtures with tool versions and
  hashes.
- Done when: A5-A7 and B7-B8 run from the manifest; `--check f11` fails on
  stale or missing certificates.

## Parallel lanes

Lane 1: 11.1 then 11.2, 11.3 in parallel, then 11.5. Lane 2: 11.6 after 11.1,
then 11.7. Lane 3: 11.4 after 11.1. Lane 4: 11.8a from day one, 11.8b
alongside each package. 11.5 and 11.7 are the integration points.

## Exit gate

```bash
uv run pytest -q tests/unit tests/property -k "decision or netbenefit or ope or rollout or anytime"
uv run python -m benchmarks.qualification --check f11
uv run python -m benchmarks.qualification --f11-referents    # Criteo, Hillstrom, simulated
```

Numbers (Wilson 95%; provisional, re-fixed from the first full run which is
published before the gate is switched on):

- Threshold: median excess cost over the oracle <= 1% of the oracle cost at
  `n_dev >= 5000`.
- Net benefit: simultaneous band covers the true curve in >= 93% (Wilson
  lower) of `R = 2000`; false `useful` rate <= 0.05 + binomial tolerance on
  null scores.
- OPE: DR coverage Wilson lower >= 0.93 where ESS/n >= 0.2; support-violation
  refusal >= 95% (Wilson lower >= 0.90), false refusal <= 5%; the Criteo and
  Hillstrom biased-subsample check covers the randomized value in >= 90% of
  200 subsamples.
- Fairness: false "satisfied" <= 5% (Wilson upper <= 8%); gap interval
  coverage >= 93%; infeasible constraint sets refused in 100% of constructed
  cases.
- Sequential: ever-promote rate under the boundary null with peeking at
  every unit <= `alpha_promote` + binomial tolerance at `R = 10000`; false
  rollback <= `alpha_rollback` + tolerance; sample-ratio check size <= 0.01 +
  tolerance and power >= 0.95 at 2% distortion with `n = 100000`.
- A/A on randomized Criteo and Hillstrom logs: false promotion <=
  `alpha_promote` + tolerance over `R = 1000` relabelings.
- Budget: assigned levels sum to at most `alpha_promote` exactly; ledger
  replay bit-identical on Python 3.11 and 3.14.
- Outside-party reproduction of the A/A and optional-stopping results; if it
  has not happened the phase is not closed.

## Evidence required

Closed-form comparison tables; coverage and refusal tables with Wilson
bounds; peeking-rate comparison between the new rule and the current
bootstrap rule; expected sample number versus fixed-horizon table;
detection-delay curves; A/A and positive-control reports from real
randomized data; fairness frontier plots with intervals; budget replay log;
certificate file with generator hash; outside-party report.

## Risks

- **Costs are inputs, not facts.** A wrong cost ratio gives a correct answer
  to the wrong question. Mitigation: costs are attested, tiered, and the
  report shows the range over which the decision is unchanged.
- **Anytime-valid intervals are wider.** The price is measured (B3) and
  reported so users see why a fixed-horizon test would have stopped earlier.
- **Overlapping experiments and e-LOND.** The guarantee for asynchronous
  starts is unverified; the plan defaults to the discount-only rule
  (valid by the union bound, lower power) until the paper is read.
- **Non-stationary effects.** The confidence sequence covers the running
  average; the report says so. Drift worlds in B9 measure the consequence.
- **Randomization is assumed.** Logs without recorded assignment
  probabilities cannot produce a `confirmed` decision claim; the instrument
  refuses rather than estimates `mu` silently.
- **Fairness is contested.** The instrument measures and enforces only what
  the charter declares and does not choose a criterion; the legal
  admissibility of group thresholds is an attestation field.
- **Dataset licences and availability** (Criteo non-commercial, Hillstrom
  unverified): manifests are `candidate` until checked; failing referents are
  replaced openly, never silently.

## Explicitly out of scope

Adaptive allocation and bandit control, Bayesian decision rules without
frequentist error guarantees, interference-aware designs, reinforcement
learning evaluation, dynamic treatment regimes, learning of fairness
criteria, estimating causal effects from observational logs (F10), data
shift handling (F6), the HTML report.
