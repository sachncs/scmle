# F11: Decision layer

Vocabulary and principles: [`../../plan/00-canon.md`](../../plan/00-canon.md).
Built in [Phase 11](../../plan/phase-11-decision-layer.md); needs F1 (charter),
F2 (custody), F3 (budget, e-value rules), F6 (shift) and, for any policy that
acts on a treatment effect from non-randomized logs, F10.

Status of this file: a specification of what is missing. Statements about
existing code carry `path:line`, read in the repo on the date of writing.
Anything marked *unverified* was recalled, not checked against a source.

## Purpose

The instrument ends where a model is scored. A data scientist's job continues:
turn scores into actions given what errors cost, say whether the new policy is
better than the old one with an error rate that survives repeated looks,
design the experiment that would show it, and decide to promote, hold or roll
back. This feature adds that layer with the same discipline as F3: a
pre-declared decision model, one-shot evidence on untouched data, an exact
error budget, and a measured operating characteristic for every gate.

## Decision it serves

For a model or policy `pi` against the incumbent `pi0`: **act on it**
(promote), **keep watching** (hold), or **revert** (rollback), and at what
decision threshold or per-group rule it operates. Also: whether the
fairness constraints the charter declares are met, at what price.

## What exists (verified) and what does not

| Item | Evidence |
|---|---|
| A cost objective for the binary threshold. `DecisionSpec` has `threshold_objective` (`none`, `f1`, `accuracy`, `balanced_accuracy`, `cost`) and two positive costs; it lives in the run spec, not in the charter. | `scmle/contracts/tasks.py:493-500` |
| `CostObjective` scores minus the mean of `c_FP * FP + c_FN * FN` per decision; the threshold is a grid search over 0.01..0.99 on out-of-fold development probabilities, ties to the point nearest 0.5. No interval, no comparison with treat-all or treat-none, no check that probabilities are calibrated. | `scmle/models/thresholds.py:80-98`, `:109-135`, `:19` |
| The threshold is chosen on development data only and the test set is not consulted. | `scmle/core/stages.py:1472-1527`, `scmle/models/thresholds.py:1-5` |
| The charter's decision map binds checks to a closed set of data actions; it has no costs and no decision model. `AnalysisCharter` has `min_effect`, `k_max`, `power` and the error budget. | `scmle/data/charter/contracts.py:35-42`, `:148-162`, `:280-312` |
| Fairness is described, not decided: group rates, parity gap flagged above 0.10, four-fifths ratio below 0.80 (constants without certificates). | `scmle/evaluation/fairness.py:22-23`, `:186-188`, module docstring |
| Fairness mitigation by group thresholds or reweighing, off by default, judged on out-of-fold dev predictions against `trigger_gap=0.05`, `min_reduction=0.5`, `max_accuracy_cost=0.03`. | `scmle/contracts/tasks.py:503-529`, `scmle/evaluation/mitigation.py:110-178`, `scmle/evaluation/reweighing.py:17`, applied at `scmle/core/stages.py:1610` |
| Small-group warning and blocking thresholds on subgroup reports. | `scmle/contracts/evaluation.py:586-600` |
| A/B analysis of two models served to different requests: 300 bootstrap resamples per arm, 95% percentile interval of the metric advantage; `adopt` if the lower end > 0, `keep` if the upper end < 0, else `inconclusive`; fewer than 100 rows in an arm is `inconclusive`. A fixed-sample interval re-evaluated on accumulating data has no stated error guarantee. | `scmle/monitoring/abtest.py:23-24`, `:99-166`, decision at `:149-156` |
| Champion-challenger: paired bootstrap, `promote` iff the lower end of the 95% interval > 0. | `scmle/monitoring/champion.py:19-70`, `:165-170` |
| Lifecycle: a staged challenger needs an independent signed approval to become champion; rollback fires when the whole 95% interval of the new champion's advantage lies below `-tolerance` (default 0.01) with at least 100 labeled rows. The check is re-run on fresh rows and has no error budget. | `scmle/governance/lifecycle.py:39-55`, `:193-213`, `:328-406`, rule at `:365` |
| Shadow evaluation compares disagreement rate and probability gap; outcomes are settled later by the same bootstrap. | `scmle/monitoring/shadow.py:21-35` |
| Uplift scoring: Qini curve, top-30% uplift, average effect; no policy value, no off-policy estimator. | `scmle/evaluation/uplift.py:31-90` |
| Two-proportion sample size from a pooled-variance normal approximation, with Bonferroni over `k_max`; the closed-form power functions. | `scmle/data/charter/feasibility.py:94-135` |
| E-values are accepted by the online rules (`Evidence(p=, e=)`), e-LOND rejects when `e * level >= 1`, with the calibrator `kappa p^(kappa-1)` at kappa=0.5. | `scmle/data/ledger/online_fdr.py:174-195`, `:238-260`, `:356-416` |
| The exact error budget: `ConfirmationBudget`, `ExplorationBudget`, `ErrorBudget` in `Fraction`s; `spend` deducts exploration wealth only and raises when exhausted; `split_budget`, `confirmation_alpha`. | `scmle/data/ledger/budget.py:66-102`, `:137`, `:245`, `:277`, `:329` |
| Held-out evaluation set is opened once by the final stage. | `scmle/core/stages.py:1529`; `scmle/governance/vault.py:56` |

Not present anywhere searched (`scmle/evaluation`, `monitoring`, `governance`,
`inference`, `models/thresholds.py`): net benefit or decision curves,
off-policy evaluation, a behaviour-policy log schema, policy learning,
A/B design (power, randomization unit, sample-ratio check), any sequential or
anytime-valid test, any decision-budget accounting, any constraint-aware
policy optimisation, and a documented promote/hold/rollback rule with an
error guarantee. (A grep for `net_benefit`, `decision_curve` found nothing;
IPW appears only in `scmle/evaluation/sequential.py` for causal panel
estimation.)

## Behaviour

Proposed charter addition (all optional blocks; absence means the layer does
not run and no decision claim is made):

```python
DecisionModel(
    actions=("act", "no_act"),
    # payoffs per outcome, in a stated unit; costs are positive numbers
    cost_fp=5.0,
    cost_fn=100.0,
    unit="EUR per case",
    # plausible range of the cost ratio, used for the decision-curve claim
    ratio_range=(0.02, 0.15),  # threshold-probability range [t_lo, t_hi]
    baselines=("treat_none", "treat_all", "incumbent"),
    treatment_cost=None,  # for causal policies: cost per treated unit
)
FairnessConstraint(
    attribute="sex",
    criterion="equal_opportunity",
    max_gap=0.05,
    mechanism="group_thresholds",
)  # explicit, never default
RolloutPlan(
    unit="user",
    assignment_prob=0.5,
    primary="conversion",
    guardrails=(Guardrail("latency_p95", margin=20.0),),
    delta_min=0.005,
    tolerance=0.01,
    n_max=200_000,
    alpha_promote="1/50",
    alpha_rollback="1/100",
)
```

Pipeline:

1. **Thresholding / policy fit on `X`/development data** under the declared
   `DecisionModel`; frozen and hashed.
2. **Offline evidence on the final evaluation set, once**: net benefit with
   intervals against the baselines over `ratio_range`; or, for logged
   decisions, the off-policy value difference with its interval.
3. **Fairness constraints** evaluated on the same frozen decision rule.
4. **Rollout design** from the plan: sample size from power for a fixed
   horizon, or expected sample number and `n_max` for the sequential design;
   sample-ratio check defined.
5. **Online decision**: the promote/hold/rollback rule on anytime-valid
   evidence, with its own spend from a decision budget.
6. All steps write to the ledger and `decision_report.json`,
   `policy_value.json`, `rollout_design.json`, `rollout_decisions.jsonl`.

## Method

### A. Cost-sensitive thresholding

For a binary action with `cost_fp`, `cost_fn` and a calibrated probability
`p = P(y=1 | x)`, the expected cost of acting is `(1-p) cost_fp` and of not
acting is `p cost_fn`; act iff `p >= t*` with

```
t* = cost_fp / (cost_fp + cost_fn)
```

(derived here; the same rule is in the cost-sensitive learning literature,
for example Elkan 2001, *unverified*). It is optimal only if `p` is
calibrated, so the instrument checks calibration on development data first
(the existing `CalibrationReport`, `scmle/evaluation/diagnostics.py:143`,
reports Brier and ECE descriptively; F11 adds a calibration gate with a
measured false-alarm rate, or recalibrates with isotonic or Platt scaling
fitted on development data only). The grid search at `thresholds.py:109`
stays as the fallback when calibration fails; the report says which path
ran. Outcome: a threshold, its expected-cost estimate and a bootstrap
interval (clustered by entity when the split is grouped), and the
cost-ratio range over which the same decision rule stays better than the
baselines.

### B. Decision curves and net benefit with intervals

For threshold probability `t` (the odds `t/(1-t)` encode the cost ratio):

```
NB(t) = TP/n - FP/n * t / (1 - t)
```

(Vickers and Elkin 2006, *unverified* from memory; the formula is an
algebraic rewrite of the cost rule above with `cost_fp / cost_fn = t/(1-t)`
and is checked by a unit test against it). Baselines: treat none
(`NB = 0`) and treat all (`NB = prevalence - (1-prevalence) t/(1-t)`).
Intervals: cluster bootstrap over entities with `B` taken from the charter's
`permutations_b`, simultaneous over the declared range `[t_lo, t_hi]` by the
sup-statistic of the bootstrap (a pointwise interval at each `t` would claim
more than a decision range needs). Claim `useful`: the lower simultaneous
bound of `NB_model(t) - max(NB_all(t), NB_none(t))` is above 0 for every `t`
in `ratio_range`. This is evaluated once on the final evaluation set
(`stages.py:1529`), with threshold and model frozen beforehand, so the
interval is valid for the frozen rule; the guarantee is about the frozen
rule on that population.

### C. Off-policy evaluation for logged decisions

Setting: a log of `(x_i, a_i, y_i)` produced by a behaviour policy
`mu(a | x)` with recorded or estimated propensities; a target policy `pi`
fitted on `X`. Value `V(pi) = E[Y(pi(X))]`.

```
IPW:   V_hat = mean( 1[a_i = pi(x_i)] * y_i / mu(a_i | x_i) )
SNIPS: V_hat = sum(w_i y_i) / sum(w_i),   w_i = 1[a_i = pi(x_i)] / mu(a_i | x_i)
DR:    V_hat = mean( q_hat(x_i, pi(x_i))
                     + 1[a_i = pi(x_i)] (y_i - q_hat(x_i, a_i)) / mu(a_i | x_i) )
```

(Dudik, Langford and Li 2011 for the doubly robust form, *unverified*; the
unbiasedness of IPW and DR under correct `mu` or `q` is textbook algebra
and is tested by simulation.) Required diagnostics, each a gate with a
measured operating characteristic:

- **support**: `pi` must put mass only where `mu(a | x) >= eps_mu` (declared);
  otherwise the value is not identified and the instrument refuses;
- **weights**: Kish effective sample size `(sum w)^2 / sum w^2` over `n`;
  below the declared floor, refuse (code to reuse:
  `scmle/evaluation/sequential.py:952`);
- **clipping**: weights clipped at `M` trade bias for variance. With
  non-negative rewards clipped IPW is a lower bound on the value, so the
  clipped estimate may be used for a conservative claim; the clipping level
  is declared in the charter, not tuned on the outcome;
- **q_hat** is cross-fitted on development data and frozen before `C`/final
  evaluation;
- the interval for DR comes from the influence function (the per-row term
  minus its mean), for IPW from the same form; a bootstrap alternative is
  reported; empirical-Bernstein or similar finite-sample lower bounds
  (Thomas, Theocharous and Ghavamzadeh 2015, *unverified*) are an option for
  conservative claims.

If `mu` is estimated (observational logs), identification is a F10 problem:
the estimator is F10's strategy A and the claim inherits its tier cap. Only
logs from a randomization with recorded probabilities give an estimate that
is tiered `confirmed` by the final evaluation alone.

**Policy optimisation** is restricted to a pre-declared class (a threshold on
an estimated effect `tau_hat(x) > c`, or a tree of depth at most 2) and is
done on development data by maximising the DR value minus a complexity
penalty fixed in the charter. Using a rich class on a small log produces a
policy that wins on the training log and not elsewhere (a known failure of
policy learning; the regret bounds of Kitagawa and Tetenov 2018 and Athey and
Wager 2021 are the relevant references, both *unverified*). The held-out
value is the evidence, not the training value.

### D. A/B tests and staged rollouts

**Design** (written before any traffic):

- randomization unit and assignment by a salted hash of the unit id, so
  assignment is reproducible from the log; clustering factor if the unit is
  not the analysis unit (design-effect inflation of the sample size);
- sample size for a fixed-horizon test from power: for proportions
  `n_required_two_proportions` (`feasibility.py:118`), for means
  `n_required_mean` (`:79`); both are the existing closed forms;
- for the sequential design, the expected sample number at the planned
  effect and `delta_min` from simulation, and `n_max`; the cost of
  anytime-validity relative to the fixed-horizon `n` is reported, not hidden;
- **sample-ratio mismatch** check: exact binomial test of the assignment
  counts against `assignment_prob`; a rejection halts inference
  (`hold_invalid`) because the randomization or logging is broken. Its false
  alarm rate is measured (Acceptance B6).

**Anytime-valid inference.** For a bounded outcome `y in [0, 1]` (rescale
otherwise) and known assignment probability `q`, define for each unit

```
z_i = a_i y_i / q  -  (1 - a_i) y_i / (1 - q)         # E[z_i] = Delta, the mean difference
```

`z_i` is bounded in `[-1/(1-q), 1/q]`, and its mean is the effect because
assignment is randomized. Apply a confidence sequence for a bounded mean
that is valid at all sample sizes and stopping times simultaneously: the
betting-based confidence sequences of Waudby-Smith and Ramdas, "Estimating
means of bounded random variables by betting", J. R. Stat. Soc. B (2024),
arXiv:2010.09686 (existence, topic and journal verified), or the
time-uniform sub-Gaussian / empirical-Bernstein sequences of Howard,
Ramdas, McAuliffe and Sekhon, "Time-uniform, nonparametric, nonasymptotic
confidence sequences", Ann. Statist. (2021), arXiv:1810.08240 (verified). The
test form: a nonnegative supermartingale (e-process) `E_t` for the null
`Delta <= delta_0` has `P(sup_t E_t >= 1/alpha) <= alpha` by Ville's
inequality; this e-process view of anytime-valid inference is surveyed in
Ramdas, Grunwald, Vovk and Shafer, "Game-theoretic statistics and safe
anytime-valid inference", Statistical Science 38(4), 2023, arXiv:2210.01948
(existence verified). The specific betting construction and the claim that
the pseudo-observation `z_i` meets the boundedness condition are mine and
must be validated by the Acceptance B1-B3 simulations, not by citation.

**Reuse of the repo's machinery.** An e-process value at a stopping time is
an e-value, so it enters the ledger as `Evidence(e=...)`
(`online_fdr.py:174`). Across a stream of experiments and metrics the
controlled quantity is the false discovery rate over promotions: the e-LOND
rule (`online_fdr.py:356`) controls FDR at `alpha` under arbitrary
dependence between e-values; its assumptions in the repo carry a verified
citation (Xu and Ramdas, arXiv:2311.06412, Theorem 1). Whether the e-LOND
guarantee holds when experiments overlap in time, so that the rejection
count `|R_{j-1}|` is not final when experiment `j` starts, is not settled by
the repo's docstring and is *unverified*. Until read from the paper, the
rule is: experiment `j` uses level `alpha_promote * gamma_j` (discount
`gamma_j` summing to at most one, `Discount` at `online_fdr.py:138`) without
the rejection boost. This is valid by the union bound over experiments and
costs power; the boost is enabled only when experiments are strictly
sequential, or after the paper is checked.

### E. The promote / hold / rollback rule

State per (champion, challenger) pair, evidence accumulating in rollout
order. Let `Delta` be the primary-metric advantage of the challenger over the
champion in the metric's favourable direction, `delta_min` the smallest
improvement worth acting on (from the decision model), `tau` the
rollback tolerance.

Promotion hypotheses, all must be rejected (intersection-union):

```
H0_primary:   Delta <= delta_min            e-process E_primary
H0_guard_g:   harm_g >= margin_g  for each guardrail g   e-process E_g
```

**promote** at the first time `t` with
`E_primary(t) >= 1/level_j` and `E_g(t) >= 1/level_j` for every `g`, where
`level_j` is the experiment's level from the decision budget, and the
sample-ratio check has not rejected, and the independent approval required
by `LifecycleManager.promote_challenger` (`lifecycle.py:193`) is present
(statistics select the candidate; a named human releases it, unchanged).

Error guarantee. If any one null in the intersection is true, the chance
that the promotion fires is at most that component's chance of ever
crossing `1/level_j`, which is at most `level_j` by Ville's inequality
(derivation: the promotion event is contained in the crossing event of the
component whose null is true; no correction across components is needed for
an intersection-union test, which is the standard size argument and is
re-derived in the docstring). Therefore
`P(promote | challenger not worth delta_min, or harmful on some guardrail) <= level_j`
at any stopping rule, including peeking at every unit and stopping when
the evidence looks good. Across experiments the FDR (or, with the discount
rule above, the family-wise probability) of false promotions is at most
`alpha_promote`.

**rollback** after promotion, monitoring continues on fresh traffic with the
previous champion held out at probability `1 - q_rb`:

```
H0_rb:  Delta_new_vs_old >= -tau       (new is not worse than tolerance)
```

**rollback** at the first time `E_rb(t) >= 1/alpha_rollback`. Guarantee:
`P(rollback | new champion no worse than tau) <= alpha_rollback` over the
whole monitoring lifetime. This replaces the current rule
(`lifecycle.py:365`, interval re-evaluated on each check without
accounting for repeated checks). The converse error (failing to roll back a
bad champion) is not controlled by `alpha`; it is described by the measured
detection delay (expected rows until rollback for a given true loss), which
is a certificate (B4).

**hold** otherwise, until `n_max`, after which the decision is
`hold_final`: the champion stays; the report states the observed lower
confidence bound and the number of rows. `hold_invalid` applies to a
sample-ratio rejection or a logging failure; no claim is made.

Budget accounting. A `DecisionBudget` (new, beside `ErrorBudget`, in
`scmle/data/ledger/budget.py`) holds `alpha_promote` and `alpha_rollback` as
`Fraction`s; each promotion experiment is assigned its `level_j` before
traffic and recorded in the ledger; each started experiment spends its
level whether or not it ends in a promotion (the level is not recycled when
the test fails to promote, unless the e-LOND boost is verified and enabled).
`spend` (`budget.py:245`) currently deducts exploration wealth only; the
decision budget needs its own spend that raises `BudgetExhaustedError` the
same way. The rollback budget is per promoted champion, not shared.

Assumptions of the guarantee, stated in the report: randomized assignment
with the recorded probability; units do not interfere; the effect is
stationary over the monitoring window (otherwise the confidence sequence
covers the running average of `Delta_t`, not the current value); the
primary outcome is observed for all assigned units or missing outcomes are
handled by a declared rule; the guardrail margins and `delta_min` come from
the decision model and are not tested.

Not randomized (shadow mode, or champion-challenger on the same rows,
`champion.py`, `shadow.py`): the same e-process is applied to the paired
per-row loss difference `d_i` bounded in `[-L, L]` for a declared bound `L`;
the evidence is about the predictive metric, not about a downstream
outcome under the policy, and is labelled that way.

### F. Fairness as explicit constraints

Default: none. `FairnessConstraint` appears only in the charter. When
present it is a constraint on the frozen decision rule, not a bonus
objective:

```
maximize   NB(rule)   subject to   gap_g(rule) <= max_gap  for each declared constraint
```

where `gap` is the declared criterion (demographic parity, equal
opportunity, equalized odds) between groups of the attribute. Mechanisms,
chosen in the charter: group-specific thresholds (the machinery at
`mitigation.py:80` finds cuts for equal selection or equal TPR) or
reweighing (`reweighing.py:17`, Kamiran and Calders 2012 per its docstring).
The constraint is solved on development data; **satisfaction is a claim
tested on the final evaluation set**: the upper confidence bound of the gap
(Newcombe-type or bootstrap interval for the difference of two group rates)
must be `<= max_gap`. A point estimate below `max_gap` is not satisfaction.
Groups below the small-group blocking threshold
(`contracts/evaluation.py:586`) are reported as `not_assessable`, not as
passing.

The report always contains the trade-off: net benefit versus gap along the
frontier of the constraint (a sweep of `max_gap`), the price of the
constraint `NB(unconstrained) - NB(constrained)` with an interval, and, per
group, selection rate, TPR, FPR and calibration. If the declared criteria are
mutually infeasible when group base rates differ (the impossibility results
of Kleinberg, Mullainathan and Raghavan 2016 and Chouldechova 2017, both
*unverified* from memory), the instrument refuses the constraint set and
says which pair conflicts, using the measured base rates. Whether group
thresholds are permitted at all is a legal and policy question: the charter
carries it as an attestation field and the instrument does not decide it.

### Cut, and why

- Learning the fairness criterion from data or choosing one automatically.
- Multi-armed bandits and contextual bandit exploration policies; the layer
  evaluates and monitors policies, it does not run an adaptive allocation.
- Bayesian rollout decisions with priors (no error guarantee in the sense
  required by P2).
- Interference-aware designs (switchbacks, geo experiments), future work.
- Dynamic treatment regimes and reinforcement-learning OPE.

## Tiers for decision claims

- `confirmed`: the claim was tested once on the final evaluation set with the
  rule frozen (net benefit, policy value, fairness satisfaction) or, online,
  an e-process crossed its level with the sample-ratio check passing;
  and, for any claim about a treatment effect of a policy, the logs are
  randomized with recorded probabilities or the F10 tier of the effect is
  `confirmed`.
- `attested`: cost model, margins, or logging propensities are supplied by a
  named human without independent verification; evidence passed.
- `unverified`: defaults were used (for example the default threshold),
  calibration failed and the grid fallback ran, or support/ESS gates failed
  and a diagnostic estimate was shown.

Costs themselves are never verified by the instrument; they are attested
inputs and the report shows the range of cost ratios over which the
decision would not change.

## Guarantees and assumptions

- Offline: for a frozen rule and the final evaluation set opened once, the
  simultaneous interval over `ratio_range` has its stated coverage
  asymptotically; the finite-`n` coverage is measured (B2).
- Online: the promote and rollback errors stated in E hold at any stopping
  time under randomization, no interference and stationarity, and under the
  budget accounting above. They do not hold when peeking is done through a
  different statistic than the registered e-process.
- OPE: unbiasedness needs correct `mu` (IPW) or correct `mu` or `q` (DR);
  nothing in the data confirms `mu` for observational logs.
- None of this says the chosen costs reflect reality, or that an effect
  measured on one population holds on another (F6 covers shift, not
  transport).

## Failure modes and degradation

| Situation | Behaviour |
|---|---|
| No `DecisionModel` in the charter | The layer does not run; thresholding falls back to today's default and is tiered `unverified`. |
| Probabilities not calibrated | Recalibrate on development data; if still failing, grid fallback and `unverified`. |
| Decision-curve interval contains 0 somewhere in the range | Claim `not_useful_in_range`; report the sub-range where it holds. |
| Support violation or ESS below floor in OPE | Refuse the value; report ESS and the actions with no support. |
| Sample-ratio mismatch | `hold_invalid`; no inference. |
| Outcome missing differentially across arms | Report; the e-process is computed on observed units and labelled `missingness_not_handled`; tier `unverified`. |
| Decision budget exhausted | New promotions are not allowed; watching and rollback still run (separate budget). |
| Group below blocking size | Constraint `not_assessable`; run not blocked. |
| Constraint set infeasible | Refuse with the conflicting pair; no constrained policy issued. |

## Repo integration

- Charter: `DecisionModel`, `FairnessConstraint`, `RolloutPlan` beside
  `SetSizes` (`charter/contracts.py:193`); costs move from `DecisionSpec`
  (`contracts/tasks.py:493`) into the charter; the run spec's fields stay as
  the source only when no charter exists.
- Thresholding: `choose_decision_threshold` (`stages.py:1472`) reads the
  charter cost model and writes the calibration decision, the expected-cost
  interval and the ratio range.
- Final evaluation: `FinalEvaluationStage` (`stages.py:1529`) computes net
  benefit, policy value and fairness satisfaction on the sealed set.
- Online: `LifecycleManager` (`lifecycle.py:102`) gains a rollout driver; the
  bootstrap paths `abtest.compare_arms` and `ChampionChallenger.compare`
  remain as descriptive reports and no longer decide; the rollback rule at
  `lifecycle.py:365` is replaced by the e-process rule; independent approval
  (`lifecycle.py:193`) is unchanged.
- Budget: `DecisionBudget` and its spend in `ledger/budget.py`; ledger rows
  (`ledger.py:641`, `:669`) for each experiment and e-process snapshot.
- Package: new `scmle/decision/` (`threshold.py`, `netbenefit.py`, `ope.py`,
  `design.py`, `anytime.py`, `rollout.py`, `constraints.py`).
  `evaluation/fairness.py` and `mitigation.py` become clients of the
  constraint module; their constants stop gating.

## Acceptance criteria (machine-checkable)

Intervals Wilson 95% unless stated; `R` per world as stated; thresholds
provisional until the first full run is published.

**Offline**

A1. *Bayes threshold.* In simulated worlds with known `p(y|x)`, the chosen
threshold's excess expected cost over the oracle `t*` has median
<= 1% of the oracle cost at `n_dev >= 5000`; the full distribution is
published. Referent: the closed form `t*`.
A2. *Net benefit identity.* with `t = cost_fp / (cost_fp + cost_fn)`,
`NB(t) = (cost_none - cost_rule) / cost_fn` to 1e-12 on random tables, where
`cost_none = cost_fn * prevalence` is the mean cost of treating no one
(algebra check, derived here), and `NB(t)`
equals the closed-form binormal value within Monte Carlo error.
A3. *Interval coverage.* Simultaneous interval over `ratio_range` covers the
true curve in >= 93% (Wilson lower bound) of `R = 2000` replicates at
`n = 2000` and `n = 20000`, independent and clustered rows.
A4. *False usefulness.* With a score independent of the label, the claim
`useful` is issued in <= 0.05 + tolerance of replicates; power versus effect
size is reported.
A5. *OPE.* On simulated bandits with known policy value: bias and coverage of
IPW, SNIPS, DR with and without clipping, reported with Wilson bounds; DR
coverage Wilson lower >= 0.93 at ESS/n >= 0.2; the refusal gate fires in
>= 95% (Wilson lower >= 0.90) of worlds with planted support violations and
in <= 5% of worlds without.
A6. *Real randomized referent.* On the Criteo uplift data and on the
Hillstrom email data (both randomized; descriptions below), build a
non-random logging policy by subsampling with known selection probabilities
that depend on covariates; DR value of a fixed target policy estimated from
the biased log must have an interval covering the value estimated from the
randomized full data in >= 90% of repeated subsamples (`R = 200`, Wilson).
A7. *Fairness verdicts.* In worlds with planted group gaps of known size,
the verdict "constraint satisfied" is wrong (satisfied when true gap
> `max_gap`) in <= 5% (Wilson upper <= 8%); the gap interval covers the
truth in >= 93%. Optimal group thresholds on binormal groups match the
closed-form optimum to the grid resolution.
A8. *Infeasible constraints* are refused in 100% of constructed cases with
unequal base rates, and not refused when base rates are equal.

**Online**

B1. *Type I error under optional stopping.* Under `Delta = delta_min`
(boundary null), peeking after every unit up to `n_max`, `R = 10000`: the
share of runs that ever promote is <= `alpha_promote` + binomial tolerance.
The same experiment with `abtest.compare_arms` called after every batch is
run and its realized false-adopt rate is published, not assumed.
B2. *Intersection-union.* With the primary null false but one guardrail
null true, promotion rate <= that guardrail's level (same tolerance).
B3. *Power and cost of anytime-validity.* Expected sample number to promote
at `Delta = 2 * delta_min` versus the fixed-horizon `n` from
`feasibility.py`; ratio published per metric type (binary, bounded mean).
B4. *Rollback.* False rollback <= `alpha_rollback` + tolerance under
`Delta = -tau`; detection delay (median and 90th percentile of rows to
rollback) for true losses of `2 tau` and `4 tau`.
B5. *Budget algebra.* Sum of assigned levels never exceeds `alpha_promote`
(exact `Fraction` equality), a spend past the budget raises, ledger replay
reproduces levels bit for bit.
B6. *Sample-ratio check.* Size <= 0.01 + tolerance under correct
assignment; power >= 0.95 at a planted 2% allocation distortion with
`n = 100000`.
B7. *A/A on real data.* Take the randomized Criteo and Hillstrom logs and
randomly relabel arms (A/A) repeatedly (`R = 1000`, random stream order):
false promotion rate <= `alpha_promote` + tolerance (Wilson upper bound
reported).
B8. *Positive control.* On the Hillstrom data using the real randomized
arms, the instrument's promotion and the sequential stopping time are
reported against the fixed-sample result computed by a reference
implementation; agreement of direction and rough magnitude is a pass
condition, the numbers are not asserted in advance.
B9. *End-to-end lifecycle.* Simulated champion/challenger with drift: false
promote and false rollback within budget; `hold_final` correct when the
effect is below `delta_min`.
B10. *Determinism and import graph.* Same report hash on Python 3.11 and
3.14; no decision module reads the sealed set outside the one-shot call.

## External referent

None is the author or the user.

- **Randomized experiment outcomes.** Criteo uplift dataset (Diemert,
  Betlei, Renaudin and Amini, "A Large Scale Benchmark for Uplift Modeling",
  AdKDD & TargetAd workshop, London, 2018; dataset page at
  ailab.criteo.com; verified by search: user rows, features f0-f11, a
  treatment indicator, binary visit and conversion; randomly projected,
  non-uniformly sub-sampled; licence CC BY-NC-SA 4.0; row counts disagree
  across listings, so the manifest records the downloaded hash and count;
  the licence is non-commercial and must be checked before any use). An
  extended 2021 version is arXiv:2111.10106 (listing seen, not read).
  Hillstrom's MineThatData email challenge data (three randomized arms;
  existence and details *unverified*, to be checked and hashed).
- **Simulated environments with known policy value.** Bandit simulators
  written for the benchmark with analytic `V(pi)`. The Open Bandit Dataset
  and its pipeline (Saito et al., ZOZO, 2021) are a candidate for logged
  data from a randomized policy (*unverified*; enters the manifest as
  `candidate`).
- **Closed forms.** Bayes threshold, binormal net benefit and optimal group
  thresholds, the betting and mixture confidence-sequence guarantees
  (papers above) and Ville's inequality.
- **Reference implementations (frozen outputs).** R `dcurves` or `rmda` for
  decision curves, an OPE library such as Open Bandit Pipeline for IPW/DR
  (all named from memory, *unverified*), `confseq` for confidence sequences
  (existence *unverified*), and the repo's own `onlineFDR` oracles for the
  e-LOND levels (F3).
- **Outside replication** of A6, B1 and B7 by a party that is neither the
  author nor the user, in Phase 11's exit.
