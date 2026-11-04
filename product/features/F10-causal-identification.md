# F10: Causal identification

Vocabulary and principles: [`../../plan/00-canon.md`](../../plan/00-canon.md).
Built in [Phase 10](../../plan/phase-10-causal-identification.md); needs F1
(charter), F2 (custody), F3 (ledger, confirmation), F4 (admissibility) and
F6 (stability and shift).

Status of this file: a specification of what is missing. Nothing named in
"Proposed" exists in the repo. Statements about existing code carry
`path:line` and were read in the repo on the date of writing.

## Purpose

Today a charter can declare `kind="causal"`, and the instrument then does
almost nothing causal. This feature adds the missing step between "the
analyst declares a causal question" and "a causal claim is issued": choose an
identification strategy only from what the charter declares, run it with
diagnostics whose operating characteristics are measured, say how wrong the
answer can be under unmeasured confounding, refuse when nothing is identified,
and tier the result honestly.

The held-out set `C` can only certify sampling error. It cannot certify that
the estimand equals a causal effect. This file defines what `confirmed` means
for a causal claim and why it is weaker than a statement of truth.

## Decision it serves

Whether a causal effect estimate may (a) be reported at all, (b) be reported
as `attested` or `confirmed`, or (c) be withheld as `not_identified`, and
which downstream action a failed falsification triggers (decision map, F1).
An associational estimate for a causal question is never labelled causal.

## What exists (verified) and what does not

| Item | Evidence |
|---|---|
| Estimand kinds are `predictive` and `causal`; the type carries `kind`, `population`, `target`, `prediction_moment`, `intercurrent_events`, `summary_measure` and nothing else. No treatment, no instrument, no time or unit index, no assumption list. | `scmle/data/charter/contracts.py:31`, `:80-98` |
| For a causal estimand the admissibility stage is skipped with the note "causal estimand without a declared graph: admissibility not run". | `scmle/data/admissibility/qualify.py:157-166` |
| A DAG is held as a hashed input marked `hypothesis`; nothing in the package confirms it. | `scmle/data/admissibility/dag.py:1-12`, `:37` |
| d-separation, back-door check and smallest adjustment set by brute force over at most 16 candidates. | `dag.py:78` (separated), `:244` (`backdoor_valid`), `:278` (`adjustment_set`), `:38` |
| Causal field rule: quarantine descendants of the treatment, flag when no adjustment set exists, quarantine colliders. Only reachable with a graph. | `scmle/data/admissibility/rules.py:214-293` |
| Conditional-independence falsification of the DAG's implied independences, Bonferroni level `alpha / m`, verdicts `falsified` / `not_falsified` / `skipped`; ledger tier `unverified`; the report states "tests can falsify an implication, never confirm the graph". | `scmle/data/admissibility/independence.py:309-384`, `:344`, `:399` |
| For `TaskType.UPLIFT`, a propensity check: cross-validated logistic AUC of treatment on covariates, "randomized" if AUC <= 0.6 (a constant with no certificate), arms need >= 5% share. | `scmle/evaluation/uplift.py:22-23`, `:112-145`; called at `scmle/core/stages.py:424-432` |
| Qini curve and average treated-minus-control difference (valid only under randomization or unconfoundedness, as the module says). | `scmle/evaluation/uplift.py:31-90`, `:25-28` |
| A bounded sequential IPW estimator for sustained binary treatment on balanced panels with a positivity-support check, effective sample size, and an explicit list of what it does not do (no dynamic regimes, no g-computation, no nuisance-uncertainty propagation). | `scmle/evaluation/sequential.py:1-12`, `:500`, `:939`, `:952` |
| Tiers `confirmed` / `attested` / `unverified`; claims carry a tier; confirmation yields `confirmed` only after `C` is tested once. | `scmle/contracts/qualification.py:25`, `:194`; `scmle/data/ledger/confirm.py:301-304` |
| Wilson interval and measured-certificate machinery to reuse. | `scmle/data/detectors/certificates.py:73`; `scmle/contracts/qualification.py:261` |
| Decision actions are a closed set; none means "refuse a claim". | `scmle/data/charter/contracts.py:35-42` |

Not present anywhere in `scmle/` (grep for estimator names was empty in the
admissibility, ledger and evaluation packages): any treatment-effect
estimator with an interval other than the raw uplift difference, overlap
diagnostics other than the AUC heuristic and the sequential positivity
check, difference-in-differences, instrumental variables, regression
discontinuity, synthetic control, sensitivity analysis, negative controls
for causal claims, and a refusal path. `statsmodels` is not a dependency
(`pyproject.toml:27-38`: numpy, pandas, pyarrow, pydantic, scikit-learn,
scipy, joblib, threadpoolctl), so linear-model inference is written in
numpy/scipy and checked against frozen reference outputs (see F7 oracles).

## Behaviour

Proposed charter addition (an optional block on `AnalysisEstimand`, bump
charter `FORMAT_VERSION`, `contracts.py:30`):

```python
CausalSpec(
    treatment="coupon",
    outcome="spend_30d",
    estimand="ATE",  # ATE | ATT | LATE | ATT_DID | ATT_SC | LATE_RD
    unit="customer_id",
    time="week",  # optional, required by DiD / SC
    instrument=None,
    running_variable=None,
    cutoff=None,
    dag="dag.json",  # hashed hypothesis (dag.py), optional
    assumptions=(  # closed vocabulary, each one attested
        "unconfoundedness_given:age,region,tenure",
        "positivity",
        "no_interference",
    ),
    strategies=("backdoor_aipw",),  # priority order, fixed in the charter
    negative_controls=(  # pre-registered, attested to share confounding
        NegativeControl(kind="outcome", variable="spend_prior_year"),
    ),
    strata=("region",),  # replication strata, or "period"
    sensitivity=SensitivitySpec(benchmark="strongest_observed", max_bias=0.2),
)
```

Pipeline, run once per charter on `X` unless stated:

1. **Select** (pure function of the charter, not of the data). A strategy is
   eligible only if every assumption in its required set is in
   `CausalSpec.assumptions` and the data design it needs is present (panel
   for DiD and SC, instrument column for IV, running variable for RD). The
   first eligible strategy in the charter's priority order is primary; other
   eligible strategies run as the declared multiverse (F6). Selection does
   not look at outcomes, so it is not a specification search.
2. **Refuse** when no strategy is eligible: write
   `identification_report.json` with `status="not_identified"`, the list of
   missing assumptions per strategy, and no effect estimate. Any associational
   summary printed alongside is labelled `associational`. Refusal does not
   stall the run (P5).
3. **Diagnose design** on `X` (overlap, instrument strength, density at the
   cutoff, pre-trends, pre-fit). Failure follows the decision map: usually
   `downgrade_tier` or `refuse_claim` (new action).
4. **Fit nuisances on `X`**, freeze them, hash them into the ledger.
5. **Estimate on `C` once** through F3's confirmation (one pre-registered
   hypothesis per estimand; subgroup or heterogeneity hypotheses are further
   ledger entries and pay the multiplicity cost).
6. **Falsification battery** (negative controls, placebos, replication), also
   registered before `C` is opened.
7. **Sensitivity** to unmeasured confounding on the estimate.
8. **Tier** (below).

Artifacts: `identification_plan.json` (strategy, assumptions, why eligible or
not), `identification_report.json` (diagnostics, estimate, interval, n),
`falsification_report.json`, `sensitivity_report.json`,
`causal_claims.json`. Each is hashed into the ledger head like F3's files.

## Method

One strategy per identification family, no survey. For each: estimand,
estimator, mandatory diagnostics, and what it cannot do.

### A. Back-door adjustment with doubly robust estimation (primary for observational cross-sections)

Assumptions required: `unconfoundedness_given(S)`, `positivity`,
`no_interference`. If a DAG is declared, `S` must satisfy
`Dag.backdoor_valid` (`dag.py:244`) and every member of `S` must pass the
existing causal admissibility rule (`rules.py:214`), otherwise the strategy is
ineligible. If no DAG is declared the declared `S` is an unchecked
attestation and the claim cannot exceed `attested`.

Estimator: cross-fitted augmented IPW (the DML / doubly robust score;
Chernozhukov et al., "Double/Debiased Machine Learning for Treatment and
Causal Parameters", arXiv:1608.00060 — title and summary verified: orthogonal
score plus cross-fitting gives root-n estimators with valid intervals).
With outcome regressions `m_t(x) = E[Y | T=t, X=x]` and propensity
`e(x) = P(T=1 | X=x)`:

```
psi_i = m_1(x_i) - m_0(x_i)
        + T_i (Y_i - m_1(x_i)) / e(x_i)
        - (1 - T_i)(Y_i - m_0(x_i)) / (1 - e(x_i))
theta_hat = mean(psi_i)
se        = sd(psi_i) / sqrt(n)        # interval: theta_hat +/- z_{1-a/2} se
```

Nuisance learners are fitted on `X` and frozen; the score on `C` is then a
sample mean of i.i.d. terms given the frozen nuisances, so the interval is
valid for `C` without further cross-fitting. (This is the reason the
nuisance models never see `C`; when `|X|` is too small to fit them well the
interval is still valid but wider.) The ATT variant replaces the score
accordingly (formula to be written from the source in WP-10.2; *unverified*
here). Validity needs rates for the nuisance error product; the conditions
are from memory (*unverified*) and the docstring must cite the theorem.

Overlap and positivity diagnostics (all measured on `X`, then repeated on
`C` as a check that cannot change the plan):

- minimum and maximum of `e_hat`, share of units with `e_hat` outside
  `[eps, 1 - eps]`;
- Kish effective sample size of the IPW weights, `(sum w)^2 / sum w^2`
  (`scmle/evaluation/sequential.py:952` computes this quantity for the
  sequential estimator and is the code to reuse);
- standardized mean differences after weighting for every covariate in `S`;
- the existing randomization AUC (`uplift.py:112`) stays as a descriptive
  note and is no longer a gate.

Trimming to the overlap region changes the estimand (to the effect on the
overlap population). The charter must declare a trimming rule in advance;
the report names the changed estimand. A common default of trimming at
`e_hat < 0.1` or `> 0.9` comes from Crump, Hotz, Imbens and Mitnik (2009) as
I recall it (*unverified*); the gate here is not that constant but the
measured certificate (below).

Cannot do: handle unmeasured confounding (that is section G), interference,
or time-varying confounding (the bounded estimator at `sequential.py:500`
covers one narrow case).

### B. Difference-in-differences with event-study diagnostics

Assumptions: `parallel_trends`, `no_anticipation`, `no_interference`, and for
staggered adoption an estimator that avoids negative weights on already
treated units. Estimator: group-time ATT with never-treated or
not-yet-treated comparison units (Callaway and Sant'Anna 2021, J. of
Econometrics; two-way fixed effects bias under staggered timing, Goodman-Bacon
2021, de Chaisemartin and D'Haultfoeuille 2020; all *unverified* from memory).
Diagnostics: event-study coefficients for pre-periods with simultaneous
confidence bands, a joint Wald test of the pre-period coefficients, and the
count of pre-periods. A pre-trend test that passes does not support parallel
trends well: pre-testing has low power and distorts inference after
selection (Roth 2022, *unverified*). Therefore the pre-trend test is
descriptive. The gating quantity is the honest sensitivity interval of
Rambachan and Roth, "A More Credible Approach to Parallel Trends", Review of
Economic Studies 90(5), 2555-2591, 2023 (bibliographic details verified):
the post-treatment violation is bounded by a multiple `M` of the largest
pre-period deviation, and the identified set for the effect is reported for
each `M`. The charter declares `M_max`; the claim holds at tier `confirmed`
only if the interval excludes 0 at `M_max`.

Cannot do: a single treated unit with no donor panel (use G below), or
treatment anticipated by the units.

### C. Instrumental variables

Assumptions: `instrument_relevance` (testable), `exclusion` and
`independence` (not testable; attested), and `monotonicity` for the LATE
reading (Imbens and Angrist 1994, *unverified*). Estimator: 2SLS for one
endogenous regressor, with the just-identified Wald ratio as the special case.
Mandatory diagnostics:

- first-stage strength by the effective F statistic that is robust to
  heteroskedasticity (Olea and Pflueger 2013, *unverified*); the critical
  values from Stock and Yogo (2005) are not hard-coded because their
  values were not verified; the weak-instrument threshold is the one the
  certificate measures (see Acceptance, A10);
- a weak-instrument-robust interval (Anderson-Rubin inversion) reported
  whenever F is below the declared threshold, instead of the 2SLS Wald
  interval;
- overidentification test (Sargan / Hansen J) when there are more
  instruments than endogenous regressors. Passing it does not support
  exclusion for a just-identified model and for a single-invalid-instrument
  pattern; the report says so.

Estimand is LATE for compliers, not ATE. The charter's `estimand` must say
`LATE`; if it says `ATE` and only IV is eligible, the strategy is ineligible.

### D. Regression discontinuity

Assumptions: `continuity_at_cutoff` of potential outcomes, `no_manipulation`
of the running variable at the cutoff. Estimator: local linear regression
with a data-driven bandwidth and robust bias-corrected interval (Calonico,
Cattaneo and Titiunik 2014, *unverified*; R `rdrobust` is the intended
reference). Diagnostics: density test of the running variable at the cutoff
(McCrary 2008; Cattaneo, Jansson and Ma 2020; both *unverified*), covariate
balance at the cutoff, placebo cutoffs on the same side, bandwidth and
kernel sensitivity. Estimand is the effect at the cutoff only; it is never
extended to the population.

### E. Synthetic control

Assumptions: the treated unit's counterfactual is in the convex hull of the
donors given the pre-period fit; no interference to donors. Estimator:
non-negative weights summing to one that minimise pre-period distance
(Abadie, Diamond and Hainmueller 2010, *unverified*). Inference by
placebo-in-space permutation: re-assign the treatment to each donor, compute
the post/pre RMSPE ratio, and take `p = rank / (J + 1)`. The smallest
attainable p-value is `1 / (J + 1)`; with `alpha = 0.05` at least 19 donors
are required, and the instrument refuses with fewer. With one treated unit
there is nothing to put in `C` (the unit is the sample), so the best tier is
`attested`; this is a stated limit, not a bug.

### F. Randomized or encouraged-design data

If the charter declares `randomized_assignment` with a recorded assignment
mechanism, the estimator is the difference in means with regression
adjustment, and the back-door machinery is not needed. The AUC check of
`uplift.py:112` becomes a balance diagnostic with a measured false-alarm rate
(a randomization check test has a known size under the null: it must reject
about `alpha` of the time on randomized data). This is also the source of
benchmark truth (section "External referent").

### G. Sensitivity to unmeasured confounding

Reported for every strategy that rests on unconfoundedness (A) and, in
analogue form, for B (via `M`) and C (via violation of exclusion):

- **E-value** (VanderWeele and Ding, Annals of Internal Medicine 167(4),
  268-274, 2017; existence, definition as "minimum strength of association on
  the risk-ratio scale that an unmeasured confounder would need with both
  treatment and outcome to explain away the association", and the advice to
  report it for the point estimate and for the confidence limit nearest the
  null, are verified). The closed form `RR + sqrt(RR (RR - 1))` for `RR >= 1`
  is from memory and must be checked against the paper and the R `EValue`
  package at the start of WP-10.5.
- **Robustness value and partial-R2 bounds** (Cinelli and Hazlett, "Making
  Sense of Sensitivity: Extending Omitted Variable Bias", JRSS-B 82(1),
  39-67, 2020; reference list cites the R package `sensemakr`; both
  verified). They state how strong a confounder must be, in partial R2 with
  treatment and with outcome, to change the conclusion, and bound confounder
  strength by benchmarking against observed covariates. The charter's
  `benchmark="strongest_observed"` means: the confounder is allowed to be
  as strong as the strongest observed covariate in `S`. Benchmarking is an
  assumption about the unobserved, and it is reported as one.
- **Rosenbaum bounds** for matched designs (Rosenbaum 2002, *Observational
  Studies*, *unverified*): the sensitivity parameter `Gamma` at which the
  conclusion flips. Only reported when the charter selects a matched design;
  not implemented otherwise (cut).

None of these bounds proves absence of confounding. Their gate is: the
conclusion survives a confounder up to the charter's declared strength
(`max_bias`), and the gate's operating characteristic is measured on worlds
with planted unobserved confounders (Acceptance A6).

## What `confirmed` means for a causal claim

The three tiers of canon P4 are kept. For a `kind="causal"` claim they are
defined as follows (a one-paragraph addition to the canon is needed; it is
not made here).

- `unverified`: the strategy ran on defaults, or an assumption it needs was
  not declared, or the falsification battery could not run.
- `attested`: all assumptions the strategy needs are declared, each carries a
  named human attestation with date (provenance of the kind F4 already
  records), design diagnostics passed, but the battery is absent or
  incomplete, or `C` cannot be used (single treated unit).
- `confirmed`: all of the following hold, each recorded in
  `causal_claims.json`:
  1. the estimate and its interval were computed on `C`, opened once (F3);
  2. every design diagnostic of the strategy passed at its certified
     threshold;
  3. the pre-registered **falsification battery** passed: for each declared
     negative-control outcome (an outcome the treatment cannot affect but
     that shares the confounding structure) the interval for the "effect"
     includes 0, tested on `C` with Holm at `alpha_falsify`; for each
     negative-control exposure likewise; for each placebo (placebo treatment
     time, placebo cutoff, placebo donor assignment) the placebo effect is
     not distinguishable from 0;
  4. **replication**: the estimate computed separately in each declared
     stratum (site, region, period; the forward holdout `F` of F2 serves as
     a period) has the same sign and a heterogeneity test across strata does
     not reject at a declared level, with a minimum of two strata;
  5. the sensitivity gate (section G) holds at the declared `max_bias`.

What this does and does not mean. `confirmed` says: under the attested
assumptions, the interval has its nominal coverage (asymptotically; measured
at finite `n` by the benchmark), and the battery would have flagged a
violation of size at least `delta_min` with the measured power, and did not.
It does not say the assumptions are true. A violation smaller than
`delta_min`, or one that the negative controls do not share (an unmeasured
confounder that affects the outcome but not the negative-control outcome),
passes undetected; the attestation that the negative control shares the
confounding is itself untestable (canon P6). The held-out set alone never
yields `confirmed` for a causal claim: condition 1 is necessary and not
sufficient.

When no negative control is declared, the maximum tier is `attested`.

## Error control for the causal claim

- One pre-registered primary hypothesis per estimand: `theta = 0` against a
  two-sided alternative, tested on `C` once under the charter's
  `alpha_confirm` (`scmle/data/ledger/budget.py:66`). Subgroup, dose and
  heterogeneity hypotheses are registered in the F3 ledger
  (`scmle/data/ledger/ledger.py:641` `register`) and share the confirmation
  multiplicity rule.
- The falsification battery has its own level `alpha_falsify` split from the
  charter's error budget by the existing `split_budget`
  (`budget.py:277`); it controls the probability that a valid design is
  flagged (false alarm), not the probability of missing an invalid one. Its
  sensitivity to planted violations is a certificate (A5).
- Sequential monitoring of a causal estimate over time is out of scope here
  and handled by F11 using the same e-value machinery.

## Guarantees and assumptions

- Coverage of the primary interval is nominal under: the strategy's declared
  assumptions, i.i.d. or clustered sampling in `C` consistent with F2's
  grouped split, frozen nuisances not fitted on `C`, and overlap bounded away
  from zero. None of the identification assumptions is guaranteed by the
  instrument.
- `Dag` remains a hypothesis (`dag.py:37`). This feature does not add a way
  to confirm a DAG. Causal discovery algorithms are cut: their output is a
  data-selected graph and would need its own confirmation which the data
  cannot give.
- The refusal rule is a guarantee about the instrument's output, not about
  nature: it never prints a causal estimate unless an eligible strategy
  exists.

## Failure modes and degradation

| Situation | Behaviour |
|---|---|
| No strategy eligible | `status="not_identified"`; no causal estimate; decision-map action `refuse_claim`; run continues. |
| `adjustment_set` finds no set (`dag.py:278`) | Strategy A ineligible; message names the open back-door path. |
| More than 16 candidate covariates for the set search (`dag.py:38`) | Charter must supply `S`; search is not run; attested only. |
| Propensity near 0 or 1 (ESS below the certified floor) | Trim by the declared rule and report the changed estimand, or refuse if the trimmed population is under the declared minimum share. |
| Weak instrument (F below certified threshold) | Report the weak-instrument-robust interval only; cap tier at `attested`. |
| Density test rejects at the cutoff | RD ineligible; `refuse_claim`. |
| Fewer than 19 donors for `alpha = 0.05` | SC refuses. |
| Negative control rejects | Claim cannot be `confirmed`; decision-map action `downgrade_tier`; report which control. |
| Strata disagree in sign | Claim stays `attested`; heterogeneity reported. |
| `C` too small for the minimum detectable effect | Feasibility (`charter/feasibility.py:189`) downgrades scope; claim `unverified`. |
| Treatment or outcome is a predictive-only charter | F10 does not run. |

## Repo integration

- Charter: add `CausalSpec` beside `AnalysisEstimand`
  (`scmle/data/charter/contracts.py:80`) and a `refuse_claim` member of
  `DecisionAction` (`:35`); `AnalysisCharter` (`:280`) already hashes the
  estimand, so the causal spec enters the charter hash without a new
  mechanism.
- Admissibility: replace the skip at `qualify.py:157-166` by a call into the
  identification selector when the estimand has a `CausalSpec`; reuse
  `Dag.backdoor_valid` and `rules.decide_causal`; keep the CI falsification
  (`independence.falsify`) and report its result in `identification_report.json`.
- Stage: after the claim ledger (`scmle/core/stages.py:516`, `run_claim_ledger`
  at `:518`), run the identification stage; it reads `X` for design,
  `C` through F2/F3 custody for the estimate, `F` for period replication.
- Ledger: primary and subgroup hypotheses through `Ledger.register`; e-values
  from the estimate (for example a likelihood-ratio e-value from the
  asymptotic normal interval) are accepted by `online_fdr.Evidence`
  (`online_fdr.py:174`) if exploration of many heterogeneity hypotheses is
  allowed.
- Certificates: new store `f10_certificates.json` beside
  `f6_certificates.json`, produced by `benchmarks/qualification/f10_certify.py`
  following the pattern of `f6_certify.py`; a diagnostic gates only for the
  defect sizes whose outcome passed (as in the README of the harness).
- Package: new `scmle/data/identification/` (`select.py`, `aipw.py`, `did.py`,
  `iv.py`, `rd.py`, `synth.py`, `sensitivity.py`, `battery.py`, `tier.py`).
  The existing `evaluation/uplift.py` and `evaluation/sequential.py` are
  kept; the sequential estimator is registered as a narrow strategy
  `sequential_ipw`.

## Cut, and why

- Causal discovery for graph selection: unconfirmable.
- Matching and entropy-balancing variants: AIPW covers the observational
  cross-section; Rosenbaum bounds only if a matched design is chartered.
- Mediation, dynamic treatment regimes, continuous treatments beyond a
  dose-response with an explicit estimand: separate features.
- Conditional average effects as a gate: allowed only as ledger hypotheses.
- Automatic "best method" selection by outcome fit: it is a specification
  search and voids the guarantee.

## Acceptance criteria (machine-checkable)

Fixed in advance: replicate count `R = 2000` per world unless stated,
seeds from the charter, intervals Wilson at 95%.

1. **Selector is data-blind.** Property test: permuting outcome values
   leaves the selected strategy and the refusal decision unchanged; the
   selector module does not import any estimator.
2. **Refusal worlds.** On 200 generated charters with a declared effect
   question and assumptions that match no strategy, the instrument prints no
   causal estimate in 200/200 (Wilson lower bound >= 0.98). On 200 charters
   with an eligible strategy it does not refuse (Wilson lower bound >= 0.98).
3. **Coverage when assumptions hold.** Semi-synthetic worlds (real `X`
   covariates from the F7 corpus, planted treatment assignment and known
   `tau`) for each strategy: empirical coverage of the 95% interval with its
   Wilson interval; pass if the Wilson lower bound >= 0.93 for A, B, C, D at
   the benchmark `n` and the bias in units of `se` is within 0.1 on
   average. The thresholds are provisional; they are re-fixed from the
   first full run before the gate is switched on, and the first-run values
   are published with the result.
4. **Overlap worlds.** As planted overlap shrinks (min propensity 0.5, 0.1,
   0.02, 0.005), coverage and ESS are reported as curves; the overlap gate's
   false-refuse rate (valid overlap flagged) is <= 0.05 and its recall on the
   worst two settings is >= 0.8, both with Wilson bounds.
5. **Falsification battery operating characteristic.** Plant a violation of
   size `delta` (a hidden confounder shared with the negative-control
   outcome; a parallel-trends violation of slope `delta`; an invalid
   instrument with direct effect `delta`; sorting at the cutoff). Report
   power versus `delta` with Wilson bounds and the smallest `delta_min` for
   which power >= 0.8. False-alarm rate with no violation <= `alpha_falsify`
   plus the binomial tolerance used in F3's exit gate.
6. **Sensitivity bound validity.** In worlds with a planted unobserved
   confounder of known partial R2 below the declared benchmark, the
   Cinelli-Hazlett bounded interval contains the true effect in >= 95%
   (Wilson lower bound >= 0.93); with the confounder above the benchmark the
   rate at which the claim is still `confirmed` is reported, not gated.
7. **False-confirm rate.** In worlds with a violated assumption of size
   >= `delta_min` that the battery is certified to detect, the share of
   runs that reach `confirmed` is <= 0.05 (Wilson upper bound <= 0.08). In
   worlds with a violation the battery cannot see (unshared confounding),
   the rate is reported as the known blind spot and the tier is capped at
   `attested` when no negative control exists.
8. **Held-out alone is insufficient.** A test constructs a world with
   confounding by an unobserved variable, no negative control, and a
   significant estimate on `C`; the tier is not `confirmed`.
9. **Reference agreement.** Estimates and intervals equal the reference
   implementation on fixed inputs to the stated tolerance: AIPW against a
   hand-computed score on a toy table (exact), 2SLS and Anderson-Rubin
   against closed-form 2x2 cases, local linear RD against `rdrobust` frozen
   output (1e-6), synthetic control weights against a convex-program
   reference, E-value and robustness value against `EValue` and `sensemakr`
   frozen output (1e-9). R packages are not available in CI; outputs are
   frozen with their versions and hashes under `tests/fixtures/oracles/`
   (F7 convention).
10. **Experimental gold standards.** On the LaLonde / NSW benchmark (see
    below), the instrument's output on the observational comparison (CPS or
    PSID controls) is scored against the experimental estimate: the point
    estimate's bias and whether the claim was refused, capped or `confirmed`.
    Pass condition: a `confirmed` tier is never issued when the 95% interval
    for the observational estimate excludes the experimental estimate, on
    all the declared comparison groups; refusals and caps are reported.
11. **Determinism.** Identical `identification_report.json` hash on Python
    3.11 and 3.14 for a seeded run.
12. **Import graph.** No estimator imports from custody `C` APIs outside the
    confirmation function (as in WP-3.4).

## External referent

None of these is the author or the user.

- **Experimental truth.** Randomized experiments with an observational
  comparison group: the National Supported Work data with the CPS and PSID
  comparison groups (LaLonde 1986, American Economic Review; Dehejia and
  Wahba 1999, JASA; existence and general use as a benchmark are from memory,
  *unverified*; the true experimental estimate must be read from the data and
  recorded in the manifest, not from this document). Tennessee STAR class-size
  data (Krueger 1999, *unverified*) with an observational sample built by
  selecting units into treatment by a covariate rule, so that the randomized
  contrast is the truth.
- **Competition generators with known effects.** The 2016 Atlantic Causal
  Inference Conference data challenge, "Is Your SATT Where It's At?", and its
  analysis: Dorie, Hill, Shalit, Scott and Cervone, "Automated versus
  do-it-yourself methods for causal inference: Lessons learned from a data
  analysis competition", Statistical Science 34(1), 43-68, 2019, arXiv:1707.02641
  (verified: 30 competitors, two tracks, flexible response-surface methods
  performed best). The number of simulation settings, the later ACIC 2018,
  2019 and 2022 challenges and the IHDP semi-synthetic benchmark of Hill
  (2011) are from memory (*unverified*); each corpus enters the manifest as a
  `candidate` until existence, licence and hash are checked (see
  `plan/validation-referents.md`).
- **Reference implementations (outputs frozen with versions).** R `grf` or
  `DoubleML` for AIPW, `did` for group-time ATT, `HonestDiD` for
  Rambachan-Roth, `rdrobust` and `rddensity`, `AER` or `ivmodel` for IV,
  `Synth`, `EValue`, `sensemakr` (the last verified as cited by the
  Cinelli-Hazlett paper; the others are named from memory, *unverified*).
- **Replication by an outside party** in Phase 10's exit: someone other than
  the authors runs the gold-standard benchmark from the manifest and files
  the result.
