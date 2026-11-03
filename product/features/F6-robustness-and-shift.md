# F6: Robustness and shift

Canon: [`../../plan/00-canon.md`](../../plan/00-canon.md). Claims are
**verified** (cite `path:line`) or **proposed**.

## Purpose

Decide which exploration findings are stable enough to spend confirmation on
(F3), and show where the analysis sample under-represents the deployment
population, so the validation design reflects it.

## Decision it serves

| Output | Action |
|---|---|
| Stability verdict per finding | only `stable` findings go to confirmation; others are reported as `fragile` |
| Shift verdict per feature / overall | change the validation design (stratified or reweighted evaluation, wider CI); record coverage gaps; block when the charter's shift ceiling is exceeded |
| Forward check on `F` | final tier for temporal data: a claim that fails forward is labelled `unstable over time` |

## Behaviour

- Reads `X` and the charter's declarations; touches `C` only through F3's
  unlock, and `F` only for the forward check.
- Artifacts: `stability_report.json` (specification curves, perturbation
  intervals, choice-to-result attribution), `shift_report.json` (overall test,
  per-feature table, per-period drift, subgroup coverage),
  `decision_log.jsonl` entries.
- Findings arrive from the ledger (F3); results return as a `stable` /
  `fragile` flag plus the numbers behind it.

## Method A: multiverse stability (PCS)

**One method:** a pre-declared specification set plus a survival rule, with
joint inference by permutation.

1. **Declare before looking.** The charter lists the defensible choices and
   their levels: imputation method, outlier rule, encoding, inclusion
   criteria, MNAR delta range (F5). The set is hashed in the charter. Adding a
   specification after exploration is recorded as a charter amendment and
   downgrades the tier to `unverified`. This closes the "enumerate afterwards"
   loophole.
2. **Re-run** each candidate finding over the full grid (or a seeded random
   subset of fixed size when the grid is large; the subset size is a charter
   parameter).
3. **Perturb.** Add bootstrap resamples within each specification (grouped or
   block bootstrap when `X` has groups or time) for PCS-style intervals.
4. **Survival rule** (numbers are charter parameters, not constants here):
   a finding is `stable` iff
   - the sign agrees in at least `s_min` of specifications, and
   - the effect size lies within a band of `+/- b` of its median across them,
   - and the bootstrap interval excludes zero in at least `q_min` of them.
5. **Joint inference (specification curve).** Permute the target (within
   groups or blocks), recompute the whole curve B times, and compare the
   observed median effect and the share of significant specifications to the
   permuted distribution. This answers "is the curve itself more than chance"
   without treating the specifications as independent.
6. **Attribution.** Report which choice moves which result (effect of each
   factor on the estimate), so a fragile finding says why.

Cut: exhaustive multiverse over every column-level choice; Bayesian model
averaging. The declared grid is the contract.

## Method B: representativeness and shift

Needs a **deployment sample**. If the charter supplies none, the shift check
returns status `unverifiable`, not `pass`, and compares only against any
stated marginals (P5). The existing code requires `reference_uri` and skips
otherwise (`stages.py` `add_quality_notes`, ~`stages.py:518-535`).

1. **Overall:** classifier two-sample test. Today a gradient-boosting
   classifier gives an out-of-fold AUC compared with a fixed `0.65`
   (`representativeness.py:23,139-182`). Proposed: keep the classifier, add a
   **permutation p-value** (permute the side labels, refit or rescore B times)
   and a bootstrap or DeLong interval for the AUC; gate on the lower bound
   exceeding `0.5 + delta`, `delta` from the charter, not on `0.65`.
2. **Second, independent statistic:** kernel MMD with a permutation (or wild
   bootstrap) p-value, bandwidth by the median heuristic fixed before the test
   (*proposed*). Agreement of the two raises confidence; disagreement is
   reported.
3. **Per feature:** KS for numeric, chi-square for categorical, effect size
   alongside, all p-values entered in the ledger so the online rule controls
   FDR across them (replaces ad hoc comparison; today
   `column_gap` is a statistic with a numeric p-value only,
   `representativeness.py:126-136`).
4. **PSI only if calibrated.** PSI against a fixed cut (`> 0.25`,
   `monitoring/reference.py:37`) is not a test. The repo already subtracts a
   sampling-noise floor, `1.5 * (bins - 1) * (1/n_ref + 1/n_new)`
   (`reference.py:77-98`); that floor was verified by simulation per its
   docstring. Proposed: calibrate PSI by permutation per column, or demote it
   to descriptive.
5. **Temporal drift:** split `X` into ordered periods and test adjacent
   period pairs with the same two-sample machinery, with an embargo gap
   (charter) and one family for FDR. Repeated monitoring in production uses
   sequential error control (outside F6's scope; see monitoring work).
6. **Subgroup coverage:** per sensitive or charter-named subgroup, report
   sample share versus deployment share with an exact binomial interval; a
   coverage gap is where the validation design must stratify.
7. **Forward check on `F`** (temporal data): fit nothing on `F`; evaluate the
   surviving claims and report effect, interval and sign on `F`. Opened once
   (F2 custody).

Cut: t-SNE/UMAP overlays; t-test per feature with no multiplicity control;
label-shift estimation (deferred, needs deployment labels).

## Guarantees and assumptions

- Stability is relative to the **declared** specification set; it is not a
  claim about all defensible analyses.
- Shift tests have power only to the extent the deployment sample is a random
  draw of the deployment population; selection into that sample is a stated
  assumption.
- Two-sample tests detect covariate shift, not label or concept shift.
- The permutation p-values are exact under exchangeability within the permuted
  blocks; clustered or temporal data must permute within blocks or the null is
  wrong.

## Failure modes and degradation

| Failure | Behaviour |
|---|---|
| No deployment sample | shift status `unverifiable`; design note; tier `unverified` |
| Grid too large for the time budget | seeded subset of declared size, recorded |
| Fewer than `min` rows per period | period dropped with reason, not pooled |
| No `F` set (non-temporal data) | forward check absent; claim tier stays `confirmed`, never `forward-checked` |
| Specifications disagree on sign | finding `fragile`, reported, not confirmed |

## Repo integration (verified)

Reuses `scmle/data/profiling/representativeness.py`,
`scmle/monitoring/reference.py` (PSI and noise floor, lines 27-42, 77-98),
and the call site in `scmle/core/stages.py`. New code in proposed
`scmle/data/robustness/`. Uses F2 for set access, F3 for ledger entries and
unlock, F5 for the defect-handling choices that form the grid.

## Acceptance criteria (machine-checkable)

- [ ] A specification added after exploration changes the charter hash and
      lowers the tier (test).
- [ ] A finding with a flipped sign in more than `1 - s_min` of
      specifications is `fragile`.
- [ ] Under a label-permuted copy, `stable` findings number at most the
      negative-control floor over 20 seeds.
- [ ] Two-sample test p-values are uniform on same-distribution splits
      (Kolmogorov-Smirnov over 200 runs, level 0.01), including grouped and
      temporal data.
- [ ] Planted covariate shift of known size is detected with power >= the
      charter target; no shift gives false-alarm rate <= `alpha`.
- [ ] Missing deployment sample yields `unverifiable`, never `pass`.

## External referent

Reference implementations as differential oracles (scipy / R for KS, chi-square,
MMD permutation; published specification-curve and PCS procedures), shifts
with known ground truth from the F7 simulator, and the forward-in-time
outcome itself: whether stable findings later hold on `F` and in production.
