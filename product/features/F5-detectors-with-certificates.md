# F5: Detectors with certificates

Canon: [`../../plan/00-canon.md`](../../plan/00-canon.md). Status of every
claim below: **verified** (cites `path:line`) or **proposed**.

## Purpose

Turn the data-defect detectors (missingness, label noise, duplicates,
sentinels, outliers) from heuristics into instruments whose error rates are
measured, on file and checked before a verdict may gate a run (P1).

## Decision it serves

Each detector is bound in the charter's decision map (F1) to one action:

| Detector | Action it may trigger |
|---|---|
| Missingness | add a was-missing flag; choose the imputation strategy; block when the MNAR tipping point is inside the plausible range |
| Label noise | quarantine suspect rows from training; widen the confirmation set's CI; block above a charter noise ceiling |
| Duplicates / near-duplicates | merge into one split group (F2); drop exact copies |
| Sentinels | recode to missing before any other detector runs |
| Outlier triage | keep, cap, or exclude rows, per provenance class |

A detector with no bound action, or no certificate, is descriptive only.

## The certificate

A **certificate** is a stored record of one detector's measured operating
characteristics. It is produced by the F7 harness, never by the detector.

1. **Plant.** Copy the exploration set `X`. Inject defects of one type at a
   known rate and location (flip labels, blank cells under a chosen
   mechanism, insert sentinels, duplicate rows with jitter, inflate values).
2. **Measure recall** per defect type: planted defects found / planted. Report
   the Wilson 95% interval; the gate compares the **lower bound** to a
   threshold declared in the charter before the run.
3. **Measure the false-alarm rate** by running the detector on an unplanted
   copy (and on label-permuted copies for label noise). Report the Wilson
   upper bound.
4. **Store** `detector_certificates.json`: detector version, data fingerprint,
   defect type, rate, n planted, recall interval, false-alarm interval, seed.
5. **Gate.** A detector may gate only if its certificate is present for the
   current dataset family, recall lower bound >= charter threshold and
   false-alarm upper bound <= charter ceiling. Otherwise its output is
   downgraded to descriptive and the claim tier is `unverified`.

**Honest limit.** Planted defects are synthetic and tend to be easier than
real ones, so certified recall is **optimistic**. The report must say so next
to every recall figure, and F7 separately measures recall on real corpora
with adjudicated defects (see F7) to quantify the gap.

## Behaviour

- Input: the exploration set `X` only (P3). Detectors never read `C` or `F`.
- Output per detector: `Check` records (canon) plus a ledger entry (F3), a
  `Certificate` reference, and a decision from the decision map.
- Order of execution: sentinels, duplicates, missingness, labels, outliers.
  Sentinels run first because a code such as 999 hides in every later
  statistic.
- Artifacts: `detector_certificates.json`, `error_mechanism_report.json`
  (inside `eda_report.html`), entries in `decision_log.jsonl`.

## Method (one per detector; what was cut)

**Missingness.** Mechanism by evidence, not by assertion.
- MCAR: Little's test on the multivariate pattern (*proposed*; no MCAR test
  exists today, grep found none).
- MAR: fit an indicator model, missing-or-not on the other features, with
  out-of-fold AUC against a permutation null. Keep the existing per-column
  chi-square or rank tests with Cramer's V as the cheap screen
  (`scmle/data/profiling/missingness.py:51-73`, effect floor line 22).
- MNAR: untestable from the data; report a **tipping-point** analysis. Shift
  imputed values by delta across a charter-declared range and report the delta
  at which the conclusion flips. Gate only on that range, never on a claim of
  MCAR/MAR.
- Informative-missing flag: kept as is (the profile already feeds
  `informative_missing`, `profiler.py:358`).
- Imputation strategy is selected by held-out imputation error and **fit
  inside CV folds** only; a property test (F7) asserts the fitted state is
  unchanged when held-out rows are perturbed.
- Cut: MICE/MissForest comparison grids. One strategy per run, chosen by
  held-out error, with the multiverse in F6 covering the alternatives.

**Label noise.** Two detectors that must agree to gate.
- Confident learning: exists in simplified form, out-of-fold probabilities
  from a gradient-boosting model with per-class mean own-probability
  thresholds (`label_noise.py:109-167`, thresholds lines 149-153).
- Second detector, in-sample margin / area-under-margin (*proposed*); a row is
  a suspect only if both flag it, which lowers the false-alarm rate.
- Rate: replace the fixed `NOTE_RATE = 0.05` note (`label_noise.py:26`) by an
  interval. With an audited random sample of size m, combine with the detector
  scores by prediction-powered inference for a valid interval on the true
  noise rate. Without an audit, report the **detector-only** estimate with a
  stated model-dependence caveat and tier `unverified`.
- Existing limits: the module caps at `MAX_ROWS = 5000` (line 25) and covers
  numeric features only (`stages.py` passes numeric columns); the cap must be
  recorded as a sampling-error bound.
- Cut: noise-transition-matrix estimation for every class pair; kept only for
  the binary/multiclass rate.

**Duplicates and near-duplicates.**
- Exact: hash after rounding floats to 6 decimals (`duplicates.py:14,18-39`);
  group-aware split already moves copies together (`duplicates.py:49-103`).
- Near-duplicates (*proposed*): blocking keys plus MinHash on tokenised
  strings and a distance cut set from the planted-jitter certificate, not
  chosen by hand. Same entity under different IDs is found by blocking on the
  non-ID columns.
- Report duplicate rate with an exact binomial interval; label conflicts
  among duplicates are reported against the noise rate above.

**Sentinels.** Find values with an implausible point mass: a spike at a
round or repeated value that a smooth density fit (kernel or spline on the
rest of the column) cannot explain, tested by a one-sided binomial against the
fitted mass (*proposed*). Strings ("N/A", "-1") use exact-match dictionaries
declared in the charter plus a frequency test. Recode to missing, log the
recode.

**Outlier triage.** The question is *what is it*, not *how far out is it*.
- Today: Tukey IQR share (`columns.py:38,49-60`) is a statistic only.
- Proposed: classify each flagged row as error, real extreme or different
  population using **provenance** (F4): the field's instrument range, the
  write path, a cross-field contract violation. A distance score only ranks
  candidates (Isolation Forest, LOF); it never assigns the class.
- Rows without provenance are `unverified` and kept, never silently dropped.

## Demoted to descriptive-only (P1)

Kept in the EDA report, not gating, because they have no bound action or no
external reference: normality, kurtosis, multimodality and zero-inflation
tests; PCA / t-SNE / UMAP structure; mutual-information tables; pairwise
association matrices; correlation-based redundancy (`fingerprint.py:29,65`);
co-missingness (`fingerprint.py:32`); VIF and condition number (reported, no
gate); skew and quantile summaries (`fingerprint.py:310`). They inform the
reader and feed F3 as *candidate* hypotheses only.

## Guarantees and assumptions

- A gating detector has a certificate with intervals, not a point claim.
- False-alarm control across the detectors uses one family in the error
  budget (F1); per-column missingness tests move from Bonferroni
  (`missingness.py:104`) to the ledger's online rule (F3).
- Assumes planted defect rates and mechanisms are representative; real
  recall is measured separately (F7) and may be lower.
- MNAR conclusions are conditional on the declared delta range.

## Failure modes and degradation

| Failure | Behaviour |
|---|---|
| No certificate for this data family | descriptive only, tier `unverified` |
| Too few rows or one class | detector declines with a recorded reason (current code raises, `label_noise.py:131-136`) |
| Recall lower bound below threshold | detector blocked from gating; run continues |
| Over time budget | recorded as skipped, never silent |

## Repo integration (verified)

Existing code to migrate behind the `Check`/certificate interface:
`scmle/data/profiling/{missingness,label_noise}.py`,
`scmle/data/splitting/duplicates.py`, outlier statistic in
`scmle/data/validation/columns.py`. Called from `core/stages.py` after
profiling on the development partition (`stages.py:400`, `add_quality_notes`
at ~`stages.py:480-530`). New code lives in proposed `scmle/data/detectors/`.

## Acceptance criteria (machine-checkable)

- [ ] Every gating detector has a certificate; a missing one downgrades the
      detector in a test.
- [ ] False-alarm upper bound <= charter ceiling on 20 seeds of clean data.
- [ ] Recall lower bound >= threshold for each planted defect type.
- [ ] Label-noise suspects require agreement of both detectors.
- [ ] Imputation state is invariant to held-out perturbation (property test).
- [ ] Sentinel recode runs before every other detector (order test).

## External referent

Adjudicated label errors in public corpora (human-relabelled sets), documented
real data defects in OpenML/TabArena tasks, and the R/`cleanlab` reference
implementations as differential oracles; see F7. Neither the author nor the
user judges whether a detector is right.
