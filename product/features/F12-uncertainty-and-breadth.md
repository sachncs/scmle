# F12: Uncertainty and problem-type breadth

Vocabulary and principles: [`../../plan/00-canon.md`](../../plan/00-canon.md).
Built in [Phase 12](../../plan/phase-12-uncertainty-and-breadth.md). Needs F2
(custody, splits), F3 (ledger, tiers) and the F7 harness. The canon's feature
table stops at F7; adding F12 and F13 there is a separate edit, not made here.

Why this file exists: a person doing a data scientist's job does not hand
over a point prediction. They state how wrong it can be, say for whom it is
more wrong, and use a validation scheme that matches the data's structure
(time, entities, queries, censoring). SCMLE has parts of both. This file
records which parts are present (verified in code), which are missing, and
the one method proposed for each gap, with a measured gate and an external
referent for each (P1).

## Purpose

(a) Give every deployed prediction an interval or set whose coverage is
guaranteed in finite samples under a stated assumption, measured on data the
calibration never saw, reported by slice, and downgraded in tier when the
assumption is not supported. (b) Close the problem-type gaps where today's
validation or metric is the wrong one for the data, so that "the system
picked a model" means something for forecasts, rankings, survival times,
panels, labels sets, text and images.

## Decision it serves

(a) Whether a coverage statement may ship in the bundle as `confirmed`,
`attested` or `unverified` (P4), and whether an interval may be used by a
downstream rule (F13 abstention). (b) Whether a task type may be run
unattended at all: a type without a certified protocol and a passed
benchmark gate is run, but every result is tiered `unverified` and F13
escalates it.

## What exists today (verified)

### Uncertainty

| Piece | Where | What it does and does not do |
|---|---|---|
| Finite-sample conformal quantile | `scmle/evaluation/conformal.py:23-39` | `rank = ceil((n+1)*coverage)`; returns infinity when `rank > n`. Correct split-conformal quantile. |
| Regression interval | `conformal.py:59-72`, `:122-136` | Absolute-residual score, one margin for every row. Marginal only. |
| Classification set | `conformal.py:75-101`, `:138-158` | Score `1 - p(true class)`, one pooled margin. A label is kept when `1 - p <= margin` (`:150`). Marginal over classes, not class-conditional. |
| Conformalized quantile regression | `scmle/evaluation/quantile.py:21-22` (fixed 0.05/0.95 base levels), `:38-73` (`QuantileBand`), `:90-96` (interval), `:98-127` (`coverage_curve`) | Adaptive width. Coverage curve is overall, not by slice. |
| Calibration source | `scmle/core/stages.py:2363-2442` (`conformal_calibration`), `:2444-2505` (`quantile_intervals`) | Scores are pooled out-of-fold over the K development folds (`:2405`), but the served model is a separate refit on all development rows (`:2497-2503`). That is not split conformal (calibration model differs from served model) and not CV+ (no per-fold models at prediction time). **No theorem as stated covers this exact construction.** |
| Skipped task types | `stages.py:2380-2389` | Survival, ordinal, ranking, multilabel, multioutput, forecast, uplift return `None`: no interval or set at all. |
| Skipped fold structure | `stages.py:2405` | Folds come from whatever the split strategy gave, including temporal and group folds. Nothing records that exchangeability is violated there. |
| Serving | `scmle/artifacts/bundle.py:1155-1204` (`predict_interval`), `:1528-1549` (`predict_set`), `:61` (`CONFORMAL_FILE`) | Methods `residual` and `quantile`; no `weights` argument, no slice report. |
| Uncertainty ranking | `bundle.py:1509-1526` | Margin between top two classes, or quantile-band width, for active selection. A ranking, not a guarantee. |
| Existing coverage gate | `benchmarks/reliability.py:287-304` | One synthetic regression set, one seed; a **90%** interval passes at empirical coverage **>= 0.85** (`:292`, `:303`). Too loose to detect a 4-point undercoverage and has no operating characteristic (fails P1 as a gate). Unit tests: `tests/unit/test_conformal.py:16`, `tests/integration/test_quantile_intervals.py:29`. |
| Slice machinery | `scmle/evaluation/slices.py:16-21` (`MIN_SUPPORT = 30`, `BANDS = 4`), `:24-77` | Scores a point metric per slice and lists worst gaps. Does not compute coverage or any interval around the slice score. |
| Calibration of probabilities | `scmle/evaluation/reliability.py:11` (top-label curve, 10 bins) | Descriptive; no confidence interval on bins. |
| Shift diagnostic | `scmle/evaluation/shift.py:26-31` (`AdversarialValidation`), `:93` (verdict) | An AUC, most-shifted features and (since 2026-11-06) the clipped, size-corrected odds of its out-of-fold discriminator with their effective sample size; the weights a bundle uses are still estimated by `attach_shift_weights`. |

Built since this section was written (library code, simulation gates in
`benchmarks/qualification/uncertainty_certificates.json`, see `CHANGELOG.md`):
class-conditional conformal sets (`ConformalCalibrator.class_margins`),
weighted conformal and clipped-odds shift weights, `cv_plus_bounds`, measured
coverage with Clopper-Pearson and Holm slice flags (`evaluation/coverage.py`),
proper scores, covariate-dependent IPCW, MinT, query-bootstrap NDCG, the task
gate status file. **In the run since 2026-11-04:** the fold models and
residuals are stored for regression with fewer than 1000 calibration rows
(`method="cv_plus"`, `evaluation/cv_plus.py`), shift weights can be attached
to a bundle (`shift=True`), a `coverage_report.json` is measured on the final
test rows (`evaluation/coverage_run.py`), and `unit_of_prediction:
new_entity` in the charter calibrates on one score per entity. **Still not
in the run:** the seven task types above still get no interval. Since 2026-11-06 the coverage claim (`coverage_claim.json`: levels,
methods, tolerance, rule) is hashed and registered in the custody log and the
event log before the final rows are released, the report names its hash and
`claims verify` checks it; above 1000 rows the pooled out-of-fold scores are
the served source by design (`conformal.scores` event). With the charter's
`uncertainty.coverage_on_confirmation` (format 11) the coverage claim is also
a registered claim of the ledger, tested on the one opening of `C` after the
model and its calibration are frozen (`coverage_confirmation.json`, status
`confirmed` or `unverified`, checked by `claims verify`), and the per-slice
margin flag (G5b) is certified.

### Problem-type breadth

| Type | Exists (verified) | Gap |
|---|---|---|
| Forecast | Task `forecast` (`scmle/contracts/tasks.py:23-35`); `ForecastMetric` with `mase`/`mae`/`smape` (`scmle/evaluation/forecast.py:44-58`); per-step backtest against last-value and seasonal-naive baselines and per-origin errors (`scmle/evaluation/backtest.py:18-24`, `:53`, `:80`); `RollingSplit` and horizon embargo (`scmle/data/splitting/splitters.py:581-602`, `:653`); lag/rolling feature nodes (`scmle/features/feature_graph.py:354-424`) | Point forecasts only. `MASE` here divides by the no-change error **on the same evaluation rows** (`forecast.py:44-47`), not by the in-sample naive error as in the usual definition; values are not comparable with published M4/M5 numbers. No CRPS/WIS/pinball. No hierarchy or reconciliation. No interval (skipped at `stages.py:2386`). |
| Ranking | Task `ranking`; `ndcg`, `map`, `mrr`, `precision_at_10`, `recall_at_10` (`scmle/contracts/evaluation.py:106-108`; `scmle/evaluation/ranking.py:22-109`); query-grouped split (`splitters.py:502`, `:744`); XGBoost `rank:ndcg` (`scmle/models/ranking/estimators.py:98`) | Point estimates with no query-level interval; no stated handling of queries without a relevant item; gain convention and tie handling not pinned to a reference (`ranking.py` imports `sklearn.metrics`; confirm at phase start which function is used). |
| Survival | Task `survival`; Harrell C (`scmle/evaluation/concordance.py:14-48`, thinned above 4000 rows at `:11`); IPCW Brier and dynamic AUC (`scmle/evaluation/survival.py:51-128`); Cox, discrete-time boosting, competing risks, time-varying (`scmle/models/survival/estimators.py:94,179`, `competing.py`, `timevarying.py`); event-stratified split (`splitters.py:331`) | The censoring curve is a **marginal** Kaplan-Meier (`survival.py:24-28`), i.e. it assumes censoring independent of covariates. Censoring probability floored at 0.05 (`:21`, `:37`), a bias the report does not state. Harrell C depends on the censoring distribution; a truncated, IPCW-weighted version is the usual remedy (Uno et al. 2011; from memory, **unverified**). No horizon calibration. No interval (skipped at `stages.py:2381`). |
| Multilabel | Task `multilabel`; `f1_macro_labels`, `f1_micro_labels`, `hamming`, `subset_accuracy` (`contracts/evaluation.py:109-111`); `MultilabelStratifiedSplit` (`splitters.py:370`); estimators `scmle/models/multitarget/` | One global threshold per label; no per-label support rule; no per-label uncertainty (skipped at `stages.py:2384`). |
| Grouped / panel | `GroupSplit` (`splitters.py:495`); temporal split with embargo scaled by series count (`:586-602`); group/block bootstrap in model comparison (`scmle/monitoring/champion.py:110-120`, `:147-150`); balanced-panel sequential IPW for causal contrasts (`scmle/evaluation/sequential.py:1-12`) | No distinction in the estimand between "new entity" and "future of a known entity"; interval calibration is row-wise, so for clustered data it is not covered by an exchangeability argument. |
| Text / image | `DocumentClassifier` (`scmle/models/documents.py:150`) and `ImageClassifier` (`scmle/models/images.py:133`); `multimodal` extra is `pillow` only (`pyproject.toml:66`); `embeddings` extra is `sentence-transformers` (`pyproject.toml:53`) | `images.py:1-6` says the classifier uses deterministic resized pixels and "deliberately" does not claim a production vision model. No near-duplicate-aware split. |
| Unsupervised | `scmle/unsupervised/clustering.py` (clustering with bootstrap stability) | Only clustering. No anomaly detection, no dimension reduction. |

## Behaviour

Proposed interface (extends `ModelBundle`, `scmle/artifacts/bundle.py`):

```python
bundle.predict_interval(data, coverage=0.9, method="cv_plus")  # regression
bundle.predict_interval(data, coverage=0.9, method="cqr")  # exists
bundle.predict_set(data, coverage=0.9, conditional="class")  # Mondrian
bundle.attach_shift_weights(calibration_rows, target_rows)  # built
bundle.predict_interval(data, coverage=0.9, shift=True)  # weighted, built
bundle.coverage_report  # dict set by the run from the final test rows
```

Artifacts: `coverage_report.json` (per nominal level: marginal coverage,
Clopper-Pearson interval, mean width; per slice: n, coverage, interval,
Holm-adjusted flag; per class for sets), `coverage_claim.json` (the claim
and its tier), `conformal.json` (existing; now carrying fold models' hashes
and the method name). The coverage claim is a claim in the F3 sense: it is
registered before `C` is read, tested once on `C`, and carries a tier.

## Method (one rigorous method per need)

Notation: level `1 - a`; `n` calibration points; scores `s_i` computed by
models that did not see point `i`.

### A1. Marginal coverage: CV+ in place of pooled out-of-fold scores

Replace the current construction (pooled out-of-fold scores with a separate
final refit) by CV+ (Barber, Candes, Ramdas, Tibshirani, "Predictive
inference with the jackknife+"; arXiv:1905.02928, title and authors read
2026-11-04). The abstract confirms: rigorous coverage for exchangeable data
and any algorithm that treats training points symmetrically; CV+ is the
K-fold variant. **The numeric statement (coverage `>= 1 - 2a` in the worst
case, near `1 - a` in practice) is from memory and unverified; the page read
did not state it. Read Theorems 1 and 4 before coding and cite them in the
docstring.**

For a test point `x`: with fold models `f_{-k}` and fold-of-`i` index `k(i)`,
the interval is the `ceil((1-a)(n+1))`-th smallest of `f_{-k(i)}(x) + R_i`
and the `floor(a(n+1))`-th smallest of `f_{-k(i)}(x) - R_i`, where
`R_i = |y_i - f_{-k(i)}(x_i)|`. The bundle stores the K fold models (hashed
in the manifest) in addition to the served model's pipeline; the served
point prediction is the K-model mean or the refit, declared in the claim.

Where a dedicated calibration split exists and `n_cal` is large (default
`n_cal >= 1000`), plain split conformal is used (`conformal.py:23-39` is
already correct), because its coverage law is exact and simpler to certify:
for continuous scores, coverage conditional on the calibration set follows
`Beta(n+1-l, l)` with `l = floor((n+1)*a)` (standard result, from memory,
**unverified**; the Phase 12 simulation tests it rather than assuming it).

Cut: jackknife+ itself (n refits, no gain at the sizes where K-fold is used),
jackknife+-after-bootstrap, full conformal, and any "conformal" variant
without a stated theorem.

### A2. Heteroscedastic regression: keep CQR

CQR (Romano, Patterson, Candes; arXiv:1905.03222, abstract read: "valid
coverage" guarantee, formal statement not on the abstract page, so the
exact bound is **unverified**). Already built (`quantile.py`). Change: the
calibration scores come from the CV+ folds (A1) rather than the pooled
out-of-fold construction at `stages.py:2480-2504`, and the base quantile
levels are tied to the requested level instead of fixed at 0.05/0.95
(`quantile.py:21-22`).

### A3. Conditional coverage: class-conditional by construction, slices by measurement

Exact conditional coverage given `X = x` cannot be guaranteed without
assumptions on the distribution (a known impossibility result for
distribution-free methods; from memory, **unverified**, citation to be
located). So:

- **By label (classification): Mondrian/class-conditional conformal.** A
  separate margin per true class, computed from the calibration points of
  that class only. Guarantee: coverage `>= 1 - a` within each class, given
  exchangeability inside the class. Needs `n_k >= ceil(1/a) - 1` points in
  class `k`, otherwise the margin is infinite (follows from `rank > n` at
  `conformal.py:37`); classes below that count get the full label set and
  the report says so. Reference for the idea: Sadinle, Lei, Wasserman, "Least
  ambiguous set-valued classifiers with bounded error levels"
  (arXiv:1609.00451, read 2026-11-04: set-valued classifiers with a
  user-set coverage level and minimal expected set size; the page does not
  use the term "class-conditional", so the exact class-wise statement is
  **unverified**).
- **By slice (everything else): measure, do not promise.** Slices come from
  `slices.py` (support `>= 30`, `slices.py:19`). For slice `g` with `m_g`
  test points and `k_g` covered, report the Clopper-Pearson interval at
  level `1 - delta_g`. A slice is flagged under-covered when its upper limit
  is below `1 - a`. The family of slices is fixed before `C` is read and
  tested with Holm at familywise `0.05`. A flag changes a decision (see
  decision map below), so its false-flag rate and recall are certificates
  (P1).

### A4. Covariate shift: weighted conformal; temporal drift: detect and escalate

Weighted conformal prediction (Tibshirani, Barber, Candes, Ramdas;
arXiv:1904.06019, read 2026-11-04) is for test and training covariate
distributions that differ; it needs the likelihood ratio between them, known
or estimated from unlabeled test covariates. The abstract does not state the
coverage form; with an **estimated** ratio the guarantee is approximate and
that is how it is reported. Weights `w(x)` come from a probabilistic
classifier between calibration and target covariates (`shift.py` reports
the clipped odds `p/(1-p)` of its out-of-fold discriminator and the effective
sample size `(sum w)^2 / sum w^2`, descriptively).

Time-dependent drift is not covariate shift with a fixed ratio. For it the
method is not a new interval but a detector plus F13: the system reports
realised coverage on a rolling matured-label window, and a coverage
confidence sequence (F13) escalates. Cut: adaptive conformal inference
(Gibbs and Candes, arXiv:2106.00170, abstract read: long-run coverage
"irrespective of the true data generating process") and the non-exchangeable
weighted variant of Barber et al. (arXiv:2202.13415, abstract read: "less
loss of coverage" under drift, no formal bound stated on the page). Both are
reasonable, but each adds a tuning parameter whose operating characteristic
needs its own certificate; they are the first candidates if A4's detect and
escalate proves too conservative.

### A5. Coverage claim and tier

- `confirmed`: marginal coverage tested once on `C` (or on `F` for temporal
  data), lower Clopper-Pearson limit `>= 1 - a - tol`, `tol` fixed in the
  charter (default `0.02`). The theorem's assumption (exchangeability) is
  then not what is trusted; the held-out measurement is.
- `attested`: no clean held-out set (small data) and a named person attests
  the data are exchangeable with deployment (same population, no time
  structure). Interval ships; tier says `attested`.
- `unverified`: neither. Interval ships with the label.

### B. Problem-type breadth: one method per type

For each type: the splitting and leakage rule, the metric, and the gate.

**B1. Forecast.** Protocol: rolling origin (`RollingSplit`) with embargo
`>= horizon` steps per series, never random CV; features built only from
rows at or before the origin (feature graph already does this). Metric:
**weighted interval score** over a fixed quantile grid, reported with the
point metric. WIS is a proper score from central prediction intervals and
approximates CRPS (Bracher, Ray, Gneiting, Reich; arXiv:2005.12881; title,
authors and both statements read 2026-11-04). The CRPS definition and
propriety (Gneiting and Raftery 2007) are **from memory, unverified**.
Interval source: quantile forecasts calibrated by A1 on rolling-origin
residuals **per horizon step**; the temporal-drift caveat of A4 applies and
is printed. Hierarchy: when the charter declares a hierarchy (bottom series
plus aggregation matrix `S`), reconcile with trace-minimisation (MinT;
Wickramasuriya, Athanasopoulos, Hyndman 2019; **from memory, unverified**,
paper not located in this session). Bottom-up is kept as the baseline that
MinT must beat. Cut: top-down variants, ensembles of reconciliation methods,
neural forecasters as a new family (they enter through the existing model
registry, not through F12). MASE for gating is redefined in the standard
way (in-sample naive scale per series) and the current same-rows variant is
renamed; this is a metric change recorded in the changelog.

**B2. Ranking.** Protocol: split by query, never by row (already
`splitters.py:502,744`); queries whose items are all irrelevant are
excluded and counted, not scored as zero. Metric: NDCG@10 with a
**query-level bootstrap** interval; gain and tie convention pinned to a
reference implementation. Cut: unbiased learning-to-rank from click logs,
conformal ranking.

**B3. Survival.** Protocol: event-stratified split, entities with several
spells grouped (`GroupSplit`), administrative censoring defined by the data
cut-off date in the charter, not per row. Metric: IPCW concordance
truncated at a charter horizon `tau` (Uno et al. 2011; **unverified**),
IPCW Brier at `tau`, and calibration at `tau` (predicted vs
Kaplan-Meier risk by decile). The censoring model is a Cox model on
covariates when the charter does not declare censoring independent of
covariates; the marginal Kaplan-Meier version (`survival.py:24`) remains as
the declared-independent case. The floor at 0.05 (`survival.py:21`) is
reported as `n_floored` per evaluation. Cut: survival conformal, deep
survival models as a new family.

**B4. Multilabel.** Protocol: iterative stratification (`splitters.py:370`);
labels with fewer than 50 positives in the development set are reported
descriptively and excluded from the primary metric. Metric: macro-F1 with a
label-bootstrap interval, plus per-label calibration. Threshold per label
chosen on out-of-fold probabilities (as `models/thresholds.py:184` does for
the binary case). Uncertainty: per-label A3 coverage. Cut: label-dependence
models, set-level (joint) conformal guarantees.

**B5. Grouped / panel.** The charter (F1) declares the unit of prediction:
`new_entity` or `known_entity_future`. `new_entity`: group split; interval
calibration uses calibration points from entities not in training and
**one score per entity cluster is the exchangeable unit** (the effective
`n` is the number of entities). `known_entity_future`: temporal split within
entity with embargo (built: the entity is not a split unit, `C` is time blocks
with an embargo, the final rows are the latest period, and the run refuses a
charter without the entity and time columns or with an entity-disjoint
validation; not yet measured on a real panel). Metric intervals by cluster bootstrap (as
`champion.py:110-120` already does for model comparison). Cut: mixed-effects
prediction as a model family.

**B6. Text and image.** Protocol: before splitting, group near-duplicates
(exact hash plus embedding cosine above a threshold fixed on a planted-
duplicate calibration set); split by group. Model: frozen pretrained
encoder plus the existing linear head (`embeddings` extra for text; an image
encoder behind the `multimodal` extra for images, to be chosen and
license-checked at phase start). Metric and gate: see below. Cut:
fine-tuning, detection, segmentation, generation, OCR.

## Gates still open

Thresholds are fixed before a gate run; seeds and a hash of this file are
recorded. Gates G1, G2 (simulation), G4, G6, G9 (simulation), G11(i) and
G13 (simulation) are in the certificate file. Open:

| # | What is open | External referent |
|---|---|---|
| G2 | Differential: CV+ intervals equal MAPIE's (not installed; existence and version to verify) on three frozen datasets to `1e-9`; the jackknife+ paper's datasets. | MAPIE; the paper. |
| G3 | **Missed**: CQR in the highest-noise decile covered 0.864 (floor 0.87, 150 replicates). Re-register a design before it claims conditional coverage. | Romano et al. datasets (names unverified). |
| G4 | Differential against a class-conditional reference (MAPIE mode, if any). | Sadinle et al. construction. |
| G5 | **Missed**: familywise false-flag 0.0755 (bound 0.0597); the Holm flag against the nominal level stays unused. Its redesign with a margin fixed in advance, G5b, passed (see the certificate file). | Exact binomial test; Holm. |
| G6 | Real covariate-shift data; tier of the estimated-ratio claim follows A5. | Closed-form ratio; the paper. |
| G7 | 5 regression and 3 classification datasets with time or group structure; coverage on `C`/`F`; gate only on "no `confirmed` claim below `1 - a - tol` on `F`"; every violation a published miss. | Held-out `F`. |
| G8 | Reproduce the official Naive and Seasonal-naive scores on an M4 subset (MASE, sMAPE, WIS) to the published precision; then system minus best baseline with a series bootstrap, upper limit `< 0` or the forecast is `unverified`. M5 with 9 quantiles. | M4/M5 data and published tables (details unverified). |
| G9 | MinT on the M5 hierarchy and WIS (simulation used squared error); `hts`/`fable` outputs on frozen inputs. | R reconciliation packages. |
| G10 | NDCG@10 equal to `sklearn.metrics.ndcg_score` on frozen lists to `1e-9`, and the interval beside published LambdaMART/LightGBM figures on one public LTR set. | Published results; sklearn. |
| G11 | (ii) GBSG2/Rotterdam and METABRIC-style data: agreement with `scikit-survival` `concordance_index_ipcw` and R `survival`/`pec` to `1e-6`. | scikit-survival; R. |
| G12 | Multilabel sets (Yeast, Scene; unverified): per-label coverage, macro-F1 interval, thresholded versus 0.5. | `scikit-multilearn`/sklearn metrics. |
| G13 | Wire entity-level calibration; the simulation did not show row-wise calibration to be anti-conservative, so that claim is unsupported. | Random-effects predictive law. |
| G14 | Plant duplicates in an image and a text set; recovery `>= 0.95` at false-merge `<= 0.01`; leak inflation removed by the grouped split. | Planted truth; published duplicate audits (unverified). |
| G15 | Frozen encoder + linear head within a stated margin of the model card's linear-probe figure. | The encoder's published result. |

## Decision map (what each check does)

| Check | Action it triggers |
|---|---|
| Coverage claim fails A5 `confirmed` | `interval_tier` -> `attested` or `unverified`; F13 treats the interval as non-binding. |
| Slice flagged under-covered (G5 certified range) | Interval ships with the slice named in `coverage_report.json`; F13 raises that slice's escalation rate. |
| Minority class below minimum `n_k` | Full label set returned for that class; claim restricted. |
| Duplicate grouper merges `> x%` of rows | Split is group-based; if leak inflation measured on dev exceeds bootstrap error, text/image result tiers `unverified`. |
| Forecast gate G8 fails | Forecast released as `unverified`, point forecast only. |
| Task type without a passed gate | Run completes, every output `unverified`. |

## Guarantees and assumptions

- Marginal coverage `>= 1 - a` (split) holds if calibration and test points
  are exchangeable and the score has no ties (ties make it conservative).
  CV+ adds that the learner treats training points symmetrically.
- Under temporal drift, covariate shift without correction, group
  dependence with row-wise calibration, or a calibration set used for model
  selection, **the guarantee does not hold**. The held-out measurement on
  `C`/`F` is what the tier rests on.
- Conditional coverage is measured, not promised (A3), except class-
  conditional under exchangeability within the class.
- The slice false-flag rate is controlled only for the pre-fixed slice family.
- Gate values are conditional on the simulated worlds; real data may violate
  them. Hence G7 is a recorded-miss policy and not a pass criterion.

## Failure modes and degradation

| Situation | Behaviour |
|---|---|
| `n_cal` too small for the level | Infinite margin (existing `conformal.py:37`); interval reported as unbounded, never silently capped. |
| Time or group structure in the data | Row-wise calibration is refused; protocol switches per B1/B5 or tier is `unverified`. |
| Density-ratio effective sample size `< 50` | Weighted conformal refused; escalation to F13. |
| Exchangeability test statistic on `C` rejects (coverage below tolerance) | Claim `unverified`; interval kept, labelled. |
| Time budget spent | Existing `conformal.skipped` event (`stages.py:2353-2361`); coverage claim absent, not faked. |
| Hierarchy not declared | No reconciliation; forecasts are independent per series, stated. |
| Reference data (M4/M5, LTR, survival sets) not retrievable or hash mismatch | Corpus stays `candidate`; the gate it feeds does not run and the type stays `unverified`. |

## Repo integration

- Extend `ConformalCalibrator` (`conformal.py:43`) with `by_class` margins and
  `weights`; keep `finite_sample_quantile` (`:23`) as the single quantile
  routine.
- Replace the calibration loops at `stages.py:2405-2442` and `:2480-2504`
  with a fold-model store; add `coverage_report` as a stage after the final
  evaluation, using `C` through F2's custody API (opened once, F3 rules).
- Extend `SliceReport` (`slices.py:24`) rather than writing a second slicer.
- `benchmarks/reliability.py:287` is tightened to the G1 criterion or
  replaced by it; the 0.85 threshold is removed.
- New: `scmle/evaluation/proper_scores.py` (WIS, pinball, CRPS-from-
  quantiles), `scmle/evaluation/reconcile.py`, `scmle/evaluation/ipcw.py`
  (covariate-dependent censoring), `scmle/data/splitting/neardup.py`.
  Gates in `benchmarks/qualification/uncertainty_certify.py` (proposed
  name) writing `uncertainty_certificates.json`, same pattern as
  `f6_certify.py`/`f6_certificates.json`.

## Acceptance criteria still open (machine-checkable)

1. G2 and G4 differentials pass against a reference implementation, or the
   documented tie-handling difference.
2. (Built for a charter that asks for it: the coverage claim is registered
   before `C` is read and tested on its one opening.)
3. (Built: `tests/integration/test_cv_plus_bundle.py` round-trips the fold
   models and refuses a changed file.)
4. G8: our MASE, sMAPE and WIS reproduce the official Naive and Seasonal-naive
   scores on the M4 subset; mismatch fails the build.
5. G10, G11(ii), G12 differential tests pass against their references.
6. G14: planted duplicates are recovered; leak inflation is shown, then
   removed by the grouped split.
7. (Built: the slice flag reports `flag_certified` false below the certified
   size of G5b.)
8. A type not `passed` in `task_gate_status.json` yields only `unverified`
   outputs *from the run* (`task_gate.tier_ceiling` exists and is tested over
   `TaskType`, but no run output calls it yet).

## External referent

For (a): the closed-form coverage law (checked against the paper that states
it); MAPIE or another independent conformal implementation for differential
tests (availability to verify); the published datasets of the jackknife+,
CQR and weighted-conformal papers; the held-out `C`/`F` sets. For (b): the
M4 and M5 competition data and published benchmark tables, public LTR sets
with published results, survival data with published analyses plus
`scikit-survival` and R `survival`/`pec`, the encoder model card for
G15. A domain owner's recorded attestation (validation-referents section 5)
is the referent for the unit of prediction (`new_entity` vs
`known_entity_future`) and the censoring horizon `tau`: statistics cannot
choose them.

## Verification status of literature claims

Read (abstract pages only): 1905.02928 (jackknife+), 1905.03222 (CQR),
1904.06019 (weighted conformal), 2202.13415 (beyond exchangeability),
2106.00170 (adaptive conformal inference), 1609.00451 (least-ambiguous sets),
2005.12881 (WIS). **No theorem statement or constant was verified.** From
memory and unverified: the `1 - 2a` jackknife+/CV+ bound; the Beta law of
split-conformal coverage (G1 now measures it: mean in the derived band and
Kolmogorov-Smirnov `p` above 0.001 against `Beta(451, 50)`, in simulation);
the impossibility of distribution-free conditional coverage; CRPS propriety;
MinT (the shrinkage intensity and the formulas are written from memory);
Uno et al. 2011 (the pair weights are derived, not copied; checked by G11(i)
only); M4/M5 details and scores; every named dataset.

## Cut, and why

Jackknife+ and full conformal (cost, no gain at these sizes); ACI and
non-exchangeable weighting (extra tuning parameter needing its own
certificate); conformal for survival, ranking, multilabel-joint (no
verified method read); top-down and ensemble reconciliation; click-model
learning to rank; fine-tuned vision and language models; anomaly detection
and dimension reduction (not needed for the stated decisions).
