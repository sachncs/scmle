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
| Shift diagnostic | `scmle/evaluation/shift.py:26-31` (`AdversarialValidation`), `:93` (verdict) | An AUC and most-shifted features. It does not output density-ratio weights. |

Not found anywhere under `scmle/` (grep of `jackknife`, `mondrian`, `cqr`,
`weighted conformal`, `crps`, `pinball`, `winkler`): jackknife+/CV+,
class-conditional conformal, covariate-shift-weighted conformal, any
probabilistic-forecast score.

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
bundle.predict_interval(data, coverage=0.9, method="cv_plus")      # regression
bundle.predict_interval(data, coverage=0.9, method="cqr")          # exists
bundle.predict_set(data, coverage=0.9, conditional="class")        # Mondrian
bundle.predict_interval(data, coverage=0.9, shift_weights=w)       # weighted
report = bundle.coverage_report(frame_c, target_c, slices=...)     # on C, once
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
classifier between calibration and target covariates (`shift.py:26` already
trains the discriminator; it must be extended to output clipped odds
`p/(1-p)` and report the effective sample size `(sum w)^2 / sum w^2`).

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
entity with embargo. Metric intervals by cluster bootstrap (as
`champion.py:110-120` already does for model comparison). Cut: mixed-effects
prediction as a model family.

**B6. Text and image.** Protocol: before splitting, group near-duplicates
(exact hash plus embedding cosine above a threshold fixed on a planted-
duplicate calibration set); split by group. Model: frozen pretrained
encoder plus the existing linear head (`embeddings` extra for text; an image
encoder behind the `multimodal` extra for images, to be chosen and
license-checked at phase start). Metric and gate: see below. Cut:
fine-tuning, detection, segmentation, generation, OCR.

## Gates: measured operating characteristic and external referent

All "R" are replicate counts; every interval is exact binomial or bootstrap
as stated. Thresholds are fixed before the gate run (hash of this file and
seeds recorded).

| # | Quantity | Operating characteristic measured | Pass criterion | External referent |
|---|---|---|---|---|
| G1 | Split-conformal marginal coverage | `R = 2000` replicates, `n_cal = 500`, `a = 0.1`, test set 20,000 per replicate, scores continuous. Mean conditional coverage. | Mean in `[0.9 - 3*se, 0.9 + 1/(n+1) + 3*se]`; with the Beta law above, `se` is about `0.0003` (derived by hand from the formula; recompute in the harness). Per-replicate coverage vs `Beta(451, 50)`: KS `p > 0.001`. | Closed-form coverage law (above, to be checked against the paper that states it). |
| G2 | CV+ marginal coverage | Same design, K = 10, then 5. Worst-case world: unstable learner (1-NN). | Empirical coverage `>= 1 - a - 3*se`. The `1 - 2a` worst-case bound is not the pass criterion. | Differential: intervals equal MAPIE's CV+ (existence and version to be verified) on frozen data to `1e-9`; the jackknife+ paper's datasets if retrievable. |
| G3 | CQR | Heteroscedastic simulation (noise sd varies 5x across `x`); R = 2000. | Marginal coverage as G1; coverage in the highest-noise decile `>= 1 - a - 0.03`, and split-conformal with constant width shown to undercover there (documents why CQR). | Romano et al. benchmark datasets (names from memory, **unverified**). |
| G4 | Class-conditional sets | Imbalanced 5-class simulation, minority prior 2%; R = 2000. | Per-class coverage `>= 1 - a - 3*se` for classes with `n_k >=` the minimum; pooled-margin sets shown to under-cover the minority class. | Sadinle et al. construction; scikit-learn-compatible reference (MAPIE, if its class-conditional mode exists; to verify). |
| G5 | Slice flag | (i) Homoscedastic exchangeable worlds, 8 slices: familywise false-flag rate. (ii) A world with one slice at 3x noise: recall by slice `n`. | (i) `<= 0.05 + 2*sqrt(0.05*0.95/R)` (about 0.0597 at R = 2000). (ii) Recall lower bound `>= 0.80` at the smallest `n` that achieves it; slice flag gates only for `n` above that, below it is descriptive. | Exact binomial test (scipy); Holm per textbook. |
| G6 | Weighted conformal | Gaussian mean shift in 3 features, density ratio known in closed form; then ratio estimated by classifier. R = 2000. | Known ratio: coverage on the shifted test `>= 1 - a - 3*se`; unweighted shown to undercover by a reported margin. Estimated ratio: coverage reported with interval; claim tier follows A5 (if below, `unverified`). | Closed-form ratio; Tibshirani et al. paper. |
| G7 | Coverage claim on real data | 5 public regression and 3 classification datasets with a time-ordered or grouped structure where available; coverage on `C`/`F`. | Report. Gate only on: no `confirmed` coverage claim whose realised coverage on `F` is below `1 - a - tol`; any violation is a published miss (validation-referents section 3). | Held-out `F` set; the datasets themselves. |
| G8 | Forecast | M4 subset (Hourly and Daily frequencies; sizes and horizons to be read from the M4 data release) and M5 (hierarchy, 9 quantiles). First, reproduce the **official benchmark scores of Naive and Seasonal-naive** with our MASE/sMAPE/WIS implementations. | Metric implementation matches published benchmark scores to the precision the organisers published (target 3 decimals, to be confirmed). Then: the system's WIS and OWA are reported with a series-bootstrap interval; gate is "interval upper limit of (system minus best of Naive/Seasonal-naive/Theta) `< 0`" for the frequencies where the charter asks for forecasting; otherwise the forecast is released as `unverified`. | M4 competition data and published benchmark table (Makridakis et al., Int. J. Forecasting 2020) and M5 (Makridakis et al. 2022): **dataset names, scores and years from memory, unverified; a WebFetch to the publisher returned HTTP 403 for M5 in this session.** Verify at phase start. |
| G9 | Reconciliation | Hierarchical simulation with known covariance structure; and the M5 hierarchy. | MinT WIS `<=` bottom-up WIS on the simulation where the covariance is non-diagonal; coherent forecasts (aggregation constraint residual `< 1e-8`). | `hts`/`fable` (R) reconciliation outputs on frozen inputs (to verify availability). |
| G10 | Ranking | One public LTR set (MSLR-WEB10K or Yahoo LTR or LETOR; **names unverified**) with the standard folds. | NDCG@10 implementation equals the reference (`sklearn.metrics.ndcg_score` on frozen lists, `1e-9`); the system's NDCG@10 interval vs the published LambdaMART/LightGBM figures reported, no claim of beating them. | Published LTR results; sklearn. |
| G11 | Survival | (i) Weibull proportional-hazards simulation with covariate-dependent censoring where the true C and true Brier are computable. (ii) GBSG2 / Rotterdam and METABRIC style data (names from memory, **unverified**). | (i) Bias of the covariate-censoring IPCW C-index relative to the true truncated C within Monte Carlo error; marginal-KM version shown biased when censoring depends on covariates. (ii) Agreement of our estimates with `scikit-survival` `concordance_index_ipcw` and R `survival`/`pec` to `1e-6` on frozen inputs. | scikit-survival, R `survival`, `pec` (versions and availability to verify). |
| G12 | Multilabel | Public multilabel sets (Yeast, Scene; **unverified**) and a simulated one with known label dependence. | Per-label coverage per A3; macro-F1 interval; thresholded vs 0.5 reported. | `scikit-multilearn`/`scikit-learn` reference metrics. |
| G13 | Panel | Random-intercept simulation; exact prediction interval for a new entity is known in closed form. | Row-wise calibration shown to be anti-conservative for `new_entity` (same logic as F3 criterion 5); entity-wise calibration within `3*se` of nominal. | Closed-form random-effects predictive distribution. |
| G14 | Near-duplicate split | Plant `p` in `{1%, 5%, 20%}` near-duplicates across a random split on an image and a text set. | Measure score inflation from the leak; the duplicate grouper recovers `>= 0.95` of planted pairs at false-merge rate `<= 0.01`; grouped split removes the inflation to within bootstrap error. | Planted truth; published near-duplicate audits (CIFAR duplicate studies; **unverified**). |
| G15 | Image/text baseline | Frozen encoder + linear head on one public image set and one text set. | Within a stated margin of the encoder's published linear-probe accuracy (figure from the encoder's model card, to be recorded with hash); no claim beyond that. | The encoder's published result. |

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

## Acceptance criteria (machine-checkable)

1. G1 passes at `R = 2000`; the pass is a unit test with the seed recorded.
2. G2: CV+ intervals equal the reference implementation's on three frozen
   datasets to `1e-9` (or the documented tie-handling difference), and the
   worst-case world meets its bound.
3. G3 and G4 show the baseline (constant width; pooled margin) failing where
   the new method passes. A gate with no failing control is not accepted.
4. G5 certificate file lists false-flag and recall with intervals by slice
   size; the slice check refuses to gate below its certified size.
5. G6 coverage under known shift is within tolerance; unweighted shown
   worse. Estimated-ratio run records effective sample size.
6. A coverage claim is registered before `C` is read; a second read of `C`
   raises (F2/F3 mechanism, reused).
7. `ModelBundle` round-trip preserves fold-model hashes; tamper on one fold
   model fails verification (extends the existing checksum at
   `bundle.py:933-945`).
8. G8: our MASE, sMAPE and WIS reproduce the official Naive and Seasonal-
   naive scores on the M4 subset; mismatch fails the build.
9. G10-G13 differential tests pass against their references; G11(i) shows
   the marginal-KM IPCW biased under covariate-dependent censoring.
10. G14: planted duplicates are recovered; leak inflation is shown, then
    removed by the grouped split.
11. Every task type has a row in `task_gate_status.json` (`passed`,
    `failed`, or `no_referent`); a type not `passed` yields only
    `unverified` outputs (enforced by a test over `TaskType`).

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

Read in this session (2026-11-04, arXiv abstract pages): 1905.02928
(jackknife+), 1905.03222 (CQR), 1904.06019 (weighted conformal), 2202.13415
(beyond exchangeability), 2106.00170 (adaptive conformal inference),
1609.00451 (least-ambiguous sets), 2005.12881 (WIS). Only abstracts were
read, so **no theorem statement or constant was verified**. From memory and
unverified: the `1 - 2a` jackknife+/CV+ bound; the Beta law of split-
conformal coverage and the `1/(n+1)` upper bound; the impossibility of
distribution-free conditional coverage; CRPS propriety (Gneiting and
Raftery 2007); MinT (Wickramasuriya et al. 2019); Uno et al. 2011; M4/M5
details and scores; every named dataset above.

## Cut, and why

Jackknife+ and full conformal (cost, no gain at these sizes); ACI and
non-exchangeable weighting (extra tuning parameter needing its own
certificate); conformal for survival, ranking, multilabel-joint (no
verified method read); top-down and ensemble reconciliation; click-model
learning to rank; fine-tuned vision and language models; anomaly detection
and dimension reduction (not needed for the stated decisions).
