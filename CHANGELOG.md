# Changelog

## Unreleased

* Data qualification foundations (no behavior change to existing runs): record
  types `Check`, `Claim` and `Certificate` with evidence tiers
  (`scmle.contracts.qualification`), an exact-arithmetic error budget that keeps
  exploration wealth and the confirmation level as separate types
  (`scmle.data.ledger.budget`), and keyed, order-independent random streams
  (`scmle.core.randomness`). None are top-level exports yet.
* Qualification verification harness (benchmarks/qualification): defect injector, semi-synthetic simulator, frozen scipy-generated reference oracles, corpus manifest, certificate store that refuses stale or missing certificates for gating detectors; mutation scope now covers the qualification modules.
* Analysis charter and custody (opt-in; runs without a charter are unchanged).
  `scmle charter init|check|sign` and `MLProject(charter=...)` /
  `--charter` fix the question, an ICH E9(R1)-style estimand, the decision map,
  set sizes and the error budget before profiling, with a SHA-256 hash and an
  optional HMAC signature. A charter run carves a confirmation set `C` out of
  the development rows (group- and duplicate-aware, time blocks with an embargo
  for temporal data) and seals it in a separate vault process, even with
  `enforcement="cooperative"`; profiling and trials see exploration rows only.
  Closed-form power gives a `feasible`, `downgrade` or `infeasible` verdict:
  when the set is too small, or the carve would leave a validation fold
  empty, the run completes as `exploratory_unconfirmed` with no `C`.
  New artifacts: `analysis_charter.json`, `error_budget.json`,
  `feasibility.json`, `split_manifest.json` (v2, full 64-hex SHA-256) and the
  hash-chained `custody_log.jsonl`. The `MLProject` signature snapshot was
  regenerated for the new optional `charter` argument. Custody is
  tamper-evident on one machine, not adversary-proof; see `docs/SECURITY.md`.
* Claim ledger and confirmation (charter runs): hash-chained append-only
  `hypothesis_ledger.jsonl`, online FDR rules (e-LOND default, valid under
  arbitrary dependence; LORD++ and SAFFRON labelled independence-only),
  permuted-target negative controls, and one-time confirmation of exploration
  survivors by BY (default), BH, Holm or Bonferroni. New artifacts:
  `negative_control.json`, `confirmed_claims.json`. `--check fdr` runs the
  simulation gate. Thresholdout is not implemented.
* Admissibility and provenance (`scmle.data.admissibility`): field provenance,
  as-of reconstruction, admissibility rules by estimand, falsify-only DAG
  checks, leakage tests and reconciliation, with certificates in
  `benchmarks/qualification/f4_certificates.json`. The leakage skill test stays
  descriptive until noisy-copy leaks are certified.
* Detectors with certificates (`scmle.data.detectors`): missingness, label
  noise, sentinels, duplicates and outlier triage. `--write-f5` regenerates
  `detector_certificates.json`; `--check` verifies it. Certificates cover one
  corpus; label-noise detection is descriptive.
* Thread safety: the estimator and metric registries, experiment store, model
  registry, key registry, monitoring journals, shadow evaluator and bundle
  history are safe to use from several threads in one process (see
  `docs/ARCHITECTURE.md`, Concurrency). Separate processes sharing one
  registry or run directory are still not coordinated.
* The L1 feature selector no longer fails on multiclass targets.
* The public API snapshot (`tests/fixtures/public_api.json`) was regenerated
  after the `scmle` rename, which had left it stale; the set of exports is
  unchanged. `scripts/check_api_compatibility.py --write` regenerates it.
* Replay verification now reports `different` when a trial's status changed
  between the original run and the rerun; it used to report `exact`.
* Registered TabPFN engines refuse `sample_weight` instead of ignoring it, and
  a skipped conformal calibration is recorded as a `conformal.skipped` event.
* Model bundles now checksum `schema.json` and use bundle format 2. Legacy
  format 1 bundles are refused because their schema was not authenticated.

## 0.1.0b1

First public release.

* Distribution name `scmle`; import name and command `scmle`.
  Environment variables are `SCMLE_SIGNING_KEY` and `SCMLE_TRUSTED`; the base
  exception is `ScmleError`.
* Windows, macOS and Linux are supported and tested in CI. Helper processes
  (the test-set vault and the PyTorch worker) connect over an authenticated
  local channel instead of an inherited file descriptor; trial supervision
  uses `spawn` and `psutil` on Windows; all text files are read and written
  as UTF-8 with LF line endings so signed bundles verify on every platform.
* The HTTP serving adapter refuses request bodies above 16 MiB with 413.
* Added `SECURITY.md`, a code of conduct, issue and pull-request templates,
  Dependabot, and a release workflow that checks the tag against the package
  version and changelog and creates a GitHub release.

* `FairnessSpec(mitigation="reweighing")` weights development rows so group and
  label are independent in training; locked rows are untouched.
* `selection.sensitivity_repeats` runs the sensitivity analysis on the selected
  model and stores it in `evaluation["sensitivity"]`.
* Clustering (`scmle.unsupervised.clustering.find_clusters`): k-means, Gaussian
  mixture and agglomerative at several counts, scored by silhouette,
  Davies-Bouldin, Calinski-Harabasz and bootstrap stability.
* Competing risks (`CauseSpecificCox`, Aalen-Johansen `cumulative_incidence`)
  and time-varying covariates (`TimeVaryingCox`, counting-process rows).
* The profile stage now reports label noise and, given `reference=`, a gap
  between the development rows and a reference sample.
* Fairness reweighing (`reweighing_weights`): training weights that remove
  the association of a sensitive group and the label, used through
  `weight_column`.
* Dirichlet calibration (`bundle.calibrate(..., method="dirichlet")`) for
  multiclass models.
* Sensitivity analysis (`scmle.evaluation.sensitivity`): training (seed,
  resample), evaluation (split) and performance (input noise) sensitivity.
* `RollbackPolicy.tolerance` now defaults to 0.01 so routine retraining noise
  does not undo promotions.
* Data quality: label-noise detection (`find_label_noise`) and
  representativeness against a reference sample (`compare_to_reference`).
* Lifecycle: shadow evaluation (`ShadowEvaluator`), A/B analysis with an
  independent bootstrap (`compare_arms`), and rollback criteria
  (`LifecycleManager.check_rollback`, `previous` registry alias).
* Package layout follows the section 3 module tree under a flat `scmle/`;
  modules carry their real names, with no compatibility aliases.
* Logging replaces `print` in the library, CLI, benchmarks and examples.
* PyTorch model families (MLP, residual, TabM-style ensemble,
  FT-Transformer, RealMLP-style) computed in a dedicated worker process;
  warm-up and cosine schedule, automatic batch size, gradient clipping and a
  training-health report; `deeplearning` extra.
* Bagging adapters; early stopping for LightGBM, XGBoost and CatBoost.
* Fit-health verdicts (healthy, overfit, underfit, unstable) per trial.
* Paired Bayesian tie handling in selection (`uncertainty="bayesian"`).
* Diagnostics: slices, robustness stress tests, adversarial validation,
  partial dependence, per-prediction attribution; reliability section in the
  report.
* Conformal prediction intervals and sets stored in the bundle.
* Automatic feature construction and L1 selection as tested strategies.
* Profile-driven recommendations; run audit; `scmle.predict` / `scmle.load_model`.
* Under hard enforcement the locked test rows live in a separate vault
  process.
* Rolling-origin validation and `data.feature_delays`.
* `models.exclude_models` / `models.exclude_families`.
* Datetime columns become calendar and trend features; free-text columns are
  encoded as fold-local TF-IDF terms.
* Missing-value indicator and iterative-imputation strategies; arcsinh target
  transform for long-tailed regression targets.
* `scmle.register_metric` for user-defined metrics.
* Greedy weighted blending searched alongside stacking.
* Fairness report (`data.sensitive_columns`), out-of-distribution flags in
  the bundle, a learning curve, monotonic constraints
  (`data.monotonic_constraints`).
* Per-row training weights (`data.weight_column`, `--weight-column`): used
  directly by models that accept `sample_weight`, otherwise by a weighted
  draw of training rows. Test metrics are also reported weighted
  (`weighted_test_metrics`).
* Conformalized quantile intervals (`predict_interval(method="quantile")`):
  widths follow the row's noise; the default residual method is unchanged.
* `data.time_as_feature` offers the time column to the model (opt-in).
* Unlabeled rows: `data.unlabeled` (`block`, `ignore`, `pseudo_label`);
  self-training is judged in cross-validation and used only if it wins;
  `bundle.rank_for_labeling` picks the rows to label next.
* Survival analysis: `task="survival"` with `event_column`; Cox and
  discrete-time boosted models, concordance index, `predict_survival`.
* Keras network (`keras_mlp`; Keras 3 on the PyTorch backend, in the same
  worker, with the `deeplearning` extra).
* Validation folds are scored with the weight column (`selection.weighted`),
  so selection agrees with the weighted test metrics.
* Benchmarks: `--seeds N` repeats each dataset and reports the advantage over
  the reference with a 95% interval; the verdict and the regression gate use
  the interval, so noise alone neither wins nor fails.
* Multiclass reliability report (top-label calibration error), calibration
  error before/after in `bundle.calibrate`, and a quantile-interval coverage
  curve on the locked rows (`evaluation["quantile_coverage"]`).
* ONNX export (`bundle.export_onnx`, `scmle export-onnx`, `onnx` extra)
  with a parity check; unsupported models are refused by name. `scmle
  serve` runs the HTTP server; schema violations return 422 and `/health`
  reports the model version and checksum.
* `scmle.register_estimator` registers any scikit-learn-compatible estimator
  class as a model (conformance-checked, picklable for trial processes).
* Ordinal targets: `task="ordinal"` with `ordinal_order`; thresholded
  regressors give ordered labels and probabilities, scored by quadratic
  weighted kappa and mean absolute class error.
* Survival: event-stratified splits and censoring-weighted integrated Brier
  score and dynamic AUC beside the concordance index. Competing risks and
  time-varying covariates are not built.
* Duplicate rows (equal features) are kept on one side of the holdout and of
  every fold; the manifest notes how many groups were kept together.
* Missingness report: per-column tests of whether gaps relate to the target
  (development rows, Bonferroni-corrected); missing-value flags are planned
  and built only for the flagged columns.
* `bundle.environment_lock()` writes the build environment as pinned
  requirements; a dependency mismatch at load carries it in the error.
* `bundle.counterfactual`: the smallest change to actionable columns that
  gives a wanted class or value range; unneeded changes are pruned and other
  columns never change.
* Lifecycle retraining policy: one retrain per drift event (data digest),
  a row cooldown, a minimum of labeled rows, and an audit record for every
  review, skip, rejection and promotion with its evidence.
* Production for any model: `bundle.export_service` / `scmle package`
  write a service folder (bundle, pinned requirements, build environment,
  server script, Dockerfile). ONNX also covers LightGBM and XGBoost with
  `onnxmltools`; refusals point to the service package.
* Fairness mitigation (`fairness=FairnessSpec(mitigation="group_thresholds")`):
  per-group thresholds judged on validation, reported on the locked rows,
  logged in the decision log. Reweighing and constrained fitting are not built.
* Learning to rank: `task="ranking"` with `query_column`; pointwise,
  LightGBM and XGBoost rankers, NDCG@10 and MAP@10, query-disjoint splits and
  `bundle.rank`.
* Text embeddings: `text_encoding="embedding"` encodes free text with a
  sentence-embedding model in the PyTorch worker (`embeddings` extra), each
  distinct string embedded once, falling back to TF-IDF with a warning.
* Multi-label and multi-output targets: pass a list of target columns with
  `task="multilabel"` or `"multioutput"`; iterative stratification keeps rare
  labels in every fold; `bundle.predict_frame` returns one column per target.
* Forecasting: `task="forecast"` with `horizon`, `time_column` and optional
  `group_column` and `seasonal_period`; lag and rolling features are added
  automatically, the split embargoes the horizon, MASE against no-change is
  the default metric, and `bundle.forecast(history)` forecasts any series.
  The backtest report gives per-step errors against last-value and
  seasonal-naive baselines and per-origin scores.
* Uplift modelling: `task="uplift"` with `treatment_column`; T-, S- and
  X-learners, Qini coefficient and uplift@30%, treatment/outcome-stratified
  splits, a propensity check that flags non-randomized assignment, and the
  no-unmeasured-confounding assumption stated in the report.
* Files larger than memory: `memory_budget_gb` streams a Parquet or CSV file
  in batches. Locked rows are chosen by hashing row positions, development
  uses a bounded sample, and every locked row is scored by streaming the
  file. `training="stream"` keeps incremental models (`sgd_logistic`,
  `sgd_regression`) and updates the final one over the whole development
  partition; other models are rejected with a stated reason.
* Multi-node Ray: `executor="ray"` with `cluster_address`; trial results
  match a single-node run, hard limits hold on workers, and a lost worker is
  retried once and then recorded as a failed trial instead of ending the run.
* `ProblemSpec`: state the decision, population and success criteria;
  the criteria are checked on the locked rows and shown in the summary
  (`result.meets_criteria`).
* Evaluation completeness: MRR and precision/recall@10 for ranking, optional
  nested cross-validation (`selection.nested_cv`) reporting the optimism of
  the winning score, a warning for exactly equal classes, and a note that
  validation and test keep the natural class mix.
* Bias–variance report (`selection.bias_variance_replicates`): bootstrap
  decomposition of the selected model's error with a nearest-neighbour noise
  estimate, a verdict and advice. It reproduces resampling variance only; the
  limitation is stated in the module and the report.
* PyTorch networks: optimizer choice (`adam`, `adamw`, `sgd` with momentum,
  `rmsprop`, full-batch `lbfgs`), Huber and focal losses, label smoothing and
  gradient accumulation.
* Fixes found on real data: the forecasting backtest crashed under hard
  isolation (the seasonal baseline now comes from split time), and the
  tight-budget rule rejected the boosted learners of the newer tasks.
* Tuning breadth: a grid optimizer, Hyperband brackets (`search.pruning`,
  `search.eta`) and Pareto selection (`selection.pareto`) that reports the
  score / latency / size front and picks the fastest model within a score
  tolerance.
* Optional TabPFN adapter (`tabpfn` extra; needs an accepted license).
* Benchmarks: Hugging Face source and suites (`--suite`).
* Post-selection diagnostics are bounded by a share of the time budget.

## 0.1.0b1

First public beta.

* Typed, versioned project specification; deterministic config hash.
* Leakage-safe workflow: locked test set, fold-local fitting, split
  strategies (stratified, random, group, temporal), leakage hypotheses
  including outcome-dependent recording.
* Model registry with scikit-learn, LightGBM and XGBoost adapters, cross-fitted
  stacking, pruned hyperparameter search (random or Optuna), data strategies
  (class weighting, resampling, calibration, log and interaction features),
  threshold optimization, serving-constraint selection.
* Diagnostics and explainability; decision log; HTML report.
* Resumable runs, SQLite (and MLflow) tracking, parallel trials (process
  pool or Ray), meta-learning from run history.
* Signed, checksummed bundles; refuses unauthenticated loading by default.
* Monitoring (drift with sampling-noise floor, delayed labels, advisor),
  champion-challenger, model registry, batch and HTTP serving, lifecycle
  manager.
* SVM, kNN (size-limited), CatBoost adapters; SMOTE-family and opt-in jitter
  augmentation strategies; probability calibration strategy.
* Full dataset fingerprint (shape, types, missingness, target, features,
  relationships, structure, quality, operational cost, risk).
* Structure-aware bootstrap intervals; resource governor; policy
  checkpoints; data-access ledger and audit events; explicit inference
  column mapping; replay verification.
* Declarative, versioned feature-transformation graph (18 operations,
  fold-fitted statistics, datetime parts, per-entity lag / diff / rolling
  features) packaged in the bundle.
* Arrow-compatible interchange (Arrow, Polars, Feather); Gaussian-copula
  synthetic data with a fidelity evaluator; neural-network family;
  feature-availability contract and temporal embargo.
* MLflow tracking adapter and Ray execution backend.
* CLI: `plan`, `run`, `resume`, `predict`.
* Benchmarks on nine open datasets with a regression gate.
* Python 3.11 or newer (tested 3.11 to 3.14).
* Enforced resource limits: each trial runs in a supervised process that is
  killed past its time ceiling or memory limit.
* The saved model carries per-entity history, so lag / rolling features work
  on a single new record; serving adapters keep the history current.
* Profiling, feature selection and leakage checks use the development
  partition only; trials record what each fold fit and scored on and the
  ledger verifies it against the split manifest.
* Adapter capabilities drive preprocessing and planning (native categorical
  handling, no imputation for models that handle missing values, sparse
  input); high-cardinality, regression rare-target and short-budget policies.
* Group- and time-aware inner cross-validation for calibration and stacking.
* Selection rules: tie-breaker, tolerance, one-standard-error rule, hardware
  safety factor; Nadeau-Bengio corrected confidence interval.
* Plugin discovery gated by the conformance suite; registry authenticates
  bundles; tracking lineage with artifact checksums; save-and-reload parity
  verification; optional probability calibration at inference.
* Workflow compiler (stages declare dependencies).
* Admissibility and detectors in charter runs (runs without a charter are
  unchanged): leakage suspicion tests, the admissibility table and the F5
  detector pipeline now run on the exploration set `X` only, and write
  `admissibility_table.json` and `detector_certificates.json` (the
  certificates the run used). Each leakage check is chained into the ledger as a new
  `diagnostic` row that spends no wealth. Certificates are read from
  `benchmarks/qualification/`, which the installed package does not ship; when
  the files are missing every detector is descriptive. The shipped leakage
  certificate is not gating (the noisy-copy leak type is descriptive), so a
  suspicion is reported but quarantines nothing yet. The old profiler
  constants `LEAK_ASSOCIATION`, `MISSINGNESS_LEAK` and the `LEAKAGE_MARKER`
  string match remain: they still drive `leakage_policy="block"` for runs
  without a charter.
* Robustness and shift (`scmle.data.robustness`; runs without a charter are
  unchanged). A charter run re-runs each exploration survivor over the
  multiverse the charter declared (imputation, outlier rule, inclusion; other
  declared choices are reported as not run) with bootstrap perturbation
  intervals, a survival rule (sign, size band, interval) and a permutation test
  of the whole specification curve, and compares `X` with the `reference`
  sample by a classifier two-sample test and a kernel MMD test with permutation
  p-values, per-feature KS or chi-square under BH, and subgroup coverage. New
  artifacts: `stability_report.json` and `shift_report.json`. Without a
  deployment sample the shift status is `unverifiable`, never `pass`. The
  forward-in-time check on the holdout is implemented (`forward.forward_check`,
  opens the set once) but a run reports it as skipped, because the final test
  set is opened only by the final evaluation. Certificates are in
  `benchmarks/qualification/f6_certificates.json` (`f6_certify.py`): the
  classifier test is certified (false alarms 24 in 1000 nulls, power 0.94 at a
  planted AUC of 0.60 with 2000 rows a side); the MMD test is measured and
  stays descriptive because its false-alarm upper bound (0.0765) is just over
  the pre-registered 0.075; the stability rule is certified on semi-synthetic
  plants only. Stability verdicts do not yet remove findings from the
  confirmation set, per-period drift and calibrated PSI are not implemented,
  and the per-feature p-values are adjusted within the report, not entered in
  the ledger. The old `0.65` representativeness constant is unchanged.
