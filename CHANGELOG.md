# Changelog

## Unreleased

* Dirichlet calibration (`bundle.calibrate(..., method="dirichlet")`) for
  multiclass models.
* Sensitivity analysis (`mml.evaluation.sensitivity`): training (seed,
  resample), evaluation (split) and performance (input noise) sensitivity.
* `RollbackPolicy.tolerance` now defaults to 0.01 so routine retraining noise
  does not undo promotions.
* `docs/CURRICULUM.md`: the eight modelling stages mapped to modules, tests
  and equations, with one executed example per stage.
* Data quality: label-noise detection (`find_label_noise`) and
  representativeness against a reference sample (`compare_to_reference`).
* Lifecycle: shadow evaluation (`ShadowEvaluator`), A/B analysis with an
  independent bootstrap (`compare_arms`), and rollback criteria
  (`LifecycleManager.check_rollback`, `previous` registry alias).
* Package layout follows the section 3 module tree under a flat `mml/`;
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
* Profile-driven recommendations; run audit; `mml.predict` / `mml.load_model`.
* Under hard enforcement the locked test rows live in a separate vault
  process.
* Rolling-origin validation and `data.feature_delays`.
* `models.exclude_models` / `models.exclude_families`.
* Datetime columns become calendar and trend features; free-text columns are
  encoded as fold-local TF-IDF terms.
* Missing-value indicator and iterative-imputation strategies; arcsinh target
  transform for long-tailed regression targets.
* `mml.register_metric` for user-defined metrics.
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
* ONNX export (`bundle.export_onnx`, `magicml export-onnx`, `onnx` extra)
  with a parity check; unsupported models are refused by name. `magicml
  serve` runs the HTTP server; schema violations return 422 and `/health`
  reports the model version and checksum.
* `mml.register_estimator` registers any scikit-learn-compatible estimator
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
* Production for any model: `bundle.export_service` / `magicml package`
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
