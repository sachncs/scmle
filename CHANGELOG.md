# Changelog

## Unreleased

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
