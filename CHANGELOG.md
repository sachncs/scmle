# Changelog

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
