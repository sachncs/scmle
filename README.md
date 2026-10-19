# super-charge-ml

[![CI](https://github.com/sachncs/magicml/actions/workflows/ci.yml/badge.svg)](https://github.com/sachncs/magicml/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/super-charge-ml)](https://pypi.org/project/super-charge-ml/)
[![Python](https://img.shields.io/pypi/pyversions/super-charge-ml)](https://pypi.org/project/super-charge-ml/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Autonomous, leakage-safe ML lifecycle framework for tabular data. Runs on
Linux, macOS and Windows with Python 3.11 or newer (tested on 3.11 to 3.14,
including the free-threaded 3.14 build). The design is described in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

```bash
pip install super-charge-ml                  # core
pip install "super-charge-ml[boosting]"      # + LightGBM, XGBoost
pip install "super-charge-ml[all]"           # most optional extras
```

On macOS, the `boosting` extra needs the OpenMP runtime (`brew install libomp`).

The PyPI distribution is `super-charge-ml`; the import name and the command
are both `scml`. **Status: beta.** The public API may change between `0.x`
releases; see [`CHANGELOG.md`](CHANGELOG.md).
The supported entry points are documented in
[`docs/API_STABILITY.md`](docs/API_STABILITY.md).
The release gate is tracked in
[`docs/PRODUCTION_READINESS.md`](docs/PRODUCTION_READINESS.md).

```python
import logging

import scml

logging.basicConfig(level=logging.INFO)
project = scml.MLProject("customers.parquet", target="churn", metric="roc_auc")
logging.info(project.plan().summary())  # inspect before any training
result = project.run()  # profile, split, search, test, package
logging.info("\n%s", result.leaderboard())
result.save("artifacts/churn", signing_key="...")  # signed bundle
result.predict("new_customers.parquet")
```

CLI: `scml plan | run | resume | predict` (see `scml --help`).
More in [`docs/TUTORIAL.md`](docs/TUTORIAL.md).

## What it does

1. **Profiles** the data: task resolution, column kinds, a full dataset
   fingerprint, and leakage *hypotheses* (including outcome-dependent
   recording such as Titanic's `boat` / `body`).
2. **Splits** without leakage (stratified / random / group / temporal) behind
   a locked test set that opens once.
3. **Engineers features** declaratively (`FeatureGraphSpec`: ratios, bins,
   clipping, datetime parts, lags and rolling statistics per entity), with
   every statistic learned on the training fold only.
4. **Searches** every compatible model family (scikit-learn, LightGBM,
   XGBoost, CatBoost, SVM, kNN), then evaluates data strategies on the
   original validation folds (class weighting, random and SMOTE-family
   resampling, copula augmentation, calibration, log / interaction features), tunes the winners
   with pruned search (random or Optuna), and assembles a cross-fitted stack
   that is adopted only if it beats its best member. Trials run in parallel
   (process pool or Ray) with failure isolation and resumable checkpoints.
5. **Selects** by cross-validation under optional serving constraints (model
   size, p95 latency), optimizes the decision threshold on out-of-fold
   predictions against an explicit cost or metric, evaluates once on the
   locked test set, and explains the result (diagnostics, permutation
   importance, structure-aware confidence intervals).
6. **Packages** a signed, checksummed bundle with schema and a monitoring
   reference, plus a decision log with the evidence behind every choice.
7. **Operates**: monitors drift and delayed-label performance, compares a
   challenger against the champion, versions and promotes models, and serves
   them in batch or over HTTP.

## Guarantees

- **Leakage-safe by construction.** Preprocessing, resampling, calibration
  and stacking are part of the candidate and fitted inside each fold;
  `tests/leakage` proves the search is invariant to the test partition.
- **Auditable.** Every run writes its configuration, split, events, trials,
  decision log, data-access ledger (verified against the split), report and
  bundle; approval checkpoints can gate the plan and the selection.
- **Conservative.** Ambiguous targets, unparseable times, too few groups or
  minority examples, impossible constraints and rejected approvals put the
  run in `BLOCKED` instead of guessing.
- **Secure by default.** Bundles are pickle-based, so loading is refused
  unless the HMAC signature verifies or you pass `trusted=True`; checksums
  are verified before anything is unpickled ([`docs/SECURITY.md`](docs/SECURITY.md)).
- **Honest about limits.** CPU / time / memory limits are enforced (no
  trial starts past a limit; every trial runs in a supervised process that is
  killed at a hard time ceiling or when its resident memory exceeds the limit) and each run reports how every limit was applied and what was
  observed; reruns are classified `exact`, `comparable` or `different`.

## Benchmarks on open datasets

`python -m benchmarks.run_benchmarks` runs the full workflow on real data and
compares it with an independent plain-scikit-learn `HistGradientBoosting`
model fitted on the **same development rows** and scored on the **same locked
test rows** (lower is better for log loss and RMSE). Results from
`benchmarks/results.json` (400 s budget, 2 workers, every optional extra
installed):

| dataset | rows | task | metric | selected | cv | test | reference |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| breast_cancer | 569 | binary | roc_auc | torch_realmlp | 0.9955 | 0.9940 | 0.9901 |
| wine | 178 | multiclass | log_loss | stacking | 0.0647 | 0.1049 | 0.0099 |
| digits | 1797 | multiclass | log_loss | catboost | 0.0840 | 0.0759 | 0.1394 |
| diabetes | 442 | regression | rmse | torch_tabm | 55.42 | 50.43 | 57.01 |
| credit-g (OpenML) | 1000 | binary | roc_auc | stacking | 0.8047 | 0.7823 | 0.7479 |
| titanic (OpenML) | 1309 | binary | roc_auc | lightgbm | 0.9944 | 0.9920 | 0.9910 |
| kc1 defects (OpenML) | 2109 | binary, 15% positive | pr_auc | stacking | 0.4773 | 0.4991 | 0.5249 |
| adult income (OpenML, 12k) | 12000 | binary | roc_auc | torch_tabm / calibrated | 0.9112 | 0.9138 | 0.9066 |
| california housing (OpenML, 10k) | 10000 | regression | rmse | stacking | 48795 | 47321 | 49378 |

Honest reading: ahead of the reference on 7 of 9. The two exceptions are
wine, whose locked test set has only 36 rows so one confident mistake
dominates log loss, and kc1, where the selected stack trails the reference on
the locked test rows by about 0.026 PR-AUC (a 15%-positive set with few
positives in the test rows, so the score is noisy). Titanic's near-perfect score comes from leaked columns (`boat`,
`body`): the profiler flags both, and `leakage_policy="block"` refuses to
proceed. Choosing the best of many trials carries a mild optimism bias, which
is why the locked-test score is the number to trust.

More datasets: on 20 Hugging Face Hub tables the framework is ahead of the same reference on 16 of 20, and on 21 further OpenML tables on 13 of 21
(including extreme class imbalance, chronological, mixed-type and
wide data). The full tables, and where it is behind, are in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## Layout

| Package | Responsibility |
| --- | --- |
| `scml.contracts` | Typed contracts: pydantic configuration, frozen dataclasses for data, metrics, artifacts, events, ABCs |
| `scml.data` | `readers` (CSV, Parquet, Arrow / Polars interchange), `profiling`, `validation` (column kinds, target and task), `curation` (fold-local preprocessing), `splitting` (leakage-safe and structure-aware strategies) |
| `scml.features` | Versioned, fitted feature graph (row-wise and series nodes) |
| `scml.sampling` | Resamplers, synthesis, and the data strategies evaluated inside folds |
| `scml.models` | `ModelAdapter` hierarchy and registry, `sklearn` and `boosting` adapters, conformance, thresholds, stacking |
| `scml.optimization` | Ask/tell optimizers (random, Optuna) |
| `scml.experimentation` | Trials, tuning, strategy and ensemble search, experiment stores (SQLite) |
| `scml.intelligence` | Dataset fingerprint, `Planner` and `PlanningRule`s, recommendations, meta-learning from run history |
| `scml.evaluation` | Metric catalog, selection policies, diagnostics, resource measurement |
| `scml.explainability` | Permutation importance, partial dependence, per-prediction occlusion |
| `scml.core` | State machine, workflow compiler, `Stage` hierarchy, policy checkpoints, limits, execution backends, configuration files, orchestrator |
| `scml.artifacts` | Signed model bundle, run store, report, replay verification |
| `scml.inference` | Column mapping, input validation, batch / HTTP serving |
| `scml.monitoring` | Reference profiles, drift, delayed-label performance, champion-challenger |
| `scml.governance` | Access ledger, content-addressed model registry, lifecycle manager, run audit |
| `scml.integrations` | Optional MLflow and Ray adapters |
| `scml.api` | `MLProject`, `RunResult`, `load_model` / `predict` |

Extension points are polymorphic: add a `TableReader`, `ColumnKind`,
`SplitStrategy`, `DataStrategy`, `ModelAdapter`, `Optimizer`, `PlanningRule`,
`SelectionPolicy`, `Diagnostic`, `PolicyCheckpoint`, `ExecutionBackend`,
`ExperimentStore`, `ServingAdapter` or `Stage` subclass; no core edits are
needed ([`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)). Third-party adapters
can also register via the `scml.models` entry-point group and must pass
`scml.models.conformance.check_conformance`.

## Conventions

Google Python style (80 columns, Google docstrings, module imports), enforced
by `ruff` and `mypy`. No underscore-prefixed names anywhere, enforced by
`tests/unit/test_conventions.py`.

```bash
uv sync --extra boosting --extra optimization
uv run ruff check . && uv run ruff format --check .
uv run mypy scml benchmarks
uv run pytest
uv run python -m benchmarks.run_benchmarks --offline --output results.json
uv run python -m benchmarks.check_regression results.json
```

Quality gates in CI: lint and types; the lowest supported dependency
versions; a base install without extras; the MLflow / Ray integrations; the
benchmark regression gate; `bandit` and `pip-audit`; and a wheel build
installed into a clean environment.

## Known limits

* Memory is sampled while a trial runs (every 0.1 s), so a very fast spike
  can overshoot once before the trial is killed. The trial's memory includes
  every process it started, and those are killed with it. `resources.enforcement: cooperative` turns
  supervision off.
* Series features (lag / rolling) at prediction time use the recent rows
  the model stored at training time (`series_history.parquet`, checksummed in
  the bundle) plus rows scored so far by the serving adapters; a record
  before an entity's stored history gets missing values by construction.
* Only the target is looked at before the split (to resolve the task and, for
  stratified splits, to choose the test rows). All other profiling, feature
  selection, leakage detection and the fingerprint use the development
  partition alone.
* The HTTP serving adapter is a minimal reference server (no authentication
  or TLS); run it behind a gateway.
* PyTorch model families ship with the `deeplearning` extra and run in a
  separate worker process (two OpenMP runtimes in one process crash); Keras
  and pretrained tabular foundation models (TabPFN) are not bundled, but
  plug in through the same `ModelAdapter` contract.
* `gpu="required"` blocks a run unless PyTorch can see a CUDA device.
* The test-set vault keeps the locked rows out of trial processes. It cannot
  stop a process from reading the original data file when a path was given,
  and a caller's own in-memory DataFrame still holds every row.
* Ray and CatBoost have no free-threaded (3.14t) wheels yet.
* Windows: peak-memory reporting counts this process only (finished child
  processes are not reported), and the free-threaded build is not tested.
