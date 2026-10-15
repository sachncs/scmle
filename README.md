# magicml

Autonomous, leakage-safe ML lifecycle framework for tabular data. Python 3.14
only (standard and free-threaded builds). The architecture is specified in
[`product.md`](product.md); the importable package is `mml`.

```python
import mml

project = mml.MLProject("customers.parquet", target="churn", metric="roc_auc")
print(project.plan().summary())  # inspect before any training
result = project.run()  # profile, split, search, test, package
print(result.leaderboard())
result.save("artifacts/churn", signing_key="...")  # signed bundle
result.predict("new_customers.parquet")
```

CLI: `magicml plan | run | resume | predict` (see `magicml --help`).
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
- **Honest about limits.** CPU / time / memory limits are enforced
  cooperatively and each run reports how every limit was applied and what was
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
| breast_cancer | 569 | binary | roc_auc | svm / log_skewed | 0.9950 | 0.9957 | 0.9901 |
| wine | 178 | multiclass | log_loss | stacking | 0.0546 | 0.0480 | 0.0099 |
| digits | 1797 | multiclass | log_loss | lightgbm | 0.0806 | 0.0919 | 0.1394 |
| diabetes | 442 | regression | rmse | ridge | 55.76 | 53.59 | 57.01 |
| credit-g (OpenML) | 1000 | binary | roc_auc | stacking | 0.8054 | 0.7849 | 0.7479 |
| titanic (OpenML) | 1309 | binary | roc_auc | catboost | 0.9939 | 0.9947 | 0.9910 |
| kc1 defects (OpenML) | 2109 | binary, 15% positive | pr_auc | stacking | 0.4967 | 0.5313 | 0.5249 |
| adult income (OpenML, 12k) | 12000 | binary | roc_auc | stacking | 0.9106 | 0.9154 | 0.9066 |
| california housing (OpenML, 10k) | 10000 | regression | rmse | stacking | 47866 | 47169 | 49378 |

Honest reading: ahead of the reference on 8 of 9. The exception is wine,
whose locked test set has only 36 rows, where one confident mistake dominates
log loss. Titanic's near-perfect score comes from leaked columns (`boat`,
`body`): the profiler flags both, and `leakage_policy="block"` refuses to
proceed. Choosing the best of many trials carries a mild optimism bias, which
is why the locked-test score is the number to trust.

## Layout

| Package | Responsibility |
| --- | --- |
| `mml.contracts` | Typed contracts (pydantic config, frozen dataclasses, ABCs) |
| `mml.data` | Readers, column kinds, profiling and fingerprint, split strategies, pipelines, data strategies |
| `mml.models` | `ModelAdapter` hierarchy, registry, conformance, resampling, thresholds, stacking |
| `mml.optimization` | Ask/tell optimizers (random, Optuna) |
| `mml.intelligence` | `Planner`, `PlanningRule`s, meta-learning from run history |
| `mml.evaluation` | Metric catalog, selection policies, diagnostics, resource measurement |
| `mml.core` | State machine, `Stage` hierarchy, trials, tuner, strategy and ensemble search, checkpoints, limits, execution backends, orchestrator |
| `mml.tracking` | Experiment stores (SQLite, MLflow) |
| `mml.artifacts` | Signed model bundle, run store, ledger, report, replay verification |
| `mml.monitoring` | Reference profiles, drift, delayed-label performance, champion-challenger |
| `mml.deployment` | Content-addressed model registry, batch / HTTP serving, lifecycle manager |
| `mml.api` | `MLProject`, `RunResult` |

Extension points are polymorphic: add a `TableReader`, `ColumnKind`,
`SplitStrategy`, `DataStrategy`, `ModelAdapter`, `Optimizer`, `PlanningRule`,
`SelectionPolicy`, `Diagnostic`, `PolicyCheckpoint`, `ExecutionBackend`,
`ExperimentStore`, `ServingAdapter` or `Stage` subclass; no core edits are
needed ([`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)). Third-party adapters
can also register via the `mml.models` entry-point group and must pass
`mml.models.conformance.check_conformance`.

## Conventions

Google Python style (80 columns, Google docstrings, module imports), enforced
by `ruff` and `mypy`. No underscore-prefixed names anywhere, enforced by
`tests/unit/test_conventions.py`.

```bash
uv sync --extra boosting --extra optimization
uv run ruff check . && uv run ruff format --check .
uv run mypy mml benchmarks
uv run pytest
uv run python -m benchmarks.run_benchmarks --offline --output results.json
uv run python -m benchmarks.check_regression results.json
```

Quality gates in CI: lint and types; the lowest supported dependency
versions; a base install without extras; the MLflow / Ray integrations; the
benchmark regression gate; `bandit` and `pip-audit`; and a wheel build
installed into a clean environment.

## Known limits

* Time and memory limits are cooperative (see Guarantees): a running trial is
  never interrupted, and worker processes are not individually memory-capped.
* Profiling statistics (imbalance ratio, skew, association scores) are
  computed on the whole dataset, as `product.md` orders profiling before
  splitting; stratified splits use labels to choose the test rows.
* The HTTP serving adapter is a minimal reference server (no authentication
  or TLS); run it behind a gateway.
* Series (lag / rolling) features are computed on the batch you score, which
  must therefore contain each entity's recent history.
* Large deep-learning frameworks are not bundled; the neural family is a
  scikit-learn MLP, and PyTorch / Keras plug in through the same
  `ModelAdapter` contract.
* No GPU-capable model is bundled, so `gpu="required"` blocks a run.
* Ray and CatBoost have no free-threaded (3.14t) wheels yet.
* The PyPI names `mlforge` and `magicml` are both taken; choose a distribution
  name before publishing.
