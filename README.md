# magicml

Autonomous, leakage-safe ML lifecycle framework for tabular data.
Python 3.14 only. The architecture is specified in [`product.md`](product.md);
the importable package is `mml`.

```python
import mml

project = mml.MLProject("customers.parquet", target="churn", metric="roc_auc")
print(project.plan().summary())  # inspect before any training
result = project.run()  # profile, split, search, test, package
print(result.leaderboard())
result.save("artifacts/churn_model")
result.predict("new_customers.parquet")
```

CLI: `magicml plan|run|predict` (see `magicml --help`).

## What it does

1. **Profiles** the data, resolves the task, and raises *hypotheses* about
   leakage (including outcome-dependent recording such as Titanic's `boat`).
2. **Splits** without leakage (stratified / random / group / temporal) and
   locks the test set until selection is frozen.
3. **Screens** every compatible model family, then tries data strategies
   (class weighting, fold-internal resampling, log / interaction features) on
   the original validation folds, then **tunes** the winners with pruned
   search (random or Optuna). Trials run in parallel (`workers=`).
4. **Selects** by cross-validation, evaluates once on the locked test set,
   and **packages** a checksummed bundle. Every decision, with its evidence,
   is in `diagnostics/decision_log.json`.
5. **Resumes** interrupted runs (`MLProject.resume`) reusing finished trials;
   all runs and trials land in a local SQLite tracking database.

## Guarantees

- **Leakage-safe by construction.** The test partition sits behind
  `LockedTestSet`, opened once. Preprocessing and any resampling are part of
  the candidate and fitted inside each cross-validation fold.
- **Auditable.** Every run writes `config.json`, split manifest,
  `events.jsonl`, trials, leaderboard, evaluation, decision log, report and
  model bundle; every state change is an event.
- **Conservative.** Ambiguous targets, unparseable time columns, too few
  groups or minority examples put the run in `BLOCKED` instead of guessing.
- **Safe artifacts.** Bundles carry a schema and SHA-256 checksums and verify
  dependency compatibility on load. They are pickle-based: load trusted
  bundles only.

## Benchmarks on open datasets

`python -m benchmarks.run_benchmarks` runs the full workflow on real data and
compares with an independent plain-scikit-learn `HistGradientBoosting` model
fitted on the **same development rows** and scored on the **same locked test
rows** (lower is better for log loss and RMSE). Results from
`benchmarks/results.json` (300 s budget, 2 workers):

| dataset | rows | task | metric | selected | cv | test | reference |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| breast_cancer | 569 | binary | roc_auc | logistic_regression/log_skewed | 0.9940 | 0.9960 | 0.9901 |
| wine | 178 | multiclass | log_loss | lightgbm | 0.0561 | 0.0126 | 0.0099 |
| digits | 1797 | multiclass | log_loss | lightgbm | 0.0818 | 0.0984 | 0.1394 |
| diabetes | 442 | regression | rmse | ridge | 55.76 | 53.59 | 57.01 |
| credit-g (OpenML) | 1000 | binary | roc_auc | random_forest | 0.7961 | 0.8102 | 0.7479 |
| titanic (OpenML) | 1309 | binary | roc_auc | xgboost | 0.9942 | 0.9955 | 0.9910 |
| kc1 defects (OpenML) | 2109 | binary, 15% positive | pr_auc | random_forest | 0.4796 | 0.4638 | 0.5249 |
| adult income (OpenML, 12k) | 12000 | binary | roc_auc | lightgbm/class_weight | 0.9097 | 0.9144 | 0.9066 |
| california housing (OpenML, 10k) | 10000 | regression | rmse | hist_gradient_boosting | 49602 | 48796 | 49378 |

Honest reading: ahead of the reference on 7 of 9, within noise on wine (36
test rows) and behind on kc1 (about 65 test positives; the gap is under one
cross-validation standard deviation). Titanic's near-perfect score comes from
leaked columns (`boat`, `body`): the profiler flags both, and
`leakage_policy="block"` refuses to proceed. Selecting the best of many trials
carries a mild optimism bias, which is why the locked-test score is the number
to trust.

## Layout

| Package | Responsibility |
| --- | --- |
| `mml.contracts` | Typed contracts (pydantic config, frozen dataclasses, ABCs) |
| `mml.data` | Readers, column kinds, profiling, split strategies, pipelines, data strategies |
| `mml.models` | `ModelAdapter` hierarchy (sklearn, LightGBM, XGBoost), registry, conformance, resampling |
| `mml.optimization` | Ask/tell optimizers (random, Optuna) |
| `mml.intelligence` | `Planner` and composable `PlanningRule`s |
| `mml.evaluation` | Metric catalog, selection policies |
| `mml.core` | State machine, `Stage` hierarchy, trial runner, tuner, strategy search, execution backends, orchestrator |
| `mml.tracking` | Experiment stores (SQLite) |
| `mml.artifacts` | Model bundle, run store, report |
| `mml.api` | `MLProject`, `RunResult` |

Extension points are polymorphic: add a `TableReader`, `ColumnKind`,
`SplitStrategy`, `DataStrategy`, `ModelAdapter`, `Optimizer`, `PlanningRule`,
`SelectionPolicy`, `ExecutionBackend`, `ExperimentStore` or `Stage` subclass;
no core edits are needed. Third-party adapters can also register via the
`mml.models` entry-point group and must pass
`mml.models.conformance.check_conformance`.

## Conventions

Google Python style (80 columns, Google docstrings, module imports), enforced
by `ruff` and `mypy`. No underscore-prefixed names, enforced by
`tests/unit/test_conventions.py`.

```bash
uv sync --extra optimization --extra boosting
uv run ruff check . && uv run ruff format --check .
uv run mypy mml
uv run pytest
```

## Not yet implemented

Ensembles / stacking, probability calibration and threshold optimization,
meta-learning, deployment and monitoring adapters, MLflow and Ray adapters
(roadmap M3 remainder and later in `product.md`). CatBoost is its own extra
because it has no CPython 3.14 wheel yet. The PyPI names `mlforge` and
`magicml` are both taken; choose a distribution name before publishing.
