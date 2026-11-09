# SCMLE

> Mathematics meets engineering.

[![CI](https://github.com/sachncs/scmle/actions/workflows/ci.yml/badge.svg)](https://github.com/sachncs/scmle/actions/workflows/ci.yml)
![Status](https://img.shields.io/badge/status-beta-orange)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

SCMLE is a curation-first machine learning lifecycle for teams that need
reproducible decisions, not just a high score. It connects statistical
discipline with production engineering: typed contracts, leakage-safe
evaluation, signed artifacts, operational checkpoints, and monitoring.

Runs on Linux, macOS and Windows with Python 3.11 or newer (tested on 3.11 to
3.14, including the free-threaded 3.14 build). New to machine learning? Start with
[Start here](docs/START_HERE.md). Otherwise start with the
[quickstart](docs/TUTORIAL.md), then use the [architecture guide](docs/ARCHITECTURE.md)
to understand how the system is composed.

```bash
# not on PyPI yet: install from GitHub
pip install "scmle @ git+https://github.com/sachncs/scmle"  # core
pip install "scmle[boosting] @ git+https://github.com/sachncs/scmle"  # + LightGBM, XGBoost
pip install "scmle[all] @ git+https://github.com/sachncs/scmle"  # most optional extras
```

On macOS, the `boosting` extra needs the OpenMP runtime (`brew install libomp`).

The PyPI distribution is `scmle`; the import name and the command
are both `scmle`. **Status: beta.** The public API may change between `0.x`
releases; see [`CHANGELOG.md`](CHANGELOG.md).
The supported entry points are documented in
[`docs/API_STABILITY.md`](docs/API_STABILITY.md).

```python
import logging

import scmle.api.project

logging.basicConfig(level=logging.INFO)
project = scmle.api.project.Project(
    "customers.parquet", target="churn", metric="roc_auc"
)
logging.info(project.plan().summary())  # inspect before any training
result = project.run()  # profile, split, search, test, package
logging.info("\n%s", result.leaderboard())
result.save("artifacts/churn", signing_key="...")  # signed with your key
result.predict("new_customers.parquet")
```

CLI: `scmle plan | run | resume | catalog | predict | serve | export-onnx | package`
(see [`docs/CLI.md`](docs/CLI.md)).
More in [`docs/TUTORIAL.md`](docs/TUTORIAL.md).

## Documentation map

| Need | Start here |
| --- | --- |
| Build your first model, no experience needed | [`docs/START_HERE.md`](docs/START_HERE.md) |
| Look up a word | [`docs/GLOSSARY.md`](docs/GLOSSARY.md) |
| Decide what to predict, measure and avoid | [`docs/DECISIONS.md`](docs/DECISIONS.md) |
| Read a report, tiers and claims | [`docs/READING_RESULTS.md`](docs/READING_RESULTS.md) |
| Understand a refusal, gate or alarm | [`docs/WHEN_THINGS_GO_WRONG.md`](docs/WHEN_THINGS_GO_WRONG.md) |
| Know the limits | [`docs/WHAT_SCMLE_CANNOT_DO.md`](docs/WHAT_SCMLE_CANNOT_DO.md) |
| Understand leakage flags and their measured error rates | [`docs/LEAKAGE.md`](docs/LEAKAGE.md) |
| Browse every page | [`docs/README.md`](docs/README.md) |
| Learn the product | [`docs/TUTORIAL.md`](docs/TUTORIAL.md), [`examples/`](examples/README.md) |
| Use the command line | [`docs/CLI.md`](docs/CLI.md) |
| Know what a run writes | [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md) |
| Understand the system | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Choose an input and validation strategy | [`docs/OPERATING_LOOP.md`](docs/OPERATING_LOOP.md) |
| Know what runs unattended and what still needs a person | [`docs/AUTONOMY.md`](docs/AUTONOMY.md) |
| Integrate the Python API | [`docs/API_STABILITY.md`](docs/API_STABILITY.md) |
| Run and recover safely | [`docs/OPERATIONS_REHEARSAL.md`](docs/OPERATIONS_REHEARSAL.md) |
| Review security boundaries | [`docs/SECURITY.md`](docs/SECURITY.md) |
| Inspect dependency policy | [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md) |
| Compare benchmark results and reliability checks | [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md), [`benchmarks/`](benchmarks/README.md) |
| Cut a release | [`docs/RELEASING.md`](docs/RELEASING.md) |

The hosted documentation is available at
[`sachncs.github.io/scmle`](https://sachncs.github.io/scmle/).

## What it does

1. **Profiles** the data: task resolution, column kinds, a full dataset
   fingerprint, and leakage *suspicions* from a permutation screen with
   Benjamini-Yekutieli error control (including outcome-dependent recording
   such as Titanic's `boat` / `body`). A suspicion is not proof; only
   mechanisms with a passing certificate may block a run.
2. **Splits** without leakage (stratified / random / group / temporal) behind
   a locked test set that opens once.
3. **Engineers features** declaratively (`FeatureGraph`: ratios, bins,
   clipping, datetime parts, lags and rolling statistics per entity), with
   every statistic learned on the training fold only.
4. **Searches** the planned compatible model families (scikit-learn,
   LightGBM, XGBoost, CatBoost, SVM, kNN) within the configured time budget
   (the plan lists what it included, rejected and deferred), then evaluates data strategies on the
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
6. **Packages** a checksummed bundle (signed when you configure a signing
   key) with schema and a monitoring
   reference, plus a decision log with the evidence behind every choice.
7. **Operates**: monitors drift and delayed-label performance, compares a
   challenger against the champion, versions and promotes models, and serves
   them in batch or over HTTP.

## Guarantees

- **Leakage-safe by construction.** Preprocessing, resampling, calibration
  and stacking are part of the candidate and fitted inside each fold;
  `tests/leakage` checks, on the tested scenarios, that the search result does
  not change when the held-out rows are perturbed (evidence under those
  conditions, not a proof for every input).
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

## Does it keep its promises?

`python -m benchmarks.reliability` checks nine properties on synthetic data
with known answers: same seed gives the same result, replay agrees, noise
labels never score well, a planted leak is found, signed bundles round-trip
and reject tampering, 90% intervals cover, drift is flagged only when real, and
one crashing candidate cannot sink a run. See
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

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
`body`): the leakage screen flags both (`boat` by skill, missingness and record
purity; `body` by record purity only, because it is recorded for 9% of
passengers, all of whom died), and `leakage_policy="block"` (opt-in; the default is
`"warn"`) refuses to proceed. See [`docs/LEAKAGE.md`](docs/LEAKAGE.md). Choosing the best of many trials carries a mild optimism bias, which
is why the locked-test score is the number to trust.

More datasets: on 20 Hugging Face Hub tables the framework is ahead of the same reference on 16 of 20, and on 21 further OpenML tables on 13 of 21
(including extreme class imbalance, chronological, mixed-type and
wide data). The full tables, and where it is behind, are in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## Layout

| Package | Responsibility |
| --- | --- |
| `scmle.contracts` | Typed contracts: pydantic configuration, frozen dataclasses for data, metrics, artifacts, events, ABCs |
| `scmle.data` | `readers` (CSV, Parquet, Arrow / Polars interchange), `profiling`, `validation` (column kinds, target and task), `curation` (fold-local preprocessing), `splitting` (leakage-safe and structure-aware strategies) |
| `scmle.features` | Versioned, fitted feature graph (row-wise and series nodes) |
| `scmle.sampling` | Resamplers, synthesis, and the data strategies evaluated inside folds |
| `scmle.models` | `Model` hierarchy and registry, `sklearn` and `boosting` adapters, conformance, thresholds, stacking |
| `scmle.optimization` | Ask/tell optimizers (random, Optuna) |
| `scmle.experimentation` | Trials, tuning, strategy and ensemble search, experiment stores (SQLite) |
| `scmle.intelligence` | Dataset fingerprint, `Planner` and `Refiner`s, recommendations, meta-learning from run history |
| `scmle.evaluation` | Metric catalog, selection policies, diagnostics, resource measurement |
| `scmle.explainability` | Permutation importance, partial dependence, per-prediction occlusion |
| `scmle.core` | State machine, workflow compiler, `Stage` hierarchy, policy checkpoints, limits, execution backends, configuration files, orchestrator |
| `scmle.artifacts` | Signed model bundle, run store, report, replay verification |
| `scmle.inference` | Column mapping, input validation, batch / HTTP serving |
| `scmle.monitoring` | Reference profiles, drift, delayed-label performance, champion-challenger |
| `scmle.governance` | Access ledger, content-addressed model registry, lifecycle manager, run audit |
| `scmle.integrations` | Optional MLflow and Ray adapters |
| `scmle.unsupervised` | Clustering model search and scoring |
| `scmle.durability` | Atomic, crash-safe file writes used by the stores |
| `scmle.api` | `Project`, `RunResult`, `load_model` / `predict` |

Extension points are polymorphic: add a `TableReader`, `ColumnKind`,
`Partitioner`, `Strategy`, `Model`, `Optimizer`, `Refiner`,
`Selector`, `Diagnostic`, `Reviewer`, `Executor`,
`ExperimentStore`, `Server` or `Stage` subclass; no core edits are
needed ([`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)). Third-party adapters
can also register via the `scmle.models` entry-point group and must pass
`scmle.models.conformance.check_conformance`.

## Conventions

Google Python style (80 columns, Google docstrings, module imports), enforced
by `ruff` and `mypy`. No underscore-prefixed names anywhere, enforced by
`tests/unit/test_conventions.py`.

```bash
uv sync --extra boosting --extra optimization --extra multimodal
uv run ruff check . && uv run ruff format --check .
uv run mypy scmle benchmarks
uv run pytest
uv run python -m benchmarks.run_benchmarks --offline --output results.json
uv run python -m benchmarks.check_regression results.json
uv run python -m benchmarks.reliability
```

Quality gates in CI: lint and types; the lowest supported dependency
versions; a base install without extras; the MLflow / Ray integrations; the
benchmark regression and reliability gates; `bandit` and `pip-audit`; and a wheel build
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
* The HTTP serving adapter is a minimal reference server: it verifies bearer
  tokens when configured (and refuses a non-loopback bind without them) but
  has no TLS; terminate TLS at an ingress.
* PyTorch and Keras model families ship with the `deeplearning` extra and run
  in a separate worker process (two OpenMP runtimes in one process crash).
  TabPFN (the `tabpfn` extra) stays extension-only and needs its own license
  and token; other pretrained models plug in through the `Model`
  contract.
* `gpu="required"` blocks a run unless PyTorch can see a CUDA device.
* The test-set vault keeps the locked rows out of trial processes. It cannot
  stop a process from reading the original data file when a path was given,
  and a caller's own in-memory DataFrame still holds every row.
* Ray and CatBoost have no free-threaded (3.14t) wheels yet.
* Threads: the public API may be called from several threads of one process
  (registries, runs, prediction, stores and journals; see "Concurrency" in
  `docs/ARCHITECTURE.md`). Locks are per process, so separate processes that
  share one registry, journal or run directory are not coordinated, and
  mutating a shared `Bundle` (`calibrate`) while others predict is not
  supported.
* Windows: peak-memory reporting counts this process only (finished child
  processes are not reported), and the free-threaded build is not tested.
