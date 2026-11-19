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
3.14, including the free-threaded 3.14 build). On Windows, CI is set up to run the
unit, property and leakage tests and the end-to-end test (their results on
Windows have not been checked here); serving, fault-injection
and deployment scenarios have only been run on Linux and macOS. New to
machine learning? Start with
[Start here](docs/START_HERE.md). Otherwise start with the
[quickstart](docs/TUTORIAL.md), then use the [architecture guide](docs/ARCHITECTURE.md)
to understand how the system is composed.

```bash
# not on PyPI yet: install from GitHub, into a fresh virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install "scmle @ git+https://github.com/sachncs/scmle"  # core
pip install "scmle[boosting] @ git+https://github.com/sachncs/scmle"  # + LightGBM, XGBoost
```

On macOS, the `boosting` extra needs the OpenMP runtime (`brew install libomp`).
`pip install "scmle[all] @ git+https://github.com/sachncs/scmle"` adds most
optional extras. It does not install on the free-threaded 3.14 build, where
Ray and CatBoost have no wheel yet; there, install the extras you need one by
one.

The PyPI distribution is `scmle`; the import name and the command
are both `scmle`. **Status: beta.** The public API may change between `0.x`
releases; see [`CHANGELOG.md`](CHANGELOG.md).
The supported entry points are documented in
[`docs/API_STABILITY.md`](docs/API_STABILITY.md).

A first run, on scikit-learn's breast-cancer table written to a Parquet file.
Run it as a script: on macOS and Windows the `if __name__` guard is required,
because the search trains in helper processes.

```python
import logging

import sklearn.datasets

import scmle.api.project


def main():
    logging.basicConfig(level=logging.INFO)
    table = sklearn.datasets.load_breast_cancer(as_frame=True).frame
    table.to_parquet("cancer.parquet")
    project = scmle.api.project.Project(
        "cancer.parquet", target="target", metric="roc_auc", time_budget=60
    )
    logging.info(project.plan().summary())  # inspect before any training
    result = project.run()  # profile, split, search, test, package
    logging.info("\n%s", result.leaderboard())
    result.save("artifacts/cancer", signing_key="your-own-secret")
    logging.info(result.predict(table.drop(columns="target").head()))


if __name__ == "__main__":
    main()
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
| Understand a stopped run, a gate or an alarm | [`docs/WHEN_THINGS_GO_WRONG.md`](docs/WHEN_THINGS_GO_WRONG.md) |
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
   such as Titanic's `boat` / `body`). A suspicion is not proof, so by
   default a flagged column is reported and the run goes on; with
   `leakage_policy="block"` any flagged column blocks the run.
2. **Splits** without leakage (stratified / random / group / temporal) behind
   a locked test set that opens once.
3. **Engineers features** declaratively (`Derivation`: ratios, bins,
   clipping, datetime parts, lags and rolling statistics per entity), with
   every statistic learned on the training fold only.
4. **Searches** the planned compatible model families (scikit-learn,
   LightGBM, XGBoost, CatBoost, SVM, kNN) within the configured time budget.
   The plan gives a reason for every technique it includes, rejects or
   defers. The search then evaluates data strategies on the
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
- **Honest about limits.** CPU / time / memory limits are enforced: the
  first trial always runs, so a result exists, and no later trial starts past
  the time budget or the memory limit; under the default hard enforcement
  every trial runs in a supervised process that is killed at a hard time
  ceiling or when its resident memory exceeds the limit. Each run reports how
  every limit was applied and what was observed; reruns are classified
  `exact`, `comparable` or `different`.

## Does it keep its promises?

`python -m benchmarks.reliability` checks nine properties on synthetic data
with known answers: same seed gives the same result, replay agrees, noise
labels score near a ROC AUC of 0.5, a planted leak is found, signed bundles
round-trip and reject tampering, 90% intervals cover, drift is flagged only
when real, and one crashing candidate cannot sink a run. See
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
| `scmle.contracts` | Typed contracts: pydantic configuration, frozen dataclasses for data, metrics, artifacts, events, ABCs, typed errors |
| `scmle.certificates` | Certificate stores written by the qualification harness and read by the gates |
| `scmle.durability`, `scmle.envelope` | Crash-safe file writes; authenticated envelopes for pickled payloads |
| `scmle.runtime` | Randomness, retries, limits, budgets, logs, workers, the scheduler and its `Executor`s, the Ray backend |
| `scmle.optimization` | Ask/tell optimizers (random, Optuna) |
| `scmle.unsupervised` | Clustering search, stability and validation, and the model behind `scmle cluster` |
| `scmle.storage` | Object stores, store conformance, experiment stores (SQLite, MLflow) |
| `scmle.data` | `readers` (CSV, TSV, JSON, Excel, Parquet, Feather, Arrow / Polars interchange), `profiling`, `validation`, `splitting`, `feature_graph` (versioned, fitted row-wise and series features), `catalog` |
| `scmle.preparation` | Fold-local preprocessing, selection, resampling, synthesis and the data strategies evaluated inside folds |
| `scmle.models` | `Model` hierarchy and registry, scikit-learn, boosting and deep-learning adapters, conformance, thresholds, stacking |
| `scmle.evaluation` | Metric catalog, selection policies, diagnostics, conformal and bootstrap intervals, fairness, resource measurement |
| `scmle.explainability`, `scmle.inference`, `scmle.assurance` | Explanations; column mapping, input validation and novelty; the ledger, charter, leakage detectors and custody |
| `scmle.causal`, `scmle.decision` | Causal identification; thresholds, net benefit and rollout decisions |
| `scmle.artifacts` | Signed model bundle, run store, report, replay verification |
| `scmle.monitoring`, `scmle.governance` | Drift, delayed-label performance, champion-challenger; audit ledger, model registry, lifecycle promotion |
| `scmle.pipeline` | Run engine, stages, trials, tuning, strategy and ensemble search, dataset fingerprint, `Planner` and `Refiner`s, recommendations |
| `scmle.deploy`, `scmle.security`, `scmle.trail` | Deployment contract and packaging; service authentication and audit; the question-to-outcome trail |
| `scmle.api` | `Project`, `Curator`, `ModalityProject`, `load_model` / `predict`, `clustering.cluster` / `evaluate` |
| `scmle.operations`, `scmle.runners`, `scmle.serving`, `scmle.workflow` | Operating a deployed model; portable training runners; the prediction servers; workflow definitions and a local executor |
| `scmle.cli`, `scmle.rehearsal` | The `scmle` command (also `python -m scmle`); the demo and fault rehearsals |

The rows run from the bottom layer to the top; each package imports only the
layers below it (`pyproject.toml`, checked by `lint-imports`).

Extension points are polymorphic: add a `Family`, `Partitioner`, `Strategy`,
`Model`, `Optimizer`, `Refiner`, `Selector`, `Diagnostic`, `Reviewer`,
`Executor`, `Store`, `Server` or `Stage` subclass
([`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)). A `tabular.Reader` subclass is
the exception: `tabular.Loader` lists its readers in a fixed default tuple,
and the CLI and `Project` build that default loader, so a new file format
needs an edit to `scmle/data/readers/tabular.py` before they read it; only
code that builds its own `Loader(readers=...)` can use it without one.
Third-party adapters can also register via the `scmle.models` entry-point
group and must pass `scmle.models.conformance.comply`.

## Conventions

Google Python style (80 columns, Google docstrings, module imports), checked
by `ruff` and `mypy`. No underscore-prefixed names anywhere, enforced by
`tests/unit/test_conventions.py::test_no_private_names` and `tests/unit/test_conventions.py::test_no_private_file_names`.

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
* The test-set vault keeps the locked rows out of trial processes. It does
  not stop a process from reading the original data file when a path was given,
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
