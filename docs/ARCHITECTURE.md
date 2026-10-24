# SCMLE architecture

SCMLE is a modular monolith with three planes that stay separate. The design
keeps the public surface small, moves variability behind typed contracts, and
lets concrete implementations evolve without making the orchestrator know
about every model, reader, or serving runtime.

| Plane | Decides / does | Code |
| --- | --- | --- |
| Control | what to do, when and why | `scmle.core` (orchestrator, stages, state machine), `scmle.intelligence` (planner, meta-learning) |
| Execution | transformations, training, tuning, evaluation | `scmle.data`, `scmle.models`, `scmle.optimization`, `scmle.experimentation.trials`, `scmle.core.scheduler` |
| Evidence | artifacts, measurements, decisions, provenance | `scmle.artifacts`, `scmle.experimentation.stores`, `scmle.monitoring`, `scmle.governance`, `scmle.inference` |

Modules interact only through the typed contracts in `scmle.contracts`.

## The run

`Orchestrator` walks a tuple of `Stage` objects under a `StateMachine`
(`CREATED -> VALIDATING -> PROFILING -> PLANNING -> TRAINING -> SELECTING ->
TESTING -> PACKAGING -> COMPLETED`, with `FAILED`, `CANCELLED`, `BLOCKED` as
exits). Each transition is an event in `events.jsonl` and a row update in the
tracking database.

1. **Ingest** accepts common open tabular formats (CSV/TSV, JSON records,
   newline-delimited JSON, Parquet, Feather/Arrow), pandas/Arrow-compatible
   in-memory tables, and bounded streaming sources. It fingerprints the data
   and maps infinities to missing values (reported as findings).
2. **Profile** resolves the task, splits the data (`SplitStrategy`) behind a
   `LockedTestSet`, then classifies columns (`ColumnKind`) and raises leakage
   *hypotheses* (value association, outcome-dependent recording) from the
   development rows only.
3. **Plan** asks the `Planner` (a chain of `PlanningRule`s) for candidates,
   recording every included, rejected and deferred technique.
4. **Train** screens every compatible `ModelAdapter`, then evaluates data
   strategies (`DataStrategy`), tunes the winners with pruned search, and
   assembles a cross-fitted stack. Every candidate is a complete pipeline
   (preprocessing + estimator) evaluated on the same folds, in parallel via an
   `ExecutionBackend`, with failure isolation and checkpoints.
5. **Select** ranks by cross-validation (`SelectionPolicy`) and applies serving
   constraints (size, p95 latency) measured on the running machine.
6. **Test** refits the frozen candidate, optimizes the decision threshold on
   out-of-fold predictions, opens the locked test set exactly once, and runs
   diagnostics (confusion, calibration, thresholds, residuals, permutation
   importance).
7. **Package** writes the signed model bundle (schema, checksums, monitoring
   reference profile), leaderboard, decision log and HTML report.

## Governance

* **Policy checkpoints** (`PolicyCheckpoint`) gate the run after planning and
  after selection. The selection gate fires *before the test set is opened*.
  Every verdict is a `policy.approved` / `policy.rejected` event.
* **Audit events** record the feature set and exclusions (`features.selected`),
  the split (`split.created`), the frozen selection (`selection.frozen`) and
  the single release of the test set (`test.opened`).
* **Data-access ledger** (`diagnostics/data_access.json`) states what every
  fitted operation could see and *verifies* it against the split manifest.
* **Resource governor** applies CPU / time / memory limits (every trial supervised in its own
  process, killed at a hard time ceiling or memory limit) and
  reports, per limit, how it was enforced and what was observed.
* **Replay verification** (`MLProject.verify_replay`) classifies a rerun as
  `exact`, `comparable` or `different`.

## Leakage guarantees

* The test partition is held by `LockedTestSet`; it opens once, with a token
  issued after selection.
* Preprocessing, resampling, calibration and stacking are fitted *inside*
  each training fold because they are part of the candidate estimator.
* Thresholds are chosen from out-of-fold validation predictions only.
* Strategy and ensemble decisions are made on validation folds that keep the
  original distribution.
* Confidence intervals on the test metric resample the independent units of
  observation (rows, whole groups, or contiguous time blocks).
* `tests/leakage` proves it: rewriting the test partition's labels and
  features leaves every trial, fold score and selection unchanged.

Known, documented limit: only the target is looked at before the split (to
resolve the task and, for stratified splits, to choose the test rows).
Profiling, feature selection, leakage detection and the dataset fingerprint
use the development partition alone; `tests/leakage` checks that a leak
present only in the test rows is not a profile finding.

## Extension points and polymorphism

Add a focused implementation of a contract; no core edits are needed. The
framework uses polymorphism where it buys a real product capability: model
families, data readers, split strategies, execution backends, diagnostics, and
serving adapters can vary behind the same lifecycle. It intentionally avoids
deep wrapper hierarchies and speculative interfaces; a contract should own a
decision boundary, not merely rename a function call.

| To add | Subclass | Register |
| --- | --- | --- |
| a file format | `TableReader` | `DatasetLoader(readers=...)` |
| a column family | `ColumnKind` | `columns.KINDS` |
| a feature operation | `FeatureOp` | `feature_graph.OPS` |
| a synthetic-data generator | `SyntheticGenerator` | `GeneratorOverSampler.generator_class` |
| a validation protocol | `SplitStrategy` | `splitting.STRATEGIES` |
| a model family | `ModelAdapter` | `ModelRegistry.register` or the `scmle.models` entry point |
| a data strategy | `DataStrategy` | `strategies.STRATEGIES` |
| a planning policy | `PlanningRule` | `Planner(rules=...)` |
| a search algorithm | `Optimizer` | `optimizers.OPTIMIZERS` |
| a selection rule | `SelectionPolicy` | `Orchestrator(policy=...)` |
| a diagnostic | `Diagnostic` | `diagnostics.run_suite(suite=...)` |
| an executor | `ExecutionBackend` | `execution.create_backend` |
| a tracker | `ExperimentStore` | `Orchestrator(tracker=...)` |
| a serving surface | `ServingAdapter` | direct use |
| a workflow stage | `Stage` | `Orchestrator(stages=...)` |

Third-party model adapters must pass
`scmle.models.conformance.check_conformance`.

## Conventions

Google Python style (80 columns, Google docstrings, module imports), checked
by `ruff` and `mypy`. No identifier, attribute or file name starts with an
underscore (`tests/unit/test_conventions.py`).

## Concurrency

The public API can be called from several threads of one process, including
on the free-threaded build (`tests/unit/test_thread_safety.py`).

* **Registries.** `register_estimator` / `unregister_estimator`
  (`models.registry.USER_ADAPTERS`) and `register_metric`
  (`MetricCatalog`) check and change under a lock, so a duplicate name has
  exactly one winner. Readers take a snapshot: `ModelRegistry.with_builtins`
  copies the user adapters once, and `MetricCatalog` replaces its tables
  rather than editing them, so a run that is scoring never sees a metric
  vanish halfway. A registration made while a run is in flight may or may
  not be visible to that run; register before you start.
* **Runs.** Each `MLProject.run()` owns its context, run directory (made
  with an atomic `mkdir`) and seeded random state, so parallel runs give the
  results a sequential run with the same seed gives. Runs that share an
  `output_dir` share one `experiments.sqlite`, whose connections are
  per call. `limits.forkserver_environment` reference-counts its change to
  `VECLIB_MAXIMUM_THREADS` so overlapping trials restore the original value.
* **Prediction.** `ModelBundle.predict`, `predict_proba` and `score` only
  read the bundle. `remember` (series history) is serialized by
  `history_lock`; `calibrate` and other methods that replace fitted parts are
  setup steps, not safe to run beside predictions.
* **Files.** `durability.path_lock(path)` gives every writer of one file the
  same re-entrant lock, and `durability.append_text` appends whole records.
  The model registry store, lifecycle state, incident journal, alert
  journal, prediction log and SQLite schema migration use them. Their
  read-modify-write steps (assign a version, move an alias, chain an
  incident event, dedupe an alert) are one critical section each.
* **Lazy singletons.** `worker.client()` creates the PyTorch worker client
  once; requests to the worker are serialized by its own lock.

Limits: locks are per process. Separate processes writing one registry,
journal or run directory are not coordinated (SQLite and `os.replace` keep
single files intact, but a read-modify-write can still lose an update).
`ShadowEvaluator` and `SigningKeyRegistry` are thread safe; the
`Embedder` cache and `optuna` verbosity live in worker or global state that
only a single caller drives. One `RunContext` must not be used by two
threads. Registry locks are never held while a user estimator runs, but a
`LifecycleManager.review` holds its state lock for the whole retrain.

