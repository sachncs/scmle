# Architecture

scml is a modular monolith with three planes that stay separate:

| Plane | Decides / does | Code |
| --- | --- | --- |
| Control | what to do, when and why | `scml.core` (orchestrator, stages, state machine), `scml.intelligence` (planner, meta-learning) |
| Execution | transformations, training, tuning, evaluation | `scml.data`, `scml.models`, `scml.optimization`, `scml.experimentation.trials`, `scml.core.scheduler` |
| Evidence | artifacts, measurements, decisions, provenance | `scml.artifacts`, `scml.experimentation.stores`, `scml.monitoring`, `scml.governance`, `scml.inference` |

Modules interact only through the typed contracts in `scml.contracts`.

## The run

`Orchestrator` walks a tuple of `Stage` objects under a `StateMachine`
(`CREATED -> VALIDATING -> PROFILING -> PLANNING -> TRAINING -> SELECTING ->
TESTING -> PACKAGING -> COMPLETED`, with `FAILED`, `CANCELLED`, `BLOCKED` as
exits). Each transition is an event in `events.jsonl` and a row update in the
tracking database.

1. **Ingest** reads CSV/Parquet, fingerprints the data, maps infinities to
   missing values (reported as findings).
2. **Profile** resolves the task, classifies columns (`ColumnKind`), and raises
   leakage *hypotheses* (value association, outcome-dependent recording).
3. **Plan** splits the data (`SplitStrategy`) behind a `LockedTestSet` and asks
   the `Planner` (a chain of `PlanningRule`s) for candidates, recording every
   included, rejected and deferred technique.
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

Known, documented limits: profiling statistics (imbalance ratio, skew,
leakage associations) are computed on the whole dataset, as the design specification
orders profiling before splitting; stratified splits legitimately use labels
to choose the test rows.

## Extension points

Add a subclass; no core edits are needed.

| To add | Subclass | Register |
| --- | --- | --- |
| a file format | `TableReader` | `DatasetLoader(readers=...)` |
| a column family | `ColumnKind` | `columns.KINDS` |
| a feature operation | `FeatureOp` | `feature_graph.OPS` |
| a synthetic-data generator | `SyntheticGenerator` | `GeneratorOverSampler.generator_class` |
| a validation protocol | `SplitStrategy` | `splitting.STRATEGIES` |
| a model family | `ModelAdapter` | `ModelRegistry.register` or the `scml.models` entry point |
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
`scml.models.conformance.check_conformance`.

## Conventions

Google Python style (80 columns, Google docstrings, module imports), checked
by `ruff` and `mypy`. No identifier, attribute or file name starts with an
underscore (`tests/unit/test_conventions.py`).
