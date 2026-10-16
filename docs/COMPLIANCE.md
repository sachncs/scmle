# Traceability: `product.md` -> code and tests

Status is **Implemented**, **Partial** (what is missing is stated) or
**Extension point** (the contract exists, no implementation ships).
`tests/unit/test_compliance.py` fails if any cited file or test is missing.

**Executable verification:** `tests/spec/test_product_spec.py` holds one direct
check per normative statement and table of `product.md`, each named after the
spec lines it covers (`test_l<from>_l<to>_...`); it runs in the normal suite.

## 1. Architecture decisions

| Requirement | Status | Evidence |
| --- | --- | --- |
| Python 3.11+ (tested 3.11 to 3.14, standard and free-threaded) | Implemented | `pyproject.toml`; CI matrix; suites run on 3.11 and 3.14 builds |
| Open-source PyPI package, Apache-2.0 | Implemented | `LICENSE`, `.github/workflows/release.yml` |
| Modular monolith with plugin contracts | Implemented | `mml/contracts/models.py`, `tests/unit/test_component_contracts.py` |
| Local-first, distributed-capable | Implemented | `mml/core/scheduler.py`, `tests/integration/test_parallel.py`, `tests/integration/test_integrations.py` |
| Declarative Python API | Implemented | `mml/api/project.py`, `tests/integration/test_end_to_end.py` |
| Pydantic typed configuration | Implemented | `mml/contracts/tasks.py`, `tests/unit/test_tasks.py` |
| Arrow-compatible tabular abstraction | Implemented | Arrow tables, Polars frames / lazyframes and anything with the Arrow stream protocol in; Arrow out; Feather and Parquet files (`mml/data/readers/interchange.py`, `tests/integration/test_interchange.py`) |
| Scikit-learn estimator protocol | Implemented | `mml/contracts/models.py`, `tests/unit/test_registry.py` |
| Workflow as a DAG: stages declare dependencies, a compiler validates and orders them | Implemented | `mml/core/compiler.py`, `tests/unit/test_component_contracts.py::test_the_workflow_is_a_compiled_dag_of_declared_dependencies` |
| Multi-fidelity, budget-aware optimization | Implemented | `mml/experimentation/tuning.py`, `tests/integration/test_tuning.py` |
| Content-addressed, versioned artifacts | Implemented | `mml/governance/registry.py`, `tests/integration/test_deployment.py` |
| Event-driven telemetry and monitoring adapters | Implemented | `mml/contracts/events.py`, `mml/integrations/mlflow.py`, `tests/integration/test_integrations.py` |
| Reproducibility: config, data fingerprints, seeds, environments | Implemented | `mml/artifacts/reproducibility.py`, `tests/integration/test_reproducibility.py` |
| Deployment is an optional adapter | Implemented | `mml/inference/serving.py`, `tests/integration/test_deployment.py` |
| No mandatory Kubernetes / cloud / DB / GPU / tracker | Implemented | base install in `.github/workflows/ci.yml`, `tests/unit/test_component_contracts.py` |
| Control / execution / evidence planes separate | Implemented | `docs/ARCHITECTURE.md` |

## 2-5. Modules, contracts and the domain model

| Requirement | Status | Evidence |
| --- | --- | --- |
| Package organization (api, contracts, core, intelligence, data, models, optimization, evaluation, artifacts, monitoring, deployment, integrations) | Implemented | `docs/ARCHITECTURE.md` (integrations live in `mml/integrations/mlflow.py` and `mml/core/scheduler.py`) |
| Modules interact only through typed contracts | Implemented | `tests/unit/test_component_contracts.py` |
| `DataSpec` / `ValidationSpec` / `ResourceSpec` with cross-field validation | Implemented | `tests/unit/test_tasks.py` |
| Config serializable, hashable after canonicalization, saved with every run | Implemented | `tests/unit/test_tasks.py`, `tests/integration/test_end_to_end.py` |
| `DatasetManifest` / `DatasetProfile` / `ExperimentPlan` / `TrialResult` | Implemented | `mml/contracts/data.py`, `mml/contracts/experiments.py`, `tests/unit/test_contracts.py` |
| Capability discovery, error categories, versioned contracts | Implemented | `mml/contracts/errors.py`, `tests/unit/test_registry.py` |
| Ingestion: immutable input, manifest | Implemented | `tests/unit/test_readers.py` |
| Profiling: no fitted state | Implemented | `tests/unit/test_profiling.py` |
| Splitting: immutable manifest, no entity/time leakage | Implemented | `tests/unit/test_splitting.py`, `tests/property/test_invariants.py` |
| Curation / feature engineering fitted on training only | Implemented | `tests/unit/test_preprocessing.py`, `tests/leakage/test_leakage.py` |
| Sampling touches only the training fold | Implemented | `tests/unit/test_resampling.py` |
| Model registry excludes unsupported models | Implemented | `tests/unit/test_registry.py` |
| Optimizer: enforced budget, trial isolation | Implemented | `tests/integration/test_recovery.py`, `tests/integration/test_resource_limits.py` |
| Selector with defined policy | Implemented | `tests/unit/test_evaluation.py` |
| Artifact builder: loadable bundle, training/serving parity | Implemented | `tests/unit/test_bundle.py` |
| Inference: schema checked | Implemented | `tests/unit/test_bundle.py` |
| Monitoring: missing labels explicit | Implemented | `tests/integration/test_monitoring.py` |
| Conformance tests for plugins | Implemented | `mml/models/conformance.py`, `tests/unit/test_registry.py` |

## 6. Execution and lifecycle

| Requirement | Status | Evidence |
| --- | --- | --- |
| Full execution flow incl. locked test and one-time final evaluation | Implemented | `mml/core/stages.py`, `tests/leakage/test_leakage.py` |
| State machine with BLOCKED / FAILED / CANCELLED | Implemented | `tests/unit/test_state.py`, `tests/integration/test_checkpoints.py` |
| Transition events with run id, time, config version, reason, artifacts | Implemented | `tests/unit/test_contracts.py`, `tests/integration/test_governance.py` |
| Idempotent retries, checkpointing, restart from valid artifacts | Implemented | `tests/integration/test_resume.py` |
| Cancellation and budget exhaustion | Implemented | `tests/integration/test_recovery.py` |
| Consistent failure reporting with recoverable artifacts | Implemented | `tests/integration/test_recovery.py` |
| Deterministic ordering, parallel independent trials, resource isolation | Implemented | `tests/integration/test_parallel.py` |

## 7. Dataset fingerprinting

| Requirement | Status | Evidence |
| --- | --- | --- |
| All eleven dimensions (shape ... risk) | Implemented | `mml/intelligence/fingerprint.py`, `tests/unit/test_fingerprint.py` |
| ML difficulty (baseline vs best) | Implemented | `benchmarks/run_benchmarks.py` records baseline and selected CV; `mml/core/stages.py` stores `baseline_cv_score` |
| Preserves uncertainty (association is a hypothesis) | Implemented | `tests/unit/test_profiling.py` |
| Decision policy examples (dummy/linear/tree, high-cardinality, imbalance, tight budget) | Implemented | `mml/intelligence/planner.py`, `tests/unit/test_planner.py` |
| Planner proposes; evaluation decides | Implemented | `tests/integration/test_strategy_search.py` |

## 8. Model selection and tuning

| Requirement | Status | Evidence |
| --- | --- | --- |
| Stages A-E (baselines, screening, optimization, ensembles, select/package) | Implemented | `tests/integration/test_tuning.py`, `tests/integration/test_ensembles.py` |
| Baselines, linear, trees, HistGradientBoosting | Implemented | `tests/unit/test_registry.py` |
| XGBoost, LightGBM, CatBoost | Implemented | `tests/unit/test_boosting.py` |
| SVM and kNN when dataset size permits | Implemented | `tests/unit/test_registry.py` |
| Adapter declares capabilities (tasks, sparse, probabilities, native categoricals, missing values, early stopping, memory, serialization); planner reads them | Implemented | `tests/unit/test_contracts.py`, `tests/unit/test_planner.py`, `tests/spec/test_product_spec.py` |
| Optuna as optional backend behind our interface | Implemented | `mml/optimization/optimizers.py`, `tests/unit/test_optimizers.py` |
| Deep learning as model-family plugins | Implemented | `tests/unit/test_registry.py` (MLP family and third-party GPU-flagged adapter) |

## 9. Balancing and augmentation

| Requirement | Status | Evidence |
| --- | --- | --- |
| No rebalancing baseline, class weights, random over/under | Implemented | `tests/integration/test_strategy_search.py` |
| SMOTE variants for compatible tabular data | Implemented | `tests/unit/test_strategies.py`, `tests/integration/test_strategy_search.py` |
| Synthetic tabular generation as research plugin | Implemented | `SyntheticGenerator` contract, Gaussian-copula generator and `FidelityEvaluator` (`mml/sampling/synthesis.py`, `tests/unit/test_synthesis.py`); opt-in `copula_augment` strategy judged on the original folds (`tests/integration/test_strategy_search.py`) |
| Calibration evaluated separately | Implemented | `tests/integration/test_decisions.py` |
| Threshold optimization with explicit objective | Implemented | `tests/unit/test_thresholds.py`, `tests/integration/test_decisions.py` |
| Sampling inside folds, evidence on original validation distribution | Implemented | `tests/unit/test_resampling.py` |
| Regression does not reuse classification resampling | Implemented | `tests/unit/test_strategies.py` |

## 10. Evaluation

| Requirement | Status | Evidence |
| --- | --- | --- |
| Metrics per task and secondary diagnostics | Implemented | `tests/unit/test_evaluation.py`, `tests/unit/test_diagnostics.py` |
| User-selectable primary objective | Implemented | `tests/integration/test_end_to_end.py` |
| Confidence respects independent units (groups / time) | Implemented | `tests/unit/test_diagnostics.py` |
| Selection constraints (latency, size) and tie-breaker | Implemented | `tests/integration/test_constraints.py` |

## 11. Leakage prevention

| Requirement | Status | Evidence |
| --- | --- | --- |
| Immutable split manifest of row ids | Implemented | `tests/unit/test_splitting.py` |
| Separate fit/transform permissions | Implemented | `tests/unit/test_preprocessing.py` (fitting happens only inside fold pipelines) |
| Record of which data each fitted operation accessed | Implemented | `mml/governance/ledger.py`, `tests/integration/test_governance.py` |
| Locked final-test evaluator | Implemented | `tests/unit/test_splitting.py`, `tests/leakage/test_leakage.py` |
| Audit events for split / exclusion / feature changes | Implemented | `tests/integration/test_governance.py` |
| Temporal / entity data split by deployment scenario | Implemented | `tests/property/test_invariants.py` |

## 12-13. Artifacts and inference

| Requirement | Status | Evidence |
| --- | --- | --- |
| Run directory layout | Implemented | `tests/integration/test_end_to_end.py` |
| Artifact format specified; untrusted artifacts not loaded | Implemented | `docs/SECURITY.md`, `tests/unit/test_bundle.py` |
| Checksums and dependency compatibility | Implemented | `tests/unit/test_bundle.py` |
| Exact replay vs statistically comparable reruns | Implemented | `tests/integration/test_reproducibility.py` |
| Prediction through the whole fitted pipeline | Implemented | `tests/unit/test_bundle.py` |
| Schema validation, explicit column mapping, unknown categories, missing-feature diagnostics, batch, structured errors | Implemented | `tests/unit/test_bundle.py`, `tests/integration/test_deployment.py` |
| Local and batch inference first; serving later | Implemented | `tests/integration/test_deployment.py` |

## 14-17. Stack, roadmap, backlog, tests

| Requirement | Status | Evidence |
| --- | --- | --- |
| uv / pyproject, Pydantic, PyArrow, Pandas, scikit-learn, boosting, imbalanced-learn, Optuna, SQLite, MLflow, joblib, Ray, HTML+JSON reports, pytest+Hypothesis, ruff+mypy, GitHub Actions | Implemented | `pyproject.toml`, `.github/workflows/ci.yml` |
| Optional extras; no heavy imports until a model is built | Implemented | `tests/unit/test_component_contracts.py` |
| Milestones M0-M4 deliverables | Implemented | `CHANGELOG.md`, `docs/TUTORIAL.md` |
| Backlog P0 (project API, ingestion, validation strategies, leakage-safe pipelines, baselines, metrics, artifact round trip) | Implemented | `tests/integration/test_end_to_end.py` |
| Backlog P1 (registry, tuning, tracking, automated decisions, imbalance, explainability) | Implemented | `tests/integration/test_strategy_search.py`, `tests/integration/test_resume.py`, `tests/unit/test_diagnostics.py` |
| Backlog P2 (ensembles, advanced augmentation, meta-learning, deployment, monitoring) | Implemented | `tests/integration/test_ensembles.py`, `tests/integration/test_meta_learning.py`, `tests/integration/test_monitoring.py`, `tests/integration/test_deployment.py` |
| Contract, statistical, leakage, property, artifact, recovery, benchmark tests | Implemented | `tests/unit/test_component_contracts.py`, `tests/property/test_invariants.py`, `tests/leakage/test_leakage.py`, `benchmarks/check_regression.py` |
| Quality gates (fold-local state, manifests and decision logs, round-trip, limits, terminal-state reporting, plugin conformance, benchmark suite) | Implemented | `tests/integration/test_resource_limits.py`, `tests/integration/test_recovery.py`, `tests/unit/test_benchmark_gate.py` |
| Security review, dependency policy, packaging | Implemented | `docs/SECURITY.md`, `docs/DEPENDENCIES.md`, `.github/workflows/release.yml` |

## 18-20. First slice, API, future commitments

| Requirement | Status | Evidence |
| --- | --- | --- |
| v0.1 scope and the five first-sprint outcomes | Implemented | `README.md`, `docs/ARCHITECTURE.md` |
| Frozen API: `plan()`, `run(plan)`, `leaderboard()`, `save`, `predict`, `task="classification"` alias | Implemented | `tests/integration/test_end_to_end.py` |
| Deep learning as plugins | Implemented | `tests/unit/test_registry.py` |
| Time-series: separate validation and feature-availability contracts | Implemented | `tests/unit/test_splitting.py`, `tests/unit/test_profiling.py`, `tests/integration/test_end_to_end.py` (embargoed temporal validation; `unavailable_at_prediction`); lag / diff / rolling / expanding features per group in time order with no future rows (`tests/unit/test_feature_graph.py`, `tests/integration/test_feature_graph_runs.py`) |
| Augmentation: pluggable generator and evaluator interfaces | Implemented | `mml/sampling/resampling.py` (generator = `Resampler`), strategy evaluation = `DataStrategy` |
| Feature engineering: versioned, fitted transformation graph | Implemented | declarative `FeatureGraphSpec` (versioned, fingerprinted, acyclic by construction), 18 operations, statistics learned on the training fold, packaged in the bundle (`mml/features/graph.py`, `tests/unit/test_feature_graph.py`, `tests/integration/test_feature_graph_runs.py`) |
| Meta-learning: fingerprint + experiment history | Implemented | `tests/integration/test_meta_learning.py` |
| Human approval: policy checkpoints | Implemented | `tests/integration/test_checkpoints.py` |
| Production monitoring: prediction/label event schema | Implemented | `tests/integration/test_monitoring.py` |
| Distributed execution: scheduler-independent task definitions | Implemented | `mml/experimentation/trials.py`, `tests/integration/test_integrations.py` |
| Governance: audit trail for every automated decision | Implemented | `tests/integration/test_governance.py`, `tests/integration/test_strategy_search.py` |
| Benchmarking: independent suite and comparison harness | Implemented | `benchmarks/run_benchmarks.py`, `tests/integration/test_open_datasets.py` |

## Independent-audit follow-ups

Defects an independent reviewer found against `product.md`, and where each is
now closed.

| Requirement | Status | Evidence |
| --- | --- | --- |
| Resume reuses checkpoints only for the same data | Implemented | `mml/core/stages.py::refuse_stale_resume`, `tests/integration/test_resume.py::test_resume_refuses_checkpoints_computed_on_different_data` |
| Profiling, feature selection and leakage decisions use the development partition only | Implemented | `mml/core/stages.py::make_split`, `tests/leakage/test_leakage.py::test_profile_and_feature_plan_never_see_the_locked_test_rows` |
| Ambiguous / degenerate targets BLOCK the run | Implemented | `mml/data/profiling/profiler.py::resolve_auto`, `tests/integration/test_edge_cases.py::test_degenerate_targets_block_the_run_with_a_clear_reason` |
| Installed plugins are discovered and must pass the conformance suite | Implemented | `mml/models/registry.py::load_plugins`, `tests/unit/test_registry.py::test_a_default_orchestrator_discovers_conforming_plugins_only` |
| Adapter capabilities drive preprocessing and planning (missing values, sparse input, native categoricals, serializability) and are verified by the conformance suite | Implemented | `mml/data/curation/preprocessing.py::default_options`, `mml/models/conformance.py::check_declared_capabilities`, `tests/unit/test_capability_use.py` |
| High-cardinality categories: native-categorical models first, regularized target encoding evaluated | Implemented | `mml/intelligence/planner.py::HighCardinalityRule`, `mml/sampling/strategies.py::TargetEncodingStrategy`, `tests/unit/test_capability_use.py` |
| Short time budgets reduce search complexity | Implemented | `mml/contracts/tasks.py::effective_search`, `tests/unit/test_capability_use.py::test_short_budget_reduces_search_complexity_and_is_recorded` |
| Regression has its own rare-target policy (inverse-density weights); classification resampling is not reused | Implemented | `mml/sampling/strategies.py::RareTargetStrategy`, `tests/unit/test_rare_target.py` |
| Model selection: primary objective, uncertainty-aware ties, tie-breaker, serving constraints with a hardware safety factor | Implemented | `mml/evaluation/selection.py::BestMeanPolicy`, `mml/evaluation/resources.py`, `tests/unit/test_evaluation.py` |
| Confidence respects dependent folds (Nadeau-Bengio) and structure (grouped / blocked bootstraps) | Implemented | `mml/evaluation/confidence.py`, `mml/monitoring/champion.py`, `tests/unit/test_confidence.py`, `tests/integration/test_monitoring.py::test_comparison_resamples_whole_groups_and_time_blocks` |
| Inner cross-validation for calibration and stacking follows group and time structure | Implemented | `mml/data/splitting/inner_cv.py`, `tests/unit/test_inner_cv.py` |
| Group folds keep every class in every fold; temporal folds are listed latest first | Implemented | `mml/data/splitting/splitters.py`, `tests/unit/test_splitting.py` |
| The saved bundle is reloaded and verified against the in-memory model before a run completes | Implemented | `mml/artifacts/bundle.py::verify_round_trip`, `tests/unit/test_bundle.py` |
| Tracking records trial strategy, the full configuration and artifact checksums | Implemented | `mml/experimentation/stores.py`, `tests/integration/test_resume.py::test_tracking_database_records_runs_trials_and_lineage` |
| The data-access ledger verifies what each fold actually received | Implemented | `mml/governance/ledger.py::observed_access`, `tests/integration/test_governance.py::test_ledger_checks_what_trials_actually_received` |
| The registry authenticates bundles at registration and on load | Implemented | `mml/governance/registry.py`, `tests/integration/test_deployment.py` |
| Fingerprint covers ML difficulty, text / id types, scale and missingness patterns | Implemented | `mml/intelligence/fingerprint.py`, `tests/unit/test_fingerprint.py` |
| Inference: structured result, optional probability calibration, library compatibility checks | Implemented | `mml/artifacts/bundle.py::score`, `::calibrate`, `tests/unit/test_bundle.py` |
| Series features have history at prediction time | Implemented | `mml/features/graph.py::compute_with_history`, `tests/integration/test_feature_graph_runs.py` |
| Resource limits are enforced: every trial supervised in its own process, killed past the time ceiling or memory limit | Implemented | `mml/core/limits.py::TrialSupervisor`, `tests/unit/test_trial_supervision.py`, `tests/integration/test_resource_limits.py::test_a_trial_past_the_hard_ceiling_is_killed_and_the_run_continues` |
| CLI exposes validation, embargo, strategies, enforcement and nested specifications | Implemented | `mml/cli.py`, `tests/integration/test_cli.py::test_config_file_and_flags_reach_the_full_specification` |

Deliberate deviation: the package layout in section 3 is a recommendation; the
code uses a flat `mml/` package (no `src/`), as the project owner instructed.

## 21. Risks

| Risk | Mitigation shipped | Evidence |
| --- | --- | --- |
| Excessive automation produces invalid models | Validation invariants, BLOCKED states | `tests/integration/test_end_to_end.py` |
| Slower than existing AutoML | Budget-aware search, baselines | `tests/integration/test_resource_limits.py`, `benchmarks/results.json` |
| Inconsistent transforms at inference | Whole fitted graph serialized | `tests/unit/test_bundle.py` |
| Leakage inflates performance | Partition-aware execution, adversarial tests | `tests/leakage/test_leakage.py` |
| Dependency conflicts | Extras, capability checks, tested lower bounds | `docs/DEPENDENCIES.md` |
| Meta-learning overfits benchmarks | Same-dataset history excluded | `tests/integration/test_meta_learning.py` |
| Artifact incompatibility | Explicit schema and version policy | `tests/unit/test_bundle.py` |
| Broad scope delays releases | Vertical slice first | `CHANGELOG.md` |
