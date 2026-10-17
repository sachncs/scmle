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
| Event-driven telemetry and monitoring adapters | Implemented | `mml/contracts/events.py`, `mml/integrations/mlflow_store.py`, `tests/integration/test_integrations.py` |
| Reproducibility: config, data fingerprints, seeds, environments | Implemented | `mml/artifacts/reproducibility.py`, `tests/integration/test_reproducibility.py` |
| Deployment is an optional adapter | Implemented | `mml/inference/serving.py`, `tests/integration/test_deployment.py` |
| No mandatory Kubernetes / cloud / DB / GPU / tracker | Implemented | base install in `.github/workflows/ci.yml`, `tests/unit/test_component_contracts.py` |
| Control / execution / evidence planes separate | Implemented | `docs/ARCHITECTURE.md` |

## 2-5. Modules, contracts and the domain model

| Requirement | Status | Evidence |
| --- | --- | --- |
| Package organization (api, contracts, core, intelligence, data, models, optimization, evaluation, artifacts, monitoring, deployment, integrations) | Implemented | `docs/ARCHITECTURE.md` (integrations live in `mml/integrations/mlflow_store.py` and `mml/core/scheduler.py`) |
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
| Quality gates (fold-local state, manifests and decision logs, round-trip, limits, terminal-state reporting, plugin conformance, benchmark suite) | Implemented | `tests/integration/test_resource_limits.py`, `tests/integration/test_recovery.py`, `tests/benchmarks/test_benchmark_gate.py` |
| Security review, dependency policy, packaging | Implemented | `docs/SECURITY.md`, `docs/DEPENDENCIES.md`, `.github/workflows/release.yml` |

## 18-20. First slice, API, future commitments

| Requirement | Status | Evidence |
| --- | --- | --- |
| v0.1 scope and the five first-sprint outcomes | Implemented | `README.md`, `docs/ARCHITECTURE.md` |
| Frozen API: `plan()`, `run(plan)`, `leaderboard()`, `save`, `predict`, `task="classification"` alias | Implemented | `tests/integration/test_end_to_end.py` |
| Deep learning as plugins | Implemented | `tests/unit/test_registry.py` |
| Time-series: separate validation and feature-availability contracts | Implemented | `tests/unit/test_splitting.py`, `tests/unit/test_profiling.py`, `tests/integration/test_end_to_end.py` (embargoed temporal validation; `unavailable_at_prediction`); lag / diff / rolling / expanding features per group in time order with no future rows (`tests/unit/test_feature_graph.py`, `tests/integration/test_feature_graph_runs.py`) |
| Augmentation: pluggable generator and evaluator interfaces | Implemented | `mml/sampling/resampling.py` (generator = `Resampler`), strategy evaluation = `DataStrategy` |
| Feature engineering: versioned, fitted transformation graph | Implemented | declarative `FeatureGraphSpec` (versioned, fingerprinted, acyclic by construction), 18 operations, statistics learned on the training fold, packaged in the bundle (`mml/features/feature_graph.py`, `tests/unit/test_feature_graph.py`, `tests/integration/test_feature_graph_runs.py`) |
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
| Ambiguous / degenerate targets BLOCK the run | Implemented | `mml/data/validation/target.py::resolve_auto`, `tests/integration/test_edge_cases.py::test_degenerate_targets_block_the_run_with_a_clear_reason` |
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
| Series features have history at prediction time | Implemented | `mml/features/feature_graph.py::compute_with_history`, `tests/integration/test_feature_graph_runs.py` |
| Resource limits are enforced: every trial supervised in its own process, killed past the time ceiling or memory limit | Implemented | `mml/core/limits.py::TrialSupervisor`, `tests/unit/test_trial_supervision.py`, `tests/integration/test_resource_limits.py::test_a_trial_past_the_hard_ceiling_is_killed_and_the_run_continues` |
| CLI exposes validation, embargo, strategies, enforcement and nested specifications | Implemented | `mml/cli.py`, `tests/integration/test_cli.py::test_config_file_and_flags_reach_the_full_specification` |

Deliberate deviation: section 3 recommends a `src` layout; the module tree is
followed under a flat `mml/` package (no `src/`), as the project owner instructed.

## Beyond the first slice

Capabilities added after the first vertical slice, each with its tests.

| Requirement | Status | Evidence |
| --- | --- | --- |
| Deep learning registered as model-family plugins: PyTorch MLP, residual network, TabM-style ensemble, FT-Transformer, a RealMLP-style network and a Keras network (Keras 3 on the PyTorch backend), all computed in a worker process so the framework never imports torch | Implemented | `mml/models/deeplearning/networks.py`, `mml/models/deeplearning/keras_engine.py`, `mml/models/deeplearning/worker.py::WorkerClient`, `tests/unit/test_keras.py`, `tests/unit/test_deeplearning.py::test_the_framework_process_never_imports_torch`, `tests/unit/test_deeplearning.py::test_torch_shares_a_process_with_xgboost_and_lightgbm` |
| Training health: warm-up and cosine learning-rate schedule, automatic batch size, gradient clipping, divergence and vanishing / exploding detection | Implemented | `mml/models/deeplearning/training.py::TrainingMonitor`, `tests/unit/test_training.py::test_non_finite_values_mean_divergence_and_stop_the_run` |
| Bagging for unstable learners | Implemented | `mml/models/sklearn/bagging.py::BaggedAdapter`, `tests/unit/test_bagging.py::test_the_bagger_wraps_the_base_model_and_applies_both_parameter_sets` |
| Early stopping of boosting on an internal holdout | Implemented | `mml/models/boosting/early_stopping.py::HoldoutEarlyStopping`, `tests/unit/test_early_stopping.py::test_lightgbm_stops_well_before_the_round_limit` |
| Under- and over-training: a fit-health verdict per trial and for the selected model | Implemented | `mml/evaluation/fit_health.py::assess`, `tests/unit/test_fit_health.py::test_a_model_that_memorises_is_overfit` |
| Paired Bayesian comparison drives tie handling in selection | Implemented | `mml/evaluation/comparison.py::compare_paired`, `mml/evaluation/selection.py::BestMeanPolicy`, `tests/unit/test_evaluation.py::test_bayesian_selection_prefers_the_simpler_model_when_gain_is_noise` |
| Stress tests: score under missing values, noise and unseen categories | Implemented | `mml/evaluation/robustness.py::RobustnessReport`, `tests/unit/test_robustness.py::test_every_scenario_is_scored_and_noise_hurts_a_signal_model` |
| Subgroup performance with the worst slices surfaced | Implemented | `mml/evaluation/slices.py::SliceReport`, `tests/unit/test_slices.py::test_the_worst_slice_is_the_region_the_model_cannot_learn` |
| Adversarial validation between development and test rows | Implemented | `mml/evaluation/shift.py::AdversarialValidation`, `tests/unit/test_shift.py::test_a_moved_feature_is_detected_and_named` |
| Conformal prediction intervals and sets with a stated coverage, stored in the bundle | Implemented | `mml/evaluation/conformal.py::ConformalCalibrator`, `tests/unit/test_conformal.py::test_regression_intervals_cover_at_the_requested_rate`, `tests/integration/test_end_to_end.py::test_regression_bundles_return_calibrated_intervals` |
| Explanations: permutation importance, partial dependence, per-prediction attribution | Implemented | `mml/explainability/local.py::OcclusionExplainer`, `tests/unit/test_explainability.py::test_partial_dependence_follows_the_driving_feature` |
| Automatic feature construction (differences, ratios) and L1 feature selection, tested as strategies | Implemented | `mml/data/curation/selection.py::SparseSelector`, `tests/unit/test_selection.py::test_the_sparse_selector_drops_noise_and_keeps_signal`, `tests/unit/test_strategies.py::test_constructed_ratios_let_a_linear_model_learn_a_ratio_target` |
| Profile-driven recommendations for the user | Implemented | `mml/intelligence/recommendations.py::recommend`, `tests/unit/test_recommendations.py::test_severe_imbalance_suggests_metric_and_threshold` |
| Locked test rows held by a separate vault process under hard enforcement | Implemented | `mml/governance/vault.py::SealedTestSet`, `tests/unit/test_vault.py::test_the_rows_are_released_once_to_the_token_holder`, `tests/leakage/test_leakage.py::test_hard_enforcement_moves_the_test_rows_out_of_reach` |
| Run audit against lifecycle and locked-test invariants | Implemented | `mml/governance/audit.py::RunAuditor`, `tests/integration/test_audit.py::test_a_clean_run_passes_and_lists_its_decisions` |
| Rolling-origin validation and a feature-availability contract for series data | Implemented | `mml/data/splitting/splitters.py::RollingSplit`, `mml/contracts/tasks.py::ProjectSpec`, `tests/unit/test_splitting.py::test_rolling_windows_slide_forward_and_always_validate_on_the_future` |
| Model and family exclusion | Implemented | `mml/intelligence/planner.py::ExcludedModelsRule`, `tests/unit/test_planner.py::test_excluded_models_and_families_are_not_planned_and_are_explained` |
| Predict from a saved bundle without the training project | Implemented | `mml/api/predict.py::predict`, `tests/integration/test_predict_api.py::test_a_saved_bundle_predicts_without_the_project` |
| Datetime columns become calendar and trend features computed inside the pipeline, so inference accepts raw timestamps | Implemented | `mml/data/profiling/profiler.py::DatasetProfiler`, `tests/unit/test_profiling.py::test_datetime_columns_become_calendar_and_trend_features`, `tests/integration/test_end_to_end.py::test_datetime_features_train_and_predict_from_raw_timestamps` |
| Free-text columns are encoded as fold-local TF-IDF terms | Implemented | `mml/data/validation/columns.py::TextKind`, `tests/integration/test_end_to_end.py::test_free_text_is_encoded_and_scored_from_raw_strings` |
| Missing-value strategies (indicators, iterative imputation) and an arcsinh target transform are tested like any other strategy | Implemented | `mml/sampling/strategies.py::MissingIndicatorStrategy`, `mml/sampling/strategies.py::TargetTransformStrategy`, `tests/unit/test_strategies.py::test_missing_indicators_append_a_flag_for_features_with_informative_gaps`, `tests/unit/test_strategies.py::test_target_transform_helps_a_linear_model_on_a_long_tailed_target` |
| User-defined metrics (`mml.register_metric`) are optimised, selected and reported | Implemented | `mml/contracts/metrics.py::FunctionMetric`, `tests/integration/test_custom_metric.py::test_a_custom_cost_metric_is_optimised_and_reported` |
| Greedy weighted blending is searched alongside stacking | Implemented | `mml/models/ensembles.py::BlendingAdapter`, `tests/unit/test_blending.py::test_the_blender_puts_its_weight_on_the_member_that_predicts_best` |
| Fairness report across named sensitive columns | Implemented | `mml/evaluation/fairness.py::FairnessReport`, `tests/unit/test_fairness.py::test_a_rule_that_favours_one_group_is_flagged_with_its_disparities` |
| Bundles flag rows unlike the training data | Implemented | `mml/inference/novelty.py::NoveltyDetector`, `tests/unit/test_novelty.py::test_a_single_wild_value_is_flagged_and_the_feature_is_named` |
| A learning curve says whether more data would help | Implemented | `mml/evaluation/learning_curve.py::compute`, `tests/unit/test_learning_curve.py::test_judging_curves_directly` |
| Monotonic constraints are enforced by the models that support them and others are not planned | Implemented | `mml/models/wrappers.py::MonotoneEstimator`, `mml/intelligence/planner.py::MonotonicRule`, `tests/integration/test_monotonic.py::test_constrained_predictions_never_fall_as_the_feature_rises` |
| Post-selection diagnostics are bounded by a share of the time budget | Implemented | `mml/explainability/importance.py::PermutationImportance`, `tests/unit/test_explainability.py::test_importance_stops_at_its_deadline_and_says_it_was_partial` |
| Per-row training weights (`weight_column`), also used to score validation folds and the test metrics | Implemented | `mml/models/weighting.py::WeightedEstimator`, `tests/integration/test_weights.py::test_validation_scores_are_weighted_and_say_so` |
| Survival analysis for right-censored data: Cox and boosted-hazard models, concordance, integrated Brier score and dynamic AUC, event-stratified splits (competing risks and time-varying covariates are not built) | Implemented | `mml/models/survival/estimators.py::CoxRegressor`, `mml/evaluation/survival.py::integrated_brier`, `tests/integration/test_survival.py::test_time_dependent_metrics_are_reported_on_the_locked_rows`, `tests/unit/test_survival_split.py::test_the_event_rate_is_steady_across_test_and_folds` |
| Ordinal targets with thresholded regressors, quadratic weighted kappa and mean absolute class error | Implemented | `mml/models/ordinal/estimators.py::CumulativeThresholdRegressor`, `tests/integration/test_ordinal_task.py::test_predictions_are_the_users_labels_in_the_stated_order` |
| Unlabeled rows: ignore, or self-train only when cross-validation says it helps; rank rows to label next | Implemented | `mml/sampling/pseudo_labeling.py::judge`, `tests/integration/test_unlabeled.py::test_pseudo_labeling_is_judged_and_recorded` |
| Conformalized quantile intervals, multiclass reliability and a coverage curve on the locked rows | Implemented | `mml/evaluation/quantile.py::QuantileIntervals`, `tests/integration/test_calibration_reports.py::test_quantile_coverage_tracks_the_nominal_level_on_locked_rows` |
| Copies of one record kept on one side of every split | Implemented | `mml/data/splitting/duplicates.py::keep_together`, `tests/unit/test_duplicate_groups.py::test_no_duplicate_group_straddles_the_holdout_or_any_fold` |
| Informative-missingness report; missing-value flags only where gaps carry signal | Implemented | `mml/data/profiling/missingness.py::assess`, `tests/integration/test_missingness_plan.py::test_random_gaps_do_not_earn_missing_value_flags` |
| Production for any model: a service folder with Dockerfile and pinned environment; ONNX for the models the converter can express | Implemented | `mml/inference/packaging.py::write_service`, `mml/inference/onnx_export.py::export`, `tests/integration/test_service_package.py::test_the_generated_server_scores_rows_like_the_bundle`, `tests/unit/test_onnx_export.py::test_a_regressor_matches_on_scaled_values` |
| Register any scikit-learn-compatible estimator in one call | Implemented | `mml/models/custom.py::register_estimator`, `tests/integration/test_register_estimator.py::test_a_registered_classifier_competes_in_a_run` |
| Counterfactual explanations that change only actionable columns | Implemented | `mml/explainability/counterfactual.py::CounterfactualSearch`, `tests/unit/test_counterfactual.py::test_columns_that_are_not_actionable_never_change` |
| Debounced, audited retraining loop | Implemented | `mml/governance/lifecycle.py::RetrainPolicy`, `tests/integration/test_deployment.py::test_repeated_checks_of_the_same_drift_retrain_exactly_once` |
| Build-environment lock attached to compatibility errors | Implemented | `mml/artifacts/bundle.py::installed_distributions`, `tests/unit/test_bundle.py::test_a_version_mismatch_arrives_with_the_lock_attached` |
| Fairness mitigation by group thresholds, judged on validation and reported on the locked rows (reweighing and constrained fitting are not built) | Implemented | `mml/evaluation/mitigation.py::judge`, `tests/integration/test_fairness_mitigation.py::test_the_decision_is_logged_with_its_evidence` |
| Repeated-seed benchmarks with confidence intervals | Implemented | `benchmarks/intervals.py::mean_interval`, `tests/benchmarks/test_intervals.py::test_verdicts_come_from_the_interval` |
| Learning to rank with query-disjoint splits and NDCG / MAP | Implemented | `mml/models/ranking/estimators.py::LibraryRanker`, `mml/evaluation/ranking.py::mean_ndcg`, `tests/integration/test_ranking_task.py::test_no_query_appears_in_both_development_and_locked_rows`, `tests/unit/test_ranking.py::test_ndcg_matches_scikit_learn_on_one_query` |
| Pretrained sentence embeddings for text columns, computed in the worker with a TF-IDF fallback | Implemented | `mml/data/curation/embedding.py::TextEmbedder`, `mml/models/deeplearning/embeddings.py::Embedder`, `tests/unit/test_embeddings.py::test_embeddings_generalise_to_wording_tfidf_never_saw`, `tests/unit/test_embeddings.py::test_the_framework_process_never_imports_torch` |
| Multi-label and multi-output targets with iterative stratification and per-target or joint models | Implemented | `mml/data/splitting/iterative.py::iterative_assignment`, `mml/models/multitarget/wrapper.py::MultiTargetEstimator`, `tests/integration/test_multitarget_task.py::test_folds_keep_every_label_when_the_data_allow`, `tests/unit/test_iterative.py::test_every_subset_gets_its_share_of_every_label` |
| Time-series forecasting with a horizon embargo, baselines and a per-step backtest | Implemented | `mml/data/curation/forecasting.py::add_targets`, `mml/evaluation/backtest.py::report`, `mml/artifacts/bundle.py::ModelBundle`, `tests/integration/test_forecast_task.py::test_the_model_beats_no_change_and_the_seasonal_baseline_at_every_step`, `tests/integration/test_forecast_task.py::test_training_targets_never_reach_into_the_locked_period` |
| Uplift / treatment-effect modelling with Qini scoring and a randomization check | Implemented | `mml/models/uplift/estimators.py::TwoModelLearner`, `mml/evaluation/uplift.py::check_propensity`, `tests/integration/test_uplift_task.py::test_estimated_uplift_tracks_the_true_effect_in_a_randomized_trial`, `tests/integration/test_uplift_task.py::test_confounded_assignment_is_flagged` |
| Datasets larger than memory: hashed locked rows, a bounded development sample, streamed scoring of every locked row, incremental training | Implemented | `mml/data/readers/streaming.py::sample_pass`, `mml/evaluation/replay.py::PredictionReplay`, `mml/intelligence/planner.py::IncrementalRule`, `tests/unit/test_streaming.py::test_a_file_far_larger_than_the_budget_is_read_in_bounded_memory`, `tests/integration/test_streamed_run.py::test_every_locked_row_is_scored_by_streaming_the_file` |
| Trials on a multi-node Ray cluster with single-node-identical results, enforced limits and lost-worker handling (GPU scheduling is not exercised) | Implemented | `mml/integrations/ray_backend.py::RayBackend`, `mml/experimentation/trials.py::lost_trial`, `tests/integration/test_ray_cluster.py::test_a_multi_node_run_reproduces_the_single_node_trials`, `tests/integration/test_ray_cluster.py::test_a_lost_worker_becomes_a_failed_item_not_a_dead_run` |
| Problem statement with success criteria checked against the locked rows | Implemented | `mml/evaluation/criteria.py::check`, `tests/integration/test_problem_spec.py::test_missed_criteria_are_named_with_the_gap`, `tests/unit/test_criteria.py::test_a_floor_for_higher_is_better_and_a_ceiling_for_lower_is_better` |
| Nested cross-validation for the optimism of the tuning procedure, MRR and precision/recall@k, equal-class warning | Implemented | `mml/evaluation/nested.py::nested_score`, `mml/evaluation/ranking.py::mean_reciprocal_rank`, `tests/unit/test_nested.py::test_the_honest_estimate_on_pure_noise_is_near_chance_and_optimism_shows`, `tests/integration/test_evaluation_extras.py::test_nested_validation_reports_the_optimism_of_the_winning_score` |
| Bias–variance decomposition with an irreducible-noise estimate | Implemented | `mml/evaluation/bias_variance.py::decompose`, `tests/unit/test_bias_variance.py::test_a_straight_line_on_a_curve_is_bias_dominated`, `tests/integration/test_evaluation_extras.py::test_the_bias_variance_report_names_the_dominant_error` |
| Optimizer, loss and gradient-accumulation options for the PyTorch networks | Implemented | `mml/models/deeplearning/engines.py::TabularEngine`, `tests/unit/test_optimizers.py::test_every_first_order_optimizer_learns`, `tests/unit/test_optimizers.py::test_huber_loss_resists_outliers_that_pull_the_squared_loss` |
| Grid search, Hyperband and multi-objective (score, latency, size) selection | Implemented | `mml/optimization/optimizers.py::GridOptimizer`, `mml/experimentation/tuning.py::Tuner`, `mml/core/stages.py::pareto_report`, `tests/unit/test_tuning_breadth.py::test_hyperband_brackets_start_cheap_and_always_end_with_the_full_cost`, `tests/integration/test_tuning_and_pareto.py::test_the_pareto_front_is_reported_and_the_fast_model_wins_a_tie` |
| Shadow evaluation, A/B analysis and rollback criteria for promoted models | Implemented | `mml/monitoring/shadow.py::ShadowEvaluator`, `mml/monitoring/abtest.py::compare_arms`, `mml/governance/lifecycle.py::LifecycleManager`, `tests/unit/test_lifecycle_extras.py::test_a_clearly_worse_new_champion_is_rolled_back` |
| Label-noise detection and sample-vs-reference representativeness (run in the profile stage; findings appear in the quality findings and report) | Implemented | `mml/data/profiling/label_noise.py::find_label_noise`, `tests/integration/test_quality_checks_in_run.py::test_noisy_labels_and_a_shifted_reference_are_reported`, `mml/data/profiling/representativeness.py::compare_to_reference`, `tests/unit/test_data_quality_checks.py::test_flipped_labels_are_found_first` |
| Training, evaluation and performance sensitivity of a model's score (library function) | Implemented | `mml/evaluation/sensitivity.py::analyze`, `tests/unit/test_sensitivity.py::test_the_report_names_the_dominant_source_and_serializes` |
| Dirichlet calibration of multiclass probabilities | Implemented | `mml/evaluation/dirichlet.py::DirichletCalibrator`, `tests/unit/test_reliability.py::test_dirichlet_calibration_lowers_error_and_survives_a_round_trip` |
| Training-time fairness by reweighing (weights supplied through `weight_column`) | Implemented | `mml/evaluation/reweighing.py::reweighing_weights`, `tests/integration/test_reweighing.py::test_a_project_trained_on_the_weights_selects_more_evenly` |
| Clustering with internal validity indices, stability and external scores (library functions, not a pipeline task) | Implemented | `mml/unsupervised/clustering.py::find_clusters`, `tests/unit/test_clustering.py::test_three_separated_groups_are_found_and_judged_stable` |
| Competing risks (cause-specific Cox, Aalen-Johansen incidence) | Implemented | `mml/models/survival/competing.py::CauseSpecificCox`, `tests/unit/test_competing_timevarying.py::test_cause_specific_cox_recovers_the_effect_and_orders_risk` |
| Time-varying covariates in a Cox model (counting-process rows) | Implemented | `mml/models/survival/timevarying.py::TimeVaryingCox`, `tests/unit/test_competing_timevarying.py::test_time_varying_cox_recovers_the_effect_of_a_switching_covariate` |
| TabPFN foundation model as an optional worker-backed adapter (license acceptance required; exercised only against a stand-in package, not the real weights) | Extension point | `mml/models/deeplearning/foundation.py::TabPfnEngine`, `tests/unit/test_foundation.py::test_the_proxy_trains_and_predicts_through_the_worker` |
| Benchmarks draw on Hugging Face and OpenML tables as well as the core nine | Implemented | `benchmarks/datasets.py::HuggingFaceDataset`, `tests/benchmarks/test_datasets.py::test_hugging_face_tables_point_at_files_in_known_repositories` |

Known limits of the isolation: the vault protects the locked rows from code
in trial processes and plugins that run in them. It cannot stop a process
from reading the original data file if the caller passed a path, and a
caller's own in-memory DataFrame still contains every row.

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
