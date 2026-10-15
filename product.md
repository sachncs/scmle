# Technical Architecture Specification

Autonomous ML Lifecycle Framework · Architecture v0.1 · October 2026

Product: Open-source, Python-first machine learning lifecycle framework

Working name: `mlforge` (placeholder; package-name availability not verified)

Initial focus: Tabular classification and regression

Long-term scope: Intelligent data preparation, statistical analysis, feature engineering, model discovery, experimentation, optimization, evaluation, explainability, inference, deployment, monitoring, and lifecycle management.

## 1. Architecture decisions

The framework should be built around one central principle:

The user specifies the ML objective; the framework determines, executes, validates, and documents an appropriate workflow.

The system must be autonomous where decisions are technically defensible, conservative where the data is ambiguous, and transparent about every transformation and experiment.

### Non-negotiable design decisions

| Decision                | Specification                                                   |
| ----------------------- | --------------------------------------------------------------- |
| Language                | Python 3.11+ initially                                          |
| Distribution            | Open-source PyPI package                                        |
| Architecture            | Modular monolith with plugin contracts                          |
| Execution               | Local-first, distributed-capable                                |
| Primary API             | Declarative Python                                              |
| Configuration           | Pydantic-based typed configuration                              |
| Data representation     | Arrow-compatible tabular abstraction                            |
| Model interface         | Scikit-learn-compatible estimator protocol                      |
| Workflow representation | Directed acyclic graph (DAG)                                    |
| Experiment optimization | Multi-fidelity, budget-aware                                    |
| Artifact management     | Content-addressed, versioned                                    |
| Observability           | Event-driven telemetry and monitoring adapters                  |
| Reproducibility         | Versioned configuration, data fingerprints, seeds, environments |
| Deployment              | Optional adapter; not a mandatory dependency                    |
| Open-source license     | Apache-2.0 candidate, subject to dependency review              |

We should avoid mandatory Kubernetes, cloud services, databases, GPUs, or hosted experiment trackers for the base installation.

A user should be able to install the framework on a laptop and train a model without setting up infrastructure.

## 2. System architecture

PUBLIC API

Python SDK · CLI · Project Configuration · Events

ORCHESTRATION KERNEL

Workflow Compiler · State Machine · Execution Scheduler

Policy Engine · Resource Budget Manager · Recovery Controller

DECISION INTELLIGENCE

Dataset Fingerprint → Candidate Plans → Evaluation → Adaptive Search

Data intelligence

Schema, quality, leakage

Data preparation

Cleaning, encoding, imputation

Feature engineering

Generation, selection, transformations

Sampling / augmentation

Weights, resampling, synthesis

Model search

Families, estimators, architectures

Optimization

Hyperparameters, early stopping

Evaluation

Metrics, robustness, calibration

Lifecycle

Artifacts, inference, monitoring

EXECUTION ADAPTERS

Local · Multiprocessing · Ray (optional)

SHARED PERSISTENCE

Datasets · Trials · Events · Reports · Pipelines · Models · Lineage

Three architectural layers should remain separate:

- Control plane: Decides what to do, when, and why.
- Execution plane: Performs transformations, training, tuning, and evaluation.
- Evidence plane: Stores artifacts, measurements, decisions, logs, and provenance.

This separation is essential for future distributed execution and advanced decision intelligence.

## 3. Package organization

I recommend a `src` layout with a small core package and optional integration packages.

```

mlforge/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── mlforge/
│       ├── __init__.py
│       ├── api/
│       │   ├── project.py
│       │   ├── results.py
│       │   └── predict.py
│       ├── contracts/
│       │   ├── data.py
│       │   ├── tasks.py
│       │   ├── models.py
│       │   ├── metrics.py
│       │   ├── artifacts.py
│       │   └── events.py
│       ├── core/
│       │   ├── config.py
│       │   ├── orchestrator.py
│       │   ├── compiler.py
│       │   ├── scheduler.py
│       │   ├── state.py
│       │   └── policies.py
│       ├── intelligence/
│       │   ├── fingerprint.py
│       │   ├── planner.py
│       │   ├── recommendations.py
│       │   └── knowledge.py
│       ├── data/
│       │   ├── readers/
│       │   ├── profiling/
│       │   ├── validation/
│       │   ├── curation/
│       │   └── splitting/
│       ├── features/
│       ├── sampling/
│       ├── models/
│       │   ├── registry.py
│       │   ├── sklearn/
│       │   └── boosting/
│       ├── optimization/
│       ├── experimentation/
│       ├── evaluation/
│       ├── explainability/
│       ├── artifacts/
│       ├── inference/
│       ├── monitoring/
│       ├── governance/
│       └── integrations/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── leakage/
│   ├── property/
│   └── benchmarks/
├── examples/
├── docs/
└── benchmarks/

```

A particular module must never reach directly into another module's internal implementation. Interactions should occur through stable protocols and typed data contracts.

This matters because eventually we may replace Pandas with Polars, switch optimization engines, or introduce new estimator libraries without rewriting the orchestrator.

# Part II — Formal module contracts

## 4. The core domain model

Every ML project should have an explicit, versioned specification.

```
from enum import Enumfrom typing import Literalfrom pydantic import BaseModel, Fieldclass TaskType(str, Enum):    BINARY = "binary_classification"    MULTICLASS = "multiclass_classification"    REGRESSION = "regression"class DataSpec(BaseModel):    uri: str    target: str    id_columns: list[str] = Field(default_factory=list)    group_column: str | None = None    time_column: str | None = None    excluded_columns: list[str] = Field(default_factory=list)class ValidationSpec(BaseModel):    strategy: Literal[        "auto", "stratified", "random",        "group", "temporal"    ] = "auto"    test_fraction: float = Field(default=0.20, gt=0, lt=1)    folds: int = Field(default=5, ge=2)    seed: int = 42class ResourceSpec(BaseModel):    time_budget_seconds: int = Field(default=1800, gt=0)    memory_limit_gb: float = Field(default=8.0, gt=0)    cpu_limit: int | None = None    gpu: Literal["auto", "disabled", "required"] = "auto"
```

The implementation needs cross-field validation in addition to Pydantic's individual field constraints. For example, group validation needs a group column; binary-classification metrics require a binary target; and temporal validation needs an unambiguous time definition.

The configuration should be serializable, hashable after canonicalization, and saved with every project run.

## 5. Key internal interfaces

These are the foundational contracts that all engines should implement.

```
from dataclasses import dataclassfrom typing import Any, Mapping, Protocol@dataclass(frozen=True)class DatasetManifest:    dataset_id: str    schema_hash: str    data_hash: str    row_count: int    column_count: int    target: str@dataclass(frozen=True)class DatasetProfile:    manifest: DatasetManifest    column_statistics: Mapping[str, Any]    quality_findings: tuple[str, ...]    risk_findings: tuple[str, ...]@dataclass(frozen=True)class ExperimentPlan:    plan_id: str    task: str    validation_strategy: str    candidate_models: tuple[str, ...]    optimization_budget_seconds: int@dataclass(frozen=True)class TrialResult:    trial_id: str    model_name: str
```

These are architectural interface sketches, not yet production-complete contracts.

The production contracts should additionally define schema types, capability discovery, fit/predict context, cancellation, resource consumption, deterministic configuration hashes, error categories, serialization formats, and compatibility versions.

### Module input/output contract matrix

| Module              | Input                             | Output                        | Key invariant                       |
| ------------------- | --------------------------------- | ----------------------------- | ----------------------------------- |
| Ingestion           | URI + reader settings             | Dataset handle + manifest     | Input remains immutable             |
| Profiling           | Dataset handle                    | Statistical profile           | No fitted model state               |
| Splitting           | Dataset + task + policy           | Immutable split manifest      | No entity/time leakage              |
| Curation            | Training partition                | Fitted transformation plan    | Learn statistics on train only      |
| Feature engineering | Training folds                    | Fitted feature graph          | No cross-fold leakage               |
| Sampling            | Training fold                     | Rebalanced training fold      | Validation distributions untouched  |
| Model registry      | Task + capability constraints     | Compatible model candidates   | Unsupported models excluded         |
| Optimizer           | Candidate space + validation plan | Trial results                 | Enforced budget and trial isolation |
| Evaluation          | Predictions + labels              | Metrics + diagnostics         | Fixed scoring protocol              |
| Selector            | Valid trial results               | Selected candidate            | Defined selection policy            |
| Artifact builder    | Fitted pipeline + metadata        | Loadable bundle               | Training/serving parity             |
| Inference           | Bundle + new dataset              | Predictions + diagnostics     | Schema checked                      |
| Monitoring          | Inference and outcome events      | Drift/performance assessments | Missing labels explicitly handled   |

The interface contracts should be enforced through automated conformance tests so third-party plugin authors cannot accidentally violate the framework's expectations.

# Part III — Execution and decision lifecycle

## 6. Complete execution DAG

The framework should compile the user request into an executable directed acyclic graph.

1\. Ingest and validate dataset

2\. Inspect schema and dataset characteristics

3\. Establish immutable split manifest

Train / cross-validation / locked test

Training branch

Fit transformations and candidate models within training folds

Test branch

Retain unchanged and unavailable to search until final evaluation

4\. Baselines → Candidate selection → Tuning

Training partitions and validation folds only

5\. Select configuration and refit on permitted development data

6\. One-time final test evaluation

Evaluate the frozen candidate without further tuning

7\. Artifact verification and registration

8\. Inference and monitoring

Optional deployment, drift checks, candidate retraining

### Runtime state machine

Every workflow should have an explicit state.

`CREATED → VALIDATING → PROFILING → PLANNING → TRAINING → SELECTING → TESTING → PACKAGING → COMPLETED`

Terminal or suspended states include `FAILED`, `CANCELLED`, and `BLOCKED`.

`BLOCKED` is important: sometimes the framework cannot safely continue automatically. Examples include an ambiguous target, invalid temporal ordering, insufficient minority-class examples, or an unresolvable leakage risk.

Each state transition should create an event containing the run ID, timestamp, configuration version, reason, and artifact references.

The orchestrator must support:

- Idempotent retries where technically safe.
- Cancellation and budget exhaustion.
- Checkpointing completed trials.
- Restarting from saved valid artifacts.
- Consistent failure reporting.
- Deterministic dependency ordering.
- Parallel execution of independent trials.
- Resource isolation between competing trials.

## 7. Dataset fingerprinting

The fingerprint is a major input to decision intelligence.

It should include more than basic descriptive statistics.

| Dimension      | Measurements                                                     |
| -------------- | ---------------------------------------------------------------- |
| Shape          | Rows, columns, effective samples, sparsity                       |
| Types          | Numeric, categorical, datetime, text, IDs                        |
| Missingness    | Rates, patterns, feature associations                            |
| Target         | Distribution, imbalance, cardinality, skew                       |
| Features       | Cardinality, entropy, scale, outliers                            |
| Relationships  | Correlations, redundancy, feature–target associations            |
| Data structure | Groups, temporal sequences, repeated entities                    |
| Quality        | Duplicates, invalid values, inconsistent labels                  |
| ML difficulty  | Baseline performance, potential nonlinearity, class separability |
| Operational    | Memory footprint, read speed, feature availability               |
| Risk           | Leakage indicators, extreme rarity, potential distribution shift |

The engine should preserve uncertainty. For example, correlation with the target may indicate a useful predictor or a source of leakage; statistics alone generally cannot establish which.

### Decision-policy example

A candidate policy might be:

```
if profile.is_classification:    include("dummy_baseline")    include("linear_model")    include("tree_ensemble")if profile.has_high_cardinality_categories:    consider("native_categorical_boosting")    consider("regularized_target_encoding")if profile.has_severe_imbalance:    evaluate("class_weighting")    evaluate("resampling_inside_cv")if resources.time_budget_seconds < 300:    prioritize("fast_models")    reduce_search_complexity()
```

This is intentionally candidate generation, not an assertion that any technique will improve performance.

A good planner first decides what experiments are worth trying; the evaluation layer decides whether they actually help.

## 8. Adaptive model selection and tuning

I recommend the following strategy for the first implementation.

Stage A — Establish baselines

Dummy, regularized linear, tree-based baseline

Stage B — Screen model families

Low-cost trials, shared folds, compatible metrics

Stage C — Optimize promising candidates

Conditional hyperparameter spaces, pruning, early stopping

Stage D — Evaluate optional ensembles

Out-of-fold predictions, latency constraints, no leakage

Stage E — Select and package

Metric, uncertainty, model size, inference cost, reliability

### Model family registry — initial implementation

| Model family       | Initial candidates                     |
| ------------------ | -------------------------------------- |
| Baselines          | DummyClassifier, DummyRegressor        |
| Linear             | LogisticRegression, Ridge, ElasticNet  |
| Trees              | DecisionTree, RandomForest, ExtraTrees |
| Histogram boosting | HistGradientBoosting                   |
| Gradient boosting  | XGBoost, LightGBM, CatBoost            |
| Other              | SVM and kNN, when dataset size permits |

A model adapter must declare supported tasks, sparse-input support, probability outputs, native categorical handling, missing-value capabilities, early stopping, memory characteristics, and serialization support.

The planner uses those capabilities instead of hard-coding library-specific conditions.

Existing systems provide useful integration points: AutoGluon supports comprehensive tabular fitting, while FLAML exposes time-budget-aware automated search.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://auto.gluon.ai&sz=32)

AutoGluon 1.5.0 documentation

+1

I would use Optuna as an optional tuning backend and build our own lightweight orchestration interface around it. This prevents the optimizer library from becoming the framework's architecture.

## 9. Dataset balancing and augmentation

This subsystem deserves independent design because inappropriate balancing can make a model worse or produce misleading evaluations.

The proposed search space:

| Technique                    | Default policy                                    |
| ---------------------------- | ------------------------------------------------- |
| No rebalancing               | Always retain as baseline                         |
| Class/sample weighting       | Evaluate for compatible models                    |
| Random undersampling         | Conditional candidate                             |
| Random oversampling          | Conditional candidate                             |
| SMOTE variants               | Optional for compatible tabular datasets          |
| Synthetic tabular generation | Research/advanced plugin initially                |
| Calibration                  | Evaluate separately when probabilities matter     |
| Threshold optimization       | Use validation predictions and explicit objective |

Sampling occurs inside each training fold, after splitting, never on the full dataset before cross-validation.

Augmentation should require evidence that it improves evaluation under the original, unsampled validation distribution.

For regression, rare-target handling requires a different policy; classification-specific resampling should not be reused indiscriminately.

# Part IV — Evaluation, reproducibility, and inference

## 10. Evaluation contract

The framework should produce standardized performance results and diagnostics.

| Task                  | Primary candidate metrics             | Secondary diagnostics                             |
| --------------------- | ------------------------------------- | ------------------------------------------------- |
| Binary classification | ROC-AUC, PR-AUC, log loss, F1         | Calibration, confusion matrix, threshold analysis |
| Multiclass            | Log loss, macro-F1, balanced accuracy | Per-class performance, confusion matrix           |
| Regression            | MAE, RMSE, R²                         | Residuals, bias, error by segment                 |

The user should be able to select a primary objective. The framework can propose one when not supplied, but it should not assume that accuracy, F1, or ROC-AUC always represents the user's real operational costs.

Statistical confidence should reflect independent units of observation. Cross-validation fold scores alone are not necessarily independent observations, and uncertainty estimates need to respect group or temporal structure.

The model selector should also support constraints:

```
selection = {    "maximize": "roc_auc",    "constraints": {        "p95_inference_ms": 50,        "model_size_mb": 100    },    "tie_breaker": "simpler_model"}
```

These constraints must be evaluated on representative hardware before a candidate can be declared compliant.

## 11. Data leakage prevention

Leakage prevention must be a framework-level invariant, not an optional afterthought.

The test dataset should be hidden behind a restricted handle. Trial code should have no access to it, and feature transformations should be fitted independently inside training folds.

The same principle is explicitly recommended in Scikit-learn's data-leakage guidance.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://scikit-learn.org&sz=32)

scikit-learn 1.9.1 documentation

I recommend five structural protections:

1. A split manifest containing immutable row identifiers.
2. Separate `fit` and `transform` permissions for dataset partitions.
3. A pipeline graph recording which data each fitted operation accessed.
4. A locked final test evaluator callable only after model selection.
5. Audit events for all changes to splits, exclusions, and feature definitions.

For temporal or entity-level data, splitting should follow the actual deployment scenario. A simple random split must not be used merely because it produces better scores.

## 12. Artifact and reproducibility contract

A completed run should produce the following directory or equivalent object-store structure:

```

runs/
└── run_20261015_001/
    ├── config.json
    ├── environment.lock
    ├── dataset_manifest.json
    ├── dataset_profile.json
    ├── validation_manifest.json
    ├── execution_plan.json
    ├── events.jsonl
    ├── trials/
    │   ├── trial_001.json
    │   └── trial_002.json
    ├── leaderboard.parquet
    ├── evaluation.json
    ├── diagnostics/
    ├── model/
    │   ├── pipeline.artifact
    │   ├── model_manifest.json
    │   └── schema.json
    └── report.html

```

The artifact format must specify serialization and loading behavior; `.artifact` is a logical placeholder, not a file format.

For Python-based estimator serialization, serialized model files should be treated as trusted executable artifacts, because formats such as Pickle and Joblib can execute code during loading. Untrusted artifacts should not be loaded.

Model bundles also need dependency compatibility checks and verifiable checksums.

Perfect bit-for-bit reproducibility cannot be guaranteed for every GPU, parallel backend, or external library. The framework should record reproducibility conditions and distinguish exact replay from statistically comparable reruns.

## 13. Inference contract

Prediction should be performed through the entire fitted pipeline, not the estimator alone.

```
model = result.load_best_model()predictions = model.predict(    "new_customers.parquet")probabilities = model.predict_proba(    "new_customers.parquet")
```

The inference subsystem should provide schema validation, column mapping according to explicit policy, unknown-category handling, missing-feature diagnostics, batch execution, optional probability calibration, and structured errors.

An example prediction result could contain:

```
{    "prediction": 1,    "probability": 0.87,    "model_version": "v1",    "schema_valid": True}
```

This is an illustrative response shape, not a real model prediction.

The first release should support local and batch inference. API serving can follow in the production lifecycle phase.

# Part V — Implementation strategy

## 14. Recommended technology stack

| Layer                    | Primary choice                   | Rationale                                           |
| ------------------------ | -------------------------------- | --------------------------------------------------- |
| Package management       | `uv`, `pyproject.toml`           | Fast, standards-based development workflows         |
| Typed configuration      | Pydantic                         | Validated, serializable contracts                   |
| Tabular interoperability | PyArrow                          | Columnar format and cross-engine data interchange   |
| Data processing          | Pandas initially; Polars adapter | Broad ML compatibility with expansion path          |
| Statistical modeling     | Scikit-learn                     | Familiar, stable estimator and pipeline conventions |
| Boosting                 | LightGBM, XGBoost, CatBoost      | Competitive tabular model families                  |
| Sampling                 | Imbalanced-learn                 | Established resampling implementations              |
| Tuning                   | Optuna adapter                   | Search-space flexibility and pruning                |
| Tracking                 | SQLite + local artifact store    | Zero-service local execution                        |
| External tracking        | MLflow adapter                   | Existing experiment and registry ecosystem          |
| Parallel execution       | Joblib/process backend           | Simple first-release execution                      |
| Distributed execution    | Ray adapter later                | Optional scale-out                                  |
| Reports                  | HTML + JSON                      | Human- and machine-readable results                 |
| Monitoring               | Metrics/events abstraction       | No forced dependency on a hosted platform           |
| Testing                  | Pytest, Hypothesis               | Unit, property-based, integration testing           |
| Code quality             | Ruff, mypy or Pyright            | Linting and type analysis                           |
| CI/CD                    | GitHub Actions                   | Repeatable public builds and release checks         |

MLflow already supports model versioning, lineage, aliases, and registry workflows; using it as an optional backend avoids rebuilding a mature external system.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://mlflow.org&sz=32)

MLflow AI Platform

### Dependency strategy

Keep the base installation lightweight:

```
pip install mlforge
```

Optional capabilities should be exposed through extras, for example:

```
pip install "mlforge[boosting]"
pip install "mlforge[optimization]"
pip install "mlforge[tracking]"
pip install "mlforge[distributed]"
pip install "mlforge[all]"
```

These commands describe the intended future packaging interface; the proposed package has not yet been published.

A critical rule is that installed integrations must be discoverable at runtime. A model registry entry should not cause imports of GPU or deep-learning dependencies unless that model is actually selected.

## 15. Development roadmap

The most efficient approach is to deliver small, independently useful releases with increasingly sophisticated intelligence.

## M0

Weeks 1–2

Foundation and contracts

Repository structure, packaging, configuration, domain models, data loaders, task validation, error taxonomy, event contracts, baseline test infrastructure.

Deliverable: Installable library with typed project configuration and reliable CSV/Parquet ingestion.

## M1

Weeks 3–5

Working end-to-end ML pipeline

Dataset profiling, validation, splitting, preprocessing pipelines, baseline models, cross-validation, metrics, artifact saving, batch inference.

Deliverable: `MLProject(...).run()` trains a defensible classification or regression pipeline and saves an inference-ready artifact.

## M2

Weeks 6–8

Automated search and experimentation

Model registry, estimator adapters, hyperparameter spaces, optimization, pruning, experiment persistence, leaderboard, resume logic.

Deliverable: Multi-model training and budget-aware optimization with reproducible comparisons.

## M3

Weeks 9–11

Data and feature intelligence

Intelligent preprocessing candidate selection, imbalance handling, feature engineering experiments, diagnostics, decision logs, basic explainability.

Deliverable: Framework evaluates alternative data strategies rather than applying a fixed preprocessing recipe.

## M4

Weeks 12–14

Release hardening and benchmark validation

Leakage tests, edge cases, cancellation, artifacts, tutorials, benchmark suite, security review, dependency policy, package publishing.

Deliverable: Public beta suitable for experimental and non-critical production evaluation.

These are planning estimates for a small experienced team, not confirmed delivery dates. A single developer or a team unfamiliar with the chosen libraries could require substantially longer.

The first production-oriented release would follow, adding deployment adapters, model monitoring, drift analysis, champion–challenger management, and lifecycle automation.

## 16. Implementation backlog, in priority order

| Priority | Feature                | Acceptance criterion                                              |
| -------- | ---------------------- | ----------------------------------------------------------------- |
| P0       | Project API            | Validated project spec and typed run result                       |
| P0       | Data ingestion         | CSV and Parquet with schema and error reporting                   |
| P0       | Validation strategy    | Correct stratified, random, group, and temporal protocols         |
| P0       | Leakage-safe pipelines | Fold-local transformations and sampling                           |
| P0       | Baseline models        | Classification and regression end to end                          |
| P0       | Metrics                | Correct, task-specific evaluation                                 |
| P0       | Artifact persistence   | Save/load round-trip prediction equivalence                       |
| P1       | Model registry         | Third-party estimators registered without core edits              |
| P1       | Hyperparameter tuning  | Budget-aware trials with failure isolation                        |
| P1       | Experiment tracking    | Run, trial, configuration, and artifact lineage                   |
| P1       | Automated decisions    | Plans explain applicable and rejected techniques                  |
| P1       | Imbalance handling     | Candidate strategies compared on original validation distribution |
| P1       | Explainability         | Basic feature importance and error diagnostics                    |
| P2       | Ensembles              | Leakage-safe out-of-fold stacking                                 |
| P2       | Advanced augmentation  | Explicit experimental policy                                      |
| P2       | Meta-learning          | Historical runs influence search ordering                         |
| P2       | Deployment             | Serving adapters and model promotion                              |
| P2       | Monitoring             | Drift, delayed-label metrics, retraining evaluation               |

P0 means functionality required for the first credible end-to-end release.

## 17. Testing and engineering quality gates

The testing strategy matters as much as the model-search code.

### Minimum required test categories

Contract tests: Verify that every estimator adapter, sampler, profiler, and tracker implements its declared API.

Statistical tests: Validate splitting, class distributions, metric orientation, target-label consistency, and feature transformations.

Leakage tests: Construct adversarial datasets where a deliberately leaked feature or improper preprocessing could produce inflated performance.

Property-based tests: Exercise randomized shapes, missingness patterns, category cardinalities, duplicate rows, and edge cases.

Artifact tests: Compare predictions before and after serialization, verify schema checks, and enforce artifact compatibility.

Recovery tests: Interrupt trials, simulate exhausted compute budgets, and confirm successful recovery of completed artifacts.

Benchmark tests: Evaluate representative classification and regression datasets under fixed compute budgets.

### Initial quality gates

Every transformation learns fitted state only from permitted training data

Every completed run contains a machine-readable manifest and decision log

Every saved model loads and reproduces predictions within declared tolerances

Every experiment respects configured resource limits or reports enforcement limitations

Every failed or interrupted run reports its terminal state and recoverable artifacts

Every public estimator plugin passes the shared contract test suite

Every release passes a fixed regression benchmark suite

No arbitrary performance-improvement percentage should be promised before a benchmark baseline exists.

Instead, early releases should prioritize correctness and reliable execution. Later optimization releases can target demonstrable gains in the accuracy–compute trade-off.

# Part VI — First executable vertical slice

## 18. Exact scope of the first implementation

I recommend that we begin with a deliberately constrained but fully working pipeline.

Version 0.1 — Minimum viable ML engine

CSV/Parquet data loading

IN SCOPE

Binary and multiclass classification

IN SCOPE

Regression

IN SCOPE

Data profiling and diagnostics

IN SCOPE

Leakage-safe train/validation/test splitting

IN SCOPE

Automatic numeric/categorical preprocessing

IN SCOPE

Baseline and tree/linear model comparison

IN SCOPE

Cross-validation and model selection

IN SCOPE

Local experiment persistence

IN SCOPE

Saved end-to-end inference pipeline

IN SCOPE

Advanced augmentation

LATER

Distributed execution

LATER

Production serving and monitoring

LATER

Neural architecture search

LATER

### First sprint deliverables

The first sprint should produce five concrete outcomes.

1. A published architecture specification containing the module responsibilities, invariants, and contracts defined above.
2. An installable package skeleton with working build configuration and automated tests.
3. A data ingestion and profiling subsystem that returns a validated dataset manifest and statistical profile.
4. A leakage-aware split manager implementing stratified, random, group, and basic temporal strategies.
5. A minimal end-to-end training path using simple Scikit-learn estimators, reusable preprocessing, and saved artifacts.

That vertical slice establishes the central abstraction before introducing sophisticated optimization.

## 19. The API I would freeze first

The core API should remain stable even as the internal intelligence changes.

```
from mlforge import MLProjectproject = MLProject(    data="customers.parquet",    target="churn",    task="classification",    time_budget=1800,    metric="roc_auc")# Analyze data and construct the workflow.plan = project.plan()# Inspect recommendations and warnings.print(plan.summary())# Execute the workflow.result = project.run()# Review experiments.print(result.leaderboard())# Persist a loadable pipeline.result.save("artifacts/churn_model")# Run inference.predictions = result.predict("new_customers.parquet")
```

For compatibility, `task="classification"` should be a public convenience alias. Internally, the target validator resolves and records whether the task is binary or multiclass.

The `plan()` method is particularly valuable: users can examine the proposed pipeline before any expensive training begins.

The `run()` method should also accept an approved or modified execution plan, making the system suitable for both fully automatic and governed ML workflows.

## 20. Decisions to preserve for future expansion

Several structural choices should be made now, even though their implementations belong to later releases.

| Concern               | Architectural commitment                                      |
| --------------------- | ------------------------------------------------------------- |
| Deep learning         | Model architectures registered as model-family plugins        |
| Time-series ML        | Separate validation and feature-availability contracts        |
| Data augmentation     | Pluggable generator and evaluator interfaces                  |
| Feature engineering   | Versioned, fitted transformation graph                        |
| Meta-learning         | Dataset fingerprint and experiment-history store              |
| Human approval        | Policy checkpoints within the orchestrator                    |
| Production monitoring | Prediction/label event schema                                 |
| Distributed execution | Scheduler-independent workflow task definitions               |
| Governance            | Audit trail for every automated decision                      |
| Benchmarking          | Independent dataset suite and reproducible comparison harness |

These commitments keep future expansion possible without forcing us to implement everything in the first release.

## 21. Principal technical risks

The highest risks are not library installation or implementing model adapters. They are architectural and statistical.

| Risk                                         | Mitigation                                                          |
| -------------------------------------------- | ------------------------------------------------------------------- |
| Excessive automation produces invalid models | Validation invariants and explicit blocking conditions              |
| Search becomes slower than existing AutoML   | Baselines, budget-aware search, profiling of orchestration overhead |
| Inconsistent transformations at inference    | Serialize the complete fitted preprocessing/model graph             |
| Model performance inflated by leakage        | Partition-aware execution and adversarial tests                     |
| Integrations create dependency conflicts     | Optional extras, capability checks, pinned tested combinations      |
| Meta-learning overfits benchmark datasets    | Isolated evaluation datasets and provenance                         |
| Model artifacts become incompatible          | Explicit artifact schemas and compatibility policy                  |
| Broad scope delays useful releases           | Complete tabular vertical slice first                               |

The most important early design decision is to make a model candidate a complete, independently evaluable pipeline, not merely an estimator.

A candidate must include its preprocessing choices, feature graph, sampling policy, estimator, hyperparameters, evaluation configuration, and runtime requirements.

That enables legitimate comparison between end-to-end approaches rather than between algorithms trained on arbitrarily different data.

## Final architecture recommendation

The framework should have five foundational abstractions:

Dataset

Immutable data identity, schema, semantics, partitions, and provenance.

Plan

Declarative, auditable description of the proposed ML workflow and decision policies.

Experiment

An isolated execution of a complete candidate pipeline under fixed validation and resource constraints.

Model Bundle

A validated, portable inference pipeline with schema, metadata, and compatibility requirements.

Lifecycle

Versioned relationships between datasets, experiments, artifacts, deployments, and monitoring outcomes.

These five abstractions, supported by a deterministic orchestrator and an extensible decision planner, form the architectural foundation.

The immediate engineering priority is to implement `Dataset → Plan → Experiment → Model Bundle` end to end. That gives us a working, testable framework. Lifecycle orchestration and adaptive intelligence can then be introduced without redesigning the foundation.

This specification is a proposed architecture, not a claim that the package or its interfaces have already been implemented.
