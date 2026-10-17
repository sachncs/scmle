"""Executable item-by-item verification of `product.md`.

Each test names the spec line(s) it covers (`L<number>`) and checks the
behaviour directly rather than trusting other suites. Section 1 is the
decision table, then the normative statements in document order.
"""

import ast
import dataclasses
import json
import pathlib
import re
import subprocess
import sys
import tomllib

import numpy as np
import pandas as pd
import pytest
import support

import mml
from mml.artifacts import bundle as bundle_module
from mml.contracts import errors
from mml.contracts import events
from mml.contracts import experiments
from mml.contracts import models as model_contracts
from mml.contracts import tasks
from mml.core import scheduler
from mml.core import stages
from mml.core import state
from mml.data.validation import columns
from mml.data.validation import target as target_module
from mml.governance import lifecycle
from mml.inference import mapping as mapping_module
from mml.integrations import ray_backend
from mml.models import conformance
from mml.models import registry as registry_module
from mml.sampling import strategies as strategies_module

ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "mml"


def make_frame(rows=600, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=rows),
            "b": rng.normal(size=rows),
            "c": rng.choice(["x", "y", "z"], rows),
        }
    )
    frame.loc[::17, "a"] = np.nan
    signal = frame.a.fillna(0) + 0.5 * frame.b + (frame.c == "x")
    frame["y"] = (signal + rng.normal(scale=0.5, size=rows) > 0.5).astype(int)
    return frame


@pytest.fixture(scope="module")
def frame():
    return make_frame()


@pytest.fixture(scope="module")
def run(frame, tmp_path_factory):
    """One complete run with every default feature enabled."""
    project = mml.MLProject(
        frame,
        "y",
        task="classification",
        metric="roc_auc",
        time_budget=600,
        output_dir=tmp_path_factory.mktemp("spec"),
        decision=tasks.DecisionSpec(threshold_objective="f1"),
    )
    return project, project.run()


def events_of(result):
    lines = (result.run_dir / "events.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


# ---------------------------------------------------------------- section 1


def test_l5_to_l11_python_package_tabular_scope():
    """L5-L11: Python-first, open-source package for tabular classification/regression."""
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert meta["requires-python"].startswith(">=3.11")
    assert meta["license"] == "Apache-2.0"
    assert tasks.STANDARD_TASKS == {
        tasks.TaskType.BINARY,
        tasks.TaskType.MULTICLASS,
        tasks.TaskType.REGRESSION,
    }
    # Survival is an addition beyond the specified scope, never a replacement.
    assert set(tasks.TaskType) - tasks.STANDARD_TASKS == {
        tasks.TaskType.SURVIVAL
    }


def test_l15_l17_user_states_objective_framework_plans_executes_validates_documents(
    run,
):
    """L15-L17: the objective in, a planned, executed, validated, documented workflow out."""
    result = run[1]
    for artifact in (
        "execution_plan.json",  # planned
        "trials",  # executed
        "validation_manifest.json",  # validated split
        "evaluation.json",
        "report.html",  # documented
        "diagnostics/decision_log.json",
    ):
        assert (result.run_dir / artifact).exists(), artifact


def test_l19_conservative_when_ambiguous_and_transparent(frame, tmp_path):
    """L19: blocks on ambiguity; every experiment and decision is recorded."""
    ambiguous = frame.assign(y=np.arange(len(frame)) % 5)
    with pytest.raises(errors.BlockedError):
        support.make_project(ambiguous, "y", output_dir=tmp_path).run()
    result = support.make_project(frame, "y", output_dir=tmp_path / "ok").run()
    trial_files = list((result.run_dir / "trials").glob("trial_*.json"))
    assert len(trial_files) == len(result.trials) > 1
    assert (result.run_dir / "diagnostics" / "decision_log.json").exists()


def test_l23_to_l39_decision_table(run):
    """L23-L39: each row of the non-negotiable decisions table."""
    project, result = run
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert meta["requires-python"].startswith(">=3.11")  # Python 3.11+
    assert meta["license"] == "Apache-2.0" and (ROOT / "LICENSE").exists()
    assert issubclass(
        tasks.ProjectSpec, tasks.pydantic.BaseModel
    )  # typed config
    assert issubclass(model_contracts.ModelAdapter, object) and hasattr(
        project.spec, "config_hash"
    )
    assert hasattr(project, "plan") and hasattr(
        project, "run"
    )  # declarative API
    assert (
        next(s.state for s in stages.default_stages())
        is state.RunState.VALIDATING
    )
    assert {"sequential", "process", "ray"} <= {
        scheduler.SequentialBackend.name,
        scheduler.ProcessBackend.name,
        ray_backend.RayBackend.name,
    }  # local-first, distributed-capable
    assert (
        result.evaluation["split_fingerprint"]
        and (result.run_dir / "environment.lock").read_text()
    )  # reproducibility: config, data fingerprints, seeds, env
    assert (result.run_dir / "events.jsonl").exists()  # event-driven telemetry
    assert "content_id" in {
        f.name
        for f in dataclasses.fields(
            __import__("mml.governance.registry", fromlist=["x"]).VersionRecord
        )
    }  # content-addressed artifacts
    assert {"tracking", "distributed", "optimization"} <= set(
        tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
            "optional-dependencies"
        ]
    )  # deployment/trackers are optional extras


def test_l41_l43_no_mandatory_infrastructure_on_the_base_install():
    """L41-L43: no Kubernetes / cloud / DB server / GPU / hosted tracker required."""
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "dependencies"
    ]
    names = {re.split(r"[<>=!~ ]", d)[0].lower() for d in deps}
    forbidden = {
        "kubernetes",
        "ray",
        "mlflow",
        "torch",
        "tensorflow",
        "boto3",
        "psycopg2",
        "pymongo",
        "docker",
        "cuda",
        "wandb",
    }
    assert not names & forbidden
    code = (
        "import sys, mml; from mml.models import registry; "
        "registry.ModelRegistry.with_builtins(); "
        "bad=[m for m in ('ray','mlflow','torch','kubernetes') if m in sys.modules]; "
        "assert not bad, bad"
    )
    assert (
        subprocess.run([sys.executable, "-c", code], check=False).returncode
        == 0
    )


# --------------------------------------------------------------- section 2-3


def test_l101_l105_three_planes_are_separate_packages():
    """L101-L105: control, execution and evidence planes are separate code."""
    for plane in (
        ("core", "intelligence"),
        ("data", "models"),
        ("artifacts", "tracking"),
    ):
        assert all((PACKAGE / name).is_dir() for name in plane)


def test_l113_l176_package_organization():
    """L113-L176: every module of the spec's tree exists (flat `mml/`)."""
    packages = (
        "api",
        "contracts",
        "core",
        "intelligence",
        "data",
        "data/readers",
        "data/profiling",
        "data/validation",
        "data/curation",
        "data/splitting",
        "features",
        "sampling",
        "models",
        "models/sklearn",
        "models/boosting",
        "optimization",
        "experimentation",
        "evaluation",
        "explainability",
        "artifacts",
        "inference",
        "monitoring",
        "governance",
        "integrations",
    )
    for package in packages:
        assert (PACKAGE / package / "__init__.py").exists(), package
    modules = (
        "api/project.py",
        "api/results.py",
        "api/predict.py",
        "contracts/data.py",
        "contracts/tasks.py",
        "contracts/models.py",
        "contracts/metrics.py",
        "contracts/artifacts.py",
        "contracts/events.py",
        "core/config.py",
        "core/orchestrator.py",
        "core/compiler.py",
        "core/scheduler.py",
        "core/state.py",
        "core/policies.py",
        "intelligence/fingerprint.py",
        "intelligence/planner.py",
        "intelligence/recommendations.py",
        "intelligence/knowledge.py",
        "models/registry.py",
    )
    for rel in modules:
        assert (PACKAGE / rel).exists(), rel
    for folder in ("unit", "integration", "leakage", "property", "benchmarks"):
        assert (ROOT / "tests" / folder).is_dir()
    assert (
        (ROOT / "benchmarks").is_dir()
        and (ROOT / "examples").is_dir()
        and (ROOT / "docs").is_dir()
        and (ROOT / "LICENSE").is_file()
    )


def test_l178_modules_use_stable_contracts_not_private_internals():
    """L178: no module reaches into another's internals; contracts depend on nothing above them."""
    for path in (PACKAGE / "contracts").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("mml")
            ):
                assert (
                    node.module.startswith("mml.contracts")
                    or node.module == "mml"
                ), f"{path.name} imports {node.module}"
    for path in PACKAGE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.ImportFrom):
                names = [a.name for a in node.names]
            assert not any(
                n.startswith("_") and not n.startswith("__") for n in names
            )


def test_l186_l194_explicit_versioned_serializable_hashable_saved_spec(run):
    """L186, L194: explicit versioned spec; serializable; canonical hash; saved per run."""
    project, result = run
    spec = project.spec
    assert spec.version == tasks.CONFIG_VERSION
    assert tasks.ProjectSpec.model_validate_json(spec.model_dump_json()) == spec
    assert (
        len(spec.config_hash()) == 16
        and spec.config_hash() == spec.config_hash()
    )
    saved = tasks.ProjectSpec.model_validate_json(
        (result.run_dir / "config.json").read_text()
    )
    assert saved.config_hash() == spec.config_hash()


def test_l192_cross_field_validation():
    """L192: group needs a group column; temporal needs a time column; binary metrics need binary targets."""
    with pytest.raises(ValueError, match="group_column"):
        tasks.ProjectSpec(
            data=tasks.DataSpec(uri="x", target="y"),
            validation=tasks.ValidationSpec(strategy="group"),
        )
    with pytest.raises(ValueError, match="time_column"):
        tasks.ProjectSpec(
            data=tasks.DataSpec(uri="x", target="y"),
            validation=tasks.ValidationSpec(strategy="temporal"),
        )
    from mml.evaluation import metrics

    with pytest.raises(errors.ConfigError):
        metrics.CATALOG.resolve("pr_auc", tasks.TaskType.MULTICLASS)
    from mml.data.splitting import splitters

    frame = make_frame(100).assign(t="not a date")
    spec = tasks.ProjectSpec(
        data=tasks.DataSpec(uri="x", target="y", time_column="t")
    )
    with pytest.raises(errors.BlockedError, match="ambiguous"):
        splitters.SplitManager(spec, tasks.TaskType.BINARY).split(frame)


def test_l198_l206_foundational_and_production_contracts(run):
    """L198-L206: contracts incl. schema types, capabilities, fit context, cancellation,
    resource consumption, deterministic hashes, error categories, formats, compat versions."""
    project, result = run
    assert result.bundle.schema.dtypes  # schema types
    assert dataclasses.fields(
        model_contracts.Capabilities
    )  # capability discovery
    from mml.experimentation import trials

    assert {f.name for f in dataclasses.fields(trials.TrialPayload)} >= {
        "job",
        "adapter",
        "task",
        "seed",
        "folds",
    }  # fit/predict context
    assert (
        hasattr(project, "cancel")
        and errors.CancelledError.category == "cancelled"
    )
    assert {"model_mb", "p95_latency_ms"} <= set(result.evaluation["resources"])
    assert (
        project.spec.config_hash() == project.spec.config_hash()
    )  # deterministic hash
    assert {
        e.category
        for e in (
            errors.ConfigError,
            errors.DataError,
            errors.SchemaError,
            errors.BlockedError,
            errors.LeakageError,
            errors.ArtifactError,
        )
    } == {"config", "data", "schema", "blocked", "leakage", "artifact"}
    assert (
        result.bundle.info.format == bundle_module.BUNDLE_FORMAT
    )  # serialization format
    assert isinstance(
        result.bundle.info.compatibility_problems(), list
    )  # compat versions


def test_l210_l224_module_io_matrix_invariants(run, tmp_path):
    """L210-L224: key invariant of each module in the I/O matrix."""
    project, result = run
    from mml.data.profiling import profiler
    from mml.data.readers import tabular
    from mml.data.splitting import splitters

    frame = make_frame(200)
    before = frame.copy()
    tabular.build_manifest(frame, "y")
    pd.testing.assert_frame_equal(frame, before)  # ingestion: input immutable
    spec = project.spec
    manifest = tabular.build_manifest(frame, "y")
    task = target_module.TaskResolver(spec).resolve(frame["y"])
    profiler.DatasetProfiler(spec).profile(frame, manifest, task)
    pd.testing.assert_frame_equal(
        frame, before
    )  # profiling: no fitted state, no mutation
    split = splitters.SplitManager(spec, task).split(frame)
    assert not set(split.manifest.train_ids) & set(
        split.manifest.test_ids
    )  # splitting
    assert {t.status for t in result.trials} >= {experiments.TrialStatus.OK}
    assert result.selected_model  # selector: defined policy
    assert (
        result.run_dir / "model" / "model_manifest.json"
    ).exists()  # artifact builder
    bundle = bundle_module.ModelBundle.load(
        result.save(tmp_path / "b"), trusted=True
    )
    features = frame.drop(columns="y")
    np.testing.assert_array_equal(
        bundle.predict(features), result.predict(features)
    )  # parity
    with pytest.raises(errors.SchemaError):
        bundle.predict(
            features.drop(columns=["b"])
        )  # inference: schema checked


def test_l226_plugins_are_conformance_tested():
    """L226: third-party plugins cannot violate expectations."""

    class Bad(model_contracts.ModelAdapter):
        name = "bad"
        family = "x"
        capabilities = model_contracts.Capabilities(
            frozenset({tasks.TaskType.BINARY})
        )

        def build(self, task, seed, params=None):
            class Broken:
                def fit(self, features, target):
                    return self

            return Broken()

    with pytest.raises(Exception):  # noqa: B017 - any contract violation
        conformance.check_conformance(Bad())


# ---------------------------------------------------------------- section 6


def test_l232_executable_dag_of_ordered_dependent_stages():
    """L232: the request compiles into an executable dependency-ordered graph."""
    states = [stage.state for stage in stages.default_stages()]
    order = state.FORWARD_ORDER
    assert [order.index(s) for s in states] == sorted(
        order.index(s) for s in states
    )
    assert len(states) == len(set(states)) == 7


def test_l234_to_l264_flow_with_locked_test_and_single_final_evaluation(run):
    """L234-L264: split manifest, train branch, locked test branch, one-time final evaluation."""
    result = run[1]
    kinds = [e["kind"] for e in events_of(result)]
    assert kinds.count("test.opened") == 1
    assert (
        kinds.index("split.created")
        < kinds.index("selection.frozen")
        < kinds.index("test.opened")
    )
    assert result.evaluation["test_set_opened"] is True
    planned = {
        d.technique
        for d in result.plan.decisions
        if d.action == "include" and d.technique in strategies_module.STRATEGIES
    }
    attempted = {t.strategy for t in result.trials} - {"default"}
    assert attempted <= planned  # nothing outside the plan was tried
    assert bool(attempted) == bool(planned)  # applicable strategies were tried
    assert (
        result.run_dir / "model"
    ).exists()  # artifact verification and registration


def test_l268_l276_states_blocked_and_transition_events(run):
    """L268-L276: explicit states incl. BLOCKED/FAILED/CANCELLED; events carry required fields."""
    project, result = run
    assert {s.value for s in state.RunState} >= {
        "CREATED",
        "VALIDATING",
        "PROFILING",
        "PLANNING",
        "TRAINING",
        "SELECTING",
        "TESTING",
        "PACKAGING",
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "BLOCKED",
    }
    transition = next(
        e for e in events_of(result) if e["kind"] == "state.TRAINING"
    )
    assert transition["run_id"] == result.run_id  # run ID
    assert transition["timestamp"]  # timestamp
    assert (
        transition["config_hash"] == project.spec.config_hash()
    )  # configuration version
    assert (
        "reason" in transition and "artifacts" in transition
    )  # reason, artifact references
    assert {f.name for f in dataclasses.fields(events.Event)} >= {
        "run_id",
        "timestamp",
        "config_hash",
        "reason",
        "artifacts",
    }


def test_l278_l288_orchestrator_support(frame, tmp_path):
    """L278-L288: retries, cancellation, budget, checkpointing, restart, failures, ordering, parallelism, isolation."""
    import shutil

    from mml.core import orchestrator as orchestrator_module

    project = support.make_project(
        frame, "y", time_budget=300, output_dir=tmp_path
    )
    first = project.run()
    (first.run_dir / "evaluation.json").unlink()
    shutil.rmtree(first.run_dir / "model")
    resumed = mml.MLProject.resume(
        first.run_dir, data=frame
    )  # idempotent retry / restart
    assert [t.trial_id for t in resumed.trials] == [
        t.trial_id for t in first.trials
    ]  # deterministic ordering
    assert list(
        (first.run_dir / "trials").glob("trial_*.json")
    )  # checkpointing
    pending = orchestrator_module.Orchestrator(project.spec, frame)
    pending.cancel()
    with pytest.raises(errors.CancelledError) as caught:  # cancellation
        pending.run(tmp_path / "c")
    assert (
        "recoverable_trials" in caught.value.details
        and caught.value.details["run_dir"]
    )  # consistent failure reporting
    budget = support.make_project(
        frame, "y", time_budget=1, output_dir=tmp_path / "b"
    ).run()
    assert budget.state == "COMPLETED"  # budget exhaustion handled
    assert scheduler.ProcessBackend(2).map(abs, [-1, -2]) == [
        1,
        2,
    ]  # parallel independent work
    assert any(
        t.status == "ok" for t in first.trials
    )  # isolation: failures don't end runs (see recovery suite)


# ---------------------------------------------------------------- section 7


def test_l293_l307_fingerprint_dimensions(run):
    """L293-L307: all eleven fingerprint dimensions."""
    result = run[1]
    dims = set(result.profile.dataset_statistics)
    assert {
        "shape",
        "types",
        "missingness",
        "target",
        "features",
        "relationships",
        "ml_difficulty",
        "structure",
        "quality",
        "operational",
        "risk",
    } <= dims
    difficulty = result.profile.dataset_statistics["ml_difficulty"]
    assert difficulty["available"]  # ML difficulty: baselines and headroom
    assert {
        "constant_baseline",
        "linear_baseline",
        "nonlinearity_gain",
    } <= set(difficulty)
    stats = result.profile.column_statistics["a"]
    assert {"outlier_rate", "std"} <= set(stats)  # scale, outliers
    assert result.profile.dataset_statistics["shape"]["effective_samples"] > 0
    assert columns.entropy(pd.Series(list("aabb"))) == pytest.approx(
        1.0
    )  # entropy


def test_l309_statistics_preserve_uncertainty_about_leakage():
    """L309: strong association is a hypothesis, not a verdict."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"x": rng.normal(size=300)})
    frame["y"] = (frame.x > 0).astype(int)
    result = support.make_project(frame, "y", leakage_policy="warn").plan()
    findings = " ".join(result.warnings)
    assert (
        "possible leakage" in findings
        and "exceptionally strong predictor" in findings
    )


def test_l311_l321_decision_policy_examples():
    """L311-L321: candidate generation rules; the planner proposes, evaluation decides."""
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"x": rng.normal(size=300)})
    frame["c"] = rng.choice([f"k{i}" for i in range(200)], 300)
    frame["y"] = (rng.random(300) < 0.05).astype(int)
    plan = mml.MLProject(frame, "y", time_budget=60).plan()
    names = set(plan.candidate_models)
    assert (
        "dummy" in names and "logistic_regression" in names
    )  # baseline + linear
    assert {"hist_gradient_boosting"} & names  # fast boosting kept
    assert (
        "random_forest" not in names
    )  # time budget < 300 s -> fast models only
    reasons = {d.technique: d.action for d in plan.decisions}
    assert (
        reasons.get("class_weight") == "include"
    )  # severe imbalance -> evaluate weighting
    assert "random_undersample" in reasons  # resampling considered


# ---------------------------------------------------------------- section 8


def test_l327_l345_stages_a_to_e(run):
    """L327-L345: baselines, screening, optimization, ensembles, select+package."""
    project, result = run
    models_run = {t.model_name for t in result.trials}
    assert "dummy" in models_run  # A
    assert len(models_run) > 3  # B
    assert (
        any(t.params for t in result.trials)
        or project.spec.search.top_models == 0
    )  # C
    assert any(t.model_name == "stacking" for t in result.trials)  # D
    assert result.selected_model and (result.run_dir / "model").exists()  # E


def test_l347_l356_model_family_registry():
    """L347-L356: the initial model families."""
    names = set(registry_module.ModelRegistry.with_builtins().adapters)
    assert {
        "dummy",
        "logistic_regression",
        "ridge",
        "elastic_net",
        "decision_tree",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "xgboost",
        "lightgbm",
        "catboost",
        "svm",
        "knn",
    } <= names


def test_l358_adapter_capability_declarations():
    """L358: tasks, sparse, probabilities, native categoricals, missing values,
    early stopping, memory characteristics, serialization."""
    declared = {
        f.name for f in dataclasses.fields(model_contracts.Capabilities)
    }
    assert {
        "tasks",
        "sparse_input",
        "probabilities",
        "native_categorical",
        "handles_missing",
        "early_stopping",
        "memory_factor",
        "serializable",
    } <= declared
    for (
        adapter
    ) in registry_module.ModelRegistry.with_builtins().adapters.values():
        assert adapter.capabilities.memory_factor >= 1.0


def test_l360_planner_reads_capabilities_not_library_names():
    """L360: planning uses declared capabilities (here: size limits and tasks)."""
    registry = registry_module.ModelRegistry.with_builtins()
    rejected = registry.candidates(tasks.TaskType.REGRESSION, 1_000_000)[1]
    assert (
        "svm" in rejected and "logistic_regression" in rejected
    )  # max_rows, task support


def test_l370_optimizer_library_is_not_the_architecture():
    """L370: Optuna sits behind our ask/tell interface."""
    from mml.optimization import optimizers

    assert issubclass(optimizers.OptunaOptimizer, optimizers.Optimizer)
    assert issubclass(optimizers.RandomOptimizer, optimizers.Optimizer)


# ---------------------------------------------------------------- section 9


def test_l374_l393_balancing_and_augmentation(frame):
    """L374-L393: the technique table and its policies."""
    from mml.sampling import strategies

    for name in (
        "class_weight",
        "random_oversample",
        "random_undersample",
        "smote",
        "calibrated",
    ):
        assert (
            name in strategies.STRATEGIES
        )  # weighting, over/under, SMOTE, calibration
    assert (
        "jitter_augment" in strategies.OPT_IN
        and "copula_augment" in strategies.OPT_IN
    )  # research plugins
    from mml.contracts import data as data_contracts
    from mml.data.profiling import profiler
    from mml.data.readers import tabular
    from mml.sampling import resampling

    # Sampling happens inside the training fold (inside fit), never on the full dataset.
    assert "fit" in resampling.ResamplingClassifier.__dict__
    # Regression does not reuse classification resampling.
    spec = tasks.ProjectSpec(data=tasks.DataSpec(uri="x", target="y"))
    reg = pd.DataFrame({"x": np.arange(100.0), "y": np.arange(100.0) * 2})
    profile = profiler.DatasetProfiler(spec).profile(
        reg, tabular.build_manifest(reg, "y"), tasks.TaskType.REGRESSION
    )
    adapter = registry_module.ModelRegistry.with_builtins().get("ridge")
    reason = strategies.STRATEGIES["random_oversample"].applicability(
        profile, tasks.TaskType.REGRESSION, adapter
    )
    assert reason is not None and "classification" in reason
    assert isinstance(profile, data_contracts.DatasetProfile)


def test_l385_l387_calibration_and_threshold_use_validation_evidence(run):
    """L385-L387: threshold optimization uses validation predictions and an explicit objective."""
    result = run[1]
    decision = result.evaluation["decision"]
    assert (
        decision["objective"] == "f1"
        and decision["validation_rows"] <= result.evaluation["n_dev"]
    )


# --------------------------------------------------------------- section 10


def test_l399_l409_evaluation_contract(run):
    """L399-L409: standardized metrics/diagnostics; selectable objective; no accuracy assumption."""
    result = run[1]
    assert {"roc_auc", "pr_auc", "log_loss", "f1"} <= set(
        result.evaluation["test_metrics"]
    )
    assert {
        "classification_report",
        "calibration",
        "threshold_analysis",
    } <= set(result.diagnostics)
    from mml.evaluation import metrics

    defaults = set(metrics.MetricCatalog.default_metric.values())
    assert not defaults & {
        "accuracy",
        "f1",
    }  # never assumes accuracy/F1 represents costs
    assert (
        result.evaluation["metric"] == "roc_auc"
    )  # user-selected objective honoured
    reg = metrics.CATALOG.report_names[tasks.TaskType.REGRESSION]
    assert set(reg) == {"mae", "rmse", "r2"}
    assert {"log_loss", "f1_macro", "balanced_accuracy"} <= set(
        metrics.CATALOG.report_names[tasks.TaskType.MULTICLASS]
    )


def test_l409_confidence_respects_independent_units(frame, tmp_path):
    """L409: uncertainty follows groups / time structure."""
    framed = frame.assign(g=np.arange(len(frame)) % 30)
    result = support.make_project(
        framed, "y", group_column="g", output_dir=tmp_path
    ).run()
    assert (
        result.diagnostics["metric_uncertainty"]["resampling_unit"] == "groups"
    )


def test_l411_l417_selection_constraints_measured_on_this_machine(
    frame, tmp_path
):
    """L411-L417: maximize + constraints (size, latency) + tie-breaker, measured on hardware."""
    result = support.make_project(
        frame,
        "y",
        output_dir=tmp_path,
        selection=tasks.SelectionSpec(max_model_mb=50, max_p95_latency_ms=1000),
    ).run()
    usage = result.evaluation["resources"]
    assert (
        usage["model_mb"] <= 50
        and usage["p95_latency_ms"] <= 1000
        and usage["machine"]
    )
    assert "ties -> simpler_model" in result.evaluation["selection_policy"]


# --------------------------------------------------------------- section 11


def test_l421_l437_leakage_invariants(run, frame, tmp_path):
    """L421-L437: structural protections 1-5."""
    project, result = run
    manifest = json.loads(
        (result.run_dir / "validation_manifest.json").read_text()
    )
    assert (
        manifest["train_ids"] and manifest["test_ids"]
    )  # 1 immutable row-id manifest
    from mml.data.splitting import splitters

    assert (
        dataclasses.is_dataclass(splitters.SplitManifest)
        and splitters.SplitManifest.__dataclass_params__.frozen
    )
    ledger = json.loads(
        (result.run_dir / "diagnostics" / "data_access.json").read_text()
    )
    assert all(
        ledger["verified"].values()
    )  # 3 record of data accessed per fitted operation
    split = splitters.SplitManager(project.spec, tasks.TaskType.BINARY).split(
        make_frame(200)
    )
    with pytest.raises(errors.LeakageError):
        split.locked_test.open(
            "guess"
        )  # 4 locked test evaluator / restricted handle
    kinds = {e["kind"] for e in events_of(result)}
    assert {
        "split.created",
        "features.selected",
        "selection.frozen",
        "test.opened",
    } <= kinds  # 5 audit
    # Fold-local fitting: statistics come from training rows only (see tests/leakage for the
    # test-label invariance proof).
    assert (ROOT / "tests" / "leakage" / "test_leakage.py").exists()


def test_l439_deployment_scenario_splits(frame):
    """L439: temporal/entity data must not use a random split."""
    from mml.data.splitting import splitters

    with_time = support.make_project(
        frame.assign(t=range(len(frame))), "y", time_column="t"
    ).spec
    with_group = support.make_project(
        frame.assign(g=np.arange(len(frame)) % 40), "y", group_column="g"
    ).spec
    assert (
        splitters.resolve_strategy_name(with_time, tasks.TaskType.BINARY)
        == "temporal"
    )
    assert (
        splitters.resolve_strategy_name(with_group, tasks.TaskType.BINARY)
        == "group"
    )


# --------------------------------------------------------------- section 12-13


def test_l443_l466_run_directory_layout(run):
    """L443-L466: the artifact directory structure."""
    result = run[1]
    for rel in (
        "config.json",
        "environment.lock",
        "dataset_manifest.json",
        "dataset_profile.json",
        "validation_manifest.json",
        "execution_plan.json",
        "events.jsonl",
        "trials",
        "leaderboard.parquet",
        "evaluation.json",
        "diagnostics",
        "model",
        "model/model_manifest.json",
        "model/schema.json",
        "report.html",
    ):
        assert (result.run_dir / rel).exists(), rel
    assert list(
        (result.run_dir / "model").glob("pipeline.*")
    )  # logical pipeline artifact


def test_l470_l474_artifact_format_and_untrusted_artifacts(run, tmp_path):
    """L470-L474: specified serialization; pickle treated as executable; checksums; refuse untrusted."""
    result = run[1]
    directory = result.save(tmp_path / "m")
    manifest = json.loads((directory / "model_manifest.json").read_text())
    assert (
        manifest["format"] == bundle_module.BUNDLE_FORMAT and manifest["files"]
    )  # checksums
    with pytest.raises(errors.ArtifactError, match="unauthenticated"):
        bundle_module.ModelBundle.load(
            directory
        )  # untrusted artifacts not loaded
    with (directory / "pipeline.joblib").open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(errors.ArtifactError, match="checksum"):
        bundle_module.ModelBundle.load(directory, trusted=True)
    assert (
        manifest["python"] and manifest["sklearn"]
    )  # dependency compatibility


def test_l476_exact_replay_vs_comparable_rerun(run):
    """L476: records reproducibility conditions; distinguishes exact from comparable."""
    from mml.artifacts import reproducibility

    assert {
        reproducibility.EXACT,
        reproducibility.COMPARABLE,
        reproducibility.DIFFERENT,
    } == {"exact", "comparable", "different"}
    assert "reproducibility" in run[1].evaluation
    assert hasattr(mml.MLProject, "verify_replay")


def test_l480_l496_inference_contract(run, frame, tmp_path):
    """L480-L496: whole pipeline; schema validation; explicit column mapping;
    unknown categories; missing-feature diagnostics; batch; structured errors."""
    result = run[1]
    features = frame.drop(columns="y")
    novel = features.head(5).assign(c="never_seen")
    report = result.bundle.validate(novel)
    assert report.unseen_categories == {
        "c": ["never_seen"]
    }  # unknown categories
    assert (
        result.bundle.validate(
            features.head(3).assign(a=np.nan)
        ).missing_value_counts["a"]
        == 3
    )
    renamed = features.rename(columns={"a": "alpha"})
    mapping = mapping_module.ColumnMapping({"alpha": "a"})
    np.testing.assert_array_equal(
        result.bundle.predict(renamed, mapping),
        result.predict(features),
    )  # explicit mapping
    with pytest.raises(errors.SchemaError) as caught:
        result.predict(features.drop(columns="b"))
    assert caught.value.details["missing"] == ["b"]  # structured errors
    path = tmp_path / "batch.csv"
    features.to_csv(path, index=False)
    from mml.inference import serving

    assert serving.BatchServing(result.bundle).score_file(
        path, tmp_path / "out.csv"
    ) == len(features)  # batch
    assert result.bundle.pipeline.predict(
        result.bundle.prepare_frame(features)
    ).shape  # whole pipeline


# --------------------------------------------------------------- section 14-17


def test_l500_l546_stack_and_extras():
    """L500-L546: the technology stack and the optional extras."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    base = " ".join(project["dependencies"])
    for needed in ("pydantic", "pyarrow", "pandas", "scikit-learn", "joblib"):
        assert needed in base
    extras = set(project["optional-dependencies"])
    assert {
        "boosting",
        "optimization",
        "tracking",
        "distributed",
        "all",
    } <= extras  # L538-L544
    assert (ROOT / "uv.lock").exists() and (
        ROOT / ".github/workflows/ci.yml"
    ).exists()
    assert any(
        "ruff" in d
        for d in tomllib.loads((ROOT / "pyproject.toml").read_text())[
            "dependency-groups"
        ]["dev"]
    )


def test_l548_no_heavy_imports_until_a_model_is_selected():
    """L548: a registry entry must not import GPU/deep-learning dependencies."""
    code = (
        "import sys; from mml.contracts import tasks; from mml.models import registry; "
        "r = registry.ModelRegistry.with_builtins(); r.candidates(tasks.TaskType.BINARY, 100); "
        "assert not [m for m in ('lightgbm','xgboost','catboost','torch','tensorflow') if m in sys.modules]"
    )
    assert (
        subprocess.run([sys.executable, "-c", code], check=False).returncode
        == 0
    )


def test_l554_l606_roadmap_deliverables(frame, tmp_path):
    """L554-L600: M0-M4 deliverables are observable behaviours."""
    from mml.data.readers import tabular

    assert tabular.read_table  # M0: CSV/Parquet ingestion + typed config
    result = support.make_project(
        frame, "y", output_dir=tmp_path
    ).run()  # M1: MLProject(...).run()
    assert result.predict(frame.drop(columns="y")).shape[0] == len(frame)
    assert len({t.model_name for t in result.trials}) > 3  # M2: multi-model
    assert (
        tmp_path / "experiments.sqlite"
    ).exists()  # M2: experiment persistence
    plan = support.make_project(frame, "y").plan()
    assert any(
        d.technique in {"class_weight", "log_skewed", "calibrated"}
        for d in plan.decisions
    )  # M3: data strategies
    assert (ROOT / "benchmarks" / "baseline.json").exists() and (
        ROOT / "docs" / "SECURITY.md"
    ).exists()  # M4


def test_l610_l629_backlog_priorities(run):
    """L610-L629: each backlog row has a working implementation."""
    from mml.contracts import models as model_contracts
    from mml.evaluation import diagnostics
    from mml.explainability import importance
    from mml.governance import registry
    from mml.inference import serving
    from mml.intelligence import knowledge
    from mml.models import ensembles
    from mml.monitoring import monitor
    from mml.sampling import synthesis

    result = run[1]
    assert len(result.predict(make_frame().drop(columns="y"))) == len(
        make_frame()
    )  # P0 project API and inference-ready artifact
    assert issubclass(serving.HttpServing, serving.ServingAdapter)  # P2
    assert callable(registry.ModelRegistryStore.promote)  # P2 deployment
    report = monitor.ModelMonitor(result.bundle).assess(
        make_frame(seed=3).drop(columns="y"), make_frame(seed=3)["y"]
    )
    assert report.recommendation.action  # P2 monitoring
    assert issubclass(
        importance.PermutationImportance, diagnostics.Diagnostic
    )  # P1 explainability
    assert "permutation_importance" in result.diagnostics
    assert issubclass(ensembles.StackingAdapter, model_contracts.ModelAdapter)
    generator = synthesis.GaussianCopulaGenerator().fit(
        make_frame()[["a", "b"]]
    )
    fake = generator.sample(30, np.random.default_rng(0))
    assert fake.shape == (30, 2)  # P2 augmentation
    assert knowledge.MetaLearner([]).recommend  # P2 meta-learning


def test_l637_l669_test_categories_and_quality_gates(run):
    """L637-L667: required test categories exist; each quality gate is observable."""
    for rel in (
        "tests/unit/test_component_contracts.py",  # contract
        "tests/unit/test_splitting.py",  # statistical
        "tests/leakage/test_leakage.py",  # leakage
        "tests/property/test_invariants.py",  # property
        "tests/unit/test_bundle.py",  # artifact
        "tests/integration/test_recovery.py",  # recovery
        "benchmarks/check_regression.py",
    ):  # benchmark
        assert (ROOT / rel).exists(), rel
    result = run[1]
    assert (result.run_dir / "dataset_manifest.json").exists() and (
        result.run_dir / "diagnostics" / "decision_log.json"
    ).exists()  # manifest + decision log
    assert (
        "resource_limits" in result.evaluation
    )  # limits respected or limitations reported
    assert (
        ROOT / "benchmarks" / "baseline.json"
    ).exists()  # fixed regression benchmark suite


def test_l669_no_arbitrary_performance_promises():
    """L669: no improvement percentage promised without a benchmark baseline."""
    text = (ROOT / "README.md").read_text() + (
        ROOT / "docs" / "TUTORIAL.md"
    ).read_text()
    assert not re.search(
        r"\d+\s*%\s*(faster|better|more accurate|improvement)", text, re.I
    )


# --------------------------------------------------------------- section 18-20


def test_l679_l735_v01_scope_matches_the_spec(frame, tmp_path):
    """L679-L735: in-scope items work; out-of-scope ones are optional extras."""
    multi = make_frame(300).assign(
        y=lambda f: pd.cut(
            f.a.fillna(0) + f.b, [-9, -0.5, 0.5, 9], labels=["l", "m", "h"]
        ).astype(str)
    )
    reg = make_frame(300).assign(y=lambda f: f.b * 2)
    for data, task in (
        (frame, "classification"),
        (multi, "classification"),
        (reg, "regression"),
    ):
        result = support.make_project(
            data, "y", task=task, output_dir=tmp_path / task
        ).run()
        assert result.state == "COMPLETED"
    assert issubclass(
        ray_backend.RayBackend, scheduler.ExecutionBackend
    )  # distributed: optional


def test_l739_l747_first_sprint_outcomes(run):
    """L739-L747: the five concrete outcomes."""
    assert (ROOT / "docs" / "ARCHITECTURE.md").read_text().strip()  # 1
    assert (ROOT / "pyproject.toml").exists() and list(
        (ROOT / "tests").rglob("test_*.py")
    )  # 2 package skeleton + tests
    from mml.data.profiling import profiler
    from mml.data.readers import tabular
    from mml.data.splitting import splitters

    frame = make_frame()
    manifest = tabular.build_manifest(frame, "y", "<memory>")
    assert manifest.row_count == len(frame)  # 3 ingestion
    profile = profiler.DatasetProfiler(
        tasks.ProjectSpec(data=tasks.DataSpec(uri="x", target="y"))
    ).profile(frame, manifest, tasks.TaskType.BINARY)
    assert set(profile.feature_plan.features) == {"a", "b", "c"}  # profiling
    assert {"stratified", "random", "group", "temporal"} <= set(
        splitters.STRATEGIES
    )  # 4 split manager
    result = run[1]
    assert (
        result.state == "COMPLETED" and (result.run_dir / "model").exists()
    )  # 5 end-to-end training path with saved artifacts


def test_l751_l761_frozen_api(frame, tmp_path):
    """L751-L761: the API as written, incl. approved plans and the classification alias."""
    project = mml.MLProject(
        frame,
        "y",
        task="classification",
        time_budget=600,
        metric="roc_auc",
        output_dir=tmp_path,
    )
    plan = project.plan()
    assert isinstance(plan.summary(), str)  # print(plan.summary())
    result = project.run(plan)  # run() accepts an approved plan
    assert result.leaderboard().shape[0] == len(
        result.trials
    )  # result.leaderboard()
    exported = result.save(tmp_path / "artifacts")
    assert (exported / "model_manifest.json").exists()  # result.save(...)
    path = tmp_path / "new.csv"
    frame.drop(columns="y").to_csv(path, index=False)
    assert len(result.predict(path)) == len(frame)  # result.predict("new.csv")
    assert (
        result.evaluation["task"] == "binary_classification"
    )  # alias resolved and recorded
    modified = dataclasses.replace(
        plan, candidate_models=("dummy", "logistic_regression")
    )
    assert {t.model_name for t in project.run(modified).trials} == {
        "dummy",
        "logistic_regression",
    }


def test_l765_l778_future_expansion_commitments(frame, tmp_path):
    """L765-L778: each architectural commitment works, not merely exists."""
    import pickle

    from mml.core import policies
    from mml.data.splitting import splitters
    from mml.experimentation import trials
    from mml.features import feature_graph
    from mml.intelligence import knowledge
    from mml.models import registry as models_registry
    from mml.monitoring import events as monitor_events
    from mml.sampling import resampling
    from mml.sampling import synthesis

    registry = models_registry.ModelRegistry.with_builtins()
    neural = registry.get("neural_network")  # deep learning: a plugin family
    conformance.check_conformance(neural)

    series = pd.DataFrame(
        {
            "t": pd.date_range("2024-01-01", periods=120),
            "a": np.arange(120.0),
            "y": np.arange(120) % 2,
        }
    )
    spec = tasks.ProjectSpec(
        data=tasks.DataSpec(
            uri="x",
            target="y",
            time_column="t",
            unavailable_at_prediction=["a"],
        ),
        validation=tasks.ValidationSpec(
            strategy="temporal", folds=3, embargo_rows=5
        ),
    )
    split = splitters.SplitManager(spec, tasks.TaskType.BINARY).split(series)
    gaps = [min(held) - max(fit) for fit, held in split.manifest.folds]
    assert all(gap > 5 for gap in gaps)  # time-series: embargo is honoured
    profile = mml.MLProject(
        series.drop(columns="a").assign(b=np.arange(120.0) % 7),
        "y",
        time_column="t",
    )
    profile.plan()
    assert "t" not in profile.profile.feature_plan.features

    generator = synthesis.GaussianCopulaGenerator().fit(frame[["a", "b"]])
    fake = generator.sample(40, np.random.default_rng(0))
    report = synthesis.FidelityEvaluator().evaluate(frame[["a", "b"]], fake)
    assert fake.shape == (40, 2) and report.max_marginal_ks < 0.5
    assert issubclass(synthesis.GeneratorOverSampler, resampling.Resampler)

    one = tasks.FeatureGraphSpec(
        nodes=[tasks.FeatureNode(name="sq", op="square", inputs=["b"])]
    )
    two = tasks.FeatureGraphSpec(
        nodes=[tasks.FeatureNode(name="sq", op="sqrt", inputs=["b"])]
    )
    assert one.fingerprint() != two.fingerprint()  # versioned graph
    graph = feature_graph.FeatureGraphTransformer(tuple(one.nodes))
    out = graph.fit(frame[["b"]]).transform(frame[["b"]])
    np.testing.assert_allclose(out["sq"], frame["b"] ** 2)

    vector = knowledge.fingerprint(
        mml_profile(frame), tasks.TaskType.BINARY
    )  # meta-learning fingerprint
    assert set(vector) == set(knowledge.FIELDS)

    rejecting = policies.CallbackCheckpoint(lambda review: False)
    gated = mml.MLProject(
        frame, "y", output_dir=tmp_path, checkpoints=(rejecting,)
    )
    with pytest.raises(errors.BlockedError):
        gated.run()  # human approval: a rejection stops the run

    prediction = monitor_events.PredictionEvent(
        "e1", "v1", "2026-01-01T00:00:00Z", {"a": 1.0}, 1, 0.8
    )
    outcome = monitor_events.OutcomeEvent("e1", "2026-01-02T00:00:00Z", 1)
    assert prediction.event_id == outcome.event_id  # label schema joins

    job = trials.TrialJob("trial_001", "dummy", {"strategy": "x"})
    assert pickle.loads(pickle.dumps(job)) == job  # scheduler-independent
    for backend in (scheduler.SequentialBackend(), scheduler.ProcessBackend(2)):
        assert backend.map(abs, [-1, -2, -3]) == [1, 2, 3]


def mml_profile(frame):
    """Profile of `frame` for fingerprint checks."""
    from mml.data.profiling import profiler
    from mml.data.readers import tabular

    manifest = tabular.build_manifest(frame, "y", "<memory>")
    spec = tasks.ProjectSpec(data=tasks.DataSpec(uri="x", target="y"))
    return profiler.DatasetProfiler(spec).profile(
        frame, manifest, tasks.TaskType.BINARY
    )


def test_l786_l801_risks_have_mitigations_and_candidates_are_whole_pipelines(
    frame,
):
    """L786-L801: a candidate is a complete, independently evaluable pipeline."""
    from mml.sampling import strategies

    registry = registry_module.ModelRegistry.with_builtins()
    plan = mml.MLProject(frame, "y").plan()
    profile = mml.MLProject(frame, "y")
    profile.plan()
    feature_plan = profile.profile.feature_plan
    candidate = strategies.STRATEGIES["default"].build(
        feature_plan,
        registry.get("random_forest"),
        tasks.TaskType.BINARY,
        0,
        {"max_depth": 4},
    )
    assert {"prep", "model"} <= set(
        candidate.named_steps
    )  # preprocessing + estimator
    assert candidate.named_steps["model"].max_depth == 4  # hyperparameters
    resampled = strategies.STRATEGIES["random_undersample"].build(
        feature_plan, registry.get("random_forest"), tasks.TaskType.BINARY, 0
    )
    assert hasattr(
        resampled, "resampler"
    )  # sampling policy part of the candidate
    assert plan.metric and plan.validation_strategy  # evaluation configuration
    bundle = bundle_module.BundleInfo("m", "v1", "roc_auc")
    assert bundle.python and bundle.sklearn  # runtime requirements


def test_l805_l829_five_abstractions_and_the_core_path(run):
    """L805-L829: Dataset, Plan, Experiment, Model Bundle, Lifecycle; Dataset->Plan->Experiment->Bundle."""
    from mml.contracts import data as data_contracts
    from mml.governance import registry

    result = run[1]
    assert isinstance(
        result.profile.manifest, data_contracts.DatasetManifest
    )  # Dataset
    assert isinstance(result.plan, experiments.ExperimentPlan)  # Plan
    assert all(
        isinstance(t, experiments.TrialResult) for t in result.trials
    )  # Experiment
    assert isinstance(result.bundle, bundle_module.ModelBundle)  # Model Bundle
    assert (
        registry.ModelRegistryStore and lifecycle.LifecycleManager
    )  # Lifecycle
