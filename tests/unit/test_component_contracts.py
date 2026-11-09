"""Contract tests: every pluggable component honours its declared API."""

import abc
import inspect
import subprocess
import sys
import textwrap

import pytest

from scmle.contracts import models
import scmle.contracts.experiments
from scmle.core import policies
from scmle.core import scheduler
from scmle.core import stages
from scmle.data.readers import tabular
from scmle.data.splitting import splitters
from scmle.data.validation import columns
from scmle.evaluation import diagnostics
from scmle.evaluation import selection
from scmle.experimentation import stores
from scmle.features import feature_graph
from scmle.inference import serving
from scmle.intelligence import planner
from scmle.models import registry
from scmle.models import thresholds
from scmle.monitoring import monitor
from scmle.optimization import optimizers
from scmle.sampling import resampling
from scmle.sampling import strategies
from scmle.sampling import synthesis


def concrete_subclasses(base, *modules):
    found = []
    for module in modules:
        for member in dict(
            inspect.getmembers(module, inspect.isclass)
        ).values():
            if (
                issubclass(member, base)
                and member is not base
                and member.__module__ == module.__name__
            ):
                found.append(member)
    return found


@pytest.mark.parametrize(
    ("base", "modules"),
    [
        (tabular.TableReader, [tabular]),
        (columns.ColumnKind, [columns]),
        (splitters.Partitioner, [splitters]),
        (strategies.Strategy, [strategies]),
        (diagnostics.Diagnostic, [diagnostics]),
        (planner.Refiner, [planner]),
        (selection.Selector, [selection]),
        (stages.Stage, [stages]),
        (policies.Reviewer, [policies]),
        (resampling.Resampler, [resampling]),
        (resampling.IndexResampler, [resampling]),
        (feature_graph.FeatureOp, [feature_graph]),
        (synthesis.SyntheticGenerator, [synthesis]),
        (thresholds.Objective, [thresholds]),
        (monitor.Advisor, [monitor]),
        (scheduler.Executor, [scheduler]),
        (optimizers.Optimizer, [optimizers]),
        (stores.ExperimentStore, [stores]),
        (serving.Server, [serving]),
    ],
    ids=lambda value: getattr(value, "__name__", ""),
)
def test_every_concrete_component_implements_its_abstract_interface(
    base, modules
):
    implementations = concrete_subclasses(base, *modules)
    assert implementations, f"no implementations of {base.__name__} found"
    concrete = [cls for cls in implementations if not inspect.isabstract(cls)]
    assert concrete, f"{base.__name__} has no concrete implementation"
    for cls in implementations:
        if inspect.isabstract(
            cls
        ):  # An intermediate base: needs a concrete heir.
            assert any(issubclass(other, cls) for other in concrete), (
                cls.__name__
            )


def test_every_model_adapter_declares_complete_metadata():
    adapters = registry.Catalog.with_builtins(include_user=False).adapters
    assert len(adapters) >= 40
    names = set()
    for adapter in adapters.values():
        assert not inspect.isabstract(type(adapter))
        assert adapter.name and adapter.family and adapter.capabilities.tasks
        assert adapter.name not in names
        names.add(adapter.name)


def test_strategy_registry_entries_are_self_describing():
    for name, strategy in strategies.STRATEGIES.items():
        assert strategy.name == name and strategy.description
    assert set(strategies.DEFAULT_NAMES) | strategies.OPT_IN | {
        scmle.contracts.experiments.DEFAULT_STRATEGY
    } == set(strategies.STRATEGIES)


def test_one_stage_per_lifecycle_state_in_order():
    states = [stage.state for stage in stages.default_stages()]
    assert len(states) == len(set(states)) == 7


def test_abstract_bases_cannot_be_instantiated():
    for base in (
        tabular.TableReader,
        columns.ColumnKind,
        splitters.Partitioner,
        strategies.Strategy,
        resampling.IndexResampler,
        models.Model,
        stores.ExperimentStore,
    ):
        assert inspect.isabstract(base)
        assert isinstance(base, abc.ABCMeta)
        with pytest.raises(TypeError):
            base()


def run_isolated(code):
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_registering_and_planning_never_imports_optional_libraries():
    """Spec: a registry entry must not import GPU / heavy dependencies."""
    result = run_isolated(
        """
        import sys
        import scmle
        from scmle.contracts import tasks
        from scmle.models import registry
        reg = registry.Catalog.with_builtins()
        reg.candidates(tasks.Task.BINARY, 1000)
        heavy = ["lightgbm", "xgboost", "catboost", "optuna", "imblearn",
                 "mlflow", "ray", "torch", "tensorflow"]
        loaded = [name for name in heavy if name in sys.modules]
        assert not loaded, loaded
        """
    )
    assert result.returncode == 0, result.stderr


def test_a_model_is_imported_only_when_it_is_built():
    pytest.importorskip("lightgbm")
    result = run_isolated(
        """
        import sys
        from scmle.contracts import tasks
        from scmle.models import registry
        reg = registry.Catalog.with_builtins()
        assert "lightgbm" not in sys.modules
        reg.get("lightgbm").build(tasks.Task.BINARY, 0)
        assert "lightgbm" in sys.modules
        """
    )
    assert result.returncode == 0, result.stderr


def test_the_workflow_is_a_compiled_dag_of_declared_dependencies():
    from scmle.contracts import errors
    from scmle.core import compiler

    default = stages.default_stages()
    kinds = [type(s) for s in default]
    assert kinds[0] is stages.IngestStage and kinds[-1] is stages.PackageStage
    shuffled = list(reversed(default))
    assert [type(s) for s in compiler.compile_workflow(shuffled)] == kinds
    with pytest.raises(errors.ConfigError, match="requires"):
        compiler.compile_workflow(default[1:])
    with pytest.raises(errors.ConfigError, match="more than once"):
        compiler.compile_workflow([*default, default[0]])

    class Left:
        requires = ()

    class Right:
        requires = (Left,)

    Left.requires = (Right,)
    with pytest.raises(errors.ConfigError, match="cycle"):
        compiler.compile_workflow([Left(), Right()])


def test_a_replacement_stage_subclass_satisfies_dependencies():
    from scmle.core import compiler

    class Custom(stages.TrainStage):
        pass

    swapped = [
        Custom() if isinstance(s, stages.TrainStage) else s
        for s in stages.default_stages()
    ]
    ordered = compiler.compile_workflow(swapped)
    assert [type(s) for s in ordered].count(Custom) == 1
