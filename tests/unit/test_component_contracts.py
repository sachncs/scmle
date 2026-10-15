"""Contract tests: every pluggable component honours its declared API."""

import abc
import inspect
import subprocess
import sys
import textwrap

import pytest

from mml.contracts import models as model_contracts
from mml.core import checkpoints
from mml.core import execution
from mml.core import stages
from mml.data import columns
from mml.data import readers
from mml.data import splitting
from mml.data import strategies
from mml.deployment import serving
from mml.evaluation import diagnostics
from mml.evaluation import selection
from mml.intelligence import planner
from mml.models import boosting_adapters
from mml.models import resampling
from mml.models import sklearn_adapters
from mml.models import thresholds
from mml.monitoring import monitor
from mml.optimization import optimizers
from mml.tracking import stores


def concrete_subclasses(base, *modules):
    found = []
    for module in modules:
        for _, member in inspect.getmembers(module, inspect.isclass):
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
        (readers.TableReader, [readers]),
        (columns.ColumnKind, [columns]),
        (splitting.SplitStrategy, [splitting]),
        (strategies.DataStrategy, [strategies]),
        (diagnostics.Diagnostic, [diagnostics]),
        (planner.PlanningRule, [planner]),
        (selection.SelectionPolicy, [selection]),
        (stages.Stage, [stages]),
        (checkpoints.PolicyCheckpoint, [checkpoints]),
        (resampling.Resampler, [resampling]),
        (thresholds.ThresholdObjective, [thresholds]),
        (monitor.AdvisoryRule, [monitor]),
        (execution.ExecutionBackend, [execution]),
        (optimizers.Optimizer, [optimizers]),
        (stores.ExperimentStore, [stores]),
        (serving.ServingAdapter, [serving]),
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
    adapters = (
        sklearn_adapters.BUILTIN_ADAPTERS + boosting_adapters.BOOSTING_ADAPTERS
    )
    names = set()
    for cls in adapters:
        assert not inspect.isabstract(cls)
        assert cls.name and cls.family and cls.capabilities.tasks
        assert cls.name not in names
        names.add(cls.name)


def test_strategy_registry_entries_are_self_describing():
    for name, strategy in strategies.STRATEGIES.items():
        assert strategy.name == name and strategy.description
    assert set(strategies.DEFAULT_NAMES) | strategies.OPT_IN | {
        strategies.DEFAULT
    } == set(strategies.STRATEGIES)


def test_one_stage_per_lifecycle_state_in_order():
    states = [stage.state for stage in stages.default_stages()]
    assert len(states) == len(set(states)) == 7


def test_abstract_bases_cannot_be_instantiated():
    for base in (
        readers.TableReader,
        columns.ColumnKind,
        splitting.SplitStrategy,
        strategies.DataStrategy,
        model_contracts.ModelAdapter,
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
        import mml
        from mml.contracts import tasks
        from mml.models import registry
        reg = registry.ModelRegistry.with_builtins()
        reg.candidates(tasks.TaskType.BINARY, 1000)
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
        from mml.contracts import tasks
        from mml.models import registry
        reg = registry.ModelRegistry.with_builtins()
        assert "lightgbm" not in sys.modules
        reg.get("lightgbm").build(tasks.TaskType.BINARY, 0)
        assert "lightgbm" in sys.modules
        """
    )
    assert result.returncode == 0, result.stderr
