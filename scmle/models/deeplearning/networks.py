"""PyTorch model-family adapters (`deeplearning` extra).

`Network` holds the construction logic once; a network that only differs in
its data (architecture, search space, capabilities) is a configured instance,
such as `MLP`, not a subclass. `Jepa` and `KerasMlp` stay classes because
they override `build` (and `Jepa` also `rejection_reason` and `assemble`),
and `TabPfn` builds different estimators.
"""

from collections.abc import Mapping
import dataclasses
from typing import Any

from sklearn import pipeline

from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import models
from scmle.contracts import search
from scmle.contracts import tasks
from scmle.data.curation import preprocessing
from scmle.models.deeplearning import contracts
from scmle.models.deeplearning import estimators
from scmle.models.deeplearning import foundation
from scmle.models.deeplearning import jepa
from scmle.models.deeplearning import worker

MIN_JEPA_ROWS = 16
COMMON_SPACE = (
    search.FloatParameter("dropout", 0.0, 0.4),
    search.FloatParameter("learning_rate", 1e-4, 1e-2, log=True),
    search.FloatParameter("weight_decay", 1e-6, 1e-2, log=True),
    search.ChoiceParameter("batch_size", ("auto", 64, 128, 256)),
    search.ChoiceParameter("schedule", ("cosine", "constant")),
)


class Network(models.Model):
    """A PyTorch tabular network family.

    Configure it with keyword arguments, or subclass it and declare the same
    names as class attributes. An argument left out keeps the class-level
    value, so both styles work.

    Attributes:
        architecture: Name in `architectures.ARCHITECTURES`.
        family_contract: Modalities the family accepts.
    """

    family = "deep_learning"
    requires: tuple[str, ...] = ("torch",)
    architecture: str
    family_contract: contracts.ModelFamilyContract = (
        contracts.TABULAR_TORCH_CONTRACT
    )
    capabilities = models.Capabilities(
        tasks.STANDARD_TASKS,
        probabilities=True,
        early_stopping=True,
        needs_scaling=True,
        gpu=True,
        max_rows=500_000,
        cost=4,
        memory_factor=4.0,
    )

    def __init__(
        self,
        *,
        name: str | None = None,
        architecture: str | None = None,
        search_space: search.SearchSpace | None = None,
        capabilities: models.Capabilities | None = None,
        capability_family: str | None = None,
        family_contract: contracts.ModelFamilyContract | None = None,
        auto_candidate: bool | None = None,
    ) -> None:
        """Sets the declared data on this instance.

        Args:
            name: Unique registry name.
            architecture: Name in `architectures.ARCHITECTURES`.
            search_space: Tunable hyperparameters.
            capabilities: Declared capabilities.
            capability_family: Release-boundary identity, if stricter than
                the family.
            family_contract: Modalities the family accepts.
            auto_candidate: Whether the adapter enters automatic search.
        """
        given: dict[str, Any] = {
            "name": name,
            "architecture": architecture,
            "search_space": search_space,
            "capabilities": capabilities,
            "capability_family": capability_family,
            "family_contract": family_contract,
            "auto_candidate": auto_candidate,
        }
        for attribute, value in given.items():
            if value is not None:
                setattr(self, attribute, value)

    def gpu_usable(self) -> bool:
        """Whether PyTorch can see a CUDA device (asked of the worker).

        Returns:
            True if the adapter's dependencies are installed and the worker
            reports that CUDA is available, else False.
        """
        return self.is_available() and bool(
            worker.client().request("probe", ())
        )

    def build(
        self,
        task: tasks.Task,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Creates an unfitted network estimator.

        Args:
            task: Resolved task type.
            seed: Random seed.
            params: Search-space overrides.

        Returns:
            A scikit-learn compatible estimator.

        Raises:
            ConfigError: If PyTorch is not installed.
        """
        if not self.is_available():
            raise errors.ConfigError(
                f"{self.name}: install the 'deeplearning' extra (torch)"
            )
        settings = {
            "architecture": self.architecture,
            "random_state": seed,
            **(params or {}),
        }
        if task.is_classification:
            return estimators.TorchClassifier(**settings)
        return estimators.TorchRegressor(**settings)


# Fully connected network.
MLP = Network(
    name="torch_mlp",
    architecture="mlp",
    search_space=search.SearchSpace(
        (
            search.ChoiceParameter(
                "hidden_sizes", ((64,), (128,), (128, 64), (256, 128))
            ),
            *COMMON_SPACE,
        )
    ),
)


# Residual network with layer normalization.
RESIDUAL_MLP = Network(
    name="torch_residual",
    architecture="residual",
    search_space=search.SearchSpace(
        (
            search.ChoiceParameter(
                "hidden_sizes", ((64,), (64, 64), (128, 128), (128, 128, 128))
            ),
            *COMMON_SPACE,
        )
    ),
)


# BatchEnsemble MLP (TabM-style): many cheap members, averaged.
TABM = Network(
    name="torch_tabm",
    architecture="tabm",
    search_space=search.SearchSpace(
        (
            search.ChoiceParameter(
                "hidden_sizes", ((64,), (128, 64), (256, 128))
            ),
            *COMMON_SPACE,
        )
    ),
)


# Experimental tabular FT-Transformer over numeric feature tokens.
#
# `family` stays `deep_learning` for existing reporting and strategy
# behavior. `capability_family` is the stricter release-boundary identity: it
# resolves to the experimental, tabular-only transformer declaration and must
# not inherit the supported generic deep-learning capability.
FT_TRANSFORMER = Network(
    name="torch_ft_transformer",
    architecture="ft_transformer",
    capability_family="transformer",
    family_contract=contracts.TABULAR_TRANSFORMER_CONTRACT,
    auto_candidate=False,
    capabilities=dataclasses.replace(
        Network.capabilities, max_rows=100_000, cost=5, memory_factor=8.0
    ),
    search_space=search.SearchSpace(
        (
            search.ChoiceParameter("hidden_sizes", ((32,), (64,), (64, 64))),
            *COMMON_SPACE,
        )
    ),
)


# MLP with learned scaling, smooth clipping and PReLU (RealMLP-style).
ROBUST_MLP = Network(
    name="torch_realmlp",
    architecture="robust_mlp",
    search_space=search.SearchSpace(
        (
            search.ChoiceParameter(
                "hidden_sizes", ((128,), (256, 128), (256, 256))
            ),
            *COMMON_SPACE,
        )
    ),
)


class Jepa(Network):
    """Experimental tabular JEPA representation plus supervised head.

    The adapter composes the existing worker-backed numeric JEPA estimator
    with a small supervised PyTorch head. It is intentionally a complete
    candidate pipeline so cross-validation, packaging, and prediction all
    use the same representation state. This is tabular-only experimental
    support; it is not a multimodal or pretrained JEPA foundation model.
    """

    name = "torch_jepa"
    family = "deep_learning"
    capability_family = "jepa"
    family_contract = contracts.TABULAR_JEPA_CONTRACT
    auto_candidate = False
    requires: tuple[str, ...] = ("torch",)
    capabilities = dataclasses.replace(
        Network.capabilities,
        max_rows=100_000,
        cost=5,
        memory_factor=8.0,
    )
    search_space = search.SearchSpace(
        (
            search.ChoiceParameter("hidden_size", (32, 64, 128)),
            search.ChoiceParameter("embedding_size", (16, 32, 64)),
            search.FloatParameter("mask_probability", 0.15, 0.5),
            search.FloatParameter("ema_decay", 0.95, 0.999),
            search.FloatParameter("variance_weight", 0.05, 0.3),
            search.FloatParameter("learning_rate", 1e-4, 1e-2, log=True),
            search.FloatParameter("weight_decay", 1e-6, 1e-2, log=True),
            search.ChoiceParameter("epochs", (20, 40, 80)),
            search.ChoiceParameter("batch_size", ("auto", 64, 128)),
        )
    )

    def rejection_reason(
        self, task: tasks.Task, row_count: int | None = None
    ) -> str | None:
        """Rejects datasets too small for representation cross-validation.

        Args:
            task: Resolved task type.
            row_count: Dataset size, if known.

        Returns:
            The base adapter's reason if it has one; otherwise a reason when
            `row_count` is below 16; otherwise None.
        """
        reason = super().rejection_reason(task, row_count)
        if reason is not None:
            return reason
        if row_count is not None and row_count < MIN_JEPA_ROWS:
            return (
                "tabular JEPA requires at least 16 rows for its training "
                "and validation boundaries"
            )
        return None

    def build(
        self,
        task: tasks.Task,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> pipeline.Pipeline:
        """Builds a JEPA representation followed by a prediction head.

        Args:
            task: Resolved binary, multiclass, or regression task.
            seed: Seed shared by the representation and prediction head.
            params: Optional search overrides shared by both steps where
                their parameter names overlap.

        Returns:
            An unfitted sklearn pipeline.

        Raises:
            errors.ConfigError: If PyTorch is not installed.
        """
        if not self.is_available():
            raise errors.ConfigError(
                f"{self.name}: install the 'deeplearning' extra (torch)"
            )
        values = dict(params or {})
        representation_names = {
            "hidden_size",
            "embedding_size",
            "mask_probability",
            "learning_rate",
            "weight_decay",
            "epochs",
            "batch_size",
            "ema_decay",
            "variance_weight",
        }
        head_names = {
            "learning_rate",
            "weight_decay",
            "epochs",
            "batch_size",
        }
        representation_params = {
            name: value
            for name, value in values.items()
            if name in representation_names
        }
        representation_params["random_state"] = seed
        head_params = {
            name: value for name, value in values.items() if name in head_names
        }
        head_params.update({"hidden_sizes": (32,), "random_state": seed})
        head = (
            estimators.TorchClassifier(**head_params)
            if task.is_classification
            else estimators.TorchRegressor(**head_params)
        )
        return pipeline.Pipeline(
            [
                (
                    "representation",
                    jepa.JepaEstimator(**representation_params),
                ),
                ("model", head),
            ]
        )

    def assemble(
        self,
        plan: data.FeaturePlan,
        task: tasks.Task,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> pipeline.Pipeline:
        """Builds the raw-frame pipeline used by project training.

        The shared feature front end turns the declared tabular columns into
        a finite numeric matrix. JEPA then learns only from that matrix, and
        the supervised head consumes its embeddings. Keeping this composition
        in the adapter makes it available to every strategy, package, and
        prediction path without a special-case orchestrator branch.

        Args:
            plan: Feature plan describing the declared columns.
            task: Resolved task type.
            seed: Random seed.
            params: Optional search overrides, as for `build`.

        Returns:
            An unfitted pipeline: the feature front end followed by the JEPA
            representation and prediction head.
        """
        options = preprocessing.default_options(plan, self)
        front_end = preprocessing.build_front_end(plan, options)
        representation_pipeline = self.build(task, seed, params)
        return pipeline.Pipeline(
            [("prep", front_end), *representation_pipeline.steps]
        )


class KerasMlp(Network):
    """Keras dense network, run on the PyTorch backend in the same worker."""

    name = "keras_mlp"
    requires = ("torch", "keras")
    search_space = search.SearchSpace(
        (
            search.ChoiceParameter(
                "hidden_sizes", ((64,), (128,), (128, 64), (256, 128))
            ),
            search.FloatParameter("dropout", 0.0, 0.4),
            search.FloatParameter("learning_rate", 1e-4, 1e-2, log=True),
            search.FloatParameter("l2", 1e-6, 1e-2, log=True),
            search.ChoiceParameter("batch_size", (64, 128, 256)),
        )
    )

    def build(
        self,
        task: tasks.Task,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Creates an unfitted Keras estimator.

        Args:
            task: Resolved task type.
            seed: Random seed.
            params: Search-space overrides passed to the estimator.

        Returns:
            An unfitted Keras classifier or regressor.

        Raises:
            ConfigError: If PyTorch or Keras is not installed.
        """
        if not self.is_available():
            raise errors.ConfigError(
                f"{self.name}: install the 'deeplearning' extra (torch, keras)"
            )
        settings = {"random_state": seed, **(params or {})}
        if task.is_classification:
            return estimators.KerasClassifier(**settings)
        return estimators.KerasRegressor(**settings)


class TabPfn(models.Model):
    """Pretrained tabular foundation model (`tabpfn` extra, license required).

    Needs the `tabpfn` package and an accepted Prior Labs license for the
    weights (`TABPFN_TOKEN`). Without them its trials fail with the library's
    own message and the rest of the run continues.
    """

    name = "tabpfn"
    family = "foundation"
    requires = ("tabpfn",)
    capabilities = models.Capabilities(
        tasks.STANDARD_TASKS,
        probabilities=True,
        gpu=True,
        max_rows=foundation.MAX_ROWS,
        cost=5,
        memory_factor=6.0,
    )

    def rejection_reason(
        self, task: tasks.Task, row_count: int | None = None
    ) -> str | None:
        """Keeps TabPFN outside the automatic beta candidate set.

        TabPFN requires an accepted third-party model-weight license, token
        custody, and evidence that is not part of the current tabular beta
        support boundary. It remains available through its explicit adapter
        and foundation-engine contract, but installing the package alone
        must not make ordinary projects depend on that external agreement.

        Args:
            task: Resolved task type.
            row_count: Dataset size, if known.

        Returns:
            The base adapter's reason if it has one; otherwise always the
            extension-only reason, so it is never an automatic candidate.
        """
        missing = super().rejection_reason(task, row_count)
        if missing is not None:
            return missing
        return (
            "foundation model is extension-only until accepted weight "
            "license, token custody, and complete release evidence exist"
        )

    def gpu_usable(self) -> bool:
        """Whether PyTorch can see a CUDA device (asked of the worker).

        Returns:
            True if the adapter's dependencies are installed and the worker
            reports that CUDA is available, else False.
        """
        return self.is_available() and bool(
            worker.client().request("probe", ())
        )

    def build(
        self,
        task: tasks.Task,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Creates an unfitted TabPFN estimator.

        Args:
            task: Resolved task type.
            seed: Random seed.
            params: Search-space overrides passed to the estimator.

        Returns:
            An unfitted TabPFN classifier or regressor.

        Raises:
            ConfigError: If the `tabpfn` package is not installed.
        """
        if not self.is_available():
            raise errors.ConfigError(
                f"{self.name}: install the 'tabpfn' extra (tabpfn)"
            )
        settings = {"random_state": seed, **(params or {})}
        if task.is_classification:
            return estimators.TabPfnClassifier(**settings)
        return estimators.TabPfnRegressor(**settings)


DEEP_LEARNING_ADAPTERS: tuple[models.Model, ...] = (
    TabPfn(),
    MLP,
    RESIDUAL_MLP,
    TABM,
    FT_TRANSFORMER,
    Jepa(),
    ROBUST_MLP,
    KerasMlp(),
)
