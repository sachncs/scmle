"""PyTorch model-family adapters (`deeplearning` extra)."""

from collections.abc import Mapping
import dataclasses
from typing import Any
from typing import ClassVar

from sklearn import pipeline

from scmle.contracts import data as data_contracts
from scmle.contracts import errors
from scmle.contracts import models
from scmle.contracts import search
from scmle.contracts import tasks
from scmle.models.deeplearning import contracts
from scmle.models.deeplearning import estimators
from scmle.models.deeplearning import foundation
from scmle.models.deeplearning import jepa
from scmle.models.deeplearning import worker

ALL_TASKS = tasks.STANDARD_TASKS
COMMON_SPACE = (
    search.FloatParameter("dropout", 0.0, 0.4),
    search.FloatParameter("learning_rate", 1e-4, 1e-2, log=True),
    search.FloatParameter("weight_decay", 1e-6, 1e-2, log=True),
    search.ChoiceParameter("batch_size", ("auto", 64, 128, 256)),
    search.ChoiceParameter("schedule", ("cosine", "constant")),
)


class TorchAdapter(models.ModelAdapter):
    """A PyTorch tabular network family.

    Attributes:
        architecture: Name in `architectures.ARCHITECTURES`.
    """

    family = "deep_learning"
    requires: ClassVar[tuple[str, ...]] = ("torch",)
    architecture: ClassVar[str]
    family_contract: ClassVar[contracts.ModelFamilyContract] = (
        contracts.TABULAR_TORCH_CONTRACT
    )
    capabilities = models.Capabilities(
        ALL_TASKS,
        probabilities=True,
        early_stopping=True,
        needs_scaling=True,
        gpu=True,
        max_rows=500_000,
        cost=4,
        memory_factor=4.0,
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

    def supports_modality(self, modality: str) -> bool:
        """Returns whether this adapter has a declared modality boundary.

        Args:
            modality: A modality name or `Modality` value.

        Returns:
            True if the family contract declares the modality.
        """
        return self.family_contract.supports_modality(modality)

    def require_modality(self, modality: str) -> None:
        """Fail closed when an adapter is asked to model another modality.

        The family contract raises ValueError if it does not declare the
        modality.

        Args:
            modality: A modality name or `Modality` value.
        """
        self.family_contract.require_modality(modality)

    def build(
        self,
        task: tasks.TaskType,
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


class TorchMlpAdapter(TorchAdapter):
    """Fully connected network."""

    name = "torch_mlp"
    architecture = "mlp"
    search_space = search.SearchSpace(
        (
            search.ChoiceParameter(
                "hidden_sizes", ((64,), (128,), (128, 64), (256, 128))
            ),
            *COMMON_SPACE,
        )
    )


class TorchResidualAdapter(TorchAdapter):
    """Residual network with layer normalization."""

    name = "torch_residual"
    architecture = "residual"
    search_space = search.SearchSpace(
        (
            search.ChoiceParameter(
                "hidden_sizes", ((64,), (64, 64), (128, 128), (128, 128, 128))
            ),
            *COMMON_SPACE,
        )
    )


class TorchEnsembleAdapter(TorchAdapter):
    """BatchEnsemble MLP (TabM-style): many cheap members, averaged."""

    name = "torch_tabm"
    architecture = "tabm"
    search_space = search.SearchSpace(
        (
            search.ChoiceParameter(
                "hidden_sizes", ((64,), (128, 64), (256, 128))
            ),
            *COMMON_SPACE,
        )
    )


class TorchTransformerAdapter(TorchAdapter):
    """Experimental tabular FT-Transformer over numeric feature tokens.

    ``family`` stays ``deep_learning`` for existing reporting and strategy
    behavior. ``capability_family`` is the stricter release-boundary identity:
    it resolves to the experimental, tabular-only transformer declaration and
    must not inherit the supported generic deep-learning capability.
    """

    name = "torch_ft_transformer"
    architecture = "ft_transformer"
    capability_family = "transformer"
    family_contract = contracts.TABULAR_TRANSFORMER_CONTRACT
    auto_candidate = False
    capabilities = dataclasses.replace(
        TorchAdapter.capabilities, max_rows=100_000, cost=5, memory_factor=8.0
    )

    search_space = search.SearchSpace(
        (
            search.ChoiceParameter("hidden_sizes", ((32,), (64,), (64, 64))),
            *COMMON_SPACE,
        )
    )


class TorchJepaAdapter(TorchAdapter):
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
    requires: ClassVar[tuple[str, ...]] = ("torch",)
    capabilities = dataclasses.replace(
        TorchAdapter.capabilities,
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
        self, task: tasks.TaskType, row_count: int | None = None
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
        if row_count is not None and row_count < 16:
            return (
                "tabular JEPA requires at least 16 rows for its training "
                "and validation boundaries"
            )
        return None

    def build(
        self,
        task: tasks.TaskType,
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
        plan: data_contracts.FeaturePlan,
        task: tasks.TaskType,
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
        from scmle.data.curation import preprocessing

        options = preprocessing.default_options(plan, self)
        front_end = preprocessing.build_front_end(plan, options)
        representation_pipeline = self.build(task, seed, params)
        return pipeline.Pipeline(
            [("prep", front_end), *representation_pipeline.steps]
        )


class TorchRobustMlpAdapter(TorchAdapter):
    """MLP with learned scaling, smooth clipping and PReLU (RealMLP-style)."""

    name = "torch_realmlp"
    architecture = "robust_mlp"
    search_space = search.SearchSpace(
        (
            search.ChoiceParameter(
                "hidden_sizes", ((128,), (256, 128), (256, 256))
            ),
            *COMMON_SPACE,
        )
    )


class KerasMlpAdapter(TorchAdapter):
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
        task: tasks.TaskType,
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


class TabPfnAdapter(models.ModelAdapter):
    """Pretrained tabular foundation model (`tabpfn` extra, license required).

    Needs the `tabpfn` package and an accepted Prior Labs license for the
    weights (`TABPFN_TOKEN`). Without them its trials fail with the library's
    own message and the rest of the run continues.
    """

    name = "tabpfn"
    family = "foundation"
    requires = ("tabpfn",)
    capabilities = models.Capabilities(
        ALL_TASKS,
        probabilities=True,
        gpu=True,
        max_rows=foundation.MAX_ROWS,
        cost=5,
        memory_factor=6.0,
    )

    def rejection_reason(
        self, task: tasks.TaskType, row_count: int | None = None
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
        task: tasks.TaskType,
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


DEEP_LEARNING_ADAPTERS: tuple[type[models.ModelAdapter], ...] = (
    TabPfnAdapter,
    TorchMlpAdapter,
    TorchResidualAdapter,
    TorchEnsembleAdapter,
    TorchTransformerAdapter,
    TorchJepaAdapter,
    TorchRobustMlpAdapter,
    KerasMlpAdapter,
)
