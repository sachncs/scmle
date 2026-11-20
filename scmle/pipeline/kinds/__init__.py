"""Data kinds: what the shared stages need to know about one kind of data.

A `Kind` is the one seam between the shared stages and a kind of data. It
reads the declared source (`read`), reports what profiling finds in the
development rows (`survey`), states the split constraints as the splitter
that keeps related rows on one side (`constrain`), builds the featuriser
that turns rows into model inputs (`featurise`), names what a served model's
outputs decode to (`decode`) and bounds the seconds featurising and
profiling take (`cost`). Ingestion reads through it, profiling surveys and
splits through it, and packaging decodes through it.

Each kind also declares its own capability records (`declare`), which
`roster.enrol` enrols by iterating `KINDS`. `Tabular` is the first kind;
the others become table rows (`rows.Rows`, and `media.Media` for files).
A new kind is a class in `kinds/<name>.py` with these seven methods, added
to `KINDS` under its modality name; `pick` resolves the kind a data block
declares and refuses one that has none.
"""

from collections.abc import Mapping
from typing import Any
from typing import Protocol
from typing import runtime_checkable

import pandas

from scmle.contracts import capabilities
from scmle.contracts import tasks
import scmle.contracts.data
from scmle.data.readers import streaming
from scmle.data.readers import tabular
from scmle.data.splitting import splitters
from scmle.pipeline import profiler
from scmle.pipeline.kinds import audio
from scmle.pipeline.kinds import document
from scmle.pipeline.kinds import geospatial
from scmle.pipeline.kinds import graph
from scmle.pipeline.kinds import image
from scmle.pipeline.kinds import raster
from scmle.pipeline.kinds import video
from scmle.preparation import preprocessing

GIGABYTE = 1e9
"""Bytes in the gigabyte `memory_budget_gb` is stated in."""

TABULAR_BETA_TASK_FAMILIES = (
    (
        tasks.Task.BINARY,
        (
            "baseline",
            "linear",
            "tree",
            "boosting",
            "kernel",
            "neighbors",
            "neural",
            "bagging",
            "deep_learning",
        ),
    ),
    (
        tasks.Task.MULTICLASS,
        (
            "baseline",
            "linear",
            "tree",
            "boosting",
            "kernel",
            "neighbors",
            "neural",
            "bagging",
            "deep_learning",
        ),
    ),
    (
        tasks.Task.REGRESSION,
        (
            "baseline",
            "linear",
            "tree",
            "boosting",
            "kernel",
            "neighbors",
            "neural",
            "bagging",
            "deep_learning",
        ),
    ),
    (tasks.Task.SURVIVAL, ("baseline", "linear", "boosting")),
    (tasks.Task.ORDINAL, ("baseline", "linear", "tree", "boosting")),
    (tasks.Task.RANKING, ("baseline", "linear", "boosting")),
    (
        tasks.Task.MULTILABEL,
        ("baseline", "linear", "tree", "boosting"),
    ),
    (
        tasks.Task.MULTIOUTPUT,
        ("baseline", "linear", "tree", "boosting"),
    ),
    (tasks.Task.FORECAST, ("baseline", "linear", "tree", "boosting")),
    (tasks.Task.UPLIFT, ("baseline", "linear", "causal")),
)
"""Explicit tabular model-family paths covered by the current beta."""


TABULAR_BETA_EVALUATIONS = {
    tasks.Task.BINARY: capabilities.Evaluation.STANDARD,
    tasks.Task.MULTICLASS: capabilities.Evaluation.STANDARD,
    tasks.Task.REGRESSION: capabilities.Evaluation.STANDARD,
    tasks.Task.SURVIVAL: capabilities.Evaluation.SURVIVAL,
    tasks.Task.ORDINAL: capabilities.Evaluation.STANDARD,
    tasks.Task.RANKING: capabilities.Evaluation.RANKING,
    tasks.Task.MULTILABEL: capabilities.Evaluation.MULTILABEL,
    tasks.Task.MULTIOUTPUT: capabilities.Evaluation.MULTIOUTPUT,
    tasks.Task.FORECAST: capabilities.Evaluation.FORECASTING,
    tasks.Task.UPLIFT: capabilities.Evaluation.UPLIFT,
}
"""Canonical evaluation contract for each tabular task path."""


def fairnesses(task: tasks.Task) -> tuple[capabilities.Fairness, ...]:
    """Lists the fairness declarations a tabular task path is registered with.

    Args:
        task: The task type.

    Returns:
        No fairness constraint, plus mitigation for binary, multiclass and
        regression tasks.
    """
    if task in (
        tasks.Task.BINARY,
        tasks.Task.MULTICLASS,
        tasks.Task.REGRESSION,
    ):
        return (
            capabilities.Fairness.NONE,
            capabilities.Fairness.MITIGATION,
        )
    return (capabilities.Fairness.NONE,)


def advertise(
    task: tasks.Task,
    model_family: str,
    fairness: capabilities.Fairness,
) -> capabilities.Capability:
    """Builds the supported library-inference capability of one tabular path.

    Args:
        task: The task type.
        model_family: The model family name.
        fairness: The fairness declaration.

    Returns:
        The supported capability, with its failure modes, evidence link and
        limitations.
    """
    return capabilities.Capability(
        key=capabilities.Coordinate(
            modality=capabilities.Modality.TABULAR,
            task=task,
            family=model_family,
            evaluation=TABULAR_BETA_EVALUATIONS[task],
            uncertainty=capabilities.Uncertainty.NONE,
            fairness=fairness,
            explainability=capabilities.Explainability.NONE,
            inference=capabilities.Inference.LIBRARY,
        ),
        status=capabilities.Status.SUPPORTED,
        extras=(
            ("deeplearning",) if model_family == "deep_learning" else ("core",)
        ),
        resources=capabilities.Resources(memory=1.0),
        modes=(
            "malformed or schema-incompatible input",
            "model-family dependency unavailable",
            "resource limits exceeded",
        ),
        links=("tests/integration/test_capability_gate.py",),
        limitations=(
            "tabular modality only",
            "library inference only",
            "no portable explanation contract",
            "no predictive uncertainty contract",
        ),
    )


def experimental_tabular_capability(
    task: tasks.Task,
    model_family: str,
    fairness: capabilities.Fairness,
    failure_modes: tuple[str, ...],
    evidence_links: tuple[str, ...],
    limitations: tuple[str, ...],
) -> capabilities.Capability:
    """Builds the experimental capability of one deep tabular path.

    Args:
        task: The task type.
        model_family: The model family name.
        fairness: The fairness declaration.
        failure_modes: Known failures of the path.
        evidence_links: Tests that evidence the path.
        limitations: User-visible boundaries of the path.

    Returns:
        The experimental capability, which needs the deeplearning extra, 4 GB
        of memory and at most 100,000 rows.
    """
    return capabilities.Capability(
        key=capabilities.Coordinate(
            modality=capabilities.Modality.TABULAR,
            task=task,
            family=model_family,
            evaluation=capabilities.Evaluation.STANDARD,
            uncertainty=capabilities.Uncertainty.NONE,
            fairness=fairness,
            explainability=capabilities.Explainability.NONE,
            inference=capabilities.Inference.LIBRARY,
        ),
        status=capabilities.Status.EXPERIMENTAL,
        extras=("deeplearning",),
        resources=capabilities.Resources(
            memory=4.0,
            rows=100_000,
        ),
        modes=failure_modes,
        links=evidence_links,
        limitations=limitations,
    )


EXPERIMENTAL = (
    (
        capabilities.Family.TRANSFORMER.value,
        (
            ("torch runtime or tabular transformer adapter is unavailable"),
            "numeric feature-token training or evaluation fails",
            ("resource limits or experimental evidence are incomplete"),
        ),
        (
            "tests/integration/test_capability_gate.py",
            "tests/unit/test_deeplearning.py",
        ),
        (
            "experimental tabular FT-Transformer only",
            "requires explicit models.experimental_models opt-in",
            ("not a text, image, audio, video, or multimodal transformer"),
            "no production support guarantee",
        ),
    ),
    (
        capabilities.Family.JEPA.value,
        (
            "torch runtime or tabular JEPA adapter is unavailable",
            (
                "numeric representation training or downstream "
                "head evaluation fails"
            ),
            ("resource limits or experimental evidence are incomplete"),
        ),
        (
            "tests/integration/test_capability_gate.py",
            "tests/integration/test_experimental_models.py",
            "tests/unit/test_jepa_contract.py",
        ),
        (
            (
                "experimental tabular numeric representation plus "
                "supervised head only"
            ),
            "requires explicit models.experimental_models opt-in",
            ("not a text, image, audio, video, graph, or multimodal JEPA"),
            "no production support guarantee",
        ),
    ),
)
"""The experimental tabular families: name, failure modes, evidence links and
limitations."""


@runtime_checkable
class Kind(Protocol):
    """What the shared stages need of one kind of data.

    Attributes:
        name: The modality the kind serves, as a data block declares it.
    """

    name: str

    def read(
        self, spec: tasks.Config
    ) -> tuple[
        pandas.DataFrame, tuple[streaming.Feed, streaming.Totals] | None
    ]:
        """Reads the declared source.

        Args:
            spec: The run's specification; its data block names the source.

        Returns:
            The rows, and the chunked source with its totals when the rows
            are a sample kept from one streaming pass, else None.
        """
        ...

    def survey(
        self,
        spec: tasks.Config,
        frame: pandas.DataFrame,
        manifest: scmle.contracts.data.Manifest,
        task: tasks.Task,
        seconds: float = 0.0,
        total: int | None = None,
    ) -> scmle.contracts.data.Portrait:
        """Profiles rows into findings and a feature plan.

        Args:
            spec: The run's specification.
            frame: The rows to profile (the development rows).
            manifest: The identity of the whole dataset.
            task: The resolved task.
            seconds: Seconds the read took.
            total: Rows in the whole dataset when `frame` is a subset.

        Returns:
            The portrait: findings, statistics and the feature plan.
        """
        ...

    def constrain(
        self, spec: tasks.Config, task: tasks.Task
    ) -> splitters.Splitter:
        """States the split constraints of the data.

        Args:
            spec: The run's specification.
            task: The resolved task.

        Returns:
            The splitter that keeps the rows the constraints tie together
            on one side of every split.
        """
        ...

    def featurise(
        self,
        plan: scmle.contracts.data.Blueprint,
        options: scmle.contracts.data.Preprocessing,
    ) -> Any:
        """Builds the unfitted featuriser of a feature plan.

        Args:
            plan: The feature plan.
            options: The preprocessing switches.

        Returns:
            A scikit-learn transformer from rows to model inputs.
        """
        ...

    def decode(self, labels: tuple[Any, ...] | None) -> tuple[Any, ...] | None:
        """Names the labels a served model's outputs decode to.

        Args:
            labels: The labels the run recorded, lowest first (ordinal
                tasks), or None.

        Returns:
            The labels per output column, or None when the model's own
            classes name them.
        """
        ...

    def cost(
        self, rows: int, columns: int, classes: int, budget: float
    ) -> float:
        """Bounds the seconds featurising and profiling take.

        Args:
            rows: Development rows.
            columns: Feature columns.
            classes: Classes of the target; below two means regression.
            budget: The run's time budget in seconds.

        Returns:
            An upper bound on the seconds.
        """
        ...

    def declare(self) -> tuple[capabilities.Capability, ...]:
        """Declares the kind's capability records.

        Returns:
            Every record the kind runs with, which `roster.enrol` enrols.
        """
        ...


class Tabular:
    """Rows and columns read from a table file or given in memory.

    Attributes:
        name: "tabular".
    """

    name = "tabular"

    @staticmethod
    def overflows(data: tasks.Dataset) -> bool:
        """Whether the file is bigger than the memory budget.

        Args:
            data: The data block, giving the source and the optional memory
                budget in gigabytes.

        Returns:
            True if a budget is set and the source's estimated size exceeds
            it; False if no budget is set.
        """
        if data.memory_budget_gb is None:
            return False
        size = streaming.locate(data.uri).footprint()
        return size > data.memory_budget_gb * GIGABYTE

    def read(
        self, spec: tasks.Config
    ) -> tuple[
        pandas.DataFrame, tuple[streaming.Feed, streaming.Totals] | None
    ]:
        """Reads the table whole, or in one pass when it overflows.

        A streamed pass assigns each row to the locked or the development
        side by a hash of its position and keeps a capped sample of each.

        Args:
            spec: The run's specification.

        Returns:
            The rows (with a `split_side` column when streamed) and the
            stream, or None when the file was read whole.
        """
        data = spec.data
        if not self.overflows(data):
            return tabular.read(data.uri), None
        source = streaming.locate(data.uri)
        development, locked = streaming.budgets(
            source.shape()[1],
            data.memory_budget_gb or 1.0,
            spec.validation.test_fraction,
        )
        frame, totals = streaming.sweep(
            source,
            development,
            locked,
            spec.validation.test_fraction,
            spec.validation.seed,
        )
        return frame, (source, totals)

    def survey(
        self,
        spec: tasks.Config,
        frame: pandas.DataFrame,
        manifest: scmle.contracts.data.Manifest,
        task: tasks.Task,
        seconds: float = 0.0,
        total: int | None = None,
    ) -> scmle.contracts.data.Portrait:
        """Profiles the rows with the column profiler.

        Args:
            spec: The run's specification.
            frame: The rows to profile.
            manifest: The identity of the whole dataset.
            task: The resolved task.
            seconds: Seconds the read took.
            total: Rows in the whole dataset when `frame` is a subset.

        Returns:
            The profiler's portrait (`profiler.Profiler.profile`).
        """
        return profiler.Profiler(spec).profile(
            frame, manifest, task, seconds, total
        )

    def constrain(
        self, spec: tasks.Config, task: tasks.Task
    ) -> splitters.Splitter:
        """The splitter of the declared protocol (groups, time, strata).

        Args:
            spec: The run's specification.
            task: The resolved task.

        Returns:
            The splitter.
        """
        return splitters.Splitter(spec, task)

    def featurise(
        self,
        plan: scmle.contracts.data.Blueprint,
        options: scmle.contracts.data.Preprocessing,
    ) -> Any:
        """The column transformer of the feature plan.

        Args:
            plan: The feature plan.
            options: The preprocessing switches.

        Returns:
            The unfitted column transformer
            (`preprocessing.build`).
        """
        return preprocessing.build(plan, options)

    def decode(self, labels: tuple[Any, ...] | None) -> tuple[Any, ...] | None:
        """Ordinal ranks decode to the recorded labels; else the classes.

        Args:
            labels: The ordinal labels, lowest first, or None.

        Returns:
            `labels`.
        """
        return labels

    def cost(
        self, rows: int, columns: int, classes: int, budget: float
    ) -> float:
        """The profiler's bound on the seconds of data preparation.

        Args:
            rows: Development rows.
            columns: Feature columns.
            classes: Classes of the target.
            budget: The run's time budget in seconds.

        Returns:
            `profiler.estimate`.
        """
        return profiler.estimate(rows, columns, classes, budget)

    def declare(self) -> tuple[capabilities.Capability, ...]:
        """The supported and experimental tabular capabilities.

        Only the library inference path is declared, with no portable
        explanation or predictive-uncertainty promise.

        Returns:
            A supported record per task, family and fairness of
            `TABULAR_BETA_TASK_FAMILIES` (`advertise`), and the experimental
            records of `EXPERIMENTAL`, each with its failure modes, evidence
            links and limitations.
        """
        supported = (
            advertise(task, family, fairness)
            for task, families in TABULAR_BETA_TASK_FAMILIES
            for family in families
            for fairness in fairnesses(task)
        )
        experimental = (
            experimental_tabular_capability(
                task, family, fairness, failures, evidence, limitations
            )
            for family, failures, evidence, limitations in EXPERIMENTAL
            for task in (
                tasks.Task.BINARY,
                tasks.Task.MULTICLASS,
                tasks.Task.REGRESSION,
            )
            for fairness in fairnesses(task)
        )
        return (*supported, *experimental)


TABULAR = Tabular()
"""The tabular kind; rows given in memory are read as a table."""

KINDS: Mapping[str, Kind] = {
    kind.name: kind
    for kind in (
        TABULAR,
        document.DOCUMENT,
        graph.GRAPH,
        image.IMAGE,
        video.VIDEO,
        audio.AUDIO,
        raster.RASTER,
        geospatial.GEOSPATIAL,
    )
}
"""Every registered kind, by the modality name a data block declares."""


def pick(data: tasks.Dataset) -> Kind:
    """The kind of a declared data source.

    Args:
        data: The data block.

    Returns:
        The kind registered for its modality.

    Raises:
        UnsupportedError: If no kind is registered for the modality.
    """
    found = KINDS.get(data.modality)
    if found is None:
        raise capabilities.UnsupportedError(
            "file-backed modality has no registered ingestion adapter",
            {
                "status": "unsupported",
                "reason": "file-backed modality has no ingestion adapter",
                "modality": data.modality,
                "uri": data.uri,
                "supported_modalities": sorted(KINDS),
            },
        )
    return found
