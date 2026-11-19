"""The built-in capability registry of the tabular path.

What each tabular task family is declared to support (`enrol`), and the
declarations for experimental and extension models. The planning stage
checks every candidate against this registry.
"""

from scmle.contracts import capabilities
from scmle.contracts import tasks

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
    (tasks.Task.RANKING, ("baseline", "boosting")),
    (
        tasks.Task.MULTILABEL,
        ("baseline", "linear", "tree", "boosting"),
    ),
    (
        tasks.Task.MULTIOUTPUT,
        ("baseline", "linear", "tree", "boosting"),
    ),
    (tasks.Task.FORECAST, ("baseline", "linear", "tree", "boosting")),
    (tasks.Task.UPLIFT, ("baseline", "causal")),
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
            model_family=model_family,
            evaluation=TABULAR_BETA_EVALUATIONS[task],
            uncertainty=capabilities.Uncertainty.NONE,
            fairness=fairness,
            explainability=capabilities.Explainability.NONE,
            inference=capabilities.Inference.LIBRARY,
        ),
        status=capabilities.Status.SUPPORTED,
        required_extras=(
            ("deeplearning",) if model_family == "deep_learning" else ("core",)
        ),
        resources=capabilities.Resources(memory_gb=1.0),
        failure_modes=(
            "malformed or schema-incompatible input",
            "model-family dependency unavailable",
            "resource limits exceeded",
        ),
        evidence_links=("tests/integration/test_capability_gate.py",),
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
            model_family=model_family,
            evaluation=capabilities.Evaluation.STANDARD,
            uncertainty=capabilities.Uncertainty.NONE,
            fairness=fairness,
            explainability=capabilities.Explainability.NONE,
            inference=capabilities.Inference.LIBRARY,
        ),
        status=capabilities.Status.EXPERIMENTAL,
        required_extras=("deeplearning",),
        resources=capabilities.Resources(
            memory_gb=4.0,
            max_rows=100_000,
        ),
        failure_modes=failure_modes,
        evidence_links=evidence_links,
        limitations=limitations,
    )


EXPERIMENTAL = (
    (
        capabilities.ModelFamily.TRANSFORMER.value,
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
        capabilities.ModelFamily.JEPA.value,
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


def enrol() -> capabilities.Registry:
    """Builds the explicit capability registry for the tabular beta.

    The registry intentionally declares only the library inference path with
    no portable explanation or predictive-uncertainty promise. Modalities,
    task/model paths, and contract dimensions not listed here are rejected
    before fitting rather than being inferred from an adapter's existence.

    Returns:
        A registry of the supported and experimental tabular capabilities,
        each with its failure modes, evidence links and limitations.
    """
    records: list[capabilities.Capability] = []
    for task, families in TABULAR_BETA_TASK_FAMILIES:
        for model_family in families:
            records.extend(
                advertise(task, model_family, fairness)
                for fairness in fairnesses(task)
            )
    for (
        model_family,
        failure_modes,
        evidence_links,
        limitations,
    ) in EXPERIMENTAL:
        for task in (
            tasks.Task.BINARY,
            tasks.Task.MULTICLASS,
            tasks.Task.REGRESSION,
        ):
            records.extend(
                experimental_tabular_capability(
                    task,
                    model_family,
                    fairness,
                    failure_modes,
                    evidence_links,
                    limitations,
                )
                for fairness in fairnesses(task)
            )
    return capabilities.Registry.seed(records)


def extension_capability(
    key: capabilities.Coordinate,
) -> capabilities.Capability:
    """Describes an explicitly supplied adapter outside the beta guarantee.

    User-registered and test adapters are allowed to exercise the extension
    boundary, but they must never inherit the supported status of a built-in
    family. The plan records this distinction before the adapter is fitted.

    Args:
        key: The capability key of the supplied adapter.

    Returns:
        A capability with extension-only status, no required extras and the
        limitations that the built-in guarantee does not cover it.
    """
    return capabilities.Capability(
        key=key,
        status=capabilities.Status.EXTENSION_ONLY,
        required_extras=(),
        resources=capabilities.Resources(memory_gb=1.0),
        failure_modes=(
            "adapter conformance or dependency failure",
            "resource limits exceeded",
            "missing end-to-end lifecycle evidence",
        ),
        evidence_links=("explicit adapter registration",),
        limitations=(
            "not covered by the built-in production-beta guarantee",
            "model-specific evaluation and recovery evidence is caller-owned",
        ),
    )
