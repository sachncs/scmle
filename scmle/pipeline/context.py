"""Mutable state shared by the stages of one run."""

import dataclasses
import logging
from typing import Any
from typing import Self
from typing import TYPE_CHECKING
from typing import TypeVar

import pandas

from scmle.artifacts import results
import scmle.artifacts.bundle
from scmle.assurance.custody import pipeline
from scmle.assurance.custody import sealed
from scmle.contracts import errors
from scmle.contracts import experiments
from scmle.contracts import lineage
from scmle.contracts import tasks
import scmle.contracts.data
import scmle.contracts.metrics
from scmle.data.readers import streaming
from scmle.data.splitting import splitters
import scmle.pipeline.journal
import scmle.pipeline.recommendations
import scmle.pipeline.services
from scmle.runtime import limits

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")
BRIDGE = {
    "governor": "services",
    "capability_registry": "services",
    "trials": "journal",
    "outcomes": "journal",
    "trial_counter": "journal",
    "skipped": "journal",
}
"""Product fields that live on the services or the journal, by owner.

`products.Product.capture` and `apply` read and set a product's fields by
name on the context; these names reach the owner instead. The bridge goes
when the products stop naming context attributes (phase 4, commits 12 to 22).
"""


def require(value: T | None, name: str) -> T:
    """Returns a stage product, failing loudly if its stage has not run.

    Args:
        value: The possibly unset context field.
        name: Field name used in the error message.

    Returns:
        `value`, narrowed to non-None.

    Raises:
        ScmleError: If the producing stage has not executed.
    """
    if value is None:
        raise errors.ScmleError(
            f"'{name}' is unavailable: its stage has not run yet"
        )
    return value


@dataclasses.dataclass
class Context:
    """The services, the journal and the products made so far in a run.

    The services and the journal are set at construction; the remaining
    fields are filled by stages in order. A stage reads its inputs from
    `services`, records through `journal`, and sets the fields below.

    Attributes:
        services: The run's fixed inputs and collaborators.
        journal: Where the run's files, events, trials and decisions go.
        frame: Ingested dataset.
        read_seconds: Time spent reading the dataset.
        cleaning_notes: Data repairs made at ingestion (reported as quality
            findings).
        manifest: Dataset identity.
        evidence_graph: Graph of the run's evidence, replaced by the one
            built when the run is packaged.
        task: Resolved task type.
        profile: Dataset profile.
        cluster_column: Entity column whose rows are not independent in the
            final evaluation (set for `known_entity_future`, where entities
            are not split units but their rows are correlated).
        curation: Curation report as first written, kept so that later
            stages can amend it when they change the feature plan.
        split: Split result with development data and locked test.
        custody: Three-way split outcome when a charter governs the run.
        custody_vault: Holds the sealed confirmation set of a charter run.
        plan: Experiment plan.
        recommendations: Advice for the user derived from the profile.
        metric: Primary metric.
        selected: Winning trial.
        diagnostics: Post-selection diagnostic reports by name.
        reference_scores: Model scores / predicted classes on the held-out
            rows, the reference for output-drift monitoring.
        coverage_claim_sha256: Hash of the coverage claim written for the
            run, or None when no claim was made.
        pending_ledger: The exploration of a charter run whose opening of the
            confirmation set waits for the model selection, or None.
        refits: Candidates already refit on development data during
            constraint checks, by trial id (reused by the final evaluation).
        final_pipeline: Selected pipeline refit on development data.
        shipped: The model the bundle serves: the selected pipeline refit
            on every row after the test, or `final_pipeline`.
        test_rows: The released locked rows and labels, kept for reports
            written after the test was opened.
        stream: The chunked source and totals when the file was streamed.
        pareto_front: Score / latency / size of measured candidates.
        seasonal_baseline: Seasonal-naive forecasts for the locked rows,
            computed at split time (forecasting).
        propensity: How well the covariates predict treatment (uplift).
        fairness_mitigation: Evidence for group thresholds, applied or not.
        ordinal_labels: For ordinal tasks, the labels from lowest to highest
            (the data's target holds their integer ranks).
        unlabeled: Rows without a target, set aside by the `unlabeled` policy.
        ingest: Result of declared-source ingestion, or None when the project
            declared none.
        semi_supervised: Record of the self-training judgement, if any.
        evaluation: Final evaluation record.
        bundle: Packaged model.
        result: Final run result.
    """

    services: scmle.pipeline.services.Services
    journal: scmle.pipeline.journal.Journal
    frame: pandas.DataFrame | None = None
    read_seconds: float = 0.0
    cleaning_notes: list[str] = dataclasses.field(default_factory=list)
    manifest: scmle.contracts.data.DatasetManifest | None = None
    evidence_graph: lineage.Graph = dataclasses.field(
        default_factory=lineage.Graph
    )
    task: tasks.Task | None = None
    profile: scmle.contracts.data.Portrait | None = None
    curation: scmle.contracts.data.Curation | None = None
    cluster_column: str | None = None
    split: splitters.Split | None = None
    custody: pipeline.Outcome | None = None
    custody_vault: sealed.Vault | None = None
    plan: experiments.Plan | None = None
    recommendations: tuple[
        scmle.pipeline.recommendations.Recommendation, ...
    ] = ()
    metric: scmle.contracts.metrics.Metric | None = None
    selected: experiments.Trial | None = None
    diagnostics: dict[str, dict[str, Any]] = dataclasses.field(
        default_factory=dict
    )
    reference_scores: pandas.Series | None = None
    refits: dict[str, Any] = dataclasses.field(default_factory=dict)
    pending_ledger: Any = None
    coverage_claim_sha256: str | None = None
    final_pipeline: Any = None
    shipped: Any = None
    test_rows: tuple[pandas.DataFrame, pandas.Series] | None = None
    stream: tuple[streaming.Feed, streaming.Totals] | None = None
    pareto_front: list[dict[str, Any]] | None = None
    seasonal_baseline: Any = None
    propensity: dict[str, Any] | None = None
    fairness_mitigation: dict[str, Any] | None = None
    ordinal_labels: tuple[Any, ...] | None = None
    unlabeled: pandas.DataFrame | None = None
    ingest: Any = None
    semi_supervised: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    bundle: scmle.artifacts.bundle.Bundle | None = None
    result: results.Run | None = None

    if not TYPE_CHECKING:
        # Hidden from the type checker, so it still flags every attribute a
        # context does not have.

        def __getattr__(self, name: str) -> Any:
            """Reads a bridged product field from its owner (`BRIDGE`).

            Args:
                name: The attribute normal lookup did not find.

            Returns:
                The owner's attribute.

            Raises:
                AttributeError: If `name` is not bridged, or its owner is not
                    set yet.
            """
            owner = BRIDGE.get(name)
            if owner is None or owner not in vars(self):
                raise AttributeError(name)
            return getattr(vars(self)[owner], name)

        def __setattr__(self, name: str, value: Any) -> None:
            """Sets an attribute; a bridged product field is set on its owner.

            The services are frozen, so a bridged services field gives the
            context a copy of them with the field replaced.

            Args:
                name: The attribute.
                value: Its new value.
            """
            owner = BRIDGE.get(name)
            if owner == "journal":
                setattr(self.journal, name, value)
            elif owner == "services":
                self.services = dataclasses.replace(
                    self.services, **{name: value}
                )
            else:
                object.__setattr__(self, name, value)

    @classmethod
    def start(cls, given: scmle.pipeline.services.Services) -> Self:
        """Starts the context of a run with an empty journal.

        Args:
            given: The run's services.

        Returns:
            A context with no products yet.
        """
        return cls(
            given,
            scmle.pipeline.journal.Journal(
                given.spec, given.store, given.log, given.tracker
            ),
        )

    @property
    def regulator(self) -> limits.Governor:
        """The governor, which caps the threads while a stage runs."""
        return self.services.governor

    def check_cancelled(self) -> None:
        """Raises if cancellation was requested (`Services.check`).

        Raises:
            CancelledError: Carrying the run directory in its details.
        """
        self.services.check()

    def amend(self, spec: tasks.Config) -> None:
        """Replaces the specification the later stages read and events carry.

        The charter's entity block and fairness reweighing amend the data
        block this way today; fixing the specification for the whole run is
        STG-3 (phase 4, commit 11).

        Args:
            spec: The amended specification.
        """
        self.services = dataclasses.replace(self.services, spec=spec)
        self.journal.spec = spec

    def close(self) -> None:
        """Releases resources owned by a prepared or executing context.

        A preview still creates a locked test-set boundary during profiling.
        The context owns that boundary until the caller has opened it or the
        preview is finished, so every public path can close it deterministically
        instead of relying on process garbage collection.
        """
        if self.split is not None:
            self.split.locked_test.close()
        if self.custody_vault is not None:
            self.custody_vault.close()
