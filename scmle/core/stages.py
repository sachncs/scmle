"""Workflow stages.

Each `Stage` subclass owns one state of the run lifecycle and implements
`execute`; the orchestrator treats them uniformly. Because stages only read
and write a `RunContext`, the ingest/profile prefix is reused for the
side-effect-free plan preview.
"""

import abc
import dataclasses
import datetime
import hashlib
import json
import logging
import time
from typing import Any
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn import base

import scmle
from scmle.api import results
from scmle.artifacts import bundle as bundle_module
from scmle.artifacts import explanation_report
from scmle.artifacts import report
from scmle.contracts import capabilities
from scmle.contracts import data as data_contracts
from scmle.contracts import errors
from scmle.contracts import evaluation as evaluation_contracts
from scmle.contracts import evidence as evidence_contracts
from scmle.contracts import experiments
from scmle.contracts import explanations
from scmle.contracts import lineage
from scmle.contracts import reliability
from scmle.contracts import tasks
from scmle.core import causal_claim
from scmle.core import compiler
from scmle.core import context as context_module
from scmle.core import decision_reports
from scmle.core import ingest_stage
from scmle.core import randomness
from scmle.core import state as run_state
from scmle.data.admissibility import attestations
from scmle.data.admissibility import qualify
from scmle.data.charter import contracts as charter_contracts
from scmle.data.charter import ingest as charter_ingest
from scmle.data.charter import sealing as charter_sealing
from scmle.data.curation import forecasting
from scmle.data.curation import preprocessing
from scmle.data.custody import pipeline as custody_pipeline
from scmle.data.custody import sealed as custody_sealed
from scmle.data.framing import baseline as framing_baseline
from scmle.data.framing import claim as framing_claim
from scmle.data.framing import draft as framing_draft
from scmle.data.identification import run as identification_run
from scmle.data.ledger import budget as ledger_budget
from scmle.data.ledger import confirm as ledger_confirm
from scmle.data.ledger import explore as ledger_explore
from scmle.data.ledger import governed as ledger_governed
from scmle.data.profiling import label_noise
from scmle.data.profiling import profiler
from scmle.data.profiling import representativeness
from scmle.data.readers import registry as source_registry
from scmle.data.readers import streaming
from scmle.data.readers import tabular
from scmle.data.robustness import html_report as robustness_html
from scmle.data.robustness import prefilter as robustness_prefilter
from scmle.data.robustness import qualify as robustness_qualify
from scmle.data.robustness import report as robustness_report
from scmle.data.robustness import stability as robustness_stability
from scmle.data.splitting import splitters
from scmle.data.validation import target as target_module
from scmle.decision import netbenefit as decision_netbenefit
from scmle.decision import threshold as decision_threshold
from scmle.evaluation import backtest
from scmle.evaluation import bias_variance
from scmle.evaluation import confidence
from scmle.evaluation import conformal
from scmle.evaluation import coverage as coverage_module
from scmle.evaluation import coverage_confirm
from scmle.evaluation import coverage_run
from scmle.evaluation import coverage_slices
from scmle.evaluation import criteria
from scmle.evaluation import cv_plus
from scmle.evaluation import diagnostics
from scmle.evaluation import fit_health
from scmle.evaluation import learning_curve
from scmle.evaluation import metrics as metric_catalog
from scmle.evaluation import mitigation
from scmle.evaluation import nested
from scmle.evaluation import policies
from scmle.evaluation import quantile
from scmle.evaluation import release
from scmle.evaluation import replay
from scmle.evaluation import resources
from scmle.evaluation import reweighing
from scmle.evaluation import sensitivity
from scmle.evaluation import suite
from scmle.evaluation import uplift
from scmle.experimentation import ensemble_search
from scmle.experimentation import strategy_search
from scmle.experimentation import trials
from scmle.experimentation import tuning
from scmle.explainability import suite as explanation_suite
from scmle.features import feature_graph
from scmle.governance import attestation
from scmle.governance import ledger
from scmle.governance import model_card
from scmle.governance import sbom
from scmle.governance import vault
from scmle.inference import novelty
from scmle.intelligence import knowledge
from scmle.intelligence import planner as planner_module
from scmle.intelligence import recommendations
from scmle.models import thresholds
from scmle.monitoring import reference as reference_module
from scmle.sampling import pseudo_labeling
from scmle.sampling import strategies

COVERAGE_LEVELS = (0.5, 0.8, 0.9, 0.95)
DIAGNOSTIC_BUDGET_SHARE = 0.25
DIAGNOSTIC_MIN_SECONDS = 30.0
BASELINE = "dummy"
require = context_module.require


LOGGER = logging.getLogger(__name__)


class Stage(abc.ABC):
    """One step of the workflow.

    Attributes:
        state: Lifecycle state the run is in while the stage executes.
        requires: Stages that must run before this one (the workflow is
            ordered by `compiler.compile_workflow`).
    """

    state: ClassVar[run_state.RunState]
    requires: ClassVar[tuple[type, ...]] = ()

    @abc.abstractmethod
    def describe(self, context: context_module.RunContext) -> str:
        """Returns the reason recorded on entering the stage.

        Args:
            context: The run context.

        Returns:
            The reason text attached to the stage's state transition.
        """

    @abc.abstractmethod
    def execute(self, context: context_module.RunContext) -> None:
        """Performs the stage, reading and writing the context.

        Args:
            context: The run context; earlier stages' products are read from it
                and this stage's products are written to it.
        """


PARITY_ROWS = 200


class IngestStage(Stage):
    """Reads the dataset and fingerprints it."""

    state = run_state.RunState.VALIDATING

    def describe(self, context: context_module.RunContext) -> str:
        """Names the stage.

        Args:
            context: The run context; not consulted.

        Returns:
            The text "ingest dataset".
        """
        return "ingest dataset"

    @staticmethod
    def must_stream(data: tasks.DataSpec) -> bool:
        """Whether the file is bigger than the memory budget.

        Args:
            data: The data specification, giving the source and the optional
                memory budget in gigabytes.

        Returns:
            True if a memory budget is set and the source's estimated size
            exceeds it; False if no budget is set.
        """
        if data.memory_budget_gb is None:
            return False
        size = streaming.source_for(data.uri).estimated_bytes()
        return size > data.memory_budget_gb * 1e9

    def read_stream(
        self, context: context_module.RunContext, data: tasks.DataSpec
    ) -> pd.DataFrame:
        """One pass over the file: locked rows by hash, a development sample.

        The streaming source and the pass totals are stored in `context.stream`.

        Args:
            context: The run context, supplying the validation settings.
            data: The data specification naming the source and memory budget.

        Returns:
            The development sample together with the locked rows.
        """
        source = streaming.source_for(data.uri)
        columns = source.shape()[1]
        development, locked = streaming.sample_caps(
            columns,
            data.memory_budget_gb or 1.0,
            context.spec.validation.test_fraction,
        )
        frame, totals = streaming.sample_pass(
            source,
            development,
            locked,
            context.spec.validation.test_fraction,
            context.spec.validation.seed,
        )
        context.stream = (source, totals)
        return frame

    def execute(self, context: context_module.RunContext) -> None:
        """Loads data (copying in-memory input) and builds the manifest.

        Args:
            context: The run context; receives the frame, manifest and cleaning
                notes.

        Raises:
            UnsupportedCapabilityError: If the data is not tabular and no
                in-memory frame was supplied, since no other modality has an
                ingestion adapter.
        """
        data = context.spec.data
        if data.modality != "tabular" and context.input_frame is None:
            raise capabilities.UnsupportedCapabilityError(
                "file-backed modality has no registered ingestion adapter",
                {
                    "status": "unsupported",
                    "reason": "file-backed modality has no ingestion adapter",
                    "modality": data.modality,
                    "uri": data.uri,
                    "supported_modalities": ["tabular"],
                },
            )
        started = time.monotonic()
        if context.input_frame is not None:
            frame = context.input_frame.copy()
        elif self.must_stream(data):
            frame = self.read_stream(context, data)
        else:
            frame = tabular.read_table(data.uri)
        if data.horizon:
            frame = forecasting.add_targets(frame, data)
        context.read_seconds = time.monotonic() - started
        context.manifest = tabular.build_manifest(
            frame, data.target, data.uri, data.modality
        )
        self.persist_source_snapshot(context, frame)
        frame, infinite = tabular.replace_infinities(frame)
        context.cleaning_notes = [
            f"column '{name}': {count} infinite value(s) treated as missing"
            for name, count in infinite.items()
        ]
        if context.stream is not None:
            totals = context.stream[1]
            context.cleaning_notes.append(
                f"streamed {totals.rows} rows; {totals.development_sample} "
                f"development rows kept in memory, {totals.locked} rows locked"
            )
        graph = context.spec.feature_graph
        if graph.nodes:
            feature_graph.validate_graph(graph, set(frame.columns), data)
            ordered = feature_graph.split_nodes(graph)[0]
            frame = feature_graph.SeriesFeatureBuilder(
                ordered, data.group_column, data.time_column
            ).apply(frame)
            context.emit(
                "features.graph",
                f"feature graph v{graph.version} {graph.fingerprint()}",
                data={
                    "fingerprint": graph.fingerprint(),
                    "nodes": [n.model_dump(mode="json") for n in graph.nodes],
                },
            )
        context.frame = frame
        self.refuse_stale_resume(context)
        context.write_json("dataset_manifest.json", context.manifest.to_dict())
        context.track(
            dataset_id=context.manifest.dataset_id,
            data_hash=context.manifest.data_hash,
        )

    def persist_source_snapshot(
        self, context: context_module.RunContext, frame: pd.DataFrame
    ) -> None:
        """Persist source identity and reject breaking resume evolution.

        Args:
            context: The run context, holding the manifest and the run store.
            frame: The loaded data, whose column names and dtypes are recorded.

        Raises:
            ConfigError: If the run is resuming and the source's schema changed
                incompatibly since the earlier snapshot.
        """
        manifest = require(context.manifest, "manifest")
        snapshot = source_registry.SourceSnapshot.from_manifest(
            manifest,
            {
                str(name): str(dtype)
                for name, dtype in zip(frame.columns, frame.dtypes, strict=True)
            },
            datetime.datetime.now(datetime.UTC).isoformat(),
        )
        registry = source_registry.SourceRegistry()
        if context.store is not None:
            saved = context.store.path("source_registry.json")
            if saved.exists():
                registry = source_registry.SourceRegistry.from_json(
                    saved.read_text(encoding="utf-8")
                )
        compatibility = registry.compatibility(snapshot)
        if (
            compatibility.previous_snapshot_id is not None
            and not compatibility.compatible
        ):
            raise errors.ConfigError(
                "source schema changed incompatibly while resuming: "
                f"{compatibility.to_dict()}"
            )
        registry.register(snapshot)
        record = registry.to_dict()
        record["current_snapshot_id"] = snapshot.snapshot_id
        record["schema_compatibility"] = compatibility.to_dict()
        context.write_json("source_registry.json", record)
        context.emit(
            "source.snapshot",
            "source identity and schema evolution recorded",
            ("source_registry.json",),
            {
                "source_id": snapshot.source_id,
                "snapshot_id": snapshot.snapshot_id,
                "schema_status": compatibility.status,
            },
        )

    def refuse_stale_resume(self, context: context_module.RunContext) -> None:
        """Rejects resuming trials computed on different data.

        Args:
            context: The run context, holding the checkpoints and manifest.

        Raises:
            ConfigError: If the run has checkpoints and the data hash differs
                from the one recorded when they were computed.
        """
        if context.store is None or not context.checkpoints:
            return
        saved = context.store.path("dataset_manifest.json")
        if not saved.exists():
            return
        previous = json.loads(saved.read_text(encoding="utf-8")).get(
            "data_hash"
        )
        current = require(context.manifest, "manifest").data_hash
        if previous != current:
            raise errors.ConfigError(
                "cannot resume: the data changed since the checkpointed "
                "trials were computed",
                {"saved": previous, "current": current},
            )


@dataclasses.dataclass(frozen=True, slots=True)
class PendingLedger:
    """A charter run's exploration, finished, with `C` still sealed.

    Attributes:
        explored: The exploration, control and second stage (from `X` only).
        task: The resolved task.
        charter: The verified charter.
        view: The exploration rows.
        structure: Group, time and baseline columns (never candidates).
        graph: The declared DAG, or None.
        found: The qualification of the exploration rows.
        stage: The stability stage.
        signed: Whether the charter signature was verified.
        baseline: The baseline claim about the logistic rule, or None.
        causal: The primary causal claim, or None.
        subject: The rule the baseline claim is about once the model is
            selected, or None.
        holds_for_coverage: Whether `C` stays sealed until the bundle and its
            calibration exist, so the coverage claim can be tested on it.
    """

    explored: ledger_governed.Explored
    task: tasks.TaskType
    charter: charter_contracts.AnalysisCharter
    view: Any
    structure: tuple[str, ...]
    graph: Any
    found: qualify.Qualification
    stage: Any
    signed: bool
    baseline: ledger_confirm.ClaimRunner | None
    causal: ledger_confirm.ClaimRunner | None
    subject: Any = None
    holds_for_coverage: bool = False


class ProfileStage(Stage):
    """Resolves the task, splits the data, and profiles the development rows.

    Only the target is looked at to resolve the task and to split (class
    counts for stratification, groups and time for the other protocols). Every
    statistic that drives a modelling decision (column statistics, feature
    selection, leakage hypotheses, the fingerprint) is then computed on the
    development partition alone, so the locked test rows never influence
    them.
    """

    requires = (IngestStage,)
    state = run_state.RunState.PROFILING

    def describe(self, context: context_module.RunContext) -> str:
        """Names the stage.

        Args:
            context: The run context; not consulted.

        Returns:
            The text "profile dataset".
        """
        return "profile dataset"

    def set_aside_unlabeled(
        self, context: context_module.RunContext, frame: pd.DataFrame
    ) -> pd.DataFrame:
        """Applies the `unlabeled` policy to rows with a missing target.

        Args:
            context: The run context; receives the unlabeled rows and the
                labeled frame.
            frame: The data, some of whose target values are missing.

        Returns:
            The labeled rows. The others wait in `context.unlabeled` (their
            weight, if the data has one, defaults to 1).

        Raises:
            BlockedError: Under the default `block` policy.
        """
        data = context.spec.data
        missing = frame[data.target].isna()
        if data.unlabeled == "block":
            raise errors.BlockedError(
                f"target '{data.target}' has {int(missing.sum())} missing "
                "values; set data.unlabeled to 'ignore' or 'pseudo_label' to "
                "use the unlabeled rows"
            )
        rows = frame[missing].drop(columns=[data.target]).reset_index(drop=True)
        if data.weight_column and data.weight_column in rows:
            rows[data.weight_column] = rows[data.weight_column].fillna(1.0)
        context.unlabeled = rows
        context.cleaning_notes.append(
            f"{len(rows)} row(s) without a target were set aside "
            f"(policy '{data.unlabeled}')"
        )
        context.frame = frame[~missing].reset_index(drop=True)
        return context.frame

    def execute(self, context: context_module.RunContext) -> None:
        """Resolves task/metric, profiles, and enforces the leakage policy.

        Args:
            context: The run context; receives the task, metric, profile,
                curation report and the artifacts written for them.

        Raises:
            BlockedError: If curation leaves unresolved blocking findings, or
                `leakage_policy` is `block` and a leakage hypothesis was raised
                or a screened column could not be checked.
            ConfigError: If `pseudo_label` is used for a non-classification
                task, the approved plan's task differs from the one the data
                resolves to, or a success criterion names a metric that is not
                reported.
            DataError: If a declared group, time or weight column is missing
                from the data.
        """
        spec = context.spec
        frame = require(context.frame, "frame")
        target = frame[spec.data.target]
        if target.isna().any():
            frame = self.set_aside_unlabeled(context, frame)
            target = frame[spec.data.target]
        task = target_module.TaskResolver(spec).resolve(target)
        if (
            context.unlabeled is not None
            and spec.data.unlabeled == "pseudo_label"
            and not task.is_classification
        ):
            raise errors.ConfigError(
                "unlabeled='pseudo_label' supports classification only"
            )
        approved = context.approved_plan
        if approved is not None and approved.task != task.value:
            raise errors.ConfigError(
                f"the approved plan is for {approved.task} but the data "
                f"resolves to {task.value}"
            )
        if task is tasks.TaskType.SURVIVAL:
            target_module.check_events(frame, spec.data.event_column)
        if task is tasks.TaskType.UPLIFT:
            target_module.check_treatment(frame, spec.data.treatment_column)
        if task in (
            tasks.TaskType.MULTILABEL,
            tasks.TaskType.MULTIOUTPUT,
            tasks.TaskType.FORECAST,
        ):
            target_module.check_targets(frame, spec.data.targets, task)
        if task is tasks.TaskType.ORDINAL:
            frame = target_module.encode_ordinal(
                frame, spec.data.target, spec.data.ordinal_order
            )
            context.frame = frame
            context.ordinal_labels = tuple(spec.data.ordinal_order)
        context.task = task
        context.metric = context.catalog.resolve(spec.primary_metric, task)
        unknown = sorted(
            set(spec.problem.success_criteria)
            - set(context.catalog.report_names[task])
        )
        if unknown:
            raise errors.ConfigError(
                f"success criteria name unreported metrics {unknown}; "
                f"available: {sorted(context.catalog.report_names[task])}"
            )
        for role, column in (
            ("group", spec.data.group_column),
            ("time", spec.data.time_column),
            ("weight", spec.data.weight_column),
        ):
            if column and column not in frame.columns:
                raise errors.DataError(
                    f"{role} column '{column}' not found; "
                    f"columns: {list(frame.columns)[:20]}"
                )
        frame = self.synthesize_features(context, frame)
        split = self.make_split(context, task, frame)
        spec = context.spec  # reweighing may have set the weight column
        development = split.dev_features.assign(
            **{spec.data.target: split.dev_target}
        )
        profile = profiler.DatasetProfiler(spec).profile(
            development,
            require(context.manifest, "manifest"),
            task,
            context.read_seconds,
            total_rows=len(frame),
        )
        profile = attach_feature_graph(spec, profile)
        if context.cleaning_notes:
            profile = dataclasses.replace(
                profile,
                quality_findings=profile.quality_findings
                + tuple(context.cleaning_notes),
            )
        profile = self.add_quality_notes(spec, task, development, profile)
        if task is tasks.TaskType.UPLIFT:
            propensity = uplift.check_propensity(
                split.dev_features.drop(
                    columns=[spec.data.treatment_column], errors="ignore"
                ),
                split.dev_features[spec.data.treatment_column].to_numpy(),
                spec.validation.seed,
            )
            context.propensity = propensity.to_dict()
            finding = f"treatment assignment: {propensity.note}"
            profile = dataclasses.replace(
                profile,
                quality_findings=(*profile.quality_findings, finding),
            )
        source_rows = (
            context.stream[1].rows if context.stream is not None else len(frame)
        )
        curation = data_contracts.CurationReport(
            manifest=require(context.manifest, "manifest"),
            source_rows=source_rows,
            profiled_rows=len(development),
            read_seconds=context.read_seconds,
            streamed=context.stream is not None,
            cleaning_notes=tuple(context.cleaning_notes),
            quality_findings=tuple(
                data_contracts.CurationFinding.from_observation(
                    finding, "quality"
                )
                for finding in profile.quality_findings
            ),
            risk_findings=tuple(
                data_contracts.CurationFinding.from_observation(finding, "risk")
                for finding in profile.risk_findings
            ),
            feature_columns={
                "numeric": profile.feature_plan.numeric,
                "categorical": profile.feature_plan.categorical,
                "text": profile.feature_plan.text,
            },
            dropped_columns=profile.feature_plan.dropped,
        )
        card = data_contracts.DatasetCard.from_curation(curation)
        context.profile = profile
        context.curation = curation
        context.write_json("dataset_profile.json", profile.to_dict())
        context.write_json("curation_report.json", curation.to_dict())
        context.write_json("dataset_card.json", card.to_dict())
        context.emit(
            "curation.completed",
            "dataset curation findings and feature decisions recorded",
            (
                "curation_report.json",
                "dataset_profile.json",
                "dataset_card.json",
            ),
            {
                "source_rows": source_rows,
                "profiled_rows": len(development),
                "quality_findings": len(profile.quality_findings),
                "risk_findings": len(profile.risk_findings),
            },
        )
        if not curation.release_ready:
            finding_ids = tuple(
                finding.finding_id
                for finding in curation.unresolved_blocking_findings
            )
            raise errors.BlockedError(
                "unresolved blocking curation findings",
                {"finding_ids": list(finding_ids)},
            )
        context.emit(
            "features.selected",
            "feature set and exclusions fixed from the profile",
            ("dataset_profile.json",),
            {
                "numeric": list(profile.feature_plan.numeric),
                "categorical": list(profile.feature_plan.categorical),
                "dropped": dict(profile.feature_plan.dropped),
            },
        )
        context.track(
            fingerprint=knowledge.fingerprint(profile, task), task=task.value
        )
        if profile.unchecked_columns:
            if spec.leakage_policy == "block":
                raise errors.BlockedError(
                    f"WARNING {profile.screen_warning}; leakage_policy="
                    "'block' does not train on unchecked suspicion (use "
                    "'warn' to proceed with the columns unchecked)",
                    {"leakage": dict(profile.leakage)},
                )
            LOGGER.warning(
                "WARNING %s; the test score is provisional",
                profile.screen_warning,
            )
        if spec.leakage_policy == "block":
            leaks = profile.blocking_leaks
            if leaks:
                raise errors.BlockedError(
                    "unresolved leakage risk (certified screen, "
                    f"Benjamini-Yekutieli level {profile.leakage['alpha']}): "
                    + ", ".join(leaks),
                    {"leakage": dict(profile.leakage)},
                )
        elif profile.leakage.get("suspects"):
            LOGGER.warning(
                "leakage suspect(s) %s stay in the features; the test score "
                "is provisional until a person settles them (declare the "
                "column unavailable at prediction, or attest it)",
                ", ".join(sorted(set(map(str, profile.leakage["suspects"])))),
            )
        self.run_claim_ledger(context, task)

    def synthesize_features(
        self, context: context_module.RunContext, frame: pd.DataFrame
    ) -> pd.DataFrame:
        """Adds the synthesized features before the data are split.

        Runs when the data spec names an `ingest_plan`, which needs a charter
        with an `ingest` block. The plan reads the database through the as-of
        views, so no feature uses a fact recorded after its row's cutoff; the
        confirmation set does not exist yet. A unit verdict of `coarser` or
        `unknown` blocks the run (raised by the ingest stage as `BlockedError`).

        Args:
            context: The run context, holding the specification and run store.
            frame: The loaded data.

        Returns:
            The frame with the synthesized columns appended, or `frame` itself
            when no ingest plan and no charter ingest block are set.

        Raises:
            ConfigError: If only one of the ingest plan and the charter block
                is set, or a synthesized name collides with a column.
        """
        data = context.spec.data
        path = context.spec.charter_path
        block = (
            charter_sealing.read_verified(str(path)).charter.ingest
            if path
            else None
        )
        if not data.ingest_plan:
            if block is not None:
                raise errors.ConfigError(
                    "the charter has an ingest block but data.ingest_plan is "
                    "not set"
                )
            return frame
        if block is None:
            raise errors.ConfigError(
                "data.ingest_plan needs a charter with an ingest block"
            )
        result = ingest_stage.run(
            ingest_stage.load_plan(data.ingest_plan),
            block,
            frame,
            context.store.directory if context.store is not None else None,
        )
        clash = sorted(set(result.features.columns) & set(frame.columns))
        if clash:
            raise errors.ConfigError(
                f"synthesized features collide with columns: {clash}"
            )
        merged = pd.concat(
            [frame.reset_index(drop=True), result.features], axis=1
        )
        if block.entity is not None:
            merged = self.assign_entity_groups(context, block.entity, merged)
        context.frame = merged
        context.ingest = result
        for name, document in result.documents.items():
            context.write_json(name, document)
        context.emit(
            "ingest.completed",
            f"{result.declared} features synthesized through the as-of views",
            (ingest_stage.MANIFEST, ingest_stage.PROVENANCE),
            result.summary,
        )
        return merged

    def assign_entity_groups(
        self,
        context: context_module.RunContext,
        entity: charter_ingest.EntitySpec,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Makes the resolved entity the split's group column.

        Args:
            context: The run context; its specification gains the group column.
            entity: The charter's entity block.
            frame: The data with the synthesized features.

        Returns:
            The frame with the resolved entity column added.

        Raises:
            ConfigError: If the data spec names a group column, or the
                resolved column collides with a column of the table.
        """
        data = context.spec.data
        if data.group_column:
            raise errors.ConfigError(
                f"data.group_column {data.group_column!r} conflicts with the "
                "charter's entity block, which supplies the group column "
                f"{entity.column!r}"
            )
        if entity.column in frame.columns:
            raise errors.ConfigError(
                f"the entity group column {entity.column!r} collides with a "
                "column of the table"
            )
        groups, document = ingest_stage.resolve_entities(frame, entity)
        frame = frame.assign(**{entity.column: groups})
        context.spec = context.spec.model_copy(
            update={
                "data": data.model_copy(update={"group_column": entity.column})
            }
        )
        context.write_json("entity_resolution.json", document)
        context.emit(
            "ingest.entities",
            f"{document['records']} rows resolved to {document['groups']} "
            f"entities; the split groups by {entity.column!r}",
            ("entity_resolution.json",),
            {"tier": document["tier"], "groups": document["groups"]},
        )
        return frame

    def run_claim_ledger(
        self, context: context_module.RunContext, task: tasks.TaskType
    ) -> None:
        """Explores `X` under the ledger and confirms once on `C`.

        Runs only for a charter run. Exploration reads the exploration view; the
        sealed confirmation set is handed to the confirmation stage as an
        argument and is opened at most once, after the survivors are frozen in
        the ledger. Tasks other than classification and regression are recorded
        as skipped: their findings stay unconfirmed. When the baseline claim is
        about the selected model, or the coverage claim needs the calibrated
        model, the confirmation is deferred until the model is frozen.

        Args:
            context: The run context, holding the custody outcome and vault.
            task: The resolved task.
        """
        outcome = context.custody
        vault = context.custody_vault
        if outcome is None or vault is None:
            return
        if task not in tasks.STANDARD_TASKS:
            context.emit(
                "ledger.skipped",
                f"the claim ledger does not cover {task.value} tasks; "
                "findings stay unconfirmed",
            )
            return
        spec = context.spec
        verified = charter_sealing.read_verified(str(spec.charter_path))
        charter = verified.charter
        data = spec.data
        view = outcome.view
        run_dir = context.store.directory if context.store is not None else None
        structure = tuple(
            c
            for c in (data.group_column, data.time_column, data.baseline_column)
            if c
        )
        block = charter.estimand.causal
        graph = causal_claim.load_dag(block.dag) if block is not None else None
        found = self.qualify_exploration(
            context, task, charter, view, structure, graph
        )
        rules = charter.rules
        stage = robustness_prefilter.StabilityStage(
            lambda columns: self.run_robustness(
                context, task, charter, view, columns, structure
            ),
            requested=bool(rules and rules.stability_filter),
            rows=len(view.features),
            seed=charter.seed,
        )
        # Declarations are validated before the expensive exploration.
        baseline = self.baseline_runner(
            context, task, charter, view, structure, signed=verified.signed
        )
        causal = self.causal_runner(context, charter, view, graph, found)
        explored = ledger_governed.explore_governed(
            charter=charter,
            features=view.features,
            target=view.target,
            classification=task.is_classification,
            run_dir=run_dir,
            exclude=structure + tuple(found.quarantined),
            strata=ledger_governed.strata_labels(
                view.features, data.group_column, data.time_column
            ),
            rule=chartered_rule(
                "ledger_rule",
                spec.ledger_rule,
                None if rules is None else rules.ledger_rule,
                "e-lond",
            ),
            policy=chartered_rule(
                "failure_policy",
                spec.confirmation_failure_policy,
                None if rules is None else rules.failure_policy,
                "unconfirmed",
            ),
            diagnostics=found.checks,
            second_stage=stage,
        )
        pending = PendingLedger(
            explored,
            task,
            charter,
            view,
            structure,
            graph,
            found,
            stage,
            verified.signed,
            baseline,
            causal,
            holds_for_coverage=coverage_on_confirmation(charter, task),
        )
        selected = bool(
            data.baseline_column and data.baseline_subject == "selected_model"
        )
        if selected or pending.holds_for_coverage:
            context.pending_ledger = pending
            context.emit(
                "ledger.deferred",
                "the confirmation set stays sealed until "
                + (
                    "the model and its calibration are frozen, so the "
                    "coverage claim can be tested on it"
                    if pending.holds_for_coverage
                    else "the model selection is frozen, so the baseline "
                    "claim can be about that model"
                ),
            )
            return
        self.complete_claim_ledger(context, pending)

    def complete_claim_ledger(
        self,
        context: context_module.RunContext,
        pending: "PendingLedger",
        subject: framing_claim.Rule | None = None,
        extra: ledger_confirm.ClaimRunner | None = None,
    ) -> None:
        """Registers the claims and opens `C` once, then the reports.

        Args:
            context: The run context.
            pending: The exploration, finished on `X` alone.
            subject: The rule the baseline claim is about, when it is not
                the logistic rule frozen on `X`.
            extra: A further registered claim tested in the same opening
                (the coverage claim).
        """
        charter = pending.charter
        view = pending.view
        rules = charter.rules
        baseline = pending.baseline
        if subject is not None:
            baseline = self.baseline_runner(
                context,
                pending.task,
                charter,
                view,
                pending.structure,
                signed=pending.signed,
                subject=subject,
            )
        runner = ledger_confirm.merge_runners(baseline, pending.causal, extra)
        vault = require(context.custody_vault, "custody vault")
        run = ledger_governed.confirm_explored(
            pending.explored,
            source=vault.sets.get("C"),
            confirmation_rule=ledger_budget.MultiplicityRule(
                chartered_rule(
                    "confirmation_rule",
                    context.spec.confirmation_rule,
                    None if rules is None else rules.confirmation_rule,
                    "by",
                )
            ),
            policy=chartered_rule(
                "failure_policy",
                context.spec.confirmation_failure_policy,
                None if rules is None else rules.failure_policy,
                "unconfirmed",
            ),
            claim_runner=runner,
        )
        self.write_registered_claims(context, run, runner)
        self.check_ingest_family(context, run, pending.found)
        context.emit(
            "ledger.completed",
            run.confirmation.reason,
            (
                "hypothesis_ledger.jsonl",
                "negative_control.json",
                "confirmed_claims.json",
            ),
            {
                **run.summary(),
                "rules": "chartered" if rules else "not chartered",
            },
        )
        self.assess_robustness(context, charter, run, pending.stage)
        self.assess_framing(context, charter, view)
        self.assess_identification(
            context, charter, view, run, pending.graph, pending.found
        )

    def baseline_runner(
        self,
        context: context_module.RunContext,
        task: tasks.TaskType,
        charter: charter_contracts.AnalysisCharter,
        view: Any,
        structure: tuple[str, ...],
        *,
        signed: bool,
        subject: framing_claim.Rule | None = None,
    ) -> ledger_confirm.ClaimRunner | None:
        """Freezes the baseline rule on `X` and registers the claim.

        Runs when the data spec names `baseline_column`. The rule is fitted
        on the exploration rows, the selection of outcomes is checked on those
        rows and on the rows set aside for a missing target, and the claim is
        handed to the ledger, which registers it by hash and has it tested on
        the one opening of `C`. With `subject` (the selected model, frozen
        before this call) the claim is about that model and the logistic rule
        is reported next to it as the reference.

        Args:
            context: The run context, holding the profile and unlabeled rows.
            task: The resolved task.
            charter: The analysis charter, supplying the cost model.
            view: The exploration view (features and target).
            structure: Group, time and baseline columns, which are not used as
                rule inputs.
            signed: Whether the charter is signed, which bears on the ceiling
                that a drafted framing may claim.
            subject: The frozen rule of the selected model, or None to make the
                claim about the logistic rule itself.

        Returns:
            The runner that registers and evaluates the baseline claim, or None
            if the data spec names no `baseline_column`.

        Raises:
            ConfigError: If the charter has no cost model, the task is not
                binary or the column is missing.
        """
        data = context.spec.data
        column = data.baseline_column
        if not column:
            return None
        model = charter.cost_model
        if model is None or task is not tasks.TaskType.BINARY:
            raise errors.ConfigError(
                "data.baseline_column needs a binary task and a charter with "
                "a cost_model"
            )
        if column not in view.features.columns:
            raise errors.ConfigError(
                f"baseline column '{column}' is not in the exploration rows"
            )
        classes = sorted(view.target.dropna().unique(), key=str)
        outcome = ledger_explore.encode_target(view.target, True, classes)
        plan = require(context.profile, "profile").feature_plan
        numeric = [
            c
            for c in plan.numeric
            if c in view.features.columns and c not in structure
        ]
        rule = framing_claim.freeze_rule(
            view.features,
            list(outcome.to_numpy(float)),
            numeric,
            model.threshold,
            charter.seed,
        )
        decisions = view.features[column].to_numpy(dtype=float)
        seen = np.ones(len(decisions), dtype=bool)
        if context.unlabeled is not None and column in context.unlabeled:
            extra = context.unlabeled[column].to_numpy(dtype=float)
            decisions = np.concatenate([decisions, extra])
            seen = np.concatenate([seen, np.zeros(len(extra), dtype=bool)])
        selection = framing_baseline.selective_labels(seen, decisions)
        ceiling, why = framing_claim.source_ceiling(
            data.framing_draft, charter.charter_hash, model, signed=signed
        )
        context.write_json(framing_claim.PROVENANCE, why)
        return framing_claim.build_runner(
            rule=rule if subject is None else subject,
            reference=None if subject is None else rule,
            model=model,
            alpha=float(charter.error_budget.confirmation.alpha_confirm),
            baseline_column=column,
            group_column=data.group_column,
            classes=classes,
            selection=selection.to_dict(),
            ceiling=ceiling,
            seed=charter.seed,
        )

    def check_ingest_family(
        self,
        context: context_module.RunContext,
        run: ledger_governed.GovernedRun,
        found: qualify.Qualification,
    ) -> None:
        """Requires the ledger to hold exactly the declared synthesized family.

        A mismatch between the declared family size and the registered
        hypotheses blocks the run (the multiplicity check raises
        `BlockedError`).

        Args:
            context: The run context, holding the ingest result.
            run: The governed run whose ledger registered the hypotheses.
            found: The exploration qualification, whose quarantined columns
                are exempt.
        """
        result = context.ingest
        if result is None:
            return
        columns = tuple(str(c) for c in result.features.columns)
        registered = {
            h.spec.column
            for h in run.ledger.hypotheses.values()
            if h.spec.column
        }
        skipped = {column for column, ignored in run.plan.skipped}
        ingest_stage.check_multiplicity(
            result.declared,
            columns,
            registered,
            set(found.quarantined) | skipped,
        )

    def write_registered_claims(
        self,
        context: context_module.RunContext,
        run: ledger_governed.GovernedRun,
        runner: ledger_confirm.ClaimRunner | None,
    ) -> None:
        """Writes the report files of the registered claims.

        A claim that `C` could not test (no confirmation set, or a failure)
        is written as not evaluated, never left out.

        Args:
            context: The run context that stores the documents.
            run: The governed run, holding the confirmation's documents.
            runner: The claim runner, or None if no claim was registered, in
                which case nothing is written.
        """
        if runner is None:
            return
        documents = dict(run.confirmation.documents)
        status = {
            claim_id: record["status"]
            for claim_id, record in (run.confirmation.registered or {}).items()
        }
        if framing_claim.DOCUMENT not in documents and any(
            i.kind == framing_claim.KIND for i in runner.registered
        ):
            documents[framing_claim.DOCUMENT] = {
                "schema_version": 1,
                "status": ledger_confirm.NOT_EVALUATED,
                "issued": False,
                "reasons": [
                    "the confirmation set was not available to test the claim"
                ],
            }
        for name, document in documents.items():
            context.write_json(name, document)
        context.emit(
            "claims.registered",
            "registered claims were evaluated on the one opening of C",
            tuple(documents),
            {"status": status},
        )

    def run_robustness(
        self,
        context: context_module.RunContext,
        task: tasks.TaskType,
        charter: charter_contracts.AnalysisCharter,
        view: Any,
        survivors: tuple[str, ...],
        structure: tuple[str, ...],
    ) -> robustness_qualify.Robustness:
        """Stability of the survivors and shift against a reference sample.

        Reads the exploration view and the data spec's reference sample, if
        any; never `C` and never `F`. It runs between exploration and the
        opening of `C`, so its inputs are exploration rows only.

        Args:
            context: The run context, holding the profile and specification.
            task: The resolved task.
            charter: The analysis charter, supplying the stability rule, seed
                and multiverse choices.
            view: The exploration view (features and target).
            survivors: Columns that survived exploration.
            structure: Group, time and baseline columns.

        Returns:
            The stability and shift assessment of the survivors.
        """
        data = context.spec.data
        profile = require(context.profile, "profile")
        plan = profile.feature_plan
        candidates = [*plan.numeric, *plan.categorical]
        codes = ledger_explore.encode_target(
            view.target, task.is_classification
        )
        kind = ledger_explore.target_kind(view.target, task.is_classification)
        reference = (
            tabular.read_table(data.reference_uri)
            if data.reference_uri
            else None
        )
        rules = charter.rules
        return robustness_qualify.run(
            view.features,
            codes,
            survivors=list(survivors),
            multiclass=kind == ledger_explore.MULTICLASS,
            choices=charter.multiverse,
            seed=charter.seed,
            permutations=charter.error_budget.permutations_b,
            strata=ledger_governed.strata_labels(
                view.features, data.group_column, data.time_column
            ),
            reference=reference,
            candidates=candidates,
            sensitive=tuple(data.sensitive_columns),
            temporal=bool(data.time_column),
            rule=(
                robustness_stability.DEFAULT_RULE
                if rules is None
                else robustness_stability.Rule(
                    rules.s_min, rules.band, rules.q_min
                )
            ),
        )

    def assess_robustness(
        self,
        context: context_module.RunContext,
        charter: charter_contracts.AnalysisCharter,
        run: ledger_governed.GovernedRun,
        stage: robustness_prefilter.StabilityStage,
    ) -> None:
        """Writes `stability_report.json` and `shift_report.json`.

        The assessment itself ran as the second stage, before `C` was
        opened, on the exploration set only. Without a reference sample the
        shift status is `unverifiable`, never `pass`. Fragile survivors were
        withdrawn from the confirmation only if the charter's rules ask for
        it and the stability rule is certified for its thresholds; the
        report says which.

        Args:
            context: The run context that stores the reports.
            charter: The analysis charter.
            run: The governed run.
            stage: The stability stage, holding the assessment it ran.

        Raises:
            BlockedError: If the stability stage never ran.
        """
        found = stage.found
        if found is None:  # pragma: no cover - run_governed always calls it
            raise errors.BlockedError("the second stage did not run")
        stability_document = robustness_report.stability_report(
            found, stage.note if stage.requested else None
        )
        shift_document = robustness_report.shift_report(found)
        robustness_report.validate_stability(stability_document)
        robustness_report.validate_shift(shift_document)
        context.write_json("stability_report.json", stability_document)
        context.write_json("shift_report.json", shift_document)
        context.write_text(
            "eda_report.html",
            robustness_html.render(stability_document, shift_document),
        )
        context.emit(
            "robustness.completed",
            "stability and shift were assessed on the exploration set",
            ("stability_report.json", "shift_report.json", "eda_report.html"),
            {
                "stability": stability_document["status"],
                "stability_counts": stability_document["counts"],
                "fragile": list(found.fragile),
                "stability_gate": found.stability_gate.value,
                "stability_filter": stage.note,
                "shift": shift_document["status"],
                "shift_reason": shift_document["reason"],
                "forward": found.forward.status,
            },
        )

    def assess_framing(
        self,
        context: context_module.RunContext,
        charter: charter_contracts.AnalysisCharter,
        view: Any,
    ) -> None:
        """Checks that the charter's words name real columns.

        Reads the column names of the exploration view and the target
        column, never row values. Writes `framing_link.json`. The result is
        descriptive: a name that matches no column is `unresolved`, and
        nothing in the run is gated on it.

        Args:
            context: The run context that stores the report.
            charter: The analysis charter whose target, moment and unit are
                matched.
            view: The exploration view, supplying the column names.
        """
        data = context.spec.data
        columns = [*view.features.columns, data.target]
        document = framing_draft.link_report(charter, columns)
        context.write_json("framing_link.json", document)
        context.emit(
            "framing.completed",
            "the charter's target, moment and unit were matched to columns",
            ("framing_link.json",),
            {k: v["status"] for k, v in document["fields"].items()},
        )

    def causal_runner(
        self,
        context: context_module.RunContext,
        charter: charter_contracts.AnalysisCharter,
        view: Any,
        graph: Any,
        found: qualify.Qualification,
    ) -> ledger_confirm.ClaimRunner | None:
        """Registers the primary causal hypothesis for the one opening.

        The strategy is chosen from the charter and the declared DAG alone.
        The estimate, the falsification battery, the replication, the
        sensitivity assessment and the tier are computed on the frame the
        confirmation releases, in the same opening that tests the bivariate
        family; nothing opens `C` for them.

        Args:
            context: The run context, supplying the target name.
            charter: The analysis charter, with its causal block and budget.
            view: The exploration view (features and target).
            graph: The declared DAG, or None.
            found: The exploration qualification, whose DAG checks are
                copied into the report.

        Returns:
            The claim runner, or None for a predictive charter and for a
            charter no strategy identifies (no estimate can exist).
        """
        block = charter.estimand.causal
        if block is None:
            return None
        frame = view.features.copy()
        frame[context.spec.data.target] = view.target
        return causal_claim.build_runner(
            block,
            frame,
            context.spec.data.target,
            charter.error_budget,
            dag=graph,
            seed=charter.seed,
            dag_report=found.dag,
            charter_hash=charter.charter_hash,
        )

    def assess_identification(
        self,
        context: context_module.RunContext,
        charter: charter_contracts.AnalysisCharter,
        view: Any,
        run: ledger_governed.GovernedRun,
        graph: Any,
        found: qualify.Qualification,
    ) -> None:
        """Reports the identification of a causal charter.

        Runs only when the charter's estimand carries a causal block. When the
        confirmation evaluated the registered claim, its five report files
        were written with the other registered claims, bound to the ledger
        head, and only the event is emitted here. Otherwise (no strategy
        identifies the charter, no confirmation set exists, or the claim could
        not be evaluated) the strategy is chosen from the charter, the design
        is diagnosed on `X`, and the claim is `unverified` or
        `not_identified` with no estimate.

        Args:
            context: The run context that stores the documents.
            charter: The analysis charter.
            view: The exploration view (features and target).
            run: The governed run, holding the confirmation's documents.
            graph: The declared DAG, or None.
            found: The exploration qualification, whose DAG checks are used.
        """
        block = charter.estimand.causal
        if block is None:
            return
        documents = run.confirmation.documents
        if causal_claim.CLAIM_FILE in documents:
            claims = documents[causal_claim.CLAIM_FILE]
            context.emit(
                "identification.completed",
                f"causal claim status {claims['status']}, tier {claims['tier']}"
                " (estimated on the single opening of C)",
                tuple(causal_claim.DOCUMENTS),
                {
                    "status": claims["status"],
                    "tier": claims["tier"],
                    "estimated_on_c": True,
                },
            )
            return
        frame = view.features.copy()
        frame[context.spec.data.target] = view.target
        outcome = identification_run.design_only(
            block,
            frame,
            charter.error_budget,
            dag=graph,
            seed=charter.seed,
            dag_checks=found.dag,
        )
        for name, document in outcome.documents.items():
            context.write_json(name, document)
        claims = outcome.documents["causal_claims.json"]
        context.emit(
            "identification.completed",
            f"causal claim status {claims['status']}, tier {claims['tier']}",
            tuple(outcome.documents),
            {
                "status": claims["status"],
                "tier": claims["tier"],
                "estimated_on_c": False,
            },
        )

    def qualify_exploration(
        self,
        context: context_module.RunContext,
        task: tasks.TaskType,
        charter: charter_contracts.AnalysisCharter,
        view: Any,
        structure: tuple[str, ...],
        graph: Any = None,
    ) -> qualify.Qualification:
        """Runs the admissibility tests and detectors on `X` only.

        Writes `admissibility_table.json` and `detector_certificates.json`
        (the certificates the run used). A leakage suspicion quarantines a
        field only when the leakage screen has a current certificate for the
        mechanism that raised it, and a valid per-column attestation
        (`spec.leakage_attestations`) keeps such a field, tier `attested`,
        with a ledger diagnostic. Quarantined fields leave the feature plan,
        with the reason logged.

        Args:
            context: The run context, holding the profile; the feature plan,
                curation report and dataset card are amended when fields are
                quarantined.
            task: The resolved task.
            charter: The analysis charter.
            view: The exploration view (features and target).
            structure: Group, time and baseline columns.
            graph: The declared DAG, or None.

        Returns:
            The qualification: admissibility table, detector results,
            quarantined fields with reasons, and leakage suspects.

        Raises:
            BlockedError: If every feature would be quarantined.
            ConfigError: If the leakage attestations file is invalid.
        """
        spec = context.spec
        try:
            given = (
                attestations.load(spec.leakage_attestations)
                if spec.leakage_attestations
                else ()
            )
        except ValueError as error:
            raise errors.ConfigError(
                f"invalid leakage attestations: {error}"
            ) from error
        found = qualify.run(
            view.features,
            view.target,
            kind=charter.estimand.kind,
            classification=task.is_classification,
            seed=charter.seed,
            structure=structure,
            group_column=spec.data.group_column,
            time_column=spec.data.time_column,
            attestations=given,
            scope=charter.charter_hash,
            causal=charter.estimand.causal,
            dag=graph,
            declared=() if context.ingest is None else context.ingest.records,
        )
        written: list[str] = []
        if found.table is not None:
            context.write_json(
                "admissibility_table.json", found.table.to_dict()
            )
            written.append("admissibility_table.json")
        if found.f5_store is not None:
            context.write_json(
                "detector_certificates.json", found.f5_store.to_dict()
            )
            written.append("detector_certificates.json")
        quarantined = found.quarantined
        profile = require(context.profile, "profile")
        plan = profile.feature_plan
        if quarantined:
            kept = [f for f in plan.features if f not in quarantined]
            if not kept:
                raise errors.BlockedError(
                    "admissibility quarantined every feature",
                    {"fields": sorted(quarantined)},
                )
            gone = frozenset(quarantined)
            plan = dataclasses.replace(
                plan,
                numeric=tuple(c for c in plan.numeric if c not in gone),
                categorical=tuple(c for c in plan.categorical if c not in gone),
                text=tuple(c for c in plan.text if c not in gone),
                monotonic={
                    k: v for k, v in plan.monotonic.items() if k not in gone
                },
                high_cardinality=tuple(
                    c for c in plan.high_cardinality if c not in gone
                ),
                dropped={
                    **plan.dropped,
                    **{
                        name: f"quarantined by admissibility: {why}"
                        for name, why in quarantined.items()
                        if name in plan.features
                    },
                },
            )
            profile = dataclasses.replace(profile, feature_plan=plan)
            context.profile = profile
            context.write_json("dataset_profile.json", profile.to_dict())
            curation = context.curation
            if curation is not None:
                amended = dataclasses.replace(
                    curation,
                    feature_columns={
                        "numeric": plan.numeric,
                        "categorical": plan.categorical,
                        "text": plan.text,
                    },
                    dropped_columns=plan.dropped,
                )
                context.curation = amended
                context.write_json("curation_report.json", amended.to_dict())
                context.write_json(
                    "dataset_card.json",
                    data_contracts.DatasetCard.from_curation(amended).to_dict(),
                )
                written.extend(["curation_report.json", "dataset_card.json"])
        context.emit(
            "admissibility.completed",
            "admissibility and detectors ran on the exploration set",
            tuple(written),
            {
                "leakage_gate": found.leakage_gate.value,
                "quarantined": quarantined,
                "suspects": sorted(found.suspects),
                "detector_status": {
                    detector: {d: s.value for d, s in by_defect.items()}
                    for detector, by_defect in found.detectors.status.items()
                },
                "skipped_detectors": dict(found.detectors.skipped),
                "notes": list(found.notes),
            },
        )
        return found

    @staticmethod
    def add_quality_notes(
        spec: tasks.ProjectSpec,
        task: tasks.TaskType,
        development: pd.DataFrame,
        profile: data_contracts.DatasetProfile,
    ) -> data_contracts.DatasetProfile:
        """Appends label-noise and representativeness findings.

        Args:
            spec: The project specification.
            task: The resolved task; label noise is assessed for binary and
                multiclass tasks only.
            development: The development rows.
            profile: The dataset profile to extend.

        Returns:
            The profile with the new findings and their evidence appended, or
            `profile` itself if there are none.
        """
        present = set(development.columns)
        columns = [c for c in profile.feature_plan.numeric if c in present]
        categorical = [
            c for c in profile.feature_plan.categorical if c in present
        ]
        notes: list[str] = []
        evidence: list[data_contracts.QualityEvidence] = []
        if columns and task in (
            tasks.TaskType.BINARY,
            tasks.TaskType.MULTICLASS,
        ):
            observation = label_noise.assess_label_noise(
                development[columns],
                development[spec.data.target],
                spec.validation.seed,
            )
            if observation.note:
                notes.append(observation.note)
            evidence.append(observation.evidence)
        if spec.data.reference_uri:
            reference = tabular.read_table(spec.data.reference_uri)
            observation = representativeness.assess_representativeness(
                development[columns + categorical],
                reference,
                spec.validation.seed,
            )
            if observation.note:
                notes.append(observation.note)
            evidence.append(observation.evidence)
        if not notes and not evidence:
            return profile
        return dataclasses.replace(
            profile,
            quality_findings=(*profile.quality_findings, *notes),
            quality_evidence=(*profile.quality_evidence, *evidence),
        )

    @staticmethod
    def seal(locked: splitters.LockedTestSet) -> splitters.LockedTestSet:
        """Moves the test rows into a vault process.

        Args:
            locked: The resident test set just produced by the split.

        Returns:
            The sealed set; this process keeps only the row ids. The
            resident object is emptied so no reference to the rows remains.
        """
        assert isinstance(locked, splitters.ResidentTestSet)  # nosec B101
        sealed = vault.SealedTestSet(locked.features, locked.target, locked.ids)
        locked.features = locked.features.iloc[0:0]
        locked.target = locked.target.iloc[0:0]
        return sealed

    def apply_reweighing(
        self,
        context: context_module.RunContext,
        task: tasks.TaskType,
        split: splitters.SplitResult,
    ) -> None:
        """Weights development rows so group and label are independent.

        The weights come from development rows only; the locked rows are
        untouched. The weight column then trains the models as any other. Does
        nothing unless reweighing is the requested mitigation; for tasks other
        than binary and multiclass it records a rejection instead.

        Args:
            context: The run context; its specification gains the weight column.
            task: The resolved task.
            split: The split whose development features receive the weights.
        """
        fairness = context.spec.fairness
        if fairness.mitigation != "reweighing":
            return
        column = fairness.column or context.spec.data.sensitive_columns[0]
        label = f"fairness:{column}"
        if task not in (tasks.TaskType.BINARY, tasks.TaskType.MULTICLASS):
            context.add_outcome(
                label,
                experiments.DecisionAction.REJECT,
                "reweighing applies to binary and multiclass tasks only",
            )
            return
        weights = reweighing.reweighing_weights(
            split.dev_features[column].astype(str), split.dev_target
        )
        split.dev_features[tasks.FAIRNESS_WEIGHT] = weights
        context.spec = context.spec.model_copy(
            update={
                "data": context.spec.data.model_copy(
                    update={"weight_column": tasks.FAIRNESS_WEIGHT}
                )
            }
        )
        context.add_outcome(
            label,
            experiments.DecisionAction.INCLUDE,
            "development rows reweighted so group and label are independent "
            f"(weights from {weights.min():.2f} to {weights.max():.2f})",
        )

    def apply_charter_custody(
        self,
        context: context_module.RunContext,
        task: tasks.TaskType,
        frame: pd.DataFrame,
        split: splitters.SplitResult,
    ) -> splitters.SplitResult:
        """Carves and seals the confirmation set under the charter.

        The charter is verified (hash, and signature when a key is
        configured) before any row is partitioned. The returned split holds
        exploration rows only, so the profiler and every trial that follows
        read nothing from the confirmation set.

        Limits: the vault is a separate process on this machine. This process
        drops its own reference to the sealed rows, but the caller's DataFrame
        or the original file still holds every row.

        Args:
            context: The run context; receives the custody outcome and vault
                and the files that describe them.
            task: The resolved task.
            frame: The full data before partitioning.
            split: The split before the confirmation rows are carved out.

        Returns:
            The split reduced to exploration rows.

        Raises:
            ConfigError: If the charter's unit of prediction is `new_entity`
                and the data spec has no group column.
        """
        path = str(context.spec.charter_path)
        verified = charter_sealing.read_verified(path)
        charter = verified.charter
        if (
            charter.unit_of_prediction == charter_contracts.NEW_ENTITY
            and not context.spec.data.group_column
        ):
            raise errors.ConfigError(
                "the charter's unit_of_prediction is new_entity, which needs "
                "data.group_column so that entities are split and calibrated "
                "as units"
            )
        self.check_known_entity(context, charter, split)
        log = custody_sealed.OpenLog(
            context.store.directory / "custody_log.jsonl"
            if context.store is not None
            else None
        )
        vault_set = custody_sealed.CustodyVault(log)
        data_hash = (
            context.manifest.data_hash
            if context.manifest is not None
            else hashlib.sha256(
                pd.util.hash_pandas_object(frame).to_numpy().tobytes()
            ).hexdigest()
        )
        outcome = custody_pipeline.apply_custody(
            charter=charter,
            split=split,
            frame=frame,
            spec=context.spec,
            task=task,
            data_hash=data_hash,
            vault=vault_set,
        )
        context.custody = outcome
        context.custody_vault = vault_set
        context.write_json(
            "analysis_charter.json", charter_contracts.load_json(path)
        )
        context.write_json("error_budget.json", charter.error_budget.to_dict())
        context.write_json("feasibility.json", outcome.feasibility.to_dict())
        context.write_json("split_manifest.json", outcome.manifest.to_dict())
        context.emit(
            "custody.sealed",
            outcome.feasibility.reason,
            (
                "analysis_charter.json",
                "split_manifest.json",
                "feasibility.json",
                "custody_log.jsonl",
            ),
            {
                "charter_hash": charter.charter_hash,
                "charter_signed": verified.signed,
                "manifest_sha256": outcome.manifest.sha256,
                "scope": outcome.manifest.scope,
                "exploration_rows": len(outcome.manifest.exploration_ids),
                "confirmation_rows": len(outcome.manifest.confirmation_ids),
                "embargo_rows": len(outcome.manifest.embargo_ids),
                "isolation": dict(outcome.manifest.isolation),
            },
        )
        return outcome.split

    def check_known_entity(
        self,
        context: context_module.RunContext,
        charter: charter_contracts.AnalysisCharter,
        split: splitters.SplitResult,
    ) -> None:
        """Requires the data a known-entity-future charter needs.

        A later row of an entity seen in training is predicted, so the
        entity column and the time column must be named and validation must
        be time-ordered: an entity-disjoint split would test the wrong unit
        (that is `new_entity`).

        Args:
            context: The run context, supplying the data spec.
            charter: The analysis charter; nothing is checked unless its unit
                of prediction is `known_entity_future`.
            split: The split, whose strategy must be time-ordered.

        Raises:
            ConfigError: If a column is missing or the split is entity-disjoint.
        """
        if charter.unit_of_prediction != charter_contracts.KNOWN_ENTITY_FUTURE:
            return
        data = context.spec.data
        for role, column in (
            ("group_column", data.group_column),
            ("time_column", data.time_column),
        ):
            if not column:
                raise errors.ConfigError(
                    "the charter's unit_of_prediction is known_entity_future, "
                    f"which needs data.{role} (the entity and the time order)"
                )
        if split.manifest.strategy not in custody_pipeline.TEMPORAL:
            raise errors.ConfigError(
                "known_entity_future needs time-ordered validation "
                "(temporal or rolling), not "
                f"'{split.manifest.strategy}': an entity-disjoint split tests "
                "new entities"
            )
        context.cluster_column = data.group_column
        context.emit(
            "unit.known_entity_future",
            "entities are not split units: their rows sit on both sides, "
            "the confirmation rows are time blocks with an embargo and the "
            "final test rows are the latest period",
            data={"entity_column": data.group_column},
        )

    def make_split(
        self,
        context: context_module.RunContext,
        task: tasks.TaskType,
        frame: pd.DataFrame,
    ) -> splitters.SplitResult:
        """Splits the dataset and records the manifest.

        Args:
            context: The run context; receives the split and its manifest.
            task: The resolved task.
            frame: The data to split.

        Returns:
            The split; under a charter, or when resources are hard-enforced, its
            locked rows are sealed in a separate vault process.
        """
        split = splitters.SplitManager(context.spec, task).split(frame)
        data = context.spec.data
        if task is tasks.TaskType.FORECAST and data.seasonal_period:
            # Computed now: under hard isolation the frame held afterwards has
            # no locked rows, but the seasonal baseline needs their history.
            context.seasonal_baseline = backtest.seasonal_naive(
                frame,
                np.asarray(split.manifest.test_ids),
                data.value_column or "",
                data.group_column,
                len(data.targets),
                data.seasonal_period,
            )
        chartered = bool(context.spec.charter_path)
        if chartered:
            split = self.apply_charter_custody(context, task, frame, split)
        self.apply_reweighing(context, task, split)
        isolation = "in-process protocol"
        # A charter requests custody regardless of the resource-enforcement
        # mode: sealing the rows must not depend on trial supervision.
        if context.resource_governor.hard or chartered:
            split.locked_test = self.seal(split.locked_test)
            context.frame = split.dev_features.assign(
                **{context.spec.data.target: split.dev_target}
            )
            isolation = "separate vault process"
        context.split = split
        context.write_json("validation_manifest.json", split.manifest.to_dict())
        context.track(split_fingerprint=split.manifest.fingerprint)
        context.emit(
            "split.created",
            f"{split.manifest.strategy} split",
            ("validation_manifest.json",),
            {
                "fingerprint": split.manifest.fingerprint,
                "strategy": split.manifest.strategy,
                "development_rows": len(split.manifest.train_ids),
                "test_rows": len(split.manifest.test_ids),
                "test_set_isolation": isolation,
                "notes": list(split.manifest.notes),
            },
        )
        return split


def chartered_rule(
    name: str, requested: str, chartered: str | None, default: str
) -> str:
    """The rule a charter run uses: the charter's when it fixes one.

    A run specification that names a different, non-default value than the
    charter does is a conflict and is refused; the charter is what was signed.

    Args:
        name: Field name, for the message.
        requested: Value in the run specification.
        chartered: Value in the charter's rules, or None without rules.
        default: The specification's default.

    Returns:
        The charter's value when it fixes one, otherwise `requested`.

    Raises:
        ConfigError: If the specification contradicts the charter.
    """
    if chartered is None:
        return requested
    if requested not in (chartered, default):
        raise errors.ConfigError(
            f"the run specification sets {name}={requested!r} but the "
            f"charter fixes {chartered!r}; change the specification or "
            "re-sign the charter"
        )
    return chartered


def attach_feature_graph(
    spec: tasks.ProjectSpec, profile: data_contracts.DatasetProfile
) -> data_contracts.DatasetProfile:
    """Adds the feature graph to the profile's feature plan.

    Row-wise nodes become numeric model features computed inside the
    pipeline; ordered nodes are already columns of the profiled frame. Raw
    columns the graph reads but the model does not use are recorded as
    `extra_inputs`, so inference knows what the caller must supply.

    Args:
        spec: Project specification.
        profile: Profile whose plan lacks the graph.

    Returns:
        A profile whose feature plan carries the graph.
    """
    graph = spec.feature_graph
    if not graph.nodes:
        return profile
    ordered, rowwise = feature_graph.split_nodes(graph)
    plan = profile.feature_plan
    computed = {n.name for n in graph.nodes} | plan.derived_names
    needed = feature_graph.required_raw_inputs(graph.nodes, computed)
    if ordered:
        needed += [
            c
            for c in (spec.data.group_column, spec.data.time_column)
            if c and c not in needed
        ]
    used = set(plan.features) | {n.name for n in rowwise}
    extra = tuple(
        dict.fromkeys(
            [*plan.extra_inputs, *(c for c in needed if c not in used)]
        )
    )
    new_plan = dataclasses.replace(
        plan,
        numeric=plan.numeric + tuple(n.name for n in rowwise),
        derived=plan.derived + tuple(rowwise),
        series=tuple(ordered),
        extra_inputs=extra,
    )
    return dataclasses.replace(profile, feature_plan=new_plan)


def build_plan(
    context: context_module.RunContext,
) -> experiments.ExperimentPlan:
    """Creates and validates the experiment plan from a profiled context.

    The planner remains responsible for adapter compatibility and search
    policy. The capability registry is the separate release boundary: every
    candidate in the resulting plan must have an explicit complete capability
    declaration before the execution plan can reach ``TrainStage``.

    Args:
        context: The run context, holding the profile, task and manifest.

    Returns:
        The validated experiment plan.

    Raises:
        capabilities.UnsupportedCapabilityError: If any planned candidate or
            requested fairness contract is not explicitly supported.
    """
    approved = context.approved_plan
    planner = planner_module.Planner(context.registry, context.catalog)
    plan = planner.plan(
        planner_module.PlanningContext(
            context.spec,
            require(context.profile, "profile"),
            require(context.task, "task"),
            approved.candidate_models if approved is not None else None,
            context.history,
        )
    )
    task = require(context.task, "task")
    manifest = require(context.manifest, "manifest")
    for model_name in plan.candidate_models:
        adapter = context.registry.get(model_name)
        model_family = adapter.capability_family or adapter.family
        key = capability_key(context, task, model_family)
        allowed_statuses = (
            {
                capabilities.CapabilityStatus.SUPPORTED,
                capabilities.CapabilityStatus.EXPERIMENTAL,
            }
            if model_name in context.spec.models.experimental_models
            else None
        )
        try:
            context.capability_registry.require(
                key,
                allowed_statuses=allowed_statuses,
                row_count=manifest.row_count,
            )
        except capabilities.UnsupportedCapabilityError:
            if model_family not in {"custom", "test"}:
                raise
            if context.capability_registry.find(key) is None:
                context.capability_registry.register(
                    context_module.extension_capability(key)
                )
            context.capability_registry.require(
                key,
                allowed_statuses={capabilities.CapabilityStatus.EXTENSION_ONLY},
                row_count=manifest.row_count,
            )
    return plan


def capability_key(
    context: context_module.RunContext,
    task: tasks.TaskType,
    model_family: str,
) -> capabilities.CapabilityKey:
    """Builds the complete capability request for a planned candidate.

    The current beta exposes tabular data through the library inference path.
    Selection uncertainty is deliberately not translated into predictive
    uncertainty: it is a model-selection policy, not an output contract.

    Args:
        context: The run context, supplying the manifest and specification.
        task: The resolved task.
        model_family: Family of the candidate model.

    Returns:
        The capability key requested for the candidate.

    Raises:
        capabilities.UnsupportedCapabilityError: If the dataset modality is
            unknown, or the requested fairness behavior is not implemented
            for ``task``.
        ConfigError: If the manifest's modality differs from the data spec's.
    """
    manifest = require(context.manifest, "manifest")
    try:
        modality = capabilities.Modality(manifest.modality)
    except ValueError as error:
        raise capabilities.UnsupportedCapabilityError(
            "unsupported dataset modality before training",
            {
                "status": "unsupported",
                "reason": "dataset manifest has an unknown modality",
                "modality": manifest.modality,
            },
        ) from error
    if manifest.modality != context.spec.data.modality:
        raise errors.ConfigError(
            "dataset manifest modality does not match the project data "
            f"spec: {manifest.modality!r} != {context.spec.data.modality!r}"
        )
    fairness = (
        capabilities.Fairness.NONE
        if context.spec.fairness.mitigation == "none"
        else capabilities.Fairness.MITIGATION
    )
    key = capabilities.CapabilityKey(
        modality=modality,
        task=task,
        model_family=model_family,
        evaluation=context_module.TABULAR_BETA_EVALUATIONS[task],
        uncertainty=capabilities.Uncertainty.NONE,
        fairness=fairness,
        explainability=capabilities.Explainability.NONE,
        inference=capabilities.Inference.LIBRARY,
    )
    if (
        context.spec.fairness.mitigation == "group_thresholds"
        and task is not tasks.TaskType.BINARY
    ):
        context.capability_registry.raise_unsupported(
            key,
            "group-threshold mitigation is implemented for binary tasks only",
        )
    return key


def advise(context: context_module.RunContext) -> tuple[Any, ...]:
    """Derives and stores the recommendations for the user.

    Args:
        context: A context whose profile is populated.

    Returns:
        The recommendations, also kept on the context.
    """
    advice = recommendations.recommend(
        require(context.profile, "profile"), context.spec
    )
    context.recommendations = advice
    return advice


class PlanStage(Stage):
    """Builds the experiment plan from the development-set profile."""

    requires = (ProfileStage,)
    state = run_state.RunState.PLANNING

    def describe(self, context: context_module.RunContext) -> str:
        """Names the stage.

        Args:
            context: The run context; not consulted.

        Returns:
            The text "build execution plan".
        """
        return "build execution plan"

    def execute(self, context: context_module.RunContext) -> None:
        """Plans candidates.

        Writes the execution plan, the capability registry, the decision log and
        the recommendations.

        Args:
            context: The run context; receives the execution plan.
        """
        plan = build_plan(context)
        context.plan = plan
        context.write_json("execution_plan.json", plan.to_dict())
        context.write_json(
            "capability_registry.json",
            context.capability_registry.to_dict(),
        )
        context.write_json(
            "diagnostics/decision_log.json",
            {"plan": [d.to_dict() for d in plan.decisions], "outcomes": []},
        )
        advice = advise(context)
        context.write_json(
            "diagnostics/recommendations.json",
            [item.to_dict() for item in advice],
        )
        context.emit(
            "plan.recommendations",
            f"{len(advice)} recommendation(s) for the user",
            ("diagnostics/recommendations.json",),
        )


class TrainStage(Stage):
    """Cross-validates every candidate within the time budget."""

    requires = (PlanStage,)
    state = run_state.RunState.TRAINING

    def describe(self, context: context_module.RunContext) -> str:
        """Reports the number of candidates.

        Args:
            context: The run context, holding the execution plan.

        Returns:
            The candidate count as text.
        """
        plan = require(context.plan, "plan")
        return f"{len(plan.candidate_models)} candidates"

    def execute(self, context: context_module.RunContext) -> None:
        """Screens candidates, then tunes the most promising ones.

        Candidates run in chunks of `backend.workers`; the budget is checked
        before every chunk but the first, and results are recorded in
        candidate order regardless of the backend.

        Args:
            context: The run context, holding the execution plan; receives the
            trials.
        """
        runner = trials.TrialRunner(context)
        plan = require(context.plan, "plan")
        names = plan.candidate_models
        for start in range(0, len(names), runner.workers):
            context.check_cancelled()
            chunk = names[start : start + runner.workers]
            ids = context.allocate_trial_ids(len(chunk))
            reason = context.resource_governor.skip_reason(first=start == 0)
            if reason is not None:
                outcomes = [
                    runner.skip(trial_id, name, reason)
                    for trial_id, name in zip(ids, chunk, strict=True)
                ]
            else:
                jobs = [
                    trials.TrialJob(trial_id, name)
                    for trial_id, name in zip(ids, chunk, strict=True)
                ]
                outcomes = runner.run_many(jobs)
            for outcome in outcomes:
                context.record_trial(outcome)
        strategy_search.StrategySearch(context, runner).search(
            list(context.trials)
        )
        tuning.Tuner(context, runner).tune(list(context.trials))
        ensemble_search.EnsembleSearch(context, runner).search(
            list(context.trials)
        )


class SelectStage(Stage):
    """Applies the selection policy and the serving constraints."""

    requires = (TrainStage,)
    state = run_state.RunState.SELECTING

    def describe(self, context: context_module.RunContext) -> str:
        """Names the policy.

        Args:
            context: The run context, holding the selection policy.

        Returns:
            The selection policy's description.
        """
        return context.policy.description

    def announce(self, context: context_module.RunContext) -> None:
        """Audits that the selection is frozen before the test set opens.

        Args:
            context: The run context, holding the selected trial.
        """
        selected = require(context.selected, "selected")
        context.emit(
            "selection.frozen",
            f"{selected.model_name}/{selected.strategy}",
            (f"trials/{selected.trial_id}.json",),
            {
                "trial_id": selected.trial_id,
                "params": selected.params,
                "cv_score": selected.score_mean,
            },
        )

    def limits(
        self, context: context_module.RunContext
    ) -> resources.ResourceLimits:
        """Builds the serving limits from the specification.

        Args:
            context: The run context, supplying the selection settings.

        Returns:
            The model-size and latency limits.
        """
        spec = context.spec.selection
        return resources.ResourceLimits(
            spec.max_model_mb,
            spec.max_p95_latency_ms,
            spec.latency_safety_factor,
        )

    def select_on_front(
        self,
        context: context_module.RunContext,
        checked: list[experiments.TrialResult],
        limits: resources.ResourceLimits,
    ) -> None:
        """Measures every checked candidate and picks from the Pareto front.

        Among candidates that meet the limits and score within
        `pareto_tolerance` of the best, the fastest wins (smaller size breaks
        ties). The whole front is reported so the trade-off is visible.

        Args:
            context: The run context; receives the front and the selection.
            checked: The best-ranked trials to measure.
            limits: The serving limits.

        Raises:
            BlockedError: If no checked candidate satisfies the limits.
        """
        split = require(context.split, "split")
        task = require(context.task, "task")
        plan = require(context.profile, "profile").feature_plan
        metric = require(context.metric, "metric")
        sign = 1.0 if metric.greater_is_better else -1.0
        rows = []
        for trial in checked:
            candidate = strategies.get_strategy(trial.strategy).build(
                plan,
                context.registry.get(trial.model_name),
                task,
                context.spec.validation.seed,
                trial.params,
            )
            candidate.fit(split.dev_features, split.dev_target)
            usage = resources.measure(candidate, split.dev_features)
            rows.append(
                Measured(trial, candidate, usage, limits.violations(usage))
            )
        feasible = [r for r in rows if not r.broken]
        if not feasible:
            raise errors.BlockedError(
                f"none of the top {len(checked)} candidates satisfies the "
                "serving constraints; relax the limits or allow more candidates"
            )
        best = max(sign * r.trial.score_mean for r in feasible)
        near = [
            r
            for r in feasible
            if best - sign * r.trial.score_mean
            <= context.spec.selection.pareto_tolerance
        ]
        chosen = min(
            near,
            key=lambda r: (r.usage.p95_latency_ms, r.usage.model_mb),
        )
        context.pareto_front = pareto_report(rows, chosen, sign)
        trial = chosen.trial
        context.add_outcome(
            f"{trial.model_name}/{trial.strategy}",
            experiments.DecisionAction.INCLUDE,
            f"fastest within {context.spec.selection.pareto_tolerance} of the "
            f"best score ({chosen.usage.p95_latency_ms:.1f} ms p95)",
        )
        context.refits[trial.trial_id] = chosen.candidate
        context.selected = trial
        self.announce(context)

    def execute(self, context: context_module.RunContext) -> None:
        """Chooses the model, then opens a deferred confirmation.

        If the confirmation was deferred, it is completed here once the model
        is frozen, unless the coverage claim still waits for calibration.
        Raises `BlockedError` (through `choose`) if no checked candidate
        satisfies the constraints.

        Args:
            context: The run context, holding any pending ledger.
        """
        self.choose(context)
        pending = context.pending_ledger
        if pending is None:
            return
        selected = require(context.selected, "selected")
        split = require(context.split, "split")
        data = context.spec.data
        subject = None
        if data.baseline_column and data.baseline_subject == "selected_model":
            subject = self.selected_rule(context, pending, selected, split)
        if pending.holds_for_coverage:
            context.pending_ledger = dataclasses.replace(
                pending, subject=subject
            )
            return
        context.pending_ledger = None
        ProfileStage().complete_claim_ledger(context, pending, subject)

    def selected_rule(
        self,
        context: context_module.RunContext,
        pending: "PendingLedger",
        selected: experiments.TrialResult,
        split: Any,
    ) -> framing_claim.SelectedModelRule:
        """The selected model as the rule the baseline claim is about.

        Args:
            context: The run context; the refit pipeline is cached in it.
            pending: The deferred ledger, supplying the view and charter.
            selected: The selected trial.
            split: The split whose development rows the model is refit on if
                no refit exists.

        Returns:
            The selected model with its positive class, the cost model's
            threshold and a description of how it was fitted.
        """
        pipeline = context.refits.get(selected.trial_id)
        if pipeline is None:
            pipeline = strategies.get_strategy(selected.strategy).build(
                require(context.profile, "profile").feature_plan,
                context.registry.get(selected.model_name),
                require(context.task, "task"),
                context.spec.validation.seed,
                selected.params,
            )
            pipeline.fit(split.dev_features, split.dev_target)
            context.refits[selected.trial_id] = pipeline
        model = require(pending.charter.cost_model, "cost_model")
        classes = sorted(pending.view.target.dropna().unique(), key=str)
        return framing_claim.SelectedModelRule(
            pipeline,
            classes[-1],
            float(model.threshold),
            {
                "model": selected.model_name,
                "strategy": selected.strategy,
                "params": selected.params,
                "trial_id": selected.trial_id,
                "split": split.manifest.fingerprint,
                "seed": context.spec.validation.seed,
            },
        )

    def choose(self, context: context_module.RunContext) -> None:
        """Chooses the best trial that satisfies the serving constraints.

        Candidates are walked in policy order; each is refit and measured on
        this machine, and the first one within every limit wins.

        Args:
            context: The run context; receives the selected trial.

        Raises:
            BlockedError: If no checked candidate satisfies the constraints.
        """
        ranked = context.policy.rank(
            context.trials, require(context.metric, "metric"), context.registry
        )
        limits = self.limits(context)
        spec = context.spec.selection
        if not limits.active and not spec.pareto:
            context.selected = ranked[0]
            self.announce(context)
            return
        split = require(context.split, "split")
        task = require(context.task, "task")
        plan = require(context.profile, "profile").feature_plan
        checked = ranked[: context.spec.selection.max_candidates_checked]
        if spec.pareto:
            self.select_on_front(context, checked, limits)
            return
        for trial in checked:
            candidate = strategies.get_strategy(trial.strategy).build(
                plan,
                context.registry.get(trial.model_name),
                task,
                context.spec.validation.seed,
                trial.params,
            )
            candidate.fit(split.dev_features, split.dev_target)
            usage = resources.measure(candidate, split.dev_features)
            broken = limits.violations(usage)
            label = f"{trial.model_name}/{trial.strategy}"
            if broken:
                context.add_outcome(
                    label, experiments.DecisionAction.REJECT, "; ".join(broken)
                )
                continue
            context.add_outcome(
                label,
                experiments.DecisionAction.INCLUDE,
                f"within serving limits ({usage.model_mb:.2f} MB, p95 "
                f"{usage.p95_latency_ms:.1f} ms on {usage.machine})",
            )
            context.refits[trial.trial_id] = candidate
            context.selected = trial
            self.announce(context)
            return
        raise errors.BlockedError(
            f"none of the top {len(checked)} candidates satisfies the "
            "serving constraints; relax the limits or allow more candidates"
        )


@dataclasses.dataclass
class Measured:
    """A candidate refit on the development rows and measured for serving.

    Attributes:
        trial: The trial it came from.
        candidate: The fitted pipeline.
        usage: Its measured size and latency.
        broken: Serving limits it violates.
    """

    trial: experiments.TrialResult
    candidate: Any
    usage: resources.ResourceUsage
    broken: list[str]


def pareto_report(
    rows: list[Measured], chosen: Measured, sign: float
) -> list[dict[str, Any]]:
    """Scores / latency / size of each measured candidate, with dominance.

    Args:
        rows: The measured candidates.
        chosen: The candidate that was selected.
        sign: 1 if a higher score is better, -1 if a lower score is better.

    Returns:
        One record per candidate, in order: model, strategy, score, p95
        latency, size, limit violations, whether another candidate is at
        least as good on all three and better on one, and whether it was
        chosen.
    """
    points = [
        (
            sign * r.trial.score_mean,
            r.usage.p95_latency_ms,
            r.usage.model_mb,
        )
        for r in rows
    ]

    def dominated(index: int) -> bool:
        a = points[index]
        return any(
            b[0] >= a[0]
            and b[1] <= a[1]
            and b[2] <= a[2]
            and (b[0] > a[0] or b[1] < a[1] or b[2] < a[2])
            for other, b in enumerate(points)
            if other != index
        )

    return [
        {
            "model": r.trial.model_name,
            "strategy": r.trial.strategy,
            "score": r.trial.score_mean,
            "p95_latency_ms": r.usage.p95_latency_ms,
            "model_mb": r.usage.model_mb,
            "violations": list(r.broken),
            "dominated": dominated(index),
            "chosen": r is chosen,
        }
        for index, r in enumerate(rows)
    ]


def reference_scores(
    fitted: Any, features: pd.DataFrame, task: tasks.TaskType
) -> pd.Series:
    """Model outputs on held-out rows, used as the output-drift reference.

    Args:
        fitted: The final candidate.
        features: Held-out features.
        task: Resolved task type.

    Returns:
        Positive-class probabilities for binary tasks, predicted labels for
        multiclass tasks and predictions for regression.
    """
    if task is tasks.TaskType.BINARY:
        return pd.Series(fitted.predict_proba(features)[:, 1], name="score")
    predicted = np.asarray(fitted.predict(features))
    if predicted.ndim == 2:
        predicted = predicted[:, 0]  # The primary target stands for all.
    return pd.Series(predicted, name="prediction")


def coverage_on_confirmation(
    charter: charter_contracts.AnalysisCharter, task: tasks.TaskType
) -> bool:
    """Whether the charter asks for the coverage claim on `C` for this task.

    Args:
        charter: The analysis charter.
        task: The resolved task.

    Returns:
        True if the charter's uncertainty block asks for it and the task
        has a primary coverage choice.
    """
    block = charter.uncertainty
    return (
        block is not None
        and block.coverage_on_confirmation
        and coverage_confirm.primary_choice(task.value) is not None
    )


def declared_slice_columns(
    context: context_module.RunContext,
) -> tuple[str, ...]:
    """The slice columns the charter's uncertainty block declares.

    Args:
        context: The run context, supplying the charter path.

    Returns:
        The declared columns; empty if there is no charter or no
        uncertainty block.
    """
    path = context.spec.charter_path
    if not path:
        return ()
    block = charter_sealing.read_verified(str(path)).charter.uncertainty
    return () if block is None else block.slice_columns


def slice_labels(
    context: context_module.RunContext, features: pd.DataFrame
) -> dict[str, Any]:
    """Slice label of each final test row for every declared column.

    Bands of a numeric column are cut at quantiles of the development rows.
    A column the rows lack, or that has too many levels, maps to None.

    Args:
        context: The run context, supplying the split and charter.
        features: The final test rows.

    Returns:
        A mapping from each declared slice column to its labels, or None.
    """
    split = require(context.split, "split")
    found: dict[str, Any] = {}
    for column in declared_slice_columns(context):
        found[column] = (
            coverage_slices.labels_for(
                features[column], split.dev_features.get(column)
            )
            if column in features.columns
            else None
        )
    return found


def chartered_decision_model(
    context: context_module.RunContext,
) -> charter_contracts.AnalysisCharter | None:
    """The charter when it declares a decision layer, else None.

    Args:
        context: The run context, supplying the charter path.

    Returns:
        The charter if it declares both a decision layer and a cost model,
        else None.
    """
    path = context.spec.charter_path
    if path is None:
        return None
    charter = charter_sealing.read_verified(str(path)).charter
    if charter.decision is None or charter.cost_model is None:
        return None
    return charter


def chartered_threshold(
    fitted: Any,
    probabilities: np.ndarray,
    actual: np.ndarray,
    charter: charter_contracts.AnalysisCharter,
    seed: int,
    rows: int,
) -> tuple[Any, dict[str, Any]]:
    """Thresholds from the chartered cost model with the calibration gate.

    The rule is chosen on out-of-fold development probabilities only and is
    frozen; a recalibrating rule is expressed as the equivalent cut on the raw
    probability, so the served classifier needs no extra stage.

    Args:
        fitted: The fitted classifier to wrap.
        probabilities: Out-of-fold development probabilities.
        actual: The observed labels.
        charter: The analysis charter with its decision layer and cost model.
        seed: Seed for the calibration replicates.
        rows: Number of validation rows, recorded in the description.

    Returns:
        The thresholded classifier and a description of the objective, the
        raw threshold and the cost model's hash.

    Raises:
        ScmleError: If the charter has no decision layer or cost model, or
            the chosen rule has no raw threshold.
    """
    layer = charter.decision
    model = charter.cost_model
    if layer is None or model is None:
        raise errors.ScmleError("the charter has no decision layer")
    chosen = decision_threshold.choose_threshold(
        probabilities,
        actual.astype(float),
        model,
        level=layer.model.calibration_level,
        replicates=charter.error_budget.permutations_b,
        seed=seed,
    )
    cut = chosen.rule.raw_threshold
    if cut is None:
        raise errors.ScmleError("the chosen rule has no raw threshold")
    description = {
        "objective": "chartered_cost",
        "threshold": cut,
        "validation_rows": rows,
        "f11": chosen.to_dict(),
        "cost_model_hash": model.model_hash,
    }
    return thresholds.ThresholdClassifier.around_fitted(fitted, cut), (
        description
    )


def chartered_net_benefit(
    context: context_module.RunContext,
    pipeline: Any,
    features: pd.DataFrame,
    target: pd.Series,
    decision: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Net benefit of the frozen rule on the locked test set, once.

    Registered by the rule's hash before the table is read; the guard refuses
    a second call. The band is simultaneous over the chartered range of
    threshold probabilities and clustered by the group column of a grouped
    split. The claim is `useful` or `not_useful_in_range`; it is tiered
    `attested` when every cost came from a person and `unverified` otherwise.

    Args:
        context: The run context, supplying the charter and split.
        pipeline: The fitted classifier.
        features: The final test rows.
        target: The final test labels.
        decision: The chartered threshold description, or None.

    Returns:
        The net-benefit report with its tier, cost model hash and rule
        hash, or None if the charter has no decision layer or the decision
        carries no chartered rule.

    Raises:
        ScmleError: If the charter has no cost model.
    """
    charter = chartered_decision_model(context)
    if charter is None or decision is None or "f11" not in decision:
        return None
    model = charter.cost_model
    if model is None:
        raise errors.ScmleError("the charter has no cost model")
    positive = pipeline.classes_[1]
    split = require(context.split, "split")
    clusters = (
        features[context.spec.data.group_column]
        if split.manifest.strategy == "group" and context.spec.data.group_column
        else None
    )
    guard = decision_netbenefit.FinalNetBenefit(decision["f11"]["rule_hash"])
    report = guard.evaluate(
        decision["f11"]["rule_hash"],
        (target == positive).to_numpy(dtype=float),
        (np.asarray(pipeline.predict(features)) == positive).astype(float),
        (model.t_lo, model.t_hi),
        clusters=clusters,
        alpha=float(charter.error_budget.confirmation.alpha_confirm),
        replicates=max(charter.error_budget.permutations_b, 500),
        seed=charter.seed,
    )
    tier = "unverified" if model.unverified_entries else "attested"
    return {
        **report.to_dict(),
        "tier": "unverified"
        if decision["f11"]["tier"] == "unverified"
        else tier,
        "cost_model_hash": model.model_hash,
        "rule_hash": decision["f11"]["rule_hash"],
    }


def choose_decision_threshold(
    context: context_module.RunContext, fitted: Any
) -> tuple[Any, dict[str, Any] | None]:
    """Optimizes the binary decision threshold on out-of-fold predictions.

    Only development data is used: each validation row is scored by a model
    that never saw it, so the locked test set stays untouched.

    Args:
        context: Run context holding the split and the selected trial.
        fitted: The selected candidate, already refit on development data.

    Returns:
        The estimator to evaluate (wrapped with the threshold when one was
        optimized) and a description of the decision, or None when disabled
        or not applicable.
    """
    decision = context.spec.decision
    task = require(context.task, "task")
    chartered = chartered_decision_model(context)
    if task is not tasks.TaskType.BINARY or (
        decision.threshold_objective == "none" and chartered is None
    ):
        return fitted, None
    split = require(context.split, "split")
    selected = require(context.selected, "selected")
    strategy = strategies.get_strategy(selected.strategy)
    adapter = context.registry.get(selected.model_name)
    plan = require(context.profile, "profile").feature_plan
    seed = context.spec.validation.seed
    probabilities, positions = thresholds.out_of_fold_probabilities(
        lambda: strategy.build(plan, adapter, task, seed, selected.params),
        split.dev_features,
        split.dev_target,
        split.manifest.folds,
    )
    positive = fitted.classes_[1]
    actual = (split.dev_target.iloc[positions] == positive).to_numpy()
    if chartered is not None:
        return chartered_threshold(
            fitted, probabilities, actual, chartered, seed, len(positions)
        )
    kind = thresholds.OBJECTIVES[decision.threshold_objective]
    objective = (
        kind(decision.false_positive_cost, decision.false_negative_cost)
        if decision.threshold_objective == "cost"
        else kind()
    )
    threshold, score = thresholds.optimize_threshold(
        probabilities, actual, objective
    )
    description = {
        "objective": decision.threshold_objective,
        "threshold": threshold,
        "validation_score": score,
        "validation_rows": len(positions),
    }
    return thresholds.ThresholdClassifier.around_fitted(fitted, threshold), (
        description
    )


class FinalEvaluationStage(Stage):
    """Refits the frozen candidate and opens the locked test set once."""

    requires = (SelectStage,)
    state = run_state.RunState.TESTING

    def describe(self, context: context_module.RunContext) -> str:
        """Names the selected model.

        Args:
            context: The run context, holding the selected trial.

        Returns:
            The reason text naming the model that is evaluated.
        """
        selected = require(context.selected, "selected")
        return f"final one-time test evaluation of {selected.model_name}"

    @staticmethod
    def diagnostic_seconds(context: context_module.RunContext) -> float:
        """Time the post-selection diagnostics may use in total.

        A share of the time budget, never less than a minimum, so wide
        datasets cannot turn explanation into the slowest step of the run.

        Args:
            context: The run context, supplying the time budget.

        Returns:
            The total seconds the diagnostics may use: a fixed share of the time
            budget, but never less than the minimum.
        """
        budget = context.spec.resources.time_budget_seconds
        return max(DIAGNOSTIC_MIN_SECONDS, DIAGNOSTIC_BUDGET_SHARE * budget)

    def selected_health(
        self,
        context: context_module.RunContext,
        selected: experiments.TrialResult,
    ) -> dict[str, Any] | None:
        """Fit-health record of the selected trial, if it can be judged.

        Args:
            context: The run context, holding the trials and the metric.
            selected: The selected trial.

        Returns:
            The fit-health assessment as a dict, or None if the trial cannot be
            judged.
        """
        health = fit_health.assess_trial(
            selected,
            context.trials,
            require(context.metric, "metric").greater_is_better,
        )
        return None if health is None else health.to_dict()

    def mitigate_fairness(
        self,
        context: context_module.RunContext,
        selected: experiments.TrialResult,
        pipeline: Any,
        decision: dict[str, Any] | None,
    ) -> Any:
        """Gives each group its own threshold when that is judged worthwhile.

        The judgement uses out-of-fold development predictions only. The
        outcome (applied or not) is always written to the decision log.

        Args:
            context: The run context; receives the decision-log entry and the
                mitigation record.
            selected: The selected trial, used to rebuild the model for the
                out-of-fold predictions.
            pipeline: The fitted final model.
            decision: The chartered or default threshold description, or None;
                its threshold is the base cut of the groups (0.5 if None).

        Returns:
            A group-threshold classifier when applied, else `pipeline`.
        """
        fairness = context.spec.fairness
        task = require(context.task, "task")
        if fairness.mitigation != "group_thresholds":
            return pipeline
        column = fairness.column or context.spec.data.sensitive_columns[0]
        label = f"fairness:{column}"
        if task is not tasks.TaskType.BINARY:
            context.add_outcome(
                label,
                experiments.DecisionAction.REJECT,
                "group thresholds apply to binary classification only",
            )
            return pipeline
        split = require(context.split, "split")
        strategy = strategies.get_strategy(selected.strategy)
        adapter = context.registry.get(selected.model_name)
        plan = require(context.profile, "profile").feature_plan
        seed = context.spec.validation.seed
        probabilities, positions = thresholds.out_of_fold_probabilities(
            lambda: strategy.build(plan, adapter, task, seed, selected.params),
            split.dev_features,
            split.dev_target,
            split.manifest.folds,
        )
        actual = (
            split.dev_target.iloc[positions] == pipeline.classes_[1]
        ).to_numpy()
        assigner = thresholds.GroupAssigner.fit(split.dev_features[column])
        groups = assigner.assign(
            split.dev_features[column].iloc[positions]
        ).reset_index(drop=True)
        base_cut = decision["threshold"] if decision else 0.5
        result = mitigation.judge(
            probabilities,
            groups,
            actual,
            column,
            fairness.criterion,
            base_cut,
            fairness.trigger_gap,
            fairness.min_reduction,
            fairness.max_accuracy_cost,
        )
        context.add_outcome(
            label,
            experiments.DecisionAction.INCLUDE
            if result.applied
            else experiments.DecisionAction.REJECT,
            result.reason,
        )
        context.fairness_mitigation = result.to_dict()
        if not result.applied:
            return pipeline
        return thresholds.GroupThresholdClassifier(
            pipeline, column, assigner, result.thresholds, base_cut
        )

    def stream_refit(
        self,
        context: context_module.RunContext,
        pipeline: Any,
    ) -> Any:
        """Passes the whole development partition through an incremental model.

        The preprocessing stays as fitted on the development sample; the
        final model keeps learning from every development row, batch by
        batch, in one pass over the file.

        Args:
            context: The run context, holding the stream and settings.
            pipeline: The fitted final model.

        Returns:
            `pipeline`, updated in place when the data is streamed for training,
            otherwise unchanged.
        """
        if context.stream is None or context.spec.data.training != "stream":
            return pipeline
        source = context.stream[0]
        data, seed = context.spec.data, context.spec.validation.seed
        fraction = context.spec.validation.test_fraction
        preprocessing_steps, model = pipeline[:-1], pipeline.steps[-1][1]
        classes = getattr(pipeline, "classes_", None)
        offset, rows_seen = 0, 0
        for batch in source.batches():
            positions = np.arange(offset, offset + len(batch))
            offset += len(batch)
            side = streaming.side_of(positions, fraction, seed)
            part = batch[side == streaming.DEVELOPMENT]
            if part.empty:
                continue
            features = part.drop(columns=[data.target])
            matrix = preprocessing_steps.transform(features)
            if classes is None:
                model.partial_fit(matrix, part[data.target])
            else:
                model.partial_fit(matrix, part[data.target], classes=classes)
            rows_seen += len(part)
        context.emit(
            "training.streamed",
            f"final model updated on {rows_seen} streamed development rows",
            data={"rows": rows_seen},
        )
        return pipeline

    def streamed_locked_metrics(
        self,
        context: context_module.RunContext,
        pipeline: Any,
    ) -> dict[str, Any] | None:
        """Scores every locked row of a streamed file, batch by batch.

        Args:
            context: The run context, holding the stream and catalog.
            pipeline: The fitted final model.

        Returns:
            The number of locked rows, the total number of rows and the metrics
            on the locked rows, or None if the data was not streamed.
        """
        if context.stream is None:
            return None
        source, totals = context.stream
        data, seed = context.spec.data, context.spec.validation.seed
        fraction = context.spec.validation.test_fraction
        task = require(context.task, "task")
        labels, guesses, scores = [], [], []
        offset = 0
        for batch in source.batches():
            positions = np.arange(offset, offset + len(batch))
            offset += len(batch)
            side = streaming.side_of(positions, fraction, seed)
            part = batch[side == streaming.LOCKED]
            if part.empty:
                continue
            features = part.drop(columns=[data.target])
            labels.append(part[data.target].to_numpy())
            guesses.append(np.asarray(pipeline.predict(features)))
            if task.is_classification:
                scores.append(np.asarray(pipeline.predict_proba(features)))
        truth = pd.Series(np.concatenate(labels))
        replay_class = (
            replay.ClassifierPredictions
            if task.is_classification
            else replay.RegressorPredictions
        )
        stand_in = replay_class(
            np.concatenate(guesses),
            np.vstack(scores) if scores else None,
            getattr(pipeline, "classes_", None),
        )
        metrics = context.catalog.evaluate(
            stand_in, pd.DataFrame(index=truth.index), truth, task
        )
        return {
            "rows": len(truth),
            "of_total": totals.rows,
            "metrics": metrics,
        }

    def uplift_report(
        self,
        context: context_module.RunContext,
        fitted: Any,
        features: pd.DataFrame,
        outcome: pd.Series,
    ) -> dict[str, Any] | None:
        """Treatment-effect evidence on the released rows, with its caveat.

        Args:
            context: The run context, supplying the task, treatment column and
                propensity check.
            fitted: The fitted model whose predictions are the predicted effect.
            features: The released rows, including the treatment column.
            outcome: The observed outcome of those rows.

        Returns:
            The stated assumption, propensity check, average effect, mean
            predicted effect and a sampled Qini curve, or None if the task is
            not uplift.
        """
        if require(context.task, "task") is not tasks.TaskType.UPLIFT:
            return None
        arm = features[context.spec.data.treatment_column].to_numpy(dtype=int)
        predicted = np.asarray(fitted.predict(features), dtype=float)
        rows, curve = uplift.qini_curve(outcome.to_numpy(float), arm, predicted)
        keep = np.unique(np.linspace(0, len(rows) - 1, 50).astype(int))
        return {
            "assumption": uplift.ASSUMPTION,
            "propensity": context.propensity,
            "average_effect": uplift.average_effect(
                outcome.to_numpy(float), arm
            ),
            "mean_predicted_effect": float(predicted.mean()),
            "qini_curve": {
                "rows": rows[keep].tolist(),
                "incremental": curve[keep].tolist(),
            },
        }

    def mitigation_on_locked_rows(
        self,
        context: context_module.RunContext,
        before: Any,
        after: Any,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> None:
        """Adds the gap and accuracy on the released rows to the evidence.

        Does nothing unless a group-threshold mitigation was applied.

        Args:
            context: The run context, holding the mitigation record.
            before: The model without the group thresholds.
            after: The model with the group thresholds.
            features: The released rows.
            target: The labels of the released rows.
        """
        record = context.fairness_mitigation
        if record is None or not record["applied"]:
            return
        column = record["column"]
        assigner = after.assigner
        groups = assigner.assign(features[column]).reset_index(drop=True)
        positive = after.classes_[1]
        actual = (target == positive).to_numpy()
        flags = {
            name: np.asarray(model.predict(features)) == positive
            for name, model in (("before", before), ("after", after))
        }
        criterion = record["criterion"]
        record["locked_rows"] = {
            "gap_before": mitigation.gap(
                groups, flags["before"], actual, criterion
            ),
            "gap_after": mitigation.gap(
                groups, flags["after"], actual, criterion
            ),
            "accuracy_before": float((flags["before"] == actual).mean()),
            "accuracy_after": float((flags["after"] == actual).mean()),
            "rows": len(features),
        }

    def self_train(
        self,
        context: context_module.RunContext,
        selected: experiments.TrialResult,
        pipeline: Any,
    ) -> Any:
        """Adds confident pseudo-labels to the final fit when they help.

        The judgement uses only development folds. The unlabeled rows never
        touch the locked test rows.

        Args:
            context: The run context, holding the unlabeled rows and split.
            selected: The selected trial, used to build the template model.
            pipeline: The fitted final model.

        Returns:
            The pipeline to carry forward: the self-trained refit when
            cross-validation accepted it, otherwise `pipeline` unchanged.
        """
        data = context.spec.data
        unlabeled = context.unlabeled
        if data.unlabeled != "pseudo_label" or unlabeled is None:
            return pipeline
        split = require(context.split, "split")
        task = require(context.task, "task")
        metric = require(context.metric, "metric")
        template = strategies.get_strategy(selected.strategy).build(
            require(context.profile, "profile").feature_plan,
            context.registry.get(selected.model_name),
            task,
            context.spec.validation.seed,
            selected.params,
        )
        outcome = pseudo_labeling.judge(
            template,
            split.dev_features,
            split.dev_target,
            split.manifest.folds,
            unlabeled,
            metric.scorer(
                metric_catalog.positive_label(split.dev_target, task)
            ),
            data.pseudo_label_confidence,
            lambda: bool(context.resource_governor.skip_reason(first=False)),
        )
        record: dict[str, Any] = {
            "unlabeled_rows": len(unlabeled),
            "confidence": data.pseudo_label_confidence,
            "outcome": None if outcome is None else outcome.to_dict(),
            "used": False,
        }
        context.semi_supervised = record
        if outcome is None or not outcome.accepted:
            return pipeline
        augmented, count = pseudo_labeling.augmented_fit(
            pipeline,
            split.dev_features,
            split.dev_target,
            unlabeled,
            data.pseudo_label_confidence,
        )
        record["used"] = count > 0
        record["pseudo_rows"] = count
        return augmented

    def register_coverage_claim(
        self, context: context_module.RunContext
    ) -> None:
        """Fixes and hashes the coverage claim before the final rows open.

        Written to `coverage_claim.json`, announced by a `coverage.registered`
        event and, in a charter run, appended to the custody log. A task with
        no interval or set has no claim.

        Args:
            context: The run context that stores the claim and its hash.
        """
        split = require(context.split, "split")
        charter_hash = None
        if context.spec.charter_path:
            charter_hash = charter_sealing.read_verified(
                str(context.spec.charter_path)
            ).charter_hash
        claim = coverage_run.claim_document(
            task=require(context.task, "task").value,
            levels=COVERAGE_LEVELS,
            tolerance=coverage_module.DEFAULT_TOLERANCE,
            split_fingerprint=split.manifest.fingerprint,
            charter_hash=charter_hash,
            slice_columns=declared_slice_columns(context),
        )
        if claim is None:
            return
        sha = coverage_run.claim_hash(claim)
        context.write_json(coverage_run.CLAIM_FILE, claim)
        context.coverage_claim_sha256 = sha
        vault = context.custody_vault
        if vault is not None:
            vault.log.append(
                "final_test",
                coverage_run.REGISTRATION_EVENT,
                "scmle.coverage",
                {"sha256": sha},
            )
        context.emit(
            "coverage.registered",
            "coverage claim fixed before the final test rows are released",
            (coverage_run.CLAIM_FILE,),
            {"sha256": sha},
        )

    def register_decision_files(
        self, context: context_module.RunContext
    ) -> None:
        """Appends the hashes of the four decision files to the custody log.

        Does nothing without a custody vault or a stored run.

        Args:
            context: The run context, holding the vault and the run directory.
        """
        vault = context.custody_vault
        if vault is None or context.store is None:
            return
        vault.log.append(
            "final_test",
            decision_reports.REGISTRATION_EVENT,
            "scmle.decision",
            {"files": decision_reports.file_digests(context.store.directory)},
        )

    def write_decision_reports(
        self,
        context: context_module.RunContext,
        pipeline: Any,
        features: pd.DataFrame,
        target: pd.Series,
        decision: dict[str, Any] | None,
        net_benefit: dict[str, Any] | None,
    ) -> None:
        """Policy value, fairness satisfaction and the rollout design.

        Runs only when the charter has a decision block and the task is
        binary, once, on the locked rows that were just opened. It writes
        `policy_value.json`, `rollout_design.json`, `rollout_decisions.jsonl`
        and `decision_report.json`.

        Args:
            context: The run context that stores the reports.
            pipeline: The fitted final model.
            features: The released rows.
            target: The labels of the released rows.
            decision: The chartered threshold description, or None.
            net_benefit: The net-benefit report, or None.
        """
        charter = chartered_decision_model(context)
        if charter is None or charter.decision is None:
            return
        if require(context.task, "task") is not tasks.TaskType.BINARY:
            return
        layer = charter.decision
        data = context.spec.data
        positive = pipeline.classes_[1]
        outcome = (target == positive).to_numpy(dtype=float)
        actions = (np.asarray(pipeline.predict(features)) == positive).astype(
            float
        )
        logged = (
            features[data.logged_action_column].to_numpy(dtype=float)
            if data.logged_action_column
            and data.logged_action_column in features
            else None
        )
        propensity = (
            features[data.propensity_column].to_numpy(dtype=float)
            if data.propensity_column and data.propensity_column in features
            else None
        )
        value = decision_reports.policy_value(
            layer, actions, outcome, logged, propensity
        )
        satisfaction = decision_reports.fairness(
            layer,
            features,
            outcome,
            actions,
            float(charter.error_budget.confirmation.alpha_confirm),
        )
        design_record = None
        built = decision_reports.rollout_design(
            charter, float(np.mean(outcome)) if len(outcome) else 0.0
        )
        report = decision_reports.decision_report(
            charter,
            decision,
            net_benefit,
            value,
            satisfaction,
            None if built is None else built[0],
        )
        context.write_json(decision_reports.POLICY_FILE, value)
        written = ["decision_report.json", decision_reports.POLICY_FILE]
        if decision is not None:
            context.write_json(decision_reports.THRESHOLD_FILE, decision)
            written.append(decision_reports.THRESHOLD_FILE)
        if net_benefit is not None:
            context.write_json(decision_reports.NET_BENEFIT_FILE, net_benefit)
            written.append(decision_reports.NET_BENEFIT_FILE)
        if built is not None:
            design_record, log = built
            context.write_json("rollout_design.json", design_record)
            if context.store is not None:
                log.write(context.store.directory / "rollout_decisions.jsonl")
            written += ["rollout_design.json", "rollout_decisions.jsonl"]
        context.write_json("decision_report.json", report)
        self.register_decision_files(context)
        context.emit(
            "decision.reported",
            "policy value, fairness satisfaction and rollout design written",
            tuple(written),
            {
                "policy_value": value.get("status"),
                "fairness": [f["status"] for f in satisfaction],
            },
        )

    def execute(self, context: context_module.RunContext) -> None:
        """Refits on all development data and evaluates on the test set.

        Opens the locked rows once, after the selection is frozen, and
        records the test metrics, diagnostics and the evaluation document in
        the context.

        Args:
            context: The run context; receives the final model, the test rows
                and the evaluation.
        """
        context.check_cancelled()
        split = require(context.split, "split")
        selected = require(context.selected, "selected")
        task = require(context.task, "task")
        metric = require(context.metric, "metric")
        pipeline = context.refits.get(selected.trial_id)
        if pipeline is None:
            pipeline = strategies.get_strategy(selected.strategy).build(
                require(context.profile, "profile").feature_plan,
                context.registry.get(selected.model_name),
                task,
                context.spec.validation.seed,
                selected.params,
            )
            pipeline.fit(split.dev_features, split.dev_target)
        pipeline = self.self_train(context, selected, pipeline)
        pipeline = self.stream_refit(context, pipeline)
        pipeline, decision = choose_decision_threshold(context, pipeline)
        unmitigated = pipeline
        pipeline = self.mitigate_fairness(context, selected, pipeline, decision)
        self.register_coverage_claim(context)
        locked = split.locked_test
        test_features, test_target = locked.open(locked.issue_token())
        if (
            context.spec.data.weight_column == tasks.FAIRNESS_WEIGHT
            and tasks.FAIRNESS_WEIGHT not in test_features
        ):
            # Reweighing weights development rows only; locked rows carry a
            # neutral weight so the pipeline finds the column it expects.
            test_features = test_features.assign(**{tasks.FAIRNESS_WEIGHT: 1.0})
        context.test_rows = (test_features, test_target)
        self.mitigation_on_locked_rows(
            context, unmitigated, pipeline, test_features, test_target
        )
        context.emit(
            "test.opened",
            "locked test set released once, after selection was frozen",
            data={
                "rows": len(test_features),
                "fingerprint": split.manifest.fingerprint,
            },
        )
        baseline = next(
            (
                t
                for t in context.trials
                if t.model_name == BASELINE
                and t.status is experiments.TrialStatus.OK
            ),
            None,
        )
        context.final_pipeline = pipeline
        usage = resources.measure(pipeline, split.dev_features)
        test_metrics = context.catalog.evaluate(
            pipeline, test_features, test_target, task
        )
        weight_column = context.spec.data.weight_column
        weighted_test_metrics = (
            context.catalog.evaluate(
                pipeline,
                test_features,
                test_target,
                task,
                test_features[weight_column].to_numpy(dtype=float),
            )
            if weight_column and weight_column != tasks.FAIRNESS_WEIGHT
            else None
        )
        context.track(
            selected_strategy=selected.strategy,
            selected_model=selected.model_name,
            metric=metric.name,
            cv_score=selected.score_mean,
            test_metrics=test_metrics,
        )
        context.reference_scores = reference_scores(
            pipeline, test_features, task
        )
        context.diagnostics = diagnostics.run_suite(
            diagnostics.DiagnosticInput(
                pipeline,
                test_features,
                test_target,
                task,
                metric,
                require(context.profile, "profile").feature_plan,
                context.spec.validation.seed,
                groups=(
                    test_features[context.spec.data.group_column]
                    if (
                        split.manifest.strategy == "group"
                        or context.cluster_column
                    )
                    and context.spec.data.group_column
                    else None
                ),
                temporal=split.manifest.strategy == "temporal",
                reference_features=split.dev_features,
                sensitive_columns=tuple(context.spec.data.sensitive_columns),
            ),
            suite.full_suite(),
            self.diagnostic_seconds(context),
        )
        for name, report_body in context.diagnostics.items():
            context.write_json(f"diagnostics/{name}.json", report_body)
        net_benefit = chartered_net_benefit(
            context, pipeline, test_features, test_target, decision
        )
        self.write_decision_reports(
            context, pipeline, test_features, test_target, decision, net_benefit
        )
        context.evaluation = {
            "run_id": context.store.run_id if context.store else "",
            "task": task.value,
            "metric": metric.name,
            "metric_greater_is_better": metric.greater_is_better,
            "selection_policy": context.policy.description,
            "selected_model": selected.model_name,
            "selected_trial_id": selected.trial_id,
            "selected_params": selected.params,
            "selected_strategy": selected.strategy,
            "selected_cv_score": selected.score_mean,
            "selected_cv_std": selected.score_std,
            "fit_health": self.selected_health(context, selected),
            "cv_confidence": confidence.cv_interval(
                selected.fold_scores,
                float(np.mean([len(fold[0]) for fold in split.manifest.folds])),
                float(np.mean([len(fold[1]) for fold in split.manifest.folds])),
                split.manifest.strategy,
            ),
            "baseline_cv_score": baseline.score_mean if baseline else None,
            "test_metrics": test_metrics,
            "weighted_test_metrics": weighted_test_metrics,
            "semi_supervised": context.semi_supervised,
            "fairness_mitigation": context.fairness_mitigation,
            "execution_backend": context.backend.name,
            "pareto_front": context.pareto_front,
            "evaluation_prevalence": {
                "natural": True,
                "note": "resampling strategies change only training folds; "
                "validation and test rows keep their real class mix",
                "strategy": selected.strategy,
            },
            "success_criteria": [
                r.to_dict()
                for r in criteria.check(
                    context.spec.problem.success_criteria,
                    test_metrics,
                    context.catalog,
                )
            ],
            "streamed_locked": self.streamed_locked_metrics(context, pipeline),
            "uplift": self.uplift_report(
                context, pipeline, test_features, test_target
            ),
            "n_dev": len(split.dev_features),
            "n_test": len(test_features),
            "split_fingerprint": split.manifest.fingerprint,
            "split_strategy": split.manifest.strategy,
            "decision": decision,
            "net_benefit": net_benefit,
            "resources": usage.to_dict(),
            "resource_limits": context.resource_governor.report(),
            "test_set_opened": locked.opened,
            "diagnostics": sorted(context.diagnostics),
            "elapsed_seconds": time.monotonic() - context.started_at,
            "reproducibility": (
                "same seed/data/environment replays; threaded backends may "
                "differ"
            ),
        }


class PackageStage(Stage):
    """Writes the model bundle, leaderboard and report."""

    requires = (FinalEvaluationStage,)
    state = run_state.RunState.PACKAGING

    def evidence_index(
        self,
        context: context_module.RunContext,
        store: Any,
        policy: evaluation_contracts.EvaluationPolicy,
    ) -> evidence_contracts.EvidenceIndex:
        """Collects accepted evidence for artifacts already written.

        Acceptance is deliberately narrow: an automated record is accepted
        only when the referenced artifact exists and its digest is captured.
        The evidence limitation records that an independent human review is
        not represented by this automated run.

        Args:
            context: The run context, supplying the manifest, task and
                configuration hash.
            store: The run store holding the artifacts.
            policy: The evaluation policy, naming the evidence references that
                a release requires.

        Returns:
            An index with one accepted record for each required reference whose
            artifact exists; missing artifacts are omitted.
        """
        manifest = require(context.manifest, "manifest")
        raw_inputs = {
            "snapshot_id": manifest.snapshot_id or manifest.data_hash,
            "config_hash": context.spec.config_hash(),
        }
        records: list[evidence_contracts.EvidenceRecord] = []
        for reference in release.required_evidence_references(policy):
            path = store.path(reference)
            if not path.exists():
                continue
            records.append(
                evidence_contracts.EvidenceRecord(
                    criterion_id=reference,
                    claim=(
                        f"artifact exists and matches run output: {reference}"
                    ),
                    scope={
                        "task": require(context.task, "task").value,
                        "modality": manifest.modality,
                    },
                    raw_inputs=raw_inputs,
                    command_or_test=(
                        "scmle automated release evidence collector"
                    ),
                    artifact={
                        "path": reference,
                        "sha256": bundle_module.file_sha256(path),
                    },
                    result=evidence_contracts.EvidenceResult.PASS,
                    reviewer="scmle.automation",
                    observed_at=datetime.datetime.now(datetime.UTC).isoformat(),
                    limitations=(
                        "automated artifact-presence and digest check; "
                        "independent human review is not represented",
                    ),
                    decision=evidence_contracts.EvidenceDecision.ACCEPTED,
                )
            )
        return evidence_contracts.EvidenceIndex(records)

    def release_evaluation(
        self,
        context: context_module.RunContext,
        store: Any,
        evaluation: dict[str, Any],
    ) -> None:
        """Evaluates explicit success criteria against release evidence.

        Writes the release policy, evidence and evaluation files into the run
        store and records the results in `evaluation`. When no success criteria
        were declared, the evaluation is blocked with status
        `policy_not_declared`; thresholds are never invented from observed
        metrics.

        Args:
            context: The run context, holding the task, metric and diagnostics.
            store: The run store receiving the files.
            evaluation: The evaluation document; the release policy,
                evaluation and evidence entries are added to it.
        """
        task = require(context.task, "task")
        metric = require(context.metric, "metric")
        policy = policies.release_policy_from_criteria(
            task,
            metric.name,
            metric.greater_is_better,
            context.spec.problem.success_criteria,
        )
        if policy is None:
            record = {
                "status": "policy_not_declared",
                "passed": False,
                "blocked": True,
                "blocking_reasons": [
                    "no explicit success criteria were declared; release "
                    "thresholds were not invented from observed metrics",
                ],
                "findings": [],
            }
            evaluation["release_policy"] = None
            evaluation["release_evaluation"] = record
            evaluation["release_evidence"] = {"records": [], "ready": False}
            store.write_json("release_evaluation.json", record)
            return

        evidence_index = self.evidence_index(context, store, policy)
        signing_key = attestation.resolve_key(None)
        if signing_key is not None:
            evidence_index.sign(signing_key)
        declared_metrics = {metric.name for metric in policy.metrics}
        release_metrics = {
            name: value
            for name, value in evaluation["test_metrics"].items()
            if name in declared_metrics
        }
        outcome = release.evaluate_release(
            policy,
            release_metrics,
            context.diagnostics,
            {"fairness": context.diagnostics.get("fairness", {})},
            evidence_index,
        )
        outcome = release.apply_reliability_gate(
            outcome, evaluation.get("reliability_plan")
        )
        verification = False
        try:
            release_attestation = attestation.ReleaseAttestation.from_dict(
                evaluation["release_attestation"]
            )
            verification = release_attestation.verify(
                evaluation["provenance"],
                evaluation["sbom"],
                signing_key=attestation.resolve_key(None),
            )
        except (KeyError, TypeError, ValueError):
            verification = False
        outcome = release.apply_attestation_gate(
            outcome, evaluation.get("release_attestation"), verification
        )
        evidence_verified = signing_key is not None and evidence_index.verify(
            signing_key
        )
        outcome = release.apply_evidence_index_gate(
            outcome, evidence_index, evidence_verified
        )
        evaluation["release_policy"] = policy.to_dict()
        evaluation["release_evaluation"] = outcome.to_dict()
        evaluation["release_evidence"] = evidence_index.to_dict()
        store.write_json("evaluation_policy.json", policy.to_dict())
        store.write_json("release_evidence.json", evidence_index.to_dict())
        store.write_json("release_evaluation.json", outcome.to_dict())

    def explanation_evidence(
        self,
        context: context_module.RunContext,
        store: Any,
    ) -> dict[str, dict[str, Any]]:
        """Persist portable evidence for explanations produced by the run.

        Args:
            context: The run context, holding the diagnostics, manifest and
                split.
            store: The run store; the evidence is written under `explanations/`.

        Returns:
            A mapping from diagnostic name to its evidence record, for each
            explanation diagnostic that ran without error and has a report.
        """
        manifest = require(context.manifest, "manifest")
        split = require(context.split, "split")
        model_manifest = store.path(f"model/{bundle_module.MANIFEST_FILE}")
        bundle_digest = bundle_module.file_sha256(model_manifest)
        snapshot = explanations.DatasetSnapshot(
            identifier=manifest.dataset_id,
            version=manifest.snapshot_id or manifest.data_hash,
            digest=manifest.data_hash,
        )
        feature_graph = explanations.FeatureGraphReference(
            version=context.spec.feature_graph.version,
            fingerprint=context.spec.feature_graph.fingerprint(),
        )
        diagnostics_by_name = {
            diagnostic.name: diagnostic
            for diagnostic in explanation_suite.default_suite()
        }
        evidence: dict[str, dict[str, Any]] = {}
        for name, diagnostic in diagnostics_by_name.items():
            result = context.diagnostics.get(name)
            if result is None or "error" in result:
                continue
            report_path = store.path(f"diagnostics/{name}.json")
            if not report_path.exists():
                continue
            limitations = [
                "associational evidence is not causal evidence",
                "stability and fidelity require separate validation",
            ]
            if result.get("truncated"):
                limitations.append("feature coverage was truncated by budget")
            record = explanation_suite.ExplanationEvidenceBuilder().build(
                diagnostic,
                result,
                model_fingerprint=bundle_digest,
                bundle_digest=bundle_digest,
                input_fingerprint=split.manifest.fingerprint,
                dataset_snapshot=snapshot,
                feature_graph=feature_graph,
                method_parameters={"seed": context.spec.validation.seed},
                timestamp=datetime.datetime.now(datetime.UTC),
                software_version=scmle.__version__,
                uncertainty={
                    "status": "not_estimated",
                    "reason": "explanation evidence does not estimate "
                    "prediction uncertainty",
                },
                stability={
                    "status": "not_estimated",
                    "reason": "repeatability must be evaluated separately",
                },
                limitations=tuple(limitations),
                causal_warning=(
                    "This explanation describes model associations; it does "
                    "not establish that changing a feature causes an outcome."
                ),
                privacy_decision=explanations.PrivacyDecision(
                    decision="allowed",
                    reason=(
                        "diagnostic output contains aggregate feature evidence"
                    ),
                ),
                evidence_links=(
                    explanations.EvidenceLink(
                        relation="diagnostic-report",
                        uri=f"diagnostics/{name}.json",
                        digest=bundle_module.file_sha256(report_path),
                    ),
                ),
            )
            evidence[name] = record.to_dict()
            store.write_json(f"explanations/{name}.json", record.to_dict())
            store.write_text(
                f"explanations/{name}.html",
                explanation_report.ExplanationHtmlReport().render(
                    record, title=f"{name} explanation"
                ),
            )
        return evidence

    def provenance_record(
        self,
        context: context_module.RunContext,
        store: Any,
        bundle: bundle_module.ModelBundle,
    ) -> dict[str, Any]:
        """Builds a content-linked provenance record for the packaged run.

        The record is also written to `provenance.json`.

        Args:
            context: The run context, supplying the manifest and configuration.
            store: The run store, supplying the run id and model manifest.
            bundle: The packaged model bundle, supplying the model and
                environment information.

        Returns:
            The provenance record: dataset, model manifest digest, environment
            and signature presence.
        """
        manifest = require(context.manifest, "manifest")
        model_manifest = store.path(f"model/{bundle_module.MANIFEST_FILE}")
        record = {
            "schema_version": 1,
            "run_id": store.run_id,
            "software_version": scmle.__version__,
            "config_hash": context.spec.config_hash(),
            "dataset": {
                "snapshot_id": manifest.snapshot_id or manifest.data_hash,
                "dataset_id": manifest.dataset_id,
                "data_hash": manifest.data_hash,
                "raw_source_hash": manifest.lineage.object_hashes.get(
                    "raw-source"
                ),
            },
            "model": {
                "manifest_path": f"model/{bundle_module.MANIFEST_FILE}",
                "manifest_sha256": bundle_module.file_sha256(model_manifest),
                "model_name": bundle.info.model_name,
                "model_version": bundle.info.model_version,
            },
            "environment": {
                "python": bundle.info.python,
                "platform": bundle.info.platform,
                "libraries": bundle.info.libraries,
                "lock": bundle.info.environment_lock(),
            },
            "signature": {
                "path": f"model/{bundle_module.SIGNATURE_FILE}",
                "present": store.path(
                    f"model/{bundle_module.SIGNATURE_FILE}"
                ).exists(),
            },
        }
        store.write_json("provenance.json", record)
        return record

    def sbom_record(
        self,
        context: context_module.RunContext,
        store: Any,
        bundle: bundle_module.ModelBundle,
    ) -> dict[str, Any]:
        """Builds and persists the packaged environment's SBOM.

        Args:
            context: The run context; not consulted.
            store: The run store receiving `sbom.json`.
            bundle: The packaged model bundle, supplying the environment.

        Returns:
            The software bill of materials.
        """
        record = sbom.build(
            bundle.info.environment,
            run_id=store.run_id,
            software_version=scmle.__version__,
        )
        store.write_json("sbom.json", record)
        return record

    def reliability_plan_record(
        self,
        context: context_module.RunContext,
        store: Any,
    ) -> dict[str, Any]:
        """Persist the complete, unexecuted production fault plan.

        Args:
            context: The run context; not consulted.
            store: The run store receiving `fault_injection_plan.json`.

        Returns:
            The fault-injection plan of the production beta, marked as not
            executed with the release blocked, since the fault plan declares
            required evidence and does not claim that the scenarios ran.
        """
        del context
        record = (
            reliability.FaultInjectionPlanRegistry.production_beta().to_dict()
        )
        record["execution_status"] = "planned"
        record["release_blocked"] = True
        record["limitation"] = (
            "The plan declares required fault evidence; it does not claim "
            "that the scenarios have executed."
        )
        store.write_json("fault_injection_plan.json", record)
        return record

    def describe(self, context: context_module.RunContext) -> str:
        """Names the stage.

        Args:
            context: The run context; not consulted.

        Returns:
            The text "write bundle and reports".
        """
        return "write bundle and reports"

    def learning_curve(
        self,
        context: context_module.RunContext,
        selected: experiments.TrialResult,
    ) -> dict[str, Any] | None:
        """Learning curve of the selected candidate on its first fold.

        Args:
            context: The run context, holding the split, metric and budget.
            selected: The selected trial, used to build the model.

        Returns:
            The curve as a record, or None when the budget is spent or the
            folds are too small.
        """
        if context.resource_governor.skip_reason(first=False):
            return None
        split = require(context.split, "split")
        template = strategies.get_strategy(selected.strategy).build(
            require(context.profile, "profile").feature_plan,
            context.registry.get(selected.model_name),
            require(context.task, "task"),
            context.spec.validation.seed,
            selected.params,
        )
        metric = require(context.metric, "metric")
        fit_positions, evaluate_positions = split.manifest.folds[0]
        curve = learning_curve.compute(
            template,
            split.dev_features,
            split.dev_target,
            fit_positions,
            evaluate_positions,
            metric.scorer(
                metric_catalog.positive_label(
                    split.dev_target, require(context.task, "task")
                )
            ),
            metric.greater_is_better,
            lambda: bool(context.resource_governor.skip_reason(first=False)),
        )
        return None if curve is None else curve.to_dict()

    def note_conformal_skipped(
        self, context: context_module.RunContext, reason: str
    ) -> None:
        """Records why the bundle will have no prediction intervals or sets.

        Args:
            context: The run context that receives the event.
            reason: Why calibration was skipped.
        """
        context.emit(
            "conformal.skipped",
            f"conformal calibration skipped: {reason}",
            data={"reason": reason},
        )

    def conformal_calibration(
        self,
        context: context_module.RunContext,
        selected: experiments.TrialResult,
        bundle: bundle_module.ModelBundle,
    ) -> conformal.ConformalCalibrator | None:
        """Calibrates intervals or sets from out-of-fold predictions.

        The selected candidate is refit on each cross-validation fold's
        training rows and scored on its held-out rows, so every calibration
        score comes from a model that never saw that row.

        Args:
            context: The run context, holding the split, budget and charter.
            selected: The selected trial, used to build the fold models.
            bundle: The model bundle; the fold models are kept on it when the
                CV+ rule applies.

        Returns:
            The calibrator, or None when the task has no conformal margin, the
            time budget is spent or too few held-out rows exist.
        """
        task = require(context.task, "task")
        if task in (
            tasks.TaskType.SURVIVAL,
            tasks.TaskType.ORDINAL,
            tasks.TaskType.RANKING,
            tasks.TaskType.MULTILABEL,
            tasks.TaskType.MULTIOUTPUT,
            tasks.TaskType.FORECAST,
            tasks.TaskType.UPLIFT,
        ):
            return None  # Conformal margins need point predictions or labels.
        reason = context.resource_governor.skip_reason(first=False)
        if reason:
            self.note_conformal_skipped(context, reason)
            return None
        split = require(context.split, "split")
        template = strategies.get_strategy(selected.strategy).build(
            require(context.profile, "profile").feature_plan,
            context.registry.get(selected.model_name),
            task,
            context.spec.validation.seed,
            selected.params,
        )
        held_labels: list[Any] = []
        held_scores: list[Any] = []
        held_entities: list[Any] = []
        fold_models: list[Any] = []
        classes = list(bundle.classes or ())
        entity_column = self.entity_calibration_column(context)
        for fit_positions, evaluate_positions in split.manifest.folds:
            reason = context.resource_governor.skip_reason(first=False)
            if reason:
                self.note_conformal_skipped(context, reason)
                return None
            fit = np.asarray(fit_positions)
            evaluate = np.asarray(evaluate_positions)
            candidate = base.clone(template).fit(
                split.dev_features.iloc[fit], split.dev_target.iloc[fit]
            )
            rows = split.dev_features.iloc[evaluate]
            held_labels.extend(split.dev_target.iloc[evaluate])
            if entity_column is not None:
                held_entities.extend(rows[entity_column].astype(str))
            if task is tasks.TaskType.REGRESSION:
                held_scores.extend(candidate.predict(rows))
                fold_models.append(candidate)
                continue
            known = list(candidate.classes_)
            probabilities = candidate.predict_proba(rows)
            held_scores.extend(
                [
                    probabilities[i, known.index(c)] if c in known else 0.0
                    for c in classes
                ]
                for i in range(len(rows))
            )
        if entity_column is not None:
            chosen = conformal.one_row_per_entity(
                held_entities,
                randomness.component_rng(
                    context.spec.validation.seed, "conformal.entity", 0
                ),
            )
            held_labels = [held_labels[i] for i in chosen]
            held_scores = [held_scores[i] for i in chosen]
            fold_models = []
            context.emit(
                "conformal.entity_wise",
                f"calibrated on one score for each of {len(chosen)} "
                "entities (unit_of_prediction: new_entity)",
                data={"entities": len(chosen)},
            )
        if len(held_labels) < conformal.MIN_CALIBRATION_ROWS:
            self.note_conformal_skipped(
                context,
                f"only {len(held_labels)} "
                f"{'entities' if entity_column else 'held-out rows'}; need "
                f"{conformal.MIN_CALIBRATION_ROWS}",
            )
            return None
        if task is tasks.TaskType.REGRESSION:
            self.keep_cv_plus(
                context,
                bundle,
                fold_models,
                np.asarray(held_labels, dtype=float),
                np.asarray(held_scores, dtype=float),
            )
            return conformal.ConformalCalibrator.for_regression(
                held_labels, held_scores
            )
        return conformal.ConformalCalibrator.for_classification(
            held_labels, np.asarray(held_scores), classes
        )

    def entity_calibration_column(
        self, context: context_module.RunContext
    ) -> str | None:
        """The group column when the charter's unit is a new entity.

        Intervals for a new entity treat one score per entity as the
        exchangeable unit, so calibration keeps one random held-out
        row of each entity. Returns None without a charter or for any other
        unit, and the row-wise calibration applies.

        Args:
            context: The run context, supplying the charter path and data spec.

        Returns:
            The group column name, or None when row-wise calibration applies.
        """
        path = context.spec.charter_path
        if not path:
            return None
        charter = charter_sealing.read_verified(str(path)).charter
        if charter.unit_of_prediction != charter_contracts.NEW_ENTITY:
            return None
        return context.spec.data.group_column

    def keep_cv_plus(
        self,
        context: context_module.RunContext,
        bundle: bundle_module.ModelBundle,
        models: list[Any],
        observed: np.ndarray,
        predicted: np.ndarray,
    ) -> None:
        """Stores the fold models for CV+ intervals when the rule applies.

        CV+ needs each calibration row held out in exactly one fold; with
        `MIN_ROWS_FOR_SPLIT` or more rows the pooled margin is used instead,
        and a model set above `MAX_BYTES` is not kept. The reason is
        emitted either way.

        Args:
            context: The run context, holding the split.
            bundle: The model bundle that receives the ensemble.
            models: The fold models, one per cross-validation fold.
            observed: The held-out labels.
            predicted: The held-out predictions.
        """
        split = require(context.split, "split")
        held = [fold[1] for fold in split.manifest.folds]
        plan = cv_plus.plan_scores(
            held,
            len(split.dev_features),
            len(observed),
            models,
            self.cv_plus_cap(context),
        )
        context.emit(
            "conformal.scores",
            f"calibration scores are {plan.mode.replace('_', ' ')}",
            data={"source": plan.mode, "rows": len(observed)},
        )
        if plan.folds is None:
            context.emit(
                "cv_plus.skipped",
                f"CV+ ensemble not stored: {plan.reason}",
                data={"reason": plan.reason},
            )
            return
        bundle.cv_plus = cv_plus.CvPlusEnsemble(
            tuple(models), plan.folds, np.abs(observed - predicted)
        )

    def complete_held_ledger(
        self,
        context: context_module.RunContext,
        bundle: bundle_module.ModelBundle,
    ) -> None:
        """Opens `C` once the bundle and its calibration are frozen.

        Only for a charter that asked for the coverage claim on `C`: the
        ledger was held back, and now registers the coverage claim with the
        others and tests everything in the one opening.

        Args:
            context: The run context, holding the pending ledger.
            bundle: The frozen model bundle, whose conformal calibration the
                coverage claim tests.
        """
        pending = context.pending_ledger
        if pending is None:
            return
        context.pending_ledger = None
        task = require(context.task, "task")
        runner = None
        if context.coverage_claim_sha256 is not None:
            runner = coverage_confirm.build_runner(
                bundle, task.value, context.coverage_claim_sha256
            )
        if runner is None:
            context.write_json(
                coverage_confirm.DOCUMENT,
                coverage_confirm.not_evaluated(
                    "the bundle holds no conformal calibration, so there is "
                    "no interval or set to test on C"
                ),
            )
        ProfileStage().complete_claim_ledger(
            context, pending, pending.subject, runner
        )

    def cv_plus_cap(self, context: context_module.RunContext) -> int:
        """The most bytes of fold models to keep (charter's, else default).

        Args:
            context: The run context, supplying the charter path.

        Returns:
            The byte limit from the charter's uncertainty block, or the default
            limit if the charter sets none.
        """
        path = context.spec.charter_path
        if path:
            block = charter_sealing.read_verified(str(path)).charter.uncertainty
            if block is not None and block.cv_plus_max_bytes is not None:
                return block.cv_plus_max_bytes
        return cv_plus.MAX_BYTES

    def quantile_intervals(
        self, context: context_module.RunContext
    ) -> quantile.QuantileIntervals | None:
        """Fits and calibrates the low and high quantile models (regression).

        Both models are boosted trees on the standard preprocessing. Each
        cross-validation fold fits them on its training rows and records how
        far its held-out rows fall outside the band; the final pair is fit on
        all development rows.

        Args:
            context: The run context, holding the split, plan and budget.

        Returns:
            The intervals, or None for classification, when the time budget
            is spent or too few held-out rows exist.
        """
        task = require(context.task, "task")
        if task is not tasks.TaskType.REGRESSION:
            return None
        if context.resource_governor.skip_reason(first=False):
            return None
        split = require(context.split, "split")
        plan = require(context.profile, "profile").feature_plan
        adapter = context.registry.get("hist_gradient_boosting")
        seed = context.spec.validation.seed

        def build(level: float) -> Any:
            return preprocessing.build_pipeline(
                plan,
                adapter,
                task,
                seed,
                {"loss": "quantile", "quantile": level},
            )

        observed: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for fit_positions, evaluate_positions in split.manifest.folds:
            if context.resource_governor.skip_reason(first=False):
                return None
            fit = np.asarray(fit_positions)
            evaluate = np.asarray(evaluate_positions)
            train_x = split.dev_features.iloc[fit]
            train_y = split.dev_target.iloc[fit]
            low, high = quantile.predicted_band(
                build(quantile.LOWER_LEVEL).fit(train_x, train_y),
                build(quantile.UPPER_LEVEL).fit(train_x, train_y),
                split.dev_features.iloc[evaluate],
            )
            observed.extend(split.dev_target.iloc[evaluate])
            lows.extend(low)
            highs.extend(high)
        if len(observed) < conformal.MIN_CALIBRATION_ROWS:
            return None
        return quantile.QuantileIntervals(
            build(quantile.LOWER_LEVEL).fit(
                split.dev_features, split.dev_target
            ),
            build(quantile.UPPER_LEVEL).fit(
                split.dev_features, split.dev_target
            ),
            quantile.QuantileBand.calibrate(observed, lows, highs),
        )

    def bias_variance(
        self,
        context: context_module.RunContext,
        selected: experiments.TrialResult,
    ) -> dict[str, Any] | None:
        """Splits the selected model's error into bias and variance.

        Args:
            context: The run context, holding the split and settings.
            selected: The selected trial.

        Returns:
            The decomposition; a record with a `skipped` reason when the task is
            not binary, multiclass or regression, or the budget was spent; None
            when no replicates are configured.
        """
        replicates = context.spec.selection.bias_variance_replicates
        task = require(context.task, "task")
        if not replicates:
            return None
        if task not in (
            tasks.TaskType.BINARY,
            tasks.TaskType.MULTICLASS,
            tasks.TaskType.REGRESSION,
        ):
            return {"skipped": f"not defined for {task.value}"}
        split = require(context.split, "split")
        adapter = context.registry.get(selected.model_name)
        strategy = strategies.get_strategy(selected.strategy)
        plan = require(context.profile, "profile").feature_plan
        seed = context.spec.validation.seed
        found = bias_variance.decompose(
            lambda: strategy.build(plan, adapter, task, seed, selected.params),
            split.dev_features,
            split.dev_target,
            task.is_classification,
            replicates,
            seed,
            lambda: bool(context.resource_governor.skip_reason(first=False)),
        )
        if found is None:
            return {"skipped": "time budget spent or too few predictions"}
        return found.to_dict()

    def sensitivity(
        self,
        context: context_module.RunContext,
        selected: experiments.TrialResult,
    ) -> dict[str, Any] | None:
        """Spread of the selected model's score over training, split, noise.

        Args:
            context: The run context, holding the split and settings.
            selected: The selected trial.

        Returns:
            The spread report with its dominant source; a record with a
            `skipped` reason for other tasks or when the budget was spent; None
            when no repeats are configured.
        """
        repeats = context.spec.selection.sensitivity_repeats
        task = require(context.task, "task")
        if not repeats:
            return None
        if task not in (
            tasks.TaskType.BINARY,
            tasks.TaskType.MULTICLASS,
            tasks.TaskType.REGRESSION,
        ):
            return {"skipped": f"not defined for {task.value}"}
        if context.resource_governor.skip_reason(first=False):
            return {"skipped": "time budget spent"}
        split = require(context.split, "split")
        adapter = context.registry.get(selected.model_name)
        strategy = strategies.get_strategy(selected.strategy)
        plan = require(context.profile, "profile").feature_plan
        seed = context.spec.validation.seed
        metric = require(context.metric, "metric")
        scorer = metric.scorer(
            metric_catalog.positive_label(split.dev_target, task)
        )
        report = sensitivity.analyze(
            lambda: strategy.build(plan, adapter, task, seed, selected.params),
            split.dev_features,
            split.dev_target,
            scorer,
            repeats,
            seed,
        )
        return {**report.to_dict(), "dominant": report.dominant()}

    def nested_validation(
        self,
        context: context_module.RunContext,
        selected: experiments.TrialResult,
    ) -> dict[str, Any] | None:
        """Estimates the optimism of the winning score by nested validation.

        Args:
            context: The run context, holding the split and settings.
            selected: The selected trial.

        Returns:
            The nested-score record, with the mean and optimism in the metric's
            natural direction; a record with a `skipped` reason when the
            validation strategy is not random or stratified, the budget was
            spent or there were too few rows; None when nested validation is
            off.
        """
        if not context.spec.selection.nested_cv:
            return None
        split = require(context.split, "split")
        if split.manifest.strategy not in {"random", "stratified"}:
            return {
                "skipped": f"not defined for {split.manifest.strategy} "
                "validation"
            }
        if context.resource_governor.skip_reason(first=False):
            return {"skipped": "time budget spent"}
        task = require(context.task, "task")
        adapter = context.registry.get(selected.model_name)
        strategy = strategies.get_strategy(selected.strategy)
        plan = require(context.profile, "profile").feature_plan
        seed = context.spec.validation.seed
        metric = require(context.metric, "metric")
        positive = metric_catalog.positive_label(split.dev_target, task)
        result = nested.nested_score(
            lambda params: strategy.build(plan, adapter, task, seed, params),
            adapter.search_space,
            split.dev_features,
            split.dev_target,
            [(f[0], f[1]) for f in split.manifest.folds],
            metric.scorer(positive),
            task.is_classification,
            seed,
            should_stop=lambda: bool(
                context.resource_governor.skip_reason(first=False)
            ),
        )
        if result is None:
            return {"skipped": "time budget spent or too few rows"}
        sign = 1.0 if metric.greater_is_better else -1.0
        record = result.to_dict()
        record["mean_natural"] = sign * result.mean
        record["optimism_natural"] = sign * result.optimism
        return record

    def forecast_backtest(
        self,
        context: context_module.RunContext,
        selected: experiments.TrialResult,
    ) -> dict[str, Any] | None:
        """Per-step errors against baselines, and one row per rolling origin.

        The locked rows give the headline comparison. Each cross-validation
        fold (an origin) refits the selected candidate on its past and scores
        the future that followed.

        Args:
            context: The run context, holding the split and test rows.
            selected: The selected trial.

        Returns:
            The report, or None when the time budget is spent.
        """
        if context.resource_governor.skip_reason(first=False):
            return None
        split = require(context.split, "split")
        data = context.spec.data
        template = strategies.get_strategy(selected.strategy).build(
            require(context.profile, "profile").feature_plan,
            context.registry.get(selected.model_name),
            require(context.task, "task"),
            context.spec.validation.seed,
            selected.params,
        )
        fitted = []
        for fit_positions, evaluate_positions in split.manifest.folds:
            if context.resource_governor.skip_reason(first=False):
                return None
            fit = np.asarray(fit_positions)
            held = np.asarray(evaluate_positions)
            model = base.clone(template).fit(
                split.dev_features.iloc[fit], split.dev_target.iloc[fit]
            )
            fitted.append(
                (
                    model,
                    split.dev_features.iloc[held],
                    split.dev_target.iloc[held],
                )
            )
        headline = None
        if context.test_rows is not None:
            features, target = context.test_rows
            headline = backtest.report(
                context.final_pipeline,
                features,
                target,
                context.seasonal_baseline,
            )
        return {
            "steps": headline,
            "origins": backtest.origins(fitted),
            "seasonal_period": data.seasonal_period,
        }

    def build_evidence_graph(
        self,
        context: context_module.RunContext,
        store: Any,
    ) -> lineage.EvidenceGraph:
        """Builds the content-linked graph for one completed package.

        Args:
            context: The run context, supplying the manifest, split and trial.
            store: The run store holding the artifacts that are hashed.

        Returns:
            The graph from the raw source through the model and the reports.
        """
        manifest = require(context.manifest, "manifest")
        split = require(context.split, "split")
        selected = require(context.selected, "selected")
        graph = lineage.EvidenceGraph()

        raw_digest = manifest.lineage.object_hashes.get(
            "raw-source", manifest.data_hash
        )
        graph.add_node(
            lineage.EvidenceNode(
                "raw-source",
                lineage.EvidenceNodeKind.RAW_SOURCE,
                raw_digest,
                {"uri": manifest.uri},
            )
        )
        graph.add_node(
            lineage.EvidenceNode(
                "snapshot",
                lineage.EvidenceNodeKind.SNAPSHOT,
                manifest.snapshot_id or manifest.data_hash,
                {"dataset_id": manifest.dataset_id},
            )
        )
        graph.add_edge(
            lineage.EvidenceEdge("raw-source", "snapshot", "materializes")
        )
        graph.add_node(
            lineage.EvidenceNode(
                "curation",
                lineage.EvidenceNodeKind.CURATION,
                bundle_module.file_sha256(store.path("curation_report.json")),
                {"artifact": "curation_report.json"},
            )
        )
        graph.add_edge(lineage.EvidenceEdge("snapshot", "curation", "curated"))

        if context.spec.feature_graph.nodes:
            graph.add_node(
                lineage.EvidenceNode(
                    "feature-graph",
                    lineage.EvidenceNodeKind.FEATURE_GRAPH,
                    context.spec.feature_graph.fingerprint(),
                    {"version": str(context.spec.feature_graph.version)},
                )
            )
            graph.add_edge(
                lineage.EvidenceEdge("curation", "feature-graph", "transforms")
            )
            split_parent = "feature-graph"
        else:
            split_parent = "curation"

        graph.add_node(
            lineage.EvidenceNode(
                "split",
                lineage.EvidenceNodeKind.SPLIT,
                split.manifest.fingerprint,
                {"artifact": "validation_manifest.json"},
            )
        )
        graph.add_edge(
            lineage.EvidenceEdge(split_parent, "split", "partitions")
        )

        pipeline_path = store.path("model/pipeline.joblib")
        model_manifest_path = store.path(f"model/{bundle_module.MANIFEST_FILE}")
        graph.add_node(
            lineage.EvidenceNode(
                "model",
                lineage.EvidenceNodeKind.MODEL,
                bundle_module.file_sha256(pipeline_path),
                {"model_family": selected.model_name},
            )
        )
        graph.add_edge(lineage.EvidenceEdge("split", "model", "trains"))
        graph.add_node(
            lineage.EvidenceNode(
                "evaluation",
                lineage.EvidenceNodeKind.EVALUATION,
                bundle_module.file_sha256(store.path("evaluation.json")),
                {"artifact": "evaluation.json"},
            )
        )
        graph.add_edge(lineage.EvidenceEdge("model", "evaluation", "evaluates"))

        explanation_paths = sorted(
            store.path("explanations").glob("*.json")
            if store.path("explanations").exists()
            else ()
        )
        if explanation_paths:
            explanation_identity = hashlib.sha256(
                "\n".join(
                    f"{path.name}:{bundle_module.file_sha256(path)}"
                    for path in explanation_paths
                ).encode("utf-8")
            ).hexdigest()
            graph.add_node(
                lineage.EvidenceNode(
                    "explanations",
                    lineage.EvidenceNodeKind.EXPLANATION,
                    explanation_identity,
                    {"artifact_count": str(len(explanation_paths))},
                )
            )
            graph.add_edge(
                lineage.EvidenceEdge("model", "explanations", "explains")
            )
            graph.add_edge(
                lineage.EvidenceEdge(
                    "evaluation", "explanations", "contextualizes"
                )
            )

        graph.add_node(
            lineage.EvidenceNode(
                "bundle",
                lineage.EvidenceNodeKind.BUNDLE,
                bundle_module.file_sha256(model_manifest_path),
                {"artifact": "model/" + bundle_module.MANIFEST_FILE},
            )
        )
        graph.add_edge(lineage.EvidenceEdge("model", "bundle", "packages"))
        graph.add_edge(lineage.EvidenceEdge("evaluation", "bundle", "gates"))
        if explanation_paths:
            graph.add_edge(
                lineage.EvidenceEdge("explanations", "bundle", "includes")
            )
        graph.validate()
        context.evidence_graph = graph
        return graph

    def execute(self, context: context_module.RunContext) -> None:
        """Packages artifacts and assembles the `RunResult`.

        Args:
            context: The run context; receives the bundle and the result.
        """
        store = context.run_store
        split = require(context.split, "split")
        selected = require(context.selected, "selected")
        evaluation = require(context.evaluation, "evaluation")
        profile = require(context.profile, "profile")
        bundle = bundle_module.ModelBundle.create(
            context.final_pipeline,
            profile.feature_plan,
            split.dev_features,
            context.spec.data.target,
            require(context.task, "task").value,
            require(context.metric, "metric").name,
            selected.model_name,
            model_params=selected.params,
            strategy=selected.strategy,
            decision_threshold=(evaluation["decision"] or {}).get("threshold"),
            graph_fingerprint=context.spec.feature_graph.fingerprint()
            if context.spec.feature_graph.nodes
            else "",
            group_column=context.spec.data.group_column,
            time_column=context.spec.data.time_column,
            history_source=context.frame,
            class_labels=context.ordinal_labels,
            reference=reference_module.ReferenceProfile.build(
                split.dev_features,
                profile.feature_plan,
                require(context.reference_scores, "reference_scores"),
                str(require(context.reference_scores, "reference_scores").name),
                evaluation["test_metrics"],
            ),
        )
        evaluation["learning_curve"] = self.learning_curve(context, selected)
        evaluation["nested_cv"] = self.nested_validation(context, selected)
        evaluation["bias_variance"] = self.bias_variance(context, selected)
        evaluation["sensitivity"] = self.sensitivity(context, selected)
        bundle.conformal = self.conformal_calibration(context, selected, bundle)
        bundle.quantile = self.quantile_intervals(context)
        self.complete_held_ledger(context, bundle)
        if require(context.task, "task") is tasks.TaskType.FORECAST:
            evaluation["forecast_backtest"] = self.forecast_backtest(
                context, selected
            )
            context.write_json(
                "diagnostics/forecast_backtest.json",
                evaluation["forecast_backtest"],
            )
        if bundle.quantile is not None and context.test_rows is not None:
            evaluation["quantile_coverage"] = bundle.quantile.coverage_curve(
                *context.test_rows, COVERAGE_LEVELS
            )
        if context.test_rows is not None:
            measured = coverage_run.run_report(
                bundle,
                *context.test_rows,
                COVERAGE_LEVELS,
                claim_sha256=context.coverage_claim_sha256,
                slices=slice_labels(context, context.test_rows[0]),
            )
            if measured is not None:
                bundle.coverage_report = measured
                context.write_json("coverage_report.json", measured)
                context.emit(
                    "coverage.reported",
                    "coverage of the stored intervals or sets was measured "
                    "on the final test rows",
                    ("coverage_report.json",),
                    {"methods": sorted(measured["methods"])},
                )
        bundle.novelty = novelty.NoveltyDetector.fit(
            split.dev_features,
            bundle.schema.numeric,
            bundle.schema.categorical,
            context.spec.validation.seed,
        )
        bundle.save(store.path("model"))
        card = model_card.ModelCard.from_run(
            require(context.manifest, "manifest"),
            profile,
            bundle,
            evaluation,
            bundle_module.file_sha256(
                store.path(f"model/{bundle_module.MANIFEST_FILE}")
            ),
        )
        evaluation["model_card"] = card.to_dict()
        store.write_json("model_card.json", card.to_dict())
        evaluation["reliability_plan"] = self.reliability_plan_record(
            context, store
        )
        evaluation["sbom"] = self.sbom_record(context, store, bundle)
        evaluation["provenance"] = self.provenance_record(
            context, store, bundle
        )
        release_attestation = attestation.ReleaseAttestation.from_records(
            evaluation["provenance"],
            evaluation["sbom"],
            signing_key=attestation.resolve_key(None),
        ).to_dict()
        evaluation["release_attestation"] = release_attestation
        store.write_json("release_attestation.json", release_attestation)
        evaluation["explanation_evidence"] = self.explanation_evidence(
            context, store
        )
        evaluation["artifact_verification"] = bundle.verify_round_trip(
            store.path("model"), split.dev_features.head(PARITY_ROWS)
        )
        store.write_json(
            "diagnostics/data_access.json",
            ledger.build_ledger(
                split,
                context.trials,
                int(split.locked_test.opened),
                evaluation["decision"] is not None,
            ),
        )
        store.write_json(
            "diagnostics/decision_log.json",
            {
                "plan": [
                    d.to_dict() for d in require(context.plan, "plan").decisions
                ],
                "outcomes": [d.to_dict() for d in context.outcomes],
                "selection": {
                    "model": selected.model_name,
                    "strategy": selected.strategy,
                    "params": selected.params,
                    "policy": context.policy.description,
                },
            },
        )
        self.release_evaluation(context, store, evaluation)
        store.write_json("evaluation.json", evaluation)
        evidence_graph = self.build_evidence_graph(context, store)
        graph_record = evidence_graph.to_dict()
        graph_record["graph_digest"] = evidence_graph.digest()
        store.write_json("evidence_graph.json", graph_record)
        result = results.RunResult(
            run_id=store.run_id,
            run_dir=store.directory,
            state=run_state.RunState.COMPLETED.value,
            plan=require(context.plan, "plan"),
            profile=profile,
            trials=context.trials,
            selected_model=selected.model_name,
            evaluation=evaluation,
            bundle=bundle,
            diagnostics=context.diagnostics,
        )
        context.track(artifacts=self.lineage(store))
        board = result.leaderboard()
        board.to_parquet(store.path("leaderboard.parquet"), index=False)
        store.write_text(
            "report.html",
            report.HtmlReport().render(result, board),
        )
        context.bundle, context.result = bundle, result

    def lineage(self, store: Any) -> dict[str, str]:
        """Checksums of the artifacts a run produced, for tracking lineage.

        Args:
            store: The run store whose artifacts are hashed.

        Returns:
            A mapping from artifact path to its SHA-256 checksum, for the
            known artifacts that exist.
        """
        names = (
            "config.json",
            "dataset_manifest.json",
            "dataset_card.json",
            "model_card.json",
            "source_registry.json",
            "fault_injection_plan.json",
            "provenance.json",
            "sbom.json",
            "release_attestation.json",
            "release_evidence.json",
            "parameter_estimation.json",
            "evidence_graph.json",
            "evaluation.json",
            f"model/{bundle_module.MANIFEST_FILE}",
            f"model/{bundle_module.MODEL_FILE}",
            "explanations/permutation_importance.html",
            "explanations/partial_dependence.html",
            "explanations/pairwise_interaction.html",
        )
        return {
            name: bundle_module.file_sha256(store.path(name))
            for name in names
            if store.path(name).exists()
        }


def default_stages() -> tuple[Stage, ...]:
    """Returns the standard stage sequence, one per lifecycle state."""
    return compiler.compile_workflow(
        (
            IngestStage(),
            ProfileStage(),
            PlanStage(),
            TrainStage(),
            SelectStage(),
            FinalEvaluationStage(),
            PackageStage(),
        )
    )
