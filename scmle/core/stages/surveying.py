"""Profiling and curation of the development rows."""

import dataclasses
from typing import Any

import pandas

from scmle.contracts import tasks
import scmle.contracts.data
import scmle.core.context
from scmle.core.stages import labelling
from scmle.data.profiling import label_noise
from scmle.data.profiling import profiler
from scmle.data.profiling import representativeness
from scmle.data.readers import tabular
from scmle.evaluation import uplift
from scmle.features import feature_graph


class Surveying(labelling.Labelling):
    """The development rows profiled, curated and annotated.

    Part of `ProfileStage`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    def survey(
        self,
        context: scmle.core.context.Context,
        spec: tasks.Config,
        task: tasks.Task,
        frame: pandas.DataFrame,
        split: Any,
    ) -> tuple[Any, pandas.DataFrame]:
        """Profiles the development rows of a split.

        Args:
            context: The run context, holding the manifest and cleaning notes.
            spec: The project specification.
            task: The resolved task.
            frame: The whole loaded data.
            split: The split whose development rows are profiled.

        Returns:
            The profile, with the feature graph, cleaning notes, quality notes
            and (for an uplift task) the propensity finding attached, and the
            development frame.
        """
        development = split.dev_features.assign(
            **{spec.data.target: split.dev_target}
        )
        profile = profiler.Profiler(spec).profile(
            development,
            scmle.core.context.require(context.manifest, "manifest"),
            task,
            context.read_seconds,
            total_rows=len(frame),
        )
        profile = graft(spec, profile)
        if context.cleaning_notes:
            profile = dataclasses.replace(
                profile,
                quality_findings=profile.quality_findings
                + tuple(context.cleaning_notes),
            )
        profile = self.annotate(spec, task, development, profile)
        if task is tasks.Task.UPLIFT:
            propensity = uplift.randomised(
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
        return profile, development

    def curate(
        self,
        context: scmle.core.context.Context,
        frame: pandas.DataFrame,
        development: pandas.DataFrame,
        profile: Any,
    ) -> None:
        """Writes the curation report and dataset card.

        Curation findings are advisory observations for a person to review;
        the conditions that stop a run are the registered validation and
        split gates (`gates.toml`). A blocking check here could never fire,
        because every observation is recorded as non-blocking (GATE-1).

        Args:
            context: The run context; receives the profile, the curation
                report and the artifacts written for them.
            frame: The whole loaded data.
            development: The development rows that were profiled.
            profile: The dataset profile.
        """
        source_rows = (
            context.stream[1].rows if context.stream is not None else len(frame)
        )
        curation = scmle.contracts.data.Curation(
            manifest=scmle.core.context.require(context.manifest, "manifest"),
            source_rows=source_rows,
            profiled_rows=len(development),
            read_seconds=context.read_seconds,
            streamed=context.stream is not None,
            cleaning_notes=tuple(context.cleaning_notes),
            quality_findings=tuple(
                scmle.contracts.data.Finding.glean(finding, "quality")
                for finding in profile.quality_findings
            ),
            risk_findings=tuple(
                scmle.contracts.data.Finding.glean(finding, "risk")
                for finding in profile.risk_findings
            ),
            feature_columns={
                "numeric": profile.feature_plan.numeric,
                "categorical": profile.feature_plan.categorical,
                "text": profile.feature_plan.text,
            },
            dropped_columns=profile.feature_plan.dropped,
        )
        card = scmle.contracts.data.Datasheet.distill(curation)
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

    @staticmethod
    def annotate(
        spec: tasks.Config,
        task: tasks.Task,
        development: pandas.DataFrame,
        profile: scmle.contracts.data.Portrait,
    ) -> scmle.contracts.data.Portrait:
        """Appends label-noise and representativeness findings.

        Representativeness is assessed only when the data specification names a
        reference sample.

        Args:
            spec: The project specification.
            task: The resolved task; label noise is assessed for binary and
                multiclass tasks only, and only if numeric features exist.
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
        evidence: list[dict[str, object]] = []
        if columns and task in (
            tasks.Task.BINARY,
            tasks.Task.MULTICLASS,
        ):
            observation = label_noise.inspect(
                development[columns],
                development[spec.data.target],
                spec.validation.seed,
                allowance=spec.resources.time_budget_seconds
                * label_noise.SHARE,
                threads=spec.resources.cpu_limit,
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


def graft(
    spec: tasks.Config, profile: scmle.contracts.data.Portrait
) -> scmle.contracts.data.Portrait:
    """Adds the feature graph to the profile's feature plan.

    Row-wise nodes become numeric model features computed inside the
    pipeline; ordered nodes are already columns of the profiled frame. Raw
    columns the graph reads but the model does not use are recorded as
    `extra_inputs`, so inference knows what the caller must supply.

    Args:
        spec: Project specification.
        profile: Profile whose plan lacks the graph.

    Returns:
        A profile whose feature plan carries the graph, or `profile` itself if
        the specification has no feature graph nodes.
    """
    graph = spec.feature_graph
    if not graph.nodes:
        return profile
    ordered, rowwise = feature_graph.segregate(graph)
    plan = profile.feature_plan
    computed = {n.name for n in graph.nodes} | plan.engineered
    needed = feature_graph.roots(graph.nodes, computed)
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
