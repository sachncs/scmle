"""Admissibility of a charter run's exploration."""

import dataclasses
from typing import Any

import numpy

from scmle.assurance.admissibility import attestations
from scmle.assurance.admissibility import qualify
from scmle.assurance.charter import contracts
import scmle.assurance.framing.baseline
import scmle.assurance.framing.claim
from scmle.assurance.ledger import confirm
from scmle.assurance.ledger import explore
from scmle.contracts import errors
from scmle.contracts import tasks
import scmle.contracts.data
import scmle.pipeline.context
from scmle.pipeline.stages import identification


class Admission(identification.Identification):
    """Admissibility of the exploration and the baseline claim's runner.

    Part of `ProfileStage`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    def baseline_runner(
        self,
        context: scmle.pipeline.context.Context,
        task: tasks.Task,
        charter: contracts.AnalysisCharter,
        view: Any,
        structure: tuple[str, ...],
        *,
        signed: bool,
        subject: scmle.assurance.framing.claim.Rule | None = None,
    ) -> confirm.Docket | None:
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
                binary, the column is missing from the exploration rows, or
                the rule cannot be fitted (no numeric input column, or fewer
                than two observed outcome classes).
        """
        data = context.spec.data
        column = data.baseline_column
        if not column:
            return None
        model = charter.costs
        if model is None or task is not tasks.Task.BINARY:
            raise errors.ConfigError(
                "data.baseline_column needs a binary task and a charter with "
                "a cost_model"
            )
        if column not in view.features.columns:
            raise errors.ConfigError(
                f"baseline column '{column}' is not in the exploration rows"
            )
        classes = sorted(view.target.dropna().unique(), key=str)
        outcome = explore.vectorize(
            view.target, classification=True, classes=classes
        )
        plan = scmle.pipeline.context.require(
            context.profile, "profile"
        ).feature_plan
        numeric = [
            c
            for c in plan.numeric
            if c in view.features.columns and c not in structure
        ]
        rule = scmle.assurance.framing.claim.solidify(
            view.features,
            list(outcome.to_numpy(float)),
            numeric,
            model.threshold,
            charter.seed,
        )
        decisions = view.features[column].to_numpy(dtype=float)
        seen = numpy.ones(len(decisions), dtype=bool)
        if context.unlabeled is not None and column in context.unlabeled:
            extra = context.unlabeled[column].to_numpy(dtype=float)
            decisions = numpy.concatenate([decisions, extra])
            seen = numpy.concatenate(
                [seen, numpy.zeros(len(extra), dtype=bool)]
            )
        selection = scmle.assurance.framing.baseline.selective_labels(
            seen, decisions
        )
        ceiling, why = scmle.assurance.framing.claim.source_ceiling(
            data.framing_draft, charter.charter_hash, model, signed=signed
        )
        context.write_json(scmle.assurance.framing.claim.PROVENANCE, why)
        return scmle.assurance.framing.claim.rig(
            rule=rule if subject is None else subject,
            reference=None if subject is None else rule,
            model=model,
            alpha=float(charter.budget.confirmation.alpha_confirm),
            baseline_column=column,
            group_column=data.group_column,
            classes=classes,
            selection=selection.to_dict(),
            ceiling=ceiling,
            seed=charter.seed,
        )

    def qualify_exploration(
        self,
        context: scmle.pipeline.context.Context,
        task: tasks.Task,
        charter: contracts.AnalysisCharter,
        view: Any,
        structure: tuple[str, ...],
        graph: Any = None,
    ) -> qualify.Qualification:
        """Runs the admissibility tests and detectors on `X` only.

        Writes `admissibility_table.json` and `detector_certificates.json`
        (the certificates the run used). A leakage suspicion quarantines a
        field only when the leakage screen has a current, passing certificate
        for the mechanism that raised it, and a valid per-column attestation
        (`spec.leakage_attestations`) keeps such a field, tier `attested`,
        with a ledger diagnostic. Quarantined fields leave the feature plan;
        the profile, curation report and dataset card are rewritten and the
        reasons are kept in the `dropped` entries of the feature plan and in
        the emitted event.

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
        profile = scmle.pipeline.context.require(context.profile, "profile")
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
                    scmle.contracts.data.Datasheet.distill(amended).to_dict(),
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
