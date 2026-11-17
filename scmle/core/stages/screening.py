"""Leakage screening and feature synthesis of the profile stage."""

import logging
from typing import Any

import pandas

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.core import ingest_stage
import scmle.core.context
from scmle.core.stages import surveying
from scmle.data.charter import ingest
from scmle.data.charter import sealing

LOGGER = logging.getLogger(__name__)


class Screening(surveying.Surveying):
    """The leakage policy enforced and the feature set synthesised.

    Part of `ProfileStage`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    def enforce_leakage_policy(self, spec: tasks.Config, profile: Any) -> None:
        """Applies the leakage policy to the profile's leakage evidence.

        Args:
            spec: The project specification.
            profile: The dataset profile.

        Raises:
            BlockedError: If `leakage_policy` is `block` and a leakage
                hypothesis was raised or a screened column could not be
                checked.
        """
        if profile.untested:
            if spec.leakage_policy == "block":
                raise errors.BlockedError(
                    f"WARNING {profile.admonition}; leakage_policy="
                    "'block' does not train on unchecked suspicion (use "
                    "'warn' to proceed with the columns unchecked)",
                    {"leakage": dict(profile.leakage)},
                )
            LOGGER.warning(
                "WARNING %s; the test score is provisional",
                profile.admonition,
            )
        probe = profile.leakage.get("probe", {})
        missed = sorted(d for d, hit in probe.items() if not hit)
        if missed:
            LOGGER.warning(
                "WARNING on these rows the leakage screen did not find the "
                "planted %s: a leak of that kind may remain undetected",
                ", ".join(missed),
            )
        if spec.leakage_policy == "block":
            leaks = profile.blockers
            if leaks:
                raise errors.BlockedError(
                    "unresolved leakage risk (permutation screen, "
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

    def synthesize(
        self, context: scmle.core.context.Context, frame: pandas.DataFrame
    ) -> pandas.DataFrame:
        """Adds the synthesized features before the data are split.

        Does nothing unless the data spec names an `ingest_plan`, which needs
        a charter with an `ingest` block. The plan reads the database through
        the as-of views, so no feature uses a fact recorded after its row's
        cutoff; the confirmation set does not exist yet. A unit verdict of
        `coarser` or `unknown`, or views that do not reproduce the reconciled
        totals, block the run (the ingest stage raises `BlockedError`).

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
            sealing.read_verified(str(path)).charter.ingest if path else None
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
            ingest_stage.recipe(data.ingest_plan),
            block,
            frame,
            context.store.directory if context.store is not None else None,
        )
        clash = sorted(set(result.features.columns) & set(frame.columns))
        if clash:
            raise errors.ConfigError(
                f"synthesized features collide with columns: {clash}"
            )
        merged = pandas.concat(
            [frame.reset_index(drop=True), result.features], axis=1
        )
        if block.entity is not None:
            merged = self.regroup(context, block.entity, merged)
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

    def regroup(
        self,
        context: scmle.core.context.Context,
        entity: ingest.Entity,
        frame: pandas.DataFrame,
    ) -> pandas.DataFrame:
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
