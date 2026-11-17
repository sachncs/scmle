"""Sealing and impounding the locked test rows."""

import hashlib

import pandas

from scmle.assurance.charter import contracts
from scmle.assurance.charter import sealing
from scmle.assurance.custody import vault
import scmle.assurance.custody.pipeline
import scmle.assurance.custody.sealed
from scmle.contracts import errors
from scmle.contracts import experiments
from scmle.contracts import tasks
import scmle.core.context
from scmle.core.stages import exploration
from scmle.data.splitting import splitters
from scmle.evaluation import reweighing


class Impounding(exploration.Exploration):
    """The locked test rows sealed, reweighed and impounded.

    Part of `ProfileStage`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    @staticmethod
    def seal(locked: splitters.Lockbox) -> splitters.Lockbox:
        """Moves the test rows into a vault process.

        Args:
            locked: The resident test set just produced by the split.

        Returns:
            The sealed set; this process keeps only the row ids. The
            resident object is emptied so no reference to the rows remains.
        """
        assert isinstance(locked, splitters.ResidentTestSet)  # noqa: S101 -- narrows the type # nosec B101
        sealed = vault.SealedTestSet(locked.features, locked.target, locked.ids)
        locked.features = locked.features.iloc[0:0]
        locked.target = locked.target.iloc[0:0]
        return sealed

    def reweigh(
        self,
        context: scmle.core.context.Context,
        task: tasks.Task,
        split: splitters.Split,
    ) -> None:
        """Weights development rows so group and label are independent.

        The weights come from development rows only; the locked rows are
        untouched. The weight column then trains the models as any other. Does
        nothing unless reweighing is the requested mitigation; for tasks other
        than binary and multiclass it records a rejection instead. The group
        is the fairness column, or the first sensitive column if none is set.

        Args:
            context: The run context; its specification gains the weight column.
            task: The resolved task.
            split: The split whose development features receive the weights
                (the weight column is added to them in place).
        """
        fairness = context.spec.fairness
        if fairness.mitigation != "reweighing":
            return
        column = fairness.column or context.spec.data.sensitive_columns[0]
        label = f"fairness:{column}"
        if task not in (tasks.Task.BINARY, tasks.Task.MULTICLASS):
            context.enter(
                label,
                experiments.Edict.REJECT,
                "reweighing applies to binary and multiclass tasks only",
            )
            return
        weights = reweighing.weigh(
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
        context.enter(
            label,
            experiments.Edict.INCLUDE,
            "development rows reweighted so group and label are independent "
            f"(weights from {weights.min():.2f} to {weights.max():.2f})",
        )

    def impound(
        self,
        context: scmle.core.context.Context,
        task: tasks.Task,
        frame: pandas.DataFrame,
        split: splitters.Split,
    ) -> splitters.Split:
        """Carves and seals the confirmation set under the charter.

        The charter is verified (hash, and signature when a key is
        configured) before any row is partitioned, and a charter that fails
        verification is refused. The returned split holds exploration rows
        only, so the profiler and every trial that follows read nothing from
        the confirmation set. The checks of `check_known_entity` also apply.

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
        verified = sealing.read_verified(path)
        charter = verified.charter
        if (
            charter.unit_of_prediction == contracts.NEW_ENTITY
            and not context.spec.data.group_column
        ):
            raise errors.ConfigError(
                "the charter's unit_of_prediction is new_entity, which needs "
                "data.group_column so that entities are split and calibrated "
                "as units"
            )
        self.check_known_entity(context, charter, split)
        log = scmle.assurance.custody.sealed.Journal(
            context.store.directory / "custody_log.jsonl"
            if context.store is not None
            else None
        )
        vault_set = scmle.assurance.custody.sealed.Vault(log)
        data_hash = (
            context.manifest.data_hash
            if context.manifest is not None
            else hashlib.sha256(
                pandas.util.hash_pandas_object(frame).to_numpy().tobytes()
            ).hexdigest()
        )
        outcome = scmle.assurance.custody.pipeline.sequester(
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
        context.write_json("analysis_charter.json", contracts.deserialize(path))
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
        context: scmle.core.context.Context,
        charter: contracts.AnalysisCharter,
        split: splitters.Split,
    ) -> None:
        """Requires the data a known-entity-future charter needs.

        A later row of an entity seen in training is predicted, so the
        entity column and the time column must be named and validation must
        be time-ordered: an entity-disjoint split would test the wrong unit
        (that is `new_entity`). The entity column becomes the context's
        cluster column.

        Args:
            context: The run context, supplying the data spec.
            charter: The analysis charter; nothing is checked unless its unit
                of prediction is `known_entity_future`.
            split: The split, whose strategy must be time-ordered.

        Raises:
            ConfigError: If a column is missing or the split strategy is not
                time-ordered.
        """
        if charter.unit_of_prediction != contracts.KNOWN_ENTITY_FUTURE:
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
        if (
            split.manifest.strategy
            not in scmle.assurance.custody.pipeline.TEMPORAL
        ):
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
