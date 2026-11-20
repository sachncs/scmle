"""Sealing and impounding the locked test rows."""

import hashlib
from typing import Any

import pandas

from scmle.assurance.charter import contracts
from scmle.assurance.charter import sealing
from scmle.assurance.custody import vault
import scmle.assurance.custody.pipeline
import scmle.assurance.custody.sealed
from scmle.contracts import errors
from scmle.contracts import experiments
from scmle.contracts import tasks
from scmle.data.splitting import splitters
from scmle.evaluation import reweighing
from scmle.pipeline import engine
from scmle.pipeline.stages import exploration
from scmle.pipeline.stages import ingestion


class Impounding(exploration.Exploration):
    """The locked test rows sealed, reweighed and impounded.

    Part of `Profiling`, which inherits these methods unchanged;
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
        assert isinstance(locked, splitters.lockbox.Resident)  # noqa: S101 -- narrows the type # nosec B101
        sealed = vault.SealedTestSet(locked.features, locked.target, locked.ids)
        locked.features = locked.features.iloc[0:0]
        locked.target = locked.target.iloc[0:0]
        return sealed

    def reweigh(
        self,
        board: engine.Board,
        task: tasks.Task,
        split: splitters.Split,
    ) -> None:
        """Weights development rows so group and label are independent.

        The weights come from development rows only; the locked rows are
        untouched. The weight column then trains the models as any other. Does
        nothing unless reweighing is the requested mitigation; for tasks other
        than binary, multiclass and regression it records a rejection
        instead. A regression outcome is balanced by its equal-count bins
        (`reweighing.discretise`). The group is the fairness column, or the
        first sensitive column if none is set.

        Args:
            board: The run's board; its specification gains the weight column.
            task: The resolved task.
            split: The split whose development features receive the weights
                (the weight column is added to them in place).
        """
        fairness = board.services.spec.fairness
        if fairness.mitigation != "reweighing":
            return
        column = (
            fairness.column or board.services.spec.data.sensitive_columns[0]
        )
        label = f"fairness:{column}"
        if task not in (
            tasks.Task.BINARY,
            tasks.Task.MULTICLASS,
            tasks.Task.REGRESSION,
        ):
            board.journal.enter(
                label,
                experiments.Edict.REJECT,
                "reweighing applies to binary, multiclass and regression "
                "tasks only",
            )
            return
        levels = (
            reweighing.discretise(split.target)
            if task is tasks.Task.REGRESSION
            else split.target
        )
        weights = reweighing.weigh(split.features[column].astype(str), levels)
        split.features[tasks.FAIRNESS_WEIGHT] = weights
        board.amend(
            board.services.spec.model_copy(
                update={
                    "data": board.services.spec.data.model_copy(
                        update={"weight_column": tasks.FAIRNESS_WEIGHT}
                    )
                }
            )
        )
        board.journal.enter(
            label,
            experiments.Edict.INCLUDE,
            "development rows reweighted so group and label are independent "
            f"(weights from {weights.min():.2f} to {weights.max():.2f})",
        )

    def sequester(
        self,
        board: engine.Board,
        task: tasks.Task,
        frame: pandas.DataFrame,
        split: splitters.Split,
    ) -> exploration.Custody:
        """Carves and seals the confirmation set under the charter.

        The charter is verified (hash, and signature when a key is
        configured) before any row is partitioned, and a charter that fails
        verification is refused. The custody's split holds exploration rows
        only, so the profiler and every trial that follows read nothing from
        the confirmation set. The checks of `recognize` also apply.
        If the carving or its records fail, or the run is interrupted, the
        vault is closed before the error leaves: nothing else holds it yet.

        Limits: the vault is a separate process on this machine. This process
        drops its own reference to the sealed rows, but the caller's DataFrame
        or the original file still holds every row.

        Args:
            board: The run's board, supplying the specification and the
                manifest; receives the files that describe the custody.
            task: The resolved task.
            frame: The full data before partitioning.
            split: The split before the confirmation rows are carved out.

        Returns:
            The custody: the outcome, its vault and the entity column.

        Raises:
            ConfigError: If the charter's unit of prediction is `new_entity`
                and the data spec has no group column.
        """
        path = str(board.services.spec.charter_path)
        verified = sealing.read(path)
        charter = verified.charter
        if (
            charter.purpose == contracts.NEW_ENTITY
            and not board.services.spec.data.group_column
        ):
            raise errors.ConfigError(
                "the charter's unit_of_prediction is new_entity, which needs "
                "data.group_column so that entities are split and calibrated "
                "as units"
            )
        column = self.recognize(board, charter, split)
        log = scmle.assurance.custody.sealed.Journal(
            board.services.store.directory / "custody_log.jsonl"
            if board.services.store is not None
            else None
        )
        intake = board.find(ingestion.Intake)
        data_hash = (
            intake.manifest.digests["data"]
            if intake is not None
            else hashlib.sha256(
                pandas.util.hash_pandas_object(frame).to_numpy().tobytes()
            ).hexdigest()
        )
        held = scmle.assurance.custody.sealed.Vault(log)
        try:
            outcome = scmle.assurance.custody.pipeline.sequester(
                charter=charter,
                split=split,
                frame=frame,
                spec=board.services.spec,
                task=task,
                data_hash=data_hash,
                vault=held,
            )
            self.chronicle(board, path, verified, outcome)
        except BaseException:
            # handled: the vault is closed and the error re-raised
            held.close()
            raise
        return exploration.Custody(outcome, held, column)

    def chronicle(
        self,
        board: engine.Board,
        path: str,
        verified: Any,
        outcome: Any,
    ) -> None:
        """Writes the charter, error budget, feasibility and split manifest.

        Args:
            board: The run's board that receives the files and the event.
            path: The charter file.
            verified: The verified charter and whether it is signed.
            outcome: The custody outcome of the sealing.
        """
        board.journal.write(
            "analysis_charter.json", contracts.deserialize(path)
        )
        board.journal.write(
            "error_budget.json", verified.charter.budget.to_dict()
        )
        board.journal.write("feasibility.json", outcome.feasibility.to_dict())
        board.journal.write("split_manifest.json", outcome.manifest.to_dict())
        board.journal.emit(
            "custody.sealed",
            outcome.feasibility.reason,
            (
                "analysis_charter.json",
                "split_manifest.json",
                "feasibility.json",
                "custody_log.jsonl",
            ),
            {
                "charter_hash": verified.charter.digest,
                "charter_signed": verified.signed,
                "manifest_sha256": outcome.manifest.sha256,
                "scope": outcome.manifest.scope,
                "exploration_rows": len(outcome.manifest.sets.exploration),
                "confirmation_rows": len(outcome.manifest.sets.confirmation),
                "embargo_rows": len(outcome.manifest.sets.embargo),
                "isolation": dict(outcome.manifest.isolation),
            },
        )

    def recognize(
        self,
        board: engine.Board,
        charter: contracts.Charter,
        split: splitters.Split,
    ) -> str | None:
        """Requires the data a known-entity-future charter needs.

        A later row of an entity seen in training is predicted, so the
        entity column and the time column must be named and validation must
        be time-ordered: an entity-disjoint split would test the wrong unit
        (that is `new_entity`). The entity column becomes the run's cluster
        column (`exploration.Custody.column`).

        Args:
            board: The run's board, supplying the data spec.
            charter: The analysis charter; nothing is checked unless its unit
                of prediction is `known_entity_future`.
            split: The split, whose strategy must be time-ordered.

        Returns:
            The entity column, or None unless the charter predicts known
            entities' future rows.

        Raises:
            ConfigError: If a column is missing or the split strategy is not
                time-ordered.
        """
        if charter.purpose != contracts.KNOWN_ENTITY_FUTURE:
            return None
        data = board.services.spec.data
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
        board.journal.emit(
            "unit.known_entity_future",
            "entities are not split units: their rows sit on both sides, "
            "the confirmation rows are time blocks with an embargo and the "
            "final test rows are the latest period",
            data={"entity_column": data.group_column},
        )
        return data.group_column
