"""The final evaluation stage on the locked test set."""

import dataclasses
from typing import Any
from typing import ClassVar

import numpy
import pandas

from scmle.assurance.custody import journal
from scmle.contracts import experiments
from scmle.contracts import models
from scmle.contracts import records
from scmle.contracts import tasks
from scmle.models import thresholds
from scmle.pipeline import engine
from scmle.pipeline import products
from scmle.pipeline.stages import cost
from scmle.pipeline.stages import examination
from scmle.pipeline.stages import ingestion
from scmle.pipeline.stages import labelling
from scmle.pipeline.stages import profiling
from scmle.pipeline.stages import scoring
from scmle.pipeline.stages import selecting
from scmle.pipeline.stages import summary
from scmle.preparation import pseudo_labeling
from scmle.preparation import strategies
import scmle.runtime.state


@dataclasses.dataclass(frozen=True)
class Exam(products.Derived):
    """The final model and its one test-set evaluation; never written.

    It holds the opened test rows. A run whose test set was opened is not
    resumed (`journal.Journal.admit`), so it is never needed again.

    Attributes:
        evaluation: The evaluation record.
        final_pipeline: The model refit on the development data.
        shipped: The model the bundle serves: the same choice refit on
            every row, or `final_pipeline` when it was not refit.
        test_rows: The opened test rows and labels.
        pseudolabels: The self-training judgement of the final fit.
        remedy: The group-threshold judgement of the final model.
        decision: The decision threshold, its net benefit and the coverage
            claim's hash.
        readings: The reference scores and diagnostics of the test rows.
    """

    evaluation: records.Layout | None
    final_pipeline: models.Estimator | None
    shipped: models.Estimator | None
    test_rows: tuple[pandas.DataFrame, pandas.Series] | None
    pseudolabels: examination.Pseudolabels = dataclasses.field(
        default_factory=examination.Pseudolabels
    )
    remedy: examination.Remedy = dataclasses.field(
        default_factory=examination.Remedy
    )
    decision: examination.Decision = dataclasses.field(
        default_factory=examination.Decision
    )
    readings: examination.Readings = dataclasses.field(
        default_factory=examination.Readings
    )

    def apply(self, board: object) -> None:
        """Holds the exam and each of its parts on the board.

        Args:
            board: The run's board.
        """
        for part in (
            self.pseudolabels,
            self.remedy,
            self.decision,
            self.readings,
        ):
            part.apply(board)
        super().apply(board)

    def summary(self) -> object:
        """The evaluation record and whether the test rows were opened.

        Returns:
            The evaluation and the opening.
        """
        return {"evaluation": self.evaluation, "opened": bool(self.test_rows)}


class FinalEvaluationStage(summary.Summary):
    """Refits the frozen candidate and opens the locked test set once."""

    requires: ClassVar[tuple[type, ...]] = (selecting.SelectStage,)
    makes: ClassVar[type[products.Product] | None] = Exam
    state: ClassVar[scmle.runtime.state.Progress] = (
        scmle.runtime.state.Progress.TESTING
    )

    def describe(self, board: engine.Board) -> str:
        """Names the selected model.

        Args:
            board: The run's board, holding the selected trial.

        Returns:
            The text "final one-time test evaluation of" and the model name.
        """
        selected = products.require(
            board.get(selecting.Choice).selected, "selected"
        )
        return f"final one-time test evaluation of {selected.model_name}"

    def execute(self, board: engine.Board) -> Exam:
        """Refits on all development data and evaluates on the test set.

        Reuses the refit made during selection if there is one. Applies
        pseudo-labeling, streamed training, the decision threshold and the
        fairness mitigation, registers the coverage claim, then opens the
        locked rows once, after the selection is frozen. The bundle's model
        is the choice refit on every row (`ship`). The locked rows are first
        wrapped so that opening them is recorded in the run's custody
        journal (`journal.Custodian`).

        Args:
            board: The run's board; its journal records events and files.

        Returns:
            The final model, the test rows and the evaluation, with the
            pseudo-labels, the remedy, the decision and the readings.
        """
        board.services.check()
        partition = board.get(profiling.Partition)
        split = partition.split
        if not isinstance(split.test, journal.Custodian):
            split.test = journal.Custodian(
                split.test,
                journal.Journal.locate(board.services.workspace.directory),
            )
        selected = products.require(
            board.get(selecting.Choice).selected, "selected"
        )
        task = partition.task
        pipeline = self.refit(board, selected, task)
        pipeline, pseudolabels = self.pseudolabel(board, selected, pipeline)
        pipeline = self.relearn(board, pipeline)
        pipeline, threshold = scoring.choose_decision_threshold(board, pipeline)
        unmitigated = pipeline
        pipeline, remedy = self.mitigate(board, selected, pipeline, threshold)
        claim = self.pledge(board)
        locked = split.test
        rows = self.unlock(board, threshold)
        remedy = self.corroborate(remedy, unmitigated, pipeline, *rows)
        usage = cost.meter(board, pipeline, split.features)
        test_metrics = self.measure(board, pipeline, *rows)
        shipped, refit = self.ship(board, pipeline, rows, pseudolabels)
        readings = self.examine(board, pipeline, *rows)
        benefit = scoring.appraise(board, pipeline, *rows, threshold)
        decision = examination.Decision(threshold, benefit, claim)
        self.decide(board, pipeline, *rows, decision)
        evaluation = self.document(
            board,
            pipeline,
            rows,
            test_metrics,
            (pseudolabels, remedy),
            {"decision": threshold, "net_benefit": benefit},
        ) | {
            "resources": usage.to_dict(),
            "resource_limits": board.services.governor.report(),
            "skipped_for_time": board.journal.skipped,
            "test_set_opened": locked.opened,
            "shipped": refit,
            "diagnostics": sorted(readings.diagnostics),
            "elapsed_seconds": board.services.elapsed(),
            "reproducibility": (
                "same seed/data/environment replays; threaded backends may "
                "differ"
            ),
        }
        return Exam(
            evaluation,
            pipeline,
            shipped,
            rows,
            pseudolabels,
            remedy,
            decision,
            readings,
        )

    def refit(
        self,
        board: engine.Board,
        selected: experiments.Trial,
        task: tasks.Task,
    ) -> Any:
        """The selected pipeline fitted on all development data.

        Args:
            board: The run's board.
            selected: The selected trial.
            task: The resolved task.

        Returns:
            The refit made during selection if there is one, else a new fit.
        """
        pipeline = board.get(selecting.Choice).refits.get(selected.trial_id)
        partition = board.get(profiling.Partition)
        if pipeline is None:
            split = partition.split
            pipeline = strategies.lookup(selected.strategy).build(
                partition.profile.feature_plan,
                board.services.registry.get(selected.model_name),
                task,
                board.services.spec.validation.seed,
                selected.params,
                featurise=board.get(ingestion.Intake).kind.featurise,
            )
            pipeline.fit(split.features, split.target)
        return pipeline

    def ship(
        self,
        board: engine.Board,
        evaluated: Any,
        rows: tuple[pandas.DataFrame, pandas.Series],
        pseudolabels: examination.Pseudolabels,
    ) -> tuple[Any, records.Layout]:
        """The frozen choice refit on development and test rows, to serve.

        Selection and the test score are done, so the held-out rows have no
        further use as a judge; a model fitted on every row is expected to
        be at least as good as the one evaluated, as a practitioner refits
        on all data after choosing. The same configuration, pseudo-labels,
        decision threshold and group thresholds are kept; rows keep their
        ingested order. The reported test metrics stay those of
        `evaluated`, which never saw the test rows, so they are a
        conservative estimate for the served model. The refit is skipped,
        on record, when a charter governs the run (its confirmatory claim
        is about the evaluated model itself), when training was streamed,
        or when it would not end before the packaging reserve.

        Args:
            board: The run's board, holding the split and the selection.
            evaluated: The evaluated model, wrapped as it was scored.
            rows: The opened test rows and labels.
            pseudolabels: The final fit's self-training judgement; the
                refit self-trains again when the pseudo-labels were used.

        Returns:
            The model to serve and the record of the refit (`refit`, the
            `rows` it was fitted on and the `reason` it was skipped).
        """
        partition = board.get(profiling.Partition)
        split = partition.split
        selected = products.require(
            board.get(selecting.Choice).selected, "selected"
        )
        reason = self.hold(board, len(rows[0]))
        if reason:
            board.journal.forgo("refit on every row", reason)
            return evaluated, {
                "refit": False,
                "rows": len(split.features),
                "reason": reason,
            }
        features, target = pool(split, rows)
        task = partition.task
        fitted = strategies.lookup(selected.strategy).build(
            partition.profile.feature_plan,
            board.services.registry.get(selected.model_name),
            task,
            board.services.spec.validation.seed,
            selected.params,
            featurise=board.get(ingestion.Intake).kind.featurise,
        )
        fitted.fit(features, target)
        unlabeled = board.get(labelling.Resolution).unlabeled
        if (pseudolabels.record or {}).get("used") and unlabeled is not None:
            fitted = pseudo_labeling.selftrain(
                fitted,
                features,
                target,
                unlabeled,
                board.services.spec.data.pseudo_label_confidence,
            )[0]
        board.journal.emit(
            "model.shipped",
            f"selected model refit on all {len(features)} rows to serve",
            data={"rows": len(features)},
        )
        return thresholds.rewrap(evaluated, fitted), {
            "refit": True,
            "rows": len(features),
            "reason": "",
        }

    def hold(self, board: engine.Board, tested: int) -> str | None:
        """Why the selected model must not be refit on every row, or None.

        Args:
            board: The run's board, holding the split, selection and
                governor.
            tested: Number of test rows the refit would add.

        Returns:
            The reason: a charter governs the run, training was streamed, or
            the refit would not end before the packaging reserve.
        """
        if board.services.spec.charter_path is not None:
            return "a charter governs the run"
        if (
            board.get(ingestion.Intake).stream is not None
            and board.services.spec.data.training == "stream"
        ):
            return "training was streamed"
        split = board.get(profiling.Partition).split
        selected = products.require(
            board.get(selecting.Choice).selected, "selected"
        )
        developed = max(len(split.features), 1)
        return board.services.governor.overdue(
            cost.refitcost(board, selected) * (developed + tested) / developed,
            board.services.governor.reserve.parts.get("package", 0.0),
        )

    def unlock(
        self,
        board: engine.Board,
        decision: records.Layout | None,
    ) -> tuple[pandas.DataFrame, pandas.Series]:
        """Opens the locked rows once, bound to the frozen selection.

        Emits `test.opened` with the rows released and the split's
        fingerprint.

        Args:
            board: The run's board, holding the split and the selection.
            decision: The decision threshold the selection is frozen with.

        Returns:
            The test features and labels.
        """
        split = board.get(profiling.Partition).split
        selected = products.require(
            board.get(selecting.Choice).selected, "selected"
        )
        locked = split.test
        binding = scoring.pin(selected, decision)
        locked.bind(binding)
        features, target = locked.open(locked.mint(), binding)
        if (
            board.services.spec.data.weight_column == tasks.FAIRNESS_WEIGHT
            and tasks.FAIRNESS_WEIGHT not in features
        ):
            # Reweighing weights development rows only; locked rows carry a
            # neutral weight so the pipeline finds the column it expects.
            features = features.assign(**{tasks.FAIRNESS_WEIGHT: 1.0})
        board.journal.emit(
            "test.opened",
            "locked test set released once, after selection was frozen",
            data={
                "rows": len(features),
                "fingerprint": split.manifest.fingerprint,
            },
        )
        return features, target


def pool(
    split: Any, rows: tuple[pandas.DataFrame, pandas.Series]
) -> tuple[pandas.DataFrame, pandas.Series]:
    """Development and test rows together, in their ingested order.

    Args:
        split: The split, holding the development rows and the manifest.
        rows: The opened test features and labels.

    Returns:
        Every row's features and labels, ordered by row id.
    """
    order = numpy.argsort(
        numpy.concatenate([split.manifest.training, split.manifest.test]),
        kind="stable",
    )
    parts = [split.features, rows[0], split.target, rows[1]]
    parts = [part.copy(deep=False) for part in parts]
    for part in parts:
        # `pandas.concat` compares the parts' `attrs`, which fails when a
        # user's frame keeps a Series there; the pooled rows need none.
        part.attrs = {}
    features = pandas.concat(parts[:2], ignore_index=True)
    target = pandas.concat(parts[2:], ignore_index=True)
    return (
        features.iloc[order].reset_index(drop=True),
        target.iloc[order].reset_index(drop=True),
    )
