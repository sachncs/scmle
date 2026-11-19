"""Scores and the decision threshold of the final evaluation."""

from collections.abc import Mapping
import hashlib
import json
from typing import Any

import numpy
import pandas

from scmle.assurance.charter import contracts
from scmle.assurance.charter import sealing
from scmle.contracts import errors
from scmle.contracts import records
from scmle.contracts import tasks
from scmle.decision import netbenefit
import scmle.decision.threshold
from scmle.pipeline import engine
from scmle.pipeline import products
from scmle.pipeline.stages import cost
from scmle.pipeline.stages import ingestion
from scmle.pipeline.stages import profiling
from scmle.pipeline.stages import selecting
from scmle.preparation import strategies

COVERAGE_LEVELS = (0.5, 0.8, 0.9, 0.95)


MATRIX_DIMENSIONS = 2


def reference_scores(
    fitted: Any, features: pandas.DataFrame, task: tasks.Task
) -> pandas.Series:
    """Returns model outputs on held-out rows, the output-drift reference.

    Args:
        fitted: The final candidate.
        features: Held-out features.
        task: Resolved task type.

    Returns:
        Positive-class probabilities for binary tasks; for every other task the
        predictions, taking the first column if they are two-dimensional.
    """
    if task is tasks.Task.BINARY:
        return pandas.Series(fitted.predict_proba(features)[:, 1], name="score")
    predicted = numpy.asarray(fitted.predict(features))
    if predicted.ndim == MATRIX_DIMENSIONS:
        predicted = predicted[:, 0]  # The primary target stands for all.
    return pandas.Series(predicted, name="prediction")


def slicers(
    board: engine.Board,
) -> tuple[str, ...]:
    """Returns the slice columns the charter's uncertainty block declares.

    Args:
        board: The run's board, supplying the charter path.

    Returns:
        The declared columns; empty if there is no charter or no
        uncertainty block.
    """
    path = board.services.spec.charter_path
    if not path:
        return ()
    block = sealing.read_verified(str(path)).charter.uncertainty
    return () if block is None else block.slices


def decisional(
    board: engine.Board,
) -> contracts.AnalysisCharter | None:
    """Returns the charter when it declares a decision layer, else None.

    Args:
        board: The run's board, supplying the charter path.

    Returns:
        The charter if it declares both a decision layer and a cost model,
        else None.
    """
    path = board.services.spec.charter_path
    if path is None:
        return None
    charter = sealing.read_verified(str(path)).charter
    if charter.decision is None or charter.costs is None:
        return None
    return charter


def appraise(
    board: engine.Board,
    pipeline: Any,
    features: pandas.DataFrame,
    target: pandas.Series,
    decision: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Returns the net benefit of the frozen rule on the locked test set.

    A guard registered with the rule's hash evaluates the band in a single
    call. The band is simultaneous over the chartered range of threshold
    probabilities and clustered by the group column of a grouped split. The
    claim is `useful` or `not_useful_in_range`; it is tiered `attested` when
    every cost came from a person and the rule's own tier is not
    `unverified`, and `unverified` otherwise.

    Args:
        board: The run's board, supplying the charter and split.
        pipeline: The fitted classifier.
        features: The final test rows.
        target: The final test labels.
        decision: The chartered threshold description, or None.

    Returns:
        The net-benefit report with its tier, cost model hash and rule
        hash, or None if the charter has no decision layer and cost model or
        the decision carries no chartered rule.

    Raises:
        ScmleError: If the charter has no cost model; the earlier check of
            `decisional` already returns None then, so this
            guards the type.
    """
    charter = decisional(board)
    if charter is None or decision is None or "f11" not in decision:
        return None
    model = charter.costs
    if model is None:
        raise errors.ScmleError("the charter has no cost model")
    positive = pipeline.classes_[1]
    split = board.get(profiling.Partition).split
    clusters = (
        features[board.services.spec.data.group_column]
        if split.manifest.strategy == "group"
        and board.services.spec.data.group_column
        else None
    )
    guard = netbenefit.FinalNetBenefit(decision["f11"]["rule_hash"])
    report = guard.evaluate(
        decision["f11"]["rule_hash"],
        (target == positive).to_numpy(dtype=float),
        (numpy.asarray(pipeline.predict(features)) == positive).astype(float),
        (model.low, model.high),
        clusters=clusters,
        alpha=float(charter.budget.confirmation.alpha_confirm),
        replicates=max(charter.budget.permutations, 500),
        seed=charter.seed,
    )
    tier = "unverified" if model.unreviewed else "attested"
    return {
        **report.to_dict(),
        "tier": "unverified"
        if decision["f11"]["tier"] == "unverified"
        else tier,
        "cost_model_hash": model.model_hash,
        "rule_hash": decision["f11"]["rule_hash"],
    }


def pin(selected: Any, decision: Mapping[str, Any] | None) -> str:
    """Digest of the frozen selection the test set is released for.

    The single release of the locked rows is bound to it (GATE-3), so the
    release cannot serve a different model, setting or threshold.

    Args:
        selected: The selected trial.
        decision: The chosen decision threshold, or None.

    Returns:
        SHA-256 of the trial id, model, strategy, parameters and threshold,
        as canonical JSON.
    """
    frozen = {
        "trial": selected.trial_id,
        "model": selected.model_name,
        "strategy": selected.strategy,
        "params": selected.params,
        "decision": decision,
    }
    text = json.dumps(frozen, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def choose_decision_threshold(
    board: engine.Board, fitted: Any
) -> tuple[Any, records.Layout | None]:
    """Optimizes the binary decision threshold on out-of-fold predictions.

    Only development data is used (`scmle.decision.threshold.choose`): each
    validation row is scored by a model that never saw it, so the locked
    test set stays untouched.

    Args:
        board: Run board holding the split and the selected trial.
        fitted: The selected candidate, already refit on development data.

    Returns:
        The estimator to evaluate (wrapped with the threshold when one was
        optimized) and a description of the decision. For a task other than
        binary, or with no objective and no chartered decision layer, the
        estimator is returned unchanged with None, as it is (on record) when
        the out-of-fold fits would not end before the packaging reserve. A
        charter's cost model takes precedence over the specification's
        objective and is always applied.
    """
    decision = board.services.spec.decision
    partition = board.get(profiling.Partition)
    task = partition.task
    chartered = decisional(board)
    if task is not tasks.Task.BINARY or (
        decision.threshold_objective == "none" and chartered is None
    ):
        return fitted, None
    split = partition.split
    selected = products.require(
        board.get(selecting.Choice).selected, "selected"
    )
    if chartered is None:
        governor = board.services.governor
        reason = governor.overdue(
            cost.foldcost(selected) * len(split.manifest.folds),
            governor.reserve.parts.get("package", 0.0),
        )
        if reason is not None:
            board.journal.forgo("decision threshold", reason)
            return fitted, None
    strategy = strategies.lookup(selected.strategy)
    adapter = board.services.registry.get(selected.model_name)
    plan = partition.profile.feature_plan
    seed = board.services.spec.validation.seed
    return scmle.decision.threshold.choose(
        fitted,
        lambda: strategy.build(
            plan,
            adapter,
            task,
            seed,
            selected.params,
            featurise=board.get(ingestion.Intake).kind.featurise,
        ),
        (split.dev_features, split.dev_target),
        split.manifest.folds,
        decision,
        chartered,
        seed,
    )
