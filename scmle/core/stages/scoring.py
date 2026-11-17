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
from scmle.contracts import tasks
import scmle.core.context
from scmle.core.stages import cost
from scmle.decision import netbenefit
import scmle.decision.threshold
from scmle.evaluation import coverage_confirm
from scmle.models import thresholds
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


def coverage_on_confirmation(
    charter: contracts.AnalysisCharter, task: tasks.Task
) -> bool:
    """Returns whether the charter asks for the coverage claim on `C`.

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
        and coverage_confirm.principal(task.value) is not None
    )


def slicers(
    context: scmle.core.context.Context,
) -> tuple[str, ...]:
    """Returns the slice columns the charter's uncertainty block declares.

    Args:
        context: The run context, supplying the charter path.

    Returns:
        The declared columns; empty if there is no charter or no
        uncertainty block.
    """
    path = context.spec.charter_path
    if not path:
        return ()
    block = sealing.read_verified(str(path)).charter.uncertainty
    return () if block is None else block.slice_columns


def decisional(
    context: scmle.core.context.Context,
) -> contracts.AnalysisCharter | None:
    """Returns the charter when it declares a decision layer, else None.

    Args:
        context: The run context, supplying the charter path.

    Returns:
        The charter if it declares both a decision layer and a cost model,
        else None.
    """
    path = context.spec.charter_path
    if path is None:
        return None
    charter = sealing.read_verified(str(path)).charter
    if charter.decision is None or charter.cost_model is None:
        return None
    return charter


def freeze(
    fitted: Any,
    probabilities: numpy.ndarray,
    actual: numpy.ndarray,
    charter: contracts.AnalysisCharter,
    seed: int,
    rows: int,
) -> tuple[Any, dict[str, Any]]:
    """Wraps a classifier in the chartered cost model's frozen threshold.

    The rule is chosen on out-of-fold development probabilities only and is
    frozen: the cost model's Bayes cut if the probabilities pass the calibration
    check, a recalibrated cut if they pass after recalibration, otherwise a
    cut searched on a grid. A recalibrating rule is expressed as the
    equivalent cut on the raw probability, so the served classifier needs no
    extra stage.

    Args:
        fitted: The fitted classifier to wrap.
        probabilities: Out-of-fold development probabilities.
        actual: The observed labels.
        charter: The analysis charter with its decision layer and cost model.
        seed: Seed for the calibration check and its replicates.
        rows: Number of validation rows, recorded in the description.

    Returns:
        The thresholded classifier and a description holding the objective,
        the raw threshold, the number of validation rows, the rule's own
        record (`f11`) and the cost model's hash.

    Raises:
        ScmleError: If the charter has no decision layer or cost model, or
            the chosen rule has no raw threshold.
    """
    layer = charter.decision
    model = charter.cost_model
    if layer is None or model is None:
        raise errors.ScmleError("the charter has no decision layer")
    chosen = scmle.decision.threshold.operate(
        probabilities,
        actual.astype(float),
        model,
        level=layer.model.calibration_level,
        replicates=charter.error_budget.permutations_b,
        seed=seed,
    )
    cut = chosen.rule.uncalibrated
    if cut is None:
        raise errors.ScmleError("the chosen rule has no raw threshold")
    description = {
        "objective": "chartered_cost",
        "threshold": cut,
        "validation_rows": rows,
        "f11": chosen.to_dict(),
        "cost_model_hash": model.model_hash,
    }
    return thresholds.Classifier.around(fitted, cut), (description)


def appraise(
    context: scmle.core.context.Context,
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
        context: The run context, supplying the charter and split.
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
    charter = decisional(context)
    if charter is None or decision is None or "f11" not in decision:
        return None
    model = charter.cost_model
    if model is None:
        raise errors.ScmleError("the charter has no cost model")
    positive = pipeline.classes_[1]
    split = scmle.core.context.require(context.split, "split")
    clusters = (
        features[context.spec.data.group_column]
        if split.manifest.strategy == "group" and context.spec.data.group_column
        else None
    )
    guard = netbenefit.FinalNetBenefit(decision["f11"]["rule_hash"])
    report = guard.evaluate(
        decision["f11"]["rule_hash"],
        (target == positive).to_numpy(dtype=float),
        (numpy.asarray(pipeline.predict(features)) == positive).astype(float),
        (model.t_lo, model.t_hi),
        clusters=clusters,
        alpha=float(charter.error_budget.confirmation.alpha_confirm),
        replicates=max(charter.error_budget.permutations_b, 500),
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
    context: scmle.core.context.Context, fitted: Any
) -> tuple[Any, dict[str, Any] | None]:
    """Optimizes the binary decision threshold on out-of-fold predictions.

    Only development data is used: each validation row is scored by a model
    that never saw it, so the locked test set stays untouched.

    Args:
        context: Run context holding the split and the selected trial.
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
    decision = context.spec.decision
    task = scmle.core.context.require(context.task, "task")
    chartered = decisional(context)
    if task is not tasks.Task.BINARY or (
        decision.threshold_objective == "none" and chartered is None
    ):
        return fitted, None
    split = scmle.core.context.require(context.split, "split")
    selected = scmle.core.context.require(context.selected, "selected")
    if chartered is None:
        governor = context.regulator
        reason = governor.overdue(
            cost.foldcost(selected) * len(split.manifest.folds),
            governor.reserve.parts.get("package", 0.0),
        )
        if reason is not None:
            context.forgo("decision threshold", reason)
            return fitted, None
    strategy = strategies.lookup(selected.strategy)
    adapter = context.registry.get(selected.model_name)
    plan = scmle.core.context.require(context.profile, "profile").feature_plan
    seed = context.spec.validation.seed
    probabilities, positions = thresholds.heldout(
        lambda: strategy.build(plan, adapter, task, seed, selected.params),
        split.dev_features,
        split.dev_target,
        split.manifest.folds,
    )
    positive = fitted.classes_[1]
    actual = (split.dev_target.iloc[positions] == positive).to_numpy()
    if chartered is not None:
        return freeze(
            fitted, probabilities, actual, chartered, seed, len(positions)
        )
    kind = thresholds.OBJECTIVES[decision.threshold_objective]
    objective = (
        kind(decision.false_positive_cost, decision.false_negative_cost)
        if decision.threshold_objective == "cost"
        else kind()
    )
    threshold, score = thresholds.argmax(probabilities, actual, objective)
    description = {
        "objective": decision.threshold_objective,
        "threshold": threshold,
        "validation_score": score,
        "validation_rows": len(positions),
    }
    return thresholds.Classifier.around(fitted, threshold), (description)
