"""The exploration phase: one registered hypothesis per feature.

For every usable feature the exploration plan registers a bivariate
hypothesis (feature against target) *before* any statistic is computed, then
the ledger tests them in registration order under the online rule. The plan is
a function of the feature names and the target kind only, never of any
result.

Nothing here imports custody or reads the confirmation set: the functions take
exploration data as arguments and cannot reach anything else (an import-graph
test enforces it for the whole package).
"""

from collections.abc import Sequence
import dataclasses
from typing import Any

import numpy as np
import pandas as pd

from scmle.data.ledger import bivariate
from scmle.data.ledger import ledger as ledger_module
from scmle.data.ledger import online_fdr

NUMERIC = "numeric"
BINARY = "binary"
MULTICLASS = "multiclass"
FAMILY = "bivariate"


@dataclasses.dataclass(frozen=True, slots=True)
class Plan:
    """Hypotheses to register, and the features that were left out.

    Attributes:
        specs: One specification per usable feature, in column order.
        skipped: (column, reason) for each feature not tested.
        target_kind: `numeric`, `binary` or `multiclass`.
    """

    specs: tuple[ledger_module.HypothesisSpec, ...]
    skipped: tuple[tuple[str, str], ...]
    target_kind: str


def target_kind(target: pd.Series, classification: bool) -> str:
    """Decides how the target is treated.

    Args:
        target: The exploration target.
        classification: Whether the task is classification.

    Returns:
        `numeric` for regression, otherwise `binary` when the target has at
        most two distinct non-missing values and `multiclass` when it has
        more.
    """
    if not classification:
        return NUMERIC
    return BINARY if target.nunique(dropna=True) <= 2 else MULTICLASS


def encode_target(
    target: pd.Series,
    classification: bool,
    classes: Sequence[Any] | None = None,
) -> pd.Series:
    """The target as floats: class codes for classification.

    Codes follow the sorted order of the classes (or `classes` when given, so
    the confirmation set is coded exactly as exploration was), so they do not
    depend on the order in which rows appear. Missing labels stay missing.

    Args:
        target: The target to encode.
        classification: Whether to code class labels (otherwise the target
            is cast to float).
        classes: Fixed class order for the codes, or None to use the sorted
            distinct labels of `target`.

    Returns:
        A float series with the index of `target`: the class code of each
        label for classification (NaN where the label is missing or not in
        `classes`), or the values themselves for regression.

    Raises:
        ValueError: If a regression target is not numeric.
    """
    if classification:
        order = (
            sorted(target.dropna().unique(), key=str)
            if classes is None
            else list(classes)
        )
        codes = {label: float(i) for i, label in enumerate(order)}
        return target.map(codes).astype(float)
    if not pd.api.types.is_numeric_dtype(target):
        raise ValueError("a regression target must be numeric")
    return target.astype(float)


def plan_exploration(
    features: pd.DataFrame, kind: str, exclude: tuple[str, ...] = ()
) -> Plan:
    """Chooses the hypotheses from the feature names and the target kind.

    Args:
        features: Exploration features.
        kind: `numeric`, `binary` or `multiclass`.
        exclude: Columns that are structure, not candidates (group, time).

    Returns:
        The plan. Non-numeric features, all-missing and constant features are
        listed in `skipped`: they are descriptive only.
    """
    statistic = "kruskal" if kind == MULTICLASS else "spearman"
    specs = []
    skipped = []
    for name in features.columns:
        if name in exclude:
            skipped.append((str(name), "structure column, not a candidate"))
            continue
        column = features[name]
        if not pd.api.types.is_numeric_dtype(column):
            skipped.append((str(name), "not numeric: descriptive only"))
            continue
        values = column.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) == 0 or float(np.ptp(finite)) == 0.0:
            skipped.append((str(name), "constant or entirely missing"))
            continue
        specs.append(
            ledger_module.HypothesisSpec.create(
                name=f"{name}~target",
                family=FAMILY,
                statistic=statistic,
                null=(
                    f"{name} is not monotonically associated with the target"
                    if statistic == "spearman"
                    else f"{name} has the same distribution in every class"
                ),
                column=str(name),
            )
        )
    return Plan(tuple(specs), tuple(skipped), kind)


def run_exploration(
    ledger: ledger_module.Ledger,
    features: pd.DataFrame,
    codes: pd.Series,
    plan: Plan,
) -> tuple[ledger_module.Result, ...]:
    """Registers every planned hypothesis, then tests them in order.

    Args:
        ledger: The run's ledger.
        features: Exploration features.
        codes: Encoded exploration target.
        plan: The plan.

    Returns:
        The recorded results, in registration order.
    """
    handles = [ledger.register(spec) for spec in plan.specs]
    return tuple(ledger.test(handle, features, codes) for handle in handles)


def count_discoveries(
    config: online_fdr.RuleConfig,
    plan: Plan,
    features: pd.DataFrame,
    codes: pd.Series,
) -> int:
    """Counts the rejections of the whole exploration on the given data.

    It applies the same statistics and the same online rule as
    `run_exploration`, without writing a ledger; the negative controls call it
    once per re-run.

    Args:
        config: Configuration of the online rule to build.
        plan: The exploration plan whose hypotheses are tested in order.
        features: Exploration features.
        codes: Encoded exploration target.

    Returns:
        The number of hypotheses the online rule rejects. A hypothesis
        arriving when the rule's level is zero is given ``p = 1``.
    """
    rule = config.build()
    state = rule.initial()
    target = codes.to_numpy(dtype=float)
    rejected = 0
    for spec in plan.specs:
        if rule.level(state) == 0:
            p = 1.0
        else:
            column = features[spec.column].to_numpy(dtype=float)
            p = bivariate.run_statistic(
                spec.statistic, column, target, spec.direction
            ).p
        step = rule.step(state, online_fdr.Evidence(p=p))
        rejected += int(step.rejected)
        state = step.state
    return rejected
