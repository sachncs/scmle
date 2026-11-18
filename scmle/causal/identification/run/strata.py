"""Replication across strata and the Holm family of declared subgroups."""

from fractions import Fraction
from typing import Any

import numpy
import pandas

from scmle.assurance.charter import causal
from scmle.assurance.ledger import multiplicity
import scmle.assurance.ledger.budget
from scmle.causal.identification import battery
import scmle.causal.identification.certified
from scmle.causal.identification.run.estimator import Estimator

MIN_STRATUM_ROWS = 50


def replication(
    adapter: Estimator,
    c: pandas.DataFrame,
    level: float,
    pooled: float,
    alpha: float,
) -> dict[str, Any]:
    """Re-estimates inside each declared stratum and tests agreement.

    Args:
        adapter: The strategy's adapter.
        c: The confirmation rows.
        level: Confidence level.
        pooled: The pooled estimate whose sign the strata must share.
        alpha: Level of the heterogeneity test.

    Returns:
        The replication result per stratifying column under ``by_column``
        (a skip reason when the column is absent from `c` or has fewer than
        two usable strata), and ``passed``: None when no column had two usable
        strata, otherwise whether every column that did replicated. A stratum
        is usable when it has `MIN_STRATUM_ROWS` rows and `adapter.estimate`
        gives a finite positive standard error without a `ValueError`. The
        adapter's `design` and `weak` are restored afterwards, so the report
        keeps the values from the pooled estimate on `C`.
    """
    results: dict[str, Any] = {}
    passed: bool | None = None
    saved_design = dict(adapter.design)
    saved_weak = adapter.weak
    for column in adapter.spec.strata:
        if column not in c.columns:
            results[column] = {"skipped": "column not in C"}
            continue
        values: dict[str, tuple[float, float]] = {}
        for stratum, part in c.groupby(column):
            if len(part) < MIN_STRATUM_ROWS:
                continue
            try:
                est = adapter.estimate(part, level)
            except ValueError:
                # handled: documented unusable stratum; may yield skipped reason
                continue
            if numpy.isfinite(est.se) and est.se > 0:
                values[f"{column}={stratum}"] = (est.estimate, est.se)
        if len(values) < battery.MIN_STRATA:
            results[column] = {"skipped": "fewer than two usable strata"}
            continue
        rep = battery.replicate(values, pooled, alpha)
        results[column] = rep.to_dict()
        passed = rep.passed if passed is None else (passed and rep.passed)
    adapter.design = saved_design
    adapter.weak = saved_weak
    return {"by_column": results, "passed": passed}


def examine(
    adapter: Estimator,
    c: pandas.DataFrame,
    level: float,
    column: str,
    value: str,
) -> dict[str, Any]:
    """Tests one declared subgroup's effect against zero.

    Args:
        adapter: The strategy's adapter.
        c: The confirmation rows.
        level: Confidence level.
        column: The subgroup's column.
        value: The subgroup's value of `column`, as text.

    Returns:
        The hypothesis row: the column, value and Wald p-value, with the
        estimate, interval and rows when the subgroup was estimated, or the
        reason it was skipped (its p-value is then 1).
    """
    row: dict[str, Any] = {
        "column": column,
        "value": value,
        "skipped": None,
    }
    part = None
    if column not in c.columns:
        row["skipped"] = "column not in C"
    else:
        part = c[c[column].astype(str) == value]
        if len(part) < MIN_STRATUM_ROWS:
            row["skipped"] = f"fewer than {MIN_STRATUM_ROWS} rows"
    p_value = 1.0
    if part is not None and row["skipped"] is None:
        try:
            est = adapter.estimate(part, level)
        except ValueError as error:
            # handled: returned as failed finding/problem/refusal with
            # error text
            row["skipped"] = str(error)
        else:
            if numpy.isfinite(est.se) and est.se > 0:
                test = battery.wald(
                    f"{column}={value}",
                    "subgroup",
                    est.estimate,
                    est.se,
                )
                p_value = test.p_value
                row.update(
                    estimate=est.estimate,
                    se=est.se,
                    low=est.low,
                    high=est.high,
                    n=est.n,
                )
            else:
                row["skipped"] = "no usable standard error"
    row["p_value"] = p_value
    return row


def stratify(
    adapter: Estimator,
    c: pandas.DataFrame,
    level: float,
    subgroups: tuple[causal.Subgroup, ...],
) -> dict[str, Any]:
    """Estimates the effect in each declared subgroup, adjusted as one family.

    The family is every declared (column, value) pair, fixed by the charter
    and registered with the primary claim before `C` is opened. Each
    hypothesis `theta_g = 0` is tested by a Wald statistic of the strategy's
    own estimator on the rows of `C` in the subgroup, and the p-values are
    adjusted by Holm's step-down at the primary level `1 - level`, which
    controls the familywise error rate under any dependence. A subgroup that
    cannot be estimated (column absent, fewer than `MIN_STRATUM_ROWS` rows, no
    usable standard error) stays in the family with p = 1, so a missing
    subgroup never shrinks the correction. The result describes
    heterogeneity; it does not enter the tier.

    Args:
        adapter: The strategy's adapter.
        c: The confirmation rows.
        level: Confidence level; the family level is `1 - level`.
        subgroups: The declared subgroups.

    Returns:
        A mapping with the rule, the family level `alpha` (`1 - level`), the
        family size, the guarantee `statement`, one row per hypothesis under
        `hypotheses` (estimate, p-value, adjusted p-value, decision, or why
        it was skipped) and the names of the `rejected` subgroups. The
        adapter's `design` is restored afterwards; its `weak` flag is not
        (`identify` restores it).
    """
    saved = dict(adapter.design)
    rows = [
        examine(adapter, c, level, group.column, value)
        for group in subgroups
        for value in group.values  # noqa: PD011 -- `Subgroup.values` is a dataclass field, not a pandas attribute
    ]
    adapter.design = saved
    alpha = Fraction(1.0 - level).limit_denominator(10**9)
    adjusted = multiplicity.adjust(
        [r["p_value"] for r in rows],
        scmle.assurance.ledger.budget.Correction.HOLM,
        alpha,
    )
    for row, p_adj, flag in zip(
        rows, adjusted.adjusted, adjusted.rejected, strict=True
    ):
        row["adjusted_p"] = p_adj
        row["rejected"] = bool(flag)
    return {
        "rule": scmle.assurance.ledger.budget.Correction.HOLM.value,
        "alpha": float(alpha),
        "family_size": len(rows),
        "statement": multiplicity.STATEMENTS[
            scmle.assurance.ledger.budget.Correction.HOLM
        ],
        "hypotheses": rows,
        "rejected": [
            f"{r['column']}={r['value']}" for r in rows if r["rejected"]
        ],
    }


def certify(found: dict[str, Any], named: tuple[str, ...] | None) -> None:
    """Labels the subgroup report as gating or only descriptive, in place.

    The Holm family is certified by `subgroup_certificates.json` (gate
    `subgroup_holm`): both its error rates and its power met their
    pre-registered floors on this code. With the certificate the word is
    `gating`, and a subgroup with at least the certified number of rows is
    marked `certified_size`; without it the family is `descriptive`. The tier
    is not read from this: the tier rule has no subgroup input.

    Args:
        found: The `stratify` result; gets `gate`, `statement_gate`
            and a `certified_size` flag on each hypothesis row.
        named: The certified gates the caller names (as `identify`'s
            `certified`), or None to read the shipped certificate. When
            given, it alone decides the status, but `min_rows` still comes
            from the shipped certificate (None when it is absent, so every
            estimated subgroup then counts as certified size).
    """
    gate = scmle.causal.identification.certified.subgroup_gate(stratify)
    if named is not None:
        listed = scmle.causal.identification.certified.SUBGROUP_GATE in named
        gate = scmle.causal.identification.certified.SubgroupGate(
            scmle.causal.identification.certified.GATING
            if listed
            else scmle.causal.identification.certified.DESCRIPTIVE,
            "named among the certified gates"
            if listed
            else "the caller named the certified gates and this is not one",
            gate.min_rows,
        )
    found["gate"] = gate.to_dict()
    gating = gate.status == scmle.causal.identification.certified.GATING
    for row in found["hypotheses"]:
        size = row.get("n")
        row["certified_size"] = bool(
            gating
            and size is not None
            and (gate.min_rows is None or size >= gate.min_rows)
        )
    found["statement_gate"] = (
        "gating: the family-wise error and the power at the certified size "
        "met their pre-registered floors (a result for a subgroup under "
        "`min_rows` is still descriptive)"
        if gating
        else "descriptive: the family-wise error and power of the subgroup "
        "tests are not certified, so read a rejected level as exploratory; "
        f"{gate.reason}"
    )


def subgroup_results(
    adapter: Estimator,
    spec: causal.Question,
    c: pandas.DataFrame,
    level: float,
    named: tuple[str, ...] | None,
) -> dict[str, Any] | None:
    """Estimates and labels the declared subgroups.

    The adapter's `weak` flag is restored afterwards.

    Args:
        adapter: The strategy's estimator.
        spec: The charter's causal block.
        c: The confirmation rows.
        level: Confidence level.
        named: The certified gates the caller names, or None to read the
            shipped certificate (see `certify`).

    Returns:
        The labelled subgroup report, or None when no subgroups are
        declared or there is a single treated unit.
    """
    weak_before_subgroups = adapter.weak
    found = (
        stratify(adapter, c, level, spec.subgroups)
        if spec.subgroups and not adapter.single_unit
        else None
    )
    adapter.weak = weak_before_subgroups
    if found is not None:
        certify(found, named)
    return found
