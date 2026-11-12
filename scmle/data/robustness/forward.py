"""The forward-in-time check on the holdout `F`.

It runs after the survivors are fixed and only for temporal data. Nothing is
fitted on `F`: each surviving finding's rank correlation is re-estimated on
the forward rows and compared with its sign on `X`. `F` is a sealed set that
opens once; this module asks for the release token once, opens once, and
the custody log records the open under the actor `f6-forward`.

`C` tests replication; `F` tests stability over time. The two are never
merged, and a forward result never changes a `confirmed` tier.

Labels, per finding, by the bootstrap interval of the effect on `F`:

* `holds`: the interval excludes zero on the same side as the effect on `X`;
* `unstable_over_time`: it excludes zero on the opposite side;
* `inconclusive`: it contains zero (the forward set may simply be small).

With non-temporal data, no `F`, or no surviving finding, the check is
skipped with a reason and, in the last case, `F` is not opened.
"""

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
import itertools
import math
import operator
from typing import Any

import numpy
import pandas

from scmle.contracts import qualification
from scmle.core import randomness
from scmle.data.ledger import confirm
from scmle.data.robustness import stability

ACTOR = "f6-forward"
CHECK_ID = "f6-forward"
HOLDS = "holds"
UNSTABLE = "unstable_over_time"
INCONCLUSIVE = "inconclusive"
BOOTSTRAP = 1000
CONFIDENCE = 0.95


@dataclasses.dataclass(frozen=True, slots=True)
class ForwardFinding:
    """One finding evaluated on the forward set.

    Attributes:
        name: The finding (feature name).
        effect_x: Its effect on the exploration set.
        effect_f: Its effect on `F`, or None when not defined.
        label: `holds`, `unstable_over_time` or `inconclusive`.
        check: The `Check` record on `F`, or None when not defined.
    """

    name: str
    effect_x: float
    effect_f: float | None
    label: str
    check: qualification.Check | None

    def to_dict(self) -> dict[str, Any]:
        """Converts the record to a JSON-ready dictionary.

        Returns:
            The fields by name, with `check` as its own dictionary or None.
        """
        return {
            "name": self.name,
            "effect_x": self.effect_x,
            "effect_f": self.effect_f,
            "label": self.label,
            "check": None if self.check is None else self.check.to_dict(),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class ForwardCheck:
    """Outcome of the forward check.

    Attributes:
        status: `checked` or `skipped`.
        reason: One sentence saying why.
        opened: Whether `F` was opened (at most once).
        findings: One record per finding when checked.
    """

    status: str
    reason: str
    opened: bool
    findings: tuple[ForwardFinding, ...]

    def to_dict(self) -> dict[str, Any]:
        """Converts the record to a JSON-ready dictionary.

        Returns:
            The fields by name, with `findings` as a list of dictionaries.
        """
        return {
            "status": self.status,
            "reason": self.reason,
            "opened": self.opened,
            "findings": [f.to_dict() for f in self.findings],
        }


def skipped(reason: str) -> ForwardCheck:
    """Builds a result that says the check did not run, and why.

    Args:
        reason: One sentence saying why the check was skipped.

    Returns:
        A `skipped` result with `opened` False and no findings.
    """
    return ForwardCheck(
        status="skipped", reason=reason, opened=False, findings=()
    )


def label_for(effect_x: float, low: float, high: float) -> str:
    """Labels a finding from its forward interval and its sign on `X`.

    Args:
        effect_x: The finding's effect on the exploration set; only its sign
            is used (zero counts as positive).
        low: Lower end of the interval of the effect on `F`.
        high: Upper end of the interval of the effect on `F`.

    Returns:
        `holds` when the interval excludes zero on the same side as
        `effect_x`, `unstable_over_time` when it excludes zero on the
        opposite side, and `inconclusive` when it contains zero.
    """
    sign = math.copysign(1.0, effect_x)
    if low > 0.0 or high < 0.0:
        return HOLDS if (low > 0.0) == (sign > 0.0) else UNSTABLE
    return INCONCLUSIVE


def evaluate_finding(
    name: str,
    effect_x: float,
    values: numpy.ndarray,
    target: numpy.ndarray,
    seed: int,
    bootstrap: int,
) -> ForwardFinding:
    """Re-estimates one finding on the forward rows and labels it.

    The interval is a percentile bootstrap of the rank correlation, widened
    if needed so that it contains the point estimate.

    Args:
        name: The finding (feature name).
        effect_x: Its effect on the exploration set.
        values: The feature values on the forward rows.
        target: The encoded target on the forward rows, aligned with
            `values`.
        seed: Run seed from which the bootstrap generator is derived.
        bootstrap: Number of resamples.

    Returns:
        The finding with its forward effect, label and `Check` record. When
        the rank correlation is not defined on the forward rows, the effect
        and the `Check` are None and the label is `inconclusive`.
    """
    effect = stability.rank_effect(values, target)
    if not math.isfinite(effect):
        return ForwardFinding(name, effect_x, None, INCONCLUSIVE, None)
    rng = randomness.component_rng(seed, f"{CHECK_ID}.{name}", 0)
    rows = len(values)
    picks = list(
        map(
            operator.call,
            itertools.repeat(
                lambda: rng.integers(0, rows, size=rows), bootstrap
            ),
        )
    )
    draws = numpy.array(
        [stability.rank_effect(values[p], target[p]) for p in picks]
    )
    draws = draws[numpy.isfinite(draws)]
    tail = (1.0 - CONFIDENCE) / 2.0 * 100.0
    low, high = (
        (float(v) for v in numpy.percentile(draws, (tail, 100.0 - tail)))
        if draws.size
        else (effect, effect)
    )
    low, high = min(low, effect), max(high, effect)
    label = label_for(effect_x, low, high)
    check = qualification.Check(
        check_id=f"{CHECK_ID}:{name}",
        null_hypothesis="the finding's effect on the forward set is zero",
        statistic=effect,
        p_value=None,
        interval=(low, high),
        effect=effect,
        n=rows,
        decision=(
            qualification.Decision.WARN
            if label == UNSTABLE
            else qualification.Decision.PASS
        ),
        set_id=qualification.SetId.FORWARD,
        seed_path=randomness.seed_path(seed, f"{CHECK_ID}.{name}", 0),
        tier=qualification.Tier.UNVERIFIED,
    )
    return ForwardFinding(name, effect_x, effect, label, check)


def forward_check(
    source: confirm.ConfirmationSource | None,
    effects: Mapping[str, float],
    encode: Callable[[pandas.Series], numpy.ndarray],
    *,
    temporal: bool,
    seed: int,
    bootstrap: int = BOOTSTRAP,
) -> ForwardCheck:
    """Opens `F` once and evaluates the surviving findings on it.

    Args:
        source: The sealed forward set, or None when there is none.
        effects: Finding (feature name) to its effect on `X`. The findings
            must already be fixed; nothing is selected here.
        encode: Turns the forward target into the codes used on `X`.
        temporal: Whether the data have a time order. Non-temporal data skip
            the check.
        seed: Run seed.
        bootstrap: Resamples for the interval.

    Returns:
        The result. `F` is opened at most once and not at all when the check
        is skipped. A finding whose feature is missing from `F` is reported
        as `inconclusive` with no forward effect.
    """
    if not temporal:
        return skipped("the data are not temporal: no forward check")
    if source is None:
        return skipped("no forward set was sealed")
    if not effects:
        return skipped("no surviving finding to check; F was not opened")
    token = source.issue_token()
    features, target = source.open(token, ACTOR)
    codes = encode(target)
    findings = []
    for name, effect_x in effects.items():
        if name not in features.columns:
            findings.append(
                ForwardFinding(name, effect_x, None, INCONCLUSIVE, None)
            )
            continue
        findings.append(
            evaluate_finding(
                name,
                effect_x,
                features[name].to_numpy(dtype=float),
                numpy.asarray(codes, dtype=float),
                seed,
                bootstrap,
            )
        )
    return ForwardCheck(
        status="checked",
        reason="F was opened once and nothing was fitted on it",
        opened=True,
        findings=tuple(findings),
    )
