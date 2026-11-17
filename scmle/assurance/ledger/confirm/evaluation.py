"""Claims and checks built from the frozen survivors, and the gates."""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from fractions import Fraction
from typing import Any

import pandas

from scmle.assurance.ledger import bivariate
from scmle.assurance.ledger import budget
from scmle.assurance.ledger import multiplicity
from scmle.assurance.ledger.confirm import records
import scmle.assurance.ledger.ledger
from scmle.contracts import qualification
from scmle.runtime import randomness


def assumptions(
    rule: budget.Correction, size: int, withdrawn: int = 0
) -> tuple[str, ...]:
    """Lists the assumptions a confirmed claim depends on.

    Args:
        rule: Multiplicity rule used for the family.
        size: Number of survivors in the tested family.
        withdrawn: Number of survivors withdrawn before `C` was opened.

    Returns:
        One sentence per assumption; a sentence on the withdrawal is added
        when `withdrawn` is positive.
    """
    lines = [
        (
            f"{multiplicity.STATEMENTS[rule]} (rule {rule.value}, family of "
            f"{size} survivors fixed before C was opened)"
        ),
        (
            "rows of C are independent of rows of X (grouped split by entity, "
            "embargo for temporal data)"
        ),
        "C was opened once and its rows were used only by this confirmation",
        (
            "each statistic is valid under its null on C: Spearman and Kruskal "
            "assume independent rows"
        ),
        "exploration FDR is a triage budget, not part of this guarantee",
    ]
    if withdrawn:
        lines.insert(
            1,
            f"{withdrawn} survivor(s) were withdrawn before C was opened by "
            "a rule computed from the exploration set only; the family above "
            "is the survivors that remain",
        )
    return tuple(lines)


def statement(spec: scmle.assurance.ledger.ledger.Test) -> str:
    """Writes the sentence of a claim.

    Args:
        spec: The hypothesis specification.

    Returns:
        The hypothesis name followed by the null it is evidence against.
    """
    return f"{spec.name}: evidence against '{spec.null}'"


def disclaim(
    ledger: scmle.assurance.ledger.ledger.Ledger, why: str
) -> tuple[qualification.Claim, ...]:
    """Builds claims for survivors that were not (or could not be) confirmed.

    Args:
        ledger: The ledger whose survivors are frozen.
        why: Reason, recorded in each claim's assumptions.

    Returns:
        One unverified claim per survivor, in survivor order.

    Raises:
        LedgerError: If the survivors have not been frozen.
    """
    survivors = ledger.survivor_set
    if survivors is None:
        raise scmle.assurance.ledger.ledger.LedgerError(
            "freeze the survivors first"
        )
    claims = []
    for hypothesis_id in survivors.hypothesis_ids:
        result = ledger.results[hypothesis_id]
        spec = ledger.hypotheses[hypothesis_id].spec
        claims.append(
            qualification.Claim(
                identifier=hypothesis_id,
                statement=statement(spec),
                checks=(f"{hypothesis_id}:X",),
                adjusted_p_value=None,
                effect=result.check.effect if result.check else 0.0,
                tier=qualification.Tier.UNVERIFIED,
                assumptions=(
                    f"not confirmed on C: {why}",
                    (
                        "exploration finding only; it does not change feature "
                        "admission or preprocessing"
                    ),
                ),
            )
        )
    return tuple(claims)


def withdraw(
    ledger: scmle.assurance.ledger.ledger.Ledger, reasons: Mapping[str, str]
) -> dict[str, qualification.Claim]:
    """Builds unverified claims for survivors withdrawn before `C` opened.

    Args:
        ledger: The ledger holding the survivors' results.
        reasons: Hypothesis id to the sentence that withdrew it.

    Returns:
        Hypothesis id to an unverified claim that records the reason.
    """
    claims = {}
    for hypothesis_id, why in reasons.items():
        result = ledger.results[hypothesis_id]
        spec = ledger.hypotheses[hypothesis_id].spec
        claims[hypothesis_id] = qualification.Claim(
            identifier=hypothesis_id,
            statement=statement(spec),
            checks=(f"{hypothesis_id}:X",),
            adjusted_p_value=None,
            effect=result.check.effect if result.check else 0.0,
            tier=qualification.Tier.UNVERIFIED,
            assumptions=(
                f"withdrawn before C was opened, not tested on C: {why}",
                (
                    "exploration finding only; it does not change feature "
                    "admission or preprocessing"
                ),
            ),
        )
    return claims


def stage_two_record(
    ledger: scmle.assurance.ledger.ledger.Ledger, withdrawal: records.Withdrawal
) -> dict[str, Any]:
    """Builds the accounting of a withdrawal, hashed over the surviving family.

    Args:
        ledger: The ledger; its survivors are frozen if they are not yet.
        withdrawal: The withdrawal to account for.

    Returns:
        A mapping with the survivors and their hash, the withdrawn ids with
        reasons, the remaining family and its hash, the rule and the
        guarantee statement.

    Raises:
        LedgerError: If a withdrawn id is not a frozen survivor.
    """
    survivors = ledger.freeze()
    unknown = sorted(set(withdrawal.reasons) - set(survivors.hypothesis_ids))
    if unknown:
        raise scmle.assurance.ledger.ledger.LedgerError(
            f"withdrawn ids {unknown} are not frozen survivors"
        )
    family = [
        i for i in survivors.hypothesis_ids if i not in withdrawal.reasons
    ]
    return {
        "survivors_sha256": survivors.sha256,
        "survivors": list(survivors.hypothesis_ids),
        "withdrawn": dict(sorted(withdrawal.reasons.items())),
        "family": family,
        "family_sha256": scmle.assurance.ledger.ledger.digest(
            {"survivors_sha256": survivors.sha256, "family": family}
        ),
        "rule": dict(withdrawal.rule),
        "statement": records.STAGE_TWO_STATEMENT,
    }


def verify(
    ledger: scmle.assurance.ledger.ledger.Ledger,
    rows: tuple[pandas.DataFrame, pandas.Series],
    rule: budget.Correction,
    alpha: Fraction,
    encode: Callable[[pandas.Series], Any],
    stage_two: Mapping[str, Any] | None = None,
) -> tuple[tuple[qualification.Claim, ...], tuple[qualification.Check, ...]]:
    """Tests the survivors on the opened rows and builds checks and claims.

    Args:
        ledger: The ledger with frozen survivors.
        rows: The opened rows of `C`.
        rule: Multiplicity rule.
        alpha: Confirmation level.
        encode: Target encoder.
        stage_two: Accounting from `stage_two_record`; the survivors in its
            ``withdrawn`` mapping are left out of the tested family and get
            unverified claims instead.

    Returns:
        One claim per frozen survivor, in survivor order (withdrawn ones
        included), and the checks on `C`, which cover only the tested
        family.

    Raises:
        LedgerError: If the survivors have not been frozen.
    """
    survivors = ledger.survivor_set
    if survivors is None:
        raise scmle.assurance.ledger.ledger.LedgerError(
            "freeze the survivors first"
        )
    features, target = rows
    withdrawn = {} if stage_two is None else dict(stage_two["withdrawn"])
    ids = tuple(i for i in survivors.hypothesis_ids if i not in withdrawn)
    codes = encode(target)
    outcomes = measure(ledger, ids, features, codes)
    adjustment = multiplicity.adjust([o.p for o in outcomes], rule, alpha)
    checks = []
    claims = []
    for index, hypothesis_id in enumerate(ids):
        spec = ledger.hypotheses[hypothesis_id].spec
        outcome = outcomes[index]
        check = examine(
            ledger, hypothesis_id, index, outcome, adjustment.rejected[index]
        )
        checks.append(check)
        claims.append(
            qualification.Claim(
                identifier=hypothesis_id,
                statement=statement(spec),
                checks=(f"{hypothesis_id}:X", f"{hypothesis_id}:C"),
                adjusted_p_value=adjustment.adjusted[index],
                effect=outcome.effect,
                tier=check.tier,
                assumptions=assumptions(rule, len(ids), len(withdrawn)),
            )
        )
    if withdrawn:
        by_id = {c.identifier: c for c in claims}
        by_id.update(withdraw(ledger, withdrawn))
        claims = [by_id[i] for i in survivors.hypothesis_ids]
    return tuple(claims), tuple(checks)


def measure(
    ledger: scmle.assurance.ledger.ledger.Ledger,
    ids: Sequence[str],
    features: pandas.DataFrame,
    codes: Any,
) -> list[bivariate.Outcome]:
    """Runs each hypothesis' registered statistic on the opened rows.

    Args:
        ledger: The ledger holding the specifications.
        ids: The hypotheses of the tested family, in survivor order.
        features: The opened features of `C`; the column named by each
            specification is read, or the first column if it names none.
        codes: The encoded target of `C`.

    Returns:
        One outcome per hypothesis, in the order of `ids`.
    """
    outcomes = []
    for hypothesis_id in ids:
        spec = ledger.hypotheses[hypothesis_id].spec
        column = features[spec.column] if spec.column else features.iloc[:, 0]
        outcomes.append(
            bivariate.run_statistic(
                spec.statistic,
                column.to_numpy(dtype=float),
                codes,
                spec.direction,
            )
        )
    return outcomes


def examine(
    ledger: scmle.assurance.ledger.ledger.Ledger,
    hypothesis_id: str,
    index: int,
    outcome: bivariate.Outcome,
    passed: bool,
) -> qualification.Check:
    """Builds the check on `C` of one hypothesis of the tested family.

    Args:
        ledger: The ledger holding the specification and the seed.
        hypothesis_id: The hypothesis.
        index: Its position in the tested family.
        outcome: Its statistic on `C`.
        passed: Whether the multiplicity rule rejected its null.

    Returns:
        The check, at tier `confirmed` when `passed`, otherwise `unverified`.
    """
    spec = ledger.hypotheses[hypothesis_id].spec
    return qualification.Check(
        identifier=f"{hypothesis_id}:C",
        null=spec.null,
        statistic=outcome.statistic,
        p_value=outcome.p,
        interval=None,
        effect=outcome.effect,
        n=outcome.n,
        decision=(
            qualification.Decision.WARN
            if passed
            else qualification.Decision.PASS
        ),
        subset=qualification.Subset.CONFIRMATION,
        seeds=randomness.seed_path(
            ledger.seed, f"confirm.{spec.statistic}", index
        ),
        tier=(
            qualification.Tier.CONFIRMED
            if passed
            else qualification.Tier.UNVERIFIED
        ),
    )


def default_encoding(target: pandas.Series) -> Any:
    """Encodes the target as floats; classes are coded in sorted order.

    Args:
        target: The opened confirmation target.

    Returns:
        A float array: the values of a numeric target, or the code of each
        label otherwise (labels sorted as text), NaN where missing.
    """
    if pandas.api.types.is_numeric_dtype(target):
        return target.to_numpy(dtype=float)
    classes = sorted(target.dropna().unique(), key=str)
    codes = {label: float(i) for i, label in enumerate(classes)}
    return target.map(codes).to_numpy(dtype=float)


def block(
    ledger: scmle.assurance.ledger.ledger.Ledger,
    require_control: bool,
) -> tuple[str, str, tuple[qualification.Claim, ...]] | None:
    """Decides why the bivariate family cannot be promoted, if it cannot.

    The gates apply in this order: no survivors, no control, control floor
    not exceeded.

    Args:
        ledger: The ledger; its survivors are frozen if they are not yet.
        require_control: Whether a missing negative control blocks promotion.

    Returns:
        None when the family can be promoted, otherwise the outcome code, the
        sentence and the unverified claims.
    """
    survivors = ledger.freeze()
    if not survivors.hypothesis_ids:
        return (
            records.NO_SURVIVORS,
            "no hypothesis survived exploration; C was not opened",
            (),
        )
    control = ledger.control
    if control is None and require_control:
        return (
            records.NO_CONTROL,
            "no negative control was recorded; no claim is promoted",
            disclaim(ledger, "no negative control"),
        )
    if control is not None and not control["exceeds"]:
        return (
            records.FLOOR_NOT_EXCEEDED,
            str(control["message"]) + "; no claim is promoted",
            disclaim(
                ledger, "findings do not exceed the false-discovery floor"
            ),
        )
    return None
