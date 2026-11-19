"""Reads the scorecard thresholds and decides the statistical rows.

The thresholds live in `acceptance.toml` at the repository root (plan
section 7). Each is a `Variable` with a value, unit, kind, claim and
derivation; a statistical variable also has a rule, `Ceiling` (a rate at
most the value) or `Floor` (a rate at least the value). Both rules decide a
row the same way, by a one-sided Clopper-Pearson bound at `alpha`, and both
report the smallest sample that can pass at all and the sample needed for a
given power, so the harness refuses to report a pass on fewer trials than
the rule can support.

    python -m benchmarks.acceptance            # validate, print the table
"""

import dataclasses
import json
import math
import pathlib
import sys
import tomllib

from scipy import stats

from scmle.evaluation import coverage

PATH = pathlib.Path(__file__).resolve().parents[1] / "acceptance.toml"
KINDS = frozenset({"statistical", "service", "response", "policy"})
FIELDS = frozenset({"value", "unit", "kind", "claim", "derivation", "rule"})
LIMIT = 100_000


def size(null: float, truth: float, alpha: float, power: float) -> int:
    """Trials needed to tell a true rate from the null rate.

    The normal approximation for a one-sided test of one proportion:
    `[z(1-alpha) sqrt(p0 (1-p0)) + z(power) sqrt(p1 (1-p1))]^2 / (p0-p1)^2`.

    Args:
        null: The rate the row is decided against, `p0`.
        truth: The rate assumed true, `p1`; must differ from `null`.
        alpha: One-sided error rate.
        power: Chance of passing when `truth` holds.

    Returns:
        The number of trials, rounded up.

    Raises:
        ValueError: If `truth` equals `null`.
    """
    if truth == null:
        raise ValueError("the true rate must differ from the null rate")
    spread = float(stats.norm.ppf(1 - alpha)) * math.sqrt(null * (1 - null))
    spread += float(stats.norm.ppf(power)) * math.sqrt(truth * (1 - truth))
    return math.ceil(spread**2 / (null - truth) ** 2)


def search(
    rule: "Ceiling | Floor", truth: float, alpha: float, power: float
) -> int:
    """Smallest sample whose exact chance of passing reaches `power`.

    The normal approximation in `size` can fall well short near 0 or 1 (at
    a true rate of 0.01 against a ceiling of 0.05 it gives 123 trials, whose
    exact power is 0.65), so the plan is the exact one; `size` is only
    where the search stops looking.

    Args:
        rule: The rule deciding the row.
        truth: The rate assumed true; on the passing side of the bound.
        alpha: One-sided error rate.
        power: Chance of passing wanted.

    Returns:
        The number of trials.

    Raises:
        ValueError: If `truth` is not on the passing side of the bound, or
            no sample up to `LIMIT` trials reaches `power`.
    """
    passing = (
        truth < rule.bound if isinstance(rule, Ceiling) else truth > rule.bound
    )
    if not passing:
        raise ValueError("the true rate must be on the passing side")
    trials = rule.minimum(alpha)
    while rule.chance(truth, trials, alpha) < power:
        trials += 1
        if trials > LIMIT:
            raise ValueError(f"no sample up to {LIMIT} trials reaches {power}")
    return trials


@dataclasses.dataclass(frozen=True)
class Ceiling:
    """A rate that passes when its upper bound is at most `bound`.

    Attributes:
        bound: The largest rate allowed, in (0, 1).
    """

    bound: float

    def passes(self, hits: int, trials: int, alpha: float) -> bool:
        """Decides the row from its counts.

        Args:
            hits: Trials that showed the event (a false block, a refusal).
            trials: Trials run.
            alpha: One-sided error rate.

        Returns:
            Whether `BetaInv(1 - alpha; hits + 1, trials - hits)` is at
            most `bound`. False with no trials.
        """
        if trials == 0:
            return False
        return self.limit(hits, trials, alpha) <= self.bound

    def limit(self, hits: int, trials: int, alpha: float) -> float:
        """The one-sided upper bound the row is decided by.

        Args:
            hits: Trials that showed the event.
            trials: Trials run.
            alpha: One-sided error rate.

        Returns:
            `BetaInv(1 - alpha; hits + 1, trials - hits)`; 1 with no trials.
        """
        return coverage.clopper_pearson(hits, trials, 1 - 2 * alpha)[1]

    def minimum(self, alpha: float) -> int:
        """Smallest number of trials that can pass, with no event at all.

        Args:
            alpha: One-sided error rate.

        Returns:
            `ceil(ln(alpha) / ln(1 - bound))`.
        """
        return math.ceil(math.log(alpha) / math.log1p(-self.bound))

    def chance(self, truth: float, trials: int, alpha: float) -> float:
        """Exact chance of passing when the true rate is `truth`.

        Fewer events never make the row fail, so the passing counts are
        `0..k` for the largest passing `k`.

        Args:
            truth: True rate of the event.
            trials: Trials run.
            alpha: One-sided error rate.

        Returns:
            `P(hits <= k)` under `Binomial(trials, truth)`; 0 if no count
            passes.
        """
        most = -1
        while most < trials and self.passes(most + 1, trials, alpha):
            most += 1
        return float(stats.binom.cdf(most, trials, truth)) if most >= 0 else 0.0

    def plan(self, truth: float, alpha: float, power: float) -> int:
        """Trials needed to pass with `power` when the true rate is `truth`.

        Args:
            truth: True rate, below `bound`.
            alpha: One-sided error rate.
            power: Chance of passing.

        Returns:
            The smallest number of trials, from `minimum` up, whose exact
            chance of passing is at least `power`.
        """
        return search(self, truth, alpha, power)


@dataclasses.dataclass(frozen=True)
class Floor:
    """A rate that passes when its lower bound is at least `bound`.

    Attributes:
        bound: The smallest rate allowed, in (0, 1).
    """

    bound: float

    def passes(self, hits: int, trials: int, alpha: float) -> bool:
        """Decides the row from its counts.

        Args:
            hits: Trials that succeeded (a leak caught, an interval covering).
            trials: Trials run.
            alpha: One-sided error rate.

        Returns:
            Whether `BetaInv(alpha; hits, trials - hits + 1)` is at least
            `bound`. False with no trials.
        """
        if trials == 0:
            return False
        return self.limit(hits, trials, alpha) >= self.bound

    def limit(self, hits: int, trials: int, alpha: float) -> float:
        """The one-sided lower bound the row is decided by.

        Args:
            hits: Trials that succeeded.
            trials: Trials run.
            alpha: One-sided error rate.

        Returns:
            `BetaInv(alpha; hits, trials - hits + 1)`; 0 with no trials.
        """
        return coverage.clopper_pearson(hits, trials, 1 - 2 * alpha)[0]

    def minimum(self, alpha: float) -> int:
        """Smallest number of trials that can pass, with every trial a hit.

        Args:
            alpha: One-sided error rate.

        Returns:
            `ceil(ln(alpha) / ln(bound))`.
        """
        return math.ceil(math.log(alpha) / math.log(self.bound))

    def chance(self, truth: float, trials: int, alpha: float) -> float:
        """Exact chance of passing when the true rate is `truth`.

        More hits never make the row fail, so the passing counts are
        `k..trials` for the smallest passing `k`.

        Args:
            truth: True success rate.
            trials: Trials run.
            alpha: One-sided error rate.

        Returns:
            `P(hits >= k)` under `Binomial(trials, truth)`; 0 if no count
            passes.
        """
        least = trials + 1
        while least > 0 and self.passes(least - 1, trials, alpha):
            least -= 1
        if least > trials:
            return 0.0
        return float(stats.binom.sf(least - 1, trials, truth))

    def plan(self, truth: float, alpha: float, power: float) -> int:
        """Trials needed to pass with `power` when the true rate is `truth`.

        Args:
            truth: True rate, above `bound`.
            alpha: One-sided error rate.
            power: Chance of passing.

        Returns:
            The smallest number of trials, from `minimum` up, whose exact
            chance of passing is at least `power`.
        """
        return search(self, truth, alpha, power)


RULES: dict[str, type[Ceiling] | type[Floor]] = {
    "ceiling": Ceiling,
    "floor": Floor,
}


@dataclasses.dataclass(frozen=True)
class Variable:
    """One threshold and the reason for its value.

    Attributes:
        name: Its key in `acceptance.toml`.
        value: The threshold.
        unit: What the value counts.
        kind: `statistical`, `service`, `response` or `policy`.
        claim: The scorecard claim or claims it decides.
        derivation: Why the value is what it is.
        rule: For a statistical variable, how a row is decided.
    """

    name: str
    value: float
    unit: str
    kind: str
    claim: str
    derivation: str
    rule: Ceiling | Floor | None = None


def parse(name: str, table: dict[str, object]) -> Variable:
    """Builds one variable from its table, refusing anything malformed.

    Args:
        name: The table's key.
        table: Its fields.

    Returns:
        The variable.

    Raises:
        ValueError: For an unknown or missing field, an unknown kind, a
            value that is not a finite number, an empty text field, a
            statistical variable without a rule or with a value outside
            (0, 1), or a rule on another kind.
    """
    unknown = set(table) - FIELDS
    missing = FIELDS - {"rule"} - set(table)
    if unknown or missing:
        raise ValueError(
            f"{name}: unknown fields {sorted(unknown)}, "
            f"missing fields {sorted(missing)}"
        )
    value = table["value"]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name}: value must be a number")
    if not math.isfinite(value):
        raise ValueError(f"{name}: value must be finite")
    for key in ("unit", "kind", "claim", "derivation"):
        text = table[key]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{name}: {key} must be non-empty text")
    kind = str(table["kind"])
    if kind not in KINDS:
        raise ValueError(f"{name}: kind must be one of {sorted(KINDS)}")
    return Variable(
        name,
        float(value),
        str(table["unit"]),
        kind,
        str(table["claim"]),
        str(table["derivation"]),
        pick(name, kind, table.get("rule"), float(value)),
    )


def pick(
    name: str, kind: str, rule: object, value: float
) -> "Ceiling | Floor | None":
    """Builds the rule of a statistical variable.

    Args:
        name: The variable, for messages.
        kind: Its kind.
        rule: Its `rule` field, or None.
        value: Its value, the rule's bound.

    Returns:
        The rule, or None for a kind that is not statistical.

    Raises:
        ValueError: If a statistical kind has no rule or an unknown one, a
            rule is given for another kind, or the bound is outside (0, 1).
    """
    if (kind == "statistical") != (rule is not None):
        raise ValueError(f"{name}: a rule belongs to statistical kinds only")
    if rule is None:
        return None
    if rule not in RULES:
        raise ValueError(f"{name}: rule must be one of {sorted(RULES)}")
    if not 0 < value < 1:
        raise ValueError(f"{name}: a statistical value lies in (0, 1)")
    return RULES[str(rule)](value)


def load(path: pathlib.Path = PATH) -> dict[str, Variable]:
    """Reads and validates every variable.

    Args:
        path: The thresholds file.

    Returns:
        Variable name to variable, in file order.

    Raises:
        ValueError: If a table is malformed or `alpha` or `power` is
            missing or outside (0, 1).
    """
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    found = {}
    for name, table in document.items():
        if not isinstance(table, dict):
            raise ValueError(f"{name}: every entry is a table")
        found[name] = parse(name, table)
    for name in ("alpha", "power"):
        if name not in found or not 0 < found[name].value < 1:
            raise ValueError(f"{name} must be present and lie in (0, 1)")
    return found


def ceiling(name: str, variables: dict[str, Variable] | None = None) -> Ceiling:
    """The `Ceiling` rule of one statistical variable.

    Args:
        name: The variable's key.
        variables: As returned by `load`; read from the file when None.

    Returns:
        The rule.

    Raises:
        ValueError: If the variable has no ceiling rule.
    """
    found = (load() if variables is None else variables)[name].rule
    if not isinstance(found, Ceiling):
        raise ValueError(f"{name} must have the ceiling rule")
    return found


def floor(name: str, variables: dict[str, Variable] | None = None) -> Floor:
    """The `Floor` rule of one statistical variable.

    Args:
        name: The variable's key.
        variables: As returned by `load`; read from the file when None.

    Returns:
        The rule.

    Raises:
        ValueError: If the variable has no floor rule.
    """
    found = (load() if variables is None else variables)[name].rule
    if not isinstance(found, Floor):
        raise ValueError(f"{name} must have the floor rule")
    return found


def margin(level: float, trials: int, alpha: float, power: float) -> float:
    """Smallest size margin a correctly sized test passes with `power`.

    A size gate runs a test of nominal level `L` on `n` null replicates and
    passes when the exact upper bound of its rejection rate is at most
    `L (1 + margin)`. A test whose true size is `L` rejects
    `Binomial(n, L)` times; it passes with chance at least `power` exactly
    when the ceiling admits the `power` quantile `k` of that count, so the
    smallest margin is `BetaInv(1 - alpha; k + 1, n - k) / L - 1`. A
    ceiling of `L` itself (margin 0) fails such a test most of the time.

    Args:
        level: Nominal level of the test, `L`.
        trials: Null replicates the gate runs, `n`.
        alpha: One-sided error rate of the bound.
        power: Chance of passing wanted for a correctly sized test.

    Returns:
        The margin, a share of `level`.
    """
    count = int(stats.binom.ppf(power, trials, level))
    while stats.binom.cdf(count, trials, level) < power:
        count += 1
    return Ceiling(level).limit(count, trials, alpha) / level - 1


def sized(
    level: float, variables: dict[str, Variable] | None = None
) -> Ceiling:
    """The ceiling of a size gate: the level times one plus `size_margin`.

    Args:
        level: Nominal level of the test the gate measures.
        variables: As returned by `load`; read from the file when None.

    Returns:
        `Ceiling(level * (1 + size_margin))`.
    """
    found = load() if variables is None else variables
    return Ceiling(level * (1 + found["size_margin"].value))


def describe(variables: dict[str, Variable]) -> list[dict[str, object]]:
    """Lists every variable, with the minimum sample of statistical ones.

    Args:
        variables: As returned by `load`.

    Returns:
        One JSON-ready row per variable.
    """
    alpha = variables["alpha"].value
    return [
        {
            "name": v.name,
            "value": v.value,
            "unit": v.unit,
            "kind": v.kind,
            "claim": v.claim,
            "minimum": None if v.rule is None else v.rule.minimum(alpha),
        }
        for v in variables.values()
    ]


def main() -> int:
    """Validates the thresholds file and prints its table.

    Returns:
        0; a malformed file raises.
    """
    print(json.dumps(describe(load()), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
