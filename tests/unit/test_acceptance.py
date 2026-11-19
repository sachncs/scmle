"""The scorecard thresholds are well formed and decided by exact bounds."""

import math
import re

import pytest
from scipy import stats

from benchmarks import acceptance

VARIABLES = acceptance.load()
ALPHA = VARIABLES["alpha"].value
STATISTICAL = [n for n, v in VARIABLES.items() if v.rule is not None]


def best(rule, trials):
    """The most favourable count for a rule: no event, or every trial a hit."""
    return 0 if isinstance(rule, acceptance.Ceiling) else trials


def searched(rule, alpha):
    """Smallest n that passes, found by trying every n in turn."""
    trials = 1
    while not rule.passes(best(rule, trials), trials, alpha):
        trials += 1
    return trials


def power(rule, truth, trials, alpha):
    """Exact chance that the rule passes when the true rate is `truth`."""
    return sum(
        stats.binom.pmf(hits, trials, truth)
        for hits in range(trials + 1)
        if rule.passes(hits, trials, alpha)
    )


def test_the_shipped_file_loads_and_every_variable_has_a_reason():
    assert len(VARIABLES) >= 40
    for variable in VARIABLES.values():
        assert variable.derivation.strip()
        assert variable.kind in acceptance.KINDS


def test_the_plan_defaults_are_the_file_defaults():
    expected = {
        "alpha": 0.05,
        "power": 0.80,
        "practitioners": 1,
        "leak_recall_min": 0.90,
        "false_block_max": 0.05,
        "refusal_rate_max": 0.05,
        "refusal_precision_min": 0.90,
        "interval_coverage_min": 0.93,
        "null_skill_max": 0.05,
        "speedup_min": 5,
        "practitioner_noninferiority_margin": 0.02,
        "study_look_every": 4,
        "bootstrap_resamples": 10000,
        "mutation_score_min": 0.85,
        "function_lines_max": 80,
    }
    assert {k: VARIABLES[k].value for k in expected} == expected


@pytest.mark.parametrize("name", STATISTICAL)
def test_the_minimum_sample_is_the_smallest_that_can_pass(name):
    rule = VARIABLES[name].rule
    least = rule.minimum(ALPHA)
    assert least == searched(rule, ALPHA)
    assert not rule.passes(best(rule, least - 1), least - 1, ALPHA)


def test_every_quoted_minimum_matches_the_computed_one():
    for variable in VARIABLES.values():
        quoted = re.search(r"(\d+) [a-z ]+to pass at all", variable.derivation)
        if variable.rule is not None and quoted:
            assert int(quoted.group(1)) == variable.rule.minimum(ALPHA), (
                variable.name
            )


def test_the_ceiling_bound_is_the_beta_quantile():
    rule = acceptance.Ceiling(0.05)
    hits, trials = 3, 200
    upper = stats.beta.ppf(1 - ALPHA, hits + 1, trials - hits)
    assert rule.passes(hits, trials, ALPHA) == (upper <= 0.05)
    assert rule.passes(0, 59, ALPHA)
    assert not rule.passes(0, 58, ALPHA)
    assert not rule.passes(0, 0, ALPHA)


def test_the_limit_is_the_one_sided_bound_each_rule_compares():
    ceiling, floor = acceptance.Ceiling(0.01), acceptance.Floor(0.80)
    upper = stats.beta.ppf(1 - ALPHA, 16, 985)
    assert ceiling.limit(15, 1000, ALPHA) == pytest.approx(upper)
    assert not ceiling.passes(15, 1000, ALPHA)
    lower = stats.beta.ppf(ALPHA, 382, 119)
    assert floor.limit(382, 500, ALPHA) == pytest.approx(lower)
    assert floor.passes(382, 500, ALPHA) == (lower >= 0.80)
    assert ceiling.limit(0, 0, ALPHA) == 1.0
    assert floor.limit(0, 0, ALPHA) == 0.0


def test_a_named_rule_is_read_with_its_kind_checked():
    assert acceptance.ceiling("false_block_max") == acceptance.Ceiling(0.05)
    assert acceptance.floor("leak_recall_min", VARIABLES) == acceptance.Floor(
        0.90
    )
    with pytest.raises(ValueError, match="ceiling"):
        acceptance.ceiling("leak_recall_min", VARIABLES)
    with pytest.raises(ValueError, match="floor"):
        acceptance.floor("alpha", VARIABLES)


def test_the_floor_bound_is_the_beta_quantile():
    rule = acceptance.Floor(0.90)
    hits, trials = 95, 100
    lower = stats.beta.ppf(ALPHA, hits, trials - hits + 1)
    assert rule.passes(hits, trials, ALPHA) == (lower >= 0.90)
    assert rule.passes(29, 29, ALPHA)
    assert not rule.passes(28, 28, ALPHA)
    assert not rule.passes(0, 0, ALPHA)


def test_the_planned_size_follows_the_power_formula():
    z = stats.norm.ppf
    null, truth = 0.05, 0.10
    by_hand = (
        z(0.95) * math.sqrt(null * (1 - null))
        + z(0.80) * math.sqrt(truth * (1 - truth))
    ) ** 2 / (null - truth) ** 2
    assert acceptance.size(null, truth, 0.05, 0.80) == math.ceil(by_hand)
    assert acceptance.size(null, truth, 0.05, 0.80) == 150
    assert VARIABLES["null_tasks_min"].value >= 150


def test_a_plan_is_never_below_the_minimum():
    rule = acceptance.Ceiling(0.05)
    assert rule.plan(0.0001, ALPHA, 0.8) == rule.minimum(ALPHA)
    with pytest.raises(ValueError, match="differ"):
        acceptance.size(0.05, 0.05, ALPHA, 0.8)


@pytest.mark.parametrize(
    ("rule", "truth"),
    [(acceptance.Ceiling(0.05), 0.01), (acceptance.Floor(0.90), 0.97)],
)
def test_a_planned_sample_reaches_its_exact_power(rule, truth):
    trials = rule.plan(truth, ALPHA, 0.80)
    assert power(rule, truth, trials, ALPHA) >= 0.80
    assert rule.chance(truth, trials, ALPHA) == pytest.approx(
        power(rule, truth, trials, ALPHA)
    )
    assert power(rule, truth, trials - 1, ALPHA) < 0.80 or trials == (
        rule.minimum(ALPHA)
    )


def test_the_normal_approximation_alone_underpowers_near_zero():
    rule = acceptance.Ceiling(0.05)
    approximate = acceptance.size(0.05, 0.01, ALPHA, 0.80)
    assert power(rule, 0.01, approximate, ALPHA) < 0.80
    assert rule.plan(0.01, ALPHA, 0.80) > approximate


def test_a_true_rate_on_the_failing_side_cannot_be_planned():
    with pytest.raises(ValueError, match="passing side"):
        acceptance.Ceiling(0.05).plan(0.06, ALPHA, 0.80)
    with pytest.raises(ValueError, match="passing side"):
        acceptance.Floor(0.90).plan(0.90, ALPHA, 0.80)


@pytest.mark.parametrize(
    ("rule", "truth"),
    [(acceptance.Ceiling(0.05), 0.05), (acceptance.Floor(0.90), 0.90)],
)
def test_a_rule_passes_at_most_alpha_of_the_time_at_its_bound(rule, truth):
    for trials in (59, 120, 400):
        assert power(rule, truth, trials, ALPHA) <= ALPHA + 1e-12


def table(**fields):
    """A well-formed policy variable, with `fields` replacing its own."""
    base = {
        "value": 0.5,
        "unit": "u",
        "kind": "policy",
        "claim": "c",
        "derivation": "d",
    }
    return {**base, **fields}


BAD = [
    ({"value": 1}, "missing fields"),
    (table(kind="k"), "kind must be"),
    (table(value="x"), "must be a number"),
    (table(value=True), "must be a number"),
    (table(value=math.inf), "finite"),
    (table(derivation=" "), "non-empty"),
    (table(kind="statistical"), "statistical kinds only"),
    (table(rule="floor"), "statistical kinds only"),
    (table(kind="statistical", rule="band"), "rule must be"),
    (table(kind="statistical", rule="floor", value=1.5), "lies in"),
    (table(extra=1), "unknown fields"),
]


@pytest.mark.parametrize(("fields", "message"), BAD)
def test_a_malformed_variable_is_refused(fields, message):
    with pytest.raises(ValueError, match=message):
        acceptance.parse("x", fields)


def test_a_file_without_alpha_is_refused(tmp_path):
    path = tmp_path / "acceptance.toml"
    path.write_text(
        '[power]\nvalue = 0.8\nunit = "p"\nkind = "policy"\n'
        'claim = "all"\nderivation = "d"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="alpha"):
        acceptance.load(path)


def test_a_ceiling_at_the_level_fails_most_correctly_sized_tests():
    """The old size rule: a test whose true size is its level rarely passes."""
    assert power(acceptance.Ceiling(0.01), 0.01, 2000, ALPHA) < 0.1
    assert power(acceptance.Ceiling(0.05), 0.05, 2000, ALPHA) < 0.1


@pytest.mark.parametrize(
    ("level", "trials"), [(0.01, 2000), (0.05, 500), (0.05, 1000), (0.05, 2000)]
)
def test_the_margin_is_the_smallest_that_passes_a_sized_test_with_power(
    level, trials
):
    found = acceptance.margin(level, trials, ALPHA, 0.8)
    assert found > 0
    exact = acceptance.Ceiling(level * (1 + found))
    assert power(exact, level, trials, ALPHA) >= 0.8
    short = acceptance.Ceiling(level * (1 + found) - 1e-9)
    assert power(short, level, trials, ALPHA) < 0.8


def test_a_size_ceiling_is_the_level_times_one_plus_the_margin():
    margin = VARIABLES["size_margin"].value
    assert acceptance.sized(0.01, VARIABLES).bound == pytest.approx(
        0.01 * (1 + margin)
    )
    assert acceptance.sized(0.05).bound == pytest.approx(0.05 * (1 + margin))


def test_a_size_ceiling_still_fails_a_test_of_twice_its_level():
    rule = acceptance.sized(0.01, VARIABLES)
    assert rule.passes(24, 2000, ALPHA)
    assert not rule.passes(25, 2000, ALPHA)
    assert power(rule, 0.02, 2000, ALPHA) < 0.01


def test_the_table_lists_the_minimum_of_statistical_rows():
    rows = {r["name"]: r for r in acceptance.describe(VARIABLES)}
    assert rows["false_block_max"]["minimum"] == 59
    assert rows["leak_recall_min"]["minimum"] == 29
    assert rows["speedup_min"]["minimum"] is None
    assert acceptance.main() == 0
