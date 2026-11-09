"""Conditional-independence tests that falsify but never confirm."""

import math

import numpy
import pandas
import pytest
from scipy import stats

from scmle.contracts.qualification import Decision
from scmle.contracts.qualification import SetId
from scmle.contracts.qualification import Tier
from scmle.data.admissibility import dag
from scmle.data.admissibility import independence


def sem_frame(n, seed, extra_edge=False):
    """Linear Gaussian data from a -> b -> c, a -> d (optionally a -> c)."""
    rng = numpy.random.default_rng(seed)
    a = rng.normal(size=n)
    b = 0.8 * a + rng.normal(size=n)
    c = 0.8 * b + (0.7 * a if extra_edge else 0.0) + rng.normal(size=n)
    dd = 0.6 * a + rng.normal(size=n)
    return pandas.DataFrame({"a": a, "b": b, "c": c, "d": dd})


CHAIN = dag.Dag.build([("a", "b"), ("b", "c"), ("a", "d")], "expert")


def precision_partial(frame, x, y, given):
    cols = [x, y, *given]
    cov = numpy.cov(frame[cols].to_numpy().T)
    prec = numpy.linalg.inv(cov)
    return -prec[0, 1] / math.sqrt(prec[0, 0] * prec[1, 1])


@pytest.mark.parametrize("given", [[], ["a"], ["a", "d"]])
def test_partial_correlation_matches_precision_matrix_and_scipy(given):
    frame = sem_frame(300, 1, extra_edge=True)
    r = independence.partial_correlation(frame, "b", "c", given)
    assert r == pytest.approx(
        precision_partial(frame, "b", "c", given), abs=1e-10
    )
    if given:
        z = frame[given].to_numpy()
        design = numpy.column_stack([numpy.ones(len(frame)), z])
        rb = (
            frame["b"]
            - design @ numpy.linalg.lstsq(design, frame["b"], rcond=None)[0]
        )
        rc = (
            frame["c"]
            - design @ numpy.linalg.lstsq(design, frame["c"], rcond=None)[0]
        )
        assert r == pytest.approx(stats.pearsonr(rb, rc).statistic, abs=1e-10)
    else:
        assert r == pytest.approx(
            stats.pearsonr(frame["b"], frame["c"]).statistic
        )


@pytest.mark.parametrize("given", [[], ["a"], ["a", "d"]])
def test_fisher_z_p_value_follows_the_published_formula(given):
    frame = sem_frame(200, 2, extra_edge=True)
    result = independence.fisher_z(frame, "b", "c", given)
    r = precision_partial(frame, "b", "c", given)
    z = math.sqrt(200 - len(given) - 3) * 0.5 * math.log((1 + r) / (1 - r))
    assert result.statistic == pytest.approx(z, abs=1e-9)
    assert result.p_value == pytest.approx(
        2 * (1 - stats.norm.cdf(abs(z))), abs=1e-12
    )
    assert result.effect == pytest.approx(r, abs=1e-10)
    assert result.n == 200


def test_fisher_z_needs_enough_rows():
    frame = sem_frame(6, 0)
    independence.fisher_z(
        frame, "a", "b", ["c", "d"]
    )  # 6 - 2 - 3 = 1 is enough
    with pytest.raises(ValueError, match="too few"):
        independence.fisher_z(frame.iloc[:5], "a", "b", ["c", "d"])  # 0 is not


def test_perfect_correlation_gives_a_zero_p_value():
    frame = pandas.DataFrame(
        {"x": numpy.arange(20.0), "y": 2 * numpy.arange(20.0) + 1}
    )
    result = independence.fisher_z(frame, "x", "y", [])
    assert result.p_value == 0.0
    assert math.isinf(result.statistic)


def test_a_constant_column_is_an_error():
    frame = pandas.DataFrame({"x": numpy.arange(20.0), "y": numpy.ones(20)})
    with pytest.raises(ValueError, match="no spread"):
        independence.partial_correlation(frame, "x", "y", [])


def discrete_frame(seed=0, n=600):
    rng = numpy.random.default_rng(seed)
    s = rng.integers(0, 3, n)
    x = (rng.random(n) < 0.1 + 0.4 * s).astype(int)
    y = (rng.random(n) < 0.1 + 0.4 * s).astype(int)
    return pandas.DataFrame({"x": x, "y": y, "s": s})


def test_stratified_chi_square_equals_the_sum_of_scipy_per_stratum_statistics():
    frame = discrete_frame()
    result = independence.stratified_chi2(frame, "x", "y", ["s"])
    total = 0.0
    df = 0
    for key, rows in frame.groupby("s"):
        assert key in (0, 1, 2)
        table = pandas.crosstab(rows["x"], rows["y"])
        total += stats.chi2_contingency(table, correction=False)[0]
        df += 1
    assert result.statistic == pytest.approx(total, rel=1e-10)
    assert result.df == df
    assert result.p_value == pytest.approx(stats.chi2.sf(total, df), rel=1e-9)
    assert result.skipped_strata == 0
    # x and y are related only through s, so conditioning removes it
    assert result.p_value > 0.001
    marginal = independence.stratified_chi2(frame, "x", "y", [])
    assert marginal.p_value < 1e-6


def test_degenerate_strata_are_skipped_and_counted():
    frame = pandas.DataFrame(
        {
            "x": [0, 1, 0, 1, 0, 0, 0, 0],
            "y": [0, 1, 1, 0, 1, 1, 1, 1],
            "s": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    result = independence.stratified_chi2(frame, "x", "y", ["s"])
    assert result.skipped_strata == 1
    assert result.df == 1
    nothing = independence.stratified_chi2(frame.iloc[4:], "x", "y", ["s"])
    assert nothing.df == 0 and nothing.p_value == 1.0


def test_dispatch_and_refusals():
    num = sem_frame(100, 0)
    assert (
        independence.run_test(num, "a", "b", []).kind
        is independence.CIKind.FISHER_Z
    )
    disc = discrete_frame().astype("category")
    assert (
        independence.run_test(disc, "x", "y", ["s"]).kind
        is independence.CIKind.STRATIFIED_CHI2
    )
    mixed = disc.assign(w=numpy.arange(len(disc), dtype=float))
    with pytest.raises(ValueError, match="mixed"):
        independence.run_test(mixed, "x", "w", [])
    with pytest.raises(ValueError, match="complete"):
        independence.run_test(
            num.assign(a=numpy.where(num.index == 0, numpy.nan, num["a"])),
            "a",
            "b",
            [],
        )
    wide = pandas.DataFrame(
        {"x": numpy.arange(30).astype(str), "y": numpy.arange(30).astype(str)}
    )
    with pytest.raises(ValueError, match="too many levels"):
        independence.run_test(wide, "x", "y", [])


def test_implied_statements_of_the_chain():
    statements, skipped = independence.implied_statements(CHAIN)
    texts = {s.text() for s in statements}
    assert "c is independent of a given b" in texts
    assert "c is independent of d given b" in texts
    assert "b is independent of d given a" in texts
    assert len(texts) == 4 and len(statements) == 4
    assert skipped == []
    for s in statements:  # every statement is a true d-separation
        assert CHAIN.d_separated([s.x], [s.y], s.given)


def test_latent_parents_skip_the_node():
    graph = dag.Dag.build(
        [("U", "a"), ("a", "b"), ("b", "c")], "expert", latent=["U"]
    )
    skipped = independence.implied_statements(graph)[1]
    assert skipped == ["a"]


def test_a_true_graph_is_rarely_falsified():
    hits = 0
    runs = 200
    for seed in range(runs):
        report = independence.falsify(CHAIN, sem_frame(300, seed))
        hits += bool(report.falsified)
    # family-wise level 0.05; 99% band for 200 runs is about 0.05 + 0.04
    assert hits / runs <= 0.05 + 2.576 * math.sqrt(0.05 * 0.95 / runs)


def test_a_wrong_graph_is_falsified_when_data_contradict_it():
    hits = sum(
        bool(
            independence.falsify(
                CHAIN, sem_frame(500, s, extra_edge=True)
            ).falsified
        )
        for s in range(40)
    )
    assert hits >= 38


def test_report_never_confirms_and_records_every_test():
    report = independence.falsify(CHAIN, sem_frame(300, 3), run_seed=7)
    assert report.dag_hash == CHAIN.digest()
    assert report.level == pytest.approx(0.05 / len(report.outcomes))
    assert len(report.checks) == len(report.outcomes)
    assert all(o.verdict.value != "confirmed" for o in report.outcomes)
    assert {v.value for v in independence.Verdict} == {
        "falsified",
        "not_falsified",
        "skipped",
    }
    summary = independence.describe(report)
    assert summary["status"] == "hypothesis"
    assert "confirm" not in " ".join(k for k in summary if k != "note")
    for check in report.checks:
        assert check.set_id is SetId.EXPLORATION
        assert check.tier is Tier.UNVERIFIED
        assert check.decision in (Decision.PASS, Decision.WARN)
        assert check.seed_path[0] == 7
    assert len({c.check_id for c in report.checks}) == len(report.checks)
    assert len({c.seed_path for c in report.checks}) == len(report.checks)


def test_falsification_threshold_is_inclusive(monkeypatch):
    def fake(frame, x, y, given):
        return independence.CITest(
            independence.CIKind.FISHER_Z, 1.0, level, 0.1, len(frame), 0, 0
        )

    statements = independence.implied_statements(CHAIN)[0]
    level = 0.05 / len(statements)
    monkeypatch.setattr(independence, "run_test", fake)
    report = independence.falsify(CHAIN, sem_frame(50, 0))
    assert len(report.falsified) == len(statements)
    level = level * 1.0001
    report = independence.falsify(CHAIN, sem_frame(50, 0))
    assert len(report.falsified) == 0


def test_untestable_statements_are_skipped_not_passed():
    frame = sem_frame(100, 0)
    frame.loc[0, "b"] = numpy.nan
    report = independence.falsify(CHAIN, frame)
    skipped = [
        o for o in report.outcomes if o.verdict is independence.Verdict.SKIPPED
    ]
    assert skipped and all(o.check is None and o.reason for o in skipped)
    assert independence.describe(report)["skipped"] == len(skipped)


def test_arguments_are_validated():
    frame = sem_frame(50, 0)
    for alpha in (0.0, 1.0, float("nan")):
        with pytest.raises(ValueError):
            independence.falsify(CHAIN, frame, alpha=alpha)
    with pytest.raises(ValueError, match="not in the data"):
        independence.falsify(CHAIN, frame.drop(columns="d"))
