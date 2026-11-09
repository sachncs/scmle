"""Missingness detectors against brute force, scipy and closed forms."""

import numpy
import pandas
import pytest
from scipy import optimize
from scipy import stats
from sklearn import metrics

from scmle.data.detectors import missingness


def gaussian_with_gaps(seed=0, n=80, p=3, rate=0.25, mar=False):
    rng = numpy.random.default_rng(seed)
    cov = numpy.array([[1.0, 0.6, 0.3], [0.6, 1.5, 0.4], [0.3, 0.4, 2.0]])[
        :p, :p
    ]
    data = rng.multivariate_normal(numpy.arange(p, dtype=float), cov, size=n)
    out = data.copy()
    if mar:
        hide = rng.random(n) < 1.0 / (1.0 + numpy.exp(-3.0 * data[:, 0]))
        out[hide, 1] = numpy.nan
    else:
        tail = out[:, 1:]
        tail[rng.random((n, p - 1)) < rate] = numpy.nan
    return out


def observed_nll(params, values):
    p = values.shape[1]
    mean = params[:p]
    chol = numpy.zeros((p, p))
    chol[numpy.tril_indices(p)] = params[p:]
    cov = chol @ chol.T + 1e-12 * numpy.eye(p)
    total = 0.0
    for row in values:
        seen = ~numpy.isnan(row)
        if seen.any():
            total -= stats.multivariate_normal(
                mean[seen], cov[numpy.ix_(seen, seen)]
            ).logpdf(row[seen])
    return total


def brute_force_mle(values):
    p = values.shape[1]
    chol0 = numpy.diag(numpy.nanstd(values, axis=0))
    start = numpy.concatenate(
        [numpy.nanmean(values, axis=0), chol0[numpy.tril_indices(p)]]
    )
    best = optimize.minimize(
        observed_nll,
        start,
        args=(values,),
        method="BFGS",
        options={"gtol": 1e-9, "maxiter": 2000},
    )
    mean = best.x[:p]
    chol = numpy.zeros((p, p))
    chol[numpy.tril_indices(p)] = best.x[p:]
    return mean, chol @ chol.T


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_em_matches_brute_force_maximum_likelihood(seed):
    values = gaussian_with_gaps(seed)
    fit = missingness.em_normal(values)
    mean, cov = brute_force_mle(values)
    assert fit.converged
    numpy.testing.assert_allclose(fit.mean, mean, atol=2e-3)
    numpy.testing.assert_allclose(fit.covariance, cov, atol=2e-3)


def test_em_on_complete_data_is_sample_moments():
    values = numpy.random.default_rng(3).normal(size=(40, 3))
    fit = missingness.em_normal(values)
    numpy.testing.assert_allclose(fit.mean, values.mean(axis=0), atol=1e-9)
    numpy.testing.assert_allclose(
        fit.covariance, numpy.cov(values.T, ddof=0), atol=1e-6
    )


def test_em_rejects_bad_input():
    with pytest.raises(ValueError, match="two-dimensional"):
        missingness.em_normal(numpy.zeros(5))
    bad = numpy.array([[1.0, numpy.nan], [2.0, numpy.nan], [3.0, numpy.nan]])
    with pytest.raises(ValueError, match="at least one observed"):
        missingness.em_normal(bad)


def test_little_statistic_matches_naive_formula_at_brute_force_mle():
    values = gaussian_with_gaps(5)
    frame = pandas.DataFrame(values, columns=list("abc"))
    result = missingness.little_mcar_test(frame, transform="none")
    mean, cov = brute_force_mle(values)
    d2, pj = 0.0, 0
    for group in dict(
        list(frame.groupby(frame.isna().apply(tuple, axis=1)))
    ).values():
        seen = ~group.isna().iloc[0].to_numpy()
        gap = group.to_numpy()[:, seen].mean(axis=0) - mean[seen]
        d2 += (
            len(group)
            * gap
            @ numpy.linalg.inv(cov[numpy.ix_(seen, seen)])
            @ gap
        )
        pj += int(seen.sum())
    assert result.statistic == pytest.approx(d2, abs=5e-3)
    assert result.df == pj - 3
    assert result.p_value == pytest.approx(stats.chi2.sf(d2, pj - 3), abs=5e-3)
    assert result.rows == len(frame) and result.columns == ("a", "b", "c")


def test_little_degrees_of_freedom_by_hand():
    # patterns: all observed (k=3) x2, c missing (k=2), b,c missing (k=1)
    data = numpy.array(
        [
            [1, 2, 3],
            [2, 1, 5],
            [3, 5, numpy.nan],
            [4, 3, numpy.nan],
            [5, numpy.nan, numpy.nan],
            [6, numpy.nan, numpy.nan],
            [7, 8, 9],
            [2, 9, 1],
        ],
        dtype=float,
    )
    result = missingness.little_mcar_test(
        pandas.DataFrame(data, columns=list("abc")), transform="none"
    )
    assert result.patterns == 3
    assert result.df == 3 + 2 + 1 - 3


def test_little_returns_none_without_gaps_or_columns():
    complete = pandas.DataFrame(
        numpy.random.default_rng(0).normal(size=(30, 3))
    )
    assert missingness.little_mcar_test(complete) is None
    one = pandas.DataFrame({"a": [1.0, numpy.nan, 3.0, 4.0]})
    assert missingness.little_mcar_test(one) is None


def test_little_validates_arguments():
    frame = pandas.DataFrame({"a": [1.0, 2.0, numpy.nan], "b": ["x", "y", "z"]})
    with pytest.raises(ValueError, match="transform"):
        missingness.little_mcar_test(frame, ["a"], transform="log")
    with pytest.raises(ValueError, match="not numeric"):
        missingness.little_mcar_test(frame, ["a", "b"])


def test_little_p_values_are_uniform_under_mcar_normal_data():
    cov = numpy.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
    ps = []
    for rep in range(300):
        rng = numpy.random.default_rng(rep)
        x = rng.multivariate_normal(numpy.zeros(3), cov, size=120)
        x[rng.random(120) < 0.2, 1] = numpy.nan
        x[rng.random(120) < 0.2, 2] = numpy.nan
        ps.append(
            missingness.little_mcar_test(
                pandas.DataFrame(x), transform="none"
            ).p_value
        )
    assert stats.kstest(ps, "uniform").pvalue > 0.001
    assert numpy.mean(numpy.array(ps) <= 0.05) == pytest.approx(0.05, abs=0.04)


def test_little_rejects_when_missingness_depends_on_observed_value():
    frame = pandas.DataFrame(gaussian_with_gaps(1, n=400, mar=True))
    assert missingness.little_mcar_test(frame, transform="none").p_value < 1e-4


def test_normal_scores_are_monotone_symmetric_and_keep_gaps():
    column = numpy.array([5.0, 1.0, numpy.nan, 3.0, 9.0])
    out = missingness.normal_scores(column[:, None])[:, 0]
    assert numpy.isnan(out[2])
    seen = out[~numpy.isnan(out)]
    assert list(numpy.argsort(seen)) == list(
        numpy.argsort(column[~numpy.isnan(column)])
    )
    assert seen.sum() == pytest.approx(0.0, abs=1e-12)
    assert seen.max() == pytest.approx(stats.norm.ppf((4 - 0.5) / 4))


def test_auc_from_ranks_matches_sklearn():
    rng = numpy.random.default_rng(0)
    flag = rng.random(100) < 0.3
    score = rng.normal(size=100) + flag
    assert missingness.auc_from_ranks(score, flag) == pytest.approx(
        metrics.roc_auc_score(flag, score)
    )
    tied = numpy.zeros(100)
    assert missingness.auc_from_ranks(tied, flag) == pytest.approx(0.5)


def indicator_frame(dependent, n=200, seed=0):
    rng = numpy.random.default_rng(seed)
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    target = y.copy()
    p = 1 / (1 + numpy.exp(-2.5 * x)) if dependent else numpy.full(n, 0.3)
    target[rng.random(n) < p] = numpy.nan
    return pandas.DataFrame({"x": x, "y": y, "t": target})


def test_indicator_model_detects_dependence_and_respects_null():
    hit = missingness.indicator_model_test(indicator_frame(True), "t", 0, 99)
    assert hit.p_value <= 0.02 and hit.auc > 0.7
    assert hit.column == "t" and hit.missing > 0
    nulls = [
        missingness.indicator_model_test(
            indicator_frame(False, seed=s), "t", s, 99
        ).p_value
        for s in range(40)
    ]
    assert numpy.mean(numpy.array(nulls) <= 0.05) <= 0.2
    assert min(nulls) >= 1 / 100


def test_indicator_model_is_deterministic():
    frame = indicator_frame(True)
    assert missingness.indicator_model_test(
        frame, "t", 3, 19
    ) == missingness.indicator_model_test(frame, "t", 3, 19)


def test_indicator_model_boundaries():
    base = pandas.DataFrame(
        {"x": numpy.random.default_rng(0).normal(size=100), "t": 1.0}
    )
    few = base.copy()
    few.loc[: missingness.MIN_MISSING - 2, "t"] = numpy.nan
    assert missingness.indicator_model_test(few, "t", 0, 9) is None
    enough = base.copy()
    enough.loc[: missingness.MIN_MISSING - 1, "t"] = numpy.nan
    assert (
        missingness.indicator_model_test(enough, "t", 0, 9).missing
        == missingness.MIN_MISSING
    )
    crowded = base.iloc[
        : missingness.MIN_MISSING + missingness.MIN_ROWS - 1
    ].copy()
    crowded.loc[: missingness.MIN_MISSING - 1, "t"] = numpy.nan
    assert missingness.indicator_model_test(crowded, "t", 0, 9) is None
    ok = base.iloc[: missingness.MIN_MISSING + missingness.MIN_ROWS].copy()
    ok.loc[: missingness.MIN_MISSING - 1, "t"] = numpy.nan
    assert missingness.indicator_model_test(ok, "t", 0, 9) is not None
    alone = enough[["t"]]
    assert missingness.indicator_model_test(alone, "t", 0, 9) is None
    with pytest.raises(ValueError, match="permutations"):
        missingness.indicator_model_test(enough, "t", 0, 0)
    with pytest.raises(ValueError, match="folds"):
        missingness.indicator_model_test(enough, "t", 0, 9, folds=1)


def test_pattern_table_counts_patterns():
    frame = pandas.DataFrame(
        {"a": [1, numpy.nan, 3, numpy.nan], "b": [1, 2, numpy.nan, numpy.nan]}
    )
    table = missingness.pattern_table(frame)
    assert table["count"].sum() == 4 and len(table) == 4
    assert list(table["count"]) == sorted(table["count"], reverse=True)


def test_numeric_columns_skip_constants_text_and_booleans():
    frame = pandas.DataFrame(
        {"a": [1.0, 2.0], "c": [1.0, 1.0], "s": ["x", "y"], "b": [True, False]}
    )
    assert missingness.numeric_columns(frame) == ["a"]


def test_assess_mechanism_reports_and_rejects():
    report = missingness.assess_mechanism(indicator_frame(True, n=300), seed=1)
    assert report.mcar_rejected and report.tier == "unverified"
    assert report.rows_checked == report.input_rows == 300
    assert report.to_dict()["alpha"] == missingness.ALPHA
    clean = pandas.DataFrame(numpy.random.default_rng(0).normal(size=(60, 3)))
    quiet = missingness.assess_mechanism(clean)
    assert not quiet.mcar_rejected and quiet.little is None
    with pytest.raises(ValueError, match="alpha"):
        missingness.assess_mechanism(clean, alpha=1.0)
    with pytest.raises(ValueError, match="alpha"):
        missingness.assess_mechanism(clean, alpha=0.0)


def test_assess_mechanism_records_caps_instead_of_failing(monkeypatch):
    frame = indicator_frame(False, n=300)
    monkeypatch.setattr(missingness, "MAX_ROWS", 100)
    report = missingness.assess_mechanism(frame, permutations=9)
    assert report.rows_checked == 100 and report.input_rows == 300
    assert any("sampled" in s for s in report.skipped)
    monkeypatch.setattr(missingness, "MAX_COLUMNS", 2)
    report = missingness.assess_mechanism(frame, permutations=9)
    assert any("over budget" in s for s in report.skipped)
    assert report.little is None and report.indicators == ()
    monkeypatch.setattr(missingness, "MAX_COLUMNS", 40)
    monkeypatch.setattr(missingness, "MAX_INDICATOR_COLUMNS", 0)
    report = missingness.assess_mechanism(frame, permutations=9)
    assert any("indicator models" in s for s in report.skipped)


def test_informative_columns_use_the_existing_screen():
    rng = numpy.random.default_rng(0)
    target = pandas.Series(rng.integers(0, 2, 400))
    gap = target == 1
    frame = pandas.DataFrame({"a": numpy.where(gap, numpy.nan, 1.0), "b": 1.0})
    assert missingness.informative_columns(frame, ["a", "b"], target, True) == [
        "a"
    ]


def test_tipping_point_matches_closed_form_and_grid():
    rng = numpy.random.default_rng(0)
    observed = rng.normal(1.0, 1.0, 100)
    k, null = 20, 0.5
    delta = missingness.mean_tipping_delta(observed, k, null)
    full = (observed.sum() + k * (observed.mean() + delta)) / (
        len(observed) + k
    )
    assert full == pytest.approx(null)
    result = missingness.tipping_point(
        observed, k, lambda v: v.mean() > null, -10.0, 10.0, steps=2001
    )
    assert result.baseline is True and result.inside_range
    assert result.flip_delta is not None
    assert result.flip_delta == pytest.approx(delta, abs=0.02)
    assert result.flip_delta <= delta + 1e-9
    narrow = missingness.tipping_point(
        observed, k, lambda v: v.mean() > null, -1.0, 1.0, steps=101
    )
    assert narrow.flip_delta is None and not narrow.inside_range
    assert (narrow.low, narrow.high, narrow.steps) == (-1.0, 1.0, 101)


def test_tipping_point_validates():
    f = lambda v: True  # noqa: E731
    with pytest.raises(ValueError, match="observed"):
        missingness.tipping_point(numpy.array([]), 1, f, -1, 1)
    with pytest.raises(ValueError, match="observed"):
        missingness.tipping_point(numpy.ones(3), 0, f, -1, 1)
    with pytest.raises(ValueError, match="low < high"):
        missingness.tipping_point(numpy.ones(3), 1, f, 1, 1)
    with pytest.raises(ValueError, match="steps"):
        missingness.tipping_point(numpy.ones(3), 1, f, -1, 1, steps=1)


@pytest.mark.parametrize("strategy", missingness.STRATEGIES)
def test_fold_imputer_state_ignores_held_out_rows(strategy):
    rng = numpy.random.default_rng(0)
    values = rng.normal(size=(60, 4))
    values[rng.random(values.shape) < 0.1] = numpy.nan
    folds = missingness.fold_imputers(values, strategy, seed=2)
    for train, held, fitted in folds:
        assert not set(train) & set(held)
        changed = values.copy()
        changed[held] = 1e6
        again = missingness.fit_one(changed, strategy, train)
        probe = values[train][:5]
        numpy.testing.assert_array_equal(
            fitted.transform(probe), again.transform(probe)
        )


def test_imputation_errors_and_selection():
    rng = numpy.random.default_rng(0)
    base = rng.normal(size=(200, 1))
    values = numpy.hstack(
        [base + 0.05 * rng.normal(size=(200, 1)) for col in range(4)]
    )
    values[rng.random(values.shape) < 0.1] = numpy.nan
    errors = missingness.imputation_errors(values, seed=0)
    assert set(errors) == set(missingness.STRATEGIES)
    assert errors["knn"] < errors["mean"]
    assert missingness.select_imputer(values, seed=0) == "knn"
    with pytest.raises(ValueError, match="share"):
        missingness.imputation_errors(values, share=1.0)
    with pytest.raises(ValueError, match="strategy"):
        missingness.make_imputer("magic")
