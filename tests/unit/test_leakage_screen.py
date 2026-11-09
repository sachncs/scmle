"""The leakage screen: references, error control, plants and gating."""

import dataclasses
from fractions import Fraction
import math

import numpy
import pandas
import pytest
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score

from scmle.data.admissibility import leakage
from scmle.data.admissibility import screen
from scmle.data.detectors import certificates
from scmle.data.ledger import multiplicity
import scmle.data.ledger.budget

CLS = "classification"
FAST = leakage.Config(
    skill_ceiling=0.95,
    association_ceiling=0.9,
    permutations=199,
    bootstrap=100,
)


def world(seed, n=300, columns=4):
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame(
        {f"x{i}": rng.normal(size=n) for i in range(columns)}
    )
    frame["y"] = rng.integers(0, 2, n)
    return frame


def run(frame, **options):
    return screen.screen(frame, "y", CLS, FAST, **options)


def suspects(report):
    return {(f.feature, f.mechanism) for f in report.suspects}


# ---- building blocks against references ------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_within_fold_auc_matches_sklearn_per_fold(seed):
    rng = numpy.random.default_rng(seed)
    n = 120
    truth = rng.integers(0, 2, n)
    score = numpy.round(rng.normal(size=n) + truth, 1)
    tests = [numpy.arange(0, 60), numpy.arange(60, 120)]
    wins, pairs = screen.fold_concordance(
        score[None, :], (truth == 1)[None, :], tests
    )
    expected = [roc_auc_score(truth[t], score[t]) for t in tests]
    sizes = [(truth[t] == 1).sum() * (truth[t] == 0).sum() for t in tests]
    assert wins[0] / pairs[0] == pytest.approx(
        numpy.average(expected, weights=sizes), abs=1e-12
    )


def test_a_pooled_auc_would_differ_from_the_within_fold_one():
    rng = numpy.random.default_rng(0)
    truth = rng.integers(0, 2, 200)
    score = rng.normal(size=200) + numpy.repeat([0.0, 3.0], 100)
    tests = [numpy.arange(100), numpy.arange(100, 200)]
    within = screen.auc_rows(score[None, :], (truth == 1)[None, :], tests)[0]
    assert within != pytest.approx(roc_auc_score(truth, score), abs=1e-3)


def test_auc_is_nan_when_no_fold_has_both_classes():
    out = screen.auc_rows(
        numpy.arange(4.0)[None, :],
        numpy.ones((1, 4), dtype=bool),
        [numpy.arange(2), numpy.arange(2, 4)],
    )
    assert math.isnan(out[0])


def test_the_ridge_fit_matches_numpy_least_squares_on_a_fold():
    rng = numpy.random.default_rng(1)
    n = 80
    x = rng.normal(size=n)
    y = 2 * x + rng.normal(scale=0.1, size=n)
    phi = screen.basis(pandas.Series(x))
    splits = [
        (numpy.arange(40), numpy.arange(40, 80)),
        (numpy.arange(40, 80), numpy.arange(40)),
    ]
    pred = screen.oof_fit(y[None, :], phi, splits)[0]
    train, test = splits[0]
    centred = y[train] - y[train].mean()
    gram = phi[train].T @ phi[train] + screen.RIDGE * numpy.eye(phi.shape[1])
    beta = numpy.linalg.solve(gram, phi[train].T @ centred)
    expected = y[train].mean() + phi[test] @ beta
    assert pred[test] == pytest.approx(expected, abs=1e-9)


def test_a_linear_relation_is_inside_the_numeric_basis():
    x = pandas.Series(numpy.linspace(-3, 3, 200))
    phi = screen.basis(x)
    beta = numpy.linalg.lstsq(phi, x.to_numpy(), rcond=None)[0]
    assert phi @ beta == pytest.approx(x.to_numpy(), abs=1e-9)


def test_basis_keeps_missing_and_categories_apart():
    numeric = pandas.Series(numpy.r_[numpy.linspace(0, 1, 30), [numpy.nan] * 5])
    phi = screen.basis(numeric)
    assert phi[-1, -1] == 1.0 and phi[:30, -1].sum() == 0.0
    assert phi.sum(axis=1) == pytest.approx(numpy.ones(35))
    codes = screen.basis(pandas.Series([*list("abcab"), None]))
    assert codes.shape == (6, 4)
    assert codes.sum(axis=1).tolist() == [1.0] * 6


def test_constant_column_has_a_single_basis_column():
    assert screen.basis(pandas.Series([1.0] * 50)).shape[1] == 1
    assert screen.basis(pandas.Series(["a"] * 50)).shape[1] == 1


def test_permutations_stay_inside_their_strata():
    strata = numpy.repeat([0, 1, 2], [5, 7, 3])
    rng = numpy.random.default_rng(0)
    index = screen.batch_permutations(strata, 50, rng)
    assert index.shape == (50, 15)
    for row in index:
        assert sorted(row) == list(range(15))
        assert (strata[row] == strata).all()
    assert len({tuple(r) for r in index}) > 40


def test_permutations_are_global_with_one_stratum():
    index = screen.batch_permutations(
        numpy.zeros(30, dtype=int), 20, numpy.random.default_rng(1)
    )
    assert (index != numpy.arange(30)).any(axis=1).all()


def test_strata_follow_groups_blocks_or_nothing():
    assert screen.strata_of(6, None, None).tolist() == [0] * 6
    assert screen.strata_of(6, None, 2).tolist() == [0, 0, 1, 1, 2, 2]
    groups = numpy.array(["b", "a", "b", "a"])
    assert screen.strata_of(4, groups, None).tolist() == [0, 1, 0, 1]


def test_required_permutations_reach_the_by_threshold():
    alpha = Fraction(1, 20)
    for family in (1, 2, 10, 30):
        need = screen.required_permutations(family, alpha)
        h = multiplicity.harmonic_exact(family)
        assert Fraction(1, need + 1) <= alpha / (family * h)
        assert Fraction(1, need) > alpha / (family * h)
    assert screen.required_permutations(0, alpha) == 0


def test_purity_bound_is_clopper_pearson():
    classes = numpy.array([0] * 40 + [1] * 60)
    recorded = numpy.zeros(100, dtype=bool)
    recorded[:30] = True
    bound = screen.purity_bound(recorded, classes, FAST, None)
    assert bound == pytest.approx(0.05 ** (1 / 30), abs=1e-12)
    mixed = numpy.zeros(100, dtype=bool)
    mixed[35:65] = True
    expected = stats.beta.ppf(0.05, 25, 6)
    assert screen.purity_bound(mixed, classes, FAST, None) == pytest.approx(
        expected
    )


def test_purity_bound_counts_groups_not_rows_when_grouped():
    classes = numpy.zeros(40, dtype=int)
    recorded = numpy.ones(40, dtype=bool)
    groups = numpy.repeat(numpy.arange(4), 10)
    one_per_group = screen.purity_bound(recorded, classes, FAST, groups)
    assert one_per_group == pytest.approx(0.05 ** (1 / 4))
    assert one_per_group < screen.purity_bound(recorded, classes, FAST, None)


def test_chi_square_rows_match_scipy():
    rng = numpy.random.default_rng(3)
    missing = rng.random(90) < 0.3
    labels = rng.integers(0, 3, 90)
    totals = numpy.bincount(labels).astype(float)
    table = pandas.crosstab(missing, labels).to_numpy()
    expected = stats.chi2_contingency(table, correction=False)[0]
    got = screen.chi_square_rows(missing, labels[None, :], totals)[0]
    assert got == pytest.approx(expected, rel=1e-12)


# ---- the screen ------------------------------------------------------------


def test_a_clean_table_has_no_suspect_and_a_planted_copy_has_one():
    frame = world(0)
    assert run(frame).suspects == ()
    leaky = frame.assign(leak=frame.y * 3.0 + 1)
    report = run(leaky)
    assert suspects(report) == {("leak", "skill")}
    (found,) = report.suspects
    assert found.statistic == pytest.approx(1.0)
    assert found.lower_bound > FAST.skill_ceiling
    assert found.adjusted_p <= 0.05 and found.significant


def test_a_strong_legitimate_predictor_is_significant_but_not_a_suspect():
    frame = world(1, n=600)
    frame["strong"] = frame.y + numpy.random.default_rng(1).normal(
        scale=1.0, size=600
    )
    report = run(frame)
    (finding,) = [
        f for f in report.findings if f.feature == "strong" and f.significant
    ]
    assert finding.statistic < FAST.skill_ceiling
    assert finding.significant and not finding.suspect
    assert report.suspects == ()


def test_the_family_counts_every_column_and_every_mechanism():
    frame = world(2, columns=3)
    frame.loc[::4, "x0"] = numpy.nan
    report = run(frame)
    mechanisms = [f.mechanism for f in report.findings]
    assert report.family == len(report.findings) == 5
    assert mechanisms.count(screen.SKILL) == 3
    assert mechanisms.count(screen.MISSING) == 1
    assert mechanisms.count(screen.PURITY) == 1
    assert report.permutations >= screen.required_permutations(
        report.family, Fraction(1, 20)
    )


def test_by_decisions_equal_the_exact_adjustment_of_the_reported_p_values():
    frame = world(3).assign(leak=lambda f: f.y + 0.0)
    report = run(frame)
    ps = [f.p_value if f.complete else 1.0 for f in report.findings]
    adjustment = multiplicity.adjust(
        ps, scmle.data.ledger.budget.Correction.BY, Fraction(1, 20)
    )
    assert [f.significant for f in report.findings] == list(adjustment.rejected)


def test_early_stopping_leaves_the_decisions_unchanged(monkeypatch):
    frame = world(4).assign(leak=lambda f: f.y + 0.0)
    frame["weak"] = frame.y * 0.3 + numpy.random.default_rng(4).normal(size=300)
    early = run(frame)
    original = screen.draw_permutations

    def never_stop(pending, labels, binned, strata, total, alpha, seed):
        # A huge level puts the stopping ceiling out of reach.
        return original(
            pending, labels, binned, strata, total, Fraction(10**9), seed
        )

    monkeypatch.setattr(screen, "draw_permutations", never_stop)
    full = run(frame)
    assert any(not f.complete for f in early.findings)
    assert all(f.complete for f in full.findings)
    assert [f.significant for f in early.findings] == [
        f.significant for f in full.findings
    ]
    assert suspects(early) == suspects(full)


def test_the_screen_is_deterministic_in_the_seed():
    frame = world(5).assign(leak=lambda f: f.y * 1.0)
    one, two = run(frame), run(frame)
    assert one == two
    other = screen.screen(
        frame, "y", CLS, dataclasses.replace(FAST, run_seed=9)
    )
    assert suspects(other) == suspects(one)


def test_a_noisy_copy_and_a_threshold_rule_are_suspects():
    rng = numpy.random.default_rng(6)
    frame = world(6, n=500)
    flip = rng.random(500) < 0.02
    frame["noisy"] = numpy.where(flip, 1 - frame.y, frame.y)
    frame["threshold"] = frame.y + rng.normal(scale=0.25, size=500)
    assert suspects(run(frame)) == {
        ("noisy", screen.SKILL),
        ("threshold", screen.SKILL),
    }


def test_a_value_recorded_only_for_one_class_is_a_suspect():
    rng = numpy.random.default_rng(7)
    frame = world(7, n=400)
    frame["body"] = numpy.where(
        (frame.y == 0) & (rng.random(400) < 0.3),
        rng.normal(size=400),
        numpy.nan,
    )
    report = run(frame)
    assert ("body", screen.PURITY) in suspects(report)
    found = next(f for f in report.suspects if f.mechanism == screen.PURITY)
    assert found.statistic == 1.0
    assert found.lower_bound == pytest.approx(
        0.05 ** (1 / int(frame.body.notna().sum()))
    )


def test_missing_only_for_one_class_is_a_missingness_suspect():
    rng = numpy.random.default_rng(8)
    frame = world(8, n=400)
    frame["boat"] = numpy.where(frame.y == 1, rng.normal(size=400), numpy.nan)
    report = run(frame)
    assert ("boat", screen.MISSING) in suspects(report)
    found = next(f for f in report.suspects if f.mechanism == screen.MISSING)
    assert found.statistic == pytest.approx(1.0)


def test_random_gaps_are_not_a_suspect():
    frame = world(9, n=400)
    frame.loc[numpy.random.default_rng(9).random(400) < 0.3, "x0"] = numpy.nan
    assert run(frame).suspects == ()


def test_classification_with_more_than_two_classes():
    rng = numpy.random.default_rng(10)
    frame = world(10, n=450)
    frame["y"] = rng.integers(0, 3, 450)
    frame["leak"] = frame.y * 2.0
    report = run(frame)
    assert suspects(report) == {("leak", screen.SKILL)}


def test_regression_copy_and_a_strong_but_noisy_predictor():
    rng = numpy.random.default_rng(111)
    frame = world(11, n=300)
    frame["y"] = rng.normal(size=300)
    frame["leak"] = frame.y * 2.0 + 1.0
    frame["x1"] = frame.y + rng.normal(scale=1.0, size=300)
    report = screen.screen(frame, "y", "regression", FAST)
    assert suspects(report) == {("leak", screen.SKILL)}
    strong = next(f for f in report.findings if f.feature == "x1")
    assert strong.significant and not strong.suspect


def test_groups_keep_entities_whole_and_still_find_a_leak():
    frame = world(12, n=400)
    groups = pandas.Series(numpy.arange(400) // 8)
    frame["leak"] = frame.y + 0.0
    report = run(frame, groups=groups)
    assert suspects(report) == {("leak", screen.SKILL)}
    assert report.findings


def test_time_blocks_run_and_flag_a_leak():
    frame = world(13, n=400)
    frame["leak"] = frame.y + 0.0
    report = run(frame, block_size=20)
    assert suspects(report) == {("leak", screen.SKILL)}


def test_a_null_outcome_inside_groups_is_not_flagged():
    rng = numpy.random.default_rng(14)
    n = 400
    groups = pandas.Series(numpy.arange(n) // 8)
    frame = pandas.DataFrame(
        {"x0": numpy.repeat(rng.normal(size=n // 8), 8) + rng.normal(size=n)}
    )
    base = numpy.repeat(rng.integers(0, 2, n // 8), 8)
    frame["y"] = base
    report = run(frame, groups=groups)
    assert report.suspects == ()


def test_too_few_rows_or_a_constant_target_say_why_and_flag_nothing():
    small = run(world(15, n=40))
    assert small.suspects == () and "too few rows" in small.notes[0]
    constant = world(15).assign(y=1)
    report = run(constant)
    assert report.suspects == () and "screen not run" in report.notes[0]
    flat = screen.screen(world(15).assign(y=2.0), "y", "regression", FAST)
    assert flat.suspects == () and "constant" in flat.notes[0]


def test_a_family_too_large_to_resolve_is_reported_not_guessed(monkeypatch):
    monkeypatch.setattr(screen, "MAX_PERMUTATIONS", 300)
    frame = world(16, columns=12).assign(leak=lambda f: f.y + 0.0)
    report = run(frame)
    assert report.suspects == () and report.permutations == 0
    assert "permutations" in report.notes[0]


def test_unknown_columns_and_both_scopes_are_refused():
    frame = world(17)
    with pytest.raises(ValueError, match="not feature columns"):
        run(frame, features=["ghost"])
    with pytest.raises(ValueError, match="not both"):
        run(frame, groups=pandas.Series(numpy.arange(300)), block_size=5)
    with pytest.raises(ValueError, match="same length"):
        run(frame, groups=pandas.Series([1, 2]))


def test_large_tables_are_subsampled_without_losing_the_leak():
    frame = world(18, n=screen.MAX_ROWS + 500, columns=2)
    frame["leak"] = frame.y + 0.0
    report = run(frame)
    assert report.rows == screen.MAX_ROWS
    assert suspects(report) == {("leak", screen.SKILL)}


def test_subsample_keeps_groups_whole_and_time_order():
    n = screen.MAX_ROWS * 2
    groups = numpy.repeat(numpy.arange(n // 4), 4)
    rows = screen.subsample(n, groups, None, 0)[0]
    assert len(rows) >= screen.MAX_ROWS
    kept = groups[rows]
    for g in numpy.unique(kept):
        assert (kept == g).sum() == 4
    rows, block = screen.subsample(n, None, 40, 0)
    assert (numpy.diff(rows) > 0).all() and block == 20
    all_rows, same = screen.subsample(100, None, 7, 0)
    assert all_rows.tolist() == list(range(100)) and same == 7


# ---- size ------------------------------------------------------------------


def test_the_global_null_is_flagged_no_more_often_than_alpha():
    significant = 0
    worlds = 60
    for seed in range(worlds):
        frame = world(1000 + seed, n=200, columns=5)
        report = screen.screen(
            frame,
            "y",
            CLS,
            dataclasses.replace(FAST, bootstrap=20, run_seed=seed),
        )
        assert report.suspects == ()
        significant += any(f.significant for f in report.findings)
    # BY at 0.05: expect at most about 3 of 60 worlds; 9 would be p < 0.001.
    assert significant <= 8


# ---- gating ----------------------------------------------------------------


def passing_store():
    return certificates.Store.load(screen.STORE_PATH)


def test_the_shipped_certificate_is_current_for_the_screen_code():
    store = passing_store()
    state = store.state(screen.DETECTOR_ID, screen.code_digest())
    assert state is certificates.Freshness.CURRENT
    gates = screen.gate_statuses(store, leakage.CERTIFIED, CLS, 2)
    assert set(gates) == set(screen.MECHANISM_DEFECTS)
    assert set(gates.values()) == {certificates.GateStatus.GATING}


def test_only_the_certified_target_and_settings_may_gate():
    store = passing_store()
    for task, classes in ((CLS, 3), ("regression", 0)):
        gates = screen.gate_statuses(store, leakage.CERTIFIED, task, classes)
        assert set(gates.values()) == {certificates.GateStatus.DESCRIPTIVE}
    other = dataclasses.replace(leakage.CERTIFIED, skill_ceiling=0.5)
    gates = screen.gate_statuses(store, other, CLS, 2)
    assert set(gates.values()) == {certificates.GateStatus.DESCRIPTIVE}
    seeded = dataclasses.replace(leakage.CERTIFIED, run_seed=11)
    gates = screen.gate_statuses(store, seeded, CLS, 2)
    assert set(gates.values()) == {certificates.GateStatus.GATING}
    gates = screen.gate_statuses(None, leakage.CERTIFIED, CLS, 2)
    assert set(gates.values()) == {certificates.GateStatus.DESCRIPTIVE}


def test_a_stale_certificate_does_not_gate(monkeypatch):
    store = passing_store()
    monkeypatch.setattr(screen, "code_digest", lambda: "0" * 64)
    gates = screen.gate_statuses(store, leakage.CERTIFIED, CLS, 2)
    assert set(gates.values()) == {certificates.GateStatus.DESCRIPTIVE}


def test_a_mechanism_without_a_passing_type_stays_descriptive(monkeypatch):
    store = passing_store()
    monkeypatch.setattr(
        screen,
        "MECHANISM_DEFECTS",
        {**screen.MECHANISM_DEFECTS, screen.PURITY: ("not_certified",)},
    )
    gates = screen.gate_statuses(store, leakage.CERTIFIED, CLS, 2)
    assert gates[screen.PURITY] is certificates.GateStatus.DESCRIPTIVE
    assert gates[screen.SKILL] is certificates.GateStatus.GATING


def test_without_a_store_a_suspect_warns_and_cannot_block():
    frame = world(19).assign(leak=lambda f: f.y + 0.0)
    report = run(frame)
    (found,) = report.suspects
    assert (
        found.gate is certificates.GateStatus.DESCRIPTIVE and not found.blocking
    )
    assert report.blocking == ()
    assert "warning only" in report.sentences()[0]


def test_with_the_shipped_certificate_a_suspect_blocks():
    frame = world(20).assign(leak=lambda f: f.y + 0.0)
    report = screen.screen(
        frame, "y", CLS, leakage.CERTIFIED, store=passing_store()
    )
    assert [f.feature for f in report.blocking] == ["leak"]
    text = report.sentences()[0]
    assert "may block" in text and "not proof" in text


def test_the_report_is_json_ready_and_says_it_is_not_proof():
    import json

    frame = world(21).assign(leak=lambda f: f.y + 0.0)
    document = run(frame).to_dict()
    assert "not proof" in document["note"]
    assert document["rule"] == "benjamini-yekutieli"
    assert document["suspects"] == ["leak"]
    assert document["blocking"] == []
    assert json.loads(json.dumps(document)) == document


def test_screen_table_orders_by_time_and_respects_groups():
    frame = world(22, n=300).assign(leak=lambda f: f.y + 0.0)
    frame["t"] = numpy.random.default_rng(22).permutation(300)
    frame["g"] = numpy.arange(300) % 30
    by_time = screen.screen_table(
        frame, "y", CLS, ["x0", "leak"], 0, time_column="t"
    )
    by_group = screen.screen_table(
        frame, "y", CLS, ["x0", "leak"], 0, group_column="g", time_column="t"
    )
    assert {f.feature for f in by_time.suspects} == {"leak"}
    assert {f.feature for f in by_group.suspects} == {"leak"}
    assert by_time.rows == by_group.rows == 300


def test_r2_matches_sklearn_for_the_observed_regression_fit():
    rng = numpy.random.default_rng(23)
    n = 200
    x = rng.normal(size=n)
    y = x + rng.normal(size=n)
    folds = screen.fold_splits(n, 5, None, None, 0)
    problem = screen.Problem(
        y, "regression", 0, folds, float(((y - y.mean()) ** 2).sum())
    )
    phi = screen.basis(pandas.Series(x))
    got = screen.skill_statistic(problem, phi)(y[None, :])[0]
    pred = screen.oof_fit(y[None, :], phi, folds)[0]
    assert got == pytest.approx(r2_score(y, pred), abs=1e-12)


def test_a_wide_multiclass_table_is_declined_quickly_and_says_so():
    import time

    import numpy
    import pandas

    from scmle.data.admissibility import screen

    rng = numpy.random.default_rng(0)
    frame = pandas.DataFrame(
        rng.normal(size=(600, 64)), columns=[f"c{i}" for i in range(64)]
    )
    frame["y"] = rng.integers(0, 10, 600)
    started = time.monotonic()
    report = screen.screen_table(
        frame,
        "y",
        "classification",
        [c for c in frame if c != "y"],
        0,
        time_budget_seconds=120,
    )
    assert time.monotonic() - started < 20
    assert not [f for f in report.findings if f.suspect]
    assert any("limit is" in note for note in report.notes)
    assert len(report.unchecked) == 64
