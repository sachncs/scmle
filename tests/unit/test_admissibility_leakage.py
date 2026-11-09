"""Leakage suspicion tests against scipy/sklearn references and plants."""

import dataclasses
import math

import numpy
import pandas
import pytest
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score

from scmle.contracts import qualification
from scmle.contracts.qualification import Decision
from scmle.contracts.qualification import SetId
from scmle.contracts.qualification import Tier
from scmle.data.admissibility import leakage
from scmle.data.detectors import certificates

CLS = "classification"
GATING = certificates.GateStatus.GATING
FAST = leakage.LeakageConfig(
    skill_ceiling=0.95,
    association_ceiling=0.9,
    permutations=39,
    bootstrap=100,
)


def binary_world(seed, n=300):
    rng = numpy.random.default_rng(seed)
    y = pandas.Series(rng.integers(0, 2, n), name="y")
    noise = pandas.Series(rng.normal(size=n), name="noise")
    return y, noise


# ---- references -----------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_binary_auc_matches_sklearn_with_ties(seed):
    rng = numpy.random.default_rng(seed)
    truth = rng.integers(0, 2, 200)
    score = numpy.round(rng.normal(size=200) + truth, 1)  # many ties
    assert leakage.binary_auc(truth, score) == pytest.approx(
        roc_auc_score(truth, score), abs=1e-12
    )


def test_binary_auc_needs_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        leakage.binary_auc(numpy.zeros(5, dtype=int), numpy.arange(5.0))


def test_skill_of_matches_sklearn_for_each_task():
    rng = numpy.random.default_rng(0)
    y3 = rng.integers(0, 3, 300)
    prob = rng.dirichlet(numpy.ones(3), 300)
    assert leakage.skill_of(prob, y3, CLS) == pytest.approx(
        roc_auc_score(y3, prob, multi_class="ovr", average="macro")
    )
    y2 = rng.integers(0, 2, 300)
    p2 = rng.dirichlet(numpy.ones(2), 300)
    assert leakage.skill_of(p2, y2, CLS) == pytest.approx(
        roc_auc_score(y2, p2[:, 1])
    )
    yr = rng.normal(size=300)
    pr = yr + rng.normal(size=300)
    assert leakage.skill_of(pr, yr, "regression") == pytest.approx(
        r2_score(yr, pr)
    )
    with pytest.raises(ValueError, match="constant"):
        leakage.skill_of(pr, numpy.ones(300), "regression")


def test_permutation_p_value_formula_and_tie_direction():
    null = [0.5, 0.6, 0.7, 0.8]
    assert leakage.permutation_p(0.9, null) == pytest.approx(1 / 5)
    assert leakage.permutation_p(0.7, null) == pytest.approx(3 / 5)  # >= counts
    assert leakage.permutation_p(0.0, null) == pytest.approx(1.0)
    assert leakage.permutation_p(0.71, null) == pytest.approx(2 / 5)


def test_cramers_v_matches_scipy_association():
    table = numpy.array([[20, 5], [8, 40]])
    expected = stats.contingency.association(
        table, method="cramer", correction=False
    )
    assert leakage.cramers_v(table) == pytest.approx(expected)
    wide = numpy.array([[10, 20, 30], [30, 20, 10]])
    assert leakage.cramers_v(wide) == pytest.approx(
        stats.contingency.association(wide, method="cramer", correction=False)
    )
    assert leakage.cramers_v(numpy.zeros((2, 2))) == 0.0


def test_permute_within_keeps_groups_blocks_and_is_a_permutation():
    rng = numpy.random.default_rng(0)
    values = numpy.arange(12)
    groups = numpy.array(list("aaabbbbcccdd"))
    order = leakage.permute_within(values, rng, groups, None)
    assert sorted(order) == list(range(12))
    assert (groups[order] == groups).all()
    block = leakage.permute_within(values, numpy.random.default_rng(1), None, 5)
    assert sorted(block[:5]) == list(range(5))
    assert sorted(block[5:10]) == list(range(5, 10))
    assert sorted(block[10:]) == [10, 11]
    free = leakage.permute_within(
        values, numpy.random.default_rng(2), None, None
    )
    assert sorted(free) == list(range(12)) and list(free) != list(range(12))


def test_design_handles_numeric_categorical_and_missing():
    num = leakage.design(pandas.Series([1.0, numpy.nan, 3.0]))
    assert num.shape == (3, 1) and numpy.isnan(num[1, 0])
    cat = leakage.design(pandas.Series(["a", None, "b", "a"]))
    assert cat.shape == (4, 3)  # a, b, <missing>
    assert cat.sum(axis=1).tolist() == [1, 1, 1, 1]
    many = leakage.design(pandas.Series([str(i % 80) for i in range(400)]))
    assert many.shape[1] == leakage.MAX_LEVELS + 1  # pooled "<other>"
    boolean = leakage.design(pandas.Series([True, False, True]))
    assert boolean.shape == (3, 2)


def test_target_encoding_rules():
    assert leakage.encode_target(
        pandas.Series(["b", "a", "b"]), CLS
    ).tolist() == [
        1,
        0,
        1,
    ]
    with pytest.raises(ValueError, match="two classes"):
        leakage.encode_target(pandas.Series([1, 1, 1]), CLS)
    with pytest.raises(ValueError, match="missing"):
        leakage.encode_target(pandas.Series([0, numpy.nan, 1]), CLS)
    with pytest.raises(ValueError, match="task"):
        leakage.encode_target(pandas.Series([0, 1]), "ranking")


def test_config_boundaries():
    ok = leakage.LeakageConfig(0.9, 0.5, alpha=0.05, permutations=20)
    assert ok.permutations == 20
    with pytest.raises(ValueError, match="too few"):
        leakage.LeakageConfig(0.9, 0.5, alpha=0.04, permutations=20)
    leakage.LeakageConfig(0.9, 0.5, alpha=0.04, permutations=24)  # B + 1 == 25
    with pytest.raises(ValueError):
        leakage.LeakageConfig(0.9, 0.5, permutations=19)
    for bad in (0.0, 1.5, float("nan")):
        with pytest.raises(ValueError):
            leakage.LeakageConfig(bad, 0.5)
        with pytest.raises(ValueError):
            leakage.LeakageConfig(0.9, bad)
    leakage.LeakageConfig(1.0, 1.0)
    for field in ("alpha", "confidence"):
        for bad in (0.0, 1.0):
            with pytest.raises(ValueError):
                leakage.LeakageConfig(0.9, 0.5, **{field: bad})
    with pytest.raises(ValueError):
        leakage.LeakageConfig(0.9, 0.5, folds=1)


# ---- the skill test -------------------------------------------------------


def test_null_features_are_rarely_suspect_and_p_values_are_valid():
    runs = 150
    small = 0
    suspects = 0
    for seed in range(runs):
        y, noise = binary_world(seed)
        cfg = dataclasses.replace(FAST, run_seed=seed)
        out = leakage.skill_test(noise, y, CLS, cfg)
        small += out.p_value <= 0.05
        suspects += out.suspect
    band = 0.05 + 2.576 * math.sqrt(0.05 * 0.95 / runs)
    assert small / runs <= band
    assert suspects == 0


def planted(kind, seed, n=300):
    y = binary_world(seed, n)[0]
    rng = numpy.random.default_rng(seed + 1)
    if kind == "copy":
        x = y.astype(float)
    elif kind == "noisy":
        flip = rng.random(n) < 0.01
        x = y.where(~flip, 1 - y).astype(float)
    elif kind == "threshold":
        x = y + rng.normal(scale=0.25, size=n)
    else:  # recorded only for one class
        x = pandas.Series(rng.normal(size=n))
        x[y == 0] = numpy.nan
    return pandas.Series(numpy.asarray(x, dtype=float), name=kind), y


@pytest.mark.parametrize("kind", ["copy", "noisy", "threshold", "missing"])
def test_planted_leaks_are_suspects(kind):
    for seed in range(6):
        x, y = planted(kind, seed)
        out = leakage.skill_test(
            x, y, CLS, dataclasses.replace(FAST, run_seed=seed)
        )
        assert out.suspect, (kind, seed, out.skill, out.lower_bound)
        assert out.p_value == pytest.approx(1 / 40)
        assert out.lower_bound > FAST.skill_ceiling


def test_a_strong_legitimate_predictor_below_the_ceiling_is_not_a_suspect():
    rng = numpy.random.default_rng(3)
    y = pandas.Series(rng.integers(0, 2, 400))
    x = pandas.Series(
        y + rng.normal(scale=1.0, size=400), name="x"
    )  # AUC ~ 0.76
    out = leakage.skill_test(x, y, CLS, FAST)
    assert out.p_value <= 0.05 and 0.6 < out.skill < 0.9
    assert not out.suspect


def test_suspect_requires_the_p_value_and_a_strictly_higher_bound(monkeypatch):
    x, y = planted("copy", 0)
    monkeypatch.setattr(
        leakage, "lower_bound", lambda *a, **k: FAST.skill_ceiling
    )
    assert not leakage.skill_test(x, y, CLS, FAST).suspect  # bound == ceiling
    monkeypatch.setattr(
        leakage, "lower_bound", lambda *a, **k: FAST.skill_ceiling + 1e-9
    )
    assert leakage.skill_test(x, y, CLS, FAST).suspect
    monkeypatch.setattr(leakage, "permutation_p", lambda *a, **k: FAST.alpha)
    assert leakage.skill_test(x, y, CLS, FAST).suspect  # p == alpha counts
    monkeypatch.setattr(
        leakage, "permutation_p", lambda *a, **k: FAST.alpha + 1e-9
    )
    assert not leakage.skill_test(x, y, CLS, FAST).suspect


def test_few_rows_report_low_power_and_never_a_suspect():
    x, y = planted("copy", 0, n=99)
    out = leakage.skill_test(x, y, CLS, FAST)
    assert out.low_power and not out.suspect
    assert out.check.decision is Decision.WARN
    x, y = planted("copy", 0, n=100)
    assert not leakage.skill_test(x, y, CLS, FAST).low_power


def test_a_rare_class_is_low_power():
    y = pandas.Series([1] * 4 + [0] * 296)
    x = y.astype(float).rename("x")
    assert leakage.skill_test(x, y, CLS, FAST).low_power  # 4 < 5 folds


def test_the_decision_is_demoted_until_the_detector_is_certified():
    x, y = planted("copy", 0)
    soft = leakage.skill_test(x, y, CLS, FAST)
    hard = leakage.skill_test(x, y, CLS, FAST, gate=GATING)
    assert soft.check.decision is Decision.WARN
    assert hard.check.decision is Decision.QUARANTINE
    yy, clean = binary_world(0)
    assert leakage.skill_test(
        clean, yy, CLS, FAST, gate=GATING
    ).check.decision is (Decision.PASS)


def test_the_ledger_record_is_complete():
    x, y = planted("copy", 0)
    cfg = dataclasses.replace(FAST, run_seed=11)
    out = leakage.skill_test(x, y, CLS, cfg)
    check = out.check
    assert check.check_id == "leakage.skill.copy"
    assert check.set_id is SetId.EXPLORATION and check.tier is Tier.UNVERIFIED
    assert check.p_value == out.p_value and check.n == 300
    assert check.seed_path[0] == 11
    assert check.interval is not None and check.interval[0] == out.lower_bound
    assert check.effect == pytest.approx(out.skill - out.null_mean)
    assert qualification.Check.from_dict(check.to_dict()) == check


def test_results_are_deterministic_and_seed_dependent():
    y, noise = binary_world(0)
    a = leakage.skill_test(noise, y, CLS, FAST)
    b = leakage.skill_test(noise, y, CLS, FAST)
    c = leakage.skill_test(noise, y, CLS, dataclasses.replace(FAST, run_seed=5))
    assert a == b
    assert (a.p_value, a.lower_bound) != (c.p_value, c.lower_bound)


def test_regression_target_is_supported():
    rng = numpy.random.default_rng(0)
    y = pandas.Series(rng.normal(size=300))
    leak = (y + rng.normal(scale=0.05, size=300)).rename("leak")
    cfg = dataclasses.replace(FAST, skill_ceiling=0.9)
    assert leakage.skill_test(leak, y, "regression", cfg).suspect
    noise = pandas.Series(rng.normal(size=300), name="n")
    assert not leakage.skill_test(noise, y, "regression", cfg).suspect


def test_groups_and_blocks_are_respected():
    rng = numpy.random.default_rng(0)
    groups = pandas.Series(numpy.repeat(numpy.arange(30), 10))
    y = pandas.Series(rng.integers(0, 2, 300))
    x = y.astype(float).rename("x")
    grouped = leakage.skill_test(x, y, CLS, FAST, groups=groups)
    assert grouped.suspect
    blocked = leakage.skill_test(x, y, CLS, FAST, block_size=50)
    assert blocked.suspect
    with pytest.raises(ValueError, match="not both"):
        leakage.skill_test(x, y, CLS, FAST, groups=groups, block_size=50)
    with pytest.raises(ValueError, match="same length"):
        leakage.skill_test(x.iloc[:10], y, CLS, FAST)


def test_group_level_signal_is_not_suspicious_under_within_group_permutation():
    # The outcome is constant within a group, so permuting within groups
    # changes nothing: the test must not call a group-constant feature a leak.
    rng = numpy.random.default_rng(0)
    groups = pandas.Series(numpy.repeat(numpy.arange(40), 8))
    level = rng.integers(0, 2, 40)
    y = pandas.Series(level[groups.to_numpy()])
    x = y.astype(float).rename("x")
    out = leakage.skill_test(x, y, CLS, FAST, groups=groups)
    assert out.p_value == pytest.approx(1.0)
    assert not out.suspect


# ---- missingness ----------------------------------------------------------


def missing_pair(table):
    rows = []
    for (gone, label), k in zip(
        [(0, 0), (0, 1), (1, 0), (1, 1)], numpy.ravel(table), strict=True
    ):
        rows += [(numpy.nan if gone else 1.0, label)] * int(k)
    frame = pandas.DataFrame(rows, columns=["f", "y"])
    return frame["f"], frame["y"]


def test_missingness_uses_fisher_for_small_2x2_and_chi2_otherwise():
    cfg = dataclasses.replace(FAST, min_rows=1)
    small = numpy.array([[12, 3], [2, 8]])  # expected count below 5 in a cell
    f, y = missing_pair(small)
    out = leakage.missingness_test(f, y, CLS, cfg)
    assert out is not None and out.method == "fisher"
    assert out.p_value == pytest.approx(stats.fisher_exact(small.T).pvalue)
    big = numpy.array([[60, 40], [20, 80]])
    f, y = missing_pair(big)
    out = leakage.missingness_test(f, y, CLS, cfg)
    assert out is not None and out.method == "chi2"
    table = pandas.crosstab(f.isna(), y).to_numpy()
    assert out.p_value == pytest.approx(
        stats.chi2_contingency(table, correction=False)[1]
    )
    assert out.cramers_v == pytest.approx(leakage.cramers_v(table))
    assert out.n_missing == int(f.isna().sum())


def test_missingness_suspect_boundaries_are_inclusive(monkeypatch):
    f, y = missing_pair(numpy.array([[100, 0], [0, 100]]))
    out = leakage.missingness_test(f, y, CLS, FAST)
    assert (
        out is not None and out.suspect and out.cramers_v == pytest.approx(1.0)
    )
    strict = dataclasses.replace(FAST, association_ceiling=1.0)
    assert leakage.missingness_test(f, y, CLS, strict).suspect  # V == ceiling
    f2, y2 = missing_pair(numpy.array([[100, 0], [40, 60]]))
    out2 = leakage.missingness_test(f2, y2, CLS, FAST)
    assert out2 is not None and out2.cramers_v < 0.9
    assert not out2.suspect
    assert out2.p_value < 1e-10
    few = leakage.missingness_test(f.iloc[:50], y.iloc[:50], CLS, FAST)
    assert few is None or not few.suspect


def test_missingness_test_is_skipped_when_there_is_nothing_to_test():
    y = pandas.Series([0, 1] * 50)
    assert (
        leakage.missingness_test(pandas.Series(numpy.ones(100)), y, CLS, FAST)
        is None
    )
    assert (
        leakage.missingness_test(
            pandas.Series(numpy.nan, index=range(100)), y, CLS, FAST
        )
        is None
    )
    with pytest.raises(ValueError, match="same length"):
        leakage.missingness_test(pandas.Series([1.0]), y, CLS, FAST)


def test_missingness_with_a_continuous_target_is_binned():
    rng = numpy.random.default_rng(0)
    y = pandas.Series(rng.normal(size=400))
    f = pandas.Series(numpy.where(y > 0.5, numpy.nan, 1.0))
    cfg = dataclasses.replace(FAST, association_ceiling=0.8)
    out = leakage.missingness_test(f, y, "regression", cfg)
    assert out is not None and out.suspect


# ---- conditional test and scan -------------------------------------------


def test_conditional_test_explains_away_a_feature_carried_by_others():
    rng = numpy.random.default_rng(0)
    n = 400
    z = rng.normal(size=n)
    y = pandas.Series((rng.random(n) < 1 / (1 + numpy.exp(-6 * z))).astype(int))
    x = pandas.Series(z + rng.normal(scale=0.05, size=n), name="x")
    others = pandas.DataFrame({"z": z})
    marginal = leakage.skill_test(x, y, CLS, FAST)
    assert marginal.skill > 0.85
    out = leakage.conditional_test(x, others, y, CLS, FAST)
    assert not out.keeps_signal and out.p_value > 0.05
    assert out.strata >= 2


def test_conditional_test_keeps_the_signal_of_a_leak_given_the_others():
    y, noise = binary_world(0, 400)
    others = pandas.DataFrame({"n": noise})
    leak = y.astype(float).rename("leak")
    out = leakage.conditional_test(leak, others, y, CLS, FAST)
    assert out.keeps_signal and out.p_value == pytest.approx(1 / 40)
    assert out.check.decision is Decision.WARN
    nothing = leakage.conditional_test(leak, others.iloc[:, :0], y, CLS, FAST)
    assert nothing.strata == 1 and nothing.keeps_signal


def test_strata_come_from_a_tree_on_the_other_features():
    rng = numpy.random.default_rng(0)
    z = pandas.DataFrame({"z": rng.normal(size=200)})
    num = pandas.Series(z["z"].to_numpy() * 2, name="x")
    labels = leakage.strata_labels(num, z, FAST)
    assert 2 <= len(numpy.unique(labels)) <= FAST.n_strata
    cat = pandas.Series(numpy.where(z["z"] > 0, "hi", "lo"), name="c")
    labels = leakage.strata_labels(cat, z, FAST)
    assert len(numpy.unique(labels)) >= 2
    assert set(leakage.strata_labels(num, z.iloc[:, :0], FAST)) == {0}
    # missingness is not a stratum: the feature may move its gaps
    gappy = num.copy()
    gappy.iloc[:20] = numpy.nan
    assert len(set(leakage.strata_labels(gappy, z, FAST))) <= FAST.n_strata


def test_scan_reports_every_feature_and_never_claims_proof():
    y, noise = binary_world(0, 300)
    frame = pandas.DataFrame({"noise": noise, "leak": y.astype(float), "y": y})
    report = leakage.scan(frame, "y", CLS, FAST)
    assert report.suspects == {"noise": False, "leak": True}
    leak = report.assessments[1]
    assert leak.escalated and leak.conditional is not None
    assert report.assessments[0].conditional is None
    doc = report.to_dict()
    assert "not proof" in doc["note"]
    assert doc["gate"] == "descriptive"
    assert [f["feature"] for f in doc["features"]] == ["noise", "leak"]
    ids = [c.check_id for c in report.checks]
    assert ids.count("leakage.skill.noise") == 1
    assert "leakage.conditional.leak" in ids
    assert len(ids) == len(set(ids))
    only = leakage.scan(frame, "y", CLS, FAST, features=["noise"])
    assert list(only.suspects) == ["noise"]
    with pytest.raises(ValueError, match="not feature columns"):
        leakage.scan(frame, "y", CLS, FAST, features=["y"])


def test_missing_for_class_is_found_by_both_tests():
    x, y = planted("missing", 0)
    frame = pandas.DataFrame({"x": x, "y": y})
    out = leakage.assess_feature(frame["x"], frame.iloc[:, :0], y, CLS, FAST)
    assert out.missingness is not None and out.missingness.suspect
    assert out.skill.suspect and out.suspect


# ---- gating from certificates --------------------------------------------


def store_for(digest, defects=leakage.LEAK_DEFECTS, passed=True):
    results = []
    for defect in defects:
        results.append(
            qualification.Certificate(
                leakage.DETECTOR_ID,
                defect,
                200,
                200 if passed else 100,
                1000,
                0,
                (0, "t", 0),
            )
        )
    cert = certificates.certify(
        leakage.DETECTOR_ID,
        digest,
        "b" * 64,
        [0],
        results,
        certificates.CertificationPolicy(0.9, 0.05, 0.95),
    )
    return certificates.CertificateStore([cert])


def test_gate_requires_current_passing_certificates_for_every_leak_type():
    digest = leakage.code_digest()
    good = store_for(digest)
    assert leakage.gate_status(good, leakage.CERTIFIED) is GATING
    descriptive = certificates.GateStatus.DESCRIPTIVE
    assert (
        leakage.gate_status(certificates.CertificateStore(), leakage.CERTIFIED)
        is descriptive
    )
    assert (
        leakage.gate_status(store_for("c" * 64), leakage.CERTIFIED)
        is descriptive
    )  # stale
    assert (
        leakage.gate_status(store_for(digest, passed=False), leakage.CERTIFIED)
        is descriptive
    )
    missing_type = store_for(digest, leakage.LEAK_DEFECTS[:-1])
    assert leakage.gate_status(missing_type, leakage.CERTIFIED) is descriptive


def test_gate_requires_the_certified_settings_and_a_binary_target():
    good = store_for(leakage.code_digest())
    descriptive = certificates.GateStatus.DESCRIPTIVE
    assert (
        leakage.gate_status(
            good, dataclasses.replace(leakage.CERTIFIED, run_seed=9)
        )
        is GATING
    )
    for change in (
        {"skill_ceiling": 0.96},
        {"alpha": 0.1},
        {"permutations": 99},
        {"max_leaf_nodes": 4},
    ):
        cfg = dataclasses.replace(leakage.CERTIFIED, **change)
        assert leakage.gate_status(good, cfg) is descriptive
    assert (
        leakage.gate_status(good, leakage.CERTIFIED, "regression")
        is descriptive
    )
    assert (
        leakage.gate_status(good, leakage.CERTIFIED, CLS, n_classes=3)
        is descriptive
    )


def test_code_digest_is_a_sha256_of_the_module_source():
    assert len(leakage.code_digest()) == 64
    assert leakage.code_digest() == leakage.code_digest()


def test_enough_rows_boundaries_are_exact():
    balanced = numpy.array([0, 1] * 50)
    assert leakage.enough_rows(balanced, CLS, FAST)  # 100 rows, 50 per class
    assert not leakage.enough_rows(balanced[:98], CLS, FAST)
    five = numpy.array([1] * 5 + [0] * 95)
    four = numpy.array([1] * 4 + [0] * 96)
    assert leakage.enough_rows(five, CLS, FAST)  # 5 == folds
    assert not leakage.enough_rows(four, CLS, FAST)
    assert leakage.enough_rows(numpy.arange(100.0), "regression", FAST)
    assert not leakage.enough_rows(numpy.arange(99.0), "regression", FAST)


def test_lower_bound_is_the_requested_bootstrap_quantile():
    rng = numpy.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    score = y * 0.5 + rng.random(200) * 0.8
    pred = numpy.column_stack([1 - score, score])
    loose = dataclasses.replace(FAST, confidence=0.5)
    tight = dataclasses.replace(FAST, confidence=0.99)
    mid = leakage.lower_bound(
        pred, y, CLS, loose, None, numpy.random.default_rng(1)
    )
    low = leakage.lower_bound(
        pred, y, CLS, tight, None, numpy.random.default_rng(1)
    )
    point = leakage.skill_of(pred, y, CLS)
    assert low < mid <= point + 0.02
    # cluster bootstrap resamples whole groups
    groups = numpy.repeat(numpy.arange(20), 10)
    by_group = leakage.lower_bound(
        pred, y, CLS, tight, groups, numpy.random.default_rng(1)
    )
    assert by_group <= point


def test_a_target_with_one_class_in_every_draw_gives_no_bound():
    y = numpy.zeros(20, dtype=int)
    pred = numpy.column_stack([numpy.ones(20), numpy.zeros(20)])
    assert (
        leakage.lower_bound(
            pred, y, CLS, FAST, None, numpy.random.default_rng(0)
        )
        == -math.inf
    )
