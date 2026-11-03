"""Label-noise detectors against cleanlab output, closed forms, simulation."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from scmle.data.detectors import labels as lab

FIXTURE = Path(__file__).parents[1] / "fixtures" / "f5_oracles.json"


@pytest.mark.parametrize("index", [0, 1, 2])
def test_confident_learning_matches_frozen_cleanlab_mask(index):
    case = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"][index]
    probs = np.array(case["probabilities"])
    codes = np.array(case["labels"])
    flags, thresholds = lab.confident_learning_flags(probs, codes)
    np.testing.assert_array_equal(flags, np.array(case["issues"]))
    assert flags.sum() > 0
    np.testing.assert_allclose(
        thresholds, [probs[codes == j, j].mean() for j in range(probs.shape[1])]
    )


def test_assignment_by_hand():
    probs = np.array(
        [[0.9, 0.1, 0.0], [0.2, 0.5, 0.3], [0.34, 0.33, 0.33], [0.1, 0.1, 0.8]]
    )
    thresholds = np.array([0.5, 0.5, 0.5])
    assert list(lab.confident_assignment(probs, thresholds)) == [0, 1, -1, 2]
    exact = np.array([[0.5, 0.5]])
    assert lab.confident_assignment(exact, np.array([0.5, 0.5]))[0] == 0


def test_row_with_no_confident_class_is_not_an_issue():
    probs = np.array(
        [[0.6, 0.4], [0.4, 0.6], [0.45, 0.55], [0.55, 0.45], [0.5, 0.5]]
    )
    codes = np.array([0, 1, 0, 1, 0])
    flags, thresholds = lab.confident_learning_flags(probs, codes)
    assert thresholds[0] == pytest.approx((0.6 + 0.45 + 0.5) / 3)
    assert thresholds[1] == pytest.approx((0.6 + 0.45) / 2)
    assert list(flags) == [False, False, True, True, False]


def test_encode_returns_sorted_classes():
    classes, codes = lab.encode(pd.Series(["b", "a", "b"]))
    assert list(classes) == ["a", "b"] and list(codes) == [1, 0, 1]


def test_ppi_mean_matches_independent_formula():
    rng = np.random.default_rng(0)
    big = rng.random(500) < 0.3
    small_f = rng.random(60) < 0.3
    small_y = small_f ^ (rng.random(60) < 0.1)
    estimate, (low, high) = lab.ppi_mean(big, small_f, small_y, 0.9)
    rect = small_f.astype(float) - small_y
    expected = big.mean() - rect.mean()
    var = big.astype(float).var(ddof=1) / 500 + rect.var(ddof=1) / 60
    half = stats.norm.ppf(0.95) * math.sqrt(var)
    assert estimate == pytest.approx(expected)
    assert (high - low) / 2 == pytest.approx(half)


def test_ppi_interval_covers_the_true_mean_at_nominal_rate():
    truth, covered, reps = 0.2, 0, 600
    for rep in range(reps):
        rng = np.random.default_rng(rep)
        y = rng.random(400) < truth
        f = y ^ (rng.random(400) < 0.15)
        audit = rng.choice(400, 80, replace=False)
        rest = np.setdiff1d(np.arange(400), audit)
        interval = lab.ppi_mean(f[rest], f[audit], y[audit], 0.95)[1]
        low, high = interval
        covered += low <= y.mean() <= high
    assert covered / reps == pytest.approx(0.95, abs=0.04)


def test_ppi_mean_validates():
    ok = np.ones(5)
    with pytest.raises(ValueError, match="audited"):
        lab.ppi_mean(ok, np.ones(1), np.ones(1))
    with pytest.raises(ValueError, match="audited"):
        lab.ppi_mean(np.ones(1), ok, ok)
    with pytest.raises(ValueError, match="audited"):
        lab.ppi_mean(ok, ok, np.ones(4))
    with pytest.raises(ValueError, match="confidence"):
        lab.ppi_mean(ok, ok, ok, 1.0)
    with pytest.raises(ValueError, match="confidence"):
        lab.ppi_mean(ok, ok, ok, 0.0)
    lab.ppi_mean(
        np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([1.0, 0.0])
    )


def test_noise_rate_without_audit_is_flagged_unverified():
    flags = np.zeros(100, dtype=bool)
    flags[:10] = True
    rate = lab.noise_rate(flags)
    assert rate.method == "detector_only" and rate.tier == "unverified"
    assert rate.estimate == pytest.approx(0.1) and rate.audited == 0
    assert rate.interval[0] < 0.1 < rate.interval[1]
    assert "not the noise rate" in rate.caveat


def test_noise_rate_with_audit_uses_prediction_powered_inference():
    rng = np.random.default_rng(2)
    truth = rng.random(1000) < 0.1
    flags = truth ^ (rng.random(1000) < 0.05)
    positions = rng.choice(1000, 100, replace=False)
    audit = pd.Series(truth[positions], index=positions)
    rate = lab.noise_rate(flags, audit)
    assert rate.method == "prediction_powered" and rate.tier == "attested"
    assert (
        rate.audited == 100
        and 0.0 <= rate.interval[0] <= rate.interval[1] <= 1.0
    )
    assert rate.interval[0] <= truth.mean() <= rate.interval[1]


@pytest.mark.parametrize(
    ("audited", "method"),
    [
        (1, "detector_only"),
        (2, "prediction_powered"),
        (8, "prediction_powered"),
        (9, "detector_only"),
    ],
)
def test_noise_rate_audit_size_boundaries(audited, method):
    flags = np.arange(10) % 3 == 0
    audit = pd.Series(np.arange(audited) % 2 == 0, index=np.arange(audited))
    assert lab.noise_rate(flags, audit).method == method


def test_noise_rate_rejects_duplicate_audit_positions():
    audit = pd.Series([True, False, True], index=[1, 1, 2])
    with pytest.raises(ValueError, match="unique"):
        lab.noise_rate(np.zeros(20, dtype=bool), audit)


def test_hoeffding_bound_formula():
    assert lab.hoeffding_bound(5000) == pytest.approx(
        math.sqrt(math.log(40.0) / 10000.0)
    )
    assert lab.hoeffding_bound(5000, 0.99) > lab.hoeffding_bound(5000)


def separable(n=300, flip=0.1, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    x = rng.normal(size=(n, 4)) + 3.0 * y[:, None]
    wrong = rng.random(n) < flip
    return (
        pd.DataFrame(x, columns=list("abcd")),
        pd.Series(np.where(wrong, 1 - y, y)),
        wrong,
    )


def test_margin_flags_mark_rows_the_small_model_cannot_fit():
    x, y, wrong = separable()
    codes = y.to_numpy()
    flags = lab.margin_flags(x, codes, 0)
    assert flags[wrong].mean() > 0.8
    assert flags[~wrong].mean() < 0.1


def test_detect_requires_agreement_and_reports_overlap():
    x, y, wrong = separable()
    report = lab.detect(x, y, seed=0)
    both = set(report.confident_learning) & set(report.margin)
    assert set(report.suspects) == both
    assert 0.0 <= report.agreement <= 1.0
    found = set(report.suspects) & set(np.flatnonzero(wrong))
    assert len(found) / wrong.sum() > 0.7
    assert len(set(report.suspects) - set(np.flatnonzero(wrong))) < 10
    assert (
        report.rate.method == "detector_only" and report.sampling_bound == 0.0
    )
    assert report.rows_checked == report.input_rows == len(x)
    assert report.to_dict()["rows_checked"] == len(x)
    assert lab.detect(x, y, seed=0) == report


def test_detect_declines_small_or_single_class_data():
    x, y = separable(n=lab.MIN_ROWS - 1)[:2]
    with pytest.raises(ValueError, match="50"):
        lab.detect(x, y)
    x, y = separable(n=lab.MIN_ROWS)[:2]
    lab.detect(x, y)
    with pytest.raises(ValueError, match="two classes"):
        lab.detect(x, pd.Series(np.zeros(len(x), dtype=int)))
    y2 = y.copy()
    y2.iloc[:] = 0
    y2.iloc[0] = 1
    with pytest.raises(ValueError, match="at least two rows"):
        lab.detect(x, y2)


def test_detect_records_the_sampling_cap_and_keeps_audited_rows(monkeypatch):
    x, y, wrong = separable(n=200)
    monkeypatch.setattr(lab, "MAX_ROWS", 120)
    audit = pd.Series(wrong[:30], index=np.arange(30))
    report = lab.detect(x, y, seed=0, audit=audit)
    assert report.rows_checked == 120 and report.input_rows == 200
    assert report.sampling_bound == pytest.approx(lab.hoeffding_bound(120))
    assert report.rate.method == "prediction_powered"
    assert report.rate.audited == 30
    capped = lab.detect(x, y, seed=0)
    assert capped.rows_checked == 120 and capped.rate.audited == 0
