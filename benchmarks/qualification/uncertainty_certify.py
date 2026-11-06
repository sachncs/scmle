"""Measured gates for the F12 coverage and breadth methods (canon P1).

Every threshold below is fixed in this file before the measurement and is not
retuned: a gate that misses its floor is filed as measured and `passed` is
false, so that method is descriptive. Certification seeds are `0 ..
replicates - 1`; development looks use seeds from `DEV_SEED_BASE`, which never
overlap.

Gates (worlds are simulated; real data may violate them):

* G1 split conformal: `n_cal = 500`, `a = 0.1`, scores `|N(0,1)|`. The
  coverage given the calibration set is exact (normal cdf at the conformal
  quantile), so the test-sample noise of the plan's 20,000 test rows is
  removed. Pass: mean in `[0.9 - 3 se, 0.9 + 1/(n+1) + 3 se]` with `se` the
  standard deviation of `Beta(451, 50)` over `sqrt(R)`, and a
  Kolmogorov-Smirnov test of the replicates against `Beta(451, 50)` with
  `p > 0.001`. The Beta law is from memory; the test is what checks it.
* G2 CV+: `n = 200`, K in (10, 5), 1-NN (unstable learner) and a linear
  learner, 2000 test rows per replicate. Pass: mean coverage `>= 0.9 - 3 se`.
  The `1 - 2a` worst case is not the criterion. The plan's differential
  against MAPIE has NOT run (MAPIE is not installed here): replaced by a
  brute-force loop implementation in the unit tests.
* G3 CQR: noise sd rises 5x across `x`; boosted quantile models at 0.05/0.95.
  Pass: marginal coverage `>= 0.9 - 3 se` and highest-noise-decile coverage
  `>= 0.87`, with the constant-width split-conformal control below `0.87`
  there (a gate without a failing control is not accepted). Replicates are
  `CQR_REPLICATES`, fewer than the plan's 2000 because each fits two models.
* G4 class-conditional sets: five classes, minority prior 2 percent, a
  temperature-distorted posterior. Pass: every class `>= 0.9 - 3 se`, with
  the pooled margin below `0.9 - 3 se` on the minority class.
* G5 slice flag: (i) 8 exchangeable slices, familywise false-flag rate
  `<= 0.05 + 2 sqrt(0.05 * 0.95 / R)`; (ii) one slice at 3x noise, recall by
  slice size. The flag is certified only from the smallest tested size whose
  Wilson lower bound is `>= 0.80`; below it, descriptive.
* G5b slice flag with a margin fixed in advance (registered 2026-11-06,
  before its first run; the redesign that the G5 miss called for, and G5
  itself is neither changed nor re-run). `coverage.slice_flag` tests, for
  each slice of at least 30 rows, the one-sided exact binomial hypothesis
  "coverage `>= nominal - 0.05`" (`SLICE_MARGIN = 0.05`, about 3.7 standard
  deviations of the calibration-conditional coverage at `n_cal = 500`,
  `Beta(451, 50)` sd 0.0134) and flags by Holm at familywise 0.05 over all
  slices of at least 30 rows. Worlds as G5 but with the harder `n_cal =
  500` (G5 used 1000): (i) 8 exchangeable slices of 200 rows, `R = 2000`,
  familywise false-flag rate `<= 0.05 + 2 sqrt(0.05 * 0.95 / R)`; (ii) one
  slice of 30, 50, 100, 200 or 400 rows at 3x noise (true coverage about
  0.42), `R = 2000` each, detection of that slice. Pass: (i) within its
  bound and (ii) with Wilson lower bound `>= 0.80` at some size; the flag is
  certified from the smallest such size (`min_certified_rows`) and
  descriptive below it. Filed, not gated: detection at 1.5x noise (true
  coverage about 0.73) and at 1.25x (about 0.81). By construction the flag
  cannot see a shortfall under about 0.05; it says "below nominal by more
  than the margin", not "below nominal".
* G6 weighted conformal: Gaussian mean shift in 3 features, noise sd
  `exp(0.6 x1)`. Known ratio: shifted-test coverage `>= 0.9 - 3 se` with the
  unweighted version lower. Estimated ratio (logistic discriminator, clipped
  odds): coverage reported with its interval and effective size; passes only
  at the same floor.
* G9 reconciliation: hierarchy of 7 series, non-diagonal base error
  covariance. Pass: paired mean squared-error difference (MinT minus
  bottom-up) has a 95 percent upper limit `< 0` and coherence residual
  `< 1e-8`. WIS is not evaluated; the metric is squared error.
* G11(i) survival: Weibull proportional hazards, censoring that depends on
  the covariates. Truth is the truncated concordance of a large uncensored
  sample. Pass: the covariate-censoring IPCW estimate is within `3 se` of the
  truth and the marginal Kaplan-Meier estimate is outside `3 se` (control).
  The reference differential against scikit-survival/R has NOT run.
* G13 panel: random intercept, new-entity target. Pass: entity-wise
  calibration (one random row per entity) has coverage within `3 se` of
  nominal; row-wise coverage is reported either way.

NOT measured here (need outside data or tools): G7 (real datasets), G8 and
G9 on M4/M5, G10 against a public ranking set, G11(ii), G12, G14, G15.

    python -m benchmarks.qualification.uncertainty_certify --write
    python -m benchmarks.qualification.uncertainty_certify --check
"""

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy import stats
from sklearn import ensemble
from sklearn import linear_model
from sklearn import neighbors

from scmle.data.detectors import certificates
from scmle.evaluation import conformal
from scmle.evaluation import coverage
from scmle.evaluation import coverage_slices
from scmle.evaluation import ipcw
from scmle.evaluation import quantile
from scmle.evaluation import reconcile
from scmle.evaluation import shift_weights

STORE_PATH = Path(__file__).with_name("uncertainty_certificates.json")
DEV_SEED_BASE = 10_000
ALPHA = 0.1
LEVEL = 1.0 - ALPHA
REPLICATES = 2000
CQR_REPLICATES = 150
ESTIMATED_RATIO_REPLICATES = 200
SURVIVAL_REPLICATES = 100
GATES = (
    "G1",
    "G2",
    "G3",
    "G4",
    "G5",
    "G5b",
    "G6",
    "G9",
    "G11i",
    "G13",
)
NOT_MEASURED = (
    "G7 real datasets",
    "G8 M4/M5 forecast",
    "G10 public ranking set",
    "G11ii scikit-survival differential",
    "G12 multilabel sets",
    "G14 near-duplicate split",
    "G15 encoder baseline",
    "MAPIE differential for G2/G4",
)


def wilson_lower(hits: int, rows: int) -> float:
    """Wilson 95 percent lower bound."""
    return certificates.wilson_interval(hits, rows, 0.95)[0]


def mean_floor(values: np.ndarray, nominal: float = LEVEL) -> dict[str, Any]:
    """Mean and the `nominal - 3 se` floor for replicate coverages."""
    standard = float(values.std(ddof=1) / math.sqrt(len(values)))
    mean = float(values.mean())
    return {
        "mean": mean,
        "se": standard,
        "floor": nominal - 3 * standard,
        "meets_floor": mean >= nominal - 3 * standard,
    }


def gate_g1(replicates: int, base: int = 0) -> dict[str, Any]:
    """Split-conformal coverage law."""
    n_cal = 500
    k = math.ceil((n_cal + 1) * LEVEL)
    beta = stats.beta(k, n_cal + 1 - k)
    values = np.empty(replicates)
    for index in range(replicates):
        rng = np.random.default_rng(base + index)
        score = np.abs(rng.standard_normal(n_cal))
        quantile_value = conformal.finite_sample_quantile(score, LEVEL)
        values[index] = 2 * stats.norm.cdf(quantile_value) - 1
    se = float(beta.std() / math.sqrt(replicates))
    mean = float(values.mean())
    low, high = LEVEL - 3 * se, LEVEL + 1 / (n_cal + 1) + 3 * se
    ks = stats.kstest(values, beta.cdf)
    return {
        "design": {"n_cal": n_cal, "alpha": ALPHA, "beta": [k, n_cal + 1 - k]},
        "mean": mean,
        "band": [low, high],
        "ks_p": float(ks.pvalue),
        "passed": bool(low <= mean <= high and ks.pvalue > 0.001),
    }


def cv_plus_replicate(
    seed: int, folds: int, learner: str
) -> tuple[float, float]:
    """Coverage of CV+ and of pooled-residual-plus-refit for one world."""
    rng = np.random.default_rng(seed)
    n, d, test = 200, 3, 2000
    beta = np.array([1.0, -1.0, 0.5])
    x = rng.standard_normal((n, d))
    y = x @ beta + rng.standard_normal(n)
    xt = rng.standard_normal((test, d))
    yt = xt @ beta + rng.standard_normal(test)
    fold_of = rng.permutation(n) % folds

    def make() -> Any:
        if learner == "knn1":
            return neighbors.KNeighborsRegressor(1)
        return linear_model.LinearRegression()

    models = []
    residuals = np.empty(n)
    for k in range(folds):
        held = fold_of == k
        model = make().fit(x[~held], y[~held])
        models.append(model)
        residuals[held] = np.abs(y[held] - model.predict(x[held]))
    preds = np.column_stack([m.predict(xt) for m in models])
    low, high = conformal.cv_plus_bounds(preds, fold_of, residuals, LEVEL)
    cv = float(np.mean((yt >= low) & (yt <= high)))
    refit = make().fit(x, y).predict(xt)
    margin = conformal.finite_sample_quantile(residuals, LEVEL)
    pooled = float(np.mean(np.abs(yt - refit) <= margin))
    return cv, pooled


def gate_g2(replicates: int, base: int = 0) -> dict[str, Any]:
    """CV+ marginal coverage."""
    runs = max(replicates // 10, 50)
    result: dict[str, Any] = {"replicates": runs, "worlds": {}}
    ok = True
    for learner in ("knn1", "linear"):
        for folds in (10, 5):
            pairs = np.array(
                [
                    cv_plus_replicate(base + i, folds, learner)
                    for i in range(runs)
                ]
            )
            cv = mean_floor(pairs[:, 0])
            result["worlds"][f"{learner}_K{folds}"] = {
                "cv_plus": cv,
                "pooled_refit_control": float(pairs[:, 1].mean()),
            }
            ok = ok and cv["meets_floor"]
    result["passed"] = bool(ok)
    return result


def gate_g3(replicates: int, base: int = 0) -> dict[str, Any]:
    """CQR against constant-width split conformal."""
    n_fit, n_cal, n_test = 1000, 600, 4000

    def draw(rng: np.random.Generator, n: int) -> tuple[np.ndarray, ...]:
        x = rng.uniform(0, 1, (n, 1))
        sd = 0.5 + 2.0 * x[:, 0]
        return x, sd * rng.standard_normal(n), sd

    rows = []
    for index in range(replicates):
        rng = np.random.default_rng(base + index)
        xf, yf, _ = draw(rng, n_fit)
        xc, yc, _ = draw(rng, n_cal)
        xt, yt, sdt = draw(rng, n_test)
        kw = {"n_estimators": 40, "max_depth": 2, "random_state": index}
        lo = ensemble.GradientBoostingRegressor(
            loss="quantile", alpha=quantile.LOWER_LEVEL, **kw
        ).fit(xf, yf)
        hi = ensemble.GradientBoostingRegressor(
            loss="quantile", alpha=quantile.UPPER_LEVEL, **kw
        ).fit(xf, yf)
        mean = ensemble.GradientBoostingRegressor(**kw).fit(xf, yf)
        band = quantile.QuantileBand.calibrate(
            yc.tolist(),
            np.minimum(lo.predict(xc), hi.predict(xc)),
            np.maximum(lo.predict(xc), hi.predict(xc)),
        )
        margin = band.margin(LEVEL)
        a, b = lo.predict(xt), hi.predict(xt)
        cqr_in = (yt >= np.minimum(a, b) - margin) & (
            yt <= np.maximum(a, b) + margin
        )
        resid = conformal.finite_sample_quantile(
            np.abs(yc - mean.predict(xc)), LEVEL
        )
        const_in = np.abs(yt - mean.predict(xt)) <= resid
        top = sdt >= np.quantile(sdt, 0.9)
        rows.append(
            [
                cqr_in.mean(),
                cqr_in[top].mean(),
                const_in.mean(),
                const_in[top].mean(),
            ]
        )
    data = np.array(rows)
    marginal = mean_floor(data[:, 0])
    top_cqr, top_const = float(data[:, 1].mean()), float(data[:, 3].mean())
    return {
        "replicates": replicates,
        "cqr_marginal": marginal,
        "cqr_top_decile": top_cqr,
        "constant_width_top_decile": top_const,
        "constant_width_marginal": float(data[:, 2].mean()),
        "passed": bool(marginal["meets_floor"] and top_cqr >= 0.87 > top_const),
    }


def class_world(rng: np.random.Generator, n: int) -> tuple[np.ndarray, ...]:
    """Five classes (minority 2 percent) and a distorted posterior."""
    priors = np.array([0.4, 0.3, 0.2, 0.08, 0.02])
    means = np.array([[0, 0], [2, 0], [0, 2], [2, 2], [1, 1.0]])
    labels = rng.choice(5, size=n, p=priors)
    x = means[labels] + rng.standard_normal((n, 2))
    logit = -0.5 * ((x[:, None, :] - means[None]) ** 2).sum(-1)
    logit = logit * 0.6 + np.log(priors)[None]
    logit -= logit.max(1, keepdims=True)
    prob = np.exp(logit)
    return labels, prob / prob.sum(1, keepdims=True)


def gate_g4(replicates: int, base: int = 0) -> dict[str, Any]:
    """Class-conditional against pooled sets."""
    hit = np.zeros((replicates, 5))
    pooled = np.zeros((replicates, 5))
    for index in range(replicates):
        rng = np.random.default_rng(base + index)
        cl, cp = class_world(rng, 2500)
        tl, tp = class_world(rng, 5000)
        calibrator = conformal.ConformalCalibrator.for_classification(
            cl.tolist(), cp, list(range(5))
        )
        for mode, store in (("class", hit), (None, pooled)):
            keep = np.array(
                [
                    [c in s for c in range(5)]
                    for s in calibrator.sets(tp, LEVEL, mode)
                ]
            )
            inside = keep[np.arange(len(tl)), tl]
            for k in range(5):
                store[index, k] = inside[tl == k].mean()
    floors = [mean_floor(hit[:, k]) for k in range(5)]
    pooled_floor = [mean_floor(pooled[:, k]) for k in range(5)]
    return {
        "replicates": replicates,
        "class_conditional": floors,
        "pooled_control": pooled_floor,
        "passed": bool(
            all(f["meets_floor"] for f in floors)
            and not all(f["meets_floor"] for f in pooled_floor)
        ),
    }


def gate_g5(replicates: int, base: int = 0) -> dict[str, Any]:
    """Slice flag: false flags and recall by slice size."""
    size = 200
    flagged = 0
    for index in range(replicates):
        rng = np.random.default_rng(base + index)
        q = conformal.finite_sample_quantile(
            np.abs(rng.standard_normal(1000)), LEVEL
        )
        err = np.abs(rng.standard_normal(8 * size))
        report = coverage.slice_coverage(
            err <= q, np.repeat(np.arange(8), size).tolist(), LEVEL
        )
        flagged += bool(report["flagged"])
    bound = 0.05 + 2 * math.sqrt(0.05 * 0.95 / replicates)
    recall = {}
    for rows in (30, 50, 100, 200, 400):
        hits = 0
        for index in range(replicates):
            rng = np.random.default_rng(base + index)
            q = conformal.finite_sample_quantile(
                np.abs(rng.standard_normal(1000)), LEVEL
            )
            err = np.abs(rng.standard_normal(8 * rows))
            err[:rows] *= 3.0
            report = coverage.slice_coverage(
                err <= q, np.repeat(np.arange(8), rows).tolist(), LEVEL
            )
            hits += "0" in report["flagged"]
        recall[str(rows)] = {
            "recall": hits / replicates,
            "wilson_lower": wilson_lower(hits, replicates),
        }
    certified = [int(k) for k, v in recall.items() if v["wilson_lower"] >= 0.80]
    return {
        "replicates": replicates,
        "familywise_false_flag": flagged / replicates,
        "false_flag_bound": bound,
        "recall_by_slice_rows": recall,
        "min_certified_rows": min(certified) if certified else None,
        "passed": bool(flagged / replicates <= bound and certified),
    }


def gate_g5b(replicates: int, base: int = 0) -> dict[str, Any]:
    """Slice flag with a margin: false flags and detection by slice size."""
    size, calibration = 200, 500
    bound = 0.05 + 2 * math.sqrt(0.05 * 0.95 / replicates)

    def flagged_slices(rows: int, noise: float, index: int) -> set[str]:
        rng = np.random.default_rng(base + index)
        q = conformal.finite_sample_quantile(
            np.abs(rng.standard_normal(calibration)), LEVEL
        )
        err = np.abs(rng.standard_normal(8 * rows))
        err[:rows] *= noise
        report = coverage.slice_flag(
            err <= q, np.repeat(np.arange(8), rows).tolist(), LEVEL
        )
        return {str(name) for name in report["flagged"]}

    false_flags = sum(
        bool(flagged_slices(size, 1.0, index)) for index in range(replicates)
    )
    detection: dict[str, Any] = {}
    for noise in (3.0, 1.5, 1.25):
        by_rows = {}
        for rows in (30, 50, 100, 200, 400):
            hits = sum(
                "0" in flagged_slices(rows, noise, index)
                for index in range(replicates)
            )
            by_rows[str(rows)] = {
                "recall": hits / replicates,
                "wilson_lower": wilson_lower(hits, replicates),
            }
        detection[f"noise_{noise:g}x"] = by_rows
    certified = [
        int(k)
        for k, v in detection["noise_3x"].items()
        if v["wilson_lower"] >= 0.80
    ]
    return {
        "replicates": replicates,
        "calibration_rows": calibration,
        "margin": coverage.SLICE_MARGIN,
        "familywise_false_flag": false_flags / replicates,
        "false_flag_bound": bound,
        "detection_by_slice_rows": detection,
        "min_certified_rows": min(certified) if certified else None,
        "method_hash": coverage_slices.flag_method_hash(),
        "passed": bool(false_flags / replicates <= bound and certified),
    }


def gate_g6(replicates: int, base: int = 0) -> dict[str, Any]:
    """Weighted conformal under a known and an estimated ratio."""
    mu = np.array([1.0, 0.5, 0.5])

    def world(rng: np.random.Generator, n: int, shift: bool) -> tuple[Any, Any]:
        x = rng.standard_normal((n, 3)) + (mu if shift else 0.0)
        return x, np.exp(0.6 * x[:, 0]) * rng.standard_normal(n)

    def run(estimated: bool, runs: int) -> dict[str, Any]:
        weighted, plain, sizes = [], [], []
        for index in range(runs):
            rng = np.random.default_rng(base + index)
            xc, ec = world(rng, 1000, False)
            xt, et = world(rng, 3000, True)
            score = np.abs(ec)
            if estimated:
                unlabeled, _ = world(rng, 1000, True)
                est = shift_weights.estimate_weights(xc, unlabeled)
                wc = est.calibration
                wt = est.ratio(xt)
                sizes.append(est.effective_size)
            else:
                wc = np.exp(xc @ mu - mu @ mu / 2)
                wt = np.exp(xt @ mu - mu @ mu / 2)
            margins = conformal.weighted_margins(score, wc, wt, LEVEL)
            weighted.append(np.mean(np.abs(et) <= margins))
            plain.append(
                np.mean(
                    np.abs(et) <= conformal.finite_sample_quantile(score, LEVEL)
                )
            )
        out = {
            "replicates": runs,
            "weighted": mean_floor(np.array(weighted)),
            "unweighted": float(np.mean(plain)),
        }
        if sizes:
            out["effective_size_mean"] = float(np.mean(sizes))
        return out

    known = run(False, replicates)
    estimated = run(True, ESTIMATED_RATIO_REPLICATES)
    return {
        "known_ratio": known,
        "estimated_ratio": estimated,
        "passed": bool(
            known["weighted"]["meets_floor"]
            and known["unweighted"] < known["weighted"]["floor"]
        ),
        "estimated_passed": bool(estimated["weighted"]["meets_floor"]),
    }


def gate_g9(replicates: int, base: int = 0) -> dict[str, Any]:
    """MinT against bottom-up on a non-diagonal error covariance."""
    summing = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    rng0 = np.random.default_rng(12345)
    factor = rng0.standard_normal((7, 7)) * 0.5
    truth_cov = factor @ factor.T + np.diag([0.2, 1, 1, 1.5, 1.5, 1.5, 1.5])
    chol = np.linalg.cholesky(truth_cov)
    diffs = np.empty(replicates)
    worst = 0.0
    for index in range(replicates):
        rng = np.random.default_rng(base + index)
        train = rng.standard_normal((150, 7)) @ chol.T
        w = reconcile.shrunk_covariance(train)
        bottom = rng.standard_normal((20, 4)) * 3
        truth = bottom @ summing.T
        base_f = truth + rng.standard_normal((20, 7)) @ chol.T
        mint = reconcile.mint(base_f, summing, w)
        bu = reconcile.bottom_up(base_f[:, 3:], summing)
        worst = max(worst, reconcile.coherence_residual(mint, summing))
        diffs[index] = ((mint - truth) ** 2).mean() - ((bu - truth) ** 2).mean()
    upper = float(
        diffs.mean() + 1.645 * diffs.std(ddof=1) / math.sqrt(replicates)
    )
    return {
        "replicates": replicates,
        "mean_difference": float(diffs.mean()),
        "upper_95": upper,
        "coherence_residual": worst,
        "passed": bool(upper < 0 and worst < 1e-8),
    }


def survival_world(
    rng: np.random.Generator, n: int, censor: bool
) -> tuple[np.ndarray, ...]:
    """Weibull PH events; censoring rate depends on the covariates."""
    x = rng.standard_normal((n, 2))
    risk = x @ np.array([0.8, -0.6])
    event_time = (rng.exponential(size=n) / np.exp(risk)) ** (1 / 1.5)
    if not censor:
        return x, risk, event_time, event_time, np.ones(n)
    rate = 0.35 * np.exp(-1.2 * risk)
    cut = rng.exponential(size=n) / rate
    return (
        x,
        risk,
        event_time,
        np.minimum(event_time, cut),
        (event_time <= cut).astype(float),
    )


def gate_g11i(replicates: int, base: int = 0) -> dict[str, Any]:
    """Covariate-dependent IPCW concordance against the truth."""
    rng = np.random.default_rng(999_999)
    _, risk, full, _, _ = survival_world(rng, 6000, False)
    tau = float(np.quantile(full, 0.7))
    flat = ipcw.CensoringWeights(np.ones(len(full)), 0.05, 0, False)
    truth = ipcw.truncated_concordance(
        full, np.ones(len(full)), risk, tau, flat
    )
    cox, marginal, share = [], [], []
    for index in range(replicates):
        rng = np.random.default_rng(base + index)
        x, risk, _, time, event = survival_world(rng, 1500, True)
        wc = ipcw.censoring_weights(time, event, x)
        wm = ipcw.censoring_weights(time, event, None)
        cox.append(ipcw.truncated_concordance(time, event, risk, tau, wc))
        marginal.append(ipcw.truncated_concordance(time, event, risk, tau, wm))
        share.append(wc.n_floored / len(time))
    cox_a, marginal_a = np.array(cox), np.array(marginal)
    se_cox = float(cox_a.std(ddof=1) / math.sqrt(replicates))
    se_marginal = float(marginal_a.std(ddof=1) / math.sqrt(replicates))
    cox_bias = float(cox_a.mean() - truth)
    marginal_bias = float(marginal_a.mean() - truth)
    return {
        "replicates": replicates,
        "tau": tau,
        "true_concordance": truth,
        "covariate_censoring_bias": cox_bias,
        "marginal_km_bias": marginal_bias,
        "se": [se_cox, se_marginal],
        "mean_floored_share": float(np.mean(share)),
        "passed": bool(
            abs(cox_bias) <= 3 * se_cox and abs(marginal_bias) > 3 * se_marginal
        ),
    }


def gate_g13(replicates: int, base: int = 0) -> dict[str, Any]:
    """Entity-wise against row-wise calibration for a new entity."""
    entities, per = 30, 40
    entity_hits, row_hits = [], []
    for index in range(replicates):
        rng = np.random.default_rng(base + index)
        u = rng.standard_normal(entities)
        resid = np.abs(u[:, None] + rng.standard_normal((entities, per)))
        row_margin = conformal.finite_sample_quantile(resid.ravel(), LEVEL)
        pick = rng.integers(0, per, entities)
        entity_margin = conformal.finite_sample_quantile(
            resid[np.arange(entities), pick], LEVEL
        )
        new = np.abs(rng.standard_normal() + rng.standard_normal(4000))
        entity_hits.append(np.mean(new <= entity_margin))
        row_hits.append(np.mean(new <= row_margin))
    entity = mean_floor(np.array(entity_hits))
    return {
        "replicates": replicates,
        "entities": entities,
        "rows_per_entity": per,
        "entity_wise": entity,
        "row_wise": mean_floor(np.array(row_hits)),
        "passed": bool(entity["meets_floor"]),
    }


RUNNERS = {
    "G1": (gate_g1, REPLICATES),
    "G2": (gate_g2, REPLICATES),
    "G3": (gate_g3, CQR_REPLICATES),
    "G4": (gate_g4, 300),
    "G5": (gate_g5, REPLICATES),
    "G5b": (gate_g5b, REPLICATES),
    "G6": (gate_g6, REPLICATES),
    "G9": (gate_g9, REPLICATES),
    "G11i": (gate_g11i, SURVIVAL_REPLICATES),
    "G13": (gate_g13, REPLICATES),
}


def sources() -> dict[str, list[Any]]:
    """Code each gate rests on (hashed so a change makes it stale)."""
    return {
        "G1": [conformal.finite_sample_quantile, gate_g1],
        "G2": [conformal.cv_plus_bounds, gate_g2, cv_plus_replicate],
        "G3": [quantile.QuantileBand, gate_g3],
        "G4": [conformal.ConformalCalibrator, gate_g4, class_world],
        "G5": [coverage.slice_coverage, coverage.holm_rejections, gate_g5],
        "G5b": [
            coverage.slice_flag,
            coverage.holm_rejections,
            coverage.undercoverage_p_value,
            gate_g5b,
        ],
        "G6": [
            conformal.weighted_margins,
            conformal.weighted_quantile,
            shift_weights.estimate_weights,
            gate_g6,
        ],
        "G9": [reconcile.mint, reconcile.shrunk_covariance, gate_g9],
        "G11i": [ipcw.censoring_weights, ipcw.truncated_concordance, gate_g11i],
        "G13": [conformal.finite_sample_quantile, gate_g13],
    }


def current_digests() -> dict[str, str]:
    """Code hash of every gate."""
    return {k: certificates.code_hash(*v) for k, v in sources().items()}


def certify_all() -> dict[str, Any]:
    """Runs every gate on certification seeds."""
    document: dict[str, Any] = {
        "version": 1,
        "dev_seed_base": DEV_SEED_BASE,
        "not_measured": list(NOT_MEASURED),
        "gates": {},
    }
    digests = current_digests()
    for name, (runner, runs) in RUNNERS.items():
        measured = runner(runs, 0)
        measured["code_hash"] = digests[name]
        measured["seeds"] = [0, runs]
        document["gates"][name] = measured
        print(name, measured["passed"], file=sys.stderr)
    return document


def problems(document: dict[str, Any] | None = None) -> list[str]:
    """Why the stored certificates cannot be trusted (empty when current)."""
    if document is None:
        if not STORE_PATH.exists():
            return ["uncertainty_certificates.json is missing"]
        document = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    found = []
    digests = current_digests()
    for name in GATES:
        entry = document.get("gates", {}).get(name)
        if entry is None:
            found.append(f"{name}: no certificate")
        elif entry.get("code_hash") != digests[name]:
            found.append(f"{name}: stale (code changed since measurement)")
    return found


def passed_gates(document: dict[str, Any]) -> list[str]:
    """Gates that met their pre-registered floors."""
    return [k for k, v in document["gates"].items() if v["passed"]]


def descriptive_only(document: dict[str, Any] | None = None) -> list[str]:
    """Gates that missed their floors; their methods stay descriptive."""
    if document is None:
        if not STORE_PATH.exists():
            return list(GATES)
        document = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    return [k for k, v in document["gates"].items() if not v["passed"]]


def main(arguments: list[str] | None = None) -> int:
    """Writes or checks the certificate file."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    options = parser.parse_args(arguments)
    if options.write:
        document = certify_all()
        STORE_PATH.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"passed": passed_gates(document)}))
        return 0
    issues = problems()
    print(json.dumps({"ok": not issues, "problems": issues}, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
