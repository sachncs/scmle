"""Why CQR missed the G3 conditional-coverage floor: a filed analysis (F12).

Written on 2026-11-07 before it was run. This is not a gate and changes no
certificate: G3 stays a published miss (`cqr_top_decile` 0.8637 against the
floor 0.87 at 150 replicates, `uncertainty_certificates.json`), and a
re-registered gate would need its own pre-registration. The question here is
what the miss is made of. Two readings are open:

* sampling noise: with 150 replicates the mean of the highest-noise-decile
  coverage has a standard error of `se_150`, so 0.8637 may be within noise of
  0.87; and
* model capacity: G3 fits 40 depth-2 trees per quantile, which may be too
  little to follow a noise sd that rises fivefold, so the band is too narrow
  where the noise is largest and the single additive margin of conformal
  calibration cannot repair a conditional shortfall.

Design (same world and same calibration as `gate_g3`, development seeds
`DEV_SEED_BASE` onwards, which never overlap the certification seeds): the
gate's configuration (40 trees, depth 2) at `REPLICATES = 600`, and the
variants of more trees and depth listed in `VARIANTS`, each at `REPLICATES`.
For each: the mean and standard error of the highest-noise-decile coverage
of CQR, its marginal coverage, and the same decile for the constant-width
split-conformal control. No threshold is applied to the output; it is read
against 0.87 in the text of `docs`/CHANGELOG, and the reading is stated there.

    python -m benchmarks.qualification.cqr_analysis --write
"""

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from sklearn import ensemble

from benchmarks.qualification import uncertainty_certify as cert
from scmle.evaluation import conformal
from scmle.evaluation import quantile

OUTPUT_PATH = Path(__file__).with_name("cqr_analysis.json")
REPLICATES = 600
FLOOR = 0.87
VARIANTS: tuple[dict[str, Any], ...] = (
    {"name": "gate", "n_estimators": 40, "max_depth": 2},
    {"name": "more_trees", "n_estimators": 200, "max_depth": 2},
    {"name": "deeper", "n_estimators": 40, "max_depth": 4},
    {"name": "both", "n_estimators": 200, "max_depth": 4},
)


def draw(rng: np.random.Generator, n: int) -> tuple[np.ndarray, ...]:
    """The G3 world: noise sd rises from 0.5 to 2.5 across x."""
    x = rng.uniform(0, 1, (n, 1))
    sd = 0.5 + 2.0 * x[:, 0]
    return x, sd * rng.standard_normal(n), sd


def replicate(index: int, base: int, trees: int, depth: int) -> list[float]:
    """CQR and control coverage on one replicate (marginal, top decile)."""
    rng = np.random.default_rng(base + index)
    xf, yf, ignored = draw(rng, 1000)
    xc, yc, ignored = draw(rng, 600)
    xt, yt, sdt = draw(rng, 4000)
    kw = {"n_estimators": trees, "max_depth": depth, "random_state": index}
    lo = ensemble.GradientBoostingRegressor(
        loss="quantile", alpha=quantile.LOWER_LEVEL, **kw
    ).fit(xf, yf)
    hi = ensemble.GradientBoostingRegressor(
        loss="quantile", alpha=quantile.UPPER_LEVEL, **kw
    ).fit(xf, yf)
    mean = ensemble.GradientBoostingRegressor(**kw).fit(xf, yf)
    low_c = np.minimum(lo.predict(xc), hi.predict(xc))
    high_c = np.maximum(lo.predict(xc), hi.predict(xc))
    band = quantile.QuantileBand.calibrate(yc.tolist(), low_c, high_c)
    margin = band.margin(cert.LEVEL)
    a, b = lo.predict(xt), hi.predict(xt)
    inside = (yt >= np.minimum(a, b) - margin) & (
        yt <= np.maximum(a, b) + margin
    )
    resid = conformal.finite_sample_quantile(
        np.abs(yc - mean.predict(xc)), cert.LEVEL
    )
    plain = np.abs(yt - mean.predict(xt)) <= resid
    top = sdt >= np.quantile(sdt, 0.9)
    return [
        float(inside.mean()),
        float(inside[top].mean()),
        float(plain[top].mean()),
    ]


def summarize(values: np.ndarray) -> dict[str, float]:
    """Mean and standard error of a column of replicate results."""
    return {
        "mean": float(values.mean()),
        "se": float(values.std(ddof=1) / np.sqrt(len(values))),
    }


def analyse(replicates: int = REPLICATES) -> dict[str, Any]:
    """Runs every variant on development seeds."""
    started = time.monotonic()
    report: dict[str, Any] = {
        "registered": "2026-11-07",
        "note": "analysis of a published miss; no gate, no certificate",
        "g3_floor": FLOOR,
        "replicates": replicates,
        "seeds": [cert.DEV_SEED_BASE, cert.DEV_SEED_BASE + replicates],
        "variants": {},
    }
    for variant in VARIANTS:
        data = np.array(
            [
                replicate(
                    i,
                    cert.DEV_SEED_BASE,
                    variant["n_estimators"],
                    variant["max_depth"],
                )
                for i in range(replicates)
            ]
        )
        report["variants"][variant["name"]] = {
            "n_estimators": variant["n_estimators"],
            "max_depth": variant["max_depth"],
            "cqr_marginal": summarize(data[:, 0]),
            "cqr_top_decile": summarize(data[:, 1]),
            "constant_width_top_decile": summarize(data[:, 2]),
        }
    report["seconds"] = round(time.monotonic() - started, 1)
    return report


def main(arguments: list[str] | None = None) -> int:
    """Writes the analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", required=True)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    options = parser.parse_args(arguments)
    document = analyse(options.replicates)
    OUTPUT_PATH.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(document["variants"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
