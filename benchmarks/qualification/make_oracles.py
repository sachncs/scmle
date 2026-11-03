"""Generates the frozen oracle files. Run offline, never in CI.

    uv run python -m benchmarks.qualification.make_oracles

Every expected value comes from scipy (and numpy for the digamma identity),
never from the code under test. R-based references (`onlineFDR`, `knockoff`)
are not produced here because R is not available; see `oracles.PENDING_R`.
Editing this file changes its hash, which fails the freshness test until the
fixtures are regenerated and the change is reviewed.
"""

import argparse
from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy import special
from scipy import stats

from benchmarks.qualification import oracles

TOLERANCE = {"rel": 1e-8}
SAMPLE_DECIMALS = 4
Case = dict[str, Any]


@dataclasses.dataclass(frozen=True)
class Family:
    """How one oracle is built.

    Attributes:
        reference: The scipy call that produces the expected values.
        inputs: Returns (case id, inputs) pairs.
        compute: Maps one case's inputs to its expected outputs.
    """

    reference: str
    inputs: Callable[[], list[tuple[str, dict[str, Any]]]]
    compute: Callable[[dict[str, Any]], dict[str, Any]]


def sample(
    seed: int, size: int, shift: float = 0.0, scale: float = 1.0
) -> list:
    """Reproducible normal sample rounded to fixed decimals."""
    rng = np.random.default_rng(seed)
    values = rng.normal(loc=shift, scale=scale, size=size)
    return [round(float(v), SAMPLE_DECIMALS) for v in values]


def numbered(prefix: str, items: Sequence[dict[str, Any]]) -> list:
    """Pairs each input with a stable id."""
    return [(f"{prefix}-{i}", dict(item)) for i, item in enumerate(items)]


def proportion_inputs(pairs: Sequence[tuple[int, int, float]]) -> list:
    """Cases for a binomial proportion interval."""
    return numbered(
        "ci",
        [{"successes": k, "n": n, "confidence": c} for k, n, c in pairs],
    )


def wilson_inputs() -> list:
    """Wilson interval cases, including both edges and n = 1."""
    return proportion_inputs(
        [
            (0, 10, 0.95),
            (1, 10, 0.95),
            (5, 10, 0.95),
            (10, 10, 0.95),
            (37, 120, 0.95),
            (37, 120, 0.99),
            (250, 1000, 0.9),
            (3, 200, 0.95),
            (190, 200, 0.95),
            (1, 1, 0.95),
        ]
    )


def interval(method: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Builds the computation for a named binomial interval method."""

    def compute(given: dict[str, Any]) -> dict[str, Any]:
        found = stats.binomtest(given["successes"], given["n"]).proportion_ci(
            confidence_level=given["confidence"], method=method
        )
        return {"low": float(found.low), "high": float(found.high)}

    return compute


def clopper_inputs() -> list:
    """Exact-interval cases."""
    return proportion_inputs(
        [
            (0, 10, 0.95),
            (10, 10, 0.95),
            (5, 10, 0.95),
            (37, 120, 0.95),
            (3, 200, 0.99),
        ]
    )


def chi2_inputs() -> list:
    """Contingency tables, 2x2 up to 3x3."""
    tables = [
        [[10, 20], [30, 25]],
        [[12, 5, 9], [7, 14, 8]],
        [[50, 30], [20, 40]],
        [[3, 9, 4], [8, 2, 11], [6, 7, 5]],
        [[40, 41], [39, 40]],
    ]
    return numbered("table", [{"table": t} for t in tables])


def chi2_compute(given: dict[str, Any]) -> dict[str, Any]:
    """Pearson chi-square without continuity correction."""
    result = stats.chi2_contingency(given["table"], correction=False)
    return {
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "dof": float(result.dof),
    }


def fisher_inputs() -> list:
    """2x2 tables with no empty margin cell that makes the ratio infinite."""
    tables = [
        [[3, 1], [1, 3]],
        [[10, 2], [3, 15]],
        [[8, 5], [7, 2]],
        [[20, 30], [25, 25]],
    ]
    return numbered("table", [{"table": t} for t in tables])


def fisher_compute(given: dict[str, Any]) -> dict[str, Any]:
    """Two-sided Fisher exact test."""
    result = stats.fisher_exact(given["table"])
    return {
        "odds_ratio": float(result.statistic),
        "pvalue": float(result.pvalue),
    }


def two_sample_inputs() -> list:
    """Pairs of samples with and without a location shift."""
    specs = [(1, 2, 0.0), (3, 4, 0.5), (5, 6, 1.0), (7, 8, 0.2)]
    return numbered(
        "pair",
        [
            {
                "x": sample(a, 28),
                "y": sample(b, 31, shift=shift, scale=1.2),
            }
            for a, b, shift in specs
        ],
    )


def ks_compute(given: dict[str, Any]) -> dict[str, Any]:
    """Two-sample Kolmogorov-Smirnov test (scipy's automatic method)."""
    result = stats.ks_2samp(given["x"], given["y"])
    return {
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
    }


def mannwhitney_compute(given: dict[str, Any]) -> dict[str, Any]:
    """Two-sided Mann-Whitney U test."""
    result = stats.mannwhitneyu(given["x"], given["y"], alternative="two-sided")
    return {
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
    }


def kruskal_inputs() -> list:
    """Three groups of different size and location."""
    specs = [(11, 0.0), (21, 0.4), (31, 0.0)]
    return numbered(
        "groups",
        [
            {
                "groups": [
                    sample(seed + i, size, shift=shift * i)
                    for i, size in enumerate((12, 15, 10))
                ]
            }
            for seed, shift in specs
        ],
    )


def kruskal_compute(given: dict[str, Any]) -> dict[str, Any]:
    """Kruskal-Wallis H test."""
    result = stats.kruskal(*given["groups"])
    return {
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
    }


def paired_inputs() -> list:
    """Pairs with weak, moderate and strong monotone association."""
    items = []
    for seed, strength in ((41, 0.0), (42, 0.5), (43, 1.5), (44, -1.0)):
        x = sample(seed, 25)
        noise = sample(seed + 100, 25)
        y = [
            round(strength * a + b, SAMPLE_DECIMALS)
            for a, b in zip(x, noise, strict=True)
        ]
        items.append({"x": x, "y": y})
    return numbered("pair", items)


def correlation(function: Callable[..., Any]) -> Callable[[dict], dict]:
    """Builds the computation for a scipy correlation test."""

    def compute(given: dict[str, Any]) -> dict[str, Any]:
        result = function(given["x"], given["y"])
        return {
            "statistic": float(result.statistic),
            "pvalue": float(result.pvalue),
        }

    return compute


def fdr_inputs() -> list:
    """P-value vectors under both adjustment methods."""
    vectors = [
        [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205],
        [0.01, 0.04, 0.03, 0.005, 0.5, 0.2, 0.9, 0.0001, 0.07],
        [0.2, 0.4, 0.6, 0.8],
    ]
    items = [{"pvalues": v, "method": m} for v in vectors for m in ("bh", "by")]
    return numbered("pvalues", items)


def fdr_compute(given: dict[str, Any]) -> dict[str, Any]:
    """Benjamini-Hochberg or Benjamini-Yekutieli adjusted p-values."""
    adjusted = stats.false_discovery_control(
        given["pvalues"], method=given["method"]
    )
    return {"adjusted": [float(v) for v in adjusted]}


def harmonic_inputs() -> list:
    """Harmonic numbers across the exact and asymptotic ranges."""
    sizes = [1, 2, 3, 10, 100, 1000, 10000, 10001, 100000, 1000000]
    return numbered("n", [{"n": n} for n in sizes])


def harmonic_compute(given: dict[str, Any]) -> dict[str, Any]:
    """H_n = digamma(n + 1) + Euler's constant, an independent route."""
    value = float(special.digamma(given["n"] + 1) + np.euler_gamma)
    return {"value": value}


FAMILIES: dict[str, Family] = {
    "chi2_contingency": Family(
        "scipy.stats.chi2_contingency(table, correction=False)",
        chi2_inputs,
        chi2_compute,
    ),
    "clopper_pearson": Family(
        "scipy.stats.binomtest(k, n).proportion_ci(method='exact')",
        clopper_inputs,
        interval("exact"),
    ),
    "fdr_control": Family(
        "scipy.stats.false_discovery_control(pvalues, method)",
        fdr_inputs,
        fdr_compute,
    ),
    "fisher_exact": Family(
        "scipy.stats.fisher_exact(table)", fisher_inputs, fisher_compute
    ),
    "harmonic_number": Family(
        "scipy.special.digamma(n + 1) + numpy.euler_gamma",
        harmonic_inputs,
        harmonic_compute,
    ),
    "kendalltau": Family(
        "scipy.stats.kendalltau(x, y)",
        paired_inputs,
        correlation(stats.kendalltau),
    ),
    "kruskal": Family(
        "scipy.stats.kruskal(*groups)", kruskal_inputs, kruskal_compute
    ),
    "ks_2samp": Family(
        "scipy.stats.ks_2samp(x, y)", two_sample_inputs, ks_compute
    ),
    "mannwhitneyu": Family(
        "scipy.stats.mannwhitneyu(x, y, alternative='two-sided')",
        two_sample_inputs,
        mannwhitney_compute,
    ),
    "pearsonr": Family(
        "scipy.stats.pearsonr(x, y)",
        paired_inputs,
        correlation(stats.pearsonr),
    ),
    "spearmanr": Family(
        "scipy.stats.spearmanr(x, y)",
        paired_inputs,
        correlation(stats.spearmanr),
    ),
    "wilson_interval": Family(
        "scipy.stats.binomtest(k, n).proportion_ci(method='wilson')",
        wilson_inputs,
        interval("wilson"),
    ),
}


def document(name: str) -> dict[str, Any]:
    """Builds the full oracle record for one family, with its hashes."""
    family = FAMILIES[name]
    cases: list[Case] = [
        {"id": case_id, "inputs": given, "expected": family.compute(given)}
        for case_id, given in family.inputs()
    ]
    record: dict[str, Any] = {
        "oracle": name,
        "reference": family.reference,
        "tool": {"numpy": np.__version__, "scipy": scipy.__version__},
        "generator": "benchmarks/qualification/make_oracles.py",
        "generator_sha256": oracles.generator_sha256(),
        "tolerance": dict(TOLERANCE),
        "cases": cases,
    }
    record["content_sha256"] = oracles.content_digest(record)
    return record


def write_all(directory: Path) -> list[Path]:
    """Writes every oracle file into `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name in sorted(FAMILIES):
        path = directory / f"{name}.json"
        text = json.dumps(
            document(name), sort_keys=True, indent=1, allow_nan=False
        )
        path.write_text(text + "\n", encoding="utf-8")
        written.append(path)
    return written


def main(arguments: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=oracles.ORACLE_DIR)
    options = parser.parse_args(arguments)
    for path in write_all(options.out):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
