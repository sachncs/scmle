"""The Phase 1 gate: every harness piece, exercised once, with controls.

`run_smoke` returns a named list of checks. Two of them are negative
controls: a corrupted mask must be caught, and a deliberately liberal test
must fall outside the size band. A passing smoke run therefore shows that
the harness can fail, not only that it did not.
"""

from collections.abc import Callable
import dataclasses
from typing import Any

import pandas as pd
from scipy import stats
from sklearn.datasets import load_diabetes

from benchmarks.qualification import corpora
from benchmarks.qualification import inject
from benchmarks.qualification import ledger_checks
from benchmarks.qualification import oracles
from benchmarks.qualification import simulate
from benchmarks.qualification import standins
from scmle.contracts import qualification
from scmle.data.detectors import certificates
from scmle.data.ledger import budget

ALPHA = 0.05
SMOKE_ROWS = 300
ORACLE_REL_TOL = 1e-8
FULL_RUNS = 20000
POWER_RUNS = 400
FDR_RUNS = 300
SABOTAGES = (None, "size")


@dataclasses.dataclass(frozen=True)
class SmokeCheck:
    """One named check and what it found."""

    name: str
    ok: bool
    detail: str


@dataclasses.dataclass(frozen=True)
class SmokeReport:
    """All checks of a smoke run."""

    checks: tuple[SmokeCheck, ...]

    @property
    def ok(self) -> bool:
        """Whether every check passed."""
        return all(check.ok for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return {
            "ok": self.ok,
            "checks": [dataclasses.asdict(check) for check in self.checks],
        }


def injection_cases(
    frame: pd.DataFrame,
) -> dict[str, Callable[[], inject.Injection]]:
    """One call per injector, on the demo table."""
    return {
        "label_flip": lambda: inject.flip_labels(frame, "target", 0.1, 0),
        "missing_mcar": lambda: inject.inject_missing(
            frame, ["mean radius"], 0.1, 0
        ),
        "missing_mar": lambda: inject.inject_missing(
            frame,
            ["mean radius"],
            0.1,
            0,
            mechanism="mar",
            driver="mean texture",
        ),
        "missing_mnar": lambda: inject.inject_missing(
            frame, ["mean radius"], 0.1, 0, mechanism="mnar"
        ),
        "sentinel": lambda: inject.inject_sentinels(
            frame, ["mean radius"], 0.1, 0
        ),
        "duplicate": lambda: inject.inject_duplicates(frame, 0.1, 0),
        "duplicate_near": lambda: inject.inject_duplicates(
            frame, 0.1, 0, near=True
        ),
        "outlier": lambda: inject.inject_outliers(
            frame, ["mean radius"], 0.05, 0
        ),
        "leak": lambda: inject.inject_leak(frame, "target", "copy", 0),
        "covariate_shift": lambda: inject.shift_covariates(
            frame, ["mean radius"], 0.3, 1.0, 0
        ),
    }


def mismatched_cells(original: pd.DataFrame, planted: inject.Injection) -> int:
    """Cells where the mask disagrees with the before/after diff."""
    diff = inject.changed_cells(original, planted.frame)
    return int((planted.mask.to_numpy() != diff.to_numpy()).sum())


def check_masks() -> list[SmokeCheck]:
    """Exactness of every mask, plus a control that corrupts one."""
    frame = standins.demo_frame().head(SMOKE_ROWS).reset_index(drop=True)
    total = 0
    cases = injection_cases(frame)
    for build in cases.values():
        total += mismatched_cells(frame, build())
    corrupted = inject.flip_labels(frame, "target", 0.1, 0)
    mask = corrupted.mask.copy()
    mask.iloc[0, 0] = not bool(mask.iloc[0, 0])
    tampered = dataclasses.replace(corrupted, mask=mask)
    caught = mismatched_cells(frame, tampered)
    return [
        SmokeCheck(
            "injector masks are exact",
            total == 0,
            f"{total} mismatched cells across {len(cases)} injectors",
        ),
        SmokeCheck(
            "mask checker detects a corrupted mask",
            caught == 1,
            f"{caught} mismatched cell found after corrupting one",
        ),
    ]


def pearson_test(liberal: bool) -> Callable[[simulate.World], float]:
    """P-value of a correlation test, optionally made too liberal."""

    def test(world: simulate.World) -> float:
        value = float(
            stats.pearsonr(world.features["bmi"], world.outcome).pvalue
        )
        return value / 2.0 if liberal else value

    return test


def check_size(runs: int, seed: int, sabotage: str | None) -> list[SmokeCheck]:
    """Empirical size of a valid test, and of a deliberately biased one."""
    data = load_diabetes(as_frame=True)

    def measure(liberal: bool) -> simulate.RejectionRate:
        worlds = simulate.null_worlds(data.data, data.target, runs, seed)
        return simulate.rejection_rate(pearson_test(liberal), worlds, ALPHA)

    result = measure(liberal=sabotage == "size")
    biased = measure(liberal=True)
    low, high = simulate.size_band(ALPHA, runs)
    interval = result.interval
    return [
        SmokeCheck(
            "null size inside the band",
            result.within_size_band,
            f"size {result.rate:.4f} over {runs} worlds, band "
            f"[{low:.4f}, {high:.4f}], Wilson [{interval[0]:.4f}, "
            f"{interval[1]:.4f}]",
        ),
        SmokeCheck(
            "biased test falls outside the band",
            not biased.within_size_band,
            f"liberal size {biased.rate:.4f}",
        ),
    ]


def check_oracles() -> SmokeCheck:
    """Oracle integrity and agreement of our statistics with scipy."""
    compared = 0
    try:
        for name in oracles.NAMES:
            document = oracles.load_oracle(name)
            if document.generator_sha256 != oracles.generator_sha256():
                raise oracles.OracleError(f"{name}: generator hash is stale")
        for case in oracles.load_oracle("wilson_interval").cases:
            given = case["inputs"]
            low, high = certificates.wilson_interval(
                given["successes"], given["n"], given["confidence"]
            )
            oracles.assert_agrees(
                {"low": low, "high": high},
                case["expected"],
                label=case["id"],
                rel_tol=ORACLE_REL_TOL,
                abs_tol=1e-15,
            )
            compared += 1
        for case in oracles.load_oracle("harmonic_number").cases:
            oracles.assert_agrees(
                budget.harmonic(case["inputs"]["n"]),
                case["expected"]["value"],
                label=case["id"],
                rel_tol=ORACLE_REL_TOL,
            )
            compared += 1
    except (oracles.OracleError, AssertionError) as error:
        return SmokeCheck("oracles load and agree", False, str(error))
    return SmokeCheck(
        "oracles load and agree",
        True,
        f"{len(oracles.NAMES)} files verified, {compared} cases agree with "
        "scipy; R oracles pending: " + ", ".join(sorted(oracles.PENDING_R)),
    )


def check_corpora() -> SmokeCheck:
    """The manifest validates and no entry claims more than it has."""
    try:
        manifest = corpora.load_manifest()
    except corpora.CorpusError as error:
        return SmokeCheck("corpus manifest is valid", False, str(error))
    verified = sum(c.status == "verified" for c in manifest)
    return SmokeCheck(
        "corpus manifest is valid",
        True,
        f"{len(manifest)} corpora, {verified} verified, "
        f"{len(manifest) - verified} candidate",
    )


def check_certificates() -> list[SmokeCheck]:
    """Stand-in certification, reproducibility and staleness."""
    first = standins.certify_all()
    second = standins.certify_all()
    lines = []
    for item in first.certificates:
        outcome = item.outcomes[0]
        cert = outcome.certificate
        lines.append(
            f"{item.detector_id}: recall {cert.recall:.3f} "
            f"(lower {outcome.recall_interval[0]:.3f}), false alarms "
            f"{cert.false_alarm_rate:.4f} "
            f"(upper {outcome.false_alarm_interval[1]:.4f}), "
            f"{'pass' if outcome.passed else 'FAIL'}"
        )
    all_passed = all(
        o.passed for item in first.certificates for o in item.outcomes
    )
    reproducible = first.certificates == second.certificates
    name = "standin-duplicate"
    standin = standins.STANDINS[name]
    altered = "0" * 64
    stale = first.state(name, altered) is certificates.CertificateState.STALE
    demoted = (
        first.status(name, altered, standin.plan.defect_type)
        is certificates.GateStatus.DESCRIPTIVE
        and certificates.demote(
            qualification.Decision.BLOCK, certificates.GateStatus.DESCRIPTIVE
        )
        is qualification.Decision.WARN
    )
    try:
        first.mark_gating(name, altered, standin.plan.defect_type)
        refused = False
    except certificates.StaleCertificateError:
        refused = True
    return [
        SmokeCheck(
            "stand-in certificates",
            all_passed and reproducible,
            "; ".join(lines) + f"; reproducible={reproducible}",
        ),
        SmokeCheck(
            "stale certificate demotes gating",
            stale and demoted and refused,
            f"stale={stale}, descriptive={demoted}, gating refused={refused}",
        ),
    ]


def check_ledger(seed: int) -> list[SmokeCheck]:
    """The claim ledger: vectorized tester, uniform p, FDR bound and power."""
    return [
        SmokeCheck(f.name, f.ok, f.detail)
        for f in ledger_checks.smoke_findings(seed)
    ]


def check_power_and_fdr(seed: int) -> list[SmokeCheck]:
    """Full mode only: a power curve and a pipeline false-discovery rate."""
    data = load_diabetes(as_frame=True)
    curve = simulate.power_curve(
        pearson_test(liberal=False),
        data.data,
        "bmi",
        simulate.EFFECT_GRID,
        POWER_RUNS,
        ALPHA,
        seed,
    )
    rates = [curve[size].rate for size in simulate.EFFECT_GRID]
    proportions = []
    for run in range(FDR_RUNS):
        effects = simulate.random_effects(
            list(data.data.columns), 3, 0.3, seed, index=run
        )
        world = simulate.planted_world(data.data, effects, seed, index=run)
        # A conditional test: the planted truth is a zero coefficient given
        # the other columns, which a marginal test does not address.
        pvalues = simulate.coefficient_pvalues(world).tolist()
        adjusted = stats.false_discovery_control(pvalues, method="bh")
        rejected = {
            c
            for c, p in zip(world.features.columns, adjusted, strict=True)
            if p <= ALPHA
        }
        proportions.append(simulate.false_discovery_proportion(rejected, world))
    summary = simulate.summarize_false_discoveries(proportions)
    return [
        SmokeCheck(
            "power rises with effect size",
            rates == sorted(rates),
            ", ".join(
                f"{s}: {r:.3f}"
                for s, r in zip(simulate.EFFECT_GRID, rates, strict=True)
            ),
        ),
        SmokeCheck(
            "pipeline false-discovery rate",
            summary.supports(ALPHA),
            f"mean {summary.mean:.4f}, upper {summary.upper:.4f} over "
            f"{summary.runs} worlds (BH at {ALPHA})",
        ),
    ]


def run_smoke(
    runs: int = 2000,
    seed: int = 0,
    full: bool = False,
    sabotage: str | None = None,
) -> SmokeReport:
    """Runs every harness check once.

    Args:
        runs: Null worlds for the size check (the plan's gate uses 2000).
        seed: Run seed.
        full: Also run the power curve and pipeline FDR (nightly).
        sabotage: `size` swaps the valid test for a liberal one, so the size
            check must fail; used to show the gate can fail.

    Returns:
        The report; `ok` is True only if every check passed.

    Raises:
        ValueError: On an unknown sabotage name.
    """
    if sabotage not in SABOTAGES:
        raise ValueError(f"sabotage must be one of {SABOTAGES}")
    qualification.count(runs, "runs", minimum=1)
    checks = check_masks()
    checks += check_size(FULL_RUNS if full else runs, seed, sabotage)
    checks.append(check_oracles())
    checks.append(check_corpora())
    checks += check_certificates()
    checks += check_ledger(seed)
    if full:
        checks += check_power_and_fdr(seed)
    return SmokeReport(tuple(checks))
