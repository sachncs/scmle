"""Runs a chartered causal analysis end to end (F10).

Order, as F10 specifies: select (a function of the charter), diagnose the
design and fit frozen nuisances on `X`, estimate on `C` once, run the
pre-registered falsification battery and the replication, assess sensitivity,
and tier the claim. The function receives `X` and `C` as frames; opening `C`
through the custody vault exactly once is the caller's job (the confirmation
function), and nothing in this module reads `C` before the estimate step or
refits anything on it.

Outputs are five JSON-ready documents named as the feature file lists them.
A claim is `not_identified` (and no estimate is printed) when no strategy is
eligible, when the overlap gate fails with no trimming declared, when
trimming leaves too few rows, or when the regression-discontinuity density
check rejects on `X`.

Single-treated-unit synthetic control has nothing to hold out: it runs on `X`
and its tier is capped at `attested`.
"""

from collections.abc import Mapping
import dataclasses
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from scmle.data.admissibility import dag as dag_module
from scmle.data.charter import causal
from scmle.data.identification import aipw
from scmle.data.identification import battery
from scmle.data.identification import did
from scmle.data.identification import iv
from scmle.data.identification import rd
from scmle.data.identification import select
from scmle.data.identification import sensitivity
from scmle.data.identification import synth
from scmle.data.identification import tier
from scmle.data.ledger import budget as budget_module

DOCUMENTS = (
    "identification_plan.json",
    "identification_report.json",
    "falsification_report.json",
    "sensitivity_report.json",
    "causal_claims.json",
)
INTERVAL_GATES = {
    "backdoor_aipw": "coverage_aipw",
    "randomized_diff": "coverage_randomized",
    "did": "coverage_did",
    "iv": "coverage_iv",
    "rd": "coverage_rd",
    "synth": "size_synth",
}
DESIGN_GATES = {
    "backdoor_aipw": ("overlap",),
    "iv": ("iv_strength",),
    "rd": ("density_rd",),
}
BATTERY_GATES = {
    "backdoor_aipw": "battery_nco",
    "randomized_diff": "battery_nco",
    "did": "battery_pretrend",
    "iv": "battery_iv",
    "rd": "battery_rd",
    "synth": "battery_synth",
}
SENSITIVITY_GATES = {
    "backdoor_aipw": "sensitivity_bound",
    "did": "sensitivity_did",
}
MIN_STRATUM_ROWS = 50
LIMITS = (
    "confirmed means: under the attested assumptions the interval has its "
    "measured coverage and the battery would have flagged a violation of the "
    "certified size; it does not say the assumptions are true, and a "
    "confounder the negative controls do not share passes undetected"
)


@dataclasses.dataclass(frozen=True, slots=True)
class Est:
    """A point estimate with its standard error and interval."""

    estimate: float
    se: float
    low: float
    high: float
    n: int
    detail: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def excludes_zero(self) -> bool:
        """Whether the interval excludes zero."""
        return self.low > 0 or self.high < 0


@dataclasses.dataclass(frozen=True)
class Outcome:
    """The result of a causal run.

    Attributes:
        documents: File name to JSON-ready content (`DOCUMENTS`).
        decision: The tier decision.
        hashes: File name to SHA-256 of the canonical JSON.
    """

    documents: dict[str, dict[str, Any]]
    decision: tier.TierDecision
    hashes: dict[str, str]


def digest(document: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of a document."""
    text = json.dumps(
        document, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stratum_frames(frame: pd.DataFrame, column: str) -> dict[str, pd.DataFrame]:
    """Rows of each stratum large enough to estimate in."""
    out = {}
    for level, part in frame.groupby(column):
        if len(part) >= MIN_STRATUM_ROWS:
            out[str(level)] = part
    return out


def wald_from(est: Est, name: str, kind: str) -> battery.Test | None:
    """A Wald test of a placebo estimate, or None without a usable se."""
    if not np.isfinite(est.se) or est.se <= 0:
        return None
    return battery.wald_test(name, kind, est.estimate, est.se)


class Adapter:
    """What a strategy needs: design check, estimator, controls, extras.

    Subclasses fix `name` and implement the hooks. `prepare` runs on `X`
    only; every later method reads `C`.
    """

    name = ""

    def __init__(
        self,
        spec: causal.CausalSpec,
        x: pd.DataFrame,
        seed: int,
        learner: str,
        alpha_falsify: float,
    ) -> None:
        """Stores the settings of the strategy."""
        self.spec = spec
        self.x = x
        self.seed = seed
        self.learner = learner
        self.alpha_falsify = alpha_falsify
        self.design: dict[str, Any] = {}
        self.refusal: str | None = None
        self.weak = False
        self.single_unit = False
        self.notes: list[str] = []

    def prepare(self) -> None:
        """Design diagnostics and frozen quantities, from `X` only."""

    def estimate(self, c: pd.DataFrame, level: float) -> Est:
        """The primary estimate on `C`."""
        raise NotImplementedError

    def estimate_for(
        self, c: pd.DataFrame, level: float, outcome: str | None = None
    ) -> Est:
        """The same estimator on `C` with another outcome column."""
        raise NotImplementedError

    def controls(self, c: pd.DataFrame, level: float) -> list[battery.Test]:
        """Falsification tests on `C` (negative controls and placebos)."""
        return []

    def sensitivity(self, c: pd.DataFrame, level: float) -> dict[str, Any]:
        """Sensitivity document; `holds` is None when unavailable."""
        return {
            "holds": None,
            "note": "no sensitivity analysis for this strategy",
        }


class BackdoorAdapter(Adapter):
    """Cross-fitted-free AIPW with frozen nuisances."""

    name = "backdoor_aipw"

    def __init__(self, *args: Any) -> None:
        """Stores the settings of the strategy."""
        super().__init__(*args)
        self.covariates = causal.covariates_of(self.spec.assumptions) or ()
        self.nuisances: aipw.Nuisances | None = None

    def prepare(self) -> None:
        """Design diagnostics and frozen quantities from `X` only."""
        spec = self.spec
        self.nuisances = aipw.fit_nuisances(
            self.x,
            self.covariates,
            spec.treatment,
            spec.outcome,
            self.seed,
            self.learner,
        )
        e, _, _ = self.nuisances.predict(self.x)
        t = aipw.binary(self.x[spec.treatment], spec.treatment)
        matrix = self.nuisances.scaler.transform(
            self.nuisances.design.matrix(self.x)
        )
        self.design = {
            "overlap_on_x": aipw.overlap_report(e, t, matrix),
            "nuisance_digest": self.nuisances.digest,
            "learner": self.learner,
        }
        passed = self.design["overlap_on_x"]["passed"]
        if not passed and spec.trim <= 0:
            self.refusal = (
                "the overlap gate failed on X and the charter declares no "
                "trimming rule"
            )

    def estimate(self, c: pd.DataFrame, level: float) -> Est:
        """The primary estimate on `C`."""
        assert self.nuisances is not None
        spec = self.spec
        result = aipw.estimate(
            self.nuisances,
            c,
            spec.treatment,
            spec.outcome,
            spec.estimand,
            level,
            spec.trim,
            spec.min_share,
        )
        self.design["overlap_on_c"] = result.overlap
        self.design["population"] = result.population
        return Est(
            result.estimate,
            result.se,
            result.low,
            result.high,
            result.n,
            {"population": result.population},
        )

    def estimate_for(
        self, c: pd.DataFrame, level: float, outcome: str | None = None
    ) -> Est:
        """The same estimator on `C` with another outcome column."""
        spec = self.spec
        nuisances = aipw.fit_nuisances(
            self.x,
            self.covariates,
            spec.treatment,
            outcome or spec.outcome,
            self.seed,
            self.learner,
        )
        result = aipw.estimate(
            nuisances,
            c,
            spec.treatment,
            outcome or spec.outcome,
            spec.estimand,
            level,
            spec.trim,
            spec.min_share,
        )
        return Est(
            result.estimate, result.se, result.low, result.high, result.n
        )

    def controls(self, c: pd.DataFrame, level: float) -> list[battery.Test]:
        """Falsification tests on `C`."""
        spec = self.spec
        tests = []
        for control in spec.negative_controls:
            if control.kind == "outcome":
                est = self.estimate_for(c, level, control.variable)
                name = f"negative_outcome:{control.variable}"
            else:
                nuisances = aipw.fit_nuisances(
                    self.x,
                    self.covariates,
                    control.variable,
                    spec.outcome,
                    self.seed,
                    self.learner,
                )
                found = aipw.estimate(
                    nuisances,
                    c,
                    control.variable,
                    spec.outcome,
                    spec.estimand,
                    level,
                    spec.trim,
                    spec.min_share,
                )
                est = Est(
                    found.estimate, found.se, found.low, found.high, found.n
                )
                name = f"negative_exposure:{control.variable}"
            test = wald_from(est, name, f"negative_{control.kind}")
            if test is not None:
                tests.append(test)
        return tests

    def sensitivity(self, c: pd.DataFrame, level: float) -> dict[str, Any]:
        """Sensitivity of the estimate to unmeasured confounding."""
        spec = self.spec
        if spec.sensitivity is None:
            return {
                "holds": None,
                "note": "no sensitivity block in the charter",
            }
        if not self.covariates:
            return {"holds": None, "note": "benchmarking needs a covariate"}
        assert self.nuisances is not None
        design = self.nuisances.design
        matrix = design.matrix(c)
        names = [f"x{j}" for j in range(matrix.shape[1])]
        frame = pd.DataFrame(matrix, columns=names)
        frame["t"] = c[spec.treatment].to_numpy(dtype=float)
        frame["y"] = c[spec.outcome].to_numpy(dtype=float)
        report = sensitivity.assess(
            frame,
            "t",
            "y",
            tuple(names),
            spec.sensitivity.benchmark,
            spec.sensitivity.strength,
            level,
        )
        out = report.to_dict()
        out["scale"] = (
            "linear-model coefficient of the treatment on the rows of C; a "
            "description of how fragile that summary is, not of the AIPW "
            "estimate itself"
        )
        return out


class RandomizedAdapter(Adapter):
    """Difference in means under a recorded assignment mechanism."""

    name = "randomized_diff"

    def estimate(self, c: pd.DataFrame, level: float) -> Est:
        """The primary estimate on `C`."""
        spec = self.spec
        r = aipw.difference_in_means(c, spec.treatment, spec.outcome, level)
        return Est(r.estimate, r.se, r.low, r.high, r.n)

    def estimate_for(
        self, c: pd.DataFrame, level: float, outcome: str | None = None
    ) -> Est:
        """The same estimator on `C` with another outcome column."""
        r = aipw.difference_in_means(
            c, self.spec.treatment, outcome or self.spec.outcome, level
        )
        return Est(r.estimate, r.se, r.low, r.high, r.n)

    def controls(self, c: pd.DataFrame, level: float) -> list[battery.Test]:
        """Falsification tests on `C`."""
        tests = []
        for control in self.spec.negative_controls:
            if control.kind != "outcome":
                continue
            est = self.estimate_for(c, level, control.variable)
            test = wald_from(
                est, f"negative_outcome:{control.variable}", "negative_outcome"
            )
            if test is not None:
                tests.append(test)
        return tests

    def sensitivity(self, c: pd.DataFrame, level: float) -> dict[str, Any]:
        """Sensitivity of the estimate to unmeasured confounding."""
        return {
            "holds": True,
            "note": "randomized assignment: no unmeasured confounding to bound",
            "applicable": False,
        }


class DidAdapter(Adapter):
    """Group-time difference in differences."""

    name = "did"

    def columns(self) -> tuple[str, str, str]:
        """The panel column names."""
        s = self.spec
        assert s.unit and s.time and s.adoption
        return s.unit, s.time, s.adoption

    def prepare(self) -> None:
        """Design diagnostics and frozen quantities from `X` only."""
        unit, time, adoption = self.columns()
        fit = did.estimate(self.x, unit, time, adoption, self.spec.outcome)
        self.design = {
            "groups": fit.groups,
            "n_never_treated": fit.n_never,
            "events": list(fit.events),
            "pretrend_p_on_x": fit.pretrend_p,
            "pre_periods_support_sensitivity": fit.max_pre_change is not None,
        }

    def fit(
        self, c: pd.DataFrame, level: float, outcome: str | None = None
    ) -> did.DidResult:
        """Fits the estimator on `c` or `frame`."""
        unit, time, adoption = self.columns()
        return did.estimate(
            c, unit, time, adoption, outcome or self.spec.outcome, level
        )

    def estimate(self, c: pd.DataFrame, level: float) -> Est:
        """The primary estimate on `C`."""
        fit = self.fit(c, level)
        return Est(
            fit.estimate,
            fit.se,
            fit.low,
            fit.high,
            len(c),
            {
                "pretrend_p": fit.pretrend_p,
                "event_study": dict(
                    zip(fit.events, fit.coefficients, strict=True)
                ),
            },
        )

    def estimate_for(
        self, c: pd.DataFrame, level: float, outcome: str | None = None
    ) -> Est:
        """The same estimator on `C` with another outcome column."""
        fit = self.fit(c, level, outcome)
        return Est(fit.estimate, fit.se, fit.low, fit.high, len(c))

    def controls(self, c: pd.DataFrame, level: float) -> list[battery.Test]:
        """Falsification tests on `C`."""
        tests = []
        fit = self.fit(c, level)
        if fit.pretrend_p is not None:
            tests.append(
                battery.Test("pretrend_wald", "placebo", fit.pretrend_p)
            )
        for control in self.spec.negative_controls:
            if control.kind == "outcome":
                est = self.estimate_for(c, level, control.variable)
                test = wald_from(
                    est,
                    f"negative_outcome:{control.variable}",
                    "negative_outcome",
                )
                if test is not None:
                    tests.append(test)
        return tests

    def sensitivity(self, c: pd.DataFrame, level: float) -> dict[str, Any]:
        """Sensitivity of the estimate to unmeasured confounding."""
        spec = self.spec
        if spec.sensitivity is None:
            return {
                "holds": None,
                "note": "no sensitivity block in the charter",
            }
        fit = self.fit(c, level)
        if fit.max_pre_change is None:
            return {"holds": None, "note": "fewer than two pre-periods"}
        m = spec.sensitivity.m_max
        low, high = fit.sensitivity(m)
        return {
            "holds": bool(low > 0 or high < 0),
            "m_max": m,
            "low": low,
            "high": high,
            "breakdown_m": fit.breakdown(),
            "max_pre_change": fit.max_pre_change,
            "note": "simplified relative-magnitudes bound, not HonestDiD",
        }


class IvAdapter(Adapter):
    """Just-identified instrumental variables."""

    name = "iv"

    def controls_columns(self) -> tuple[str, ...]:
        """Covariates used as IV controls."""
        return causal.covariates_of(self.spec.assumptions) or ()

    def fit(
        self, frame: pd.DataFrame, level: float, outcome: str | None = None
    ) -> iv.IvResult:
        """Fits the estimator on `c` or `frame`."""
        spec = self.spec
        assert spec.instrument
        return iv.estimate(
            frame,
            spec.treatment,
            outcome or spec.outcome,
            spec.instrument,
            self.controls_columns(),
            level,
        )

    def prepare(self) -> None:
        """Design diagnostics and frozen quantities from `X` only."""
        fit = self.fit(self.x, 0.95)
        self.weak = fit.weak
        self.design = {
            "first_stage_F_on_x": fit.f_stat,
            "threshold": iv.F_THRESHOLD,
            "weak_on_x": fit.weak,
            "overidentification": (
                "not applicable: one instrument, just identified"
            ),
        }

    def estimate(self, c: pd.DataFrame, level: float) -> Est:
        """The primary estimate on `C`."""
        fit = self.fit(c, level)
        self.weak = self.weak or fit.weak
        self.design["first_stage_F_on_c"] = fit.f_stat
        return Est(
            fit.estimate,
            fit.se,
            fit.low,
            fit.high,
            fit.n,
            {
                "headline": fit.headline,
                "ar_kind": fit.ar_kind,
                "estimand": "LATE",
            },
        )

    def estimate_for(
        self, c: pd.DataFrame, level: float, outcome: str | None = None
    ) -> Est:
        """The same estimator on `C` with another outcome column."""
        fit = self.fit(c, level, outcome)
        return Est(fit.estimate, fit.se, fit.wald_low, fit.wald_high, fit.n)

    def controls(self, c: pd.DataFrame, level: float) -> list[battery.Test]:
        """Falsification tests on `C`."""
        tests = []
        for control in self.spec.negative_controls:
            if control.kind == "outcome":
                est = self.estimate_for(c, level, control.variable)
                test = wald_from(
                    est,
                    f"negative_outcome:{control.variable}",
                    "negative_outcome",
                )
                if test is not None:
                    tests.append(test)
        return tests


class RdAdapter(Adapter):
    """Sharp regression discontinuity with a bandwidth frozen on `X`."""

    name = "rd"

    def __init__(self, *args: Any) -> None:
        """Stores the settings of the strategy."""
        super().__init__(*args)
        self.bandwidth = 0.0

    def balance_columns(self) -> tuple[str, ...]:
        """Predetermined covariates tested for a jump."""
        return tuple(
            c.variable
            for c in self.spec.negative_controls
            if c.kind == "outcome"
        )

    def prepare(self) -> None:
        """Design diagnostics and frozen quantities from `X` only."""
        spec = self.spec
        assert spec.running_variable and spec.cutoff is not None
        fit = rd.estimate(
            self.x,
            spec.running_variable,
            spec.outcome,
            spec.cutoff,
            covariates=self.balance_columns(),
        )
        self.bandwidth = fit.bandwidth
        self.design = {
            "bandwidth": fit.bandwidth,
            "density_p_on_x": fit.density_p,
            "alpha_density": self.alpha_falsify,
            "n_left_in_bandwidth": fit.n_left,
            "n_right_in_bandwidth": fit.n_right,
        }
        if fit.density_p < self.alpha_falsify:
            self.refusal = (
                "the density check rejects at the cutoff on X: possible "
                "manipulation of the running variable"
            )

    def fit(
        self,
        c: pd.DataFrame,
        level: float,
        outcome: str | None = None,
        covariates: tuple[str, ...] = (),
    ) -> rd.RdResult:
        """Fits the estimator on `c` or `frame`."""
        spec = self.spec
        assert spec.running_variable and spec.cutoff is not None
        return rd.estimate(
            c,
            spec.running_variable,
            outcome or spec.outcome,
            spec.cutoff,
            level,
            self.bandwidth,
            covariates,
        )

    def estimate(self, c: pd.DataFrame, level: float) -> Est:
        """The primary estimate on `C`."""
        fit = self.fit(c, level)
        return Est(
            fit.bias_corrected,
            fit.se,
            fit.low,
            fit.high,
            fit.n_left + fit.n_right,
            {"conventional": fit.estimate, "bandwidth": fit.bandwidth},
        )

    def estimate_for(
        self, c: pd.DataFrame, level: float, outcome: str | None = None
    ) -> Est:
        """The same estimator on `C` with another outcome column."""
        fit = self.fit(c, level, outcome)
        return Est(
            fit.bias_corrected,
            fit.se,
            fit.low,
            fit.high,
            fit.n_left + fit.n_right,
        )

    def controls(self, c: pd.DataFrame, level: float) -> list[battery.Test]:
        """Falsification tests on `C`."""
        fit = self.fit(c, level, covariates=self.balance_columns())
        tests = [
            battery.Test(
                f"placebo_cutoff:{name}",
                "placebo",
                float(row["p"]),
                float(row["bias_corrected"]),
                float(row["se"]),
            )
            for name, row in fit.placebos.items()
        ]
        tests.extend(
            battery.Test(
                f"covariate_balance:{name}",
                "negative_outcome",
                float(row["p"]),
                float(row["bias_corrected"]),
                float(row["se"]),
            )
            for name, row in fit.balance.items()
        )
        return tests

    def sensitivity(self, c: pd.DataFrame, level: float) -> dict[str, Any]:
        """Sensitivity of the estimate to unmeasured confounding."""
        fit = self.fit(c, level)
        stable = [
            (row["low"] <= 0 <= row["high"]) == (fit.low <= 0 <= fit.high)
            for row in fit.sensitivity.values()
        ]
        return {
            "holds": True,
            "applicable": False,
            "note": "RD has no unmeasured-confounding bound; the bandwidth "
            "sensitivity is descriptive",
            "bandwidth_sensitivity": fit.sensitivity,
            "same_conclusion_at_all_multiples": bool(all(stable)),
        }


class SynthAdapter(Adapter):
    """Synthetic control with one treated unit; runs on `X` alone."""

    name = "synth"

    def __init__(self, *args: Any) -> None:
        """Stores the settings of the strategy."""
        super().__init__(*args)
        self.single_unit = True

    def estimate(self, c: pd.DataFrame, level: float) -> Est:
        """The primary estimate on `C`."""
        spec = self.spec
        assert spec.unit and spec.time
        frame = self.x
        flag = frame[spec.treatment].to_numpy(dtype=float) > 0
        treated_units = frame.loc[flag, spec.unit].unique()
        if len(treated_units) != 1:
            raise ValueError("synthetic control needs exactly one treated unit")
        first = frame.loc[flag, spec.time].min()
        result = synth.estimate(
            frame,
            spec.unit,
            spec.time,
            spec.outcome,
            treated_units[0],
            first,
            1 - level,
        )
        self.design = {
            "donors": result.donors,
            "min_attainable_p": result.min_p,
            "pre_rmspe": result.pre_rmspe,
            "weights": result.weights,
        }
        return Est(
            result.effect,
            float("nan"),
            float("nan"),
            float("nan"),
            len(frame),
            {"p_value": result.p_value, "ratio": result.ratio},
        )

    def estimate_for(
        self, c: pd.DataFrame, level: float, outcome: str | None = None
    ) -> Est:
        """The same estimator on `C` with another outcome column."""
        raise NotImplementedError


ADAPTERS: dict[str, type[Adapter]] = {
    "backdoor_aipw": BackdoorAdapter,
    "randomized_diff": RandomizedAdapter,
    "did": DidAdapter,
    "iv": IvAdapter,
    "rd": RdAdapter,
    "synth": SynthAdapter,
}


def refusal_outcome(
    plan: select.Plan,
    spec: causal.CausalSpec,
    reason: str | None = None,
) -> Outcome:
    """The five documents of a claim that is not identified."""
    report = dict(select.refusal(plan, spec))
    if reason:
        report["reason"] = reason
    decision = tier.decide(
        tier.TierInputs(
            False,
            (),
            {},
            False,
            False,
            (),
            "",
            False,
            False,
            False,
            "",
            None,
            None,
            None,
            False,
            False,
            (),
        )
    )
    return assemble(plan, spec, report, {}, {}, decision, "", ())


def assemble(
    plan: select.Plan,
    spec: causal.CausalSpec,
    report: dict[str, Any],
    falsification: dict[str, Any],
    sens: dict[str, Any],
    decision: tier.TierDecision,
    strategy: str,
    needed: tuple[str, ...],
    attestors: Mapping[str, str] | None = None,
    certified: tuple[str, ...] = (),
) -> Outcome:
    """Builds the documents and the claim file that hashes the others."""
    documents = {
        "identification_plan.json": {
            **plan.to_dict(),
            "causal_spec": spec.to_dict(),
        },
        "identification_report.json": report,
        "falsification_report.json": falsification,
        "sensitivity_report.json": sens,
    }
    hashes = {k: digest(v) for k, v in documents.items()}
    claims = {
        "status": decision.status,
        "strategy": strategy or None,
        "estimand": spec.estimand,
        "tier": None if decision.tier is None else decision.tier.value,
        "conditions": decision.conditions,
        "reasons": list(decision.reasons),
        "action": decision.action,
        "assumptions": {a: (attestors or {}).get(a, "") for a in needed},
        "certified_gates_used": sorted(certified),
        "hashes": hashes,
        "limits": LIMITS,
    }
    documents["causal_claims.json"] = claims
    hashes["causal_claims.json"] = digest(claims)
    return Outcome(documents, decision, hashes)


def alphas(error_budget: budget_module.ErrorBudget) -> tuple[float, float]:
    """`(alpha_primary, alpha_falsify)` split from the confirmation level."""
    parts = budget_module.split_budget(error_budget, battery.FALSIFY_WEIGHTS)
    return (
        float(parts["estimate"].confirmation.alpha_confirm),
        float(parts["falsify"].confirmation.alpha_confirm),
    )


def replication(
    adapter: Adapter, c: pd.DataFrame, level: float, pooled: float, alpha: float
) -> dict[str, Any]:
    """Re-estimates inside each declared stratum and tests agreement."""
    results: dict[str, Any] = {}
    passed: bool | None = None
    for column in adapter.spec.strata:
        if column not in c.columns:
            results[column] = {"skipped": "column not in C"}
            continue
        values: dict[str, tuple[float, float]] = {}
        for label, part in stratum_frames(c, column).items():
            try:
                est = adapter.estimate(part, level)
            except ValueError:
                continue
            if np.isfinite(est.se) and est.se > 0:
                values[f"{column}={label}"] = (est.estimate, est.se)
        if len(values) < battery.MIN_STRATA:
            results[column] = {"skipped": "fewer than two usable strata"}
            continue
        rep = battery.replicate(values, pooled, alpha)
        results[column] = rep.to_dict()
        passed = rep.passed if passed is None else (passed and rep.passed)
    return {"by_column": results, "passed": passed}


def identify(
    spec: causal.CausalSpec,
    x: pd.DataFrame,
    c: pd.DataFrame,
    error_budget: budget_module.ErrorBudget,
    attestors: Mapping[str, str] | None = None,
    dag: dag_module.Dag | None = None,
    seed: int = 0,
    learner: str = "linear",
    certified: tuple[str, ...] | None = None,
) -> Outcome:
    """Selects, estimates, falsifies, assesses and tiers a causal claim.

    Args:
        spec: The charter's causal block.
        x: Exploration rows (design, nuisances, bandwidth).
        c: Confirmation rows; used for the estimate and the controls only.
        error_budget: The charter's error budget; its confirmation level is
            split into the primary level and `alpha_falsify` (weights 4:1).
        attestors: Assumption to the named attestor.
        dag: The declared DAG, if any.
        seed: Seed of the learners.
        learner: `linear` or `boosted` nuisance learners.
        certified: Gates with a current passing certificate; None loads the
            shipped certificate file, and an empty tuple certifies nothing.

    Returns:
        The five documents, their hashes and the tier decision.
    """
    plan = select.select(spec, dag)
    if plan.primary is None:
        return refusal_outcome(plan, spec)
    if certified is None:
        from scmle.data.identification import certified as loader

        certified = tuple(sorted(loader.certified_gates()))
    primary_alpha, alpha_falsify = alphas(error_budget)
    level = 1.0 - primary_alpha
    strategy = plan.primary
    adapter = ADAPTERS[strategy](spec, x, seed, learner, alpha_falsify)
    try:
        adapter.prepare()
    except ValueError as error:
        return refusal_outcome(
            plan, spec, f"design could not be assessed: {error}"
        )
    if adapter.refusal:
        return refusal_outcome(plan, spec, adapter.refusal)
    try:
        primary = adapter.estimate(c, level)
    except ValueError as error:
        return refusal_outcome(plan, spec, str(error))
    tests: list[battery.Test] = []
    notes: list[str] = []
    try:
        tests = adapter.controls(c, level)
    except ValueError as error:
        notes.append(f"controls could not run: {error}")
    for control in spec.negative_controls:
        if control.kind == "exposure" and strategy != "backdoor_aipw":
            notes.append(
                f"negative-control exposure {control.variable} is only "
                "supported for backdoor_aipw and was not run"
            )
    report_battery = battery.run_battery(tests, alpha_falsify)
    rep = (
        replication(adapter, c, level, primary.estimate, alpha_falsify)
        if spec.strata and not adapter.single_unit
        else {"by_column": {}, "passed": None}
    )
    sens: dict[str, Any] = (
        adapter.sensitivity(c, level)
        if not adapter.single_unit
        else {
            "holds": None,
            "note": "single treated unit: inference is the placebo permutation",
        }
    )
    verdict = verdict_of(spec, primary)
    verdict_sig = primary.excludes_zero() if np.isfinite(primary.low) else None
    verdict["interval_excludes_zero"] = verdict_sig
    needed = needed_assumptions(strategy, spec)
    checked = strategy != "backdoor_aipw" or plan.verdict(strategy).checked
    holds = sens.get("holds")
    facts = tier.TierInputs(
        identified=True,
        needed=needed,
        attestors=dict(attestors or {}),
        estimated_on_c=not adapter.single_unit,
        design_passed=True,
        design_gates=DESIGN_GATES.get(strategy, ()),
        interval_gate=INTERVAL_GATES[strategy],
        battery_ran=report_battery.ran,
        battery_passed=report_battery.passed,
        negative_control_declared=bool(spec.negative_controls),
        battery_gate=BATTERY_GATES[strategy],
        replication_passed=rep["passed"],
        sensitivity_holds=holds,
        sensitivity_gate=SENSITIVITY_GATES.get(strategy),
        single_unit=adapter.single_unit,
        weak_instrument=adapter.weak,
        certified=certified,
        adjustment_checked=checked,
    )
    decision = tier.decide(facts)
    if not checked:
        notes.append(
            "the adjustment set is an unchecked attestation (no DAG): the "
            "tier cannot exceed attested"
        )
    report = {
        "status": select.IDENTIFIED,
        "strategy": strategy,
        "estimand": spec.estimand,
        "level": level,
        "alpha_primary": primary_alpha,
        "alpha_falsify": alpha_falsify,
        "estimate": {
            "point": primary.estimate,
            "se": primary.se,
            "low": primary.low,
            "high": primary.high,
            "n": primary.n,
            "detail": dict(primary.detail),
        },
        "design": adapter.design,
        "multiverse_not_run": list(plan.multiverse),
        "notes": notes,
        "verdict": verdict,
    }
    falsification = {
        **report_battery.to_dict(),
        "replication": rep,
        "gate": BATTERY_GATES[strategy],
    }
    return assemble(
        plan,
        spec,
        report,
        falsification,
        sens,
        decision,
        strategy,
        needed,
        attestors,
        certified,
    )


def verdict_of(spec: causal.CausalSpec, est: Est) -> dict[str, Any]:
    """The primary hypothesis `theta = 0` stated in words."""
    return {
        "hypothesis": (
            f"{spec.estimand} of {spec.treatment} on {spec.outcome} = 0"
        ),
        "labelled": "causal, under the attested assumptions",
    }


def needed_assumptions(
    strategy: str, spec: causal.CausalSpec
) -> tuple[str, ...]:
    """The attested assumptions the chosen strategy relies on."""
    required = select.STRATEGIES[strategy].assumptions
    out = []
    for word in required:
        if word == "unconfoundedness_given":
            for item in spec.assumptions:
                if item.startswith(causal.UNCONFOUNDEDNESS_PREFIX):
                    out.append(item)
        else:
            out.append(word)
    return tuple(out)
