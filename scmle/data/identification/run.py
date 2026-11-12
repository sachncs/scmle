"""Runs a chartered causal analysis end to end.

Order: select (a function of the charter), diagnose the design and fit frozen
nuisances on `X`, estimate on `C` once, run the pre-registered falsification
battery and the replication, assess sensitivity, and tier the claim. The
function receives `X` and `C` as frames; opening `C` through the custody vault
exactly once is the caller's job (the confirmation function), and nothing in
this module reads `C` before the estimate step or refits anything on it.

Outputs are five JSON-ready documents, named in `DOCUMENTS`. A claim is
`not_identified` (and no estimate is printed) when no strategy is eligible,
when the overlap gate fails with no trimming declared, when trimming leaves
too few rows, or when the regression-discontinuity density check rejects on
`X`.

Single-treated-unit synthetic control has nothing to hold out: it runs on `X`
and its tier is capped at `attested`.
"""

from collections.abc import Mapping
import dataclasses
from fractions import Fraction
import hashlib
import json
from typing import Any
from typing import cast

import numpy
import pandas

from scmle.contracts import qualification
import scmle.data.admissibility.dag
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
import scmle.data.identification.certified
from scmle.data.ledger import budget
from scmle.data.ledger import multiplicity
import scmle.data.ledger.budget

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
    """A point estimate with its standard error and interval.

    Attributes:
        estimate: The point estimate.
        se: Its standard error (NaN when the strategy gives none).
        low: Lower interval limit.
        high: Upper interval limit.
        n: Rows used.
        detail: Strategy-specific extras for the report.
    """

    estimate: float
    se: float
    low: float
    high: float
    n: int
    detail: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def excludes_zero(self) -> bool:
        """Tests whether the interval excludes zero.

        Returns:
            True when the lower limit is above zero or the upper limit is
            below it.
        """
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
    decision: tier.Assignment
    hashes: dict[str, str]


def digest(document: Mapping[str, Any]) -> str:
    """Hashes the canonical JSON of a document.

    Args:
        document: The JSON-ready document; values that are not JSON types are
            converted with `str`.

    Returns:
        The SHA-256 hex digest of the document's JSON with sorted keys and
        compact separators.
    """
    text = json.dumps(
        document, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def wald_from(est: Est, name: str, kind: str) -> battery.Test | None:
    """Builds a Wald test of a placebo estimate.

    Args:
        est: The placebo estimate.
        name: Name of the test.
        kind: Kind of the test.

    Returns:
        The test, or None when the standard error is not finite and positive.
    """
    if not numpy.isfinite(est.se) or est.se <= 0:
        return None
    return battery.wald_test(name, kind, est.estimate, est.se)


class Estimator:
    """What a strategy needs: design check, estimator, controls, extras.

    Subclasses fix `name` and implement the hooks. `prepare` runs on `X`
    only; every later method reads `C`.

    Attributes:
        spec: The charter's causal block.
        x: Exploration rows.
        seed: Seed of the learners.
        learner: `linear` or `boosted` nuisance learners.
        alpha_falsify: Familywise level of the falsification battery.
        design: Design diagnostics for the report.
        refusal: Why the claim must be refused, or None.
        weak: Whether the instrument was found weak.
        single_unit: Whether there is a single treated unit (nothing to hold
            out).
        notes: Remarks for the report.
    """

    name = ""

    def __init__(
        self,
        spec: causal.Question,
        x: pandas.DataFrame,
        seed: int,
        learner: str,
        alpha_falsify: float,
    ) -> None:
        """Stores the settings of the strategy.

        Args:
            spec: The charter's causal block.
            x: Exploration rows.
            seed: Seed of the learners.
            learner: `linear` or `boosted` nuisance learners.
            alpha_falsify: Familywise level of the falsification battery.
        """
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
        """Runs the design diagnostics and freezes quantities, from `X` only.

        The base class does nothing; subclasses fill `design` and may set
        `refusal`.
        """

    def estimate(self, c: pandas.DataFrame, level: float) -> Est:
        """Computes the primary estimate on `C`.

        Args:
            c: The confirmation rows.
            level: Confidence level.

        Returns:
            The estimate with its standard error and interval.

        Raises:
            NotImplementedError: Always, in the base class; subclasses
                override it.
        """
        raise NotImplementedError

    def estimate_for(
        self, c: pandas.DataFrame, level: float, outcome: str | None = None
    ) -> Est:
        """Computes the same estimate on `C` with another outcome column.

        Args:
            c: The confirmation rows.
            level: Confidence level.
            outcome: The outcome column to use, or None for the charter's
                outcome.

        Returns:
            The estimate with its standard error and interval.

        Raises:
            NotImplementedError: Always, in the base class; subclasses
                override it.
        """
        raise NotImplementedError

    def controls(self, c: pandas.DataFrame, level: float) -> list[battery.Test]:  # noqa: ARG002 -- base-class hook; overriding strategies use these arguments
        """Runs the falsification tests on `C` (negative controls and placebos).

        Args:
            c: The confirmation rows.
            level: Confidence level.

        Returns:
            The tests; the base class has none.
        """
        return []

    def sensitivity(self, c: pandas.DataFrame, level: float) -> dict[str, Any]:  # noqa: ARG002 -- base-class hook; overriding strategies use these arguments
        """Assesses sensitivity to unmeasured confounding.

        Args:
            c: The confirmation rows.
            level: Confidence level.

        Returns:
            The sensitivity document; the base class returns `holds` as None
            with a note that the strategy has no sensitivity analysis.
        """
        return {
            "holds": None,
            "note": "no sensitivity analysis for this strategy",
        }


class BackdoorEstimator(Estimator):
    """AIPW with nuisances frozen on `X` and no cross-fitting."""

    name = "backdoor_aipw"

    def __init__(self, *args: Any) -> None:
        """Stores the settings, then reads the covariates from the charter.

        The nuisance models start unset.

        Args:
            *args: The `Adapter` constructor arguments.
        """
        super().__init__(*args)
        self.covariates = causal.covariates_of(self.spec.assumptions) or ()
        self.nuisances: aipw.Nuisances | None = None

    def prepare(self) -> None:
        """Fits the frozen nuisances and checks overlap, from `X` only.

        The claim is refused when the overlap gate fails on `X` and the charter
        declares no trimming rule.
        """
        spec = self.spec
        self.nuisances = aipw.fit_nuisances(
            self.x,
            self.covariates,
            spec.treatment,
            spec.outcome,
            self.seed,
            self.learner,
        )
        e = self.nuisances.predict(self.x)[0]
        t = aipw.binary(self.x[spec.treatment], spec.treatment)
        matrix = cast(
            "numpy.ndarray",
            self.nuisances.scaler.transform(
                self.nuisances.design.matrix(self.x)
            ),
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

    def estimate(self, c: pandas.DataFrame, level: float) -> Est:
        """Computes the primary AIPW estimate on `C` with the frozen nuisances.

        Args:
            c: The confirmation rows.
            level: Confidence level.

        Also records `overlap_on_c` and `population` in `design`.

        Returns:
            The estimate, with the estimated population in its detail.

        Raises:
            ValueError: If `prepare` has not run. Errors of `aipw.estimate`
                (trimming below `min_share`, a thin arm, missing outcomes)
                are also `ValueError`.
        """
        if self.nuisances is None:
            raise ValueError("prepare() must run first")
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
        self, c: pandas.DataFrame, level: float, outcome: str | None = None
    ) -> Est:
        """Computes the AIPW estimate on `C` with another outcome column.

        The nuisances are refitted on `X` for that outcome.

        Args:
            c: The confirmation rows.
            level: Confidence level.
            outcome: The outcome column, or None for the charter's outcome.

        Returns:
            The estimate with its standard error and interval.
        """
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

    def controls(self, c: pandas.DataFrame, level: float) -> list[battery.Test]:
        """Runs the declared negative controls on `C`.

        Args:
            c: The confirmation rows.
            level: Confidence level.

        Returns:
            A Wald test for each negative-control outcome and exposure whose
            estimate has a usable standard error.
        """
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

    def sensitivity(self, c: pandas.DataFrame, level: float) -> dict[str, Any]:
        """Assesses sensitivity of the estimate to unmeasured confounding.

        Args:
            c: The confirmation rows.
            level: Confidence level.

        Returns:
            The sensitivity report of the linear-model summary on `C`, with a
            note on its scale; or `holds` as None with a note when the
            charter has no sensitivity block or no covariate exists to
            benchmark against.

        Raises:
            ValueError: If the charter has a sensitivity block and covariates
                but `prepare` has not run.
        """
        spec = self.spec
        if spec.sensitivity is None:
            return {
                "holds": None,
                "note": "no sensitivity block in the charter",
            }
        if not self.covariates:
            return {"holds": None, "note": "benchmarking needs a covariate"}
        if self.nuisances is None:
            raise ValueError("prepare() must run first")
        design = self.nuisances.design
        matrix = design.matrix(c)
        names = [f"x{j}" for j in range(matrix.shape[1])]
        frame = pandas.DataFrame(matrix, columns=names)
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


class RandomizedEstimator(Estimator):
    """Difference in means under a recorded assignment mechanism."""

    name = "randomized_diff"

    def estimate(self, c: pandas.DataFrame, level: float) -> Est:
        """Computes the difference in means on `C`.

        Args:
            c: The confirmation rows.
            level: Confidence level.

        Returns:
            The estimate with its standard error and interval.
        """
        spec = self.spec
        r = aipw.difference_in_means(c, spec.treatment, spec.outcome, level)
        return Est(r.estimate, r.se, r.low, r.high, r.n)

    def estimate_for(
        self, c: pandas.DataFrame, level: float, outcome: str | None = None
    ) -> Est:
        """Computes the difference in means on `C` with another outcome column.

        Args:
            c: The confirmation rows.
            level: Confidence level.
            outcome: The outcome column, or None for the charter's outcome.

        Returns:
            The estimate with its standard error and interval.
        """
        r = aipw.difference_in_means(
            c, self.spec.treatment, outcome or self.spec.outcome, level
        )
        return Est(r.estimate, r.se, r.low, r.high, r.n)

    def controls(self, c: pandas.DataFrame, level: float) -> list[battery.Test]:
        """Runs the declared negative-control outcomes on `C`.

        Args:
            c: The confirmation rows.
            level: Confidence level.

        Returns:
            A Wald test for each negative-control outcome whose estimate has a
            usable standard error; exposures are not supported here.
        """
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

    def sensitivity(self, c: pandas.DataFrame, level: float) -> dict[str, Any]:  # noqa: ARG002 -- overrides Estimator.sensitivity, which fixes this signature
        """Reports that randomization leaves no unmeasured confounding to bound.

        Args:
            c: The confirmation rows (not read).
            level: Confidence level (not read).

        Returns:
            A document with `holds` True and `applicable` False.
        """
        return {
            "holds": True,
            "note": "randomized assignment: no unmeasured confounding to bound",
            "applicable": False,
        }


class DidEstimator(Estimator):
    """Group-time difference in differences."""

    name = "did"

    def columns(self) -> tuple[str, str, str]:
        """Reads the panel column names from the charter.

        Returns:
            The unit, time and adoption column names.

        Raises:
            ValueError: If any of them is not declared.
        """
        s = self.spec
        if not (s.unit and s.time and s.adoption):
            raise ValueError("did needs unit, time and adoption")
        return s.unit, s.time, s.adoption

    def prepare(self) -> None:
        """Estimates on `X` to record the panel and the pre-trend test.

        The results go into `design` (default level); the estimate is dropped.
        """
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
        self, c: pandas.DataFrame, level: float, outcome: str | None = None
    ) -> did.Estimate:
        """Fits the difference-in-differences estimator on the given rows.

        Args:
            c: The panel rows.
            level: Confidence level.
            outcome: The outcome column, or None for the charter's outcome.

        Returns:
            The fitted group-time result.
        """
        unit, time, adoption = self.columns()
        return did.estimate(
            c, unit, time, adoption, outcome or self.spec.outcome, level
        )

    def estimate(self, c: pandas.DataFrame, level: float) -> Est:
        """Computes the overall ATT on `C`.

        Args:
            c: The confirmation rows.
            level: Confidence level.

        Returns:
            The estimate, with the pre-trend p-value and the event-study
            coefficients in its detail.
        """
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
        self, c: pandas.DataFrame, level: float, outcome: str | None = None
    ) -> Est:
        """Computes the overall ATT on `C` with another outcome column.

        Args:
            c: The confirmation rows.
            level: Confidence level.
            outcome: The outcome column, or None for the charter's outcome.

        Returns:
            The estimate with its standard error and interval.
        """
        fit = self.fit(c, level, outcome)
        return Est(fit.estimate, fit.se, fit.low, fit.high, len(c))

    def controls(self, c: pandas.DataFrame, level: float) -> list[battery.Test]:
        """Runs the pre-trend test and negative-control outcomes on `C`.

        Args:
            c: The confirmation rows.
            level: Confidence level.

        Returns:
            The pre-trend Wald test when pre-period coefficients exist,
            followed by a Wald test for each negative-control outcome with a
            usable standard error.
        """
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

    def sensitivity(self, c: pandas.DataFrame, level: float) -> dict[str, Any]:
        """Applies the relative-magnitudes bound on `C`.

        Args:
            c: The confirmation rows.
            level: Confidence level.

        Returns:
            The widened interval at the charter's `m_max`, whether it still
            excludes zero, the breakdown multiple and the largest pre-period
            change; or `holds` as None with a note when the charter has no
            sensitivity block or fewer than two pre-periods exist.
        """
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


class IvEstimator(Estimator):
    """Just-identified instrumental variables."""

    name = "iv"

    def controls_columns(self) -> tuple[str, ...]:
        """Lists the covariates used as IV controls.

        Returns:
            The covariate columns declared in the unconfoundedness
            assumption, or an empty tuple.
        """
        return causal.covariates_of(self.spec.assumptions) or ()

    def fit(
        self, frame: pandas.DataFrame, level: float, outcome: str | None = None
    ) -> iv.Estimate:
        """Fits the just-identified IV model on the given rows.

        Args:
            frame: The rows.
            level: Confidence level.
            outcome: The outcome column, or None for the charter's outcome.

        Returns:
            The fitted IV result.

        Raises:
            ValueError: If the charter declares no instrument.
        """
        spec = self.spec
        if not spec.instrument:
            raise ValueError("iv needs an instrument")
        return iv.estimate(
            frame,
            spec.treatment,
            outcome or spec.outcome,
            spec.instrument,
            self.controls_columns(),
            level,
        )

    def prepare(self) -> None:
        """Fits on `X` (level 0.95) to record the first-stage F and strength.

        Sets `weak` and `design`.
        """
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

    def estimate(self, c: pandas.DataFrame, level: float) -> Est:
        """Computes the IV estimate on `C`.

        A weak instrument on `C` also marks the run as weak (`weak`), and
        the first-stage F on `C` is recorded in `design`.

        Args:
            c: The confirmation rows.
            level: Confidence level.

        Returns:
            The estimate, with the headline interval kind in its detail.
        """
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
        self, c: pandas.DataFrame, level: float, outcome: str | None = None
    ) -> Est:
        """Computes the IV estimate on `C` with another outcome column.

        Args:
            c: The confirmation rows.
            level: Confidence level.
            outcome: The outcome column, or None for the charter's outcome.

        Returns:
            The estimate with its standard error and the 2SLS interval limits.
        """
        fit = self.fit(c, level, outcome)
        return Est(fit.estimate, fit.se, fit.wald_low, fit.wald_high, fit.n)

    def controls(self, c: pandas.DataFrame, level: float) -> list[battery.Test]:
        """Runs the declared negative-control outcomes on `C`.

        Args:
            c: The confirmation rows.
            level: Confidence level.

        Returns:
            A Wald test for each negative-control outcome whose estimate has a
            usable standard error.
        """
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


class RdEstimator(Estimator):
    """Sharp regression discontinuity with a bandwidth frozen on `X`."""

    name = "rd"

    def __init__(self, *args: Any) -> None:
        """Stores the settings, with the bandwidth unset until it is fixed.

        Args:
            *args: The `Adapter` constructor arguments.
        """
        super().__init__(*args)
        self.bandwidth = 0.0

    def balance_columns(self) -> tuple[str, ...]:
        """Lists the predetermined covariates tested for a jump.

        Returns:
            The variables of the declared negative-control outcomes.
        """
        return tuple(
            c.variable
            for c in self.spec.negative_controls
            if c.kind == "outcome"
        )

    def prepare(self) -> None:
        """Fixes the bandwidth and checks the density at the cutoff on `X`.

        The claim is refused when the density check rejects at `alpha_falsify`.

        Raises:
            ValueError: If the charter declares no running variable or cutoff.
        """
        spec = self.spec
        if not spec.running_variable or spec.cutoff is None:
            raise ValueError("rd needs a running variable and cutoff")
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
        c: pandas.DataFrame,
        level: float,
        outcome: str | None = None,
        covariates: tuple[str, ...] = (),
    ) -> rd.Estimate:
        """Fits the discontinuity at the frozen bandwidth.

        Args:
            c: The rows to fit.
            level: Confidence level.
            outcome: The outcome column, or None for the charter's outcome.
            covariates: Covariates to test for a jump.

        Returns:
            The fitted regression discontinuity result.

        Raises:
            ValueError: If the charter declares no running variable or cutoff.
        """
        spec = self.spec
        if not spec.running_variable or spec.cutoff is None:
            raise ValueError("rd needs a running variable and cutoff")
        return rd.estimate(
            c,
            spec.running_variable,
            outcome or spec.outcome,
            spec.cutoff,
            level,
            self.bandwidth,
            covariates,
        )

    def estimate(self, c: pandas.DataFrame, level: float) -> Est:
        """Computes the bias-corrected discontinuity on `C`.

        Args:
            c: The confirmation rows.
            level: Confidence level.

        Returns:
            The bias-corrected estimate, with the conventional estimate and the
            bandwidth in its detail.
        """
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
        self, c: pandas.DataFrame, level: float, outcome: str | None = None
    ) -> Est:
        """Computes the bias-corrected discontinuity for another outcome.

        Args:
            c: The confirmation rows.
            level: Confidence level.
            outcome: The outcome column, or None for the charter's outcome.

        Returns:
            The estimate with its standard error and interval.
        """
        fit = self.fit(c, level, outcome)
        return Est(
            fit.bias_corrected,
            fit.se,
            fit.low,
            fit.high,
            fit.n_left + fit.n_right,
        )

    def controls(self, c: pandas.DataFrame, level: float) -> list[battery.Test]:
        """Runs the placebo cutoffs and covariate balance tests on `C`.

        Args:
            c: The confirmation rows.
            level: Confidence level.

        Returns:
            A test for each placebo cutoff and each covariate discontinuity
            whose fit was not degenerate.
        """
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

    def sensitivity(self, c: pandas.DataFrame, level: float) -> dict[str, Any]:
        """Reports how the estimate changes with the bandwidth.

        Args:
            c: The confirmation rows.
            level: Confidence level.

        Returns:
            A document with `holds` True and `applicable` False (a regression
            discontinuity has no unmeasured-confounding bound), the estimates at
            other bandwidth multiples, and whether each leads to the same
            conclusion about zero.
        """
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


class SynthEstimator(Estimator):
    """Synthetic control with one treated unit; runs on `X` alone."""

    name = "synth"

    def __init__(self, *args: Any) -> None:
        """Stores the settings of the strategy and marks a single treated unit.

        Args:
            *args: The `Adapter` constructor arguments.
        """
        super().__init__(*args)
        self.single_unit = True

    def estimate(self, c: pandas.DataFrame, level: float) -> Est:  # noqa: ARG002 -- overrides Estimator.estimate, which fixes this signature
        """Computes the synthetic control effect on `X`.

        With one treated unit there is nothing to hold out, so the rows of
        `X` are used and `c` is not read.

        Args:
            c: The confirmation rows (not read).
            level: Confidence level; `1 - level` is the level the donor count is
                checked against.

        Returns:
            The mean post-period gap, with NaN standard error and limits and the
            placebo p-value and ratio in its detail.

        Raises:
            ValueError: If the charter lacks unit or time, the panel does not
                hold exactly one treated unit, or `synth.estimate` refuses
                the panel (for example too few donors).
        """
        spec = self.spec
        if not (spec.unit and spec.time):
            raise ValueError("synth needs unit and time")
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
        self, c: pandas.DataFrame, level: float, outcome: str | None = None
    ) -> Est:
        """Always refuses: synthetic control has no other-outcome estimate.

        Args:
            c: The confirmation rows.
            level: Confidence level.
            outcome: The outcome column.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError


ADAPTERS: dict[str, type[Estimator]] = {
    "backdoor_aipw": BackdoorEstimator,
    "randomized_diff": RandomizedEstimator,
    "did": DidEstimator,
    "iv": IvEstimator,
    "rd": RdEstimator,
    "synth": SynthEstimator,
}


def refusal_outcome(
    plan: select.Plan,
    spec: causal.Question,
    reason: str | None = None,
    dag_checks: Mapping[str, Any] | None = None,
) -> Outcome:
    """Builds the five documents of a claim that is not identified.

    Args:
        plan: The selector's plan.
        spec: The charter's causal block.
        reason: Why the claim is refused, or None.
        dag_checks: What was found about the declared DAG, or None.

    Returns:
        The documents, their hashes and a tier decision with no tier.
    """
    report = dict(select.refusal(plan, spec))
    if reason:
        report["reason"] = reason
    if dag_checks is not None:
        report["dag"] = dict(dag_checks)
    decision = tier.decide(
        tier.Inputs(
            identified=False,
            needed=(),
            attestors={},
            estimated_on_c=False,
            design_passed=False,
            design_gates=(),
            interval_gate="",
            battery_ran=False,
            battery_passed=False,
            negative_control_declared=False,
            battery_gate="",
            replication_passed=None,
            sensitivity_holds=None,
            sensitivity_gate=None,
            single_unit=False,
            weak_instrument=False,
            certified=(),
        )
    )
    return assemble(plan, spec, report, {}, {}, decision, "", ())


def assemble(
    plan: select.Plan,
    spec: causal.Question,
    report: dict[str, Any],
    falsification: dict[str, Any],
    sens: dict[str, Any],
    decision: tier.Assignment,
    strategy: str,
    needed: tuple[str, ...],
    attestors: Mapping[str, str] | None = None,
    certified: tuple[str, ...] = (),
    subgroups: Mapping[str, Any] | None = None,
    attestation_report: Mapping[str, Any] | None = None,
) -> Outcome:
    """Builds the documents and the claim file that hashes the others.

    Args:
        plan: The selector's plan.
        spec: The charter's causal block.
        report: The identification report.
        falsification: The falsification report.
        sens: The sensitivity report.
        decision: The tier decision.
        strategy: Name of the chosen strategy, or an empty string (recorded
            as null in the claim file).
        needed: The assumptions the strategy relies on.
        attestors: Assumption to the named attestor.
        certified: Certificate gates used.
        subgroups: The labelled subgroup report, or None.
        attestation_report: What was decided about the signed attestations, or
            None.

    Returns:
        The documents (the claim file included), the tier decision and the hash
        of every document.
    """
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
    claims: dict[str, Any] = {
        "status": decision.status,
        "strategy": strategy or None,
        "estimand": spec.estimand,
        "tier": None if decision.tier is None else decision.tier.value,
        "conditions": decision.conditions,
        "reasons": list(decision.reasons),
        "action": decision.action,
        "assumptions": {a: (attestors or {}).get(a, "") for a in needed},
        "certified_gates_used": sorted(certified),
        "hashes": dict(hashes),
        "limits": LIMITS,
    }
    if subgroups is not None:
        claims["subgroups"] = {
            "rule": subgroups["rule"],
            "alpha": subgroups["alpha"],
            "family_size": subgroups["family_size"],
            "rejected": list(subgroups["rejected"]),
            "gate": subgroups["gate"]["status"],
        }
    if attestation_report is not None:
        claims["attestations"] = dict(attestation_report)
    documents["causal_claims.json"] = claims
    hashes["causal_claims.json"] = digest(claims)
    return Outcome(documents, decision, hashes)


def alphas(error_budget: budget.ErrorBudget) -> tuple[float, float]:
    """Splits the confirmation level into the primary and falsification levels.

    Args:
        error_budget: The charter's error budget.

    Returns:
        ``(alpha_primary, alpha_falsify)``, the confirmation level divided in
        the ratio 4 to 1.
    """
    parts = budget.split_budget(error_budget, battery.FALSIFY_WEIGHTS)
    return (
        float(parts["estimate"].confirmation.alpha_confirm),
        float(parts["falsify"].confirmation.alpha_confirm),
    )


def replication(
    adapter: Estimator,
    c: pandas.DataFrame,
    level: float,
    pooled: float,
    alpha: float,
) -> dict[str, Any]:
    """Re-estimates inside each declared stratum and tests agreement.

    Args:
        adapter: The strategy's adapter.
        c: The confirmation rows.
        level: Confidence level.
        pooled: The pooled estimate whose sign the strata must share.
        alpha: Level of the heterogeneity test.

    Returns:
        The replication result per stratifying column under ``by_column``
        (a skip reason when the column is absent from `c` or has fewer than
        two usable strata), and ``passed``: None when no column had two usable
        strata, otherwise whether every column that did replicated. A stratum
        is usable when it has `MIN_STRATUM_ROWS` rows and `adapter.estimate`
        gives a finite positive standard error without a `ValueError`. The
        adapter's `design` and `weak` are restored afterwards, so the report
        keeps the values from the pooled estimate on `C`.
    """
    results: dict[str, Any] = {}
    passed: bool | None = None
    saved_design = dict(adapter.design)
    saved_weak = adapter.weak
    for column in adapter.spec.strata:
        if column not in c.columns:
            results[column] = {"skipped": "column not in C"}
            continue
        values: dict[str, tuple[float, float]] = {}
        for stratum, part in c.groupby(column):
            if len(part) < MIN_STRATUM_ROWS:
                continue
            try:
                est = adapter.estimate(part, level)
            except ValueError:
                continue
            if numpy.isfinite(est.se) and est.se > 0:
                values[f"{column}={stratum}"] = (est.estimate, est.se)
        if len(values) < battery.MIN_STRATA:
            results[column] = {"skipped": "fewer than two usable strata"}
            continue
        rep = battery.replicate(values, pooled, alpha)
        results[column] = rep.to_dict()
        passed = rep.passed if passed is None else (passed and rep.passed)
    adapter.design = saved_design
    adapter.weak = saved_weak
    return {"by_column": results, "passed": passed}


def subgroup_report(
    adapter: Estimator,
    c: pandas.DataFrame,
    level: float,
    subgroups: tuple[causal.Subgroup, ...],
) -> dict[str, Any]:
    """Estimates the effect in each declared subgroup, adjusted as one family.

    The family is every declared (column, value) pair, fixed by the charter
    and registered with the primary claim before `C` is opened. Each
    hypothesis `theta_g = 0` is tested by a Wald statistic of the strategy's
    own estimator on the rows of `C` in the subgroup, and the p-values are
    adjusted by Holm's step-down at the primary level `1 - level`, which
    controls the familywise error rate under any dependence. A subgroup that
    cannot be estimated (column absent, fewer than `MIN_STRATUM_ROWS` rows, no
    usable standard error) stays in the family with p = 1, so a missing
    subgroup never shrinks the correction. The result describes
    heterogeneity; it does not enter the tier.

    Args:
        adapter: The strategy's adapter.
        c: The confirmation rows.
        level: Confidence level; the family level is `1 - level`.
        subgroups: The declared subgroups.

    Returns:
        A mapping with the rule, the family level `alpha` (`1 - level`), the
        family size, the guarantee `statement`, one row per hypothesis under
        `hypotheses` (estimate, p-value, adjusted p-value, decision, or why
        it was skipped) and the names of the `rejected` subgroups. The
        adapter's `design` is restored afterwards; its `weak` flag is not
        (`identify` restores it).
    """
    saved = dict(adapter.design)
    rows: list[dict[str, Any]] = []
    for group in subgroups:
        for value in group.values:  # noqa: PD011 -- `Subgroup.values` is a dataclass field, not a pandas attribute
            row: dict[str, Any] = {
                "column": group.column,
                "value": value,
                "skipped": None,
            }
            part = None
            if group.column not in c.columns:
                row["skipped"] = "column not in C"
            else:
                part = c[c[group.column].astype(str) == value]
                if len(part) < MIN_STRATUM_ROWS:
                    row["skipped"] = f"fewer than {MIN_STRATUM_ROWS} rows"
            p_value = 1.0
            if part is not None and row["skipped"] is None:
                try:
                    est = adapter.estimate(part, level)
                except ValueError as error:
                    row["skipped"] = str(error)
                else:
                    if numpy.isfinite(est.se) and est.se > 0:
                        test = battery.wald_test(
                            f"{group.column}={value}",
                            "subgroup",
                            est.estimate,
                            est.se,
                        )
                        p_value = test.p_value
                        row.update(
                            estimate=est.estimate,
                            se=est.se,
                            low=est.low,
                            high=est.high,
                            n=est.n,
                        )
                    else:
                        row["skipped"] = "no usable standard error"
            row["p_value"] = p_value
            rows.append(row)
    adapter.design = saved
    alpha = Fraction(1.0 - level).limit_denominator(10**9)
    adjusted = multiplicity.adjust(
        [r["p_value"] for r in rows],
        scmle.data.ledger.budget.Correction.HOLM,
        alpha,
    )
    for row, p_adj, flag in zip(
        rows, adjusted.adjusted, adjusted.rejected, strict=True
    ):
        row["adjusted_p"] = p_adj
        row["rejected"] = bool(flag)
    return {
        "rule": scmle.data.ledger.budget.Correction.HOLM.value,
        "alpha": float(alpha),
        "family_size": len(rows),
        "statement": multiplicity.STATEMENTS[
            scmle.data.ledger.budget.Correction.HOLM
        ],
        "hypotheses": rows,
        "rejected": [
            f"{r['column']}={r['value']}" for r in rows if r["rejected"]
        ],
    }


def label_subgroups(
    found: dict[str, Any], named: tuple[str, ...] | None
) -> None:
    """Labels the subgroup report as gating or only descriptive, in place.

    The Holm family is certified by `subgroup_certificates.json` (gate
    `subgroup_holm`): both its error rates and its power met their
    pre-registered floors on this code. With the certificate the word is
    `gating`, and a subgroup with at least the certified number of rows is
    marked `certified_size`; without it the family is `descriptive`. The tier
    is not read from this: the tier rule has no subgroup input.

    Args:
        found: The `subgroup_report` result; gets `gate`, `statement_gate`
            and a `certified_size` flag on each hypothesis row.
        named: The certified gates the caller names (as `identify`'s
            `certified`), or None to read the shipped certificate. When
            given, it alone decides the status, but `min_rows` still comes
            from the shipped certificate (None when it is absent, so every
            estimated subgroup then counts as certified size).
    """
    gate = scmle.data.identification.certified.subgroup_gate(subgroup_report)
    if named is not None:
        listed = scmle.data.identification.certified.SUBGROUP_GATE in named
        gate = scmle.data.identification.certified.SubgroupGate(
            scmle.data.identification.certified.GATING
            if listed
            else scmle.data.identification.certified.DESCRIPTIVE,
            "named among the certified gates"
            if listed
            else "the caller named the certified gates and this is not one",
            gate.min_rows,
        )
    found["gate"] = gate.to_dict()
    gating = gate.status == scmle.data.identification.certified.GATING
    for row in found["hypotheses"]:
        size = row.get("n")
        row["certified_size"] = bool(
            gating
            and size is not None
            and (gate.min_rows is None or size >= gate.min_rows)
        )
    found["statement_gate"] = (
        "gating: the family-wise error and the power at the certified size "
        "met their pre-registered floors (a result for a subgroup under "
        "`min_rows` is still descriptive)"
        if gating
        else "descriptive: the family-wise error and power of the subgroup "
        "tests are not certified, so read a rejected level as exploratory; "
        f"{gate.reason}"
    )


def sensitivity_of(
    adapter: Estimator, c: pandas.DataFrame, level: float
) -> dict[str, Any]:
    """Assesses sensitivity to unmeasured confounding, if it can be done.

    Args:
        adapter: The strategy's estimator.
        c: The confirmation rows.
        level: Confidence level.

    Returns:
        The sensitivity document of the strategy, or a note that a single
        treated unit has only the placebo permutation.
    """
    if adapter.single_unit:
        return {
            "holds": None,
            "note": "single treated unit: inference is the placebo permutation",
        }
    return adapter.sensitivity(c, level)


def subgroup_results(
    adapter: Estimator,
    spec: causal.Question,
    c: pandas.DataFrame,
    level: float,
    named: tuple[str, ...] | None,
) -> dict[str, Any] | None:
    """Estimates and labels the declared subgroups.

    The adapter's `weak` flag is restored afterwards.

    Args:
        adapter: The strategy's estimator.
        spec: The charter's causal block.
        c: The confirmation rows.
        level: Confidence level.
        named: The certified gates the caller names, or None to read the
            shipped certificate (see `label_subgroups`).

    Returns:
        The labelled subgroup report, or None when no subgroups are
        declared or there is a single treated unit.
    """
    weak_before_subgroups = adapter.weak
    found = (
        subgroup_report(adapter, c, level, spec.subgroups)
        if spec.subgroups and not adapter.single_unit
        else None
    )
    adapter.weak = weak_before_subgroups
    if found is not None:
        label_subgroups(found, named)
    return found


def closing_notes(
    spec: causal.Question, strategy: str, *, checked: bool
) -> list[str]:
    """Lists the notes about what the run could not do or check.

    Args:
        spec: The charter's causal block.
        strategy: Registry name of the strategy.
        checked: Whether a declared DAG verified the adjustment set.

    Returns:
        A note for each negative-control exposure that the strategy does
        not support, then a note when the adjustment set is unchecked.
    """
    notes = [
        f"negative-control exposure {control.variable} is only "
        "supported for backdoor_aipw and was not run"
        for control in spec.negative_controls
        if control.kind == "exposure" and strategy != "backdoor_aipw"
    ]
    if not checked:
        notes.append(
            "the adjustment set is an unchecked attestation (no DAG): the "
            "tier cannot exceed attested"
        )
    return notes


def identify(
    spec: causal.Question,
    x: pandas.DataFrame,
    c: pandas.DataFrame,
    error_budget: budget.ErrorBudget,
    attestors: Mapping[str, str] | None = None,
    dag: scmle.data.admissibility.dag.Dag | None = None,
    seed: int = 0,
    learner: str = "linear",
    certified: tuple[str, ...] | None = None,
    dag_checks: Mapping[str, Any] | None = None,
    attestation_report: Mapping[str, Any] | None = None,
) -> Outcome:
    """Selects, estimates, falsifies, assesses and tiers a causal claim.

    Args:
        spec: The charter's causal block.
        x: Exploration rows (design, nuisances, bandwidth).
        c: Confirmation rows; used for the estimate, controls, replication,
            sensitivity and subgroups, never to fit nuisances or a bandwidth.
        error_budget: The charter's error budget; its confirmation level is
            split into the primary level and `alpha_falsify` (weights 4:1).
        attestors: Assumption to the named attestor.
        dag: The declared DAG, if any.
        seed: Seed of the learners.
        learner: `linear` or `boosted` nuisance learners.
        certified: Gates with a current passing certificate; None loads the
            shipped certificate file, and an empty tuple certifies nothing.
        dag_checks: What the admissibility stage found about the declared DAG
            on `X` (its hash, the conditional-independence falsification, the
            adjustment members the graph rules refused); copied into the
            identification report.
        attestation_report: What `attest.apply` decided about the signed
            attestations (`Applied.to_dict()`), copied into the claim file.

    Returns:
        The five documents, their hashes and the tier decision. A
        `ValueError` from the adapter's design check or primary estimate is
        not raised: it gives the refusal documents with the error text as the
        reason, as does a failed overlap or density gate. A `ValueError` from
        the controls is recorded as a note instead.
    """
    plan = select.select(spec, dag)
    if plan.primary is None:
        return refusal_outcome(plan, spec, dag_checks=dag_checks)
    named = certified
    if certified is None:
        certified = tuple(
            sorted(scmle.data.identification.certified.certified_gates())
        )
    primary_alpha, alpha_falsify = alphas(error_budget)
    level = 1.0 - primary_alpha
    strategy = plan.primary
    adapter = ADAPTERS[strategy](spec, x, seed, learner, alpha_falsify)
    try:
        adapter.prepare()
    except ValueError as error:
        return refusal_outcome(
            plan, spec, f"design could not be assessed: {error}", dag_checks
        )
    if adapter.refusal:
        return refusal_outcome(plan, spec, adapter.refusal, dag_checks)
    try:
        primary = adapter.estimate(c, level)
    except ValueError as error:
        return refusal_outcome(plan, spec, str(error), dag_checks)
    tests: list[battery.Test] = []
    notes: list[str] = []
    try:
        tests = adapter.controls(c, level)
    except ValueError as error:
        notes.append(f"controls could not run: {error}")
    report_battery = battery.run_battery(tests, alpha_falsify)
    rep: dict[str, Any]
    if spec.strata and not adapter.single_unit:
        rep = replication(adapter, c, level, primary.estimate, alpha_falsify)
    else:
        rep = {"by_column": {}, "passed": None}
    sens = sensitivity_of(adapter, c, level)
    subgroup_found = subgroup_results(adapter, spec, c, level, named)
    needed = needed_assumptions(strategy, spec)
    checked = strategy != "backdoor_aipw" or plan.verdict(strategy).checked
    facts = tier.Inputs(
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
        sensitivity_holds=sens.get("holds"),
        sensitivity_gate=SENSITIVITY_GATES.get(strategy),
        single_unit=adapter.single_unit,
        weak_instrument=adapter.weak,
        certified=certified,
        adjustment_checked=checked,
    )
    decision = tier.decide(facts)
    notes.extend(closing_notes(spec, strategy, checked=checked))
    verdict = verdict_of(spec)
    verdict["interval_excludes_zero"] = (
        primary.excludes_zero() if numpy.isfinite(primary.low) else None
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
    if dag_checks is not None:
        report["dag"] = dict(dag_checks)
    if subgroup_found is not None:
        report["subgroups"] = subgroup_found
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
        subgroup_found,
        attestation_report,
    )


NOT_ESTIMATED = (
    "no estimate was computed on the confirmation set: it was not available "
    "to the run, or the claim could not be registered, so the claim is "
    "unverified. `identify` computes the estimate on the single opening of C "
    "when the ledger hands it the released rows (`scmle.core.causal_claim`)."
)


def design_only(
    spec: causal.Question,
    x: pandas.DataFrame,
    error_budget: budget.ErrorBudget,
    dag: scmle.data.admissibility.dag.Dag | None = None,
    seed: int = 0,
    learner: str = "linear",
    dag_checks: Mapping[str, Any] | None = None,
) -> Outcome:
    """Selects and diagnoses the design on `X`, with no estimate.

    Used by the pipeline stage, which never hands `C` to this module: the
    plan, the refusal if there is one, the design diagnostics and a claim
    file at tier `unverified`.

    Args:
        spec: The charter's causal block.
        x: Exploration rows.
        error_budget: The charter's error budget.
        dag: The declared DAG, if any.
        seed: Seed of the learners.
        learner: `linear` or `boosted` nuisance learners.
        dag_checks: What was found about the declared DAG, copied into the
            report.

    Returns:
        The five documents, their hashes and a tier decision at `unverified`
        (or the refusal documents when the claim is not identified).
    """
    plan = select.select(spec, dag)
    if plan.primary is None:
        return refusal_outcome(plan, spec, dag_checks=dag_checks)
    alpha_falsify = alphas(error_budget)[1]
    strategy = plan.primary
    adapter = ADAPTERS[strategy](spec, x, seed, learner, alpha_falsify)
    try:
        adapter.prepare()
    except ValueError as error:
        return refusal_outcome(
            plan, spec, f"design could not be assessed: {error}", dag_checks
        )
    if adapter.refusal:
        return refusal_outcome(plan, spec, adapter.refusal, dag_checks)
    report = {
        "status": select.IDENTIFIED,
        "strategy": strategy,
        "estimand": spec.estimand,
        "estimate": None,
        "design": adapter.design,
        "multiverse_not_run": list(plan.multiverse),
        "notes": [NOT_ESTIMATED],
    }
    if dag_checks is not None:
        report["dag"] = dict(dag_checks)
    decision = tier.Assignment(
        qualification.Tier.UNVERIFIED,
        "identified",
        dict.fromkeys(tier.CONDITIONS, False),
        (NOT_ESTIMATED,),
        "downgrade_tier",
    )
    return assemble(
        plan,
        spec,
        report,
        {"ran": False, "note": NOT_ESTIMATED},
        {"holds": None, "note": NOT_ESTIMATED},
        decision,
        strategy,
        needed_assumptions(strategy, spec),
    )


def verdict_of(spec: causal.Question) -> dict[str, Any]:
    """States the primary hypothesis `theta = 0` in words.

    Args:
        spec: The charter's causal block.

    Returns:
        The hypothesis sentence and how the result is labelled.
    """
    return {
        "hypothesis": (
            f"{spec.estimand} of {spec.treatment} on {spec.outcome} = 0"
        ),
        "labelled": "causal, under the attested assumptions",
    }


def needed_assumptions(strategy: str, spec: causal.Question) -> tuple[str, ...]:
    """Lists the attested assumptions the chosen strategy relies on.

    Args:
        strategy: Registry name of the strategy.
        spec: The charter's causal block.

    Returns:
        The strategy's required assumptions, with the unconfoundedness
        requirement replaced by the charter's matching assumption words.
    """
    required = select.STRATEGIES[strategy].assumptions
    out: list[str] = []
    for word in required:
        if word == "unconfoundedness_given":
            out.extend(
                item
                for item in spec.assumptions
                if item.startswith(causal.UNCONFOUNDEDNESS_PREFIX)
            )
        else:
            out.append(word)
    return tuple(out)
