"""Strategies that use exogenous variation: an instrument or a cutoff."""

from typing import Any

import pandas

from scmle.assurance.charter import causal
from scmle.causal.identification import battery
from scmle.causal.identification import iv
from scmle.causal.identification import rd
from scmle.causal.identification.run.estimator import Est
from scmle.causal.identification.run.estimator import Estimator
from scmle.causal.identification.run.estimator import waldify
from scmle.contracts import errors


class Instrument(Estimator):
    """Just-identified instrumental variables."""

    name = "iv"

    def controls_columns(self) -> tuple[str, ...]:
        """Lists the covariates used as IV controls.

        Returns:
            The covariate columns declared in the unconfoundedness
            assumption, or an empty tuple.
        """
        return causal.adjusters(self.spec.assumptions) or ()

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
            raise errors.InputError("iv needs an instrument")
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

    def reckon(
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
        for control in self.spec.controls:
            if control.kind == "outcome":
                est = self.reckon(c, level, control.variable)
                test = waldify(
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

    def pretreatment(self) -> tuple[str, ...]:
        """Lists the predetermined covariates tested for a jump.

        Returns:
            The variables of the declared negative-control outcomes.
        """
        return tuple(
            c.variable for c in self.spec.controls if c.kind == "outcome"
        )

    def prepare(self) -> None:
        """Fixes the bandwidth and checks the density at the cutoff on `X`.

        The claim is refused when the density check rejects at `alpha_falsify`.

        Raises:
            ValueError: If the charter declares no running variable or cutoff.
        """
        spec = self.spec
        if not spec.running_variable or spec.cutoff is None:
            raise errors.InputError("rd needs a running variable and cutoff")
        fit = rd.estimate(
            self.x,
            spec.running_variable,
            spec.outcome,
            spec.cutoff,
            covariates=self.pretreatment(),
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
            raise errors.InputError("rd needs a running variable and cutoff")
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

    def reckon(
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
        fit = self.fit(c, level, covariates=self.pretreatment())
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
