"""Strategies that adjust for measured confounders or rely on randomization."""

from typing import Any
from typing import cast
from typing import TYPE_CHECKING

import pandas

from scmle.assurance.charter import causal
from scmle.causal.identification import aipw
from scmle.causal.identification import battery
from scmle.causal.identification import sensitivity
from scmle.causal.identification.run.estimator import Est
from scmle.causal.identification.run.estimator import Estimator
from scmle.causal.identification.run.estimator import waldify
from scmle.contracts import errors

if TYPE_CHECKING:
    import numpy


class Backdoor(Estimator):
    """AIPW with nuisances frozen on `X` and no cross-fitting."""

    name = "backdoor_aipw"

    def __init__(self, *args: Any) -> None:
        """Stores the settings, then reads the covariates from the charter.

        The nuisance models start unset.

        Args:
            *args: The `Adapter` constructor arguments.
        """
        super().__init__(*args)
        self.covariates = causal.adjusters(self.spec.assumptions) or ()
        self.nuisances: aipw.Nuisances | None = None

    def prepare(self) -> None:
        """Fits the frozen nuisances and checks overlap, from `X` only.

        The claim is refused when the overlap gate fails on `X` and the charter
        declares no trimming rule.
        """
        spec = self.spec
        self.nuisances = aipw.prefit(
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
            "overlap_on_x": aipw.propensities(e, t, matrix),
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
            raise errors.InputError("prepare() must run first")
        spec = self.spec
        result = aipw.estimate(
            self.nuisances,
            c,
            spec.treatment,
            spec.outcome,
            spec.estimand,
            level,
            spec.trim,
            spec.retained,
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

    def reckon(
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
        nuisances = aipw.prefit(
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
            spec.retained,
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
        for control in spec.controls:
            if control.kind == "outcome":
                est = self.reckon(c, level, control.variable)
                name = f"negative_outcome:{control.variable}"
            else:
                nuisances = aipw.prefit(
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
                    spec.retained,
                )
                est = Est(
                    found.estimate, found.se, found.low, found.high, found.n
                )
                name = f"negative_exposure:{control.variable}"
            test = waldify(est, name, f"negative_{control.kind}")
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
            raise errors.InputError("prepare() must run first")
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


class Experiment(Estimator):
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
        r = aipw.contrast(c, spec.treatment, spec.outcome, level)
        return Est(r.estimate, r.se, r.low, r.high, r.n, df=r.df)

    def reckon(
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
        r = aipw.contrast(
            c, self.spec.treatment, outcome or self.spec.outcome, level
        )
        return Est(r.estimate, r.se, r.low, r.high, r.n, df=r.df)

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
        for control in self.spec.controls:
            if control.kind != "outcome":
                continue
            est = self.reckon(c, level, control.variable)
            test = waldify(
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
