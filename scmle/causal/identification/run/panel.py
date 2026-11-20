"""Strategies for panels: difference in differences and synthetic control."""

from typing import Any

import pandas

from scmle.causal.identification import battery
from scmle.causal.identification import did
from scmle.causal.identification import synth
from scmle.causal.identification.run.estimator import Est
from scmle.causal.identification.run.estimator import Estimator
from scmle.causal.identification.run.estimator import waldify
from scmle.contracts import errors


class Differences(Estimator):
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
            raise errors.InputError("did needs unit, time and adoption")
        return s.unit, s.time, s.adoption

    def prepare(self) -> None:
        """Estimates on `X` to record the panel and the pre-trend test.

        The results go into `design` (default level); the estimate is dropped.
        """
        unit, time, adoption = self.columns()
        fit = did.estimate(self.x, unit, time, adoption, self.spec.outcome)
        self.design = {
            "groups": fit.groups,
            "n_never_treated": fit.untreated,
            "events": list(fit.events),
            "pretrend_p_on_x": fit.pretrend,
            "pre_periods_support_sensitivity": fit.drift is not None,
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
                "pretrend_p": fit.pretrend,
                "event_study": dict(
                    zip(fit.events, fit.coefficients, strict=True)
                ),
            },
        )

    def reckon(
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
        if fit.pretrend is not None:
            tests.append(battery.Test("pretrend_wald", "placebo", fit.pretrend))
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
        if fit.drift is None:
            return {"holds": None, "note": "fewer than two pre-periods"}
        m = spec.sensitivity.multiple
        low, high = fit.sensitivity(m)
        return {
            "holds": bool(low > 0 or high < 0),
            "m_max": m,
            "low": low,
            "high": high,
            "breakdown_m": fit.breakdown(),
            "max_pre_change": fit.drift,
            "note": "simplified relative-magnitudes bound, not HonestDiD",
        }


class Synth(Estimator):
    """Synthetic control with one treated unit; runs on `X` alone."""

    name = "synth"

    def __init__(self, *args: Any) -> None:
        """Stores the settings of the strategy and marks a single treated unit.

        Args:
            *args: The `Adapter` constructor arguments.
        """
        super().__init__(*args)
        self.unit = True

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
            raise errors.InputError("synth needs unit and time")
        frame = self.x
        flag = frame[spec.treatment].to_numpy(dtype=float) > 0
        treated_units = frame.loc[flag, spec.unit].unique()
        if len(treated_units) != 1:
            raise errors.InputError(
                "synthetic control needs exactly one treated unit"
            )
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
            "min_attainable_p": result.floor,
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

    def reckon(
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
