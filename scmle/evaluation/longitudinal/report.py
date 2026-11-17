"""The serializable evidence of one longitudinal estimation run."""

from collections.abc import Mapping
import dataclasses
from typing import Any

import numpy

from scmle.contracts import errors
from scmle.evaluation.envelope import Envelope
from scmle.evaluation.longitudinal.declaration import Config
from scmle.evaluation.longitudinal.declaration import Estimand
from scmle.evaluation.longitudinal.declaration import PeriodATE
from scmle.evaluation.longitudinal.declaration import SHA256_HEX_LENGTH


@dataclasses.dataclass(frozen=True, slots=True)
class Estimation(Envelope):
    """Complete serializable evidence for one longitudinal estimation run.

    Attributes:
        estimand: The declared estimand.
        policy: The policy the run used.
        periods: Per-period estimates, in time order, with unique labels.
        covariance: Symmetric covariance matrix of the period estimates, one
            row and column per period in the order of `periods`.
        rows_input: Number of rows in the supplied frame.
        rows_used: Number of rows analysed.
        rows_dropped: Number of rows removed by the missingness policy.
        diagnostics: Method, overlap, and effective-sample-size details.
        limitations: Statements of what the estimates do not cover.
        reference: Record of the independent recomputation of the period
            estimates.
        raw_input_fingerprint: Lowercase SHA-256 digest of the supplied
            frame.
    """

    __module__ = "scmle.evaluation.longitudinal"
    label = "longitudinal"

    estimand: Estimand
    policy: Config
    periods: tuple[PeriodATE, ...]
    covariance: tuple[tuple[float, ...], ...]
    rows_input: int
    rows_used: int
    rows_dropped: int
    diagnostics: Mapping[str, Any]
    limitations: Mapping[str, Any]
    reference: Mapping[str, Any]
    raw_input_fingerprint: str

    def __post_init__(self) -> None:
        """Validates the report shape before serialization.

        Raises:
            ValueError: If there are no periods, period labels repeat, the
                covariance is not square over the periods, not finite, or
                not symmetric, a count is not a non-negative integer, used
                plus dropped rows do not equal input rows, or the
                fingerprint is not a lowercase SHA-256 digest.
            TypeError: If `diagnostics`, `limitations`, or `reference` is
                not a mapping.
        """
        if not self.periods:
            raise errors.InputError(
                "longitudinal result needs at least one period"
            )
        labels = tuple(item.period for item in self.periods)
        if len(set(labels)) != len(labels):
            raise errors.InputError("period labels must be unique")
        self.invertible()
        self.conserve()
        for name, item_value in (
            ("diagnostics", self.diagnostics),
            ("limitations", self.limitations),
            ("reference", self.reference),
        ):
            if not isinstance(item_value, Mapping):
                raise errors.KindError(f"{name} must be a mapping")
        fingerprint: object = self.raw_input_fingerprint  # may be non-text
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != SHA256_HEX_LENGTH
            or any(
                character not in "0123456789abcdef" for character in fingerprint
            )
        ):
            raise errors.InputError(
                "raw_input_fingerprint must be a lowercase SHA-256 digest"
            )

    def invertible(self) -> None:
        """Checks that the covariance is square, finite and symmetric.

        Raises:
            ValueError: If the covariance is not square over the periods,
                has a non-finite value, or is not symmetric.
        """
        expected = len(self.periods)
        if len(self.covariance) != expected or any(
            len(row) != expected for row in self.covariance
        ):
            raise errors.InputError("covariance must be square over periods")
        covariance = numpy.asarray(self.covariance, dtype=float)
        if not numpy.isfinite(covariance).all():
            raise errors.InputError("covariance must contain finite values")
        if not numpy.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
            raise errors.InputError("covariance must be symmetric")

    def conserve(self) -> None:
        """Checks the row counts and that used plus dropped rows add up.

        Raises:
            ValueError: If a row count is not a non-negative integer, or the
                used and dropped rows do not sum to the input rows.
        """
        counts: tuple[tuple[str, object], ...] = (
            ("rows_input", self.rows_input),
            ("rows_used", self.rows_used),
            ("rows_dropped", self.rows_dropped),
        )
        for name, count in counts:
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
            ):
                raise errors.InputError(
                    f"{name} must be a non-negative integer"
                )
        if self.rows_used + self.rows_dropped != self.rows_input:
            raise errors.InputError(
                "row accounting must reconcile to rows_input"
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible longitudinal report.

        Returns:
            All report fields keyed by field name, with nested objects
            converted to dictionaries and tuples to lists.
        """
        return {
            "estimand": self.estimand.to_dict(),
            "policy": self.policy.to_dict(),
            "periods": [item.to_dict() for item in self.periods],
            "covariance": [list(row) for row in self.covariance],
            "rows_input": self.rows_input,
            "rows_used": self.rows_used,
            "rows_dropped": self.rows_dropped,
            "diagnostics": dict(self.diagnostics),
            "limitations": dict(self.limitations),
            "reference": dict(self.reference),
            "raw_input_fingerprint": self.raw_input_fingerprint,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Estimation":
        """Rebuilds a result from its JSON-compatible representation.

        Args:
            record: Mapping in the form produced by `to_dict`.

        Returns:
            The result built from the record; field validation runs on
            construction. A missing key raises `KeyError`.
        """
        estimand_values = dict(record["estimand"])
        estimand_values["covariates"] = tuple(estimand_values["covariates"])
        estimand_values["assumptions"] = tuple(estimand_values["assumptions"])
        periods = tuple(
            PeriodATE(
                period=str(item["period"]),
                estimate=float(item["estimate"]),
                standard_error=float(item["standard_error"]),
                interval=(
                    float(item["interval"][0]),
                    float(item["interval"][1]),
                ),
                rows=int(item["rows"]),
                treated_rows=int(item["treated_rows"]),
                control_rows=int(item["control_rows"]),
                propensity_min=float(item["propensity_min"]),
                propensity_max=float(item["propensity_max"]),
                fraction_clipped=float(item["fraction_clipped"]),
                effective_sample_size_treated=float(
                    item["effective_sample_size_treated"]
                ),
                effective_sample_size_control=float(
                    item["effective_sample_size_control"]
                ),
                converged=bool(item["converged"]),
                iterations=int(item["iterations"]),
            )
            for item in record["periods"]
        )
        policy = Config(**dict(record["policy"]))
        return cls(
            Estimand(**estimand_values),
            policy,
            periods,
            tuple(
                tuple(float(value) for value in row)
                for row in record["covariance"]
            ),
            int(record["rows_input"]),
            int(record["rows_used"]),
            int(record["rows_dropped"]),
            dict(record["diagnostics"]),
            dict(record["limitations"]),
            dict(record["reference"]),
            str(record["raw_input_fingerprint"]),
        )
