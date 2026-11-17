"""The serializable evidence of one longitudinal estimation run."""

from collections.abc import Mapping
import dataclasses
import functools
from typing import Any
from typing import ClassVar

import numpy

from scmle.contracts import errors
from scmle.contracts import records
from scmle.evaluation.envelope import Counts
from scmle.evaluation.envelope import Envelope
from scmle.evaluation.envelope import fold
from scmle.evaluation.envelope import lowercase
from scmle.evaluation.envelope import unfold
from scmle.evaluation.longitudinal.declaration import Config
from scmle.evaluation.longitudinal.declaration import Estimand
from scmle.evaluation.longitudinal.declaration import PeriodATE

NESTED = {
    "rows": {
        "rows_input": "received",
        "rows_used": "used",
        "rows_dropped": "dropped",
    },
    "digests": {"raw_input_fingerprint": "frame"},
}


@dataclasses.dataclass(frozen=True, slots=True)
class Estimation(Envelope):
    """Complete serializable evidence for one longitudinal estimation run.

    Attributes:
        estimand: The declared estimand.
        policy: The policy the run used.
        periods: Per-period estimates, in time order, with unique labels.
        covariance: Symmetric covariance matrix of the period estimates, one
            row and column per period in the order of `periods`.
        rows: Rows in the supplied frame, analysed, and removed by the
            missingness policy.
        diagnostics: Method, overlap, and effective-sample-size details.
        limitations: Statements of what the estimates do not cover.
        reference: Record of the independent recomputation of the period
            estimates.
        digests: The lowercase SHA-256 digest of the supplied frame, under
            `frame`.
    """

    __module__ = "scmle.evaluation.longitudinal"
    label = "longitudinal"
    migration: ClassVar[records.Migration] = records.Migration(
        change=functools.partial(fold, groups=NESTED),
        revert=functools.partial(unfold, groups=NESTED),
    )

    estimand: Estimand
    policy: Config
    periods: tuple[PeriodATE, ...]
    covariance: tuple[tuple[float, ...], ...]
    rows: Counts
    diagnostics: Mapping[str, Any]
    limitations: Mapping[str, Any]
    reference: Mapping[str, Any]
    digests: Mapping[str, str]

    def __post_init__(self) -> None:
        """Validates the report shape before serialization.

        Raises:
            ValueError: If there are no periods, period labels repeat, the
                covariance is not square over the periods, not finite, or
                not symmetric, or the digests are not exactly the frame's
                lowercase SHA-256 digest.
            TypeError: If `rows` is not `Counts`, or `diagnostics`,
                `limitations`, or `reference` is not a mapping.
        """
        if not self.periods:
            raise errors.InputError(
                "longitudinal result needs at least one period"
            )
        labels = tuple(item.period for item in self.periods)
        if len(set(labels)) != len(labels):
            raise errors.InputError("period labels must be unique")
        self.invertible()
        if not isinstance(self.rows, Counts):
            raise errors.KindError("rows must be Counts")
        for name, item_value in (
            ("diagnostics", self.diagnostics),
            ("limitations", self.limitations),
            ("reference", self.reference),
        ):
            if not isinstance(item_value, Mapping):
                raise errors.KindError(f"{name} must be a mapping")
        if set(self.digests) != {"frame"}:
            raise errors.InputError("digests must hold exactly the frame")
        lowercase(self.digests["frame"], "digests.frame")

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
            "rows": dataclasses.asdict(self.rows),
            "diagnostics": dict(self.diagnostics),
            "limitations": dict(self.limitations),
            "reference": dict(self.reference),
            "digests": dict(self.digests),
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
            Counts(**record["rows"]),
            dict(record["diagnostics"]),
            dict(record["limitations"]),
            dict(record["reference"]),
            {key: str(value) for key, value in record["digests"].items()},
        )
