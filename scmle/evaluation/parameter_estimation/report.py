"""The serializable estimation result and one estimator's fit."""

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
from scmle.evaluation.envelope import rename
from scmle.evaluation.envelope import unfold
from scmle.evaluation.parameter_estimation.config import Config
from scmle.evaluation.parameter_estimation.declaration import Estimand
from scmle.evaluation.parameter_estimation.declaration import Estimate
from scmle.evaluation.parameter_estimation.declaration import Prior

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
    """Complete, serializable evidence for one parameter-estimation run.

    Attributes:
        estimand: The quantity the run claims to estimate.
        policy: The policy the run used.
        parameters: Coefficient estimates, in coefficient order.
        covariance: Symmetric covariance matrix of the coefficients, in the
            same order as ``parameters``.
        rows: Rows supplied, used by the fit, and removed by the
            missing-value policy.
        diagnostics: Numerical and model diagnostics of the fit.
        sensitivity: Bootstrap and other sensitivity evidence.
        reference: Result of the independent reference computation.
        digests: The lowercase SHA-256 hex digest of the raw input, under
            `frame`.
    """

    __module__ = "scmle.evaluation.parameter_estimation"
    label = "estimation"
    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            change=functools.partial(fold, groups=NESTED),
            revert=functools.partial(unfold, groups=NESTED),
        ),
        records.nest(
            {},
            parts={
                "estimand": Estimand.migrations[0],
                "policy": Config.migrations[0],
                "parameters": Estimate.migrations[0],
            },
        ),
    )

    estimand: Estimand
    policy: Config
    parameters: tuple[Estimate, ...]
    covariance: tuple[tuple[float, ...], ...]
    rows: Counts
    diagnostics: Mapping[str, Any]
    sensitivity: Mapping[str, Any]
    reference: Mapping[str, Any]
    digests: Mapping[str, str]

    def __post_init__(self) -> None:
        """Validates the report shape before it can be persisted or used.

        Raises:
            TypeError: If the estimand, policy or rows has the wrong type,
                or if diagnostics, sensitivity or reference is not a
                mapping.
            ValueError: If there are no parameters or their names repeat, if
                the covariance is not square, finite and symmetric over the
                parameters, or if the digests are not exactly the input's
                lowercase SHA-256 digest.
        """
        if not isinstance(self.estimand, Estimand):
            raise errors.KindError("estimand must be Estimand")
        if not isinstance(self.policy, Config):
            raise errors.KindError("policy must be Config")
        if not self.parameters:
            raise errors.InputError("estimation result needs parameters")
        names = tuple(item.name for item in self.parameters)
        if len(set(names)) != len(names):
            raise errors.InputError("parameter names must be unique")
        self.invertible()
        if not isinstance(self.rows, Counts):
            raise errors.KindError("rows must be Counts")
        for mapping_name, mapping_value in (
            ("diagnostics", self.diagnostics),
            ("sensitivity", self.sensitivity),
            ("reference", self.reference),
        ):
            if not isinstance(mapping_value, Mapping):
                raise errors.KindError(f"{mapping_name} must be a mapping")
        if set(self.digests) != {"frame"}:
            raise errors.InputError("digests must hold exactly the frame")
        lowercase(self.digests["frame"], "digests.frame")

    def invertible(self) -> None:
        """Checks that the covariance is square, finite and symmetric.

        Raises:
            ValueError: If the covariance is not square over all parameters,
                has a non-finite value, or is not symmetric.
        """
        expected = len(self.parameters)
        if len(self.covariance) != expected or any(
            len(row) != expected for row in self.covariance
        ):
            raise errors.InputError(
                "covariance must be square over all parameters"
            )
        covariance = numpy.asarray(self.covariance, dtype=float)
        if not numpy.isfinite(covariance).all():
            raise errors.InputError("covariance must contain finite values")
        if not numpy.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
            raise errors.InputError("covariance must be symmetric")

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible estimation report.

        Returns:
            A dictionary holding every field, with nested objects converted
            to dictionaries and matrices to lists.
        """
        return {
            "estimand": self.estimand.to_dict(),
            "policy": self.policy.to_dict(),
            "parameters": [item.to_dict() for item in self.parameters],
            "covariance": [list(row) for row in self.covariance],
            "rows": dataclasses.asdict(self.rows),
            "diagnostics": dict(self.diagnostics),
            "sensitivity": dict(self.sensitivity),
            "reference": dict(self.reference),
            "digests": dict(self.digests),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Estimation":
        """Rebuilds a result from its JSON-compatible report.

        Args:
            record: Dictionary in the form produced by `to_dict`.

        Returns:
            The estimation result described by the record.  A missing key
            or a component that fails its own validation propagates the
            error of the lookup or of that component's constructor.
        """
        # The parts were written in single words at version 3. They are read
        # at version 2, where the assumption groups are flat, and the keys
        # whose fields took single words are renamed back to the fields.
        record = records.downgrade(cls.migrations[1:], dict(record))
        estimand_record = record["estimand"]
        policy_record = record["policy"]
        parameters = tuple(
            Estimate(
                item["name"],
                float(item["estimate"]),
                float(item["standard_error"]),
                (float(item["interval"][0]), float(item["interval"][1])),
                float(item["p_value"]),
            )
            for item in record["parameters"]
        )
        estimand_values = rename(Estimand, estimand_record)
        estimand_values["features"] = tuple(estimand_values["features"])
        estimand_values["assumptions"] = tuple(estimand_record["assumptions"])
        policy_values = rename(Config, policy_record)
        for policy_field in (
            "causal_assumptions",
            "continuous_treatment_assumptions",
            "hierarchical_assumptions",
        ):
            if policy_field in policy_values:
                policy_values[policy_field] = tuple(policy_values[policy_field])
        if "bounds" in policy_values:
            policy_values["bounds"] = tuple(
                float(value) for value in policy_values["bounds"]
            )
        if policy_values.get("prior") is not None:
            prior_values = dict(policy_values["prior"])
            prior_values["mean"] = tuple(prior_values["mean"])
            prior_values["precision"] = tuple(
                tuple(row) for row in prior_values["precision"]
            )
            policy_values["prior"] = Prior(**prior_values)
        return cls(
            Estimand(**estimand_values),
            Config(**policy_values),
            parameters,
            tuple(
                tuple(float(value) for value in row)
                for row in record["covariance"]
            ),
            Counts(**record["rows"]),
            dict(record["diagnostics"]),
            dict(record["sensitivity"]),
            dict(record["reference"]),
            {key: str(value) for key, value in record["digests"].items()},
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Fit:
    """The fitted quantities of one estimator, before they are reported.

    Attributes:
        names: Parameter names, one per coefficient.
        coefficients: Point estimates.
        covariance: Covariance matrix of the estimates.
        diagnostics: Estimator diagnostics, including the likelihood.
        reference: Independent reference calculation.
        likelihood: Likelihood family of the fitted model.
        pivot: Critical value that scales the interval half-widths.
        p_values: Two-sided p-value of each coefficient.
        standard_errors: Standard errors when the estimator sets them (the
            Bayesian estimators), else None to derive them from the
            covariance.
        interval_errors: Scales of the interval half-widths when the
            estimator sets them, else None to use the standard errors.
    """

    __module__ = "scmle.evaluation.parameter_estimation"

    names: tuple[str, ...]
    coefficients: numpy.ndarray
    covariance: numpy.ndarray
    diagnostics: dict[str, Any]
    reference: dict[str, Any]
    likelihood: str
    pivot: Any
    p_values: numpy.ndarray
    standard_errors: numpy.ndarray | None = None
    interval_errors: numpy.ndarray | None = None
