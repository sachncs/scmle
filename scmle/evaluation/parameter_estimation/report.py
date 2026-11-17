"""The serializable estimation result and one estimator's fit."""

from collections.abc import Mapping
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy

from scmle.contracts import errors
from scmle.evaluation.parameter_estimation.config import Config
from scmle.evaluation.parameter_estimation.declaration import BayesianPrior
from scmle.evaluation.parameter_estimation.declaration import Estimand
from scmle.evaluation.parameter_estimation.declaration import ParameterEstimate
from scmle.evaluation.parameter_estimation.declaration import SHA256_HEX_LENGTH


def created_temporary(temporary: Path | None) -> Path:
    """Returns the temporary report file, which must have been created.

    Args:
        temporary: The path of the temporary file, or None if none was made.

    Returns:
        `temporary`.

    Raises:
        RuntimeError: If no temporary file was created.
    """
    if temporary is None:
        raise RuntimeError("temporary estimation report was not created")
    return temporary


@dataclasses.dataclass(frozen=True, slots=True)
class Estimation:
    """Complete, serializable evidence for one parameter-estimation run.

    Attributes:
        estimand: The quantity the run claims to estimate.
        policy: The policy the run used.
        parameters: Coefficient estimates, in coefficient order.
        covariance: Symmetric covariance matrix of the coefficients, in the
            same order as ``parameters``.
        rows_input: Number of rows supplied.
        rows_used: Number of rows the fit used.
        rows_dropped: Number of rows removed by the missing-value policy.
        diagnostics: Numerical and model diagnostics of the fit.
        sensitivity: Bootstrap and other sensitivity evidence.
        reference: Result of the independent reference computation.
        raw_input_fingerprint: Lowercase SHA-256 hex digest of the raw input.
    """

    __module__ = "scmle.evaluation.parameter_estimation"

    estimand: Estimand
    policy: Config
    parameters: tuple[ParameterEstimate, ...]
    covariance: tuple[tuple[float, ...], ...]
    rows_input: int
    rows_used: int
    rows_dropped: int
    diagnostics: Mapping[str, Any]
    sensitivity: Mapping[str, Any]
    reference: Mapping[str, Any]
    raw_input_fingerprint: str

    def __post_init__(self) -> None:
        """Validates the report shape before it can be persisted or used.

        Raises:
            TypeError: If the estimand or policy has the wrong type, or if
                diagnostics, sensitivity or reference is not a mapping.
            ValueError: If there are no parameters or their names repeat, if
                the covariance is not square, finite and symmetric over the
                parameters, if a row count is negative or not an integer, if
                used plus dropped rows differ from input rows, or if the
                fingerprint is not a lowercase SHA-256 digest.
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
        self.conserve()
        for mapping_name, mapping_value in (
            ("diagnostics", self.diagnostics),
            ("sensitivity", self.sensitivity),
            ("reference", self.reference),
        ):
            if not isinstance(mapping_value, Mapping):
                raise errors.KindError(f"{mapping_name} must be a mapping")
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

    def conserve(self) -> None:
        """Checks the row counts and that used plus dropped rows add up.

        Raises:
            ValueError: If a row count is negative or not an integer, or the
                used and dropped rows do not sum to the input rows.
        """
        counts: tuple[tuple[str, object], ...] = (
            ("rows_input", self.rows_input),
            ("rows_used", self.rows_used),
            ("rows_dropped", self.rows_dropped),
        )
        for name, count in counts:
            if not isinstance(count, int) or count < 0:
                raise errors.InputError(
                    f"{name} must be a non-negative integer"
                )
        if self.rows_used + self.rows_dropped != self.rows_input:
            raise errors.InputError(
                "row accounting must reconcile to rows_input"
            )

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
            "rows_input": self.rows_input,
            "rows_used": self.rows_used,
            "rows_dropped": self.rows_dropped,
            "diagnostics": dict(self.diagnostics),
            "sensitivity": dict(self.sensitivity),
            "reference": dict(self.reference),
            "raw_input_fingerprint": self.raw_input_fingerprint,
        }

    def digest(self) -> str:
        """Returns the content digest of the canonical estimation report."""
        content = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def write(self, path: str | Path) -> str:
        """Atomically writes a digest-bound estimation report.

        The report is written to a temporary file in the destination
        directory, flushed to disk and renamed over the destination. The file
        holds an envelope with `schema_version` 1, the report and its digest.
        On any failure the temporary file is removed and the error
        propagates.

        Args:
            path: Destination file; missing parent directories are created.

        Returns:
            The SHA-256 hex digest of the canonical report.

        Raises:
            RuntimeError: If the temporary file path was not recorded; the
                check cannot trigger in practice.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "schema_version": 1,
            "report": self.to_dict(),
            "digest": self.digest(),
        }
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(envelope, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            created_temporary(temporary).replace(destination)
        except BaseException:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise
        return self.digest()

    @classmethod
    def read(cls, path: str | Path) -> "Estimation":
        """Reads a report and rejects a modified or malformed envelope.

        Args:
            path: File written by `write`.

        Returns:
            The rebuilt estimation result.

        Raises:
            ValueError: If the schema version is not 1, the report is not a
                mapping, the stored digest does not match the report, or the
                rebuilt result does not reproduce the digest. A file that is
                not valid JSON raises `json.JSONDecodeError`, a subclass.
        """
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
        if envelope.get("schema_version") != 1:
            raise errors.InputError("unsupported estimation report schema")
        record = envelope.get("report")
        if not isinstance(record, Mapping):
            raise errors.InputError("estimation report must contain a mapping")
        canonical = json.dumps(
            record, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        if envelope.get("digest") != digest:
            raise errors.InputError("estimation report digest mismatch")
        report = cls.from_dict(record)
        if report.digest() != digest:
            raise errors.InputError(
                "estimation report canonicalization mismatch"
            )
        return report

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
        estimand_record = record["estimand"]
        policy_record = record["policy"]
        parameters = tuple(
            ParameterEstimate(
                item["name"],
                float(item["estimate"]),
                float(item["standard_error"]),
                (float(item["interval"][0]), float(item["interval"][1])),
                float(item["p_value"]),
            )
            for item in record["parameters"]
        )
        estimand_values = dict(estimand_record)
        estimand_values["feature_names"] = tuple(
            estimand_record["feature_names"]
        )
        estimand_values["assumptions"] = tuple(estimand_record["assumptions"])
        policy_values = dict(policy_record)
        for policy_field in (
            "causal_assumptions",
            "continuous_treatment_assumptions",
            "hierarchical_assumptions",
        ):
            if policy_field in policy_values:
                policy_values[policy_field] = tuple(policy_values[policy_field])
        if "reml_variance_bounds" in policy_values:
            policy_values["reml_variance_bounds"] = tuple(
                float(value) for value in policy_values["reml_variance_bounds"]
            )
        if policy_values.get("prior") is not None:
            prior_values = dict(policy_values["prior"])
            prior_values["mean"] = tuple(prior_values["mean"])
            prior_values["precision"] = tuple(
                tuple(row) for row in prior_values["precision"]
            )
            policy_values["prior"] = BayesianPrior(**prior_values)
        return cls(
            Estimand(**estimand_values),
            Config(**policy_values),
            parameters,
            tuple(
                tuple(float(value) for value in row)
                for row in record["covariance"]
            ),
            int(record["rows_input"]),
            int(record["rows_used"]),
            int(record["rows_dropped"]),
            dict(record["diagnostics"]),
            dict(record["sensitivity"]),
            dict(record["reference"]),
            str(record["raw_input_fingerprint"]),
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
