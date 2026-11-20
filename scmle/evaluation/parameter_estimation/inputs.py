"""Input checks, missing-value handling and input fingerprints."""

from collections.abc import Sequence
import hashlib
import json
from typing import Any

import numpy
import pandas

from scmle.contracts import errors
from scmle.evaluation.parameter_estimation.config import Config
from scmle.evaluation.parameter_estimation.declaration import CAUSAL_ESTIMATORS
from scmle.evaluation.parameter_estimation.declaration import (
    COVARIANCE_CONDITION_THRESHOLD,
)
from scmle.evaluation.parameter_estimation.declaration import Estimand
from scmle.evaluation.parameter_estimation.declaration import MATRIX_NDIM
from scmle.evaluation.parameter_estimation.declaration import MIN_CLUSTERS
from scmle.evaluation.parameter_estimation.declaration import MissingPolicy
from scmle.evaluation.parameter_estimation.declaration import (
    SMALL_CLUSTER_COUNT,
)


def admit(
    matrix: numpy.ndarray,
    values: numpy.ndarray,
    input_clusters: numpy.ndarray | None,
    policy: Config,
    estimand: Estimand,
) -> None:
    """Checks the inputs against the policy before any row is dropped.

    Args:
        matrix: Complete feature matrix.
        values: Complete outcome vector.
        input_clusters: Validated cluster labels, or None.
        policy: Estimator policy.
        estimand: Scientific estimand.

    Raises:
        ValueError: If the row counts differ, cluster labels are missing or
            not allowed, the random slope lies outside the features, or the
            estimand likelihood does not match the estimator.
    """
    if matrix.shape[0] != len(values):
        raise errors.InputError(
            "features and outcome must have the same row count"
        )
    if policy.covariance == "clustered" and input_clusters is None:
        raise errors.InputError("clustered covariance needs cluster labels")
    if policy.estimator == "hierarchical_linear" and input_clusters is None:
        raise errors.InputError("hierarchical_linear requires cluster labels")
    if policy.slope is not None and policy.slope >= matrix.shape[1]:
        raise errors.InputError(
            "random_slope_position is outside the feature matrix"
        )
    if (
        policy.estimator in {"ipw_ate", "aipw_ate", "continuous_ipw_ate"}
        and input_clusters is not None
        and policy.covariance != "clustered"
    ):
        raise errors.InputError(
            f"{policy.estimator} cluster labels require covariance='clustered'"
        )
    expected_likelihood = {
        "linear": "gaussian",
        "logistic": "bernoulli",
        "bayesian_linear": "gaussian",
        "bayesian_ate": "gaussian",
        "ipw_ate": "gaussian",
        "aipw_ate": "gaussian",
        "continuous_ipw_ate": "gaussian",
        "hierarchical_linear": "gaussian",
    }[policy.estimator]
    if estimand.likelihood != expected_likelihood:
        raise errors.InputError(
            "estimand likelihood does not match estimator: "
            f"expected {expected_likelihood}, got {estimand.likelihood}"
        )


def screen(
    matrix: numpy.ndarray,
    values: numpy.ndarray,
    clusters: Sequence[Any] | None,
    policy: Config,
) -> None:
    """Checks the rows that remain after missing values were handled.

    Args:
        matrix: Feature matrix of the kept rows.
        values: Outcome vector of the kept rows.
        clusters: The cluster labels as the caller supplied them.
        policy: Estimator policy.

    Raises:
        ValueError: If a logistic outcome is not 0 or 1, a Bayesian estimator
            was given clusters, or the treatment position of a causal
            estimator is missing or outside the feature matrix.
    """
    if policy.estimator == "logistic" and not set(
        numpy.unique(values)
    ).issubset({0.0, 1.0}):
        raise errors.InputError("logistic outcome must contain only 0 and 1")
    if (
        policy.estimator in {"bayesian_linear", "bayesian_ate"}
        and clusters is not None
    ):
        raise errors.InputError(
            f"{policy.estimator} estimation does not accept clusters"
        )
    if policy.estimator in CAUSAL_ESTIMATORS:
        if policy.treatment is None:
            raise errors.InputError(
                f"{policy.estimator} requires treatment_position"
            )
        if not 0 <= policy.treatment < matrix.shape[1]:
            raise errors.InputError(
                "treatment_position is outside the feature matrix"
            )


def imprint(
    features: Any,
    outcome: Any,
    clusters: Any | None = None,
) -> str:
    """Returns a deterministic fingerprint of the estimator's raw inputs.

    Numeric values are hashed after the same float conversion used by the
    estimator. Cluster labels retain their value types and row order. The
    fingerprint covers inputs before missing-row filtering so it can identify
    the complete evidence source independently of the fitted result.

    Args:
        features: Feature matrix, converted to float.
        outcome: Response values, flattened after conversion to float.
        clusters: Optional cluster labels, hashed as objects.

    Returns:
        The SHA-256 hex digest over the shape, dtype and values of each
        input.
    """
    matrix = numpy.asarray(features, dtype=float)
    values = numpy.asarray(outcome, dtype=float).reshape(-1)
    digest = hashlib.sha256()
    absorb(digest, "features", matrix)
    absorb(digest, "outcome", values)
    if clusters is None:
        digest.update(b"clusters:none")
    else:
        cluster_values = numpy.asarray(clusters, dtype=object)
        absorb(digest, "clusters", cluster_values)
    return digest.hexdigest()


def absorb(digest: Any, name: str, values: numpy.ndarray) -> None:
    """Adds one shape-, dtype-, and value-bound input to a digest.

    Args:
        digest: Hash object updated in place.
        name: Label bound into the descriptor of this input.
        values: Array whose shape, dtype and contents are hashed.
    """
    array = numpy.asarray(values)
    descriptor = json.dumps(
        {
            "name": name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(descriptor).to_bytes(8, "big"))
    digest.update(descriptor)
    if array.dtype.kind in "biufc":
        canonical = numpy.ascontiguousarray(
            array.astype(array.dtype.newbyteorder("<"), copy=False)
        )
        values_bytes = canonical.tobytes()
    else:
        value_hashes = pandas.util.hash_pandas_object(
            pandas.Series(array.reshape(-1)), index=False
        ).to_numpy(dtype="<u8")
        type_names = [
            f"{type(value).__module__}.{type(value).__qualname__}"
            for value in array.reshape(-1).tolist()
        ]
        values_bytes = json.dumps(
            {
                "types": type_names,
                "hashes": value_hashes.tolist(),
            },
            separators=(",", ":"),
        ).encode("utf-8")
    digest.update(len(values_bytes).to_bytes(8, "big"))
    digest.update(values_bytes)


def check(matrix: numpy.ndarray, columns: int, name: str) -> None:
    """Validates a numeric two-dimensional matrix.

    Args:
        matrix: Matrix to check.
        columns: Required number of columns.
        name: Matrix name used in the error message.

    Raises:
        ValueError: If the matrix is not two-dimensional with `columns`
            columns, or does not have more rows than `columns`.
    """
    if matrix.ndim != MATRIX_NDIM or matrix.shape[1] != columns:
        raise errors.InputError(f"{name} must have shape (rows, {columns})")
    if matrix.shape[0] <= columns:
        raise errors.InputError("estimation needs more rows than parameters")


def validate(clusters: Sequence[Any] | None, rows: int) -> numpy.ndarray | None:
    """Validates cluster-label shape before missingness processing.

    Args:
        clusters: Cluster labels, or None when the data are not clustered.
        rows: Number of rows the labels must cover.

    Returns:
        The labels as a one-dimensional object array, or None when
        `clusters` is None.

    Raises:
        ValueError: If the labels are not one-dimensional or their length
            differs from `rows`.
    """
    if clusters is None:
        return None
    cluster_values = numpy.asarray(clusters, dtype=object)
    if cluster_values.ndim != 1:
        raise errors.InputError("clusters must have shape (rows,)")
    if len(cluster_values) != rows:
        raise errors.InputError(
            "clusters must have the same row count as outcome"
        )
    return cluster_values


def condition(covariance: numpy.ndarray, dimension: int) -> dict[str, Any]:
    """Validates covariance shape and reports rank and numerical conditioning.

    Args:
        covariance: Coefficient covariance matrix.
        dimension: Required number of rows and columns.

    Returns:
        A dictionary with the identification ``status`` (``identified``,
        ``ill_conditioned`` or ``rank_deficient``), ``dimension``, numerical
        ``rank``, ``condition_number`` (None when the smallest singular value
        is zero) and the ``condition_number_threshold`` used.

    Raises:
        ValueError: If the matrix is not `dimension` by `dimension`, has a
            non-finite entry, or is not symmetric.
    """
    matrix = numpy.asarray(covariance, dtype=float)
    expected_shape = (dimension, dimension)
    if matrix.shape != expected_shape:
        raise errors.InputError(
            f"covariance must have shape ({dimension}, {dimension})"
        )
    if not numpy.isfinite(matrix).all():
        raise errors.InputError("covariance must contain finite values")
    if not numpy.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise errors.InputError("covariance must be symmetric")
    singular_values = numpy.linalg.svd(matrix, compute_uv=False)
    scale = float(singular_values[0]) if singular_values.size else 0.0
    rank_tolerance = numpy.finfo(float).eps * max(expected_shape) * scale
    rank = int(numpy.sum(singular_values > rank_tolerance))
    smallest_singular_value = (
        float(singular_values[-1]) if singular_values.size else 0.0
    )
    condition_number = (
        float(scale / smallest_singular_value)
        if smallest_singular_value > 0.0
        else None
    )
    if rank < dimension:
        status = "rank_deficient"
    elif (
        condition_number is None
        or condition_number > COVARIANCE_CONDITION_THRESHOLD
    ):
        status = "ill_conditioned"
    else:
        status = "identified"
    return {
        "status": status,
        "dimension": dimension,
        "rank": rank,
        "condition_number": condition_number,
        "condition_number_threshold": COVARIANCE_CONDITION_THRESHOLD,
    }


def prepare(
    matrix: numpy.ndarray,
    values: numpy.ndarray,
    missing: MissingPolicy,
    clusters: Any | None,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray | None, int]:
    """Applies the declared missingness policy and keeps cluster labels aligned.

    Args:
        matrix: Feature matrix.
        values: Outcome vector.
        missing: ``reject`` raises when any row is incomplete;
            ``complete_case`` drops incomplete rows.
        clusters: Optional cluster labels, one per row.

    Returns:
        The kept feature rows, the kept outcomes, the kept cluster labels
        (None when `clusters` is None) and the number of rows dropped.  A row
        is dropped when any feature or the outcome is not finite or its
        cluster label is missing.

    Raises:
        ValueError: If cluster labels are not scalar, if rows are incomplete
            and the policy is ``reject``, or if no more than the number of
            columns plus one complete rows remain.
    """
    valid = numpy.isfinite(matrix).all(axis=1) & numpy.isfinite(values)
    if clusters is not None:
        cluster_array = numpy.asarray(clusters, dtype=object)
        cluster_missing = numpy.asarray(pandas.isna(cluster_array))
        if cluster_missing.ndim != 1:
            raise errors.InputError("cluster labels must be scalar values")
        valid &= ~cluster_missing.astype(bool)
    else:
        cluster_array = None
    dropped = int((~valid).sum())
    if dropped and missing == "reject":
        raise errors.InputError(
            "missing or non-finite values require an explicit policy"
        )
    if valid.sum() <= matrix.shape[1] + 1:
        raise errors.InputError("complete rows are insufficient for estimation")
    kept_clusters = None
    if cluster_array is not None:
        kept_clusters = cluster_array[valid]
    return matrix[valid], values[valid], kept_clusters, dropped


def clustering(
    clusters: numpy.ndarray | None,
) -> dict[str, Any]:
    """Returns explicit sampling-unit diagnostics for clustered covariance.

    Args:
        clusters: Cluster label of each row, or None.

    Returns:
        A dictionary with the method name, cluster count, cluster sizes in
        first-appearance order, smallest and largest size, fraction of
        singleton clusters, the sampling-unit declaration, and
        ``small_cluster_caution``, True when there are fewer than 30
        clusters.

    Raises:
        ValueError: If `clusters` is None, the labels are not hashable, or
            there are fewer than two clusters.
    """
    if clusters is None:
        raise errors.InputError(
            "cluster labels are required for cluster diagnostics"
        )
    try:
        labels = tuple(dict.fromkeys(clusters.tolist()))
    except TypeError as error:
        raise errors.InputError(
            "cluster labels must be hashable scalar values"
        ) from error
    if len(labels) < MIN_CLUSTERS:
        raise errors.InputError(
            "clustered covariance needs at least two clusters"
        )
    sizes = [int(numpy.sum(clusters == label)) for label in labels]
    return {
        "method": "finite_sample_cluster_robust_sandwich",
        "cluster_count": len(labels),
        "cluster_sizes": sizes,
        "minimum_cluster_size": min(sizes),
        "maximum_cluster_size": max(sizes),
        "singleton_cluster_fraction": float(
            numpy.mean(numpy.asarray(sizes) == 1)
        ),
        "independent_sampling_unit": "declared_cluster_label",
        "small_cluster_caution": len(labels) < SMALL_CLUSTER_COUNT,
    }
