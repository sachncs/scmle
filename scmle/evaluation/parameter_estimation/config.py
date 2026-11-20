"""The numerical and uncertainty policy of an estimation run."""

import dataclasses
import functools
from typing import Any
from typing import ClassVar

from scmle.contracts import errors
from scmle.contracts import records
from scmle.evaluation.parameter_estimation.declaration import CAUSAL_ESTIMATORS
from scmle.evaluation.parameter_estimation.declaration import CausalAssumption
from scmle.evaluation.parameter_estimation.declaration import (
    ContinuousTreatmentAssumption,
)
from scmle.evaluation.parameter_estimation.declaration import CovarianceKind
from scmle.evaluation.parameter_estimation.declaration import (
    DEFAULT_CONTINUOUS_WEIGHT_CAP,
)
from scmle.evaluation.parameter_estimation.declaration import (
    DEFAULT_PROPENSITY_CLIP,
)
from scmle.evaluation.parameter_estimation.declaration import (
    DEFAULT_REML_VARIANCE_BOUNDS,
)
from scmle.evaluation.parameter_estimation.declaration import EstimatorKind
from scmle.evaluation.parameter_estimation.declaration import (
    HierarchicalAssumption,
)
from scmle.evaluation.parameter_estimation.declaration import HierarchicalMethod
from scmle.evaluation.parameter_estimation.declaration import MissingPolicy
from scmle.evaluation.parameter_estimation.declaration import Prior
from scmle.evaluation.parameter_estimation.declaration import (
    REQUIRED_BAYESIAN_ATE_ASSUMPTIONS,
)
from scmle.evaluation.parameter_estimation.declaration import (
    REQUIRED_CAUSAL_ASSUMPTIONS,
)
from scmle.evaluation.parameter_estimation.declaration import (
    REQUIRED_CONTINUOUS_TREATMENT_ASSUMPTIONS,
)
from scmle.evaluation.parameter_estimation.declaration import (
    REQUIRED_HIERARCHICAL_ASSUMPTIONS,
)
from scmle.evaluation.parameter_estimation.declaration import (
    REQUIRED_HIERARCHICAL_RANDOM_SLOPE_ASSUMPTIONS,
)
from scmle.evaluation.parameter_estimation.rules import Rules

ASSUMPTIONS = {
    "assumptions": {
        "causal_assumptions": "causal",
        "continuous_treatment_assumptions": "continuous",
        "hierarchical_assumptions": "hierarchical",
    }
}


def gather(layout: records.Layout) -> records.Layout:
    """Collects the assumption lists the estimator declares.

    Args:
        layout: The policy with the flat assumption lists.

    Returns:
        The policy with `assumptions.{causal, continuous, hierarchical}`;
        `assumptions` is left out when the estimator declares none.
    """
    layout = records.gather(layout, ASSUMPTIONS)
    if not layout["assumptions"]:
        del layout["assumptions"]
    return layout


@dataclasses.dataclass(frozen=True, slots=True)
class Config(Rules):
    """Numerical and uncertainty policy for an estimation run.

    Attributes:
        estimator: Estimator to run.
        covariance: Covariance estimator for the coefficients.
        confidence: Confidence level of the reported intervals, strictly
            between 0 and 1.
        missing: Whether missing values are rejected or rows with them are
            dropped.
        cluster: Name of the cluster column, or None when the data are
            not clustered.
        replicates: Number of bootstrap replicates for the
            sensitivity report; 0 disables the bootstrap.
        seed: Seed of the random number generator.
        iterations: Iteration limit of iterative fits.
        tolerance: Convergence tolerance of iterative fits.
        reference: Tolerance for agreement with the independent
            reference computation.
        prior: Conjugate prior for the Bayesian estimators, else None.
        treatment: Index of the treatment column among the features
            for the causal estimators, else None.
        clip: Propensity scores are clipped to
            ``[propensity_clip, 1 - propensity_clip]``; strictly between 0
            and 0.5.
        causal_assumptions: Declared assumptions for the binary-treatment
            causal estimators.
        continuous_treatment_assumptions: Declared assumptions for
            ``continuous_ipw_ate``.
        cap: Upper cap on the continuous-treatment weights;
            greater than 1.
        hierarchical_assumptions: Declared assumptions for
            ``hierarchical_linear``.
        folds: Number of cross-fitting folds for ``aipw_ate``;
            a positive integer.
        hierarchy: Variance-component method for
            ``hierarchical_linear``.
        bounds: Lower and upper bound of the variance
            components in the REML search.
        slope: Index of the feature carrying a random slope
            in ``hierarchical_linear``, or None for a random intercept only.
        migrations: The written layout's version steps, oldest first;
            version 2 writes `confidence`, `cluster`, `replicates`,
            `iterations`, `reference`, `treatment`, `clip`, `cap`, `folds`,
            `hierarchy`, `bounds`, `slope` and
            `assumptions.{causal, continuous, hierarchical}`.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={
                "confidence_level": "confidence",
                "cluster_name": "cluster",
                "bootstrap_replicates": "replicates",
                "max_iterations": "iterations",
                "reference_tolerance": "reference",
                "treatment_position": "treatment",
                "propensity_clip": "clip",
                "continuous_weight_cap": "cap",
                "cross_fitting_folds": "folds",
                "hierarchical_method": "hierarchy",
                "reml_variance_bounds": "bounds",
                "random_slope_position": "slope",
            },
            change=gather,
            revert=functools.partial(records.scatter, folds=ASSUMPTIONS),
        ),
    )

    __module__ = "scmle.evaluation.parameter_estimation"

    estimator: EstimatorKind
    covariance: CovarianceKind = "hc3"
    confidence: float = 0.95
    missing: MissingPolicy = "reject"
    cluster: str | None = None
    replicates: int = 0
    seed: int = 0
    iterations: int = 100
    tolerance: float = 1e-8
    reference: float = 1e-6
    prior: Prior | None = None
    treatment: int | None = None
    clip: float = DEFAULT_PROPENSITY_CLIP
    causal_assumptions: tuple[CausalAssumption, ...] = ()
    continuous_treatment_assumptions: tuple[
        ContinuousTreatmentAssumption, ...
    ] = ()
    cap: float = DEFAULT_CONTINUOUS_WEIGHT_CAP
    hierarchical_assumptions: tuple[HierarchicalAssumption, ...] = ()
    folds: int = 1
    hierarchy: HierarchicalMethod = "moment_gls"
    bounds: tuple[float, float] = DEFAULT_REML_VARIANCE_BOUNDS
    slope: int | None = None

    def __post_init__(self) -> None:
        """Validates the policy before data is touched.

        Raises:
            ValueError: If a field is outside its range, or if a field is
                set for an estimator that does not support it, or if the
                estimator's required prior, treatment position, cluster name,
                covariance kind or exact assumption set is missing or wrong.
        """
        self.vet()
        self.check()
        self.fence()
        self.relevant()
        if self.estimator in {"bayesian_linear", "bayesian_ate"}:
            self.bayesian()
        elif self.estimator in CAUSAL_ESTIMATORS:
            self.check_causal_policy()
        elif self.estimator == "hierarchical_linear":
            self.nesting()
        elif self.prior is not None:
            raise errors.InputError(
                "a Bayesian prior is supported only by bayesian_linear or "
                "bayesian_ate"
            )
        self.check_hierarchical_only_fields()

    def bayesian(self) -> None:
        """Validates the policy of a Bayesian estimator.

        Raises:
            ValueError: If the prior, covariance, bootstrap, treatment
                position, assumptions or cluster name do not suit the
                estimator.
        """
        if self.prior is None:
            raise errors.InputError(
                f"{self.estimator} estimation requires an explicit prior"
            )
        if self.covariance != "iid":
            raise errors.InputError(
                f"{self.estimator} estimation supports only iid covariance"
            )
        if self.replicates:
            raise errors.InputError(
                f"{self.estimator} bootstrap sensitivity is unsupported"
            )
        if self.estimator == "bayesian_ate":
            if self.treatment is None:
                raise errors.InputError(
                    "bayesian_ate requires treatment_position"
                )
            if (
                len(self.causal_assumptions)
                != len(set(self.causal_assumptions))
                or frozenset(self.causal_assumptions)
                != REQUIRED_BAYESIAN_ATE_ASSUMPTIONS
            ):
                raise errors.InputError(
                    "bayesian_ate requires consistency, "
                    "conditional_exchangeability, positivity, and "
                    "linearity assumptions"
                )
            if self.cluster is not None:
                raise errors.InputError(
                    "bayesian_ate does not support clustered data"
                )

    def check_causal_policy(self) -> None:
        """Validates the policy of a frequentist causal estimator.

        Raises:
            ValueError: If the prior, covariance, treatment position or
                assumption set does not suit the estimator.
        """
        if self.prior is not None:
            raise errors.InputError(
                f"a Bayesian prior is not valid for {self.estimator}"
            )
        if self.covariance == "hc3":
            raise errors.InputError(
                f"{self.estimator} supports only iid or clustered "
                "covariance; hc3 is unsupported"
            )
        if self.treatment is None:
            raise errors.InputError(
                f"{self.estimator} requires treatment_position"
            )
        if self.estimator == "continuous_ipw_ate":
            if self.causal_assumptions:
                raise errors.InputError(
                    "continuous_ipw_ate requires "
                    "continuous_treatment_assumptions"
                )
            if (
                len(self.continuous_treatment_assumptions)
                != len(set(self.continuous_treatment_assumptions))
                or frozenset(self.continuous_treatment_assumptions)
                != REQUIRED_CONTINUOUS_TREATMENT_ASSUMPTIONS
            ):
                raise errors.InputError(
                    "continuous_ipw_ate requires consistency, "
                    "conditional_exchangeability, "
                    "continuous_treatment_positivity, "
                    "linear_dose_response, and "
                    "normal_conditional_treatment_density assumptions"
                )
        elif (
            len(self.causal_assumptions) != len(set(self.causal_assumptions))
            or frozenset(self.causal_assumptions) != REQUIRED_CAUSAL_ASSUMPTIONS
        ):
            raise errors.InputError(
                f"{self.estimator} requires consistency, "
                "conditional_exchangeability, "
                "and positivity assumptions"
            )

    def nesting(self) -> None:
        """Validates the policy of the hierarchical estimator.

        Raises:
            ValueError: If the prior, covariance, cluster name or assumption
                set does not suit the estimator.
        """
        if self.prior is not None:
            raise errors.InputError(
                "a Bayesian prior is not valid for hierarchical_linear"
            )
        if self.covariance != "iid":
            raise errors.InputError(
                "hierarchical_linear supplies its own covariance model"
            )
        if not self.cluster:
            raise errors.InputError("hierarchical_linear requires cluster_name")
        required_hierarchical_assumptions = (
            REQUIRED_HIERARCHICAL_RANDOM_SLOPE_ASSUMPTIONS
            if self.slope is not None
            else REQUIRED_HIERARCHICAL_ASSUMPTIONS
        )
        if (
            len(self.hierarchical_assumptions)
            != len(set(self.hierarchical_assumptions))
            or frozenset(self.hierarchical_assumptions)
            != required_hierarchical_assumptions
        ):
            raise errors.InputError(
                "hierarchical_linear requires all assumptions for its "
                "declared random-effects structure"
            )

    def check_hierarchical_only_fields(self) -> None:
        """Rejects hierarchical-only fields set for another estimator.

        Raises:
            ValueError: If a hierarchical-only field is set for another
                estimator.
        """
        if self.estimator != "hierarchical_linear":
            if self.slope is not None:
                raise errors.InputError(
                    "random_slope_position is supported only by "
                    "hierarchical_linear"
                )
            if self.hierarchy != "moment_gls":
                raise errors.InputError(
                    "hierarchical_method is supported only by "
                    "hierarchical_linear"
                )
            if self.bounds != DEFAULT_REML_VARIANCE_BOUNDS:
                raise errors.InputError(
                    "reml_variance_bounds is supported only by "
                    "hierarchical_linear"
                )

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible policy metadata, in the current layout."""
        values = dataclasses.asdict(self)
        if self.prior is None:
            values.pop("prior")
        if self.estimator not in {
            "bayesian_ate",
            "ipw_ate",
            "aipw_ate",
            "continuous_ipw_ate",
        }:
            values.pop("treatment")
        if self.estimator not in {"bayesian_ate", "ipw_ate", "aipw_ate"}:
            values.pop("causal_assumptions")
        if self.estimator != "continuous_ipw_ate":
            values.pop("continuous_treatment_assumptions")
            values.pop("cap")
        if self.estimator not in {"ipw_ate", "aipw_ate"}:
            values.pop("clip")
        if self.estimator != "aipw_ate":
            values.pop("folds")
        if self.estimator != "hierarchical_linear":
            values.pop("hierarchical_assumptions")
            values.pop("hierarchy")
            values.pop("bounds")
            values.pop("slope")
        if self.prior is not None:
            values["prior"] = self.prior.to_dict()
        return records.upgrade(self.migrations, values, records.ORIGIN)
