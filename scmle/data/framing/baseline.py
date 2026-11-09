"""The current business process as the baseline, and the claim against it.

The baseline is the decision the business takes today, a column of past
decisions `b_i` joined to outcomes `y_i`. For a model rule `a_i` frozen on
`X` and a threshold `t` with weight `w = t / (1 - t)`,

    d_i(t)   = (a_i - b_i) * (y_i - (1 - y_i) w),
    Delta(t) = mean_i d_i(t) = NB_model(t) - NB_base(t).

`Delta(t)` is linear in `w`, so "Delta(t) > 0 on all of `[t_lo, t_hi]`" holds
if and only if it holds at both endpoints. The claim is the intersection of
the two endpoint nulls, each tested at `alpha_confirm` with a one-sided
cluster-robust t test (Berger, 1982, intersection-union: the size is at most
`alpha_confirm` with no multiplicity correction, at the cost of power; cited
from memory). Rows of one entity are not independent, so the standard error
is the cluster-robust sandwich over the grouping unit, with `G - 1` degrees
of freedom. It is asymptotic in the number of clusters; `MIN_CLUSTERS` is the
smallest count at which the measured size met its bound
(`benchmarks/qualification/framing_certify.py`), and a claim below it is not
issued.

The comparison is valid only if outcomes are observed independently of the
baseline's decision. `selective_labels` tests that. The hypothesis ledger is not
used here: its hypotheses are single-feature bivariate statistics. Instead a
`Evaluator` registers the cost model and the frozen rule by hash before
`C` is read and refuses a second evaluation.
"""

from collections.abc import Mapping
import dataclasses
import hashlib
from typing import Any

import numpy
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from scipy import stats

from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.contracts import signing
from scmle.data.framing import costs

DETECTOR_ID = "framing-selective-labels"
DEFECT_TYPE = "selective_labels"
LABEL_TOLERANCE = 0.2
LABEL_ALPHA = 0.01
MIN_ARM_ROWS = 5
MIN_CLUSTERS = 20
CHECK_PREFIX = "baseline"
CLAIM_ID = "baseline_vs_current_process"
NOT_COMPUTED = "not_computed"
MIN_VARIANCE_CLUSTERS = 2


class BaselineError(errors.ArtifactError):
    """The baseline claim was misused or cannot be made."""


@dataclasses.dataclass(frozen=True, slots=True)
class ClusterTest:
    """One-sided cluster-robust test of `mean(d) <= 0`.

    Attributes:
        mean: Mean of the per-row differences.
        se: Cluster-robust standard error of the mean.
        df: Degrees of freedom, the number of clusters minus one.
        statistic: The t statistic.
        p_value: One-sided p-value of `mean > 0`.
        n: Rows used.
        clusters: Clusters used.
    """

    mean: float
    se: float
    df: int
    statistic: float
    p_value: float
    n: int
    clusters: int


def vector(values: ArrayLike, name: str) -> NDArray[numpy.float64]:
    """Converts `values` to a non-empty 1-d float array.

    Args:
        values: The values to convert.
        name: Name of the argument, used in the error message.

    Returns:
        The values as a float array.

    Raises:
        ValueError: If the array is not 1-d or is empty.
    """
    array = numpy.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-d array")
    return array


def paired_difference(
    y: ArrayLike, a: ArrayLike, b: ArrayLike, t: float
) -> NDArray[numpy.float64]:
    """Computes the per-row gain `d_i(t)` of the model rule over the baseline.

    Args:
        y: Outcomes, 0 or 1.
        a: Actions of the model rule, 0 or 1.
        b: Decisions of the baseline, 0 or 1.
        t: Threshold, strictly between 0 and 1.

    Returns:
        The gain of each row, `(a - b) * (y - (1 - y) * t / (1 - t))`.

    Raises:
        ValueError: If `y`, `a` and `b` differ in length. Values that
            `costs.binary_vector` or `costs.odds` reject also raise it.
    """
    outcomes = costs.binary_vector(y, "y")
    model = costs.binary_vector(a, "a")
    base = costs.binary_vector(b, "b")
    if not outcomes.shape == model.shape == base.shape:
        raise ValueError("y, a and b must have the same length")
    w = costs.odds(t)
    return (model - base) * (outcomes - (1.0 - outcomes) * w)


def cluster_test(d: ArrayLike, clusters: ArrayLike) -> ClusterTest:
    """One-sided cluster-robust t test that the mean of `d` exceeds zero.

    Args:
        d: Per-row differences.
        clusters: Cluster label of each row.

    The size is not exactly the nominal level when clusters are few and of
    uneven size: with a random effect shared inside each cluster, simulated
    rejection rates at 0.05 were 0.056 for 20 clusters of 5 to 15 rows and
    0.057 for 30 clusters of 1 to 30 rows (20,000 datasets each). It is
    slightly liberal there, by about 0.006.

    Returns:
        The test, with `G - 1` degrees of freedom.

    Raises:
        ValueError: With fewer than two clusters or mismatched lengths.
    """
    values = vector(d, "d")
    codes = factorize(clusters)[0]
    if codes.shape != values.shape:
        raise ValueError("d and clusters must have the same length")
    g = int(numpy.max(codes)) + 1
    if g < MIN_VARIANCE_CLUSTERS:
        raise ValueError("at least two clusters are needed")
    n = values.size
    mean = float(values.mean())
    sums = numpy.bincount(codes, weights=values - mean, minlength=g)
    variance = g / (g - 1) * float(numpy.sum(sums**2)) / n**2
    se = float(numpy.sqrt(variance))
    if se == 0.0:
        statistic = (
            0.0 if mean == 0.0 else float(numpy.copysign(numpy.inf, mean))
        )
        p_value = 0.5 if mean == 0.0 else float(mean < 0)
    else:
        statistic = mean / se
        p_value = float(stats.t.sf(statistic, g - 1))
    return ClusterTest(mean, se, g - 1, statistic, p_value, n, g)


def factorize(clusters: ArrayLike) -> tuple[NDArray[numpy.intp], int]:
    """Codes the cluster labels as consecutive integers.

    Args:
        clusters: Cluster label of each row.

    Returns:
        The integer code of each row, in the sorted order of the labels, and
        the number of distinct clusters (0 when there are no rows).

    Raises:
        ValueError: If `clusters` is not 1-d.
    """
    labels = numpy.asarray(clusters)
    if labels.ndim != 1:
        raise ValueError("clusters must be 1-d")
    codes = numpy.unique(labels, return_inverse=True)[1]
    return codes.astype(numpy.intp), int(
        numpy.max(codes)
    ) + 1 if codes.size else 0


@dataclasses.dataclass(frozen=True, slots=True)
class Selectivity:
    """Whether outcome observation depends on the baseline decision.

    Attributes:
        flagged: Missingness differs by more than the tolerance, with
            one-sided Fisher p-value below the level, in either direction.
        missing_rate_b0: Share of rows with `b = 0` whose outcome is missing.
        missing_rate_b1: The same for `b = 1`.
        difference: `missing_rate_b0 - missing_rate_b1`.
        p_value: Fisher exact p-value for the direction observed.
        tolerance: Smallest absolute difference that counts.
        alpha: Level of the Fisher test.
    """

    flagged: bool
    missing_rate_b0: float
    missing_rate_b1: float
    difference: float
    p_value: float
    tolerance: float
    alpha: float

    def to_dict(self) -> dict[str, Any]:
        """Returns the finding as a JSON-ready record.

        Returns:
            A mapping of every field of the finding to its value.
        """
        return dataclasses.asdict(self)


def selective_labels(observed: ArrayLike, b: ArrayLike) -> Selectivity:
    """Flags an outcome that is observed depending on the baseline action.

    Args:
        observed: Whether each row's outcome is observed (True or 1).
        b: The baseline decision, 0 or 1.

    Returns:
        The finding. With fewer than `MIN_ARM_ROWS` rows in an arm it is not
        flagged and the p-value is 1, because nothing can be said.

    Raises:
        ValueError: If `observed` and `b` differ in length. A `b` that is
            not a non-empty 1-d array of zeros and ones also raises it.
    """
    seen = numpy.asarray(observed).astype(bool)
    base = costs.binary_vector(b, "b").astype(bool)
    if seen.shape != base.shape:
        raise ValueError("observed and b must have the same length")
    n1, n0 = int(base.sum()), int((~base).sum())
    if min(n1, n0) < MIN_ARM_ROWS:
        return Selectivity(
            flagged=False,
            missing_rate_b0=numpy.nan,
            missing_rate_b1=numpy.nan,
            difference=0.0,
            p_value=1.0,
            tolerance=LABEL_TOLERANCE,
            alpha=LABEL_ALPHA,
        )
    miss0 = int((~seen[~base]).sum())
    miss1 = int((~seen[base]).sum())
    rate0, rate1 = miss0 / n0, miss1 / n1
    difference = rate0 - rate1
    table = [[miss0, n0 - miss0], [miss1, n1 - miss1]]
    alternative = "greater" if difference >= 0 else "less"
    p_value = float(stats.fisher_exact(table, alternative=alternative)[1])
    flagged = abs(difference) >= LABEL_TOLERANCE and p_value <= LABEL_ALPHA
    return Selectivity(
        flagged, rate0, rate1, difference, p_value, LABEL_TOLERANCE, LABEL_ALPHA
    )


def profile_on_exploration(
    y: ArrayLike,
    b: ArrayLike,
    model: costs.DecisionCostModel,
    observed: ArrayLike | None = None,
) -> dict[str, Any]:
    """Measures the baseline on `X`: its net benefit against the two defaults.

    A `b` that is not a non-empty 1-d array of zeros and ones, or that
    differs in length from the observation mask, raises `ValueError`.

    Args:
        y: Outcomes (NaN where unobserved).
        b: The baseline decision.
        model: The cost model.
        observed: Observation mask; defaults to `y` not NaN.

    Returns:
        A JSON-ready profile with the selective-label finding, five
        thresholds spread evenly over the cost range, and the net benefit of
        acting on no row. The net benefit of acting on every observed row is
        added only when some outcome is observed. The baseline's
        own net benefit is the string `not_computed` (with a `note`) when
        labels are selective or no outcome is observed, because the observed
        rows are then a biased sample of the ones it acted on.
    """
    outcomes = numpy.asarray(y, dtype=float)
    mask = (
        ~numpy.isnan(outcomes) if observed is None else numpy.asarray(observed)
    )
    selectivity = selective_labels(mask, b)
    grid = numpy.linspace(model.t_lo, model.t_hi, 5)
    profile: dict[str, Any] = {
        "selective_labels": selectivity.to_dict(),
        "thresholds": [float(t) for t in grid],
        "net_benefit_none": [costs.net_benefit_none()] * len(grid),
    }
    base = numpy.asarray(b, dtype=float)[mask.astype(bool)]
    seen = outcomes[mask.astype(bool)]
    if seen.size:
        profile["net_benefit_all"] = [
            costs.net_benefit_all(seen, t) for t in grid
        ]
    if selectivity.flagged or not seen.size:
        profile["net_benefit_baseline"] = NOT_COMPUTED
        profile["note"] = (
            "outcome observation depends on the baseline decision: its net "
            "benefit is biased and not reported"
            if selectivity.flagged
            else "no outcome is observed"
        )
    else:
        profile["net_benefit_baseline"] = [
            costs.net_benefit(seen, base, t) for t in grid
        ]
    return profile


@dataclasses.dataclass(frozen=True, slots=True)
class Registration:
    """What is fixed before the confirmation set is read.

    Attributes:
        cost_model_hash: Hash of the cost model.
        rule_hash: Hash of the frozen action rule, as the caller computed it.
        alpha: Level of each endpoint test.
        t_lo: Lower endpoint of the range.
        t_hi: Upper endpoint of the range.
    """

    cost_model_hash: str
    rule_hash: str
    alpha: float
    t_lo: float
    t_hi: float

    @property
    def registration_hash(self) -> str:
        """The SHA-256 hex digest of the registration."""
        return hashlib.sha256(
            signing.canonical_payload(dataclasses.asdict(self))
        ).hexdigest()


def register(
    model: costs.DecisionCostModel,
    rule: Mapping[str, Any],
    alpha: float,
) -> Registration:
    """Freezes the claim: the cost model, the rule and the level.

    Args:
        model: The cost model, fixed in the charter.
        rule: A JSON-ready description of the action rule chosen on `X`.
        alpha: The confirmation level, in (0, 1).

    Returns:
        The registration, to be built before `C` is opened.

    Raises:
        ValueError: If `alpha` is not a finite number strictly between 0 and
            1.
    """
    level = qualification.probability(alpha, "alpha")
    if not 0.0 < level < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    rule_hash = hashlib.sha256(signing.canonical_payload(rule)).hexdigest()
    return Registration(
        model.model_hash, rule_hash, level, model.t_lo, model.t_hi
    )


def rule_digest(rule: Mapping[str, Any]) -> str:
    """Hashes a rule description the way `register` does.

    Args:
        rule: A JSON-ready description of the action rule.

    Returns:
        The SHA-256 hex digest of the canonical payload of `rule`.
    """
    return hashlib.sha256(signing.canonical_payload(rule)).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class Outcome:
    """Outcome of the one evaluation on `C`.

    Attributes:
        issued: Whether the claim is made (both endpoint nulls rejected).
        status: `issued`, `not_shown` or `not_issued`.
        reasons: Why the claim is not issued, if it is not.
        endpoints: One test per range endpoint (empty when not computed).
        p_value: Intersection-union p-value, the larger endpoint p-value.
        selectivity: The selective-label finding.
        registration_hash: Hash of the registration the run used.
        checks: One check per endpoint, on `C`.
        claim: The claim, tier `confirmed` only if issued and every input
            may support that tier.
    """

    issued: bool
    status: str
    reasons: tuple[str, ...]
    endpoints: tuple[ClusterTest, ...]
    p_value: float | None
    selectivity: Selectivity
    registration_hash: str
    checks: tuple[qualification.Check, ...]
    claim: qualification.Claim

    def to_dict(self) -> dict[str, Any]:
        """Returns the `baseline_report.json` document.

        Returns:
            A JSON-ready mapping of the status, the reasons, the endpoint
            tests, the selective-label finding, the checks, the claim and the
            statement of the guarantee.
        """
        return {
            "schema_version": 1,
            "status": self.status,
            "issued": self.issued,
            "reasons": list(self.reasons),
            "p_value_intersection_union": self.p_value,
            "endpoints": [dataclasses.asdict(e) for e in self.endpoints],
            "selective_labels": self.selectivity.to_dict(),
            "registration_hash": self.registration_hash,
            "checks": [c.to_dict() for c in self.checks],
            "claim": self.claim.to_dict(),
            "guarantee": (
                "Pr(claim | no true gain at some t in the range) <= alpha "
                "under independent C, read once, a rule frozen on X and "
                "outcomes observed independently of the baseline; the "
                "cluster-robust error is asymptotic in the clusters"
            ),
        }


class Evaluator:
    """Evaluates the registered claim on `C` exactly once.

    Attributes:
        registration: The registration the evaluation must match.
        used: Whether the claim has already been evaluated.
    """

    def __init__(self, registration: Registration) -> None:
        """Binds the guard to a registration.

        Args:
            registration: What was fixed before the confirmation set was
                read.
        """
        self.registration = registration
        self.used = False

    def evaluate(
        self,
        y: ArrayLike,
        a: ArrayLike,
        b: ArrayLike,
        clusters: ArrayLike,
        *,
        rule: Mapping[str, Any],
        model: costs.DecisionCostModel,
        observed: ArrayLike | None = None,
        input_tier: qualification.Tier = qualification.Tier.CONFIRMED,
        seed: int = 0,
    ) -> Outcome:
        """Tests `Delta(t) > 0` at both endpoints on the confirmation rows.

        Args:
            y: Outcomes on `C` (NaN where unobserved).
            a: The frozen rule's actions on `C`.
            b: The baseline decisions on `C`.
            clusters: Grouping unit of each row.
            rule: The rule description that was registered.
            model: The cost model that was registered.
            observed: Observation mask; defaults to `y` not NaN.
            input_tier: Highest tier the inputs allow (see
                `provenance.claim_tier`).
            seed: Run seed, recorded in the checks.

        Returns:
            The report. It is a result, not an exception, when the claim is
            not issued.

        Raises:
            BaselineError: On a second call, or when the rule or the cost
                model differ from what was registered.
        """
        if self.used:
            raise BaselineError(
                "the baseline claim was already evaluated on C; a second "
                "look at the confirmation set is not allowed"
            )
        if (
            rule_digest(rule) != self.registration.rule_hash
            or model.model_hash != self.registration.cost_model_hash
        ):
            raise BaselineError(
                "the rule or the cost model differs from what was registered "
                "before C was opened"
            )
        self.used = True
        return self.run(y, a, b, clusters, observed, input_tier, seed, model)

    def run(
        self,
        y: ArrayLike,
        a: ArrayLike,
        b: ArrayLike,
        clusters: ArrayLike,
        observed: ArrayLike | None,
        input_tier: qualification.Tier,
        seed: int,
        model: costs.DecisionCostModel,
    ) -> Outcome:
        """Computes the report once the call has been authorised.

        Args:
            y: Outcomes on `C` (NaN where unobserved).
            a: The frozen rule's actions on `C`.
            b: The baseline decisions on `C`.
            clusters: Grouping unit of each row.
            observed: Observation mask; defaults to `y` not NaN.
            input_tier: Highest tier the inputs allow.
            seed: Run seed, recorded in the checks.
            model: The cost model that was registered.

        Returns:
            The report. No endpoint test is run, and the claim is not
            issued, when outcomes are selective or the observed rows span
            fewer than `MIN_CLUSTERS` clusters.
        """
        outcomes = numpy.asarray(y, dtype=float)
        mask = (
            ~numpy.isnan(outcomes)
            if observed is None
            else numpy.asarray(observed).astype(bool)
        )
        selectivity = selective_labels(mask, b)
        labels = numpy.asarray(clusters)
        reasons: list[str] = []
        if selectivity.flagged:
            reasons.append(
                "outcomes are observed depending on the baseline decision "
                "(selective labels); the naive comparison is biased"
            )
        n_clusters = int(numpy.unique(labels[mask]).size) if mask.any() else 0
        if n_clusters < MIN_CLUSTERS:
            reasons.append(
                f"{n_clusters} clusters is below the measured minimum of "
                f"{MIN_CLUSTERS} for a calibrated test"
            )
        endpoints: tuple[ClusterTest, ...] = ()
        if not reasons:
            endpoints = tuple(
                cluster_test(
                    paired_difference(
                        outcomes[mask],
                        numpy.asarray(a)[mask],
                        numpy.asarray(b)[mask],
                        t,
                    ),
                    labels[mask],
                )
                for t in (self.registration.t_lo, self.registration.t_hi)
            )
        return self.assemble(
            endpoints, reasons, selectivity, input_tier, seed, model
        )

    def assemble(
        self,
        endpoints: tuple[ClusterTest, ...],
        reasons: list[str],
        selectivity: Selectivity,
        input_tier: qualification.Tier,
        seed: int,
        model: costs.DecisionCostModel,
    ) -> Outcome:
        """Builds the checks, the claim and the report.

        Args:
            endpoints: The tests at `t_lo` and `t_hi`, or empty when none
                was run.
            reasons: Why no test was run; empty when the tests were run.
            selectivity: The selective-label finding.
            input_tier: Highest tier the inputs allow.
            seed: Run seed, recorded in the checks.
            model: The cost model that was registered.

        Returns:
            The report. Its status is `not_issued` when there are reasons,
            `issued` when the larger endpoint p-value is at most the
            registered level, and `not_shown` otherwise. The claim keeps
            `input_tier` only when it is issued and is `unverified` if not.
        """
        alpha = self.registration.alpha
        p_value = max((e.p_value for e in endpoints), default=None)
        issued = bool(endpoints) and p_value is not None and p_value <= alpha
        if reasons:
            status = "not_issued"
        elif issued:
            status = "issued"
        else:
            status = "not_shown"
            reasons = [
                (
                    "the gain is not shown to be positive at both ends of the "
                    f"cost range at level {alpha}"
                )
            ]
        tier = input_tier if issued else qualification.Tier.UNVERIFIED
        points = (self.registration.t_lo, self.registration.t_hi)
        checks = tuple(
            qualification.Check(
                check_id=f"{CHECK_PREFIX}:delta@t={t:.4f}",
                null_hypothesis=f"Delta(t) <= 0 at t = {t:.4f}",
                statistic=e.statistic if numpy.isfinite(e.statistic) else 0.0,
                p_value=e.p_value,
                interval=None,
                effect=e.mean,
                n=e.n,
                decision=(
                    qualification.Decision.WARN
                    if e.p_value <= alpha
                    else qualification.Decision.PASS
                ),
                set_id=qualification.SetId.CONFIRMATION,
                seed_path=(seed, "framing.baseline", index),
                tier=tier,
            )
            for index, (t, e) in enumerate(zip(points, endpoints, strict=False))
        )
        claim = qualification.Claim(
            claim_id=CLAIM_ID,
            statement=(
                "net benefit of the model exceeds net benefit of the current "
                f"process over thresholds [{points[0]:.4f}, {points[1]:.4f}]"
            ),
            check_ids=tuple(c.check_id for c in checks)
            or (f"{CHECK_PREFIX}:not_computed",),
            adjusted_p_value=p_value if issued else None,
            effect=min((e.mean for e in endpoints), default=0.0),
            tier=tier,
            assumptions=(
                "outcomes observed independently of the baseline decision",
                "the action rule was frozen on X before C was opened",
                "cluster-robust standard error, asymptotic in the clusters",
                f"cost model {model.model_hash[:12]} fixed before C",
            ),
        )
        return Outcome(
            issued=issued,
            status=status,
            reasons=tuple(reasons),
            endpoints=endpoints,
            p_value=p_value,
            selectivity=selectivity,
            registration_hash=self.registration.registration_hash,
            checks=checks,
            claim=claim,
        )
