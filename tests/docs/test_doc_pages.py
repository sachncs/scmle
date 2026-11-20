"""Statements of the site's docs pages checked against the code they name.

Each test names the page lines it enforces.
"""

import json
import pathlib
import re

import numpy
import pandas
import pytest

from scmle.contracts import capabilities
from scmle.contracts import errors
from scmle.contracts import modalities
from scmle.contracts import tasks
from scmle.evaluation import fairness
from scmle.evaluation import parameter_estimation
from scmle.governance import keyring
from scmle.governance import review
from scmle.pipeline import modality_lifecycle

ROOT = pathlib.Path(__file__).resolve().parents[2]
PAGES = ROOT / "website" / "src" / "pages" / "docs"
OBSERVATIONAL = "Lalonde NSW vs PSID (observational, 614)"
# Largest Qini coefficient, as a share of the overall effect, still read as
# close to zero. `uplift.qini_coefficient` is the area between the Qini
# curve and the random line over n squared. A ranking that puts the whole
# effect in the top share q of rows lifts the curve to its end value
# n_t * effect by row q * n and keeps it there, so the area is
# n_t * effect * n * (1 - q) / 2 and the coefficient is
# (n_t / n) * effect * (1 - q) / 2, at most effect * (1 - q) / 2 since the
# treated share n_t / n is at most 1. With q = 0.9 (the effect spread over
# nine tenths of the rows, the weakest ordering still told from none) the
# bound is effect / 20.
SHARE = 0.05


def page(name):
    """The source of one docs page."""
    return (PAGES / f"{name}.astro").read_text(encoding="utf-8")


def headings(name):
    """The second-level headings of one docs page."""
    return re.findall(r"<h2[^>]*>(.*?)</h2>", page(name))


def test_the_observational_uplift_estimate_is_no_better_than_chance():
    # Enforces website/src/pages/docs/causal-decision.astro:66.
    path = ROOT / "benchmarks" / "results_real_tasks.json"
    results = json.loads(path.read_text(encoding="utf-8"))["results"]
    (found,) = [r for r in results if r["dataset"] == OBSERVATIONAL]
    assert (found["task"], found["metric"]) == ("uplift", "qini")
    assert found["note"].startswith("NOT randomized")
    # Close to zero: no larger than `SHARE` of the overall effect, the most
    # a ranking that leaves only the bottom tenth of rows without the
    # effect could score.
    (effect,) = re.findall(r"overall effect ([-+]?[0-9.]+)", found["note"])
    assert float(effect) != 0
    assert abs(found["framework"]) <= SHARE * abs(float(effect))
    assert 'id="task-types"' in page("benchmarks")
    assert "it should not be read as a causal result" in page("benchmarks")


def test_a_fairness_flag_comes_with_counts_and_prevalence_and_decides_nothing():
    # Enforces website/src/pages/docs/evidence.astro:44.
    groups = pandas.Series(["a"] * 100 + ["b"] * 100 + ["c"] * 10)
    truth = numpy.array([1, 0] * 50 + [1] * 20 + [0] * 80 + [1] * 10)
    predicted = numpy.array([1] * 100 + [0] * 100 + [1] * 10)
    report = fairness.Fairness().binary(groups, truth, predicted, 1)
    assert {name: g["rows"] for name, g in report["groups"].items()} == {
        "a": 100,
        "b": 100,
    }
    assert report["groups"]["a"]["base_rate"] == 0.5
    assert report["groups"]["b"]["base_rate"] == 0.2
    assert report["excluded_groups"] == {"c": 10}
    assert report["power_warning"] is True
    assert report["flags"]
    # The report describes; mitigating or blocking promotion is left to the
    # reader.
    assert set(report) == {
        "groups",
        "demographic_parity_difference",
        "equal_opportunity_difference",
        "equalized_odds_difference",
        "disparate_impact_ratio",
        "flags",
        "excluded_groups",
        "power_warning",
    }


def test_the_core_leaves_a_new_family_semantics_to_its_adapter():
    # Enforces website/src/pages/docs/extensions.astro:8.
    assert modality_lifecycle.Preparer.__abstractmethods__ == {
        "curate",
        "digest",
        "features",
        "label",
        "modality",
        "monitor",
        "remarks",
        "sample_id",
        "samples",
        "split",
    }
    with pytest.raises(TypeError):
        modality_lifecycle.Preparer()
    family = modalities.Bespoke("acme/point-cloud")
    assert family.value == "extension:acme/point-cloud"
    with pytest.raises(errors.InputError, match="conflicts with a built-in"):
        modalities.Bespoke("image")


def test_status_limits_renames_and_new_refusals_are_where_the_page_says():
    # Enforces website/src/pages/docs/migrations.astro:15.
    families = capabilities.Registry.seed().list_model_families()
    assert families
    for family in families:
        assert isinstance(family.status, capabilities.Status)
        assert isinstance(family.resources, capabilities.Resources)
        assert family.to_dict()["resources"]["memory"] is not None
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    breaking = changelog.split("### Breaking changes", 1)[1]
    breaking = breaking.split("\n### ", 1)[0]
    assert "renamed" in breaking
    assert "now refused" in breaking
    renames = (ROOT / "docs" / "RENAMES.md").read_text(encoding="utf-8")
    assert re.search(r"^- `\w+` -> `\w+`$", renames, re.MULTILINE)


def claim(kind):
    """One passed recomputation of a release review."""
    return review.Recomputation(
        criterion=f"{kind.value}.check",
        kind=kind,
        passed=True,
        observed="0.81",
        reference="0.80",
        uri=f"evidence/{kind.value}.json",
        notes="the accepted difference: 0.01 above the reference",
    )


def signoff(decision):
    """A release review of commit abc1234 with the given decision."""
    return review.Signoff(
        release="0.1.0b2",
        commit="abc1234",
        digests={
            "clone": "sha256:clone",
            "environment": "sha256:environment",
            "artifacts": {"wheel": "sha256:wheel"},
        },
        reviewer="reviewer@example.com",
        implementers=("engineer@example.com",),
        commands=("uv run pytest -q",),
        claims=tuple(claim(kind) for kind in review.Kind),
        observed="2026-11-19T12:00:00+00:00",
        decision=decision,
    )


def test_a_release_decision_names_its_differences_or_blocks():
    # Enforces website/src/pages/docs/migrations.astro:19.
    kinds = list(review.Kind)
    approved = signoff(review.Verdict.APPROVED).sign("release-key")
    assert approved.accepted("abc1234", kinds, "release-key")
    # A recomputation must name its difference: blank notes are refused.
    with pytest.raises(errors.InputError, match="notes must not be empty"):
        review.Recomputation(
            criterion="metric.check",
            kind=review.Kind.METRIC,
            passed=True,
            observed="0.81",
            reference="0.80",
            uri="evidence/metric.json",
            notes=" ",
        )
    blocked = signoff(review.Verdict.BLOCKED).sign("release-key")
    assert blocked.verify_signature("release-key")
    assert not blocked.accepted("abc1234", kinds, "release-key")


def test_the_estimation_report_answers_the_acceptance_questions():
    # Enforces website/src/pages/docs/modeling.astro:462.
    rng = numpy.random.default_rng(0)
    features = pandas.DataFrame(
        {"dose": rng.normal(size=200), "age": rng.normal(size=200)}
    )
    outcome = pandas.Series(1.0 + 2.0 * features["dose"])
    outcome += rng.normal(size=200)
    estimand = parameter_estimation.Estimand(
        name="dose_slope",
        outcome="outcome",
        population="adults treated in 2025",
        likelihood="gaussian",
        features=("dose", "age"),
        assumptions=("dose precedes the outcome",),
    )
    policy = parameter_estimation.Config(
        estimator="linear", replicates=50, seed=3
    )
    report = parameter_estimation.infer(
        features, outcome, estimand, policy
    ).to_dict()
    # The estimand and its population; the assumptions; the estimator, its
    # resolved parameters and convergence; the uncertainty method; what
    # reproduces it; the checks that would change the reading.
    assert report["estimand"]["name"] == "dose_slope"
    assert report["estimand"]["population"] == "adults treated in 2025"
    assert report["estimand"]["assumptions"] == ("dose precedes the outcome",)
    assert report["policy"]["estimator"] == "linear"
    assert report["policy"]["iterations"] == policy.iterations
    assert report["diagnostics"]["converged"] is True
    assert report["policy"]["covariance"] == "hc3"
    assert report["policy"]["replicates"] == 50
    assert report["policy"]["seed"] == 3
    assert re.fullmatch(r"[0-9a-f]{64}", report["digests"]["frame"])
    assert {"missingness", "sampling"} <= set(report["sensitivity"])


def test_the_evidence_page_covers_release_claims_and_production_status():
    # Enforces website/src/pages/docs/modeling.astro:477.
    assert "href={`${base}/docs/evidence/`}" in page("modeling")
    found = headings("evidence")
    assert "What blocks a release" in found
    assert "Independent review is a separate proof" in found
    assert (
        "production-proven means an independent reviewer reproduced the "
        "evidence from a clean checkout"
    ) in page("evidence")


def test_a_valid_signature_proves_neither_custody_rotation_nor_review():
    # Enforces website/src/pages/docs/security.astro:100.
    owner = "release-engineering"
    ring = keyring.Keyring(
        (
            keyring.Credential("old", owner, "2026-10-01T00:00:00+00:00"),
            keyring.Credential("new", owner, "2026-11-01T00:00:00+00:00"),
        )
    )
    payload = {"release": "candidate-1"}
    # A secret typed in place verifies as well as one from a manager.
    envelope = ring.sign(payload, "old", "typed", "2026-10-15T00:00:00+00:00")
    assert ring.verify(
        payload, envelope, {"old": "typed"}, "2026-10-20T00:00:00+00:00"
    )
    # A rotation records what the caller says, with no operator identity.
    retired = ring.retire("old", "2026-11-02T00:00:00+00:00", "said so", "new")
    assert "operator" not in retired.to_dict()
    # A validly signed review that is not approved is not a release.
    blocked = signoff(review.Verdict.BLOCKED).sign("release-key")
    assert blocked.verify_signature("release-key")
    assert not blocked.accepted("abc1234", list(review.Kind), "release-key")


def test_the_evidence_page_lists_the_claims_a_reviewer_recomputes():
    # Enforces website/src/pages/docs/security.astro:102.
    assert "href={`${base}/docs/evidence/`}" in page("security")
    text = page("evidence")
    assert "recompute at least one critical result" in text
    assert (
        "independently recomputed metric, fairness, and explanation claims"
    ) in text
    assert {"metric", "fairness", "explanation"} <= {
        k.value for k in review.Kind
    }


def test_a_declaration_carries_target_availability_split_compute_evidence():
    # Enforces website/src/pages/docs/workflows.astro:9.
    spec = tasks.Config(
        data=tasks.Dataset(
            uri="orders.csv",
            target="renewed",
            unavailable_at_prediction=("renewal_reason",),
            time_column="ordered",
        ),
        validation=tasks.Validation(strategy="temporal"),
        resources=tasks.Resources(time_budget_seconds=120),
        charter_path="charter.toml",
    )
    assert spec.data.target == "renewed"
    assert spec.data.unavailable_at_prediction == ["renewal_reason"]
    assert spec.validation.strategy == "temporal"
    assert spec.resources.time_budget_seconds == 120
    assert spec.charter_path == "charter.toml"
    with pytest.raises(ValueError, match="strategy"):
        tasks.Validation(strategy="shuffled")
