"""Estimator dispatch, the public entry point and the serialised report."""

import dataclasses
import json

import numpy
import pytest
from scipy import stats

from scmle.contracts import errors
from scmle.evaluation.envelope import Counts
from scmle.evaluation.parameter_estimation import config
from scmle.evaluation.parameter_estimation import declaration
from scmle.evaluation.parameter_estimation import estimate
from scmle.evaluation.parameter_estimation import inputs
from scmle.evaluation.parameter_estimation import linear
from scmle.evaluation.parameter_estimation import report


def estimand(features=("a", "b"), likelihood="gaussian"):
    return declaration.Estimand("n", "y", "p", likelihood, features, ("x",))


def world(rows=120, seed=0):
    rng = numpy.random.default_rng(seed)
    x = rng.normal(size=(rows, 2))
    y = 1.0 + 2.0 * x[:, 0] - 0.5 * x[:, 1] + rng.normal(scale=0.3, size=rows)
    return x, y


@pytest.fixture(scope="module")
def fitted():
    x, y = world()
    return estimate.estimate_parameters(
        x, y, estimand(), config.Config(estimator="linear", covariance="iid")
    )


def test_an_ingestion_keeps_rows_and_adds_an_intercept():
    x, y = world()
    x[0, 0] = numpy.nan
    matrix, values, clusters, design, rows, dropped, digest = estimate.ingest(
        x,
        y,
        estimand(),
        config.Config(estimator="linear", missing="complete_case"),
        None,
    )
    assert (rows, dropped) == (120, 1)
    assert matrix.shape == (119, 2)
    assert values.tolist() == y[1:].tolist()
    assert design[:, 0].tolist() == [1.0] * 119
    assert clusters is None
    assert digest == inputs.raw_input_fingerprint(x, y, None)


def test_an_ingestion_rejects_a_two_dimensional_outcome():
    x, y = world()
    with pytest.raises(
        errors.InputError, match=r"outcome must have shape \(rows,\)"
    ):
        estimate.ingest(
            x, y[:, None], estimand(), config.Config(estimator="linear"), None
        )


def test_a_compute_dispatches_by_estimator():
    x, y = world()
    design = numpy.column_stack([numpy.ones(120), x])
    fit = estimate.compute(
        x, design, y, estimand(), config.Config(estimator="linear"), None
    )
    expected = linear.ols(
        design,
        y,
        config.Config(estimator="linear"),
        None,
        ("intercept", "a", "b"),
    )
    numpy.testing.assert_allclose(fit.coefficients, expected.coefficients)
    assert fit.names == ("intercept", "a", "b")


def test_a_linear_estimation_reports_t_intervals(fitted):
    first = fitted.parameters[1]
    se = first.standard_error
    pivot = stats.t.ppf(0.975, 117)
    assert first.name == "a"
    assert first.interval == pytest.approx(
        (first.estimate - pivot * se, first.estimate + pivot * se)
    )
    assert first.estimate == pytest.approx(2.0, abs=0.1)
    assert fitted.rows == Counts(120, 120, 0)
    assert (
        fitted.diagnostics["covariance_conditioning"]["status"] == "identified"
    )
    assert fitted.sensitivity["sampling"] == {
        "status": "not_requested",
        "bootstrap_replicates": 0,
    }
    assert fitted.estimand.likelihood == "gaussian"


def test_a_clustered_estimation_adds_cluster_uncertainty():
    x, y = world()
    clusters = numpy.repeat(numpy.arange(12), 10)
    found = estimate.estimate_parameters(
        x,
        y,
        estimand(),
        config.Config(
            estimator="linear", covariance="clustered", cluster_name="c"
        ),
        clusters=clusters.tolist(),
    )
    assert found.diagnostics["cluster_uncertainty"]["cluster_count"] == 12


def test_a_summary_uses_supplied_interval_errors():
    fit = report.Fit(
        ("a",),
        numpy.array([1.0]),
        numpy.array([[4.0]]),
        {},
        {},
        "gaussian",
        2.0,
        numpy.array([0.5]),
        numpy.array([3.0]),
        numpy.array([0.5]),
    )
    diagnostics, parameters = estimate.summarise(
        fit, config.Config(estimator="linear"), None
    )
    assert parameters[0].standard_error == 3.0
    assert parameters[0].interval == (0.0, 2.0)
    assert diagnostics["covariance_conditioning"]["rank"] == 1
    plain = dataclasses.replace(fit, standard_errors=None, interval_errors=None)
    ignored, parameters = estimate.summarise(
        plain, config.Config(estimator="linear"), None
    )
    assert parameters[0].standard_error == 2.0
    assert parameters[0].interval == (-3.0, 5.0)
    only = dataclasses.replace(fit, interval_errors=None)
    ignored, parameters = estimate.summarise(
        only, config.Config(estimator="linear"), None
    )
    assert parameters[0].interval == (-5.0, 7.0)


def test_a_clustered_summary_needs_labels():
    fit = report.Fit(
        ("a",),
        numpy.array([1.0]),
        numpy.array([[1.0]]),
        {},
        {},
        "g",
        1.0,
        numpy.array([0.5]),
    )
    with pytest.raises(errors.InputError, match="cluster labels are required"):
        estimate.summarise(
            fit,
            config.Config(
                estimator="linear", covariance="clustered", cluster_name="c"
            ),
            None,
        )


def test_a_logistic_estimation_rejects_non_binary_outcomes():
    x, y = world()
    with pytest.raises(errors.InputError, match="only 0 and 1"):
        estimate.estimate_parameters(
            x,
            y,
            estimand(likelihood="bernoulli"),
            config.Config(estimator="logistic"),
        )


# report


def test_a_report_round_trips(fitted):
    record = fitted.to_dict()
    json.dumps(record)
    assert report.Estimation.from_dict(record) == fitted
    assert record["rows"] == {"received": 120, "used": 120, "dropped": 0}
    assert report.NESTED["digests"] == {"raw_input_fingerprint": "frame"}


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"estimand": None}, errors.KindError, "estimand must be Estimand"),
        ({"policy": {}}, errors.KindError, "policy must be Config"),
        (
            {"parameters": ()},
            errors.InputError,
            "estimation result needs parameters",
        ),
        (
            {"covariance": ((1.0,),)},
            errors.InputError,
            "square over all parameters",
        ),
        ({"rows": (1, 1, 0)}, errors.KindError, "rows must be Counts"),
        (
            {"sensitivity": []},
            errors.KindError,
            "sensitivity must be a mapping",
        ),
        (
            {"digests": {}},
            errors.InputError,
            "digests must hold exactly the frame",
        ),
    ],
)
def test_an_invalid_report_is_rejected(fitted, changes, error, message):
    with pytest.raises(error, match=message):
        dataclasses.replace(fitted, **changes)


def test_a_report_rejects_repeated_names_and_bad_covariance(fitted):
    first = fitted.parameters[0]
    with pytest.raises(
        errors.InputError, match="parameter names must be unique"
    ):
        dataclasses.replace(fitted, parameters=(first, first, first))
    cov = [list(row) for row in fitted.covariance]
    cov[0][1] += 1.0
    with pytest.raises(errors.InputError, match="must be symmetric"):
        dataclasses.replace(fitted, covariance=tuple(map(tuple, cov)))
    cov[0][1] = float("inf")
    with pytest.raises(errors.InputError, match="finite values"):
        dataclasses.replace(fitted, covariance=tuple(map(tuple, cov)))
