"""A run measures the coverage of its intervals and sets on the test rows."""

import json
import pathlib

import numpy as np
import pandas as pd
import pytest
import support

import scmle
from scmle.artifacts import bundle as bundle_module
from scmle.contracts import errors


def regression_frame(rows=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, rows)
    return pd.DataFrame({"x": x, "y": 2 * x + rng.normal(0, 1, rows)})


@pytest.fixture(scope="module")
def regression(tmp_path_factory):
    return support.make_project(
        regression_frame(),
        "y",
        time_budget=300,
        output_dir=tmp_path_factory.mktemp("covreg"),
    ).run()


def test_a_regression_run_writes_a_coverage_report(regression):
    path = pathlib.Path(regression.run_dir) / "coverage_report.json"
    report = json.loads(path.read_text())
    assert report["kind"] == "regression"
    assert {"residual", "quantile", "cv_plus"} <= set(report["methods"])
    entry = report["methods"]["residual"]["0.9"]
    assert entry["marginal"]["rows"] == report["rows"] > 0
    assert entry["tier"] in {"confirmed", "unverified"}
    assert regression.bundle.coverage_report == report


def test_the_report_is_saved_with_the_bundle_and_checksummed(
    regression, tmp_path
):
    regression.bundle.save(tmp_path / "copy")
    loaded = scmle.ModelBundle.load(tmp_path / "copy", trusted=True)
    assert loaded.coverage_report == regression.bundle.coverage_report
    path = tmp_path / "copy" / bundle_module.COVERAGE_FILE
    path.write_text(path.read_text().replace("0.9", "0.8", 1))
    with pytest.raises(errors.ArtifactError, match="coverage report"):
        scmle.ModelBundle.load(tmp_path / "copy", trusted=True)


def test_a_classification_run_reports_sets_by_class(tmp_path, binary_frame):
    result = support.make_project(
        binary_frame, "y", time_budget=300, output_dir=tmp_path
    ).run()
    report = result.bundle.coverage_report
    assert report is not None and report["kind"] == "classification"
    entry = report["methods"]["class_conditional"]["0.9"]
    assert set(entry["by_class"]) == {"0", "1"}
    assert entry["marginal"]["rows"] == report["rows"]
