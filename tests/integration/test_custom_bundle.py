"""A registered custom estimator and metric round-trip through a bundle.

The bundle is saved by the run that registered them and loaded in a fresh
interpreter that registered nothing, the way it is served.
"""

import json
import os
import pathlib
import subprocess
import sys

import metric_functions
import numpy
import pytest
import support
import support_components

from scmle.contracts import tasks
from scmle.evaluation import metrics
from scmle.models import registry
import scmle.models.custom

TESTS = pathlib.Path(__file__).resolve().parents[1]

LOADER = """
import json, sys
import numpy, pandas
import scmle.artifacts.bundle
from scmle.evaluation import metrics
from scmle.models import registry

bundle = scmle.artifacts.bundle.Bundle.load(sys.argv[1], trusted=True)
rows = pandas.read_csv(sys.argv[2])
numpy.save(sys.argv[3], bundle.predict_proba(rows))
print(json.dumps({
    "model": bundle.info.model_name,
    "metric": bundle.info.metric,
    "registered": "centroid" in registry.Catalog.stock().adapters
    or "business_cost" in metrics.CATALOG.metrics,
    "labels": [str(label) for label in bundle.predict(rows)],
}))
"""


@pytest.fixture
def registered():
    scmle.models.custom.register(
        "centroid", classifier=support_components.Centroid
    )
    metrics.register(
        "business_cost",
        metric_functions.business_cost,
        greater_is_better=False,
        task_types=(tasks.Task.BINARY,),
    )
    yield
    scmle.models.custom.unregister("centroid")
    metrics.CATALOG.unregister("business_cost")


def test_custom_components_round_trip_through_a_saved_bundle(
    registered, binary_frame, tmp_path
):
    result = support.make_project(
        binary_frame,
        "y",
        metric="business_cost",
        models=tasks.Candidates(exclude_models=sorted(registry.inventory())),
        budget=300,
        directory=tmp_path / "runs",
    ).run()
    saved = result.save(tmp_path / "bundle")
    rows = binary_frame.drop(columns="y")
    rows.to_csv(tmp_path / "rows.csv", index=False)

    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            LOADER,
            str(saved),
            str(tmp_path / "rows.csv"),
            str(tmp_path / "proba.npy"),
        ],
        cwd=TESTS,
        env={**os.environ, "PYTHONPATH": str(TESTS)},
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    report = json.loads(loaded.stdout.strip().splitlines()[-1])

    assert result.evaluation["metric"] == "business_cost"
    assert report["model"] == "centroid"
    assert report["metric"] == "business_cost"
    assert not report["registered"]
    numpy.testing.assert_allclose(
        numpy.load(tmp_path / "proba.npy"), result.predict_proba(rows)
    )
    assert report["labels"] == [str(label) for label in result.predict(rows)]
