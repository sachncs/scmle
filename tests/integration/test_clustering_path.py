"""Clustering from the API and the command line to a bundle that predicts."""

import json
import subprocess  # nosec B404 -- runs this interpreter on `-m scmle`
import sys

import numpy
import pandas
import pytest
from sklearn import datasets

from scmle import cli
from scmle.api import clustering
from scmle.artifacts import bundle
from scmle.contracts import errors

SEARCH = {"counts": (2, 3, 4), "methods": ("kmeans",), "rounds": 3}


def table(rows=240):
    values, truth = datasets.make_blobs(
        rows, centers=3, cluster_std=0.6, random_state=0
    )
    frame = pandas.DataFrame(values, columns=["x", "y"])
    return frame.assign(group=truth, name=[f"r{i}" for i in range(rows)])


def test_the_api_finds_the_groups_and_saves_a_bundle_that_predicts(tmp_path):
    frame = table()

    result = clustering.cluster(frame, truth="group", **SEARCH)
    folder = result.save(tmp_path / "bundle")
    model = bundle.Bundle.load(folder, trusted=True)

    assert result.report.best.clusters == 3
    assert result.report.stable
    assert result.agreement["adjusted_rand"] > 0.95
    assert set(result.metrics) == {
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "stability",
    }
    assert model.schema.task == clustering.TASK
    assert model.schema.numeric == ("x", "y")
    assert model.info.model == "kmeans"
    assert model.info.metric == "silhouette"
    assert model.classes is None
    numpy.testing.assert_array_equal(model.predict(frame), result.labels)
    record = json.loads((folder / clustering.REPORT_FILE).read_text("utf-8"))
    assert record["best"]["clusters"] == 3
    assert record["agreement"] == result.agreement
    assert record["rows"] == 240


def test_a_row_with_a_gap_still_gets_a_label():
    frame = table()
    frame.loc[0, "x"] = numpy.nan

    result = clustering.cluster(frame, **SEARCH)

    assert result.labels.shape == (240,)
    assert result.agreement == {}


def test_evaluate_scores_a_saved_bundle_on_new_rows(tmp_path):
    folder = clustering.cluster(table(), **SEARCH).save(tmp_path / "b")

    scores = clustering.evaluate(folder, table(90), truth="group", trusted=True)

    assert set(scores) == {
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "adjusted_rand",
        "normalized_mutual_info",
    }
    assert scores["adjusted_rand"] > 0.95
    assert scores["silhouette"] > 0.7


def test_a_bundle_of_another_task_is_refused(tmp_path):
    folder = clustering.cluster(table(), **SEARCH).save(tmp_path / "b")
    schema = folder / bundle.SCHEMA_FILE
    text = schema.read_text("utf-8").replace('"clustering"', '"regression"')
    schema.write_text(text, "utf-8")
    manifest = folder / bundle.MANIFEST_FILE
    record = json.loads(manifest.read_text("utf-8"))
    record["files"][bundle.SCHEMA_FILE] = bundle.shasum(schema)
    manifest.write_text(json.dumps(record), "utf-8")

    with pytest.raises(errors.ArtifactError, match="not a clustering"):
        clustering.evaluate(folder, table(), trusted=True)


def test_saving_never_overwrites_and_leaves_no_partial_folder(tmp_path):
    result = clustering.cluster(table(), **SEARCH)
    target = tmp_path / "taken"
    target.mkdir()
    (target / "keep.txt").write_text("mine", "utf-8")

    with pytest.raises(errors.InputError, match="already exists"):
        result.save(target)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["taken"]
    assert (target / "keep.txt").read_text("utf-8") == "mine"


def test_bad_requests_are_refused_before_any_work():
    frame = table()
    with pytest.raises(errors.InputError, match="unknown column"):
        clustering.cluster(frame, truth="missing", **SEARCH)
    with pytest.raises(errors.InputError, match="unknown column"):
        clustering.cluster(frame, exclude=("nope",), **SEARCH)
    with pytest.raises(errors.InputError, match="rounds"):
        clustering.cluster(frame, counts=(2,), rounds=0)


def test_the_cli_clusters_then_predicts_with_the_bundle(tmp_path):
    data = tmp_path / "rows.csv"
    table().to_csv(data, index=False)
    folder = tmp_path / "bundle"
    output = tmp_path / "labels.csv"

    status = cli.main(
        [
            "cluster",
            str(data),
            "--output",
            str(folder),
            "--counts",
            "2",
            "3",
            "4",
            "--methods",
            "kmeans",
            "--rounds",
            "3",
            "--truth",
            "group",
        ]
    )
    scored = cli.main(
        [
            "predict",
            str(folder),
            str(data),
            "--trusted",
            "--output",
            str(output),
        ]
    )

    assert (status, scored) == (cli.EXIT_OK, cli.EXIT_OK)
    record = json.loads((folder / clustering.REPORT_FILE).read_text("utf-8"))
    labels = pandas.read_csv(output)
    assert list(labels.columns) == ["prediction"]
    assert labels["prediction"].nunique() == record["best"]["clusters"] == 3
    assert record["agreement"]["adjusted_rand"] > 0.95


def test_the_cli_refuses_a_taken_output(tmp_path, caplog):
    data = tmp_path / "rows.csv"
    table().to_csv(data, index=False)
    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "file").write_text("x", "utf-8")

    status = cli.main(
        ["cluster", str(data), "--output", str(tmp_path / "taken")]
    )

    assert status == cli.EXIT_FAILED
    assert "already exists" in caplog.text


def test_python_dash_m_scmle_is_the_console_script():
    shown = subprocess.run(  # nosec B603 -- fixed arguments
        [sys.executable, "-m", "scmle", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    wrong = subprocess.run(  # nosec B603 -- fixed arguments
        [sys.executable, "-m", "scmle", "nonsense"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert shown.returncode == cli.EXIT_OK
    assert shown.stdout == cli.build().format_help()
    assert wrong.returncode == cli.EXIT_USAGE
    assert wrong.stderr.startswith("usage: scmle ")
