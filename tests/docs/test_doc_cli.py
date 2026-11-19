"""Guarantees of docs/CLI.md that no other test enforced."""

import json
import logging
import os
import pathlib
import tomllib

import numpy
import pandas
import pytest
from sklearn import datasets

from scmle import cli
from scmle.artifacts import bundle
from scmle.artifacts.bundle import tombstone
from scmle.deploy import parity
from scmle.deploy import release_state
from scmle.governance import registry
from scmle.pipeline import orchestrator

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "deploy" / "local" / "golden"


def options(parser):
    """Yields every option string of a parser and of all its sub-commands.

    Args:
        parser: The argument parser to walk.

    Yields:
        Each option string, such as `--key-file`.
    """
    for action in vars(parser)["_actions"]:
        yield from action.option_strings
        for child in getattr(action, "choices", None) or {}:
            if isinstance(action.choices, dict):
                yield from options(action.choices[child])


def test_deploy_commands_take_keys_only_as_files():
    """Enforces docs/CLI.md:243.

    Also enforces website/src/pages/docs/cli.astro:141.
    """
    deploy = next(
        action.choices["deploy"]
        for action in vars(cli.build_parser())["_actions"]
        if isinstance(getattr(action, "choices", None), dict)
    )
    named = {option for option in options(deploy) if "key" in option}
    assert "--key-file" in named
    assert all(option.endswith("-file") for option in named), named


def briefly(adapter, *unused):
    """Stands in for `runtime.hold`: starts the server, then stops it."""
    adapter.start()
    adapter.stop()


def test_serve_needs_authentication_beyond_loopback(
    tmp_path, monkeypatch, caplog
):
    """Enforces docs/CLI.md:276.

    Also enforces website/src/pages/docs/cli.astro:145.
    """
    for name in list(os.environ):
        if name.startswith("SCMLE_AUTH_"):
            monkeypatch.delenv(name)
    monkeypatch.delenv("SCMLE_ENV", raising=False)
    monkeypatch.setattr(cli.delivery.runtime, "hold", briefly)
    folder = tmp_path / "bundle"
    parity.handwrite(parity.goldens(GOLDEN).spec).save(folder)
    command = ["serve", str(folder), "--trusted", "--port", "0"]
    with caplog.at_level(logging.INFO, logger=cli.__name__):
        exposed = cli.main([*command, "--host", "0.0.0.0"])  # nosec B104
        assert exposed == cli.EXIT_FAILED
        assert "without authentication" in caplog.text
        assert "serving" not in caplog.text
        assert cli.main([*command, "--host", "127.0.0.1"]) == cli.EXIT_OK
        assert "serving" in caplog.text


def estimation_frame(rows=80):
    """A table where `outcome` is linear in `signal`.

    Args:
        rows: Number of rows.

    Returns:
        The table.
    """
    rng = numpy.random.default_rng(21)
    signal = rng.normal(size=rows)
    noise = rng.normal(scale=0.1, size=rows)
    outcome = 1.25 + 2 * signal + noise
    return pandas.DataFrame({"signal": signal, "outcome": outcome})


def estimation(tmp_path, table, likelihood="gaussian", features=("signal",)):
    """Runs `scmle estimate` on a table and a slope declaration.

    Args:
        tmp_path: A scratch directory.
        table: The data.
        likelihood: The declared likelihood.
        features: The declared features.

    Returns:
        The exit code and the output directory.
    """
    data = tmp_path / "data.csv"
    table.to_csv(data, index=False)
    declaration = tmp_path / "estimand.json"
    declaration.write_text(
        json.dumps(
            {
                "estimand": {
                    "name": "slope",
                    "outcome": "outcome",
                    "population": "the declared dataset",
                    "likelihood": likelihood,
                    "features": list(features),
                    "assumptions": ["the conditional mean is linear"],
                },
                "policy": {"estimator": "linear"},
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    out.mkdir()
    (out / "parameter_estimation.json").write_text("{}", encoding="utf-8")
    code = cli.main(
        [
            "estimate",
            str(data),
            "--estimation",
            str(declaration),
            "--output-dir",
            str(out),
        ]
    )
    return code, out


def gapped():
    table = estimation_frame()
    table.loc[0, "signal"] = numpy.nan
    return table


@pytest.mark.parametrize(
    ("table", "options"),
    [
        (estimation_frame, {"features": ("absent",)}),
        (gapped, {}),
        (estimation_frame, {"likelihood": "bernoulli"}),
        (lambda: estimation_frame(rows=2), {}),
    ],
    ids=["missing-column", "missing-values", "likelihood", "estimator"],
)
def test_a_declaration_the_data_cannot_support_is_refused_on_record(
    tmp_path, table, options
):
    """Enforces docs/CLI.md:391-395.

    Also enforces docs/ARTIFACTS.md:160.
    """
    code, out = estimation(tmp_path, table(), **options)
    assert code == cli.EXIT_BLOCKED == 2
    record = json.loads(
        (out / "parameter_estimation_refusal.json").read_text("utf-8")
    )
    assert record["status"] == "refused"
    assert record["reason"]
    assert record["category"]
    assert "details" in record
    assert record["estimand"]["name"] == "slope"
    assert record["policy"]["estimator"] == "linear"
    assert not (out / "parameter_estimation.json").exists()
    gate = tomllib.loads((ROOT / "gates.toml").read_text("utf-8"))
    assert gate["estimate"]["refusal"]["error"] == "BlockedError"


def blobs(rows=240):
    """Three well-separated groups, a known-group column and a name.

    Args:
        rows: Number of rows.

    Returns:
        The table.
    """
    values, truth = datasets.make_blobs(
        rows, centers=3, cluster_std=0.6, random_state=0
    )
    frame = pandas.DataFrame(values, columns=["x", "y"])
    return frame.assign(group=truth, name=[f"r{i}" for i in range(rows)])


def test_cluster_searches_each_method_and_count_and_writes_a_bundle(tmp_path):
    """Enforces docs/CLI.md:417-422.

    Also enforces docs/CLI.md:422-424.
    """
    data = tmp_path / "rows.csv"
    blobs().to_csv(data, index=False)
    folder = tmp_path / "segments"
    command = ["cluster", str(data), "--output", str(folder)]
    options = ["--methods", "kmeans", "agglomerative", "--rounds", "2"]
    truth = ["--truth", "group", "--exclude", "name"]
    assert cli.main([*command, *options, *truth]) == cli.EXIT_OK
    record = json.loads((folder / "clustering.json").read_text("utf-8"))
    tried = {(c["method"], c["clusters"]) for c in record["candidates"]}
    assert tried == {
        (method, count)
        for method in ("kmeans", "agglomerative")
        for count in range(2, 9)
    }
    for candidate in record["candidates"]:
        assert {
            "silhouette",
            "davies_bouldin",
            "calinski_harabasz",
            "stability",
        } <= set(candidate)
    assert record["task"] == "clustering"
    assert "group" not in record["columns"]
    assert record["agreement"]
    assert (folder / "model_manifest.json").is_file()
    # The output must be new or empty, and each count at least 2.
    assert cli.main([*command, *options]) == cli.EXIT_FAILED
    small = [*command[:3], str(tmp_path / "other"), "--counts", "1", "2"]
    assert cli.main(small) != cli.EXIT_OK


def test_retire_refuses_a_bundle_and_retires_a_registered_version(
    tmp_path, monkeypatch
):
    """Enforces docs/CLI.md:461-464.

    Also enforces docs/ARTIFACTS.md:181.
    """
    monkeypatch.setenv("SCMLE_SIGNING_KEY", "cli-doc-retire-key")
    folder = tmp_path / "model"
    model = parity.handwrite(parity.goldens(GOLDEN).spec)
    model.save(folder)
    address = registry.identify(folder)
    receipts = ["--receipts", str(tmp_path / "receipts")]
    retire = ["retire", str(folder), "--reason", "superseded", *receipts]
    assert cli.main(retire) == cli.EXIT_OK
    assert registry.identify(folder) == address
    with pytest.raises(tombstone.RetiredError):
        bundle.Bundle.load(folder, trusted=True)
    rows = tmp_path / "rows.csv"
    pandas.DataFrame([parity.goldens(GOLDEN).records[0]]).to_csv(
        rows, index=False
    )
    predict = ["predict", str(folder), str(rows), "--trusted"]
    assert cli.main(predict) == cli.EXIT_BLOCKED == 2
    (folder / tombstone.FILE).write_text("{", encoding="utf-8")
    with pytest.raises(tombstone.RetiredError):
        bundle.Bundle.load(folder, trusted=True)

    other = tmp_path / "other"
    model.save(other)
    store = registry.ModelRegistryStore(tmp_path / "registry")
    release = store.register("churn", other, trusted=True)
    command = [
        "retire",
        str(tmp_path / "registry"),
        "--model",
        "churn",
        "--version",
        str(release.version),
        "--reason",
        "drift",
        *receipts,
    ]
    assert cli.main(command) == cli.EXIT_OK
    machine = release_state.ReleaseRegistry(store)
    state = machine.lookup("churn", release.version)
    assert state is release_state.ReleaseState.RETIRED
    with pytest.raises(tombstone.RetiredError):
        bundle.Bundle.load(tmp_path / "registry" / release.path, trusted=True)


def test_run_estimation_is_kept_beside_selection_and_never_read_by_it(
    tmp_path, monkeypatch
):
    """Enforces docs/CLI.md:395-396."""
    seen = []
    run = orchestrator.Orchestrator.run

    def watching(self, output_dir, plan=None, **options):
        declared = options.get("parameter_estimation_declaration") or {}
        folder = pathlib.Path(output_dir)
        written = list(folder.rglob("parameter_estimation.json"))
        seen.append((sorted(declared), written))
        return run(self, output_dir, plan, **options)

    monkeypatch.setattr(orchestrator.Orchestrator, "run", watching)
    code = estimation(tmp_path, estimation_frame())[0]
    assert code == cli.EXIT_OK
    data = tmp_path / "data.csv"
    out = tmp_path / "runs"
    command = [
        "run",
        str(data),
        "--target",
        "outcome",
        "--task",
        "regression",
        "--budget",
        "30",
        "--estimation",
        str(tmp_path / "estimand.json"),
        "--output-dir",
        str(out),
    ]
    assert cli.main(command) == cli.EXIT_OK
    # Selection ran with the declaration only: no estimate existed yet and
    # none was handed to it.
    ((declared, written),) = seen
    assert declared == ["enabled", "estimand", "policy", "schema_version"]
    assert written == []
    assert (next(out.glob("run_*")) / "parameter_estimation.json").is_file()


def test_a_completed_run_is_refused_by_resume_with_exit_one(
    tmp_path, binary_frame, caplog
):
    """Enforces docs/CLI.md:260-261.

    Also enforces docs/TUTORIAL.md:195-196,
    website/src/pages/docs/cli.astro:80,
    website/src/pages/docs/operations.astro:40 and
    website/src/pages/docs/workflows.astro:68.
    """
    data = tmp_path / "data.csv"
    binary_frame.to_csv(data, index=False)
    out = tmp_path / "runs"
    command = ["run", str(data), "--target", "y", "--output-dir", str(out)]
    assert cli.main([*command, "--budget", "30"]) == cli.EXIT_OK
    (run,) = out.glob("run_*")
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=cli.__name__):
        code = cli.main(["resume", str(run)])
    assert code == cli.EXIT_FAILED == 1
    expected = "config error: run already completed; nothing to resume"
    assert expected in caplog.text
