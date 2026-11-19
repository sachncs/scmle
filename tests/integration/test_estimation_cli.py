"""The CLI estimate path: `scmle estimate` and `scmle run --estimation`."""

import json

import numpy
import pandas
import pytest

from scmle import cli
from scmle.api import estimation
from scmle.contracts import errors
from scmle.evaluation.parameter_estimation import Config
from scmle.evaluation.parameter_estimation import Estimand
from scmle.evaluation.parameter_estimation import Estimation


def frame():
    rng = numpy.random.default_rng(21)
    signal = rng.normal(size=80)
    outcome = 1.25 + 2.0 * signal + rng.normal(scale=0.1, size=80)
    return pandas.DataFrame({"signal": signal, "outcome": outcome})


def declaration(tmp_path, **policy):
    path = tmp_path / "estimand.json"
    path.write_text(
        json.dumps(
            {
                "estimand": {
                    "name": "slope",
                    "outcome": "outcome",
                    "population": "the declared dataset",
                    "likelihood": "gaussian",
                    "features": ["signal"],
                    "assumptions": ["the conditional mean is linear"],
                },
                "policy": {"estimator": "linear", **policy},
            }
        ),
        encoding="utf-8",
    )
    return path


def data(tmp_path, table=None):
    path = tmp_path / "data.csv"
    (frame() if table is None else table).to_csv(path, index=False)
    return path


def test_load_reads_the_single_word_layout(tmp_path):
    estimand, policy = estimation.load(declaration(tmp_path, confidence=0.9))

    assert estimand == Estimand(
        name="slope",
        outcome="outcome",
        population="the declared dataset",
        likelihood="gaussian",
        feature_names=("signal",),
        assumptions=("the conditional mean is linear",),
    )
    assert policy == Config(estimator="linear", confidence_level=0.9)


def test_load_reads_the_version_one_layout(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "estimand": {
                    "name": "slope",
                    "outcome": "outcome",
                    "population": "p",
                    "likelihood": "gaussian",
                    "feature_names": ["signal"],
                    "assumptions": ["a"],
                },
                "policy": {"estimator": "linear", "confidence_level": 0.8},
            }
        ),
        encoding="utf-8",
    )

    estimand, policy = estimation.load(path)

    assert estimand.feature_names == ("signal",)
    assert policy.confidence_level == 0.8


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json", "^parameter estimation declaration cannot be read$"),
        (
            "[]",
            (
                "^parameter estimation declaration must be a JSON object "
                "with `estimand` and `policy`$"
            ),
        ),
        (
            json.dumps({"estimand": {}, "policy": {}, "schema_version": 9}),
            "^unsupported parameter estimation declaration schema$",
        ),
        (
            json.dumps({"estimand": {"name": "x"}, "policy": {}}),
            "^parameter estimation declaration is invalid$",
        ),
    ],
)
def test_load_refuses_a_bad_file_with_a_config_error(
    tmp_path, content, message
):
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(errors.ConfigError, match=message) as caught:
        estimation.load(path)

    assert caught.value.details == {"path": str(path)}


def test_load_refuses_a_missing_file(tmp_path):
    with pytest.raises(errors.ConfigError, match="cannot be read"):
        estimation.load(tmp_path / "absent.json")


def test_the_estimate_command_writes_the_estimate_and_its_interval(
    tmp_path, caplog
):
    out = tmp_path / "out"
    caplog.set_level("INFO")

    code = cli.main(
        [
            "estimate",
            str(data(tmp_path)),
            "--estimation",
            str(declaration(tmp_path)),
            "--output-dir",
            str(out),
        ]
    )

    assert code == cli.EXIT_OK
    report = Estimation.read(out / "parameter_estimation.json")
    slope = next(p for p in report.parameters if p.name == "signal")
    assert slope.estimate == pytest.approx(2.0, abs=0.05)
    assert slope.interval[0] < 2.0 < slope.interval[1]
    assert not (out / "parameter_estimation_refusal.json").exists()
    lines = [r.getMessage() for r in caplog.records]
    assert any(
        line.startswith(f"slope signal = {slope.estimate:.6g}, 95% interval")
        for line in lines
    )


def test_a_refused_estimate_exits_blocked_with_its_reasons(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "parameter_estimation.json").write_text("{}", encoding="utf-8")
    table = frame()
    table.loc[0, "signal"] = numpy.nan

    code = cli.main(
        [
            "estimate",
            str(data(tmp_path, table)),
            "--estimation",
            str(declaration(tmp_path)),
            "--output-dir",
            str(out),
        ]
    )

    assert code == cli.EXIT_BLOCKED
    record = json.loads(
        (out / "parameter_estimation_refusal.json").read_text(encoding="utf-8")
    )
    assert record["schema_version"] == 1
    assert record["status"] == "refused"
    assert record["category"] == "data"
    assert record["reason"] == (
        "parameter estimation missing values require an explicit "
        "complete_case policy: {'signal': 1}"
    )
    assert record["estimand"]["name"] == "slope"
    assert record["policy"]["estimator"] == "linear"
    assert not (out / "parameter_estimation.json").exists()


def test_a_bad_declaration_is_a_failure_not_a_refusal(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("[]", encoding="utf-8")

    code = cli.main(
        [
            "estimate",
            str(data(tmp_path)),
            "--estimation",
            str(path),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert code == cli.EXIT_FAILED
    assert not (tmp_path / "out").exists()


def test_run_with_an_estimation_keeps_the_estimate_in_the_run(tmp_path):
    out = tmp_path / "runs"

    code = cli.main(
        [
            "run",
            str(data(tmp_path)),
            "--target",
            "outcome",
            "--task",
            "regression",
            "--budget",
            "30",
            "--estimation",
            str(declaration(tmp_path)),
            "--output-dir",
            str(out),
        ]
    )

    assert code == cli.EXIT_OK
    run = next(out.glob("run_*"))
    report = Estimation.read(run / "parameter_estimation.json")
    evaluation = json.loads((run / "evaluation.json").read_text())
    assert evaluation["parameter_estimation"] == {
        "artifact": "parameter_estimation.json",
        "digest": report.digest(),
    }
    stored = json.loads(
        (run / "parameter_estimation_declaration.json").read_text()
    )
    assert stored["enabled"] is True
    assert stored["estimand"]["name"] == "slope"


def test_run_with_a_refused_estimation_trains_nothing(tmp_path):
    out = tmp_path / "runs"
    table = frame()
    table.loc[0, "signal"] = numpy.nan

    code = cli.main(
        [
            "run",
            str(data(tmp_path, table)),
            "--target",
            "outcome",
            "--estimation",
            str(declaration(tmp_path)),
            "--output-dir",
            str(out),
        ]
    )

    assert code == cli.EXIT_FAILED
    assert not list(out.glob("run_*"))
    record = json.loads((out / "parameter_estimation_refusal.json").read_text())
    assert record["category"] == "data"
    assert record["reason"].startswith(
        "parameter estimation missing values require"
    )
