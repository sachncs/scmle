"""The parameter-estimation declaration, its checks and its records."""

import hashlib
import json
import types

import pandas
import pytest

from scmle.api import estimation
from scmle.contracts import errors
from scmle.evaluation import parameter_estimation


def estimand(features=("x", "t")):
    return parameter_estimation.Estimand(
        name="beta",
        outcome="y",
        population="all rows",
        likelihood="gaussian",
        feature_names=features,
        assumptions=("iid",),
    )


def policy(**fields):
    return parameter_estimation.Config(**{"estimator": "linear", **fields})


def ate(estimator, **fields):
    return policy(
        estimator=estimator,
        covariance="iid",
        treatment_position=1,
        causal_assumptions=(
            "consistency",
            "conditional_exchangeability",
            "positivity",
        ),
        **fields,
    )


def test_the_declaration_needs_both_or_neither():
    assert estimation.declare(None, None) == {
        "schema_version": 2,
        "enabled": False,
    }
    declared = estimation.declare(estimand(), policy())
    assert declared["enabled"] is True
    assert declared["estimand"] == estimand().to_dict()
    assert declared["policy"] == policy().to_dict()
    for half in ((estimand(), None), (None, policy())):
        with pytest.raises(errors.ConfigError, match="requires both"):
            estimation.declare(*half)


def test_the_fingerprint_is_of_compact_sorted_json():
    record = {"b": 1, "a": [1, 2]}
    assert (
        estimation.fingerprint(record)
        == hashlib.sha256(b'{"a":[1,2],"b":1}').hexdigest()
    )


def store(tmp_path, envelope, lock=None):
    digest = estimation.fingerprint(envelope)
    body = {**envelope, "digest": digest}
    (
        tmp_path / estimation.PARAMETER_ESTIMATION_DECLARATION_ARTIFACT
    ).write_text(json.dumps(body), encoding="utf-8")
    (tmp_path / estimation.PARAMETER_ESTIMATION_DECLARATION_LOCK).write_text(
        (lock or digest) + "\n", encoding="utf-8"
    )
    return tmp_path


def disabled(config="c"):
    return {
        "schema_version": 2,
        "enabled": False,
        "project_config_hash": config,
    }


def test_unwrap_returns_the_declaration_without_its_digest(tmp_path):
    assert estimation.unwrap(store(tmp_path, disabled()), "c") == disabled()


def test_unwrap_refuses_every_kind_of_damage(tmp_path):
    with pytest.raises(errors.ArtifactError, match="missing or unreadable"):
        estimation.unwrap(tmp_path, "c")
    store(tmp_path, disabled(), lock="0")
    with pytest.raises(errors.ArtifactError, match="digest mismatch"):
        estimation.unwrap(tmp_path, "c")
    store(tmp_path, disabled())
    with pytest.raises(errors.ArtifactError, match="does not match the run"):
        estimation.unwrap(tmp_path, "other")
    store(tmp_path, {**disabled(), "schema_version": 3})
    with pytest.raises(errors.ArtifactError, match="unsupported"):
        estimation.unwrap(tmp_path, "c")
    store(tmp_path, {**disabled(), "schema_version": 0})
    with pytest.raises(errors.ArtifactError, match="unsupported"):
        estimation.unwrap(tmp_path, "c")
    path = tmp_path / estimation.PARAMETER_ESTIMATION_DECLARATION_ARTIFACT
    path.write_text("[1]", encoding="utf-8")
    with pytest.raises(errors.ArtifactError, match="must be a JSON object"):
        estimation.unwrap(tmp_path, "c")
    path.write_text('{"a": 1}', encoding="utf-8")
    with pytest.raises(
        errors.ArtifactError, match="digest is missing"
    ) as caught:
        estimation.unwrap(tmp_path, "c")
    assert caught.value.details == {"run_dir": str(tmp_path)}


def test_unwrap_accepts_a_version_one_declaration(tmp_path):
    old = {**disabled(), "schema_version": 1}
    assert estimation.unwrap(store(tmp_path, old), "c") == old


def test_restore_returns_nothing_for_a_disabled_run(tmp_path):
    assert estimation.restore(store(tmp_path, disabled()), "c") == (None, None)
    store(tmp_path, {**disabled(), "estimand": {}})
    with pytest.raises(errors.ArtifactError, match="has extra fields"):
        estimation.restore(tmp_path, "c")


def enabled(**changes):
    declared = estimation.declare(estimand(), policy())
    return {**declared, "project_config_hash": "c", **changes}


def test_restore_rebuilds_an_enabled_declaration(tmp_path):
    found = estimation.restore(store(tmp_path, enabled()), "c")
    assert found == (estimand(), policy())


def test_restore_refuses_an_incomplete_or_invalid_declaration(tmp_path):
    incomplete = enabled()
    del incomplete["policy"]
    store(tmp_path, incomplete)
    with pytest.raises(errors.ArtifactError, match="incomplete"):
        estimation.restore(tmp_path, "c")
    store(tmp_path, enabled(enabled="yes"))
    with pytest.raises(errors.ArtifactError, match="incomplete"):
        estimation.restore(tmp_path, "c")
    store(tmp_path, enabled(estimand={"name": "only"}))
    with pytest.raises(errors.ArtifactError, match="is invalid"):
        estimation.restore(tmp_path, "c")


def test_rebuild_restores_tuples_and_the_prior():
    prior = parameter_estimation.BayesianPrior(
        mean=(0.0, 0.0), precision=((1.0, 0.0), (0.0, 1.0))
    )
    chosen = policy(
        estimator="bayesian_linear",
        covariance="iid",
        prior=prior,
    )
    declared = json.loads(json.dumps(estimation.declare(estimand(), chosen)))
    rebuilt_estimand, rebuilt_policy = estimation.rebuild(declared)
    assert rebuilt_estimand == estimand()
    assert rebuilt_policy == chosen
    causal = ate("ipw_ate")
    declared = json.loads(json.dumps(estimation.declare(estimand(), causal)))
    assert estimation.rebuild(declared)[1] == causal
    assert rebuilt_policy.prior.precision == ((1.0, 0.0), (0.0, 1.0))


def test_load_reads_a_declaration_file(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(json.dumps(estimation.declare(estimand(), policy())))
    assert estimation.load(path) == (estimand(), policy())
    stripped = estimation.declare(estimand(), policy())
    del stripped["schema_version"]
    path.write_text(json.dumps(stripped))
    assert estimation.load(path) == (estimand(), policy())


def test_load_refuses_unreadable_or_foreign_files(tmp_path):
    path = tmp_path / "d.json"
    with pytest.raises(errors.ConfigError, match="cannot be read") as caught:
        estimation.load(path)
    assert caught.value.details == {"path": str(path)}
    path.write_text('{"estimand": {}}')
    with pytest.raises(errors.ConfigError, match="with `estimand` and"):
        estimation.load(path)
    path.write_text('{"estimand": {}, "policy": {}, "schema_version": 9}')
    with pytest.raises(errors.ConfigError, match="unsupported"):
        estimation.load(path)
    path.write_text('{"estimand": {}, "policy": {}}')
    with pytest.raises(errors.ConfigError, match="is invalid"):
        estimation.load(path)


def test_refuse_records_the_reason_and_removes_an_old_report(tmp_path):
    (tmp_path / estimation.REPORT).write_text("{}")
    error = errors.DataError("too few rows", {"rows": 3})
    path = estimation.refuse(tmp_path / "out", estimand(), policy(), error)
    assert path == tmp_path / "out" / estimation.REFUSAL
    record = json.loads(path.read_text("utf-8"))
    assert record == {
        "schema_version": 1,
        "status": "refused",
        "reason": "too few rows",
        "category": error.category,
        "details": {"rows": 3},
        "estimand": json.loads(json.dumps(estimand().to_dict())),
        "policy": json.loads(json.dumps(policy().to_dict())),
    }
    (tmp_path / "out" / estimation.REPORT).write_text("{}")
    estimation.refuse(tmp_path / "out", estimand(), policy(), error)
    assert not (tmp_path / "out" / estimation.REPORT).exists()


def orchestrator(treatment=None):
    return types.SimpleNamespace(
        spec=types.SimpleNamespace(
            data=types.SimpleNamespace(treatment_column=treatment)
        )
    )


def test_locate_checks_the_treatment_column():
    chosen = ate("ipw_ate")
    with pytest.raises(
        errors.ConfigError, match="ipw_ate parameter estimation"
    ):
        estimation.locate(orchestrator(), estimand(), chosen)
    with pytest.raises(errors.ConfigError, match="included in estimand"):
        estimation.locate(orchestrator("z"), estimand(), chosen)
    with pytest.raises(errors.ConfigError, match="treatment_position must"):
        estimation.locate(orchestrator("t"), estimand(("t", "x")), chosen)
    assert estimation.locate(orchestrator("t"), estimand(), chosen) == "t"


def test_screen_lists_the_declared_columns_and_their_gaps():
    frame = pandas.DataFrame(
        {"x": [1.0, None], "t": [0, 1], "y": [1.0, 2.0], "g": ["a", "b"]}
    )
    complete = policy(missing="complete_case", cluster_name="g")
    assert estimation.screen(orchestrator(), estimand(), complete, frame) == (
        ["x", "t", "y", "g"],
        None,
        {"x": 1},
    )
    with pytest.raises(errors.DataError, match="complete_case policy"):
        estimation.screen(orchestrator(), estimand(), policy(), frame)
    with pytest.raises(errors.DataError, match=r"missing: \['z'\]"):
        estimation.screen(orchestrator(), estimand(("x", "z")), complete, frame)
    doubly = ate("aipw_ate", missing="complete_case")
    assert estimation.screen(orchestrator("t"), estimand(), doubly, frame)[
        1
    ] == ("t")


def test_trace_names_the_source_only_for_real_data(tmp_path):
    frame = pandas.DataFrame({"a": [1]})
    found = estimation.trace("<memory>", frame)
    assert found["source_uri"] == "<memory>"
    assert found["source_identity"] is None
    assert found["snapshot_id"]
    assert estimation.trace("<in-memory>", frame)["source_identity"] is None


def test_estimate_is_tabular_only():
    other = types.SimpleNamespace(
        spec=types.SimpleNamespace(data=types.SimpleNamespace(modality="image"))
    )
    with pytest.raises(errors.ConfigError, match="tabular modality"):
        estimation.estimate(other, estimand(), policy())
