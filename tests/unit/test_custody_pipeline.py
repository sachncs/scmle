"""`apply_custody`: the three-way split over a real `SplitManager` result."""

import typing

import numpy as np
import pandas as pd
import pytest

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.charter import feasibility as feas
from scmle.data.custody import carve
from scmle.data.custody import pipeline
from scmle.data.custody import sealed
from scmle.data.splitting import splitters
from tests.unit.test_charter_contracts import make_charter

BINARY = tasks.TaskType.BINARY
DATA_HASH = "9" * 64


def make_spec(**data):
    return tasks.ProjectSpec(
        data=tasks.DataSpec(uri="m", target="y", **data),
        validation=tasks.ValidationSpec(strategy="stratified", seed=5),
    )


def make_frame(rows=2000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.normal(size=rows),
            "z": rng.normal(size=rows),
            "y": (rng.random(rows) < 0.3).astype(int),
        }
    )


class Ran(typing.NamedTuple):
    outcome: pipeline.CustodyOutcome
    original: splitters.SplitResult
    vault: sealed.CustodyVault


def run(frame, charter=None, spec=None):
    spec = spec or make_spec()
    split = splitters.SplitManager(spec, BINARY).split(frame)
    log = sealed.OpenLog()
    vault = sealed.CustodyVault(log)
    outcome = pipeline.apply_custody(
        charter=charter or make_charter(min_effect=0.3, k_max=5),
        split=split,
        frame=frame,
        spec=spec,
        task=BINARY,
        data_hash=DATA_HASH,
        vault=vault,
    )
    return Ran(outcome, split, vault)


def test_three_sets_partition_every_row_and_c_is_sealed():
    frame = make_frame()
    outcome, original, vault = run(frame)
    manifest = outcome.manifest
    x = set(manifest.exploration_ids)
    c = set(manifest.confirmation_ids)
    e = set(manifest.embargo_ids)
    final = set(manifest.final_test_ids)
    assert x | c | e | final == set(range(len(frame)))
    assert not (x & c) and not (x & final) and not (c & final)
    assert final == set(original.manifest.test_ids)
    assert outcome.feasibility.verdict is feas.Verdict.FEASIBLE
    assert manifest.scope == "confirmatory"
    assert set(vault.sets) == {"C"}
    held = vault.sets["C"]
    assert held.ids == manifest.confirmation_ids
    rows = held.open(held.issue_token(), actor="test")
    assert np.allclose(
        rows[0]["x"].to_numpy(), frame["x"].iloc[list(held.ids)].to_numpy()
    )
    vault.close()


def test_the_development_data_is_exploration_only():
    frame = make_frame()
    ran = run(frame)
    outcome, vault = ran.outcome, ran.vault
    x = list(outcome.manifest.exploration_ids)
    assert len(outcome.split.dev_features) == len(x)
    assert outcome.split.manifest.train_ids == tuple(x)
    assert np.allclose(
        outcome.split.dev_features["x"].to_numpy(), frame["x"].iloc[x]
    )
    assert outcome.view.ids == tuple(x)
    for fit, held in outcome.split.manifest.folds:
        assert max(fit + held) < len(x)
    vault.close()


def test_too_few_rows_leave_no_confirmation_set():
    frame = make_frame(rows=120)
    outcome, original, vault = run(frame)
    assert outcome.manifest.confirmation_ids == ()
    assert outcome.manifest.scope == "exploratory_unconfirmed"
    assert outcome.feasibility.verdict is feas.Verdict.INFEASIBLE
    assert set(outcome.manifest.exploration_ids) == set(
        original.manifest.train_ids
    )
    assert not vault.sets
    vault.close()


def test_an_underpowered_confirmation_set_is_kept_but_capped():
    frame = make_frame(rows=1200)
    charter = make_charter(min_effect=0.05, k_max=10)
    ran = run(frame, charter)
    outcome, vault = ran.outcome, ran.vault
    assert outcome.feasibility.verdict is feas.Verdict.DOWNGRADE
    assert len(outcome.manifest.confirmation_ids) > 0
    assert outcome.manifest.scope == "exploratory_unconfirmed"
    vault.close()


def test_the_carve_is_reproducible_for_the_same_inputs():
    frame = make_frame()
    ran_a, ran_b = run(frame), run(frame)
    first, second = ran_a.outcome, ran_b.outcome
    vault_a, vault_b = ran_a.vault, ran_b.vault
    assert first.manifest.sha256 == second.manifest.sha256
    vault_a.close()
    vault_b.close()


def test_the_group_column_keeps_groups_out_of_both_sets():
    frame = make_frame()
    frame["g"] = np.arange(len(frame)) // 5
    spec = make_spec(group_column="g")
    split = splitters.SplitManager(spec, BINARY).split(frame)
    log = sealed.OpenLog()
    vault = sealed.CustodyVault(log)
    outcome = pipeline.apply_custody(
        charter=make_charter(min_effect=0.3, k_max=5),
        split=split,
        frame=frame,
        spec=spec,
        task=BINARY,
        data_hash=DATA_HASH,
        vault=vault,
    )
    groups = frame["g"].to_numpy()
    c = groups[list(outcome.manifest.confirmation_ids)]
    x = groups[list(outcome.manifest.exploration_ids)]
    assert len(c) > 0 and set(c).isdisjoint(x)
    vault.close()


def test_isolation_is_recorded_for_every_sealed_set():
    ran = run(make_frame())
    outcome, vault = ran.outcome, ran.vault
    assert dict(outcome.manifest.isolation)["C"] == "vault process"
    vault.close()


def test_a_refused_carve_becomes_a_run_with_no_confirmation_set(monkeypatch):
    def refuse(*args, **kwargs):
        raise errors.ConfigError(
            "carving the confirmation set left a validation fold empty"
        )

    monkeypatch.setattr(carve, "carve_confirmation", refuse)
    frame = make_frame()
    ran = run(frame)
    manifest = ran.outcome.manifest

    assert manifest.confirmation_ids == ()
    assert manifest.embargo_ids == ()
    assert manifest.scope == "exploratory_unconfirmed"
    assert set(manifest.exploration_ids) == set(ran.original.manifest.train_ids)
    assert any("not carved" in note for note in manifest.notes)
    assert ran.outcome.split.manifest.folds == ran.original.manifest.folds
    assert not ran.vault.sets
    ran.vault.close()


def test_the_manifest_commits_to_the_row_content_of_each_set():
    frame = make_frame()
    manifest = run(frame).outcome.manifest
    assert [name for name, _ in manifest.row_roots] == [
        "exploration",
        "confirmation",
        "final_test",
    ]
    manifest.verify_rows(frame)
    edited = frame.copy()
    edited.loc[manifest.confirmation_ids[0], "x"] += 1e-9
    with pytest.raises(errors.ArtifactError, match="confirmation"):
        manifest.verify_rows(edited)


def temporal_frame(phi, rows=1500):
    rng = np.random.default_rng(4)
    noise = rng.normal(size=rows)
    target = np.zeros(rows)
    for i in range(1, rows):
        target[i] = phi * target[i - 1] + noise[i]
    return pd.DataFrame(
        {"t": np.arange(rows), "x": rng.normal(size=rows), "y": target}
    )


def run_temporal(frame, charter):
    spec = tasks.ProjectSpec(
        data=tasks.DataSpec(uri="m", target="y", time_column="t"),
        validation=tasks.ValidationSpec(strategy="temporal", seed=5),
    )
    task = tasks.TaskType.REGRESSION
    split = splitters.SplitManager(spec, task).split(frame)
    vault = sealed.CustodyVault(sealed.OpenLog(), process=False)
    outcome = pipeline.apply_custody(
        charter=charter,
        split=split,
        frame=frame,
        spec=spec,
        task=task,
        data_hash=DATA_HASH,
        vault=vault,
    )
    return outcome


def test_a_serially_dependent_target_raises_the_embargo_above_the_charter():
    outcome = run_temporal(
        temporal_frame(0.9), make_charter(min_effect=0.3, k_max=5)
    )
    manifest = outcome.manifest
    assert manifest.confirmation_ids
    assert manifest.embargo_rows > 0
    assert manifest.embargo_ids
    assert any("autocorrelation" in note for note in manifest.notes)


def test_an_independent_target_keeps_the_charter_embargo():
    outcome = run_temporal(
        temporal_frame(0.0), make_charter(min_effect=0.3, k_max=5)
    )
    assert outcome.manifest.embargo_rows == 0
    assert not any("autocorrelation" in n for n in outcome.manifest.notes)
