import json

import numpy as np
import pandas as pd

from scmle import cli
from scmle.governance import deployment_ledger as dl
from scmle.governance import loop as loop_module

KEY = "person-key"


def run(*arguments):
    return cli.main([str(a) for a in arguments])


def policy_file(tmp_path):
    policy = loop_module.LoopPolicy(
        reference_risk=0.1,
        tolerance=0.05,
        r_star=0.6,
        initial_tau=0.5,
        min_matured=5,
        outcome_window=0,
        scope="m:s",
    )
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy.to_dict()))
    return path


def test_the_cli_runs_a_batch_through_step_and_a_reply(tmp_path, capsys):
    deploy = tmp_path / "deploy"
    assert (
        run(
            "loop",
            "init",
            deploy,
            "--policy",
            policy_file(tmp_path),
            "--charter-hash",
            "a" * 64,
        )
        == 0
    )
    batch = pd.DataFrame(
        {
            "case_id": [f"c{i}" for i in range(20)],
            "prediction": np.zeros(20),
            "decision": ["approve"] * 20,
        }
    )
    batch.to_csv(tmp_path / "batch.csv", index=False)
    capsys.readouterr()
    assert (
        run("loop", "observe", deploy, "--batch", tmp_path / "batch.csv") == 0
    )
    assert json.loads(capsys.readouterr().out)["routing"] == {"answered": 20}
    pd.DataFrame({"case_id": batch["case_id"], "loss": np.zeros(20)}).to_csv(
        tmp_path / "labels.csv", index=False
    )
    assert (
        run("loop", "mature", deploy, "--labels", tmp_path / "labels.csv") == 0
    )
    assert json.loads(capsys.readouterr().out)["matched"] == 20
    assert run("loop", "step", deploy) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"
    assert run("loop", "verify", deploy) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True

    resumed = loop_module.Loop.resume(deploy)
    resumed.escalate("needs_a_person", {"why": "test"})
    capsys.readouterr()
    assert run("loop", "pending", deploy) == 0
    pending = json.loads(capsys.readouterr().out)
    assert [p["question"]["action"] for p in pending] == ["needs_a_person"]
    answer = dl.Attestation.create(
        KEY,
        escalation_id=pending[0]["id"],
        question_hash=pending[0]["question_hash"],
        decision="approve",
        statement="ok",
        evidence="ticket",
        role="owner",
        name_or_id="P",
        date="2020-01-01",
        scope="m:s",
        expires="2099-01-01",
    )
    (tmp_path / "answer.json").write_text(json.dumps(answer.to_dict()))
    (tmp_path / "key.txt").write_text(KEY)
    assert (
        run(
            "loop",
            "reply",
            deploy,
            pending[0]["id"],
            "--attestation",
            tmp_path / "answer.json",
            "--key-file",
            tmp_path / "key.txt",
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["tier"] == "attested"
    assert run("loop", "pending", deploy) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_a_refused_reply_and_a_broken_ledger_fail(tmp_path, capsys):
    deploy = tmp_path / "deploy"
    run(
        "loop",
        "init",
        deploy,
        "--policy",
        policy_file(tmp_path),
        "--charter-hash",
        "a" * 64,
    )
    loop_module.Loop.resume(deploy).escalate("needs_a_person", {})
    answer = dl.Attestation.create(
        KEY,
        escalation_id="esc-1",
        question_hash="0" * 64,
        decision="approve",
        statement="ok",
        evidence="t",
        role="r",
        name_or_id="n",
        date="2020-01-01",
        scope="m:s",
        expires="2099-01-01",
    )
    (tmp_path / "a.json").write_text(json.dumps(answer.to_dict()))
    (tmp_path / "k.txt").write_text(KEY)
    code = run(
        "loop",
        "reply",
        deploy,
        "esc-1",
        "--attestation",
        tmp_path / "a.json",
        "--key-file",
        tmp_path / "k.txt",
    )
    assert code == 1
    path = deploy / dl.FILE_NAME
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], lines[0], *lines[1:]]) + "\n")
    assert run("loop", "verify", deploy) == 1


def test_a_policy_file_needs_only_the_reference_risk():
    policy = loop_module.LoopPolicy.from_dict({"reference_risk": 0.1})
    assert policy.high_impact_actions == ()
    assert policy.reference_risk == 0.1
