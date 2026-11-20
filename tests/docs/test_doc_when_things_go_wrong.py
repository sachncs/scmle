"""Guarantees of docs/WHEN_THINGS_GO_WRONG.md that no other test enforced."""

import datetime
import json
import logging
import types

import numpy
import pandas
import pytest

from benchmarks import reliability
from scmle import cli
from scmle.assurance.admissibility import provenance
from scmle.assurance.admissibility import qualify
from scmle.assurance.admissibility import rules
from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.contracts import tasks
from scmle.deploy import livetests
from scmle.governance import deployment_ledger
from scmle.governance import loop
from scmle.governance import never_list
from scmle.operations import lifecycle
from scmle.pipeline import planner
from tests import support_release

KEY = "person-key"
SYSTEM_KEY = "system-key"


def run(*arguments):
    """Runs the command-line tool.

    Args:
        *arguments: The command line, as paths or text.

    Returns:
        The exit code.
    """
    return cli.main([str(a) for a in arguments])


def policy(tmp_path):
    """Writes a small loop policy.

    Args:
        tmp_path: The folder to write it in.

    Returns:
        The path of the policy file.
    """
    settings = loop.Preregistration(
        reference=0.1,
        tolerance=0.05,
        target=0.6,
        tau=0.5,
        matured=5,
        maturity=0,
        scope="m:s",
    )
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(settings.to_dict()))
    return path


def opened(tmp_path):
    """Starts a loop on the command line with one pending escalation.

    Args:
        tmp_path: The folder to work in.

    Returns:
        The deployment folder and the pending escalation.
    """
    deploy = tmp_path / "deploy"
    code = run(
        "loop",
        "init",
        deploy,
        "--policy",
        policy(tmp_path),
        "--charter-hash",
        "a" * 64,
    )
    assert code == cli.EXIT_OK
    loop.Loop.resume(deploy).escalate("needs_a_person", {"why": "test"})
    (pending,) = loop.Loop.resume(deploy).survey()
    return deploy, pending


@pytest.mark.parametrize(
    "error",
    [
        errors.ConfigError("bad option value"),
        errors.DataError("bad data"),
        errors.SchemaError("bad columns"),
        errors.ArtifactError("refused bundle"),
    ],
)
def test_every_handled_failure_exits_one(monkeypatch, error):
    """Enforces docs/WHEN_THINGS_GO_WRONG.md:21.

    Also enforces website/src/learn/when-things-go-wrong.html:10.
    """
    command = next(c for c in cli.COMMANDS if c.name == "catalog")

    def execute(args):
        raise error

    monkeypatch.setattr(command, "execute", execute)
    assert cli.main(["catalog"]) == 1 == cli.EXIT_FAILED


def test_a_blocked_cli_run_exits_two_and_leaves_the_test_set_sealed(
    tmp_path, caplog
):
    """Enforces docs/WHEN_THINGS_GO_WRONG.md:83.

    Also enforces website/src/learn/when-things-go-wrong.html:50.
    """
    frame = reliability.binary_data(200)
    frame["y"] = numpy.arange(len(frame)) % 5
    path = tmp_path / "d.csv"
    frame.to_csv(path, index=False)
    with caplog.at_level(logging.ERROR):
        code = run(
            "run", path, "--target", "y", "--output-dir", tmp_path / "runs"
        )
    assert code == 2
    assert "blocked:" in caplog.text
    (log,) = (tmp_path / "runs").glob("run_*/events.jsonl")
    kinds = [json.loads(line)["kind"] for line in log.read_text().splitlines()]
    assert kinds[-1] == "state.BLOCKED"
    assert "test.opened" not in kinds


def test_no_model_that_can_enforce_the_monotonic_rule_blocks():
    """Enforces docs/WHEN_THINGS_GO_WRONG.md:171.

    Also enforces website/src/learn/when-things-go-wrong.html:87.
    """
    context = types.SimpleNamespace(
        spec=types.SimpleNamespace(
            data=types.SimpleNamespace(constraints={"a": 1})
        ),
        task=tasks.Task.BINARY,
    )
    candidates = [
        types.SimpleNamespace(
            name="forest", family="tree", param=None
        ),
        types.SimpleNamespace(
            name="prior", family="baseline", param=None
        ),
    ]
    with pytest.raises(errors.BlockedError, match="no available model"):
        planner.Monotonicity().apply(context, candidates, [])


def test_an_unverified_feature_is_admitted_not_kept_out():
    """Enforces docs/WHEN_THINGS_GO_WRONG.md:269.

    Also enforces website/src/learn/when-things-go-wrong.html:125.
    """
    rng = numpy.random.default_rng(0)
    features = pandas.DataFrame(
        {"a": rng.normal(size=300), "b": rng.normal(size=300)}
    )
    target = pandas.Series(
        (features.a + rng.normal(size=300) > 0).astype(int), name="y"
    )
    found = qualify.run(
        features,
        target,
        kind="predictive",
        classification=True,
        seed=0,
        scope="charter-hash",
        today=datetime.date(2026, 6, 1),
    )
    assert found.quarantined == {}
    tiers = {d.name: d.tier for d in found.table.decisions}
    assert tiers == {
        "a": qualification.Tier.UNVERIFIED,
        "b": qualification.Tier.UNVERIFIED,
    }
    assert {d.admission for d in found.table.decisions} == {
        rules.Admission.ADMIT_FLAGGED
    }


def test_an_attestation_that_expires_before_its_date_is_refused():
    """Enforces docs/WHEN_THINGS_GO_WRONG.md:316.

    Also enforces website/src/learn/leakage.html:128.
    Also enforces website/src/learn/when-things-go-wrong.html:150.
    """
    fields = (
        "A. Owner",
        "data owner",
        "recorded early",
        "ticket",
        "2026-02-01",
    )
    with pytest.raises(ValueError, match="expires must not precede date"):
        provenance.Attestation(*fields, "scope", "2026-01-31")
    same = provenance.Attestation(*fields, "scope", "2026-02-01")
    assert same.effective(datetime.date(2026, 2, 1))


def test_the_rollback_tolerance_defaults_to_one_hundredth():
    """Enforces docs/WHEN_THINGS_GO_WRONG.md:575.

    Also enforces website/src/learn/when-things-go-wrong.html:239-240.
    """
    assert lifecycle.Trigger().tolerance == 0.01


def test_a_second_reply_to_the_same_escalation_is_refused(tmp_path, caplog):
    """Enforces docs/WHEN_THINGS_GO_WRONG.md:742.

    Also enforces website/src/learn/when-things-go-wrong.html:335.
    """
    deploy, pending = opened(tmp_path)
    answer = deployment_ledger.Attestation.create(
        KEY,
        escalation=pending.id,
        digests={"question": pending.question_hash},
        decision="approve",
        statement="ok",
        evidence="ticket",
        role="owner",
        attestor="P",
        date="2020-01-01",
        scope="m:s",
        expires="2099-01-01",
    )
    (tmp_path / "answer.json").write_text(json.dumps(answer.to_dict()))
    (tmp_path / "key.txt").write_text(KEY)
    reply = (
        "loop",
        "reply",
        deploy,
        pending.id,
        "--attestation",
        tmp_path / "answer.json",
        "--key-file",
        tmp_path / "key.txt",
    )
    assert run(*reply) == cli.EXIT_OK
    with caplog.at_level(logging.ERROR):
        assert run(*reply) == cli.EXIT_FAILED
    expected = f"reply refused: {pending.id!r} is not a pending escalation"
    assert expected in caplog.text


def test_every_one_of_the_eleven_never_list_actions_needs_a_person(tmp_path):
    """Enforces docs/WHEN_THINGS_GO_WRONG.md:766.

    Also enforces website/src/learn/when-things-go-wrong.html:345.
    """
    assert len(never_list.Forbidden) == 11
    settings = loop.Preregistration(reference=0.1, tau=0.5, scope="m:s")
    guarded = loop.Loop.open(
        tmp_path / "deploy",
        types.SimpleNamespace(charter_hash="a" * 64),
        settings,
        system_keys=[SYSTEM_KEY],
    )
    calls = []
    for action in never_list.Forbidden:
        with pytest.raises(never_list.UnattestedError):
            guarded.perform(action, {"case": action.value}, calls.append)
    assert calls == []


def test_verify_cannot_see_that_the_newest_rows_were_removed(tmp_path, capsys):
    """Enforces docs/WHEN_THINGS_GO_WRONG.md:770.

    Also enforces website/src/learn/when-things-go-wrong.html:347.
    """
    deploy, ignored = opened(tmp_path)
    path = deploy / deployment_ledger.FILE_NAME
    lines = path.read_text().splitlines()
    assert len(lines) > 1
    capsys.readouterr()
    assert run("loop", "verify", deploy) == cli.EXIT_OK
    whole = json.loads(capsys.readouterr().out)
    path.write_text("\n".join(lines[:-1]) + "\n")
    assert run("loop", "verify", deploy) == cli.EXIT_OK
    cut = json.loads(capsys.readouterr().out)
    assert cut["ok"] is True and cut["rows"] == whole["rows"] - 1


def answered(document, path, auth, **fields):
    """Changes one recorded exchange of the staging responses.

    Args:
        document: The recorded responses, changed in place.
        path: The request path of the exchange.
        auth: Which caller made it (`none`, `valid`, `readonly`, ...).
        **fields: The fields to set; `body_update` merges into the body.

    Returns:
        The changed document.
    """
    for item in document["exchanges"]:
        if (item["path"], item["auth"]) == (path, auth):
            for key, value in fields.items():
                if key == "body_update":
                    item["body"].update(value)
                else:
                    item[key] = value
    return document


def problems(document=None, **changes):
    """Runs the live checks on recorded responses.

    Args:
        document: The recorded responses; the clean ones if not given.
        **changes: Changes to the check target, such as `base_url`.

    Returns:
        Every problem found, from all four checks.
    """
    results = support_release.run_live(document, **changes)
    assert [r.name for r in results] == list(livetests.CHECK_NAMES)
    return [p for r in results for p in r.problems]


def test_each_live_failure_names_the_documented_problem():
    """Enforces docs/WHEN_THINGS_GO_WRONG.md:811.

    Also enforces website/src/learn/when-things-go-wrong.html:363.
    """
    clean = support_release.recorded
    other = {"artifact_digest": "sha256:" + "1" * 64}
    cases = [
        (
            answered(clean(), "/readyz", "none", status=503),
            "/readyz answered 503 and is not ready",
        ),
        (
            answered(clean(), "/predict", "valid", body_update=other),
            "the answering model is not the released model",
        ),
        (
            answered(clean(), "/predict", "none", status=200),
            "/predict with no token answered 200",
        ),
        (
            answered(
                clean(),
                "/predict",
                "none",
                body={"error": "Traceback (most recent call last)"},
            ),
            "the refusal with no token leaks a traceback",
        ),
        (
            answered(clean(), "/predict", "readonly", status=200),
            "a read-only token reached /predict: 200",
        ),
    ]
    for document, expected in cases:
        assert expected in problems(document)
    plain = problems(base_url="http://staging.example.test")
    assert "the service is not reached over HTTPS" in plain


def test_verify_record_of_a_file_that_cannot_be_read_exits_one(tmp_path):
    """Enforces docs/WHEN_THINGS_GO_WRONG.md:828.

    Also enforces website/src/learn/when-things-go-wrong.html:367.
    """
    (tmp_path / "key").write_text("a-key")
    code = run(
        "deploy",
        "verify-record",
        tmp_path / "missing.json",
        "--key-file",
        tmp_path / "key",
    )
    assert code == 1
    (tmp_path / "broken.json").write_text("{not json")
    code = run(
        "deploy",
        "verify-record",
        tmp_path / "broken.json",
        "--key-file",
        tmp_path / "key",
    )
    assert code == 1
