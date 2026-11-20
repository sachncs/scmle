"""The delivery, governance, intake and rollout commands of the CLI.

Each command is built on its own parser; the subsystems a command wraps are
stood in for where running them for real would need a deployment, so the
tests pin the command's own wiring: arguments, dispatch, output and errors.
"""

import io
import json
import types

import numpy
import pandas
import pytest

import scmle.artifacts.bundle
from scmle.cli import command
from scmle.cli import delivery
from scmle.cli import governance
from scmle.cli import intake
from scmle.cli import rollout
from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.governance import deployment_ledger
import scmle.models.registry
from scmle.preparation import preprocessing


def parse(kind, argv):
    parser = command.Parser(prog="scmle")
    kind().configure(parser)
    return parser.parse_args(argv)


@pytest.fixture
def saved(tmp_path):
    plan = data.Blueprint(("num",), ("cat",))
    rng = numpy.random.default_rng(0)
    features = pandas.DataFrame(
        {"num": rng.normal(size=80), "cat": rng.choice(["a", "b"], 80)}
    )
    target = (features.num > 0).astype(int)
    adapter = scmle.models.registry.Catalog.stock().get("logistic_regression")
    pipe = preprocessing.instantiate(plan, adapter, tasks.Task.BINARY, 0).fit(
        features, target
    )
    made = scmle.artifacts.bundle.Bundle.create(
        pipe, plan, features, "y", "binary_classification", "roc_auc", "lr"
    )
    made.save(tmp_path / "bundle")
    features.head(5).to_csv(tmp_path / "rows.csv", index=False)
    return made, features.head(5), tmp_path


def test_predict_writes_predictions_and_probabilities(saved):
    made, rows, root = saved
    args = parse(
        delivery.Prediction,
        [
            str(root / "bundle"),
            str(root / "rows.csv"),
            "--output",
            str(root / "out.csv"),
            "--trusted",
        ],
    )
    assert delivery.Prediction().execute(args) == command.EXIT_OK
    written = pandas.read_csv(root / "out.csv")
    assert list(written.columns) == ["prediction", "proba_0", "proba_1"]
    assert written.prediction.tolist() == made.predict(rows).tolist()
    numpy.testing.assert_allclose(
        written[["proba_0", "proba_1"]].to_numpy(),
        made.predict_proba(rows),
        rtol=1e-6,
    )


def test_predict_maps_renamed_columns(saved):
    rows, root = saved[1], saved[2]
    rows.rename(columns={"num": "NUM "}).to_csv(root / "odd.csv", index=False)
    args = parse(
        delivery.Prediction,
        [
            str(root / "bundle"),
            str(root / "odd.csv"),
            "--output",
            str(root / "out.csv"),
            "--trusted",
            "--rename",
            "NUM =num",
        ],
    )
    delivery.Prediction().execute(args)
    assert len(pandas.read_csv(root / "out.csv")) == 5


def test_predict_refuses_an_unsigned_bundle_without_trust(saved):
    root = saved[2]
    args = parse(
        delivery.Prediction, [str(root / "bundle"), str(root / "rows.csv")]
    )
    assert args.output == "predictions.csv"
    assert args.trusted is False
    assert args.rename == []
    with pytest.raises(errors.ArtifactError, match="unauthenticated"):
        delivery.Prediction().execute(args)


def test_batch_scores_the_file_through_the_serving_path(saved, monkeypatch):
    root = saved[2]
    calls = []

    class Batcher:
        def __init__(self, model, on_null, on_unseen):
            calls.append((model.info.model_name, on_null, on_unseen))

        def score(self, data, output, chunk_rows, resume):
            calls.append((data, output, chunk_rows, resume))
            return 5

    monkeypatch.setattr(delivery.server, "Batcher", Batcher)
    args = parse(
        delivery.Batch,
        [
            str(root / "bundle"),
            str(root / "rows.csv"),
            "--output",
            str(root / "batch.csv"),
            "--trusted",
            "--on-null",
            "reject",
        ],
    )
    assert (args.chunk_rows, args.on_null, args.on_unseen) == (
        10_000,
        "reject",
        "report",
    )
    assert delivery.Batch().execute(args) == command.EXIT_OK
    assert calls == [
        ("lr", "reject", "report"),
        (str(root / "rows.csv"), str(root / "batch.csv"), 10_000, True),
    ]
    calls.clear()
    args = parse(
        delivery.Batch,
        [
            str(root / "bundle"),
            str(root / "rows.csv"),
            "--output",
            str(root / "batch.csv"),
            "--trusted",
            "--restart",
            "--chunk-rows",
            "7",
        ],
    )
    assert delivery.Batch().execute(args) == command.EXIT_OK
    assert calls[1] == (
        str(root / "rows.csv"),
        str(root / "batch.csv"),
        7,
        False,
    )


def test_export_and_package_pass_their_options(monkeypatch, tmp_path):
    calls = []
    model = types.SimpleNamespace(
        read_frame=lambda path: f"frame:{path}",
        export_onnx=lambda output, check, tolerance, unverified: (
            calls.append((output, check, tolerance, unverified)) or output
        ),
    )
    monkeypatch.setattr(
        delivery.bundle.Bundle,
        "load",
        lambda path, trusted: calls.append((path, trusted)) or model,
    )
    args = parse(delivery.Export, ["b"])
    assert (args.output, args.check, args.trusted) == (
        "model.onnx",
        None,
        False,
    )
    assert delivery.Export().execute(args) == command.EXIT_OK
    default = delivery.onnx_export.DEFAULT_TOLERANCE
    assert calls == [("b", False), ("model.onnx", None, default, False)]
    calls.clear()
    delivery.Export().execute(
        parse(delivery.Export, ["b", "--check", "c", "--tolerance", "0.01"])
    )
    assert calls[-1] == ("model.onnx", "frame:c", 0.01, False)
    delivery.Export().execute(parse(delivery.Export, ["b", "--unverified"]))
    assert calls[-1] == ("model.onnx", None, default, True)
    monkeypatch.setattr(
        delivery.packaging,
        "scaffold",
        lambda bundle, output: calls.append(("scaffold", output)) or output,
    )
    args = parse(delivery.Packaging, ["b", "--trusted"])
    assert args.output == "service"
    assert delivery.Packaging().execute(args) == command.EXIT_OK
    assert calls[-2:] == [("b", True), ("scaffold", "service")]


def test_serve_hands_the_server_to_the_serving_loop(monkeypatch):
    events = []

    class Server:
        def __init__(self, model, host, port, gateway, monitor, drain):
            events.append(("init", host, port, drain))

    model = types.SimpleNamespace(info=types.SimpleNamespace(model_name="m"))
    monkeypatch.setattr(
        delivery.bundle.Bundle, "load", lambda path, trusted: model
    )
    monkeypatch.setattr(delivery.server, "Service", Server)
    monkeypatch.setattr(
        delivery.runtime, "fetch", lambda path, trusted: ("fetch", trusted)
    )
    monkeypatch.setattr(
        delivery.runtime,
        "hold",
        lambda adapter, monitor, path, interval, fetch: events.append(
            ("hold", path, interval, fetch)
        ),
    )
    args = parse(delivery.Serve, ["b", "--drain-seconds", "2"])
    assert (args.host, args.port) == ("127.0.0.1", 8000)
    assert args.signals_interval == delivery.runtime.INTERVAL
    assert delivery.Serve().execute(args) == command.EXIT_OK
    assert events == [
        ("init", "127.0.0.1", 8000, 2.0),
        ("hold", None, delivery.runtime.INTERVAL, ("fetch", False)),
    ]


def test_catalog_reads_a_file_and_refuses_a_missing_one(tmp_path):
    (tmp_path / "raw.bin").write_bytes(b"\x00\x01")
    reader = intake.Cataloging()
    assert reader.read(str(tmp_path / "raw.bin")) == b"\x00\x01"
    with pytest.raises(errors.DataError, match="failed to read raw bytes"):
        reader.read(str(tmp_path / "none.bin"))


def test_catalog_reads_standard_input_as_bytes(monkeypatch):
    monkeypatch.setattr(
        intake.sys, "stdin", types.SimpleNamespace(buffer=io.BytesIO(b"abc"))
    )
    assert intake.Cataloging().read("-") == b"abc"
    monkeypatch.setattr(intake.sys, "stdin", io.StringIO("text"))
    with pytest.raises(errors.DataError, match="binary stream"):
        intake.Cataloging().read("-")


def test_catalog_needs_exactly_one_input(tmp_path):
    both = parse(intake.Cataloging, ["x.csv", "--raw-bytes", "y"])
    with pytest.raises(errors.ConfigError, match="not both"):
        intake.Cataloging().execute(both)
    neither = parse(intake.Cataloging, [])
    with pytest.raises(errors.ConfigError, match="tabular input or --raw"):
        intake.Cataloging().execute(neither)


def test_catalog_prints_and_writes_the_evidence(tmp_path, capsys):
    pandas.DataFrame({"a": [1, 2], "y": [0, 1]}).to_csv(
        tmp_path / "t.csv", index=False
    )
    args = parse(
        intake.Cataloging,
        [str(tmp_path / "t.csv"), "--output", str(tmp_path / "cat.json")],
    )
    assert intake.Cataloging().execute(args) == command.EXIT_OK
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads((tmp_path / "cat.json").read_text("utf-8"))
    assert printed["source"]["uri"] == (tmp_path / "t.csv").resolve().as_uri()


def test_ingest_options_have_zero_tolerances_by_default():
    args = parse(intake.IngestCommand, ["--sqlite", "d", "--tables", "a,b"])
    assert (args.eps_key, args.eps_null, args.eps_fk) == (0.0, 0.0, 0.0)
    assert args.output == "ingest"
    assert args.max_key_size == intake.grain.LEVELS
    assert args.max_queries is None


def test_charter_init_writes_a_template_once(tmp_path):
    path = tmp_path / "charter.json"
    args = parse(intake.Charter, ["init", str(path)])
    assert intake.Charter().execute(args) == command.EXIT_OK
    assert json.loads(path.read_text("utf-8")) == (intake.contracts.template())
    assert path.read_text("utf-8").endswith("}\n")
    with pytest.raises(errors.ConfigError, match="exists; pass --force"):
        intake.Charter().execute(args)
    forced = parse(intake.Charter, ["init", str(path), "--force"])
    assert intake.Charter().execute(forced) == command.EXIT_OK


def test_charter_sign_needs_a_key(monkeypatch, tmp_path):
    monkeypatch.delenv("SCMLE_SIGNING_KEY", raising=False)
    args = parse(intake.Charter, ["sign", str(tmp_path / "c.json")])
    with pytest.raises(errors.ConfigError, match="no signing key"):
        intake.Charter().execute(args)
    (tmp_path / "key").write_text("  \n")
    empty = parse(
        intake.Charter,
        ["sign", str(tmp_path / "c.json"), "--key-file", str(tmp_path / "key")],
    )
    with pytest.raises((errors.ConfigError, ValueError)):
        intake.Charter().execute(empty)


def test_frame_requires_its_inputs():
    args = parse(
        intake.Frame,
        ["req.txt", "--source", "s", "--data", "d", "--output", "o"],
    )
    assert args.parse_dates == ""
    assert args.cost_model is None
    with pytest.raises(SystemExit) as caught:
        parse(intake.Frame, ["req.txt"])
    assert caught.value.code == command.EXIT_USAGE


def test_a_key_file_must_hold_a_key(tmp_path):
    (tmp_path / "k").write_text("  secret \n")
    assert rollout.Deployment.read(str(tmp_path / "k")) == "secret"
    (tmp_path / "e").write_text(" \n")
    with pytest.raises(errors.ConfigError, match="holds no key"):
        rollout.Deployment.read(str(tmp_path / "e"))


class Controller:
    made = []

    def __init__(self, deploy_dir, verification_keys=None):
        self.deploy_dir = deploy_dir
        self.keys = verification_keys
        self.halted = []
        self.revoked = []
        self.granted = []
        self.saved = False
        self.loop = types.SimpleNamespace(
            suspend=lambda kind, detail: self.halted.append((kind, detail)),
            save_state=lambda: setattr(self, "saved", True),
        )
        self.envelopes = types.SimpleNamespace(
            revoke=self.revoke, grant=self.granted.append
        )
        Controller.made.append(self)

    def revoke(self, envelope_id, reason):
        if envelope_id == "bad":
            raise deployment_ledger.MisuseError("unknown envelope")
        self.revoked.append((envelope_id, reason))

    def tick(self, inbox):
        return types.SimpleNamespace(to_dict=lambda: {"inbox": inbox})

    def status(self):
        return {"envelopes": ["e1"], "state": "running"}


@pytest.fixture
def controller(monkeypatch):
    Controller.made = []
    monkeypatch.setattr(
        rollout.scmle.operations.controller, "Controller", Controller
    )
    return Controller


def test_operate_tick_reads_the_keys_and_prints_the_result(
    controller, tmp_path, capsys
):
    (tmp_path / "keys.json").write_text('{"k1": "v1"}')
    args = parse(
        rollout.Operation,
        [
            "tick",
            "dep",
            "--inbox",
            "in",
            "--keys-file",
            str(tmp_path / "keys.json"),
        ],
    )
    assert rollout.Operation().execute(args) == command.EXIT_OK
    assert json.loads(capsys.readouterr().out) == {"inbox": "in"}
    assert controller.made[-1].keys == {"k1": "v1"}
    rollout.Operation().execute(
        parse(rollout.Operation, ["tick", "dep", "--inbox", "in"])
    )
    assert controller.made[-1].keys is None


def test_operate_status_halt_and_revoke(controller, capsys):
    rollout.Operation().execute(parse(rollout.Operation, ["status", "dep"]))
    assert json.loads(capsys.readouterr().out) == {
        "envelopes": ["e1"],
        "state": "running",
    }
    rollout.Operation().execute(
        parse(rollout.Operation, ["halt", "dep", "--reason", "storm"])
    )
    made = controller.made[-1]
    assert made.halted == [("operator_halt", {"reason": "storm"})]
    assert made.saved is True
    capsys.readouterr()
    rollout.Operation().execute(
        parse(rollout.Operation, ["revoke", "dep", "e1", "--reason", "r"])
    )
    assert controller.made[-1].revoked == [("e1", "r")]
    assert json.loads(capsys.readouterr().out) == ["e1"]
    with pytest.raises(errors.ConfigError, match="revoke refused: unknown"):
        rollout.Operation().execute(
            parse(rollout.Operation, ["revoke", "dep", "bad", "--reason", "r"])
        )


def test_operate_open_maps_a_ledger_fault(monkeypatch):
    def broken(deploy_dir, verification_keys=None):
        raise deployment_ledger.MisuseError("chain broken")

    monkeypatch.setattr(
        rollout.scmle.operations.controller, "Controller", broken
    )
    with pytest.raises(errors.ArtifactError) as caught:
        rollout.Operation.open("dep")
    assert str(caught.value) == "cannot open the deployment: chain broken"


def test_deploy_verify_record_reports_validity(monkeypatch, tmp_path, capsys):
    (tmp_path / "key").write_text("k")
    record = types.SimpleNamespace(
        verify=lambda key: key == "k", digest=lambda: "d"
    )
    monkeypatch.setattr(
        rollout.scmle.deploy.record, "read_record", lambda path: record
    )
    args = parse(
        rollout.Deployment,
        ["verify-record", "r.json", "--key-file", str(tmp_path / "key")],
    )
    assert rollout.Deployment().execute(args) == command.EXIT_OK
    assert json.loads(capsys.readouterr().out) == {"valid": True, "digest": "d"}
    record.verify = lambda key: False
    assert rollout.Deployment().execute(args) == command.EXIT_FAILED


def test_deploy_gate_blocks_a_refusal_and_writes_an_allowed_record(
    monkeypatch, tmp_path, capsys
):
    (tmp_path / "key").write_text("k")
    written = []
    decision = types.SimpleNamespace(
        allowed=False, record="rec", to_dict=lambda: {"allowed": False}
    )
    monkeypatch.setattr(rollout.release, "gather", lambda path: path)
    monkeypatch.setattr(rollout.gate, "admit", lambda c, key: decision)
    monkeypatch.setattr(rollout.gate, "certify", lambda c, key: decision)
    monkeypatch.setattr(
        rollout.scmle.deploy.record,
        "write_record",
        lambda output, record: written.append((output, record)),
    )
    argv = ["gate", "admit", "m.json", "--key-file", str(tmp_path / "key")]
    assert (
        rollout.Deployment().execute(parse(rollout.Deployment, argv))
        == command.EXIT_BLOCKED
    )
    decision.allowed = True
    assert (
        rollout.Deployment().execute(parse(rollout.Deployment, argv))
        == command.EXIT_OK
    )
    assert written == []
    certify = ["gate", "certify", "m.json", "--key-file", str(tmp_path / "key")]
    rollout.Deployment().execute(
        parse(rollout.Deployment, [*certify, "--output", "o.json"])
    )
    assert written == [("o.json", "rec")]
    assert json.loads(capsys.readouterr().out.split("}\n")[0] + "}") == {
        "allowed": False
    }


def test_deploy_live_options():
    args = parse(
        rollout.Deployment,
        [
            "live",
            "--url",
            "https://x",
            "--token-file",
            "t",
            "--environment",
            "prod",
            "--commit",
            "c",
            "--container-digest",
            "cd",
            "--model-digest",
            "md",
            "--output",
            "o",
        ],
    )
    assert args.golden == "deploy/local/golden"
    assert args.timeout == 30.0
    assert args.read_only_token_file is None


def test_trail_keys_come_from_a_file_or_the_environment(monkeypatch, tmp_path):
    (tmp_path / "keys.json").write_text('{"analyst": "a", "reviewer": 1}')
    assert governance.Trail.keys(str(tmp_path / "keys.json")) == {
        "analyst": "a",
        "reviewer": "1",
    }
    monkeypatch.setenv("SCMLE_SIGNING_KEY", "from-env")
    assert governance.Trail.keys(None) == {"analyst": "from-env"}
    monkeypatch.delenv("SCMLE_SIGNING_KEY")
    assert governance.Trail.keys(None) == {"analyst": None}


def report(ok):
    return types.SimpleNamespace(
        ok=ok,
        steps=2,
        head="0123456789abcdef",
        findings=[types.SimpleNamespace(kind="sealed", step="s1", detail="d")],
        to_dict=lambda: {"ok": ok},
    )


def test_trail_verify_prints_the_findings_and_fails_when_unverified(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        governance.chain, "verify", lambda path, keys: report(True)
    )
    args = parse(governance.Trail, ["verify", "t"])
    assert governance.Trail().execute(args) == command.EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        f"{'sealed':<30}s1: d",
        "2 steps, head 0123456789ab; verified",
    ]
    monkeypatch.setattr(
        governance.chain, "verify", lambda path, keys: report(False)
    )
    with pytest.raises(errors.ArtifactError, match="trail verification"):
        governance.Trail().execute(
            parse(governance.Trail, ["verify", "t", "--json"])
        )
    assert json.loads(capsys.readouterr().out) == {"ok": False}


def test_trail_show_renders(monkeypatch, capsys):
    monkeypatch.setattr(
        governance.scmle.trail.show, "render", lambda path: f"trail {path}"
    )
    assert (
        governance.Trail().execute(parse(governance.Trail, ["show", "t"])) == 0
    )
    assert capsys.readouterr().out == "trail t\n"


def test_claims_list_formats_each_kind_of_row(monkeypatch, capsys):
    rows = [
        {
            "claim_id": "h1",
            "tier": "confirmed",
            "feature": "age",
            "adjusted_p_value": 0.012345,
            "effect": 1.5,
            "withdrawn": False,
        },
        {
            "claim_id": "h2",
            "tier": "unverified",
            "feature": "zip",
            "adjusted_p_value": None,
            "effect": -0.25,
            "withdrawn": True,
        },
        {
            "claim_id": "b1",
            "tier": "unverified",
            "kind": "baseline",
            "status": "met",
            "identification": "",
        },
        {
            "claim_id": "c1",
            "tier": "confirmed",
            "kind": "causal",
            "status": "met",
            "identification": "iv",
        },
    ]
    monkeypatch.setattr(governance.claims_audit, "inventory", lambda path: rows)
    governance.Claims().execute(parse(governance.Claims, ["list", "r"]))
    assert capsys.readouterr().out.splitlines() == [
        f"h1  {'confirmed':<10}  {'age':<24}  adj.p=0.01235  effect=1.5",
        (
            f"h2  {'unverified':<10}  {'zip':<24}  adj.p=-  effect=-0.25"
            "  withdrawn before C"
        ),
        f"b1  {'unverified':<10}  registered baseline  status=met",
        (
            f"c1  {'confirmed':<10}  registered causal  status=met"
            "  identification=iv"
        ),
        "4 claim(s)",
    ]
    governance.Claims().execute(
        parse(governance.Claims, ["list", "r", "--tier", "confirmed", "--json"])
    )
    shown = json.loads(capsys.readouterr().out)
    assert [r["claim_id"] for r in shown] == ["h1", "c1"]


def test_claims_verify_prints_steps_and_fails_on_a_failed_step(
    monkeypatch, capsys, tmp_path
):
    found = types.SimpleNamespace(status="pass", name="n", detail="d")
    result = types.SimpleNamespace(
        ok=True, verified=1, findings=[found], to_dict=lambda: {"ok": True}
    )
    keys = []
    monkeypatch.setattr(
        governance.claims_audit,
        "verify",
        lambda path, key: keys.append(key) or result,
    )
    (tmp_path / "key").write_text(" k \n")
    args = parse(
        governance.Claims,
        ["verify", "r", "--key-file", str(tmp_path / "key")],
    )
    assert governance.Claims().execute(args) == command.EXIT_OK
    assert keys == ["k"]
    assert capsys.readouterr().out.splitlines() == [
        f"{'PASS':<8}n: d",
        "1 of 1 steps passed",
    ]
    result.ok = False
    with pytest.raises(errors.ArtifactError, match="claims verification"):
        governance.Claims().execute(parse(governance.Claims, ["verify", "r"]))
    assert keys[-1] is None
    assert capsys.readouterr().out.splitlines()[-1] == (
        "1 of 1 steps passed; verification FAILED"
    )


def test_claims_show_prints_one_claim(monkeypatch, capsys):
    monkeypatch.setattr(
        governance.claims_audit,
        "inspect",
        lambda path, claim: {"claim": claim, "run": path},
    )
    governance.Claims().execute(parse(governance.Claims, ["show", "r", "h1"]))
    assert json.loads(capsys.readouterr().out) == {"claim": "h1", "run": "r"}


class Loop:
    resumed = []

    @classmethod
    def resume(cls, deploy_dir):
        cls.resumed.append(deploy_dir)
        return cls()

    def observe(self, frame, predictions, decisions):
        self.seen = (predictions, decisions)
        Loop.observed = self.seen
        return types.SimpleNamespace(
            batch=3,
            routing=types.SimpleNamespace(status=["auto", "auto", "review"]),
        )

    def mature(self, losses):
        Loop.matured = losses
        return len(losses)

    def step(self):
        return types.SimpleNamespace(
            status="ok",
            mechanism=types.SimpleNamespace(value="elond"),
            events=[("a", 1)],
            escalation=None,
        )

    def pending_escalations(self):
        return [
            types.SimpleNamespace(
                id="e1",
                question="q",
                question_hash="h",
                evidence=("x",),
                proposed_default="hold",
            )
        ]


@pytest.fixture
def loop(monkeypatch):
    Loop.resumed = []
    monkeypatch.setattr(governance.scmle.governance.loop, "Loop", Loop)
    return Loop


def test_loop_observe_counts_the_routing(loop, tmp_path, capsys):
    pandas.DataFrame({"prediction": [1, 0, 1], "decision": [1, 1, 0]}).to_csv(
        tmp_path / "b.csv", index=False
    )
    args = parse(
        governance.Loop,
        [
            "observe",
            "dep",
            "--batch",
            str(tmp_path / "b.csv"),
        ],
    )
    assert governance.Loop().execute(args) == command.EXIT_OK
    assert json.loads(capsys.readouterr().out) == {
        "batch": 3,
        "routing": {"auto": 2, "review": 1},
    }
    assert loop.observed == ([1, 0, 1], [1, 1, 0])
    assert loop.resumed == ["dep"]


def test_loop_mature_step_and_pending(loop, tmp_path, capsys):
    pandas.DataFrame({"case_id": [7, 8], "loss": [0.5, 1.0]}).to_csv(
        tmp_path / "l.csv", index=False
    )
    governance.Loop().execute(
        parse(
            governance.Loop,
            ["mature", "dep", "--labels", str(tmp_path / "l.csv")],
        )
    )
    assert json.loads(capsys.readouterr().out) == {"matched": 2}
    assert loop.matured == {"7": 0.5, "8": 1.0}
    governance.Loop().execute(parse(governance.Loop, ["step", "dep"]))
    assert json.loads(capsys.readouterr().out) == {
        "status": "ok",
        "mechanism": "elond",
        "events": [["a", 1]],
        "escalation": None,
    }
    governance.Loop().execute(parse(governance.Loop, ["pending", "dep"]))
    assert json.loads(capsys.readouterr().out) == [
        {
            "id": "e1",
            "question": "q",
            "question_hash": "h",
            "evidence": ["x"],
            "proposed_default": "hold",
        }
    ]


def test_loop_verify_maps_a_broken_ledger(monkeypatch, capsys):
    def broken(path):
        raise deployment_ledger.MisuseError("bad chain")

    monkeypatch.setattr(
        governance.deployment_ledger.DeploymentLedger, "load", broken
    )
    with pytest.raises(errors.ArtifactError) as caught:
        governance.Loop().execute(parse(governance.Loop, ["verify", "dep"]))
    assert str(caught.value) == "ledger verification failed: bad chain"
    monkeypatch.setattr(
        governance.deployment_ledger.DeploymentLedger,
        "load",
        lambda path: types.SimpleNamespace(rows=[1, 2], head="h"),
    )
    governance.Loop().execute(parse(governance.Loop, ["verify", "dep"]))
    assert json.loads(capsys.readouterr().out) == {
        "ok": True,
        "rows": 2,
        "head": "h",
    }
