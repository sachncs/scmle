"""The prediction service behind the gateway: scopes, tenants, limits, audit."""

import json
import pathlib
import re
import shutil
import urllib.error
import urllib.request

import pytest

from scmle.contracts import errors
from scmle.deploy import runtime
from scmle.deploy import verify
from scmle.governance import deployment_ledger
from scmle.inference import parity
from scmle.inference import serving
from scmle.security import access
from scmle.security import audit
from scmle.security import tokens
from tests import support_identity as idp

ROOT = pathlib.Path(__file__).resolve().parents[2]
KEY = "security-test-signing-key-31c9"
# Hex digests in the log can contain short words, so the rejected credential
# must be a string no digest can contain.
BAD_TOKEN = "not-a-token-zz"


@pytest.fixture(scope="module")
def reference():
    golden = parity.load_golden(ROOT / "deploy" / "local" / "golden")
    return parity.reference_bundle(golden.spec)


def make_gateway(clock, **options):
    check = tokens.TokenVerifier(
        tokens.KeySet(
            [
                tokens.VerificationKey.from_jwk(idp.hs_jwk("hs-1")),
                tokens.VerificationKey.from_jwk(idp.rsa_jwk("k1", "rs-1")),
            ]
        ),
        idp.ISSUER,
        idp.AUDIENCE,
        clock=clock,
    )
    log = audit.AuditLog.open(None, clock=clock)
    return access.Gateway(check, audit=log, **options)


def bearer(clock, **claims):
    return tokens.sign_hs256(
        idp.HS_SECRET, "hs-1", idp.claims_at(clock.now, **claims)
    )


def call(base, path, token=None, payload=None):
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(base + path, data, headers)
    try:
        with urllib.request.urlopen(request, timeout=10) as reply:
            return reply.status, json.loads(reply.read()), dict(reply.headers)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read()), dict(error.headers)


@pytest.fixture
def service(reference):
    clock = idp.Clock()
    gateway = make_gateway(clock, tenant="acme")
    adapter = serving.HttpServing(reference, gateway=gateway)
    adapter.start()
    try:
        yield adapter, gateway, clock
    finally:
        adapter.stop()


def record_of(reference):
    return [dict.fromkeys(reference.features, 0.5)]


def test_probes_are_public_and_everything_else_needs_a_token(service):
    adapter, ignored, clock = service
    assert call(adapter.address, "/livez")[0] == 200
    assert call(adapter.address, "/readyz")[0] == 200
    for path in ("/health", "/schema"):
        status, body, headers = call(adapter.address, path)
        assert status == 401 and body["error"] == "authentication required"
        assert headers["WWW-Authenticate"].startswith("Bearer")
    assert call(adapter.address, "/predict", None, {"records": []})[0] == 401
    assert call(adapter.address, "/health", bearer(clock))[0] == 200


def test_predict_works_with_the_scope_and_not_without(service, reference):
    adapter, ignored, clock = service
    payload = {"records": record_of(reference)}
    ok = call(adapter.address, "/v1/predict", bearer(clock), payload)
    assert ok[0] == 200 and ok[1]["record_count"] == 1
    read_only = bearer(clock, scope="model:read")
    status, body, ignored = call(
        adapter.address, "/predict", read_only, payload
    )
    assert status == 403 and body["error"] == "forbidden"


def test_error_bodies_never_say_why(service, reference):
    adapter, ignored, clock = service
    payload = {"records": record_of(reference)}
    bodies = []
    for token in (
        bearer(clock, exp=int(clock.now) - 3600),
        bearer(clock, aud="x"),
        bearer(clock) + "tamper",
        "garbage",
        None,
    ):
        status, body, ignored = call(
            adapter.address, "/predict", token, payload
        )
        assert status == 401
        bodies.append({k: v for k, v in body.items() if k != "request_id"})
    assert all(b == bodies[0] for b in bodies)
    text = json.dumps(bodies[0])
    for word in ("expired", "audience", "signature", "hs-1", "issuer"):
        assert word not in text


def test_a_token_of_another_tenant_is_forbidden(service, reference):
    adapter, gateway, clock = service
    payload = {"records": record_of(reference)}
    other = bearer(clock, tenant="globex")
    assert call(adapter.address, "/predict", other, payload)[0] == 403
    none = bearer(clock, tenant=None)
    assert call(adapter.address, "/predict", none, payload)[0] == 403
    reasons = [r["body"]["reason"] for r in gateway.audit.rows(audit.ACCESS)]
    assert reasons == ["tenant_mismatch", "tenant_mismatch"]


def test_the_rate_limit_answers_429_with_retry_after(reference):
    clock = idp.Clock()
    ticks = idp.Clock(0.0)
    gateway = make_gateway(
        clock, limiter=access.RateLimiter(1.0, 2, clock=ticks)
    )
    adapter = serving.HttpServing(reference, gateway=gateway)
    adapter.start()
    try:
        token = bearer(clock)
        codes = [
            call(adapter.address, "/health", token)[0] for ignored in range(3)
        ]
        assert codes == [200, 200, 429]
        status, ignored, headers = call(adapter.address, "/health", token)
        assert status == 429 and int(headers["Retry-After"]) >= 1
        ticks.now += 2.0
        assert call(adapter.address, "/health", token)[0] == 200
    finally:
        adapter.stop()


def test_guessing_credentials_is_slowed(reference):
    clock = idp.Clock()
    ticks = idp.Clock(0.0)
    gateway = make_gateway(
        clock, failure_limiter=access.RateLimiter(1.0, 3, clock=ticks)
    )
    adapter = serving.HttpServing(reference, gateway=gateway)
    adapter.start()
    try:
        codes = [
            call(adapter.address, "/health", f"guess{n}")[0] for n in range(5)
        ]
        assert codes == [401, 401, 401, 429, 429]
    finally:
        adapter.stop()


def test_the_request_size_limit_still_applies_behind_the_gateway(
    reference,
):
    clock = idp.Clock()
    adapter = serving.HttpServing(
        reference, gateway=make_gateway(clock), max_request_bytes=64
    )
    adapter.start()
    try:
        big = {"records": record_of(reference) * 20}
        assert call(adapter.address, "/predict", bearer(clock), big)[0] == 413
    finally:
        adapter.stop()


def test_audit_records_activation_access_and_admin_in_a_valid_chain(
    reference, tmp_path
):
    clock = idp.Clock()
    path = tmp_path / "audit.jsonl"
    log = audit.AuditLog.open(path, clock=clock)
    check = tokens.TokenVerifier(
        tokens.KeySet([tokens.VerificationKey.from_jwk(idp.hs_jwk("hs-1"))]),
        idp.ISSUER,
        idp.AUDIENCE,
        clock=clock,
    )
    gateway = access.Gateway(check, audit=log)
    adapter = serving.HttpServing(reference, gateway=gateway)
    adapter.start()
    try:
        call(adapter.address, "/health", bearer(clock))
        call(adapter.address, "/health", BAD_TOKEN)
    finally:
        adapter.stop()
    gateway.rotate(
        tokens.KeySet(
            [tokens.VerificationKey.from_jwk(idp.rsa_jwk("k1", "rs-2"))]
        ),
        actor="ops",
        reason="scheduled",
    )
    loaded = deployment_ledger.DeploymentLedger.load(path)
    rows = [r["body"] for r in loaded.rows if r["kind"] == "audit"]
    assert [r["event"] for r in rows] == [
        "activation",
        "access",
        "access",
        "admin",
    ]
    assert rows[0]["artifact_digest"] == adapter.artifact_digest
    assert rows[1]["outcome"] == "allowed" and rows[1]["actor"] == "alice"
    assert rows[2]["outcome"] == "denied"
    assert rows[2]["reason"] == "malformed"
    assert rows[3]["kids_before"] == ["hs-1"] and rows[3]["kids_after"] == [
        "rs-2"
    ]
    assert "Bearer" not in path.read_text() and BAD_TOKEN not in json.dumps(
        rows[2:3]
    )


def test_an_edited_audit_row_breaks_the_chain(tmp_path):
    log = audit.AuditLog.open(tmp_path / "a.jsonl")
    log.admin("ops", "one")
    log.admin("ops", "two")
    lines = (tmp_path / "a.jsonl").read_text().splitlines()
    lines[1] = lines[1].replace('"one"', '"uno"')
    (tmp_path / "a.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(deployment_ledger.DeploymentLedgerError):
        deployment_ledger.DeploymentLedger.load(tmp_path / "a.jsonl")


def test_a_failed_activation_is_audited(tmp_path):
    clock = idp.Clock()
    gateway = make_gateway(clock)

    def failing(readiness):
        raise errors.ArtifactError("bundle is tampered with")

    adapter = serving.HttpServing(loader=failing, gateway=gateway)
    adapter.start()
    try:
        adapter.wait_until_settled(10)
    finally:
        adapter.stop()
    row = gateway.audit.rows(audit.ACTIVATION)[0]["body"]
    assert row["outcome"] == "denied" and "tampered" in row["reason"]


def test_no_unauthenticated_server_can_listen_beyond_loopback(reference):
    with pytest.raises(errors.ConfigError, match="without authentication"):
        serving.HttpServing(reference, host="0.0.0.0")  # nosec B104
    for host in ("127.0.0.1", "::1", "localhost"):
        serving.HttpServing(reference, host=host)
    clock = idp.Clock()
    serving.HttpServing(
        reference,
        host="0.0.0.0",
        gateway=make_gateway(clock),  # nosec B104
    )


def test_the_runtime_default_host_needs_authentication():
    plain = runtime.settings({})
    assert plain["host"] == "0.0.0.0" and plain["gateway"] is None  # nosec B104
    with pytest.raises(errors.ConfigError, match="without authentication"):
        access.require_authentication_for(plain["host"], plain["gateway"])
    secured = runtime.settings(
        {
            "SCMLE_AUTH_HS256_SECRET": idp.HS_SECRET.decode(),
            "SCMLE_AUTH_ISSUER": idp.ISSUER,
            "SCMLE_AUTH_AUDIENCE": idp.AUDIENCE,
            "SCMLE_TENANT": "acme",
        }
    )
    access.require_authentication_for(secured["host"], secured["gateway"])
    assert secured["gateway"].tenant == "acme"


def test_authentication_settings_need_an_issuer_and_audience():
    with pytest.raises(errors.ConfigError, match="ISSUER"):
        access.gateway_from_environment(
            {"SCMLE_AUTH_HS256_SECRET": idp.HS_SECRET.decode()}
        )
    assert access.gateway_from_environment({}) is None


def test_the_shipped_compose_file_publishes_no_port_and_secures_the_apis():
    text = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    assert not re.search(r"^\s*ports:", text, re.MULTILINE)
    demo = (ROOT / "deploy" / "local" / "demo.env").read_text()
    for name in ("SCMLE_AUTH_HS256_SECRET", "SCMLE_AUTH_ISSUER"):
        assert name in demo
    services = re.split(r"\n  (?=[a-z-]+:\n)", text.split("services:", 1)[1])
    for block in services:
        if "serve.py" in block:
            assert "demo.env" in block, block.splitlines()[0]


def test_the_generated_image_does_not_turn_authentication_off(
    reference, tmp_path
):
    folder = tmp_path / "svc"
    reference.export_service(folder, signing_key=KEY)
    for name in ("Dockerfile", "serve.py", "README.md"):
        text = (folder / name).read_text(encoding="utf-8").lower()
        assert "auth_disabled" not in text and "no-auth" not in text


def test_a_tampered_artifact_never_loads_in_production(reference, tmp_path):
    folder = tmp_path / "svc"
    reference.export_service(folder, signing_key=KEY)
    environ = {"SCMLE_ENV": "production", "SCMLE_SIGNING_KEY": KEY}
    assert verify.verify_service(folder, environ).bundle_signed
    for name in (
        "model/pipeline.joblib",
        "requirements.lock",
        "provenance.json",
    ):
        copy = tmp_path / ("copy-" + name.replace("/", "-"))
        shutil.copytree(folder, copy)
        path = copy / name
        if name == "provenance.json":
            document = json.loads(path.read_text(encoding="utf-8"))
            document["statement"]["predicate"]["builder"] = "someone-else"
            path.write_text(json.dumps(document), encoding="utf-8")
        else:
            with path.open("ab") as handle:
                handle.write(b"\n ")
        with pytest.raises(errors.ArtifactError):
            verify.verify_service(copy, environ)
    wrong = {**environ, "SCMLE_SIGNING_KEY": "another-key-entirely-0000"}
    with pytest.raises(errors.ArtifactError):
        verify.verify_service(folder, wrong)


def test_the_modality_service_is_gated_the_same_way(tmp_path):
    from scmle.inference import modality_serving
    from scmle.models import modality_lifecycle
    from tests.unit import test_modality_serving as helpers

    reference = helpers.adapter()
    report = modality_lifecycle.ModalityLifecycle().run(
        reference, tmp_path / "package", signing_key="serving-key"
    )
    predictor = modality_lifecycle.ModalityPredictor.from_package(
        reference, report.package.path, "serving-key"
    )
    open_host = modality_serving.ModalityHttpServing(
        predictor,
        helpers.DocumentDecoder(),
        host="0.0.0.0",  # nosec B104
    )
    with pytest.raises(errors.ConfigError, match="without authentication"):
        open_host.start()
    clock = idp.Clock()
    service = modality_serving.ModalityHttpServing(
        predictor, helpers.DocumentDecoder()
    )
    service.set_gateway(make_gateway(clock))
    service.start()
    try:
        assert call(service.address, "/livez")[0] == 200
        assert call(service.address, "/health")[0] == 401
        assert call(service.address, "/health", bearer(clock))[0] == 200
        payload = {"samples": []}
        assert call(service.address, "/predict", None, payload)[0] == 401
        assert (
            call(service.address, "/predict", bearer(clock), payload)[0] == 422
        )
    finally:
        service.stop()
