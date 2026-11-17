"""Guarantees of docs/TENANT_ISOLATION.md that no other test enforced."""

import json
import pathlib
import time
import urllib.error
import urllib.request

import pytest

from scmle.contracts import errors
from scmle.data.store import objectstore
from scmle.governance import deployment_ledger
from scmle.inference import parity
from scmle.inference import serving
from scmle.runners import roles
from scmle.security import access
from scmle.security import audit
from scmle.security import tokens
from tests import support_identity

ROOT = pathlib.Path(__file__).resolve().parents[2]
# A feature value no digest, id or timestamp of the audit file can contain.
MARKER = 0.918273645546372


@pytest.fixture(scope="module")
def reference():
    golden = parity.goldens(ROOT / "deploy" / "local" / "golden")
    return parity.handwrite(golden.spec)


def gateway(clock, log=None, **options):
    check = tokens.Verifier(
        tokens.Keystore([tokens.Key.jwk(support_identity.hs_jwk("hs-1"))]),
        support_identity.ISSUER,
        support_identity.AUDIENCE,
        clock=clock,
    )
    log = log or audit.Log.open(None, clock=clock)
    return access.Gateway(check, audit=log, tenant="acme", **options)


def bearer(clock, **claims):
    return tokens.sign_hs256(
        support_identity.HS_SECRET,
        "hs-1",
        support_identity.claims_at(clock.now, **claims),
    )


def call(base, path, token=None, payload=None):
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(base + path, data, headers)
    try:
        with urllib.request.urlopen(request, timeout=10) as reply:
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


@pytest.fixture
def service(reference):
    clock = support_identity.Clock()
    gate = gateway(clock)
    adapter = serving.HttpServing(reference, gateway=gate)
    adapter.start()
    try:
        yield adapter, gate, clock
    finally:
        adapter.stop()


def records(reference, value=0.5):
    return [dict.fromkeys(reference.features, value)]


def without_id(body):
    return {k: v for k, v in body.items() if k != "request_id"}


def test_the_tenant_comes_from_the_token_not_the_request(service, reference):
    """Enforces docs/TENANT_ISOLATION.md:18 (both sentences).

    A token of another tenant stays forbidden whether the request names this
    deployment's tenant in the path, the query or the body; a token of this
    tenant cannot move to another tenant by naming it either.
    """
    adapter, gate, clock = service
    payload = {"records": records(reference)}
    foreign = bearer(clock, tenant="globex")
    for path, body in (
        ("/predict", payload),
        ("/predict?tenant=acme", payload),
        ("/acme/predict", payload),
        ("/predict", {**payload, "tenant": "acme"}),
    ):
        assert call(adapter.address, path, foreign, body)[0] == 403, path
    own = bearer(clock)
    assert call(adapter.address, "/predict", own, payload)[0] == 200
    assert call(adapter.address, "/predict?tenant=globex", own, payload)[0] in (
        400,
        404,
    )
    assert call(
        adapter.address, "/predict", own, {**payload, "tenant": "globex"}
    )[0] in (400, 404)
    allowed = [
        r["body"]
        for r in gate.audit.rows(audit.ACCESS)
        if r["body"]["outcome"] == "allowed"
    ]
    assert {r["tenant"] for r in allowed} == {"acme"}


def test_the_probes_disclose_only_liveness_readiness_and_identity(
    service, reference
):
    """Enforces docs/TENANT_ISOLATION.md:23 (what the public paths say)."""
    adapter = service[0]
    status, alive = call(adapter.address, "/livez")
    assert status == 200
    assert set(without_id(alive)) <= {"status", "busy", "stuck", "misses"}
    status, ready = call(adapter.address, "/readyz")
    assert status == 200
    assert set(without_id(ready)) <= {
        "ready",
        "phase",
        "completed",
        "notes",
        "reason",
        "workers",
        "model_version",
        "artifact_digest",
    }
    text = json.dumps([alive, ready])
    for secret in ("acme", "hs-1", "alice", *reference.features):
        assert secret not in text


def test_before_it_is_ready_the_probe_gives_no_model_identity():
    """Enforces docs/TENANT_ISOLATION.md:23 (version and digest once ready)."""

    def failing(readiness):
        raise errors.ArtifactError("bundle is tampered with")

    adapter = serving.HttpServing(
        loader=failing, gateway=gateway(support_identity.Clock())
    )
    adapter.start()
    try:
        adapter.loading.join(10)
        status, body = call(adapter.address, "/readyz")
    finally:
        adapter.stop()
    assert status == 503
    assert "model_version" not in body and "artifact_digest" not in body


def test_refusals_use_three_fixed_messages(reference):
    """Enforces docs/TENANT_ISOLATION.md:45 and docs/TENANT_ISOLATION.md:46.

    A missing scope and a foreign tenant give the same 403 body, so the body
    tells neither which check failed nor whether the other tenant exists.
    """
    clock = support_identity.Clock()
    ticks = support_identity.Clock(0.0)
    gate = gateway(clock, limiter=access.RateLimiter(1.0, 1, clock=ticks))
    adapter = serving.HttpServing(reference, gateway=gate)
    adapter.start()
    try:
        payload = {"records": records(reference)}
        unauthenticated = call(adapter.address, "/predict", None, payload)
        scope = call(
            adapter.address,
            "/predict",
            bearer(clock, scope="model:read"),
            payload,
        )
        tenant = call(
            adapter.address,
            "/predict",
            bearer(clock, tenant="globex", sub="bob"),
            payload,
        )
        own = bearer(clock, sub="carol")
        call(adapter.address, "/health", own)
        flood = call(adapter.address, "/health", own)
    finally:
        adapter.stop()
    assert unauthenticated[0] == 401
    assert unauthenticated[1]["error"] == "authentication required"
    assert scope[0] == tenant[0] == 403
    assert without_id(scope[1]) == without_id(tenant[1])
    assert scope[1]["error"] == "forbidden"
    assert flood[0] == 429 and flood[1]["error"] == "too many requests"
    for body in (unauthenticated[1], scope[1], tenant[1], flood[1]):
        text = json.dumps(without_id(body))
        for word in ("globex", "acme", "hs-1", "scope", "tenant", "model:"):
            assert word not in text


def test_audit_rows_carry_the_request_id_and_never_a_body(reference, tmp_path):
    """Enforces docs/TENANT_ISOLATION.md:49."""
    clock = support_identity.Clock()
    path = tmp_path / "audit.jsonl"
    gate = gateway(clock, audit.Log.open(path, clock=clock))
    adapter = serving.HttpServing(reference, gateway=gate)
    adapter.start()
    token = bearer(clock)
    try:
        payload = {"records": records(reference, MARKER)}
        call(adapter.address, "/predict", token, payload)
        call(adapter.address, "/predict", token + "x", payload)
    finally:
        adapter.stop()
    text = path.read_text()
    assert str(MARKER) not in text and repr(MARKER)[:10] not in text
    assert token not in text and "Bearer" not in text
    loaded = deployment_ledger.DeploymentLedger.load(path)
    rows = [r["body"] for r in loaded.rows if r["kind"] == "audit"]
    access_rows = [r for r in rows if r["event"] == "access"]
    assert [r["outcome"] for r in access_rows] == ["allowed", "denied"]
    for row in access_rows:
        assert row["request_id"].startswith("srv-")
        assert row["route"] == "/predict"
    assert access_rows[0]["actor"] == "alice"


def admin_rows(path, count):
    log = audit.Log.open(path)
    for index in range(count):
        log.admin("ops", f"step {index}")
    return path.read_text().splitlines()


def test_the_chain_catches_edits_gaps_and_reordering_but_not_a_cut_tail(
    tmp_path,
):
    """Enforces docs/TENANT_ISOLATION.md:69."""
    path = tmp_path / "a.jsonl"
    lines = admin_rows(path, 4)
    gap = [lines[0], *lines[2:]]
    swapped = [lines[0], lines[2], lines[1], lines[3]]
    for changed in (gap, swapped):
        path.write_text("\n".join(changed) + "\n")
        with pytest.raises(deployment_ledger.MisuseError):
            deployment_ledger.DeploymentLedger.load(path)
    path.write_text("\n".join(lines[:-1]) + "\n")
    cut = deployment_ledger.DeploymentLedger.load(path)
    assert len(cut.rows) == len(lines) - 1


def test_the_revocation_file_named_by_the_environment_is_read_per_request(
    tmp_path,
):
    """Enforces docs/TENANT_ISOLATION.md:84 and docs/TENANT_ISOLATION.md:66."""
    revoked = tmp_path / "revoked.json"
    revoked.write_text(json.dumps({"tokens": [], "subjects": []}))
    gate = access.gateway_from_environment(
        {
            "SCMLE_AUTH_HS256_SECRET": support_identity.HS_SECRET.decode(),
            "SCMLE_AUTH_ISSUER": support_identity.ISSUER,
            "SCMLE_AUTH_AUDIENCE": support_identity.AUDIENCE,
            "SCMLE_AUTH_REVOKED_FILE": str(revoked),
        }
    )
    claims = support_identity.claims_at(time.time())
    header = "Bearer " + tokens.sign_hs256(
        support_identity.HS_SECRET, "env", claims
    )

    def status():
        try:
            gate.check("GET", "/health", header, "127.0.0.1", "r")
        except access.DeniedError as denied:
            return denied.status
        return 200

    assert status() == 200
    revoked.write_text(json.dumps({"tokens": [claims["jti"]]}))
    assert status() == 401
    revoked.write_text(json.dumps({"subjects": ["someone-else"]}))
    assert status() == 200
    revoked.write_text("{ not json")
    assert status() == 401
    revoked.unlink()
    assert status() == 401


def test_serving_and_training_roles_keep_to_their_prefixes(tmp_path):
    """Enforces docs/TENANT_ISOLATION.md:41."""
    inner = objectstore.LocalObjectStore(str(tmp_path / "store"))
    inner.put("bundles/model/v1/manifest.json", b"{}")
    serving_store = roles.RoleStore(inner, roles.SERVING_ROLE)
    training_store = roles.RoleStore(inner, roles.TRAINING_ROLE)
    assert serving_store.get("bundles/model/v1/manifest.json")[0] == b"{}"
    serving_store.put("audit/trail.jsonl", b"row")
    for key in (
        "runs/r1/manifest.json",
        "artifacts/r1/checkpoint.bin",
        "bundles/model/v2/manifest.json",
    ):
        with pytest.raises(roles.RoleDeniedError):
            serving_store.put(key, b"x")
    training_store.put("runs/r1/manifest.json", b"{}")
    training_store.put("artifacts/r1/checkpoint.bin", b"x")
    with pytest.raises(roles.RoleDeniedError):
        training_store.get("bundles/model/v1/manifest.json")
    with pytest.raises(roles.RoleDeniedError):
        training_store.put("bundles/model/v2/manifest.json", b"x")
    assert not inner.exists("bundles/model/v2/manifest.json")
