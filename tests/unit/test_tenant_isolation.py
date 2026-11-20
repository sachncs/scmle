"""Tenant isolation at the storage layer and in token verification."""

import json
import os

import pytest

from scmle import cli
from scmle.contracts import errors
from scmle.contracts import experiments
from scmle.security import access
from scmle.security import tokens
from scmle.storage import conformance
from scmle.storage import objectstore
from scmle.storage import stores
from scmle.storage import tenancy
from tests import support_identity

links = pytest.mark.skipif(
    os.name == "nt", reason="symbolic links need privileges on Windows"
)


@pytest.fixture
def shared(tmp_path):
    return objectstore.Disk(tmp_path / "root")


@pytest.fixture
def pair(shared):
    acme = tenancy.Tenancy(shared, "acme")
    globex = tenancy.Tenancy(shared, "globex")
    globex.put("secret.json", b"globex")
    return acme, globex


def test_the_conformance_suite_passes_inside_a_tenancy(shared):
    report = conformance.certify(tenancy.Tenancy(shared, "acme"))
    assert [r.name for r in report.results if not r.passed] == []
    assert shared.catalog() == []


def test_keys_are_namespaced_and_metadata_is_relative(shared, pair):
    acme = pair[0]
    meta = acme.put("bundles/v1.json", b"acme")
    assert meta.key == "bundles/v1.json"
    assert shared.get("tenants/acme/bundles/v1.json")[0] == b"acme"
    assert acme.get("bundles/v1.json")[1].key == "bundles/v1.json"
    assert acme.uri("a") == shared.location + "/tenants/acme/a"
    assert [m.key for m in acme.catalog()] == ["bundles/v1.json"]


@pytest.mark.parametrize(
    "key",
    [
        "../globex/secret.json",
        "a/../../globex/secret.json",
        "/tenants/globex/secret.json",
        "..%2Fglobex%2Fsecret.json",
        "%2e%2e/globex/secret.json",
        "..\\globex\\secret.json",
        "a//secret.json",
        "./secret.json",
        "secret.json\x00",
        chr(0xFF0E) * 2 + "/globex/secret.json",
    ],
)
def test_a_crafted_key_is_refused_for_every_operation(pair, key):
    acme, globex = pair
    operations = (
        lambda: acme.get(key),
        lambda: acme.head(key),
        lambda: acme.put(key, b"x"),
        lambda: acme.put(key, b"x", if_absent=True),
        lambda: acme.delete(key),
        lambda: acme.catalog(key + "/"),
    )
    for operation in operations:
        with pytest.raises(objectstore.StoreError):
            operation()
    assert globex.get("secret.json")[0] == b"globex"


@pytest.mark.parametrize(
    "key", ["tenants/globex/secret.json", "TENANTS/globex/secret.json"]
)
def test_a_key_naming_the_namespace_root_is_a_tenant_error(pair, key):
    acme, globex = pair
    for operation in (
        lambda: acme.get(key),
        lambda: acme.put(key, b"x"),
        lambda: acme.delete(key),
        lambda: acme.catalog("tenants/"),
    ):
        with pytest.raises(tenancy.TenantError):
            operation()
    assert globex.get("secret.json")[0] == b"globex"


def test_one_tenant_sees_nothing_of_another(pair):
    acme, globex = pair
    assert acme.catalog() == []
    with pytest.raises(objectstore.AbsentError):
        acme.get("secret.json")
    assert not acme.exists("secret.json")
    acme.put("secret.json", b"acme")
    acme.delete("secret.json")
    assert globex.get("secret.json")[0] == b"globex"


def test_a_prefix_never_matches_a_longer_tenant_name(shared):
    tenancy.Tenancy(shared, "acme2").put("x", b"2")
    assert tenancy.Tenancy(shared, "acme").catalog() == []


@pytest.mark.parametrize(
    "name",
    ["", "Acme", "ac/me", "..", "-acme", "acme-", "ac me", "a" * 64, None],
)
def test_an_invalid_tenant_name_is_refused(shared, name):
    with pytest.raises(tenancy.TenantError):
        tenancy.Tenancy(shared, name)


def test_the_longest_tenant_name_is_accepted(shared):
    name = "a" * tenancy.MAX_TENANT_LENGTH
    assert tenancy.Tenancy(shared, name).put("x", b"1").key == "x"


@links
def test_a_symlinked_directory_into_another_tenant_is_refused(shared, pair):
    acme, globex = pair
    (shared.root / "tenants" / "acme").mkdir()
    (shared.root / "tenants" / "acme" / "door").symlink_to(
        shared.root / "tenants" / "globex", target_is_directory=True
    )
    for operation in (
        lambda: acme.get("door/secret.json"),
        lambda: acme.put("door/secret.json", b"x"),
        lambda: acme.delete("door/secret.json"),
    ):
        with pytest.raises(objectstore.EscapeError):
            operation()
    assert acme.catalog() == []
    assert globex.get("secret.json")[0] == b"globex"


@links
def test_a_symlinked_file_into_another_tenant_is_refused(shared, pair):
    acme, globex = pair
    (shared.root / "tenants" / "acme").mkdir()
    (shared.root / "tenants" / "acme" / "copy.json").symlink_to(
        shared.root / "tenants" / "globex" / "secret.json"
    )
    with pytest.raises(objectstore.EscapeError):
        acme.get("copy.json")
    with pytest.raises(objectstore.EscapeError):
        acme.put("copy.json", b"x")
    assert acme.catalog() == []
    assert globex.get("secret.json")[0] == b"globex"


@links
def test_a_symlinked_tenant_directory_is_refused(shared, pair):
    acme = pair[0]
    (shared.root / "tenants" / "acme").symlink_to(
        shared.root / "tenants" / "globex", target_is_directory=True
    )
    with pytest.raises(objectstore.EscapeError):
        acme.get("secret.json")
    assert acme.catalog() == []


class Leaky(objectstore.Disk):
    """A backend that ignores the prefix of a listing."""

    def catalog(self, prefix=""):
        return super().catalog("")


def test_a_backend_that_ignores_the_prefix_cannot_leak(tmp_path):
    inner = Leaky(tmp_path / "leaky")
    tenancy.Tenancy(inner, "globex").put("secret.json", b"globex")
    with pytest.raises(tenancy.TenantError):
        tenancy.Tenancy(inner, "acme").catalog()


def test_adopt_moves_legacy_objects_and_resumes(shared):
    shared.put("bundles/v1.json", b"one")
    shared.put("runs/r1.json", b"two")
    tenancy.Tenancy(shared, "globex").put("keep.json", b"g")
    acme = tenancy.Tenancy(shared, "acme")
    # An interrupted earlier run copied one object but did not remove it.
    shared.put("tenants/acme/bundles/v1.json", b"one")
    assert acme.adopt() == ["bundles/v1.json", "runs/r1.json"]
    assert [m.key for m in acme.catalog()] == [
        "bundles/v1.json",
        "runs/r1.json",
    ]
    assert [m.key for m in shared.catalog()] == [
        "tenants/acme/bundles/v1.json",
        "tenants/acme/runs/r1.json",
        "tenants/globex/keep.json",
    ]
    assert acme.adopt() == []


def test_adopt_refuses_to_overwrite_a_different_object(shared):
    shared.put("a.json", b"legacy")
    acme = tenancy.Tenancy(shared, "acme")
    acme.put("a.json", b"newer")
    with pytest.raises(objectstore.ConflictError):
        acme.adopt()
    assert shared.get("a.json")[0] == b"legacy"
    assert acme.get("a.json")[0] == b"newer"


def test_the_store_adopt_command_moves_and_resumes(shared, capsys):
    shared.put("bundles/v1.json", b"one")
    shared.put("runs/r1.json", b"two")
    # An interrupted earlier run copied one object but did not remove it.
    shared.put("tenants/acme/bundles/v1.json", b"one")
    root = str(shared.root)
    command = ["store", "adopt", "--root", root, "--tenant", "acme"]
    assert cli.main(command) == cli.EXIT_OK
    assert capsys.readouterr().out.split() == [
        "bundles/v1.json",
        "runs/r1.json",
    ]
    assert [m.key for m in shared.catalog()] == [
        "tenants/acme/bundles/v1.json",
        "tenants/acme/runs/r1.json",
    ]
    assert cli.main(command) == cli.EXIT_OK
    assert capsys.readouterr().out == ""
    invalid = ["store", "adopt", "--root", root, "--tenant", "../x"]
    assert cli.main(invalid) == cli.EXIT_FAILED


def test_confine_follows_the_environment(shared):
    assert tenancy.confine(shared, {}) is shared
    confined = tenancy.confine(shared, {"SCMLE_TENANT": "acme"})
    assert isinstance(confined, tenancy.Tenancy)
    assert confined.tenant == "acme"
    with pytest.raises(tenancy.TenantError):
        tenancy.confine(shared, {"SCMLE_TENANT": "../x"})


def trial():
    return experiments.Trial(
        identifier="t1",
        model="m",
        status=experiments.Termination.OK,
        metric="r2",
        score=0.5,
        deviation=0.1,
        fit=1.0,
        params={},
        folds=[0.5],
    )


def test_experiment_stores_of_two_tenants_never_share_a_run(tmp_path):
    path = tmp_path / "tracking.db"
    acme = stores.Sqlite(path, tenant="acme")
    globex = stores.Sqlite(path, tenant="globex")
    assert acme.path == tmp_path / "tenants" / "acme" / "tracking.db"
    acme.commence("r1", "h-acme", "dir", "2026-01-01T00:00:00Z")
    acme.transcribe("r1", trial())
    globex.commence("r1", "h-globex", "dir", "2026-01-02T00:00:00Z")
    globex.revise("r1", state="FAILED")
    assert [r["config_hash"] for r in acme.runs()] == ["h-acme"]
    assert acme.runs()[0]["state"] == "CREATED"
    assert globex.trials("r1") == []
    assert len(acme.trials("r1")) == 1
    legacy = stores.Sqlite(path)
    assert legacy.path == path and legacy.tenant is None
    assert legacy.runs() == []


def test_an_experiment_store_refuses_a_crafted_tenant(tmp_path):
    with pytest.raises(tenancy.TenantError):
        stores.Sqlite(tmp_path / "t.db", tenant="../acme")


@links
def test_an_experiment_store_refuses_a_symlinked_tenant(tmp_path):
    (tmp_path / "tenants" / "globex").mkdir(parents=True)
    (tmp_path / "tenants" / "acme").symlink_to(
        tmp_path / "tenants" / "globex", target_is_directory=True
    )
    with pytest.raises(objectstore.EscapeError):
        stores.Sqlite(tmp_path / "t.db", tenant="acme")


def bound(kid, tenant):
    return support_identity.hs_jwk(kid, tenant=tenant)


def verifier(clock, *entries):
    return tokens.Verifier(
        tokens.Keystore([tokens.Key.jwk(entry) for entry in entries]),
        support_identity.ISSUER,
        support_identity.AUDIENCE,
        clock=clock,
    )


def token(clock, kid="hs-1", **claims):
    return tokens.sign(
        support_identity.HS_SECRET,
        kid,
        support_identity.claims_at(clock.now, **claims),
    )


def test_a_tenant_bound_key_verifies_only_its_tenant():
    clock = support_identity.Clock()
    check = verifier(clock, bound("hs-1", "acme"))
    assert check.verify(token(clock)).tenant == "acme"
    for claims in ({"tenant": "globex"}, {"tenant": None}, {"tenant": ""}):
        with pytest.raises(tokens.TokenError) as raised:
            check.verify(token(clock, **claims))
        assert raised.value.reason == tokens.WRONG_TENANT


def test_a_shared_key_bound_twice_cannot_carry_a_token_across():
    # The same secret listed twice under two kids, each bound to a tenant:
    # a token naming one tenant under the other's kid is refused.
    clock = support_identity.Clock()
    check = verifier(
        clock, bound("acme-1", "acme"), bound("globex-1", "globex")
    )
    assert check.verify(token(clock, kid="acme-1")).tenant == "acme"
    with pytest.raises(tokens.TokenError) as raised:
        check.verify(token(clock, kid="globex-1"))
    assert raised.value.reason == tokens.WRONG_TENANT


@pytest.mark.parametrize("value", ["", 7, ["acme"]])
def test_a_malformed_key_tenant_is_a_config_error(value):
    with pytest.raises(errors.ConfigError):
        tokens.Key.jwk(bound("hs-1", value))


def gateway(clock, tenant="acme", *entries):
    keys = entries or (support_identity.hs_jwk("hs-1"),)
    return access.Gateway(verifier(clock, *keys), tenant=tenant)


def admitted(gate, clock, **claims):
    return gate.check(
        "GET", "/health", "Bearer " + token(clock, **claims), "127.0.0.1", "r"
    )


def test_confine_gives_the_caller_only_its_own_tenant(shared, pair):
    clock = support_identity.Clock()
    gate = gateway(clock)
    principal = admitted(gate, clock)
    store = gate.confine(shared, principal)
    assert store.tenant == "acme"
    with pytest.raises(objectstore.AbsentError):
        store.get("secret.json")
    with pytest.raises(objectstore.StoreError):
        store.get("../globex/secret.json")


def test_confine_refuses_a_foreign_or_missing_tenant(shared):
    clock = support_identity.Clock()
    open_gate = gateway(clock, None)
    foreign = admitted(open_gate, clock, tenant="globex")
    for gate, principal, reason in (
        (gateway(clock), foreign, "tenant_mismatch"),
        (open_gate, admitted(open_gate, clock, tenant=None), "no_tenant"),
        (
            open_gate,
            admitted(open_gate, clock, tenant="Acme"),
            "invalid_tenant",
        ),
    ):
        with pytest.raises(access.DeniedError) as raised:
            gate.confine(shared, principal)
        assert raised.value.status == 403
        assert raised.value.reason == reason
        assert raised.value.body == {
            "error": "forbidden",
            "category": "forbidden",
        }
    assert open_gate.confine(shared, foreign).tenant == "globex"


def test_a_token_of_one_tenant_is_refused_by_the_other_deployment():
    clock = support_identity.Clock()
    acme = gateway(clock, "acme", bound("hs-1", "acme"))
    globex = gateway(clock, "globex", bound("hs-1", "globex"))
    header = "Bearer " + token(clock)
    assert (
        acme.check("GET", "/health", header, "127.0.0.1", "r").tenant == "acme"
    )
    with pytest.raises(access.DeniedError) as raised:
        globex.check("GET", "/health", header, "127.0.0.1", "r")
    assert raised.value.status == 401
    assert raised.value.reason == tokens.WRONG_TENANT


def environment(tmp_path, tenant, *entries):
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps({"keys": list(entries)}))
    return {
        "SCMLE_AUTH_KEYS_FILE": str(keys),
        "SCMLE_AUTH_ISSUER": support_identity.ISSUER,
        "SCMLE_AUTH_AUDIENCE": support_identity.AUDIENCE,
        "SCMLE_TENANT": tenant,
    }


def test_the_environment_refuses_a_key_of_another_tenant(tmp_path):
    entries = (bound("hs-1", "acme"), bound("hs-2", "globex"))
    with pytest.raises(errors.ConfigError, match=r"\['hs-2'\]"):
        access.assemble(environment(tmp_path, "acme", *entries))
    gate = access.assemble(
        environment(tmp_path, "acme", entries[0], support_identity.hs_jwk("x"))
    )
    assert gate is not None and gate.tenant == "acme"


def test_the_environment_refuses_an_invalid_tenant_name(tmp_path):
    with pytest.raises(errors.ConfigError):
        access.assemble(
            environment(tmp_path, "Acme", support_identity.hs_jwk("hs-1"))
        )
