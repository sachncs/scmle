"""Keys and revocations fetched over the network, with a local server."""

import contextlib
import http.server
import json
import threading

import pytest

from scmle.contracts import errors
from scmle.security import remote
from scmle.security import tokens
from tests import support_identity


class Documents:
    """What the stand-in identity provider serves, and how often."""

    def __init__(self):
        self.routes = {}
        self.hits = {}
        self.broken = False


@contextlib.contextmanager
def serve(documents):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            documents.hits[self.path] = documents.hits.get(self.path, 0) + 1
            body = documents.routes.get(self.path)
            if documents.broken or body is None:
                self.send_response(503 if documents.broken else 404)
                self.end_headers()
                return
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, message_format, *args):
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def provider(base, documents, keys=("k1",)):
    documents.routes["/.well-known/openid-configuration"] = {
        "issuer": base,
        "jwks_uri": f"{base}/jwks.json",
    }
    documents.routes["/jwks.json"] = {
        "keys": [support_identity.rsa_jwk(name, f"rs-{name}") for name in keys]
    }


def test_discovery_reads_the_jwks_uri_of_the_named_issuer():
    documents = Documents()
    with serve(documents) as base:
        provider(base, documents)
        found = remote.discover(base, allow_http=True)
        assert found["jwks_uri"] == f"{base}/jwks.json"


def test_discovery_refuses_a_document_for_another_issuer():
    documents = Documents()
    with serve(documents) as base:
        provider(base, documents)
        documents.routes["/.well-known/openid-configuration"]["issuer"] = (
            "https://evil.test"
        )
        with pytest.raises(errors.ConfigError):
            remote.discover(base, allow_http=True)


def test_plain_http_and_other_schemes_are_refused_by_default():
    for url in ("http://127.0.0.1:1/x", "file:///etc/passwd", "ftp://x/y"):
        with pytest.raises(errors.ConfigError):
            remote.fetch_json(url)


def test_a_fetch_failure_or_a_non_json_body_is_a_config_error():
    documents = Documents()
    with serve(documents) as base:
        with pytest.raises(errors.ConfigError):
            remote.fetch_json(f"{base}/missing", allow_http=True)
        documents.routes["/text"] = "not an object"
        assert remote.fetch_json(f"{base}/text", allow_http=True) == (
            "not an object"
        )
        documents.broken = True
        with pytest.raises(errors.ConfigError):
            remote.fetch_json(f"{base}/jwks.json", allow_http=True)


def test_a_verifier_accepts_a_token_signed_by_a_fetched_key():
    clock = support_identity.Clock()
    documents = Documents()
    with serve(documents) as base:
        provider(base, documents)
        keys = remote.KeySet(f"{base}/jwks.json", clock=clock, allow_http=True)
        check = tokens.Verifier(
            keys,
            support_identity.ISSUER,
            support_identity.AUDIENCE,
            clock=clock,
        )
        token = support_identity.rs256_token(
            "k1", "rs-k1", support_identity.claims_at(clock.now)
        )
        assert check.verify(token).key_id == "rs-k1"


def test_the_key_set_is_cached_until_the_ttl_and_then_refreshed():
    clock = support_identity.Clock()
    documents = Documents()
    with serve(documents) as base:
        provider(base, documents)
        keys = remote.KeySet(
            f"{base}/jwks.json", ttl=60, clock=clock, allow_http=True
        )
        keys.get("rs-k1")
        keys.get("rs-k1")
        assert documents.hits["/jwks.json"] == 1
        clock.now += 61
        keys.get("rs-k1")
        assert documents.hits["/jwks.json"] == 2


def test_rotation_is_picked_up_for_an_unknown_kid_but_not_every_call():
    clock = support_identity.Clock()
    documents = Documents()
    with serve(documents) as base:
        provider(base, documents, keys=("k1",))
        keys = remote.KeySet(
            f"{base}/jwks.json",
            ttl=3600,
            min_refresh=10,
            clock=clock,
            allow_http=True,
        )
        provider(base, documents, keys=("k1", "k2"))
        with pytest.raises(tokens.TokenError):
            keys.get("rs-k2")  # inside min_refresh: no new fetch
        assert documents.hits["/jwks.json"] == 1
        clock.now += 11
        assert keys.get("rs-k2").kid == "rs-k2"
        assert documents.hits["/jwks.json"] == 2


def test_a_key_set_that_cannot_be_fetched_fails_closed_after_max_stale():
    clock = support_identity.Clock()
    documents = Documents()
    with serve(documents) as base:
        provider(base, documents)
        keys = remote.KeySet(
            f"{base}/jwks.json",
            ttl=60,
            max_stale=300,
            clock=clock,
            allow_http=True,
        )
        documents.broken = True
        clock.now += 100  # past the ttl, inside max_stale: the cache serves
        assert keys.get("rs-k1").kid == "rs-k1"
        clock.now += 400  # past max_stale: nothing is trusted
        with pytest.raises(tokens.TokenError) as raised:
            keys.get("rs-k1")
        assert raised.value.reason == remote.KEYS_UNAVAILABLE


def test_the_first_fetch_must_succeed():
    documents = Documents()
    with serve(documents) as base, pytest.raises(errors.ConfigError):
        remote.KeySet(f"{base}/jwks.json", allow_http=True)


def revoke(documents, **fields):
    documents.routes["/revoked.json"] = {
        "tokens": [],
        "subjects": [],
        "key_ids": [],
        **fields,
    }


def test_two_replicas_see_a_revocation_within_the_ttl():
    clock = support_identity.Clock()
    documents = Documents()
    with serve(documents) as base:
        provider(base, documents)
        revoke(documents)
        replicas = []
        for ignored in range(2):
            keys = remote.KeySet(
                f"{base}/jwks.json", clock=clock, allow_http=True
            )
            lists = remote.Revocations(
                f"{base}/revoked.json", ttl=30, clock=clock, allow_http=True
            )
            replicas.append(
                tokens.Verifier(
                    keys,
                    support_identity.ISSUER,
                    support_identity.AUDIENCE,
                    revocations=lists,
                    clock=clock,
                )
            )
        token = support_identity.rs256_token(
            "k1", "rs-k1", support_identity.claims_at(clock.now)
        )
        assert all(r.verify(token).subject == "alice" for r in replicas)
        revoke(documents, subjects=["alice"])
        assert all(r.verify(token) for r in replicas)  # cached, not yet seen
        clock.now += 31
        for replica in replicas:
            with pytest.raises(tokens.TokenError) as raised:
                replica.verify(token)
            assert raised.value.reason == tokens.REVOKED


def test_a_revocation_source_that_stays_down_stops_accepting_tokens():
    clock = support_identity.Clock()
    documents = Documents()
    with serve(documents) as base:
        provider(base, documents)
        revoke(documents)
        keys = remote.KeySet(f"{base}/jwks.json", clock=clock, allow_http=True)
        lists = remote.Revocations(
            f"{base}/revoked.json",
            ttl=30,
            max_stale=120,
            clock=clock,
            allow_http=True,
        )
        check = tokens.Verifier(
            keys,
            support_identity.ISSUER,
            support_identity.AUDIENCE,
            revocations=lists,
            clock=clock,
        )
        first = support_identity.rs256_token(
            "k1", "rs-k1", support_identity.claims_at(clock.now)
        )
        assert check.verify(first).subject == "alice"  # the list is fetched
        documents.broken = True
        clock.now += 60  # stale but inside max_stale: still served
        token = support_identity.rs256_token(
            "k1", "rs-k1", support_identity.claims_at(clock.now)
        )
        assert check.verify(token).subject == "alice"
        clock.now += 200
        fresh = support_identity.rs256_token(
            "k1", "rs-k1", support_identity.claims_at(clock.now)
        )
        with pytest.raises(tokens.TokenError) as raised:
            check.verify(fresh)
        assert raised.value.reason == tokens.REVOCATION_UNAVAILABLE


def test_a_malformed_revocation_document_is_refused():
    clock = support_identity.Clock()
    documents = Documents()
    with serve(documents) as base:
        documents.routes["/revoked.json"] = ["not", "an", "object"]
        with pytest.raises(errors.ConfigError):
            remote.Revocations(
                f"{base}/revoked.json", clock=clock, allow_http=True
            )()
