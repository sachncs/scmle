"""The scmle image: pinned base, hashed wheel, and the registry protocol."""

import hashlib
import http.server
import json
import pathlib
import re
import shutil
import subprocess
import threading
import types

import pytest

from scmle.contracts import errors
from scmle.deploy import base_image
from scmle.deploy import packaging

ROOT = pathlib.Path(__file__).resolve().parents[2]
PLACEHOLDER = "sha256:" + "0" * 64
INDEX = "application/vnd.oci.image.index.v1+json"
DIGEST = "sha256:" + "b" * 64


@pytest.fixture(scope="module")
def dockerfile():
    return (ROOT / "Dockerfile").read_text(encoding="utf-8")


def blocks(text):
    """The shell scripts of the Dockerfile's `RUN case` steps."""
    found = re.findall(r"RUN (case .*?esac)", text, flags=re.DOTALL)
    return [block.replace("\\\n", " ") for block in found]


def sh(script, **env):
    return subprocess.run(
        ["sh", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_dockerfile_defaults_to_the_recorded_pin(dockerfile):
    (default,) = re.findall(r"(?m)^ARG BASE_IMAGE=(\S+)$", dockerfile)
    assert (
        default == "python:3.12-slim@" + base_image.PINNED["python:3.12-slim"]
    )
    assert base_image.PINNED_AT == "2026-11-19"
    for image, digest in base_image.PINNED.items():
        assert re.fullmatch(r"python:3\.1[1-4]-slim", image)
        assert (
            re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
            and digest != PLACEHOLDER
        )


def test_the_dockerfile_installs_hashed_and_runs_as_non_root(dockerfile):
    assert "--require-hashes --no-deps" in dockerfile
    assert (
        "-r /tmp/install/requirements.lock -r /tmp/install/wheel.lock"
        in dockerfile
    )
    assert "COPY --from=wheels ${WHEEL} /tmp/install/" in dockerfile
    assert "pip check" in dockerfile
    assert "USER 10001:10001" in dockerfile
    assert "latest" not in dockerfile and "SCMLE_TRUSTED" not in dockerfile


def test_the_dockerfile_refuses_an_unpinned_base_and_an_unhashed_wheel(
    dockerfile,
):
    base, wheel = blocks(dockerfile)
    assert base.index(PLACEHOLDER[7:]) < base.index("*@sha256:*)")
    if shutil.which("sh") is None:
        return
    assert (
        sh(base, BASE_IMAGE="python:3.12-slim@" + PLACEHOLDER).returncode == 1
    )
    assert sh(base, BASE_IMAGE="python:3.12-slim").returncode == 1
    assert sh(base, BASE_IMAGE="python:3.12-slim@" + DIGEST).returncode == 0
    wheel = wheel.split(" && ")[0]
    good = {"WHEEL": "scmle-1.0-py3-none-any.whl", "WHEEL_SHA256": "c" * 64}
    assert sh(wheel, **good).returncode == 0
    assert sh(wheel, **{**good, "WHEEL_SHA256": ""}).returncode == 1
    assert (
        sh(wheel, **{**good, "WHEEL": "other-1.0-py3-none-any.whl"}).returncode
        == 1
    )


def test_the_wheel_is_named_with_its_hash(tmp_path):
    wheel = tmp_path / "scmle-1.2.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel bytes")
    assert packaging.arguments(wheel) == {
        "WHEEL": "scmle-1.2.0-py3-none-any.whl",
        "WHEEL_SHA256": hashlib.sha256(b"wheel bytes").hexdigest(),
    }
    with pytest.raises(errors.ArtifactError, match="not an scmle wheel"):
        packaging.arguments(tmp_path / "other-1.0-py3-none-any.whl")
    with pytest.raises(errors.ArtifactError, match="cannot read"):
        packaging.arguments(tmp_path / "scmle-9.0-py3-none-any.whl")


def test_the_build_command_passes_the_wheel_context_and_its_hash(tmp_path):
    wheel = tmp_path / "scmle-1.2.0-py3-none-any.whl"
    wheel.write_bytes(b"w")
    sha = hashlib.sha256(b"w").hexdigest()
    assert packaging.recipe(wheel, "scmle:1.2.0", context=ROOT) == [
        "docker",
        "build",
        "--build-context",
        f"wheels={tmp_path.resolve()}",
        "--build-arg",
        "WHEEL=scmle-1.2.0-py3-none-any.whl",
        "--build-arg",
        f"WHEEL_SHA256={sha}",
        "--tag",
        "scmle:1.2.0",
        str(ROOT),
    ]
    pinned = packaging.recipe(
        wheel, "t", base="python:3.13-slim", platform="linux/arm64"
    )
    assert pinned[8:12] == [
        "--build-arg",
        "BASE_IMAGE=python@" + base_image.PINNED["python:3.13-slim"],
        "--platform",
        "linux/arm64",
    ]
    with pytest.raises(errors.ConfigError, match="no recorded pin"):
        packaging.recipe(wheel, "t", base="python:3.9-slim")
    assert packaging.main(["image", "--wheel", "nope.whl", "--tag", "t"]) == 1


def test_a_generated_service_dockerfile_defaults_to_the_pin(tmp_path):
    info = types.SimpleNamespace(model_name="m", model_version="1")
    packaging.furnish(tmp_path, info, "python:3.12-slim")
    text = (tmp_path / "Dockerfile").read_text(encoding="utf-8")
    pin = base_image.PINNED["python:3.12-slim"]
    assert f"ARG BASE_IMAGE=python:3.12-slim@{pin}\n" in text
    packaging.furnish(tmp_path, info, "python:3.9-slim")
    text = (tmp_path / "Dockerfile").read_text(encoding="utf-8")
    assert f"ARG BASE_IMAGE=python:3.9-slim@{PLACEHOLDER}\n" in text


def test_the_pinned_command_prints_the_build_argument(capsys):
    assert base_image.main(["pinned", "--image", "python:3.12-slim"]) == 0
    pin = base_image.PINNED["python:3.12-slim"]
    assert capsys.readouterr().out == f"python@{pin}\n"
    assert base_image.main(["pinned", "--image", "python:3.9-slim"]) == 1
    assert "is not pinned" in capsys.readouterr().err


class Registry(http.server.BaseHTTPRequestHandler):
    """A registry speaking distribution-spec v2 with bearer tokens."""

    realm = ""

    def log_message(self, *arguments):
        pass

    def do_GET(self):
        if not self.path.startswith("/token?"):
            self.send_error(404)
            return
        query = self.path.split("?", 1)[1]
        body = json.dumps({"token": "t0k" if "service=local" in query else ""})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())

    def do_HEAD(self):
        manifests = {
            "/v2/library/multi/manifests/1": INDEX,
            "/v2/library/single/manifests/1": (
                "application/vnd.oci.image.manifest.v1+json"
            ),
        }
        if self.headers.get("Authorization") != "Bearer t0k":
            self.send_response(401)
            self.send_header(
                "WWW-Authenticate",
                f'Bearer realm="{self.realm}",service="local",'
                'scope="repository:library/multi:pull"',
            )
            self.end_headers()
            return
        if self.path not in manifests:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", manifests[self.path])
        self.send_header("Docker-Content-Digest", DIGEST)
        self.end_headers()


@pytest.fixture
def registry():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Registry)
    host = f"127.0.0.1:{server.server_address[1]}"
    Registry.realm = f"http://{host}/token"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield host
    server.shutdown()
    server.server_close()


def test_resolving_follows_the_bearer_challenge_of_a_real_registry(registry):
    assert base_image.origin(registry) == f"http://{registry}"
    assert base_image.origin("ghcr.io") == "https://ghcr.io"
    assert base_image.query(f"{registry}/library/multi:1", timeout=5) == DIGEST
    record = base_image.resolve(f"{registry}/library/multi:1")
    assert record["digest"] == DIGEST
    with pytest.raises(errors.ArtifactError, match="not a multi-architecture"):
        base_image.query(f"{registry}/library/single:1", timeout=5)
    with pytest.raises(errors.ArtifactError, match="answered 404"):
        base_image.query(f"{registry}/library/gone:1", timeout=5)


def test_a_challenge_must_be_bearer_over_http_or_https():
    base = "https://registry.example"
    with pytest.raises(errors.ArtifactError, match="unsupported registry"):
        base_image.authorize('Basic realm="https://x"', base, 1)
    with pytest.raises(errors.ArtifactError, match="unsupported registry"):
        base_image.authorize('Bearer realm="file:///etc/passwd"', base, 1)
    with pytest.raises(errors.ArtifactError, match="unsupported registry"):
        base_image.authorize('Bearer realm="http://evil/token"', base, 1)
