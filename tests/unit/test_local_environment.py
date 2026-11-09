"""The local environment is explicit files that agree with each other."""

import json
import pathlib
import re
import shutil
import subprocess
import urllib.error
import urllib.request

import pandas
import pytest
import yaml

from scmle.contracts import errors
from scmle.contracts import observability
from scmle.deploy import demo
from scmle.deploy import mocks
from scmle.deploy import provenance
from scmle.deploy import runtime
from scripts import make_demo_data

ROOT = pathlib.Path(__file__).resolve().parents[2]
PLACEHOLDER = "0" * 64


@pytest.fixture(scope="module")
def compose():
    return yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))


def test_the_compose_file_has_the_four_roles(compose):
    services = compose["services"]
    assert {"api", "object-store", "metrics-sink", "test-runner"} <= set(
        services
    )
    assert services["test-runner"]["profiles"] == ["test"]
    assert set(compose["volumes"]) == {"object-store", "metrics", "work"}


def test_every_service_uses_the_one_pinned_platform_image(compose):
    for name, service in compose["services"].items():
        assert service["image"] == "scmle-local:dev", name
        assert service["platform"] == "${SCMLE_PLATFORM:-linux/amd64}", name


def test_long_running_services_are_hardened_and_limited(compose):
    for name in ("api", "golden-api", "object-store", "metrics-sink"):
        service = compose["services"][name]
        assert service["read_only"] is True, name
        assert service["cap_drop"] == ["ALL"], name
        assert service["user"] == "10001:10001", name
        assert "no-new-privileges:true" in service["security_opt"], name
        assert service["mem_limit"] and service["cpus"], name
        assert service["tmpfs"], name
    api = compose["services"]["api"]
    assert api["environment"]["SCMLE_ENV"] == "production"
    assert "SCMLE_TRUSTED" not in json.dumps(compose)


def test_dependencies_name_real_services_and_conditions(compose):
    services = compose["services"]
    for name, service in services.items():
        for dependency, rule in service.get("depends_on", {}).items():
            assert dependency in services, (name, dependency)
            assert rule["condition"] in {
                "service_healthy",
                "service_started",
                "service_completed_successfully",
            }
            if rule["condition"] == "service_healthy":
                assert "healthcheck" in services[dependency], dependency


def test_the_compose_file_holds_no_secret(compose):
    text = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    assert "SIGNING_KEY" not in text
    env_file = (ROOT / "deploy" / "local" / "demo.env").read_text()
    assert "published fixture" in env_file.lower()
    api = compose["services"]["api"]
    assert api["env_file"] == "deploy/local/demo.env"


def test_compose_validates_with_the_docker_cli_when_present():
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("docker is not installed")
    result = subprocess.run(
        [docker, "compose", "-f", str(ROOT / "compose.yaml"), "config", "-q"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_the_local_dockerfile_refuses_the_placeholder_and_runs_non_root():
    text = (ROOT / "deploy" / "Dockerfile").read_text(encoding="utf-8")
    assert f"@sha256:{PLACEHOLDER}" in text
    assert "--require-hashes" in text and "--no-deps" in text
    assert "USER 10001:10001" in text
    assert "SCMLE_TRUSTED" not in text and "latest" not in text
    block = text[text.index("RUN case") : text.index("ENV PYTHON")]
    script = block.removeprefix("RUN ").replace("\\\n", " ")
    assert block.index(PLACEHOLDER) < block.index("*@sha256:*)")
    if shutil.which("sh") is None:
        return
    placeholder = subprocess.run(
        ["sh", "-c", script],
        env={"BASE_IMAGE": f"python:3.12-slim@sha256:{PLACEHOLDER}"},
        capture_output=True,
        check=False,
    )
    assert placeholder.returncode == 1


def test_the_locked_requirements_are_hashed_and_cover_the_core():
    text = (ROOT / "deploy" / "requirements.lock").read_text(encoding="utf-8")
    entries = re.split(r"(?m)^(?=[A-Za-z0-9])", text)
    entries = [entry for entry in entries if "==" in entry.split("\n")[0]]
    assert len(entries) >= 15
    for entry in entries:
        assert "--hash=sha256:" in entry, entry.split("\n")[0]
    names = {entry.split("==")[0] for entry in entries}
    for core in ("numpy", "pandas", "scikit-learn", "scipy", "pydantic"):
        assert core in names
    assert "pytest" not in names


def test_the_lock_matches_the_project_dependencies():
    import tomllib

    project = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["dependencies"]
    lock = (ROOT / "deploy" / "requirements.lock").read_text(encoding="utf-8")
    for requirement in project:
        name = re.split(r"[<>=;\s]", requirement)[0].lower().replace("_", "-")
        if name == "psutil":
            continue  # win32 only: the lock keeps its marker
        assert f"\n{name}==" in "\n" + lock, name


def test_the_demo_data_matches_its_generator():
    assert make_demo_data.main(["--check"]) == 0
    train = pandas.read_csv(ROOT / "deploy" / "local" / "data" / "train.csv")
    assert list(train.columns) == [
        "age",
        "balance",
        "tenure_months",
        "plan",
        "churned",
    ]
    assert set(train["churned"]) == {0, 1} and len(train) == 800


def test_the_make_targets_and_the_guide_agree():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "LOCAL.md").read_text(encoding="utf-8")
    for target in ("demo", "demo-compose", "image", "lock", "lock-check"):
        assert re.search(rf"(?m)^{target}:", makefile), target
        assert f"make {target}" in guide, target
    assert "docker was not run" in guide.lower() or "not run" in guide.lower()


def test_the_object_store_round_trips_and_checks_digests(tmp_path):
    store = mocks.ObjectStore(tmp_path / "store")
    store.start()
    try:
        data = b"model bytes"
        etag = mocks.put_object(store.address, "models/a/b.bin", data)
        assert etag == provenance.sha256_digest(data)
        assert mocks.get_object(store.address, "models/a/b.bin", etag) == data
        with pytest.raises(errors.ArtifactError, match="does not match"):
            mocks.get_object(
                store.address, "models/a/b.bin", "sha256:" + "1" * 64
            )
        request = urllib.request.Request(
            f"{store.address}/models/a/c.bin",
            b"x",
            {mocks.DIGEST_HEADER: "sha256:" + "2" * 64},
            method="PUT",
        )
        with pytest.raises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request)
        assert raised.value.code == 400
        raised.value.close()
        assert not (tmp_path / "store" / "models" / "a" / "c.bin").exists()
        listing = json.loads(
            urllib.request.urlopen(f"{store.address}/models?list").read()
        )
        assert [item["key"] for item in listing["objects"]] == [
            "models/a/b.bin"
        ]
        for path in ("/models/../../etc/passwd", "/onlybucket"):
            with pytest.raises(urllib.error.HTTPError) as bad:
                urllib.request.urlopen(f"{store.address}{path}")
            assert bad.value.code in {400, 404}
            bad.value.close()
        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(f"{store.address}/models/none")
        assert missing.value.code == 404
        missing.value.close()
    finally:
        store.stop()


def test_safe_keys_refuse_traversal():
    assert str(mocks.safe_key("/bucket/a/b")) == "bucket/a/b"
    for bad in ("/bucket", "/bucket/../x", "/"):
        with pytest.raises(ValueError):
            mocks.safe_key(bad)


def test_the_metrics_sink_stores_valid_samples_and_refuses_bad_ones(tmp_path):
    sink = mocks.MetricsSink(tmp_path / "m.jsonl")
    sink.start()
    try:
        sample = observability.MetricSample(
            "latency_p95",
            observability.SignalKind.LATENCY,
            12.5,
            "ms",
            "2026-11-05T00:00:00+00:00",
            {"model": "churn"},
        )
        assert mocks.post_metrics(sink.address, [sample]) == 1
        stored = json.loads(
            urllib.request.urlopen(f"{sink.address}/metrics").read()
        )
        assert stored["count"] == 1
        assert stored["samples"][0]["value"] == 12.5
        request = urllib.request.Request(
            f"{sink.address}/metrics",
            json.dumps({"name": "x"}).encode(),
            {"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request)
        assert raised.value.code == 400
        raised.value.close()
        assert len(sink.samples()) == 1
    finally:
        sink.stop()


def test_runtime_settings_read_the_environment_and_reject_garbage():
    defaults = runtime.settings({})
    assert defaults["port"] == 8000 and defaults["max_concurrent"] == 8
    assert defaults["on_null"] == "impute"
    custom = runtime.settings(
        {"PORT": "9001", "SCMLE_MAX_CONCURRENT": "2", "SCMLE_ON_NULL": "reject"}
    )
    assert custom["port"] == 9001 and custom["max_concurrent"] == 2
    assert custom["on_null"] == "reject"
    with pytest.raises(errors.ConfigError, match="PORT"):
        runtime.settings({"PORT": "eighty"})


def test_the_container_healthcheck_follows_readiness(tmp_path):
    from scmle.artifacts import bundle
    from scmle.inference import parity
    from scmle.inference import serving

    golden = parity.load_golden(ROOT / "deploy" / "local" / "golden")
    folder = parity.reference_bundle(golden.spec).save(tmp_path / "bundle")
    model = bundle.ModelBundle.load(folder, trusted=True)
    adapter = serving.HttpServing(model, port=0)
    adapter.start()
    try:
        port = str(adapter.httpd.server_address[1])
        assert runtime.healthcheck({"PORT": port}) == 0
    finally:
        adapter.stop()
    assert runtime.healthcheck({"PORT": port}) == 1


def test_the_transcript_passes_only_when_every_step_does(capsys):
    transcript = demo.Transcript()
    assert not transcript.passed
    transcript.record("one", 0.0, ok=True, detail=1)
    assert transcript.passed
    transcript.record("two", 0.0, ok=False)
    assert not transcript.passed
    assert transcript.to_dict()["steps"][1]["ok"] is False
    assert "[FAIL] two" in capsys.readouterr().out


def test_a_folder_survives_the_object_store_round_trip(tmp_path):
    source = tmp_path / "source"
    (source / "model").mkdir(parents=True)
    (source / "model" / "a.bin").write_bytes(b"abc")
    (source / "serve.py").write_text("print()")
    store = mocks.ObjectStore(tmp_path / "store")
    store.start()
    try:
        etags = demo.upload_folder(store.address, source, "svc/v1")
        target = tmp_path / "target"
        demo.download_folder(store.address, "svc/v1", etags, target)
        assert (target / "model" / "a.bin").read_bytes() == b"abc"
        etags["serve.py"] = "sha256:" + "3" * 64
        with pytest.raises(errors.ArtifactError):
            demo.download_folder(store.address, "svc/v1", etags, target)
    finally:
        store.stop()
