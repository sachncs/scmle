"""The generated service is hardened: pinned, locked, verified, secret-free."""

import datetime
import json
import pathlib
import shutil
import subprocess
import sys

import joblib
import pytest

import scmle
from scmle.artifacts import bundle as bundle_module
from scmle.contracts import errors
from scmle.deploy import attest
from scmle.deploy import base_image
from scmle.deploy import context
from scmle.deploy import lockfile
from scmle.deploy import provenance
from scmle.deploy import verify
from scmle.deploy import vuln_gate
from scmle.governance import sbom
from scmle.inference import parity

ROOT = pathlib.Path(__file__).resolve().parents[2]
KEY = "hardening-test-signing-key-8f3a"
PLACEHOLDER = "sha256:" + "0" * 64


@pytest.fixture(scope="module")
def reference():
    golden = parity.load_golden(ROOT / "deploy" / "local" / "golden")
    return parity.reference_bundle(golden.spec)


@pytest.fixture(scope="module")
def signed_service(reference, tmp_path_factory):
    folder = tmp_path_factory.mktemp("hardening") / "service"
    reference.export_service(folder, signing_key=KEY)
    return folder


def copy_of(service, tmp_path):
    target = tmp_path / "copy"
    shutil.copytree(service, target)
    return target


def production(**extra):
    return {"SCMLE_ENV": "production", "SCMLE_SIGNING_KEY": KEY, **extra}


def test_the_folder_has_every_hardening_file(signed_service):
    names = {path.name for path in signed_service.iterdir()}
    assert {
        "Dockerfile",
        ".dockerignore",
        "build.sh",
        "requirements.txt",
        "requirements.lock",
        "environment.lock",
        "sbom.cdx.json",
        "provenance.json",
        "vuln-policy.json",
        "serve.py",
        "README.md",
        "model",
    } <= names


def test_the_dockerfile_pins_locks_and_drops_privileges(signed_service):
    text = (signed_service / "Dockerfile").read_text(encoding="utf-8")
    assert f"@{PLACEHOLDER}" in text and "-slim@sha256:" in text
    assert "pip install --no-cache-dir --no-deps -r requirements.lock" in text
    assert "pip check" in text
    assert "useradd --system --uid 10001" in text
    assert "USER 10001:10001" in text
    assert "SCMLE_ENV=production" in text
    assert "SCMLE_TRUSTED" not in text
    assert "HEALTHCHECK" in text and "healthcheck" in text
    assert "latest" not in text


def test_no_generated_text_documents_the_trusted_escape_hatch(signed_service):
    for name in ("README.md", "Dockerfile", "build.sh", "serve.py"):
        assert "SCMLE_TRUSTED" not in (signed_service / name).read_text()


def test_the_placeholder_digest_stops_the_build(signed_service):
    text = (signed_service / "Dockerfile").read_text(encoding="utf-8")
    block = text[text.index("RUN case") : text.index("ENV PYTHON")]
    script = block.removeprefix("RUN ").replace("\\\n", " ")

    def run(image):
        return subprocess.run(
            ["sh", "-c", script],
            env={"BASE_IMAGE": image},
            capture_output=True,
            text=True,
            check=False,
        )

    assert block.index(PLACEHOLDER) < block.index("*@sha256:*)")
    assert "exit 1" in block
    if shutil.which("sh") is None:
        return
    stopped = run(f"python:3.12-slim@{PLACEHOLDER}")
    assert stopped.returncode == 1 and "placeholder" in stopped.stderr
    assert run("python:3.12-slim").returncode == 1
    assert run("python:3.12-slim@sha256:" + "a" * 64).returncode == 0


def test_the_lock_is_the_runtime_closure_not_the_dev_environment(
    signed_service,
):
    lines = (signed_service / "requirements.lock").read_text().splitlines()
    pins = {
        line.split("==")[0]: line.split("==")[1]
        for line in lines
        if "==" in line
    }
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "scmle"):
        assert name in pins
    for name in ("pytest", "ruff", "mypy", "bandit"):
        assert name not in pins
    assert all("==" in line or line.startswith("#") for line in lines if line)
    environment = (signed_service / "environment.lock").read_text()
    assert len(environment.splitlines()) > len(lines)


def test_the_lock_follows_requirements_through_fake_metadata():
    graph = {
        "app": ["lib>=1", "extra-only; extra == 'x'"],
        "lib": ["leaf", "py-only; python_version < '2'"],
        "leaf": [],
    }
    versions = {"app": "1", "lib": "2", "leaf": "3"}
    closure = lockfile.runtime_closure(
        ["app"],
        requires=lambda name: graph[name],
        installed=lambda name: versions[name],
    )
    assert closure == {"app": "1", "leaf": "3", "lib": "2"}
    unresolved = []
    closure = lockfile.runtime_closure(
        ["app"],
        versions={"lib": "9"},
        requires=lambda name: graph[name],
        installed=lambda name: versions[name],
        unresolved=unresolved,
    )
    assert closure["lib"] == "9"


def test_a_missing_distribution_is_an_error():
    def absent(name):
        raise lockfile.metadata.PackageNotFoundError(name)

    with pytest.raises(errors.ArtifactError, match="neither installed"):
        lockfile.runtime_closure(["ghost"], installed=absent)


def test_the_sbom_lists_libraries_the_model_and_the_base_image(signed_service):
    document = json.loads((signed_service / "sbom.cdx.json").read_text())
    kinds = {item["type"] for item in document["components"]}
    assert {"library", "machine-learning-model", "container"} <= kinds
    model = next(
        item
        for item in document["components"]
        if item["type"] == "machine-learning-model"
    )
    digest = provenance.bundle_digest(signed_service / "model")
    assert model["hashes"][0]["content"] == digest.removeprefix("sha256:")
    names = {item["name"] for item in document["components"]}
    assert "numpy" in names and "pytest" not in names
    assert document["bomFormat"] == "CycloneDX"


def test_the_service_sbom_is_deterministic_for_one_timestamp():
    pins = {"numpy": "2.0.0", "pandas": "3.0.0"}
    arguments = {
        "model_name": "m",
        "model_version": "v1",
        "bundle_digest": "sha256:" + "a" * 64,
        "base_image": "python:3.12-slim",
        "software_version": "1",
        "observed_at": "2026-01-01T00:00:00+00:00",
    }
    assert sbom.build_service(pins, **arguments) == sbom.build_service(
        pins, **arguments
    )
    with pytest.raises(ValueError, match="sha256"):
        sbom.build_service(pins, **{**arguments, "bundle_digest": "abc"})
    assert sbom.parse_pins("# c\nnumpy==2.0.0\nbad line\nPandas==3 # x\n") == {
        "numpy": "2.0.0",
        "pandas": "3",
    }


def test_source_date_epoch_fixes_the_sbom_time(monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "86400")
    assert sbom.source_timestamp().startswith("1970-01-02T00:00:00")
    monkeypatch.delenv("SOURCE_DATE_EPOCH")
    assert datetime.datetime.fromisoformat(sbom.source_timestamp())


def test_the_provenance_verifies_and_names_the_bundle(signed_service):
    document = attest.read_document(signed_service / "provenance.json")
    body, verified = attest.verify_document(document, KEY)
    assert verified is True
    digest = provenance.bundle_digest(signed_service / "model")
    assert body["subject"][0]["digest"]["sha256"] == digest.removeprefix(
        "sha256:"
    )
    names = {item["name"] for item in body["predicate"]["materials"]}
    assert {"Dockerfile", "requirements.lock", "sbom.cdx.json"} <= names
    assert attest.material_problems(signed_service, body) == []
    with pytest.raises(errors.ArtifactError, match="signature is invalid"):
        attest.verify_document(document, "another-key")
    forged = json.loads(json.dumps(document))
    forged["statement"]["predicate"]["builder"] = "someone else"
    with pytest.raises(errors.ArtifactError, match="signature is invalid"):
        attest.verify_document(forged, KEY)
    assert (
        attest.verify_document({**document, "signature": None}, KEY)[1] is False
    )


def test_production_verification_accepts_a_signed_service(signed_service):
    verified = verify.verify_service(signed_service, production())
    assert verified.bundle_signed and verified.provenance_verified
    assert verified.production and not verified.trusted
    assert verified.bundle_digest == provenance.bundle_digest(
        signed_service / "model"
    )


def test_trusted_is_refused_in_production_everywhere(
    signed_service, reference, tmp_path, monkeypatch
):
    with pytest.raises(
        errors.ArtifactError, match="SCMLE_TRUSTED=1 is refused"
    ):
        verify.verify_service(signed_service, production(SCMLE_TRUSTED="1"))
    folder = reference.save(tmp_path / "unsigned-bundle")
    monkeypatch.setenv("SCMLE_ENV", "production")
    with pytest.raises(
        errors.ArtifactError, match="trusted loading is refused"
    ):
        bundle_module.ModelBundle.load(folder, trusted=True)
    monkeypatch.setenv("SCMLE_ENV", "staging")
    assert bundle_module.ModelBundle.load(folder, trusted=True)


def test_verification_happens_before_anything_is_deserialized(
    signed_service, tmp_path, monkeypatch
):
    def forbidden(*arguments, **keywords):
        raise AssertionError("deserialization before verification")

    monkeypatch.setattr(joblib, "load", forbidden)
    monkeypatch.setattr(bundle_module.joblib, "load", forbidden)
    corrupted = copy_of(signed_service, tmp_path)
    with (corrupted / "model" / "pipeline.joblib").open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(errors.ArtifactError, match="checksum mismatch"):
        verify.verify_service(corrupted, production())
    assert verify.verify_service(signed_service, production())


def test_an_unsigned_bundle_never_verifies_in_production(
    signed_service, tmp_path
):
    unsigned = copy_of(signed_service, tmp_path)
    (unsigned / "model" / "signature.json").unlink()
    with pytest.raises(errors.ArtifactError, match="unauthenticated"):
        verify.verify_service(unsigned, production())
    with pytest.raises(errors.ArtifactError, match="unauthenticated"):
        verify.verify_service(unsigned, {"SCMLE_ENV": "staging"})
    accepted = verify.verify_service(
        unsigned, {"SCMLE_ENV": "staging", "SCMLE_TRUSTED": "1"}
    )
    assert accepted.trusted and not accepted.bundle_signed


def test_a_wrong_key_is_refused(signed_service):
    with pytest.raises(errors.ArtifactError, match="signature is invalid"):
        verify.verify_service(
            signed_service,
            {"SCMLE_ENV": "production", "SCMLE_SIGNING_KEY": "wrong"},
        )


def test_production_needs_verifiable_provenance(signed_service, tmp_path):
    missing = copy_of(signed_service, tmp_path)
    (missing / "provenance.json").unlink()
    with pytest.raises(errors.ArtifactError, match="provenance"):
        verify.verify_service(missing, production())
    assert verify.verify_service(
        missing, {"SCMLE_ENV": "staging", "SCMLE_SIGNING_KEY": KEY}
    )


def test_changed_build_materials_are_refused(signed_service, tmp_path):
    changed = copy_of(signed_service, tmp_path)
    with (changed / "requirements.lock").open("a") as handle:
        handle.write("evil-package==1.0\n")
    with pytest.raises(errors.ArtifactError, match="materials changed"):
        verify.verify_service(changed, production())


def test_provenance_for_another_bundle_is_refused(signed_service, tmp_path):
    swapped = copy_of(signed_service, tmp_path)
    attest.write_provenance(
        swapped, "sha256:" + "9" * 64, "someone", "python:3.12-slim", KEY
    )
    with pytest.raises(errors.ArtifactError, match="different bundle digest"):
        verify.verify_service(swapped, production())


def test_an_image_attestation_binds_image_to_provenance(
    signed_service, tmp_path
):
    folder = copy_of(signed_service, tmp_path)
    image = "sha256:" + "d" * 64
    lock = {"image": "python:3.12-slim", "digest": "sha256:" + "a" * 64}
    attest.write_image_attestation(folder, image, lock, KEY)
    body = attest.verify_image(folder, image, KEY)
    assert body["predicate"]["base_image"] == lock
    with pytest.raises(errors.ArtifactError, match="different image"):
        attest.verify_image(folder, "sha256:" + "e" * 64, KEY)
    with pytest.raises(errors.ArtifactError, match="signature is invalid"):
        attest.verify_image(folder, image, "other-key")
    attest.write_image_attestation(folder, image, lock, None)
    with pytest.raises(errors.ArtifactError, match="not verified"):
        attest.verify_image(folder, image, KEY)
    attest.write_image_attestation(folder, image, lock, KEY)
    (folder / "provenance.json").write_text("{}")
    with pytest.raises(errors.ArtifactError, match="provenance file differs"):
        attest.verify_image(folder, image, KEY)


def test_the_build_context_holds_no_secret(signed_service):
    assert context.scan(signed_service, [KEY]) == []
    everything = b"".join(
        path.read_bytes()
        for path in signed_service.rglob("*")
        if path.is_file()
    )
    assert KEY.encode() not in everything


def test_the_context_scan_finds_planted_secrets(signed_service, tmp_path):
    folder = copy_of(signed_service, tmp_path)
    (folder / ".env").write_text("TOKEN=abc")
    (folder / "id_rsa").write_text("x")
    assert ".env" not in {p.name for p in context.included_files(folder)}
    assert context.scan(folder, [KEY]) == []
    (folder / "notes.txt").write_text(f"the key is {KEY}")
    (folder / "deploy.pem.txt").write_text(
        "-----BEGIN RSA PRIVATE KEY-----\nabc"
    )
    findings = context.scan(folder, [KEY])
    assert any(
        "notes.txt" in item and "secret value" in item for item in findings
    )
    assert any("private key" in item for item in findings)
    with pytest.raises(errors.ArtifactError, match="may hold secrets"):
        context.require_clean(folder, [KEY])
    (folder / ".dockerignore").unlink()
    assert any(
        ".dockerignore is missing" in item for item in context.scan(folder)
    )
    assert any("credential file" in item for item in context.scan(folder))


def test_a_dockerfile_cannot_bake_in_a_secret(signed_service, tmp_path):
    folder = copy_of(signed_service, tmp_path)
    with (folder / "Dockerfile").open("a") as handle:
        handle.write("ENV API_TOKEN=hunter2\n")
    assert any("API_TOKEN" in item for item in context.scan(folder))


def test_the_context_check_command(signed_service, tmp_path):
    assert context.main(["check", str(signed_service)]) == 0
    folder = copy_of(signed_service, tmp_path)
    (folder / "leak.txt").write_text("-----BEGIN PRIVATE KEY-----")
    assert context.main(["check", str(folder)]) == 1


def test_base_image_references_are_split_and_validated():
    assert base_image.split_image("python:3.12-slim") == (
        "registry-1.docker.io",
        "library/python",
        "3.12-slim",
    )
    assert base_image.split_image("ghcr.io/org/app:1.0") == (
        "ghcr.io",
        "org/app",
        "1.0",
    )
    assert base_image.split_image("org/app:1.0")[1] == "org/app"
    for bad in ("python", "python@" + PLACEHOLDER, "python:3.12@sha256:ab"):
        with pytest.raises(errors.ConfigError):
            base_image.split_image(bad)


def test_resolving_uses_the_registry_digest_and_refuses_the_placeholder():
    digest = "sha256:" + "b" * 64
    record = base_image.resolve("python:3.12-slim", fetch=lambda image: digest)
    assert record["digest"] == digest and record["image"] == "python:3.12-slim"
    with pytest.raises(errors.ConfigError, match="placeholder"):
        base_image.resolve("python:3.12-slim", fetch=lambda image: PLACEHOLDER)
    with pytest.raises(errors.ConfigError, match="digest"):
        base_image.resolve("python:3.12-slim", fetch=lambda image: "latest")


def test_the_lock_commands(tmp_path, capsys):
    lock = tmp_path / "base.lock.json"
    assert base_image.main(["check", "--lock", str(lock)]) == 1
    lock.write_text(
        json.dumps({"image": "python:3.12-slim", "digest": PLACEHOLDER})
    )
    assert base_image.main(["check", "--lock", str(lock)]) == 1
    digest = "sha256:" + "c" * 64
    lock.write_text(json.dumps({"image": "python:3.12-slim", "digest": digest}))
    assert base_image.main(["check", "--lock", str(lock)]) == 0
    capsys.readouterr()
    assert base_image.main(["argument", "--lock", str(lock)]) == 0
    assert capsys.readouterr().out.strip() == f"python@{digest}"


def audit(*vulnerabilities):
    return {
        "dependencies": [
            {
                "name": "numpy",
                "version": "2.0.0",
                "vulns": list(vulnerabilities),
            },
            {"name": "pandas", "version": "3.0.0", "vulns": []},
        ]
    }


POLICY = {"fail_at_or_above": "HIGH", "exceptions": []}
TODAY = datetime.date(2026, 11, 5)


def test_the_gate_passes_a_clean_report_and_fails_any_pip_audit_finding():
    clean = vuln_gate.evaluate(audit(), POLICY, today=TODAY)
    assert clean["passed"] and clean["packages_scanned"] == 2
    found = vuln_gate.evaluate(
        audit({"id": "PYSEC-1", "fix_versions": ["2.0.1"]}), POLICY, today=TODAY
    )
    assert not found["passed"] and found["blocking"][0]["id"] == "PYSEC-1"


def test_the_gate_applies_severity_and_expiring_exceptions():
    trivy = {
        "Results": [
            {
                "Packages": [{"Name": "openssl", "Version": "3.0"}],
                "Vulnerabilities": [
                    {
                        "VulnerabilityID": "CVE-1",
                        "PkgName": "openssl",
                        "InstalledVersion": "3.0",
                        "Severity": "MEDIUM",
                    },
                    {
                        "VulnerabilityID": "CVE-2",
                        "PkgName": "openssl",
                        "InstalledVersion": "3.0",
                        "Severity": "CRITICAL",
                    },
                ],
            }
        ]
    }
    verdict = vuln_gate.evaluate(trivy, POLICY, today=TODAY)
    assert [item["id"] for item in verdict["blocking"]] == ["CVE-2"]
    allowed = {
        **POLICY,
        "exceptions": [
            {"id": "CVE-2", "reason": "not reachable", "expires": "2026-12-01"}
        ],
    }
    assert vuln_gate.evaluate(trivy, allowed, today=TODAY)["passed"]
    expired = vuln_gate.evaluate(
        trivy, allowed, today=datetime.date(2027, 1, 1)
    )
    assert not expired["passed"]


def test_the_gate_fails_closed():
    with pytest.raises(vuln_gate.GateError, match="no packages"):
        vuln_gate.evaluate({"dependencies": []}, POLICY, today=TODAY)
    with pytest.raises(vuln_gate.GateError, match="unrecognised"):
        vuln_gate.evaluate({"something": 1}, POLICY, today=TODAY)
    with pytest.raises(vuln_gate.GateError, match="fail_at_or_above"):
        vuln_gate.evaluate(audit(), {"fail_at_or_above": "NEVER"}, today=TODAY)
    with pytest.raises(vuln_gate.GateError, match="lacks 'reason'"):
        vuln_gate.evaluate(
            audit(),
            {**POLICY, "exceptions": [{"id": "x", "expires": "2030-01-01"}]},
            today=TODAY,
        )


def test_the_gate_checks_that_the_scan_covered_the_sbom(signed_service):
    document = json.loads((signed_service / "sbom.cdx.json").read_text())
    covered = {
        "dependencies": [
            {"name": item["name"], "version": item["version"], "vulns": []}
            for item in document["components"]
            if item["type"] == "library"
        ]
    }
    assert vuln_gate.evaluate(covered, POLICY, document, TODAY)["passed"]
    covered["dependencies"] = [
        item for item in covered["dependencies"] if item["name"] != "numpy"
    ]
    verdict = vuln_gate.evaluate(covered, POLICY, document, TODAY)
    assert not verdict["passed"] and verdict["sbom_components_not_scanned"]


def test_the_gate_script_exit_codes(tmp_path):
    report = tmp_path / "scan.json"
    command = [sys.executable, str(ROOT / "scripts" / "vuln_gate.py")]
    report.write_text(json.dumps(audit()))
    assert (
        subprocess.run(
            [*command, str(report)], check=False, capture_output=True
        ).returncode
        == 0
    )
    report.write_text(json.dumps(audit({"id": "X-1"})))
    assert (
        subprocess.run(
            [*command, str(report)], check=False, capture_output=True
        ).returncode
        == 1
    )
    report.write_text("not json")
    assert (
        subprocess.run(
            [*command, str(report)], check=False, capture_output=True
        ).returncode
        == 2
    )


def test_the_shipped_policy_is_valid():
    policy = json.loads((ROOT / "deploy" / "vuln-policy.json").read_text())
    vuln_gate.check_policy(policy, TODAY)
    assert policy["fail_at_or_above"] == "HIGH"


def test_the_framework_may_be_unscanned_only_while_its_exception_lives(
    signed_service,
):
    document = json.loads((signed_service / "sbom.cdx.json").read_text())
    scanned = {
        "dependencies": [
            {"name": item["name"], "version": item["version"], "vulns": []}
            for item in document["components"]
            if item["type"] == "library" and item["name"] != "scmle"
        ]
    }
    assert "scmle" in {item["name"] for item in document["components"]}
    waived = vuln_gate.evaluate(
        scanned, vuln_gate.DEFAULT_POLICY, document, TODAY
    )
    assert waived["passed"] and not waived["sbom_components_not_scanned"]
    expired = vuln_gate.evaluate(
        scanned, vuln_gate.DEFAULT_POLICY, document, datetime.date(2028, 1, 1)
    )
    assert not expired["passed"]
    assert expired["sbom_components_not_scanned"] == [
        f"scmle=={scmle.__version__}"
    ]
    no_waiver = vuln_gate.evaluate(scanned, POLICY, document, TODAY)
    assert not no_waiver["passed"]
