"""The deployment contract: valid specs round-trip, invalid ones are refused."""

import dataclasses

import pytest

from scmle.contracts import errors
from scmle.deploy import contract

IMAGE = "sha256:" + "a" * 64
BUNDLE = "sha256:" + "b" * 64


def make_spec(**changes):
    values = {
        "version": "1.2.0",
        "environment": "staging",
        "image": contract.ImageRef("registry.example/scmle-svc", IMAGE),
        "bundle": contract.BundleRef("churn", "v3", BUNDLE, True),
        "resources": contract.Resources(0.5, 1.0, 256, 512),
        "config": {"PORT": "8000", "SCMLE_ENV": "staging"},
        "secrets": (
            contract.SecretRef("SCMLE_SIGNING_KEY", "env", "SIGNING_KEY_SRC"),
        ),
    }
    values.update(changes)
    return contract.Deployment(**values)


class FakeAdapter(contract.Provider):
    """Runs nothing; reports exactly what it was given, or a drift."""

    provider = "fake"

    def __init__(self, drift=None, architectures=None):
        self.drift = drift or {}
        self.architectures = architectures or (
            contract.Architecture.AMD64,
            contract.Architecture.ARM64,
        )
        self.last = None

    def supported_architectures(self):
        return self.architectures

    def apply(self, spec):
        values = {
            "spec_digest": spec.digest(),
            "image_digest": spec.image.digest,
            "bundle_digest": spec.bundle.digest,
            "model_version": spec.bundle.model_version,
            "ready": True,
            "secret_names": tuple(s.name for s in spec.secrets),
        }
        values.update(self.drift)
        self.last = contract.Runtime(**values)
        return self.last

    def observe(self):
        return self.last


def test_a_spec_round_trips_and_has_a_stable_digest():
    spec = make_spec()
    rebuilt = contract.Deployment.from_dict(spec.to_dict())
    assert rebuilt == spec
    assert rebuilt.digest() == spec.digest()
    assert spec.digest().startswith("sha256:")
    assert spec.image.reference == f"registry.example/scmle-svc@{IMAGE}"


def test_the_digest_changes_with_any_field():
    spec = make_spec()
    other = make_spec(
        resources=dataclasses.replace(spec.resources, memory_limit_mb=1024)
    )
    assert other.digest() != spec.digest()


def test_config_order_does_not_change_the_digest():
    first = make_spec(config={"A": "1", "B": "2"})
    second = make_spec(config={"B": "2", "A": "1"})
    assert first.digest() == second.digest()


@pytest.mark.parametrize(
    "digest", ["latest", "sha256:abc", "sha256:" + "G" * 64]
)
def test_an_image_must_be_pinned_by_a_real_digest(digest):
    with pytest.raises(errors.ConfigError, match="digest"):
        contract.ImageRef("registry.example/x", digest)


def test_architectures_must_be_known_and_unique():
    with pytest.raises(ValueError):
        contract.ImageRef("registry.example/x", IMAGE, ("sparc",))
    with pytest.raises(errors.ConfigError, match="unique"):
        contract.ImageRef("registry.example/x", IMAGE, ("amd64", "amd64"))


def test_secrets_are_references_not_values():
    with pytest.raises(errors.ConfigError, match="plain reference"):
        contract.SecretRef("TOKEN", "store", "abc=def")
    with pytest.raises(errors.ConfigError, match="plain reference"):
        contract.SecretRef("TOKEN", "store", "-----BEGIN PRIVATE KEY-----")
    with pytest.raises(errors.ConfigError, match="absolute"):
        contract.SecretRef("TOKEN", "file", "relative/path")
    with pytest.raises(errors.ConfigError, match="variable name"):
        contract.SecretRef("TOKEN", "env", "lower")


def test_a_secret_looking_config_key_is_refused():
    with pytest.raises(errors.ConfigError, match="looks like a secret"):
        make_spec(config={"API_TOKEN": "x"})
    with pytest.raises(errors.ConfigError, match="key material"):
        make_spec(config={"NOTE": "-----BEGIN RSA PRIVATE KEY-----"})


def test_a_name_cannot_be_both_config_and_secret():
    with pytest.raises(errors.ConfigError, match="both"):
        make_spec(
            config={"OTHER": "2"},
            secrets=(contract.SecretRef("OTHER", "env", "OTHER_SRC"),),
        )


def test_production_refuses_trusted_loading_and_unsigned_bundles():
    with pytest.raises(errors.ConfigError, match="SCMLE_TRUSTED"):
        make_spec(environment="production", config={"SCMLE_TRUSTED": "1"})
    with pytest.raises(errors.ConfigError, match="SCMLE_TRUSTED"):
        make_spec(config={"SCMLE_ENV": "production", "SCMLE_TRUSTED": "1"})
    with pytest.raises(errors.ConfigError, match="signed"):
        make_spec(
            environment="production",
            bundle=contract.BundleRef("churn", "v3", BUNDLE, False),
        )
    assert make_spec(environment="staging", config={"SCMLE_TRUSTED": "1"})


def test_resources_need_limits_above_requests_and_isolation():
    with pytest.raises(errors.ConfigError, match="cpu_limit"):
        contract.Resources(1.0, 0.5, 256, 512)
    with pytest.raises(errors.ConfigError, match="memory_limit_mb"):
        contract.Resources(0.5, 1.0, 512, 256)
    with pytest.raises(errors.ConfigError, match="read-only"):
        contract.Resources(0.5, 1.0, 256, 512, read_only_root=False)
    with pytest.raises(errors.ConfigError, match="positive"):
        contract.Resources(0.0, 1.0, 256, 512)
    with pytest.raises(errors.ConfigError, match="positive"):
        contract.Resources(True, 1.0, 256, 512)


def test_health_needs_distinct_endpoints_and_sane_timing():
    with pytest.raises(errors.ConfigError, match="different"):
        contract.Probes("/z", "/z")
    with pytest.raises(errors.ConfigError, match="start with"):
        contract.Probes("livez", "/readyz")
    with pytest.raises(errors.ConfigError, match="period"):
        contract.Probes(period_seconds=1.0, timeout_seconds=2.0)


def test_other_spec_formats_are_refused():
    record = make_spec().to_dict()
    record["format"] = 99
    with pytest.raises(errors.ConfigError, match="format"):
        contract.Deployment.from_dict(record)


def test_a_conforming_adapter_deploys():
    spec = make_spec()
    adapter = FakeAdapter()
    report = adapter.deploy(spec)
    assert contract.conformance_problems(spec, report) == []
    assert adapter.observe() is report


def test_an_adapter_that_drifts_is_refused_with_every_reason():
    spec = make_spec()
    drifted = FakeAdapter(
        {
            "image_digest": "sha256:" + "c" * 64,
            "ready": False,
            "secret_names": ("EXTRA",),
            "run_as_non_root": False,
            "memory_limit_mb": 4096,
        }
    )
    with pytest.raises(errors.ArtifactError) as raised:
        drifted.deploy(spec)
    problems = raised.value.details["problems"]
    joined = "; ".join(problems)
    for expected in (
        "image digest",
        "not ready",
        "required secrets",
        "undeclared secrets",
        "runs as root",
        "memory limit",
    ):
        assert expected in joined


def test_an_adapter_must_cover_every_image_architecture():
    adapter = FakeAdapter(architectures=(contract.Architecture.AMD64,))
    with pytest.raises(errors.ConfigError, match="arm64"):
        adapter.deploy(make_spec())
