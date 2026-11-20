"""Behaviour of the deployment spec, its references and runtime conformance."""

import dataclasses
import hashlib
import json

import pytest

from scmle.contracts import errors
from scmle.contracts import records
from scmle.deploy.contract import deployment
from scmle.deploy.contract import references
from scmle.deploy.contract import runtime

IMAGE = "sha256:" + "a" * 64
BUNDLE = "sha256:" + "b" * 64
OTHER = "sha256:" + "c" * 64


def spec(**changes):
    values = {
        "version": "1.0",
        "environment": "staging",
        "image": references.Image("repo/serve", IMAGE),
        "bundle": references.Bundle("churn", "3", BUNDLE, True),
        "resources": deployment.Resources(
            deployment.Quota(0.5, 256), deployment.Quota(1.0, 512)
        ),
        "config": {"LOG_LEVEL": "info"},
        "secrets": (
            references.Secret("API_TOKEN", "env", "API_TOKEN_SOURCE"),
            references.Secret("EXTRA", "file", "/run/x", required=False),
        ),
    }
    values.update(changes)
    return deployment.Deployment(**values)


def report(made, **changes):
    values = {
        "spec": made.digest(),
        "image": IMAGE,
        "digest": BUNDLE,
        "version": "3",
        "ready": True,
        "secrets": ("API_TOKEN",),
    }
    values.update(changes)
    return runtime.Runtime(**values)


# ---- references ------------------------------------------------------------


def test_reference_constants_are_the_documented_patterns():
    assert references.PRIVATE_KEY_MARKER == "-----BEGIN"
    assert references.Architecture.AMD64 == "amd64"
    assert references.Source.ENVIRONMENT == "env"
    assert [e.value for e in references.Environment] == [
        "local",
        "staging",
        "production",
    ]


@pytest.mark.parametrize(
    "value",
    [IMAGE.upper(), "sha256:" + "a" * 63, "sha256:" + "a" * 65, "latest", 5],
)
def test_require_digest_refuses_anything_but_a_sha256_digest(value):
    with pytest.raises(errors.ConfigError, match="f must be 'sha256:'"):
        references.require(value, "f")


def test_require_digest_returns_a_digest():
    assert references.require(IMAGE, "f") == IMAGE


def test_simple_accepts_names_up_to_128_characters():
    assert references.simple("a" * 128, "n") == "a" * 128
    for bad in ("a" * 129, "-a", "", "a b", None):
        with pytest.raises(errors.ConfigError, match="n is not a valid name"):
            references.simple(bad, "n")


@pytest.mark.parametrize(
    "value", [0, -1, True, float("inf"), float("nan"), "1"]
)
def test_require_positive_refuses_non_positive_or_odd_values(value):
    with pytest.raises(
        errors.ConfigError, match="x must be positive and finite"
    ):
        references.enforce(value, "x")


def test_require_positive_returns_small_positive_values():
    assert references.enforce(1e-9, "x") == 1e-9
    assert references.enforce(3, "x") == 3


def test_an_image_reference_converts_and_sorts_architectures():
    image = references.Image("repo", IMAGE, ("arm64", "amd64"))
    assert image.architectures == (
        references.Architecture.ARM64,
        references.Architecture.AMD64,
    )
    assert image.reference == f"repo@{IMAGE}"
    assert image.to_dict() == {
        "repository": "repo",
        "digest": IMAGE,
        "architectures": ["amd64", "arm64"],
    }
    assert references.Image.from_dict(image.to_dict()).to_dict() == (
        image.to_dict()
    )


@pytest.mark.parametrize("architectures", [(), ("amd64", "amd64")])
def test_an_image_needs_unique_architectures(architectures):
    with pytest.raises(errors.ConfigError, match="non-empty and unique"):
        references.Image("repo", IMAGE, architectures)


def test_an_image_with_an_unknown_architecture_is_a_value_error():
    with pytest.raises(ValueError):
        references.Image("repo", IMAGE, ("s390x",))


def test_a_bundle_reference_validates_and_round_trips():
    bundle = references.Bundle("churn", "1.2+b", BUNDLE, False)
    assert bundle.to_dict() == {
        "model": "churn",
        "version": "1.2+b",
        "digest": BUNDLE,
        "signed": False,
    }
    assert references.Bundle.from_dict(bundle.to_dict()) == bundle
    with pytest.raises(
        errors.ConfigError, match=r"bundle\.version is not valid"
    ):
        references.Bundle("churn", "-1", BUNDLE, True)
    with pytest.raises(errors.ConfigError, match=r"bundle\.signed must be"):
        references.Bundle("churn", "1", BUNDLE, 1)
    with pytest.raises(errors.ConfigError, match=r"bundle\.digest"):
        references.Bundle("churn", "1", "sha256:x", True)


@pytest.mark.parametrize(
    ("source", "locator", "message"),
    [
        ("env", "", "locator must be a plain reference"),
        ("env", " PAD", "locator must be a plain reference"),
        ("env", "A\nB", "locator must be a plain reference"),
        ("env", "A=B", "locator must be a plain reference"),
        ("store", "-----BEGIN key", "locator must be a plain reference"),
        ("env", "lower", "env locator must be a variable name"),
        ("file", "relative/path", "file locator must be an absolute path"),
        ("store", "bad name", "locator is not a valid name"),
    ],
)
def test_a_secret_reference_refuses_inline_or_unsuitable_locators(
    source, locator, message
):
    with pytest.raises(errors.ConfigError, match=message):
        references.Secret("TOKEN", source, locator)


def test_a_secret_name_must_be_an_upper_case_variable():
    with pytest.raises(errors.ConfigError, match="upper-case variable"):
        references.Secret("token", "env", "TOKEN")


def test_a_secret_reference_round_trips_and_defaults_to_required():
    made = references.Secret("TOKEN", "store", "vault/path-1")
    assert made.source is references.Source.STORE
    assert made.to_dict() == {
        "name": "TOKEN",
        "source": "store",
        "locator": "vault/path-1",
        "required": True,
    }
    record = {"name": "TOKEN", "source": "file", "locator": "/a"}
    assert references.Secret.from_dict(record).required is True


# ---- deployment ------------------------------------------------------------


def test_deployment_constants_are_the_documented_values():
    assert deployment.SPEC_FORMAT == 2
    assert deployment.PORT_LIMIT == 65536
    assert deployment.TRUSTED_VARIABLE == "SCMLE_TRUSTED"
    assert deployment.ENVIRONMENT_VARIABLE == "SCMLE_ENV"
    assert "KEY" in deployment.SECRET_WORDS


@pytest.mark.parametrize(
    ("requests", "limits", "message"),
    [
        ((2.0, 256), (1.0, 512), "limits.cpu must be at least requests.cpu"),
        ((0.5, 600), (1.0, 512), "limits.memory must be at least"),
        ((0, 256), (1.0, 512), "resources.requests.cpu must be positive"),
        ((0.5, 256), (1.0, 0), "resources.limits.memory must be positive"),
    ],
)
def test_resources_limits_must_dominate_requests(requests, limits, message):
    with pytest.raises(errors.ConfigError, match=message):
        deployment.Resources(
            deployment.Quota(*requests), deployment.Quota(*limits)
        )


def test_resources_equal_to_requests_are_allowed_and_round_trip():
    made = deployment.Resources(
        deployment.Quota(1.0, 512), deployment.Quota(1.0, 512)
    )
    assert made.to_dict() == {
        "requests": {"cpu": 1.0, "memory": 512},
        "limits": {"cpu": 1.0, "memory": 512},
        "scratch": 64,
        "immutable": True,
        "rootless": True,
        "concurrency": 8,
    }
    assert deployment.Resources.from_dict(made.to_dict()) == made


@pytest.mark.parametrize(
    "changes",
    [{"immutable": False}, {"rootless": False}],
)
def test_resources_need_isolation(changes):
    with pytest.raises(
        errors.ConfigError, match="runs non-root on a read-only"
    ):
        deployment.Resources(
            deployment.Quota(1.0, 1), deployment.Quota(1.0, 1), **changes
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"liveness": "live"}, "health.liveness must start with '/'"),
        ({"readiness": "ready"}, "health.readiness must start with '/'"),
        ({"readiness": "/livez"}, "must be different endpoints"),
        ({"timeout": 10.5}, "probe timeout exceeds the probe period"),
        ({"threshold": 0}, "health.threshold must be positive"),
        ({"grace": -1}, "health.grace must be positive"),
    ],
)
def test_probes_refuse_bad_paths_and_timings(changes, message):
    with pytest.raises(errors.ConfigError, match=message):
        deployment.Probes(**changes)


def test_probes_allow_a_timeout_equal_to_the_period():
    made = deployment.Probes(timeout=10.0)
    assert deployment.Probes.from_dict(made.to_dict()) == made
    assert made.to_dict()["liveness"] == "/livez"


def test_canonize_is_compact_sorted_ascii_json():
    assert deployment.canonize({"b": 1, "a": "é"}) == b'{"a":"\\u00e9","b":1}'


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"lower": "x"}, "config key 'lower' is not valid"),
        ({"LEVEL": 1}, "config LEVEL must be text"),
        ({"DB_PASSWORD": "x"}, "config DB_PASSWORD looks like a secret"),
        ({"API_KEY": "x"}, "looks like a secret"),
        ({"CERT": "-----BEGIN RSA"}, "config CERT holds key material"),
    ],
)
def test_nonsecret_refuses_secrets_in_config(config, message):
    with pytest.raises(errors.ConfigError, match=message):
        deployment.nonsecret(config)


def test_a_deployment_sorts_its_config_and_round_trips():
    made = spec(config={"ZED": "1", "ALPHA": "2"})
    assert list(made.config) == ["ALPHA", "ZED"]
    record = made.to_dict()
    assert record["format"] == 2
    assert [s["name"] for s in record["secrets"]] == ["API_TOKEN", "EXTRA"]
    assert record["port"] == 8000
    assert deployment.Deployment.from_dict(record) == made


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"version": "-bad"}, "version is not valid"),
        ({"port": 0}, "port must be between 1 and 65535"),
        ({"port": 65536}, "port must be between 1 and 65535"),
        (
            {
                "secrets": (
                    references.Secret("A", "env", "B"),
                    references.Secret("A", "env", "C"),
                )
            },
            "secret names must be unique",
        ),
        (
            {
                "config": {"API_TOKEN": "x"},
            },
            "looks like a secret",
        ),
        (
            {
                "config": {"LEVEL": "1"},
                "secrets": (references.Secret("LEVEL", "env", "B"),),
            },
            "names are both config and secret: LEVEL",
        ),
    ],
)
def test_a_deployment_refuses_invalid_specs(changes, message):
    with pytest.raises(errors.ConfigError, match=message):
        spec(**changes)


def test_ports_at_the_edges_are_accepted():
    assert spec(port=1).port == 1
    assert spec(port=65535).port == 65535


def test_production_needs_a_signed_untrusted_bundle():
    unsigned = references.Bundle("churn", "3", BUNDLE, False)
    with pytest.raises(errors.ConfigError, match="requires a signed bundle"):
        spec(environment="production", bundle=unsigned)
    with pytest.raises(errors.ConfigError, match="requires a signed bundle"):
        spec(bundle=unsigned, config={"SCMLE_ENV": "production"})
    with pytest.raises(errors.ConfigError, match="SCMLE_TRUSTED=1 is refused"):
        spec(environment="production", config={"SCMLE_TRUSTED": "1"})
    assert (
        spec(bundle=unsigned, config={"SCMLE_TRUSTED": "1"}).bundle == unsigned
    )
    assert spec(environment="production", config={"SCMLE_TRUSTED": "0"})


def test_a_newer_spec_format_is_refused():
    # Enforces website/src/pages/docs/api-stability.astro:36.
    record = spec().to_dict()
    record["format"] = 3
    with pytest.raises(
        errors.ConfigError, match="unsupported deployment spec format 3"
    ):
        deployment.Deployment.from_dict(record)


def test_the_digest_is_anchored_at_format_one():
    made = spec()
    digest = made.digest()
    assert digest.startswith("sha256:")
    assert len(digest) == 71
    assert spec().digest() == digest
    assert spec(port=8001).digest() != digest
    old = deployment.Deployment.from_dict(
        json.loads(
            json.dumps(
                {
                    "format": 1,
                    "version": "1.0",
                    "environment": "staging",
                    "image": made.image.to_dict(),
                    "bundle": {
                        "model": "churn",
                        "model_version": "3",
                        "digest": BUNDLE,
                        "signed": True,
                    },
                    "resources": {
                        "cpu_request": 0.5,
                        "memory_request_mb": 256,
                        "cpu_limit": 1.0,
                        "memory_limit_mb": 512,
                        "scratch_mb": 64,
                        "read_only_root": True,
                        "run_as_non_root": True,
                        "max_concurrent_requests": 8,
                    },
                    "health": {
                        "liveness_path": "/livez",
                        "readiness_path": "/readyz",
                        "period_seconds": 10.0,
                        "timeout_seconds": 2.0,
                        "failure_threshold": 3,
                        "startup_grace_seconds": 60.0,
                    },
                    "config": {"LOG_LEVEL": "info"},
                    "secrets": [s.to_dict() for s in made.secrets],
                    "port": 8000,
                }
            )
        )
    )
    assert old == made
    assert old.digest() == digest


def test_the_digest_hashes_the_canonical_format_one_layout():
    made = spec()
    layout = made.to_dict()
    del layout["format"]

    layout = records.downgrade(made.migrations, layout)
    layout["format"] = records.ORIGIN
    expected = hashlib.sha256(deployment.canonize(layout)).hexdigest()
    assert made.digest() == "sha256:" + expected


# ---- runtime ---------------------------------------------------------------


def test_a_runtime_report_needs_digests():
    with pytest.raises(errors.ConfigError, match=r"report\.spec_digest"):
        runtime.Runtime("x", IMAGE, BUNDLE, "1", True)
    with pytest.raises(errors.ConfigError, match=r"report\.image_digest"):
        runtime.Runtime(IMAGE, "x", BUNDLE, "1", True)
    with pytest.raises(errors.ConfigError, match=r"report\.bundle_digest"):
        runtime.Runtime(IMAGE, IMAGE, "x", "1", True)


def test_a_conforming_report_has_no_problems():
    made = spec()
    assert runtime.audit(made, report(made)) == []
    exact = report(made, memory=512, limit=1.0)
    assert runtime.diff(made, exact) == []


def test_divergences_list_identity_differences_in_order():
    made = spec()
    seen = report(
        made,
        spec=OTHER,
        image=OTHER,
        digest=OTHER,
        version="4",
        ready=False,
    )
    assert runtime.divergences(made, seen) == [
        "runtime was started from a different spec",
        f"image digest {OTHER} != {IMAGE}",
        f"bundle digest {OTHER} != {BUNDLE}",
        "model version differs from the spec",
        "service is not ready",
    ]


def test_mounts_list_missing_and_undeclared_secrets():
    made = spec()
    assert runtime.mounts(made, report(made, secrets=("ZZ", "AA"))) == [
        "required secrets not mounted: API_TOKEN",
        "undeclared secrets mounted: AA, ZZ",
    ]
    optional = report(made, secrets=("API_TOKEN", "EXTRA"))
    assert runtime.mounts(made, optional) == []


def test_diff_lists_isolation_and_limit_differences():
    made = spec()
    seen = report(
        made,
        immutable=False,
        unprivileged=False,
        memory=513,
        limit=1.01,
    )
    assert runtime.diff(made, seen) == [
        "root filesystem is writable",
        "container runs as root",
        "memory limit exceeds the spec",
        "cpu limit exceeds the spec",
    ]


class Fake(runtime.Provider):
    provider = "fake"

    def __init__(self, seen, architectures=("amd64", "arm64")):
        self.seen = seen
        self.architectures = tuple(
            references.Architecture(a) for a in architectures
        )

    def support(self):
        return self.architectures

    def apply(self, spec):
        return self.seen

    def observe(self):
        return self.seen


def test_a_provider_deploys_a_conforming_spec():
    made = spec()
    assert Fake(report(made)).deploy(made) == report(made)


def test_a_provider_refuses_an_architecture_it_cannot_run():
    made = spec()
    with pytest.raises(errors.ConfigError, match="fake cannot run arm64"):
        Fake(report(made), ("amd64",)).deploy(made)
    anonymous = Fake(report(made), ("amd64",))
    anonymous.provider = ""
    with pytest.raises(errors.ConfigError, match="adapter cannot run arm64"):
        anonymous.deploy(made)


def test_a_provider_refuses_a_report_that_does_not_conform():
    made = spec()
    with pytest.raises(errors.ArtifactError) as caught:
        Fake(report(made, ready=False)).deploy(made)
    assert str(caught.value) == (
        "deployment does not match its spec: service is not ready"
    )
    assert caught.value.details == {"problems": ["service is not ready"]}


def test_a_runtime_report_defaults_to_isolation():
    made = report(spec())
    assert made.immutable is True
    assert made.unprivileged is True
    assert made.memory is None
    assert dataclasses.replace(made, ready=False).ready is False
