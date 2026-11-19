"""Guarantees of docs/API_STABILITY.md that no other test enforced."""

import json
import pathlib

import numpy
import pandas
import pytest

from scmle.assurance.admissibility import screen
from scmle.contracts import data
from scripts import check_api_compatibility

SNAPSHOT = (
    pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "public_api.json"
)
CONTRACT = (
    "BundleRef",
    "Deployment",
    "ImageRef",
    "Probes",
    "Provider",
    "Resources",
    "Runtime",
    "SecretRef",
    "conformance_problems",
)
SECURITY = {
    "scmle.security.tokens": (
        "Key",
        "Keystore",
        "Principal",
        "RevocationList",
        "Verifier",
    ),
    "scmle.security.access": ("DeniedError", "Gateway", "RateLimiter"),
    "scmle.security.audit": ("Log",),
}
GOVERNED = (
    "scmle.runtime.retry",
    "scmle.storage.objectstore",
    "scmle.data.store.processing",
    "scmle.data.store.quality",
    "scmle.data.store.snapshot",
)


def surfaces():
    """The `surfaces` block of the checked-in snapshot.

    Returns:
        A mapping from module name to its pinned names and signatures.
    """
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))["surfaces"]


def test_the_deployment_contract_is_pinned_and_every_change_is_reported():
    """Enforces docs/API_STABILITY.md:202.

    Also enforces website/src/pages/docs/api-stability.astro:34.
    """
    pinned = surfaces()["scmle.deploy.contract"]
    for name in CONTRACT:
        assert pinned[name]["signature_sha256"], name
    expected = check_api_compatibility.current_snapshot()
    changed = json.loads(json.dumps(expected))
    block = changed["surfaces"]["scmle.deploy.contract"]
    block["Extra"] = dict(block["Runtime"])
    block.pop("Provider")
    block["Deployment"]["signature_sha256"] = "changed"
    found = check_api_compatibility.differences(expected, changed)
    assert "surface added: scmle.deploy.contract.Extra" in found
    assert "surface removed: scmle.deploy.contract.Provider" in found
    assert "surface contract changed: scmle.deploy.contract.Deployment" in found


@pytest.mark.parametrize("module", sorted(SECURITY))
def test_the_security_names_are_pinned_in_the_snapshot(module):
    """Enforces docs/API_STABILITY.md:209."""
    pinned = surfaces()[module]
    for name in SECURITY[module]:
        assert pinned[name]["signature_sha256"], name


@pytest.mark.parametrize("module", GOVERNED)
def test_the_governed_data_modules_are_pinned_in_the_snapshot(module):
    """Enforces docs/API_STABILITY.md:221."""
    pinned = surfaces()[module]
    assert pinned
    assert all(entry["signature_sha256"] for entry in pinned.values())


def test_a_portrait_has_an_empty_leakage_field_and_writes_its_key():
    """Enforces docs/API_STABILITY.md:54.

    Also enforces website/src/pages/docs/api-stability.astro:62.
    """
    portrait = data.Portrait(
        manifest=data.DatasetManifest(
            "ds", {"schema": "schema", "data": "data"}, 3, 1, "y"
        ),
        column_statistics={},
        quality_findings=(),
        risk_findings=(),
        target_summary={},
        feature_plan=data.Blueprint(("x",), ()),
    )
    assert portrait.leakage == {}
    assert portrait.blockers == ()
    assert portrait.to_dict()["leakage"] == {}


def test_a_multiclass_copy_of_the_target_blocks_in_a_run_screen():
    """Enforces docs/API_STABILITY.md:54 and docs/ARTIFACTS.md:32.

    The screen a run uses lets every mechanism gate, so a suspect blocks for
    a three-class target too, not only for a binary one.

    Also enforces website/src/pages/docs/api-stability.astro:64.
    Also enforces website/src/pages/docs/artifacts.astro:15.
    """
    rng = numpy.random.default_rng(0)
    frame = pandas.DataFrame({f"x{i}": rng.normal(size=300) for i in range(3)})
    frame["y"] = rng.choice(["a", "b", "c"], 300)
    frame["leak"] = frame["y"].map({"a": 0.0, "b": 1.0, "c": 2.0})
    found = screen.screen_table(
        frame, "y", "classification", ["x0", "x1", "x2", "leak"], 7
    )
    assert set(found.gates.values()) == {screen.certificates.Authority.GATING}
    assert [f.feature for f in found.blocking] == ["leak"]
    assert found.to_dict()["blocking"] == ["leak"]
