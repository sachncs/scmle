"""The parity script sees every shipped certificate store (ERR-1)."""

import importlib.util
import pathlib

from scmle.assurance import coverage_slices
from scmle.assurance.admissibility import screen
import scmle.certificates
from scmle.data.identification import certified

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "parity.py"
STORES = {
    "detector_certificates.json",
    "f10_certificates.json",
    "f11_certificates.json",
    "f4_certificates.json",
    "f6_certificates.json",
    "framing_certificates.json",
    "ingest_certificates.json",
    "loop_certificates.json",
    "screen_certificates.json",
    "subgroup_certificates.json",
    "uncertainty_certificates.json",
}


def load():
    """Imports the script as a module, since `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location("parity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_store_ships_inside_the_package():
    shipped = {p.name for p in scmle.certificates.DIRECTORY.glob("*.json")}
    assert shipped == STORES
    package = ROOT / "scmle"
    assert scmle.certificates.DIRECTORY.is_relative_to(package)


def test_every_reader_defaults_to_the_shipped_folder():
    folder = scmle.certificates.DIRECTORY
    assert screen.STORE_PATH.parent == folder
    assert certified.STORE_PATH.parent == folder
    assert certified.SUBGROUP_STORE_PATH.parent == folder
    assert coverage_slices.STORE_PATH.parent == folder


def test_the_survey_finds_every_gate_family():
    found = load().survey()
    assert set(found["stores"]) == STORES
    assert found["detectors"] == "loaded"
    assert found["robustness"] == "loaded"
    assert found["screen"] == "current"
    assert found["identification"]
    assert load().main() == 0


def test_a_missing_store_fails(monkeypatch, tmp_path):
    module = load()
    monkeypatch.setattr(scmle.certificates, "DIRECTORY", tmp_path)
    assert module.main() == 1
