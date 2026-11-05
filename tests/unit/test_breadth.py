"""Breadth slices: definitions are well formed and nothing unsupported is
in the production registry.

The definitions under ``benchmarks/breadth/`` are written before any slice is
built. These tests keep them honest (nothing run, nothing fixed, nothing
claimed) and keep the production registry to what has evidence.
"""

import copy
import pathlib
import re

import pytest

from benchmarks import breadth
from scmle.contracts import capabilities
from scmle.contracts import modalities
from scmle.contracts import tasks
from scmle.core import context
from scmle.models import registry

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "docs" / "BREADTH.md"
UNSUPPORTED_NAMES = frozenset(
    {
        "kmeans",
        "k_means",
        "gaussian_mixture",
        "agglomerative",
        "dbscan",
        "hdbscan",
        "apriori",
        "fp_growth",
        "eclat",
        "pca",
        "kernel_pca",
        "tsne",
        "umap",
        "svd",
        "nmf",
        "autoencoder",
        "arima",
        "sarima",
        "prophet",
        "ets",
        "bert",
        "resnet",
        "vit",
        "wav2vec",
        "clip",
        "gnn",
        "gcn",
        "graphsage",
        "ray_train",
        "horovod",
        "kafka_consumer",
    }
)
FAMILIES = {
    capabilities.ModelFamily.TRANSFORMER.value,
    capabilities.ModelFamily.TEXT_TRANSFORMER.value,
    capabilities.ModelFamily.VISION_TRANSFORMER.value,
    capabilities.ModelFamily.AUDIO_TRANSFORMER.value,
    capabilities.ModelFamily.VIDEO_TRANSFORMER.value,
    capabilities.ModelFamily.MULTIMODAL_TRANSFORMER.value,
    capabilities.ModelFamily.JEPA.value,
}


def test_the_definition_files_are_exactly_the_declared_slices():
    found = {path.stem for path in breadth.FOLDER.glob("*.json")}
    assert found == set(breadth.SLICES)


@pytest.mark.parametrize("name", breadth.SLICES)
def test_each_definition_is_well_formed(name):
    definition = breadth.load(name)
    assert breadth.problems(name, definition) == []
    assert tuple(definition["benchmarks"]) == breadth.SECTIONS
    assert definition["status"] == "definition_only"


@pytest.mark.parametrize("name", breadth.SLICES)
def test_nothing_is_run_fixed_or_declared(name):
    definition = breadth.load(name)
    assert definition["deployment_requirement"]["declared"] is False
    assert definition["deployment_requirement"]["declared_by"] is None
    for section in definition["benchmarks"].values():
        assert section["status"] == "not_run"
        assert all(v is None for v in section["parameters"].values())


def mutated(change):
    """A copy of the clustering definition after ``change``."""
    definition = copy.deepcopy(breadth.load("clustering"))
    change(definition)
    return breadth.problems("clustering", definition)


def test_a_malformed_definition_is_reported():
    def claim(d):
        d["status"] = "qualified"

    def run(d):
        d["benchmarks"]["drift"]["status"] = "passed"

    def fix(d):
        d["benchmarks"]["robustness"]["parameters"]["ari_floor"] = 0.7

    def drop(d):
        del d["benchmarks"]["explanation"]

    def blank(d):
        d["benchmarks"]["monitoring"]["metric"] = " "

    def cite(d):
        d["evidence_in_repo"].append("scmle/does_not_exist.py")

    def declare(d):
        d["deployment_requirement"]["declared"] = True

    def silent(d):
        d["not_claimed"] = []

    def stages(d):
        d["lifecycle_required"] = ["ingestion"]

    for change, word in (
        (claim, "status"),
        (run, "not_run"),
        (fix, "fixed"),
        (drop, "exactly"),
        (blank, "empty"),
        (cite, "does not exist"),
        (declare, "declared"),
        (silent, "disclaim"),
        (stages, "lifecycle"),
    ):
        assert any(word in item for item in mutated(change)), change.__name__


def test_every_supported_capability_is_tabular_library_inference():
    found = context.tabular_beta_capability_registry().list_capabilities()
    supported = [
        c for c in found if c.status is capabilities.CapabilityStatus.SUPPORTED
    ]
    assert supported
    for capability in supported:
        assert capability.key.modality is capabilities.Modality.TABULAR
        assert capability.key.inference is capabilities.Inference.LIBRARY


def test_no_pretrained_or_multimodal_family_is_supported():
    found = context.tabular_beta_capability_registry().list_capabilities()
    for capability in found:
        if capability.key.model_family in FAMILIES:
            assert (
                capability.status is capabilities.CapabilityStatus.EXPERIMENTAL
            )


def test_streaming_inference_is_not_declared_supported():
    found = context.tabular_beta_capability_registry().list_capabilities()
    assert not [
        c
        for c in found
        if c.key.inference is capabilities.Inference.STREAMING
        and c.status is capabilities.CapabilityStatus.SUPPORTED
    ]


def test_unsupported_algorithms_are_not_in_the_model_registry():
    assert UNSUPPORTED_NAMES.isdisjoint(registry.builtin_names())


def test_unsupervised_families_are_not_task_types():
    values = {task.value for task in tasks.TaskType}
    for word in ("clustering", "association", "dimensionality"):
        assert not [v for v in values if word in v]


def test_an_extension_adapter_never_inherits_supported():
    key = capabilities.CapabilityKey(
        modality=capabilities.Modality.IMAGE,
        task=tasks.TaskType.BINARY,
        model_family="acme_cnn",
        evaluation=capabilities.Evaluation.STANDARD,
        uncertainty=capabilities.Uncertainty.NONE,
        fairness=capabilities.Fairness.NONE,
        explainability=capabilities.Explainability.NONE,
        inference=capabilities.Inference.LIBRARY,
    )
    declared = context.extension_capability(key)
    assert declared.status is capabilities.CapabilityStatus.EXTENSION_ONLY


def test_only_the_contract_module_can_name_a_supported_modality():
    named = []
    for path in (ROOT / "scmle").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "ModalityStatus.SUPPORTED" in text:
            named.append(path.relative_to(ROOT).as_posix())
    assert named == ["scmle/contracts/modalities.py"]
    assert modalities.ModalityStatus.SUPPORTED.value == "supported"


def table_rows():
    """Rows of the status table in docs/BREADTH.md, by slice."""
    rows = {}
    for line in DOCUMENT.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\|\s*`(\w+)`\s*\|(.*)\|\s*$", line)
        if match:
            rows[match.group(1)] = [
                c.strip() for c in match.group(2).split("|")
            ]
    return rows


def test_the_status_table_covers_every_slice_and_qualifies_none():
    rows = table_rows()
    assert set(rows) == set(breadth.SLICES)
    for name, cells in rows.items():
        assert cells[0] == f"`{breadth.load(name)['built']}`", name
        assert cells[1] == "no", name


def test_the_document_does_not_claim_exactly_once_streaming():
    text = DOCUMENT.read_text(encoding="utf-8")
    assert "Exactly-once streaming is not claimed" in text
    assert "no slice is qualified" in text.lower()
