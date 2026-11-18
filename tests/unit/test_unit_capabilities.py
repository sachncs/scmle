"""Behaviour of capability declarations, the catalog and the registry."""

import json

import pytest

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.contracts.capabilities import axes
from scmle.contracts.capabilities import catalog
from scmle.contracts.capabilities import declarations
from scmle.contracts.capabilities import registry


def coordinate(**changes):
    values = {
        "modality": axes.Modality.TABULAR,
        "task": tasks.Task.BINARY,
        "model_family": "tree",
        "evaluation": axes.Evaluation.CALIBRATED,
        "uncertainty": axes.Uncertainty.NONE,
        "fairness": axes.Fairness.NONE,
        "explainability": axes.Explainability.GLOBAL,
        "inference": axes.Inference.LIBRARY,
    }
    values.update(changes)
    return declarations.Coordinate(**values)


def capability(status=axes.Status.SUPPORTED, **changes):
    values = {
        "key": coordinate(),
        "status": status,
        "required_extras": ("core",),
        "resources": declarations.Resources(
            cpu_cores=2, memory_gb=4.0, disk_gb=1.0
        ),
        "failure_modes": ("fails",),
        "evidence_links": ("tests/x.py",),
    }
    values.update(changes)
    return declarations.Capability(**values)


def family(**changes):
    values = {
        "model_family": "tree",
        "status": axes.Status.EXPERIMENTAL,
        "modalities": (axes.Modality.TABULAR,),
        "required_extras": ("core",),
        "resources": declarations.Resources(),
        "failure_modes": ("fails",),
    }
    values.update(changes)
    return declarations.Declaration(**values)


ENOUGH = declarations.Environment(
    available_extras=frozenset({"core"}),
    cpu_cores=2,
    memory_gb=4.0,
    disk_gb=1.0,
)


# ---- declarations ----------------------------------------------------------


def test_resources_are_stored_in_single_words():
    assert declarations.RESOURCES.renames["cpu_cores"] == "cpu"
    made = declarations.Resources(max_rows=10, notes=("b", "a", "b"))
    assert made.notes == ("a", "b")
    assert made.to_dict() == {
        "cpu": 1,
        "memory": 1.0,
        "disk": 0.0,
        "gpu": False,
        "network": False,
        "rows": 10,
        "notes": ["a", "b"],
    }


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"cpu_cores": 1.5}, errors.KindError, "cpu_cores must be an integer"),
        ({"cpu_cores": 0}, errors.InputError, "cpu_cores must be at least 1"),
        (
            {"memory_gb": 0.0},
            errors.InputError,
            "memory_gb must be greater than 0",
        ),
        ({"memory_gb": float("inf")}, errors.InputError, "memory_gb must be"),
        ({"disk_gb": -0.1}, errors.InputError, "disk_gb must not be negative"),
        ({"disk_gb": float("nan")}, errors.InputError, "disk_gb must not be"),
        ({"max_rows": 0}, errors.InputError, "max_rows must be at least 1"),
        (
            {"notes": (" ",)},
            errors.InputError,
            "resource notes must not be empty",
        ),
    ],
)
def test_resources_refuse_impossible_assumptions(changes, error, message):
    with pytest.raises(error, match=message):
        declarations.Resources(**changes)


def test_resources_accept_their_edges():
    made = declarations.Resources(cpu_cores=1, disk_gb=0.0, max_rows=1)
    assert made.max_rows == 1


def test_a_coordinate_dumps_every_dimension():
    assert coordinate().to_dict() == {
        "modality": "tabular",
        "task": "binary_classification",
        "family": "tree",
        "evaluation": "calibrated",
        "uncertainty": "none",
        "fairness": "none",
        "explainability": "global",
        "inference": "library",
    }


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        (
            {"modality": "tabular"},
            errors.KindError,
            "modality must be Modality",
        ),
        ({"task": "binary"}, errors.KindError, "task must be Task"),
        (
            {"inference": "library"},
            errors.KindError,
            "inference must be Inference",
        ),
        (
            {"model_family": 3},
            errors.KindError,
            "model_family must be a string",
        ),
        ({"model_family": ""}, errors.InputError, "must not be empty"),
        ({"model_family": "Tree"}, errors.InputError, "must be lowercase"),
        ({"model_family": " tree"}, errors.InputError, "must be lowercase"),
    ],
)
def test_a_coordinate_refuses_loose_values(changes, error, message):
    with pytest.raises(error, match=message):
        coordinate(**changes)


def test_a_family_declaration_canonicalizes_its_lists():
    made = family(
        modalities=(
            axes.Modality.TEXT,
            axes.Modality.IMAGE,
            axes.Modality.TEXT,
        ),
        required_extras=("b", "a", "a"),
        limitations=("z", "y"),
    )
    assert made.modalities == (axes.Modality.IMAGE, axes.Modality.TEXT)
    assert made.required_extras == ("a", "b")
    assert made.to_dict() == {
        "family": "tree",
        "status": "experimental",
        "modalities": ["image", "text"],
        "extras": ["a", "b"],
        "resources": declarations.Resources().to_dict(),
        "failures": ["fails"],
        "links": [],
        "limitations": ["y", "z"],
    }
    assert made.accommodates(axes.Modality.TEXT)
    assert not made.accommodates(axes.Modality.TABULAR)
    with pytest.raises(errors.KindError, match="modality must be Modality"):
        made.accommodates("text")


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        (
            {"model_family": 1},
            errors.KindError,
            "model_family must be a string",
        ),
        ({"model_family": ""}, errors.InputError, "must be lowercase"),
        ({"model_family": "Tree"}, errors.InputError, "must be lowercase"),
        ({"status": "experimental"}, errors.KindError, "status must be Status"),
        ({"resources": {}}, errors.KindError, "resources must be Resources"),
        ({"modalities": ()}, errors.InputError, "at least one value"),
        ({"modalities": ("tabular",)}, errors.KindError, "Modality values"),
        (
            {"failure_modes": ("",)},
            errors.InputError,
            "failure_modes must contain",
        ),
        ({"limitations": (1,)}, errors.InputError, "limitations must contain"),
    ],
)
def test_a_family_declaration_refuses_bad_values(changes, error, message):
    with pytest.raises(error, match=message):
        family(**changes)


def test_an_environment_records_what_was_observed():
    made = declarations.Environment(
        available_extras={"a"}, cpu_cores=1, memory_gb=0.0
    )
    assert made.available_extras == frozenset({"a"})
    assert made.gpu_available is None
    with pytest.raises(errors.InputError, match="non-empty strings"):
        declarations.Environment(available_extras={""})
    with pytest.raises(errors.InputError, match="cpu_cores must be at least 1"):
        declarations.Environment(cpu_cores=0)
    with pytest.raises(errors.InputError, match="memory_gb must be finite"):
        declarations.Environment(memory_gb=-1.0)
    with pytest.raises(errors.InputError, match="disk_gb must be finite"):
        declarations.Environment(disk_gb=float("inf"))


def test_a_reason_dumps_its_fields():
    assert declarations.Reason("c", "m", 1, 2).to_dict() == {
        "code": "c",
        "message": "m",
        "expected": 1,
        "observed": 2,
    }
    assert declarations.Reason("c", "m").to_dict()["expected"] is None


def test_a_capability_keeps_its_declaration_order_and_dumps():
    made = capability(limitations=["b", "a"])
    assert made.limitations == ("b", "a")
    record = made.to_dict()
    assert record["key"]["family"] == "tree"
    assert record["status"] == "supported"
    assert record["extras"] == ["core"]
    assert record["failures"] == ["fails"]
    assert record["links"] == ["tests/x.py"]
    assert record["resources"]["cpu"] == 2


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"key": {}}, errors.KindError, "key must be Coordinate"),
        ({"status": "supported"}, errors.KindError, "status must be Status"),
        ({"resources": None}, errors.KindError, "resources must be Resources"),
        ({"failure_modes": ()}, errors.InputError, "at least one value"),
        (
            {"required_extras": (" ",)},
            errors.InputError,
            "required_extras must",
        ),
        (
            {"evidence_links": ()},
            errors.InputError,
            "must declare evidence_links",
        ),
    ],
)
def test_a_capability_refuses_incomplete_declarations(changes, error, message):
    with pytest.raises(error, match=message):
        capability(**changes)


def test_an_experimental_capability_may_lack_evidence():
    made = capability(axes.Status.EXPERIMENTAL, evidence_links=())
    assert made.evidence_links == ()


def test_the_default_families_cover_the_transformer_families():
    names = [
        d.model_family for d in declarations.DEFAULT_MODEL_FAMILY_DECLARATIONS
    ]
    assert names == [f.value for f in axes.ModelFamily]
    assert declarations.UnsupportedError.category == "capability"
    assert issubclass(declarations.UnsupportedError, errors.ConfigError)


# ---- catalog ---------------------------------------------------------------


def test_a_catalog_refuses_duplicates_and_wrong_types():
    made = catalog.Catalog([capability()], [family()])
    with pytest.raises(
        errors.ConfigError, match="capability already registered"
    ):
        made.register(capability())
    with pytest.raises(
        errors.ConfigError, match="model family already registered: tree"
    ):
        made.enroll(family())
    with pytest.raises(errors.KindError, match="capability must be Capability"):
        made.register({})
    with pytest.raises(errors.KindError, match="declaration must be"):
        made.enroll({})
    with pytest.raises(errors.KindError, match="model_family must be a string"):
        made.find(1)
    with pytest.raises(
        errors.KindError, match="environment must be Environment"
    ):
        made.bind({})


def test_a_catalog_lists_in_a_deterministic_order():
    second = capability(key=coordinate(model_family="boost"))
    made = catalog.Catalog(
        [capability(), second], [family(), family(model_family="alpha")]
    )
    assert [c.key.model_family for c in made.catalogue()] == ["boost", "tree"]
    assert [d.model_family for d in made.list_model_families()] == [
        "alpha",
        "tree",
    ]
    assert made.find("tree").status is axes.Status.EXPERIMENTAL
    assert made.find("none") is None
    assert made.runtime_environment is None


def test_audit_passes_a_capable_environment():
    made = catalog.Catalog()
    assert made.audit(capability(), ENOUGH) == ()
    assert made.audit(capability(), None) == ()


def test_audit_lists_missing_extras_and_short_resources():
    made = catalog.Catalog()
    poor = declarations.Environment(
        available_extras=frozenset({"z"}),
        cpu_cores=1,
        memory_gb=3.9,
        disk_gb=None,
    )
    found = made.audit(capability(), poor)
    assert [r.code for r in found] == [
        "missing_extra",
        "insufficient_cpu_cores",
        "insufficient_memory_gb",
        "unknown_disk_gb",
    ]
    assert found[0].expected == ["core"]
    assert found[0].observed == ["z"]
    assert (found[1].expected, found[1].observed) == (2, 1)
    assert found[3].message == "disk_gb is required to verify resources"


def test_audit_checks_gpu_and_network_only_when_required():
    made = catalog.Catalog()
    needs = capability(
        required_extras=(),
        resources=declarations.Resources(
            gpu_required=True, network_required=True
        ),
    )
    unknown = declarations.Environment(cpu_cores=1, memory_gb=1.0, disk_gb=0.0)
    assert [r.code for r in made.audit(needs, unknown)] == [
        "unknown_gpu_available",
        "unknown_network_available",
    ]
    absent = declarations.Environment(
        cpu_cores=1,
        memory_gb=1.0,
        disk_gb=0.0,
        gpu_available=False,
        network_available=False,
    )
    found = made.audit(needs, absent)
    assert [r.code for r in found] == [
        "missing_gpu_available",
        "missing_network_available",
    ]
    assert found[0].message == "declared GPU resource is unavailable"
    present = declarations.Environment(
        cpu_cores=1,
        memory_gb=1.0,
        disk_gb=0.0,
        gpu_available=True,
        network_available=True,
    )
    assert made.audit(needs, present) == ()
    plain = capability(required_extras=(), resources=declarations.Resources())
    assert made.audit(plain, unknown) == ()


def test_audit_checks_a_row_limit_without_an_environment():
    made = catalog.Catalog()
    limited = capability(resources=declarations.Resources(max_rows=100))
    assert [r.code for r in made.audit(limited, None)] == ["unknown_row_count"]
    assert made.audit(limited, None, 100) == ()
    over = made.audit(limited, None, 101)
    assert [(r.code, r.expected, r.observed) for r in over] == [
        ("row_limit_exceeded", 100, 101)
    ]


def test_audit_uses_the_bound_environment_and_checks_types():
    made = catalog.Catalog()
    made.bind(declarations.Environment(cpu_cores=1, memory_gb=1.0, disk_gb=0.0))
    assert (
        next(r.code for r in made.audit(capability(), None)) == "missing_extra"
    )
    with pytest.raises(errors.KindError, match="capability must be Capability"):
        made.audit({}, None)
    with pytest.raises(
        errors.KindError, match="environment must be Environment"
    ):
        made.audit(capability(), {})


# ---- registry --------------------------------------------------------------


def test_the_schema_version_counts_the_migrations():
    assert registry.CAPABILITY_SCHEMA_VERSION == 2
    assert len(registry.MIGRATIONS) == 1


def test_a_seeded_registry_holds_the_default_families():
    made = registry.Registry.seed()
    assert made.find("jepa").status is axes.Status.EXPERIMENTAL
    assert made.catalogue() == ()


def test_a_registered_supported_capability_crosses_the_boundary():
    made = registry.Registry([capability()])
    assert made.diagnose(coordinate()) == ()
    assert (
        made.require(coordinate(), environment=ENOUGH)
        is made.records[coordinate()]
    )


def test_an_unregistered_key_is_refused_with_its_diagnostics():
    made = registry.Registry.seed()
    key = coordinate(model_family="vision_transformer")
    found = made.diagnose(key)
    assert [r.code for r in found] == [
        "missing_declaration",
        "model_family_modality_unsupported",
        "model_family_status_not_allowed",
    ]
    assert found[0].observed == {"model_family_status": "extension-only"}
    assert found[1].expected == ["image", "video"]
    assert found[2].expected == ["supported"]
    with pytest.raises(declarations.UnsupportedError) as caught:
        made.require(key)
    assert str(caught.value).startswith(
        "unsupported capability before training: no declaration exists"
    )
    details = caught.value.details
    assert details["status"] == "unsupported"
    assert details["key"] == key.to_dict()
    assert len(details["diagnostics"]) == 3
    assert details["registered_keys"] == []


def test_an_unknown_family_only_lacks_a_declaration():
    found = registry.Registry().diagnose(coordinate(model_family="mystery"))
    assert [r.code for r in found] == ["missing_declaration"]
    assert found[0].observed == {"model_family_status": None}


def test_a_family_in_an_allowed_status_and_modality_adds_nothing():
    made = registry.Registry.seed()
    key = coordinate(model_family="jepa")
    found = made.diagnose(key, {axes.Status.EXPERIMENTAL})
    assert [r.code for r in found] == ["missing_declaration"]


def test_an_experimental_capability_needs_an_explicit_status():
    made = registry.Registry([capability(axes.Status.EXPERIMENTAL)])
    found = made.diagnose(coordinate())
    assert [r.code for r in found] == ["status_not_allowed"]
    assert found[0].message == (
        "registered as experimental; accepted statuses do not include it"
    )
    assert found[0].expected == ["supported"]
    allowed = {axes.Status.EXPERIMENTAL}
    assert (
        made.require(coordinate(), allowed).status is axes.Status.EXPERIMENTAL
    )


def test_diagnose_checks_resources_and_row_limits():
    made = registry.Registry([capability()])
    poor = declarations.Environment(cpu_cores=1, memory_gb=1.0, disk_gb=0.0)
    codes = [r.code for r in made.diagnose(coordinate(), environment=poor)]
    assert codes[0] == "missing_extra"
    limited = registry.Registry(
        [capability(resources=declarations.Resources(max_rows=5))]
    )
    assert [r.code for r in limited.diagnose(coordinate())] == [
        "unknown_row_count"
    ]
    assert limited.diagnose(coordinate(), row_count=5) == ()
    assert made.diagnose(coordinate(), row_count=10**9) == ()
    with pytest.raises(
        declarations.UnsupportedError, match="exceeds capability row"
    ):
        limited.require(coordinate(), row_count=6)


def test_diagnose_checks_argument_types():
    made = registry.Registry()
    with pytest.raises(errors.KindError, match="key must be Coordinate"):
        made.diagnose({})
    with pytest.raises(
        errors.KindError, match="environment must be Environment"
    ):
        made.diagnose(coordinate(), environment={})
    with pytest.raises(errors.KindError, match="allowed_statuses must contain"):
        made.diagnose(coordinate(), ["supported"])


def test_a_registry_dumps_canonical_json():
    made = registry.Registry([capability()], [family()])
    record = made.to_dict()
    assert record["schema_version"] == 2
    assert record["model_families"][0]["family"] == "tree"
    assert record["capabilities"][0]["key"]["family"] == "tree"
    text = made.canonize()
    assert made.to_json() == text
    assert json.loads(text) == record
    assert " " not in text.replace("tests/x.py", "")


def test_refuse_always_raises_with_the_registered_keys():
    made = registry.Registry([capability()])
    with pytest.raises(declarations.UnsupportedError) as caught:
        made.refuse(coordinate(model_family="x"), "because")
    assert "because; key=" in str(caught.value)
    assert caught.value.details["registered_keys"] == [coordinate().to_dict()]
    assert caught.value.details["diagnostics"] == []
