"""Every gate in the code is registered, and every registered gate exists (C1)."""

import pytest

from benchmarks import registry

GATES = registry.load()


def test_the_registry_and_the_code_agree():
    assert registry.compare(GATES, registry.scan()) == []
    assert registry.main() == 0


def test_every_gate_says_how_it_passes_blocks_and_is_overridden():
    for gate in GATES.values():
        assert gate.passes and gate.blocks and gate.override, gate.name
        assert set(gate.claims) <= registry.CLAIMS


def test_an_unregistered_raise_is_found(tmp_path):
    package = tmp_path / "scmle"
    package.mkdir()
    (package / "planted.py").write_text(
        "from scmle.contracts import errors\n\n"
        "class Door:\n"
        "    def open(self):\n"
        "        raise errors.BlockedError('shut')\n",
        encoding="utf-8",
    )
    sites = registry.scan(package)
    assert sites == {"scmle.planted:Door.open": "BlockedError"}
    assert registry.compare({}, sites) == [
        "unregistered gate: scmle.planted:Door.open raises BlockedError"
    ]


def test_a_stale_or_mistyped_entry_is_found():
    gate = registry.Gate(
        "door",
        "scmle.planted:Door.open",
        "LeakageError",
        ("C1",),
        "p",
        "b",
        "o",
    )
    assert registry.compare({"door": gate}, {}) == [
        "registered gate no longer raises: door (scmle.planted:Door.open)"
    ]
    assert registry.compare(
        {"door": gate}, {"scmle.planted:Door.open": "BlockedError"}
    ) == ["door: registered as LeakageError, raises BlockedError"]


def table(**fields):
    """A well-formed gate, with `fields` replacing its own."""
    base = {
        "site": "scmle.a:b",
        "error": "BlockedError",
        "claim": "C1",
        "passes": "p",
        "blocks": "b",
        "override": "none",
    }
    return {**base, **fields}


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"site": "scmle.a:b"}, "missing fields"),
        (table(extra=1), "unknown fields"),
        (table(passes=" "), "non-empty"),
        (table(error="ValueError"), "error must be"),
        (table(claim="C99"), "unknown claim"),
        (table(site="scmle.a.b"), "module:qualname"),
        (table(variables="x"), "must be a list"),
    ],
)
def test_a_malformed_gate_is_refused(fields, message):
    with pytest.raises(ValueError, match=message):
        registry.parse("x", fields)


def test_a_site_registered_twice_is_refused(tmp_path):
    entry = (
        'site = "scmle.a:b"\nerror = "BlockedError"\nclaim = "C1"\n'
        'passes = "p"\nblocks = "b"\noverride = "none"\n'
    )
    path = tmp_path / "gates.toml"
    path.write_text(f"[one]\n{entry}\n[two]\n{entry}", encoding="utf-8")
    with pytest.raises(ValueError, match="twice"):
        registry.load(path)
