"""Fingerprints cover all the product code a measurement loads."""

from benchmarks import sources


def tree(root, files):
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def test_the_closure_follows_imports_transitively(tmp_path):
    tree(
        tmp_path,
        {
            "scmle/__init__.py": "",
            "scmle/a.py": "from scmle import b\nimport os\n",
            "scmle/b.py": "from . import c\n",
            "scmle/c.py": "import scmle.d.e\n",
            "scmle/d/__init__.py": "",
            "scmle/d/e.py": "x = 1\n",
            "scmle/unused.py": "",
            "bench/run.py": "from scmle import a\n",
        },
    )
    assert sources.closure(["bench.run"], tmp_path) == [
        "scmle",
        "scmle.a",
        "scmle.b",
        "scmle.c",
        "scmle.d",
        "scmle.d.e",
    ]


def changed(monkeypatch, tmp_path, name):
    """Makes `locate` see an edited copy of one product module."""
    original = sources.locate(name)
    copy = tmp_path / "copy.py"
    copy.write_text(original.read_text() + "\nEDITED = 1\n")
    real = sources.locate

    def edited(target, root=sources.ROOT):
        return copy if target == name else real(target, root)

    monkeypatch.setattr(sources, "locate", edited)


def test_a_change_anywhere_in_the_closure_changes_the_digest(
    monkeypatch, tmp_path
):
    before = sources.digest(sources.closure)
    changed(monkeypatch, tmp_path, "scmle.assurance.detectors.hashing")
    assert sources.digest(sources.closure) != before


def test_a_module_run_as_main_is_followed_by_its_real_name(monkeypatch):
    import importlib.machinery
    import types

    from benchmarks.qualification import framing_certify

    script = types.ModuleType("__main__")
    script.__spec__ = importlib.machinery.ModuleSpec(
        framing_certify.__name__, None
    )
    assert sources.name(script) == framing_certify.__name__
    assert sources.name(framing_certify) == framing_certify.__name__
