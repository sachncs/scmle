"""Claim C10 evidence: the mutation score of a pre-registered sample."""

import json

from benchmarks import acceptance
from benchmarks.evidence import mutation


def module(name: str, killed: int, decided: int) -> dict:
    counts = dict.fromkeys(mutation.STATUSES, 0)
    counts["killed"] = killed
    counts["survived"] = decided - killed
    return {
        "module": name,
        "tests": [],
        "counts": counts,
        "decided": decided,
        "not_run": 0,
        "files": mutation.fingerprint([name]),
    }


def document(modules: list[dict] | None = None) -> dict:
    """A current document; by default every planned mutant killed."""
    least, planned = mutation.required()
    each = -(-planned // least)
    names = mutation.population()[:least]
    modules = modules or [module(n, each, each) for n in names]
    return {
        "floor": acceptance.load()[mutation.VARIABLE].value,
        "code": mutation.digest(),
        "modules": modules,
    }


def test_the_design_comes_from_the_floor_rule():
    least, planned = mutation.required()
    assert least == 19
    assert planned == 281
    assert mutation.design() == (19, 2 * 15)


def test_the_draw_is_stratified_seeded_and_stable():
    paths = mutation.population()
    sample = mutation.draw(paths, 19)
    assert sample == mutation.draw(paths, 19)
    assert len(set(sample)) == 19
    assert len({mutation.stratum(p) for p in sample}) > 1


def test_the_frame_drops_modules_that_cannot_decide(tmp_path):
    package = tmp_path / "scmle" / "part"
    package.mkdir(parents=True)
    (tmp_path / "tests" / "unit").mkdir(parents=True)
    body = "def f(x):\n    return x + 1\n"
    (package / "__init__.py").write_text("from scmle.part.used import f\n")
    (package / "__main__.py").write_text(body)
    (package / "used.py").write_text(body)
    (package / "lonely.py").write_text(body)
    (package / "flat.py").write_text("X = 1\n")
    (tmp_path / "tests" / "unit" / "test_used.py").write_text(
        "from scmle.part import used\nimport scmle.part.flat\n"
    )
    paths = mutation.population(tmp_path)
    kept, dropped = mutation.frame(paths, tmp_path)
    assert kept == ["scmle/part/used.py"]
    assert dropped == {
        "entry point": ["scmle/part/__main__.py"],
        "no mutants": ["scmle/part/__init__.py", "scmle/part/flat.py"],
        "untested": ["scmle/part/lonely.py"],
    }


def test_a_package_init_takes_the_package_name():
    assert (
        mutation.dotted("scmle/core/stages/__init__.py") == "scmle.core.stages"
    )
    assert mutation.dotted("scmle/cli.py") == "scmle.cli"


def test_allocation_is_proportional_by_largest_remainder():
    taken = mutation.allocate({"a": 6, "b": 3, "c": 1}, 5)
    assert taken == {"a": 3, "b": 2, "c": 0}
    assert sum(mutation.allocate({"a": 1, "b": 1}, 1).values()) == 1


def test_strata_are_top_level_packages():
    assert mutation.stratum("scmle/assurance/ledger/budget.py") == "data"
    assert mutation.stratum("scmle/cli.py") == "root"


def test_imports_see_both_forms():
    source = "import scmle.a.b\nfrom scmle.c import d\nfrom . import e\n"
    found = mutation.imports(source)
    assert {"scmle.a.b", "scmle.c", "scmle.c.d"} <= found
    assert "e" not in found


def test_the_mutant_draw_is_seeded_and_capped():
    names = [f"m{i}" for i in range(100)]
    assert mutation.pick(names, 3, 10) == mutation.pick(names, 3, 10)
    assert len(mutation.pick(names, 3, 10)) == 10
    assert mutation.pick(names[:4], 0, 10) == sorted(names[:4])


def test_only_a_test_failure_is_a_kill():
    assert mutation.classify(1) == "killed"
    assert mutation.classify(0) == "survived"
    assert mutation.classify(33) == "no tests"
    assert mutation.classify(36) == "timeout"
    assert mutation.classify(-11) == "crash"
    assert mutation.classify(None) is None


def test_a_current_document_with_every_mutant_killed_passes():
    assert mutation.problems(document()) == []
    text, verdict = mutation.summary(document())
    assert verdict == "pass"
    assert "lower bound" in text


def test_a_low_score_fails():
    least, planned = mutation.required()
    each = -(-planned // least)
    names = mutation.population()[:least]
    low = document([module(n, each * 3 // 4, each) for n in names])
    assert mutation.problems(low) == []
    assert mutation.summary(low)[1] == "fail"


def test_too_few_modules_or_mutants_is_not_measured():
    names = mutation.population()[:18]
    few = document([module(n, 30, 30) for n in names])
    issues = mutation.problems(few)
    assert any("none can pass below 19" in i for i in issues)
    assert mutation.summary(few)[1] == "not measured"
    names = mutation.population()[:19]
    thin = document([module(n, 10, 10) for n in names])
    assert any("planned" in i for i in mutation.problems(thin))


def test_changed_files_code_or_floor_are_not_measured():
    changed = document()
    changed["modules"][0] = changed["modules"][0] | {"files": "0" * 64}
    assert any("changed since run" in i for i in mutation.problems(changed))
    stale = document() | {"code": "0" * 64}
    assert "mutation: measured with other code" in mutation.problems(stale)
    moved = document() | {"floor": 0.5}
    assert any("floor" in i for i in mutation.problems(moved))


def test_nothing_stored_is_not_measured(tmp_path, monkeypatch):
    monkeypatch.setattr(mutation, "PATH", tmp_path / "mutation.json")
    assert mutation.summary() == ("no measurement", "not measured")
    assert mutation.problems() == ["mutation.json is missing"]
    assert mutation.main(["--check"]) == 1


def test_check_reads_the_stored_document(tmp_path, monkeypatch, capsys):
    path = tmp_path / "mutation.json"
    path.write_text(json.dumps(document()), encoding="utf-8")
    monkeypatch.setattr(mutation, "PATH", path)
    assert mutation.main(["--check"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_a_kill_set_that_reaches_nothing_counts_as_no_tests():
    assert mutation.idle(
        "Stopping early, because we could not find any "
        "test case for any mutant."
    )
    skipped = "7 skipped in 0.02s\nFAILED: Unable to force test failures"
    assert mutation.idle(skipped)
    broken = "1 failed, 7 skipped in 0.02s\nUnable to force test failures"
    assert not mutation.idle(broken)
