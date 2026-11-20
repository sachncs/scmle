"""Statements of `docs/preregistration` checked against the code.

Each test names the registration lines it enforces. A registration states
the analysis a claim will be decided by; where a part of it is not built
yet, the test checks that the missing part keeps its clause from passing
rather than letting the claim pass without it.
"""

import ast
import collections
import json
import math
import pathlib
import subprocess

import numpy
import pandas
import pytest

from benchmarks import acceptance
from benchmarks import scorecard
from benchmarks.evidence import budget
from benchmarks.evidence import coverage
from benchmarks.evidence import custody
from benchmarks.evidence import extensibility
from benchmarks.evidence import guarantees
from benchmarks.evidence import mutation
from benchmarks.evidence import reading
from benchmarks.evidence import refusal
from benchmarks.evidence import reproducibility
from benchmarks.evidence import snippets
from benchmarks.sota import scmle_runner
from scmle.contracts import errors
import scmle.evaluation.coverage

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "benchmarks" / "evidence"
# The commit that added the registrations; its parent holds the results
# scored before them.
REGISTERED = "e2af7e58"


@pytest.fixture(scope="module")
def claims():
    """Scorecard rows by claim number, such as `C10`."""
    return {row.claim.split()[0]: row for row in scorecard.rows()}


def clause(row, words):
    """The one clause of a row whose criterion holds `words`."""
    (found,) = [c for c in row.clauses if words in c.criterion]
    return found


def stored(name):
    """A stored evidence document of `benchmarks/evidence`."""
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def seeds(module):
    """The first argument of every `default_rng` call in a module."""
    tree = ast.parse(pathlib.Path(module.__file__).read_text("utf-8"))
    return [
        node.args[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "default_rng"
    ]


def counts(**given):
    """A mutmut module record with the given status counts."""
    return {
        "counts": dict.fromkeys(mutation.STATUSES, 0) | given,
        "not_run": 0,
    }


def test_mutation_is_a_floor_at_alpha_and_no_crash_is_a_kill():
    # Enforces docs/preregistration/c10.md:38.
    variables = acceptance.load()
    alpha = variables["alpha"].value
    floor = mutation.rule()
    assert floor == variables["mutation_score_min"].rule
    assert isinstance(floor, acceptance.Floor)
    assert variables["mutation_score_min"].value == 0.85
    lower = scmle.evaluation.coverage.clopper_pearson(50, 59, 1 - 2 * alpha)
    assert mutation.bound(50, 59) == pytest.approx(float(lower[0]))
    # No exit code is classified as a correct refusal: a crash stays a
    # crash and the pooled score counts it as not killed.
    assert mutation.classify(-11) == "crash"
    assert mutation.classify(2) == "crash"
    assert "refusal" not in mutation.STATUSES
    pooled = mutation.pool([counts(killed=1, crash=1)])
    assert (pooled["killed"], pooled["decided"]) == (1, 2)
    assert pooled["score"] == 0.5


def test_each_registered_c10_failure_fails_or_holds_back_the_row(
    claims, monkeypatch, tmp_path
):
    # Enforces docs/preregistration/c10.md:61.
    variables = acceptance.load()
    alpha = variables["alpha"].value
    limit = int(variables["change_cost_files_max"].value)
    assert limit == 3
    point = {
        "works": True,
        "needed": {"a.py": True, "b.py": True, "c.py": True, "d.py": True},
        "tests": {"passed": True},
        "files": ["a.py", "b.py", "c.py", "d.py"],
    }
    assert extensibility.decide(point, limit)[0] == "fail"
    point["files"] = point["files"][:3]
    assert extensibility.decide(point, limit) == ("pass", [])
    # An item over budget fails the size clause.
    counted = json.loads(scorecard.STRUCTURE.read_text(encoding="utf-8"))
    over = tmp_path / "structure.json"
    over.write_text(json.dumps(counted | {"functions_over": 1}), "utf-8")
    monkeypatch.setattr(scorecard, "STRUCTURE", over)
    sizes = clause(scorecard.Row("C10", tuple(scorecard.structure())), "lines")
    assert sizes.verdict == "fail"
    # A module below the floor, or under the minimum, cannot pass.
    floor = mutation.rule()
    least = floor.minimum(alpha)
    assert not floor.passes(least - 1, least - 1, alpha)
    assert not floor.passes(80, 100, alpha)
    # An unclassified crash is a decided mutant that is not killed.
    assert mutation.pool([counts(crash=1)])["score"] == 0.0
    # Contract exceptions, tests that cannot fail and CI time: until each
    # is measured its clause holds the row back, and a failed clause fails
    # the row.
    row = claims["C10"]
    for words in ("import-layer", "cannot fail", "pull-request"):
        assert clause(row, words).verdict in ("fail", scorecard.UNMEASURED)
    assert row.verdict != "pass"
    failed = scorecard.Clause("x", "1", "e", "fail")
    passed = scorecard.Clause("y", "1", "e", "pass")
    assert scorecard.Row("C0", (passed, failed)).verdict == "fail"
    assert variables["ci_minutes_max"].value == 15


def test_comprehension_is_a_registered_floor_with_no_reading_yet(claims):
    # Enforces docs/preregistration/c11.md:16.
    variables = acceptance.load()
    assert isinstance(variables["comprehension_min"].rule, acceptance.Floor)
    assert variables["comprehension_min"].value == 0.9
    assert variables["non_specialists"].value == 5
    answered = clause(claims["C11"], "correct on reading the result")
    assert answered.verdict == scorecard.UNMEASURED
    assert answered.evidence == ""
    assert claims["C11"].verdict == scorecard.UNMEASURED


def test_the_c12_statistics_are_counted_as_registered(claims, monkeypatch):
    # Enforces docs/preregistration/c12.md:29.
    found = [{"tests": ["t.py::test_a"]}, {"tests": []}, {"tests": []}]
    assert guarantees.tally(found) == {
        "guarantees": 3,
        "linked": 1,
        "unlinked": 2,
    }
    monkeypatch.setattr(guarantees, "problems", lambda document: [])
    document = {"counts": guarantees.tally(found)}
    value, verdict = guarantees.summary(document)
    assert value.startswith("1/3 documented guarantees linked")
    assert verdict == "fail"
    records = [{"status": s} for s in ("ran", "ran", "failed", "output")]
    ran = {"counts": snippets.tally(records)}
    assert snippets.summary(ran)[0].startswith("2/3 snippets run")
    # Planted statements are not counted yet, and the link audit has no
    # variable: it is reported and decides nothing.
    row = claims["C12"]
    assert clause(row, "planted").verdict == scorecard.UNMEASURED
    assert len(row.clauses) == 3
    assert not [name for name in acceptance.load() if "audit" in name]


def test_each_registered_c12_failure_fails_or_holds_back_the_row(
    claims, monkeypatch
):
    # Enforces docs/preregistration/c12.md:51.
    monkeypatch.setattr(guarantees, "problems", lambda document: [])
    # An unlinked guarantee, and one citing a test that does not exist.
    text = "It never fails (`tests/a.py::test_gone`). It always logs.\n"
    found = guarantees.extract(text, "docs/SAMPLE.md")
    for item in found:
        item |= {"tests": [], "missing": item["cited"]}
    assert [g["missing"] for g in found] == [["tests/a.py::test_gone"], []]
    document = {"counts": guarantees.tally(found)}
    assert guarantees.summary(document)[1] == "fail"
    # A snippet that fails in a fresh process.
    failed = {"counts": snippets.tally([{"status": "failed"}])}
    assert snippets.summary(failed)[1] == "fail"
    # Planted statements and the link audit are not measured, so the claim
    # cannot pass.
    assert clause(claims["C12"], "planted").verdict == scorecard.UNMEASURED
    assert claims["C12"].verdict != "pass"


def test_performance_targets_are_the_service_and_response_variables(claims):
    # Enforces docs/preregistration/c13.md:49.
    variables = acceptance.load()
    targets = {
        name
        for name, variable in variables.items()
        if variable.kind in ("service", "response")
    }
    assert targets == {
        "ingest_overhead_max",
        "memory_multiplier_max",
        "first_feedback_seconds_max",
        "progress_seconds_max",
        "profile_seconds_max",
        "batch_throughput_fraction_min",
        "extras_cost_max",
        "rerun_fraction_max",
    }
    # No upper-bound rule exists yet for any target, so the stage clause
    # cannot be decided and holds the claim back.
    # Enforces docs/preregistration/c13.md:85-88.
    assert all(variables[name].rule is None for name in targets)
    assert variables["perf_repeats"].value == 5
    stages = clause(claims["C13"], "every stage's speed and memory ratio")
    assert stages.verdict == scorecard.UNMEASURED
    assert claims["C13"].verdict != "pass"


def precision(records):
    """The precision counts `refusal.tally` makes of some records."""
    return refusal.tally(records, refusal.catalog())["precision"]


def tasks():
    """A real task of the deciding split and a planted task."""
    placed = refusal.splits()
    catalog = refusal.catalog()
    real = sorted(
        name
        for name, task in catalog.items()
        if task.group == "real" and placed.get(name) == refusal.DECIDING
    )
    planted = sorted(n for n, t in catalog.items() if t.group == "planted")
    return real[0], catalog[planted[0]]


def run(name, outcome, gate=""):
    """One run record."""
    return {"name": name, "outcome": outcome, "gate": gate}


def test_precision_counts_every_refusal_in_envelope_and_planted():
    # Enforces docs/preregistration/c14.md:37.
    real, planted = tasks()
    records = [
        run(real, "refused", planted.expected[0]),
        run(planted.name, "refused", planted.expected[0]),
        run(real, "model"),
    ]
    found = precision(records)
    assert (found["refusals"], found["justified"]) == (2, 1)


def test_a_wrong_gate_or_no_defect_is_an_unjustified_refusal():
    # Enforces docs/preregistration/c14.md:67.
    real, planted = tasks()
    wrong = run(planted.name, "refused", "no.such.gate")
    assert precision([wrong]) | {"rubric": ""} == {
        "refusals": 1,
        "justified": 0,
        "rubric": "",
    }
    clean = run(real, "refused", planted.expected[0])
    assert precision([clean])["justified"] == 0
    right = run(planted.name, "refused", planted.expected[0])
    assert precision([right])["justified"] == 1


def test_no_refusal_is_validator_reviewed_before_the_split_opens():
    # Enforces docs/preregistration/c14.md:72.
    assert refusal.DECIDING == "validation"
    placed = refusal.splits()
    sealed = sorted(
        name
        for name, task in refusal.catalog().items()
        if task.group == "real" and placed.get(name) == "acceptance"
    )
    assert sealed
    counted = refusal.tally([run(sealed[0], "refused")], refusal.catalog())
    assert counted["real"]["deciding"] == {"trials": 0, "hits": 0}
    assert "not validator-reviewed" in counted["precision"]["rubric"]
    assert (
        "the sealed acceptance split and the validator's review of each "
        "refusal against a written rubric (plan section 6)"
    ) in stored("refusal.json")["not_measured"]


def test_the_real_tables_are_split_by_custody_and_all_are_run():
    # Enforces docs/preregistration/c14.md:81-87.
    assert (refusal.BUDGET, refusal.SECONDS) == (30, 3000)
    placed = refusal.splits()
    catalog = refusal.catalog()
    found = collections.Counter(
        (task.group, placed.get(name)) for name, task in catalog.items()
    )
    real = {s: n for (g, s), n in found.items() if g == "real"}
    assert real == {
        "development": 40,
        "validation": 53,
        "acceptance": 54,
        None: 7,
    }
    assert found[("simulated", "acceptance")] == 22
    assert refusal.DECIDING == "validation"
    # The stored run holds outcomes of acceptance tasks, so that split is
    # not sealed.
    names = [r["name"] for r in stored("refusal.json")["records"]]
    assert [n for n in names if placed.get(n) == "acceptance"]


def history(path):
    """A file as it was just before the registrations were added."""
    shallow = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    if shallow.stdout.strip() == "true":
        pytest.skip("corpus 'git-history' is not available locally")
    shown = subprocess.run(
        ["git", "show", f"{REGISTERED}^:{path}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(shown.stdout)


def test_the_c14_result_before_registration_decides_nothing():
    # Enforces docs/preregistration/c14.md:93.
    earlier = history("benchmarks/evidence/refusal.json")
    real = earlier["counts"]["real"]
    assert (real["refusals"], real["trials"], real["failures"]) == (0, 65, 1)
    assert "deciding" not in real
    # Its code is not the code measured now, so it cannot be the result
    # the scorecard reads.
    assert earlier["code_hash"] != stored("refusal.json")["code_hash"]
    with pytest.raises(errors.InputError, match="no deciding counts"):
        refusal.summary(earlier)


def test_coverage_measures_three_families_and_names_the_other_two():
    # Enforces docs/preregistration/c3.md:19.
    assert coverage.FAMILIES == (
        "binary_rows",
        "regression_rows",
        "binary_groups",
    )
    assert "multiclass log loss" in coverage.NOT_MEASURED
    assert "temporal (block) resampling" in coverage.NOT_MEASURED
    assert (coverage.TRAIN_ROWS, coverage.TEST_ROWS) == (400, 300)
    assert coverage.FRESH_ROWS == 100_000


def test_calibration_and_single_spend_are_policy_clauses(claims):
    # Enforces docs/preregistration/c3.md:53.
    row = claims["C3"]
    spend = clause(row, "spent once per analysis")
    assert spend.verdict == scorecard.UNMEASURED
    assert spend.evidence == ""
    calibrated = clause(row, "simulation-based calibration")
    assert calibrated.evidence == "benchmarks/evidence/calibration.json"
    statistical = {
        name
        for name, variable in acceptance.load().items()
        if variable.rule is not None
    }
    assert not {n for n in statistical if "calibrat" in n or "spend" in n}
    assert row.verdict != "pass"


def test_the_study_size_cannot_be_computed_before_its_pilot(claims):
    # Enforces docs/preregistration/c4.md:67.
    variables = acceptance.load()
    assert variables["study_pilot_tasks"].value == 3
    assert variables["study_pilot_tasks"].kind == "policy"
    for name in ("alpha", "power", "practitioner_noninferiority_margin"):
        assert variables[name].kind == "policy", name
    (practitioner,) = [
        s for s in stored("custody.json")["sets"] if s["name"] == "practitioner"
    ]
    assert practitioner["items"] == {}
    assert practitioner["needs"] == {}
    study = clause(claims["C4"], "practitioner study")
    assert study.verdict == scorecard.UNMEASURED
    assert stored("custody.json")["seed"] == custody.SEED


def test_the_awkward_corpus_holds_each_registered_kind_of_file():
    # Enforces docs/preregistration/c5.md:16.
    corpus = reading.corpus()
    groups = {case.group for case in corpus}
    assert {
        "encoding",
        "delimiter",
        "quoting",
        "mixed types",
        "empty columns",
        "duplicate headers",
        "huge numbers",
        "dates",
        "truncated",
    } <= groups
    names = {case.name for case in corpus}
    assert {"leading_zeros", "beyond_int64", "uint64_max"} <= names
    for case in corpus:
        # An expected table has its columns and rows; an expected refusal
        # has no columns.
        assert case.columns is None or case.rows is not None, case.name
    assert {case.columns is None for case in corpus} == {True, False}
    # The other registered formats are named as not measured, not dropped.
    assert reading.NOT_MEASURED == (
        "Excel, Parquet and Feather files",
        (
            "the streaming reader (streaming.Csv), database connectors other "
            "than SQLite"
        ),
    )


def copies():
    """Forty entities of five identical rows each."""
    rng = numpy.random.default_rng(0)
    base = pandas.DataFrame(
        {
            "entity": numpy.arange(40),
            "x": rng.normal(size=40),
            "y": numpy.tile([0, 1], 20),
        }
    )
    return base.loc[base.index.repeat(5)].reset_index(drop=True)


def test_identical_rows_may_cross_but_copies_of_an_entity_never_do():
    # Enforces docs/preregistration/c5.md:84.
    frame = copies()
    free = reading.hygiene(frame, "y", {})
    assert free["refused"] == "" and free["splits"] > 1
    assert free["duplicate"] > 0
    assert not reading.crossed(free)
    grouped = reading.hygiene(frame, "y", {"group": "entity"})
    assert grouped["refused"] == "" and grouped["splits"] > 1
    assert (grouped["entity"], grouped["copy"]) == (0, 0)
    # A copy of one declared entity crossing is a violation, except where a
    # temporal split keeps a known entity's past for training.
    crossing = dict.fromkeys(reading.CROSSINGS, 0) | {"error": "", "copy": 1}
    assert reading.crossed(crossing | {"strategy": "group", "entities": ""})
    assert not reading.crossed(
        crossing | {"strategy": "temporal", "entities": "known"}
    )
    assert reading.crossed(
        crossing | {"strategy": "temporal", "entities": "unseen"}
    )


def test_the_c6_data_today_come_from_one_development_seed():
    # Enforces docs/preregistration/c6.md:23.
    assert (reproducibility.SEED, reproducibility.ROWS) == (7, 600)
    drawn = seeds(reproducibility)
    assert drawn
    assert all(isinstance(s, ast.Name) and s.id == "SEED" for s in drawn)
    measured = stored("reproducibility.json")["platforms"].values()
    assert measured
    assert {(p["seed"], p["rows"]) for p in measured} == {(7, 600)}


def compliance(practice, real):
    """A budget document with the given practice and real counts."""
    good, runs = real
    return {
        "settings": budget.settings(),
        "practicerule": budget.OLDRULE,
        "compliance": {"hits": practice[0], "trials": practice[1]},
        "resume": {"identical": 3, "trials": 3},
        "real": {"hits": good, "trials": runs, "refused": 0, "runs": []},
    }


def test_the_seeded_runs_are_practice_and_decide_nothing():
    # Enforces docs/preregistration/c7.md:29.
    # Enforces docs/preregistration/c7.md:86-88.
    assert budget.RUNS_MAX == 400
    runs = stored("budget.json")["compliance"]["runs"]
    assert [r["seed"] for r in runs] == list(range(budget.RUNS_MAX))
    assert stored("budget.json")["practicerule"] == budget.OLDRULE
    least = acceptance.load()[budget.VARIABLE].rule.minimum(
        acceptance.load()["alpha"].value
    )
    # Every practice run within budget, no real run: nothing is decided.
    value, verdict = budget.summary(compliance((400, 400), (0, 0)))
    assert verdict == scorecard.UNMEASURED
    assert "practice project (old rule, not deciding): 400 of 400" in value
    # Every practice run over, every real run within: the row passes.
    passing = compliance((0, 400), (least, least))
    assert budget.summary(passing)[1] == "pass"


def test_the_real_family_is_unsplit_with_size_scaled_budgets():
    # Enforces docs/preregistration/c7.md:74-85.
    found = budget.tables()
    assert {source for source, ident in found} == {"quality", "refusal"}
    assert [i for s, i in found if s == "refusal"] == list(refusal.EXTRA)
    assert [budget.allot(c) for c in (600, 10_000, 10_001, 100_001)] == [
        30,
        30,
        60,
        120,
    ]
    assert budget.seeded("quality", 7, 0) == scmle_runner.seed_for(7, 0, 0)
    assert budget.seeded("refusal", 7, 0) == scmle_runner.seed_for(7, 0, 0)
    source = pathlib.Path(budget.__file__).read_text(encoding="utf-8")
    assert "custody" not in source
    assert "validation" not in source
    least = acceptance.load()[budget.VARIABLE].rule.minimum(
        acceptance.load()["alpha"].value
    )
    assert least == 299
    assert math.ceil(least / len(found)) > 1
    assert math.ceil(least / len(found)) <= budget.PASSES
    assert budget.REAL_SECONDS == 3600.0


def test_each_registered_c9_count_is_a_clause_not_yet_counted(claims):
    # Enforces docs/preregistration/c9.md:34.
    row = claims["C9"]
    for words in (
        "unauthenticated deserialisation",
        "no raw rows, categories or secrets in artifacts",
        "a forget operation",
        "OpenSSF Scorecard",
        "known high vulnerability",
    ):
        found = clause(row, words)
        assert found.verdict == scorecard.UNMEASURED, words
    assert acceptance.load()["openssf_score_min"].value == 8
    assert row.verdict == scorecard.UNMEASURED
