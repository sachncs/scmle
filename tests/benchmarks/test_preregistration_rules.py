"""The registered analyses of `docs/preregistration` hold in the code.

Each test names the registration lines it enforces. The sample-size tables
themselves are recomputed by `test_preregistration`; these tests check the
statements around them: the decision rules, the planned counts, the index
of claims, and the seed streams kept back for validation.
"""

import ast
import math
import pathlib
import re
import subprocess

import pytest

from benchmarks import acceptance
from benchmarks import scorecard
from benchmarks.evidence import coverage
from benchmarks.evidence import faults
from benchmarks.evidence import leakage
from benchmarks.evidence import quality
from benchmarks.evidence import reading
from benchmarks.evidence import refusal
from benchmarks.sota import split
from scmle.contracts import errors
from scmle.data.readers import tabular
from tests.benchmarks import test_evidence_faults
from tests.benchmarks import test_evidence_quality
from tests.benchmarks import test_evidence_reading

ROOT = pathlib.Path(__file__).resolve().parents[2]
FOLDER = ROOT / "docs" / "preregistration"
SOURCES = ("benchmarks", "scmle", "scripts", "tests")
# The four Clopper-Pearson rows decided on the acceptance split in stage D.
STAGE = {
    "leak_recall_min": 0.97,
    "false_block_max": 0.01,
    "refusal_rate_max": 0.01,
    "refusal_precision_min": 0.97,
}
INDEX = re.compile(
    r"^\| (C\d+ [^|]+?) \| \[(c\d+\.md)\]\(c\d+\.md\) \| ([^|]+?) \| "
    r"([^|]+?) \| [^|]+ \|$",
    re.MULTILINE,
)
ROW = re.compile(
    r"^\| `(\w+)` \| (?:floor|ceiling) \| ([\d.]+) \| ([\d.]+) \| \d+ \| "
    r"(\d+) \|$",
    re.MULTILINE,
)
STREAM = re.compile(r"""["'][\w.]+\.(?:validation|acceptance)["']""")


def registered(claim: str) -> str:
    """The text of one registration."""
    return (FOLDER / f"{claim}.md").read_text(encoding="utf-8")


def claims() -> dict[str, scorecard.Row]:
    """Scorecard rows by claim number, such as `C2`."""
    return {row.claim.split()[0]: row for row in scorecard.rows()}


def criteria(claim: str) -> list[str]:
    """The criterion of each clause the scorecard decides a claim on."""
    return [clause.criterion for clause in claims()[claim].clauses]


def test_registrations_are_not_kept_under_plan():
    # Enforces docs/preregistration/README.md:12.
    tracked = subprocess.run(
        ["git", "ls-files", "plan"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert tracked.stdout == ""
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert not [line for line in ignored if line.strip().strip("/") == "plan"]
    if (ROOT / "plan").exists():
        subprocess.run(
            ["git", "check-ignore", "-q", "plan/"], cwd=ROOT, check=True
        )
    assert len(list(FOLDER.glob("c*.md"))) == 14


def test_the_index_rows_match_the_scorecard_and_the_files():
    # Enforces docs/preregistration/README.md:21.
    # Enforces docs/preregistration/README.md:27.
    # Enforces docs/preregistration/README.md:33.
    rows = INDEX.findall(registered("README"))
    found = claims()
    assert len(rows) == len(found) == 14
    for title, name, decided, modules in rows:
        row = found[title.split()[0]]
        assert row.claim == title
        assert registered(name.removesuffix(".md")).startswith(f"# {title}: ")
        section = registered(name.removesuffix(".md")).split("## Split")[1]
        first = section.strip().split(".")[0].split(",")[0].lower()
        if first.startswith("none applies"):
            assert decided.startswith("stage D")
        else:
            assert decided.split(";")[0] == first.split()[0], title
        named = set(re.findall(r"`([\w/]+\.py)`", modules))
        assert all((ROOT / path).exists() for path in named), title
        stored = {
            clause.evidence.removesuffix(".json") + ".py"
            for clause in row.clauses
            if clause.evidence
        }
        assert stored <= named, title
        assert bool(named) == (modules != "none"), title


def test_no_validation_or_acceptance_stream_is_drawn():
    # Enforces docs/preregistration/README.md:44.
    # Enforces docs/preregistration/c3.md:38.
    drawn = [
        f"{path.relative_to(ROOT)}: {match.group()}"
        for top in SOURCES
        for path in (ROOT / top).rglob("*.py")
        for match in STREAM.finditer(path.read_text(encoding="utf-8"))
    ]
    assert drawn == []
    tree = ast.parse(pathlib.Path(coverage.__file__).read_text("utf-8"))
    seeds = [
        node.args[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "default_rng"
    ]
    assert seeds
    assert all(isinstance(s, ast.List) and len(s.elts) == 2 for s in seeds)


def test_holm_sizes_the_four_stage_d_rows():
    # Enforces docs/preregistration/README.md:62.
    # Enforces docs/preregistration/c2.md:49.
    variables = acceptance.load()
    alpha = variables["alpha"].value
    power = variables["power"].value
    assert alpha / 4 == pytest.approx(0.0125)
    adjusted = {
        name
        for claim in ("c2", "c14")
        for name, truth, level, planned in ROW.findall(registered(claim))
        if float(level) == pytest.approx(alpha / 4)
    }
    assert adjusted == set(STAGE)
    for name, truth in STAGE.items():
        rule = variables[name].rule
        assert rule is not None
        unadjusted = rule.plan(truth, alpha, power)
        assert 0.25 <= rule.chance(truth, unadjusted, alpha / 4) <= 0.35


@pytest.mark.parametrize("name", [*STAGE, "mutation_score_min"])
def test_fewer_trials_than_the_minimum_cannot_pass(name):
    # Enforces docs/preregistration/c2.md:50.
    # Enforces docs/preregistration/c14.md:45.
    # Enforces docs/preregistration/c10.md:54.
    variables = acceptance.load()
    rule = variables[name].rule
    alpha = variables["alpha"].value
    assert rule is not None
    for level in (alpha, alpha / 4):
        least = rule.minimum(level)
        best = (
            (lambda n: n)
            if isinstance(rule, acceptance.Floor)
            else (lambda n: 0)
        )
        assert not rule.passes(best(least - 1), least - 1, level)
        assert rule.passes(best(least), least, level)
    if name == "mutation_score_min":
        assert rule.minimum(alpha) == 19


def test_the_registered_true_rates_size_the_tables():
    # Enforces docs/preregistration/c2.md:54.
    # Enforces docs/preregistration/c14.md:50.
    for claim in ("c2", "c14"):
        rows = ROW.findall(registered(claim))
        assert rows
        for row in rows:
            assert float(row[1]) == STAGE[row[0]], (claim, row[0])


def test_the_refusal_split_sizes_follow_from_the_plan():
    # Enforces docs/preregistration/c14.md:59.
    variables = acceptance.load()
    alpha = variables["alpha"].value
    power = variables["power"].value
    rate = variables["refusal_rate_max"].rule
    precision = variables["refusal_precision_min"].rule
    assert rate is not None and precision is not None
    assert rate.plan(0.01, alpha / 4, power) == 192
    assert precision.plan(0.97, alpha / 4, power) == 124
    assert rate.plan(0.01, alpha, power) == 124
    assert precision.plan(0.97, alpha, power) == 76


def test_the_tabarena_holdout_is_short_of_multiclass_tasks():
    # Enforces docs/preregistration/c4.md:62.
    tasks = split.load_tasks()
    types = {int(t["task_id"]): t["problem_type"] for t in tasks}
    held = [types[i] for i in split.assign(tasks)["holdout"]]
    counts = {kind: held.count(kind) for kind in set(held)}
    assert counts == {"binary": 12, "regression": 5, "multiclass": 3}
    alpha = acceptance.load()["alpha"].value
    assert quality.minimum() == math.ceil(math.log(alpha) / math.log(0.5))
    assert math.ceil(math.log(alpha / 3) / math.log(0.5)) == 6
    gaps = dict.fromkeys(quality.TASKS, [0.0] * 5)
    gaps["multiclass"] = [0.0] * counts["multiclass"]
    found = test_evidence_quality.document(gaps)
    assert quality.summary(found)[1] == "not measured"


def test_the_scorecard_decides_each_registered_criterion():
    # Enforces docs/preregistration/c1.md:8.
    # Enforces docs/preregistration/c2.md:9.
    # Enforces docs/preregistration/c5.md:8.
    # Enforces docs/preregistration/c7.md:8.
    # Enforces docs/preregistration/c10.md:8.
    # Enforces docs/preregistration/c12.md:8.
    # Enforces docs/preregistration/c14.md:9.
    value = scorecard.value
    expected = {
        "C1": [
            (
                "every gate registered with pass, block and override paths; "
                "every injected failure blocks or is a recorded override "
                f"(share {value('gate_fault_block_min')})"
            ),
            "an unregistered gate fails CI",
            "every certificate read by the code it justifies",
        ],
        "C2": [
            (
                f"recall lower bound >= {value('leak_recall_min')} on real leaks, "
                "entity-level leaks included; false-block upper bound <= "
                f"{value('false_block_max')} on clean real datasets"
            )
        ],
        "C5": [
            (
                "awkward files correct or refused; point-in-time features equal "
                f"a naive reference (tolerance {value('reference_rel_tol')}) on "
                "generated and real relational data"
            ),
            (
                "no declared entity and no tied timestamp on both sides of "
                "any split; duplicate rows that cross are reported, not "
                "counted, unless they belong to one declared entity"
            ),
        ],
        "C7": [
            "service objectives under load",
            "a 24-hour soak with flat memory",
            "a hung request cannot wedge a replica",
            "zero reset connections on SIGTERM",
            (
                "kill -9 at every stage boundary, journal append and lease "
                "operation resumes identically"
            ),
            "lease and journal protocols model-checked",
        ],
        "C10": [
            (
                "a new model family, task type, data kind, metric or reader "
                f"touches <= {value('change_cost_files_max')} files outside its "
                "package"
            ),
            "import-layer contract with zero exceptions",
            (
                f"no module, class or function over {value('module_lines_max')}/"
                f"{value('class_lines_max')}/{value('function_lines_max')} lines"
            ),
            f"mutation score >= {value('mutation_score_min')} per gated module",
            "no test that cannot fail",
            f"pull-request checks <= {value('ci_minutes_max')} minutes",
        ],
        "C12": [
            "every guarantee in docs and site linked to an enforcing test",
            "every snippet and command runs in a fresh process",
            "a planted false statement fails CI",
        ],
        "C14": [
            (
                f"refusal-rate upper bound <= {value('refusal_rate_max')}; "
                "justified share lower bound >= "
                f"{value('refusal_precision_min')}"
            )
        ],
    }
    for claim, clauses in expected.items():
        decided = criteria(claim)
        assert all(c in " ; ".join(decided) for c in clauses), claim
    budget = criteria("C7")[0]
    assert f">= {value('budget_compliance_min')}" in budget
    assert reading.WORLDS >= 1000 and reading.REAL_DATASETS >= 5
    assert isinstance(
        acceptance.load()["leak_recall_min"].rule, acceptance.Floor
    )
    assert isinstance(
        acceptance.load()["refusal_rate_max"].rule, acceptance.Ceiling
    )


def test_the_policy_limits_are_the_registered_values():
    # Enforces docs/preregistration/c10.md:42.
    variables = acceptance.load()
    limits = {
        "change_cost_files_max": 3,
        "module_lines_max": 800,
        "class_lines_max": 300,
        "function_lines_max": 80,
        "ci_minutes_max": 15,
    }
    for name, limit in limits.items():
        assert variables[name].kind == "policy", name
        assert variables[name].value == limit, name
        assert variables[name].rule is None, name


def test_unmeasured_clauses_keep_their_claims_from_passing():
    # Enforces docs/preregistration/c10.md:78.
    # Enforces docs/preregistration/c12.md:67.
    # Enforces docs/preregistration/c12.md:36.
    found = claims()
    pending = {
        "C10": ("touches", "import-layer", "cannot fail", "pull-request"),
        "C12": ("planted",),
    }
    for claim, words in pending.items():
        for word in words:
            clause = next(
                c for c in found[claim].clauses if word in c.criterion
            )
            assert clause.value == scorecard.UNMEASURED, (claim, word)
            assert clause.evidence == "", (claim, word)
        assert found[claim].verdict != "pass", claim
    passed = scorecard.Clause("x", "1", "e", "pass")
    row = scorecard.Row("C0", (passed, scorecard.pending("y")))
    assert row.verdict == scorecard.UNMEASURED


def test_every_unsatisfied_fault_unit_counts_against_the_claim():
    # Enforces docs/preregistration/c1.md:53.
    rows = test_evidence_faults.real(test_evidence_faults.records())
    first = next(i for i, r in enumerate(rows) if r["path"] == "fault")
    whole = faults.tally(rows)
    assert whole["satisfied"] == whole["units"]
    for outcome in ("silent", "failed", "not injected"):
        changed = [*rows]
        changed[first] = {**rows[first], "outcome": outcome}
        counted = faults.tally(changed)
        assert counted["units"] == whole["units"], outcome
        assert counted["satisfied"] == whole["units"] - 1, outcome
    certificate = next(
        c
        for c in claims()["C1"].clauses
        if c.criterion == "every certificate read by the code it justifies"
    )
    assert certificate.verdict == scorecard.UNMEASURED


def test_only_a_block_by_the_screen_is_a_catch():
    # Enforces docs/preregistration/c2.md:40.
    endings = ("blocked", "refused", "failed", "passed")
    records = [{"corpus": "leak", "outcome": e} for e in endings]
    assert [leakage.caught(r) for r in records] == [True, False, False, False]
    assert leakage.tally(records, "leak", leakage.caught)["hits"] == 1


def test_planted_tasks_cover_the_registered_defects():
    # Enforces docs/preregistration/c14.md:18.
    planted = [t for t in refusal.catalog().values() if t.group == "planted"]
    defects = {t.name.split("_")[1] for t in planted}
    assert {
        "constant",
        "identifier",
        "unlabeled",
        "singleton",
        "leak",
        "events",
        "arm",
        "labels",
        "queries",
    } <= defects
    assert min(int(t.name.rsplit("_", 1)[1]) for t in planted) <= 12
    assert all(t.expected for t in planted)


def test_a_refused_readable_file_is_reported_and_not_a_failure(monkeypatch):
    # Enforces docs/preregistration/c5.md:42.
    # Enforces docs/preregistration/c5.md:66.
    found = test_evidence_reading.document()
    found["corpus"] |= {
        "correct": found["corpus"]["correct"] - 1,
        "refused": 1,
        "over_refused": 1,
    }
    text, verdict = reading.summary(found)
    assert verdict != "fail"
    cases = found["corpus"]["cases"]
    assert f"awkward files {cases}: {cases - 1} correct, 1 refused" in text
    assert "(1 readable), 0 silently wrong" in text
    assert "simulated worlds equal the naive reference" in text
    assert "real relational datasets 5 of 5 equal" in text
    assert "0 cross (table runs with a split sharing" in text


def test_a_crash_other_than_a_data_error_is_not_a_refusal(
    monkeypatch, tmp_path
):
    # Enforces docs/preregistration/c5.md:61.
    case = reading.text("ok", "g", "a\n1\n", ("a",), ((1,),))

    def crash(path):
        raise ValueError("reader crashed")

    monkeypatch.setattr(tabular, "read", crash)
    with pytest.raises(ValueError, match="reader crashed"):
        reading.read(case, tmp_path, 1e-8)

    def refuse(path):
        raise errors.DataError("unreadable")

    monkeypatch.setattr(tabular, "read", refuse)
    assert reading.read(case, tmp_path, 1e-8)["outcome"] == "refused"
