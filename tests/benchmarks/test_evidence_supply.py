"""C9 supply-chain evidence: corpus, static workflow checks, live checks
through a fake shell, the attack report parser and the verdicts."""

import json
import pathlib
import subprocess

import pytest

from benchmarks.evidence import supply
from scmle.security import leaks

WORKFLOW = supply.WORKFLOW.read_text(encoding="utf-8")
COMMIT = "f7dd8c54c2067bafc12ca7a55595d5ee9b75204a"
UPLOAD = "      - uses: pypa/gh-action-pypi-publish@"
GATE = (
    "          uv run python -m scmle.deploy.vuln_gate audit.json "
    "--policy deploy/vuln-policy.json\n"
)
AUDIT = (
    "          uv run pip-audit --requirement audit-requirements.txt "
    "--no-deps --disable-pip --format json --output audit.json "
    "|| status=$?\n"
)
GATED = "      - name: No known high vulnerability in the locked dependencies\n"


def step(uses):
    """The release workflow with one more publish step, before upload."""
    return WORKFLOW.replace(UPLOAD, f"      - uses: {uses}\n{UPLOAD}")


def test_the_corpus_is_seeded_and_plants_every_kind_of_value():
    made = supply.table("binary-0", "binary")
    again = supply.table("binary-0", "binary")
    frame = made["frame"]
    assert frame.equals(again["frame"])
    assert made["secrets"] == again["secrets"]
    assert list(frame.columns) == [
        "a",
        "b",
        "age",
        "name",
        "ident",
        "note",
        "colour",
        "region",
        "y",
    ]
    assert frame.shape == (supply.ROWS, 9)
    assert frame["name"].nunique() == supply.ROWS
    assert frame["ident"].nunique() == supply.ROWS
    counts = frame["region"].value_counts()
    rare = sorted(counts[counts < leaks.FLOOR].tolist())
    assert rare == [1, 2]
    assert len(made["secrets"]) == supply.ROWS + 1
    assert made["secrets"][-1] == made["key"]
    assert all(
        t in n for t, n in zip(made["secrets"][:-1], frame["note"], strict=True)
    )
    assert set(frame["y"]) == {0, 1}
    assert set(supply.table("multiclass-0", "multiclass")["frame"]["y"]) == {
        0,
        1,
        2,
    }
    assert not supply.table("binary-1", "binary")["frame"].equals(frame)


def test_each_task_has_one_validation_and_one_development_table():
    assert supply.splits() == {
        "binary-0": "validation",
        "binary-1": "development",
        "regression-0": "validation",
        "regression-1": "development",
        "multiclass-0": "validation",
        "multiclass-1": "development",
    }
    other = supply.splits(seed=1)
    for task in supply.TASKS:
        assert sorted(other[f"{task}-{i}"] for i in range(2)) == [
            "development",
            "validation",
        ]


def test_the_release_workflow_passes_every_static_check():
    assert {k: v["ok"] for k, v in supply.workflow(WORKFLOW).items()} == {
        "provenance": True,
        "signing": True,
        "audit": True,
        "pinning": True,
    }


@pytest.mark.parametrize(
    ("change", "failed"),
    [
        (
            lambda t: t.replace(f"@v2.1.0 # {COMMIT}", f"@{COMMIT}"),
            "provenance",
        ),
        (lambda t: t.replace("upload-assets: true", ""), "provenance"),
        (
            lambda t: t.replace(
                "sigstore/gh-action-sigstore-python@790bc6befb9d733738f18d8f895854b453640ec9",
                "sigstore/gh-action-sigstore-python@v3",
            ),
            "signing pinning",
        ),
        (lambda t: t.replace("scmle.deploy.vuln_gate", "true"), "audit"),
        (lambda t: t.replace(GATE, ""), "audit"),
        (
            lambda t: t.replace(GATE, GATE.replace("uv run", "# uv run")),
            "audit",
        ),
        (lambda t: t.replace(AUDIT, ""), "audit"),
        (
            lambda t: t.replace(GATED, GATED + "        if: ${{ false }}\n"),
            "audit",
        ),
        (
            lambda t: t.replace(
                GATED, GATED + "        continue-on-error: true\n"
            ),
            "audit",
        ),
        (
            lambda t: step(
                "slsa-framework/slsa-github-generator/.github/workflows/"
                "generator_generic_slsa3.yml@main"
            ),
            "pinning",
        ),
        (
            lambda t: step(
                "slsa-framework/slsa-github-generator/.github/actions/x@v2.1.0"
            ),
            "pinning",
        ),
        (
            lambda t: t.replace(
                "generator_generic_slsa3.yml@v2.1.0", "x@v2.1.0"
            ),
            "provenance pinning",
        ),
        (
            lambda t: t.replace(
                "needs.publish.outputs.hashes", "needs.other.outputs.hashes"
            ),
            "provenance",
        ),
        (lambda t: t + "\njobs: [\n", "provenance signing audit pinning"),
        (
            lambda t: t.replace(
                "actions/checkout@11d5960a326750d5838078e36cf38b85af677262",
                "actions/checkout@v4",
            ),
            "pinning",
        ),
    ],
)
def test_a_weakened_workflow_fails_its_check(change, failed):
    found = supply.workflow(change(WORKFLOW))
    assert [k for k, v in found.items() if not v["ok"]] == failed.split()


def test_the_steps_are_found_in_the_parse_not_in_comments():
    assert "(scmle.deploy.vuln_gate)" in WORKFLOW
    assert WORKFLOW.index("(scmle.deploy.vuln_gate)") < WORKFLOW.index(GATE)
    assert supply.workflow(WORKFLOW.replace(GATE, ""))["audit"]["ok"] is False
    moved = WORKFLOW.replace(GATE, "").replace(
        "          gh release create", GATE + "          gh release create"
    )
    assert supply.workflow(moved)["audit"]["ok"] is False


def test_an_unpinned_action_is_named():
    text = step("someone/action@main")
    assert supply.workflow(text)["pinning"]["detail"] == (
        "every other action pinned by commit; loose: someone/action@main"
    )


class Shell:
    """A fake `gh`, `git` and `uv` that answers from a table."""

    def __init__(self, answers=None, audit=None):
        self.answers = answers or {}
        self.audit = audit
        self.calls = []

    def __call__(self, command):
        self.calls.append(command)
        key = " ".join(command)
        if command[:2] == ["uv", "export"]:
            pathlib.Path(command[-1]).write_text("x==1\n", encoding="utf-8")
            return ""
        if "pip-audit" in command:
            output = pathlib.Path(command[command.index("--output") + 1])
            output.write_text(json.dumps(self.audit), encoding="utf-8")
            if any(d["vulns"] for d in self.audit["dependencies"]):
                raise subprocess.CalledProcessError(1, command)
            return ""
        if key not in self.answers:
            raise subprocess.CalledProcessError(1, command, stderr="no answer")
        return self.answers[key]


def test_the_generator_tag_must_still_name_the_recorded_commit():
    ask = f"gh api repos/{supply.GENERATOR}/commits/v2.1.0 --jq .sha"
    assert supply.tagged(WORKFLOW, Shell({ask: COMMIT + "\n"})) == {
        "ok": True,
        "detail": f"v2.1.0 is {COMMIT}, recorded {COMMIT}",
    }
    moved = supply.tagged(WORKFLOW, Shell({ask: "0" * 40}))
    assert moved["ok"] is False
    assert supply.tagged("no generator", Shell())["ok"] is False


def rulesets(*sets):
    listed = "gh api repos/o/r/rulesets?targets=tag&includes_parents=true"
    answers = {listed: json.dumps([{"id": i} for i in range(len(sets))])}
    for i, each in enumerate(sets):
        answers[f"gh api repos/o/r/rulesets/{i}"] = json.dumps(each)
    return Shell(answers)


def ruleset(include=("refs/tags/v*",), rules=("deletion", "update"), **extra):
    return {
        "name": "release tags",
        "target": "tag",
        "enforcement": "active",
        "conditions": {"ref_name": {"include": list(include), "exclude": []}},
        "rules": [{"type": r} for r in rules],
        **extra,
    }


def test_a_tag_ruleset_must_forbid_deletion_and_update():
    assert supply.protected("o/r", "v1.2.3", rulesets(ruleset())) == {
        "ok": True,
        "detail": "ruleset release tags",
    }
    assert supply.protected(
        "o/r", "v1.2.3", rulesets(ruleset(include=["~ALL"]))
    )["ok"]
    for weak in (
        ruleset(rules=["deletion"]),
        ruleset(include=["refs/tags/release-*"]),
        ruleset(enforcement="evaluate"),
        ruleset(target="branch"),
    ):
        assert not supply.protected("o/r", "v1.2.3", rulesets(weak))["ok"]
    assert supply.protected("o/r", "v1.2.3", rulesets()) == {
        "ok": False,
        "detail": "no active tag ruleset forbids deleting or moving "
        "refs/tags/v1.2.3 (0 tag rulesets)",
    }
    with pytest.raises(ValueError, match="no GitHub repository"):
        supply.protected("", "v1", Shell())


def report(*vulns):
    return {
        "dependencies": [
            {"name": "pkg", "version": "1.0", "vulns": list(vulns)},
            {"name": "other", "version": "2.0", "vulns": []},
        ]
    }


def test_the_audit_passes_a_clean_lock_and_blocks_any_finding():
    assert supply.audited(Shell(audit=report())) == {
        "ok": True,
        "detail": "2 packages scanned, 0 blocking",
    }
    finding = {"id": "PYSEC-1", "fix_versions": ["1.1"], "aliases": []}
    assert supply.audited(Shell(audit=report(finding))) == {
        "ok": False,
        "detail": "2 packages scanned, 1 blocking (PYSEC-1)",
    }


def test_a_live_check_that_cannot_run_fails_and_says_why():
    found = supply.attempt(lambda: supply.protected("o/r", "v1", Shell()))
    assert found == {"ok": False, "detail": "could not check: no answer"}
    empty = Shell(audit={"dependencies": []})
    found = supply.attempt(lambda: supply.audited(empty))
    assert found["ok"] is False
    assert found["detail"].startswith("gate refused the report")


def test_the_origin_remote_names_the_repository():
    for url in (
        "https://github.com/o/r.git",
        "git@github.com:o/r.git",
        "https://github.com/o/r",
    ):
        shell = Shell({"git remote get-url origin": url + "\n"})
        assert supply.origin(shell) == "o/r"
    assert supply.origin(Shell()) == ""


def test_the_release_clause_measures_static_and_live_checks():
    shell = rulesets()
    shell.answers[
        f"gh api repos/{supply.GENERATOR}/commits/v2.1.0 --jq .sha"
    ] = COMMIT
    shell.audit = report()
    record = supply.Release(shell, "o/r").measure()
    assert record["repository"] == "o/r"
    assert {k: v["ok"] for k, v in record["checks"].items()} == {
        "provenance": True,
        "signing": True,
        "audit": True,
        "pinning": True,
        "generator": True,
        "protection": False,
        "vulnerabilities": True,
    }
    text, verdict = supply.Release().judge(record)
    assert verdict == "fail"
    assert text.startswith("6 of 7 checks pass (protection: no active tag")


JUNIT = """<testsuites><testsuite>
<testcase name="test_every_site_has_an_attack"/>
<testcase name="test_attack_reaches_the_sink[a.b:f]"/>
<testcase name="test_unauthenticated_bytes_never_run[a.b:f-unsigned]"/>
<testcase name="test_unauthenticated_bytes_never_run[a.b:f-wrong-key]"/>
<testcase name="test_attack_reaches_the_sink[a.c:g]"/>
<testcase name="test_unauthenticated_bytes_never_run[a.c:g-unsigned]">
<failure message="ran"/></testcase>
<testcase name="test_unauthenticated_bytes_never_run[a.c:g-wrong-key]">
<skipped/></testcase>
<testcase name="test_attack_reaches_the_sink[a.d:h]">
<failure message="refused before the sink"/></testcase>
<testcase name="test_unauthenticated_bytes_never_run[a.d:h-unsigned]"/>
<testcase name="test_unauthenticated_bytes_never_run[a.d:h-wrong-key]"/>
<testcase name="test_unauthenticated_bytes_never_run[a.e:i-unsigned]"/>
<testcase name="test_unauthenticated_bytes_never_run[a.r:s-no-input]"/>
</testsuite></testsuites>"""


def test_attack_outcomes_are_read_per_site():
    assert supply.outcomes(JUNIT) == {
        "a.b:f": {"passed": 2, "failed": 0},
        "a.c:g": {"passed": 0, "failed": 2},
        "a.d:h": {"passed": 0, "failed": 2},
        "a.e:i": {"passed": 0, "failed": 1},
        "a.r:s": {"passed": 1, "failed": 0},
    }


def site(key, verdict="authenticated"):
    return {"key": key, "line": 1, "sink": "joblib.load", "verdict": verdict}


def test_the_deserialisation_verdict():
    clean = {
        "sites": [site("a:f"), site("b:g", "roundtrip")],
        "attacks": {
            "a:f": {"passed": 2, "failed": 0},
            "b:g": {"passed": 1, "failed": 0},
        },
    }
    assert supply.Deserialisation().judge(clean) == (
        (
            "2 sites: 1 authenticated, 0 forwarded, 1 roundtrip, "
            "0 unauthenticated; attacks refused at 2 of 2 sites"
        ),
        "pass",
    )
    broken = {
        "sites": [site("a:f", "unauthenticated"), site("b:g")],
        "attacks": {
            "a:f": {"passed": 2, "failed": 0},
            "b:g": {"passed": 1, "failed": 0},
        },
    }
    assert supply.Deserialisation().judge(broken) == (
        (
            "2 sites: 1 authenticated, 0 forwarded, 0 roundtrip, "
            "1 unauthenticated; attacks refused at 1 of 2 sites "
            "(open: a:f, b:g)"
        ),
        "fail",
    )


def test_the_deserialisation_clause_measures_the_real_package():
    record = supply.Deserialisation().measure()
    assert len(record["sites"]) == 8
    verdicts = {s["key"]: s["verdict"] for s in record["sites"]}
    assert verdicts["scmle.models.conformance:smoke"] == "roundtrip"
    assert {
        key: tried["passed"] for key, tried in record["attacks"].items()
    } == {
        key: 1 if verdict == "roundtrip" else 2
        for key, verdict in verdicts.items()
    }
    assert all(not t["failed"] for t in record["attacks"].values())
    assert supply.Deserialisation().judge(record)[1] == "pass"


def table(split, found, error=""):
    return {
        "table": f"t-{split}",
        "split": split,
        "error": error,
        "leaks": {
            k: sum(f["count"] for f in found if f["kind"] == k)
            for k in leaks.LEAKS
        },
        "found": found,
    }


def test_the_artifacts_verdict_fails_on_a_validation_leak_only():
    leak = {"path": "run/a.json", "kind": "row", "count": 3}
    record = {"tables": [table("validation", []), table("development", [leak])]}
    assert supply.Artifacts().judge(record) == (
        (
            "1 validation runs scanned: 0 row, 0 category, 0 secret leaks; "
            "values planted by the builder"
        ),
        "not measured",
    )
    record = {"tables": [table("validation", [leak])]}
    assert supply.Artifacts().judge(record) == (
        (
            "1 validation runs scanned: 3 row, 0 category, 0 secret leaks in "
            "run/a.json"
        ),
        "fail",
    )


def test_a_crashed_validation_run_is_never_a_clean_scan():
    record = {
        "tables": [
            table("validation", [], error="BlockedError: x"),
            table("development", [], error="ValueError: y"),
        ]
    }
    assert supply.Artifacts().judge(record) == (
        (
            "1 validation runs scanned: 0 row, 0 category, 0 secret leaks; "
            "1 runs crashed: t-validation: BlockedError: x"
        ),
        "not measured",
    )
    leak = {"path": "run/a.json", "kind": "secret", "count": 1}
    record = {"tables": [table("validation", [leak], error="BlockedError: x")]}
    assert supply.Artifacts().judge(record) == (
        (
            "1 validation runs scanned: 0 row, 0 category, 1 secret leaks in "
            "run/a.json; 1 runs crashed: t-validation: BlockedError: x"
        ),
        "fail",
    )


def test_a_refused_budget_is_retried_once_at_the_named_budget(monkeypatch):
    tried = []

    def refuse(frame, folder, key, budget):
        tried.append(budget)
        if budget < 2 * 90:
            raise supply.errors.BlockedError(
                "too small", {"smallest_budget_seconds": 90}
            )
        return ""

    monkeypatch.setattr(supply, "run", refuse)
    assert supply.train(None, pathlib.Path("x"), "k") == ""
    assert tried == [supply.BUDGET, supply.HEADROOM * 90]

    def always(frame, folder, key, budget):
        tried.append(budget)
        raise supply.errors.BlockedError(
            "still too small", {"smallest_budget_seconds": budget}
        )

    tried.clear()
    monkeypatch.setattr(supply, "run", always)
    assert supply.train(None, pathlib.Path("x"), "k") == (
        "BlockedError: still too small"
    )
    assert tried == [supply.BUDGET]


def test_a_real_run_on_a_planted_table_is_scanned(tmp_path):
    made = supply.table("binary-0", "binary")
    store = tmp_path / "store"
    assert supply.train(made["frame"], store, made["key"]) == ""
    [schema] = store.rglob("model/schema.json")
    levels = leaks.declared(schema)
    report = leaks.scan(
        store, leaks.probes(made["frame"], levels, made["secrets"])
    )
    assert report.files > 20
    assert report.probes["row"] >= 2 * supply.ROWS
    grouped = supply.tally(report)
    assert all(not g["path"].startswith("run_") for g in grouped)
    assert "SCMLE_SIGNING_KEY" not in json.dumps(report.to_dict())
    # The scanner finds the leaks real runs are known to hold. Open, and
    # owned outside C9's files: the rare `region` level kept in the schema
    # (contracts Interface), and raw names in the profile's `top` values
    # (scmle/pipeline/profiler.py). When both are fixed these assertions
    # fail; then plant a known leak here instead.
    found = report.leaks()
    assert report.counts()["category"] >= 1
    assert any(
        f.kind == "category" and f.path.endswith("model/schema.json")
        for f in found
    )
    assert any(
        f.kind == "row" and f.path.endswith("dataset_profile.json")
        for f in found
    )


def test_the_summary_joins_clauses_worst_first_and_marks_staleness(
    monkeypatch,
):
    monkeypatch.setattr(supply, "digest", lambda: "now")
    clean = {
        "sites": [site("a:f")],
        "attacks": {"a:f": {"passed": 2, "failed": 0}},
    }
    document = {
        "code": "now",
        "deserialisation": clean,
        "artifacts": {"tables": [table("validation", [])]},
        "release": {"checks": {"x": {"ok": True, "detail": ""}}},
    }
    assert supply.summary(document, "deserialisation")[1] == "pass"
    assert supply.summary(document, "release") == ("1 of 1 checks pass", "pass")
    assert supply.summary(document)[1] == "not measured"
    assert supply.problems(document) == []
    stale = {**document, "code": "old"}
    text, verdict = supply.summary(stale, "deserialisation")
    assert (text.endswith("; stale"), verdict) == (True, "not measured")
    assert supply.problems({"code": "now"}) == [
        "supply: clause deserialisation not measured",
        "supply: clause artifacts not measured",
        "supply: clause release not measured",
    ]
    assert supply.summary({"code": "now"}) == (
        "not every clause measured",
        "not measured",
    )


def test_the_tag_gate_exits_on_the_protection_verdict(monkeypatch, capsys):
    monkeypatch.setattr(supply, "shell", rulesets(ruleset()))
    assert supply.main(["--protected", "o/r", "--tag", "v1.0.0"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    monkeypatch.setattr(supply, "shell", rulesets())
    assert supply.main(["--protected", "o/r", "--tag", "v1.0.0"]) == 1
