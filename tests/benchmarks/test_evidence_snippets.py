"""C12 evidence: every snippet and command in the docs runs in a fresh process."""

import pathlib
import subprocess

import pytest

from benchmarks.evidence import snippets
from scmle.contracts import errors

PAGE = """# Title

Save this as `hello.py`:

```python
print("hello")
```

Then run it:

```bash runnable
mkdir -p work && cd work
export GREETING=hi
python ../hello.py
```

```text
hello
```

```bash
test "$GREETING" = hi && test "$(basename "$PWD")" = work
```
"""


def run(page, tmp_path, name="docs/PAGE.md", limit=60.0):
    return snippets.follow(snippets.extract(page, name), tmp_path, limit)


def test_every_fenced_block_is_a_snippet_with_its_language_and_marks():
    # Enforces docs/preregistration/c12.md:16.
    found = snippets.extract(
        PAGE + "\n```bash background\nsleep 60\n```\n"
        "\n```bash refused\nfalse\n```\n",
        "docs/PAGE.md",
    )
    assert [s.language for s in found] == [
        "python",
        "bash",
        "text",
        "bash",
        "bash",
        "bash",
    ]
    assert [s.line for s in found[:3]] == [5, 11, 17]
    assert found[0].name == "hello.py"
    assert found[1].name == ""
    assert found[4].background
    assert found[5].refused
    assert found[2].kind == "output"


def test_an_unclosed_fence_is_refused():
    with pytest.raises(errors.InputError, match="unclosed"):
        snippets.extract("```bash\nls\n", "docs/PAGE.md")


def test_site_blocks_are_read_from_markup_literals_and_set_html():
    page = (
        '---\nconst install = `<span class="cmd">$ </span>pip --version`;\n'
        "---\n<p>Save it as <code>make.py</code>.</p>\n"
        '<pre><code class="language-python">{`print("a\\\\n")`}</code></pre>\n'
        '<pre><code class="language-bash" set:html={install}></code></pre>\n'
        "<pre><code>a &lt; b</code></pre>\n"
        '<pre><code class="language-bash" data-snippet="refused">'
        "false</code></pre>\n"
    )
    found = snippets.extract(page, "website/src/pages/x.astro")
    assert [s.language for s in found] == ["python", "bash", "", "bash"]
    assert found[0].text == 'print("a\\n")\n'
    assert found[0].name == "make.py"
    assert found[1].text == "pip --version\n"
    assert found[2].text == "a < b\n"
    assert found[3].refused


def test_a_set_html_without_its_literal_is_refused():
    page = "<pre><code set:html={gone}></code></pre>\n"
    with pytest.raises(errors.InputError, match="gone"):
        snippets.extract(page, "website/src/pages/x.astro")


def test_a_page_runs_top_to_bottom_carrying_the_terminal(tmp_path):
    # Enforces docs/preregistration/c12.md:24-25.
    # Enforces docs/preregistration/c12.md:70-74 (fresh folder and process
    # per page and block, carried terminal) and c12.md:74-76 (outputs are
    # counted, not run).
    records = run(PAGE, tmp_path)
    assert [r["status"] for r in records] == ["ran", "ran", "output", "ran"]
    assert snippets.tally(records) == {
        "snippets": 3,
        "ran": 3,
        "failed": 0,
        "skipped": 0,
        "outputs": 1,
        "synopses": 0,
    }


def test_each_page_starts_in_an_empty_folder_without_scmle_variables(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SCMLE_SIGNING_KEY", "leaked")
    page = (
        "```bash\n"
        'test -z "$(ls -A)" && test "$HOME" = "$PWD"'
        ' && test -z "${SCMLE_SIGNING_KEY:-}"\n'
        "touch left.txt\n```\n"
    )
    first = run(page, tmp_path)
    second = run(page, tmp_path)
    assert [r["status"] for r in first + second] == ["ran", "ran"]


def test_a_failing_command_fails_its_snippet_even_when_the_last_passes(
    tmp_path,
):
    (record,) = run("```bash\nfalse\necho done\n```\n", tmp_path)
    assert record["status"] == "failed"
    assert record["reason"].startswith("line 1 exited 1")


def test_a_failure_the_page_shows_with_its_exit_code_is_not_a_failure(
    tmp_path,
):
    page = (
        "```bash\nls missing-file \\\n  other\n"
        'echo "exit code: $?"\n```\n'
        "```bash refused\nfalse\n```\n"
        "```bash refused\ntrue\n```\n"
    )
    records = run(page, tmp_path)
    assert [r["status"] for r in records] == ["ran", "ran", "failed"]
    assert "exited 0" in records[2]["reason"]


@pytest.mark.parametrize(
    ("block", "status"),
    [
        ("```python\nraise SystemExit(3)\n```\n", "failed"),
        ("```python\nprint(undefined_name)\n```\n", "failed"),
        ('```json\n{"a": 1}\n```\n', "ran"),
        ('```json\n{"a": ...}\n```\n', "failed"),
        ("```toml\na = 1\n```\n", "ran"),
        ("```toml\na = \n```\n", "failed"),
        ("```\nls\n```\n", "failed"),
        ("```yaml\na: 1\n```\n", "failed"),
    ],
)
def test_each_language_runs_as_its_kind_and_unlabelled_code_fails(
    tmp_path, block, status
):
    (record,) = run(block, tmp_path)
    assert record["status"] == status


def test_a_saved_script_no_command_runs_is_run_at_the_end(tmp_path):
    page = "Save this as `boom.py`:\n\n```python\nraise SystemExit(1)\n```\n"
    (record,) = run(page, tmp_path)
    assert record["status"] == "failed"


def test_a_powershell_block_is_skipped_without_pwsh_and_fails_the_clause(
    tmp_path, monkeypatch
):
    # Enforces docs/preregistration/c12.md:76-77.
    monkeypatch.setattr(snippets.shutil, "which", lambda *a, **k: None)
    (record,) = run("```powershell\n$x = 1\n```\n", tmp_path)
    assert record["status"] == "skipped"
    counts = snippets.tally([record])
    assert snippets.summary({"counts": counts})[1] == "fail"


def test_a_synopsis_is_listed_not_run_and_needs_a_placeholder(tmp_path):
    page = (
        "```bash synopsis\nscmle trail show <run> [--json]\n```\n"
        "```bash synopsis\nfalse\n```\n"
    )
    records = run(page, tmp_path)
    assert [r["status"] for r in records] == ["synopsis", "failed"]
    assert "placeholder" in records[1]["reason"]
    counts = snippets.tally(records)
    assert (counts["synopses"], counts["snippets"]) == (1, 1)
    assert "1 usage synopses not run" in snippets.summary({"counts": counts})[0]


def test_the_published_url_reaches_the_working_tree_snapshot(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "marker.txt").write_text("uncommitted\n")
    source = snippets.snapshot(root, tmp_path)
    page = (
        f"```bash\ngit clone -q {snippets.ORIGIN}\ncat scmle/marker.txt\n```\n"
    )
    found = snippets.extract(page, "docs/PAGE.md")
    (record,) = snippets.follow(found, tmp_path, 60.0, source)
    assert record["status"] == "ran", record["reason"]


def test_a_snippet_past_the_cap_fails(tmp_path):
    (record,) = run("```bash\nsleep 5\n```\n", tmp_path, limit=0.5)
    assert record["status"] == "failed"
    assert "cap" in record["reason"]


def test_a_background_service_must_stay_up_and_is_stopped(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(snippets, "SETTLE", 0.5)
    page = (
        "```bash background\nsleep 30\n```\n```bash background\nexit 4\n```\n"
    )
    records = run(page, tmp_path)
    assert [r["status"] for r in records] == ["ran", "failed"]


def test_the_cap_scales_with_the_largest_documented_budget():
    # Enforces docs/preregistration/c12.md:73-74.
    found = snippets.extract(
        "```bash\nscmle run a.csv --budget 600\n```\n"
        "```python\nbudget = 9000\n```\n",
        "docs/PAGE.md",
    )
    assert snippets.cap(found) == snippets.MARGIN * 600
    assert snippets.cap(found[1:]) == snippets.FLOOR


def test_a_skipped_or_failed_snippet_fails_the_clause():
    def document(**counts):
        base = {"ran": 2, "failed": 0, "skipped": 0, "outputs": 1}
        base.update(counts)
        base["snippets"] = base["ran"] + base["failed"] + base["skipped"]
        return {"counts": base}

    assert snippets.summary(document())[1] == "pass"
    assert snippets.summary(document(skipped=1))[1] == "fail"
    assert snippets.summary(document(failed=1))[1] == "fail"
    assert snippets.summary(document(ran=0))[1] == "not measured"


def test_every_scanned_page_declares_a_language_for_every_block():
    root = pathlib.Path(__file__).resolve().parents[2]
    found = snippets.gather(root)
    undeclared = [f"{s.doc}:{s.line}" for s in found if not s.kind]
    assert len(found) > 100
    assert undeclared == []


def test_the_listing_prints_every_snippet(capsys):
    assert snippets.main(["--list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("README.md:") for line in lines)
