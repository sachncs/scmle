"""C12 evidence: every snippet and command in the docs runs in a fresh process.

The second clause of C12 (`docs/preregistration/c12.md`). The population is
that of `benchmarks.evidence.guarantees`: `README.md`, every Markdown file
under `docs/` and the site source under `website/src`.

What a snippet is: every fenced code block of a Markdown file, and every
`<pre><code>` block of a site page (its text, a `{`...`}` template literal,
or the front-matter template literal a `set:html={name}` names). Each one
declares its language: the fence's info string, or a `language-<name>`
class on the `<code>` element. The language decides what running means:

* `bash`, `sh`, `shell`, `console`: the block is a shell script, run by
  `bash` with `pipefail`. It fails when any command fails, unless the next
  command of the block reads `$?` (the page shows that exit code on
  purpose). A leading `$ ` prompt is dropped.
* `python`: the block is a script, run by the `python` found on the page's
  `PATH`.
* `json` and `toml`: the block is a document; it runs when it parses.
* `powershell`: run by `pwsh` where it is installed, and `skipped`
  otherwise.
* `text` (and `output`): shown output, not code. It is counted under
  `outputs` and not run.

A block that declares no language, or one not listed, fails: nothing may
leave the population by being unlabelled.

How a page runs: as a reader would follow it, top to bottom, in a fresh
empty folder that is also `HOME`, with no `SCMLE_*` variable set. What the
page finds does not depend on the machine or on how the measurement was
started (`Bench`): first on `PATH` is a fresh environment of its own holding
a snapshot of the working tree (`snapshot`) installed, not editable, with its
locked dependencies and no extra (`install`), so a page that needs an extra
must install it as a reader would; then the tools of `TOOLS`; then only
`os.defpath`. The pages install the package from its GitHub URL (`ORIGIN`);
that URL reaches the same snapshot (through the `HOME` folder's git
configuration), so a run measures the code the stored result names, not the
last published commit nor whatever another checkout or environment holds.
Each snippet is a fresh process; what a reader's terminal keeps between
commands (the working directory and the exported variables) carries from one
shell snippet to the next. A Python, JSON or TOML block whose lead-in
paragraph says to save it under a name of its kind (`save ...
read_results.py`) is written there; a Python file is then counted as run
when a later shell snippet of the page runs it, and is run directly
otherwise. Three marks, a word after the language in a fence's info string
or in a `data-snippet="..."` attribute on `<code>`, state what the page
shows. `background`: the block starts a service the page leaves running; it
runs when it is still alive `SETTLE` seconds after starting, and it is
stopped when the page ends. `refused`: the page shows the command being
refused; it runs when it exits non-zero, and fails when it succeeds.
`synopsis`: the block is a usage pattern with placeholders (`<sha>`,
`[--budget SECONDS]`, `...`), not a command; it is not run, and it is
counted and listed under `synopses` so the report shows every one. A
synopsis with no placeholder fails: it is a command, and a command runs.

The time cap: `MARGIN` times the largest `--budget` a shell snippet sets
(never below `FLOOR`); a snippet still running at the cap fails.

Verdict: `pass` only when at least one snippet was found and every one ran;
`fail` when one failed or was skipped. A run depends on the product, so the
stored result goes stale when a scanned doc, the package source, the lock
file or this module changes.

    python -m benchmarks.evidence.snippets --list           # extract only
    python -m benchmarks.evidence.snippets --report [DOC]   # run, print
    python -m benchmarks.evidence.snippets --write
    python -m benchmarks.evidence.snippets --check
"""

import argparse
from collections.abc import Callable
import dataclasses
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
from typing import Any

from benchmarks.evidence import guarantees
from scmle.assurance.detectors import certificates
from scmle.contracts import errors

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = Path(__file__).with_name("snippets.json")
KINDS = {
    "bash": "shell",
    "sh": "shell",
    "shell": "shell",
    "console": "shell",
    "python": "python",
    "json": "json",
    "toml": "toml",
    "powershell": "powershell",
    "text": "output",
    "output": "output",
}
SAVED = {"python": ".py", "json": ".json", "toml": ".toml"}
# The cap scales with the longest search a page asks for: a run spends its
# `--budget` on the search, and profiling, the one test and packaging took
# well under the search on the tables the docs use (the site quickstart's
# 60 s run finished in about 30 s), so twice the budget leaves the rest of
# the run as much time again. FLOOR covers a page with no budget: the
# slowest command without one, `pip install .` of the package with its
# locked dependencies, took under 150 s on a cold cache.
MARGIN = 2.0
FLOOR = 300.0
# A documented service imports the package and verifies the bundle before it
# listens; that took under 5 s on the author's machine, doubled for a loaded
# CI runner.
SETTLE = 10.0
# Bytes of a failing snippet's output kept as its reason.
TAIL = 600
# The repository's published URL, as the pages clone and install it.
ORIGIN = "https://github.com/sachncs/scmle"
# The tools a page finds besides its own environment and `os.defpath`: the
# pages clone with git and develop with uv. Anything else a page runs (a
# container engine, PowerShell) must be on `os.defpath` of every machine
# alike, or its block fails or is skipped the same everywhere.
TOOLS = ("uv", "git")
# Variables that only point the tools at caches the machine already holds:
# they change how fast a page runs, never what it finds.
CACHES = {
    "UV_CACHE_DIR": ("cache", "dir"),
    "UV_PYTHON_INSTALL_DIR": ("python", "dir"),
}
FENCE = re.compile(r"^(\s*)(```+|~~~+)\s*([^\n`]*)$")
BLOCK = re.compile(
    r"<pre\b[^>]*>\s*<code\b([^>]*)>(.*?)</code>\s*</pre>",
    re.DOTALL | re.IGNORECASE,
)
LANGUAGE = re.compile(r"\blanguage-([\w-]+)")
MARKS = re.compile(r"data-snippet\s*=\s*\"([^\"]*)\"")
SOURCE = re.compile(r"set:html\s*=\s*\{\s*(\w+)\s*\}")
LITERAL = re.compile(r"\A\s*\{\s*`(.*)`\s*\}\s*\Z", re.DOTALL)
TAG = re.compile(r"<[^>]+>")
PROMPT = re.compile(r"^\$ ", re.MULTILINE)
BUDGET = re.compile(r"--budget\s+(\d+(?:\.\d+)?)")
NAMED = re.compile(r"(?:`|<code>)([\w./-]+\.(?:py|json|toml))(?:`|</code>)")
CHECKED = re.compile(r"\$\?")
PLACEHOLDER = re.compile(r"<[^<>\n]+>|\[[^\]\n]+\]|\.\.\.")
PREAMBLE = (
    "set -o pipefail\n"
    'trap \'printf "%s %s\\n" "$LINENO" "$?" >> "$SNIPPET_STATE/errors"\' ERR\n'
    'trap \'pwd > "$SNIPPET_STATE/cwd"; env -0 > "$SNIPPET_STATE/env"\' EXIT\n'
)


@dataclasses.dataclass(frozen=True)
class Snippet:
    """One code block of a page.

    Attributes:
        doc: The page's path relative to the repository root.
        line: 1-based line of the block's opening fence or `<pre>`.
        language: The declared language, empty when none is declared.
        text: The block's code.
        lead: The text of the paragraph just above the block.
        background: Whether the page leaves this service running.
        refused: Whether the page shows this command being refused.
        synopsis: Whether the block is a usage pattern, not a command.
    """

    doc: str
    line: int
    language: str
    text: str
    lead: str = ""
    background: bool = False
    refused: bool = False
    synopsis: bool = False

    @property
    def kind(self) -> str:
        """`shell`, `python`, `json`, `toml`, `powershell`, `output` or ``."""
        return KINDS.get(self.language, "")

    @property
    def name(self) -> str:
        """The file name the lead-in says to save the block as, or ``."""
        suffix = SAVED.get(self.kind)
        if suffix is None or "save" not in self.lead.lower():
            return ""
        found = [n for n in NAMED.findall(self.lead) if n.endswith(suffix)]
        return Path(found[-1]).name if found else ""


def fences(text: str, name: str) -> list[Snippet]:
    """The fenced code blocks of a Markdown page.

    Args:
        text: The Markdown source.
        name: The page's path relative to the repository root.

    Returns:
        One snippet per fenced block, in order.

    Raises:
        InputError: If a fence is never closed.
    """
    found = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        opening = FENCE.match(lines[index])
        if opening is None:
            index += 1
            continue
        indent, marker, info = opening.groups()
        words = info.split()
        body = []
        close = index + 1
        while close < len(lines) and not lines[close].strip().startswith(
            marker
        ):
            body.append(lines[close].removeprefix(indent))
            close += 1
        if close == len(lines):
            message = f"{name}:{index + 1}: unclosed code fence"
            raise errors.InputError(message)
        found.append(
            Snippet(
                doc=name,
                line=index + 1,
                language=words[0].lower() if words else "",
                text="\n".join(body) + "\n",
                lead=lead(lines, index),
                background="background" in words[1:],
                refused="refused" in words[1:],
                synopsis="synopsis" in words[1:],
            )
        )
        index = close + 1
    return found


def lead(lines: list[str], index: int) -> str:
    """The paragraph just above a line, joined into one string.

    Args:
        lines: The page's lines.
        index: 0-based index of the block's first line.

    Returns:
        The non-blank lines above `index` up to the previous blank line.
    """
    above: list[str] = []
    cursor = index - 1
    while cursor >= 0 and not lines[cursor].strip():
        cursor -= 1
    while cursor >= 0 and lines[cursor].strip():
        above.insert(0, lines[cursor].strip())
        cursor -= 1
    return " ".join(above)


def blocks(text: str, name: str) -> list[Snippet]:
    """The `<pre><code>` blocks of a site page.

    Args:
        text: The page's source.
        name: The page's path relative to the repository root.

    Returns:
        One snippet per block, in order.

    Raises:
        InputError: If a `set:html` names no template literal of the page.
    """
    found = []
    for match in BLOCK.finditer(text):
        attributes, body = match.groups()
        source = SOURCE.search(attributes)
        if source is not None:
            literal = re.search(
                r"\b" + source.group(1) + r"\s*=\s*`(.*?)`;", text, re.DOTALL
            )
            if literal is None:
                message = (
                    f"{name}: set:html names {source.group(1)}, "
                    "which is not a template literal of the page"
                )
                raise errors.InputError(message)
            body = literal.group(1)
        else:
            inner = LITERAL.match(body)
            if inner is not None:
                body = inner.group(1)
        body = body.replace("\\\\", "\\").replace("\\`", "`")
        code = PROMPT.sub("", html.unescape(TAG.sub("", body)))
        language = LANGUAGE.search(attributes)
        marks = MARKS.search(attributes)
        words = marks.group(1).split() if marks else []
        above = [
            line for line in text[: match.start()].splitlines() if line.strip()
        ] or [""]
        found.append(
            Snippet(
                doc=name,
                line=text.count("\n", 0, match.start()) + 1,
                language=language.group(1).lower() if language else "",
                text=code.rstrip("\n") + "\n",
                lead=" ".join(html.unescape(above[-1]).split()),
                background="background" in words,
                refused="refused" in words,
                synopsis="synopsis" in words,
            )
        )
    return found


def extract(text: str, name: str) -> list[Snippet]:
    """The snippets of one page, Markdown or site source.

    Args:
        text: The page's source.
        name: Its path relative to the repository root; a name not ending
            in `.md` is read as a site page.

    Returns:
        The page's snippets, in order.
    """
    return fences(text, name) if name.endswith(".md") else blocks(text, name)


def gather(root: Path) -> list[Snippet]:
    """Every snippet of every scanned page.

    Args:
        root: Repository root.

    Returns:
        The snippets, page by page in the order of `guarantees.collect`.
    """
    found: list[Snippet] = []
    for path in guarantees.collect(root):
        name = path.relative_to(root).as_posix()
        found += extract(path.read_text(encoding="utf-8"), name)
    return found


def cap(snippets: list[Snippet]) -> float:
    """The time cap: `MARGIN` times the largest documented `--budget`.

    Args:
        snippets: The population.

    Returns:
        Seconds, never below `FLOOR`.
    """
    budgets = [
        float(b)
        for s in snippets
        if s.kind == "shell"
        for b in BUDGET.findall(s.text)
    ]
    return max(FLOOR, MARGIN * max(budgets, default=0.0))


def deliberate(text: str, line: int) -> bool:
    """Whether the command at a line is followed by one that reads `$?`.

    Args:
        text: The shell snippet.
        line: 1-based line, within the snippet, where the failing command
            starts.

    Returns:
        True when the next command of the snippet reads the exit status.
    """
    lines = text.splitlines()
    cursor = line - 1
    while cursor < len(lines) and lines[cursor].rstrip().endswith("\\"):
        cursor += 1
    for following in lines[cursor + 1 :]:
        if following.strip() and not following.strip().startswith("#"):
            return CHECKED.search(following) is not None
    return False


@dataclasses.dataclass(frozen=True)
class Bench:
    """What every page is given, the same on every machine.

    Attributes:
        source: The working-tree snapshot (`snapshot`) each page installs
            and `ORIGIN` reaches.
        tools: A folder of links to the tools of `TOOLS`.
        variables: The cache locations of `CACHES`.
    """

    source: Path
    tools: Path
    variables: tuple[tuple[str, str], ...] = ()


def furnish(source: Path, base: Path) -> Bench:
    """Gathers the tools and caches every page is given.

    Args:
        source: The working-tree snapshot.
        base: A scratch directory.

    Returns:
        The bench.

    Raises:
        InputError: If a tool of `TOOLS` is not installed.
    """
    tools = base / "tools"
    tools.mkdir()
    for name in TOOLS:
        found = shutil.which(name)
        if found is None:
            message = f"snippets: the pages need {name}, which is not installed"
            raise errors.InputError(message)
        (tools / name).symlink_to(found)
    variables = []
    for variable, arguments in CACHES.items():
        done = subprocess.run(
            [str(tools / "uv"), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode == 0 and done.stdout.strip():
            variables.append((variable, done.stdout.strip()))
    return Bench(source, tools, tuple(variables))


def install(bench: Bench, target: Path) -> Path:
    """A fresh environment holding the snapshot and its locked dependencies.

    `uv venv --seed`, then `uv sync --frozen --no-dev --no-editable`: like a
    reader's virtual environment it has `pip`; the lock file decides every
    other version, the snapshot is installed as a wheel (so nothing reads
    another checkout), and no extra or development tool is in it.

    Args:
        bench: The bench.
        target: The environment's folder (created).

    Returns:
        Its folder of programs, to put first on `PATH`.

    Raises:
        InputError: If the snapshot cannot be installed.
    """
    variables = {
        "HOME": str(target.parent),
        "PATH": os.pathsep.join([str(bench.tools), os.defpath]),
        "UV_PROJECT_ENVIRONMENT": str(target),
        **dict(bench.variables),
    }
    uv = str(bench.tools / "uv")
    for command in (
        [uv, "venv", "--seed", "--allow-existing", str(target)],
        [uv, "sync", "--frozen", "--no-dev", "--no-editable", "--inexact"],
    ):
        done = subprocess.run(
            [*command, "--quiet", "--project", str(bench.source)],
            cwd=bench.source,
            env=variables,
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode != 0:
            message = (
                f"snippets: cannot install the snapshot: {done.stderr[-TAIL:]}"
            )
            raise errors.InputError(message)
    return target / ("Scripts" if os.name == "nt" else "bin")


class Terminal:
    """A reader's terminal: one page, followed top to bottom.

    Attributes:
        home: The empty folder the page starts in (also `HOME`).
        state: Where each snippet leaves its working directory and
            exported variables.
        cwd: The working directory the next snippet starts in.
        env: The exported variables the next snippet starts with.
        limit: The time cap in seconds.
        services: Background services still running.
        saved: Python files saved but not yet run, by name, with the
            snippet and the path they were saved at.
    """

    def __init__(
        self,
        base: Path,
        limit: float,
        source: Path | None = None,
        bench: Bench | None = None,
    ) -> None:
        """Opens a terminal in a fresh folder under `base`.

        Args:
            base: A scratch directory.
            limit: The time cap in seconds.
            source: A git repository that `ORIGIN` reaches from this
                terminal (`snapshot`), or None to leave `ORIGIN` alone.
            bench: What the page is given (its own environment and the
                tools); None gives the measuring interpreter's folder
                instead, for a test of the terminal itself.
        """
        if bench is not None:
            source = bench.source
        self.home = Path(tempfile.mkdtemp(dir=base, prefix="page-"))
        if source is not None:
            redirect = "".join(
                f'[url "{source.as_uri()}"]\n\tinsteadOf = {url}\n'
                for url in (ORIGIN, ORIGIN + ".git")
            )
            (self.home / ".gitconfig").write_text(redirect, encoding="utf-8")
        self.state = Path(tempfile.mkdtemp(dir=base, prefix="state-"))
        self.cwd = self.home
        if bench is None:
            found = [str(Path(sys.executable).parent)]
            variables: dict[str, str] = {}
        else:
            target = Path(tempfile.mkdtemp(dir=base, prefix="environment-"))
            found = [str(install(bench, target)), str(bench.tools)]
            variables = dict(bench.variables)
        self.env = {
            **variables,
            "HOME": str(self.home),
            "PATH": os.pathsep.join([*found, os.defpath]),
            "LANG": "C.UTF-8",
            "TMPDIR": str(self.state),
            "SNIPPET_STATE": str(self.state),
        }
        self.limit = limit
        self.services: list[subprocess.Popen[bytes]] = []
        self.saved: dict[str, tuple[Snippet, Path]] = {}

    def run(self, snippet: Snippet) -> dict[str, Any]:
        """Runs one snippet as its language says.

        Args:
            snippet: The snippet.

        Returns:
            Its record: `doc`, `line`, `language`, `status` (`ran`,
            `failed`, `skipped`, `output`, `synopsis` or `saved`),
            `seconds` and `reason`.
        """
        started = time.monotonic()
        kind = snippet.kind
        if kind == "output":
            status, reason = "output", ""
        elif snippet.synopsis and PLACEHOLDER.search(snippet.text):
            status, reason = "synopsis", "a usage pattern, not run"
        elif snippet.synopsis:
            status, reason = "failed", "a synopsis shows no placeholder"
        elif kind == "":
            status = "failed"
            reason = f"undeclared or unknown language {snippet.language!r}"
        elif kind == "shell":
            status, reason = self.shell(snippet)
        elif kind == "powershell":
            status, reason = self.powershell(snippet)
        else:
            status, reason = self.document(snippet)
        return {
            "doc": snippet.doc,
            "line": snippet.line,
            "language": snippet.language,
            "status": status,
            "seconds": round(time.monotonic() - started, 1),
            "reason": reason[-TAIL:],
        }

    def execute(self, command: list[str]) -> tuple[int | None, str]:
        """Runs a command in the terminal, within the time cap.

        Args:
            command: The program and its arguments.

        Returns:
            The exit code (None when the cap was reached) and the output.
        """
        try:
            done = subprocess.run(
                command,
                cwd=self.cwd if self.cwd.is_dir() else self.home,
                env=self.env,
                capture_output=True,
                timeout=self.limit,
                check=False,
                start_new_session=True,
            )
        except subprocess.TimeoutExpired as expired:
            output = (expired.stdout or b"") + (expired.stderr or b"")
            return None, output.decode("utf-8", "replace")
        output = done.stdout + done.stderr
        return done.returncode, output.decode("utf-8", "replace")

    def shell(self, snippet: Snippet) -> tuple[str, str]:
        """Runs a shell snippet, or starts it as a background service.

        Args:
            snippet: A snippet of kind `shell`.

        Returns:
            The status and the reason.
        """
        script = self.state / f"snippet-{snippet.line}.sh"
        script.write_text(PREAMBLE + snippet.text, encoding="utf-8")
        trapped = self.state / "errors"
        trapped.unlink(missing_ok=True)
        if snippet.background:
            return self.start(script)
        code, output = self.execute(["bash", str(script)])
        self.carry()
        for name in list(self.saved):
            pattern = r"\bpython3?\s+[\w./-]*\b" + re.escape(name)
            if re.search(pattern, snippet.text):
                del self.saved[name]
        failures = []
        if trapped.exists():
            failures = judge(snippet, trapped.read_text(encoding="utf-8"))
        if code is None or not snippet.refused:
            return self.verdict(
                code if code or not failures else 1, output, failures
            )
        if code != 0:
            return "ran", ""
        return "failed", "the page shows a refusal, but it exited 0"

    def verdict(
        self, code: int | None, output: str, failures: list[str] | None = None
    ) -> tuple[str, str]:
        """The status and reason of a finished process.

        Args:
            code: Its exit code, None when the cap stopped it.
            output: What it printed.
            failures: The failing commands found inside it, if any.

        Returns:
            `ran` for exit code 0, else `failed` with why and the output.
        """
        if code is None:
            return (
                "failed",
                f"still running at the {self.limit:.0f} s cap\n" + output,
            )
        if code == 0:
            return "ran", ""
        return "failed", (
            "; ".join(failures or []) or f"exited {code}"
        ) + "\n" + output

    def start(self, script: Path) -> tuple[str, str]:
        """Starts a background service and checks that it stays up.

        Args:
            script: The snippet's script.

        Returns:
            `ran` when it is alive after `SETTLE` seconds, else `failed`
            with its output.
        """
        log = self.state / f"{script.stem}.log"
        with log.open("wb") as sink:
            service = subprocess.Popen(
                ["bash", str(script)],
                cwd=self.cwd,
                env=self.env,
                stdout=sink,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        time.sleep(SETTLE)
        if service.poll() is not None:
            return "failed", "exited on start\n" + log.read_text(
                encoding="utf-8", errors="replace"
            )
        self.services.append(service)
        return "ran", ""

    def powershell(self, snippet: Snippet) -> tuple[str, str]:
        """Runs a PowerShell snippet where `pwsh` is installed.

        Args:
            snippet: A snippet of kind `powershell`.

        Returns:
            The status and the reason; `skipped` without `pwsh`.
        """
        program = shutil.which("pwsh", path=self.env["PATH"])
        if program is None:
            return "skipped", "pwsh is not installed on this platform"
        code, output = self.execute(
            [program, "-NoProfile", "-NonInteractive", "-Command", snippet.text]
        )
        return self.verdict(code, output)

    def document(self, snippet: Snippet) -> tuple[str, str]:
        """Saves a named block, parses a document, or runs a Python block.

        Args:
            snippet: A snippet of kind `python`, `json` or `toml`.

        Returns:
            The status and the reason.
        """
        name = snippet.name
        if name:
            target = self.cwd / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(snippet.text, encoding="utf-8")
        if snippet.kind != "python":
            return parse(snippet)
        if name:
            self.saved[name] = (snippet, self.cwd / name)
            return "saved", ""
        script = self.state / f"snippet-{snippet.line}.py"
        script.write_text(snippet.text, encoding="utf-8")
        return self.python(script)

    def python(self, script: Path) -> tuple[str, str]:
        """Runs a Python file with the `python` on the page's `PATH`.

        Args:
            script: The file.

        Returns:
            The status and the reason.
        """
        program = shutil.which("python", path=self.env["PATH"]) or "python"
        return self.verdict(*self.execute([program, str(script)]))

    def carry(self) -> None:
        """Takes the working directory and variables the last snippet left."""
        cwd = self.state / "cwd"
        if cwd.exists():
            self.cwd = Path(cwd.read_text(encoding="utf-8").strip())
        dump = self.state / "env"
        if dump.exists():
            pairs = dump.read_bytes().decode("utf-8", "replace").split("\0")
            carried = dict(p.split("=", 1) for p in pairs if "=" in p)
            carried.pop("PWD", None)
            carried.pop("OLDPWD", None)
            carried.pop("SHLVL", None)
            carried["SNIPPET_STATE"] = str(self.state)
            self.env = carried

    def close(self) -> None:
        """Stops every background service the page left running."""
        for service in self.services:
            try:
                os.killpg(service.pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
            try:
                service.wait(timeout=SETTLE)
            except subprocess.TimeoutExpired:
                os.killpg(service.pid, signal.SIGKILL)
        self.services.clear()


def judge(snippet: Snippet, trapped: str) -> list[str]:
    """The commands of a shell snippet that failed when they should not.

    Args:
        snippet: The shell snippet.
        trapped: What its `ERR` trap wrote: one `line status` per failure,
            the line counted in the script with its preamble.

    Returns:
        `line N exited S` for each failure the next command does not read
        (`deliberate`).
    """
    offset = PREAMBLE.count("\n")
    found = []
    for record in trapped.splitlines():
        number, _, status = record.partition(" ")
        line = int(number) - offset
        if not deliberate(snippet.text, line):
            found.append(f"line {line} exited {status}")
    return found


def parse(snippet: Snippet) -> tuple[str, str]:
    """Parses a JSON or TOML snippet.

    Args:
        snippet: A snippet of kind `json` or `toml`.

    Returns:
        `ran` when it parses, else `failed` with the parser's message.
    """
    try:
        if snippet.kind == "json":
            json.loads(snippet.text)
        else:
            tomllib.loads(snippet.text)
    except (json.JSONDecodeError, tomllib.TOMLDecodeError) as error:
        return "failed", f"not {snippet.kind.upper()}: {error}"
    return "ran", ""


def follow(
    snippets: list[Snippet],
    base: Path,
    limit: float,
    source: Path | None = None,
    bench: Bench | None = None,
) -> list[dict[str, Any]]:
    """Runs one page's snippets in order in a fresh terminal.

    Args:
        snippets: The page's snippets.
        base: A scratch directory.
        limit: The time cap in seconds.
        source: The working-tree snapshot `ORIGIN` reaches, or None.
        bench: What the page is given (`Terminal`).

    Returns:
        One record per snippet. A saved Python file that no later shell
        snippet ran is run directly at the end of the page.
    """
    terminal = Terminal(base, limit, source, bench)
    records: list[dict[str, Any]] = []
    try:
        records.extend(terminal.run(snippet) for snippet in snippets)
        for snippet, path in terminal.saved.values():
            status, reason = terminal.python(path)
            for record in records:
                if (record["doc"], record["line"]) == (
                    snippet.doc,
                    snippet.line,
                ):
                    record.update(status=status, reason=reason[-TAIL:])
    finally:
        terminal.close()
    for record in records:
        if record["status"] == "saved":
            record["status"] = "ran"
    return records


def snapshot(root: Path, base: Path) -> Path:
    """A git repository of the working tree, for `ORIGIN` to reach.

    Every file git tracks or would track (ignored files left out) is copied
    as it is on disk and committed once, so uncommitted edits are in it.

    Args:
        root: Repository root.
        base: A scratch directory.

    Returns:
        The snapshot repository's folder.
    """
    target = base / "source"
    listed = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")
    for name in filter(None, listed.split("\0")):
        path = root / name
        if path.is_file():
            copy = target / name
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, copy)
    author = ["-c", "user.name=snippets", "-c", "user.email=snippets@localhost"]
    for command in (
        ["init", "-q"],
        ["add", "-A"],
        [*author, "commit", "-q", "-m", "working tree"],
    ):
        subprocess.run(["git", *command], cwd=target, check=True)
    return target


def tally(records: list[dict[str, Any]]) -> dict[str, int]:
    """Counts snippets by outcome.

    Args:
        records: Records from `follow`.

    Returns:
        `snippets` (code blocks, outputs and synopses excluded), `ran`,
        `failed`, `skipped`, `outputs` and `synopses`.
    """
    count = {
        status: sum(1 for r in records if r["status"] == status)
        for status in ("ran", "failed", "skipped", "output", "synopsis")
    }
    return {
        "snippets": count["ran"] + count["failed"] + count["skipped"],
        "ran": count["ran"],
        "failed": count["failed"],
        "skipped": count["skipped"],
        "outputs": count["output"],
        "synopses": count["synopsis"],
    }


def fingerprint(root: Path) -> str:
    """Hash of everything a run depends on.

    Args:
        root: Repository root.

    Returns:
        64 lowercase hexadecimal characters over this module's code, the
        scanned pages, the package source and the lock file.
    """
    digest = hashlib.sha256()
    digest.update(Path(__file__).read_bytes())
    digest.update(Path(guarantees.__file__).read_bytes())
    paths = guarantees.collect(root)
    paths += sorted((root / "scmle").rglob("*.py"))
    paths += [root / n for n in ("pyproject.toml", "uv.lock")]
    for path in paths:
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def measure(
    root: Path = ROOT,
    only: str | None = None,
    echo: Callable[[list[dict[str, Any]]], None] | None = None,
) -> dict[str, Any]:
    """Runs every snippet of every page, page by page.

    The code and content hashes are taken before anything runs, so the
    result names the code and docs it was measured with even when a file
    changes during the run.

    Args:
        root: Repository root.
        only: A page path to run alone (for a report; never written).
        echo: Called with each page's records as soon as the page ends.

    Returns:
        The document written to `RESULTS_PATH`.
    """
    started = time.monotonic()
    code = certificates.code_hash(
        fences,
        blocks,
        deliberate,
        cap,
        follow,
        snapshot,
        furnish,
        install,
        tally,
    )
    content = fingerprint(root)
    snippets = gather(root)
    limit = cap(snippets)
    pages: dict[str, list[Snippet]] = {}
    for snippet in snippets:
        if only is None or snippet.doc == only:
            pages.setdefault(snippet.doc, []).append(snippet)
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="snippets-") as base:
        source = snapshot(root, Path(base))
        bench = furnish(source, Path(base))
        for page in pages.values():
            done = follow(page, Path(base), limit, bench=bench)
            records += done
            if echo is not None:
                echo(done)
    return {
        "version": 1,
        "claim": "C12",
        "clause": "every snippet and command runs in a fresh process",
        "definition": {
            "docs": "the pages benchmarks.evidence.guarantees scans",
            "kinds": KINDS,
            "cap_seconds": limit,
            "platform": sys.platform,
            "environment": (
                "per page: uv venv --seed, uv sync --frozen --no-dev "
                "--no-editable of the working-tree snapshot; then "
                + ", ".join(TOOLS)
                + "; then "
                "os.defpath"
            ),
        },
        "counts": tally(records),
        "snippets": records,
        "content_hash": content,
        "seconds": round(time.monotonic() - started, 1),
        "code_hash": code,
    }


def summary(document: dict[str, Any] | None = None) -> tuple[str, str]:
    """The scorecard value text and verdict.

    Args:
        document: A results document; read from `RESULTS_PATH` when None.

    Returns:
        The value text and `pass`, `fail` or `not measured` (missing,
        stale or empty).
    """
    if document is None:
        if not RESULTS_PATH.exists():
            return "snippets.json is missing", "not measured"
        document = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        if document.get("content_hash") != fingerprint(ROOT):
            return "snippets: result is for other docs or code", "not measured"
    counts = document["counts"]
    if counts["snippets"] == 0:
        return "no snippet found", "not measured"
    value = (
        f"{counts['ran']}/{counts['snippets']} snippets run in a fresh "
        f"process ({counts['failed']} failed, {counts['skipped']} skipped; "
        f"{counts.get('synopses', 0)} usage synopses not run)"
    )
    clean = counts["failed"] == 0 and counts["skipped"] == 0
    return value, "pass" if clean else "fail"


def show(records: list[dict[str, Any]]) -> None:
    """Prints each failed, skipped or synopsis snippet of a page.

    Args:
        records: One page's records.
    """
    for record in records:
        if record["status"] in ("failed", "skipped", "synopsis"):
            reason = record["reason"].strip().splitlines() or [""]
            where = f"{record['doc']}:{record['line']}"
            print(f"{where}\t{record['status']}\t{reason[0]}", flush=True)


def main(arguments: list[str] | None = None) -> int:
    """Lists, reports, writes or checks the snippet result.

    Args:
        arguments: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        0 on success; for `--check` and `--report`, 1 when a snippet
        failed or the stored result is stale.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--report", nargs="?", const="", metavar="DOC")
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    options = parser.parse_args(arguments)
    if options.list:
        for snippet in gather(ROOT):
            print(f"{snippet.doc}:{snippet.line}\t{snippet.language or '-'}")
        return 0
    if options.report is not None:
        document = measure(only=options.report or None, echo=show)
        print(json.dumps(document["counts"]))
        return 0 if summary(document)[1] == "pass" else 1
    if options.write:
        document = measure()
        RESULTS_PATH.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        value, verdict = summary(document)
        print(json.dumps({"value": value, "verdict": verdict}))
        return 0
    value, verdict = summary()
    print(json.dumps({"value": value, "verdict": verdict}))
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
