"""No exception handler swallows an error without saying so (task 2.1).

Every `except` in the package must re-raise, call a logger, or carry a
marker comment, `# handled: <where the error goes>`, on its `except` line or
on the comment lines before the first statement of its body, naming the
visible outcome that replaces the error (a returned failure, a recorded
finding, a documented fallback). A handler with none of the three hides an
error; the register calls these swallows (ERR-24).
"""

import ast
import io
import pathlib
import tokenize

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCES = sorted((ROOT / "scmle").rglob("*.py"))
LOGGERS = frozenset(
    {"debug", "info", "warning", "warn", "error", "exception", "critical"}
)
MARKER = "# handled:"


def markers(text):
    """Line number to the text of a `# handled:` comment on that line."""
    found = {}
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.COMMENT and MARKER in token.string:
            found[token.start[0]] = token.string.split(MARKER, 1)[1].strip()
    return found


def swallows(text):
    """Lines of handlers that neither raise, log nor carry a marker."""
    marked = markers(text)
    found = []
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, ast.ExceptHandler):
            continue
        inner = [n for statement in node.body for n in ast.walk(statement)]
        raises = any(isinstance(n, ast.Raise) for n in inner)
        logs = any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in LOGGERS
            for n in inner
        )
        reasons = [
            marked.get(line, "")
            for line in range(node.lineno, node.body[0].lineno)
        ]
        if not (raises or logs or any(reasons)):
            found.append(node.lineno)
    return sorted(found)


@pytest.mark.parametrize(
    "path", SOURCES, ids=lambda p: str(p.relative_to(ROOT))
)
def test_no_handler_swallows_an_error(path):
    lines = swallows(path.read_text(encoding="utf-8"))
    assert not lines, f"{path.name}: handlers at lines {lines}"


def test_the_rule_sees_a_swallow_and_accepts_the_three_ways_out():
    text = (
        "try:\n    x()\nexcept ValueError:\n    pass\n"
        "try:\n    x()\nexcept ValueError:\n    raise\n"
        "try:\n    x()\nexcept ValueError:\n    LOGGER.warning('w')\n"
        "try:\n    x()\nexcept ValueError:  # handled: returned as a failure\n"
        "    y = 1\n"
        "try:\n    x()\nexcept ValueError:  # handled:\n    pass\n"
        "try:\n    x()\nexcept ValueError:\n"
        "    # handled: returned as a failed finding, with the\n"
        "    # error text\n    y = 1\n"
    )
    assert swallows(text) == [3, 19]
