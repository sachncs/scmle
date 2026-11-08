"""Text artifacts are written with LF on every platform.

`Path.write_text` and a text-mode `open` translate "\\n" to "\\r\\n" on
Windows unless `newline="\\n"` is given. The project's own wrappers
(`RunStore.write_text(relative, text)`, `RunContext.write_text`) already write
LF; a call with two positional arguments, the second not a string literal, is
one of those and is not checked. The bytes of a written file then
differ between platforms while the in-memory hashes agree, which broke the
replay-identity test on Windows. A new writer without `newline` fails here.
"""

import ast
import pathlib

PACKAGE = pathlib.Path(__file__).resolve().parents[2] / "scmle"
WRITE_MODES = ("w", "a", "x", "+")


def mode_of(call: ast.Call, position: int) -> str | None:
    """The literal mode of an `open` call, from its position or keyword."""
    if len(call.args) > position and isinstance(
        call.args[position], ast.Constant
    ):
        value = call.args[position].value
        return value if isinstance(value, str) else None
    for keyword in call.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            value = keyword.value.value
            return value if isinstance(value, str) else None
    return None


def is_wrapper_call(call: ast.Call) -> bool:
    """Whether `call` is `store.write_text(relative, text)` of the project."""
    if len(call.args) < 2:
        return False
    second = call.args[1]
    return not (
        isinstance(second, ast.Constant) and isinstance(second.value, str)
    )


def offenders(source: str) -> list[int]:
    """Line numbers of text writes in `source` that do not set `newline`."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        has_newline = any(keyword.arg == "newline" for keyword in node.keywords)
        if (
            has_newline
            and isinstance(function, ast.Attribute)
            and function.attr == "write_text"
            and is_wrapper_call(node)
        ):
            found.append(node.lineno)  # the wrapper takes no `newline`
            continue
        if has_newline:
            continue
        if (
            isinstance(function, ast.Attribute)
            and function.attr == "write_text"
        ):
            if not is_wrapper_call(node):
                found.append(node.lineno)
            continue
        mode = None
        if isinstance(function, ast.Name) and function.id == "open":
            mode = mode_of(node, 1)
        elif isinstance(function, ast.Attribute) and function.attr == "open":
            mode = mode_of(node, 0)
        if mode and "b" not in mode and any(c in mode for c in WRITE_MODES):
            found.append(node.lineno)
    return found


def test_the_lint_flags_each_unsafe_writer_and_passes_the_safe_ones():
    unsafe = (
        "p.write_text('x')\n"
        "open('f', 'w')\n"
        "p.open(mode='a')\n"
        "p.open('w+')\n"
        "store.write_text('a.txt', text, newline='\\n')\n"
    )
    safe = (
        "p.write_text('x', newline='\\n')\n"
        "open('f', 'w', newline='\\n')\n"
        "open('f')\n"
        "open('f', 'r')\n"
        "open('f', 'wb')\n"
        "p.write_bytes(b'x')\n"
        "store.write_text('a.txt', text)\n"
    )
    assert offenders(unsafe) == [1, 2, 3, 4, 5]
    assert offenders(safe) == []


def test_every_text_writer_in_the_package_sets_newline():
    found = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        lines = offenders(path.read_text(encoding="utf-8"))
        if lines:
            found[str(path.relative_to(PACKAGE.parent))] = lines
    assert not found, f'text write without newline="\\n": {found}'
