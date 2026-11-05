"""A small reader for the Terraform (HCL) files under ``deploy/terraform``.

It reads blocks and attributes, nothing else: no expression is evaluated,
every attribute keeps its source text. That is enough for a policy check over
files this repository writes (one attribute per line, no heredocs). It is not
a general HCL parser and it is not a substitute for ``terraform validate``;
text it cannot read raises `HclError` instead of being skipped.
"""

from collections.abc import Iterator
import dataclasses
import pathlib
import re

QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)"')
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


class HclError(ValueError):
    """The text is outside the subset this reader understands."""


@dataclasses.dataclass(slots=True)
class Block:
    """One block: ``kind "label" ... { attributes and child blocks }``.

    Attributes:
        kind: Block type (``resource``, ``statement``, ``default_action``).
        labels: The quoted labels after the type.
        attributes: Attribute name to its source text.
        children: Nested blocks in source order.
        path: File the block is in.
        line: 1-based line of the block's first token.
    """

    kind: str
    labels: tuple[str, ...]
    attributes: dict[str, str]
    children: list["Block"]
    path: str
    line: int

    def child(self, kind: str) -> list["Block"]:
        """Returns the direct children of one type."""
        return [item for item in self.children if item.kind == kind]

    def walk(self) -> Iterator["Block"]:
        """Yields this block and every descendant."""
        yield self
        for item in self.children:
            yield from item.walk()

    @property
    def address(self) -> str:
        """``type.name`` for a resource or data block, else the kind."""
        if self.kind in ("resource", "data") and len(self.labels) == 2:
            return ".".join(self.labels)
        return self.kind


def literal(text: str | None) -> bool | int | float | str | list | None:
    """Reads an attribute text as a literal, or None if it is an expression.

    A string with an interpolation is not a literal. Lists of literals are
    returned as lists.
    """
    if text is None:
        return None
    value = text.strip()
    if value == "true":
        return True
    if value == "false":
        return False
    if NUMBER.fullmatch(value):
        return float(value) if "." in value else int(value)
    if value.startswith('"') and value.endswith('"'):
        inner = QUOTED.findall(value)
        if len(inner) == 1 and "${" not in inner[0]:
            return inner[0]
        return None
    if value.startswith("[") and value.endswith("]"):
        items = [item for item in split_top_level(value[1:-1]) if item]
        parsed = [literal(item) for item in items]
        if any(item is None for item in parsed):
            return None
        return parsed
    return None


def split_top_level(text: str) -> list[str]:
    """Splits a list body on commas outside strings and brackets."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            end = string_end(text, index)
            current.append(text[index:end])
            index = end
            continue
        if char in "[{(":
            depth += 1
        elif char in "]})":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        elif char not in "\n":
            current.append(char)
        index += 1
    parts.append("".join(current).strip())
    return parts


def string_end(text: str, start: int) -> int:
    """Index after the string that opens at ``text[start]``.

    Raises:
        HclError: If the string is not closed.
    """
    index = start + 1
    depth = 0
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if text.startswith("${", index):
            depth += 1
            index += 2
            continue
        if char == "}" and depth > 0:
            depth -= 1
        elif char == '"' and depth == 0:
            return index + 1
        index += 1
    raise HclError("unterminated string")


class Reader:
    """Cursor over one file's text."""

    def __init__(self, text: str, path: str) -> None:
        """Starts at the beginning of ``text``."""
        self.text = text
        self.path = path
        self.position = 0

    def line(self) -> int:
        """1-based line of the cursor."""
        return self.text.count("\n", 0, self.position) + 1

    def skip(self, newlines: bool = True) -> None:
        """Skips spaces, comments and (optionally) line breaks."""
        while self.position < len(self.text):
            char = self.text[self.position]
            if char == "\n" and not newlines:
                return
            if char in " \t\r\n":
                self.position += 1
            elif char == "#" or self.text.startswith("//", self.position):
                end = self.text.find("\n", self.position)
                self.position = len(self.text) if end < 0 else end
            elif self.text.startswith("/*", self.position):
                end = self.text.find("*/", self.position)
                if end < 0:
                    raise HclError(f"{self.path}: unterminated comment")
                self.position = end + 2
            else:
                return

    def identifier(self) -> str:
        """Reads a name."""
        match = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*").match(
            self.text, self.position
        )
        if match is None:
            raise HclError(
                f"{self.path}:{self.line()}: expected a name, found "
                f"{self.text[self.position : self.position + 20]!r}"
            )
        self.position = match.end()
        return match.group(0)

    def expression(self) -> str:
        """Reads an attribute value up to the end of its line."""
        start = self.position
        depth = 0
        while self.position < len(self.text):
            char = self.text[self.position]
            if char == '"':
                self.position = string_end(self.text, self.position)
                continue
            if self.text.startswith("<<", self.position):
                raise HclError(f"{self.path}:{self.line()}: heredoc")
            if char == "#" or self.text.startswith("//", self.position):
                end = self.text.find("\n", self.position)
                text = self.text[start : self.position]
                self.position = len(self.text) if end < 0 else end
                return text.strip()
            if char in "[{(":
                depth += 1
            elif char in "]})":
                depth -= 1
                if depth < 0:
                    break
            elif char == "\n" and depth == 0:
                break
            self.position += 1
        return self.text[start : self.position].strip()

    def body(self, block: Block) -> None:
        """Reads attributes and child blocks until the closing brace."""
        while True:
            self.skip()
            if self.position >= len(self.text):
                raise HclError(f"{self.path}: block {block.kind} not closed")
            if self.text[self.position] == "}":
                self.position += 1
                return
            line = self.line()
            name = self.identifier()
            self.skip(newlines=False)
            if self.text[self.position : self.position + 1] == "=":
                self.position += 1
                self.skip(newlines=False)
                block.attributes[name] = self.expression()
                continue
            block.children.append(self.block(name, line))

    def block(self, kind: str, line: int) -> Block:
        """Reads the labels and body of a block whose type was read."""
        labels: list[str] = []
        while True:
            self.skip(newlines=False)
            if self.text[self.position : self.position + 1] == '"':
                end = string_end(self.text, self.position)
                labels.append(self.text[self.position + 1 : end - 1])
                self.position = end
            elif self.text[self.position : self.position + 1] == "{":
                self.position += 1
                break
            else:
                raise HclError(
                    f"{self.path}:{self.line()}: expected a label or '{{'"
                )
        found = Block(kind, tuple(labels), {}, [], self.path, line)
        self.body(found)
        return found


def parse(text: str, path: str = "<memory>") -> list[Block]:
    """Parses one file into its top-level blocks.

    Raises:
        HclError: If the text is outside the supported subset.
    """
    reader = Reader(text, path)
    blocks: list[Block] = []
    while True:
        reader.skip()
        if reader.position >= len(text):
            return blocks
        line = reader.line()
        name = reader.identifier()
        reader.skip(newlines=False)
        if text[reader.position : reader.position + 1] == "=":
            raise HclError(f"{path}:{line}: top-level attribute {name}")
        blocks.append(reader.block(name, line))


def parse_directory(root: str | pathlib.Path) -> list[Block]:
    """Parses every ``.tf`` file under ``root``, in path order."""
    blocks: list[Block] = []
    for path in sorted(pathlib.Path(root).rglob("*.tf")):
        blocks.extend(parse(path.read_text(encoding="utf-8"), str(path)))
    return blocks
