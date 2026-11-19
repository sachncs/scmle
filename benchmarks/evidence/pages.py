"""Text units of the website source, for the C12 guarantee scan.

The site (`website/src`) is written in Astro pages and HTML fragments, not
Markdown, so `benchmarks.evidence.guarantees` reads it through this module.
A page's text lives in three places, and all three are read:

1. The markup: each paragraph, list item, card, cell or other block is one
   unit; a table row is one unit (its cells joined by `|`) and a row of
   header cells only is skipped, like a Markdown header row. Text inside
   `pre`, `script`, `style` and `svg` is skipped. Inline `code` is wrapped
   in backticks and link text in `[...](...)`, so the guarantee scan hides
   them exactly as it does in Markdown.
2. Quoted string literals in the front matter (between the opening `---`
   lines) and in `{...}` expressions of the markup: an array holding only
   strings is one table row (how `DocTable` rows are written); any other
   string is one unit. Template literals are code samples and are skipped,
   and so are the module paths of `import` lines and the `head={...}`
   array of a table, which is a header row.
3. HTML fragments (`website/src/learn/*.html`) are markup only.

Units come back in the shape `guarantees.chunk` uses, so the same sentence
split and guarantee words apply; a unit's section is the slug (or `id`) of
the nearest heading above it.
"""

from collections.abc import Iterator
import html.parser
import re
from typing import Any

BLOCKS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "caption",
        "dd",
        "details",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "section",
        "summary",
        "table",
        "tbody",
        "thead",
        "ul",
    }
)
SKIPPED = frozenset({"pre", "script", "style", "svg", "template"})
HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
CELLS = frozenset({"td", "th"})
OPENERS = {"(": ")", "[": "]", "{": "}"}
FRONT = re.compile(r"\A---[ \t]*\n(.*?\n)---[ \t]*(?:\n|\Z)", re.DOTALL)
CODEBLOCK = re.compile(
    r"<(style|script)\b.*?</\1\s*>", re.DOTALL | re.IGNORECASE
)
ESCAPES = {"n": " ", "t": " "}
SPACE = re.compile(r"\s+")
IMPORT = re.compile(r"^[ \t]*(?:import\b|\}\s*from\b)[^\n]*", re.MULTILINE)
HEAD = re.compile(r"\shead=\s*\Z")


def slugify(heading: str) -> str:
    """The anchor of a heading, as `guarantees.slugify` computes it.

    Args:
        heading: Heading text.

    Returns:
        Lowercase text with punctuation removed and spaces as hyphens.
    """
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def blank(text: str) -> str:
    """The text with every character but newlines turned to a space.

    Args:
        text: Any text.

    Returns:
        A string of the same length and the same line breaks.
    """
    return re.sub(r"[^\n]", " ", text)


class Reader(html.parser.HTMLParser):
    """Collects block units, table rows and headings from HTML.

    Attributes:
        units: Collected units: `start`, `end`, `kind` (`prose` or `row`)
            and `text`.
        headings: (line, slug) of each heading, in order.
    """

    def __init__(self, first: int = 1) -> None:
        """Starts an empty reader.

        Args:
            first: Line number of the first line of the fed text.
        """
        super().__init__(convert_charrefs=True)
        self.first = first
        self.units: list[dict[str, Any]] = []
        self.headings: list[tuple[int, str]] = []
        self.parts: list[str] = []
        self.start = 0
        self.end = 0
        self.row = 0
        self.skip = 0
        self.heading: list[str] | None = None
        self.anchor = ""
        self.cells: list[str] | None = None
        self.header = True

    def line(self) -> int:
        """The source line of the current parser position."""
        return self.first + self.getpos()[0] - 1

    def flush(self) -> None:
        """Closes the current unit, keeping it when it holds text."""
        text = SPACE.sub(" ", "".join(self.parts)).strip()
        if text and self.cells is not None:
            self.cells.append(text)
        elif text:
            self.units.append(
                {
                    "start": self.start,
                    "end": self.end,
                    "kind": "prose",
                    "text": text,
                }
            )
        self.parts = []
        self.start = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        """Opens skipped regions, headings, rows, cells and blocks."""
        if tag in SKIPPED:
            self.skip += 1
        elif self.skip:
            return
        elif tag in HEADINGS:
            self.flush()
            self.heading = []
            self.anchor = dict(attrs).get("id") or ""
        elif tag == "tr":
            self.flush()
            self.cells = []
            self.header = True
            self.row = self.line()
        elif tag in CELLS:
            self.flush()
            self.header = self.header and tag == "th"
        elif tag in BLOCKS:
            self.flush()
        elif tag == "code":
            self.text("`")
        elif tag == "a":
            self.text("[")

    def handle_endtag(self, tag: str) -> None:
        """Closes what `handle_starttag` opened."""
        if tag in SKIPPED:
            self.skip = max(0, self.skip - 1)
        elif self.skip:
            return
        elif tag in HEADINGS and self.heading is not None:
            name = SPACE.sub(" ", "".join(self.heading)).strip()
            self.headings.append((self.line(), self.anchor or slugify(name)))
            self.heading = None
        elif tag == "tr" and self.cells is not None:
            self.close()
        elif tag in CELLS or tag in BLOCKS:
            self.flush()
        elif tag == "code":
            self.text("`")
        elif tag == "a":
            self.text("](link)")

    def close(self) -> None:
        """Ends a table row: one unit unless every cell is a header."""
        self.flush()
        cells, self.cells = self.cells or [], None
        if cells and not self.header:
            self.units.append(
                {
                    "start": self.row,
                    "end": self.line(),
                    "kind": "row",
                    "text": "| " + " | ".join(cells) + " |",
                }
            )

    def done(self) -> None:
        """Flushes whatever is still open at the end of the input."""
        if self.cells is not None:
            self.close()
        self.flush()

    def text(self, data: str) -> None:
        """Adds text to the open heading or unit."""
        if self.heading is not None:
            self.heading.append(data)
            return
        if not self.parts:
            self.start = self.line()
        self.parts.append(data)
        self.end = self.line() + data.count("\n")

    def handle_data(self, data: str) -> None:
        """Adds visible text outside skipped regions."""
        if not self.skip:
            self.text(data)


def read(text: str, first: int = 1) -> Reader:
    """Parses HTML into units and headings.

    Args:
        text: HTML source (a page body or a fragment).
        first: Line number of the first line of `text`.

    Returns:
        The finished reader.
    """
    reader = Reader(first)
    reader.feed(text)
    reader.done()
    return reader


def flatten(text: str) -> str:
    """The visible text of an HTML fragment as one line.

    Args:
        text: HTML of one cell or string.

    Returns:
        Its units' text joined by spaces.
    """
    return " ".join(u["text"].strip("| ") for u in read(text).units)


def tokens(code: str) -> Iterator[tuple[int, str, str]]:
    """The tokens of JavaScript that matter for finding its strings.

    Comments, whitespace and commas are skipped.

    Args:
        code: JavaScript source.

    Yields:
        (index, kind, text): kind `string` (a quoted literal, its body in
        `text`), `template`, `open`, `shut` or `other` (any other
        character).
    """
    index = 0
    while index < len(code):
        char = code[index]
        stop = index + 1
        kind = "other"
        if code.startswith(("//", "/*"), index):
            end = "\n" if code[index + 1] == "/" else "*/"
            found = code.find(end, index + 2)
            stop = len(code) if found < 0 else found + len(end)
            kind = ""
        elif char in "\"'":
            stop = close(code, index)
            kind = "string"
        elif char == "`":
            stop = close(code, index)
            kind = "template"
        elif char in OPENERS:
            kind = "open"
        elif char in OPENERS.values():
            kind = "shut"
        elif char.isspace() or char == ",":
            kind = ""
        if kind:
            yield index, kind, code[index:stop]
        index = stop


def scan(code: str, first: int) -> list[dict[str, Any]]:
    """Units from the quoted string literals of JavaScript.

    Args:
        code: JavaScript source (front matter or one expression).
        first: Line number of the first line of `code`.

    Returns:
        One `row` unit per array that holds only strings, one `prose`
        unit per other string with visible text.
    """
    units: list[dict[str, Any]] = []
    loose: dict[str, Any] = {"pure": False, "strings": []}
    stack = [loose]
    for index, kind, text in tokens(code):
        if kind == "string":
            line = first + code.count("\n", 0, index)
            stack[-1]["strings"].append((line, decode(text[1:-1])))
        elif kind == "open":
            stack[-1]["pure"] = False
            stack.append({"pure": text == "[", "strings": []})
        elif kind == "shut" and len(stack) > 1:
            units += gather(stack.pop())
        else:
            stack[-1]["pure"] = False
    while stack:
        units += gather(stack.pop())
    return units


def close(code: str, index: int) -> int:
    """The index just after the string literal opening at `index`."""
    quote = code[index]
    position = index + 1
    while position < len(code):
        if code[position] == "\\":
            position += 2
            continue
        if code[position] == quote:
            return position + 1
        if code[position] == "\n" and quote != "`":
            return position
        position += 1
    return len(code)


def decode(value: str) -> str:
    """A JavaScript string body with its common escapes resolved."""
    return re.sub(r"\\(.)", lambda m: ESCAPES.get(m[1], m[1]), value)


def gather(group: dict[str, Any]) -> list[dict[str, Any]]:
    """Units from one bracket group's direct strings."""
    strings = [(line, flatten(value)) for line, value in group["strings"]]
    strings = [(line, value) for line, value in strings if value]
    if not strings:
        return []
    if group["pure"] and len(strings) > 1:
        return [
            {
                "start": strings[0][0],
                "end": strings[-1][0],
                "kind": "row",
                "text": "| " + " | ".join(v for _, v in strings) + " |",
            }
        ]
    return [
        {"start": line, "end": line, "kind": "prose", "text": value}
        for line, value in strings
    ]


def expressions(markup: str) -> list[tuple[int, int]]:
    """Spans of the top-level `{...}` expressions of Astro markup."""
    spans = []
    index = 0
    while index < len(markup):
        if markup[index] != "{":
            index += 1
            continue
        depth = 0
        position = index
        while position < len(markup):
            char = markup[position]
            if char in "\"'`":
                position = close(markup, position)
                continue
            depth += {"{": 1, "}": -1}.get(char, 0)
            position += 1
            if depth == 0:
                break
        spans.append((index, position))
        index = position
    return spans


def units(text: str, name: str) -> list[dict[str, Any]]:
    """Every text unit of one site source file.

    Args:
        text: The file's source.
        name: Its path; `.astro` files have front matter and expressions,
            anything else is read as plain HTML.

    Returns:
        Units with `start`, `end`, `section`, `kind` and `lines` (one
        string), sorted by line.
    """
    found: list[dict[str, Any]] = []
    body = CODEBLOCK.sub(lambda m: blank(m[0]), text)
    if name.endswith(".astro"):
        front = FRONT.match(body)
        if front:
            found += scan(IMPORT.sub(lambda m: blank(m[0]), front[1]), 2)
            body = blank(front[0]) + body[front.end() :]
        for begin, stop in reversed(expressions(body)):
            first = 1 + body.count("\n", 0, begin)
            if not HEAD.search(body, max(0, begin - 16), begin):
                found += scan(body[begin + 1 : stop - 1], first)
            body = body[:begin] + blank(body[begin:stop]) + body[stop:]
    reader = read(body)
    found += reader.units
    marks = sorted(reader.headings)
    result = []
    for unit in sorted(found, key=lambda u: (u["start"], u["end"])):
        above = [slug for line, slug in marks if line <= unit["start"]]
        result.append(
            {
                "start": unit["start"],
                "end": unit["end"],
                "section": above[-1] if above else "",
                "kind": unit["kind"],
                "lines": [unit["text"]],
            }
        )
    return result
