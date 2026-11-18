"""Language detection and page and section segmentation of text."""

import re

from scmle.data.modalities.document.elements import Span

DEVANAGARI_MIN_SHARE = 0.25
ASCII_LETTER_MIN_SHARE = 0.60


def detect(content: str) -> str:
    """Returns a conservative language code using Unicode script ratios.

    Args:
        content: Text to classify.

    Returns:
        ``hi`` if at least 25% of the letters are Devanagari, else ``en`` if
        at least 60% are ASCII, else ``unknown`` (also when there are no
        letters).
    """
    letters = [character for character in content if character.isalpha()]
    if not letters:
        return "unknown"
    devanagari = sum("\u0900" <= character <= "\u097f" for character in letters)
    ascii_letters = sum(character.isascii() for character in letters)
    if devanagari / len(letters) >= DEVANAGARI_MIN_SHARE:
        return "hi"
    if ascii_letters / len(letters) >= ASCII_LETTER_MIN_SHARE:
        return "en"
    return "unknown"


def segment(content: str) -> tuple[Span, ...]:
    """Builds page and Markdown-heading spans from character offsets.

    Args:
        content: Document text; form feeds separate pages.

    Returns:
        Page spans for every non-empty page, followed by section spans for
        each Markdown heading (levels 1 to 6). A section runs from its
        heading to the next heading or the end, and carries the page number
        of its heading.
    """
    spans: list[Span] = []
    page_starts = [0]
    page_ends: list[int] = []
    for match in re.finditer("\f", content):
        page_ends.append(match.start())
        page_starts.append(match.end())
    page_ends.append(len(content))
    for page_number, (start, end) in enumerate(
        zip(page_starts, page_ends, strict=True), start=1
    ):
        if end > start:
            spans.append(
                Span(
                    kind="page",
                    start=start,
                    end=end,
                    label=f"page {page_number}",
                    page_number=page_number,
                )
            )

    headings = list(
        re.finditer(
            r"(?m)(?:^|(?<=\f))[^\S\f\n]{0,3}#{1,6}[^\S\f\n]+([^\f\n]+?)\s*(?=\f|$)",
            content,
        )
    )
    for index, heading in enumerate(headings):
        end = (
            headings[index + 1].start()
            if index + 1 < len(headings)
            else len(content)
        )
        label = heading.group(1).strip()
        page_number = content.count("\f", 0, heading.start()) + 1
        spans.append(
            Span(
                kind="section",
                start=heading.start(),
                end=end,
                label=label,
                page_number=page_number,
            )
        )
    return tuple(spans)
