"""String similarity and phonetic codes used to compare and block records."""

import math
import re
from typing import Any

CLEAN = re.compile(r"[^a-z0-9]+")


SOUNDEX_CODES = {
    **dict.fromkeys("bfpv", "1"),
    **dict.fromkeys("cgjkqsxz", "2"),
    **dict.fromkeys("dt", "3"),
    "l": "4",
    **dict.fromkeys("mn", "5"),
    "r": "6",
}


def normalize(value: Any) -> str:
    """Normalizes a value for comparison.

    Args:
        value: Any value; None and NaN count as missing.

    Returns:
        The lower-case alphanumerics of the value as text, or an empty
        string for a missing value.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return CLEAN.sub("", str(value).lower())


def jaro(a: str, b: str) -> float:
    """Computes the Jaro similarity of two strings.

    Args:
        a: First string.
        b: Second string.

    Returns:
        A similarity in [0, 1]; 1.0 for equal strings and 0.0 when either
        is empty (and they differ) or no characters match.
    """
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    reach = max(max(len(a), len(b)) // 2 - 1, 0)
    taken = [False] * len(b)
    left = []
    for i, ch in enumerate(a):
        for j in range(max(0, i - reach), min(len(b), i + reach + 1)):
            if not taken[j] and b[j] == ch:
                taken[j] = True
                left.append(ch)
                break
    if not left:
        return 0.0
    right = [b[j] for j in range(len(b)) if taken[j]]
    swaps = sum(x != y for x, y in zip(left, right, strict=True)) / 2
    m = len(left)
    return (m / len(a) + m / len(b) + (m - swaps) / m) / 3.0


def jarowinkler(a: str, b: str) -> float:
    """Computes the Jaro-Winkler similarity with the usual 0.1 prefix scale.

    Args:
        a: First string.
        b: Second string.

    Returns:
        The Jaro similarity raised for a common prefix of up to four
        characters; in [0, 1].
    """
    base = jaro(a, b)
    prefix = 0
    for x, y in zip(a[:4], b[:4], strict=False):
        if x != y:
            break
        prefix += 1
    return base + prefix * 0.1 * (1.0 - base)


def soundex(text: str) -> str:
    """Computes the American Soundex code of the letters in a text.

    Args:
        text: The text; non-letters are ignored.

    Returns:
        A four-character code (a letter and three digits), or an empty
        string when the text has no letters.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return ""
    out = letters[0].upper()
    last = SOUNDEX_CODES.get(letters[0], "")
    for ch in letters[1:]:
        code = SOUNDEX_CODES.get(ch, "")
        if code and code != last:
            out += code
        if ch not in "hw":
            last = code
    return (out + "000")[:4]
