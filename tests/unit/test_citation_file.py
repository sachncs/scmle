"""`CITATION.cff` carries the fields GitHub and `cffconvert` require.

`cffconvert` is not a dependency, so the required keys of CFF 1.2.0 are
checked here and the version is compared with the package's.
"""

from pathlib import Path
import re

import scmle

ROOT = Path(__file__).parents[2]
REQUIRED = ("cff-version", "message", "title", "authors")


def top_level(text: str) -> dict[str, str]:
    """Top-level `key: value` lines of a flat YAML document."""
    found = {}
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z-]+):\s*(.*)$", line)
        if match:
            found[match.group(1)] = match.group(2).strip().strip('"')
    return found


def test_citation_file_has_the_required_keys_and_the_package_version():
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    keys = top_level(text)
    for name in REQUIRED:
        assert name in keys
    assert keys["cff-version"] == "1.2.0"
    assert keys["version"] == scmle.__version__
    assert keys["license"] == "Apache-2.0"
    assert keys["repository-code"] == "https://github.com/sachncs/scmle"
    assert re.search(r"^  - family-names: ", text, flags=re.MULTILINE)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", keys["date-released"])
