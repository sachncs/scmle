"""The pre-registrations of plan section 6 agree with `acceptance.toml`.

Each claim C1 to C14 has `docs/preregistration/c<N>.md` with the sections
the plan asks for. Every sample-size row
`variable | rule | true rate | alpha | minimum | planned` is recomputed
from `benchmarks.acceptance`, so a changed threshold that would make a
registration wrong fails here.
"""

import pathlib
import re

import pytest

from benchmarks import acceptance

ROOT = pathlib.Path(__file__).resolve().parents[2]
FOLDER = ROOT / "docs" / "preregistration"
CLAIMS = range(1, 15)
SECTIONS = (
    "## Criterion",
    "## Population and data",
    "## Split",
    "## Statistic",
    "## Decision rule",
    "## Sample size",
    "## What counts as a failure",
    "## Grader",
    "## Where the current module differs",
)
ROW = re.compile(
    r"^\| `(\w+)` \| (floor|ceiling) \| ([\d.]+) \| ([\d.]+) \| (\d+) \| "
    r"(\d+) \|$",
    re.MULTILINE,
)
NAME = re.compile(r"`([a-z]+(?:_[a-z]+)+)`")


def text(claim: int) -> str:
    """The registration of one claim."""
    return (FOLDER / f"c{claim}.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("claim", CLAIMS)
def test_every_claim_has_every_section(claim):
    found = text(claim)
    assert found.startswith(f"# C{claim} ")
    assert all(s in found for s in SECTIONS)


def test_the_index_lists_every_claim():
    index = (FOLDER / "README.md").read_text(encoding="utf-8")
    assert all(f"[c{c}.md](c{c}.md)" in index for c in CLAIMS)


@pytest.mark.parametrize("claim", CLAIMS)
def test_sample_sizes_follow_from_acceptance(claim):
    variables = acceptance.load()
    power = variables["power"].value
    for name, kind, truth, alpha, least, planned in ROW.findall(text(claim)):
        rule = variables[name].rule
        assert rule is not None and kind == type(rule).__name__.lower()
        assert rule.minimum(float(alpha)) == int(least)
        assert rule.plan(float(truth), float(alpha), power) == int(planned)


@pytest.mark.parametrize("claim", CLAIMS)
def test_named_thresholds_exist(claim):
    variables = acceptance.load()
    names = {n for n in NAME.findall(text(claim)) if n in variables}
    quoted = {
        n
        for n in NAME.findall(text(claim))
        if n.endswith(("_min", "_max", "_tol", "_margin"))
    }
    assert quoted <= names
