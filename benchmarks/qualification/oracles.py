"""Frozen reference outputs and the comparison used against them.

An oracle file holds fixed inputs and the outputs that an independent tool
(scipy, here) computed for them, with the tool versions, the hash of the
generator that wrote it and a hash of its own content. Loading verifies the
content hash, so a hand-edited reference is rejected. Regenerating a file is
a reviewed change (`benchmarks/qualification/make_oracles.py`, run offline).

Agreement shows that our code matches the reference, not that either is
true: the reference can share a modelling assumption with ours.
"""

from collections.abc import Mapping
import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ORACLE_DIR = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "oracles"
)
GENERATOR = Path(__file__).with_name("make_oracles.py")
NAMES = (
    "chi2_contingency",
    "clopper_pearson",
    "fdr_control",
    "fisher_exact",
    "harmonic_number",
    "kendalltau",
    "kruskal",
    "ks_2samp",
    "mannwhitneyu",
    "pearsonr",
    "spearmanr",
    "wilson_interval",
)
# Reference outputs that need R. They are not generated here, and no file
# stands in for them: tests of the rules below stay absent until a reviewer
# generates and freezes real R output.
PENDING_R: Mapping[str, str] = {
    "onlineFDR": (
        "R is unavailable in this environment; freeze LORD++, SAFFRON and "
        "ADDIS decisions from the R `onlineFDR` package offline."
    ),
    "knockoff": (
        "R is unavailable in this environment; freeze selection sets from "
        "the R `knockoff` package offline."
    ),
}
FIELDS = frozenset(
    {
        "oracle",
        "reference",
        "tool",
        "generator",
        "generator_sha256",
        "tolerance",
        "cases",
        "content_sha256",
    }
)


class OracleError(ValueError):
    """An oracle file is missing, malformed or fails its integrity check."""


@dataclasses.dataclass(frozen=True)
class Oracle:
    """A verified oracle file.

    Attributes:
        name: Oracle name, equal to the file stem.
        reference: What produced the expected values.
        tool: Versions of the tools that produced them.
        generator_sha256: Hash of the generator source at write time.
        tolerance: Stated relative tolerance for comparisons.
        cases: Fixed inputs with their expected outputs.
        content_sha256: Hash of everything above.
    """

    name: str
    reference: str
    tool: Mapping[str, str]
    generator_sha256: str
    tolerance: Mapping[str, float]
    cases: tuple[Mapping[str, Any], ...]
    content_sha256: str


def canonical(record: Mapping[str, Any]) -> str:
    """Stable JSON used for hashing and for the files themselves."""
    return json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def content_digest(record: Mapping[str, Any]) -> str:
    """SHA-256 of a record without its own `content_sha256`."""
    body = {k: v for k, v in record.items() if k != "content_sha256"}
    return hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()


def generator_sha256() -> str:
    """SHA-256 of the generator's current source file."""
    return hashlib.sha256(GENERATOR.read_bytes()).hexdigest()


def load_oracle(name: str, directory: Path = ORACLE_DIR) -> Oracle:
    """Reads an oracle file and verifies its integrity.

    Args:
        name: One of `NAMES`.
        directory: Folder holding the files.

    Returns:
        The verified oracle.

    Raises:
        OracleError: If the name is unknown, the file is absent or has
            unexpected or missing fields, or its content hash is wrong.
    """
    if name not in NAMES:
        raise OracleError(f"'{name}' is not a known oracle")
    path = directory / f"{name}.json"
    if not path.is_file():
        raise OracleError(f"no such oracle file: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    extra = sorted(set(record) - FIELDS)
    if extra:
        raise OracleError(f"unexpected field(s) in {path.name}: {extra}")
    if "content_sha256" not in record:
        raise OracleError(f"{path.name} has no content_sha256")
    missing = sorted(FIELDS - set(record))
    if missing:
        raise OracleError(f"{path.name} lacks field(s): {missing}")
    if record["content_sha256"] != content_digest(record):
        raise OracleError(f"{path.name} fails its content hash check")
    if record["oracle"] != name:
        raise OracleError(f"{path.name} declares oracle '{record['oracle']}'")
    return Oracle(
        name=name,
        reference=record["reference"],
        tool=record["tool"],
        generator_sha256=record["generator_sha256"],
        tolerance=record["tolerance"],
        cases=tuple(record["cases"]),
        content_sha256=record["content_sha256"],
    )


def is_number(value: Any) -> bool:
    """Whether `value` is a real number (not a bool)."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def differences(
    actual: Any,
    expected: Any,
    rel_tol: float,
    abs_tol: float,
    path: str,
) -> list[tuple[str, str, float]]:
    """Disagreements as (path, description, error); empty when equal."""
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            return [(path, "keys differ", math.inf)]
        found: list[tuple[str, str, float]] = []
        for key in sorted(expected):
            found += differences(
                actual[key], expected[key], rel_tol, abs_tol, f"{path}.{key}"
            )
        return found
    if isinstance(expected, list | tuple):
        if not isinstance(actual, list | tuple) or len(actual) != len(expected):
            return [(path, "shape differs", math.inf)]
        found = []
        for position, (a, e) in enumerate(zip(actual, expected, strict=True)):
            found += differences(a, e, rel_tol, abs_tol, f"{path}[{position}]")
        return found
    if not is_number(actual) or not is_number(expected):
        return [] if actual == expected else [(path, "values differ", math.inf)]
    a, e = float(actual), float(expected)
    if math.isnan(a) or math.isnan(e):
        return (
            [] if math.isnan(a) and math.isnan(e) else [(path, "nan", math.inf)]
        )
    if math.isinf(a) or math.isinf(e):
        return [] if a == e else [(path, "infinity differs", math.inf)]
    gap = abs(a - e)
    if gap <= rel_tol * abs(e) + abs_tol:
        return []
    error = gap / abs(e) if e != 0.0 else math.inf
    return [(path, f"{a!r} vs {e!r}", error)]


def assert_agrees(
    actual: Any,
    expected: Any,
    label: str = "",
    rel_tol: float = 1e-8,
    abs_tol: float = 0.0,
) -> None:
    """Asserts nested numbers agree to a relative tolerance.

    A pair agrees when `|a - e| <= rel_tol * |e| + abs_tol`, so an expected
    zero needs `abs_tol` above zero for anything but an exact zero. Two NaNs
    or two equal infinities agree.

    Args:
        actual: Value from our code (number, list or mapping of numbers).
        expected: Reference value of the same shape.
        label: Name used in the failure message.
        rel_tol: Relative tolerance, default 1e-8.
        abs_tol: Absolute tolerance, default 0.

    Raises:
        AssertionError: Naming the count and the worst disagreement.
    """
    found = differences(actual, expected, rel_tol, abs_tol, "")
    if not found:
        return
    worst = max(found, key=lambda item: item[2])
    where = worst[0] or "<root>"
    raise AssertionError(
        f"{label or 'values'}: {len(found)} disagreement(s); worst at "
        f"{where}: {worst[1]} (rel_tol={rel_tol}, abs_tol={abs_tol})"
    )
