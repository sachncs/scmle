"""Coverage and interval width or set size per declared slice.

The slice columns are declared in the charter's `uncertainty` block, so they
are part of the registered coverage claim and fixed before the final rows are
read. For each level of a slice the report gives the number of rows, the
share covered with its Clopper-Pearson interval and the mean width (regression)
or set size (classification).

The per-slice flag (`coverage.slice_flag`) replaces the Holm flag of
`coverage.slice_coverage`, whose measured familywise false-flag rate was
0.0755 against a bound of 0.0597: each slice is tested against the nominal
level less a margin fixed in advance (0.05) and Holm is applied over the
family. It gates only if the slice-flag entry of
`uncertainty_certificates.json` passed its pre-registered floors on the
current code, and then only for slices of at least the certified number of
rows; otherwise the report labels every flag
`descriptive` and says why. The flag never changes a tier.

A slice with fewer than
`coverage.MIN_SLICE_ROWS` rows is counted and not reported. A numeric column is
cut into `BANDS` bands at quantiles of the development rows, never of the
final rows; a column with more than `MAX_LEVELS` levels cannot be sliced.
"""

from collections.abc import Sequence
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scmle.data.detectors import certificates
from scmle.evaluation import coverage

MAX_LEVELS = 12
BANDS = 4
GATING = "gating"
DESCRIPTIVE = "descriptive"
GATE = "G5b"
STORE_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "qualification"
    / "uncertainty_certificates.json"
)
NOTE = (
    "per-slice coverage with its exact interval; the flag marks a slice "
    "whose coverage is below nominal by more than the margin (Holm over the "
    "family). The Holm flag against the nominal level itself missed its "
    "pre-registered bound (G5) and is not used"
)


@dataclasses.dataclass(frozen=True, slots=True)
class FlagGate:
    """Whether the per-slice coverage flag is certified.

    Attributes:
        status: `gating` or `descriptive`.
        reason: One sentence saying why.
        min_rows: Smallest slice (rows) at which detection was certified.
    """

    status: str
    reason: str
    min_rows: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Returns the JSON-ready record.

        Returns:
            A mapping of `status`, `reason` and `min_rows`.
        """
        return dataclasses.asdict(self)


def flag_method_hash() -> str:
    """Hash of the package code the slice-flag certificate measured.

    Returns:
        The code hash of `coverage.slice_flag`, `coverage.holm_rejections`
        and `coverage.undercoverage_p_value`.
    """
    return certificates.code_hash(
        coverage.slice_flag,
        coverage.holm_rejections,
        coverage.undercoverage_p_value,
    )


def flag_gate(path: Path | None = None) -> FlagGate:
    """Whether the stored slice-flag certificate is current and passing.

    The certificate file is not part of the installed package: without it the
    flag is descriptive.

    Args:
        path: Certificate file to read; the default store when None.

    Returns:
        A `gating` gate, with the smallest certified slice size, when the
        file has an entry for the flag whose method hash matches the current
        code and which passed. Otherwise a `descriptive` gate whose reason
        says whether the file, the entry, currency or the pass is missing.
    """
    target = STORE_PATH if path is None else path
    if not target.exists():
        return FlagGate(DESCRIPTIVE, "no coverage certificate file")
    entry = (
        json.loads(target.read_text(encoding="utf-8"))
        .get("gates", {})
        .get(GATE)
    )
    if entry is None:
        return FlagGate(DESCRIPTIVE, "no certificate for the slice flag")
    if entry.get("method_hash") != flag_method_hash():
        return FlagGate(
            DESCRIPTIVE, "stale: the code changed since it was measured"
        )
    if not entry.get("passed"):
        return FlagGate(
            DESCRIPTIVE, "the certificate missed its pre-registered floor"
        )
    return FlagGate(
        GATING,
        "false-flag rate and detection met their pre-registered floors",
        entry.get("min_certified_rows"),
    )


def labels_for(
    values: pd.Series, reference: pd.Series | None = None
) -> pd.Series | None:
    """Slice label of each row, or None when the column cannot be sliced.

    Args:
        values: The column on the rows being scored.
        reference: The same column on the development rows, where the band
            edges of a numeric column are taken.

    Returns:
        A series of labels aligned with `values`: `band k` (or `missing`)
        for a numeric column with more than `MAX_LEVELS` distinct values,
        otherwise the values as text; None if a column has more than
        `MAX_LEVELS` levels and is not numeric-banded.
    """
    seen = reference if reference is not None else values
    if pd.api.types.is_numeric_dtype(seen) and not pd.api.types.is_bool_dtype(
        seen
    ):
        clean = pd.to_numeric(seen, errors="coerce").dropna()
        if clean.nunique() > MAX_LEVELS:
            edges = np.unique(
                np.quantile(clean, np.linspace(0, 1, BANDS + 1)[1:-1])
            )
            numbers = pd.to_numeric(values, errors="coerce").to_numpy(float)
            bands = np.searchsorted(edges, numbers, side="right") + 1
            labels = [
                f"band {band}" if not np.isnan(n) else "missing"
                for band, n in zip(bands, numbers, strict=True)
            ]
            return pd.Series(labels, index=values.index)
    if seen.nunique() > MAX_LEVELS or values.nunique() > MAX_LEVELS:
        return None
    return values.astype(str)


def summary(
    inside: np.ndarray,
    sizes: np.ndarray,
    labels: Sequence[Any],
    nominal: float,
    level: float = 0.95,
    gate: FlagGate | None = None,
) -> dict[str, Any]:
    """Coverage, mean size and the margin flag of each slice level.

    Args:
        inside: Whether the interval or set held, per row.
        sizes: Interval width or set size, per row.
        labels: Slice label of each row.
        nominal: The requested coverage.
        level: Confidence level of each reported interval.
        gate: The flag's certificate state; read from the stored certificate
            when omitted.

    Returns:
        For each slice of at least `coverage.MIN_SLICE_ROWS` rows: its
        counts, coverage with interval, mean size (None if any size is
        infinite), flag, flag p-value and whether the flag is certified for
        its size. Also the flagged names, the margin, the gate, the number
        of skipped slices and a note; `descriptive` is True unless the gate
        is gating.
    """
    gate = flag_gate() if gate is None else gate
    flags_found = coverage.slice_flag(inside, labels, nominal)
    gating = gate.status == GATING
    flags = np.asarray(inside, dtype=bool)
    widths = np.asarray(sizes, dtype=float)
    names = np.asarray([str(v) for v in labels])
    found: list[dict[str, Any]] = []
    skipped = 0
    for name in sorted(set(names)):
        mask = names == name
        rows = int(mask.sum())
        if rows < coverage.MIN_SLICE_ROWS:
            skipped += 1
            continue
        hits = int(flags[mask].sum())
        lower, upper = coverage.clopper_pearson(hits, rows, level)
        chosen = widths[mask]
        found.append(
            {
                "slice": name,
                "rows": rows,
                "covered": hits,
                "coverage": hits / rows,
                "lower": lower,
                "upper": upper,
                "mean_size": None
                if np.isinf(chosen).any()
                else float(chosen.mean()),
                "flag": bool(flags_found["slices"][name]["flagged"]),
                "flag_p_value": flags_found["slices"][name]["p_value"],
                "flag_certified": bool(
                    gating and (gate.min_rows is None or rows >= gate.min_rows)
                ),
            }
        )
    return {
        "descriptive": not gating,
        "nominal": nominal,
        "level": level,
        "slices": found,
        "flagged": [item["slice"] for item in found if item["flag"]],
        "flag_margin": coverage.SLICE_MARGIN,
        "flag_gate": gate.to_dict(),
        "skipped_for_support": skipped,
        "note": NOTE,
    }
