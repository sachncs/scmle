"""Runs the F4 admissibility tests and the F5 detectors on `X` for a run.

This is the wiring between the feature packages and a charter run. It reads
the exploration view only (canon P3) and returns plain results; the caller
decides what to write and what to drop.

Certificates are stored beside the qualification harness
(`benchmarks/qualification/*.json`). That directory is not part of the
installed `scmle` package, so the stores are found by path relative to a
source checkout. When a store cannot be found or read, every detector it
covers is treated as descriptive and the reason is returned, never hidden
(canon P1): descriptive suspicions raise warnings and cannot quarantine.

The charter declares no per-field provenance and no leakage ceilings, so
fields are undeclared (tier `unverified`) and decided in permissive mode:
an unverified field is admitted with a flag, never quarantined for lack of
provenance. The only route to quarantine in a charter run is a leakage
suspicion on a field with no as-of evidence, and only when the leakage
detector has a current, passing certificate for the settings used.
"""

from collections.abc import Sequence
import dataclasses
from pathlib import Path

import pandas as pd

from scmle.contracts import qualification
from scmle.data.admissibility import leakage
from scmle.data.admissibility import provenance
from scmle.data.admissibility import rules
from scmle.data.detectors import certificates
from scmle.data.detectors import pipeline

STORE_DIRECTORY = (
    Path(__file__).resolve().parents[3] / "benchmarks" / "qualification"
)
F4_STORE = "f4_certificates.json"
F5_STORE = "detector_certificates.json"
BLOCKS = 20


@dataclasses.dataclass(frozen=True)
class LoadedStore:
    """A certificate store and what to say when there is none.

    Attributes:
        store: The certificates, or None when none could be used.
        note: Why `store` is None, or an empty string.
    """

    store: certificates.CertificateStore | None
    note: str = ""


def load_store(name: str, directory: Path | None = None) -> LoadedStore:
    """Reads a certificate store, or says why it could not.

    Args:
        name: File name of the store.
        directory: Folder holding it; defaults to the qualification folder
            of the source checkout.

    Returns:
        The store, or None with a reason.
    """
    path = (directory or STORE_DIRECTORY) / name
    if not path.is_file():
        return LoadedStore(
            None, f"{name} not found at {path}: detectors are descriptive"
        )
    try:
        return LoadedStore(certificates.CertificateStore.load(path))
    except (ValueError, KeyError, OSError) as error:
        return LoadedStore(
            None, f"{name} unreadable ({error}): detectors are descriptive"
        )


@dataclasses.dataclass(frozen=True)
class Qualification:
    """What the F4 and F5 stages found on `X`.

    Attributes:
        table: Admissibility decisions, or None when they were not run.
        leakage: Leakage suspicions, or None when not run.
        leakage_gate: Whether the leakage detector may quarantine.
        detectors: F5 pipeline result.
        f5_store: The F5 certificates used, or None.
        notes: Plain statements of what was not run or not certified.
    """

    table: rules.AdmissibilityTable | None
    leakage: leakage.LeakageReport | None
    leakage_gate: certificates.GateStatus
    detectors: pipeline.PipelineResult
    f5_store: certificates.CertificateStore | None
    notes: tuple[str, ...]

    @property
    def quarantined(self) -> dict[str, str]:
        """Quarantined field -> reasons, joined."""
        if self.table is None:
            return {}
        return {
            d.name: "; ".join(d.reasons)
            for d in self.table.decisions
            if d.admission is rules.Admission.QUARANTINE
        }

    @property
    def checks(self) -> tuple[tuple[str, qualification.Check], ...]:
        """Every (source, check) pair for the ledger."""
        if self.leakage is None:
            return ()
        return tuple((leakage.DETECTOR_ID, c) for c in self.leakage.checks)


def run(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    kind: str,
    classification: bool,
    seed: int,
    structure: Sequence[str] = (),
    group_column: str | None = None,
    time_column: str | None = None,
    directory: Path | None = None,
) -> Qualification:
    """Runs leakage, admissibility and the detector pipeline on `X`.

    Args:
        features: Exploration features.
        target: Exploration target.
        kind: Estimand kind from the charter, `predictive` or `causal`.
        classification: Whether the target holds classes.
        seed: Charter seed.
        structure: Group and time columns, which are not candidates.
        group_column: Entity column, so permutations keep entities whole.
        time_column: Time column, so permutations stay within time blocks.
        directory: Folder of the certificate stores, for tests.

    Returns:
        The results. A causal estimand has no declared graph in a charter,
        so its admissibility table is skipped and said so in `notes`.
    """
    notes: list[str] = []
    f4 = load_store(F4_STORE, directory)
    f5 = load_store(F5_STORE, directory)
    notes.extend(n for n in (f4.note, f5.note) if n)
    frame = features.assign(**{str(target.name): target})
    result = pipeline.run(
        frame, str(target.name), f5.store, seed, provenance=None
    )
    candidates = [c for c in features.columns if c not in set(structure)]
    if kind != "predictive":
        notes.append(
            f"{kind} estimand without a declared graph: admissibility not run"
        )
        return Qualification(
            None,
            None,
            certificates.GateStatus.DESCRIPTIVE,
            result,
            f5.store,
            tuple(notes),
        )
    cfg = dataclasses.replace(leakage.CERTIFIED, run_seed=seed)
    task = "classification" if classification else "regression"
    gate = (
        certificates.GateStatus.DESCRIPTIVE
        if f4.store is None
        else leakage.gate_status(f4.store, cfg, task, int(target.nunique()))
    )
    if gate is not certificates.GateStatus.GATING:
        notes.append("leakage suspicion is descriptive: no certificate gates")
    groups = (
        features[group_column].astype(str)
        if group_column and group_column in features.columns
        else None
    )
    block_size = (
        max(2, len(features) // BLOCKS)
        if groups is None and time_column in features.columns
        else None
    )
    report = leakage.scan(
        frame,
        str(target.name),
        task,
        cfg,
        features=candidates,
        groups=groups,
        block_size=block_size,
        gate=gate,
    )
    records = provenance.ProvenanceTable.covering([str(c) for c in candidates])
    table = rules.build_table(
        records,
        kind,
        rules.Mode.PERMISSIVE,
        suspects=report.suspects,
        leakage_gate=gate,
    )
    return Qualification(table, report, gate, result, f5.store, tuple(notes))
