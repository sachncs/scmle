"""Runs the admissibility tests and the detectors on `X` for a run.

This is the wiring between the feature packages and a charter run. It reads
the exploration view only and returns plain results; the caller
decides what to write and what to drop.

Certificates are stored beside the qualification harness
(`benchmarks/qualification/*.json`). That directory is not part of the
installed `scmle` package, so the stores are found by path relative to a
source checkout. When a store cannot be found or read, every detector it
covers is treated as descriptive and the reason is returned, never hidden:
descriptive suspicions raise warnings and cannot quarantine.

The charter declares no per-field provenance and no leakage ceilings, so
fields are undeclared (tier `unverified`) and decided in permissive mode:
an unverified field is admitted with a flag, never quarantined for lack of
provenance. The only route to quarantine in a charter run is a leakage
suspicion on a field with no as-of evidence, and only when the leakage
screen has a current, passing certificate for the mechanism that raised it.

The leakage test is the one used without a charter (`screen.py`: a
Benjamini-Yekutieli family over cross-fitted skill, missingness and record
purity), so a run with and without a charter flags the same columns. A
certified suspect that is a strong legitimate predictor is kept only by a
valid, ledgered per-column attestation (`attestations.py`, tier `attested`).
"""

from collections.abc import Sequence
import dataclasses
import datetime
import math
from pathlib import Path

import pandas

from scmle.contracts import qualification
from scmle.data.admissibility import independence
from scmle.data.admissibility import provenance
from scmle.data.admissibility import rules
from scmle.data.admissibility import screen
import scmle.data.admissibility.attestations
import scmle.data.admissibility.dag
import scmle.data.charter.causal
from scmle.data.detectors import certificates
from scmle.data.detectors import pipeline

STORE_DIRECTORY = (
    Path(__file__).resolve().parents[3] / "benchmarks" / "qualification"
)
F5_STORE = "detector_certificates.json"
SCREEN_STORE = "screen_certificates.json"
ATTESTATION_SOURCE = "f4-leakage-attestation"
DAG_SOURCE = "f4-dag-falsification"
DAG_ALPHA = 0.05


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
    """Reads a certificate store, or says why it could not be read.

    Args:
        name: File name of the store.
        directory: Folder holding it; defaults to the qualification folder
            of the source checkout.

    Returns:
        A `LoadedStore` holding the store, or None and a note saying
        that the file is missing or unreadable (the detectors it covers are
        then descriptive).
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
    """What the admissibility tests and the detectors found on `X`.

    Attributes:
        table: Admissibility decisions, or None when they were not run.
        leakage: The leakage screen report, or None when not run.
        detectors: Detector pipeline result.
        f5_store: The detector certificates used, or None.
        notes: Plain statements of what was not run or not certified.
        diagnostics: Ledger records (source, check). For a predictive
            estimand: one per screen finding and one per attestation that
            overrode a block. For a causal estimand: the checks of the DAG
            falsification, if it ran.
        dag: For a causal estimand with a declared DAG: its hash, the
            conditional-independence falsification on `X`, and the members of
            the declared adjustment set the graph rules refused. None
            otherwise.
    """

    table: rules.AdmissibilityTable | None
    leakage: screen.ScreenReport | None
    detectors: pipeline.PipelineResult
    f5_store: certificates.CertificateStore | None
    notes: tuple[str, ...]
    diagnostics: tuple[tuple[str, qualification.Check], ...] = ()
    dag: dict[str, object] | None = None

    @property
    def leakage_gate(self) -> certificates.GateStatus:
        """`GATING` when any screen mechanism has a passing certificate."""
        if self.leakage is not None and any(
            g is certificates.GateStatus.GATING
            for g in self.leakage.gates.values()
        ):
            return certificates.GateStatus.GATING
        return certificates.GateStatus.DESCRIPTIVE

    @property
    def suspects(self) -> dict[str, bool]:
        """Leakage suspicion per column that the screen flagged."""
        if self.leakage is None:
            return {}
        return {f.feature: True for f in self.leakage.suspects}

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
        return self.diagnostics


def finding_check(
    finding: screen.Finding, rows: int, seed: int, index: int
) -> qualification.Check:
    """Builds the ledger record of one screen finding.

    The record is a diagnostic: it carries the finding's p-value, never a
    claim. A suspect whose mechanism is not certified is demoted to a
    warning.

    Args:
        finding: The screen finding.
        rows: Number of rows the screen used.
        seed: Charter seed, recorded in the seed path.
        index: Position of the finding in the report, recorded in the seed
            path.

    Returns:
        A check on the exploration set. Its decision is `quarantine` for a
        suspect from a certified mechanism (demoted to a warning
        otherwise) and `pass` for a non-suspect. Its statistic is 0 when the
        finding's statistic is not finite, and its p-value is clipped to
        [0, 1].
    """
    if finding.suspect:
        decision = certificates.demote(
            qualification.Decision.QUARANTINE, finding.gate
        )
    else:
        decision = qualification.Decision.PASS
    statistic = finding.statistic if math.isfinite(finding.statistic) else 0.0
    return qualification.Check(
        check_id=f"leakage.screen.{finding.mechanism}:{finding.feature}",
        null_hypothesis=(
            f"'{finding.feature}' has no {screen.WORDS[finding.mechanism]} "
            "with the target"
        ),
        statistic=statistic,
        p_value=min(max(finding.p_value, 0.0), 1.0),
        interval=None,
        effect=statistic,
        n=rows,
        decision=decision,
        set_id=qualification.SetId.EXPLORATION,
        seed_path=(seed, "leakage.screen", index),
        tier=qualification.Tier.UNVERIFIED,
    )


def attestation_check(
    item: scmle.data.admissibility.attestations.LeakageAttestation,
    findings: Sequence[screen.Finding],
    rows: int,
    seed: int,
    index: int,
) -> qualification.Check:
    """Builds the ledger record of an attestation that overrode a block.

    Its p-value is the smallest p-value of the screen for the column, so the
    record shows what was overridden; its tier is `attested`.

    Args:
        item: The attestation that was applied.
        findings: Screen findings for the attested column.
        rows: Number of rows the screen used.
        seed: Charter seed, recorded in the seed path.
        index: Position of the attestation among those applied, recorded in
            the seed path.

    Returns:
        A warning-level check on the exploration set whose statistic and
        effect are the largest finite statistic among `findings` (0 if none) and
        whose p-value is the smallest (1 if none), clipped to [0, 1].
    """
    given = item.attestation
    p_value = min((f.p_value for f in findings), default=1.0)
    statistic = max(
        (f.statistic for f in findings if math.isfinite(f.statistic)),
        default=0.0,
    )
    return qualification.Check(
        check_id=f"leakage.attestation:{item.column}",
        null_hypothesis=(
            f"'{item.column}' is a legitimate predictor, known "
            f"{item.lead_seconds:g} s before the prediction moment: attested "
            f"by {given.name_or_id} ({given.role}) on {given.date}, valid to "
            f"{given.expires}, evidence: {given.evidence}"
        ),
        statistic=statistic,
        p_value=min(max(p_value, 0.0), 1.0),
        interval=None,
        effect=statistic,
        n=rows,
        decision=qualification.Decision.WARN,
        set_id=qualification.SetId.EXPLORATION,
        seed_path=(seed, "leakage.attestation", index),
        tier=qualification.Tier.ATTESTED,
    )


def causal_admissibility(
    frame: pandas.DataFrame,
    candidates: Sequence[str],
    spec: scmle.data.charter.causal.CausalSpec,
    graph: scmle.data.admissibility.dag.Dag,
    seed: int,
    day: datetime.date,
    result: pipeline.PipelineResult,
    store: certificates.CertificateStore | None,
    notes: list[str],
) -> Qualification:
    """Applies the graph rules to every candidate and tests the graph.

    Each candidate gets `rules.decide_causal` (descendants of the treatment
    and colliders are quarantined), the declared adjustment set is checked
    member by member, and the graph's implied independences are tested on
    `X` only. More candidates than `MAX_SEARCH_CANDIDATES` with no declared
    adjustment set adds a note that the claim is `attested` at best.

    Args:
        frame: Exploration data with the candidates and the target.
        candidates: Candidate covariate columns.
        spec: The charter's causal block.
        graph: The declared DAG hypothesis.
        seed: Charter seed.
        day: Day on which attestations must be valid.
        result: Detector pipeline result to carry into the return value.
        store: Detector certificates used, to carry into the return value.
        notes: Notes so far. This list is appended to in place when the graph
            rules or the falsification cannot run, when the declared
            adjustment set holds a refused member, when there are too many
            candidates and no declared adjustment set, or when the graph has
            falsified implications.

    Returns:
        The qualification with the admissibility table (None when the graph
        rules could not be applied), no leakage report, and the `dag`
        report holding the graph hash, the refused adjustment members and
        the falsification outcome. Its diagnostics are the checks of the
        falsification, if it ran.
    """
    records = provenance.ProvenanceTable.covering([str(c) for c in candidates])
    members = scmle.data.charter.causal.covariates_of(spec.assumptions) or ()
    try:
        table: rules.AdmissibilityTable | None = rules.build_table(
            records,
            "causal",
            rules.Mode.PERMISSIVE,
            today=day,
            graph=graph,
            treatment=spec.treatment,
            outcome=spec.outcome,
        )
    except ValueError as error:
        table = None
        notes.append(f"graph rules were not applied: {error}")
    refused = {
        d.name: "; ".join(d.reasons)
        for d in (() if table is None else table.decisions)
        if d.name in members and d.admission is rules.Admission.QUARANTINE
    }
    if refused:
        notes.append(
            "backdoor_aipw is ineligible: the declared adjustment set holds "
            + ", ".join(sorted(refused))
        )
    if (
        not members
        and len(candidates) > scmle.data.admissibility.dag.MAX_SEARCH_CANDIDATES
    ):
        notes.append(
            "more than "
            f"{scmle.data.admissibility.dag.MAX_SEARCH_CANDIDATES} candidate "
            "covariates and no declared adjustment set: the charter must "
            "supply the set, and the claim is attested at best"
        )
    report: dict[str, object] = {
        "dag_hash": graph.digest(),
        "adjustment_members_refused": refused,
    }
    diagnostics: tuple[tuple[str, qualification.Check], ...] = ()
    try:
        found = independence.falsify(
            graph, frame, alpha=DAG_ALPHA, run_seed=seed
        )
    except (ValueError, KeyError) as error:
        report["falsification"] = {"ran": False, "reason": str(error)}
        notes.append(f"the declared DAG could not be tested on X: {error}")
    else:
        report["falsification"] = {
            "ran": True,
            "alpha": found.alpha,
            "level": found.level,
            "tested": sum(
                o.verdict is not independence.Verdict.SKIPPED
                for o in found.outcomes
            ),
            "falsified": [o.statement.text() for o in found.falsified],
            "skipped_nodes": list(found.skipped_nodes),
            "note": "a graph can only survive, never be confirmed",
        }
        diagnostics = tuple((DAG_SOURCE, check) for check in found.checks)
        if found.falsified:
            notes.append(
                f"{len(found.falsified)} implication(s) of the declared DAG "
                "were falsified on X"
            )
    return Qualification(
        table, None, result, store, tuple(notes), diagnostics, report
    )


def run(
    features: pandas.DataFrame,
    target: pandas.Series,
    *,
    kind: str,
    classification: bool,
    seed: int,
    structure: Sequence[str] = (),
    group_column: str | None = None,
    time_column: str | None = None,
    directory: Path | None = None,
    attestations: Sequence[
        scmle.data.admissibility.attestations.LeakageAttestation
    ] = (),
    scope: str = "",
    today: datetime.date | None = None,
    causal: scmle.data.charter.causal.CausalSpec | None = None,
    dag: scmle.data.admissibility.dag.Dag | None = None,
    declared: Sequence[provenance.FieldProvenance] = (),
) -> Qualification:
    """Runs leakage, admissibility and the detector pipeline on `X`.

    Args:
        features: Exploration features.
        target: Exploration target.
        kind: Estimand kind from the charter, `predictive` or `causal`; any
            value other than `predictive` takes the causal path.
        classification: Whether the target holds classes.
        seed: Charter seed.
        structure: Group and time columns, which are not candidates.
        group_column: Entity column, so permutations keep entities whole.
        time_column: Time column, so permutations stay within time blocks.
        directory: Folder of the certificate stores, for tests. The detector
            store and the screen store are read from it.
        attestations: Per-column statements that a strong predictor is
            legitimate; each counts only if its scope equals `scope` and it
            is valid on `today`.
        scope: Charter hash the attestations must name.
        today: Day on which attestations must be valid; defaults to the
            current date.
        causal: The charter's causal block, for a causal estimand.
        dag: The DAG hypothesis the causal block declares, already loaded.
        declared: Provenance records for columns the run derived (for
            example synthesized features); a record for a column that is
            not a candidate, or that is covered by an applied attestation,
            is ignored. The other columns stay undeclared. Used for a
            predictive estimand only.

    Returns:
        The results. A predictive estimand gets the leakage screen, the
        admissibility table and the ledger diagnostics. A causal estimand
        with a causal block and a declared graph gets the graph rules
        (`rules.decide_causal`) on every candidate and the falsification of
        the graph on `X`; without both, its admissibility table and leakage
        report are None and `notes` says so.
    """
    notes: list[str] = []
    f5 = load_store(F5_STORE, directory)
    notes.extend(n for n in (f5.note,) if n)
    frame = features.assign(**{str(target.name): target})
    result = pipeline.run(
        frame, str(target.name), f5.store, seed, provenance=None
    )
    candidates = [c for c in features.columns if c not in set(structure)]
    day = today or datetime.date.today()
    if kind != "predictive":
        if causal is None or dag is None:
            notes.append(
                f"{kind} estimand without a declared graph: admissibility "
                "not run"
            )
            return Qualification(None, None, result, f5.store, tuple(notes))
        return causal_admissibility(
            frame, candidates, causal, dag, seed, day, result, f5.store, notes
        )
    task = "classification" if classification else "regression"
    store = (
        screen.shipped_store()
        if directory is None
        else screen.load_store(directory / SCREEN_STORE)
    )
    if store is None:
        notes.append("screen certificates unusable: leakage is descriptive")
    report = screen.screen_table(
        frame,
        str(target.name),
        task,
        candidates,
        seed,
        group_column,
        time_column,
        store=store or certificates.CertificateStore(),
    )
    if not any(
        g is certificates.GateStatus.GATING for g in report.gates.values()
    ):
        notes.append("leakage suspicion is descriptive: no certificate gates")
    notes.extend(report.notes)
    applied = scmle.data.admissibility.attestations.apply(
        attestations, scope, day.isoformat()
    )
    for column, why in applied.rejected.items():
        notes.append(f"attestation for '{column}' ignored: {why}")
    suspect_columns = {f.feature for f in report.suspects}
    used = {
        column: item
        for column, item in applied.valid.items()
        if column in suspect_columns and column in candidates
    }
    for column in applied.valid:
        if column not in used:
            notes.append(
                f"attestation for '{column}' unused: the screen did not flag it"
            )
    attested = {i.field().name for i in used.values()}
    names = {str(c) for c in candidates}
    records = provenance.ProvenanceTable.covering(
        [str(c) for c in candidates],
        [
            *(i.field() for i in used.values()),
            *(
                f
                for f in declared
                if f.name in names and f.name not in attested
            ),
        ],
    )
    gates = {f.feature: certificates.GateStatus.GATING for f in report.blocking}
    table = rules.build_table(
        records,
        kind,
        rules.Mode.PERMISSIVE,
        today=day,
        suspects=dict.fromkeys(suspect_columns, True),
        leakage_gate=gates,
    )
    diagnostics: list[tuple[str, qualification.Check]] = [
        (
            screen.DETECTOR_ID,
            finding_check(f, report.rows, seed, index),
        )
        for index, f in enumerate(report.findings)
    ]
    diagnostics.extend(
        (
            ATTESTATION_SOURCE,
            attestation_check(
                item,
                [f for f in report.findings if f.feature == column],
                report.rows,
                seed,
                index,
            ),
        )
        for index, (column, item) in enumerate(sorted(used.items()))
    )
    return Qualification(
        table,
        report,
        result,
        f5.store,
        tuple(notes),
        tuple(diagnostics),
    )
