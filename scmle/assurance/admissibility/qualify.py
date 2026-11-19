"""Runs the admissibility tests and the detectors on `X` for a run.

This is the wiring between the feature packages and a charter run. It reads
the exploration view only and returns plain results; the caller
decides what to write and what to drop.

Certificates are written by the qualification harness and shipped with the
package (`scmle/certificates/*.json`), so an installed wheel reads the same
stores as a source checkout. When a store cannot be found or read, every
detector it covers is treated as descriptive and the reason is returned,
never hidden: descriptive suspicions raise warnings and cannot quarantine.

The charter declares no per-field provenance and no leakage ceilings, so
fields are undeclared (tier `unverified`) and decided in permissive mode:
an unverified field is admitted with a flag, never quarantined for lack of
provenance. The only route to quarantine in a charter run is a leakage
suspicion on a field with no as-of evidence.

The leakage test is the one used without a charter (`screen.py`: a
Benjamini-Yekutieli family over cross-fitted skill, missingness and record
purity), so a run with and without a charter flags the same columns. A
certified suspect that is a strong legitimate predictor is kept only by a
valid, ledgered per-column attestation (`attestations.py`, tier `attested`).
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import datetime
import math
from pathlib import Path

import pandas

from scmle.assurance.admissibility import independence
from scmle.assurance.admissibility import provenance
from scmle.assurance.admissibility import rules
from scmle.assurance.admissibility import screen
import scmle.assurance.admissibility.attestations
import scmle.assurance.admissibility.dag
import scmle.assurance.charter.causal
from scmle.assurance.detectors import certificates
from scmle.assurance.detectors import pipeline
import scmle.certificates
from scmle.contracts import errors
from scmle.contracts import qualification

F5_STORE = "detector_certificates.json"
ATTESTATION_SOURCE = "f4-leakage-attestation"
DAG_SOURCE = "f4-dag-falsification"
DAG_ALPHA = 0.05


@dataclasses.dataclass(frozen=True)
class Shelf:
    """A certificate store and what to say when there is none.

    Attributes:
        store: The certificates, or None when none could be used.
        note: Why `store` is None, or an empty string.
    """

    store: certificates.Store | None
    note: str = ""


def load(name: str, directory: Path | None = None) -> Shelf:
    """Reads a certificate store, or says why it could not be read.

    Args:
        name: File name of the store.
        directory: Folder holding it; defaults to the stores shipped with
            the package.

    Returns:
        A `Shelf` holding the store, or None and a note saying
        that the file is missing or unreadable (the detectors it covers are
        then descriptive).
    """
    path = (directory or scmle.certificates.DIRECTORY) / name
    if not path.is_file():
        return Shelf(
            None, f"{name} not found at {path}: detectors are descriptive"
        )
    try:
        return Shelf(certificates.Store.load(path))
    except (ValueError, KeyError, OSError) as error:
        # handled: returned Shelf carries unreadable reason
        return Shelf(
            None, f"{name} unreadable ({error}): detectors are descriptive"
        )


@dataclasses.dataclass(frozen=True)
class Qualification:
    """What the admissibility tests and the detectors found on `X`.

    Attributes:
        table: Admissibility decisions, or None when they were not run.
        leakage: The leakage screen report, or None when not run.
        detectors: Detector pipeline result.
        store: The detector certificates used, or None.
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

    table: rules.Admissibility | None
    leakage: screen.Screen | None
    detectors: pipeline.Findings
    store: certificates.Store | None
    notes: tuple[str, ...]
    diagnostics: tuple[tuple[str, qualification.Check], ...] = ()
    dag: dict[str, object] | None = None

    @property
    def gate(self) -> certificates.Authority:
        """`GATING` when any screen mechanism has a passing certificate."""
        if self.leakage is not None and any(
            g is certificates.Authority.GATING
            for g in self.leakage.gates.values()
        ):
            return certificates.Authority.GATING
        return certificates.Authority.DESCRIPTIVE

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


def screened(
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
        identifier=f"leakage.screen.{finding.mechanism}:{finding.feature}",
        null=(
            f"'{finding.feature}' has no {screen.WORDS[finding.mechanism]} "
            "with the target"
        ),
        statistic=statistic,
        p_value=min(max(finding.p_value, 0.0), 1.0),
        interval=None,
        effect=statistic,
        n=rows,
        decision=decision,
        subset=qualification.Subset.EXPLORATION,
        seeds=(seed, "leakage.screen", index),
        tier=qualification.Tier.UNVERIFIED,
    )


def waiver(
    item: scmle.assurance.admissibility.attestations.Affidavit,
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
        identifier=f"leakage.attestation:{item.column}",
        null=(
            f"'{item.column}' is a legitimate predictor, known "
            f"{item.seconds:g} s before the prediction moment: attested "
            f"by {given.attestor} ({given.role}) on {given.date}, valid to "
            f"{given.expires}, evidence: {given.evidence}"
        ),
        statistic=statistic,
        p_value=min(max(p_value, 0.0), 1.0),
        interval=None,
        effect=statistic,
        n=rows,
        decision=qualification.Decision.WARN,
        subset=qualification.Subset.EXPLORATION,
        seeds=(seed, "leakage.attestation", index),
        tier=qualification.Tier.ATTESTED,
    )


@dataclasses.dataclass(frozen=True)
class Sample:
    """The exploration data one run qualifies, and what it has found so far.

    Attributes:
        frame: Exploration features with the target as a column.
        target: Name of the target column.
        candidates: Candidate covariate columns (structure columns removed).
        seed: Charter seed.
        day: Day on which attestations must be valid.
        result: Detector pipeline result.
        store: Detector certificates used, or None.
        notes: Notes so far; the stages append to this list in place.
    """

    frame: pandas.DataFrame
    target: str
    candidates: list[str]
    seed: int
    day: datetime.date
    result: pipeline.Findings
    store: certificates.Store | None
    notes: list[str]

    @classmethod
    def draw(
        cls,
        features: pandas.DataFrame,
        target: pandas.Series,
        structure: Sequence[str],
        seed: int,
        today: datetime.date | None,
        shelf: Shelf,
    ) -> "Sample":
        """Joins the target to the features and runs the detector pipeline.

        Args:
            features: Exploration features.
            target: Exploration target.
            structure: Group and time columns, which are not candidates.
            seed: Charter seed.
            today: Day on which attestations must be valid; defaults to the
                current date.
            shelf: The detector certificate store and its note.

        Returns:
            The sample, its notes holding the shelf's note if any.
        """
        notes = [n for n in (shelf.note,) if n]
        frame = features.assign(**{str(target.name): target})
        result = pipeline.run(
            frame, str(target.name), shelf.store, seed, provenance=None
        )
        candidates = [c for c in features.columns if c not in set(structure)]
        day = today or datetime.datetime.now().astimezone().date()
        return cls(
            frame,
            str(target.name),
            candidates,
            seed,
            day,
            result,
            shelf.store,
            notes,
        )

    def conclude(
        self,
        table: rules.Admissibility | None,
        leakage: screen.Screen | None,
        diagnostics: tuple[tuple[str, qualification.Check], ...] = (),
    ) -> "Qualification":
        """Wraps the stage results with the sample's detectors and notes.

        Args:
            table: Admissibility decisions, or None when not run.
            leakage: The leakage screen report, or None when not run.
            diagnostics: Ledger records (source, check).

        Returns:
            The qualification.
        """
        return Qualification(
            table,
            leakage,
            self.result,
            self.store,
            tuple(self.notes),
            diagnostics,
        )


def judge(
    sample: Sample,
    spec: scmle.assurance.charter.causal.Question,
    graph: scmle.assurance.admissibility.dag.Dag,
) -> tuple[rules.Admissibility | None, dict[str, str]]:
    """Applies the graph rules and checks the declared adjustment set.

    Args:
        sample: The exploration sample; its notes are appended to when the
            graph rules cannot run, when the declared adjustment set holds a
            refused member, or when there are too many candidates and no
            declared adjustment set.
        spec: The charter's causal block.
        graph: The declared DAG hypothesis.

    Returns:
        The admissibility table (None when the graph rules could not be
        applied) and the refused adjustment members with their reasons.
    """
    candidates = sample.candidates
    records = provenance.Dictionary.covering([str(c) for c in candidates])
    members = scmle.assurance.charter.causal.adjusters(spec.assumptions) or ()
    try:
        table: rules.Admissibility | None = rules.build(
            records,
            "causal",
            rules.Mode.PERMISSIVE,
            today=sample.day,
            graph=graph,
            treatment=spec.treatment,
            outcome=spec.outcome,
        )
    except errors.InputError as error:
        # handled: appended to notes as graph rules not applied
        table = None
        sample.notes.append(f"graph rules were not applied: {error}")
    refused = {
        d.name: "; ".join(d.reasons)
        for d in (() if table is None else table.decisions)
        if d.name in members and d.admission is rules.Admission.QUARANTINE
    }
    if refused:
        sample.notes.append(
            "backdoor_aipw is ineligible: the declared adjustment set holds "
            + ", ".join(sorted(refused))
        )
    if (
        not members
        and len(candidates)
        > scmle.assurance.admissibility.dag.MAX_SEARCH_CANDIDATES
    ):
        sample.notes.append(
            "more than "
            f"{scmle.assurance.admissibility.dag.MAX_SEARCH_CANDIDATES} "
            "candidate covariates and no declared adjustment set: the "
            "charter must supply the set, and the claim is attested at best"
        )
    return table, refused


def probe(
    sample: Sample,
    graph: scmle.assurance.admissibility.dag.Dag,
) -> tuple[Mapping[str, object], tuple[tuple[str, qualification.Check], ...]]:
    """Tests the graph's implied independences on `X`.

    Args:
        sample: The exploration sample; its notes are appended to when the
            falsification cannot run or falsifies implications.
        graph: The declared DAG hypothesis.

    Returns:
        The `falsification` entry of the `dag` report, and the checks of the
        falsification (empty when it could not run).
    """
    try:
        found = independence.falsify(
            graph, sample.frame, alpha=DAG_ALPHA, run_seed=sample.seed
        )
    except (errors.InputError, KeyError) as error:
        # handled: report falsification ran False with reason, plus note
        sample.notes.append(
            f"the declared DAG could not be tested on X: {error}"
        )
        return {"ran": False, "reason": str(error)}, ()
    falsification = {
        "ran": True,
        "alpha": found.alpha,
        "level": found.level,
        "tested": sum(
            o.verdict is not independence.Verdict.SKIPPED
            for o in found.outcomes
        ),
        "falsified": [o.statement.text() for o in found.falsified],
        "skipped_nodes": list(found.nodes),
        "note": "a graph can only survive, never be confirmed",
    }
    if found.falsified:
        sample.notes.append(
            f"{len(found.falsified)} implication(s) of the declared DAG "
            "were falsified on X"
        )
    return falsification, tuple((DAG_SOURCE, c) for c in found.checks)


def admit(
    sample: Sample,
    kind: str,
    spec: scmle.assurance.charter.causal.Question | None,
    graph: scmle.assurance.admissibility.dag.Dag | None,
) -> Qualification:
    """Applies the graph rules to every candidate and tests the graph.

    Each candidate gets `rules.adjudicate` (descendants of the treatment
    and colliders are quarantined), the declared adjustment set is checked
    member by member, and the graph's implied independences are tested on
    `X` only. More candidates than `MAX_SEARCH_CANDIDATES` with no declared
    adjustment set adds a note that the claim is `attested` at best.

    Args:
        sample: The exploration sample; its notes are appended to in place.
        kind: Estimand kind from the charter, for the note when no graph is
            declared.
        spec: The charter's causal block, or None.
        graph: The declared DAG hypothesis, or None.

    Returns:
        The qualification with the admissibility table (None when the graph
        rules could not be applied), no leakage report, and the `dag`
        report holding the graph hash, the refused adjustment members and
        the falsification outcome. Its diagnostics are the checks of the
        falsification, if it ran. Without both a causal block and a graph,
        the table and the `dag` report are None and a note says so.
    """
    if spec is None or graph is None:
        sample.notes.append(
            f"{kind} estimand without a declared graph: admissibility not run"
        )
        return sample.conclude(None, None)
    table, refused = judge(sample, spec, graph)
    falsification, diagnostics = probe(sample, graph)
    report: dict[str, object] = {
        "dag_hash": graph.digest(),
        "adjustment_members_refused": refused,
        "falsification": falsification,
    }
    found = sample.conclude(table, None, diagnostics)
    return dataclasses.replace(found, dag=report)


def vouch(
    sample: Sample,
    report: screen.Screen,
    attestations: Sequence[
        scmle.assurance.admissibility.attestations.Affidavit
    ],
    scope: str,
    declared: Sequence[provenance.Pedigree],
) -> tuple[
    dict[str, scmle.assurance.admissibility.attestations.Affidavit],
    provenance.Dictionary,
]:
    """Applies the attestations that cover a flagged candidate.

    Args:
        sample: The exploration sample; its notes are appended to for every
            rejected attestation and every valid one the screen did not need.
        report: The leakage screen report.
        attestations: Per-column statements that a strong predictor is
            legitimate.
        scope: Charter hash the attestations must name.
        declared: Provenance records for columns the run derived.

    Returns:
        The attestations used, by column, and the provenance records of the
        candidates: attested fields, then declared records for candidates
        that no used attestation covers.
    """
    applied = scmle.assurance.admissibility.attestations.apply(
        attestations, scope, sample.day.isoformat()
    )
    for column, why in applied.rejected.items():
        sample.notes.append(f"attestation for '{column}' ignored: {why}")
    suspect_columns = {f.feature for f in report.suspects}
    used = {
        column: item
        for column, item in applied.valid.items()
        if column in suspect_columns and column in sample.candidates
    }
    sample.notes.extend(
        f"attestation for '{column}' unused: the screen did not flag it"
        for column in applied.valid
        if column not in used
    )
    attested = {i.field().name for i in used.values()}
    names = {str(c) for c in sample.candidates}
    records = provenance.Dictionary.covering(
        [str(c) for c in sample.candidates],
        [
            *(i.field() for i in used.values()),
            *(
                f
                for f in declared
                if f.name in names and f.name not in attested
            ),
        ],
    )
    return used, records


def diagnose(
    report: screen.Screen,
    used: dict[str, scmle.assurance.admissibility.attestations.Affidavit],
    seed: int,
) -> tuple[tuple[str, qualification.Check], ...]:
    """Builds the ledger records of the screen and the used attestations.

    Args:
        report: The leakage screen report.
        used: The attestations that overrode a block, by column.
        seed: Charter seed.

    Returns:
        One record per screen finding, then one per used attestation in
        column order.
    """
    diagnostics: list[tuple[str, qualification.Check]] = [
        (
            screen.DETECTOR_ID,
            screened(f, report.rows, seed, index),
        )
        for index, f in enumerate(report.findings)
    ]
    diagnostics.extend(
        (
            ATTESTATION_SOURCE,
            waiver(
                item,
                [f for f in report.findings if f.feature == column],
                report.rows,
                seed,
                index,
            ),
        )
        for index, (column, item) in enumerate(sorted(used.items()))
    )
    return tuple(diagnostics)


def predict(
    sample: Sample,
    classification: bool,
    group_column: str | None,
    time_column: str | None,
    attestations: Sequence[
        scmle.assurance.admissibility.attestations.Affidavit
    ],
    scope: str,
    declared: Sequence[provenance.Pedigree],
    cutoff: str | None = None,
    known: Mapping[str, str] | None = None,
) -> Qualification:
    """Runs the leakage screen and the admissibility table of a prediction.

    Args:
        sample: The exploration sample; its notes are appended to in place.
        classification: Whether the target holds classes.
        group_column: Entity column, so permutations keep entities whole.
        time_column: Time column, so permutations stay within time blocks.
        attestations: Per-column statements that a strong predictor is
            legitimate.
        scope: Charter hash the attestations must name.
        declared: Provenance records for columns the run derived.
        cutoff: The declared prediction time (`Dataset.cutoff`), or None.
        known: When each column becomes known (`Dataset.known`), or None.

    Returns:
        The qualification with the leakage screen, the admissibility table
        and the ledger diagnostics. A candidate dated after `cutoff` or
        declared known only after the outcome blocks, as in a run without a
        charter.
    """
    task = "classification" if classification else "regression"
    report = screen.screen_table(
        sample.frame,
        sample.target,
        task,
        sample.candidates,
        sample.seed,
        group_column,
        time_column,
        cutoff=cutoff,
        known=known,
    )
    if not any(
        g is certificates.Authority.GATING for g in report.gates.values()
    ):
        sample.notes.append(
            "leakage suspicion is descriptive: the screen gave no gate"
        )
    sample.notes.extend(report.notes)
    used, records = vouch(sample, report, attestations, scope, declared)
    suspect_columns = {f.feature for f in report.suspects}
    gates = {f.feature: certificates.Authority.GATING for f in report.blocking}
    table = rules.build(
        records,
        "predictive",
        rules.Mode.PERMISSIVE,
        today=sample.day,
        suspects=dict.fromkeys(suspect_columns, True),
        leakage_gate=gates,
    )
    return sample.conclude(table, report, diagnose(report, used, sample.seed))


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
        scmle.assurance.admissibility.attestations.Affidavit
    ] = (),
    scope: str = "",
    today: datetime.date | None = None,
    causal: scmle.assurance.charter.causal.Question | None = None,
    dag: scmle.assurance.admissibility.dag.Dag | None = None,
    declared: Sequence[provenance.Pedigree] = (),
    cutoff: str | None = None,
    known: Mapping[str, str] | None = None,
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
        directory: Folder of the detector certificate store, for tests.
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
        cutoff: The declared prediction time (`Dataset.cutoff`), or None.
        known: When each column becomes known (`Dataset.known`), or None.

    Returns:
        The results. A predictive estimand gets the leakage screen, the
        admissibility table and the ledger diagnostics. A causal estimand
        with a causal block and a declared graph gets the graph rules
        (`rules.adjudicate`) on every candidate and the falsification of
        the graph on `X`; without both, its admissibility table and leakage
        report are None and `notes` says so.
    """
    sample = Sample.draw(
        features,
        target,
        structure,
        seed,
        today,
        load(F5_STORE, directory),
    )
    if kind != "predictive":
        return admit(sample, kind, causal, dag)
    return predict(
        sample,
        classification,
        group_column,
        time_column,
        attestations,
        scope,
        declared,
        cutoff,
        known,
    )
