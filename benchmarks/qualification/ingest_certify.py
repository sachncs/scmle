"""Measurements and certificates for ingestion.

Two detectors have certificates in `ingest_certificates.json`:

* `ingest-torn-read` (`scmle.data.ingest.profile.consistent_read` with the
  `DbApiConnector` batch check): `NOISY` worlds mutate a table between
  batches (an update behind the read position, one ahead of it, an insert, a
  delete; each kind equally often); `QUIET` worlds are static. Policy:
  recall Wilson lower bound at least 0.99, false-alarm upper bound at most
  0.02, 95 percent intervals.
* `ingest-reconcile` (`scmle.data.ingest.reconcile`): `RECONCILE_RUNS` worlds
  per planted defect (`fanout`, `dropped_rows`, `torn_read`) and the same
  number of clean worlds. A detection is a reconciliation with an unexplained
  gap. Policy: recall lower bound at least 0.95, false-alarm upper bound at
  most 0.05. The plants are large relative to the tolerance (zero), and the
  reconciliation compares aggregates, so offsetting row errors would pass; no
  such plant is in the world.

Every other floor is measured into `ingest_measurements.json` and judged
against numbers fixed here before any measurement:

* point in time: view leaks 0 rows over `LATE_WORLDS` late-arrival and
  backfill worlds, the event-time-only baseline leaks in at least 95 percent
  of them, the vectorised code equals `ingest_naive` on `DIFFERENTIAL_CASES`
  random schemas, and `METAMORPHIC_CASES` rewrites of not-yet-known rows
  change nothing;
* unit verdict: at least 95 percent correct on `UNIT_PAIRS` synthetic
  unit/grain pairs and no `coarser` reported as `match`; near-key threshold
  flips within one row;
* joins: no auto-accepted link outside the planted truth over
  `JOIN_WORLDS` surrogate-key schemas;
* entity resolution: pair completeness and link recall at least 0.95 over
  `ER_WORLDS` synthetic registries with a labelled clerical sample;
* budget: spend never exceeds the budget on `BUDGETS` randomized budgets and
  stop boundaries repeat exactly;
* multiplicity: declared family size equals the registered count.

None of the unit, join or entity numbers has an external referent here: the
declared-key relational corpus and the labelled ER benchmarks are not
registered (`corpora.json`), so those gates stay descriptive whatever the
numbers say. Nothing below is tuned after seeing a result: a floor that is
missed is recorded as missed.

Seeds. Development seeds start at `ingest_worlds.DEV_SEED_BASE`; every
measurement here uses seeds from 0 up. The ranges do not overlap.

    python -m benchmarks.qualification.ingest_certify --write
    python -m benchmarks.qualification.ingest_certify --check
    python -m benchmarks.qualification.ingest_certify --dev
"""

import argparse
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy
import pandas

from benchmarks.qualification import ingest_checks
from benchmarks.qualification import ingest_worlds
from scmle.contracts import qualification
from scmle.core import randomness
from scmle.data.detectors import certificates
from scmle.data.ingest import asof_view
from scmle.data.ingest import budget
from scmle.data.ingest import entities
from scmle.data.ingest import grain
from scmle.data.ingest import joins
from scmle.data.ingest import plan as ingest_plan
from scmle.data.ingest import profile
from scmle.data.ingest import reconcile
from scmle.data.ingest import synthesize
from scmle.data.readers import connectors

STORE_PATH = Path(__file__).with_name("ingest_certificates.json")
MEASUREMENTS_PATH = Path(__file__).with_name("ingest_measurements.json")
TORN_ID = "ingest-torn-read"
TORN_DEFECT = "torn_read"
TORN_RUNS = 500
TORN_POLICY = certificates.CertificationPolicy(0.99, 0.02, 0.95)
RECONCILE_RUNS = 500
RECONCILE_POLICY = certificates.CertificationPolicy(0.95, 0.05, 0.95)
LATE_WORLDS = 500
DIFFERENTIAL_CASES = 1000
METAMORPHIC_CASES = 10_000
UNIT_PAIRS = 200
JOIN_WORLDS = 200
ER_WORLDS = 10
ER_ENTITIES = 300
BUDGETS = 100
MULTIPLICITY_WORLDS = 20
UNIT_FLOOR = 0.95
ER_FLOOR = 0.95
BASELINE_LEAK_SHARE = 0.95
NEAR_KEY_ROWS = 1000
NEAR_KEY_EPS = 0.01
EXTERNAL = {
    "unit_verdict": "no declared-key relational corpus registered",
    "joins": "no declared-key relational corpus registered",
    "entities": "no labelled ER benchmark registered",
}


def digest(document: Any) -> str:
    """SHA-256 of a JSON document with sorted keys.

    Args:
        document: A JSON-serialisable value; other types are converted with
            `str`.

    Returns:
        The hex digest of the document's JSON text.
    """
    text = json.dumps(document, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_jobs(
    function: Any, jobs: Sequence[tuple[Any, ...]], workers: int
) -> list[Any]:
    """Runs `function(*job)` for every job, in order.

    Args:
        function: The function to call.
        jobs: Argument tuples.
        workers: Number of parallel workers; one or fewer runs in process.

    Returns:
        The results in the order of `jobs`.
    """
    if workers <= 1:
        return [function(*job) for job in jobs]
    from joblib import delayed
    from joblib import Parallel

    return list(Parallel(n_jobs=workers)(delayed(function)(*j) for j in jobs))


# Torn read ----------------------------------------------------------------


def torn_detected(seed: int, noisy: bool) -> bool:
    """Whether a read of one world was rejected as torn.

    Args:
        seed: World seed; for a noisy world it also picks the mutation kind.
        noisy: Whether the table is mutated during the read.

    Returns:
        True if `consistent_read` raised `TornReadError`.
    """
    kind = ingest_worlds.TORN_KINDS[seed % 4] if noisy else None
    connector = ingest_worlds.torn_world(seed, kind)[0]
    try:
        profile.consistent_read(connector, 5, attempts=1)
    except profile.TornReadError:
        return True
    return False


def torn_code() -> str:
    """Hash of the code a torn-read certificate covers.

    Returns:
        The code hash of the consistent-read check and the connector batch
        check.
    """
    return certificates.code_hash(
        profile.consistent_read,
        profile.read_once,
        profile.has_total_order,
        connectors.DbApiConnector,
        connectors.digest_rows,
    )


def torn_certificate(
    offset: int, workers: int
) -> tuple[certificates.DetectorCertificate, dict[str, Any]]:
    """Measures the torn-read check on mutated and static worlds.

    Args:
        offset: Added to the world seeds.
        workers: Number of parallel workers.

    Returns:
        The certificate and the measurements (detections overall and by
        mutation kind, and false detections on static worlds).
    """
    noisy = run_jobs(
        torn_detected, [(offset + s, True) for s in range(TORN_RUNS)], workers
    )
    quiet = run_jobs(
        torn_detected, [(offset + s, False) for s in range(TORN_RUNS)], workers
    )
    result = qualification.Certificate(
        TORN_ID,
        TORN_DEFECT,
        TORN_RUNS,
        sum(noisy),
        TORN_RUNS,
        sum(quiet),
        randomness.seed_path(offset, "ingest.certify.torn", 0),
    )
    cert = certificates.certify(
        TORN_ID,
        torn_code(),
        certificates.corpus_hash(
            [digest({"kinds": ingest_worlds.TORN_KINDS, "batch_rows": 5})]
        ),
        [offset],
        [result],
        TORN_POLICY,
    )
    by_kind = {
        kind: sum(r for k, r in enumerate(noisy) if k % 4 == i)
        for i, kind in enumerate(ingest_worlds.TORN_KINDS)
    }
    return cert, {
        "mutated_worlds": TORN_RUNS,
        "static_worlds": TORN_RUNS,
        "detected": sum(noisy),
        "false_detections": sum(quiet),
        "detected_by_kind": by_kind,
        "batch_rows": 5,
        "seed_offset": offset,
    }


# Reconciliation -----------------------------------------------------------


def reconcile_flagged(kind: str, seed: int) -> bool:
    """Whether the reconciliation of one extract has an unexplained gap.

    Args:
        kind: One of `ingest_worlds.EXTRACTS`.
        seed: World seed.

    Returns:
        True if the reconciliation report blocks.
    """
    extract = ingest_worlds.reconcile_world(seed, kind)
    report = reconcile.reconcile_extract(
        extract.db.source(),
        extract.assembled,
        "orders",
        ["amount"],
        [
            reconcile.JoinCheck(
                "orders-customers", extract.left_rows, len(extract.assembled)
            )
        ],
        gate=certificates.GateStatus.GATING,
    )
    return reconcile.blocks(report)


def reconcile_certificate(
    offset: int, workers: int
) -> tuple[certificates.DetectorCertificate, dict[str, Any]]:
    """Measures the reconciliation on planted defects and clean worlds.

    Args:
        offset: Added to the world seeds.
        workers: Number of parallel workers.

    Returns:
        The certificate and the measurements (false alarms and detections by
        defect).
    """
    clean = run_jobs(
        reconcile_flagged,
        [("clean", offset + s) for s in range(RECONCILE_RUNS)],
        workers,
    )
    results = []
    counts = {}
    for kind in reconcile.DEFECT_TYPES:
        flagged = run_jobs(
            reconcile_flagged,
            [(kind, offset + s) for s in range(RECONCILE_RUNS)],
            workers,
        )
        counts[kind] = {"planted": RECONCILE_RUNS, "detected": sum(flagged)}
        results.append(
            qualification.Certificate(
                reconcile.DETECTOR_ID,
                kind,
                RECONCILE_RUNS,
                sum(flagged),
                RECONCILE_RUNS,
                sum(clean),
                randomness.seed_path(offset, "ingest.certify.reconcile", 0),
            )
        )
    cert = certificates.certify(
        reconcile.DETECTOR_ID,
        reconcile.code_digest(),
        certificates.corpus_hash(
            [
                digest(
                    {
                        "worlds": list(ingest_worlds.EXTRACTS),
                        "measure": "amount",
                    }
                )
            ]
        ),
        [offset],
        results,
        RECONCILE_POLICY,
    )
    return cert, {
        "clean_worlds": RECONCILE_RUNS,
        "false_alarms": sum(clean),
        "by_defect": counts,
        "tolerance": 0.0,
        "seed_offset": offset,
    }


# Floors on synthetic worlds -----------------------------------------------


def judged(passed: bool, referent: str | None, **fields: Any) -> dict[str, Any]:
    """A measured section: its numbers, floor verdict and gating status.

    Args:
        passed: Whether the pre-registered floor was met.
        referent: Description of the external referent, or None when there
            is none.
        **fields: The measured numbers, kept in the section.

    Returns:
        The fields plus `floor_met`, `external_referent` and a `descriptive`
        gate with its reason.
    """
    return {
        **fields,
        "floor_met": bool(passed),
        "external_referent": referent,
        "gate": "descriptive",
        "gate_reason": (
            "not bound to a charter decision map yet"
            if referent
            else "no external referent, and not bound to a decision map"
        ),
    }


def point_in_time(offset: int, workers: int) -> dict[str, Any]:
    """Late arrival, differential and metamorphic measurements.

    Args:
        offset: Added to the world seeds.
        workers: Number of parallel workers.

    Returns:
        The judged section: leaks of the view and of the baseline, and the
        differential and metamorphic results.
    """
    leaks = run_jobs(
        ingest_checks.late_leaks,
        [(offset + s,) for s in range(LATE_WORLDS)],
        workers,
    )
    differential = run_jobs(
        ingest_checks.differential_equal,
        [(offset + s,) for s in range(DIFFERENTIAL_CASES)],
        workers,
    )
    metamorphic = run_jobs(
        ingest_checks.metamorphic_unchanged,
        [(offset + s,) for s in range(METAMORPHIC_CASES)],
        workers,
    )
    view_leaked = sum(item.view for item in leaks)
    baseline_worlds = sum(item.baseline > 0 for item in leaks)
    ok = (
        view_leaked == 0
        and baseline_worlds >= BASELINE_LEAK_SHARE * LATE_WORLDS
        and all(differential)
        and all(metamorphic)
    )
    return judged(
        ok,
        "independent naive reference (ingest_naive.py, written separately)",
        late_worlds=LATE_WORLDS,
        view_leaked_rows=view_leaked,
        baseline_leaked_rows=sum(item.baseline for item in leaks),
        baseline_leaking_worlds=baseline_worlds,
        differential_cases=DIFFERENTIAL_CASES,
        differential_mismatches=differential.count(False),
        differential_tolerance=ingest_checks.RTOL,
        metamorphic_cases=METAMORPHIC_CASES,
        metamorphic_failures=metamorphic.count(False),
        limit="the guarantee is about the code: it holds only if the "
        "recorded time is true",
    )


def unit_pair(seed: int) -> tuple[str, str, str, str]:
    """(unit, shape, expected, got) for pair number `seed`.

    Args:
        seed: Pair number; it also picks the unit and shape combination.

    Returns:
        The unit, the table's true shape, the verdict that follows from the
        construction and the verdict of `grain.unit_verdict`.
    """
    combos = [(u, s) for u in ingest_worlds.UNITS for s in ingest_worlds.SHAPES]
    unit, shape = combos[seed % len(combos)]
    db, table = ingest_worlds.grain_world(seed, unit, shape)
    source = db.source()
    profiled = profile.profile_table(source, table)
    assert profiled is not None
    scan = grain.infer_keys(source, profiled)
    verdict = grain.unit_verdict(source, profiled, unit, scan)
    return (
        unit,
        shape,
        ingest_worlds.expected_verdict(unit, shape),
        verdict.verdict,
    )


def near_key_flips() -> dict[str, Any]:
    """Whether a planted near key flips at the threshold, to one row.

    Returns:
        The allowed number of duplicates, whether the column is accepted as
        a key for each tested number of duplicates, and whether the verdict
        flips exactly at the allowance.
    """
    allowed = round(NEAR_KEY_EPS * NEAR_KEY_ROWS)
    flips = {}
    for duplicates in range(allowed - 3, allowed + 4):
        values = numpy.arange(NEAR_KEY_ROWS)
        values[:duplicates] = values[NEAR_KEY_ROWS - duplicates :]
        db = ingest_worlds.new_database("nk")
        ingest_worlds.load(db, "t", pandas.DataFrame({"k": values}))
        source = db.source()
        table = profile.profile_table(source, "t")
        assert table is not None
        scan = grain.infer_keys(source, table, eps_key=NEAR_KEY_EPS)
        flips[duplicates] = bool(scan.keys)
    exact = all(flips[d] == (d <= allowed) for d in flips)
    return {
        "allowed_duplicates": allowed,
        "is_key_by_duplicates": flips,
        "exact": exact,
    }


def unit_verdicts(offset: int, workers: int) -> dict[str, Any]:
    """Unit-verdict accuracy and the zero-tolerance error.

    Args:
        offset: Added to the world seeds.
        workers: Number of parallel workers.

    Returns:
        The judged section: accuracy with its interval, `coarser` verdicts
        reported as `match`, and the near-key result.
    """
    rows = run_jobs(
        unit_pair, [(offset + s,) for s in range(UNIT_PAIRS)], workers
    )
    correct = sum(r[2] == r[3] for r in rows)
    coarser_as_match = sum(r[2] == "coarser" and r[3] == "match" for r in rows)
    flips = near_key_flips()
    ok = (
        correct / UNIT_PAIRS >= UNIT_FLOOR
        and coarser_as_match == 0
        and flips["exact"]
    )
    lo, hi = certificates.wilson_interval(correct, UNIT_PAIRS)
    return judged(
        ok,
        None,
        pairs=UNIT_PAIRS,
        correct=correct,
        accuracy_interval=[lo, hi],
        coarser_reported_as_match=coarser_as_match,
        near_key=flips,
        key_recall_precision="not measured: " + EXTERNAL["unit_verdict"],
    )


def join_world(seed: int) -> tuple[int, int, int, int]:
    """(accepted, accepted_true, truth, found_true) for one schema.

    Args:
        seed: Schema seed.

    Returns:
        The number of auto-accepted links, how many of them are real, the
        number of real links, and the number of real links found (the same
        count as the second entry).
    """
    world = ingest_worlds.surrogate_world(seed)
    source = world.db.source()
    catalog = profile.profile_catalog(source, world.tables)
    scans = {t.name: grain.infer_keys(source, t) for t in catalog.tables}
    graph = joins.discover(source, catalog, scans)
    accepted = {
        (link.child, link.child_columns[0], link.parent)
        for link in graph.accepted()
    }
    return (
        len(accepted),
        len(accepted & world.truth),
        len(world.truth),
        len(accepted & world.truth),
    )


def join_discovery(offset: int, workers: int) -> dict[str, Any]:
    """Spurious auto-accepts and recall over surrogate-key schemas.

    Args:
        offset: Added to the schema seeds.
        workers: Number of parallel workers.

    Returns:
        The judged section: counts of accepted and spurious links,
        precision and recall with intervals.
    """
    rows = run_jobs(
        join_world, [(offset + s,) for s in range(JOIN_WORLDS)], workers
    )
    accepted = sum(r[0] for r in rows)
    true_accepted = sum(r[1] for r in rows)
    truth = sum(r[2] for r in rows)
    spurious_worlds = sum(r[0] != r[1] for r in rows)
    precision = certificates.wilson_interval(true_accepted, max(accepted, 1))
    recall = certificates.wilson_interval(true_accepted, max(truth, 1))
    return judged(
        spurious_worlds == 0,
        None,
        schemas=JOIN_WORLDS,
        auto_accepted=accepted,
        spurious_accepted=accepted - true_accepted,
        schemas_with_a_spurious_accept=spurious_worlds,
        true_links=truth,
        precision_interval=list(precision),
        recall=true_accepted / max(truth, 1),
        recall_interval=list(recall),
        note="twin parents (account/user) and unnamed owner_id links are "
        "left as candidates, which costs recall",
        fk_precision_on_corpus="not measured: " + EXTERNAL["joins"],
    )


ER_FIELDS = [
    entities.Field("first"),
    entities.Field("last"),
    entities.Field("dob", fuzzy=False),
    entities.Field("addr"),
]
ER_PASSES = [
    entities.Pass("last", "soundex"),
    entities.Pass("dob"),
    entities.Pass("first", "prefix", 3),
]


def er_world(seed: int) -> dict[str, Any]:
    """Quality of one resolution against the truth of its registry.

    Args:
        seed: Registry seed.

    Returns:
        Counts of true pairs, true pairs compared, grouped pairs and false
        group pairs, the reduction ratio and the precision estimate (None
        when there is none).
    """
    frame = ingest_worlds.er_world(seed, ER_ENTITIES)
    truth = frame["entity"].to_numpy()
    result = entities.resolve(
        frame,
        ER_FIELDS,
        ER_PASSES,
        labeler=lambda i, j: bool(truth[i] == truth[j]),
        reviewer="benchmark truth",
        seed=seed,
    )
    pairs = entities.candidate_pairs(frame, ER_PASSES)[0]
    quality = entities.measure_quality(truth.tolist(), result, pairs)
    return {
        "true_pairs": quality.true_pairs,
        "compared_true": round(quality.pair_completeness * quality.true_pairs),
        "grouped_pairs": quality.grouped_pairs,
        "false_group_pairs": int(quality.false_group_pairs),
        "reduction_ratio": result.reduction_ratio,
        "precision_estimate": list(result.estimate.precision)
        if result.estimate
        else None,
    }


def entity_resolution(offset: int, workers: int) -> dict[str, Any]:
    """Pair completeness and cross-split link recall over registries.

    Args:
        offset: Added to the registry seeds.
        workers: Number of parallel workers.

    Returns:
        The judged section: pair completeness, link recall with its
        interval and false group pairs.
    """
    rows = run_jobs(
        er_world, [(offset + s,) for s in range(ER_WORLDS)], workers
    )
    truth = sum(r["true_pairs"] for r in rows)
    compared = sum(r["compared_true"] for r in rows)
    grouped = sum(r["grouped_pairs"] for r in rows)
    completeness = compared / truth
    recall = grouped / truth
    interval = certificates.wilson_interval(grouped, truth)
    return judged(
        completeness >= ER_FLOOR and recall >= ER_FLOOR,
        None,
        registries=ER_WORLDS,
        entities_each=ER_ENTITIES,
        true_pairs=truth,
        pair_completeness=completeness,
        link_recall=recall,
        link_recall_interval=list(interval),
        false_group_pairs=sum(r["false_group_pairs"] for r in rows),
        mean_reduction_ratio=float(
            numpy.mean([r["reduction_ratio"] for r in rows])
        ),
        note="the corruption is ours; not a public ER benchmark. Grouping "
        "stays unverified until a labelled benchmark is run: "
        + EXTERNAL["entities"],
    )


def budget_run(seed: int, directory: Path) -> tuple[bool, list[dict[str, str]]]:
    """(within budget, stop boundaries) for one randomized budget.

    Args:
        seed: Budget seed.
        directory: Parent folder for the run's files.

    Returns:
        Whether queries and rows scanned stayed within the budget, and the
        stop boundaries recorded by the meter.
    """
    rng = randomness.component_rng(seed, "ingest.certify.budget", 0)
    limit = budget.IngestBudget(
        max_queries=int(rng.integers(0, 800)),
        max_rows_scanned=int(rng.integers(0, 150_000)),
    )
    world = ingest_worlds.surrogate_world(0)
    plan = ingest_plan.IngestPlan(
        directory / f"b{seed}",
        world.db.factory,
        ingest_plan.IngestSettings(tables=tuple(world.tables)),
        limit,
    )
    plan.run_all()
    meter = plan.meter
    within = (
        meter.queries <= (limit.max_queries or 0)
        and meter.rows <= (limit.max_rows_scanned or 0)
        and plan.source.queries <= (limit.max_queries or 0)
    )
    return within, [stop.to_dict() for stop in meter.stops]


def budget_adherence(offset: int, directory: Path) -> dict[str, Any]:
    """Spend within budget and identical stops on repeat.

    Args:
        offset: Added to the budget seeds.
        directory: Folder for the runs (a subfolder per repeat).

    Returns:
        The judged section: runs within budget, runs with identical stop
        boundaries on repeat and runs stopped early.
    """
    first = [budget_run(offset + s, directory / "a") for s in range(BUDGETS)]
    second = [budget_run(offset + s, directory / "b") for s in range(BUDGETS)]
    within = sum(item[0] for item in first)
    same = sum(a[1] == b[1] for a, b in zip(first, second, strict=True))
    partial = sum(bool(item[1]) for item in first)
    return judged(
        within == BUDGETS and same == BUDGETS,
        "the cost model's estimate against counted source queries",
        budgets=BUDGETS,
        within_budget=within,
        identical_stop_boundaries=same,
        runs_stopped_early=partial,
        overhead="none for queries and rows (admitted before they run); "
        "wall time depends on the clock and is not covered",
    )


def multiplicity(offset: int) -> dict[str, Any]:
    """Declared family size against registered hypotheses.

    Args:
        offset: First schema seed.

    Returns:
        The judged section: worlds in which the declared size equals the
        registered count, and worlds in which a wrong size is refused.
    """
    import fractions

    from scmle.data.ledger import ledger as ledger_module
    from scmle.data.ledger import online_fdr

    equal = 0
    mismatch_caught = 0
    for seed in range(offset, offset + MULTIPLICITY_WORLDS):
        schema = ingest_checks.random_schema(seed)
        source, items = ingest_checks.views(schema)
        found = synthesize.synthesize(
            schema.targets,
            "moment",
            [source],
            [synthesize.Nest(source, items, "oid")],
            schema.lag,
        )
        ledger = ledger_module.Ledger.create(
            None,
            charter_hash="0" * 64,
            config=online_fdr.RuleConfig("lord++", fractions.Fraction(1, 20)),
            alpha_confirm=fractions.Fraction(1, 20),
            seed=seed,
        )
        synthesize.register_family(ledger, found)
        equal += (
            synthesize.check_multiplicity(found.declared_family_size, ledger)
            == found.declared_family_size
        )
        try:
            synthesize.check_multiplicity(
                found.declared_family_size + 1, ledger
            )
        except synthesize.MultiplicityError:
            mismatch_caught += 1
    return judged(
        equal == MULTIPLICITY_WORLDS and mismatch_caught == MULTIPLICITY_WORLDS,
        "ledger registration count",
        worlds=MULTIPLICITY_WORLDS,
        declared_equals_registered=equal,
        mismatch_fails_run=mismatch_caught,
    )


def code_hashes() -> dict[str, str]:
    """Code hash of each measured module group, as it is now.

    Returns:
        Mapping of group name to code hash.
    """
    return {
        "torn_read": torn_code(),
        "reconcile": reconcile.code_digest(),
        "point_in_time": certificates.code_hash(asof_view, synthesize),
        "grain": certificates.code_hash(grain),
        "joins": certificates.code_hash(joins),
        "entities": certificates.code_hash(entities),
        "budget": certificates.code_hash(budget, ingest_plan),
    }


def certify_all(
    workers: int = 1, offset: int = 0, scratch: Path | None = None
) -> tuple[certificates.CertificateStore, dict[str, Any]]:
    """Measures everything; `offset` shifts the seed range.

    Args:
        workers: Number of parallel workers.
        offset: Added to the world seeds.
        scratch: Folder for the budget runs; a temporary folder when None.

    Returns:
        The certificate store and the measurements document.
    """
    started = time.monotonic()
    store = certificates.CertificateStore()
    document: dict[str, Any] = {}
    cert, measured = torn_certificate(offset, workers)
    store.add(cert)
    document["torn_read"] = measured
    cert, measured = reconcile_certificate(offset, workers)
    store.add(cert)
    document["reconcile"] = measured
    document["point_in_time"] = point_in_time(offset, workers)
    document["unit_verdict"] = unit_verdicts(offset, workers)
    document["joins"] = join_discovery(offset, workers)
    document["entities"] = entity_resolution(offset, workers)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        document["budget"] = budget_adherence(offset, scratch or Path(tmp))
    document["multiplicity"] = multiplicity(offset)
    document["code_hashes"] = code_hashes()
    document["seconds"] = round(time.monotonic() - started, 1)
    document["development_seed_base"] = ingest_worlds.DEV_SEED_BASE
    return store, document


def current_digests() -> dict[str, str]:
    """Code hash of each certified detector as it is now.

    Returns:
        Mapping of detector id to its current code hash.
    """
    return {
        TORN_ID: torn_code(),
        reconcile.DETECTOR_ID: reconcile.code_digest(),
    }


def required_defects() -> dict[str, tuple[str, ...]]:
    """Defect types each certified detector must be certified for.

    Returns:
        Mapping of detector id to its defect types.
    """
    return {
        TORN_ID: (TORN_DEFECT,),
        reconcile.DETECTOR_ID: reconcile.DEFECT_TYPES,
    }


def audit_labels(store: certificates.CertificateStore) -> dict[str, list[str]]:
    """Detector/defect pairs by why they may not gate.

    `missing` and `stale` are failures of the file; `uncertified` pairs were
    measured and did not meet the policy, so they stay descriptive.

    Args:
        store: The certificate store.

    Returns:
        Lists of `detector/defect` names under `missing`, `stale` and
        `uncertified`.
    """
    found: dict[str, list[str]] = {
        "missing": [],
        "stale": [],
        "uncertified": [],
    }
    digests = current_digests()
    for detector, defects in required_defects().items():
        for defect in defects:
            report = certificates.audit(
                store, {detector: (digests[detector], defect)}
            )
            for label in found:
                if getattr(report, label):
                    found[label].append(f"{detector}/{defect}")
    return found


def problems(store: certificates.CertificateStore) -> list[str]:
    """Why the certificate file is unusable: missing or stale entries.

    Args:
        store: The certificate store.

    Returns:
        One message per missing or stale detector/defect pair.
    """
    labels = audit_labels(store)
    return [
        f"{item}: {kind}"
        for kind in ("missing", "stale")
        for item in labels[kind]
    ]


def measurement_problems(document: dict[str, Any]) -> list[str]:
    """Why the measurements file no longer describes the code.

    Args:
        document: The stored measurements, with their `code_hashes`.

    Returns:
        One message per module group whose stored hash differs from the
        current code.
    """
    found = []
    stored = document.get("code_hashes", {})
    for name, current in code_hashes().items():
        if stored.get(name) != current:
            found.append(f"{name}: measurements are for other code")
    return found


def live_checks() -> dict[str, Any]:
    """A small live run of the point-in-time and torn-read checks.

    Returns:
        `ok` (True when the view leaks nothing, fast and naive agree, the
        rewrite leaves features unchanged, mutated reads are caught and
        static reads are not) and the number of cases of each kind.
    """
    leaks = [ingest_checks.late_leaks(s) for s in range(20)]
    diff = [ingest_checks.differential_equal(s) for s in range(60)]
    meta = [ingest_checks.metamorphic_unchanged(s) for s in range(60)]
    torn = [torn_detected(s, True) for s in range(8)]
    quiet = [torn_detected(s, False) for s in range(8)]
    ok = (
        all(item.view == 0 for item in leaks)
        and all(diff)
        and all(meta)
        and all(torn)
        and not any(quiet)
    )
    return {"ok": ok, "late_worlds": 20, "differential": 60, "metamorphic": 60}


def development_look() -> dict[str, Any]:
    """A small run on development seeds. Writes nothing.

    Returns:
        The raw results under the keys `torn` (six worlds), `reconcile` (four
        fan-out worlds), `unit` (six worlds) and `late` (the count of late
        rows the view leaks, for six worlds).
    """
    base = ingest_worlds.DEV_SEED_BASE
    return {
        "torn": [torn_detected(base + s, s % 2 == 0) for s in range(6)],
        "reconcile": [reconcile_flagged("fanout", base + s) for s in range(4)],
        "unit": [unit_pair(base + s) for s in range(6)],
        "late": [ingest_checks.late_leaks(base + s).view for s in range(6)],
    }


def check() -> dict[str, Any]:
    """Verifies the files against the code and re-runs the live checks.

    Returns:
        `ok`, the problems found, the descriptive-only detector/defect
        pairs, the floors missed, the gates without an external referent and
        the live-check result.
    """
    store = certificates.CertificateStore.load(STORE_PATH)
    found = problems(store)
    measured = json.loads(MEASUREMENTS_PATH.read_text(encoding="utf-8"))
    found += measurement_problems(measured)
    live = live_checks()
    if not live["ok"]:
        found.append("live point-in-time or torn-read check failed")
    floors = {
        name: section["floor_met"]
        for name, section in measured.items()
        if isinstance(section, dict) and "floor_met" in section
    }
    return {
        "ok": not found,
        "problems": found,
        "descriptive_only": audit_labels(store)["uncertified"],
        "floors_missed": sorted(k for k, v in floors.items() if not v),
        "gates_without_external_referent": sorted(EXTERNAL),
        "live": live,
    }


def main(arguments: list[str] | None = None) -> int:
    """Measures (`--write`), verifies (`--check`) or looks (`--dev`).

    Args:
        arguments: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        With `--check`, 0 when the files match the code and the live checks
        pass, otherwise 1; with `--write` and `--dev`, always 0.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    action.add_argument("--dev", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    options = parser.parse_args(arguments)
    if options.dev:
        print(json.dumps(development_look(), default=str))
        return 0
    if options.check:
        report = check()
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1
    store, document = certify_all(options.workers)
    store.save(STORE_PATH)
    MEASUREMENTS_PATH.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
