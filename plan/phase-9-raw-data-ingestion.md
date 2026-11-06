# Phase 9: Raw-data ingestion and feature construction (F9)

Names follow [`00-canon.md`](00-canon.md); feature spec in
[`../product/features/F9-raw-data-ingestion.md`](../product/features/F9-raw-data-ingestion.md).

## Goal

Read raw sources (SQL, warehouse, event logs, files), infer table grain and
keys, resolve duplicate entities, discover join paths, and synthesize
relational and temporal features whose reads of other tables can only go
through a cutoff-enforcing view. The assembled table reaches F2/F4 with a
provenance table, reconciled to source totals, inside declared budgets. Every
inference that can gate has a measured operating characteristic and an
external referent (P1).

## Status

Partly built (see `CHANGELOG.md`, "Raw-data ingestion and feature
construction"). The library in `scmle/data/ingest/` (profile and consistent
reads, grain and unit verdict, join discovery, entity resolution, as-of view
and synthesis, reconciliation, budgets), `scmle ingest`, the connector fixes
(snapshot-checked batches, enforced read-only, `IncrementalEvent.ingest_time`)
and `python -m benchmarks.qualification --check ingest` exist. Two detectors
have certificates (`ingest-torn-read`, `ingest-reconcile`); every other floor
is measured on planted worlds of our own in `ingest_measurements.json`. What
is left is the outside referents, the wiring into a charter run, and a few
measurements that need data we do not have. Nothing below was run on a real
warehouse.

## Left to do

**WP-9.8 External corpora** (human lane)
- Obtain and check licence, availability and hashes for a relational
  database set with declared keys and for labelled entity-resolution
  benchmarks; register in `benchmarks/qualification/corpora.json` as
  `candidate`, flip to `verified` by the existing rule (`corpora.py:3-7`).
- Run key inference (recall and precision at `L = 3`), foreign-key discovery
  (precision of auto-accepted links) and entity resolution (pair
  completeness, link recall, pair precision) on them, set the targets from the
  first run and freeze them. Until then the grain, join and entity gates stay
  `descriptive`.
- A person outside the project writes the as-of reference again (the current
  `ingest_naive.py` was written by a separate agent from the spec, which
  does not rule out a shared misreading of the spec) and a data-engineering
  reviewer signs a checklist in the style of `validation-referents.md`.
- Where a deployment has serving-time feature logs, compare the view's
  features with them.

**WP-9.9 Wiring into a run** (after WP-9.8 only for the gates; the plumbing
can start now)
- Acceptance criterion 14 on Python 3.11 and 3.14: a charter run exists
  (`tests/integration/test_ingest_charter_run.py`, SQLite, one interpreter);
  identity across interpreters and a non-SQLite source are not measured.

**Smaller open items**
- `--check ingest` runs a small live check; the full 500/1000/10,000-case
  measurement is rerun only by `ingest_certify --write`.
- Mutation testing: 152 mutants of 762 survive in `asof_view.py` and
  `synthesize.py`; none changes the cutoff comparison (they are in messages,
  `reset_index`, `merge` defaults and the evidence record). No survivor
  bound is fixed in the build configuration.
- Wall-time stops repeat only with a fixed clock; the budget unit is rows
  scanned, not bytes, because DB-API does not report bytes.
- `many_to_many` and entity resolution do not charge the budget per step.
- Read-only is a textual allow-list and a query-only session; the
  recommended deployment uses a read-only database role, which the instrument
  cannot verify.

## Risks

- **`known_time` is often absent.** Many warehouses keep only event time or
  only current state. Then the correct outcome is a lower tier, not a pass.
  The feature manifest shows the tier per feature and per source.
- **Keys by accident.** Small tables make every high-cardinality column look
  like a key. The report carries `N` and a `small_n` flag, ties go to the
  question list, and the corpus run must show how often this still fools it.
- **Corpora may not fit.** Public relational sets have clean declared
  constraints that real warehouses lack; ER benchmarks are mostly
  bibliographic and product text. Report the corpus domain beside every
  number; claims transfer only to similar data.
- **Source load.** Aggregate push-down still scans big tables; the budget
  stops deterministically but cannot make the scan cheaper.
- **Dependency on F2's group column.** Entity resolution only helps if the
  split uses it.
- **Reference-implementation drift.** The naive as-of reference and the fast
  implementation can share a misreading of the spec.

## Exit gate

```bash
uv run pytest -q tests/unit tests/property tests/leakage -k "ingest or grain or asof_view or entity or synthes"
uv run python -m benchmarks.qualification --check ingest
uv run python -m benchmarks.qualification --check          # certificates still current
```

Plus the corpus measurements above, with Wilson intervals, and a charter run
that passes criterion 14 on 3.11 and 3.14.

## Explicitly out of scope

LLM-based semantic column interpretation or feature proposals; change-data-
capture or replication tooling; graph, image, text and other modality
feature extraction (existing `scmle/data/modalities/`); automatic repair of
dirty data (F5 detects, it does not repair); privacy and PII governance;
real-time or online feature serving; target-aware feature selection outside
the ledger; cross-database federation beyond the connectors that exist.
