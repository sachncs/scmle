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

## Entry criteria

- Phase 2 merged (entity-grouped sets, charter, signing).
- Phase 4 merged: `asof.py`, `provenance.py`, `reconciliation.py`, duplicates
  detector and its certificate (all present in the repo).
- Phase 3 merged (ledger), because synthesized features are ledger
  hypotheses.
- Phase 8 is **not** required: the unit-to-grain verdict can take the unit
  text from the charter as is. WP-8.5a later consumes this phase's key
  inference, and Phase 8 is not blocked while this phase runs.
- A test database engine available in CI (SQLite is in the standard library
  and `DbApiConnector` already supports `sqlite://`,
  `scmle/data/readers/connectors.py:597-606`); warehouse paths are tested
  through the injected callbacks.

## Verified starting points

- Connectors, retries, checkpoints: `scmle/data/readers/connectors.py:163`
  (`RetryBudget`), `:244` (`SourceCheckpoint`), `:422` (`SourceConnector`),
  `:597` (`DbApiConnector`), `:736` (`WarehouseConnector`), `:886`, `:1044`.
- Batches re-execute the query and skip rows by fetching:
  `connectors.py:677-698`, `:718`. Not snapshot-consistent by themselves
  (reasoned from the code; confirm with the WP-9.1 torn-read test).
- Only `source_identity` is used outside `readers/` and tests
  (`scmle/api/project.py:29, :421`).
- Snapshots and schema drift: `scmle/data/readers/registry.py:18, :151`.
- Event model without ingest time: `scmle/data/readers/incremental.py:75`.
- As-of audit (not construction): `scmle/data/admissibility/asof.py:59, :103,
  :204`; provenance and tiers `provenance.py:242, :304, :376`; timing rule
  `rules.py:104`.
- Reconciliation with caller-supplied totals: `reconciliation.py:98, :166,
  :263`.
- Per-column id heuristic only: `scmle/data/validation/columns.py:27, :410`.
- Duplicates within one table: `scmle/data/splitting/duplicates.py:18`,
  `scmle/data/detectors/duplicates.py:89`.
- Group splitting by a named column: `scmle/data/splitting/splitters.py:496-509`.
- Verified absences: no `.merge(` in `scmle/`, no key or FD inference, no
  inclusion dependencies, no cross-table ER, no feature synthesis, no
  ingestion budget.
- `--check` choices at `benchmarks/qualification/__main__.py:75`:
  `certificates`, `fdr`; an `ingest` choice is added here.

## Work packages

**WP-9.1 Push-down profile and consistent reads** (one agent)
- *Proposed* `scmle/data/ingest/profile.py`: catalog by aggregate queries;
  ordered, token-checked batch reads; read-only enforcement (statement
  allow-list, read-only transaction when supported).
- Done when: criterion 1 (torn read) passes on SQLite fixtures; batches
  disjoint and complete on static data; a write statement is refused.

**WP-9.2 Grain and key inference** (one agent; needs 9.1)
- *Proposed* `ingest/grain.py`: level-wise lattice to `L = 3`, `g3` by SQL
  grouped counts, ranking, unit verdict (`match | finer | coarser |
  unknown`).
- Done when: criteria 2 and 3 pass; planted near-key threshold behaviour
  verified to 1 row.

**WP-9.3 Join discovery** (one agent; needs 9.1, 9.2)
- *Proposed* `ingest/joins.py`: containment by anti-join counts, accept rules
  (declared FK or exact containment with corroboration), cardinality typing,
  path enumeration to depth 2, many-to-many never joined.
- Done when: criterion 4 passes, including 0 spurious accepted links on 200
  surrogate-key schemas.

**WP-9.4 Entity resolution** (one agent; needs 9.1; parallel with 9.2, 9.3)
- *Proposed* `ingest/entities.py`: multi-pass blocking, Fellegi-Sunter
  weights with EM, two thresholds, stratified clerical sample estimator for
  precision and recall with intervals, link-then-group output for F2.
- Done when: criterion 5 passes; pair completeness and reduction ratio
  reported per benchmark; unlabelled run is marked `unverified`.

**WP-9.5 As-of view and synthesis** (one agent; needs 9.2, 9.3; core of the
phase)
- *Proposed* `ingest/asof_view.py` and `ingest/synthesize.py`: the view is
  the only read path; `known_time` selection (transaction time, else event
  time plus declared delay, else non-versioned); primitives and windows;
  depth 2; `B_feat` cap; family size declared before exploration;
  provenance rows for every feature with the weakest source tier.
- Done when: criteria 6, 7, 8, 9, 10 and 13 pass. 6 (differential against an
  independently written naive implementation) is written by a different
  agent from the one writing the vectorised code.

**WP-9.6 Reconciliation extension** (one agent; needs 9.1; parallel with 9.5)
- *Proposed* `ingest/reconcile.py`: supply `Item.source` from pushed-down
  queries; row count, sum conservation, fan-out ratio, end-of-data feature
  totals; certify the new defect types (`fanout`, `dropped_rows`,
  `torn_read`) through the existing certificate machinery.
- Done when: criterion 11 passes and the certificate is on file.

**WP-9.7 Budgets** (one agent; needs 9.1)
- *Proposed* `ingest/budget.py`: cost estimators from the catalog, stage
  gates, deterministic stop boundaries, partial-output marking.
- Done when: criterion 12 passes.

**WP-9.8 External corpora** (human lane, starts on day 1)
- Obtain and check licence, availability and hashes for a relational
  database set with declared keys and for the ER benchmarks; register in
  `benchmarks/qualification/corpora.json` as `candidate`, flip to `verified`
  by the existing rule (`corpora.py:3-7`). Write the independent naive as-of
  reference through a person not otherwise on the phase.
- Done when: manifests verified; if the key/FK corpus cannot be obtained,
  the grain and join gates remain `descriptive` and say so.

**WP-9.9 Wiring and CLI** (one agent; last)
- `scmle ingest` (profile, grain, joins, entities, plan), hook before
  `custody` in `scmle/core/stages.py`, charter fields (format bump),
  provenance table export to F4, assembled frame to F2 with the resolved
  entity as the group column, `--check ingest`.
- Done when: criterion 14 (end-to-end multi-table fixture) passes on 3.11 and
  3.14.

## Parallel lanes

```
WP-9.8 (human, long) -----------------------------------------+
WP-9.1 --+--> WP-9.2 --> WP-9.3 --+                            |
         |                         +--> WP-9.5 --+              |
         +--> WP-9.4 (parallel) ---+             +--> WP-9.9 <--+
         +--> WP-9.6 ------------------------- -+
         +--> WP-9.7 -----------------------------+
```

9.1 first. Then 9.2, 9.4, 9.6, 9.7 in parallel. 9.3 after 9.2. 9.5 after 9.3.
9.9 last.

## Exit gate

```bash
uv run pytest -q tests/unit tests/property tests/leakage -k "ingest or grain or asof_view or entity or synthes"
uv run python -m benchmarks.qualification --check ingest
uv run python -m benchmarks.qualification --check          # certificates still current
```

Numbers (R = 500 worlds per defect or world type unless stated; Wilson
intervals, 95%):

- Torn read: detection lower bound >= 0.99; false detection on static sources
  = 0 of 500.
- Point-in-time: leaked rows = 0 across all planted late-arrival and
  backfill worlds (R = 500) **and** the event-time-only baseline leaks > 0 in
  those worlds (otherwise the world proves nothing). Differential test: exact
  equality over 1000 random schemas. Metamorphic: 0 failures in 10^4 cases.
- Key inference on the declared-key corpus: recall and precision lower bounds
  >= 0.90 (targets; set again from the first measured run, then frozen).
- Unit verdict: >= 0.95 correct; `coarser` reported as `match` in 0 cases.
- Join discovery: auto-accepted FK precision >= 0.95; 0 spurious accepts on
  200 surrogate-key schemas.
- Entity resolution: pair completeness >= 0.95; cross-split link recall >=
  0.95 on each benchmark, else grouping stays `unverified`.
- Reconciliation: planted-defect detection lower bound >= 0.95; false alarm
  upper bound <= 0.05 on clean worlds.
- Budget: spend within budget plus the declared overhead on 100 randomized
  budgets; identical stop boundaries on repeat.
- Multiplicity: declared family size equals registered `synthesized`
  hypotheses.
- Mutation testing on `asof_view.py` and `synthesize.py` (existing mutation
  setup; the surviving-mutant bound is the one fixed for F6 modules in the
  build configuration, to be checked): all mutants that weaken the cutoff
  comparison are killed.

## Evidence required

Seeded-run artifacts listed in the feature file; the corpus manifest entries
with hashes; the key, FK and ER measurement tables with intervals; the
late-arrival world definitions with the baseline-leaks demonstration; the
naive reference implementation and its author; the mutation report;
reconciliation certificate.

## Risks

- **`known_time` is often absent.** Many warehouses keep only event time or
  only current state. Then the correct outcome is a lower tier, not a pass.
  The phase must not hide this; the feature manifest shows the tier per
  feature and per source.
- **Keys by accident.** Small tables make every high-cardinality column look
  like a key. Mitigation: report `N` and cardinalities, require corroboration
  for auto-accept, send the rest to the question list.
- **Corpora may not fit.** Public relational sets have clean declared
  constraints that real warehouses lack; ER benchmarks are mostly bibliographic
  and product text. Mitigation: report the corpus domain beside every
  number; claims transfer only to similar data; planted-defect worlds cover
  mechanism, not prevalence.
- **Feature explosion.** DFS grows combinatorially. Mitigation: grammar,
  depth 2, `B_feat` cap, family size declared in advance, label-free
  pre-filters only.
- **Source load.** Aggregate push-down still scans big tables. Mitigation:
  budget stage and estimates; stop deterministically.
- **Credential and read-only handling.** Enforcement of read-only is not in
  the current connector (reasoned from a grep). Mitigation: WP-9.1 refuses
  write statements; recommended deployment uses a read-only database role,
  which the instrument cannot verify.
- **Dependency on F2's group column.** ER output only helps if the split
  uses it; WP-9.9 sets the group column from the resolved entity and tests
  that the chosen splitter receives it.
- **Reference-implementation drift.** The naive as-of reference and the fast
  implementation can share a misreading of the spec. Mitigation: different
  authors, plus the serving-log comparison where available.

## Explicitly out of scope

LLM-based semantic column interpretation or feature proposals; change-data-
capture or replication tooling; graph, image, text and other modality
feature extraction (existing `scmle/data/modalities/`); automatic repair of
dirty data (F5 detects, it does not repair); privacy and PII governance;
real-time or online feature serving; target-aware feature selection outside
the ledger; cross-database federation beyond the connectors that exist.
