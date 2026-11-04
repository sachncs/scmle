# F9: Raw-data ingestion and feature construction

Vocabulary and principles: [`../../plan/00-canon.md`](../../plan/00-canon.md).
Built in [Phase 9](../../plan/phase-9-raw-data-ingestion.md); needs F2
(entity-grouped sets), F3 (ledger), F4 (provenance, as-of), F5 (duplicates
certificate) and, for the unit, F1/F8. F8 and F9 are not yet listed in the
canon tables; the canon owner adds them there first.

## Purpose

Today the instrument starts from a single table that someone else built. F9
covers the step before: read raw sources (SQL, warehouse, event logs, files),
infer the table grain and keys, resolve duplicate entities, discover how
tables join, build features from related tables, and hand the result to
F2/F4 with **point-in-time correctness enforced by construction**, reconciled
to source totals, inside declared cost and time budgets.

## Decision it serves

- Is the unit in the charter the grain of the data, and can this table be
  keyed on it? (Mismatch blocks the run or raises a question.)
- Which tables may be joined, on which keys, and with what cardinality.
- Which synthesized features enter exploration, and which are excluded as
  unverifiable in time.
- Does the assembled table agree with the source systems, or is the run
  marked `unverified`?

## Scope statement and the LLM boundary

F9 is deliberately **LLM-free**. Every inference below is a statistical test
or a deterministic search with a measured operating characteristic. LLM-
proposed features or semantic column interpretations are out of scope; if
added later they enter only as hypotheses that go through the same
cutoff-enforcing view and the F3 ledger, and no guarantee may rest on them
without an external check (a labelled corpus of correct interpretations,
scored as in F8).

## What exists today (verified)

| Capability | State | Evidence |
|---|---|---|
| File readers: csv, tsv, json, jsonl, xlsx, parquet, feather | exists | `DatasetLoader` `scmle/data/readers/tabular.py:123`; reader classes `:43-113`; `read_table` `:199` |
| Connectors: local file, DB-API, warehouse (callbacks), object store, streaming object store | exist, batch/checkpoint/retry | `SourceConnector` `scmle/data/readers/connectors.py:422`, `LocalFileConnector` `:491`, `DbApiConnector` `:597`, `WarehouseConnector` `:736`, `ObjectStoreConnector` `:886`, `StreamingObjectStoreConnector` `:1044`; `RetryBudget` `:163` |
| Source identity, snapshot and schema-drift detection | exists | `SourceSnapshot` `scmle/data/readers/registry.py:18`; `compare_schemas` `:151`; `SourceCheckpoint` `connectors.py:244` |
| Event stream with sequence, event time, late-data policy | exists | `IncrementalEvent` `scmle/data/readers/incremental.py:75` (fields: `event_id`, `sequence`, `event_time`, `payload`; **no separate ingest/transaction time**); `LateDataPolicy` `:53` |
| As-of reconstruction of an *existing* column from a change log | exists | `point_in_time` `scmle/data/admissibility/asof.py:59` (one `pd.merge_asof`, `:103`, the only as-of join in `scmle/`); `audit_field` `:204`; `log_provenance` `:282` |
| Field provenance and tier | exists | `FieldProvenance` `scmle/data/admissibility/provenance.py:242`; `tier` `:304`; `undeclared` `:376` |
| Reconciliation of caller-supplied totals | exists | `Item` `scmle/data/admissibility/reconciliation.py:98`; `reconcile_item` `:166`; `reconcile` `:263` |
| ID-like column heuristic | exists, per column | `ID_UNIQUE_RATIO = 0.95` `scmle/data/validation/columns.py:27`, used `:410` |
| Exact and near duplicate rows within one table | exists | `duplicate_groups` `scmle/data/splitting/duplicates.py:18`; `near_flags` with blocking `scmle/data/detectors/duplicates.py:89` |
| Group-disjoint splitting on a caller-named group column | exists | `scmle/data/splitting/splitters.py:496-509` |

## What is missing (verified by reading and grepping)

1. **Connectors are not wired to a run.** Outside `readers/` and
   `tests/unit/test_connectors.py` / `test_incremental_readers.py`, only
   `scmle/api/project.py:29` imports them, and `:421` uses only
   `connectors.source_identity`. A run reads a file.
2. **Connector reads are offset-based re-executions.**
   `DbApiConnector.retrieve_rows` (`connectors.py:677-698`) re-runs the query
   for every batch and skips `offset` rows by fetching them
   (`skip_rows`, `:718`). Without a caller-supplied `ORDER BY` and a stable
   snapshot, batches can overlap or miss rows if the table changes during
   the read; the cost per batch grows with the offset. Whether the injected
   warehouse callbacks do better is up to the caller (`WarehouseConnector`
   `:736`). The "read-only" property of the query is a docstring statement
   (`:618`); no enforcement was found in the file (grep for read-only
   enforcement found none) and is *unverified*.
3. **No table-grain or key inference.** Only the per-column id heuristic
   above. No search for composite keys, no functional dependencies, no check
   that the charter's `unit` (free text, `scmle/data/charter/contracts.py`
   `:314-325`) matches a key.
4. **No cross-table entity resolution.** Duplicate detection is within one
   table. Entities that appear under different keys in two sources are not
   linked, so a grouped split by `group_column` can put the same real entity
   in `X` and `C`.
5. **No join discovery and no multi-table feature construction.** Grep over
   `scmle/` finds no `.merge(` call and no inclusion-dependency code; the
   one `merge_asof` is the audit above.
6. **As-of is audited, never constructed.** `audit_field` compares a stored
   column to a log. Nothing builds a feature as of a cutoff, and provenance
   is per column, not per derived feature.
7. **Reconciliation takes numbers from the caller.** It does not query the
   source (`Item.source` is a float, `reconciliation.py:98-113`).
8. **No cost or time budget** on ingestion beyond per-operation retry
   (`RetryBudget`, `connectors.py:163`).

## Behaviour

Proposed interface (nothing below exists):

```python
plan = IngestPlan.open(run_dir, charter, budget=IngestBudget(...))
catalog = plan.profile(connectors)        # exact aggregates pushed down
grain = plan.infer_grain(catalog, charter.unit)    # keys, FDs, verdict
links = plan.discover_joins(catalog)      # candidate FKs with scores
ents  = plan.resolve_entities(catalog)    # clusters + estimated P/R
view  = plan.as_of_view(table, key, time, cutoff_col)   # the ONLY read path
feats = synthesize(view_set, target_table, depth=2, budget=...)
rec   = plan.reconcile(assembled, sources)   # row counts, sums, fan-out
frame = plan.assemble()                   # table + provenance table
```

Artifacts: `ingest_catalog.json` (per table: row count, column types,
distinct counts, snapshot id), `grain_report.json`, `join_graph.json`
(candidate links with containment, cardinality, score, decision),
`entity_resolution.json` (parameters, thresholds, clusters, estimated
precision and recall with intervals), `feature_manifest.json` (every
synthesized feature: expression, tables read, cutoff rule, as-of source,
tier), `reconciliation.json` (existing format, extended), `ingest_budget.json`
(spent versus allowed).

## Method

One method per need. Cut items are listed in each subsection.

### 1. Snapshot-consistent, push-down reads

All statistics in sections 2 to 4 are computed by **aggregate queries run
in the source** (`COUNT`, `COUNT(DISTINCT ...)`, anti-join counts), not by
pulling rows. Reason: uniqueness and inclusion are universal statements; a
sample can refute them but cannot confirm them. If a fraction `q` of rows
share their key with another row, the expected number of such pairs seen in
a uniform sample of `m` of `N` rows scales as `(m/N)^2` per pair, so small
samples miss duplicate keys with high probability (derived by expectation
here, not verified by simulation). Row extraction for modelling uses the
existing batch connectors but requires: an explicit total order (`ORDER BY`
on the key), and a snapshot token (transaction snapshot, `AS OF`, or a
content hash taken before and after). A read whose before/after token or
row count differs is rejected as torn. The token is stored in
`SourceSnapshot.watermark` (`registry.py:18`, field `watermark`).

Cut: sampling-based key inference (cannot certify), and CDC/replication
tooling (out of scope; the event-log path uses `IncrementalEvent`).

### 2. Grain and key inference: TANE-style lattice with the g3 error

Candidate keys and functional dependencies are searched level-wise over
column subsets of size at most `L = 3` (proposed), using partition
refinement as in TANE (Huhtala et al. 1999, from memory). The tolerance
measure is `g3(X -> A)`: the minimum fraction of rows to delete so the
dependency holds exactly. A set `X` is an **approximate key** when
`g3(X -> all) <= eps_key`. In SQL, `g3(X)` for the key case is
`1 - (COUNT(DISTINCT X) / COUNT(*))` bounded below by the duplicate
fraction; the exact form is a grouped count query.

Decision rules:

- Rank candidate keys by size, then by `g3`, then by null rate; ties go to
  the human question list (F8), not to a guess.
- **Verdict against the charter unit:** the unit text maps to a key set
  (entity key, or entity key plus period for "customer-month") by matching
  against ranked candidates. The verdict is `match`, `finer` (data are at a
  finer grain than the unit: aggregation required), `coarser` (cannot build
  the unit; refuse), or `unknown` (ask).
- Null keys are counted separately; a key with nulls above `eps_null` is
  not a key.

Honest limit: whether a combination is a key *by design* or *by accident of
this snapshot* is not statistically decidable. A low-cardinality accident is
rare for wide keys, but for small tables with high-cardinality columns
every column looks like a key. The report gives `N`, the distinct counts and
a flag when `N` is small relative to the candidate's column cardinalities.

Cut: HyFD / FDX-style discovery (faster at width but less transparent for
approximate FDs and not needed at `L = 3`); statistical-FD estimation via
correlation structure; any LLM reading of column names.

### 3. Join-path discovery: exact inclusion dependencies, conservative accept

For a candidate pair (child column set `A`, parent key `B`), measure the
**containment** `|A ∩ B| / |A|` (distinct values) with an anti-join count in
the source. Candidates must satisfy: `B` is an approximate key of the parent
(section 2), types compatible, containment `>= 1 - eps_fk`, and cardinality
child-to-parent many-to-one. Accepted automatically only when (a) a declared
foreign key exists, or (b) containment is exact and a name-similarity or
value-semantics corroboration holds. Everything else is `candidate` and is
listed as a question. This is because inclusion alone is weak evidence:
surrogate integer keys `1..N` of unrelated tables include each other by
chance (a known failure of inclusion-dependency FK discovery; Rostin et
al. 2009, from memory, and the fix there was a learned classifier).

The join graph is a directed multigraph; path enumeration is limited to
depth `d <= 2` and to many-to-one steps from the target table outward
(parent attributes), and one-to-many steps (child aggregates) only through
the as-of view of section 5.

Cut: learned FK classifiers (need a training corpus; revisit if precision
below target), multi-column inclusion dependencies beyond the declared
composite keys, and automatic acceptance of many-to-many links (they are
always reported, never joined).

### 4. Entity resolution: Fellegi-Sunter with blocking, validated by sampled labels

Records across or within sources are compared on a field vector. Model
(Fellegi and Sunter 1969): for each field comparison level `k`,
`m_k = Pr(level k | match)`, `u_k = Pr(level k | non-match)`; weight
`w = sum_k log2(m_k/u_k)` under conditional independence; two thresholds
`T_lo < T_hi` give match / possible / non-match, chosen to satisfy the
tolerated error rates `mu` (false match) and `lambda` (false non-match).
Parameters by EM on unlabelled pairs (Winkler's approach, from memory) or
from a labelled sample when available. **Blocking** limits comparisons to
pairs sharing at least one key from a multi-pass set (exact keys, phonetic
or normalised-prefix keys), with measured **reduction ratio** (share of the
`N(N-1)/2` pairs avoided) and **pair completeness** (share of true matches
retained) as in Christen (2012, from memory).

Why this method: it is transparent, produces calibratable scores, and its
operating characteristics are measurable on public labelled benchmarks.
The conditional-independence assumption is known to be false in practice,
so the model's own probabilities are **not** trusted: thresholds are set,
and precision and recall are reported, from a **stratified clerical sample**
of pairs by score band, whose labels come from the benchmark (in
validation) or a named reviewer (in use). Without labels the result is
tier `unverified` and entity links are used only to *prevent leakage*
(link-then-group, which errs toward keeping more rows together), never to
merge attribute values.

Why it matters to the instrument: F2 splits by a group column
(`splitters.py:502`). An unresolved duplicate entity under two ids is a
leak across `X` and `C`. The gate quantity is therefore **cross-split link
recall**, not average pair F1.

Cut: embedding or LLM matching (no calibration, not auditable), supervised
deep ER (needs labels per deployment), and probabilistic record linkage with
dependence terms (revisit if the benchmark precision falls below target).

### 5. Point-in-time correctness by construction: the as-of view

**Rule.** Every read of a non-target table goes through
`AsOfView(table, key, event_time, known_time, cutoff)`. For each target row
`i` with entity `e_i` and prediction moment `t_i` (the charter's prediction
moment, an F1 field), the view returns only rows with
`known_time <= t_i - lag`. `known_time` is the **transaction time** (when the
system recorded the row) if the source has one; else `event_time` plus a
declared maximum recording delay `delta_rec`; else the table is
**non-versioned** and its time-dependent columns are classed as having no
as-of rule. The synthesizer's API never exposes the underlying frame, only
the view; there is no code path to read unfiltered child rows.

Why event time alone is not enough: a row with `event_time <= t_i` can be
written after `t_i` (late arrival, backfill); using it is leakage that an
event-time filter does not catch. `IncrementalEvent` has only `event_time`
and `sequence` (`incremental.py:75`), so for event logs, `known_time` comes
from the sequence/commit time when the source gives one, otherwise from the
declared delay (attested, tier `attested` at best).

Mutable dimension tables (current customer segment, current address) keep no
history: a value read now is the value *now*. Features from them are
admissible only if a change log exists (then they pass through
`asof.audit_field`, `asof.py:204`, and become tier `confirmed` if the audit
passes, `provenance.py:304-325`) or an attestation exists (tier `attested`);
otherwise they carry `provenance.undeclared` (`provenance.py:376`, tier
`unverified`) and the existing timing rule excludes them in strict mode
(`rules.py:104`).

Statistics cannot show that a feature predates a moment (canon P6); what is
enforced here is that *the code only reads rows whose recorded time
satisfies the rule*, and that recorded time is only as good as its source,
so each feature inherits the weakest tier among the tables it reads.

**Feature synthesis (Deep Feature Synthesis style; Kanter and
Veeramachaneni 2015, from memory).** A fixed grammar of primitives:
`count`, `sum`, `mean`, `min`, `max`, `std`, `nunique`, `last`, `time since
last`, `slope over window` over child rows in windows `{7, 30, 90}` days
(declared), depth at most 2, parent-attribute lookups. Cap: `B_feat` features
(charter field). Label-free filters first (null rate, constant, duplicate of
another feature) because they are allowed to read all rows (P3). The
surviving candidate set size is **declared before exploration**, so the F3
ledger plans wealth for a known family size; each feature is a hypothesis
registered in F3, with `family="synthesized"`.

Cut: genetic or learned feature construction, LLM-proposed features (see the
scope statement), target-aware feature selection before the ledger (that
would shape the model with `X`-only information outside the ledger), and
unbounded depth.

### 6. Reconciliation to source totals

After each extraction, join and aggregation:

- Row count: assembled table vs source `COUNT(*)` at the snapshot token.
- Conservation: `SUM(measure)` on the base table before and after each join
  (a many-to-one join must preserve it; a failure indicates fan-out).
- Fan-out: output rows / left rows must equal 1 for many-to-one joins;
  otherwise a many-to-many link exists.
- Feature totals: for count and sum features, the total over all target
  entities at the **end-of-data cutoff** equals the source aggregate.

Extends the existing `Item` / `reconcile` (`reconciliation.py:98, :263`) by
supplying `Item.source` from a pushed-down query instead of the caller.
Mechanisms with explanations (`Explanation`, `:52`) and tiering are reused.

Cut: row-level diff against source (cost grows with `N`; aggregates catch
the planted defects measured below), and tolerance learned from data
(tolerance is a charter field, as now).

### 7. Cost and time budgets

`IngestBudget`: maximum source-bytes scanned, maximum wall time, maximum
number of source queries, `B_feat`. Each stage estimates cost before it runs
(from catalog row counts) and runs only if the estimate fits the remaining
budget. On exhaustion the stage stops at a **declared deterministic
boundary** (end of the current table or level), records what was not done,
and the dependent outputs are marked partial and `unverified`. Nothing is
silently truncated.

Cut: adaptive sampling to save cost (violates section 1).

## Gates, operating characteristics and referents (P1)

| Gate (binds to action) | Operating characteristic measured | External referent |
|---|---|---|
| Grain verdict `coarser`/`unknown` -> `block_run` or question | Key precision and recall with Wilson intervals; unit-verdict accuracy | Databases with **declared** primary and foreign keys, constraints stripped before inference (e.g. the CTU relational learning repository; *unverified* that it is reachable and that its licence allows use) |
| Join accepted -> allowed in synthesis | FK precision and recall vs declared FKs; spurious-link rate on surrogate-key worlds | Same corpus; plus synthetic schemas with planted coincidental inclusions |
| Entity link used for grouping | Pair precision, recall, reduction ratio, pair completeness, **cross-split link recall** | Labelled ER benchmarks (Febrl-style synthetic; DBLP-ACM, Abt-Buy from the Magellan/Leipzig collections; *unverified* availability and labels) |
| Feature read allowed | Leaked-row count = 0 on planted late-arrival and backfill worlds; equality to an independent reference implementation | Naive row-by-row reference written independently of the vectorised code; serving-time feature logs where a deployment has them |
| Reconciliation block | Detection rate of planted fan-out, dropped rows, torn reads; false alarm on clean worlds (Clopper-Pearson) | Source-system totals; existing F4 reconciliation certificate (`benchmarks/qualification/f4_certificates.json`) |
| Budget stop | Adherence: spend <= budget (+ stated overhead), stop point deterministic | The cost model's own estimate vs measured spend |

## Guarantees and assumptions

- **Time rule:** no synthesized feature for target row `i` reads a source
  row whose recorded `known_time` exceeds `t_i - lag`. Guaranteed by the
  type (only the view exists), checked by tests. This is a guarantee about
  the code, not about the world: it holds only if `known_time` is true. A
  table with no trustworthy `known_time` yields a lower tier, never a pass.
- **Keys and FKs:** statements about the snapshot read, with `g3` and
  containment reported. Not statements about design intent.
- **Entity resolution:** estimated precision and recall with intervals from
  a labelled sample; unlabelled use is `unverified` and only conservative
  (more rows grouped).
- **Reconciliation:** agreement of aggregates, not of rows; it can pass with
  offsetting row-level errors.
- **Multiplicity:** synthesized features multiply hypotheses; F3 accounts for
  them only if the family size is declared in advance.

## Failure modes and degradation

| Situation | Behaviour |
|---|---|
| Torn read (token or count changes during read) | Reject the read, retry within `RetryBudget`, then `unverified` run. |
| No unique key at declared tolerance | Grain verdict `unknown`; question to a human; no synthesis. |
| Unit finer than data (`coarser`) | Refuse with the missing grain stated. |
| Table has no `known_time` and no log | Its columns are `undeclared`, excluded in strict mode; included only as `unverified` in permissive mode. |
| Late arrivals beyond `delta_rec` | Counted; if the share exceeds the declared tolerance, `known_time` source is downgraded. |
| Many-to-many link | Reported, not joined. |
| Candidate FK from coincidental inclusion | Stays `candidate`; asked, not joined. |
| ER blocking misses cross-split matches | Reported through pair completeness; grouping marked `unverified`. |
| Budget exhausted | Partial outputs marked, stage boundary recorded. |
| Source schema drift mid-run | Existing `compare_schemas` (`registry.py:151`) flags it; run stops at the stage boundary. |
| Credentials in connection strings | Never written to artifacts; the factory is injected (existing design, `connectors.py:597-640`). |

## Repo integration

- Reuse: connectors, `SourceSnapshot`/`SourceRegistry`, `IncrementalEvent`,
  `point_in_time`, `audit_field`, `FieldProvenance`, `reconcile`,
  `CertificateStore`/`GateStatus` (`scmle/data/detectors/certificates.py:58`),
  `duplicate_groups` and the blocking code in `detectors/duplicates.py`.
- Extend: `DbApiConnector` ordered, snapshot-token reads (replace
  skip-by-fetch for sources that support keyset pagination); add an enforced
  read-only mode (statement allow-list and read-only transaction where the
  driver supports it).
- New proposed package `scmle/data/ingest/` (`profile.py`, `grain.py`,
  `joins.py`, `entities.py`, `asof_view.py`, `synthesize.py`,
  `reconcile.py`, `budget.py`); new check `ingest` in
  `benchmarks/qualification/__main__.py` (choices at `:75` today:
  `certificates`, `fdr`).
- Charter additions (proposed, format bump of `contracts.py:30`):
  `ingest_budget`, `feature_budget` (`B_feat`), window set, `lag`,
  `delta_rec`, tolerance fields.
- Output feeds: the assembled frame and `feature_manifest.json` go to F2
  (split, with the resolved entity as group column) and F4 (provenance
  table built from the manifest's as-of sources).

## Acceptance criteria (machine-checkable)

1. **Torn read.** A fixture DB mutated between batches is detected in
   >= 99% of 500 runs (Wilson lower bound) and 0 undetected on a static DB
   in 500 runs; batch outputs are disjoint and complete on static data.
2. **Key inference.** On the stripped-constraints corpus, primary-key recall
   Wilson lower bound >= 0.9 and precision >= 0.9 at `L = 3`; on synthetic
   worlds with planted near-keys at `g3 = eps_key` the verdict flips at the
   threshold (tolerance 1 row). Numbers are targets to be re-set after the
   first corpus run and then frozen (*unverified* until measured).
3. **Unit verdict.** On 200 synthetic unit/grain pairs, the verdict is
   correct in >= 95% (`match`, `finer`, `coarser`), and `coarser` is never
   reported as `match` (zero tolerance for this error type).
4. **Join discovery.** FK precision >= 0.95 among auto-accepted links on the
   corpus; on surrogate-key worlds, auto-accepted spurious links = 0 in 200
   schemas; recall reported.
5. **Entity resolution.** Blocking pair completeness >= 0.95 and reduction
   ratio reported; pair precision and recall with intervals from the
   stratified labelled sample; cross-split link recall >= 0.95 on the
   benchmarks, else grouping is `unverified`.
6. **Point-in-time, differential.** Synthesized features equal an
   independent naive reference implementation exactly on 1000 random
   schemas and cutoffs.
7. **Point-in-time, metamorphic.** For every feature, replacing all child
   rows with `known_time > t_i - lag` by arbitrary values leaves the
   feature unchanged (property test, 10^4 cases, zero failures).
8. **Late-arrival world.** On planted late-arrival and backfill worlds, the
   event-time-only baseline leaks (positive leaked-row count, demonstrating
   the world is real) and the view leaks zero rows.
9. **No raw read path.** An import-graph / grep test: `synthesize` imports
   no connector or reader; only `asof_view` does.
10. **Mutable-dimension test.** A feature from a non-versioned table with no
    log or attestation is `unverified` and is excluded in strict mode.
11. **Reconciliation.** Planted fan-out, dropped-row and torn-read defects
    detected at rate >= 0.95 (Wilson lower bound over 500 each); false alarm
    on clean worlds <= 0.05; certificate recorded before the check gates.
12. **Budget.** On 100 randomized budgets, spend never exceeds the budget by
    more than the declared overhead; stop boundaries are identical on two
    runs with the same seed.
13. **Multiplicity.** The declared family size equals the count of
    registered `synthesized` hypotheses; a mismatch fails the run.
14. **End to end.** A seeded multi-table fixture (customers, orders, events
    with late rows, a mutable dimension) yields a frame with the expected
    columns, a provenance table with correct tiers, and passing
    reconciliation, identical on Python 3.11 and 3.14.

## External referent

- Declared keys and foreign keys of public relational databases, with the
  constraints stripped (ground truth is the schema authors', not ours).
- Labelled entity-resolution benchmarks with published match labels.
- An independently written reference implementation for as-of feature
  construction, and real serving-time feature logs where a deployment
  provides them (the "rawest material" listed in `plan/README.md`).
- Source-system totals owned by the data's owner.
- A data-engineering reviewer outside the project, using the checklist
  style of `plan/validation-referents.md`.
