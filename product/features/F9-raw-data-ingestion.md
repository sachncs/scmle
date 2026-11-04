# F9: Raw-data ingestion and feature construction

Vocabulary and principles: [`../../plan/00-canon.md`](../../plan/00-canon.md).
Phase: [Phase 9](../../plan/phase-9-raw-data-ingestion.md). Needs F2
(entity-grouped sets), F3 (ledger), F4 (provenance, as-of), F5 (duplicates
certificate) and, for the unit, F1/F8.

## Status

The method is built as a library plus `scmle ingest` (what it does and how it
was measured: `CHANGELOG.md`, `docs/ARCHITECTURE.md` "Raw-data ingestion",
`docs/ARTIFACTS.md`, `benchmarks/qualification/ingest_certify.py`). F9 is
deliberately LLM-free: every inference is a statistical test or a
deterministic search. LLM-proposed features or column readings, if ever
added, enter only as hypotheses through the same view and the F3 ledger, and
no guarantee may rest on them without a labelled corpus of correct
interpretations scored as in F8.

This file keeps only what is not yet true.

## What is missing

1. **Connectors are not called by a charter run.** `scmle ingest` and
   `IngestPlan` read a database; `scmle run` still reads a file. No charter
   field carries the budget, `B_feat`, windows, `lag`, `delta_rec` or the
   tolerances, and there is no stage before `custody`.
2. **No external referent for three gates.** Key and foreign-key precision
   and recall on schemas with declared constraints, and entity-resolution
   precision and recall on labelled benchmarks, have not been measured, so the
   grain, join and entity gates are descriptive (P1). The synthetic numbers in
   `ingest_measurements.json` show the mechanisms work on worlds we built; they
   say nothing about prevalence in real warehouses.
3. **The resolved entity does not reach the split**, and the synthesized
   features' provenance table is not exported to F4 by a run.
4. **No real known-time.** The view's guarantee is about the code and holds
   only if the recorded time is true. No serving-time feature log has been
   compared with the view's output.
5. **Review.** The as-of reference was written separately from the code by an
   agent, not by a person outside the project; no data-engineering reviewer
   has signed.

## Acceptance criteria still open

- **2. Key inference on a declared-key corpus**: recall and precision Wilson
  lower bounds >= 0.9 at `L = 3`; targets to be re-set after the first run
  and frozen (*unverified* until measured). The planted near-key threshold
  already flips within one row.
- **4. Join discovery on that corpus**: auto-accepted FK precision >= 0.95;
  recall reported. The surrogate-key worlds already show 0 spurious accepts
  in 200 schemas.
- **5. Entity resolution on labelled benchmarks**: pair completeness >= 0.95
  and cross-split link recall >= 0.95 on each, else grouping stays
  `unverified`.
- **14. End to end through a charter run**, identical on Python 3.11 and
  3.14: expected columns, provenance table with correct tiers, passing
  reconciliation.

## External referent

- Declared keys and foreign keys of public relational databases with the
  constraints stripped (ground truth is the schema authors', not ours);
  licence and availability *unverified*.
- Labelled entity-resolution benchmarks with published match labels.
- A re-written reference for as-of construction by a person outside the
  project, and serving-time feature logs where a deployment has them.
- Source-system totals owned by the data's owner.
- A data-engineering reviewer outside the project, using the checklist style
  of `plan/validation-referents.md`.
