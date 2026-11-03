# Phase 2: Charter and custody (F1, F2)

Names follow [`00-canon.md`](00-canon.md). Runs in parallel with Phase 1 after
Phase 0.

## Goal

A signed, hashed analysis charter fixes the question, estimand, set sizes,
error budget and decision map before any data is profiled. Rows are split into
`X`, `C` and (temporal data) `F`, with `C` and `F` held in sealed custody the
exploration code cannot read.

## Entry criteria

Phase 0 merged (`Tier`, `ErrorBudget`, seeded RNG).

## Verified starting points

- `SplitManifest` is frozen and has `fingerprint`, but the hash is
  `sha256(repr(...))` truncated to 16 hex characters
  (`scmle/data/splitting/splitters.py:32-62`). Too weak and too fragile for a
  manifest; a v2 is *proposed*, old field left untouched.
- `LockedTestSet` (abstract, `splitters.py:101`) and `ResidentTestSet`
  (`splitters.py:140`) give a one-time, token-gated release; the sealed
  variant `SealedTestSet` (`scmle/governance/vault.py:56`) moves rows to a
  separate process via `ipc.start_helper`. The docstring itself says Python
  cannot hide attributes in-process (`splitters.py:146-150`): the stated
  limit in P6.
- `stages.py` seals the test rows (`scmle/core/stages.py:551-564`) and profiles
  the development partition only (the `profile(...)` call near
  `stages.py:400`). The three-way split extends this seam.
- `scmle/data/splitting/duplicates.py` keeps duplicate groups on one side of a
  two-way split (`keep_together`); it must handle three sets.
- Signing exists: `scmle/contracts/signing.py`.
- Group and temporal splitters exist in `splitters.py`; embargo gaps do not
  (*unverified*; confirm by reading before WP-2.4).

## Lane A: Charter (F1)

**WP-2.1 Charter contracts** (one agent)
- *Proposed* `scmle/data/charter/`: `AnalysisCharter`, `AnalysisEstimand`
  (population, target, prediction moment, intercurrent events, summary
  measure, type), `DecisionMap` (check id to action), `ErrorBudget` reuse.
- Done when: canonical-JSON hashing is stable across platforms; a charter
  with a gating check absent from the decision map is rejected.

**WP-2.2 Feasibility and power** (one agent)
- *Proposed* `charter/feasibility.py`: minimum detectable effect for each set
  size and the CI width of the headline metric on `C`; verdict `feasible`,
  `downgrade` (claims capped at "exploratory, unconfirmed") or `infeasible`.
- Done when: matches closed-form power for a two-proportion and a correlation
  test to 1e-6, and an oracle fixture from Phase 1 once available.

**WP-2.3 Sign, store, CLI** (one agent; depends on 2.1)
- Sign with the existing signing contract; add `scmle charter init|check|sign`
  in `scmle/cli.py`; document in `docs/CLI.md`.
- Done when: tampering with any charter byte fails verification.

## Lane B: Custody (F2)

**WP-2.4 Three-way splitter** (one agent)
- Extend splitters with `X/C/F`, group-aware and time-ordered, with an
  embargo gap for autocorrelated data; confirmation drawn within the
  exploration period (replication), forward set drawn later (stability).
  Extend `keep_together` to three sides.
- Done when: zero group or duplicate overlap across sets (assert on all of
  the property tests), embargo respected, split is a pure function of
  (data hash, charter hash, seed).

**WP-2.5 Sealed sets and open log** (one agent)
- Generalise `SealedTestSet` to named sets with per-set tokens and a
  hash-chained open log (*proposed* `data/custody/`); `C` opens once.
- Done when: an access-attempt test fails the build if code given only `X`
  can reach `C` or `F` rows by attribute, pickling or the log; a second open
  raises `LeakageError` (the existing exception at `vault.py` open path).

**WP-2.6 Manifest v2** (one agent; depends on 2.4)
- Full 64-hex sha256 over canonical JSON of ids, strategy, seed, embargo,
  charter hash; `split_manifest.json` written next to `dataset_profile.json`.
- Done when: round trip and corruption tests; v1 manifests still readable.

**WP-2.7 Pipeline wiring** (one agent; depends on 2.1, 2.4, 2.5, 2.6)
- Call charter check and the new split from `stages.py` before profiling; feed
  only `X` to `DatasetProfiler`. Opt-in via the charter so existing runs keep
  behaviour.

## Parallel lanes

A (2.1, 2.2, 2.3) and B (2.4, 2.5, 2.6) run concurrently; 2.7 joins them.

## Exit gate

```bash
uv run pytest -q tests/unit tests/property tests/leakage \
  tests/integration/test_end_to_end.py tests/integration/test_checkpoints.py
uv run python scripts/check_api_compatibility.py tests/fixtures/public_api.json
```
Numbers: 0 overlapping ids across sets in 1,000 Hypothesis cases; sealed-set
access test passes; feasibility matches closed forms to 1e-6; existing
integration suite unchanged (0 new failures); mutation score >= 70 on
`custody/`.

## Evidence required

Signed sample charter, split manifest with hashes, access-attempt test log,
feasibility table for five reference dataset sizes.

## Risks

- Small datasets cannot afford three sets: feasibility `downgrade` is the
  designed outcome, not an error.
- Custody is tamper-evident, not adversary-proof on one machine (P6); the docs
  must say so.
- Changing split behaviour could shift old results: opt-in via charter.

## Explicitly out of scope

The ledger and confirmation test (Phase 3), provenance, detectors.
