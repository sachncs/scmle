# F2: Custody and splits

Canon: [`../../plan/00-canon.md`](../../plan/00-canon.md). Phase 2.

## Purpose

Partition the data into exploration `X`, confirmation `C` and, for temporal
data, forward holdout `F`, before any profiling, and keep `C` and `F`
physically out of reach of exploration code until a single, logged unlock.

## Decision it serves

Whether a later claim may be called `confirmed`. The `confirmed` tier is only
meaningful if no information from `C` or `F` reached the choices that produced
the claim (P2, P3).

## Behaviour

Today (verified): one holdout. `SplitManager.split` partitions, keeps
duplicate rows together for the random/stratified strategies
(`scmle/data/splitting/splitters.py:27,784`), and returns a `ResidentTestSet`
(`splitters.py:140-180`). Rows are moved to a vault process when
`context.resource_governor.hard` is set (`scmle/core/stages.py:629-635`). That
is the default: `enforcement` defaults to `"hard"` (`scmle/contracts/tasks.py:363`,
`scmle/api/project.py:473`). Isolation falls back to an in-process protocol only
when the caller passes `enforcement="cooperative"` (the test helper does, for
speed: `tests/support.py:25`). The test set opens once (`stages.py:1545-1546`).

Verified residual gaps, which F2 closes: (1) custody is coupled to a resource
supervision flag, so opting out of supervision silently opts out of custody;
(2) the caller's own DataFrame and the original file still hold every row
(`README.md:238`); (3) only the final test set has custody, not `C` or `F`.

Proposed:

```python
# proposed
manifest = custody.split(frame, charter)  # X, C, F row ids + embargo
view = custody.exploration_view(manifest)  # the ONLY frame exploration sees
vault = custody.seal(manifest)  # C, F rows -> vault process(es)
...
request = custody.UnlockRequest(  # F3 supplies the frozen list
    set="C", charter_hash=..., ledger_hash=..., claims_hash=...
)
rows = vault.open(request)  # once; logged; hash-bound
```

Artifacts: `split_manifest.json` (extends `validation_manifest.json`,
`stages.py:637`), `custody_log.jsonl`.

**Layout.**
- *Non-temporal:* group-disjoint when `group_column` is set
  (`GroupSplit`, `splitters.py:495-536`), else stratified; duplicate groups
  stay on one side. `C` is carved from the development partition.
- *Temporal:* `F` is the latest block (today's chronological holdout,
  `splitters.py:604-627`). Before `F`, the period is cut into contiguous
  blocks at least as long as the embargo; blocks are assigned to `X` or `C` at
  random from the seed; embargo rows at every block boundary are dropped. `C`
  therefore tests replication in the same period and `F` tests stability.

**Reconciliation with today's locked test (proposed, to confirm in canon).**
The existing locked test set stays the model's final-evaluation set. It *is*
`F` for temporal data. For non-temporal data there is no `F`; it is recorded in
the manifest with role `final_test`, and `C` is a separate carve-out. Reusing
`C` for the final model score would let confirmed claims shape the model that
`C` then scores.

## Method

**Blocked, group-disjoint assignment with an embargo.** Units are the
exchangeable atoms: a group, a duplicate cluster or a time block. Assignment
depends only on unit identity, time and (for stratification) labels, never on
feature values. The embargo length is the larger of
`ValidationSpec.embargo_rows` (`tasks.py:348`), the forecast horizon term in
`TemporalSplit.embargo` (`splitters.py:586-602`) and a measured
autocorrelation length (proposed, from the lag where the target ACF falls
inside its 95% band).

Why: it reuses the strategies already tested here, and it is the simplest rule
under which `C` is independent of `X` given the units. Cut: adaptive or
covariate-balanced assignment (peeks at features), per-claim bespoke splits.

## Guarantees and assumptions

- **Disjointness:** `units(X) ∩ units(C) ∩ units(F) = ∅` for groups and
  duplicate clusters, and every `X`/`C` boundary has at least `embargo` rows
  removed. Asserted at build time, recorded in the manifest.
- **Independence for F3:** if units are independent, or blocks exchangeable,
  and the selection `S` is a function of `X` only, then p-values computed on
  `C` for the hypotheses in `S` retain their null distributions. F2 provides
  the premise; F3 uses it.
- **Commitment:** the manifest holds a full SHA-256 over canonical encodings of
  the unit ids per set plus a Merkle root over per-row content hashes. A
  changed row or reassigned unit changes the root.
  Today's `fingerprint` is the first 16 hex characters of a SHA-256 over ids
  only (`splitters.py:53-64`), so it does not detect a modified row.
- **Unlock:** at most one release per set (today enforced twice: vault side
  `vault.py:30-42`, parent side `vault.py:99-101`). Proposed: the vault checks
  that the request's `charter_hash`, `ledger_hash` and `claims_hash` match
  values registered at seal time, so a changed claim list cannot open `C`.
- **Log:** `custody_log.jsonl` is hash-chained (entry `i` embeds the SHA-256
  of entry `i-1`) and each entry is HMAC-signed with the project key
  (`signing.py:36`). Events: seal, token issue, open request, grant or deny,
  close, vault exit. Truncation or edit breaks the chain on verification.

## Limits (stated, per P6)

- The vault is a same-user subprocess with a key passed on stdin
  (`scmle/core/ipc.py:1-8`). It stops accidental and in-code access. A
  process with the same OS privileges can still attach to it or read the
  original file. This is tamper-*evident*, not adversary-proof.
- `SealedTestSet` keeps its token in the orchestrator process
  (`vault.py:76,82`). Anything given a reference to that object can open it.
  Protection therefore depends on exploration code receiving only the
  `ExplorationView`, never the vault object (capability passing, proposed).
- True custody against a hostile operator needs separation: another OS user,
  a service, or a write-once object store. The manifest records which level
  was used (`in-process protocol`, `vault process`, `external`); the report
  shows it, and `confirmed` requires at least `vault process`.

## Failure modes and degradation

| Condition | Outcome |
|---|---|
| F1 feasibility fails (small n) | No `C`. `X` is analysed with nested cross-fitting; `F`/`final_test` stays sealed; claims capped at `unverified`, scope `exploratory_unconfirmed`. |
| Fewer units than `folds + 2` or too few rows after embargo | `BlockedError`, the existing pattern (`splitters.py:505-512`). |
| Time column unparseable | `BlockedError` (`splitters.py:618-622`). |
| Vault cannot start | Protocol isolation, noted in the manifest; tier caps at `attested`. |
| Second open or wrong hash | Denied, logged, `LeakageError` (existing type). |
| Class absent from `X` or `C` | `BlockedError`, reusing `splitters.py:538-578`. |

## Repo integration

- Extend: `SplitManifest` (`splitters.py:31-81`), `LockedTestSet` and
  `SealedTestSet` (`splitters.py:101-136`, `scmle/governance/vault.py:56-118`),
  `make_split`/`seal` (`stages.py:551-652`), `RunContext` close
  (`scmle/core/context.py:497`).
- Reuse: `duplicates.keep_together`, `GroupSplit`, `TemporalSplit` embargo.
- Extend `tests/unit/test_vault.py:22-64`. Proposed location:
  `scmle/data/custody/`; the vault holds several named sets.

## Acceptance criteria

- **Inaccessibility test (fails the build).** After `seal`, (a) scan
  `gc.get_objects()` for any DataFrame, Series or ndarray whose row hashes
  intersect `C` or `F`; none may exist in the orchestrator process (mirrors
  `test_this_process_keeps_only_the_ids`, `test_vault.py:40`); (b) run the
  full exploration on data whose `C` and `F` rows carry unique canary values and
  assert no canary appears in any artifact, log or report; (c) an import-graph
  test asserts that modules under `ledger/`, `detectors/` and `admissibility/`
  never import `custody.vault`.
- Property tests (Hypothesis): for random group, duplicate and time
  structures, units are disjoint across sets, embargo gaps hold, and the manifest
  root changes iff a row or assignment changes.
- Same seed and data give an identical manifest on all three platforms.
- Second `open`, wrong token and mismatched `claims_hash` are refused and
  logged; a log with one edited or deleted line fails verification.
- Temporal case: no `X` or `C` row lies within `embargo` rows of `F`, and `C`
  blocks are interleaved with `X` blocks, not all after them.

## External referent

The embargo and purging rules follow the leakage definitions in López de
Prado, *Advances in Financial Machine Learning* (ch. 7), checked by an
independent reimplementation test on synthetic autocorrelated series. The
canary-value test is the referent for "inaccessible": it passes or fails
without the author's judgement. A reviewer outside the project should attempt
to read `C` from the exploration process (red team, F7).
