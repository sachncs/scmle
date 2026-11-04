# R1: Instrument residuals

Canon: [`../../plan/00-canon.md`](../../plan/00-canon.md). Planned in
[Phase 7](../../plan/phase-7-instrument-residuals.md). Companion:
[`F7-verification-harness.md`](F7-verification-harness.md).

## Purpose

List every known open item of the implemented features F1-F6 (Phases 0-5),
each with the file it lives in, its current state as verified in the repo on
2026-11-04, and a closing criterion a machine can check. Nothing here is a new
feature. Features themselves are not re-described.

## Decision it serves

Which statements about the instrument are allowed today. An item stays here
until its closing criterion passes; while it is open, the behaviour named in
"consequence" is the stated limit (P6) and docs may not claim more.

## How to read the tables

- "Verified" means read in code or in a committed JSON file at the cited
  path. "Unverified" means not checked or not checkable offline.
- Interval numbers are Wilson 95 percent bounds from the committed
  certificate files. Floors are the pre-registered policies in the
  `*_certify.py` files (`f4_certify.py:59`, `f6_certify.py:84,88`, `f5.py:55`).

## A. Detectors that are measured but not certified (they stay descriptive)

| ID | Item | Lives in | State (verified) | Consequence now | Closing criterion |
|---|---|---|---|---|---|
| A1 | Noisy-copy leakage not certified | `benchmarks/qualification/f4_certificates.json` (`f4-leakage-skill`, `leak_noisy_copy`); `scmle/data/admissibility/leakage.py` | recall lower bound 0.6646 vs floor 0.9 (`f4_certify.py:59`). Other four leak types pass (>= 0.9812) | The leakage skill test is descriptive for noisy copies; it cannot gate | Lower bound of recall >= 0.9 on `leak_noisy_copy` with false-alarm upper bound <= 0.05 on a re-run of `python -m benchmarks.qualification --check`, with no change to the floor; or an amendment, dated before the re-run, that states why the floor is lowered |
| A2 | Replacement leakage screen has the same gap | `scmle/data/admissibility/screen_certificates.json` (`f4-leakage-screen`) | `leak_noisy_copy` recall lower bound 0.6541; copy, threshold, missing-for-class, rare-record pass | The screen can block only for the passing leak types | Same as A1 for the screen; plus a `leak_noisy_copy` recall curve versus noise level in `screen_measurements.json` so the detectable noise range is stated |
| A3 | Titanic `body` not shown to be flagged by a certified path | `tests/integration/test_open_datasets.py:76-91` asserts `boat` and `body` are flagged; `README.md:154-155` says the profiler flags both | The test used the removed fixed-cutoff profiler rule (`LEAK_ASSOCIATION`, `MISSINGNESS_LEAK` were deleted in the uncommitted `profiler.py` diff). The new screen's record-purity test (`screen.py` docstring, step 2b) is designed for this case, but no offline fixture of the real Titanic table exists and the test needs OpenML access. Unverified whether `body` is flagged after the change | README claim may be stale | A committed test with a frozen, hash-checked Titanic extract (see `corpora.json` entry `titanic-boat-body`, status `candidate`) asserts that `body` is in the screen's suspects at level 0.05, or the README sentence is changed to say it is not |
| A4 | MMD shift test not certified | `benchmarks/qualification/f6_certificates.json` (`f6-shift-mmd`) | recall lower bound 0.9874 passes; false-alarm upper bound 0.0765 vs ceiling 0.075 (`f6_certify.py:84`) | MMD is descriptive; C2ST (upper 0.0355) is certified | False-alarm upper bound <= 0.075 with a larger null B chosen from the interval width needed (the observed rate is inside the ceiling; the interval is too wide), floor unchanged |
| A5 | Label-noise detectors not certified | `benchmarks/qualification/detector_certificates.json` (`f5-label-noise`) | recall lower bound 0.7497 (symmetric) and 0.6528 (class-conditional) vs 0.80 (`f5.py:55`) | Descriptive only | Lower bounds >= 0.80 for both types, or the claim is limited to the type that passes and the other is stated as uncovered; certificate is on more than the one corpus (breast cancer) |
| A6 | F5 certificates use one corpus | `benchmarks/qualification/README.md` "Limits" | One corpus (scikit-learn breast cancer); rows judged as independent, which understates uncertainty | Certificates are optimistic | Certificates re-issued on at least 3 corpora with different feature types; unit of replication is the table, not the row, for row detectors |
| A7 | Non-charter leakage replacement not committed | working tree: `scmle/data/admissibility/screen.py`, `screen_certificates.json`, `benchmarks/qualification/screen_certify.py`, `screen_measurements.json` (untracked); `profiler.py`, `stages.py`, `contracts/data.py`, `recommendations.py` (modified) | Implemented in the working tree: a BY-controlled permutation screen replaces the fixed-threshold profiler checks; `git log` has no commit for it. Not in `setup.cfg` mutation scope. Not wired to `python -m benchmarks.qualification --check` (no `screen` in `__main__.py`). No `test_*screen*` file found under `tests/unit` or `tests/benchmarks`. Whether the existing suite passes with it is unverified | Not part of any released or committed behaviour | Committed with tests; certificate check added to `--check`; module added to `setup.cfg`; `README.md` and `docs/ARTIFACTS.md` updated; CI green |

## B. Wiring and policy gaps in built features

| ID | Item | Lives in | State (verified) | Closing criterion |
|---|---|---|---|---|
| B1 | Stability verdicts do not filter confirmation survivors | `scmle/data/robustness/report.py:99` (`confirmation_filter`: "not applied"); `scmle/core/stages.py:594-596` docstring | The ledger freezes survivors before stability runs; `fragile` is a warning | A charter field (see B6) states whether `fragile` survivors are dropped before `C` opens; if so, stability runs before the unlock and a test shows a fragile planted effect is not tested on `C`. Because stability reads `X` only, FDR on `C` is unaffected by the choice |
| B2 | Per-period drift and calibrated PSI not built | `scmle/data/robustness/report.py:123-126` (`periods` is `not_run`) | Report states "per-period drift is not implemented" | Per-period two-sample test with a permutation null and a certificate on planted drift; PSI cutoffs set from a null simulation, not a convention |
| B3 | Forward check always skipped | `scmle/data/robustness/qualify.py:198-204`; `forward.py` is implemented and certified but not called | `F` is the final test set, opened only by the final evaluation | Either the final evaluation calls `forward.check` after it opens `F` and writes the result to `shift_report.json`, or the report says plainly that the forward check is a post-run step |
| B4 | Thresholdout not implemented | `scmle/data/ledger/confirm.py:53,255-262` raises `ConfigError` | Only policy `unconfirmed` exists; `tasks.py:580` types the field as `Literal["unconfirmed"]` | Implement with recorded noise scale and query budget plus a test of its error bound, or delete the option from F3 and the docs |
| B5 | Resampling negative control not wired | `scmle/data/ledger/controls.py:347` defines `resampling_control`; only `permuted_target_control` is called (`governed.py:183`) | Feature-only findings have no negative control | Called in the charter run for feature-only findings; same 95th-percentile gate; fixture test where a planted feature-structure finding passes and a null does not |
| B6 | Charter lacks fields for ledger rule, confirmation rule, stability thresholds | `scmle/data/charter/contracts.py:280-312` (`AnalysisCharter`); the choices live in `scmle/contracts/tasks.py:574-580` and `stability.DEFAULT_RULE` | These are outside the charter hash and signature, so they can change after signing | Fields added to the charter; charter hash covers them; a test shows changing one changes `charter_hash`. This is a charter schema change: bump the schema version and keep old charters loadable |
| B7 | Multiverse covers three choices only | `scmle/data/robustness/stability.py:71-75` (`IMPLEMENTED`: imputation, outlier rule, inclusion); `applicable_choices` reports the rest as "not run" | Any other declared choice is listed as not run | Each additional choice either is implemented with a test that the specification count rises, or is removed from the charter template |
| B8 | Stability not available for multiclass targets | `scmle/data/robustness/qualify.py:138-142` | Note: Kruskal-Wallis findings are not re-run | An estimator for the multiclass finding (the ledger's Kruskal-Wallis, `bivariate.py`) with a certificate; otherwise the limit stays documented |
| B9 | F6 results are not ledger rows | `scmle/data/ledger/governed.py:176` records diagnostics from `found.checks` (`stages.py:565`) | Verified: admissibility and F5 detector `Check` objects are written as wealth-free `diagnostic` rows (`ledger.py:610-616,810`). Not verified for stability and shift: only `found.checks` is passed, and the robustness stage runs after the ledger run (`stages.py:577`) | Stability and shift `Check` objects appended as diagnostics before the ledger is closed, or the ledger documented as excluding them |
| B10 | `curation_report.json` stale after quarantine | written at `scmle/core/stages.py:468`; quarantine applied later at `:689-715` (`dataclasses.replace` on the feature plan) | `curation_report.json` and `dataset_card.json` still list quarantined fields as kept. Inferred from write order; not run | Rewritten (or amended) after quarantine; test compares the artifact's feature list with the final feature plan |
| B11 | Per-feature p-values not in ledger | `scmle/data/admissibility/screen.py` computes one permutation p-value per column but the ledger holds only registered hypotheses | Verified that `screen.py` has no ledger call; the exact content of `ScreenReport.to_dict` was not audited | Screen p-values stored in `profile.leakage` (exists) and a ledger diagnostic row per family; see A7 |
| B12 | MinHash text near-duplicates | `scmle/data/detectors/duplicates.py:13` states it is not implemented; text columns block exact matches only | Verified | Implemented, with a planted near-duplicate-text defect, a certificate and a cross-check against an exact Jaccard computation |
| B13 | Instance-dependent label noise not modelled | `scmle/data/detectors/labels.py`; certificates cover symmetric and class-conditional flips only (`detector_certificates.json`) | Verified from certificate defect types | A planted instance-dependent flip type in `inject.py` with a certificate, or an explicit statement in `ARTIFACTS.md` that this noise is not covered |
| B14 | Spec-curve plots and `eda_report.html` do not exist | no file or string `eda_report` anywhere in `scmle/`, `docs/` | Verified by search. The specification curve data is inside `stability_report.json` findings | Either a static HTML report written from `stability_report.json` and `shift_report.json` with a test that it renders without network, or the plan item is dropped |

## C. F2 custody: which stated gaps still hold

The original custody specification was removed from `product/features/` once
custody was built. Status of its stated gaps, verified by reading `scmle/data/custody/*`,
`scmle/governance/vault.py` and the tests:

| ID | F2 gap | Status |
|---|---|---|
| C1 | Custody coupled to the supervision flag | Closed for charter runs (`stages.py:850-890`, `apply_charter_custody`; CHANGELOG). Still coupled for non-charter runs (`stages.py:629-635`); that path has no `C` |
| C2 | `C` and `F` have custody | Closed for `C` and `final_test` (`pipeline.py:211-222`). For temporal data `F` is the final test set (`manifest.py:50`) |
| C3 | Caller-held copy of the data | **Open.** `docs/SECURITY.md:40-43` says a caller DataFrame or the file still holds every row. A closing option: the CLI reads the file inside the orchestrator and never returns the full frame to the caller |
| C4 | Canary-value test over all artifacts | **Open.** No canary test found (`grep canary tests` empty). `tests/leakage/test_custody_access.py` covers view reachability, pickling, log content, orchestrator keeps ids, second open (lines 88-151) |
| C5 | `gc.get_objects()` scan for rows of `C`/`F` | **Open.** Not found in tests |
| C6 | Import-graph test | **Partly closed.** `tests/leakage/test_ledger_independence.py` covers `scmle/data/ledger` only; nothing for `detectors/`, `admissibility/` or `robustness/` |
| C7 | Unlock bound to `claims_hash` and `ledger_hash` | **Open.** No `claims_hash` anywhere in `scmle/data/custody` or `scmle/governance/vault.py`; the vault enforces one open per set only |
| C8 | Log signed with HMAC | **Open for the custody log** (`sealed.py` hash-chains with `signing.canonical_payload`; no HMAC call found). The charter is optionally HMAC-signed |
| C9 | Merkle root over per-row content hashes | **Open.** Manifest v2 hashes ids (`manifest.py:120`); a changed row value does not change it. Stated in F2 as a requirement |
| C10 | Embargo from measured autocorrelation | **Open.** `carve.py` uses `embargo_rows` from the spec; no ACF code found |
| C11 | Identical manifest on three platforms | **Unverified.** CI runs the custody unit tests on Windows and macOS, but no test compares manifests across platforms |
| C12 | Custody modules in the mutation scope | **Open.** `setup.cfg` omits `scmle/data/custody/*`, `scmle/data/charter/*`, `scmle/governance/vault.py` (see F7 W6) |

Closing criterion for the group: C3-C12 each either has a failing-then-passing
test or is recorded in `docs/SECURITY.md` as an accepted limit with its
reason; and the F2 file is rewritten to describe only what is left.

## D. Evidence artifacts not yet produced

None of these is in the repo today (verified by listing).

| ID | Artifact | Produced by | Closing criterion |
|---|---|---|---|
| D1 | Signed sample run: a charter run directory with `analysis_charter.json`, ledger, custody log, claims, reports, signed by a key whose holder is not the author | A run on a public dataset, plus `scmle charter sign` | Directory committed under `examples/` (or released as an asset) with a `verify` command that exits 0 |
| D2 | Access-attempt log: a custody log containing a refused second open and a refused wrong-token attempt from a deliberate attack script | Red team (F7 W2) | Log verifies; attempts and refusals match the red-team report |
| D3 | Sample split manifest (v2) for a grouped and a temporal fixture | `split_manifest.json` from D1 | Hash-verifiable; included in docs |
| D4 | Oracle comparison table: for every p-value or decision producer, reference, tolerance, maximum observed difference | F7 W5 and the existing scipy fixtures | Generated by a script into `docs/`; CI fails if a row is missing |

## E. Citations from memory, not re-fetched

`scmle/data/ledger/online_fdr.py` already marks which e-LOND statements were
verified. The items below are cited from memory in code or docs and have not
been re-read against the source (verified from docstrings; unverified as to
correctness of the cited details):

| ID | Citation | Where | Closing criterion |
|---|---|---|---|
| E1 | e-LOND venue (AISTATS 2024) | `online_fdr.py:16` says it was not checked | Venue and theorem number read from the paper and recorded with date |
| E2 | Classifier two-sample test (C2ST); DeLong, DeLong and Clarke-Pearson AUC variance | `scmle/data/robustness/shift.py:1-12` | Source pages read; formulas compared with a reference implementation on a fixed input |
| E3 | Gretton et al. 2012 (kernel MMD, permutation test) | `shift.py` | Same |
| E4 | PCS (Yu and Kumbier) and specification curve (Simonsohn et al.) | `scmle/data/robustness/stability.py`, `multiverse.py` | Same |
| E5 | Holm 1979; Benjamini-Hochberg 1995; Benjamini-Yekutieli 2001 | `scmle/data/ledger/multiplicity.py:11-27` | Original statements of the step-up and step-down rules and the BY constant read; one worked example from each paper reproduced in a test |

Each closed row is marked "verified on <date> against <source>" in the
module docstring. No claim in a user-facing document may cite an item here as
verified until its row is closed.

## Failure modes of this plan

| Risk | Handling |
|---|---|
| Closing A1/A5 by lowering a floor | Not allowed after results are seen; only a dated amendment made before the re-run |
| Residual list grows stale | `python -m benchmarks.qualification --check` and a test that each row ID in this file still exists in its named file (proposed) |
| Fixing B6 breaks old charters | Schema version bump; old charters load with defaults and are tiered `unverified` for the new fields |

## Acceptance criteria

- [ ] Every ID above is either closed (criterion met, row deleted here) or
      accepted (limit recorded in `docs/SECURITY.md` or `docs/ARTIFACTS.md`
      with the date).
- [ ] `python -m benchmarks.qualification --check` exits 0 with the A-group
      certificates passing or the detector listed as descriptive in the docs.
- [ ] No file in `plan/` or `product/` names an implemented behaviour as
      proposed (checked by review, since F1-F6 and F2 text predates the code).

## External referent

The frozen scipy and R fixtures, the labelled corpora in
[`../../plan/validation-referents.md`](../../plan/validation-referents.md), the
primary sources in group E, and the red-team analyst for group C.
