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

Closed on 2026-11-04 and removed: the leakage screen's own certificates now
pass for every leak type including `leak_noisy_copy`
(`screen_certificates.json`; `--check` lists no descriptive screen
mechanism), the screen is committed with tests and is in `--check`, and a
charter run uses it (`admissibility/qualify.py`) instead of `leakage.scan`.
The earlier skill scan `leakage.py` is used only as a helper library by the
screen; its own certificate (`f4-leakage-skill`, noisy copy 0.6646 against
the floor 0.9) stays descriptive and no run path calls `leakage.scan`.
A strong legitimate predictor flagged by a certified mechanism is overridden
only by a ledgered per-column attestation, tier `attested`
(`admissibility/attestations.py`).


| ID | Item | Lives in | State (verified) | Consequence now | Closing criterion |
|---|---|---|---|---|---|
| A3 | Titanic `body` not shown to be flagged by a certified path | `tests/integration/test_open_datasets.py:76-91` asserts `boat` and `body` appear in `risk_findings`; `README.md:157-159` says the screen flags both | The screen's record-purity test (`screen.py`, step 2b) is designed for this case, but no offline fixture of the real Titanic table exists and the test needs OpenML access, so it skips offline. Unverified whether `body` is flagged | A committed test with a frozen, hash-checked Titanic extract (`corpora.json` entry `titanic-boat-body`, status `candidate`) asserts that `body` is a suspect at level 0.05, or the README sentence is changed to say it is not |
| A4 | MMD shift test not certified | `benchmarks/qualification/f6_certificates.json` (`f6-shift-mmd`) | recall lower bound 0.9874 passes; false-alarm upper bound 0.0765 vs ceiling 0.075 (`f6_certify.py:84`) | MMD is descriptive; C2ST (upper 0.0355) is certified | False-alarm upper bound <= 0.075 with a larger null B chosen from the interval width needed (the observed rate is inside the ceiling; the interval is too wide), floor unchanged |
| A5 | Label-noise detectors not certified | `benchmarks/qualification/detector_certificates.json` (`f5-label-noise`) | recall lower bound 0.7497 (symmetric) and 0.6528 (class-conditional) vs 0.80 (`f5.py:55`) | Descriptive only | Lower bounds >= 0.80 for both types, or the claim is limited to the type that passes and the other is stated as uncovered; certificate is on more than the one corpus (breast cancer) |
| A6 | F5 certificates use one corpus | `benchmarks/qualification/README.md` "Limits" | One corpus (scikit-learn breast cancer); rows judged as independent, which understates uncertainty | Certificates are optimistic | Certificates re-issued on at least 3 corpora with different feature types; unit of replication is the table, not the row, for row detectors |

## B. Wiring and policy gaps in built features

Closed on 2026-11-04 and removed from group B: B1 (a charter flag
`rules.stability_filter` withdraws `fragile` survivors from the family tested
on `C`, as a pre-registered second stage with its own accounting; the claim
and its assumption are in `confirm.py` and `robustness/prefilter.py`, and it
acts only while the stability rule has a current certificate for the chartered
thresholds), B6 (charter format 3 `rules`: ledger rule, confirmation rule,
failure policy, stability thresholds; older charters keep their hash and load)
and B9 (stability and shift results, including per-column shift p-values,
are wealth-free diagnostic rows written before the survivors are frozen).

| ID | Item | Lives in | State (verified) | Closing criterion |
|---|---|---|---|---|
| B2 | Per-period drift and calibrated PSI not built | `scmle/data/robustness/report.py:123-126` (`periods` is `not_run`) | Report states "per-period drift is not implemented" | Per-period two-sample test with a permutation null and a certificate on planted drift; PSI cutoffs set from a null simulation, not a convention |
| B3 | Forward check always skipped | `scmle/data/robustness/qualify.py:198-204`; `forward.py` is implemented and certified but not called | `F` is the final test set, opened only by the final evaluation | Either the final evaluation calls `forward.check` after it opens `F` and writes the result to `shift_report.json`, or the report says plainly that the forward check is a post-run step |
| B4 | Thresholdout not implemented | `scmle/data/ledger/confirm.py:53,255-262` raises `ConfigError` | Only policy `unconfirmed` exists; `tasks.py:580` types the field as `Literal["unconfirmed"]` | Implement with recorded noise scale and query budget plus a test of its error bound, or delete the option from F3 and the docs |
| B5 | Resampling negative control not wired, and wiring it is not sound today | `scmle/data/ledger/controls.py:347` defines `resampling_control`; `governed.py` calls only `permuted_target_control` | Verified. Every hypothesis the ledger registers is a feature-target test, so the wired permuted-target control already covers it; `resampling_control` mode `columns` shuffles the features against a fixed target, which is the same null in distribution for these tests and adds nothing; mode `blocks` targets drift findings, and no drift hypothesis exists in the ledger (B2). Wiring it now would add a second gate with no finding family to guard | Open until the ledger registers a feature-only finding family (per-period drift from B2, or near-duplicate text from B12). Then: call `resampling_control` for that family, same 95th-percentile gate, fixture test where a planted feature-structure finding passes and a null does not |
| B7 | Multiverse covers three choices only | `scmle/data/robustness/stability.py:71-75` (`IMPLEMENTED`: imputation, outlier rule, inclusion); `applicable_choices` reports the rest as "not run" | Any other declared choice is listed as not run. Not extended on 2026-11-04: the estimator is Spearman's rho, so monotone transforms (log, standardise) are exact no-ops and would only inflate the specification count; choices that change a rank correlation (a winsorisation percentile, a missing-value indicator, binning) must be added to `IMPLEMENTED`, and `stability.py` and `multiverse.py` are hashed into the stability certificate (`code_digest`), so any edit makes `f6_certificates.json` stale and every verdict descriptive until it is re-issued. The re-issue was not run in that pass | Each additional choice is implemented with a test that the specification count rises, its options are in the certified planted worlds (fragile and stable by construction, floors written before the re-run), and the certificate is re-issued; or the choice is removed from the charter template |
| B8 | Stability not available for multiclass targets | `scmle/data/robustness/qualify.py:138-142` | Note: Kruskal-Wallis findings are not re-run. Not built on 2026-11-04: the survival rule reads the sign and size of one signed effect, which a Kruskal-Wallis statistic does not have, so a multiclass rule (for example epsilon squared with a bootstrap interval and a size band) would be a new pre-registered rule with its own planted multiclass worlds, floors and certificate, and an edit to `stability.py` that stales the existing certificate | An estimator for the multiclass finding (the ledger's Kruskal-Wallis, `bivariate.py`) with a certificate; otherwise the limit stays documented |
| B10 | `curation_report.json` stale after quarantine | written at `scmle/core/stages.py:468`; quarantine applied later at `:689-715` (`dataclasses.replace` on the feature plan) | `curation_report.json` and `dataset_card.json` still list quarantined fields as kept. Inferred from write order; not run | Rewritten (or amended) after quarantine; test compares the artifact's feature list with the final feature plan |
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
