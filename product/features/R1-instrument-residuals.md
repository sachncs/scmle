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
A3 closed 2026-11-04: `tests/unit/test_titanic_extract.py` runs the shipped, certified screen on a frozen, hash-checked extract of OpenML Titanic (`tests/fixtures/titanic_extract.csv`, corpus `titanic-boat-body`) and asserts that `boat` and `body` are blocked, `body` by record purity only. The corpus stays a `candidate` (licence line is OpenML metadata, not a legal check).
A4 closed 2026-11-04: the MMD test was re-measured on 5000 null worlds (dated amendment in `f6_certify.py`, floors unchanged): 259 false alarms, upper bound 0.0583 against 0.075, recall lower bound 0.9874; it is certified.
A strong legitimate predictor flagged by a certified mechanism is overridden
only by a ledgered per-column attestation, tier `attested`
(`admissibility/attestations.py`).


| ID | Item | Lives in | State (verified) | Consequence now | Closing criterion |
|---|---|---|---|---|---|
| A5 | Label-noise detectors not certified | `benchmarks/qualification/detector_certificates.json` (`f5-label-noise`); `f5_corpora_measurements.json` | breast cancer: recall lower bound 0.7497 (symmetric) and 0.6528 (class-conditional) vs 0.80 (`f5.py:55`). On three other tables (2026-11-04): digits passes both (0.8488, 0.8238), iris (0.6666, 0.6146) and wine (0.5078, 0.3462) do not | Descriptive only; the weakness grows as the table gets smaller | Lower bounds >= 0.80 for both types on the corpora, or the claim is limited to the type and table size that pass and the rest is stated as uncovered |
| A6 | F5 certificates use one corpus | `benchmarks/qualification/README.md` "Limits"; `f5_corpora.py` | The shipped certificates are on breast cancer only. `f5_corpora` repeats the plants on wine, iris and digits with the same floors: duplicates and outlier triage pass on all three, label noise on digits only, and the **sentinel detector flags the clean rows of every table** (29/150, 34/178, 1797/1797 against a 5 percent ceiling). The table-level count (k of 3 tables) has a Wilson lower bound of at most 0.44, so it is evidence of robustness, not a certificate | The sentinel certificate (gating on breast cancer) does not carry over to tables with natural repeated or zero values; label-noise and sentinel claims are limited to the corpus they were certified on | Sentinel detector fixed or its gating limited, with its certificate re-issued so the clean-table false-alarm bound holds on all four tables; certificates with the table as unit of replication on enough tables (not three) to have an informative bound |

## B. Wiring and policy gaps in built features

Closed on 2026-11-04 and removed from group B: B1 (a charter flag
`rules.stability_filter` withdraws `fragile` survivors from the family tested
on `C`, as a pre-registered second stage with its own accounting; the claim
and its assumption are in `confirm.py` and `robustness/prefilter.py`, and it
acts only while the stability rule has a current certificate for the chartered
thresholds), B6 (charter format 3 `rules`: ledger rule, confirmation rule,
failure policy, stability thresholds; older charters keep their hash and load)
B10 (closed 2026-11-06: `curation_report.json` and `dataset_card.json` are rewritten after quarantine; `test_the_curation_report_and_card_list_the_final_feature_plan`) and B9 (stability and shift results, including per-column shift p-values,
are wealth-free diagnostic rows written before the survivors are frozen).

| ID | Item | Lives in | State (verified) | Closing criterion |
|---|---|---|---|---|
| B2 | Per-period drift and calibrated PSI not built | `scmle/data/robustness/report.py:123-126` (`periods` is `not_run`) | Report states "per-period drift is not implemented" | Per-period two-sample test with a permutation null and a certificate on planted drift; PSI cutoffs set from a null simulation, not a convention |
| B3 | Forward check always skipped | `scmle/data/robustness/qualify.py:198-204`; `forward.py` is implemented and certified but not called | `F` is the final test set, opened only by the final evaluation | Either the final evaluation calls `forward.check` after it opens `F` and writes the result to `shift_report.json`, or the report says plainly that the forward check is a post-run step |
| B4 | Thresholdout not implemented | `scmle/data/ledger/confirm.py:53,255-262` raises `ConfigError` | Only policy `unconfirmed` exists; `tasks.py:580` types the field as `Literal["unconfirmed"]` | Implement with recorded noise scale and query budget plus a test of its error bound, or delete the option from F3 and the docs |
| B5 | Resampling negative control not wired, and wiring it is not sound today | `scmle/data/ledger/controls.py:347` defines `resampling_control`; `governed.py` calls only `permuted_target_control` | Verified. Every hypothesis the ledger registers is a feature-target test, so the wired permuted-target control already covers it; `resampling_control` mode `columns` shuffles the features against a fixed target, which is the same null in distribution for these tests and adds nothing; mode `blocks` targets drift findings, and no drift hypothesis exists in the ledger (B2). Wiring it now would add a second gate with no finding family to guard | Open until the ledger registers a feature-only finding family (per-period drift from B2, or near-duplicate text from B12). Then: call `resampling_control` for that family, same 95th-percentile gate, fixture test where a planted feature-structure finding passes and a null does not |
| B7 | Multiverse covers three choices only | `scmle/data/robustness/stability.py:71-75` (`IMPLEMENTED`: imputation, outlier rule, inclusion); `applicable_choices` reports the rest as "not run" | Any other declared choice is listed as not run. Not extended on 2026-11-04: the estimator is Spearman's rho, so monotone transforms (log, standardise) are exact no-ops and would only inflate the specification count; choices that change a rank correlation (a winsorisation percentile, a missing-value indicator, binning) must be added to `IMPLEMENTED`, and `stability.py` and `multiverse.py` are hashed into the stability certificate (`code_digest`), so any edit makes `f6_certificates.json` stale and every verdict descriptive until it is re-issued. The re-issue was not run in that pass | Each additional choice is implemented with a test that the specification count rises, its options are in the certified planted worlds (fragile and stable by construction, floors written before the re-run), and the certificate is re-issued; or the choice is removed from the charter template |
| B8 | Stability not available for multiclass targets | `scmle/data/robustness/qualify.py:138-142` | Note: Kruskal-Wallis findings are not re-run. Not built on 2026-11-04: the survival rule reads the sign and size of one signed effect, which a Kruskal-Wallis statistic does not have, so a multiclass rule (for example epsilon squared with a bootstrap interval and a size band) would be a new pre-registered rule with its own planted multiclass worlds, floors and certificate, and an edit to `stability.py` that stales the existing certificate | An estimator for the multiclass finding (the ledger's Kruskal-Wallis, `bivariate.py`) with a certificate; otherwise the limit stays documented |
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

Closed or accepted on 2026-11-04 and removed: C3 (accepted limit, recorded in `docs/SECURITY.md`), C4 and C5 (canary and `gc` tests), C6 (import-graph test over `detectors/`, `admissibility/`, `robustness/`), C8 (HMAC on every custody log entry when a key is available), C9 (Merkle root over row content in the manifest), C10 (autocorrelation-based embargo), C7 (`confirm.py` binds the release to the family hash and the ledger head before `C` opens; a set bound to another ledger state is refused) and C12 (custody, charter, vault, attestation and Merkle modules are in the mutation scope; survivors in `tests/unit/MUTANTS_custody.md`). The statements each test establishes, and their limits, are in `docs/SECURITY.md`.

C11 closed 2026-11-06: the `cross-platform` (Windows and macOS, Python 3.11 and 3.14) and `minimal-dependencies` jobs, which run `tests/unit/test_custody_golden.py`, passed on master run 42227276889.

## D. Evidence artifacts: what the generated ones are not

Produced on 2026-11-04 by `scripts/make_phase7_evidence.py` and `python -m benchmarks.qualification.oracle_table` from the real code (nothing hand-written): `plan/evidence/phase-7/` (sample run, access-attempt log, sample manifests) and `docs/ORACLES.md`. Closed and removed: D3, and the generation part of D1, D2 and D4. What is left:

| ID | Residual | Closing criterion |
|---|---|---|
| D1 | The sample run is signed with a published demonstration key, so it is tamper evidence only; the roadmap asked for a key whose holder is not the author | A run signed by an independent holder. (`scmle claims verify plan/evidence/phase-7/signed-sample-run --key-file KEY` exited 0 with 7 of 9 steps passing and 2 skipped on 2026-11-04, where the key file holds the published demonstration key) |
| D2 | The attacker is the generator script, not the red team (F7 W2) | Log and report from the red-team analyst; attempts and refusals match |
| D4 | Six of twelve oracles (`chi2_contingency`, `fisher_exact`, `kendalltau`, `ks_2samp`, `mannwhitneyu`, `pearsonr`) have no scmle producer compared in the table; the `onlineFDR` and `knockoff` references are pending R | Each gets a producer in `oracle_table.WIRED` or is removed with a dated reason; the R references are frozen |

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
