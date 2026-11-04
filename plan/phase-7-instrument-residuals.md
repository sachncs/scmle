# Phase 7: Instrument residuals

Canon: [`00-canon.md`](00-canon.md). Items, files, states and per-item closing
criteria: [`../product/features/R1-instrument-residuals.md`](../product/features/R1-instrument-residuals.md)
(IDs A1-E5 below refer to it). Status: **in progress**. Done 2026-11-04 and removed from R1: A1, A2, A7, B11 (leakage screen committed, certified and wired into the charter path; per-column attestations override its block) and B1, B6, B9 (charter format 3 `rules`, the pre-registered stability-filter second stage, stability and shift diagnostics in the ledger, and the `scmle claims` command). B5, B7 and B8 were examined and stay open with their reasons in R1. Done 2026-11-04 and removed from R1: C3 (accepted limit), C4, C5, C6, C8, C9, C10, C12 and D3 (canary, `gc` and import-graph tests, HMAC on the custody log, Merkle roots in the manifest, the autocorrelation embargo, mutation scope, sample manifests); C7 is wired (`confirm.py` commits the family hash and the ledger head before `C` opens); C11 is not observed on other platforms, and D1, D2, D4 are generated with the limits stated in R1. The mutation run for these modules and the already-scoped ones is recorded in `tests/unit/MUTANTS_custody.md` and `tests/unit/MUTANTS_scoped.md`. Done 2026-11-04: A4 (MMD certified on 5000 nulls) and a measurement of A5/A6 on three more tables (`f5_corpora.py`; the sentinel detector fails there, see R1). Not started in this pass: A3 and the A5/A6 fixes and the E-group primary-source re-reads. This phase is not yet in
the table of `plan/README.md` or `00-canon.md`; adding a row there is a
separate edit.

## Goal

Close or formally accept every known open item of the implemented features so
that Phase 6 sign-offs rest on stated, current limits. Phase 7 precedes the
external gates in Phase 6 that depend on it (entry criterion 1 of Phase 6);
work packages 7.5 and 7.6 can run in parallel with Phase 6.

## Entry criteria

- Phases 0-5 merged (verified in `git log`; last commit
  `df5eafb` at the time of writing).
- Working tree decision on the uncommitted leakage screen (R1 A7): either it
  is committed through 7.1, or discarded. Until then the claim "the
  non-charter profiler checks have been replaced" is **pending**, not true of
  any commit.

## Work packages

| WP | Items | Scope | Depends on | Exit gate |
|---|---|---|---|---|
| 7.1 | A3 (A7, A2, B11 done) | Commit the leakage screen with tests; add its certificates to `python -m benchmarks.qualification --check`; add `screen.py` and its tests to `setup.cfg`; freeze a Titanic extract (hash, licence) and test `body`; update `README.md:154-155` and `docs/ARTIFACTS.md` to the true behaviour | none | `uv run pytest -q tests/unit tests/property tests/leakage` passes; `uv run python -m benchmarks.qualification --check` exits 0 and lists `f4-leakage-screen` |
| 7.2 | A5, A6 (A4 done, A1 superseded) | Certification gaps. Choose per item: improve the detector, increase B for the interval, or document as descriptive. No floor lowered after results; amendments are dated and precede the re-run. Re-issue certificates on at least 3 corpora for F5 | 7.1 | `--check` exits 0; each of the five items is either passing (lower bound at floor) or listed as descriptive in `docs/ARTIFACTS.md` with its numbers |
| 7.3 | B7, B8 (B1, B6 done) | Extend or prune the multiverse choices (needs a re-issued stability certificate); multiclass stability estimator or documented limit | none | `pytest tests/unit/test_robustness_*.py` pass with new tests; the stability certificate is re-issued for the edited `stability.py` |
| 7.4 | B2, B3, B4, B5, B10, B12, B13, B14 (B9 done) | Pipeline gaps: per-period drift and calibrated PSI with a certificate; forward check after final evaluation; Thresholdout implemented or removed; wire `resampling_control` once a feature-only finding family is ledgered; rewrite `curation_report.json` after quarantine; MinHash; instance-dependent noise type; HTML report or drop | 7.3 for B1-dependent parts | For each item the closing criterion in R1 passes; the shift report no longer has `periods.status == "not_run"` unless the item was dropped by a dated decision |
| 7.5 | C11 (C3-C10, C12 done) | Observe the pinned manifest hashes pass on the Windows and lowest-dependency CI jobs | none | `pytest tests/leakage tests/unit/test_custody_*.py tests/unit/test_vault.py tests/property/test_custody_properties.py` pass; a changed ledger is refused at the unlock in a charter run |
| 7.6 | D1, D2, D4 residuals, E1-E5 | Sample run signed by an independent holder; red-team log; a scmle producer for each of the six oracles that has none; re-read each cited paper and record the date and result in the module docstring | D2 needs the Phase 6 red team | `grep -rn "from memory" scmle` returns only items listed as open in R1 group E |

## Parallel lanes

7.1 and 7.3 and 7.5 are independent. 7.2 follows 7.1. 7.4 follows 7.3 for its
charter-dependent parts. 7.6 citations can start now.

## Exit gate

```bash
uv run pytest -q tests/unit tests/property tests/leakage
uv run python -m benchmarks.qualification --smoke
uv run python -m benchmarks.qualification --check
uv run ruff check . && uv run ruff format --check . && uv run mypy scmle benchmarks
./scripts/check_mutation_score.sh 80    # with custody, charter, vault, screen in scope
```

Plus: every R1 row is closed (deleted from R1) or accepted (limit written in
`docs/SECURITY.md` or `docs/ARTIFACTS.md` with a date).

## Evidence required

Certificate JSON diffs with the Wilson bounds before and after; the amendment
text for any policy change, dated before the re-run; test names for each
custody item; the frozen Titanic extract manifest entry; the sample run
directory; citation-verification notes.

## Risks

| Risk | Mitigation |
|---|---|
| Items A1, A5 cannot reach the floor with the current methods | Accept as descriptive; do not lower the floor after seeing the interval |
| Custody items C7-C9 change the vault protocol and break the Windows channel | Run the Windows and macOS custody tests in CI for the PR; keep the bounded set under the time limit (F7 W7) |
| Mutation floor 80 not reachable on custody modules | Report survivors per module (F7 W6) and justify each; do not drop modules from scope |
| Titanic extract licence unclear | Mark `corpora.json` entry `rejected` and test with a synthetic table that has the same recording structure, stated as a stand-in |
| Scope grows into new features | New features go to a new plan file; this phase only closes R1 rows |

## Out of scope

New detectors beyond those named in R1, new modalities, anything in Phase 6.
