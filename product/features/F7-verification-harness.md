# F7: Verification harness (remaining work)

Canon: [`../../plan/00-canon.md`](../../plan/00-canon.md). Residual items of
built features are in [`R1-instrument-residuals.md`](R1-instrument-residuals.md).
Planned in [Phase 6](../../plan/phase-6-external-validation-and-release.md).

This file describes only what is not built. The implemented part of F7
(defect injector, semi-synthetic simulator, frozen scipy oracles, certificate
store with code-hash staleness, `--smoke`, `--check`, `--write-f5`, F4/F5/F6
certification) is documented in `benchmarks/qualification/README.md` and
`CHANGELOG.md` and is not repeated here. Status of everything below:
**proposed**.

## Purpose

Supply the evidence the qualification instrument cannot produce about itself:
whether its verdicts predict real outcomes, whether an outsider can defeat
it, whether someone else reproduces its numbers, how it compares with the
strongest alternatives, and whether the code that computes its statistics is
protected against silent faults. Nothing here adds user-facing analysis.

## Decision it serves

- Whether the project may state "block means worse out of time", and at what
  strength (backtest).
- Whether the `confirmed` tier may be described as resistant to a motivated
  analyst (red team).
- Whether a release may be signed off (Phase 6 go/no-go).
- Whether the claim "close to the state of the art" in `benchmarks/sota` is
  supported or not (TabArena campaign).

## State at the start (verified against the repo, 2026-11-04)

| Item | State | Evidence |
|---|---|---|
| Corpora | All entries are `candidate`; no source URL, licence or hash is filled in | `benchmarks/qualification/corpora.json` |
| R oracles | Not generated; `onlineFDR` and `knockoff` listed as pending | `benchmarks/qualification/oracles.py:42-50`; frozen scipy fixtures only in `tests/fixtures/oracles/` (12 files, none from R) |
| SOTA campaign | Driver, runners and statistics exist; protocol tagged `sota-protocol-v1`; no result file is committed under `benchmarks/sota/` | `git log -- benchmarks/sota`, directory listing |
| Mutation scope | `setup.cfg` `source_paths` covers harness, ledger, admissibility, detectors, robustness modules; it does **not** list `scmle/data/custody/*`, `scmle/data/charter/*`, `scmle/governance/vault.py`, `scmle/data/ledger/__init__.py`, or the uncommitted `scmle/data/admissibility/screen.py` | `setup.cfg:1-58` |
| Mutation score | Gate script default minimum 70; nightly runs it at 70. No survivor counts are recorded in the repo | `scripts/check_mutation_score.sh:6`, `.github/workflows/full-suite.yml:41` |
| Windows CI runtime | `cross-platform` has `timeout-minutes: 20`. Last completed master run `41748161942`: Windows 3.11 took 15 min 31 s, Windows 3.14 16 min 36 s; macOS 3.11 11 min, 3.14 12 min 26 s. Earlier run `41697834426`: Windows 3.11 15 min 38 s | `.github/workflows/ci.yml:67-70`; `gh run view` job timestamps |
| Independent replication | `benchmarks/sota/PROTOCOL.md` exists; no replication record | no such file |

## Work items

### W1. Predictive-validity backtest

Question: do datasets the instrument says `block` show a larger out-of-time
generalization gap than those it says `pass`? Uses the careful design below.

- **Gap definition is fixed before any verdict is computed**: validation score
  on the earliest period minus score on the latest period under a frozen
  reference model and metric (proposed: a GBDT with default settings,
  task-appropriate metric, normalized as in the SOTA protocol).
- **Circularity.** The verdict may not use the later period or any model score
  used in the gap. The instrument sees only the earlier period; the gap uses a
  later period it never read. Label-noise or leakage detectors that train a
  model use the same earlier period only. A charter run must set `F` (or a
  sealed later block) outside the instrument's reach, and the harness must
  assert that the artifacts of the run contain no value from it.
- **Selection bias.** Corpus membership is fixed by a written inclusion rule
  and a hash list before verdicts are computed; datasets are not dropped for
  failing, and failures are counted. Report how many datasets lacked a usable
  time axis; the claim is limited to the datasets that had one.
- **Confounding.** Datasets differ in size and difficulty, so compare within
  strata (size tercile x task type) and report an adjusted effect (stratified
  Mann-Whitney or a regression with these covariates), not a raw difference.
- **Multiplicity.** One pre-registered primary contrast (block vs pass on the
  gap). Per-check contrasts are secondary, BH-adjusted, labelled exploratory.
- **Weak power is a result.** If the interval is wide, the report says the
  claim is unsupported rather than softening the threshold.

Open design risk (unverified): TabArena tasks are mostly not temporal. A
backtest needs datasets with a real time axis; the number that qualify is
unknown until the inclusion rule is applied. If fewer than 15 qualify, the
outcome is "unsupported" and Phase 6 gate 4 stays open.

### W2. Red-team protocol

- **Who.** An analyst who did not write F1-F6 and has not read the detector
  code; they see the charter spec, the CLI and the reports.
- **Task.** Given 3 datasets and a rules sheet, produce a claim that is
  false (a planted-null feature) and pushed to `confirmed`, or a planted
  leak that is admitted, using only legal inputs (charter, attestations,
  data choices, number of runs). Reading the sealed sets or editing code is out
  of scope and is logged as a custody finding instead.
- **Attack budget** (hours) and the number of full runs per attempt are set in
  advance in the rules sheet.
- **Scoring.** Each attempt is a trial. Success = a false claim reaches
  `confirmed` or a planted leak is admitted. Report successes out of attempts
  with an exact binomial interval. Pre-registered pass: zero successes and an
  upper bound consistent with `alpha_confirm` when attempts allow; with few
  attempts the report states the bound it can support. Every successful
  attack becomes a regression test and a ledger rule.
- Known attack surface to put on the rules sheet (each is a stated residual,
  see R1): re-running with different seeds or charters (the ledger spans one
  run, not an analyst's career); the caller-held copy of the data; the
  non-charter path, which has no custody.

### W3. Independent replication

A third party reruns the qualification campaign (`--smoke --full`, `--check`,
the backtest, the sota Lite tier) from a tag on a fresh machine, using only
the committed protocol. Their summary statistics must match ours within the
pre-stated tolerance, or the gap is explained in a dated amendment. Uses the
discipline of `benchmarks/sota/PROTOCOL.md` (tag before results, amendments
dated before the hold-out opens, one hold-out opening).

### W4. TabArena campaign against AutoGluon, FLAML and tuned GBDT

Run the already-written campaign, not a new one: `benchmarks/sota/campaign.py`,
`scmle_runner.py`, `baselines.py`, `analyze.py`, pinned tasks
`tasks_tabarena_v0pt1.csv` (51 tasks, development 31 / hold-out 20 per
`split.json`). Claim, thresholds, tiers (Lite, Full) and budgets (60 s, 600 s,
3600 s) are fixed in PROTOCOL.md sections 1, 4, 5 and are not restated here.
Remaining: execute the calibration step (AutoGluon `best_quality` vs
TabArena's published split results), the development run, then open the
hold-out once. This is a model-accuracy claim about the SCMLE modelling
stages, **not** about the qualification instrument; it is in F7 only because
it shares the replication machinery. Cost estimate in PROTOCOL.md section 4:
about 9 days single-machine for one system on development at all budgets and
6 splits. Losses are published with wins (PROTOCOL.md section 6, rule 6).

### W5. R oracle fixtures

Generate offline, by a reviewer with R, and freeze in
`tests/fixtures/oracles/` with the generator script, R and package versions
and a content hash (same format as the existing scipy fixtures,
`benchmarks/qualification/oracles.py`): `onlineFDR` (LORD++, SAFFRON, LOND,
ADDIS decisions and levels on fixed p-value sequences), `knockoff` (selection
sets on fixed matrices), `dagitty` (implied conditional independencies for the
DAG tests in `scmle/data/admissibility/dag.py`). Notes recorded in the
source: `onlineFDR` 2.20.0 has no e-LOND function (`online_fdr.py:73`), so
e-LOND keeps only its closed-form and simulation checks. Agreement target:
decisions exact, levels to 1e-9.

### W6. Repo-wide mutation floor

Run mutmut over the full configured scope and over the modules missing from
it, then record final survivor counts per module in a committed table
(`benchmarks/qualification/mutation_survivors.json`, proposed) with every
survivor either killed by a new test or justified in one line.

- Add to `setup.cfg` `source_paths` and `pytest_add_cli_args_test_selection`:
  `scmle/data/custody/*.py`, `scmle/data/charter/*.py` (at least
  `contracts.py`, `feasibility.py`, `sealing.py`), `scmle/governance/vault.py`,
  and `scmle/data/admissibility/screen.py` once it is committed, with their
  tests (`tests/unit/test_custody_*.py`, `test_charter_*.py`, `test_vault.py`,
  `tests/leakage/test_custody_access.py`, `tests/property/test_custody_properties.py`).
- Required counts to report: ledger, confirm, custody, admissibility,
  detectors, robustness. Floor (proposed): 100 percent of mutants that flip a
  comparison, constant or sign in `ledger/` and in p-value code killed or
  justified; score at least 80 on the whole scope (up from 70).
- Not yet measured: the current score. The first step is to measure it; the
  floor is set after, and may be lowered only by a dated change here.

### W7. Windows and macOS CI runtime headroom

Windows 3.11 and 3.14 take 15-17 minutes of a 20-minute limit; each new
F-feature test file adds to it. Before adding the W6 tests, either (a) split
the cross-platform job into two (unit vs property/leakage/end-to-end) or (b)
move slow property tests to the nightly job, keeping a Windows custody and
vault test in the bounded set (the vault uses a different channel on Windows,
see `CHANGELOG.md` 0.1.0b1). Do not raise `timeout-minutes` as the fix: the
limit exists to catch hangs. Acceptance: Windows jobs end under 14 minutes on
three consecutive master runs.

## Guarantees and assumptions

- Certificates describe performance on injected defect distributions only;
  recall on planted defects is optimistic and reported per type.
- Differential tests show agreement with a reference, not truth.
- The backtest gives association across a finite corpus, not causation.
- A red team with few attempts supports only a loose upper bound; the report
  states it.
- Replication by one party shows reproducibility, not correctness.

## Failure modes

| Failure | Handling |
|---|---|
| Corpus file hash mismatch | Corpus run refused; no partial results. |
| Fewer than 15 backtest datasets with a time axis | Report "unsupported"; gate 4 stays open. |
| R unavailable to any reviewer | W5 stays open; e-LOND and LORD++/SAFFRON keep their closed-form and simulation checks only; this is stated in the release notes. |
| Mutation run exceeds nightly time (`full-suite.yml` limit 90 min) | Split by module group across matrix jobs; do not drop modules. |
| Independent party unavailable | Release stays open; the author does not substitute. |
| TabArena hold-out loses to a reference | Published as a loss; claim marked not met. |

## Repo integration (verified paths)

- Mutation: `setup.cfg:1-58`; `scripts/check_mutation_score.sh`; nightly
  `.github/workflows/full-suite.yml:19,38-41`.
- Certificates: `benchmarks/qualification/{detector,f4,f6}_certificates.json`
  (checked by `python -m benchmarks.qualification --check`,
  `benchmarks/qualification/__main__.py:34`).
- Oracle machinery: `benchmarks/qualification/oracles.py`,
  `make_oracles.py`, `make_f5_oracles.py`.
- SOTA: `benchmarks/sota/PROTOCOL.md`, `campaign.py`, `analyze.py`.
- CI: `.github/workflows/ci.yml:67-93` (cross-platform).
- Proposed new: `benchmarks/qualification/backtest.py`,
  `benchmarks/qualification/redteam/`, `benchmarks/qualification/PROTOCOL.md`,
  `benchmarks/qualification/mutation_survivors.json`.

## Acceptance criteria (machine-checkable where stated)

- [ ] `python -m benchmarks.qualification.backtest --protocol qualification-protocol-v1`
      exits 0 and writes a report with effect, interval, strata, n and the
      number of datasets excluded for lacking a time axis.
- [ ] `benchmarks/qualification/redteam/report.json` lists attempts, successes
      and the exact binomial upper bound; every success maps to a test id.
- [ ] A replication record exists with tag, machine description and
      per-statistic differences all within the protocol tolerance.
- [ ] `benchmarks/sota` result files for the hold-out exist with SHA-256
      checksums and the claim is reported met or not met.
- [ ] R fixtures exist for `onlineFDR`, `knockoff`, `dagitty`; differential
      tests pass (levels 1e-9, decisions exact).
- [ ] `./scripts/check_mutation_score.sh 80` exits 0 with the custody, charter
      and vault modules in scope; `mutation_survivors.json` has no unjustified
      survivor in `ledger/`.
- [ ] Windows jobs of three consecutive master runs finish in under 14 minutes.

## External referent

The red-team analyst, the replicating party, later-period outcomes in the
backtest, R reference outputs, TabArena's published results and the
reference systems. See
[`../../plan/validation-referents.md`](../../plan/validation-referents.md).
