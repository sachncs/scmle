# F7: Verification harness

Canon: [`../../plan/00-canon.md`](../../plan/00-canon.md). Phases: skeleton in
Phase 1, completed in Phase 6.

## Purpose

Measure the qualification instrument itself. F7 supplies the known-answer
inputs, reference outputs and attackers that decide whether a detector may gate
(P1), and issues the certificates the other features consume. It adds no
user-facing analysis. Nothing in F1-F6 may gate a run until F7 holds a current
certificate for it.

## Decision it serves

- Whether a check is allowed to gate (P1 clause b and c).
- Whether a release may ship: CI fails when a certificate is missing, stale,
  or outside its stated tolerance.
- Whether the project may claim "block means worse" (the predictive-validity
  claim), and only at the strength the backtest supports.

## Behaviour

1. **Defect injection.** A seeded library plants known defects in a copy of a
   real table and returns the table plus a truth mask. Defect types (proposed):
   label flips at rate r, leaked column (copy, noisy copy, missing-only-for-one-
   class), duplicates and near-duplicates, sentinel values, impossible values,
   MNAR deletion with a known delta, drift between sets, planted
   target-feature effects of known size.
2. **Semi-synthetic simulator.** Real feature matrix X (so correlation, skew
   and cardinality are realistic), synthetic y drawn from a known function with
   planted effects on a known subset of columns and none on the rest. Runs the
   whole pipeline (charter, custody, ledger, confirmation) B times and reports
   empirical FDR, power, and family-wise error with Monte Carlo intervals.
3. **Differential tests.** Each p-value or statistic is compared with a
   reference implementation on frozen inputs.
4. **Null checks.** Under a true null, p-values are uniform.
5. **Mutation gate.** Seeded faults in the statistic code must be caught.
6. **Certificate store.** One JSON per detector and version:
   `{detector, code_hash, corpus_hashes, defect_type, size, recall, intervals, B,
   seed, created}`. Gating code looks up the certificate by `code_hash`; a code
   change without re-certification downgrades the check to descriptive.
7. **Predictive-validity backtest** and **red-team** (below).

## Method

- **Operating characteristics.** Size and recall are binomial proportions over
  B independent injections, reported with Wilson intervals. A detector is
  certified for a defect type only if the lower Wilson bound of recall meets
  the pre-registered floor and the upper bound of the false-alarm rate on a
  clean copy stays under the pre-registered ceiling. B is chosen from the
  interval width required, not fixed by habit.
- **Pipeline FDR.** Empirical FDP is averaged over simulated worlds. The claim
  "FDR <= alpha" is accepted only if the upper confidence limit of mean FDP is
  <= alpha plus the stated Monte Carlo slack (proposed default 0.01).
- **Differential testing.** Oracles (frozen as fixtures, with generation script
  and tool versions recorded): scipy and statsmodels for chi-square, Fisher,
  KS, Mann-Kendall-style trend and ADF/KPSS; R `onlineFDR` for LORD++,
  SAFFRON, ADDIS decisions; R `knockoff` for selection sets; `cleanlab` for
  issue ranks; the existing SOTA statistics script as a bootstrap oracle.
  Tolerance (proposed): p-values and statistics agree to relative 1e-8 where the
  algorithm is deterministic; permutation and bootstrap outputs agree in
  distribution (two-sample KS p > 0.01 over 20 seeds) and in decisions on
  fixed seeds. Decisions of online-FDR rules must match exactly.
- **Closed-form nulls.** For each test with a known null law, draw 2,000
  null datasets and apply a one-sample KS test of p-values against U(0,1) at
  level 0.001 (so a correct test fails roughly once per thousand runs), plus a
  check that P(p <= 0.05) lies in the binomial band. Seeds are fixed so CI is
  deterministic.
- **Mutation testing.** Extend the existing mutmut gate to the new packages.
  Required: 100 percent of mutants in `ledger/` and in p-value code that flip
  a comparison, constant or sign are killed or explicitly justified; the
  repo-wide floor stays at its current value.
- **Predictive-validity backtest.** See next section.

### Backtest: avoiding circularity and selection bias

Question: do datasets the instrument says `block` show a larger out-of-time
generalization gap than those it says `pass`?

- **Gap definition is fixed before any verdict is computed**: validation score
  on the earliest period minus score on the latest period under a frozen
  reference model and metric (proposed: a GBDT with default settings,
  task-appropriate metric, normalized as in the SOTA protocol).
- **Circularity.** The verdict may not use the later period or any model score
  used in the gap. The instrument sees only the earlier period; the gap uses a
  later period it never read. Label-noise or leakage detectors that train a
  model use the same earlier period only.
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

### Red-team protocol

- **Who.** An analyst who did not write F1-F6 and has not read the detector
  code; they see the charter spec, the CLI and the reports.
- **Task.** Given 3 datasets and a rules sheet, produce a claim that is
  false (a planted-null feature) and pushed to `confirmed`, or a planted
  leak that is admitted, using only legal inputs (charter, attestations,
  data choices, number of runs). Reading the sealed sets or editing code is out
  of scope and is logged as a custody finding instead.
- **Scoring.** Each attempt is a trial. Success = a false claim reaches
  `confirmed` or a planted leak is admitted. Report successes out of attempts
  with an exact binomial interval. Pre-registered pass: zero successes and an
  upper bound consistent with `alpha_confirm` when attempts allow; with few
  attempts the report states the bound it can support. Every successful
  attack becomes a regression test and a ledger rule.

## Guarantees and assumptions

- Certificates describe performance on the injected defect distributions only.
  Real defects differ; recall on planted errors is an optimistic upper
  estimate and is reported per type, never as one number.
- Differential tests show agreement with the reference, not truth. The
  reference may share a modelling assumption with ours (stated per fixture).
- Backtest gives association across a finite corpus, not causation.

## Failure modes

| Failure | Handling |
|---|---|
| Certificate stale (code hash changed) | Check demoted to descriptive; CI fails the release gate. |
| Corpus file hash mismatch | Corpus run refused; no partial results. |
| Reference tool version drift | Fixtures are frozen; regeneration is a reviewed change. |
| Mutation run exceeds CI time | Runs on the scheduled full suite, as the current gate does. |
| Backtest corpus too small | Report "unsupported", not a verdict. |

## Repo integration

Verified: the nightly full suite runs `mutmut` and a minimum-score script
(`.github/workflows/full-suite.yml:36-39`, `scripts/check_mutation_score.sh:1-23`,
default minimum 70). Pull-request CI runs `tests/unit tests/property
tests/leakage` with a 75 percent coverage floor (`.github/workflows/ci.yml`,
bounded suite step). Hypothesis and mutmut are dev dependencies
(`pyproject.toml:84,89`). Existing reliability checks include no-false-skill
and leak detection (`benchmarks/reliability.py:192-234`), the regression gate
`benchmarks/check_regression.py`, and the committed-before-results protocol
and holdout rule in `benchmarks/sota/PROTOCOL.md` sections 1-8. Existing
leakage tests: `tests/leakage/test_leakage.py`. Verified: the mutmut scope is
`setup.cfg:1-22`, which lists only reliability, release, attestation, registry,
serving and monitoring modules; no data-stage module is mutated today, so the
qualification packages must be added to `source_paths` and the test selection.

Proposed: `benchmarks/qualification/` (injection, simulator, oracles,
backtest, redteam), `benchmarks/qualification/fixtures/` (frozen references),
`certificates/*.json`, a `--check` entry mirroring `check_regression`, a
bounded PR job (small B, differential and null checks) and a nightly job (full
B, simulator, mutation, backtest on a cached corpus).

## Acceptance criteria

- [ ] Every gating check has a certificate whose `code_hash` matches the code.
- [ ] Differential tests pass at the stated tolerances for all oracles.
- [ ] KS uniformity of null p-values passes for every test at level 0.001.
- [ ] Simulated pipeline FDR upper limit <= alpha + slack; power reported per
      effect size.
- [ ] Mutation gate: no surviving comparison/sign mutant in `ledger/`.
- [ ] Backtest report states effect, interval, strata and sample limits.
- [ ] Red-team report filed with attempts, successes and bound.
- [ ] CI fails if any certificate is missing or stale.

## External referent

Labelled defect corpora, R and scipy/statsmodels reference outputs, the
red-team analyst, and later-period outcomes in the backtest. See
[`../../plan/validation-referents.md`](../../plan/validation-referents.md).
