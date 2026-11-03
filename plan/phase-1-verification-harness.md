# Phase 1: Verification harness skeleton (F7, first)

Names follow [`00-canon.md`](00-canon.md). Built before any detector so every
later detector is measured against it from its first commit (principle P1).

## Goal

A harness that can plant known defects in real data, simulate null and
alternative worlds, compare our statistics with reference implementations, and
store detector certificates. No detector needs to exist yet; the harness is
exercised with trivial stand-ins.

## Entry criteria

Phase 0 merged: `Check`, `Certificate`, `Tier`, seeded RNG derivation.

## Verified starting points

- Benchmarks live in `benchmarks/` with a regression gate
  (`benchmarks/check_regression.py`, `benchmarks/baseline.json`) and a
  pre-registration style protocol, `benchmarks/sota/PROTOCOL.md`, committed
  before results and changed only by dated amendment. Reuse that discipline.
- The 51-task TabArena list is already in
  `benchmarks/sota/tasks_tabarena_v0pt1.csv`.
- Unexplained skips fail CI via `scripts/check_pytest_skips.py` and
  `scripts/pytest_skip_manifest.json`; any corpus test that skips offline needs
  a manifest entry.
- Mutation gate: `mutmut` is a dev dependency (`pyproject.toml:89`) and
  `scripts/check_mutation_score.sh` enforces a minimum (default 70). Verified:
  the scope is set in `setup.cfg:1` (`[mutmut] source_paths`) and covers only
  reliability, release evaluation, attestation, registry, serving and
  monitoring modules. No `scmle/data/` file is in scope, so the qualification
  modules are unmutated until a work package adds them to `source_paths` and
  `pytest_add_cli_args_test_selection`.
- The test-flake policy is `scripts/check_test_stability.py` (quarantine needs
  owner, reason, expiry).

## Work packages

**WP-1.1 Defect injector** (one agent)
- *Proposed* `benchmarks/qualification/inject.py`. Defect catalogue, each
  returning the mutated copy and a ground-truth mask: label flips (symmetric,
  class-conditional), missingness (MCAR, MAR, MNAR with known mechanism),
  sentinel values, exact and near duplicates, outliers, target-proxy leakage
  (noisy copy of y, post-outcome field), covariate shift.
- Done when: every injector is seeded, idempotent for equal seeds, and has a
  test that the mask matches the actual mutation cell by cell.

**WP-1.2 Semi-synthetic simulator** (one agent)
- *Proposed* `benchmarks/qualification/simulate.py`: keep real feature
  matrices, generate y from planted effects (known coefficients, known
  null features) and null worlds by permuting y within groups or time blocks.
- Done when: it yields worlds with known truth sets and a documented effect
  size grid; null worlds reproduce the same feature dependence structure.

**WP-1.3 Reference-oracle fixtures** (one agent)
- *Proposed* `tests/fixtures/oracles/*.json` plus
  `benchmarks/qualification/make_oracles.py` (run offline, never in CI): fixed
  inputs with outputs from scipy, statsmodels and R (`onlineFDR`, `knockoff`).
  Record tool versions and generator hash in each file.
- Done when: a loader test verifies the file hashes and a comparison helper
  asserts agreement to a stated tolerance (default 1e-8 relative).

**WP-1.4 Corpus manifest and fetcher** (one agent)
- *Proposed* `benchmarks/qualification/corpora.json` schema: id, source URL,
  licence, sha256, defect ground-truth file, adjudication source, intended use
  (certificate vs backtest). Nothing large is vendored; a fetcher verifies
  hashes. Seed list: Titanic `boat`/`body` leakage, a human-adjudicated
  label-error corpus, TabArena tasks.
- Done when: schema validation test passes, hash mismatch aborts, offline mode
  skips with a manifest entry in `scripts/pytest_skip_manifest.json`.

**WP-1.5 Certificate store** (one agent)
- *Proposed* `scmle/data/detectors/certificates.py` writes
  `detector_certificates.json`: per detector and defect type, n planted,
  recall with Wilson interval, false-alarm rate on a clean copy with Wilson
  interval, keyed by (detector code hash, corpus hash, seed set).
- Done when: stale certificates (code hash changed) are rejected, and a
  detector without a certificate cannot be marked `gating` (P1).

**WP-1.6 CI integration** (one agent; waits for 1.1 to 1.5 interfaces)
- Add a bounded `qualification-harness` job to `.github/workflows/ci.yml`
  (pattern at lines 27-62) and the long simulations to
  `.github/workflows/full-suite.yml`. Add `benchmarks/qualification/` to the
  `mypy scmle benchmarks` scope (already covered by path).
- Done when: the job runs under 10 minutes on the CI runner (job limit is
  20, `ci.yml:30`).

## Parallel lanes

1.1, 1.2, 1.3, 1.4 are independent. 1.5 needs the `Certificate` type only.
1.6 last.

## Exit gate

```bash
uv run pytest -q tests/benchmarks tests/unit -k qualification
uv run python -m benchmarks.qualification --smoke
uv run python scripts/check_pytest_skips.py <junit.xml>
bash scripts/check_mutation_score.sh 70   # harness modules in mutation scope
```
Numbers: injector masks exact (0 mismatched cells); simulator null worlds give
a stand-in test with empirical size within `alpha +/- 2*sqrt(alpha*(1-alpha)/R)`
at `R = 2000`; oracle agreement <= 1e-8 relative; mutation score >= 70 on
harness code.

## Evidence required

Corpus manifest with hashes, oracle fixture files with tool versions, a sample
certificate file produced by a stand-in detector, CI run link.

## Risks

- Planted defects are synthetic, so recall is optimistic: report per defect
  type and keep the real labelled corpora as the tie-breaker (Phase 6).
- Corpus licences: manifest records licence; unlicensed data is excluded.
- R is not available in CI: oracles are generated offline and pinned by hash.

## Explicitly out of scope

Real detectors, the predictive-validity backtest, red team (Phase 6).
