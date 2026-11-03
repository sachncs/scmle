# Phase 0: Foundations

Names follow [`00-canon.md`](00-canon.md). Items marked *proposed* do not exist
yet.

## Goal

Land the shared types, the error-budget algebra and the determinism rules that
every later phase builds on, so Phases 1 and 2 can run in parallel without
renegotiating vocabulary.

## Entry criteria

- `master` is green on the CI commands in `.github/workflows/ci.yml:40-56`
  (ruff, mypy `scmle benchmarks`, API compatibility, bounded pytest with
  `--cov-fail-under=75`).
- `plan/00-canon.md` is agreed.

## Verified starting points

- Frozen, slotted dataclasses plus `StrEnum` and a `require_text` validator are
  the contract style: `scmle/contracts/evidence.py:16-60`. Mirror it.
- Seeds are a single field today: `seed: int = 42` at
  `scmle/contracts/tasks.py:347`; splits record it in `SplitManifest`
  (`scmle/data/splitting/splitters.py:32-50`).
- The public API is pinned by `tests/fixtures/public_api.json` (an `exports`
  map of name, kind and `signature_sha256`) and checked by
  `scripts/check_api_compatibility.py`.
- Name clash: `EstimandSpec` already exists in
  `scmle/evaluation/parameter_estimation.py:98`. The charter's estimand must
  use a different name (*proposed*: `AnalysisEstimand`).

## Work packages

**WP-0.1 Core types** (one agent)
- Create *proposed* `scmle/contracts/qualification.py`: `Tier`
  (`confirmed|attested|unverified`), `Check`, `Claim`, `Certificate`,
  `Decision` (`pass|warn|quarantine|block`). Frozen, slotted, `to_dict` and
  `from_dict`, validators like `require_text`.
- `Check` fields: id, null hypothesis text, statistic, p_value or interval,
  effect, n, decision, set_id (`X`/`C`/`F`/`all`), seed_path, tier.
- Done when: round-trip JSON tests, rejects NaN p-values and p outside [0, 1],
  tier cannot be `confirmed` unless `set_id == "C"`.
- Depends on: none.

**WP-0.2 Error-budget algebra** (one agent)
- Create *proposed* `scmle/data/ledger/budget.py` with `ErrorBudget`
  (`alpha_confirm`, `wealth_exploration`, `permutations_b`, detector
  tolerances) and pure functions: `split_budget`, `spend`, `remaining`,
  `confirmation_alpha(n_survivors, rule)`. Budget is immutable; spending
  returns a new budget.
- Done when: property tests (Hypothesis, as in `tests/property/`) prove spend
  never makes wealth negative, `spend` is order-independent for non-adaptive
  rules, and the canon's P2 statement (exploration and confirmation budgets do
  not add) is encoded as a type-level separation.
- Depends on: WP-0.1 (types only).

**WP-0.3 Seeded RNG discipline** (one agent)
- Create *proposed* `scmle/core/randomness.py`: derive child generators from
  `numpy.random.SeedSequence` keyed by (run seed, component name, index), so
  adding a check never perturbs another check's stream.
- Add a test that scans `scmle/data/{charter,custody,ledger,admissibility,
  detectors,robustness}` for `np.random.<fn>`, `random.` and unseeded
  `default_rng()`; any hit fails.
- Done when: same seed gives bit-identical `Check` lists in two processes;
  different component names give independent streams (correlation test).
- Depends on: none.

**WP-0.4 API-fixture handling** (one agent)
- Decide and document the rule: new qualification types are exported
  additively; no existing signature in `public_api.json` changes. Add the new
  exports and regenerate the fixture with the script (its update flag is
  *unverified*; read `scripts/check_api_compatibility.py` before relying on it).
- Update `docs/API_STABILITY.md` and the CHANGELOG "Unreleased" section.
- Done when: `check_api_compatibility.py` passes and a test asserts no
  previously pinned entry changed.
- Depends on: WP-0.1.

## Parallel lanes

All four packages run concurrently except 0.2 and 0.4, which wait only for the
type names from 0.1 (agree names first, 15 minutes of work, then fan out).

## Exit gate

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy scmle benchmarks
uv run python scripts/check_api_compatibility.py tests/fixtures/public_api.json
uv run pytest -q tests/unit tests/property tests/leakage --cov=scmle \
  --cov-fail-under=75
```
Plus: new modules at >= 95% line coverage; determinism test passes on 3.11
and 3.14 (the CI matrix ends are `3.11` and `3.14`, `ci.yml:33`).

## Evidence required

Test report, coverage report for the new modules, the regenerated fixture diff
(additions only), the RNG-scan test output.

## Risks

- API churn in a beta package: additive exports only; fixture diff reviewed.
- Seed-stream coupling would silently change old results: new code uses
  `SeedSequence`; legacy code paths keep `seed: int` untouched.

## Explicitly out of scope

Any statistical test, any splitting change, any detector, any CLI.
