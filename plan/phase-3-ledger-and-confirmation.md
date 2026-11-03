# Phase 3: Claim ledger and confirmation (F3)

Names follow [`00-canon.md`](00-canon.md). Needs Phase 2 (custody, charter);
the oracle fixtures and null simulator from Phase 1 are used when ready, and
WP-3.1 can start from published formulas without them.

## Goal

Every comparison made on `X` is a logged hypothesis that spends alpha-wealth
under an online rule; survivors are tested once on `C`, and the final claim
carries the guarantee `FDR <= alpha_confirm` (P2).

## Entry criteria

Phase 2 merged: sealed `C`, charter with `ErrorBudget`, manifest v2.

## Verified starting points

- Today's only multiplicity control is Bonferroni inside missingness:
  `threshold = ALPHA / max(len(candidates), 1)` at
  `scmle/data/profiling/missingness.py:104`, with `ALPHA = 0.01` (`:21`).
  The ledger replaces this pattern.
- Evidence records already have a pass/fail/waived and accepted/blocked model:
  `scmle/contracts/evidence.py:16-60`; the ledger should export to it.
- One-time open semantics and `LeakageError` already exist (Phase 2 notes).

## Work packages

**WP-3.1 Online FDR rules** (one agent)
- *Proposed* `scmle/data/ledger/online_fdr.py`: LOND, LORD++ and a
  dependence-robust e-value rule (e-LOND). Each is a state machine
  `(state, p_or_e) -> (decision, new_state)`; default is the robust rule.
  Exact validity conditions must be read from the source papers before
  coding; the conditions in this plan are from memory (*unverified*).
- Done when: decisions match `onlineFDR` oracle fixtures (Phase 1 WP-1.3) on
  at least 20 fixed p-value sequences; wealth never negative.

**WP-3.2 Ledger** (one agent)
- *Proposed* `data/ledger/ledger.py`: append-only `hypothesis_ledger.jsonl`,
  hash-chained, each row a `Check` plus wealth before and after. Any analysis
  code registers hypotheses through one API; unregistered tests are a lint
  error in `data/detectors/`.
- Done when: truncation or edit breaks the chain on verify; replay of the
  ledger reproduces the same decisions bit for bit.

**WP-3.3 Negative controls** (one agent)
- *Proposed* `data/ledger/controls.py`: target permuted within groups or time
  blocks (never globally on structured data), `B >= 20` repeats of the whole
  exploration run; a separate resampling control for feature-only findings.
- Gate quantity: observed discovery count versus the 95th percentile of
  permuted counts.
- Done when: on null worlds from Phase 1 the floor is <= nominal; on
  grouped and temporal worlds a global permutation is shown to be
  anti-conservative while the block permutation is not.

**WP-3.4 Confirmation** (one agent; depends on 3.1, 3.2)
- *Proposed* `data/ledger/confirm.py`: take exploration survivors, open `C`
  once, apply BH (or Holm when the charter asks for FWER) at `alpha_confirm`
  over the survivors only, assign tiers, write `confirmed_claims.json`.
  Reuse-after-failure policy: one fresh unlock is not allowed; instead a
  Thresholdout-style reusable holdout with a noise budget, or the claim stays
  `unverified`.
- Done when: a second `open` raises `LeakageError`; selection code verified
  not to import from custody `C` APIs (import-graph test).

**WP-3.5 Wiring, CLI, report stub** (one agent; depends on 3.2, 3.4)
- Hook the ledger into `stages.py` after profiling; `scmle claims` prints
  tier, p, adjusted p. Full report is Phase 6.

## Parallel lanes

3.1, 3.2, 3.3 are independent. 3.4 needs 3.1 and 3.2. 3.5 last.

## Exit gate

```bash
uv run pytest -q tests/unit tests/property tests/leakage -k "ledger or fdr"
uv run python -m benchmarks.qualification --check fdr
```
Numbers, at `R = 2000` replicates per world:
- Null world: empirical FDR <= `0.05 + 2*sqrt(0.05*0.95/R)` (about 0.0597),
  for independent and for positively correlated tests.
- Planted-effect world: power >= 0.80 where the closed-form oracle says
  >= 0.80 is attainable at that `n`.
- p-values on null features: KS statistic against Uniform(0,1) with p > 0.001
  over 5,000 draws.
- Negative control: observed discoveries exceed the permuted 95th percentile
  on every planted world; do not exceed it on null worlds in > 5% of repeats.
- Ledger replay is bit-identical on 3.11 and 3.14.

## Evidence required

Ledger and claims files from a seeded run, oracle comparison table, FDR/power
curves from the harness, import-graph test output.

## Risks

- Online-FDR conditions are subtle; a wrong rule silently voids the
  guarantee. Mitigation: oracle tests, semi-synthetic empirical FDR, robust
  rule as default.
- Exploration strictness reduces candidates; that is a feature, but report it
  so users see why a finding was not pursued.
- Small `C` gives little power: surfaced by Phase 2 feasibility, not hidden.

## Explicitly out of scope

Detectors, admissibility, multiverse stability, shift tests, the HTML report.
