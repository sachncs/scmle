# Phase 5: Robustness and shift (F6)

Canon: [`00-canon.md`](00-canon.md). Feature: `product/features/F6-*.md`.

## Goal

Decide which exploration survivors are fragile, measure where the data
under-represents the deployment population, and run the forward-in-time check,
all before the confirmation set is opened. Findings that are not stable across
the declared judgment calls are reported as fragile and are not sent to
confirmation (P2: this keeps the confirmation family small).

Verified starting point:
- Drift today is PSI minus a sampling-noise floor
  (`scmle/monitoring/reference.py:77-98`, `NOISE_TAIL = 1.5` at `:24`), with
  no p-value; the floor was checked by simulation per its docstring.
- Representativeness is an out-of-fold AUC against a fixed `0.65`
  (`scmle/data/profiling/representativeness.py:23`), no permutation null.

## Entry criteria

- Phase 3: ledger and survivors list API merged (the multiverse re-runs
  ledger entries).
- Phase 4: detectors with certificates merged (their alternatives define
  several judgment calls).
- The charter declares the multiverse (imputation, outlier rule, encoding,
  inclusion criteria, MNAR delta) before any result exists.

## Work packages (one agent each)

| WP | Scope | Creates / modifies | Depends on | Definition of done |
|---|---|---|---|---|
| 5.1 | Multiverse runner: enumerate the declared specifications, re-run each candidate finding, cache by `(spec hash, seed)` | `scmle/data/robustness/multiverse.py` (proposed) | Phase 3, 4 | Deterministic under a fixed seed; spec count and cost recorded; resumable |
| 5.2 | Stability verdict: sign consistency and size band across specs, specification curve data, bootstrap perturbation intervals (PCS) | `.../stability.py` (proposed) | 5.1 | Rule fixed in the charter: finding is `stable` iff sign agrees in >= 80% of specs and the median effect is within +-50% of the headline; others `fragile` |
| 5.3 | Two-sample shift tests: classifier two-sample test with permutation p-value and AUC interval, kernel MMD with permutation, per-feature KS/chi-square under BH | `scmle/data/robustness/shift.py` (proposed); migrates `representativeness.py` | Phase 0 | p-values uniform under null (KS test on 1000 null reps, p > 0.01); old `0.65` constant demoted to a policy default |
| 5.4 | Drift by period and calibrated PSI: permutation-calibrated threshold replacing the heuristic floor; always-valid sequential option for repeated looks | `scmle/data/robustness/drift.py` (proposed); touches `monitoring/reference.py` | 5.3 | `examples/monitor_drift.py` still runs; false-alarm rate under repeated monitoring <= alpha over 1000 simulated streams |
| 5.5 | Subgroup coverage and "unverifiable" status: with no deployment sample, compare to stated marginals only and mark representativeness `unverifiable` | `.../coverage.py` (proposed) | 5.3 | Report distinguishes `pass`, `fail`, `unverifiable`; never reports `pass` without a reference |
| 5.6 | Forward-in-time check on `F` (only when the charter has `F`): runs after survivors are fixed, uses the custody API, one open, logged | `.../forward.py` (proposed) | Phase 2 custody, 5.2 | `F` opened once; open recorded in the access log; skipped with reason when data is not temporal |
| 5.7 | Report: `stability_report.json`, `shift_report.json`, specification-curve plots, which choice drives which result | `scmle/artifacts/` (proposed new report module) | 5.2, 5.3-5.6 | Both JSON files validate against schema; plots render in `eda_report.html` |

## Parallel lanes

- Lane stability: 5.1 -> 5.2 -> (6 joins at 5.7).
- Lane shift: 5.3 -> {5.4, 5.5}.
- Lane forward: 5.6 after custody and 5.2.
- 5.7 last. Lanes touch disjoint modules.

## Exit gate

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy scmle benchmarks
uv run python scripts/check_api_compatibility.py tests/fixtures/public_api.json
uv run pytest tests -q
uv run python -m benchmarks.qualification --check --family robustness
```
Numeric thresholds (proposed):
- Two-sample test size: false-alarm rate <= alpha within the 99% binomial
  band; power >= 0.80 at a planted AUC of 0.60 with n = 2000 per side.
- Fragile detection: on semi-synthetic data with planted spec-dependent
  effects, recall of the "fragile" label >= 0.80 (Wilson lower bound) and
  stable true effects labelled stable >= 0.90.
- Sequential drift: <= alpha false alarms over repeated looks (Phase 4 style
  band).
- Multiverse run time <= the charter budget, with skipped specs recorded.

## Evidence required

`stability_report.json`, `shift_report.json`, specification curves, KS
uniformity logs for the null p-values, differential tests of MMD and KS
against scipy, and the access log showing `F` opened exactly once.

## Risks

| Risk | Mitigation |
|---|---|
| "Defensible specification" set is arbitrary, so stability is partly author's choice | Declared and hashed in the charter before results; Phase 6 red team attacks the declaration |
| Multiverse cost explodes | Cap specs per charter; time budget per spec; record skipped specs |
| No deployment sample at build time | `unverifiable` status, not a pass |
| Forward check conflated with replication | `C` tests replication, `F` tests stability; never merged |

## Explicitly out of scope

Opening `C` (Phase 3), causal adjustment-set search, model-based shift
adaptation or reweighting, label-shift correction (black-box shift estimation
is noted as a later extension), image/audio modalities.
