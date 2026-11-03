# Phase 4: Detectors and admissibility (F4, F5)

Canon: [`00-canon.md`](00-canon.md). Features: `product/features/F4-*.md`,
`F5-*.md`.

## Goal

Replace today's fixed-cutoff checks with detectors that (a) gate only after a
certificate exists (P1) and (b) report a tier (P4). F4 decides which fields may
be features and tests for leakage. F5 detects missingness mechanisms, label
noise, duplicates, sentinels and outliers.

Verified starting point (not to be assumed complete):
- Cutoffs to retire or demote to policy defaults: `LEAK_ASSOCIATION = 0.98`,
  `MISSINGNESS_LEAK = 0.9` (`scmle/data/profiling/profiler.py:21-22`),
  `NOTE_RATE = 0.05` (`scmle/data/profiling/label_noise.py:26`),
  `UNREPRESENTATIVE_AUC = 0.65`
  (`scmle/data/profiling/representativeness.py:23`).
- Leakage is blocked only through a string match: `LEAKAGE_MARKER = "leakage"`
  (`scmle/core/stages.py:93`) used at `stages.py:499` under
  `leakage_policy == "block"`. Proposed: replace with typed `Check` decisions.
- Missingness already has a chi-square/rank test with a Bonferroni alpha
  (`scmle/data/profiling/missingness.py:18-22,104`); it is the model to copy.

## Entry criteria

- Phase 1: injection library, semi-synthetic simulator and oracle fixtures
  merged (needed to issue certificates).
- Phase 2: `AnalysisCharter` decision map and the X/C/F custody API merged;
  detectors read only the `X` handle (P3).

## Work packages (one agent each)

| WP | Scope | Creates / modifies | Depends on | Definition of done |
|---|---|---|---|---|
| 4.1 | F4 as-of and provenance table: typed field records (origin, writer, overwrite, as-of, source), attestation vs log-derived tier | `scmle/data/admissibility/provenance.py`, `asof.py` (proposed); tests | P1, P2 | Every field has a tier; a field with no as-of is `unverified`; strict mode excludes it; unit tests for each tier |
| 4.2 | F4 leakage tests: out-of-fold single-feature skill with a permutation null, one-sided lower bound; missingness-leak via exact/chi-square with Cramer's V | `scmle/data/admissibility/leakage.py` (proposed); retires `leakage_findings` (`profiler.py:278`) | 4.1, P1 | Certificate on file (size <= alpha, recall >= 0.9 on planted leaks, Wilson lower bound); old constants removed |
| 4.3 | F4 DAG falsification: conditional-independence tests on `X` only, every test in the ledger, DAG stored as a hashed hypothesis | `scmle/data/admissibility/dag.py` (proposed) | 4.1, Phase 3 ledger API | Falsification-only output; no claim of confirmation; graph hash in manifest |
| 4.4 | F5 missingness: row/pattern analysis, Little MCAR (own implementation unless `statsmodels` approved), indicator model, tipping-point delta range | `scmle/data/detectors/missingness.py` (proposed), absorbs `profiling/missingness.py` | P1 | Certificate for MCAR/MAR/MNAR plants; in-fold imputation guarantee tested |
| 4.5 | F5 labels: confident learning plus a second, in-sample detector (AUM); noise rate with interval | `scmle/data/detectors/labels.py` (proposed), absorbs `label_noise.py` | P1 | Two detectors agree-or-disagree is recorded; noise-rate interval; certificate per noise type (uniform, class-conditional, instance-dependent) |
| 4.6 | F5 structure: duplicates (exact and blocked near-duplicates), sentinels (spike at round values vs smooth density), outlier triage by provenance not z-score | `scmle/data/detectors/structure.py` (proposed); reuses `splitting/duplicates.py` | P1, 4.1 | Certificates; outliers labelled `error`, `real_extreme`, `population` only when provenance supports it, else `unclassified` |
| 4.7 | Wiring: call detectors from the profile stage on `X` only, map decisions to `FeaturePlan.dropped`, replace the marker-string gate | `scmle/core/stages.py` near `:400-503`; `scmle/contracts/data.py` | 4.2, 4.4-4.6 | Integration tests; no detector gates without a certificate (enforced by a test that lists gating checks and certificates) |

## Parallel lanes

- Lane F4: 4.1 -> {4.2, 4.3}.
- Lane F5: {4.4, 4.5, 4.6} fully parallel.
- 4.7 joins both lanes last. Lanes share only the `Check` type from Phase 0.

## Exit gate

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy scmle benchmarks
uv run python scripts/check_api_compatibility.py tests/fixtures/public_api.json
uv run pytest tests -q
uv run python -m benchmarks.qualification --check --detectors all
```
Numeric thresholds (proposed, fixed in the charter before running):
- Size on null data: empirical false-alarm rate <= alpha, inside the 99%
  binomial band over >= 1000 null replicates.
- Recall on planted defects: Wilson 95% lower bound >= 0.80 per defect type,
  >= 0.90 for leakage; planted n >= 200 per type.
- Clean-copy false-alarm rate <= 0.05 for each detector.
- `grep -rn "LEAK_ASSOCIATION\|NOTE_RATE\|UNREPRESENTATIVE_AUC" scmle` returns
  only policy-default definitions.

## Evidence required

`detector_certificates.json` with one entry per detector and defect type;
`admissibility_table.json`; differential tests of p-values against scipy
(and R where available) at fixed inputs; mutation score on the new modules
>= the repo minimum (`scripts/check_mutation_score.sh`, default 70).

## Risks

| Risk | Mitigation |
|---|---|
| Planted defects are easier than real ones, so recall is optimistic | Report per type; Phase 6 backtest on real labelled corpora is the check |
| Auto-quarantine drops a legitimate strong feature | Ablation benchmark with/without quarantine; remediation reversible and logged |
| Heavy tests slow wide data | Per-check time budget; "skipped, over budget" recorded, never silent |
| No as-of data, so everything is `unverified` | Intended behaviour; tier shown in report header |

## Explicitly out of scope

Fairness proxies and selection-bias evidence (descriptive only until a
certificate exists), image/audio/graph modalities, new third-party
dependencies beyond a pending `statsmodels` decision, multiverse and shift
(Phase 5), the confirmation unlock (Phase 3).
