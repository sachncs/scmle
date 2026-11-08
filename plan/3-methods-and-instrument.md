# Stage 3: Methods and instrument

Names follow [`00-canon.md`](00-canon.md). Item classes and reasons:
[`COMPLETION_AUDIT.md`](COMPLETION_AUDIT.md), "Stage 3". Feature text:
[`../product/features.md`](../product/features.md). Depends on stage 1.

## Goal

Close or formally accept every class A open item of the built instrument and of
the framing, ingestion, causal, decision, uncertainty and loop features, so
that the external sign-offs of stage 5 rest on stated, current limits. Items
that need outside people, data or an owner decision are in stage 5, not here.

## Rules for every item

* A statistical method is a new pre-registered design: floors and seeds are
  written before the first run and are not lowered after results (P1). A
  dated amendment precedes any re-run.
* A change to a hashed source (a detector, an estimator, a certificate script)
  stales its certificate. Re-measure with the same seeds and floors in the same
  commit and say so.
* A method that misses its floor stays descriptive and the miss is published.
  Where a floor cannot be met, the closing criterion is a limit written in
  `docs/ARTIFACTS.md` or `docs/SECURITY.md` with the date, and no claim above
  it.
* New features go to a new stage item. This stage closes rows.

## Work packages

### 3.1 Instrument (R1)

| ID | Scope | Closing criterion |
|---|---|---|
| R1-A5 | Label-noise detectors: lower bounds are below 0.80 on small tables (breast cancer 0.7497 symmetric and 0.6528 class-conditional; iris and wine lower) | Lower bounds at least 0.80 for both types, or the claim is limited to the type and table size that pass and the rest is stated as uncovered |
| R1-A6 | Sentinel detector flags clean rows of natural tables (29/150, 34/178, 1797/1797 against a 5 percent ceiling); certificates use one corpus | Detector fixed or its gating limited, certificate re-issued so the clean-table false-alarm bound holds on all four tables; certificates with the table as unit of replication on enough tables to give an informative bound |
| R1-B2 | Per-period drift with a permutation null and calibrated PSI; the report states `periods` is `not_run` | Per-period two-sample test with a permutation null and a certificate on planted drift; PSI cutoffs set from a null simulation, not a convention; `periods.status` is no longer `not_run` unless a dated decision dropped it |
| R1-B5 | Resampling negative control for a feature-only finding family (`resampling_control` in `scmle/data/ledger/controls.py`); wiring it is unsound until the ledger registers such a family | After B2 (or B12) registers one: call `resampling_control` for that family at the same 95th-percentile gate; a fixture test where a planted feature-structure finding passes and a null does not |
| R1-B7 | Multiverse covers three choices (`IMPLEMENTED` in `stability.py`). The estimator is Spearman's rho, so monotone transforms are no-ops; choices that change a rank correlation (a winsorisation percentile, a missing-value indicator, binning) must be added | Each added choice is implemented with a test that the specification count rises, its options are in the certified planted worlds (floors written before the re-run), and the stability certificate is re-issued; or the choice is removed from the charter template |
| R1-B8 | Stability for multiclass targets: the survival rule reads one signed effect, which Kruskal-Wallis lacks | A new pre-registered multiclass rule (for example epsilon squared with a bootstrap interval and a size band) with planted multiclass worlds, floors and a certificate; otherwise the limit stays documented |
| R1-B12 | MinHash text near-duplicates (`scmle/data/detectors/duplicates.py`) | Implemented, with a planted near-duplicate-text defect, a certificate and a cross-check against an exact Jaccard computation |
| R1-B13 | Instance-dependent label noise is not modelled (symmetric and class-conditional flips only) | A planted instance-dependent flip type in `benchmarks/qualification/inject.py` with a certificate, or an explicit statement in `docs/ARTIFACTS.md` that it is not covered |
| R1-D4a | Producers for the six oracles that have none (`chi2_contingency`, `fisher_exact`, `kendalltau`, `ks_2samp`, `mannwhitneyu`, `pearsonr`) | Each gets a producer in `oracle_table.WIRED`, or is removed with a dated reason |
| R1-E1, R1-E3, R1-E4 | Primary-source re-reads that are openly available: e-LOND venue, Gretton et al. 2012, PCS and the specification curve | Date and result recorded in the module docstring; `grep -rn "from memory" scmle` lists only items still open in stage 5 |
| R1-acc | A test that each register row ID that names a file still resolves | The test passes, and fails when a named file is removed |

### 3.2 Framing and ingestion (F8, F9)

| ID | Scope | Closing criterion |
|---|---|---|
| F8-sel | Selective-label detector recall where the outcome is observed for 70 percent of rows with `b = 0` (weaker than the certified plants) | Recall measured by simulation on a pre-registered design and filed beside `framing_measurements.json` |
| F9-14a | Ingestion charter run identical on Python 3.11 and 3.14 | `tests/integration/test_ingest_charter_run.py` compares the provenance table and reconciliation across both interpreters |
| F9-mut | Survivor bound for `asof_view.py` and `synthesize.py` (136 of 769 scored mutants survive; none changes the cutoff comparison) | A per-module bound in `setup.cfg` |

### 3.3 Causal identification (F10)

| ID | Scope | Closing criterion |
|---|---|---|
| F10-aipw | Re-measure `coverage_aipw` (the boosted-learner world missed its floor) | New pre-registration and run; the result is published whichever way it falls |
| F10-11 | Causal reports bit-identical on Python 3.11 and 3.14 (acceptance 11) | A test comparing both interpreters |
| F10-sub | Subgroup strategies other than the difference in means; heteroscedastic outcomes | Measured on registered designs with all error rates inside the bound |
| F10-seq | Register `scmle/evaluation/sequential.py` as strategy `sequential_ipw` | Registered with its own certificate |
| F10-did | Difference in differences: Rambachan-Roth conditional procedure, covariates, not-yet-treated group, simultaneous bands | Each with a certificate before it may gate |
| F10-iv | Instrumental variables: 2SLS with several instruments, Olea-Pflueger effective F, exclusion-violation sensitivity, a battery | Certificate; until then the claim stays `attested` |
| F10-rd | Regression discontinuity: MSE-optimal bandwidth, McCrary or Cattaneo-Jansson-Ma density test | Certificate |
| F10-syn | Synthetic control: predictor covariates and a battery of its own | Certificate |
| F10-nc | Negative-control exposures beyond `backdoor_aipw` (the run refuses others) | Supported per strategy with a test |

### 3.4 Decision layer (F11)

| ID | Scope | Closing criterion |
|---|---|---|
| F11-W3 | `evaluation/fairness.py` and `mitigation.py` become clients of `decision.constraints`; their constants stop gating; solve the `reweighing` mechanism | Both files use `decision.constraints`; existing tests stay green |
| F11-W5a | Re-register and re-measure the net benefit band (coverage miss at `n = 2000`, low power) | `--check f11` passes with A3 gating under a new floor, or the band stays descriptive on purpose |
| F11-W5d | Policy trees of depth two | Certified, with the complexity penalty beyond a per-treated-row cost |
| F11-ebl | Empirical-Bernstein lower bounds for OPE | Implemented and certified |

### 3.5 Uncertainty and problem types (F12)

| ID | Scope | Closing criterion |
|---|---|---|
| F12-1c | Intervals or sets for ordinal, multioutput, multilabel and forecast runs (all return `None` now). Survival, ranking and uplift stay without intervals | Coverage measured once on untouched data per type, tiered |
| F12-3a | Wire the refusal of small-effective-sample shift weights to F13 escalation | An escalation row in the loop test |
| F12-4a | Forecast: per-horizon calibrated quantiles, WIS in the run, `mase` scale from the training history | WIS reported by a run; test |
| F12-5a | Hierarchy: charter hierarchy and summing matrix; WIS rather than squared error | Charter field and run test |
| F12-6a | Survival: charter field for `tau`; IPCW inside Brier and dynamic AUC; calibration at `tau` | Charter field and run test |
| F12-7c | G10: NDCG@10 equal to `sklearn.metrics.ndcg_score` on frozen lists | `1e-9` agreement |
| F12-7d | G12: per-label support rule, thresholds and sets | Rule certified on simulated multilabel worlds |
| F12-8a | Near-duplicate grouper (hash plus embedding cosine) and a group split (G14) | Recovery at least 0.95 at false-merge at most 0.01 on planted duplicates; leak inflation shown, then removed by the grouped split |
| F12-9c | G3 at `R = 2000` after a re-registered design (published miss 0.864 against 0.87; a single additive margin cannot equalize coverage where noise is largest, so a locally weighted or asymmetric design is a new method) | Certified, or the conditional claim stays descriptive |
| F12-G13 | Entity-level calibration wired; the claim "row-wise is anti-conservative" is unsupported by the 30-entity world | The wire statement corrected in docs and code |
| F12-py | Every certificate replays bit-identical on Python 3.11 (produced on 3.14 only) | Replay on 3.11 |

### 3.6 Closed loop (F13)

The alarm design that lets H2 and H10 pass is item 2.2 in stage 2.

| ID | Scope | Closing criterion |
|---|---|---|
| F13-0b | An independent implementation of the betting bound (exact enumeration of the martingale and Ville properties exists; no library comparison) | A pinned library comparison, or a dated decision that enumeration is the referent |
| F13-5a1 | Novelty and set size for regression bundles; a certified `novelty_bound` (a charter input today, default 0.05) | `UncertaintyFeed` handles regression; the bound has a certificate |
| F13-5a2 | An exact or conservative weighted threshold test replacing the Kish effective size and rounded-up count | Certified; H9 re-measured for the weighted case |

## Exit gate

```bash
uv run pytest -q tests/unit tests/property tests/leakage
uv run python -m benchmarks.qualification --check all
uv run ruff check . && uv run ruff format --check . && uv run mypy scmle benchmarks
```

Every row of this stage is deleted from the register (closed) or its limit is
written in `docs/ARTIFACTS.md` or `docs/SECURITY.md` with the date.

## Risks

| Risk | Mitigation |
|---|---|
| A5 or A6 cannot reach the floor with the current methods | Accept as descriptive; do not lower the floor after seeing the interval |
| Mutation floor 80 is not reachable on a module | Report survivors per module and justify each; do not drop modules from scope |
| A change to a hashed source invalidates a certificate | Re-measure with the same seeds and floors in the same commit |
| A method passes on simulation and fails on real data | Tiers follow held-out measurement, not the theorem; misses are published (stage 5 supplies the real data) |
