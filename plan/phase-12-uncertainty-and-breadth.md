# Phase 12: Uncertainty and problem-type breadth (F12)

Names follow [`00-canon.md`](00-canon.md). Feature spec:
[`../product/features/F12-uncertainty-and-breadth.md`](../product/features/F12-uncertainty-and-breadth.md).

## Goal

Every deployed prediction carries an interval or set whose coverage was
measured once on untouched data and is tiered (P4); every problem type the
system accepts has a validation protocol that respects its data structure and
a benchmark gate with an external referent, or its outputs are tiered
`unverified`.

## Status

Partly built (see `CHANGELOG.md`, "Uncertainty and problem-type breadth"). The
building blocks exist as library code with measured simulation gates in
`benchmarks/qualification/uncertainty_certificates.json`: class-conditional
and weighted conformal, CV+ bounds, coverage reports with Holm slice flags,
proper scores, covariate-censoring IPCW, MinT, query bootstrap, shift weights
and `task_gate_status.json`. What is left is wiring them into the run, the
gates that need outside data or tools, and two methods that missed their
floors. Nothing below was run on real data.

## Published misses (not retuned)

- G3: CQR coverage in the highest-noise decile 0.864 against the floor 0.87
  (150 replicates, 40-tree boosted quantile models). The conditional claim
  of CQR is descriptive until a re-registered design passes.
- G5: familywise false-flag rate of the slice flag 0.0755 against the bound
  0.0597. The flag compares each slice with the nominal level although the
  conformal quantile itself varies with the calibration sample, so slice
  coverages are correlated and the test is anti-conservative. The flag stays
  descriptive; a fix (test against the calibration-set-conditional coverage,
  or a larger tolerance fixed in advance) needs a new pre-registration.
- G13: row-wise calibration did not under-cover a new entity in the
  30-entity, 40-row world (0.9042); the claim that it is anti-conservative is
  not supported by that world.

## Work packages still open

**WP-12.0 Reference pinning** (one agent; blocks the items marked *)
- Read the theorems, not only abstracts, for Barber et al. (1905.02928),
  Romano et al. (1905.03222), Tibshirani et al. (1904.06019); record exact
  statements in the docstrings (`conformal.py` quotes the abstracts only).
- Locate and read: the source of the Beta coverage law (G1 measured it, did
  not cite it), the impossibility result for conditional coverage, Uno et al.
  2011 (`ipcw.py` is written from memory), Wickramasuriya et al. 2019
  (`reconcile.py` from memory), Gneiting and Raftery 2007, M4 and M5 papers.
- Confirm existence, licence and SHA-256 of M4, M5, one LTR set, GBSG2 or
  Rotterdam, METABRIC, Yeast, Scene; update `corpora.json` from `candidate`.
  Confirm MAPIE implements CV+, CQR and class-conditional sets, record the
  version (it is not installed in the development environment).

**WP-12.1 Proper conformal core in the run*
- Store the K fold models and serve CV+ (`conformal.cv_plus_bounds` exists);
  plain split conformal when `n_cal >= 1000`. Replace the pooled
  out-of-fold scores with a separate refit at `core/stages.py`
  (`conformal_calibration`, `quantile_intervals`); extend the bundle checksum
  to fold-model hashes and test tamper on one fold model. Charter size cap
  with fallback.
- Tie the CQR base levels to the requested level (`quantile.py` fixes
  0.05/0.95).
- Differential test against MAPIE on three frozen datasets (`1e-9`).
- Intervals or sets for ordinal, multioutput, multilabel and forecast. Survival,
  ranking and uplift stay without intervals (out of scope; uplift has no
  observed individual effect). All still return `None`.

**WP-12.2 Coverage report in the run**
- `coverage_report.json` and `coverage_claim.json` produced by a stage on
  `C`/`F` through custody, registered as a claim before `C` is read, read
  once; `ModelBundle.coverage_report`; extend `SliceReport` (`slices.py`).
- Decide the G5 redesign above before the slice flag may gate.

**WP-12.3 Shift weights in the bundle**
- `ModelBundle.predict_interval(..., shift_weights=)` and calibration weights
  stored; extend `AdversarialValidation` (`shift.py`) to emit the same odds;
  ESS refusal wired to F13 escalation. (`shift_weights.py` and the gate G6
  exist.)

**WP-12.4 Forecast protocol**
- Per-horizon calibrated quantile forecasts; WIS reporting in the run;
  `ForecastMetric` with the in-sample scale supplied from the training history
  by the stage (`mase_insample` exists, `mase` is still the same-rows score
  by default); reproduce the official M4 Naive and Seasonal-naive scores (G8)
  and the system-versus-baselines interval. Needs the M4 data release.

**WP-12.5 Hierarchical reconciliation on real data**
- Charter-declared hierarchy and summing matrix; WIS rather than squared
  error; `hts`/`fable` differential; the M5 hierarchy.

**WP-12.6 Survival**
- Charter field for `tau`; use `ipcw.censoring_weights` inside the Brier and
  dynamic AUC (the 0.05 floor and its count are in `survival.py`/`ipcw.py`);
  calibration at `tau`; scikit-survival and R `survival`/`pec` differential
  (G11(ii)) on GBSG2/Rotterdam and METABRIC.

**WP-12.7 Ranking, multilabel, panel**
- G10 against one public LTR set with published figures; per-label support
  rule and per-label thresholds and sets (G12); `unit_of_prediction` charter
  field and entity-level calibration wired into the run (the entity-wise
  score selection of G13 is a simulation only).

**WP-12.8 Near-duplicate grouping and text/image encoders** (not started)
- Duplicate grouper (hash + embedding cosine), group split; one frozen image
  and one text encoder behind the extras, licence recorded (G14, G15).

**WP-12.9 Wiring and real-data record**
- Put `task_gate.tier_ceiling` on the outputs of a run; G7 over real
  datasets with published misses; run the certification at `R = 2000` for G3
  (it ran 150) after re-registration.

## Exit gate

```bash
uv run pytest -q tests/unit tests/property tests/leakage -k "conformal or coverage or proper_score or ipcw or reconcil or neardup"
uv run python -m benchmarks.qualification --check
uv run python -m benchmarks.reliability --only interval_coverage
```

Open numbers: G7, G8 (official M4 Naive and Seasonal-naive reproduced; the
forecast claim "confirmed" needs the upper limit below zero), G9 on M5, G10,
G11(ii), G12, G14 (recovery `>= 0.95` at false-merge `<= 0.01`), G15, and
replay bit-identical on Python 3.11 and 3.14 for every certificate (the
certificates were produced on 3.14 only).

## Evidence still required

Differential tables against MAPIE, scikit-survival, R; the M4/M5 score tables
with corpus hashes; the `unverified.md` appendix for literature claims not
resolved by WP-12.0; the list of published coverage misses on `F`.

## Risks

- A gate passes on simulation and fails on real data. Mitigation: G7 is a
  recorded-miss policy; tiers follow held-out measurement, not the theorem.
- Fold-model storage inflates bundles by K times. Mitigation: size cap in the
  charter; fall back to split conformal.
- Reference corpora unavailable (licence, hash). Mitigation: the type stays
  `unverified`; no gate is faked.
- MASE: `mase` keeps its same-rows meaning for stored runs; the standard
  score is a separate kind.
- MinT covariance estimation unstable with many bottom series. Mitigation:
  shrinkage; bottom-up remains available and is the baseline.

## Explicitly out of scope

ACI and non-exchangeable conformal, conformal for survival or ranking,
fine-tuned vision/language models, detection/segmentation, click-model LTR,
mixed-effects models as a family, anomaly detection, any closed-loop
behaviour (Phase 13), and the retrain shuffle for temporal data (Phase 13).
