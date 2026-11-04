# Phase 12: Uncertainty and problem-type breadth (F12)

Names follow [`00-canon.md`](00-canon.md). Feature spec:
[`../product/features/F12-uncertainty-and-breadth.md`](../product/features/F12-uncertainty-and-breadth.md).

## Goal

Every deployed prediction carries an interval or set whose coverage was
measured once on untouched data and is tiered (P4); every problem type the
system accepts has a validation protocol that respects its data structure and
a benchmark gate with an external referent, or its outputs are tiered
`unverified`.

## Entry criteria

- Phase 3 merged (ledger, one-time open of `C`): present as
  `scmle/data/ledger/ledger.py`, `confirm.py`.
- Phase 1 harness present: `benchmarks/qualification/` with `simulate.py`,
  `inject.py`, `oracles.py`, `corpora.py`.
- Phase 2 custody for `C` and `F`.
- Phase 6 (external replication) is **not** required to start, but WP-12.9
  needs the corpus manifest procedure from `validation-referents.md`.

## Verified starting points

- Split-conformal quantile, regression interval, classification set:
  `scmle/evaluation/conformal.py:23-39`, `:59-72`, `:75-101`, `:138-158`.
- CQR with fixed 0.05/0.95 base levels: `scmle/evaluation/quantile.py:21-22`,
  `:38-96`.
- Calibration is pooled out-of-fold scores with a separate refit served
  model: `scmle/core/stages.py:2405-2442`, `:2480-2504`. No stated theorem
  covers it (F12, "What exists today").
- Seven task types return no interval or set: `stages.py:2380-2389`.
- The only coverage gate is 90% nominal with pass at >= 0.85, one dataset,
  one seed: `benchmarks/reliability.py:287-304`.
- Forecast: rolling split and embargo `scmle/data/splitting/splitters.py:581-602`,
  `:653`; same-rows MASE `scmle/evaluation/forecast.py:44-47`; baselines
  `scmle/evaluation/backtest.py:53`; no probabilistic score.
- Survival: marginal-KM IPCW with a 0.05 floor, `scmle/evaluation/survival.py:21-37`;
  Harrell C, `concordance.py:14-48`.
- Ranking split by query, `splitters.py:502,744`. Image and document
  baselines, `scmle/models/images.py:1-6`, `documents.py:150`; `multimodal`
  extra is `pillow` only, `pyproject.toml:66`.
- Case-insensitive grep for `jackknife`, `mondrian`, `crps`, `pinball`,
  `winkler`, `reconcil`, `ipcw` over `scmle/*.py` hits only unrelated code
  ("reconcile" row-accounting checks in `evaluation/longitudinal.py`,
  `sequential.py`, `parameter_estimation.py`; `inference/mapping.py:12`;
  `data/admissibility/reconciliation.py`). The IPCW logic in `survival.py`
  does not use the word.

## Work packages

**WP-12.0 Reference pinning** (one agent; blocks 12.1, 12.8)
- Read the theorems, not only abstracts, for Barber et al. (1905.02928),
  Romano et al. (1905.03222), Tibshirani et al. (1904.06019); record exact
  statements in module docstrings as `online_fdr.py:12-90` does.
- Locate and read: the source of the Beta coverage law, the impossibility
  result for conditional coverage, Uno et al. 2011, Wickramasuriya et al.
  2019, Gneiting and Raftery 2007, the M4 and M5 papers.
- Confirm existence, licence and SHA-256 of M4, M5, one LTR set, GBSG2 or
  Rotterdam, METABRIC, Yeast, Scene; update `corpora.json` status from
  `candidate`. Anything not confirmed stays `candidate` and its gate does not
  run.
- Confirm MAPIE (or an alternative) implements CV+, CQR and class-conditional
  sets; record version.
- Done when: every "unverified" in F12 is either resolved with a citation
  in the docstring or kept and listed in a `unverified.md` appendix.

**WP-12.1 Proper conformal core** (one agent; needs 12.0)
- CV+ with stored fold models; plain split conformal when `n_cal >= 1000`;
  per-class (Mondrian) margins with the `n_k` minimum; optional `weights`.
- Replace the loops at `stages.py:2405-2442`, `:2480-2504`; update
  `bundle.py:1155-1204`, `:1528-1549`; extend the checksum at
  `bundle.py:933-945` to fold models.
- Done when: G1-G4 pass on simulation and the MAPIE differential matches.

**WP-12.2 Coverage report and slice flag** (one agent; needs 12.1)
- `coverage_report.json` on `C`/`F`; Clopper-Pearson per slice; Holm over
  the pre-fixed slice family; extend `SliceReport` (`slices.py:24`).
- Registered as a claim in the ledger before `C` is read; tier per F12 A5.
- Done when: G5 certificate written, `--check` fails if stale.

**WP-12.3 Shift weights** (one agent; needs 12.1; parallel with 12.2)
- Extend `AdversarialValidation` (`shift.py:26`) to output clipped odds and
  effective sample size; weighted quantile in `conformal.py`.
- Done when: G6 passes; ESS floor refusal tested.

**WP-12.4 Proper scores and forecast protocol** (one agent; parallel with 12.1-12.3)
- `proper_scores.py` (pinball, WIS, CRPS from quantile grid); standard MASE
  with in-sample scale; rename the same-rows variant; per-horizon calibrated
  intervals.
- Done when: our scores equal the official M4 Naive and Seasonal-naive
  benchmark scores (acceptance 8).

**WP-12.5 Hierarchical reconciliation** (one agent; needs 12.4)
- MinT with shrinkage covariance, bottom-up baseline, coherence check.
- Done when: G9.

**WP-12.6 Survival** (one agent; parallel)
- Covariate-dependent censoring model, truncated IPCW C-index and Brier,
  calibration at `tau`, `n_floored` reported; charter field for `tau`.
- Done when: G11(i) shows bias of the marginal version under covariate-
  dependent censoring and none for the new one; G11(ii) differential.

**WP-12.7 Ranking, multilabel, panel** (one agent; parallel)
- Query-bootstrap CI and reference-pinned NDCG (G10); per-label support rule
  and thresholds (G12); `unit_of_prediction` charter field and entity-level
  calibration (G13).
- Done when: G10, G12, G13.

**WP-12.8 Near-duplicate grouping and text/image encoders** (one agent; needs 12.0)
- Duplicate grouper (hash + embedding cosine), group split; one frozen image
  encoder and one text encoder behind the existing extras, licence recorded.
- Done when: G14, G15.

**WP-12.9 Gate wiring and task status** (one agent; last)
- `uncertainty_certify.py`, `uncertainty_certificates.json`,
  `task_gate_status.json`, test over `TaskType` (acceptance 11); tighten or
  replace `benchmarks/reliability.py:287`.

## Parallel lanes

```
12.0 --> 12.1 --> 12.2
   |        \--> 12.3
   |--> 12.8
12.4 --> 12.5            (12.4 can start before 12.0)
12.6, 12.7               (independent of the conformal lane)
all --> 12.9
```

## Exit gate

```bash
uv run pytest -q tests/unit tests/property tests/leakage -k "conformal or coverage or proper_score or ipcw or reconcil or neardup"
uv run python -m benchmarks.qualification --check uncertainty   # proposed flag
uv run python -m benchmarks.reliability --only interval_coverage
```

Numbers (at `R = 2000` replicates per world unless stated):
- G1: mean coverage within the derived band (about `[0.8991, 0.9029]` for
  `n_cal = 500`, `a = 0.1`); KS of per-replicate coverage against the Beta
  law with `p > 0.001`.
- G2-G4: coverage within `3*se` of nominal for each method, and the named
  baseline fails its bound (constant width in the highest-noise decile;
  pooled margin on the 2% class).
- G5: familywise false-flag `<= 0.0597`; recall lower bound `>= 0.80` at the
  certified slice size, else the slice check stays descriptive.
- G6: known-ratio coverage within `3*se`; unweighted shown lower.
- G8: official Naive and Seasonal-naive scores reproduced; then system
  interval versus best baseline reported (the claim "forecast `confirmed`"
  needs the upper limit below zero).
- G9-G13: differential tolerances `1e-6` to `1e-9` as in the feature file.
- G14: planted-duplicate recovery `>= 0.95` at false-merge `<= 0.01`.
- Replay bit-identical on Python 3.11 and 3.14 for every certificate.

## Evidence required

`uncertainty_certificates.json` with code hashes; `coverage_report.json` from
a seeded run on `C`; differential tables against MAPIE, scikit-survival, R;
the benchmark score tables for M4/M5 with corpus hashes; the `unverified.md`
appendix; the list of published misses (coverage claims that failed on `F`).

## Risks

- A gate passes on simulation and fails on real data. Mitigation: G7 is a
  recorded-miss policy; tiers follow held-out measurement, not the theorem.
- Fold-model storage inflates bundles by K times. Mitigation: size cap in the
  charter; fall back to split conformal on a calibration slice when over cap.
- Reference corpora unavailable (licence, hash). Mitigation: the type stays
  `unverified`; no gate is faked.
- MASE redefinition breaks comparison with earlier runs. Mitigation: both
  metrics stored, changelog entry.
- MinT covariance estimation unstable with many bottom series. Mitigation:
  shrinkage; bottom-up remains available and is the baseline.

## Explicitly out of scope

ACI and non-exchangeable conformal, conformal for survival or ranking,
fine-tuned vision/language models, detection/segmentation, click-model LTR,
mixed-effects models as a family, anomaly detection, any closed-loop
behaviour (Phase 13).
