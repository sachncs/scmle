- C3 reference clauses: `benchmarks/evidence/statistics.py` compares product
  statistics with independent implementations, within `reference_rel_tol`
  (1e-8; error `|product - reference| / max(|reference|, 1)`). Reported
  statistics: every metric a run reports (`metrics.CATALOG.report_names`,
  all ten tasks, through `Repertoire.evaluate`) against textbook formulas
  written here, on real tables bundled with scikit-learn (breast cancer,
  wine, digits, diabetes, linnerud) and simulated worlds with ties and an
  absent ordinal level. Gate-deciding statistics (`DECIDING`, every other
  `gates.toml` gate classified in `PREDICATES`) need two references beside
  the product: the leakage screen's statistics, permutation p-values
  (recounted on the screen's own permutation stream), Benjamini-Yekutieli
  step and lower bounds; the missingness test (chi-square and Fisher);
  Clopper-Pearson limits; the thresholdout terms; `MIN_TEST_CLASS_ROWS`;
  the ONNX tolerance and row count; the p95 latency of
  `resources.measure` that `select.serving_limits` and `select.pareto_front`
  decide on, replayed through the product on tied, one-row and two-row
  timing vectors and a recorded real run (type-7 quantile by hand and
  `scipy.stats.mstats.mquantiles`). On the current code 1,145 gate values
  agree to 1.6e-13. One reported value disagrees: `qwk`
  (`scmle/evaluation/ordinal.py`) weights misses by the position of a level
  among the levels present, not by its rank, so a level absent from the
  scored rows shrinks the distances (0.710 against 0.695). Not compared yet,
  and listed in the document's `residue`: the screen's encoding and
  duplication p-values, the paired margin of a skill suspect, the ONNX
  probability gap, metric intervals and the diagnostic reports.
  `python -m benchmarks.evidence.statistics --show [--statistic NAME]`,
  `--write`, `--check`.
