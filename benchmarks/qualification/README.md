# Qualification verification harness

Measures the qualification instrument itself (feature F7, phase 1). It adds no
analysis for users; it supplies known-answer inputs, reference outputs and
certificates that decide whether a detector may gate a run.

```bash
uv run python -m benchmarks.qualification --smoke          # the Phase 1 gate
uv run python -m benchmarks.qualification --smoke --full   # nightly, longer
uv run python -m benchmarks.qualification --check          # certificates
```

| Module | Purpose |
| --- | --- |
| `inject.py` | Plants labelled defects (label flips, MCAR/MAR/MNAR missingness, sentinels, exact and near duplicates, outliers, target leakage, covariate shift) and returns a truth mask that matches the real mutation cell by cell. |
| `simulate.py` | Real feature matrices with planted effects or permuted outcomes (within groups or time blocks); empirical size, power and false-discovery rate with Monte Carlo intervals. |
| `oracles.py`, `make_oracles.py` | Frozen scipy reference outputs in `tests/fixtures/oracles/`, with tool versions, generator hash and content hash. Regenerate offline only. |
| `corpora.py`, `corpora.json` | Manifest and hash-verifying fetcher for labelled corpora. Every entry is a `candidate` until someone checks existence, licence and hash. |
| `certify.py`, `standins.py` | Plant, detect, certify. The stand-in detectors only exercise the machinery and must never gate. |
| `smoke.py` | The gate, including negative controls that must fail. |
| `f5.py`, `detector_certificates.json` | Certification of the F5 detectors in `scmle/data/detectors/` (missingness, label noise, duplicates, sentinels, outlier triage). Regenerate with `--write-f5 FILE`; `--check` fails if any certificate is stale or missing. |
| `make_f5_oracles.py` | Offline generator of the frozen cleanlab mask in `tests/fixtures/f5_oracles.json`. |
| `ingest_certify.py`, `ingest_worlds.py`, `ingest_checks.py`, `ingest_naive.py` | F9 ingestion: planted torn-read, fan-out, dropped-row, late-arrival, grain, join and entity worlds; certificates for the torn-read check and the extract reconciliation (`ingest_certificates.json`); floors on the other methods in `ingest_measurements.json`; and the row-by-row as-of reference the vectorised code is compared with. `--check ingest` fails when either file was measured on other code. |

## Limits

- F5 certificates use one corpus (scikit-learn breast cancer). Row detectors
  are judged per row, which understates uncertainty because rows of one table
  are not independent; the missingness detector is judged per replicate
  table. A detector gates only for the defect types whose outcome passed
  (recall lower bound 0.80, false-alarm upper bound 0.05, fixed in advance);
  the rest stay descriptive. See the `f5.py` docstring for the full caveats.
- Plants are synthetic, so recall measured on them is optimistic. Certificates
  are kept per defect type and are not a claim about real defects.
- Agreement with a reference shows agreement, not truth.
- The planted truth is conditional (a null column has a zero coefficient given
  the others). With correlated features a marginal test rejects such a column
  often; use a conditional test such as `simulate.coefficient_pvalues`.
- R reference outputs (`onlineFDR`, `knockoff`) are **not** generated: R is not
  available in CI. They are listed in `oracles.PENDING_R` and must be produced
  offline and frozen by a reviewer.
- No corpus is verified yet, so no real labelled corpus is used.
- Certificates are tied to a hash of the detector's own source. Anything it
  calls must be passed to `certificates.code_hash` too.
