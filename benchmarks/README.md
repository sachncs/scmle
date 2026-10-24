# Benchmarks

Reproducible comparisons of the full SCMLE workflow against an independent
reference: a plain scikit-learn `HistGradientBoosting` model fitted on the
**same development rows** and scored on the **same locked test rows**. The
point is an honest check, so cases where SCMLE is behind are reported as is.
Reading guide and published tables: [`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md).

## Files

| File | Role |
| --- | --- |
| `run_benchmarks.py` | Entry point. Runs the workflow per dataset and writes a results file. |
| `datasets.py` | Dataset registry and the `SUITES` (`core`, `extended` OpenML, `huggingface`, `all`). Loaders cache downloads; `--offline` keeps only bundled data. |
| `reference.py` | The plain `HistGradientBoosting` reference. |
| `intervals.py` | Student-t intervals and win / tie / behind verdicts across seeds. |
| `check_regression.py` | CI gate: fails if a result misses its bound in `baseline.json`. |
| `baseline.json` | Per-dataset metric and bound the gate enforces (the four bundled datasets). |
| `reliability.py` | Nine trust checks with ground truth: determinism, replay, no false skill on noise, leak detection, bundle round-trip, tamper detection, interval coverage, drift detection, failure isolation. Exit status gates a release. |
| `real_tasks.py` | Newer task types (survival, uplift, ranking, forecasting, ...) on public data, each against a simple baseline. |
| `results.json` | Committed output of the core suite, quoted in the root README. |
| `results_seeds.json` | Committed 3-seed run with intervals. |
| `results_real_tasks.json` | Committed output of `real_tasks.py`. |
| `sota/` | Pre-registered comparison against external AutoML systems: [`PROTOCOL.md`](sota/PROTOCOL.md), the pinned task table and the seeded development / hold-out split (`split.py`). No results yet. |
| `results_reliability.json` | Committed output of `reliability.py`. |

## Run

```bash
# What CI runs: bundled datasets only, no network.
uv run python -m benchmarks.run_benchmarks --offline --seeds 5 \
    --budget 120 --workers 2 --output results.json
uv run python -m benchmarks.check_regression results.json

# Wider suites (downloads data; needs the `benchmarks` extra for the Hub).
uv run python -m benchmarks.run_benchmarks --suite extended --budget 240
uv run python -m benchmarks.run_benchmarks --suite huggingface --budget 240

# Reliability: does the package keep its promises? (no downloads, minutes)
uv run python -m benchmarks.reliability

# Newer task types.
uv run python -m benchmarks.real_tasks --budget 300
```

Pass `--only NAME ...` to run a subset. Write to `results.json` (untracked
scratch) rather than `benchmarks/results.json` unless you are deliberately
refreshing the committed evidence, and record the machine and budget when you do.

## Reading a result

* `test` is the locked-test score of the selected candidate; `reference_test`
  is the plain baseline. Lower is better for `log_loss` and `rmse`.
* With `--seeds N` each seed changes the split and the models. A win needs the
  95% interval of the advantage to exclude zero; otherwise the verdict is `tie`.
* Single-seed differences in the fourth digit are noise.
* `machine` and `software` fields in each record identify the hardware and
  interpreter, so numbers from different machines are not compared blindly.
