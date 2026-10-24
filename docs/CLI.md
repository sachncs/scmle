# Command line

The `scmle` command mirrors the Python API. Global option:
`--log-level {DEBUG,INFO,WARNING,ERROR}` (before the sub-command).

| Command | Purpose |
| --- | --- |
| `scmle plan DATA --target COL` | Profile and print the proposed plan. Trains nothing. |
| `scmle run DATA --target COL` | Execute the full workflow and write `runs/run_*`. |
| `scmle resume RUN_DIR` | Continue a cancelled, failed or crashed run, reusing finished trials. |
| `scmle catalog INPUT` | Catalog a local table (or raw bytes) into a JSON artifact. |
| `scmle predict BUNDLE DATA` | Score a file with a saved bundle. |
| `scmle serve BUNDLE` | Serve a bundle over HTTP. |
| `scmle export-onnx BUNDLE` | Export an ONNX graph (needs the `onnx` extra). |
| `scmle package BUNDLE` | Write a ready-to-run service folder with a Dockerfile. |

`plan` and `run` take the same options; `scmle run --help` lists them all.
The common ones:

| Option | Meaning |
| --- | --- |
| `--target COL [COL ...]` | Target column(s). Required. |
| `--task`, `--metric` | Override the resolved task or metric. |
| `--time-budget S`, `--workers N` | Search budget in seconds and parallel trials. |
| `--validation` | `auto`, `stratified`, `random`, `group`, `temporal`, `rolling`, and more. |
| `--group-column`, `--time-column`, `--weight-column` | Data structure. |
| `--exclude`, `--id-columns`, `--unavailable-at-prediction` | Keep columns out of the features. |
| `--leakage-policy {warn,block}` | `block` stops the run on suspicious features. |
| `--exclude-models`, `--exclude-families` | Restrict the search. |
| `--output-dir`, `--seed` | Where runs go; reproducibility. |
| `--config FILE` | Read a complete project specification (the format of `config.json`). |

## Examples

```bash
scmle plan data.csv --target churn
scmle run data.csv --target churn --workers 4 --time-budget 600
scmle resume runs/run_20261015_001
scmle catalog data.csv --label-column churn --output catalog.json
cat payload.bin | scmle catalog --raw-bytes
```

## Bundles: signed versus trusted

`predict`, `serve`, `export-onnx` and `package` load a bundle. Bundles are
pickle-based, so loading requires a valid signature (set `SCMLE_SIGNING_KEY`).
`--trusted` skips that for an unsigned bundle you produced yourself; never use
it on a bundle from elsewhere. `predict` also takes `--rename INCOMING=EXPECTED`,
`--case-insensitive` and `--strip-whitespace` for explicit column mapping.

`serve` is a minimal reference server (no authentication or TLS). Bind to
loopback or run it behind a gateway; see [`SECURITY.md`](SECURITY.md).
