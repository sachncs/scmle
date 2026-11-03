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
| `scmle charter init\|check\|sign PATH` | Create, validate or sign an analysis charter (see below). |

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
| `--charter FILE` | Govern the run by an analysis charter: a sealed confirmation set is split off before profiling. |
| `--exclude-models`, `--exclude-families` | Restrict the search. |
| `--output-dir`, `--seed` | Where runs go; reproducibility. |
| `--config FILE` | Read a complete project specification (the format of `config.json`). |

## Analysis charter

A charter fixes the question, the estimand (population, target, prediction
moment, intercurrent events, summary measure), the decision map, the set sizes
and the error budget before any data is profiled.

```bash
scmle charter init charter.json        # a template that fails `check` until filled in
scmle charter check charter.json --rows 5000   # validate; report whether the
                                       # confirmation set can support the promised power
SCMLE_SIGNING_KEY=... scmle charter sign charter.json   # add an HMAC-SHA256 signature
scmle run data.csv --target churn --charter charter.json
```

`check` verifies the charter hash and, when a key is configured, the
signature; a one-byte edit fails verification. The signature proves possession
of the key, not the signer's identity. If the confirmation set is too small to
support the promised power the run still completes: the scope becomes
`exploratory_unconfirmed` and `feasibility.json` reports the detectable effect.

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
