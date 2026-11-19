# Command line

The `scmle` command mirrors the Python API. Global options (before the
sub-command): `--log-level {DEBUG,INFO,WARNING,ERROR}`, `--log-json` (the log as
one JSON object per line), `--trace` (OpenTelemetry spans over OTLP; see below)
and `--debug`, which logs every failure with its traceback, chain of causes and details.
Log lines carry the time, the level, the logger and the ids in scope as
`[run=... request=...]` (the run id during a run, the request id in the
server; the generated service writes JSON when `SCMLE_LOG_JSON` is set).

| Exit code | Meaning |
| --- | --- |
| 0 | Done. |
| 1 | Failed: a handled error (bad data, a missing file, an invalid configuration). |
| 2 | Blocked: a gate refused the run (for example a leakage suspicion under `--leakage block`, or a locked test set opened twice). |
| 64 | Usage: the command line is wrong. |
| 70 | Internal: an unexpected error, which is a defect; the traceback is always logged. |

| Command | Purpose |
| --- | --- |
| `scmle plan DATA --target COL` | Profile and print the proposed plan. Trains nothing. |
| `scmle run DATA --target COL` | Execute the full workflow and write `runs/run_*`. |
| `scmle resume RUN_DIR` | Continue a cancelled, failed or crashed run, reusing finished trials. A run whose test set was opened, or whose custody journal (`runs/custody/<run>.jsonl`, outside the run directory) is missing or edited, is refused; deleting `evaluation.json` does not change that. |
| `scmle catalog INPUT` | Catalog a local table (or raw bytes) into a JSON artifact. |
| `scmle predict BUNDLE DATA` | Score a file with a saved bundle. |
| `scmle batch BUNDLE DATA` | Score a file through the same code path as the online service, `--chunk-rows` rows at a time; writes `--output` and `<output>.manifest.json` (model version, artifact digest, row counts). An interrupted run resumes when run again (`--restart` starts over). `--on-null` and `--on-unseen` accept `reject`. |
| `scmle serve BUNDLE` | Serve a bundle over HTTP: `/livez`, `/readyz`, `/predict` (see [SERVICE_SLO](SERVICE_SLO.md)). SIGHUP reloads the bundle, SIGTERM drains and stops (see below). |
| `scmle export-onnx BUNDLE` | Export an ONNX graph (needs the `onnx` extra). |
| `scmle package BUNDLE` | Write a hardened service folder: pinned-base Dockerfile, locked requirements, SBOM, signed provenance, `build.sh`. Sign with `SCMLE_SIGNING_KEY`. |
| `scmle charter init\|check\|sign PATH` | Create, validate or sign an analysis charter (see below). |
| `scmle claims list\|show\|verify RUN_DIR` | List the claims of a charter run, show one with its hypothesis and checks, or verify the run's ledger and claims (see below). `verify` exits 1 when any step fails. |
| `scmle frame REQUEST --source S --data X --output DIR` | Draft an unsigned charter, a cost-model record and the questions only a person can answer, from a plain request. Exit 2 when refused. See [ARTIFACTS](ARTIFACTS.md). |
| `scmle trail build\|verify\|show` | Join the charter, framing, cost model, claim ledger, registry, release state and deployment ledger of one model into a hash-chained, signed trail; verify a trail folder against its own records; print it. `build` and `verify` exit 1 on any finding. See [TRAIL](TRAIL.md). |
| `scmle records upgrade FOLDER [--dry-run]` | Rewrite the stored records of a folder at the current schema version, keeping a `.bak` of each file; exit 1, with every file restored, if a seal, signature or chain fails to verify (see below). `estimate`, `cluster`, `forget`, `retire`, `delete` and `store` are described at the end of this page. |

`plan` and `run` take the same options; `scmle run --help` lists them all.
The common ones:

| Option | Meaning |
| --- | --- |
| `--target COL [COL ...]` | Target column(s). Required. |
| `--task`, `--metric` | Override the resolved task or metric. |
| `--budget S`, `--workers N` | Search budget in seconds and parallel trials. |
| `--validation` | `auto`, `stratified`, `random`, `group`, `temporal`, `rolling`, and more. |
| `--group`, `--time`, `--weight` | Data structure. |
| `--exclude`, `--identifiers`, `--unavailable` | Keep columns out of the features. |
| `--leakage {warn,block}` | Default `warn`: the run finishes, logs a warning and labels the score provisional. `block` (opt-in) stops the run on suspicious features. See [LEAKAGE](LEAKAGE.md). |
| `--charter FILE` | Govern the run by an analysis charter: a sealed confirmation set is split off before profiling. |
| `--exclude-models`, `--exclude-families` | Restrict the search. |
| `--output-dir`, `--seed` | Where runs go; reproducibility. |
| `--config FILE` | Read a complete project specification (the format of `config.json`). |

The options renamed to single words in this release (for example
`--time-budget` to `--budget`, `--group-column` to `--group`,
`--horizon-rows` to `--rolling-horizon`) still accept their old spelling
for one release; each prints a deprecation warning once.

## Following along

The examples on this page run as written, top to bottom, in an empty folder.
They use a small table with a binary target `churn` and a demonstration
signing key: `charter sign`, `claims verify`, `records upgrade` and the
commands that load or write a bundle read it from `SCMLE_SIGNING_KEY`. Save
this as `make_data.py`:

```python
import numpy
import pandas

rng = numpy.random.default_rng(0)
rows = 800
data = pandas.DataFrame(
    {
        "customer_id": numpy.arange(rows),
        "tenure": rng.integers(1, 72, rows),
        "spend": rng.normal(60, 20, rows).round(2),
        "plan": rng.choice(["basic", "plus", "pro"], rows),
    }
)
risk = -0.03 * data.tenure + 0.02 * data.spend + 0.8 * (data.plan == "basic")
data["churn"] = (risk + rng.normal(scale=0.7, size=rows) > 0.5).astype(int)
data.to_csv("data.csv", index=False)
```

```bash
python make_data.py
export SCMLE_SIGNING_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
```

## Analysis charter

A charter fixes the question, the estimand (population, target, prediction
moment, intercurrent events, summary measure), the decision map, the set sizes
and the error budget before any data is profiled.

```bash
scmle charter init charter.json        # a template that fails `check` until filled in
```

Fill in the question, the population, the target, the prediction moment, one
intercurrent event with its strategy, the unit and the deployment population,
with an editor, or save this as `fill_charter.py`:

```python
import json

with open("charter.json") as file:
    charter = json.load(file)
charter["question"] = "Which customer traits are linked to churn?"
charter["estimand"].update(
    population="every customer of data.csv",
    target="churn",
    moment="the start of the month",
    intercurrent=[{"event": "none", "strategy": "not applicable"}],
    summary="rank correlation with churn",
)
charter["unit"] = "customer"
charter["population"] = "every customer of data.csv"
with open("charter.json", "w") as file:
    json.dump(charter, file, indent=2)
```

```bash
python fill_charter.py
scmle charter check charter.json --rows 600    # validate; report whether the
                                               # confirmation set can support the promised power
scmle charter sign charter.json                # add an HMAC-SHA256 signature (SCMLE_SIGNING_KEY,
                                               # or --key-file FILE)
scmle run data.csv --target churn --identifiers customer_id --charter charter.json \
  --budget 120 --output-dir charter_runs
```

`check` verifies the charter hash and, when a key is configured, the
signature; a one-byte edit fails verification. The signature proves possession
of the key, not the signer's identity. If the confirmation set is too small to
support the promised power the run still completes: the scope becomes
`exploratory_unconfirmed` and `feasibility.json` reports the detectable effect.

Format 3 charters may carry a `rules` object that fixes, before data, the
online rule of the ledger (`ledger`), the confirmation rule
(`confirmation`), the failure policy (`failure`), whether `fragile` survivors
are withdrawn before `C` opens (`stability`) and the stability thresholds
(`agreement`, `band`, `exclusion`). The charter hash and signature cover them. A
charter without `rules` is hashed, signed and loaded exactly as before and the
run uses the defaults, reporting `"rules": "not chartered"`. A run
specification that names a different non-default rule than the charter is
refused. The filter acts only if the stability rule has a current passing
certificate for the chartered thresholds; otherwise it is recorded as
requested and not applied, and the verdicts stay warnings.

## Claims of a charter run

```bash
scmle claims list charter_runs/run_* --tier confirmed
scmle claims list charter_runs/run_* --json
scmle claims show charter_runs/run_* h000002
scmle claims verify charter_runs/run_* --json
```

`--tier` keeps the claims of one tier (`confirmed` or `unverified`) and
`--json` prints JSON. `verify` reads the key from `SCMLE_SIGNING_KEY`, or from
a file with `--key-file`.

`verify` reads the run directory only. It checks that the ledger's hash chain
verifies and every recorded decision replays, that `confirmed_claims.json`
names the ledger head, the survivor hash, the outcome and the confirmed set of
the sealed ledger, that the multiplicity rule replayed on the recorded p-values
on `C` gives the recorded adjusted p-values and tiers, that the second-stage
family (survivors minus withdrawn) and its hash are consistent and no
withdrawn survivor was tested on `C`, that the custody log verifies, froze the
selection before `C` opened and opened `C` at most once, that the charter
verifies (signature checked when a key is configured) and, if it fixes rules,
that the ledger and claims used them, that the claims registered by hash
(baseline, causal primary) are stated by the sealed ledger and their report
files hash to the values it holds, and that `negative_control.json` is the
control the ledger recorded. A step whose input is absent is reported
`skipped`, never `pass`. This proves the artifacts agree; it does not
recompute statistics from data, and a party who can rewrite every file
consistently can pass it (custody is tamper-evident, not adversary-proof).

## Unattended operation

`scmle operate` runs the controller of a deployment that `scmle loop init`
created. The forms of its commands are below; [Autonomous
operation](AUTONOMY.md) runs `grant`, `status`, `revoke` and `halt` on a
deployment, with the envelope a person signs.

```bash synopsis
scmle operate tick <deploy> --inbox <inbox> [--keys-file keys.json]
scmle operate status <deploy>
scmle operate grant <deploy> --envelope envelope.json --key-file person.key
scmle operate revoke <deploy> <envelope-id> --reason <text>
scmle operate halt <deploy> --reason <text>
```

`tick` applies each new file of `inbox/batches` and `inbox/labels` once, steps
the loop and prints what happened as JSON. Every promotion still needs a
person's approval.

## Business trail

The repository ships the records of one deployed model, from a real run of
the Titanic extract, with demonstration keys. Rebuilt into a trail and
checked:

```bash
git clone https://github.com/sachncs/scmle
sample=scmle/tests/fixtures/trail_sample
scmle trail build trail/ --run $sample/trail/retained/run \
  --framing $sample/trail/retained/framing --registry $sample/trail/retained/registry \
  --deployment $sample/trail/retained/deployment --inputs $sample/trail/inputs.json \
  --keys $sample/keys.json
scmle trail verify trail/ --keys $sample/keys.json --json
scmle trail show trail/
```

`--keys` is a JSON file mapping `analyst`, `stakeholder` and `reviewer` to the
keys that verify their signatures; without it only `SCMLE_SIGNING_KEY` is read,
as the analyst key, and the other signatures are reported as unverifiable.
Findings are `missing_link`, `broken_chain`, `unsigned_step`,
`unapproved_step`, `outcome_outside_expectation` and four more, listed in
[TRAIL](TRAIL.md).

## Release evidence and the runtime

These commands act on a service deployed over HTTPS, a release manifest with
the evidence it names, and a model registry, none of which this page creates;
their forms are below.

```bash synopsis
scmle deploy live --url https://staging.example --token-file token --environment staging \
  --commit <sha> --container-digest sha256:... --model-digest sha256:... --output checks.json
scmle deploy gate {admit,certify} manifest.json --key-file key [--output record.json]
scmle deploy verify-record record.json --key-file key
scmle deploy model-digest <registry> --model churn --version 3
scmle deploy reconcile <registry> --model churn --environment production --observation observed.json [--ledger deployment_ledger.jsonl]
```

`live` runs the health, schema, inference and security checks and exits 1 if
any fails. `gate` reads a release manifest and the evidence it names and exits
2 with every reason when the release is refused; `certify` writes the signed
deployment record.
Keys are read from files, never from the command line. `reconcile` exits 1
when the registry, the ledger and the observation disagree. The observation
file is written by the runtime adapter you provide.

## Examples

```bash
scmle plan data.csv --target churn
scmle run data.csv --target churn --workers 4 --budget 120
scmle resume runs/run_*                 # the run folder, for example runs/run_20261015_001
echo "exit code: $?"
scmle catalog data.csv --label-column churn --output catalog.json
head -c 300 data.csv > payload.bin
cat payload.bin | scmle catalog --raw-bytes
```

`resume` continues a cancelled, failed or crashed run from its finished
trials. The run here finished, so it is refused with exit code 1 and
`config error: run already completed; nothing to resume`.

## Bundles: signed versus trusted

When `SCMLE_ENV=production`, `--trusted` (and `SCMLE_TRUSTED=1`) is refused
for every command; sign the bundle and supply `SCMLE_SIGNING_KEY` instead.


`predict`, `serve`, `export-onnx` and `package` load a bundle. Bundles are
pickle-based, so loading requires a valid signature (set `SCMLE_SIGNING_KEY`).
`--trusted` skips that for an unsigned bundle you produced yourself; do not use
it on a bundle from elsewhere. It never skips a signature: a signed bundle with
no key set to verify it is refused even with `--trusted`. `predict` also takes `--rename INCOMING=EXPECTED`,
`--case-insensitive` and `--strip-whitespace` for explicit column mapping.

`serve` is a minimal reference server without TLS. On loopback it needs no
credentials; on any other address it refuses to start unless the `SCMLE_AUTH_*`
variables of [`TENANT_ISOLATION.md`](TENANT_ISOLATION.md) configure token
verification. See [`SECURITY.md`](SECURITY.md).

## Stored records

```bash
scmle records upgrade charter_runs/run_* --dry-run
scmle records upgrade charter_runs/run_*
```

Records written by an older scmle load without change (each record kind
migrates on read; see [`RECORD_KEYS.md`](RECORD_KEYS.md)). `records upgrade`
rewrites them on disk at the current schema version: the charter, the split
and dataset manifests, the feasibility, error budget and cost model records
and the claims file, in the folder and its sub-folders. It first verifies
the stored bytes (each record loads and its seal holds, the charter's hash
and signature verify with `--key-file FILE` or `SCMLE_SIGNING_KEY`, and a charter
run passes `claims verify`); if that fails it writes nothing and exits 1.
It then keeps each old file as `NAME.bak`, writes the new one and verifies
again; if that fails, every file is restored from its `.bak` and it exits 1.
`--dry-run` lists the files that would change and writes nothing.
Hash-chained logs, `negative_control.json` (its digest is in the ledger) and
the files of a sealed trail are left as stored.

## Serving lifecycle and tracing

```bash synopsis
scmle serve <bundle> [--drain-seconds S] [--signals-file signals.jsonl] \
  [--signals-interval 60] [--signals-window 300]
kill -HUP <pid>     # reload the bundle directory; the old model serves until the new one is loaded
kill -TERM <pid>    # stop accepting connections, drain the requests in flight, exit 0
scmle --trace serve <bundle>
```

`serve` runs the same loop as the packaged service. SIGTERM or SIGINT stops
accepting connections and lets the requests in flight finish for
`--drain-seconds` (by default the longest an admitted request can take: the
request timeout to read it, the scoring timeout and the request timeout to
answer). SIGHUP verifies and loads the bundle directory again and swaps it
in without downtime; a reload that fails is logged and the previous bundle
keeps serving. Every `--signals-interval` seconds the operations signals
over the last `--signals-window` seconds, drift judged against the served
bundle's training reference, are appended to `--signals-file` as one JSON
line (`at`, `signals`), or written to the log; they are written once more
on shutdown.

`--trace`, or `SCMLE_TRACING` in the environment, exports OpenTelemetry
spans over OTLP/HTTP, configured by the standard `OTEL_EXPORTER_OTLP_*`
variables: `request` (one HTTP request, joined to the caller's trace by its
`traceparent` header), `predict` (one scoring call), `batch` (one scored
file) and `stage <name>` (one pipeline stage). It needs the `tracing` extra;
without it the degraded path is reported and every span is a no-op.

## Batch scoring

```bash
scmle batch runs/run_*/model data.csv --output predictions.csv --chunk-rows 200
scmle batch runs/run_*/model data.csv --output predictions.csv --restart
```

`batch` reads and scores `--chunk-rows` rows at a time (default 10,000), so
memory holds one chunk however large the file; delimited text is read
record by record. Each scored chunk is appended to `<output>.partial` and
`<output>.progress.json` records the rows and bytes written. Running the
same command on the same input, bundle and policies after an interruption
continues after the last recorded chunk; `--restart` scores from the first
row. The finished file replaces `<output>` atomically and the progress
journal is removed.

## Parameter estimation

The declaration is a JSON object with `estimand` and `policy`, in the
layout a run stores (`schema_version` 2, the default) or the version-1
layout; a run's own `parameter_estimation_declaration.json` is accepted as
it is. For example, a slope on a table where it is 2 (save this as
`make_estimation.py`):

```python
import json

import numpy
import pandas

rng = numpy.random.default_rng(21)
signal = rng.normal(size=400)
outcome = 1.25 + 2.0 * signal + rng.normal(scale=0.5, size=400)
pandas.DataFrame({"signal": signal, "outcome": outcome}).to_csv(
    "signal.csv", index=False
)
declaration = {
    "estimand": {
        "name": "slope",
        "outcome": "outcome",
        "population": "the rows of signal.csv",
        "likelihood": "gaussian",
        "features": ["signal"],
        "assumptions": ["the conditional mean is linear"],
    },
    "policy": {"estimator": "linear"},
}
with open("declaration.json", "w") as file:
    json.dump(declaration, file, indent=2)
```

```bash
python make_estimation.py
scmle estimate signal.csv --estimation declaration.json --output-dir estimates
scmle run signal.csv --target outcome --estimation declaration.json --budget 120 \
  --output-dir estimation_runs
```

`estimate` trains no model: it writes `parameter_estimation.json` to
`--output-dir` (default `estimates`) and logs each parameter's estimate,
interval and p-value (`--treatment COL` names a treatment column). When the declaration cannot be estimated on the data
(a missing or inconsistent column, missing values under the `reject` policy,
a likelihood the estimator does not fit, or the estimator's own refusal) it
writes `parameter_estimation_refusal.json` with the reason and exits 2 (gate
`estimate.refusal`). `run --estimation` makes the same estimate beside model
selection and keeps it in the run directory; model selection never reads it.

## Fairness mitigation

`plan` and `run` take `--mitigation {none,reweighing,group_thresholds}`,
applied to the first `--sensitive` column. `reweighing` weights the
development rows so that group and label are independent (binary,
multiclass and regression; a regression outcome is balanced by its
equal-count bins); `group_thresholds` sets a decision threshold per group
(binary only). A `fairness` section in `--config` takes precedence.

## Clustering

```bash
scmle cluster data.csv --output segments/ --counts 2 3 4 \
  --methods kmeans gaussian_mixture agglomerative --rounds 3 --seed 0 \
  --exclude customer_id --truth churn
scmle predict segments/ data.csv --output segments.csv
```

`cluster` uses the numeric columns, less `--exclude` (repeatable) and
`--truth`, as features. It tries each method in `--methods` (default all
three) at each count in `--counts` (each at least 2; default 2 to 8), scores
each candidate by silhouette, Davies-Bouldin and Calinski-Harabasz and by its
stability over `--rounds` bootstrap refits (default 10), and writes a bundle
of task `clustering` with `clustering.json` beside it into `--output`, which
must be new or empty. `--truth` names a column of known groups to measure
agreement against (it is never a feature; here the churn label stands in for
one). `scmle predict` and `Bundle.predict` return cluster labels. The bundle is signed when `SCMLE_SIGNING_KEY` is set and otherwise
loads only with `--trusted`. Clustering is not qualified; see
[`BREADTH.md`](BREADTH.md).

## Forgetting, retiring and deleting

```bash
scmle forget dataset runs/ --data data.csv --reason "ticket 118" --dry-run
scmle forget dataset runs/ --data data.csv --reason "ticket 118"
scmle forget verify receipts/forget-*.json
printf '3\n17\n' > ids.txt
scmle forget subject charter_runs/ --data data.csv --column customer_id \
  --values-file ids.txt --reason "request 2026-118"
scmle retire segments/ --reason "superseded"
scmle delete estimation_runs/run_* --reason "retention expired"
```

The forms not run above:

```bash synopsis
scmle forget dataset <folder>... --digest <sha256> --reason <text>
scmle forget subject <folder>... --data <file> --column <col> --value <id> \
  [--earlier <digest>] [--retrain] --reason <text>
scmle forget verify <receipt> [--key-file key]
scmle retire <registry> --model churn --version 3 --reason "drift"
```

`forget dataset` searches the given folders for every run, model bundle,
sealed trail and `scmle batch` prediction log derived from a dataset, named
by its file (`--data`) or by any digest a run's `dataset_manifest.json`
records (`--digest`, repeatable). It removes what was derived from the data,
keeps each run's custody records (listing their digests) and retires the
models. `forget subject` first purges a data subject's rows from the source
file (CSV, TSV, Parquet or JSON Lines, rewritten atomically), matched by
`--column` against `--value` (repeatable) or `--values-file` (one per line,
kept off the shell); `--earlier` names digests of earlier versions of the
source, and `--retrain` trains each affected run again on the purged
source. Identifiers are written only as SHA-256 hashes. `retire` retires one
bundle directory, or a registered version with `--model` and `--version`:
the bundle is refused from then on (`RetiredError`, exit 2) and a registered
version's release state becomes `retired`. `delete` deletes runs or project
folders and keeps their custody records.

Every action needs `--reason`. `--dry-run` prints the plan and changes
nothing. Otherwise the plan is written under `--receipts` (default
`receipts`), each finished step is journaled, and a signed receipt is
written after the last step; running the same command again after an
interruption resumes from the journal. The key is read from `--key-file` or
`SCMLE_SIGNING_KEY`; without one nothing is changed. `forget verify` exits 1
when a receipt's signature does not verify. See
[`ARTIFACTS.md`](ARTIFACTS.md) for the files.

## Adopting a store into a tenant

```bash
mkdir -p store
scmle store adopt --root store/ --tenant acme
```

Moves every object of a local store that lies outside the tenant
namespaces into `tenants/<tenant>/`: the migration of a store written
before tenants were namespaced (see
[`TENANT_ISOLATION.md`](TENANT_ISOLATION.md)). Each object is copied with
`if_absent` and its original removed only if unchanged, so running the
command again after an interruption finishes the move; it prints each key
it moved, and nothing once the store is adopted. It exits 1 when a
different object already sits at a destination or the tenant name is
invalid.
