# Local environment

Everything here runs on one machine with no cloud account. It is the portable
core that the AWS and GCP phases build on: the same bundle, service and
contracts, run locally first.

## One command

```bash
make demo        # native: needs only the project's Python environment
```

It runs, in order, and fails if any step does not hold:

1. **train and evaluate** a good model (`deploy/local/data/train.csv`) and a
   bad release (`bad_release_train.csv`, the same rows with the label
   relationship reversed);
2. **register and package** both, promote the good one with a signed approval,
   and write the hardened service folder (signed bundle, locked requirements,
   SBOM, signed provenance);
3. **object store round trip**: upload the folder to the mock object store and
   fetch it back by digest, refusing any file whose digest differs;
4. **serve under production rules**: `SCMLE_ENV=production`, no
   `SCMLE_TRUSTED`; the service verifies, loads and warms up, and `/readyz`
   turns 200;
5. **predict** online and in batch and compare both with the bundle;
6. **golden-record parity** against the hand-written reference model;
7. **corrupted and unsigned copies never become ready** (`/livez` 200,
   `/readyz` 503, `/predict` 503);
8. **monitor and rollback drill**: promote the bad release, monitor it on
   fresh labelled data (`fresh.csv`), roll back to the good one with a signed
   approval, and serve the restored model;
9. **measure and ship metrics** to the mock metrics sink (startup, latency,
   memory).

The transcript is printed and saved as `runs/local-demo/transcript-run.json`.
Run it from the repository root (the data and golden files are read from
`deploy/local/`).

## Explicit files

Nothing is generated at run time except the models.

| File | What it is |
| --- | --- |
| `deploy/local/data/*.csv` | The four datasets. `scripts/make_demo_data.py` made them with fixed seeds; `--check` proves they still match. |
| `deploy/local/golden/` | The hand-written reference model, golden records, closed-form expected outputs and declared tolerances ([`SERVICE_SLO.md`](SERVICE_SLO.md)). |
| `deploy/local/demo.env` | The demo signing key. It is a published fixture, not a secret, and protects nothing. |
| `compose.yaml` | The API, a mock object store, a mock metrics sink and a test runner, plus the builder and the golden service. |
| `deploy/Dockerfile`, `.dockerignore` | The local image: the repository's source on hash-locked dependencies. |
| `deploy/requirements.lock` | `uv export` of `uv.lock` with hashes. `make lock` rewrites it, `make lock-check` fails if stale. |
| `deploy/vuln-policy.json` | The vulnerability gate policy. |
| `scmle/deploy/mocks.py` | The mock object store and metrics sink (standard library only). |

## In containers

```bash
make demo-compose                      # linux/amd64
make demo-compose PLATFORM=linux/arm64
```

`make image` resolves the base image tag to the digest of its multi-architecture
index (`python -m scmle.deploy.base_image resolve`, which needs network access
to the registry), writes `deploy/base-image.lock.json`, and builds with that
digest. The Dockerfile's default is a placeholder digest that the build
refuses, so an unresolved build cannot succeed by accident. Because the digest
is the index digest, amd64 and arm64 builds share one pinned base; the
dependency lock carries wheel hashes for both.

In compose the services run as user 10001 on a read-only root filesystem with
all capabilities dropped, `no-new-privileges`, memory and CPU limits, a `tmpfs`
for `/tmp`, and the API under `SCMLE_ENV=production`. The builder trains and
packages onto a shared volume; the test runner then calls the API over the
network, checks golden parity, runs the rollback drill and ships metrics.

The compose API runs the generated folder with this repository's image. The
hardened image that `scmle package` describes (its own Dockerfile, SBOM, scan
and attestation) is built separately by the folder's `build.sh`.

## What has and has not been run

Recorded plainly, because the point of this file is not to overclaim:

* Run on a developer machine: `make demo` end to end, the unit and integration
  tests, the parsing of `compose.yaml` and `docker compose config` (client
  side validation).
* Docker was not run: the Docker CLI was installed but its daemon was not
  running, so no image was built and no container started. The compose file,
  the Dockerfiles, `build.sh` and `make demo-compose` are therefore untested
  as a whole. Treat the first container run as the first test of them.
* Not run: the registry lookup of the base digest, the vulnerability scan
  against a live database, the image attestation of a real build, the arm64
  build, and the two-build digest comparison. Image reproducibility is not
  claimed.
* Nothing here touches AWS or GCP.

## Troubleshooting

* `refusing to load an unauthenticated bundle`: the service runs under
  production rules. Package with a signing key and give the service the same
  key in `SCMLE_SIGNING_KEY`.
* `BASE_IMAGE is the placeholder`: build through `make image`, not a bare
  `docker build`.
* `/readyz` stays 503: read the `reason` in its body; the service is alive on
  purpose so that you can.
