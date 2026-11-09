# Service contract and objectives

What the generated prediction service promises, and what it does not. The
service is the reference HTTP adapter in `scmle/inference/serving.py` (tabular)
and `scmle/inference/modality_serving.py` (signed modality packages). It uses
only the standard library. Authentication, scopes, per-caller rate limits and
audit are in `scmle.security` ([`TENANT_ISOLATION.md`](TENANT_ISOLATION.md));
TLS belongs to the ingress.

## Endpoints

| Path | Meaning |
| --- | --- |
| `GET /livez` | 200 while the process runs. Never depends on the model. |
| `GET /readyz` | 200 only after **verify, load and warm-up** finished, in that order. Otherwise 503 with the phase, the steps done and, after a failure, the reason. A bundle that is unsigned, tampered with or the wrong digest leaves the service alive and never ready. |
| `GET /health` | Status, model version, model checksum, artifact digest (503 until ready). Kept for existing probes. |
| `GET /schema` | Input schema, `schema_digest`, request and response schemas, limits and policies (503 until ready). |
| `POST /predict`, `POST /v1/predict` | Score records. Request schema version 1. |

`/livez` is for the orchestrator's restart decision, `/readyz` for its routing
decision. Do not point both at the same path.

## Authentication

With a gateway configured (required beyond loopback) every path except
`/livez` and `/readyz` needs `Authorization: Bearer <JWT>`: 401 without a
valid token (with `WWW-Authenticate`), 403 when the scope or tenant is wrong,
429 with `Retry-After` over the per-subject rate or after repeated failures.
The body is a fixed message and never says which check failed. Scopes:
`model:read` for `/health` and `/schema`, `model:predict` for `/predict`.
Details and limits: [`TENANT_ISOLATION.md`](TENANT_ISOLATION.md).

## Schema version 1

Request: `{"records": [{...}], "schema_version": "1" (optional),
"expected_schema_digest": "sha256:..." (optional)}`. Unknown fields are
refused. Every record must carry every required feature; a feature that is
unknown for a record is sent as `null`, never omitted.

Response (200): `schema_version`, `request_id`, `model_version`,
`artifact_digest`, `schema_digest`, `record_count`, `predictions`, `classes`
and `probabilities` (classifiers), `schema_valid`, `diagnostics` and `rows`
(one entry per record). `artifact_digest` is the SHA-256 of the canonical JSON
of the bundle manifest, which lists the checksum of every file, so it names
the exact bundle that answered. Errors carry `schema_version`, `request_id`,
`error` and `category`. The machine-readable schemas are in
`scmle.contracts.serving` and are served at `/schema`. A change that removes
or retypes a field needs a new schema version; the service refuses a
`schema_version` it does not serve with 409 `schema_change`.

## Nulls, unseen categories and schema changes

| Input | Behaviour |
| --- | --- |
| Null in a feature | Default policy `impute`: the fitted imputers handle it and the row's `rows[i].missing` lists the column. Policy `reject` (`--on-null reject`, `on_null="reject"`): 422 `data_policy`. |
| Category never seen in training | Default `report`: scored as the encoder treats unknown values and listed in `rows[i].unseen` and `diagnostics.unseen_categories`. Policy `reject`: 422 `data_policy`. |
| Extra column | Ignored and listed in `diagnostics.ignored_columns`. |
| Required key absent in some record | 422 `schema` with `details.missing` and the record positions. |
| Required column absent | 422 `schema`. |
| Non-numeric value in a numeric column | 422 `schema`. |
| Caller expects another model schema | The caller sends `expected_schema_digest`; a mismatch is 409 `schema_change` with both digests. |
| Unknown `schema_version` | 409 `schema_change`. |

## Limits

| Limit | Default | Over the limit |
| --- | --- | --- |
| Request body | 16 MiB | 413 |
| Records per request | 10,000 | 413 |
| Requests in flight | 8 | 429 with `Retry-After: 1`; the request is not queued |
| Scoring time | 30 s | 504; the caller is answered, the computation itself cannot be cancelled and holds its worker until it returns |
| Request I/O | 10 s | connection closed |

Scoring is serialized by a lock because the bundle may update its series
history, so throughput on one process is that of one scoring thread; scale by
adding replicas, not threads.

## Batch

`scmle batch BUNDLE DATA` scores a file through `Server.respond_frame`,
the same function the online path calls: the same validation, preprocessing,
model call and policies. The output CSV is written atomically with
`<output>.manifest.json` (model version, artifact digest, schema digest, row
counts, output digest).

## Objectives

Declared targets for one replica of the container with the limits in the
deployment spec (0.5 CPU request, 1 CPU limit, 512 MiB memory limit), a
tabular bundle of up to 50 MiB and requests of up to 100 records. They are
targets set before measurement and not tuned afterwards. The numbers measured
on a developer machine are not committed; shared CI runners are too noisy to
gate on them, so the repository tests the mechanisms (limits, readiness,
parity) and reports the timings.

| Objective | Target |
| --- | --- |
| Startup: process start to `/readyz` 200 | at most 30 s (verify and load dominate; grows with bundle size) |
| Latency, 1 record, p95 | at most 100 ms |
| Latency, 100 records, p95 | at most 500 ms |
| Resident memory after warm-up | at most 400 MiB |
| CPU at idle | under 5% of one core |
| Availability | not claimed: it depends on the platform and replica count |
| Error rate from the service itself (5xx) | 0 on valid requests; every 5xx is a defect |

Not claimed: latency under load on any hosting platform, autoscaling
behaviour, or availability. Those depend on the platform you run it on and
need load tests there.

## Parity

`deploy/local/golden/` holds a hand-written model, records that include nulls,
an unseen category, an extra column and extreme values, expected outputs
computed in closed form with numpy alone, and tolerances declared in advance
(labels exact; probabilities within 1e-9 inside one process, within 1e-6 across
environments). `tests/unit/test_serving_contract.py` checks the in-process,
online, batch and packaged-service paths against them;
`scripts/check_serving_parity.py --url ...` checks any running service the
same way.

## Signing key and scan exceptions are yours to configure

* **Signing key.** Set `SCMLE_SIGNING_KEY`, or point `SCMLE_SIGNING_KEY_FILE`
  at a file a secret manager mounts (the key then never sits in an
  environment variable or an image). The key signs bundles, provenance and
  image attestations and is read the same way when the service verifies them.
* **Published demo key.** `deploy/local/demo.env` holds a key that is public.
  With `SCMLE_ENV=production` the service refuses to start with it, unless `SCMLE_ALLOW_PUBLISHED_KEY=1`, which only the local demo files set.
* **What the signature proves.** The signing is HMAC, so it shows the
  statement was made by a holder of the key and was not altered; it does not
  show who built the model beyond that. Do not call an image "signed and
  verified" in release notes until a key you manage replaces the demo key and
  an independent signer is in place.
* **Scan exceptions.** `deploy/vuln-policy.json` (or `--policy PATH`) lists
  findings and unscannable packages that are waived, each with a reason and an
  expiry date. The `scmle` entry exists only because the scanner cannot see an
  unpublished package; remove it once the package is on an index.
