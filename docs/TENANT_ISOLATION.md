# Tenant isolation policy

What "tenant" means here, what isolates tenants, and what does not. It covers
the prediction service (`scmle.inference.serving`, `modality_serving`) behind
`scmle.security`. Cloud-level isolation (accounts, networks, buckets, IAM) is
declared per provider in Phase 16 and is not claimed here.

## Model

* A **tenant** is the owner of a deployment: its models, its data and its
  audit trail. A deployment serves exactly one tenant, named by
  `SCMLE_TENANT`. Two tenants never share a process, a model bundle, a
  ledger file or a signing key.
* A caller proves its tenant with the `tenant` claim of a verified token. The
  claim is read only after the signature, issuer, audience, expiry and
  revocation checks pass.
* The tenant is never taken from the request path, query or body. A caller
  cannot choose another tenant's model by naming it.

## Rules

1. **Deny by default.** A request without a valid token is refused with 401;
   the only public paths are `/livez` and `/readyz`, which disclose liveness
   and readiness (and the model version and digest once ready) and nothing
   else.
2. **Scope, then tenant.** A route needs its scope (`model:read` for
   `/health` and `/schema`, `model:predict` for `/predict`). If the
   deployment names a tenant, a token whose tenant is absent or different is
   refused with 403, whatever its scopes.
3. **No cross-tenant scope.** There is no wildcard tenant and no scope that
   spans tenants. Operating several tenants means several deployments, each
   with its own `SCMLE_TENANT`, key set and audit file.
4. **Separate keys.** Each tenant's tokens are verified against that
   tenant's key set (`SCMLE_AUTH_KEYS_FILE`). A key shared by two tenants
   would let either mint the other's tokens; do not do it.
5. **Separate storage.** Each tenant's bundles, snapshots, checkpoints and
   run manifests live under their own prefix or bucket
   (`scmle.data.store` / `scmle.runners` take the prefix from the
   deployment). The storage role of the service can read only that prefix.
6. **Separate roles.** Training and serving use different roles
   (`scmle.runners.roles`): serving can read bundles and write nothing but
   its own audit trail; training can write checkpoints and manifests and
   cannot load or activate a service.
7. **Safe errors.** A refusal says only "authentication required",
   "forbidden" or "too many requests". It never says which check failed,
   which key id was tried, or whether a tenant exists. The precise reason is
   written to the audit log.
8. **Audit.** Every activation, access decision (allowed or denied) and
   administrative action (key rotation) is a hash-chained `audit` row of the
   deployment ledger: who, what, outcome, request id; never a credential or a
   request body.
9. **Limits.** Per-subject rate limiting (default 20 requests/s, burst 40),
   per-client limiting of failed attempts (2/s, burst 10), the request size
   and record limits of `docs/SERVICE_SLO.md`.

## What this does not give you

* **TLS.** The reference server speaks plain HTTP. Terminate TLS at the
  ingress; the token travels in a header and must not cross a network
  unencrypted.
* **Protection from a compromised host.** Whoever can read the process
  memory or `SCMLE_AUTH_HS256_SECRET` can mint tokens. Prefer RS256 with the
  private key held by the identity provider: the service holds public keys
  only.
* **Revocation latency.** A revoked token is refused from the next request
  if the revocation file is current; propagating that file to every replica
  is the deployment's job. Short token lifetimes bound the exposure.
* **A tail-complete audit trail.** The chain shows that rows were not
  edited, removed from the middle or reordered, not that the newest rows
  were never removed. Keep the ledger `head` somewhere else.
* **Noisy-neighbour isolation** between processes of the same host, and
  side channels.
* **Provider-level isolation.** Not tested; it needs the cloud accounts.

## Configuration

| Variable | Meaning |
| --- | --- |
| `SCMLE_AUTH_KEYS_FILE` | JWKS file (`oct` keys give HS256, `RSA` keys give RS256; each needs a `kid`; optional `nbf` and `exp` in epoch seconds bound a key). Rotate by publishing the new key beside the old one, then set the old key's `exp`. |
| `SCMLE_AUTH_HS256_SECRET` | One HS256 secret (at least 32 bytes) with kid `env`; for local runs and small deployments. |
| `SCMLE_AUTH_ISSUER`, `SCMLE_AUTH_AUDIENCE` | Required with either key source. |
| `SCMLE_TENANT` | The tenant this deployment serves. |
| `SCMLE_AUTH_REVOKED_FILE` | JSON `{"tokens": [], "subjects": [], "key_ids": []}`, re-read on every request; an unreadable file refuses every token. |
| `SCMLE_AUDIT_FILE` | Audit ledger path; memory only when unset. |

A server that listens on anything but a loopback address without a key
source refuses to start (`ConfigError`): no shipped configuration exposes an
unauthenticated backend. The generated image defaults to every interface, so
it needs these variables.
