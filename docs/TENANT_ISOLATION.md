# Tenant isolation policy

What "tenant" means here, what isolates tenants, and what does not. It covers
the prediction service (`scmle.serving.server`, `modality_serving`) behind
`scmle.security`. Infrastructure-level isolation (accounts, networks, storage
permissions) belongs to wherever you host it and is not claimed here.

## Model

* A **tenant** is the owner of a deployment: its models, its data and its
  audit trail. A deployment serves exactly one tenant, named by
  `SCMLE_TENANT`, so one process never serves two tenants. Give each
  tenant's deployment its own model bundles, ledger file and signing key: the
  service does not check that these are kept apart.
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
   would let either mint the other's tokens; do not do it. A JWK may carry
   a `tenant` field: that key then verifies only tokens whose tenant claim
   names that tenant (otherwise `wrong_tenant`), so even a key wrongly
   shared cannot carry a token from one tenant to another.
5. **Separate storage.** Each tenant's objects live in the namespace
   `tenants/<tenant>/` of the store (`scmle.storage.tenancy.Tenancy`):
   callers name keys relative to their namespace and the prefix is added
   underneath, never taken from the key. A key that tries `..`, an absolute
   path, a backslash, a percent-encoded separator or the `tenants` root is
   refused, and a backend listing outside the namespace is refused too.
   Tenant names are lowercase DNS labels, so two never share a directory on
   a case-insensitive file system. The job entrypoint confines its store to
   `SCMLE_TENANT`, `Gateway.confine` gives a verified caller its own
   tenant's namespace only (403 for no tenant or another tenant's), and an
   experiment store keeps each tenant's runs in a database file of its own.
   A local store refuses any path through a symbolic link.
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
  ingress; the token travels in a plain header, which anyone on an
  unencrypted network path can read.
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
* **Platform-level isolation.** Not tested; it depends on where you host the service.
* **A symbolic link swapped in mid-operation.** A local store checks for
  links before each operation and then opens the path, so a process that
  can write inside the store root and swaps a directory for a link between
  the check and the open wins that one operation (the next one is caught).
  Only the store's own processes may write its root; the namespaces do not
  defend against a writer on the same file system.

## Adopting a store written before namespacing

Objects of a store written before tenants were namespaced sit at its root,
outside every namespace, and no tenant sees them. Move them into the tenant
the store served with `scmle store adopt --root <store> --tenant <tenant>`
(`Tenancy.adopt`). Each object is copied with `if_absent` and its original
removed only if unchanged, so an interrupted run is finished by running the
command again; a different object already at a destination stops it with a
`ConflictError` and changes nothing more.

## Configuration

| Variable | Meaning |
| --- | --- |
| `SCMLE_AUTH_KEYS_FILE` | JWKS file (`oct` keys give HS256, `RSA` keys give RS256; each needs a `kid`; optional `nbf` and `exp` in epoch seconds bound a key, and an optional `tenant` binds it to one tenant). Rotate by publishing the new key beside the old one, then set the old key's `exp`. |
| `SCMLE_AUTH_HS256_SECRET` | One HS256 secret (at least 32 bytes) with kid `env`; for local runs and small deployments. |
| `SCMLE_AUTH_ISSUER`, `SCMLE_AUTH_AUDIENCE` | Required with either key source. |
| `SCMLE_TENANT` | The tenant this deployment serves; the job entrypoint confines its store to `tenants/<tenant>/`. |
| `SCMLE_AUTH_REVOKED_FILE` | JSON `{"tokens": [], "subjects": [], "key_ids": []}`, re-read on every request; an unreadable file refuses every token. |
| `SCMLE_AUDIT_FILE` | Audit ledger path; memory only when unset. |

A server that listens on anything but a loopback address without a key
source refuses to start (`ConfigError`): no shipped configuration exposes an
unauthenticated backend. The generated image defaults to every interface, so
it needs these variables.
