# Security review (0.1.0 beta)

Scope: the `scmle` package, its default configuration and optional extras.
Method: manual review of every deserialization, subprocess, network, SQL and
file-write path, plus `bandit -r scmle` and `pip-audit` (both run in CI).

## Threat model

* **Untrusted model bundles.** Bundles are pickle-based (joblib). Unpickling
  executes code, so a bundle from an untrusted source is arbitrary code
  execution. This is the principal risk.
* **Untrusted datasets.** Parsed with pandas/pyarrow; malformed files raise
  `DataError`.
* **Unauthenticated or cross-tenant callers.** The HTTP adapters verify a
  bearer token (JWT, HS256 or RS256, key rotation by `kid`), check scopes and
  tenant, rate-limit and audit (`scmle.security`; policy in
  [`TENANT_ISOLATION.md`](TENANT_ISOLATION.md)). They do not terminate TLS:
  run them behind an ingress that does.

## Controls

| Risk | Control | Test |
| --- | --- | --- |
| Loading a malicious bundle | Loading is **refused unless authentic**: the manifest HMAC-SHA256 signature must verify (`$SCMLE_SIGNING_KEY` or `signing_key=`), or the caller vouches with `trusted=True` | `tests/unit/test_bundle.py` |
| Tampered model file | SHA-256 per file is verified **before** unpickling; an invalid signature is an error even with `trusted=True` | tamper and swapped-model tests |
| Dependency drift | Bundle records Python and scikit-learn versions; mismatches are refused unless allowed | version-mismatch test |
| SQL injection | Tracking SQL binds every value; dynamic column names come from a fixed whitelist | `tests/unit/test_tracking.py` |
| Resource leaks | SQLite connections are closed on every call | leak test with `ResourceWarning` as error |
| Oversized request bodies | The HTTP adapter answers 413 above 16 MiB instead of reading the body | `tests/integration/test_serve_and_export.py` |
| Slow or abandoned HTTP clients | Each connection has a positive I/O timeout and worker threads are daemonized; the adapter remains gateway-only | `tests/integration/test_serve_and_export.py` |
| Server fingerprint disclosure | The reference handler emits a stable minimal `Server` header without the Python runtime version | `tests/integration/test_serve_and_export.py` |
| Secrets | No credentials are read, stored or logged; the signing key lives only in the environment | review |
| Trusting an unsigned bundle in production | `trusted=True` and `SCMLE_TRUSTED=1` are refused when `SCMLE_ENV=production`; the generated service then needs a signed bundle and a verifying provenance statement | `tests/unit/test_service_hardening.py`, `tests/integration/test_service_package.py` |
| Serving a modified or unattested service folder | Before it deserializes anything, the service checks the bundle signature, the checksum of every listed file, the signed provenance statement (bundle digest and build materials); on failure it stays alive but never ready | same tests |
| Secrets in the image or its build context | The generated `.dockerignore` excludes credential files; `scmle.deploy.context` scans the context for key material, configured secret values and secret-looking `ENV` lines before every build | `test_the_build_context_holds_no_secret` |
| Vulnerable dependencies in the image | `scmle.deploy.vuln_gate` fails a build on a blocking finding, an empty scan, a finding without severity, an expired exception, or an SBOM component the scan did not cover. It does not scan; the scanner and its database date are the report's | `tests/unit/test_service_hardening.py` |
| Unauthenticated access | `HttpServing` and `ModalityHttpServing` refuse to listen beyond loopback without a gateway; every path except `/livez` and `/readyz` needs a verified token with the route's scope; the compose file publishes no port and gives its services the demo identity settings | `tests/unit/test_security_service.py` |
| Invalid, revoked, expired or foreign credentials | Signature, `alg` (never `none`, never confused with the key's), `kid`, key validity window, `iss`, `aud`, `exp`, `nbf`, revocation by token, subject or key; refusals say only 401, 403 or 429, never why | `tests/unit/test_security_tokens.py`, `tests/unit/test_security_service.py` |
| Guessing credentials, request floods | Per-client limit on refused attempts and per-subject request limit (429 with `Retry-After`), on top of the size, record and concurrency limits | `tests/unit/test_security_service.py` |
| Unrecorded access or administration | Activation, access and key-rotation events are `audit` rows of the hash-chained deployment ledger, without credentials or bodies | same tests |
| Supply chain | Releases use PyPI trusted publishing (OIDC), no stored token; `pip-audit` gates CI | `.github/workflows` |
| Reading the confirmation set early | With a charter, the confirmation rows are sealed in a separate vault process (even under `enforcement="cooperative"`); exploration code receives only an `ExplorationView` with no reference to a vault, token or log; a second open is refused; every attempt is written to a hash-chained log | `tests/leakage/test_custody_access.py` |
| Edited charter or split manifest | The charter carries a SHA-256 hash and an optional HMAC signature; the split manifest v2 carries a full SHA-256; either fails to load if a byte changes | `tests/unit/test_charter_signing.py`, `tests/unit/test_custody_manifest.py` |
| Signing-key lifecycle | Public key metadata records identity, owner, expiry, retirement, replacement, and emergency revocation without storing secret material | `tests/unit/test_keyring.py` |

### Limits of custody

The vault is a subprocess of the same user with a key passed on standard input.
It stops accidental and in-code access. It is **not** adversary-proof on one
machine: a process with the same operating-system privileges could attach to the
vault or read the original data file. This process drops its own reference to
the sealed rows, but a DataFrame the caller passed in, or the data file, still
holds every row. Custody against a hostile operator needs separation: another
operating-system user, a service, or a write-once object store. Do not describe
a confirmation set as sealed beyond this.

Accepted limit, recorded 2026-11-04: the caller's copy of the data (item C3)
stays. The library cannot delete a DataFrame or file that its caller holds.
The option that would close it is a CLI that reads the file inside the
orchestrator and never returns the full frame to the caller; it is not built.

### What is tested about custody, and what is not

| Guarantee | Test | Stated limit |
|---|---|---|
| No artifact of a full charter run contains the canary value of a sealed row (`C` or the final test set) | `tests/leakage/test_custody_canary.py` | Text search for decimal spellings of each value; a function of several rows, or another encoding, is not searched |
| After custody, no pandas object tracked by the garbage collector holds a canary of `C` or the final test set | `tests/leakage/test_custody_memory.py` | The caller's own copy is dropped by the test (see above); buffers outside Python, the page cache and swap are not inspected |
| Detectors, admissibility and robustness modules do not import custody, the vault, the splitters module or the orchestrator, directly or through any scmle import | `tests/leakage/test_exploration_imports.py` | Static imports only; a runtime `importlib` call would not be seen |
| The one release can be bound to a claims hash and a ledger hash; the vault refuses a release for any other pair | `tests/unit/test_vault.py`, `tests/unit/test_custody_sealed.py`, `tests/leakage/test_custody_access.py` | Available as `SealedSet.commit` and `open(..., claims_hash, ledger_hash)`. The confirmation stage does not call `commit` yet, so charter runs are not bound; the vault enforces the binding only if someone sets it |
| Every custody log entry carries an HMAC of its hash when a signing key is available (explicit or `SCMLE_SIGNING_KEY`); a forger who recomputes the hash chain fails the MAC | `tests/unit/test_custody_sealed.py` | Without a key the log is a checksum only. Removing the newest entries is not detected by the chain or the MACs; compare `OpenLog.head` with a copy kept elsewhere |
| The split manifest carries a Merkle root over the canonical content of the rows of `X`, `C` and the final set; a changed value inside a row changes the root | `tests/unit/test_custody_merkle.py`, `tests/unit/test_custody_manifest.py` | Embargo rows have no root; manifests written before this change have none |
| For a temporal split the embargo is at least the number of leading autocorrelation lags of the target outside +-1.96/sqrt(n) | `tests/unit/test_custody_carve.py`, `tests/unit/test_custody_pipeline.py` | A rule of thumb for a stationary series; later (seasonal) dependence is not counted; only the target is measured, not the features |
| Two fixtures (grouped, temporal) have pinned manifest hashes | `tests/unit/test_custody_golden.py` | Pinned on macOS arm64 only; equality on Windows and under lowest dependencies depends on the CI jobs passing and has not been observed from this machine |

Mutation survivors for the custody, charter, vault and attestation modules are
listed in `tests/unit/MUTANTS_custody.md`.

## Guidance for operators

1. Set `SCMLE_SIGNING_KEY` wherever bundles are produced and verified.
2. Rotate keys with `Bundle.rotate_signing_key(directory, old, new)`;
   verify the new key before retiring the old one. Maintain the corresponding
   `SigningKeyRegistry` record and transition it to `retired` only after
   parity is confirmed.
3. Never pass `trusted=True` (or `scmle predict --trusted`) for a bundle you
   did not produce.
4. Serve behind an authenticating gateway; bind to loopback by default.
5. Treat the model registry directory as a trust boundary: it is read with
   `trusted=True` after checksum verification at registration time.

The registry is metadata-only: the secret remains in the deployment secret
manager. An active, non-expired key may sign. A retired key cannot sign but
may verify pre-retirement evidence only when a caller explicitly requests
historical verification. A revoked key fails verification immediately,
including historical checks. This distinction is important during emergency
rotation: preserve evidence without accepting a compromised signer.

## Static analysis

`bandit` reports no open findings. Four low-risk findings were reviewed and
annotated with their rationale: a category placeholder string (not a
password), two seeded non-cryptographic random generators, and a SQL
statement whose column names come from a whitelist.

## Reporting

Please report vulnerabilities privately to the maintainers rather than in a
public issue.
