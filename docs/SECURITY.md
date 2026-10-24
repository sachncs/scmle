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
* **Multi-tenant serving.** Out of scope: the HTTP adapter is a minimal
  reference server with no authentication or TLS and must sit behind a
  gateway.

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
| Supply chain | Releases use PyPI trusted publishing (OIDC), no stored token; `pip-audit` gates CI | `.github/workflows` |
| Signing-key lifecycle | Public key metadata records identity, owner, expiry, retirement, replacement, and emergency revocation without storing secret material | `tests/unit/test_keyring.py` |

## Guidance for operators

1. Set `SCMLE_SIGNING_KEY` wherever bundles are produced and verified.
2. Rotate keys with `ModelBundle.rotate_signing_key(directory, old, new)`;
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
