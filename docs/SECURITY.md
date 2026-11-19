# Security review (0.1.0 beta)

Scope: the `scmle` package, its default configuration and optional extras.
Method: manual review of every deserialization, subprocess, network, SQL and
file-write path, plus `bandit -r scmle` and `pip-audit` (both run in CI).

## Threat model

* **Untrusted model bundles, checkpoints and packages.** All are pickle-based
  (joblib). Unpickling executes code, so a file from an untrusted source is
  arbitrary code execution. This is the principal risk.
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
| Loading a malicious bundle | Loading is **refused unless authentic**: the manifest HMAC-SHA256 signature must verify (`$SCMLE_SIGNING_KEY` or `signing_key=`), or, for an unsigned bundle only, the caller vouches with `trusted=True`; a signed bundle with no key set to verify it is refused even with `trusted=True` | `tests/unit/test_bundle.py::test_unsigned_bundles_are_refused_unless_trusted`, `tests/unit/test_bundle.py::test_signed_bundles_load_without_trust_and_detect_tampering`, `tests/unit/test_bundle.py::test_a_signed_bundle_with_no_key_is_refused_even_when_trusted` |
| Tampered model file | SHA-256 per file is verified **before** unpickling; an invalid signature is an error even with `trusted=True` | tamper and swapped-model tests |
| Malicious deep-learning checkpoint or document/image package | Checkpoints are an authenticated envelope (`scmle.envelope`): a JSON header with the payload digest and an HMAC is read and verified before any byte is unpickled. Package manifests, which record the payload digest, are signed the same way. Unsigned files, or signed files read with no key, are refused unless `trusted=True`; an invalid signature is an error even with `trusted=True` | `tests/unit/test_envelope.py::test_a_tampered_payload_is_refused_before_unpickling`, `tests/unit/test_envelope.py::test_an_unsigned_envelope_needs_trust`, `tests/unit/test_envelope.py::test_a_signed_envelope_without_a_key_needs_trust`, `tests/unit/test_envelope.py::test_a_signed_manifest_verifies_and_a_changed_one_does_not`, `tests/unit/test_envelope.py::test_an_invalid_signature_is_refused_even_when_trusted`, `tests/unit/test_envelope.py::test_a_document_package_is_checked_before_its_payload_is_loaded` (a tripwire on `joblib.load`) |
| Dependency drift | Bundle records Python and scikit-learn versions; mismatches are refused unless allowed | version-mismatch test |
| SQL injection | Tracking SQL binds every value; dynamic column names come from a fixed whitelist | `tests/unit/test_tracking.py` |
| Resource leaks | SQLite connections are closed on every call | leak test with `ResourceWarning` as error |
| Oversized request bodies | The HTTP adapter answers 413 above 16 MiB instead of reading the body | `tests/integration/test_serve_and_export.py` |
| Slow or abandoned HTTP clients | Each connection has a positive I/O timeout and worker threads are daemonized; the adapter remains gateway-only | `tests/integration/test_serve_and_export.py` |
| Server fingerprint disclosure | The reference handler emits a stable minimal `Server` header without the Python runtime version | `tests/integration/test_serve_and_export.py` |
| Secrets | No credentials are read, stored or logged; the signing key lives only in the environment | review |
| Trusting an unsigned bundle in production | `trusted=True` and `SCMLE_TRUSTED=1` are refused when `SCMLE_ENV=production`; the generated service then needs a signed bundle and a verifying provenance statement | `tests/unit/test_service_hardening.py::test_trusted_is_refused_in_production_everywhere`, `tests/unit/test_service_hardening.py::test_an_unsigned_bundle_never_verifies_in_production`, `tests/unit/test_service_hardening.py::test_production_needs_verifiable_provenance`, `tests/integration/test_service_package.py::test_production_serves_only_a_signed_verified_folder` |
| Serving a modified or unattested service folder | Before it deserializes anything, the service checks the bundle signature, the checksum of every listed file, the signed provenance statement (bundle digest and build materials); on failure it stays alive but never ready | same tests |
| Secrets in the image or its build context | The generated `.dockerignore` excludes credential files; `scmle.deploy.context` scans the context for key material, configured secret values and secret-looking `ENV` lines before every build | `test_the_build_context_holds_no_secret` |
| Vulnerable dependencies in the image | `scmle.deploy.vuln_gate` fails a build on a blocking finding, an empty scan, a finding without severity, an expired exception, or an SBOM component the scan did not cover. It does not scan; the scanner and its database date are the report's | `tests/unit/test_service_hardening.py::test_the_gate_passes_a_clean_report_and_fails_any_pip_audit_finding` (a pip-audit finding has no severity), `tests/unit/test_service_hardening.py::test_the_gate_applies_severity_and_expiring_exceptions`, `tests/unit/test_service_hardening.py::test_the_gate_fails_closed`, `tests/unit/test_service_hardening.py::test_the_gate_checks_that_the_scan_covered_the_sbom` |
| Unauthenticated access | `HttpServing` and `ModalityHttpServing` refuse to listen beyond loopback without a gateway; every path except `/livez` and `/readyz` needs a verified token with the route's scope; the compose file publishes no port and gives its services the demo identity settings | `tests/unit/test_security_service.py::test_no_unauthenticated_server_can_listen_beyond_loopback`, `tests/unit/test_security_service.py::test_the_modality_service_is_gated_the_same_way`, `tests/unit/test_security_service.py::test_probes_are_public_and_everything_else_needs_a_token`, `tests/unit/test_security_service.py::test_predict_works_with_the_scope_and_not_without`, `tests/unit/test_security_service.py::test_the_shipped_compose_file_publishes_no_port_and_secures_the_apis` |
| Invalid, revoked, expired or foreign credentials | Signature, `alg` (never `none`, never confused with the key's), `kid`, key validity window, `iss`, `aud`, `exp`, `nbf`, revocation by token, subject or key; refusals say only 401, 403 or 429, never why | `tests/unit/test_security_tokens.py::test_a_signature_from_another_key_is_refused`, `tests/unit/test_security_tokens.py::test_the_none_algorithm_and_confusion_attacks_are_refused`, `tests/unit/test_security_tokens.py::test_an_expired_key_stops_verifying_and_a_future_key_is_not_yet_valid`, `tests/unit/test_security_tokens.py::test_wrong_issuer_audience_and_missing_claims_are_refused`, `tests/unit/test_security_tokens.py::test_expired_and_future_tokens_are_refused`, `tests/unit/test_security_tokens.py::test_revoked_tokens_subjects_and_keys_are_refused`, `tests/unit/test_security_service.py::test_a_token_of_another_tenant_is_forbidden`, `tests/unit/test_security_service.py::test_error_bodies_never_say_why` |
| Guessing credentials, request floods | Per-client limit on refused attempts and per-subject request limit (429 with `Retry-After`), on top of the size, record and concurrency limits | `tests/unit/test_security_service.py::test_guessing_credentials_is_slowed`, `tests/unit/test_security_service.py::test_the_rate_limit_answers_429_with_retry_after`, `tests/unit/test_security_service.py::test_the_request_size_limit_still_applies_behind_the_gateway` |
| Unrecorded access or administration | Activation, access and key-rotation events are `audit` rows of the hash-chained deployment ledger, without credentials or bodies | same tests |
| Supply chain | Releases use PyPI trusted publishing (OIDC), no stored token; `pip-audit` gates CI | `.github/workflows` |
| Reading the confirmation set early | With a charter, the confirmation rows are sealed in a separate vault process (even under `enforcement="cooperative"`); exploration code receives only an `View` with no reference to a vault, token or log; a second open is refused; every attempt is written to a hash-chained log | `tests/integration/test_charter_run.py::test_a_charter_run_seals_c_even_in_cooperative_mode`, `tests/leakage/test_custody_access.py::test_the_view_reaches_no_sealed_object_or_token`, `tests/leakage/test_custody_access.py::test_a_second_open_is_refused_and_both_attempts_are_logged`, `tests/unit/test_custody_sealed.py::test_log_chains_entries_and_verifies` |
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
stays. The library does not delete a DataFrame or file that its caller holds.
The option that would close it is a CLI that reads the file inside the
orchestrator and does not return the full frame to the caller; it is not
built.

### What is tested about custody, and what is not

| Guarantee | Test | Stated limit |
|---|---|---|
| No artifact of a full charter run contains the canary value of a sealed row (`C` or the final test set) | `tests/leakage/test_custody_canary.py` | Text search for decimal spellings of each value; a function of several rows, or another encoding, is not searched |
| After custody, no pandas object tracked by the garbage collector holds a canary of `C` or the final test set | `tests/leakage/test_custody_memory.py` | The caller's own copy is dropped by the test (see above); buffers outside Python, the page cache and swap are not inspected |
| Detectors, admissibility and robustness modules do not import custody, the vault, the splitters module or the orchestrator, directly or through any scmle import | `tests/leakage/test_exploration_imports.py` | Static imports only; a runtime `importlib` call would not be seen |
| The one release is bound to a claims hash and a ledger hash; the vault refuses a release for any other pair, and the confirmation stage binds every charter run before it opens `C` | `tests/unit/test_vault.py::test_a_bound_vault_refuses_a_release_for_another_selection`, `tests/unit/test_custody_sealed.py::test_a_committed_set_opens_only_for_its_claims_and_ledger`, `tests/leakage/test_custody_access.py::test_a_release_for_a_different_selection_is_refused_by_the_vault`, `tests/docs/test_site_custody.py::test_the_confirmation_stage_binds_the_release_before_it_opens_c` | `Locker.commit` and `open(..., claims_hash, ledger_hash)`; the confirmation stage calls `commit` with the frozen claims hash and ledger head, unless the set was already committed. A caller that opens `C` outside the stage without a binding is not bound |
| Every custody log entry carries an HMAC of its hash when a signing key is available (explicit or `SCMLE_SIGNING_KEY`); a forger who recomputes the hash chain fails the MAC | `tests/unit/test_custody_sealed.py::test_a_keyed_log_carries_a_mac_on_every_entry`, `tests/unit/test_custody_sealed.py::test_the_environment_key_signs_when_no_key_is_given`, `tests/unit/test_custody_sealed.py::test_a_forger_who_recomputes_the_chain_fails_the_mac` | Without a key the log is a checksum only, and a WARNING says so when it is written. A signed log cannot be verified without its key: verification refuses rather than skipping the MACs. An empty, unreadable or blank key (explicit, `SCMLE_SIGNING_KEY` or `SCMLE_SIGNING_KEY_FILE`) is an error, never "no key". Removing the newest entries is not detected by the chain or the MACs; compare `Journal.head` with a copy kept elsewhere |
| The split manifest carries a Merkle root over the canonical content of the rows of `X`, `C` and the final set; a changed value inside a row changes the root | `tests/unit/test_custody_merkle.py`, `tests/unit/test_custody_manifest.py` | Embargo rows have no root; manifests written before this change have none |
| For a temporal split the embargo is at least the number of leading autocorrelation lags of the target outside +-1.96/sqrt(n) | `tests/unit/test_custody_carve.py`, `tests/unit/test_custody_pipeline.py` | A rule of thumb for a stationary series; later (seasonal) dependence is not counted; only the target is measured, not the features |
| Two fixtures (grouped, temporal) have pinned manifest hashes | `tests/unit/test_custody_golden.py` | Pinned on macOS arm64 only; equality on Windows and under lowest dependencies depends on the CI jobs passing and has not been observed from this machine |

Mutation survivors for the custody, charter, vault and attestation modules are
listed in `tests/unit/MUTANTS_custody.md`.

## Guidance for operators

1. Set `SCMLE_SIGNING_KEY` wherever bundles are produced and verified.
2. Rotate keys with `Bundle.rotate_signing_key(directory, old, new)`;
   verify the new key before retiring the old one. Maintain the corresponding
   `Keyring` record and transition it to `retired` only after
   parity is confirmed.
3. Pass `trusted=True` (or `scmle predict --trusted`) only for a bundle you
   produced yourself.
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
