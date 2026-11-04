# Phase 14: Portable core (local first)

Feature: [F14](../product/features/F14-cloud-portability-and-production-operation.md)
items 1, 2 and 5. Depends on nothing. Everything here is verifiable without a
cloud account, and every later phase depends on it.

## Work packages

- **WP-14.1 Deployment contract.** `scmle/deploy/contract.py`: one typed spec
  (image digest, bundle digest, configuration, secret references, resources,
  health, version) that every provider adapter must satisfy. Mirror the
  contract style in `scmle/contracts`. Add to the API fixture and
  `docs/API_STABILITY.md`.
- **WP-14.2 Local environment (item 1).** Locked dependencies exported from
  `uv.lock`; `compose.yaml` with the API, a mock object store, a mock metrics
  sink and a test runner; a single documented command (`scmle dev up` or a
  make target) that trains a sample model, packages it, serves predictions,
  runs monitor and executes a rollback drill. Pin amd64 and arm64. All test
  datasets, mock services and configuration are explicit files.
- **WP-14.3 Hardened artifact (item 2).** Rework the Dockerfile generated in
  `scmle/inference/packaging.py`: pinned base-image digest, locked install,
  non-root user, read-only root filesystem, resource limits. Produce an SBOM
  (extend `scmle/governance/sbom.py`), a vulnerability scan with a CI gate,
  and provenance (signed attestation). Verify signature and digest before
  deserialization. Refuse `SCMLE_TRUSTED=1` when `SCMLE_ENV=production`, and
  remove it from generated documentation and the Dockerfile. Keep secrets out
  of the image and the repository (a test greps the built context).
- **WP-14.4 Serving contract (item 5).** Versioned request and response
  schemas; distinct `/livez` and `/readyz` (ready only after verify, load and
  warm-up); concurrency, timeout and body-size limits; a batch command that
  calls the same preprocessing and model path as online serving; explicit
  handling of nulls, unseen categories and schema changes; digest and
  version in every response; a written service SLO for CPU, memory, startup
  and latency; a golden-record parity suite with declared tolerances.

## Gate

CI on a clean runner runs the local demo end to end including the rollback
drill; the image builds from pinned inputs, passes the scan gate and verifies
its own signature; the golden-record suite matches the local reference within
tolerance; readiness stays false for an unsigned or corrupted bundle.

## Evidence

`plan/evidence/phase-14/`: the demo transcript, the SBOM, scan output,
attestation, parity report.

## Not claimed

No cloud result. Image reproducibility is claimed only to the extent two
builds on the same inputs produce the same digest, measured and reported.
