# Phase 14: Portable core (local first)

Feature: [F14](../product/features/F14-cloud-portability-and-production-operation.md)
items 1, 2 and 5. The code, tests and local evidence for the work packages are
built (deployment contract, local environment, hardened artifact, serving
contract); see `docs/LOCAL.md`, `docs/SERVICE_SLO.md` and
`plan/evidence/phase-14/`. What remains is what could not be run without a
container daemon or a cloud account.

## Open

- **Container path, untested.** `compose.yaml`, `deploy/Dockerfile`, the
  generated `build.sh` and `make demo-compose` were written and the compose
  file validates, but no image was built and no container started (the Docker
  daemon was not running). Run them once on a machine with Docker, fix what
  breaks, and keep the transcript.
- **Image reproducibility.** Measure and report whether two builds on the same
  pinned inputs give the same image digest. Nothing is claimed until measured.
- **arm64.** Build and run the demo with `PLATFORM=linux/arm64`.
- **Image attestation.** `image-attestation.json` binds a real image digest to
  the provenance; it needs a real build. The attestations are HMAC-signed;
  asymmetric or keyless signing and a verifier other than this code belong to
  Phase 16.
- **Scan gate on a built image.** The gate ran on the runtime lock with
  pip-audit; run it against the image (an independent scanner such as Trivy)
  in CI.

## Gate (still open)

CI on a clean runner builds the image from pinned inputs, passes the scan gate
and verifies its own signature; the demo then runs in containers. The native
demo, the golden-record suite and the unready-for-corrupted-or-unsigned check
already run in CI (`local-demo`, the unit suite).

## Not claimed

No cloud result. No container result. Image reproducibility is claimed only to
the extent two builds on the same inputs produce the same digest, measured and
reported.
