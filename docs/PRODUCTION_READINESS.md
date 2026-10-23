# Production-readiness gate

This checklist is the release contract for the current beta line. A release
must not be described as production-ready until every required gate is green.

## Implemented controls

- [x] Pull-request CI has bounded job timeouts and a focused contract suite.
- [x] Scheduled/manual CI retains the complete test suite.
- [x] Release CI verifies tags, changelog, wheel metadata, wheel installation,
  and source-distribution installation.
- [x] Release publication is gated on the staged release preflight: build,
  focused tests, end-to-end tests, integration tests, final regression, and
  release artifact checks.
- [x] Ruff, mypy, Bandit, and pip-audit run in CI.
- [x] Model bundles require authenticated or explicitly trusted loading.
- [x] Bundle format 2 checksums `schema.json` before deserialization.
- [x] Bundle metadata and signature failures return structured artifact errors.
- [x] Run artifacts are written atomically.
- [x] HTTP reference serving bounds request bytes and record count and emits
  baseline no-cache/content-type security headers.
- [x] Dataset curation writes `curation_report.json` with findings, repairs,
  feature decisions, and source/profiled row counts.
- [x] Champion promotion and rollback require an independent approval bound to
  the exact registered bundle content and persist the complete approval record.
- [x] Common tabular, compressed, spreadsheet, Arrow, in-memory, and bounded
  streaming inputs have dedicated tests and CI coverage.

## Required before a stable production release

- [ ] Complete the supported-platform matrix on the release commit, including
  optional integrations used by the release.
- [ ] Complete the scheduled full suite and benchmark regression gate without
  timeout or unexplained skips.
- [x] Publish migration guidance for bundle format 2 and any public API
  changes.
- [x] Define and enforce a 50% coverage threshold for the bounded PR contract
  tier; the scheduled full suite publishes the broader coverage trend.
- [ ] Run the [operational rehearsal](OPERATIONS_REHEARSAL.md) covering
  interruption, resume, rollback, signing-key rotation, and monitoring
  handoff.
- [x] Review the reference HTTP adapter against the deployment threat model;
  bounded bodies, request I/O timeouts, daemonized workers, minimal response
  fingerprinting, and security headers are enforced. It remains gateway-only
  and does not provide authentication or TLS.
- [ ] Promote the package version and classifier only after the above gates are
  green and the changelog contains the release decision.

The checklist is intentionally conservative: a green unit suite is evidence
for the covered behavior, not proof of cross-platform or operational readiness.

## Atomic commits and verification order

Use [plan.md](../plan.md) as the canonical execution plan and keep each
implementation or evidence change in a small, reviewable commit. Each commit
must leave the repository buildable, state its evidence command, and record
the exact commit, environment, result, skips, warnings, and artifacts. Do not
combine implementation, release-policy, site, and final release-decision
changes.

The required order is:

1. Build all Python packages, wheels, source distributions, site pages,
   documentation, and declared optional artifacts after implementation is
   complete.
2. Run focused unit, property, contract, static, API, and changed-path tests.
3. Run supported end-to-end user journeys from input through documented
   output.
4. Run integration checks covering package boundaries, optional dependencies,
   connectors, deployment, site routes, and external boundaries.
5. Run the full regression and release gates once, on the release candidate,
   only after all implementation work and phases 1–4 are complete.

A failure blocks the next phase. Fix the implementation in a new atomic
commit and rerun the affected phase; do not use an early regression run as
evidence for a later implementation state.
