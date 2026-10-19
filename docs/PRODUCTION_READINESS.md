# Production-readiness gate

This checklist is the release contract for the current beta line. A release
must not be described as production-ready until every required gate is green.

## Implemented controls

- [x] Pull-request CI has bounded job timeouts and a focused contract suite.
- [x] Scheduled/manual CI retains the complete test suite.
- [x] Release CI verifies tags, changelog, wheel metadata, wheel installation,
  and source-distribution installation.
- [x] Ruff, mypy, Bandit, and pip-audit run in CI.
- [x] Model bundles require authenticated or explicitly trusted loading.
- [x] Bundle format 2 checksums `schema.json` before deserialization.
- [x] Bundle metadata and signature failures return structured artifact errors.
- [x] Run artifacts are written atomically.
- [x] HTTP reference serving bounds request bytes and record count and emits
  baseline no-cache/content-type security headers.
- [x] Dataset curation writes `curation_report.json` with findings, repairs,
  feature decisions, and source/profiled row counts.
- [x] Common tabular, compressed, spreadsheet, Arrow, in-memory, and bounded
  streaming inputs have dedicated tests and CI coverage.

## Required before a stable production release

- [ ] Complete the supported-platform matrix on the release commit, including
  optional integrations used by the release.
- [ ] Complete the scheduled full suite and benchmark regression gate without
  timeout or unexplained skips.
- [x] Publish migration guidance for bundle format 2 and any public API
  changes.
- [ ] Define and enforce a coverage threshold for the declared release tiers.
- [ ] Run an operational rehearsal covering interruption, resume, rollback,
  signing-key rotation, and monitoring handoff.
- [ ] Review the reference HTTP adapter against the deployment threat model;
  it remains gateway-only and does not provide authentication or TLS.
- [ ] Promote the package version and classifier only after the above gates are
  green and the changelog contains the release decision.

The checklist is intentionally conservative: a green unit suite is evidence
for the covered behavior, not proof of cross-platform or operational readiness.
