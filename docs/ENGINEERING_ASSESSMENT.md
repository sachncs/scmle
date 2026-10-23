# Engineering assessment

Assessment date: 2026-10-24

This assessment covers the repository at the current `master` tip after the
orphaned-artifact cleanup and JEPA diagnostic fix. It records executed checks,
not intended checks or historical claims that were not rerun on this tip.

## Executive assessment

The package has a strong modular-monolith architecture. Typed contracts divide
control, execution, and evidence concerns; optional integrations are isolated;
the public API and CLI have explicit entry points; and the repository has
layered unit, property, integration, contract, leakage, packaging, and CI
checks.

The implementation is suitable for continued beta use within its documented
tabular and experimental boundaries. It is not production-ready as an
unattended platform: hosted CI and Pages have repeatedly failed before any
workflow step because of the GitHub account billing/spending-limit condition,
and independent technical, security, statistical, and operational review has
not been supplied. Those are release blockers, not code defects inferred from
local tests.

## Findings register

| ID | Location | Severity | Evidence and impact | Status | Verification |
| --- | --- | --- | --- | --- | --- |
| F-001 | `scml/models/deeplearning/networks.py` | P2 | A split string literal after `return` made the JEPA small-dataset rejection message incomplete. This reduced actionable diagnostics. | Fixed | `tests/unit/test_deeplearning_contracts.py::test_jepa_rejection_reason_includes_both_training_boundaries` passed. |
| F-002 | Repository root, `docs/`, and `benchmarks/` | P2 | Orphaned planning/release evidence and two unreferenced benchmark result snapshots increased maintenance and stale-information risk. | Fixed | Deleted files have no remaining repository references; package artifact inspection found none of them. |
| F-003 | `pyproject.toml` project URLs | P3 | The changelog URL referenced the retired `main` branch while the repository uses `master`. | Fixed | Stale-branch reference scan is clean. |
| F-004 | `.github/workflows/` and GitHub account configuration | P1 | Current hosted CI and Pages runs fail before executing steps because of the account billing/spending-limit condition. Hosted platform coverage and publication cannot be verified locally. | Unresolved external blocker | Current runs `40019762196` and `40019762190` failed at hosted startup; all 20 CI jobs and the Pages build had zero executed steps. The live verifier also received HTTP 404 for all 18 configured public routes. |
| F-005 | Release governance and operations | P1 | Independent signed review, external transport/container/kernel fault evidence, external secret custody, and an approved release decision are not present. Local tests cannot manufacture these witnesses. | Unresolved external blocker | Required by the repository's release boundary and the engineering brief; no independent signed records are available in the checkout. |

The Vulture findings for `Estimator.get_params(deep=...)` and
`AsinhTarget.fit(..., unused=...)` are intentional sklearn-compatible
signature parameters. They are not dead code and were retained to preserve
the estimator protocol.

## Verification report

Executed on 2026-10-24:

- Focused regression: `22 passed`.
- Specification suite: `48 passed`.
- Full repository regression: `2721 passed, 38 skipped` in 51 minutes 40
  seconds under `-W error::ResourceWarning`.
- Full-regression skip audit: all `38` skips were approved dependency skips;
  there were zero unapproved skips, unexplained skips, or xfails.
- Ruff lint: passed.
- Ruff format check: passed for `464` files.
- mypy: no issues in `249` source files.
- Bandit: exit code `0`; only reviewed `nosec`/comment warnings were emitted.
- pip-audit: no known vulnerabilities; the editable project was skipped as
  expected.
- Package build: wheel and source distribution built successfully.
- Twine metadata validation: both artifacts passed.
- Wheel/sdist artifact inspection: removed planning artifacts were absent.
- Website verification: npm audit found zero vulnerabilities; Astro built 19
  pages; accessibility checks passed for all 19 pages; 479 internal links
  passed; and Playwright passed all 4 Chromium smoke tests.
- Hosted-control static verification: all 5 workflow files passed with 59
  pinned action references and no static-control failures.
- Live Pages verification: the configured HTTPS URL returned HTTP 404 for all
  18 required routes, so public documentation deployment is not verified.
- Stale-reference and diff checks: passed.

The full regression and its skip audit now cover the current tip. The
all-platform hosted matrix, public Pages deployment, and independent release
review remain unverified.

## Implementation summary

- Removed obsolete `plan.md`, production planning/readiness documents, and
  unreferenced benchmark snapshots.
- Removed stale references to those files and corrected the changelog URL to
  `master`.
- Corrected the JEPA rejection diagnostic without changing its public API.
- Added a regression test that locks the complete diagnostic message.
- Added no dependencies and changed no public exports.

## Residual risk register

- Hosted runner and Pages availability remain unverified until the GitHub
  account/repository configuration is repaired.
- Native Windows/macOS hosted execution and optional integration execution
  require successful hosted runs or an independent external reproduction.
- The reference HTTP adapter remains gateway-only and does not provide
  authentication or TLS.
- Experimental non-tabular and foundation-model paths remain outside the
  supported production claim.
- The current package is beta (`0.1.0b1`); no release promotion is justified
  by this assessment.

## Final engineering verdict

**Not Production Ready.** Core package behavior, static quality, security
scanning, packaging, and the modified regression paths are verified locally.
The verdict is blocked by material missing evidence—hosted CI/Pages execution
and independent operational/security/statistical review—not by an unreported
local test failure.
