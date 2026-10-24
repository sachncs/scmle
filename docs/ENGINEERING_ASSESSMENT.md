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

| ID | Location | Severity | Evidence | Impact | Recommendation | Status | Verification |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F-001 | `scml/models/deeplearning/networks.py` | P2 | A split string literal after `return` made the JEPA small-dataset rejection message incomplete. | Reduced actionable diagnostics for rejected training configurations. | Preserve the complete boundary explanation and lock it with a regression test. | Fixed | `tests/unit/test_deeplearning_contracts.py::test_jepa_rejection_reason_includes_both_training_boundaries` passed. |
| F-002 | Repository root, `docs/`, and `benchmarks/` | P2 | Orphaned planning/release evidence and unreferenced benchmark result snapshots were present outside the maintained artifact set. | Increased maintenance burden and stale-information risk. | Remove obsolete artifacts and scan code, docs, and package outputs for operational references. | Fixed | Operational-reference scan and package artifact inspection passed; the assessment itself records the deletion. |
| F-003 | `pyproject.toml` project URLs | P3 | The changelog URL referenced the retired `main` branch while the repository uses `master`. | Documentation link could lead users to a missing or incorrect page. | Keep repository URLs aligned with the canonical branch. | Fixed | Targeted stale-URL scan is clean. |
| F-006 | `scml/inference/serving.py:BatchServing.score_file` | P1 | Batch prediction output was written directly to its destination. | Interruption could leave a partial CSV visible to downstream consumers. | Publish output through the existing fsync-and-replace durability primitive. | Fixed | Serving regression test verifies the atomic writer path; durability failure tests pass. |
| F-007 | `scml/cli.py:PredictCommand.execute` | P1 | The public `scml predict --output` path had the same direct-write hazard as batch serving. | A failed CLI process could expose incomplete prediction output. | Use the atomic writer for CLI-produced artifacts. | Fixed | `test_predict_with_explicit_renames` verifies the writer path and output. |
| F-008 | `scml/inference/onnx_export.py:export` | P1 | ONNX graph export wrote serialized model bytes directly to the destination. | Interruption could expose a partial model artifact. | Use the atomic writer for exported graphs and retain failure-preservation coverage. | Fixed | Export uses the atomic writer; durability failure-preservation tests pass. The ONNX suite passed `11` tests with `4` booster-specific skips under CPython 3.13. |
| F-009 | Optional dependency resolution (`pyproject.toml`, `uv.lock`) | P1 | The all-extras environment selected `filelock 4.2.0`, which was reported with six known CVEs by `pip-audit`; all listed fixes are available in `4.2.1` or newer. | Optional Torch/transformer/TabPFN environments could install a vulnerable transitive dependency. | Constrain the affected optional extras to `filelock>=4.2.1` and lock a patched release. | Fixed | `uv.lock` selects `filelock 4.2.4`; all-extras `pip-audit` reports no known vulnerabilities. |
| F-004 | `.github/workflows/` and GitHub account configuration | P1 | Current hosted CI and Pages runs fail before executing steps because of the account billing/spending-limit condition. | Hosted platform coverage and public documentation publication cannot be verified locally. | Repair the account/repository billing condition, rerun hosted CI and Pages, and record successful job-level evidence. | Unresolved external blocker | Latest CI run `40086738224` failed at startup with all 20 jobs reporting zero executed steps; Pages run `40086738292` also failed before execution. The live verifier received HTTP 404 for all 18 configured public routes. |
| F-005 | Release governance and operations | P1 | Independent signed review, external transport/container/kernel fault evidence, external secret custody, and an approved release decision are absent. | Local tests cannot establish the required operational and governance evidence for a production release. | Obtain and retain the required independent signed reviews, external fault evidence, secret-custody proof, and release approval. | Unresolved external blocker | Required by the release boundary and engineering brief; no independent signed records are available in the checkout. |

The Vulture findings for `Estimator.get_params(deep=...)` and
`AsinhTarget.fit(..., unused=...)` are intentional sklearn-compatible
signature parameters. They are not dead code and were retained to preserve
the estimator protocol.

## Verification report

Executed on 2026-10-24:

- Focused regression: `22 passed`.
- Specification suite: `48 passed`.
- Full repository regression at current `master` commit `30d3c6c` under
  CPython 3.13: `2610 passed, 149 skipped` in 38 minutes 15 seconds under
  `-W error::ResourceWarning`.
- Full-regression skip audit: all `149` skips were approved dependency skips;
  there were zero unapproved skips, unexplained skips, or xfails.
- ONNX-specific suite under CPython 3.13 with the ONNX extra: `11 passed, 4
  skipped`; the skips require optional booster packages not installed in that
  environment.
- Vulture: the scan found only the intentional sklearn-compatible `deep` and
  `unused` signature parameters; no genuine dead code was reported.
- Runtime smoke checks: Python 3.12, 3.13, and normal CPython 3.14 each
  installed/imported the package and passed the CLI help smoke test on macOS
  arm64.
- Local reliability rehearsal: all 15 declared fault scenarios passed and
  content-addressed evidence was retained under the recorded temporary
  evidence directory. This is controlled local evidence, not proof of a
  cloud broker, physical disk, or kernel-wide failure.
- Container smoke check: Linux arm64 Python 3.13 installed the package from
  the checkout, imported it, and passed the CLI help smoke test through
  Colima/Docker. The temporary VM was stopped after the check.
- Operational and secret-boundary checks: `65 passed` across observability,
  health/readiness, monitoring, incident, vault, keyring, release-review,
  release-evaluation, and external-release contract tests.
- Optional-capability matrix with all declared extras under CPython 3.13:
  `125 passed, 0 skipped` in 9 minutes 44 seconds; the skip audit reported
  no skips or xfails. One upstream MLflow deprecation warning was emitted.
- All-extras dependency audit after the filelock remediation: no known
  vulnerabilities; only the editable project was not auditable on PyPI.
- Ruff lint: passed.
- Ruff format check: passed for `464` files.
- mypy: no issues in `249` source files.
- Bandit: exit code `0`; only reviewed `nosec`/comment warnings were emitted.
- pip-audit: no known vulnerabilities; the editable project was skipped as
  expected.
- Package build: wheel and source distribution built successfully.
- Twine metadata validation: both artifacts passed.
- Clean installation: a fresh Python 3.11 virtual environment installed the
  wheel with declared runtime dependencies; `import scml`, `import scml.cli`,
  and `scml --help` all succeeded.
- Runtime coverage: Python 3.11 was verified locally; Python 3.12 through
  3.14 remain dependent on the blocked hosted matrix.
- Wheel/sdist artifact inspection: removed planning artifacts were absent.
- Website verification: npm audit found zero vulnerabilities; Astro built 19
  pages; accessibility checks passed for all 19 pages; 479 internal links
  passed; and Playwright passed all 4 Chromium smoke tests.
- Hosted-control static verification: all 5 workflow files passed with 59
  pinned action references and no static-control failures.
- Latest hosted execution check: CI run `40086738224` and Pages run
  `40086738292` failed before any workflow step executed, so hosted evidence
  remains unavailable.
- Vulture: not executable in the current environment because the optional
  tool is not installed; the earlier scan found only intentional sklearn
  protocol parameters plus the JEPA unreachable literal, which is fixed.
- Live Pages verification: the configured HTTPS URL returned HTTP 404 for all
  18 required routes, so public documentation deployment is not verified.
- Stale-reference and diff checks: passed.

The current full regression covers the final local `master` tip. The
all-platform hosted matrix, Windows execution, public Pages deployment,
external transport/container/kernel fault evidence, secret custody, and
independent release review remain unverified.

## Implementation summary

- Files modified or removed: the five workflow files under
  `.github/workflows/`, `pyproject.toml`, the JEPA network and its unit test,
  the serving and CLI output paths with their integration tests, the ONNX
  export path, and this assessment; obsolete planning/readiness documents and
  unreferenced benchmark snapshots were removed.
- Removed obsolete `plan.md`, production planning/readiness documents, and
  unreferenced benchmark snapshots.
- Removed stale references to those files and corrected the changelog URL to
  `master`.
- Corrected the JEPA rejection diagnostic without changing its public API.
- Added a regression test that locks the complete diagnostic message.
- Routed batch, CLI, and ONNX artifact publication through the existing atomic
  durability primitive.
- Constrained optional Torch/transformer/TabPFN environments to patched
  `filelock` and regenerated `uv.lock`; no public exports changed.

## Explicit production-boundary decisions

- The reference HTTP adapter remains gateway-only. Authentication, TLS,
  transport-specific decoding, and paging must be supplied and evidenced by
  the deployment owner before production use.
- Experimental non-tabular and foundation-model paths remain explicitly
  excluded from the production-supported claim until their optional
  dependency, platform, and independent review evidence is complete.
- Version `0.1.0b1` remains beta and is not eligible for release promotion
  until hosted execution, external fault evidence, secret custody, and signed
  independent approval are complete.

## Residual risk register

- Hosted runner and Pages availability remain unverified until the GitHub
  account/repository configuration is repaired.
- Native Windows/macOS hosted execution and optional integration execution
  require successful hosted runs or an independent external reproduction.
- The free-threaded CPython 3.14 environment cannot currently resolve the
  ONNX runtime wheel on this macOS arm64 host; normal CPython 3.14 package
  import and CLI smoke checks passed.
- The local fault executor deliberately does not claim cloud, physical-disk,
  or kernel-wide failure semantics; those require environment-specific
  operators and independent review.
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
