# Phase 6: External validation and release (remaining)

Canon: [`00-canon.md`](00-canon.md). Features: F7 work items W1-W7 in
`product/features/F7-verification-harness.md`. Residual instrument items are
Phase 7 and are not repeated here. Everything below is **proposed**; nothing
in it has been done.

## Goal

Show, with evidence that is neither the author's nor the user's (P1), that the
instrument is right, then release it through the existing evidence chain. This
phase contains only what Phases 0-5 left undone: the external gates and the
release checklist.

## Verified starting point (2026-11-04)

- Phases 0-5 are in `master`: types and budget, harness, charter and custody,
  ledger and confirmation, F4 admissibility, F5 detectors, F6 robustness
  (`git log`; `CHANGELOG.md` Unreleased).
- Not published. `https://pypi.org/pypi/scmle/json` returns 404. `README.md:22-24`
  and `website/src/pages/index.astro:67-68` install with
  `pip install "scmle @ git+https://github.com/sachncs/scmle"`. `pyproject.toml`
  has `version = "0.1.0b1"`; `CHANGELOG.md` has a `## 0.1.0b1` section and a
  large `Unreleased` section; `git tag -l` lists only `sota-protocol-v1`
  (no release tag exists).
- Release pipeline exists: `.github/workflows/release.yml` (on a `v*` tag:
  verify, test, release-checks, publish by PyPI trusted publishing, external
  digest check; `docs/RELEASING.md`). It has never run to completion (no tag,
  no PyPI project).
- `SECURITY.md` exists at the repo root (supported versions, private
  reporting). `docs/SECURITY.md` carries the custody limits
  (`docs/SECURITY.md:40-43`). No `CITATION.cff` exists.
- Two controls need people: an operational rehearsal reviewed by someone other
  than the author, and a signed independent review checked by
  `scripts/verify_release_review.py` (`docs/RELEASING.md:41`).
- `benchmarks/sota/PROTOCOL.md` is a tagged pre-registration (`sota-protocol-v1`);
  the same discipline is reused here.

## Entry criteria

1. Phase 7 groups A and C closed or accepted: certificates for the gating
   detectors are on file and the open limits are written in docs. (A detector
   that is `descriptive` does not block the phase; a claim that it gates does.)
2. The uncommitted leakage screen (R1 A7) is committed, or is removed from the
   working tree; the release is cut from a clean tree.
3. `benchmarks/qualification/PROTOCOL.md` is committed and tagged
   `qualification-protocol-v1` before any backtest or replication result exists.

## Work packages

| WP | Scope | Creates / modifies | Depends on | Definition of done |
|---|---|---|---|---|
| 6.1 | Protocol: claim, corpora list, thresholds, stop rules, amendment rule (no change to thresholds or corpus assignment after results) | `benchmarks/qualification/PROTOCOL.md` | none | Tagged before results; mirrors `sota/PROTOCOL.md` |
| 6.2 | Corpora: fill every `candidate` entry in `corpora.json` (source URL, licence, SHA-256, size, adjudication source) or remove it; hold-out assigned by hash | `benchmarks/qualification/corpora.json` | 6.1 | At least 15 datasets with at least 5 per defect family, or the claim is narrowed to the families covered; hashes verified by the fetcher |
| 6.3 | Backtest (F7 W1) | `benchmarks/qualification/backtest.py` | 6.1, 6.2 | One pre-registered contrast, run once on hold-out; report with interval |
| 6.4 | Red team (F7 W2) | `benchmarks/qualification/redteam/` | Phase 7 group C | Report filed; each success is a regression test |
| 6.5 | Independent replication (F7 W3) | replication record | 6.1, 6.3 | Statistics within tolerance, or amended |
| 6.6 | TabArena campaign (F7 W4) | results under `benchmarks/sota/` | none (can start now) | Hold-out opened once; claim reported met or not met |
| 6.7 | R oracles and mutation floor (F7 W5, W6) and CI headroom (F7 W7) | fixtures, `setup.cfg`, workflow | none | Differential tests pass; `check_mutation_score.sh 80` exits 0; Windows jobs under 14 minutes |
| 6.8 | Domain-owner pilot: one real dataset run by a team that owns it; owner reviews provenance table, admissibility and mechanisms | pilot record | Phase 4 | Signed record that each mechanism is plausible or flagged; flags become issues |
| 6.9 | Release mechanics (below) | per `docs/RELEASING.md` | 6.3-6.8 | Preflight passes; tag matches version and changelog |

## Release checklist (6.9)

| # | Item | State now | Done when |
|---|---|---|---|
| 1 | Version | `0.1.0b1` in `pyproject.toml`; Unreleased changes are large | A version is chosen (proposal: `0.2.0b1`, because `0.1.0b1` has a changelog section but was never tagged or published: decide whether to reuse it) and set in `pyproject.toml` and `scmle/__init__.py` |
| 2 | Changelog | `Unreleased` section populated through F6 | Moved under the new version heading; the F6 and screen entries are present; the "Limits" bullets from R1 are included |
| 3 | Tag | none | `v<version>` annotated tag on a clean commit that CI has passed |
| 4 | API fixture | `tests/fixtures/public_api.json` was regenerated once (CHANGELOG) | `python scripts/check_api_compatibility.py tests/fixtures/public_api.json` exits 0; any intended change noted in `docs/API_STABILITY.md` |
| 5 | `SECURITY.md` | exists | Supported-versions text matches the first published version; custody limits linked |
| 6 | `CITATION.cff` | absent | Added with title, version, authors, licence, repository URL; validated by `cffconvert --validate` (tool availability unverified) |
| 7 | Docs | `docs/ARTIFACTS.md` documents the stability report; others unverified | Docs describe each artifact a charter run writes; README install lines changed from `git+https` to `pip install scmle` only after step 9 succeeds |
| 8 | Preflight | not run for this tree | `./scripts/release_preflight.sh` exits 0 |
| 9 | Publish | PyPI 404 | `release.yml` completes; `verify_external_release.py` exits 0; `pip install scmle==<version>` works in a clean venv |
| 10 | Pages | `pages.yml`, `site.yml` exist | Site install snippet updated after step 9 |
| 11 | Operational rehearsal | not done by anyone but the author | `docs/OPERATIONS_REHEARSAL.md` filled and reviewed by a non-author |
| 12 | Independent review | none | `python scripts/verify_release_review.py review.json --source-commit <sha>` exits 0 |

## Exit gate

```bash
uv run python -m benchmarks.qualification --check
uv run python -m benchmarks.qualification.backtest --protocol qualification-protocol-v1
./scripts/check_mutation_score.sh 80
./scripts/release_preflight.sh
python scripts/verify_release_review.py review.json --source-commit <sha>
python scripts/verify_external_release.py   # after publish; arguments per scripts/README.md
pip install "scmle==<version>"              # in a clean environment
```

## Go / no-go checklist (all must hold; the signer is never the author)

| # | Criterion | Threshold | Signer |
|---|---|---|---|
| 1 | Null size, every gating detector | upper Wilson bound inside the stated ceiling over at least 1000 replicates | Independent statistician |
| 2 | Recall on planted defects | Wilson 95% lower bound at least the file's floor per type (0.80; leakage 0.90); uncertified types listed as descriptive | Independent statistician |
| 3 | Real-corpus recall | at least 0.70 on hold-out corpora, with interval; or "not measured" stated | Independent reviewer |
| 4 | Predictive validity | gap(block) > gap(pass), one-sided p < 0.05, or "unsupported" stated | Independent statistician |
| 5 | Final-claim FDR | empirical FDR upper limit <= alpha_confirm + 0.01 on semi-synthetic data | Independent statistician |
| 6 | Access control | 0 successful reads of `C`/`F` in red team and tests; open custody gaps (R1 C3-C9) listed in `docs/SECURITY.md` | Red-team analyst |
| 7 | Red team | 0 open high-severity attacks | Red-team analyst |
| 8 | Replication | third-party statistics within tolerance | Replicating party |
| 9 | Domain pilot | owner sign-off recorded | Domain owner |
| 10 | Mutation score | at least 80 on the configured scope including custody, charter and vault | CI; non-author approves config |
| 11 | Release controls | rehearsal reviewed; `verify_release_review.py` exits 0 | Independent reviewer |
| 12 | Docs, API, citation | checklist items 4-7 done | Maintainer other than author |
| 13 | Citations | R1 group E closed or listed as unverified in the docs | Independent reviewer |

A "no" blocks release, or narrows the release notes to what passed. A
threshold changes only by a dated amendment committed before the hold-out
corpora are opened.

## Evidence required

Tagged protocol; corpus manifest with hashes; raw backtest results with
SHA-256 checksums; red-team report; replication record; pilot sign-off;
differential-test logs including R fixtures; `*_certificates.json`;
`mutation_survivors.json`; preflight output; the release review record; R1 D1-D4
artifacts.

## Risks

| Risk | Mitigation |
|---|---|
| Too few real labelled defects | Count and licence-check first (6.2); narrow the claim to families covered |
| Too few datasets with a time axis for the backtest | Report "unsupported"; do not relax the inclusion rule after results |
| Backtest confounded by dataset difficulty | Strata and adjusted effect (F7 W1) |
| Independent parties unavailable | Release stays open; the author does not substitute (matches `docs/RELEASING.md`) |
| First run of `release.yml` fails on a step never exercised (OIDC publishing, `pypi` environment approval) | Dry run on TestPyPI is not configured (unverified); rehearse the workflow up to the publish step on a throwaway tag in a fork |
| Version `0.1.0b1` reuse confuses readers of the changelog | Decide in item 1 and say so in the changelog |
| CI time grows (Windows 15-17 of 20 minutes) | F7 W7 before adding tests |

## Explicitly out of scope

New detectors, new modalities, hosted services, and any claim beyond the
corpora tested. SOTA model-accuracy results (6.6) make no claim about the
qualification instrument.
