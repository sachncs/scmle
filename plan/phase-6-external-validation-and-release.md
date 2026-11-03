# Phase 6: External validation and release (F7 complete)

Canon: [`00-canon.md`](00-canon.md). Feature: `product/features/F7-*.md`.

## Goal

Show the instrument is right using evidence that is neither the author's nor
the user's (P1): real labelled defects, time, outsiders and reference
implementations. Then release through the existing evidence chain.

Verified starting point:
- The release pipeline already refuses to publish without tag/version/changelog
  agreement, full tests, a benchmark regression gate, reproducible artifacts and
  an external digest check (`.github/workflows/release.yml:16-106`,
  `docs/RELEASING.md`).
- Two controls need people: an operational rehearsal reviewed by someone other
  than the author, and a signed independent review checked by
  `scripts/verify_release_review.py` (`docs/RELEASING.md`).
- `benchmarks/sota/PROTOCOL.md` is a committed, tagged pre-registration with a
  hold-out opened once, amendments dated before the hold-out opens, and stop
  rules; the same discipline is reused here.
- Mutation gate: `scripts/check_mutation_score.sh` (default minimum 70).

## Entry criteria

Phases 3, 4 and 5 merged; certificates exist for every gating detector; the
validation protocol (below) is committed and tagged `qualification-protocol-v1`
before any backtest result exists.

## Work packages (one agent each unless a person is named)

| WP | Scope | Creates / modifies | Depends on | Definition of done |
|---|---|---|---|---|
| 6.1 | Protocol: claim, corpora list, thresholds, stop rules, amendment rule (cannot change thresholds or corpus assignment after results) | `benchmarks/qualification/PROTOCOL.md` (proposed) | none | Tagged before results; mirrors `sota/PROTOCOL.md` structure |
| 6.2 | Corpora: acquire datasets with independently established defects (human-adjudicated label errors, documented leakage cases such as Titanic `boat`/`body`, published leakage post-mortems); manifest with licence, source URL, SHA-256 | `benchmarks/qualification/corpora.json` (proposed) | 6.1 | >= 15 datasets, >= 5 per defect family, licences cleared, hashes verified, hold-out corpora assigned by hash like `sota/split.py` |
| 6.3 | Predictive-validity backtest: across historical datasets, does verdict `block`/`warn`/`pass` predict out-of-time generalization gap? | `benchmarks/qualification/backtest.py` (proposed) | 6.1, 6.2 | One-sided test that gap(block) > gap(pass), reported with interval; run once on the hold-out corpora |
| 6.4 | Red-team round: an independent analyst tries to push a false finding through (p-hack the declared multiverse, split manipulation, peek at `C`) | `benchmarks/qualification/redteam/` log (proposed) | Phase 3, 5 | Written report; every successful attack becomes a regression test; a fixed attack budget (hours) set in advance |
| 6.5 | Independent replication: a third party reruns the qualification campaign from the tag on fresh machines using the protocol | replication record (proposed) | 6.1, 6.3 | Their summary statistics match ours within the pre-stated tolerance, or the gap is explained in an amendment |
| 6.6 | Domain-owner pilot: run on one real dataset from a team that owns it; owner reviews provenance table, admissibility, mechanisms | pilot record (proposed) | Phase 4 | Signed record that each mechanism is plausible or flagged; flags become issues |
| 6.7 | Reference cross-checks: differential tests against R (`onlineFDR`, `knockoff`) and scipy/statsmodels for every p-value producer; mutation run | `tests/` additions; `scripts/check_mutation_score.sh` | Phases 3-5 | Agreement within 1e-8 on fixed inputs (exact tests) or the stated Monte Carlo tolerance; mutation score >= 80 on qualification modules |
| 6.8 | Release mechanics: version bump, `CHANGELOG.md` section, `tests/fixtures/public_api.json` regeneration, `docs/API_STABILITY.md`, new docs pages, `release_preflight.sh` | per `docs/RELEASING.md` | 6.3-6.7 | Preflight passes; tag matches version and changelog |

## Parallel lanes

6.1 first. Then 6.2, 6.4, 6.6 and 6.7 run in parallel; 6.3 follows 6.2; 6.5
follows 6.3; 6.8 last. People-bound items (6.4, 6.5, 6.6) are scheduled at
phase start because they have lead time.

## Exit gate

```bash
uv run python -m benchmarks.qualification --check --family all
uv run python -m benchmarks.qualification.backtest --protocol qualification-protocol-v1
./scripts/check_mutation_score.sh 80
./scripts/release_preflight.sh
python scripts/verify_release_review.py review.json --source-commit <sha>
```

## Go / no-go checklist (all must hold; the signer is never the author)

| # | Criterion | Threshold | Signer |
|---|---|---|---|
| 1 | Null size, every gating detector | FPR <= alpha inside 99% band over >= 1000 reps | Independent statistician |
| 2 | Recall on planted defects | Wilson 95% lower bound >= 0.80 per type (leakage >= 0.90) | Independent statistician |
| 3 | Real-corpus recall | >= 0.70 on hold-out corpora, with interval | Independent reviewer |
| 4 | Predictive validity | gap(block) > gap(pass), one-sided p < 0.05 | Independent statistician |
| 5 | Final-claim FDR | empirical FDR <= alpha_confirm + 0.01 on semi-synthetic | Independent statistician |
| 6 | Access control | 0 successful reads of `C`/`F` from exploration code in red-team and tests | Red-team analyst |
| 7 | Red team | 0 open high-severity attacks | Red-team analyst |
| 8 | Replication | third-party stats within tolerance | Replicating party |
| 9 | Domain pilot | owner sign-off recorded | Domain owner |
| 10 | Mutation score | >= 80 on qualification modules | CI (non-author reviewer approves config) |
| 11 | Release controls | rehearsal reviewed; `verify_release_review.py` exits 0 | Independent reviewer |
| 12 | Docs/API | changelog section, API fixture, docs updated | Maintainer other than author |

Any "no" blocks release; a threshold is changed only by a dated amendment
committed before the hold-out corpora are opened.

## Evidence required

Tagged protocol, corpus manifest with hashes, raw backtest results with
SHA-256 checksums, red-team report, replication record, pilot sign-off,
differential-test logs, `detector_certificates.json`, preflight output.

## Risks

| Risk | Mitigation |
|---|---|
| Too few real labelled defects | Count and licence-check first (6.2); shrink claims to the families covered |
| Backtest outcome ambiguous (confounded by dataset difficulty) | Pair datasets by size/difficulty; report matched-pair interval |
| Independent parties unavailable | Release stays open; do not substitute the author (matches `RELEASING.md`) |
| Corpus defects discovered later | Amendment, not silent edit |

## Explicitly out of scope

New detectors, new modalities, SOTA model-accuracy claims (that is
`benchmarks/sota`), hosted services, and any claim beyond the corpora tested.
