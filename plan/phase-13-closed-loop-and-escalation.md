# Phase 13: Closed loop and escalation (F13)

Names follow [`00-canon.md`](00-canon.md). Feature spec:
[`../product/features/F13-closed-loop-and-escalation.md`](../product/features/F13-closed-loop-and-escalation.md).

## Goal

The deployed loop (monitor, diagnose, retrain, compare, promote, roll back)
runs on decision rules with measured false-alarm, false-promotion and
detection-delay characteristics; performance estimates stay valid under
delayed, partial and selected labels or are tiered `unverified`; and a
pre-registered selective-prediction rule decides when the system stops and
asks a human, with the human's reply recorded in a ledger as an `attested`
entry (P1, P4, P5).

## Entry criteria

- Phase 3 merged: `scmle/data/ledger/online_fdr.py` (`ELond` at `:356`),
  `ledger.py` (hash chain, replay).
- Phase 2 custody (one-time open: `scmle/data/custody/sealed.py:170-174`).
- Phase 12 WP-12.1 and 12.2 merged (class-conditional sets, coverage report)
  for the escalation score; WP-12.1 may be stubbed with the existing
  `ConformalCalibrator` to start L1-L4.
- Phase 1 harness (`benchmarks/qualification/simulate.py`, `oracles.py`).

## Verified starting points

- Fixed drift and degradation thresholds, no certificates:
  `scmle/monitoring/monitor.py:23-41`, `:171-290`.
- Performance on labelled rows only, unweighted: `monitor.py:353-405`.
- Retrain on a random shuffle of fresh rows: `scmle/governance/lifecycle.py:290-292`.
- Bootstrap promotion rule, applied at each review: `scmle/monitoring/champion.py:169`.
- Hard human gate on the champion alias, in code: `scmle/governance/registry.py:367-380`;
  rollback executes only with approval: `lifecycle.py:379-391`.
- Incidents and SLOs: `scmle/monitoring/incidents.py:262`, `slo.py:153`.
- Ledger row kinds are closed at seven, unknown kinds raise, nothing may follow
  `confirmation`: `scmle/data/ledger/ledger.py:67-73`, `:477-485`.
- `Tier.ATTESTED` exists (`scmle/contracts/qualification.py:25-30`) and nothing
  in `data/ledger/` produces it.
- No abstention, no risk-coverage code, no label-selection propensities
  (grep over `scmle/`; F13 table).

## Work packages

**WP-13.0 Reference reading** (one agent; blocks 13.3, 13.5)
- Read, not only abstracts: Waudby-Smith and Ramdas (2010.09686) for the
  assumptions on the loss sequence and the exact confidence-sequence and
  e-process statements; Geifman and El-Yaniv (1705.08500); El-Yaniv and
  Wiener (JMLR 2010); locate "Learn then Test"; Begg and Greenes; Chow.
- Pin an independent implementation of the betting bound for differential
  tests (library and version), or derive a brute-force reference.
- Verify existence, licence and SHA-256 of each replay stream; keep as
  `candidate` in `corpora.json` until then.
- Done when: every "unverified" in F13 is resolved with a docstring citation
  or listed in an appendix.

**WP-13.1 Stream simulator with truth** (one agent; start immediately)
- `benchmarks/qualification/streams.py`: abrupt, gradual, covariate-only and
  concept shift; label delay (outcome-independent and outcome-dependent);
  labelling rules with logged `pi(x)` (MAR) and outcome-dependent (MNAR);
  audit and bypass slices; feedback loop; dependent challenger sequences.
  Returns the true change points, true risk and true labelling mechanism.
- Done when: simulator self-tests recover known risks to MC error; seeds
  are recorded.

**WP-13.2 Deployment ledger and attestation** (one agent; parallel with 13.1)
- `deployment_ledger.py` reusing the chain and replay code of `ledger.py`;
  kinds `open`, `observe_summary`, `alarm`, `retrain`, `challenger_result`,
  `promotion_proposal`, `rollback_proposal`, `escalation`, `attestation`,
  `halt`. `attestation` rows carry the section 5 fields of
  `plan/validation-referents.md`, bound to `question_hash`, with expiry; the
  system's own key is refused.
- Done when: tamper tests (delete, reorder, edit) fail verification; replay
  bit-identical on Python 3.11 and 3.14.

**WP-13.3 Sequential monitors** (one agent; needs 13.0, 13.1)
- `monitoring/sequential_tests.py`: betting confidence sequence for bounded
  loss, e-process for paired differences, reverse e-process for rollback;
  wire e-LOND across challengers. Keep the existing rules as baselines under
  a flag.
- Replace the shuffle at `lifecycle.py:290-292` with F2's out-of-time splitter
  for temporal and grouped data.
- Done when: H1-H5 certificates written; existing rules' rates on the same
  streams recorded.

**WP-13.4 Label mechanism** (one agent; needs 13.1; parallel with 13.3)
- `monitoring/labels.py`: maturity by prediction time, `outcome_window`
  charter field, logged propensity and audit slice, Horvitz-Thompson loss,
  range rescaling for the sequence; `unverified` path with the negative-
  control test (no action on MNAR without audit slice); bypass-slice
  feedback test; randomization-log uniformity check.
- `ModelMonitor.performance` gains optional `propensity` and `maturity`
  (`monitor.py:353`).
- Done when: H6, H7, H8.

**WP-13.5 Selective prediction and escalation** (one agent; needs 13.0, 13.3, Phase 12.1)
- `governance/escalation.py`: score from class-conditional set size plus
  novelty flag; fixed-sequence threshold selection with exact binomial
  p-values; `tau` recalibration; `halt`; capacity budget with safe default
  (never raises `tau`); Chow cross-check; risk-coverage curve and AURC to
  `risk_coverage.json`.
- Done when: H9, H10.

**WP-13.6 Never-list enforcement and reply protocol** (one agent; needs 13.2)
- A guard on each of the 11 listed actions (F13 "Never unattended"): raises
  without a valid attestation, writes an `escalation` row. Tests for each,
  including the two already enforced (`registry.py:367-380`, custody).
- `Loop.reply` verifies signature, `question_hash`, expiry, scope.
- Done when: acceptance 7 and 9 pass.

**WP-13.7 Replay and domain-owner audit** (one agent plus one external person; needs 13.3-13.5)
- Hash-fix the held-out incident set before any tuning; run replays with
  documented delays; obtain the owner's signed audit (section 5 format);
  compute recall, escalation rate, kappa; publish misses.
- Done when: H11 reported in `escalation_audit.json` with the owner's record
  hash, or the policy is declared `unverified` in the report.

**WP-13.8 Wiring** (one agent; last)
- `scmle` CLI entry for `loop` (observe, step, pending, reply); incident
  workflow opened on `halt`; `loop_certificates.json` and `--check loop`.

## Parallel lanes

```
13.0 --> 13.3 --\
13.1 --> 13.3    >--> 13.5 --> 13.7 --> 13.8
13.1 --> 13.4 --/
13.2 --> 13.6 ------------------^
```
13.1 and 13.2 can start on day one; 13.0 runs alongside them.

## Exit gate

```bash
uv run pytest -q tests/unit tests/property tests/leakage -k "loop or escalation or sequential or deployment_ledger or label_mechanism"
uv run python -m benchmarks.qualification --check loop      # proposed flag
uv run python -m benchmarks.reliability --only drift_detection
```

Numbers, at `R = 2000` replicates per world unless stated:
- H1: P(any false alarm over `T = 5000` batches) `<= alpha_alarm +
  2*sqrt(alpha_alarm*(1-alpha_alarm)/R)`, about 0.0597 for 0.05. The existing
  fixed-threshold rule's rate on the same streams is in the certificate.
- H2: P(detect a `2 Delta` change within `D_max`) lower bound `>= 0.90`, else
  the alarm does not gate a retrain.
- H3: retrain rate on covariate-only shift `<= alpha_alarm + tolerance`.
- H4: realised false-promotion FDR over 200-challenger sequences
  `<= alpha_promo + tolerance`; per-review bootstrap shown above it.
- H5: false rollback at no difference `<= alpha_rb + tolerance`.
- H6: bias of the weighted risk within MC error for MAR with logged `pi`;
  sequence coverage `>= 1 - alpha - tolerance`; MNAR without audit slice ->
  zero retrain/promote/rollback actions, tier `unverified`.
- H9: `P(R(tau_hat) > r*) <= delta + tolerance`.
- H10: coverage is 0 after `halt`; pre-halt window reported.
- H11: recall of owner-`yes` steps `>= 0.90` lower bound at escalation rate
  `<= c_max`; held-out incident set recall `>= 0.80` at the same rate.
- Never-list: 11 of 11 guard tests pass.
- Ledger replay bit-identical on 3.11 and 3.14.

`alpha_*`, `Delta`, `Delta_promo`, `Delta_rb`, `r*`, `delta`, `c_max`, `D_max`
are charter inputs fixed before the run; the numbers above are the only
author-chosen gate values and they are statistical tolerances, not
performance claims.

## Evidence required

`loop_certificates.json` (with simulator and code hashes), the baseline-rule
comparison tables, `risk_coverage.json` for a seeded stream, a deployment
ledger from a replay with at least one `escalation` and one `attestation`
row, the owner's signed audit record and `escalation_audit.json`, the 11
guard-test outputs, import-graph evidence that escalation code cannot import
the custody `C` open API.

## Risks

- The owner's audit is subjective and costly; a single auditor yields only
  `attested`. Mitigation: ask for two; report kappa; list disagreements.
- Replay corpora may not exist in verified form. Mitigation: simulation
  gates run regardless; H11 and replay stay unmet and the policy is
  `unverified`; no gate is faked.
- Audit and bypass slices cost real outcomes (labels, bad decisions). They
  are a business decision; without them the system degrades to
  `unverified`, which is the honest outcome.
- Sequential tests are conservative (wider intervals, longer delay).
  Mitigation: delay is measured and reported; thresholds chosen by the
  charter owner, not by the system.
- The window before `halt` has unbounded excess risk under shift. It is
  measured (H10), not eliminated.
- Key custody for human signatures cannot be enforced by the software.
  Stated in the report (P6).
- Betting-CS assumptions on the loss sequence (for example dependence across
  cases within a batch) may not hold. WP-13.0 reads them; batches are
  treated as the unit if needed.

## Explicitly out of scope

Autonomous promotion without a signed approval; changing the registry's
signature rule (`registry.py:132-133`; any delegated-approval scheme is a
separate decision for a human owner); contextual bandits and off-policy
evaluation beyond Horvitz-Thompson with a logged slice; learned deferral;
label-free exchangeability martingales; causal effect estimation of the
model's decisions beyond the bypass-slice comparison; the HTML report;
forecasting and survival specifics (Phase 12).
