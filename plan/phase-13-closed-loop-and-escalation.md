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

## Status

Built 2026-11-04 as library code, a command (`scmle loop`) and simulated
gates (`benchmarks/qualification/loop_certify.py`, `loop_certificates.json`,
`python -m benchmarks.qualification --check loop`): the stream simulator
(`streams.py`), the deployment ledger (`governance/deployment_ledger.py`),
the betting alarm, paired e-processes and e-LOND across challengers
(`monitoring/sequential_tests.py`), label maturity, Horvitz-Thompson losses
and the audit and bypass slices (`monitoring/labels.py`), selective prediction
with a fixed-sequence threshold, risk-coverage curve, Chow cross-check and
capacity routing (`governance/escalation.py`), the eleven never-list guards
(`governance/never_list.py`) and `Loop` (`governance/loop.py`). Fresh
retraining rows are held out by time or group (`lifecycle.split_fresh`) and
`ModelMonitor.performance` takes `propensity` and `maturity`. Nothing below
was run on real data.

Gates that passed on simulated streams (2000 replicates unless stated): H1
(L1 false alarm 0 at the stable risk and 0.0075 on the boundary of the null;
the existing fixed rule alarmed in every stream), H3, H4 (e-LOND false
discovery rate 0, power 1.0 in a strong world; the per-review bootstrap
0.61), H5, H6 (300 runs), H7, H8 (100 runs), H9 (1000 runs: violation 0.076
against 0.119; the best-observed-risk cut 0.464).

## Published misses (not retuned)

- H2: a rise of the loss to `r_ref + 2 Delta` is found within the
  pre-registered `D_max = 50` batches in 0.35 percent of streams (median
  delay 76 batches; 33 for `3 Delta`). So the L1 alarm does not gate a
  retrain: `LoopPolicy.alarm_gates_retrain` defaults to False and an alarm
  records an `alarm` row and escalates (`investigate`). The cost is the
  betting bound's range (batch means in `[0, 1]`) and the mixture over
  restarts (`2 log(t / stride)` nats).
- H10: the selective-risk alarm finds a rise to `2 r*` within `D_max` in 13
  percent of streams (median 57 batches, about 2850 answered cases before the
  halt); `3 r*` is found in a median 25 batches. Coverage after the halt is 0
  (tested).
- H4 is satisfied trivially in the world measured (FDR 0.0): its power
  needs an advantage of 0.2 in loss. A weaker world has power near zero
  because e-LOND's levels shrink with the index; that is the price and is
  not measured here.

## Work packages still open

**WP-13.0 Reference reading** (one agent)
- Read Begg and Greenes (1983) and Chow (1957, 1970); the other sources of
  `sequential_tests.py` and `escalation.py` were read on 2026-11-05
  (`docs/LITERATURE_CHECK.md`). Replace the remaining "from memory" in
  `labels.py` and the Chow note in `escalation.py` with a citation.
- Pin an independent implementation of the betting bound (library and
  version) or derive a brute-force reference for a differential test (the
  unit tests compare only with a plain loop of the same recurrence).
- Verify existence, licence and SHA-256 of each replay stream; keep as
  `candidate` in `corpora.json` until then.

**WP-13.3a Detection delay**
- A re-registered design for the L1 and selective alarms (tighter bounds on
  batch means, per-case instead of per-batch observations, or a charter
  `D_max` the settings can meet) with a new pre-registration. Until H2 passes
  `alarm_gates_retrain` stays False.

**WP-13.5a Escalation inputs**
- Feed `Loop.observe` the F12 class-conditional set size and the novelty flag
  from a bundle (`set_size`, `novel` columns are supplied by the caller
  today), and give the out-of-support bound `novelty_bound` a certified
  value (it is a charter input, not a measured one).
- Weighted threshold selection uses Kish effective sizes and a rounded-up
  error count; replace it with an exact or conservative weighted test and
  certify it (H9 measured the unweighted case only).

**WP-13.6a Reach of the guards**
- The custody API is not routed through the guards (documented in
  `never_list.py`). An attestation stays valid for the identical question
  until it expires (no single-use marker).
- `Loop.check_feedback` needs next-period features from the caller; `step`
  does not supply them. `escalation_audit.json` is still not written (H11).

**WP-13.7 Replay and domain-owner audit** (one agent plus one external person)
- Hash-fix the held-out incident set before any tuning; run replays with
  documented delays; obtain the owner's signed audit (section 5 format of
  `validation-referents.md`); compute recall, escalation rate, kappa; publish
  misses.
- Done when: H11 is reported in `escalation_audit.json` with the owner's
  record hash, or the policy is declared `unverified` in the report (today it
  is).

**WP-13.8a Wiring left**
- A halt writes an `incident` ledger row only; connecting it to
  `monitoring/incidents.py` needs an SLO evaluation, which a halt does not
  have.
- Run the ledger replay on Python 3.11 (only 3.14 ran here).
- Include `loop_certificates.json` in the release evidence index.

## Exit gate

```bash
uv run pytest -q tests/unit tests/leakage -k "loop or escalation or sequential or deployment_ledger or label_mechanism or never_list"
uv run python -m benchmarks.qualification --check loop
```

Open numbers: H2 and H10 at their pre-registered `D_max`; H11 recall of
owner-`yes` steps `>= 0.90` (lower bound) at an escalation rate `<= c_max`,
held-out incident recall `>= 0.80`; ledger replay bit-identical on 3.11 and
3.14.

## Risks

- The owner's audit is subjective and costly; a single auditor yields only
  `attested`. Ask for two; report kappa; list disagreements.
- Replay corpora may not exist in verified form. Then H11 and replay stay
  unmet and the policy is `unverified`; no gate is faked.
- Audit and bypass slices cost real outcomes. They are a business decision;
  without them the stream degrades to `unverified`, which is the honest
  outcome.
- Sequential tests are conservative (wider intervals, longer delay); the
  delay is measured and reported, thresholds are the charter owner's, not the
  system's.
- The window before `halt` has unbounded excess risk under shift; it is
  measured (H10), not eliminated.
- Key custody for human signatures cannot be enforced by the software (P6).
  HMAC is symmetric: whoever can verify a reply can also forge it.
- The betting bound's assumptions on dependence across cases inside a batch
  may not hold; batches are the unit of observation for that reason.

## Explicitly out of scope

Autonomous promotion without a signed approval; changing the registry's
signature rule (`registry.py:132-133`); contextual bandits and off-policy
evaluation beyond Horvitz-Thompson with a logged slice; learned deferral;
label-free exchangeability martingales; causal effect estimation of the
model's decisions beyond the bypass-slice comparison; the HTML report;
forecasting and survival specifics (Phase 12).
