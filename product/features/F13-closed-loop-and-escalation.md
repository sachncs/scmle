# F13: Closed loop and escalation

Vocabulary and principles: [`../../plan/00-canon.md`](../../plan/00-canon.md).
Built in [Phase 13](../../plan/phase-13-closed-loop-and-escalation.md). Needs
F2 (custody), F3 (ledger, e-value machinery), F12 (calibrated sets and
coverage claims) and the F7 harness. As with F12, adding it to the canon's
feature table is a separate edit.

Why this file exists: a data scientist responsible for a deployed model
notices drift, decides whether it matters, retrains, checks the new model
before swapping it in, takes it back if it was wrong, knows that only some
cases ever get a label, and knows when to say "this one is not mine to
decide". SCMLE had most of the plumbing and none of the measured decision
rules. This file records what is built, what it measured, and what is left.

## Purpose

(a) Run the production loop (monitor, diagnose, retrain, compare, promote,
roll back) with decision rules whose false-alarm, false-promotion and
detection-delay characteristics are measured on streams with known truth, and
with performance estimates that stay valid when labels are late, partial or
selected. (b) Decide, by a calibrated and pre-registered rule, when the
system stops and asks a human, and record the human's reply in a ledger as an
`attested` entry.

## Decision it serves

Which actions the system may take without a person (retrain a challenger,
stage it, answer a case) and which it must hand over (move the serving
model, answer an uncertain case, act on a stream with an unverified label
mechanism). The honest alternative to "replace the data scientist" is a
measured boundary of what is left to one.

## What is built

The methods are documented where they live (module docstrings), with the
gates in `benchmarks/qualification/loop_certify.py`:

| Need | Where | Measured (simulated streams) |
|---|---|---|
| L1 alarm on realised risk; covariate drift demoted to diagnosis | `monitoring/sequential_tests.py` (`RiskAlarm`, `MixtureTest`) | H1 passed, H3 passed, **H2 missed** |
| L2 gated promotion: paired e-process, e-LOND across challengers, decided in registration order | `sequential_tests.py` (`PairedEProcess`, `PromotionSequence`), `governance/loop.py` | H4 passed (trivially in the world measured) |
| L3 rollback: reverse e-process; `tau` tightened | same, `Loop.watch_rollback` | H5 passed (false rollback 0; power only at a loss gap of 0.1) |
| L4 delayed, partial, selected labels: maturity by prediction time, Horvitz-Thompson loss, audit slice, unverified path | `monitoring/labels.py`, `ModelMonitor.performance` | H6, H7 passed |
| L5 feedback: bypass-slice comparison | `labels.bypass_test`, called by `Loop.check_feedback` (next-period features supplied by the caller) | H8 passed |
| E1 selective prediction: fixed-sequence threshold, risk-coverage curve, Chow cross-check, capacity routing, halt | `governance/escalation.py`, `Loop` | H9 passed, **H10 missed** |
| E2 system-level escalation triggers | `Loop.decide` | unit tests only |
| E3 attestation rows, reply protocol, timeouts | `governance/deployment_ledger.py`, `Loop.reply` | unit tests, ledger tamper tests |
| Never-list: eleven guards | `governance/never_list.py`, `Loop` wrappers, and a registry with an attached guard (`LifecycleManager` promotions and rollbacks) | a test per item (11 of 11); items 1 and 2 also stay behind the registry's signed approval and the vault's one-time open (tested) |
| Fresh retraining rows held out by time or group | `governance/lifecycle.split_fresh` | unit tests |

Artifacts: `deployment_ledger.jsonl`, `loop_state.json`, `risk_coverage.json`,
`label_mechanism.json` (`docs/ARTIFACTS.md`). `escalation_audit.json` is not
written (it needs the owner's audit).

## Open gates

| # | Quantity | Operating characteristic to measure | Pass criterion | External referent |
|---|---|---|---|---|
| H2 | L1 detection delay | Rises of `1.5`, `2`, `3` times `Delta` at the H1 false-alarm level. Measured: median 183, 76, 33 batches. | Detection of a `2 Delta` change within the charter's `D_max` with probability `>= 0.90` (lower Clopper-Pearson bound). Missed at `D_max = 50`; the alarm does not gate a retrain until a re-registered design passes. | Known change point in the simulator. |
| H10 | E1 under shift | Selective risk after a rise of the answered cases' risk. Measured: 13 percent within `D_max` at `2 r*`. | Detection within `D_max` with probability `>= 0.90`; coverage 0 at `halt` (passes). | Simulator truth. |
| H11 | Escalation vs a domain owner | Replay of recorded incident-like sequences. A domain owner labels each time step "a human should have been asked: yes / no / either". | Recall of owner-`yes` steps `>= 0.90` (lower bound) at an escalation rate `<= c_max`; Cohen's kappa with interval; for a held-out incident set fixed before any tuning, recall `>= 0.80` at the same rate. | The owner's signed audit (below). |

### Replay of real incident-like sequences

Existence, licence and hash of every item are **candidates, to be verified**:
public streams with a documented regime change (electricity-price and airline
concept-drift benchmarks, Wilds-style distribution-shift datasets; names from
memory, **unverified**); and, where available, a domain owner's own incident
tickets with timestamps. Each replay is fed to the loop with labels delayed
according to the stream's documented process. Metrics: time from the
documented change to the first alarm, number of alarms in documented-stable
periods, number of unnecessary retrains, and whether any never-list action was
taken. Held-out: a subset of sequences fixed by hash before any threshold is
set (the SOTA protocol's rule, `benchmarks/sota/PROTOCOL.md`, applies).

### External referent that is neither author nor user

1. **A domain owner's recorded escalation audit.** A named person with
   authority over the stream's domain, not the author and not the person
   configuring the system, retroactively marks each time step of the replay
   sequences with whether a human should have been asked, and signs the set
   in the section 5 format of `validation-referents.md` (evidence named,
   scope = stream hash, expiry). Without a second independent auditor the
   result is `attested`, not `confirmed`; with two, inter-rater kappa is
   reported and disagreements are listed rather than adjudicated by us.
2. **A held-out sequence of incidents**, defined before tuning by hash, never
   used to set thresholds.
3. Reference implementations: R `onlineFDR` for LORD++/SAFFRON levels used by
   e-LOND-adjacent comparisons (R is not installed in CI; outputs frozen
   offline), and an independent confidence-sequence implementation for the
   betting bound (library and version to be verified).

## Guarantees and assumptions

- L1/L2/L3 error rates hold at any stopping time if the matured-case batch
  means satisfy the betting construction's assumptions (bounded observations,
  conditional mean at most the null's). They are about the matured, weighted
  stream, not about the future. After an alarm the monitor restarts, so the
  bound is for the first alarm of an episode.
- Challengers are decided in registration order (e-LOND needs the order not
  to depend on the evidence); a challenger that never reaches its level
  blocks the ones behind it until its label budget ends.
- L4 is unbiased only if the logged `pi` is correct and positive
  (`pi >= epsilon`); the uniformity test checks the audit slice's rate, not
  its independence from the outcome. Wide ranges (`1 / epsilon`) cost power.
- E1's risk bound holds under exchangeability of calibration and deployment
  cases and 0/1 losses; the weighted version is an approximation. The
  selective-risk alarm is the guard when that fails; it detects with delay, so
  there is a window of unbounded excess risk before `halt`. H10 measures it;
  it is not zero.
- Escalation recall against the owner's audit is conditional on the owner's
  judgement being the right referent; it does not prove the owner is right.
- Feedback loops are detectable only with a randomized bypass slice.
- The guards are code around the entry points of `Loop`. They cannot stop a
  person or process with file or interpreter access, and key custody for the
  human's signature is outside the software's control (P6): HMAC is
  symmetric, so the verifier's key must stay where the system cannot read it.

## Failure modes and degradation

| Situation | Behaviour |
|---|---|
| Too few matured labels | `insufficient_labels`; no retrain, promotion or rollback. |
| Label mechanism `unverified` | No retrain, promotion or rollback; an alarm on the naive labelled loss halts and escalates. |
| Calibration set too small for `r*` at `delta` | `tau = +infinity` (everything escalates). |
| Human unavailable | The escalation closes at `reply_timeout` with the proposed default; the outcome is `unverified`. |
| Ledger verification fails, or state does not match the ledger | Halted; no action until a person reviews. |
| Any halt | An `incident` row (status `open`) is written to the ledger; `Hooks.open_incident` is called after it. |
| Replay corpus unavailable | H11 and replay metrics do not run; the loop stays `unverified` for production. |
| Challenger evidence never reaches threshold | No promotion; an escalation at the label budget. |

## Verification status of literature claims

Read in full on 2026-11-05 (`docs/LITERATURE_CHECK.md`): arXiv:2010.09686,
arXiv:1705.08500, JMLR 11 (2010) El-Yaniv and Wiener, arXiv:2110.01052 ("Learn
then Test"). Two claims were corrected: the plug-in bet size is the paper's
empirical-Bernstein plug-in, used here as a one-sided adaptation, and "Learn
then Test" assumes an i.i.d. calibration set. **Still unverified:** Begg and
Greenes 1983; Chow's rule (1970; only a secondary description was read); every
dataset named in the replay section; confidence-sequence library availability.

## Cut, and why

Drift-statistic-driven retraining (no certified link to loss); fixed-sample
bootstrap promotion at each review (repeated-look inflation); adaptive
conformal and conformal test martingales for label-free exchangeability
monitoring (each needs its own certificate; label-free detection stays at
`investigate`); contextual bandit or off-policy evaluation machinery beyond
Horvitz-Thompson with a logged randomized slice; learned deferral models
(learning to defer); automatic cost-ratio elicitation; self-adjusting
thresholds.
