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
decide". SCMLE has most of the plumbing and none of the measured decision
rules. This file records what exists (verified), what is missing, and one
method per need, with a measured gate and an external referent each (P1).

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

## What exists today (verified)

| Piece | Where | What it does and does not do |
|---|---|---|
| Drift thresholds | `scmle/monitoring/monitor.py:23-41` | Fixed constants: PSI 0.1/0.25, 30% of features, unseen-category rate 0.2, 30 labelled rows, 10% relative degradation. No certificate for any of them. |
| Drift statistic | `scmle/monitoring/reference.py:27` (`population_stability_index`), `:77-94` (`noise_floor`, `adjusted_psi`) | Binned PSI with a sampling-noise floor. |
| Advisory rules | `monitor.py:171-271`, `:274-290` (`advise`) | `PerformanceRule` retrain on degradation; `FeatureDriftRule` retrain when >= 30% of features drift strongly; output and unseen-category rules investigate. Strongest action wins. |
| Delayed labels | `monitor.py:353-405` | Missing outcomes mean "label not arrived"; metrics are computed on the labelled subset **unweighted** (`:364-391`) and `labeled_fraction` is reported (`:367`). With selected labels this is a biased estimate; nothing detects or corrects that. |
| Retrain trigger | `scmle/governance/lifecycle.py:241-326` (`review_locked`), `:24-35` (`RetrainPolicy`: `min_rows = 200`, `cooldown_rows = 500`) | Retrains on advice. The challenger is fit and tested on a **random shuffle** of the fresh rows (`:290-292`, `HOLDOUT_FRACTION = 0.3` at `:19`), not an out-of-time split, whatever the data's structure. |
| Champion vs challenger | `scmle/monitoring/champion.py:72-170` | Paired bootstrap (group- or block-resampled, `:110-120`, `:147-150`); promote when the lower end of the 95% interval is above zero (`:169`). A fixed-sample test applied again at every review: repeated looks inflate the false-promotion rate, unmeasured. |
| Shadow and A/B | `scmle/monitoring/shadow.py:37`, `abtest.py:99` | Disagreement counts and settled comparison; A/B verdict `adopt`/`keep`/`inconclusive`. Fixed-sample intervals. |
| Promotion gate | `scmle/governance/registry.py:326-393`; approval type `:40-52`, `:114-135` | Moving the `champion` alias raises unless a signed, independent, identity-bound `PromotionApproval` is supplied (`:367-380`). This is an existing hard human gate, enforced in code. |
| Rollback | `lifecycle.py:328-405`, `RollbackPolicy` `:39-54` | Triggers when the whole 95% interval of the new champion's advantage lies below minus the tolerance (default 0.01); executes only with an approval (`:379-391`); otherwise audits `rollback_pending_approval`. |
| Incidents and alerts | `scmle/monitoring/incidents.py:35-40`, `:262-296`; `slo.py:41`, `:153`; `alerts.py` | SLO breach opens a durable incident with owner, runbook and rollback target. Operational metrics (latency, errors), not statistical decision rules. |
| Novelty | `scmle/inference/novelty.py:31` | Isolation-forest plus far-out range fences on incoming rows. A flag, no measured operating characteristic. |
| Online-FDR machinery | `scmle/data/ledger/online_fdr.py:174-196` (`Evidence`, p or e), `:238` (`calibrate`), `:356` (`ELond`), `:558` (`RuleConfig`) | Exact-arithmetic state machines; e-LOND controls FDR under arbitrary dependence if each true null has `E[e] <= 1` and hypotheses are indexed in advance (docstring `:20-31`, read from the paper by the previous author on 2026-11-03). |
| Hash-chained ledger | `scmle/data/ledger/ledger.py:67-73` (row kinds), `:453-497` (`ingest`), `:477` ("no row may follow the confirmation"), `:478-485` (handler table; an unknown kind raises) | Seven kinds only: open, register, result, control, diagnostic, survivors, confirmation. No `attestation` kind. The ledger is closed by `confirmation`, so a production loop needs its own ledger, not appended rows. |
| Tier | `scmle/contracts/qualification.py:25-30` | `CONFIRMED`, `ATTESTED`, `UNVERIFIED` exist; nothing in `data/ledger/` currently produces `ATTESTED`. |
| Release gates | `scmle/evaluation/release.py:462-622` (reliability, attestation and evidence-index gates, `evaluate_release`); `scmle/governance/attestation.py` (model, provenance, SBOM attestation, not a human reply); `governance/review.py:98-234` (signed independent review) | Signed-document machinery that a human reply can reuse. |
| Abstention / selective prediction | grep for `abstain`, `selective`, `reject option`, `risk-coverage`, `escalat` over `scmle/` | **Absent.** The nearest items are `models/thresholds.py` (a decision threshold, not abstention) and `bundle.py:1509-1526` (ranking by uncertainty for labelling). `core/policies.py:1` ("human (or automated) approval gates in a run") exists as a run-time checkpoint, not a calibrated rule. |
| Label selection / propensities | grep for `propensity` over `scmle/` | Only for treatment assignment in uplift (`evaluation/uplift.py:91-144`) and sequential causal IPW (`evaluation/sequential.py`). Nothing for "which cases got a label". |
| Existing drift gate | `benchmarks/reliability.py:307-327` | One +3 sigma shift flagged, one clean batch quiet. A smoke test, not an operating characteristic. |

## Behaviour

Proposed interface:

```python
loop = Loop.open(deploy_dir, charter, policy)  # own hash-chained ledger
loop.observe(batch, predictions, decisions)  # logs score, route, propensity
loop.mature(labels)  # labels arrive, with delay
state = loop.step()  # may return Escalation(...)
esc = loop.pending_escalations()
loop.reply(esc.id, attestation)  # human reply -> ledger row
```

Artifacts: `deployment_ledger.jsonl` (hash-chained; kinds `open`,
`observe_summary`, `alarm`, `retrain`, `challenger_result`, `promotion_proposal`,
`rollback_proposal`, `escalation`, `attestation`, `halt`), `loop_state.json`,
`risk_coverage.json`, `label_mechanism.json`, `escalation_audit.json`.

## Method (one rigorous method per need)

### L1. Performance alarm: anytime-valid confidence sequence on realised risk

The fixed 10% relative-degradation rule (`monitor.py:393-396`) is replaced
as the **retrain trigger** by a confidence-sequence alarm on the deployed
model's realised loss, using the betting construction for bounded means
(Waudby-Smith and Ramdas, "Estimating means of bounded random variables by
betting", arXiv:2010.09686; abstract read 2026-11-04: confidence intervals
and **time-uniform** confidence sequences from nonnegative martingales; the
formal coverage statement is not on the abstract page and is **unverified**).
Per-case loss `l_t` is clipped to `[0, 1]` (or rescaled by a charter-declared
bound). The alarm fires when the lower limit of the running confidence
sequence for the mean loss exceeds `r_ref + Delta`, where `r_ref` is the
reference risk from the bundle (`reference_metrics`, `monitor.py:368`) and
`Delta` is a charter tolerance in loss units. Because the sequence is valid
at every stopping time, checking at every batch does not inflate the false
alarm probability: `P(any false alarm over all time) <= alpha_alarm`. The
fixed-sample rule in `monitor.py` has no such property and is kept only as
the baseline against which the gate is measured.

Covariate drift (PSI, adversarial validation, `shift.py:26`) is demoted to
**diagnosis**: it ranks suspects and routes to `investigate`, but does not
by itself trigger a retrain, because drift that does not move the loss needs
no retrain and the drift statistics have no certified link to loss (P1).

### L2. Gated promotion: e-process on paired loss differences

For a challenger and the champion scored on the same matured cases, let
`d_t = l_t(champion) - l_t(challenger)` in `[-1, 1]`. Null `H0`: the
challenger is not better by at least `Delta_promo` (`E[d_t] <= Delta_promo`).
A betting e-process `E_t` (nonnegative supermartingale under `H0`) is
updated every batch; propose promotion when `E_t >= 1 / alpha_promo`.
Validity at arbitrary stopping times is the property of the construction
(Ville's inequality; from memory, **unverified**, cited to be located), which
the repeated-look bootstrap in `champion.py:169` lacks.

Challengers arrive over time; each is a hypothesis registered in the
deployment ledger **before** it is scored. The decisions across challengers
use **e-LOND** (`online_fdr.py:356`), which bounds the false-promotion share
under arbitrary dependence between challengers' e-values (the docstring
conditions: `E[e] <= 1` under each true null, hypotheses indexed in
advance). Dependence is the realistic case (successive challengers share
data and lineage). The e-process final value at the stopping time is the
e-value fed to the rule.

The proposal still goes to `LifecycleManager.promote_challenger`
(`lifecycle.py:193-213`) and needs the signed approval (`registry.py:367-380`).
F13 does not weaken that gate; it makes the evidence the signer sees
quantified: the e-value, the rule's level at that hypothesis, the FDR
budget remaining, and the matured-case count. See "Never unattended".

Fix at the source: the challenger's training and test rows come from an
out-of-time split when the data are temporal or grouped (the current random
shuffle at `lifecycle.py:290-292` is wrong for such data and is replaced by
F2's splitter).

### L3. Rollback: reverse e-process in shadow

After promotion the previous champion is kept scoring in shadow
(`shadow.py:37`). A reverse e-process tests `H0: new champion is not worse
than the previous by more than Delta_rb`; crossing `1/alpha_rb` creates a
`rollback_proposal` (executed only with approval, as `lifecycle.py:379-391`
already does). This replaces the bootstrap-interval-below-tolerance check
(`:365`) as the trigger, for the same repeated-look reason as in L2.

### L4. Delayed, partial and selected labels

Three distinct problems, one estimator each.

1. **Delay.** A case is *matured* when its outcome window (charter field
   `outcome_window`) has elapsed since prediction. Only matured cases enter
   L1-L3. If the label delay depends on the outcome (slow outcomes arrive
   late), labels received so far are biased toward fast outcomes; maturity
   by prediction time, not by label arrival, avoids that. Streams that cannot
   declare an `outcome_window` are tiered `unverified`.
2. **Selection (verification bias).** When the label is observed only for a
   subset chosen by a rule (for example, only approved cases are followed
   up), the labelled subset's loss is a biased estimate of the loss over all
   cases (this is the verification-bias problem in diagnostic studies; Begg
   and Greenes 1983, **from memory, unverified**). Method: Horvitz-Thompson
   weights `1/pi(x)` where `pi(x) = P(label observed | x)` is **logged at
   decision time** by the labelling rule, with a randomized **audit slice**
   (a fraction `epsilon` of cases labelled uniformly regardless of the model,
   so `pi(x) >= epsilon` everywhere). With logged `pi`, the weighted loss is
   unbiased without a model of the selection. Weighted losses lie in
   `[0, 1/epsilon]`; the L1 sequence is run on `l_t * w_t / (1/epsilon)`
   (range-rescaled, wider intervals as the price). Weights are clipped at
   `1/epsilon` by construction, so no clipping bias is introduced.
3. **No logged propensity and no audit slice.** The performance estimate is
   labelled `unverified`; L1-L3 may raise a `halt` and an escalation but
   **may not** trigger a retrain, promotion or rollback on that stream. The
   missing-at-random assumption (label observation independent of the
   outcome given `x`) can then not be tested, and the system does not assume it.

Right-censored outcomes (event not yet observed at the cut-off) are handled
with the IPCW machinery of F12 B3, not by treating the case as a negative.

### L5. Feedback-loop detection

A model's decisions can change the data that later trains and evaluates it.
The detectable case is the one with randomization: a fraction `epsilon_f` of
decisions bypasses the model (default action or random action), logged as
such. Comparing the bypass group with the model-decided group on outcome and
on next-period features (two-sample test, with the F6 shift tests) estimates
the model's effect on the data it sees. Without any randomized slice a
feedback loop is **not detectable** from logs alone; the system says so and
tiers retraining on its own decisions' outcomes `unverified`. The audit slice
of L4 and the bypass slice may be the same slice, declared in the charter.

### E1. Escalation as selective prediction

Cases (and, at system level, actions) are routed to a human when a score
`g(x)` is below `tau`. Formulation: selective prediction with a reject option.
Coverage `phi(tau) = P(g >= tau)`, selective risk `R(tau) = E[loss | g >=
tau]`, the risk-coverage curve `{(phi(tau), R(tau))}` and its area. The
trade-off between coverage and risk is the object studied by El-Yaniv and
Wiener, "On the foundations of noise-free selective classification" (JMLR
11, 2010; venue and risk-coverage framing read 2026-11-04). Geifman and
El-Yaniv ("Selective classification for deep neural networks",
arXiv:1705.08500, abstract read 2026-11-04) build from a trained model a
selective classifier that, for a user-given risk, rejects as needed to meet
it "with high probability"; the abstract's example numbers are not used here.

**Score `g`.** Default: the F12 class-conditional conformal set size (a set
with more than one label, or an infinite regression margin, is an
escalation) combined with the novelty flag (`novelty.py:31`). Reason: the set
carries a finite-sample marginal coverage statement; softmax margin does not.

**Threshold.** Pre-registered tolerance in the charter: target selective risk
`r*`, confidence `1 - delta`, and a human capacity budget `c_max` (maximum
escalated share per period). `tau` is chosen on matured, propensity-weighted
calibration cases by fixed-sequence testing: order candidate thresholds from
most to least conservative; for each, test `H: R(tau) > r*` with an exact
(Clopper-Pearson) binomial p-value; stop at the first non-rejection; choose
the least conservative `tau` whose hypothesis was rejected. This controls the
probability that the selected threshold violates `r*` at `delta` under
exchangeability of calibration and deployment (a fixed-sequence procedure; I
recall this is the approach of "Learn then Test", Angelopoulos et al. 2021,
**unverified, not read in this session**). Decision-theoretic cross-check
(Chow's rule, 1970, **from memory, unverified**): with calibrated
probabilities and a cost ratio `c_h / c_e` (human review vs error), abstain
when `1 - max_y p(y|x) > c_h / c_e`; a mismatch between that cut and `tau` is
reported, because it means either the calibration or the declared costs are
off.

**Under shift the guarantee lapses.** `R(tau)` is monitored on matured,
weighted cases with the L1 sequence (loss restricted to answered cases). If
the sequence excludes `r*`, the system enters `halt`: every case escalates
(coverage 0) until `tau` is recalibrated on fresh matured cases and the
recalibration passes its own test. If escalations exceed `c_max`, the excess
is **not answered by the model**; it receives the charter's declared safe
default action and a backlog entry. The capacity budget never raises `tau`.

### E2. System-level escalation (stop and ask)

Beyond cases, the loop stops and asks when any of these fires (each is a
charter-registered trigger with a decision-map action): L1 alarm with an
unverified label mechanism; coverage sequence below nominal on matured
cases (F12); out-of-support rate above its certified bound (novelty);
challenger evidence insufficient at the label budget; ledger verification
failure; two consecutive rollbacks; a requested action on the never-list.
The full pending question, evidence references and the system's proposed
default are written to the `escalation` row before anything waits.

### Never unattended

The system must not, without a valid human attestation recorded in the
ledger (and for the first four, the existing signature check):

1. Move the `champion` alias (`registry.py:367-380`) or execute a rollback
   (`lifecycle.py:379-391`).
2. Open `C` or `F` a second time, or touch sealed sets (F2/F3 custody).
3. Change the charter: estimand, `alpha_*`, set sizes, decision map,
   escalation tolerance `r*`, `delta`, `c_max`, safe defaults.
4. Edit, delete, reorder or truncate ledger rows, or register a hypothesis
   after its result is visible.
5. Relax any threshold, tolerance or coverage target in response to a failing
   gate or an alarm (no self-relaxing; a tightening needs no attestation).
6. Retrain, promote or roll back on a stream whose label mechanism is
   `unverified` (L4.3), or on its own decisions' outcomes where a feedback
   loop is undetectable (L5).
7. Answer a case flagged out-of-support, or a case in a slice the F12 report
   flags under-covered, with an action classed high-impact by the charter.
8. Suppress, silence or reroute its own alerts or incidents.
9. Change the labelling or randomization policy (audit slice, bypass slice).
10. Delete data, models, bundles or run artifacts; send external
    communications; widen the data it is permitted to use.
11. Create or sign an attestation itself. A signature on an attestation must
    come from a key the system does not hold (`scmle/contracts/signing.py`;
    key custody is a deployment requirement and is stated in the report).

Enforcement is a test per item: the action, called without a valid
attestation, raises and writes an `escalation` row. Items 1 and 2 are
already enforced in code today; 3-11 are not.

### E3. Recording the human reply as an attestation

A reply is a ledger row of kind `attestation`, signed, with the fields of the
domain-owner sign-off in `plan/validation-referents.md` section 5:

```json
{"escalation_id": "...", "question_hash": "...", "decision": "approve|reject|override|cannot_decide",
 "statement": "what the person asserts", "evidence": "ticket, log table, document id",
 "role": "...", "name_or_id": "...", "date": "YYYY-MM-DD",
 "scope": "model version hash + stream id", "expires": "YYYY-MM-DD", "signature": "..."}
```

Effects: the dependent decision is tiered `attested`, never `confirmed`
(P4); the row is bound to the exact question hash, so a reply cannot be
reused for a different question; it expires. `cannot_decide` or no reply by
the charter's timeout applies the safe default and tiers the outcome
`unverified`; the run does not stall (P5). A reply that is an *answer to a
case* (a human label) is stored as `human_decision`, **not** as ground
truth; human error is estimated on an audited subsample and escalated cases
are a non-random subset, so they are not used to calibrate `tau`.

Ledger change: add kinds `attestation` and `escalation` to the handler table
(`ledger.py:478-485`) for a **separate deployment ledger** whose chain, replay
and tamper tests mirror the data ledger's; the data ledger's closure at
confirmation (`:477`) is unchanged.

## Gates: measured operating characteristic and external referent

Simulation streams have known truth (change points, label mechanism,
true risk). Replay streams are described below. Thresholds `alpha_*`, `r*`,
`delta` are inputs; everything else is measured and written to
`loop_certificates.json` before the gate run (pre-registration by hash).

| # | Quantity | Operating characteristic measured | Pass criterion | External referent |
|---|---|---|---|---|
| H1 | L1 false-alarm | `R = 2000` stable streams, `T = 5000` batches of 50 cases, loss in `[0,1]`, no change. P(any alarm). Compared with the existing rules (`monitor.py`) on the same streams. | P(any alarm) `<= alpha_alarm + 2*sqrt(alpha_alarm*(1-alpha_alarm)/R)` (about 0.0597 at 0.05). The existing fixed rule is expected to exceed it; its rate is published. | Closed form for the null stream (loss i.i.d. Bernoulli with known mean); the betting-CS paper's stated validity once read. |
| H2 | L1 detection delay | Abrupt and gradual loss increases of `Delta`, `1.5 Delta`, `2 Delta`; delay in batches, with a bootstrap interval, at the H1 false-alarm level. | Reported; gate: detection of a `2 Delta` change within the charter's `D_max` with probability `>= 0.90` (lower Clopper-Pearson bound), else the alarm does not gate the retrain and stays `investigate`. `D_max` is set by the charter before the run. | Known change point in the simulator. |
| H3 | Benign covariate drift | Covariate-only shift with unchanged conditional `P(y|x)`; and concept shift with unchanged covariates. | Retrain trigger rate on benign shift `<= alpha_alarm` (L1 ignores it); concept shift detected as H2. Existing `FeatureDriftRule` rate on the same streams published. | Known generator. |
| H4 | L2 false promotion | Sequences of 200 challengers, 80% true nulls (not better by `Delta_promo`), dependent through shared data; e-LOND over the sequence. | Realised FDR over promotions `<= alpha_promo + MC tolerance`; the bootstrap rule (`champion.py:169`) applied at every review shown to exceed it; promotion power for truly better challengers reported. | Closed-form simulator truth; Xu and Ramdas theorem (read by the previous author, `online_fdr.py:20-31`). |
| H5 | L3 rollback | New champion worse by `0`, `Delta_rb`, `2 Delta_rb`. | False-rollback rate at `0` `<= alpha_rb + MC tolerance`; missed-rollback rate and delay at `2 Delta_rb` reported. | Simulator truth. |
| H6 | L4 selection correction | Labels observed with probability `pi(x)` depending on the model score (MAR given `x`) and, in a second world, depending on the outcome (MNAR). Audit slice `epsilon in {0.02, 0.05, 0.10}`. | MAR with logged `pi`: bias of weighted risk within MC error, naive labelled-only risk shown biased (magnitude published); CS coverage of the true risk `>= 1 - alpha - MC`. MNAR without audit slice: stream is tiered `unverified` and no action is triggered (test). | Simulator truth; closed-form risk. |
| H7 | L4 maturity | Outcome-dependent label delay. | Maturity-by-prediction-time estimator unbiased; label-arrival-time estimator shown biased. | Simulator truth. |
| H8 | L5 feedback | Simulated approval loop where the model's decisions shift the covariates of later cases; bypass slice `epsilon_f`. | Detection power of the bypass comparison vs `epsilon_f`, false-detection rate `<= 0.0597` in loop-free worlds; without bypass slice the stream is tiered `unverified` (test). | Simulator truth. |
| H9 | E1 threshold | Calibration `n_cal = 2000`, `r* = 0.05`, `delta = 0.1`; R = 2000. Risk-coverage curve and AURC reported. | `P(R(tau_hat) > r*) <= delta + MC tolerance`; coverage at `tau_hat` reported with interval vs the oracle `tau` of the simulator. Naive "pick the best-F1 cut" shown to violate. | Oracle risk-coverage curve in simulation. |
| H10 | E1 under shift | Drift of size `s` after deployment. | Realised selective risk is escalated by the L1 sequence within `D_max`; time with `R > r*` before `halt` reported; coverage goes to 0 at `halt` (test). | Simulator truth. |
| H11 | Escalation vs a domain owner | Replay of recorded incident-like sequences (below). A domain owner labels each time step "a human should have been asked: yes / no / either". | Recall of owner-`yes` steps `>= 0.90` (lower bound) at an escalation rate `<= c_max`; agreement with the owner reported as Cohen's kappa with interval; for a held-out incident set fixed before any tuning, recall `>= 0.80` at the same rate. | The owner's signed audit (below). |

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

## Decision map (what each check does)

| Check | Action it triggers |
|---|---|
| L1 alarm, label mechanism verified | `retrain` (a challenger is trained and staged, no promotion). |
| L1 alarm, label mechanism `unverified` | `halt` + escalation; no retrain. |
| L2 e-value crosses `1/alpha_promo` and e-LOND rejects | `promotion_proposal` with evidence; waits for signed approval. |
| L3 e-process crosses | `rollback_proposal`; waits for approval; meanwhile `tau` tightened (allowed: tightening needs no attestation). |
| Selective risk sequence excludes `r*` | `halt`: coverage 0, all cases escalate. |
| Escalations exceed `c_max` | Excess gets the safe default and a backlog row; `tau` is not raised. |
| Never-list action requested | `escalation` row, action not performed. |
| No reply by timeout | Safe default, outcome `unverified`. |

## Guarantees and assumptions

- L1/L2/L3 error rates hold at any stopping time if the matured-case losses
  satisfy the CS/e-process assumptions (bounded losses; conditional mean
  assumptions stated in the cited construction; to be read in Phase 13).
  They are about the matured, weighted stream, not about the future.
- L4.2 is unbiased only if the logged `pi` is correct and positive
  (`pi >= epsilon`); an incorrectly logged `pi` is an undetected error, so the
  audit slice's own randomization is checked by a uniformity test on logs.
- E1's risk bound holds under exchangeability of calibration and deployment
  cases; the L1 sequence on answered cases is the guard when that fails; it
  detects with delay, so there is a window of unbounded excess risk before
  `halt`. H10 measures that window; it is not zero.
- Escalation recall against the owner's audit is conditional on the
  owner's judgement being the right referent; it does not prove the owner is
  right.
- Feedback loops are detectable only with a randomized bypass slice (L5).
- Key custody for the human's signature is outside the software's control
  (P6).

## Failure modes and degradation

| Situation | Behaviour |
|---|---|
| Too few matured labels | `insufficient_labels`; no retrain, promotion or rollback; escalation if the stream stays unverifiable beyond a charter bound. |
| Audit slice produces no labels | Same as L4.3. |
| Calibration set for `tau` too small for `r*` at `delta` | `tau = +infinity` (everything escalates); stated. |
| Human unavailable | Safe default per charter; backlog; outcomes `unverified`. |
| Ledger verification fails | `halt`; escalation; no action until a person reviews. |
| Replay corpus unavailable | Gates H11 and replay metrics do not run; loop stays `unverified` for production. |
| Challenger evidence never reaches threshold | No promotion. Not a failure. |

## Repo integration

- Reuse: `online_fdr.ELond` / `RuleConfig` (`online_fdr.py:356,558`),
  `Evidence` with `e` (`:174`); ledger chain and replay logic
  (`ledger.py:258-497`); `PromotionApproval` and `promote` (`registry.py:40,326`);
  `LifecycleManager` (`lifecycle.py:102`); `ShadowEvaluator`
  (`shadow.py:37`); incident workflow (`incidents.py:178`) for halts.
- Change: `ModelMonitor.performance` (`monitor.py:353`) takes optional
  `propensity` and `maturity` and returns weighted risk with its sequence;
  `LifecycleManager.review_locked` (`lifecycle.py:290`) replaces the random
  shuffle by F2's splitter; `check_rollback` (`lifecycle.py:328`) uses L3.
- New: `scmle/monitoring/sequential_tests.py` (CS, e-process),
  `scmle/monitoring/labels.py` (maturity, propensity, audit slice, HT
  estimator), `scmle/governance/escalation.py` (selective threshold, routes,
  never-list guards), `scmle/governance/deployment_ledger.py` (reusing the
  ledger chain), `benchmarks/qualification/streams.py` and
  `loop_certify.py` (proposed names).

## Acceptance criteria (machine-checkable)

1. H1: P(any false alarm) within the stated bound over `R = 2000` stable
   streams; the existing fixed rule's rate on the same streams is written to
   the certificate.
2. H2/H3 certificates exist; the alarm gates a retrain only if H2's recall
   bound holds; benign covariate shift triggers no retrain beyond `alpha_alarm`.
3. H4: realised false-promotion FDR within bound; bootstrap-per-review shown
   above it on the same sequences.
4. H6: bias test passes with logged propensity; MNAR without audit slice
   produces tier `unverified` and **no** retrain/promote/rollback action
   (negative-control test).
5. H9: `P(R(tau_hat) > r*) <= delta + tolerance`; naive cut shown to violate.
6. H10: at `halt` the answered fraction is 0 (test), and the pre-halt excess
   window is recorded.
7. Never-list: for each of the 11 items, a test calls the action without a
   valid attestation and asserts it raises and writes an `escalation` row;
   items 1-2 already pass today and are kept passing.
8. Deployment ledger tamper test (delete, reorder, edit) fails verification;
   replay of the ledger reproduces every decision bit for bit on Python 3.11
   and 3.14 (same standard as `online_fdr.py:1-10`).
9. An `attestation` row is rejected if its `question_hash` does not match its
   escalation, if it is expired, or if the signature key is the system's own.
10. H11: the owner-audit agreement figures are present in
    `escalation_audit.json` with the owner's signed record hash; absent
    record means the loop's escalation policy is `unverified` and the report
    says so.
11. `ModelMonitor.performance` with logged propensities returns the weighted
    estimate; without them returns `unverified` (type-level, tested).

## Verification status of literature claims

Read in this session (2026-11-04, abstracts only): arXiv:2010.09686
(betting confidence sequences; time-uniform), arXiv:1705.08500 (selective
classification with a risk guarantee "with high probability"), JMLR 11 (2010)
El-Yaniv and Wiener (risk-coverage trade-off). From the repo, read by an
earlier author and recorded in `online_fdr.py:12-90`: e-LOND Theorem 1 and
the p-to-e calibration. **From memory and unverified:** Ville's inequality
as the basis of anytime validity; Begg and Greenes 1983; "Learn then Test"
(Angelopoulos et al. 2021) and its fixed-sequence use here; Chow's rule
(1970); every dataset named in the replay section; MAPIE-like and
confidence-sequence library availability. Formal statements of the betting
bound and its assumptions on the loss sequence must be read before WP-13.3.

## Cut, and why

Drift-statistic-driven retraining (no certified link to loss); fixed-sample
bootstrap promotion at each review (repeated-look inflation); adaptive
conformal and conformal test martingales for label-free exchangeability
monitoring (each needs its own certificate; label-free detection stays at
`investigate`); contextual bandit or off-policy evaluation machinery beyond
Horvitz-Thompson with a logged randomized slice; learned deferral models
(learning to defer); automatic cost-ratio elicitation; self-adjusting
thresholds.
