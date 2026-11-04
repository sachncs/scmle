# F8: Problem framing

Vocabulary and principles: [`../../plan/00-canon.md`](../../plan/00-canon.md).
Built in [Phase 8](../../plan/phase-8-problem-framing.md); needs F1 (charter),
F3 (confirmation) and the F7 harness. F8 and F9 are not yet listed in the
canon tables; the canon owner adds them there first.

## Purpose

Today a human writes the charter. F8 turns a plain-language request plus a
description of the data source into a **draft** `AnalysisCharter` and a
**decision-cost model**, measures the current business process as the
baseline, checks feasibility, and lists the few questions only a human can
answer. It also refuses to draft when the request is too ambiguous for the
draft to be trusted.

This is the step that decides whether the rest of the instrument answers the
right question. F1 to F7 certify how a question is answered; nothing in the
repo today certifies that the question is the right one (verified below).

## Decision it serves

Two decisions:

1. **Issue or refuse.** Does the draft charter go forward unattended, go
   forward with named open questions, or stop?
2. **Is the model worth deploying relative to what the business does now?**
   Expressed as a claim "net benefit of the model exceeds net benefit of the
   current process over the declared cost range", tested once on `C` through
   the F3 ledger.

## Honest status of the LLM-assisted pieces

An LLM-drafted charter field has no error rate that the instrument can
derive. Schema validation proves the field is well formed, not that it is
the right one. Self-consistency across samples measures agreement, not
correctness, and a model can be confidently and consistently wrong.
Therefore:

- No guarantee in this feature rests on an LLM output being right.
- The guarantee-bearing pieces are LLM-free: the cost algebra, the baseline
  comparison, the feasibility formula, and the confirmation test.
- Every LLM-assisted field is tier `unverified` until a named human signs it
  (tier `attested`, canon P4). An unsigned draft may run (P5) and every claim
  that depends on it is labelled with that tier.
- The only external check proposed for the drafter itself is a held-out
  corpus of charters written by domain experts who did not see the drafter's
  output (see "Measuring framing correctness"). Until that corpus exists and
  the gate passes, the drafter is **descriptive and non-gating** (P1).

## Behaviour

Proposed interface (nothing below exists yet):

```python
request = FramingRequest(
    text="Which customers will cancel in the next 30 days? We call the "
         "top ones today.",
    source=SourceDescription(uri="warehouse://crm/accounts", schema=...),
)
draft = frame(request, data_sample=frame_x)     # X rows only (P3)
draft.charter            # AnalysisCharter (unsigned), see below
draft.cost_model         # DecisionCostModel
draft.baseline           # BaselineReport (measured on X, then on C once)
draft.feasibility        # existing feasibility.assess output
draft.questions          # list[HumanQuestion], possibly empty
draft.status             # "issued" | "issued_with_questions" | "refused"
```

Artifacts: `framing_draft.json` (every drafted field with its source:
`template`, `llm`, `data`, `human`; model id, prompt, sampling seeds and raw
samples when an LLM was used), `decision_cost_model.json`,
`baseline_report.json`, `framing_questions.json`, `framing_refusal.json`
(when refused, with the ambiguity score and the fields in conflict).

Rules: the draft is hashed and stored; the charter hash covers the final
charter, not the draft's generation process, so a run is reproducible from
the stored draft and never by regenerating it (the repo already treats an
LLM-drafted object this way: `scmle/data/admissibility/dag.py:3` stores the
draft as a hashed hypothesis with its source and prompt; `dag.py:191` takes
the prompt). The cost model is fixed in the charter before `C` is opened.

## Method

One method per need. What was cut is stated at the end of each item.

### 1. Decision-cost model: expected cost, reported as net benefit

Binary action `a in {0,1}` (act or not), outcome `y in {0,1}`, cost
`c_ay` (negative cost = benefit). Define `c_FP = c_10 - c_00` and
`c_FN = c_01 - c_11`, both required positive (otherwise one action
dominates and the problem is not a decision problem; the drafter refuses to
build a model and asks).

Expected cost per row of a rule is

```
Cost = c_00 (1-P) + c_01 P + c_FP * FP - c_FN * TP
     = const - c_FN * NB(t),     NB(t) = TP - [t/(1-t)] * FP
```

with `TP, FP` as rates over all rows, `P` the prevalence, and
`t = c_FP / (c_FP + c_FN)` the threshold implied by the cost ratio. So
minimising expected cost and maximising the **net benefit** of Vickers and
Elkin (2006, Med Decis Making 26:565) are the same problem for a binary
decision, up to the positive scale `c_FN`. (Derived here; a unit test must
check it numerically. The Vickers-Elkin definition is from memory:
*unverified* until checked against the paper.)

Choice: **expected-cost minimisation as the primary model, net benefit as the
reporting scale** (it is unit-free in `t` and comparable across baselines,
including "act on everyone" and "act on no one"). The decision curve is
`NB(t)` over a declared range `[t_lo, t_hi]` rather than a single point,
because cost ratios are rarely known to better than a factor of a few.

Only the binary-action case is specified. Cut: multi-action utility
matrices (add after the binary case has a certificate), ROC-AUC-only
framing (does not encode costs), cost curves and Brier decomposition
(different display of the same quantity), and any attempt to learn costs
from the data (costs are preferences, not data; see section 5).

### 2. Baseline: the current process, measured

The baseline is the decision the business takes today (a rule, a person, a
default), not "random" or "majority class". It enters as a column of past
decisions `b_i in {0,1}` joined to outcomes `y_i`. Then

```
NB_base(t)  = TP_b - [t/(1-t)] * FP_b          (from b_i and y_i)
NB_all(t)   = P - (1-P) * t/(1-t)
NB_none(t)  = 0
d_i(t)      = [ a_i*(y_i - (1-y_i) w) - b_i*(y_i - (1-y_i) w) ],  w = t/(1-t)
Delta(t)    = mean_i d_i(t) = NB_model(t) - NB_base(t)
```

`Delta(t)` is linear in `w` when the model's action rule `a_i` is fixed
(chosen on `X` at `t*`, not re-thresholded per `t`). So
"Delta(t) > 0 on the whole range `[t_lo, t_hi]`" holds if and only if it
holds at both endpoints; the claim is an intersection-union test of the two
endpoint nulls, each at level `alpha_confirm`. The intersection-union
construction (Berger 1982, from memory) keeps the size at most `alpha_confirm`
without a multiplicity correction, at the price of power. Standard error:
cluster-robust (sandwich) over the F2 grouping unit, because rows of one
entity are not independent. The cluster-robust SE is asymptotic; its
coverage with few clusters is *unverified* and is measured (acceptance
criterion 4).

If a baseline decision column does not exist, the baseline is
`NB_all` and `NB_none` only, the comparison claim is not made, and a human
question is raised. A baseline column with the right name is not enough;
see "selective labels" in the failure modes.

Cut: a counterfactual (off-policy) estimate of the current process from
model scores alone. That needs propensities and assumptions the instrument
cannot verify.

### 3. Feasibility: reuse the existing closed form

The per-row paired differences `d_i(t)` have mean `Delta_min` (the smallest
net-benefit gain worth acting on, supplied by the cost model's owner or
asked) and a standard deviation `s_d` estimated on `X` only. The standardised
effect is `delta = Delta_min / s_d`, and the existing function applies as is:
`n_required_mean(delta, alpha, k_max, power)` at
`scmle/data/charter/feasibility.py:79` and `assess(...)` at `:189`, whose
verdict `FEASIBLE | DOWNGRADE | INFEASIBLE` already sets the confirmatory or
unconfirmed scope (`:24-27`). For clustered data the required `n` is
multiplied by the design effect `1 + (m-1) rho`, with `m` the mean cluster
size and `rho` the intraclass correlation estimated on `X`. Both `s_d` and
`rho` are estimated from `X`, so feasibility is an estimate with sampling
error; the report states its interval and does not present it as a fact.

Cut: simulation-based power for arbitrary models (the closed form is the
referent; simulation is used only in tests to check it).

### 4. Drafting the charter: template first, LLM only to fill slots

Two drafters, both producing the same `AnalysisCharter` shape:

- **Deterministic drafter (the comparator).** Slots filled from the data
  description by rules: target column from an explicit name in the request
  or the single column whose name matches; prediction moment from the
  single timestamp column that precedes the target's window; unit from the
  key columns F9 infers (until F9 exists: the declared primary key, or ask);
  metric from the cost model; every other field blank and listed as a
  question.
- **LLM slot-filler.** Reads the request and a schema summary (names,
  types, a few non-sensitive example values) and returns JSON for the
  free-text slots only: `question`, `population`, `deployment_population`,
  `prediction_moment`, candidate `target`, intercurrent events, and the
  action/outcome wording for the cost model. Output must pass
  `AnalysisCharter.from_dict` (rejects unknown or missing fields,
  `scmle/data/charter/contracts.py:393`) and the data-side checks below.

The LLM path is kept only if it beats the deterministic drafter on the
gold corpus (next section) by a margin whose confidence interval excludes
zero. If it does not, it is cut and the deterministic drafter ships.

**Data-side checks (LLM-free, run on every draft).** Target column exists,
is complete and fits the task (existing `check_targets`,
`scmle/data/validation/target.py:79`); the prediction-moment column exists
and every target time is after it (counts violations); the population
predicate, evaluated on `X`, selects between declared bounds of the rows
(a predicate selecting 0 or all rows is flagged); the unit's key columns
are unique at the unit grain within a tolerance (F9). These catch malformed
and impossible drafts. They do not catch a well-formed wrong draft.

**Refusal rule.** Draw `k` drafts at temperature above zero (proposed
`k = 10`). For each executable slot compute a canonical value: target
column name; moment column name; unit key set; population as the set of `X`
row ids selected by the predicate; metric class. Slot disagreement is
`1 - m_s`, where `m_s` is the share of the `k` drafts agreeing with the
modal value (for populations, mean pairwise Jaccard). The ambiguity score is
`A = max_s (1 - m_s)` over slots on the critical path (target, moment, unit,
population). Refuse when `A > tau`. `tau` is not hand-set: it is the largest
value for which the **upper Clopper-Pearson bound** of the error rate among
accepted drafts on a calibration split of the gold corpus is at most the
tolerated `epsilon` (proposed 0.05). The risk-coverage curve (accepted
fraction versus error among accepted) is on file with the certificate. This
follows the selective-prediction idea (risk-coverage; El-Yaniv and
Wiener 2010, from memory). Refusal protects only to the extent that
disagreement predicts error; the held-out corpus measures that, it is not
assumed.

Cut: multi-turn agent loops, retrieval over company documents, LLM-as-judge
of draft quality (a judge is another unverified LLM output), and autonomous
choice of the cost ratio.

### 5. The "ask the human" list: only what the data cannot answer

A question is raised only if all three hold:

1. **Not derivable from the data or request** (the deterministic drafter and
   data-side checks cannot fill it).
2. **Decision-relevant**: a **tipping-point** test. Vary the unknown over its
   plausible range and recompute the verdict (issue, baseline comparison
   sign, feasibility). If the verdict is unchanged across the whole range,
   do not ask; report the range and the verdict as robust. If it flips, ask,
   and report the tipping value (the same device as the F5 missingness
   tipping point).
3. **Answerable by a named person**: a closed form (a cost ratio in a range,
   a date, a yes/no), never an open question.

Costs: when the data hold realised unit costs per outcome, they fill `c_ay`
(source `data`, tier `unverified`). When the baseline decisions are present,
the cost ratio *implied* by the baseline's operating threshold can be shown
as a hint, but the hint assumes the current process is cost-rational, which
is false in many organisations; it is displayed, labelled, and never used
as the value.

Cut: asking for confirmation of every field (questions that never change a
decision are noise), and inferring preferences from the data alone.

## Measuring framing correctness

There is no way to establish correctness of a framing from the data. The
external referent is a **gold corpus**: real problems, each with a request, a
source description, and a charter plus cost model written by a domain expert
who has not seen the drafter's output. Corpus requirements (proposed):

- At least 100 problems, so a Wilson 95% interval for a rate near 0.8 has
  half-width about 0.08 (normal approximation: `1.96*sqrt(0.8*0.2/100)`).
- Split into calibration (for `tau`) and held-out test; the test split is
  used once per drafter version.
- At least 20 problems charted independently by two experts. Their field
  agreement is the **ceiling**: a drafter cannot be judged against a gold
  charter more consistent than the experts are with each other.
- Sources: public problem statements with a host-defined target and metric
  (competition descriptions) give target, unit and metric gold, but **not**
  cost models; costs need elicitation. A public corpus with expert cost
  models is not known to the author (*unverified*; the claim that none
  exists is also unverified). If cost gold cannot be obtained, cost-model
  drafting stays non-gating and entirely human-supplied.
- Adjudicated and signed by the experts per `plan/validation-referents.md`
  sections 1 and 5; recorded in `benchmarks/qualification/corpora.json`
  with status `candidate` until verified (the existing manifest rule,
  `benchmarks/qualification/corpora.py:3-7`).

Scoring is machine-executable, not by reading prose. Per field:

| Field | Agreement test |
|---|---|
| target | same column (exact) |
| prediction moment | same column (exact) |
| unit | same key-column set (exact) |
| population | Jaccard of selected `X` row-id sets >= 0.95 |
| metric / summary measure | same class from a closed list |
| cost ratio | `t` within a factor of 2 of gold, or the range overlaps gold |
| baseline definition | same decision column |
| questions raised | precision and recall against expert-listed "needed" items |

End-to-end correct means all critical-path fields agree. Reported with Wilson
intervals: end-to-end rate over all requests, rate among accepted requests,
coverage (accepted share), refusal precision (share of refused requests that
the deterministic drafter or LLM also got wrong), and question precision and
recall. Question recall is also measured by **planted omission**: remove one
decision-critical fact from a gold request; the drafter should raise a
question naming it (recall), and should not on untouched requests (false
question rate).

## Guarantees and assumptions

- Cost algebra: exact identity given the stated costs (tested numerically).
- Baseline claim: `Pr(claim | no true gain at some t in range) <= alpha_confirm`
  under the F3 confirmation conditions (independent `C`, read once, fixed
  rule `a_i`), asymptotically in the number of clusters; finite-cluster
  coverage is measured.
- Feasibility: estimate with an interval, conditional on `s_d`, `rho`
  estimated on `X`.
- Draft correctness: **no guarantee.** Only a measured agreement rate
  against the held-out gold corpus, valid for problems drawn like the
  corpus. A problem outside the corpus's domains has no stated rate.
- The baseline comparison is valid only if outcomes are observed
  independently of the baseline's decision (see selective labels).

## Failure modes and degradation

| Situation | Behaviour |
|---|---|
| Ambiguity score above `tau` | Refuse; write `framing_refusal.json` with conflicting slots; no run. |
| Draft fails `from_dict` or data-side checks | Draft discarded; deterministic drafter result used; both recorded. |
| One action dominates (`c_FP <= 0` or `c_FN <= 0`) | No decision model; question raised; no baseline claim. |
| No baseline decision column | `NB_all` / `NB_none` only; baseline claim not made; question raised. |
| **Selective labels**: outcome observed only when the baseline acted (for example loans approved) | Detected by: outcome missing iff `b_i = 0` above tolerance. Baseline comparison tier `unverified`; claim not issued (naive comparison is biased). |
| Cost range wide enough that `Delta` changes sign inside it | Reported as such; the claim fails the intersection-union test. |
| Feasibility `DOWNGRADE` or `INFEASIBLE` | Existing behaviour: scope `exploratory_unconfirmed`. |
| Unsigned LLM-filled field | Tier `unverified` propagates to the claims that depend on it. |
| LLM unavailable | Deterministic drafter only; recorded. |
| Request names data the source does not have | Refuse with the missing columns listed. |

## Repo integration (verified unless marked proposed)

- The charter exists and is hashed: `AnalysisCharter`
  (`scmle/data/charter/contracts.py:280`), `charter_hash` (`:368`),
  `from_dict` (`:393`), `template()` (`:417`), `read_charter` (`:469`),
  signed envelope `write_charter` (`scmle/data/charter/sealing.py:55`).
- **Gap 1: nothing links the charter's words to the data.** `unit`,
  `deployment_population`, and the estimand's `population`, `target` and
  `prediction_moment` are checked only as non-empty text
  (`contracts.py:100-110` for the estimand, `:314-325` for the charter).
  No code verifies that the target names a column or that the unit is a key.
- **Gap 2: no cost model exists.** The estimand's `summary_measure` is free
  text (`contracts.py:80-98`); `min_effect` is a standardised effect
  (`:280-300`). Grep over `scmle/` for net benefit, decision curve and cost
  matrix finds only the DAG docstring mention of an LLM draft.
- **Gap 3: no LLM client in the repo.** A grep for `openai`, `anthropic`
  and `llm` over `scmle/**/*.py` finds only `scmle/data/admissibility/dag.py`
  (the precedent above). Adding a drafter adds a dependency and a
  determinism problem; the stored-draft rule handles the latter.
- Gap 4: `from_dict` rejects unknown fields (`require_fields` is used in
  `DecisionRow.from_dict`, `contracts.py:181`; the charter's own `from_dict`
  at `:393` follows the same pattern). Adding `cost_model` to the charter is
  therefore a format change: `FORMAT_VERSION = 1` (`:30`) must be bumped and
  old charters still load (proposed).
- Reused as is: `feasibility.assess` (`scmle/data/charter/feasibility.py:189`),
  `check_targets` (`scmle/data/validation/target.py:79`),
  `DecisionAction.BLOCK_RUN` for refusal (`contracts.py:35-43`), the
  corpus manifest (`benchmarks/qualification/corpora.py`), the F3 ledger
  (`scmle/data/ledger/`) for the baseline hypothesis.
- Proposed new package: `scmle/data/framing/` (`costs.py`, `baseline.py`,
  `draft.py`, `refusal.py`, `questions.py`), and a `framing` check in
  `benchmarks/qualification/__main__.py` (its `--check` choices are
  currently only `certificates` and `fdr`, line 75).
- Not wired: `stages.py` (`scmle/core/stages.py`) does not call any framing
  step. Wiring is a work package in Phase 8.

## Acceptance criteria (machine-checkable)

1. Identity test: for 10^4 random cost matrices (`c_FP, c_FN > 0`) and
   random confusion tables, `Cost` computed directly equals
   `const - c_FN * NB(t)` to 1e-9.
2. Differential test: `NB(t)` curves equal the reference implementation
   named in the external referent on fixed inputs to 1e-9, including the
   treat-all and treat-none curves.
3. Intersection-union calibration: on simulated worlds with `Delta(t)=0` at
   one endpoint and `>0` at the other, empirical rejection rate over 2000
   runs is at most `alpha_confirm + 2*sqrt(alpha*(1-alpha)/2000)`. With 20,
   50 and 200 clusters; the smallest cluster count at which the bound holds
   is recorded and used as a hard minimum in the charter lint.
4. Power: where the closed form says power >= 0.8 is attainable, the
   claim's empirical power is >= 0.8 minus the Monte Carlo tolerance.
5. Selective-label world: with outcome missing when `b_i = 0`, the detector
   flags it in >= 95% of 500 runs (Wilson lower bound) and the claim is not
   issued; on worlds with outcomes observed for all rows it flags in
   <= 5% of runs.
6. Schema test: every drafter output is rejected unless it round-trips
   `AnalysisCharter.from_dict`; 1000 fuzzed malformed outputs, 0 accepted.
7. Gold-corpus gate, held-out split, evaluated once per drafter version:
   Wilson **lower** bound of the end-to-end rate among accepted drafts is
   >= 0.90 times the inter-expert end-to-end agreement (the ceiling), and
   coverage is reported. If the lower bound is lower, the drafter stays
   non-gating. (The 0.90 factor is a proposal, to be confirmed by the
   external statistician review before the gate is enabled.)
8. The LLM slot-filler is kept only if its end-to-end rate minus the
   deterministic drafter's rate has a 95% paired-bootstrap interval above
   zero on the held-out split.
9. Planted-omission test: question recall >= 0.8 with Wilson lower bound
   reported; false question rate on untouched requests <= 0.1.
10. Reproducibility: re-running a run from `framing_draft.json` gives a
    byte-identical charter hash on Python 3.11 and 3.14 with the LLM
    disabled.
11. Unsigned LLM field: a run whose charter contains an `llm`-sourced
    field and no signature emits no claim above tier `unverified` for
    claims that depend on that field (test on a fixture).

## External referent

- Gold charters and cost models written by domain experts who did not see
  the drafter's output, with a second expert on a subset to measure the
  ceiling (the corpus is to be built; none exists in the repo:
  `benchmarks/qualification/corpora.json` lists defect corpora only,
  *unverified* that it has no framing entry).
- Reference net-benefit curves from an independent implementation: the R
  packages `rmda` or `dcurves` and Vickers' published decision-curve code
  (*unverified* that these are current and that their definitions agree
  with the formulas above; to be checked at the start of Phase 8, as the
  F3 file does for `onlineFDR`).
- A domain owner's recorded sign-off on the cost range for at least one
  deployed problem, per `plan/validation-referents.md` section 5.
- Outcome check (Phase 6 style): for problems with a later deployment, does
  the baseline comparison's verdict on `C` agree with the realised
  difference in cost after deployment?
