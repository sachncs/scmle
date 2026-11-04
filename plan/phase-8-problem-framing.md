# Phase 8: Problem framing (F8)

Names follow [`00-canon.md`](00-canon.md); feature spec in
[`../product/features/F8-problem-framing.md`](../product/features/F8-problem-framing.md).

## Goal

From a plain-language request and a source description, produce a draft
`AnalysisCharter`, a decision-cost model, a measured comparison against the
current business process, a feasibility verdict, and a short list of
questions only a human can answer, with a measured refusal rule. The
guarantee-bearing parts are LLM-free; the LLM-assisted drafter is non-gating
until a held-out expert corpus supports it (P1).

## Entry criteria

- Phase 2 merged (charter, signing): met in the repo (see below).
- Phase 3 merged (ledger, confirmation): `scmle/data/ledger/confirm.py`
  exists; run `uv run pytest -q -k "ledger or fdr"` green first.
- Phase 1 harness available for corpora and the planted-world simulator
  (`benchmarks/qualification/simulate.py`, `corpora.py`).
- A decision on whether an LLM dependency is permitted at all. If not,
  WP-8.5b is dropped and the phase still delivers everything else.

## Verified starting points

- `AnalysisCharter`, `AnalysisEstimand`, `DecisionRow`, `DecisionAction`:
  `scmle/data/charter/contracts.py:280, :80, :148, :35`; `from_dict` `:393`;
  `FORMAT_VERSION = 1` `:30`.
- Closed-form feasibility: `n_required_mean` `scmle/data/charter/feasibility.py:79`,
  `assess` `:189`.
- Target checks: `check_targets` `scmle/data/validation/target.py:79`.
- Precedent for a stored, hashed LLM draft: `scmle/data/admissibility/dag.py:3`
  and `:191`.
- Corpus manifest with `candidate` / `verified` status:
  `benchmarks/qualification/corpora.py:3-7`.
- Verified absences: no cost model, no net-benefit code, no baseline
  comparison, no LLM client, no check that a charter's `unit`, `target` or
  `prediction_moment` names real columns (grep over `scmle/`; the charter
  validates them as non-empty text only, `contracts.py:100-110, :314-325`).
- `--check` choices in `benchmarks/qualification/__main__.py:75` are
  `certificates` and `fdr` only; a `framing` choice is added in this phase.

## Work packages

**WP-8.1 Cost model and net benefit** (one agent)
- *Proposed* `scmle/data/framing/costs.py`: `DecisionCostModel`
  (`c_00, c_01, c_10, c_11`, range `[t_lo, t_hi]`, source per entry:
  `data | human | template`), validation that `c_FP, c_FN > 0`, `NB(t)`,
  treat-all, treat-none.
- Done when: acceptance criteria 1 and 2 pass (identity to 1e-9; match to the
  reference implementation, chosen and its definitions checked first).

**WP-8.2 Baseline comparison claim** (one agent; needs 8.1, Phase 3)
- *Proposed* `framing/baseline.py`: `Delta(t)` with cluster-robust SE,
  intersection-union test at the range endpoints, registered as one F3
  hypothesis per endpoint on `C` (rule `a_i` frozen from `X` first).
  Selective-label detector.
- Done when: criteria 3, 4, 5 pass; hypothesis registered before `C` is
  opened (existing ledger rule); a second evaluation on `C` raises.

**WP-8.3 Feasibility link** (one agent; needs 8.1)
- *Proposed* `framing/feasibility.py`: estimate `s_d`, `rho` on `X`, call
  `assess` with `delta = Delta_min / s_d` and the design-effect-adjusted `n`.
  Report the interval of the estimate.
- Done when: on simulated worlds the verdict agrees with the oracle power
  in >= 95% of 500 runs away from the boundary; near the boundary the
  report says "inconclusive" and the test checks that wording.

**WP-8.4 Gold corpus** (human lane, starts on day 1, longest lead)
- Recruit domain experts independent of the project; collect >= 100
  problems with request, source description, expert charter and cost model;
  >= 20 double-charted for the ceiling; split calibration / test, test
  sealed until one use per drafter version.
- Register in `corpora.json` as `candidate`; flip to `verified` only after
  licence and hash are checked (existing rule).
- Done when: a signed adjudication record per problem exists
  (`plan/validation-referents.md` section 5) and the ceiling agreement is
  computed. If this lane cannot be staffed, WP-8.5 and 8.6 deliver code and
  remain non-gating; this is stated in the release notes.

**WP-8.5a Deterministic drafter and data-side checks** (one agent; needs 8.1)
- *Proposed* `framing/draft.py`: rule-based slot filling, data-side checks
  (target exists and complete, moment before target time, population
  predicate coverage, unit uniqueness), charter assembly via `from_dict`.
  Charter format version bump for the `cost_model` field with old-charter
  loading preserved.
- Done when: criterion 6 for the deterministic path; it scores on the
  calibration split (this is the comparator).

**WP-8.5b LLM slot-filler and refusal** (one agent; needs 8.5a, 8.4
calibration split)
- *Proposed* `framing/refusal.py` and an LLM adapter behind an interface
  with a recorded-response replay mode (so tests never call the network).
  `k` samples, canonical slot values, ambiguity score, `tau` by Clopper-
  Pearson on the calibration split, risk-coverage curve written to the
  certificate.
- Done when: criteria 6, 8, 10, 11 pass; replay mode reproduces the charter
  hash; tests make zero network calls (checked by a socket-blocking
  fixture).

**WP-8.6 Question list** (one agent; needs 8.1, 8.3)
- *Proposed* `framing/questions.py`: derivable-by-data filter, tipping-point
  test over the plausible range, closed-form question templates.
- Done when: criterion 9 (planted omission) passes.

**WP-8.7 Wiring, CLI, report** (one agent; last)
- `scmle frame` (draft + questions), hook before charter signing in
  `scmle/core/stages.py`, outputs listed in the feature file, unsigned-LLM
  tier propagation (criterion 11), `--check framing` in the harness.

## Parallel lanes

```
WP-8.4 (human, long) ------------------------------+
WP-8.1 --> WP-8.2 --+                              |
   |--> WP-8.3 -----+--> WP-8.7                    |
   |--> WP-8.5a --> WP-8.5b (needs 8.4 calibration)+
   |--> WP-8.6 (needs 8.3)
```

8.1 first; then 8.2, 8.3, 8.5a in parallel; 8.4 runs throughout.

## Exit gate

```bash
uv run pytest -q tests/unit tests/property tests/leakage -k "framing or cost_model or baseline"
uv run python -m benchmarks.qualification --check framing
uv run python -m benchmarks.qualification --check          # certificates still current
```

Numbers (R = 2000 replicates per simulated world unless noted):

- Identity: `Cost` vs `const - c_FN*NB(t)` max abs error <= 1e-9 over 10^4
  draws.
- Differential: net-benefit curves match the reference to 1e-9 on fixed
  inputs.
- Intersection-union size: empirical rejection <= `0.05 + 2*sqrt(0.05*0.95/2000)`
  (about 0.0597) at the committed minimum cluster count; the minimum is
  written into the lint.
- Power: >= 0.80 where the closed form says it is attainable.
- Selective labels: detector rate >= 0.95 (Wilson lower bound) on 500
  selective-label worlds; <= 0.05 on 500 clean worlds.
- Gold corpus, held-out, one use: Wilson lower bound of accepted-draft
  end-to-end rate >= 0.90 x inter-expert ceiling; if not met, `framing` is
  built but its `GateStatus` stays `descriptive` and the release says so.
- Planted omission: question recall >= 0.80; false-question rate <= 0.10.
- Zero network calls in tests; draft replay hash-identical on 3.11 and 3.14.

## Evidence required

`framing_draft.json`, `decision_cost_model.json`, `baseline_report.json`,
`framing_questions.json` from a seeded run; the risk-coverage curve and
`tau` with its calibration split size; the gold-corpus manifest with
adjudication records; the ceiling computation; the reference-implementation
comparison table; the signed list of corpus experts and their
independence statement.

## Risks

- **Corpus does not materialise.** Without independent expert charters, no
  LLM-assisted field can be gated. Mitigation: the non-gating fallback is
  built and labelled; the guarantee-bearing parts do not depend on it.
- **Self-consistency is not correctness.** A consistently wrong LLM passes
  the refusal rule. Mitigation: refusal is calibrated and evaluated on
  held-out data, reported as risk-coverage, never as a correctness claim.
- **Gold leakage into prompts.** Test problems must not appear in any
  prompt or few-shot example. Mitigation: split sealed; test-set hashes
  checked against prompt text.
- **Cost elicitation is the weakest link.** A wrong cost ratio changes the
  verdict. Mitigation: tipping-point reporting, a range rather than a point,
  and an explicit human attestation for the range.
- **Selective labels** are common in the cases where a baseline process
  exists. Mitigation: detector plus refusing the claim; not hidden.
- **Reference definitions differ** (decision-curve packages may normalise
  differently). Mitigation: check at the start of WP-8.1; document the
  mapping.
- **Model providers change behaviour.** Replay mode freezes tests; the
  gold-corpus score is tied to a model id and re-measured on any change.

## Explicitly out of scope

Multi-action utility matrices; off-policy (counterfactual) estimation of the
current process; learning cost ratios from data; multi-turn agent
interfaces; feature construction and data ingestion (F9); causal estimand
drafting beyond filling the `kind` field (the causal DAG remains a human or
LLM hypothesis under F4); any claim that a drafted charter is correct.
