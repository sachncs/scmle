# Phase 8: Problem framing (F8), what is left

Names follow [`00-canon.md`](00-canon.md); feature spec in
[`../product/features/F8-problem-framing.md`](../product/features/F8-problem-framing.md).

Built and measured (see `CHANGELOG.md`): the cost model and net benefit, the
baseline claim as an intersection-union test with a once-only guard and a
selective-label detector, the feasibility link, the tipping-point question
list, the deterministic drafter with data-side checks, the LLM slot-filler
interface with replay, the ambiguity score and `tau` calibration code, the
`cost_model` charter field, `scmle frame`, and `--check framing`. What stays
open needs outside parties or wiring that touches other features.

## Open work packages

**WP-8.4 Gold corpus** (human lane, longest lead, nobody inside the project
can do it)
- Recruit domain experts independent of the project; collect >= 100
  problems with request, source description, expert charter and cost model;
  >= 20 double-charted for the ceiling; split calibration / test, test
  sealed until one use per drafter version.
- Register in `corpora.json` as `candidate`; flip to `verified` only after
  licence and hash are checked (existing rule).
- Done when: a signed adjudication record per problem exists
  (`plan/validation-referents.md` section 5) and the ceiling agreement is
  computed.

**WP-8.5b (remainder) Calibrate and gate the LLM path** (needs 8.4)
- A real LLM adapter behind `refusal.SlotSampler`, with the model id tied to
  the score. `refusal.calibrate_tau` exists and is tested on synthetic data;
  run it on the calibration split and write the risk-coverage curve and `tau`
  to a certificate. Until then no `tau` is on file and `frame` refuses every
  LLM draft.
- Done when: acceptance criteria 7 and 8 of the feature file pass on the
  held-out split, used once per drafter version; test-set hashes are checked
  against prompt text. If not met, the LLM path stays `descriptive` and the
  release says so.

**WP-8.7 (remainder) Wiring**
- The baseline claim in a run is about a plain linear rule frozen on `X`, not
  about the model the run selects later; testing the selected model's rule on
  `C` needs the model before `C` opens, which the stage order does not give.
- Reproducibility on Python 3.11 (criterion 10 was checked on one interpreter
  only).

**WP-8.8 Referent checks not done**
- Compare `net_benefit` curves with the R packages `rmda` or `dcurves` and
  Vickers' published code on fixed inputs (unavailable here; the tests use an
  independent loop implementation of the same definition, which checks the
  code against the stated formula, not the formula against the literature).
- Check the Berger citation against the paper (Vickers-Elkin was read on
  2026-11-05, `docs/LITERATURE_CHECK.md`).
- Statistician review of the 0.90 factor in criterion 7 before the gate is
  enabled.
- Weaker selection than the certified plants (outcome observed for 70 percent
  of rows with `b = 0`): the detector's recall there is unmeasured.
- A domain owner's recorded sign-off on the cost range for one deployed
  problem; the outcome check (does the verdict on `C` agree with realised cost
  after deployment).

## Exit gate

```bash
uv run pytest -q tests/unit tests/property tests/leakage -k "framing or cost_model or baseline"
uv run python -m benchmarks.qualification --check framing
uv run python -m benchmarks.qualification --check          # certificates still current
```

- Gold corpus, held-out, one use: Wilson lower bound of accepted-draft
  end-to-end rate >= 0.90 x inter-expert ceiling; if not met, `framing` stays
  `descriptive`.
- Planted omission against the gold corpus (the synthetic version is built and
  measured): question recall >= 0.80, false-question rate <= 0.10.
- Draft replay hash-identical on 3.11 and 3.14.

## Evidence required

The risk-coverage curve and `tau` with its calibration split size; the
gold-corpus manifest with adjudication records; the ceiling computation; the
reference-implementation comparison table; the signed list of corpus experts
and their independence statement.

## Risks

- **Corpus does not materialise.** Without independent expert charters no
  LLM-assisted field can be gated; the non-gating fallback is built and
  labelled and the guarantee-bearing parts do not depend on it.
- **Self-consistency is not correctness.** A consistently wrong LLM passes the
  refusal rule; refusal is reported as risk-coverage, never as correctness.
- **Gold leakage into prompts.** Test problems must not appear in any prompt
  or few-shot example; test-set hashes are checked against prompt text.
- **Cost elicitation is the weakest link.** A wrong cost ratio changes the
  verdict; the range and the tipping value are reported, and the range needs a
  human attestation.
- **Reference definitions differ** (decision-curve packages may normalise
  differently); document the mapping at WP-8.8.
- **Model providers change behaviour.** The gold-corpus score is tied to a
  model id and re-measured on any change.

## Explicitly out of scope

Multi-action utility matrices; off-policy (counterfactual) estimation of the
current process; learning cost ratios from data; multi-turn agent interfaces;
feature construction and data ingestion (F9); causal estimand drafting beyond
filling the `kind` field (the causal DAG remains a human or LLM hypothesis
under F4); any claim that a drafted charter is correct.
