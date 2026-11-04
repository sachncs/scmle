# F8: Problem framing, what is left

Vocabulary and principles: [`../../plan/00-canon.md`](../../plan/00-canon.md).
Plan: [Phase 8](../../plan/phase-8-problem-framing.md).

## Built

From a plain request, a source description and exploration rows (`X` only),
`scmle frame` writes a draft unsigned `AnalysisCharter` with an optional
`cost_model`, the source of every field, data-side checks, the baseline
measured on `X`, and the questions only a person can answer (`scmle.data.
framing`, `CHANGELOG.md`, `docs/ARTIFACTS.md`). The guarantee-bearing parts
are LLM-free: the cost algebra, the baseline comparison as an intersection-
union test on `C` (library call, once only), the feasibility formula and the
tipping-point questions. The numbers they were measured at are in
`benchmarks/qualification/framing_measurements.json`, checked by
`python -m benchmarks.qualification --check framing`.

## Honest status of the LLM-assisted pieces

An LLM-drafted charter field has no error rate the instrument can derive.
Schema validation proves a field is well formed, not right, and
self-consistency across samples is agreement, not correctness. No guarantee
in F8 rests on an LLM output being right. Every LLM-filled field is
`unverified` until a named human signs the charter. The slot-filler
interface, the ambiguity score and the `tau` calibration code exist, but the
gold corpus they need does not, so no `tau` is on file, `frame` refuses every
LLM draft, and the LLM path is **descriptive and non-gating** (P1). The
deterministic drafter is the comparator and ships.

## What remains

### Gold corpus (the external referent of draft correctness)

Real problems, each with a request, a source description, and a charter plus
cost model written by a domain expert who has not seen the drafter's output.
Requirements (proposed):

- At least 100 problems (a Wilson 95% interval for a rate near 0.8 has half-
  width about 0.08).
- Split into calibration (for `tau`) and held-out test; the test split is
  used once per drafter version.
- At least 20 problems charted independently by two experts; their field
  agreement is the **ceiling**.
- Public competition descriptions give target, unit and metric gold, not cost
  models; costs need elicitation. A public corpus with expert cost models is
  not known to the author (*unverified*). If cost gold cannot be obtained,
  cost-model drafting stays non-gating and entirely human-supplied.
- Adjudicated and signed per `plan/validation-referents.md` sections 1 and 5;
  recorded in `benchmarks/qualification/corpora.json` as `candidate` until
  verified.

Scoring per field is machine-executable: target, prediction moment and unit
by exact match; population by Jaccard of selected `X` row ids >= 0.95; metric
by class; cost ratio within a factor of 2 or overlapping range; baseline
column exact; questions raised by precision and recall against the experts'
"needed" items. End-to-end correct means all critical-path fields agree.
Report Wilson intervals for the end-to-end rate, the rate among accepted
requests, coverage, refusal precision and question precision and recall.

### Acceptance criteria still open

7. Gold-corpus gate, held-out split, once per drafter version: Wilson lower
   bound of the end-to-end rate among accepted drafts >= 0.90 x the
   inter-expert ceiling, coverage reported. Otherwise non-gating. (The 0.90
   factor is a proposal for statistician review.)
8. The LLM slot-filler is kept only if its end-to-end rate minus the
   deterministic drafter's has a 95% paired-bootstrap interval above zero on
   the held-out split.
9. (Corpus version) Planted-omission test on gold requests: question recall
   >= 0.8, false-question rate <= 0.1. The synthetic version passes.
10. Byte-identical charter hash on Python 3.11 and 3.14 from the stored draft
    (only one interpreter has been run).
### Wiring not done

- The baseline claim in a run concerns a linear rule frozen on `X`, not the
  model the run selects afterwards.
- The charter's `unit`, `deployment_population` and estimand text are free
  text; a charter run only reports whether target, moment and unit match
  column names (`framing_link.json`, descriptive).

### External referent not yet used

- Reference net-benefit curves from the R packages `rmda` or `dcurves` and
  Vickers' published code (*unverified* that these are current and that their
  definitions agree with `costs.py`; the tests use an independent loop
  implementation of the same formula).
- A domain owner's recorded sign-off on a cost range for one deployed problem
  (`plan/validation-referents.md` section 5).
- Outcome check: for problems with a later deployment, does the verdict on `C`
  agree with the realised difference in cost?
- The selective-label detector is certified against strong plants only (no
  outcome for rows with `b = 0` beyond 0 to 30 percent); weaker selection is
  unmeasured, and so is selection that depends on more than the baseline
  decision.
- Citations (Vickers and Elkin 2006; Berger 1982; El-Yaniv and Wiener 2010)
  were recalled, not re-read.
