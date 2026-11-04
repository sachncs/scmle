# Phase 10: Causal identification (F10)

Names follow [`00-canon.md`](00-canon.md). Feature definition:
[`../product/features/F10-causal-identification.md`](../product/features/F10-causal-identification.md).
Needs Phases 2 (charter, custody), 3 (ledger, confirmation), 4 (admissibility,
certificates) and 5 (stability). Phase 6's corpus manifest and outside-party
protocol are reused.

## Goal

A `kind="causal"` charter produces either a causal estimate with a tier that
means what F10 says it means, or an explicit `not_identified` refusal. Every
gate in the phase has a measured operating characteristic and an external
referent (P1). The deliverable is not an estimator library; it is the
selection rule, the diagnostics with certificates, the falsification battery
and the tier rule, measured against experiments where the answer is known.

## Entry criteria

- Phases 3 and 4 gates pass on the current head: ledger replay is
  bit-identical, `C` opens once, certificate stores load.
- The corpus manifest (`benchmarks/qualification/corpora.json`) accepts new
  entries; every entry below starts as `candidate`.
- Decision recorded: the one-paragraph canon addition defining the causal
  reading of the three tiers (F10, "What `confirmed` means"). The canon is
  changed first, as `00-canon.md` requires.

## Verified starting points

- Charter estimand has no treatment, instrument, index or assumption fields:
  `scmle/data/charter/contracts.py:80-98`; kinds at `:31`.
- Causal admissibility is skipped when no graph is declared:
  `scmle/data/admissibility/qualify.py:157-166`.
- `Dag` is a hashed hypothesis; `backdoor_valid` at
  `scmle/data/admissibility/dag.py:244`, `adjustment_set` at `:278`, search
  cap 16 at `:38`; `rules.decide_causal` at
  `scmle/data/admissibility/rules.py:214`.
- Falsification only, Bonferroni, never confirms:
  `scmle/data/admissibility/independence.py:309`, `:344`, `:399`.
- Uplift propensity check with an uncertified AUC constant:
  `scmle/evaluation/uplift.py:22`, `:112`; used at
  `scmle/core/stages.py:424-432`.
- Bounded sequential IPW estimator with positivity check and ESS:
  `scmle/evaluation/sequential.py:500`, `:939`, `:952`.
- Stage order: `run_claim_ledger` at `scmle/core/stages.py:518` then
  `assess_robustness` at `:579`; `qualify_exploration` at `:649`.
- Certificate pattern to copy: `benchmarks/qualification/f6_certify.py`,
  `f6_certificates.json`; Wilson interval at
  `scmle/data/detectors/certificates.py:73`.
- Dependencies are numpy, pandas, scipy, scikit-learn
  (`pyproject.toml:27-38`); no statsmodels, no causal library.

## Work packages

**WP-10.1 Charter block and selector** (one agent)
- `CausalSpec`, `NegativeControl`, `SensitivitySpec` in
  `scmle/data/charter/contracts.py`; `refuse_claim` action; charter format
  bump; `scmle/data/identification/select.py` with the strategy registry
  (required assumptions, required data design) and the refusal report.
- Done when: acceptance A1 and A2 pass (data-blind selector, refusal worlds);
  charter hash changes when any causal field changes; old charters still load.

**WP-10.2 Back-door AIPW and overlap diagnostics** (one agent; depends on 10.1)
- `aipw.py`: cross-fitted learners on `X`, frozen, score on `C`; ATE and ATT
  (write the ATT score from the source, mark verified in the docstring);
  overlap report (propensity range, share outside bounds, Kish ESS, weighted
  SMD); trimming rule declared in the charter. Reuse
  `sequential.effective_sample_size`.
- Done when: A3 (backdoor row), A4 (overlap curves), A9 toy-table equality.

**WP-10.3 DiD, IV, RD, synthetic control** (four parallel agents, each
independent, depend on 10.1)
- 10.3a `did.py`: group-time ATT, event study, honest sensitivity (`M` grid).
  Read Callaway-Sant'Anna, Roth, Rambachan-Roth before coding; replace every
  *unverified* tag in F10 section B by a checked citation or delete the claim.
- 10.3b `iv.py`: 2SLS, effective F, Anderson-Rubin interval, Hansen J.
- 10.3c `rd.py`: local linear, bias-corrected interval, density test, placebo
  cutoffs, bandwidth sensitivity.
- 10.3d `synth.py`: weights by constrained least squares, placebo-in-space
  permutation, donor-count refusal.
- Done when: per strategy A3 and A9 pass against frozen reference output.

**WP-10.4 Falsification battery and replication** (one agent; depends on 10.2
and at least one of 10.3)
- `battery.py`: negative-control outcome and exposure tests, placebo tests,
  stratum replication with a heterogeneity test; Holm at `alpha_falsify`,
  level drawn from the error budget with `split_budget`
  (`scmle/data/ledger/budget.py:277`).
- Done when: A5 (power versus planted violation size, `delta_min`) and the
  false-alarm bound hold.

**WP-10.5 Sensitivity** (one agent; depends on 10.2)
- `sensitivity.py`: E-value, robustness value and partial-R2 bounds with
  benchmarking against observed covariates; Rosenbaum bounds only for a
  chartered matched design. Verify the E-value formula and the
  Cinelli-Hazlett definitions against the papers and the reference packages
  first.
- Done when: A6 and A9 (`EValue`, `sensemakr` frozen outputs to 1e-9).

**WP-10.6 Tier rule, wiring, report** (one agent; depends on 10.1-10.5)
- `tier.py`: the five conditions for `confirmed`, `attested` and
  `unverified` as a pure function of the report files; wire after
  `run_claim_ledger` (`stages.py:518`); write `causal_claims.json` and hash
  into the ledger head; `scmle claims` prints identification status.
- Done when: A7, A8, A11, A12.

**WP-10.7 Benchmarks and certificates** (two agents; 10.7a starts at once,
10.7b after 10.2)
- 10.7a Corpus entries and manifests (hash-verified): NSW experimental and
  CPS/PSID comparison groups, Tennessee STAR, ACIC generators, IHDP. All
  `candidate` until existence, licence and hash are checked; record the
  experimental estimate from the data at ingest.
- 10.7b `benchmarks/qualification/f10_generators.py` (planted effect on real
  `X` with knobs for confounding, overlap, instrument validity, trend
  violation, sorting) and `f10_certify.py` writing `f10_certificates.json`.
- Done when: A10 and the certificate check (`--check` fails on stale or
  missing certificates).

## Parallel lanes

Lane 1: 10.1 then 10.2 then 10.4, 10.5, 10.6. Lane 2: 10.7a from day one.
Lane 3: 10.3a-d in parallel once 10.1 lands. 10.7b joins lane 1 after 10.2.

## Exit gate

```bash
uv run pytest -q tests/unit tests/property tests/leakage -k "identification or causal"
uv run python -m benchmarks.qualification --check f10
uv run python -m benchmarks.qualification --f10-gold      # NSW, STAR, ACIC
```

Numbers (all intervals Wilson 95%; the thresholds below are provisional and
are re-fixed from the first full run, with that run published, before the
gate is switched on):

- Refusal worlds: refuses in 200/200 when no strategy is eligible (Wilson
  lower >= 0.98); does not refuse in 200/200 when one is.
- Coverage of the 95% interval at `R = 2000`, assumptions satisfied: Wilson
  lower bound >= 0.93 for AIPW, DiD, IV (strong instrument), RD.
- Overlap gate: false-refuse <= 0.05; recall >= 0.8 at min propensity 0.02
  and 0.005.
- Falsification battery: false-alarm rate <= `alpha_falsify` + binomial
  tolerance at `R = 2000`; `delta_min` (power >= 0.8) stated per violation
  type with its curve.
- Sensitivity bound: contains the truth >= 95% (Wilson lower >= 0.93) when
  the planted confounder is within the declared benchmark.
- False-confirm rate in detectable-violation worlds: <= 0.05 (Wilson upper
  <= 0.08).
- Gold standard: on NSW with each declared observational comparison group,
  no `confirmed` tier when the instrument's 95% interval excludes the
  experimental estimate; the full table of (estimate, interval, tier,
  refusal) is published including the failures.
- Reference outputs equal frozen references at the tolerances in F10 A9.
- Reports are bit-identical on Python 3.11 and 3.14.
- Outside-party run of the gold-standard benchmark reproduces the table (the
  party and date are recorded; if it has not happened the phase is not
  closed).

## Evidence required

Selector property-test log; per-strategy coverage and bias tables with Wilson
intervals; overlap curves; battery power curves and `delta_min`; sensitivity
validity table; gold-standard table; `f10_certificates.json`; frozen oracle
manifest with tool versions and hashes; import-graph output; outside-party
report.

## Risks

- **Attested assumptions carry the claim.** An analyst can attest anything.
  Mitigation: the tier text says so; assumptions and attestors are in the
  claim file; `confirmed` needs a negative control, and the blind spot (a
  confounder the control does not share) is reported with its measured rate
  in A7, not hidden.
- **Benchmarks are synthetic or old.** NSW is small and heavily studied;
  ACIC generators are not the user's data. Mitigation: report as
  referents for the operating characteristics, not as proof for any
  deployment; say what population each represents.
- **Overfitting the gate to the benchmarks.** Thresholds are fixed before the
  run and the run is repeated by an outside party; held-out corpora (a
  benchmark not used while tuning) are named before tuning begins.
- **Cross-fitting and frozen nuisances reduce efficiency.** Wider intervals
  are accepted; reported.
- **Literature claims from memory.** Several are tagged *unverified* in F10.
  WP-10.3 and WP-10.5 include verifying them; an unverified tag that
  survives to the exit gate fails the gate.
- **Scope growth.** Each added strategy needs its own certificate; no
  strategy is enabled without one (P1).

## Explicitly out of scope

Causal discovery and DAG learning, mediation, dynamic regimes beyond the
existing bounded estimator, interference and spillovers, continuous-treatment
effect curves, heterogeneity as a gate, decision policies built on the
estimate (F11), sequential monitoring (F11), the HTML report.
