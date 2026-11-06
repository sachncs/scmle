# F10: Causal identification (open items)

Vocabulary and principles: [`../../plan/00-canon.md`](../../plan/00-canon.md).
Built in [Phase 10](../../plan/phase-10-causal-identification.md); needs F1
(charter), F2 (custody), F3 (ledger, confirmation), F4 (admissibility) and
F6 (stability and shift).

The built part (the charter block, the selector and refusal, AIPW, difference
in differences, instrumental variables, regression discontinuity, synthetic
control, the falsification battery, replication, sensitivity bounds, the tier
rule, the stage that writes the report files, and the simulation certificates)
is recorded in `CHANGELOG.md` and lives in `scmle/data/identification/`,
`scmle/data/charter/causal.py` and
`benchmarks/qualification/f10_certify.py`. The meaning of the three tiers for a
causal claim is in the canon. This file keeps only what is missing. Statements
about existing code carry `path` references read in the repo on the date of
writing.

## Open: what the single opening does not yet do

The estimate, battery, replication, sensitivity and tier are computed on the
frame `C` releases (`scmle/core/causal_claim.py`). Left:

- Subgroup hypotheses are registered and reported (Holm at the primary
  level), but they describe heterogeneity only: they do not enter the tier,
  their power at the subgroup sizes of a real run is not measured, and no
  certificate covers the Holm family of subgroup estimates.
- The charter has no field for attestors, so a run reports every causal claim
  at `unverified` ("no named attestor"); `identify` takes the names, the run
  does not supply them.
- `Dag.from_artifact` loads the DAG the charter's `dag` path names; whether
  that file is the one the analyst meant is not checked beyond its hash.

## Open: method gaps (each needs its own certificate before it may gate)

- Randomized data: the difference in means exists; the balance check that
  replaces the uncertified AUC rule at `evaluation/uplift.py:112` (a test with
  a measured false-alarm rate on randomized data) does not. The sequential
  estimator (`evaluation/sequential.py:500`) is not registered as the strategy
  `sequential_ipw`.
- Difference in differences: the sensitivity bound is a simplified
  relative-magnitudes bound that ignores the noise in the largest pre-period
  change, not the Rambachan-Roth conditional or fixed-length procedure; no
  covariates, no not-yet-treated comparison group, simultaneous bands are not
  computed (the pre-period Wald test is).
- Instrumental variables: one instrument and one endogenous regressor only;
  no Sargan or Hansen J, no Olea-Pflueger effective F for several instruments,
  no exclusion-violation sensitivity (so the claim stays `attested`), no
  battery (`battery_iv` is not measured).
- Regression discontinuity: the bandwidth is a cross-validated grid choice,
  not the Calonico-Cattaneo-Titiunik mean-squared-error-optimal one; the
  density check is a binomial test of counts, not McCrary or Cattaneo-Jansson-Ma.
- Synthetic control: outcomes only (no predictor covariates), one treated unit;
  no battery of its own.
- Sensitivity: the benchmark conversions of Cinelli and Hazlett are written
  from memory and apply to the linear-model coefficient of the treatment, not
  to the AIPW estimate itself; Rosenbaum bounds for a chartered matched design
  are not implemented (matching is cut unless chartered).
- Negative-control exposures are supported for `backdoor_aipw` only.

## Open: certificates that missed or were not measured

`benchmarks/qualification/f10_certificates.json` holds the first full run. A
gate that missed its floor stays descriptive and cannot count toward
`confirmed`; the miss is listed in `CHANGELOG.md` and must be re-measured
(not retuned) when the method changes. Not measured, because the data or the
tools are not in the repository:

- worlds built on real covariates (acceptance 3 asks for them);
- frozen reference outputs of `rdrobust`, `sensemakr`, `EValue`, the R `did`
  package, `HonestDiD` and `Synth`, with tool versions and hashes under
  `tests/fixtures/oracles/` (acceptance 9). The tests use closed forms, hand
  tables, a convex-program reference and worked examples recalled from the
  papers (the E-value 3.9 example, the sensemakr README numbers), which are
  weaker than frozen package output;
- the NSW (CPS and PSID comparison groups), Tennessee STAR, ACIC and IHDP
  gold standards (acceptance 10) and their corpus manifest entries, all
  `candidate` until existence, licence and hash are checked
  (`plan/validation-referents.md`);
- bit-identical reports on Python 3.11 and 3.14 (acceptance 11; a seeded run
  is identical twice on one interpreter);
- replication by an outside party.

## Open: literature claims not verified

The environment had no access to the papers. These are written from memory and
tagged unverified in the code docstrings: Callaway and Sant'Anna 2021;
Goodman-Bacon 2021; de Chaisemartin and D'Haultfoeuille 2020; Roth 2022;
Rambachan and Roth 2023 (bibliographic details checked, the procedure not);
Imbens and Angrist 1994; Olea and Pflueger 2013; Stock and Yogo 2005 (not
used); Calonico, Cattaneo and Titiunik 2014; McCrary 2008; Cattaneo, Jansson
and Ma 2020; Abadie, Diamond and Hainmueller 2010; Crump, Hotz, Imbens and
Mitnik 2009; Rosenbaum 2002; LaLonde 1986; Dehejia and Wahba 1999; Krueger
1999; Hill 2011. The formulas that the certificates test (AIPW score, the
ATT influence function, the Anderson-Rubin inversion, the E-value, the
robustness value and the benchmark conversions) are checked by simulation and
by identities in the unit tests, not against the papers. An unverified tag
that survives to the exit gate fails the gate.

## Acceptance criteria still open

3 (real-covariate worlds), 9 (frozen reference outputs for RD, sensitivity,
DiD and synthetic control), 10 (experimental gold standards), 11 (cross-version
determinism). 1, 2, 4 to 8 and 12 have tests or certificates in the repo.

## External referent

None of these is the author or the user.

- **Experimental truth.** Randomized experiments with an observational
  comparison group: the National Supported Work data with the CPS and PSID
  comparison groups (LaLonde 1986; Dehejia and Wahba 1999); the true
  experimental estimate must be read from the data and recorded in the
  manifest, not from this document. Tennessee STAR class-size data (Krueger
  1999) with an observational sample built by selecting units into treatment
  by a covariate rule, so that the randomized contrast is the truth.
- **Competition generators with known effects.** The 2016 Atlantic Causal
  Inference Conference challenge and Dorie, Hill, Shalit, Scott and Cervone,
  Statistical Science 34(1), 43-68, 2019, arXiv:1707.02641; later ACIC years
  and the IHDP benchmark of Hill (2011).
- **Reference implementations (outputs frozen with versions).** R `grf` or
  `DoubleML`, `did`, `HonestDiD`, `rdrobust` and `rddensity`, `AER` or
  `ivmodel`, `Synth`, `EValue`, `sensemakr`.
- **Replication by an outside party** of the gold-standard benchmark.
