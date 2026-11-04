# Changelog

## Unreleased

* Deployment contract (F14 item 1; `scmle.deploy.contract`). One typed,
  versioned `DeploymentSpec` (image by digest, bundle by manifest digest,
  configuration, secret references, resources, health, version) that a
  provider adapter must satisfy, and `conformance_problems`, which compares a
  platform's `RuntimeReport` with the spec on digests and limits. A secret is a
  reference and never a value; a secret-looking configuration key, key
  material, `SCMLE_TRUSTED=1` in production and an unsigned production bundle
  are refused. It is an extension surface: the snapshot in
  `tests/fixtures/public_api.json` gained an `extension_surfaces` block and no
  existing entry changed. No adapter for a real platform exists yet.
* Ingestion in a charter run (F9; `scmle.core.ingest_stage`, charter format 7).
  The charter gained the optional `ingest` block (`IngestSpec`): source budget
  (queries, rows scanned, seconds), `B_feat`, windows, lag, `delta_rec` and the
  key and foreign-key tolerances, plus `strict`. It is omitted when unset, so
  charters of format 1 to 6 keep their hash and signature (a test pins the hash
  of a charter without it); a wrapper below format 7 that carries it is
  refused. With `data.ingest_plan` (CLI `--ingest-plan`, `MLProject(ingest_plan=)`),
  a JSON file naming the database and the as-of views, a stage before the split
  runs `IngestPlan` under the charter budget, stops with the questions when the
  unit verdict is `coarser` or `unknown`, reads each table once through a
  snapshot-checked read, synthesizes the feature family at the cutoff
  `moment - lag`, joins the columns to the table, and writes
  `feature_manifest.json`, `ingest_field_provenance.json` and the plan's own
  files. The provenance records go to the admissibility stage, so the
  synthesized features carry their tiers, and after the ledger run the stage
  fails the run when the registered hypotheses differ from the declared family.
  Not done: the resolved entity does not become the split's group column,
  extract reconciliation is not run, and the run was tested on SQLite only.
* F11 reports in a charter run (`scmle.core.decision_reports`). With a decision
  block on a binary task, `FinalEvaluationStage` now writes, once, from the
  locked rows: `policy_value.json` (self-normalised IPW of the frozen rule with
  the charter's clip, support and effective-sample-size gates, from a log whose
  0/1 action is `data.logged_action_column` and recorded probability
  `data.propensity_column`; without both it is `not_assessable` and no
  propensity is estimated), the fairness satisfaction test for every declared
  constraint, `rollout_design.json` (plan hash, fixed-horizon size, simulated
  expected sample number of the anytime-valid test) with a one-row hash-chained
  `rollout_decisions.jsonl`, and `decision_report.json`. New optional fields
  `logged_action_column` and `propensity_column` on `DataSpec`, `MLProject`
  and `--logged-action-column`/`--propensity-column`; both are never features.
  Not done: `LifecycleManager` still does not call `Rollout`/`RollbackMonitor`
  (that file belongs to the Phase 13 work), no experiment assignments reach the
  F3 ledger, and `claims verify` does not check these four files.
* The causal estimate on `C` in a charter run (F10; `scmle.core.causal_claim`).
  The primary hypothesis `theta = 0` is registered in the ledger by the hash of
  the causal block, the strategy, the DAG hash, the confirmation level and the
  exploration row count, and `run.identify` is called on the frame `C` releases
  for the confirmation, in the same opening: the run now reports the estimate,
  the falsification battery, replication, sensitivity and the tier. The five
  report files are hashed into the confirmation row and so into the ledger
  head, and the sealed set is bound to the registered claim. A charter whose
  `dag` path names a stored DAG loads it (`ConfigError` on a bad file) and
  passes it to the selector, so back-door claims can pass the adjustment check.
  `qualify.run` no longer skips admissibility for a causal estimand that has a
  causal block and a graph: it applies `rules.decide_causal` to every
  candidate (descendants of the treatment and colliders are quarantined),
  reports declared adjustment members the graph refused (back-door is then
  ineligible), tests the graph's implied independences on `X`
  (`identification_report.json` key `dag`) and says when more than 16
  candidates and no declared set leave the claim `attested` at best.
  `scmle claims list` prints the identification status and `claims verify`
  checks the five files against the sealed ledger. A charter no strategy
  identifies still gets the refusal files from `X` and `C` is not opened for
  it. The `bandit` B101 asserts in `stages.py` and `identification/run.py` are
  explicit raises. Not done: subgroup hypotheses, and attestors (every run
  claim is `unverified`).
* The F8 baseline claim in a charter run (`scmle.data.framing.claim`). With
  `data.baseline_column` (CLI `--baseline-column`, `MLProject(baseline_column=)`)
  on a binary task whose charter has a `cost_model`, a rule is frozen on the
  exploration rows (standardised ridge logistic regression on the numeric
  features, acting at the cost model's implied threshold, its fitted
  parameters hashed), the claim that it beats the current process over the
  cost range is registered in the ledger by the `BaselineGuard` registration
  hash, and `C` tests both ends of the range on the one opening with the
  cluster-robust intersection-union test (`baseline_report.json`). The column
  is never a model feature. Selective labels are checked on `X` plus the rows
  set aside for a missing target; a flag there makes the claim `not_issued`
  without a test. `data.framing_draft` (CLI `--framing-draft`) names the
  `framing_draft.json` that `scmle frame` wrote, and `provenance.claim_tier`
  now caps the claim from it: an unsigned charter with an LLM-filled field, or
  a cost that came from data or a template, leaves it `unverified`
  (`framing_provenance.json`). The claim is about the frozen rule, not the
  model the run later selects; its level is `alpha_confirm` spent on this one
  claim, with no joint correction against the bivariate family.
  `scmle claims list` prints registered claims and `scmle claims verify`
  checks that their report files hash to the values the sealed ledger holds.
  A charter run without the column is unchanged.
* Registered claims in the F3 ledger (`scmle.data.ledger`). A new row kind
  `claim` registers a claim that is not a single-feature statistic (the F8
  baseline comparison, the F10 primary causal hypothesis) by the hash of its
  registration, before the survivors are frozen; it spends no wealth and never
  enters the survivors. `confirm.confirm` takes a `ClaimRunner`: the claims are
  registered, the sealed set is bound to the survivor family and the
  registrations, and the one frame `C` releases is given to the runner after
  the bivariate family was tested, so there is still one opening. A missing
  negative control or an empty family stops the bivariate promotion but not
  the registered claims. The confirmation row records each claim's status and
  the hash of every report file the runner wrote (`attachments`), binding them
  to the ledger head, and the ledger refuses a seal that omits a registered
  claim. `confirmed_claims.json` gained `registered_claims` and `attachments`
  only when a claim is registered. The baseline and causal entries above use
  it.

* Causal identification (F10; `scmle/data/identification/`, charter format 4).
  An estimand may carry a causal block (`CausalSpec`, `NegativeControl`,
  `SensitivitySpec` in `scmle.data.charter.causal`): treatment, outcome,
  estimand, design columns, attested assumptions from a closed vocabulary,
  strategy priority, negative controls, strata, sensitivity benchmark and a
  trimming rule. It is omitted when unset, so charters of format 1 to 3 keep
  their hash and signature; `DecisionAction` gained `refuse_claim`. The
  selector (`select.select`) is a function of the charter and an optional DAG
  only and imports no estimator; when nothing is eligible the claim is
  `not_identified`, with the missing assumptions per strategy, no estimate and
  the action `refuse_claim`. Strategies, numpy and scipy only: `aipw`
  (nuisances fitted on `X` and frozen, score on `C`; ATE and ATT, overlap
  report, declared trimming that names the changed estimand), `randomized_diff`,
  `did` (group-time ATT against never-treated units, event study, a simplified
  relative-magnitudes sensitivity bound, not HonestDiD), `iv` (just-identified
  2SLS, robust first-stage F, Anderson-Rubin set as headline below F = 10),
  `rd` (local linear with a bias-corrected interval, cross-validated bandwidth,
  binomial density check, placebo cutoffs, covariate balance, bandwidth
  sensitivity) and `synth` (simplex weights, placebo-in-space, refusal below
  19 donors at 0.05; one treated unit, so at most `attested`). `battery`
  (Holm at `alpha_falsify`, one fifth of the confirmation level via
  `split_budget`) and stratum replication (sign and Cochran's Q) can only
  falsify. `sensitivity` has the E-value and the Cinelli-Hazlett robustness
  value and benchmarked partial-R2 bounds. `tier.decide` applies the five
  conditions of the canon as a pure function of the report facts, and a gate
  counts only with a current passing certificate. `run.identify` writes
  `identification_plan.json`, `identification_report.json`,
  `falsification_report.json`, `sensitivity_report.json` and
  `causal_claims.json` (which hashes the other four). In a charter run the
  stage writes the plan, the design diagnostics on `X` and an `unverified`
  claim with `estimate` null (`run.design_only`): the estimate on `C` is NOT
  wired into the single opening by the claim ledger, so no pipeline run yet
  reports a causal estimate; `identify` does it when given the confirmation
  rows. A declared DAG is not loaded by the stage either, so back-door claims
  are capped at `attested`. The admissibility skip for causal estimands
  without a graph is unchanged.
  Certificates: `benchmarks/qualification/f10_certify.py` (floors written in
  its docstring before the first run, seeds 0 to R-1, Wilson 95 percent),
  `f10_certificates.json`, `python -m benchmarks.qualification --check f10`.
  Measured on simulated worlds only (`f10_generators.py`, synthetic
  covariates): refusal worlds 200/200 and 200/200; coverage of the 95 percent
  interval 0.944 (AIPW ATE, linear), 0.948 (ATT), 0.945 (difference in means),
  0.944 and 0.951 (DiD single and staggered), 0.957 (IV, strong), 0.953 (RD);
  weak-instrument headline set 0.950, weak flag recall 0.982, false weak 0;
  placebo-in-space size 0.040 (bound 0.089); the overlap gate fires 0.006 at
  minimum propensity 0.1 and 1.0 at 0.02 and 0.005. Falsification false-alarm
  rates sit at alpha_falsify = 0.01 and the smallest planted violations with
  power of at least 0.8 (Wilson lower) are: shared hidden confounder 0.5
  (negative-control outcome), trend slope 0.3 per period (pre-trend Wald),
  sorting share 0.5 (density), covariate jump 0.3 (balance and placebo
  cutoffs). Sensitivity bound: 100 percent coverage for a confounder at half
  the strongest observed covariate's strength, 80 percent AT the benchmark
  (the bound is not conservative there) and 0 percent at twice it, both
  reported, not gated. MISSED FLOOR: `coverage_aipw` (boosted learners on a
  nonlinear surface, 469 of 500 covered, Wilson lower 0.913 against the
  pre-registered 0.93; the floor at R = 500 is hard to reach even at exact
  nominal coverage, and it was not retuned), so that gate is descriptive and
  the shipped package certifies no back-door claim as `confirmed` until it is
  re-measured; the two linear worlds passed. The end-to-end false-confirm gate
  passed (0 of 300 runs `confirmed` with a hidden confounder shared with the
  negative control), but the blind spot is large: with a confounder the
  negative control does not share, and a declared DAG that omits it, 97.7
  percent of runs reached `confirmed` (clean world 97.0 percent). NOT
  measured here: worlds on real covariates, the NSW/STAR/ACIC/IHDP gold
  standards, frozen `rdrobust`/`sensemakr`/`EValue`/`did`/`HonestDiD`
  outputs, Python 3.11 versus 3.14 bit-identity, an outside-party
  replication. The Cinelli-Hazlett benchmark conversions and the literature
  claims behind the DiD, IV and RD methods are from memory and unverified
  against the papers (`product/features/F10-causal-identification.md`).

* Decision layer (Phase 11, F11), 2026-11-04. `scmle.decision`: cost-optimal
  thresholds with a calibration gate (`threshold`: binned chi-square check,
  Platt recalibration fitted on development data, the existing grid search as
  the labelled fallback, a cluster-bootstrap cost interval), net benefit of a
  frozen rule against treat none, treat all and the incumbent with a
  simultaneous cluster-bootstrap band over the chartered range of cost ratios
  and a one-shot guard (`netbenefit`), off-policy evaluation by IPW, SNIPS and
  doubly robust estimators with influence-function intervals, declared
  clipping, a support gate and an effective-sample-size gate that refuse
  instead of estimating (`ope`; non-randomized logs are not `confirmed`),
  fairness constraints with an upper-bound satisfaction test, group
  thresholds, `not_assessable` small groups and refusal of incompatible
  criteria at measured base rates (`constraints`), and an anytime-valid
  engine (`anytime`): a fixed-grid betting mixture e-process for a bounded
  mean, with pseudo-outcomes centred by a running mean fixed before each unit,
  Ville's inequality for the error rate and a time-uniform bound. On top of it
  `design` (salted-hash assignment, fixed-horizon size, simulated expected
  sample number, an exact and an anytime sample-ratio check) and `rollout`
  (promote, hold, `hold_final`, `hold_invalid` by intersection-union over the
  primary and guardrail e-processes; a rollback monitor per promoted
  champion; a hash-chained decision log). `DecisionBudget` in
  `scmle.data.ledger.budget` keeps exact `Fraction` levels: promotion levels
  by the discount-only rule (valid by the union bound; the e-LOND rejection
  boost is not used because its guarantee for overlapping experiments was not
  checked), a rollback level per champion. The charter gained an optional
  `decision` block (`scmle.data.charter.decision`: decision model, fairness
  constraints with an attestation for group thresholds, rollout plan) under
  format 5; it needs a `cost_model`, and a charter without it hashes, signs
  and loads exactly as before. In a charter run with a decision block the
  threshold comes from the chartered cost model, `evaluation.json` carries the
  calibration path and rule hash under `decision.f11`, and `net_benefit`
  holds the band computed once on the locked test set. Runs without the block
  are unchanged. Certificates: `benchmarks/qualification/f11_certify.py` with
  floors written in its docstring before the run, results in
  `f11_certificates.json`, `python -m benchmarks.qualification --check f11`.
  Measured: 19 of 20 gates met their floors. The net benefit band missed its
  floor in one setting (independent rows, n = 2000: coverage 0.9375, Wilson
  interval 0.926 to 0.947 against a lower bound floor of 0.93), so `A3` is
  descriptive and was not retuned. Other numbers: ever-promote under peeking
  at every unit at the boundary null 37 in 10000 (floor 0.02 plus tolerance;
  the rule is conservative); `abtest.compare_arms` called at 6 looks under
  no effect adopted in 10 of 100 runs (Wilson 0.055 to 0.174), published, not
  gated; false rollback 19 in 10000; the expected sample number to promote at
  twice the minimum effect is 1.15 times a null-shifted fixed-horizon size for
  a binary metric and 1.68 times for a bounded metric; the sample-ratio checks
  rejected 1 (anytime) and 2 (exact) of 1000 correct allocations at level
  0.001 and all of 1000 with a 2 percent distortion at n = 100000. Net benefit power is low: at n = 2000 the band calls a
  rule useful in 91 percent of worlds for a score separation of 1.5 and in
  none at 1.0. Hillstrom e-mail data (randomized, hash recorded, not
  committed) gave the A/A false promotion rate 2 in 1000 and promoted the
  real Men's e-mail arm in all 200 stream orders; its fixed policy treats
  98 percent of rows, so the biased-subsample check (coverage 0.975 of 200)
  says little about policy value differences. Not done: outside-party
  reproduction, the Criteo and Open Bandit referents, reference outputs from
  R `dcurves` and Open Bandit Pipeline, the `LifecycleManager` hook, the
  migration of `DecisionSpec` costs, and the report files other than the
  evaluation fields above (see the plan).

* Closed loop and escalation (Phase 13, F13), 2026-11-04. A hash-chained
  deployment ledger (`scmle.governance.deployment_ledger`), anytime-valid
  monitors (`scmle.monitoring.sequential_tests`: betting alarm mixed over
  restarts, paired e-processes, e-LOND across challengers), label maturity,
  Horvitz-Thompson losses and audit and bypass slices
  (`scmle.monitoring.labels`), selective prediction with a fixed-sequence
  threshold (`scmle.governance.escalation`), the eleven never-list guards
  (`scmle.governance.never_list`) and `scmle.governance.loop.Loop` with
  `scmle loop`. Measured on simulated streams only
  (`benchmarks/qualification/loop_certify.py`, `python -m
  benchmarks.qualification --check loop`): the false-alarm, false-promotion,
  false-rollback, selection, maturity, feedback and threshold gates passed;
  the detection delay of the alarm did not (a rise of twice the tolerance is
  found in a median 76 batches, the pre-registered limit was 50), so a policy
  does not let an alarm retrain by itself (`alarm_gates_retrain`). Not
  measured: agreement with a domain owner's signed audit, replay of recorded
  streams, Python 3.11 ledger replay. `ModelMonitor.performance` takes
  optional `propensity` and `maturity`, and fresh retraining rows are held out
  by time or by group instead of a random shuffle.
* Closed loop wiring, 2026-11-05. A halt writes an `incident` ledger row
  (`Loop.record_incident`; `Hooks.open_incident` still runs after it).
  `Loop.check_feedback` runs `labels.bypass_test` on next-period features of
  the bypass slice, records it in the ledger and `label_mechanism.json`, and
  escalates on a detected loop. `ModelRegistryStore(guard=, scope=)` and
  `attach_guard` (also `Loop.guard_registry`) put the never-list guard in front
  of every champion move, so `LifecycleManager.promote_challenger`,
  `check_rollback` and `register_run(promote=True)` need an attestation first;
  without a guard nothing changes. `ModelRegistryStore.promote` gained the
  optional `guard_detail`. `alarm_gates_retrain` still defaults to False (H2
  and H10 missed their floors; not retuned).
* Mutation scope and results for the Phase 8 to 13 modules, 2026-11-05.
  `setup.cfg` now includes `scmle/decision`, `scmle/governance/{loop,
  escalation,never_list,deployment_ledger}`, `scmle/monitoring/{sequential_tests,
  labels}`, `scmle/data/identification`, `framing/draft.py` and
  `ingest/{entities,plan}.py` (the evaluation, framing and ingestion modules
  were in it already) with `timeout_multiplier = 2`. Results and what is and
  is not triaged are in `tests/unit/MUTANTS_{loop,decision,identification,
  evaluation,framing_ingest}.md`: scores from 64 percent (`identification/
  run.py`) to 99 percent (`sequential_tests.py`), the monitoring survivors read
  one by one, the others only classified. Pinning tests: `test_monitoring_
  mutants.py` and golden-snapshot tests (`tests/unit/golden.py`, fixtures in
  `tests/fixtures/*_golden.json`) for the decision layer, identification,
  evaluation and the loop. Framing and ingestion have no new pins.
* Citations re-read, 2026-11-05 (`docs/LITERATURE_CHECK.md`). Docstrings of
  the betting monitors, the selective threshold, CV+ and weighted conformal,
  MinT, net benefit, the robustness value and entity resolution now say what
  the sources support: the plug-in bet is the paper's empirical-Bernstein
  plug-in used one-sided here, Learn then Test assumes an i.i.d. calibration
  set, the CV+ worst-case coverage is `1 - 2 alpha - sqrt(2/n)`, and the
  paper's rule for `RV_{q,a} <= 1 - 1/f_q^2` is not implemented. No code
  changed.
* Instrument residuals closed in code, 2026-11-04. The confirmation binds the
  release of `C` to the hash of the family it tests and the ledger head
  before opening it (`SealedSet.commit`, R1 C7; a set bound to another
  ledger state is refused). The MMD shift test is certified on 5000 null
  worlds (259 false alarms, upper bound 0.0583 against 0.075; the dated
  amendment precedes the re-run, floors unchanged; R1 A4). A frozen,
  hash-checked Titanic extract (`tests/fixtures/titanic_extract.csv`) shows
  the shipped screen blocks `boat` and `body` (R1 A3). The F5 detectors were
  measured on wine, iris and digits (`f5_corpora.py`): label noise misses the
  floor on the two small tables and the sentinel detector flags clean rows on
  all three, so those two certificates are not read as general (R1 A5, A6).
* Uncertainty wiring (Phase 12): fold models stored in the bundle with CV+
  intervals (`predict_interval(method="cv_plus")`), shift weights attached to
  a bundle (`attach_shift_weights`, `shift=True`), `coverage_report.json`
  measured on the final test rows, and the charter field `unit_of_prediction`
  (format 6; charters without it keep their hash) with entity-wise
  calibration for `new_entity`.

* Custody residuals and evidence artifacts (R1 C4-C10, D1-D4). Canary and
  `gc` tests (`tests/leakage/test_custody_canary.py`,
  `test_custody_memory.py`) and an import-graph test over `detectors/`,
  `admissibility/` and `robustness/` (`test_exploration_imports.py`) fail the
  build if a sealed row or the custody modules become reachable. A sealed
  set's single release can be bound to a claims hash and a ledger hash
  (`SealedSet.commit`, `open(..., claims_hash, ledger_hash)`; the vault
  refuses any other pair); the confirmation stage calls `commit` before it
  opens `C` (see the entry above). Every custody log entry carries an HMAC when
  a signing key is available (`SCMLE_SIGNING_KEY` or an explicit key); without
  a key the log is a checksum. The split manifest records a Merkle root over
  the canonical row content of `X`, `C` and the final set (`row_roots`; the
  field is omitted when empty, so older manifests keep their hash), and
  `verify_rows` checks a table against it. For temporal splits the embargo is
  raised to the number of leading lags of the target's autocorrelation outside
  the white-noise band when that exceeds the chartered `embargo_rows`; the
  manifest records the value used. Two fixtures have pinned manifest hashes
  (`test_custody_golden.py`), observed only on macOS arm64 here. The custody,
  charter, vault and attestation modules are in the mutation scope
  (`tests/unit/MUTANTS_custody.md`). Evidence generated by running the code:
  `plan/evidence/phase-7/` (a sample charter run signed with a published
  demonstration key, an access-attempt log from a scripted attack on a real
  vault, sample manifests; `scripts/make_phase7_evidence.py`) and
  `docs/ORACLES.md` (`benchmarks.qualification.oracle_table`; six oracles
  have a scmle producer, six have none yet).

* Charter format 3 and a pre-registered second stage (R1 B1, B6, B9).
  `AnalysisCharter.rules` (`scmle.data.charter.rules.RunRules`) fixes the
  ledger rule, the confirmation rule, the failure policy, `stability_filter`
  and the stability thresholds under the charter hash and signature. The field
  is omitted when unset, so charters of format 1 and 2 keep their hash and
  signature and load unchanged (a test pins the hash of one); a wrapper from a
  newer format, or an older one carrying `rules`, is refused. A run
  specification that contradicts a chartered rule is refused, and a charter
  without rules reports `"rules": "not chartered"`. With
  `stability_filter` true, survivors that the stability rule calls `fragile`
  are withdrawn from the family tested on `C` before it opens
  (`confirm.Withdrawal`, `robustness.prefilter.StabilityStage`). The ledger
  still freezes and hashes the full survivor set; the withdrawal has its own
  accounting (`stage_two`: family, family hash, rule, certificate state),
  written to the custody log before `C` opens and to the sealed confirmation
  row. The error-control statement is in `confirm.py` and is tested by a
  simulation: the filter is a function of the exploration set only, so the
  confirmation rule keeps its guarantee over the remaining family; it spends
  no alpha, can only cost power, and does not show a withdrawn finding false.
  It acts only while the stability rule has a current certificate for the
  chartered thresholds (the certificate covers `DEFAULT_RULE` only);
  otherwise it is recorded as requested and not applied. In a charter run the
  robustness stage now runs between the negative control and the opening of
  `C`, and its stability checks, the two overall shift checks and the
  per-column shift p-values are wealth-free ledger diagnostics.
* `scmle claims list|show|verify RUN_DIR` (`scmle.artifacts.claims_audit`).
  `verify` checks the ledger chain and replay, that the claims file agrees
  with the sealed ledger, the multiplicity rule replayed on the p-values on
  `C`, the second-stage family and hash, the custody log order and single open
  of `C`, the charter and chartered rules, and the negative control; a missing
  input is `skipped`, never `pass`; exit code 1 on any failure. The `MLProject`
  snapshot and public API fixture are unchanged. Mutation scope gained
  `charter/rules.py`, `robustness/prefilter.py` and `artifacts/claims_audit.py`
  (the mutation run itself was not made).
* Charter runs use the certified leakage screen (R1 B11).
  `admissibility.qualify` no longer calls the earlier
  `leakage.scan`; it runs `screen.screen_table` on the exploration set, so a
  run with and without a charter flags the same columns with one test. The
  screen's certificates (`screen_certificates.json`) decide per mechanism
  whether a suspect may quarantine; the earlier "not gating yet" statement for
  charter runs no longer holds. Every tested column is a wealth-free
  diagnostic row in the ledger with its permutation p-value. A certified
  suspect that is a strong legitimate predictor is kept only by an explicit
  per-column attestation (`scmle.data.admissibility.attestations`, option
  `--leakage-attestations`, `MLProject(leakage_attestations=...)`): a named
  attester, evidence, a validity window and the charter hash; it yields tier
  `attested`, an `admit_flagged` decision and a ledger row
  `leakage.attestation:<column>`. Expired, wrong-charter and unflagged
  attestations are ignored and reported in the run's notes. `rules.build_table`
  accepts a leakage gate per field. A run without a charter has no ledger and
  no override other than `leakage_policy="warn"`. The `MLProject` snapshot was
  regenerated for the new keyword.
* Raw-data ingestion and feature construction (F9), the parts that could be
  built and measured without outside parties or data. New package
  `scmle.data.ingest`, used by `scmle ingest` and `IngestPlan`: `profile`
  (table catalog by aggregate queries run in the source, and
  `consistent_read`, which needs an `ORDER BY`, compares a snapshot token
  before and after and rejects a torn read), `grain` (a level-wise search to
  `L = 3` for approximate keys by the `g3` error, ties listed as questions,
  and a `match` / `finer` / `coarser` / `unknown` verdict on the charter
  unit), `joins` (exact inclusion by anti-join counts; a link is accepted
  only when declared, or exact with agreeing names and no competing parent;
  many-to-many is reported and never joined), `entities` (Fellegi-Sunter with
  multi-pass blocking, EM, and precision and recall from a stratified
  clerical sample; unlabelled results are `unverified`), `asof_view` and
  `synthesize` (the view is the only read path to another table: a row is
  visible only if its recorded time is at or before `moment - lag`; ten
  primitives in declared windows, parent lookups and depth-2 nesting; the cap
  `B_feat` and the family size are declared before exploration and every kept
  feature is registered in the ledger as a `synthesized` hypothesis; each
  feature carries the weakest tier of the tables it reads), `reconcile`
  (row count, conserved sums, join fan-out and end-of-data feature totals
  against pushed-down source queries) and `budget` (queries, rows scanned,
  wall time, deterministic stop at a table or level boundary, partial output
  marked). Artifacts: `ingest_catalog.json`, `grain_report.json`,
  `join_graph.json`, `ingest_budget.json`. Fixed three defects of the
  connectors: `DbApiConnector` batches are now checked against a running
  digest of the rows before each offset (`source_changed`) instead of
  silently overlapping or skipping, and `snapshot_token()` / `snapshot_query`
  give a before-and-after token; read-only is enforced (a textual statement
  allow-list at construction, a query-only session on sqlite3, an optional
  `read_only_statement`) instead of stated in a docstring;
  `IncrementalEvent` gains an optional `ingest_time` (omitted from the record
  when empty, so stored events load unchanged); `WarehouseConnector` accepts
  a `snapshot_reader` callback. Measured in
  `benchmarks/qualification/ingest_measurements.json` and
  `ingest_certificates.json` (`ingest_certify.py`; `python -m
  benchmarks.qualification --check ingest` fails when stale): torn reads
  detected in 500 of 500 mutated worlds and 0 of 500 static ones; planted
  fan-out, dropped rows and torn extracts detected in 500 of 500 each and 0
  of 500 clean extracts flagged (certificates `ingest-torn-read`,
  `ingest-reconcile`); 0 rows leaked by the view over 500 late-arrival and
  backfill worlds where an event-time-only filter leaked in all 500; equality
  to the independent row-by-row reference `ingest_naive.py` (to a relative
  1e-9) on 1000 random schemas and 0 failures in 10,000 metamorphic rewrites;
  200 of 200 synthetic unit verdicts correct with no `coarser` reported as
  `match`, and a planted near key flipping within one row of its threshold;
  0 spurious auto-accepted links in 200 surrogate-key schemas; pair
  completeness and link recall 0.9986 on ten synthetic registries; spend
  within budget and identical stop points on 100 randomized budgets. Plants
  and registries are our own, so these are optimistic and the key, join and
  entity numbers have no external referent: **those gates stay descriptive.**
  **Not built, blocked on outside parties or on shared files:** the
  declared-key relational corpus and the labelled ER benchmarks (not
  registered in `corpora.json`), so key and foreign-key precision and recall
  on real schemas, and entity-resolution precision on a public benchmark, are
  unmeasured; the independent author of the as-of reference is a separate
  agent, not a person outside the project; serving-time feature logs; a
  data-engineering reviewer. Not wired into a charter run: the charter
  fields (`ingest_budget`, `feature_budget`, windows, `lag`, `delta_rec`), the
  stage hook before `custody`, the hand-off of the resolved entity as the
  split's group column and of the provenance table to F4 are open.
* Problem framing and the decision-cost model (F8), the parts that could be
  built and measured without outside parties. New package
  `scmle.data.framing`: `DecisionCostModel` (four costs, a threshold range,
  a source per entry; refuses a model where one action dominates) with net
  benefit, treat-all, treat-none, expected cost and re-thresholded decision
  curves; the identity `Cost = const - c_FN NB(t)` is tested to 1e-9 over
  10,000 random cost matrices. `AnalysisCharter` gains an optional
  `cost_model` (charter `FORMAT_VERSION` 2; charters without one keep their
  hash and format 1 files load). `baseline`: the model against the current
  process as an intersection-union test at the two ends of the cost range
  with a cluster-robust t test, registered by hash before `C` is read and
  evaluated once (a second call raises); a selective-label detector refuses
  the claim when outcome observation depends on the baseline decision.
  `feasibility`: the closed form applied to the gains measured on `X` with a
  design effect and a chi-square interval for `s_d`; the verdict is
  `inconclusive` when it changes across that interval. `questions`: a
  tipping-point test, so only unknowns that change the verdict are asked.
  `draft` and `refusal`: a deterministic slot-filling drafter, data-side
  checks, an LLM slot-filler interface with recorded-response replay (no
  client ships), canonical slot values, the ambiguity score, and `tau` by
  upper Clopper-Pearson bound with its risk-coverage curve. `scmle frame`
  writes `framing_draft.json`, `decision_cost_model.json`,
  `baseline_report.json`, `framing_questions.json` and `framing_refusal.json`;
  a charter run writes a descriptive `framing_link.json`. Measured in
  `benchmarks/qualification/framing_measurements.json` and
  `framing_certificates.json` (`framing_certify.py`; `python -m
  benchmarks.qualification --check framing` fails when stale): empirical size
  of the baseline claim at most 0.0597 at 20, 50 and 200 clusters in three
  null worlds (so `MIN_CLUSTERS = 20`), power 0.97 where the closed form says
  0.95, selective labels detected in 500 of 500 planted and 0 of 500 clean
  worlds (the strong plants only; certificate `framing-selective-labels`),
  feasibility agreeing with the oracle in 98.6% of 1000 runs away from the
  boundary, question recall and false-question rate on planted omissions,
  1000 malformed charter records all rejected. Plants are synthetic and
  Gaussian-like, so these are optimistic for real data. **Not built, blocked
  on outside parties:** the expert-written gold corpus (so no `tau` is on
  file and the LLM slot-filler is descriptive and non-gating, and the
  end-to-end draft correctness rate is unmeasured), the comparison with the R
  decision-curve packages (the tests use an independent loop implementation
  of the same definition), and a domain owner's sign-off on a cost range.
  The baseline claim is not registered in the F3 ledger (its hypotheses are
  single-feature statistics) and is not run by `scmle run`; unsigned-LLM tier
  capping (`provenance.claim_tier`) is a tested function that the run does not
  call yet.
* Uncertainty and problem-type breadth (F12), the parts that could be built
  and measured without outside data. New in `scmle.evaluation`: per-class
  (Mondrian) conformal margins stored beside the pooled scores
  (`ConformalCalibrator.class_margins`, `sets(..., conditional="class")`,
  `ModelBundle.predict_set(..., conditional="class")`; a class with fewer than
  `ceil(1/(1-coverage)) - 1` calibration rows is kept in every set and listed
  by `thin_classes`); weighted conformal for covariate shift
  (`weighted_quantile`, `weighted_margins`, `ConformalCalibrator.
  weighted_interval`, and `shift_weights` for clipped discriminator odds with
  an effective-sample-size refusal below 50); `cv_plus_bounds` (CV+ from
  stored fold models; the served bundle still uses pooled out-of-fold scores
  and a refit, so no theorem covers the served interval and the stage was not
  changed); `coverage` (Clopper-Pearson, Holm-corrected slice flags, class
  coverage, the coverage tier of F12 A5); `proper_scores` (pinball, interval
  score, WIS, CRPS from a quantile grid, standard MASE with the in-sample
  scale); `ipcw` (censoring weights from a Cox model on covariates, pair
  weights `1/(G(T_i|x_i) G(T_i|x_j))`, Uno-style truncated concordance, the
  number of floored weights); `survival.floored_count`; `reconcile` (bottom-up
  and MinT with a shrunk covariance); `ranking.query_bootstrap_ndcg` (queries
  are the resampling unit; queries with nothing relevant are excluded and
  counted); `task_gate` and `task_gate_status.json` (every task type is
  `passed`, `failed` or `no_referent`; only `passed` types may keep a
  `confirmed` tier). Measured gates are in
  `benchmarks/qualification/uncertainty_certificates.json`
  (`uncertainty_certify.py`, `--check` fails when stale): G1 split conformal
  (mean coverage in the derived band, Kolmogorov-Smirnov against
  `Beta(451, 50)`), G2 CV+ with 1-NN and linear learners at K = 10 and 5, G4
  class-conditional sets (the pooled margin covers the 2 percent class 0
  percent of the time), G6 weighted conformal (known ratio passes, unweighted
  covers 0.73; estimated ratio also passes), G9 MinT against bottom-up, G11(i)
  covariate-dependent IPCW (unbiased; the marginal Kaplan-Meier version is
  biased by -0.017), G13 entity-wise calibration. Two gates missed their
  pre-registered floors and are published as misses, their methods
  descriptive: G3 (CQR in the highest-noise decile covers 0.864, floor 0.87;
  constant width 0.72) and G5 (familywise false-flag rate of the slice flag
  0.0755 against a bound of 0.0597). G13 did not show row-wise calibration to
  be anti-conservative in this world. Not run, for lack of outside data or
  tools: real-data coverage (G7), M4/M5 (G8), the MAPIE, scikit-survival and
  R differentials, a public ranking set, multilabel (G12), near-duplicate
  grouping (G14) and encoder baselines (G15). Behaviour changes:
  `conformal.json` gains `class_scores` (old bundles load; class-conditional
  sets then raise); `ModelBundle.predict_set` gained `conditional`;
  `ForecastMetric` gained `scale` and the kinds `mase_same_rows` and
  `mase_insample` (`mase` remains the same-rows score, which divides by the
  no-change error on the evaluation rows and is not comparable with published
  MASE); `benchmarks/reliability.py` `interval_coverage` now needs observed
  coverage >= 0.88 (nominal minus the 0.02 tolerance) instead of 0.85 and
  records the Clopper-Pearson interval. Not changed: the covariate-free IPCW
  floor of 0.05 in `survival.brier_scores` (its count is now available), the
  fold-model bundle storage, intervals for survival, ordinal, ranking,
  multilabel, multioutput, forecast and uplift (still `None`), and the
  temporal-data retrain shuffle (Phase 13).
* Leakage screen with error control, for runs with and without a charter
  (`scmle.data.admissibility.screen`). It replaces the profiler's fixed
  cutoffs (`LEAK_ASSOCIATION = 0.98`, `MISSINGNESS_LEAK = 0.9`,
  `DatasetProfiler.leakage_findings`) and the `"leakage"` string matches in
  `core/stages.py` and `intelligence/recommendations.py`. For every candidate
  column of the training rows: a cross-fitted held-out skill (AUC, or R^2) with
  an exact permutation p-value `(b+1)/(B+1)` (within groups or time blocks when
  known), Cramer's V of missingness with Fisher or chi-square (permutation with
  groups), and a record-purity test for a value captured only for some
  outcomes; all in one Benjamini-Yekutieli family at 0.05 in exact
  arithmetic. A column is a suspect only if it is significant and the lower
  confidence bound of its effect size exceeds the pre-registered ceiling
  (skill 0.95, association 0.9): significance alone is not leakage. Guarantee:
  under the null, and conditional on the model and fold choices, the chance of
  any false alarm is at most 0.05 and the FDR of the significant list is at
  most 0.05 under arbitrary dependence; it cannot tell a strong legitimate
  predictor from a leak without provenance. The cross-fitted AUC is counted
  within folds, because pooling predictions across folds orders same-level rows
  against their labels (on the first version of this screen, a 2 percent noisy
  copy was found in 144 of 200 plants with the pooled AUC and 199 of 200
  within folds). `leakage_policy="block"` now blocks on a
  typed result (`DatasetProfile.blocking_leaks`) and only for mechanisms with a
  current passing certificate, shipped in
  `scmle/data/admissibility/screen_certificates.json`
  (`benchmarks/qualification/screen_certify.py`; measurements in
  `screen_measurements.json`). Binary classification is certified; regression
  and multiclass targets get warnings with p-values and never block. Behaviour
  changes: `dataset_profile.json` has a `leakage` object; Titanic `body` is now
  flagged by record purity (it is recorded only for passengers who died) and
  not by a missingness score; `risk.leakage_hypotheses` in the fingerprint now
  counts screen suspects (`target_association` stays a descriptive statistic
  with no cutoff); `DatasetProfile` gained the optional field `leakage`. The
  charter path (`admissibility.qualify`) still uses the earlier
  `admissibility.leakage` scan, unchanged.
* Data qualification foundations (no behavior change to existing runs): record
  types `Check`, `Claim` and `Certificate` with evidence tiers
  (`scmle.contracts.qualification`), an exact-arithmetic error budget that keeps
  exploration wealth and the confirmation level as separate types
  (`scmle.data.ledger.budget`), and keyed, order-independent random streams
  (`scmle.core.randomness`). None are top-level exports yet.
* Qualification verification harness (benchmarks/qualification): defect injector, semi-synthetic simulator, frozen scipy-generated reference oracles, corpus manifest, certificate store that refuses stale or missing certificates for gating detectors; mutation scope now covers the qualification modules.
* Analysis charter and custody (opt-in; runs without a charter are unchanged).
  `scmle charter init|check|sign` and `MLProject(charter=...)` /
  `--charter` fix the question, an ICH E9(R1)-style estimand, the decision map,
  set sizes and the error budget before profiling, with a SHA-256 hash and an
  optional HMAC signature. A charter run carves a confirmation set `C` out of
  the development rows (group- and duplicate-aware, time blocks with an embargo
  for temporal data) and seals it in a separate vault process, even with
  `enforcement="cooperative"`; profiling and trials see exploration rows only.
  Closed-form power gives a `feasible`, `downgrade` or `infeasible` verdict:
  when the set is too small, or the carve would leave a validation fold
  empty, the run completes as `exploratory_unconfirmed` with no `C`.
  New artifacts: `analysis_charter.json`, `error_budget.json`,
  `feasibility.json`, `split_manifest.json` (v2, full 64-hex SHA-256) and the
  hash-chained `custody_log.jsonl`. The `MLProject` signature snapshot was
  regenerated for the new optional `charter` argument. Custody is
  tamper-evident on one machine, not adversary-proof; see `docs/SECURITY.md`.
* Claim ledger and confirmation (charter runs): hash-chained append-only
  `hypothesis_ledger.jsonl`, online FDR rules (e-LOND default, valid under
  arbitrary dependence; LORD++ and SAFFRON labelled independence-only),
  permuted-target negative controls, and one-time confirmation of exploration
  survivors by BY (default), BH, Holm or Bonferroni. New artifacts:
  `negative_control.json`, `confirmed_claims.json`. `--check fdr` runs the
  simulation gate. Thresholdout is not implemented.
* Admissibility and provenance (`scmle.data.admissibility`): field provenance,
  as-of reconstruction, admissibility rules by estimand, falsify-only DAG
  checks, leakage tests and reconciliation, with certificates in
  `benchmarks/qualification/f4_certificates.json`. The leakage skill test stays
  descriptive until noisy-copy leaks are certified.
* Detectors with certificates (`scmle.data.detectors`): missingness, label
  noise, sentinels, duplicates and outlier triage. `--write-f5` regenerates
  `detector_certificates.json`; `--check` verifies it. Certificates cover one
  corpus; label-noise detection is descriptive.
* Thread safety: the estimator and metric registries, experiment store, model
  registry, key registry, monitoring journals, shadow evaluator and bundle
  history are safe to use from several threads in one process (see
  `docs/ARCHITECTURE.md`, Concurrency). Separate processes sharing one
  registry or run directory are still not coordinated.
* The L1 feature selector no longer fails on multiclass targets.
* The public API snapshot (`tests/fixtures/public_api.json`) was regenerated
  after the `scmle` rename, which had left it stale; the set of exports is
  unchanged. `scripts/check_api_compatibility.py --write` regenerates it.
* Replay verification now reports `different` when a trial's status changed
  between the original run and the rerun; it used to report `exact`.
* Registered TabPFN engines refuse `sample_weight` instead of ignoring it, and
  a skipped conformal calibration is recorded as a `conformal.skipped` event.
* Model bundles now checksum `schema.json` and use bundle format 2. Legacy
  format 1 bundles are refused because their schema was not authenticated.

## 0.1.0b1

First public release.

* Distribution name `scmle`; import name and command `scmle`.
  Environment variables are `SCMLE_SIGNING_KEY` and `SCMLE_TRUSTED`; the base
  exception is `ScmleError`.
* Windows, macOS and Linux are supported and tested in CI. Helper processes
  (the test-set vault and the PyTorch worker) connect over an authenticated
  local channel instead of an inherited file descriptor; trial supervision
  uses `spawn` and `psutil` on Windows; all text files are read and written
  as UTF-8 with LF line endings so signed bundles verify on every platform.
* The HTTP serving adapter refuses request bodies above 16 MiB with 413.
* Added `SECURITY.md`, a code of conduct, issue and pull-request templates,
  Dependabot, and a release workflow that checks the tag against the package
  version and changelog and creates a GitHub release.

* `FairnessSpec(mitigation="reweighing")` weights development rows so group and
  label are independent in training; locked rows are untouched.
* `selection.sensitivity_repeats` runs the sensitivity analysis on the selected
  model and stores it in `evaluation["sensitivity"]`.
* Clustering (`scmle.unsupervised.clustering.find_clusters`): k-means, Gaussian
  mixture and agglomerative at several counts, scored by silhouette,
  Davies-Bouldin, Calinski-Harabasz and bootstrap stability.
* Competing risks (`CauseSpecificCox`, Aalen-Johansen `cumulative_incidence`)
  and time-varying covariates (`TimeVaryingCox`, counting-process rows).
* The profile stage now reports label noise and, given `reference=`, a gap
  between the development rows and a reference sample.
* Fairness reweighing (`reweighing_weights`): training weights that remove
  the association of a sensitive group and the label, used through
  `weight_column`.
* Dirichlet calibration (`bundle.calibrate(..., method="dirichlet")`) for
  multiclass models.
* Sensitivity analysis (`scmle.evaluation.sensitivity`): training (seed,
  resample), evaluation (split) and performance (input noise) sensitivity.
* `RollbackPolicy.tolerance` now defaults to 0.01 so routine retraining noise
  does not undo promotions.
* Data quality: label-noise detection (`find_label_noise`) and
  representativeness against a reference sample (`compare_to_reference`).
* Lifecycle: shadow evaluation (`ShadowEvaluator`), A/B analysis with an
  independent bootstrap (`compare_arms`), and rollback criteria
  (`LifecycleManager.check_rollback`, `previous` registry alias).
* Package layout follows the section 3 module tree under a flat `scmle/`;
  modules carry their real names, with no compatibility aliases.
* Logging replaces `print` in the library, CLI, benchmarks and examples.
* PyTorch model families (MLP, residual, TabM-style ensemble,
  FT-Transformer, RealMLP-style) computed in a dedicated worker process;
  warm-up and cosine schedule, automatic batch size, gradient clipping and a
  training-health report; `deeplearning` extra.
* Bagging adapters; early stopping for LightGBM, XGBoost and CatBoost.
* Fit-health verdicts (healthy, overfit, underfit, unstable) per trial.
* Paired Bayesian tie handling in selection (`uncertainty="bayesian"`).
* Diagnostics: slices, robustness stress tests, adversarial validation,
  partial dependence, per-prediction attribution; reliability section in the
  report.
* Conformal prediction intervals and sets stored in the bundle.
* Automatic feature construction and L1 selection as tested strategies.
* Profile-driven recommendations; run audit; `scmle.predict` / `scmle.load_model`.
* Under hard enforcement the locked test rows live in a separate vault
  process.
* Rolling-origin validation and `data.feature_delays`.
* `models.exclude_models` / `models.exclude_families`.
* Datetime columns become calendar and trend features; free-text columns are
  encoded as fold-local TF-IDF terms.
* Missing-value indicator and iterative-imputation strategies; arcsinh target
  transform for long-tailed regression targets.
* `scmle.register_metric` for user-defined metrics.
* Greedy weighted blending searched alongside stacking.
* Fairness report (`data.sensitive_columns`), out-of-distribution flags in
  the bundle, a learning curve, monotonic constraints
  (`data.monotonic_constraints`).
* Per-row training weights (`data.weight_column`, `--weight-column`): used
  directly by models that accept `sample_weight`, otherwise by a weighted
  draw of training rows. Test metrics are also reported weighted
  (`weighted_test_metrics`).
* Conformalized quantile intervals (`predict_interval(method="quantile")`):
  widths follow the row's noise; the default residual method is unchanged.
* `data.time_as_feature` offers the time column to the model (opt-in).
* Unlabeled rows: `data.unlabeled` (`block`, `ignore`, `pseudo_label`);
  self-training is judged in cross-validation and used only if it wins;
  `bundle.rank_for_labeling` picks the rows to label next.
* Survival analysis: `task="survival"` with `event_column`; Cox and
  discrete-time boosted models, concordance index, `predict_survival`.
* Keras network (`keras_mlp`; Keras 3 on the PyTorch backend, in the same
  worker, with the `deeplearning` extra).
* Validation folds are scored with the weight column (`selection.weighted`),
  so selection agrees with the weighted test metrics.
* Benchmarks: `--seeds N` repeats each dataset and reports the advantage over
  the reference with a 95% interval; the verdict and the regression gate use
  the interval, so noise alone neither wins nor fails.
* Multiclass reliability report (top-label calibration error), calibration
  error before/after in `bundle.calibrate`, and a quantile-interval coverage
  curve on the locked rows (`evaluation["quantile_coverage"]`).
* ONNX export (`bundle.export_onnx`, `scmle export-onnx`, `onnx` extra)
  with a parity check; unsupported models are refused by name. `scmle
  serve` runs the HTTP server; schema violations return 422 and `/health`
  reports the model version and checksum.
* `scmle.register_estimator` registers any scikit-learn-compatible estimator
  class as a model (conformance-checked, picklable for trial processes).
* Ordinal targets: `task="ordinal"` with `ordinal_order`; thresholded
  regressors give ordered labels and probabilities, scored by quadratic
  weighted kappa and mean absolute class error.
* Survival: event-stratified splits and censoring-weighted integrated Brier
  score and dynamic AUC beside the concordance index. Competing risks and
  time-varying covariates are not built.
* Duplicate rows (equal features) are kept on one side of the holdout and of
  every fold; the manifest notes how many groups were kept together.
* Missingness report: per-column tests of whether gaps relate to the target
  (development rows, Bonferroni-corrected); missing-value flags are planned
  and built only for the flagged columns.
* `bundle.environment_lock()` writes the build environment as pinned
  requirements; a dependency mismatch at load carries it in the error.
* `bundle.counterfactual`: the smallest change to actionable columns that
  gives a wanted class or value range; unneeded changes are pruned and other
  columns never change.
* Lifecycle retraining policy: one retrain per drift event (data digest),
  a row cooldown, a minimum of labeled rows, and an audit record for every
  review, skip, rejection and promotion with its evidence.
* Production for any model: `bundle.export_service` / `scmle package`
  write a service folder (bundle, pinned requirements, build environment,
  server script, Dockerfile). ONNX also covers LightGBM and XGBoost with
  `onnxmltools`; refusals point to the service package.
* Fairness mitigation (`fairness=FairnessSpec(mitigation="group_thresholds")`):
  per-group thresholds judged on validation, reported on the locked rows,
  logged in the decision log. Reweighing and constrained fitting are not built.
* Learning to rank: `task="ranking"` with `query_column`; pointwise,
  LightGBM and XGBoost rankers, NDCG@10 and MAP@10, query-disjoint splits and
  `bundle.rank`.
* Text embeddings: `text_encoding="embedding"` encodes free text with a
  sentence-embedding model in the PyTorch worker (`embeddings` extra), each
  distinct string embedded once, falling back to TF-IDF with a warning.
* Multi-label and multi-output targets: pass a list of target columns with
  `task="multilabel"` or `"multioutput"`; iterative stratification keeps rare
  labels in every fold; `bundle.predict_frame` returns one column per target.
* Forecasting: `task="forecast"` with `horizon`, `time_column` and optional
  `group_column` and `seasonal_period`; lag and rolling features are added
  automatically, the split embargoes the horizon, MASE against no-change is
  the default metric, and `bundle.forecast(history)` forecasts any series.
  The backtest report gives per-step errors against last-value and
  seasonal-naive baselines and per-origin scores.
* Uplift modelling: `task="uplift"` with `treatment_column`; T-, S- and
  X-learners, Qini coefficient and uplift@30%, treatment/outcome-stratified
  splits, a propensity check that flags non-randomized assignment, and the
  no-unmeasured-confounding assumption stated in the report.
* Files larger than memory: `memory_budget_gb` streams a Parquet or CSV file
  in batches. Locked rows are chosen by hashing row positions, development
  uses a bounded sample, and every locked row is scored by streaming the
  file. `training="stream"` keeps incremental models (`sgd_logistic`,
  `sgd_regression`) and updates the final one over the whole development
  partition; other models are rejected with a stated reason.
* Multi-node Ray: `executor="ray"` with `cluster_address`; trial results
  match a single-node run, hard limits hold on workers, and a lost worker is
  retried once and then recorded as a failed trial instead of ending the run.
* `ProblemSpec`: state the decision, population and success criteria;
  the criteria are checked on the locked rows and shown in the summary
  (`result.meets_criteria`).
* Evaluation completeness: MRR and precision/recall@10 for ranking, optional
  nested cross-validation (`selection.nested_cv`) reporting the optimism of
  the winning score, a warning for exactly equal classes, and a note that
  validation and test keep the natural class mix.
* Bias–variance report (`selection.bias_variance_replicates`): bootstrap
  decomposition of the selected model's error with a nearest-neighbour noise
  estimate, a verdict and advice. It reproduces resampling variance only; the
  limitation is stated in the module and the report.
* PyTorch networks: optimizer choice (`adam`, `adamw`, `sgd` with momentum,
  `rmsprop`, full-batch `lbfgs`), Huber and focal losses, label smoothing and
  gradient accumulation.
* Fixes found on real data: the forecasting backtest crashed under hard
  isolation (the seasonal baseline now comes from split time), and the
  tight-budget rule rejected the boosted learners of the newer tasks.
* Tuning breadth: a grid optimizer, Hyperband brackets (`search.pruning`,
  `search.eta`) and Pareto selection (`selection.pareto`) that reports the
  score / latency / size front and picks the fastest model within a score
  tolerance.
* Optional TabPFN adapter (`tabpfn` extra; needs an accepted license).
* Benchmarks: Hugging Face source and suites (`--suite`).
* Post-selection diagnostics are bounded by a share of the time budget.

## 0.1.0b1

First public beta.

* Typed, versioned project specification; deterministic config hash.
* Leakage-safe workflow: locked test set, fold-local fitting, split
  strategies (stratified, random, group, temporal), leakage hypotheses
  including outcome-dependent recording.
* Model registry with scikit-learn, LightGBM and XGBoost adapters, cross-fitted
  stacking, pruned hyperparameter search (random or Optuna), data strategies
  (class weighting, resampling, calibration, log and interaction features),
  threshold optimization, serving-constraint selection.
* Diagnostics and explainability; decision log; HTML report.
* Resumable runs, SQLite (and MLflow) tracking, parallel trials (process
  pool or Ray), meta-learning from run history.
* Signed, checksummed bundles; refuses unauthenticated loading by default.
* Monitoring (drift with sampling-noise floor, delayed labels, advisor),
  champion-challenger, model registry, batch and HTTP serving, lifecycle
  manager.
* SVM, kNN (size-limited), CatBoost adapters; SMOTE-family and opt-in jitter
  augmentation strategies; probability calibration strategy.
* Full dataset fingerprint (shape, types, missingness, target, features,
  relationships, structure, quality, operational cost, risk).
* Structure-aware bootstrap intervals; resource governor; policy
  checkpoints; data-access ledger and audit events; explicit inference
  column mapping; replay verification.
* Declarative, versioned feature-transformation graph (18 operations,
  fold-fitted statistics, datetime parts, per-entity lag / diff / rolling
  features) packaged in the bundle.
* Arrow-compatible interchange (Arrow, Polars, Feather); Gaussian-copula
  synthetic data with a fidelity evaluator; neural-network family;
  feature-availability contract and temporal embargo.
* MLflow tracking adapter and Ray execution backend.
* CLI: `plan`, `run`, `resume`, `predict`.
* Benchmarks on nine open datasets with a regression gate.
* Python 3.11 or newer (tested 3.11 to 3.14).
* Enforced resource limits: each trial runs in a supervised process that is
  killed past its time ceiling or memory limit.
* The saved model carries per-entity history, so lag / rolling features work
  on a single new record; serving adapters keep the history current.
* Profiling, feature selection and leakage checks use the development
  partition only; trials record what each fold fit and scored on and the
  ledger verifies it against the split manifest.
* Adapter capabilities drive preprocessing and planning (native categorical
  handling, no imputation for models that handle missing values, sparse
  input); high-cardinality, regression rare-target and short-budget policies.
* Group- and time-aware inner cross-validation for calibration and stacking.
* Selection rules: tie-breaker, tolerance, one-standard-error rule, hardware
  safety factor; Nadeau-Bengio corrected confidence interval.
* Plugin discovery gated by the conformance suite; registry authenticates
  bundles; tracking lineage with artifact checksums; save-and-reload parity
  verification; optional probability calibration at inference.
* Workflow compiler (stages declare dependencies).
* Admissibility and detectors in charter runs (runs without a charter are
  unchanged): leakage suspicion tests, the admissibility table and the F5
  detector pipeline now run on the exploration set `X` only, and write
  `admissibility_table.json` and `detector_certificates.json` (the
  certificates the run used). Each leakage check is chained into the ledger as a new
  `diagnostic` row that spends no wealth. Certificates are read from
  `benchmarks/qualification/`, which the installed package does not ship; when
  the files are missing every detector is descriptive. The shipped leakage
  certificate is not gating (the noisy-copy leak type is descriptive), so a
  suspicion is reported but quarantines nothing yet. The old profiler
  constants `LEAK_ASSOCIATION`, `MISSINGNESS_LEAK` and the `LEAKAGE_MARKER`
  string match remain: they still drive `leakage_policy="block"` for runs
  without a charter.
* Robustness and shift (`scmle.data.robustness`; runs without a charter are
  unchanged). A charter run re-runs each exploration survivor over the
  multiverse the charter declared (imputation, outlier rule, inclusion; other
  declared choices are reported as not run) with bootstrap perturbation
  intervals, a survival rule (sign, size band, interval) and a permutation test
  of the whole specification curve, and compares `X` with the `reference`
  sample by a classifier two-sample test and a kernel MMD test with permutation
  p-values, per-feature KS or chi-square under BH, and subgroup coverage. New
  artifacts: `stability_report.json` and `shift_report.json`. Without a
  deployment sample the shift status is `unverifiable`, never `pass`. The
  forward-in-time check on the holdout is implemented (`forward.forward_check`,
  opens the set once) but a run reports it as skipped, because the final test
  set is opened only by the final evaluation. Certificates are in
  `benchmarks/qualification/f6_certificates.json` (`f6_certify.py`): the
  classifier test is certified (false alarms 24 in 1000 nulls, power 0.94 at a
  planted AUC of 0.60 with 2000 rows a side); the MMD test is measured and
  stays descriptive because its false-alarm upper bound (0.0765) is just over
  the pre-registered 0.075; the stability rule is certified on semi-synthetic
  plants only. Stability verdicts do not yet remove findings from the
  confirmation set, per-period drift and calibrated PSI are not implemented,
  and the per-feature p-values are adjusted within the report, not entered in
  the ledger. The old `0.65` representativeness constant is unchanged.
