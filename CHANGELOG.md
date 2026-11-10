# Changelog

All notable changes to SCMLE are recorded in this file.

The format follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/)
and the project aims to follow [Semantic Versioning](https://semver.org/).
SCMLE is a beta (`0.1.0b1`): the public API may change between `0.x` releases
(see [`docs/API_STABILITY.md`](docs/API_STABILITY.md)). Breaking changes are
marked `Breaking:` and listed first in their group.

## [Unreleased]

### Breaking changes

* **No package-root exports.** `scmle` exports only `__version__`. Every public
  name is imported from the module that defines it: `scmle.api.project.Project`,
  not `scmle.Project`. The 42 former root names, and where each now lives, are
  listed in `scripts/check_api_compatibility.py` (`ROOT_SURFACES`).
* **No aliasing.** `import x as y`, `from m import n as a`, module-level
  `NAME = other.name` assignments and type aliases are removed everywhere in the
  package, tests, benchmarks, scripts and examples, and a test fails on any new
  one. `numpy`, `pandas` and `pyarrow` are written in full.
* **Class names.** About 470 classes were renamed to remove banned words
  (`Adapter`, `Engine`, `Manager`, `Handler`, `Options`), most of the `Spec`,
  `Policy`, `Rule`, `Strategy`, `Report` and `Result` suffixes and redundant
  package prefixes. `MLProject` is now `Project`. The old-to-new map is the
  reviewed registry kept with the project plan.
* **Public-API snapshot.** `tests/fixtures/public_api.json` is schema version 2:
  pinned names are listed under their defining module.

* **Bundle format 3.** A model bundle stores the module path and class name of
  every object inside it, and those names changed, so bundles written by an
  earlier build cannot be loaded. `Bundle.load` refuses them before it
  unpickles anything, with an `ArtifactError` that says the format it reads and
  to retrain and save again. There is no converter.

### Added

- Unattended operation, tested on simulated streams only
  ([`docs/AUTONOMY.md`](docs/AUTONOMY.md)). An envelope
  (`scmle.governance.envelope`) is authority a person signs in advance for
  `train_challenger`, with a scope, dates and a use limit; the eleven
  never-list actions cannot be put in one, and every promotion still needs a
  person's approval.
- `scmle operate tick | status | grant | revoke | halt`
  (`scmle.operations.controller`) applies each inbox file exactly once, spends
  an envelope before a retrain and escalates when data or labels stop. The
  deployment ledger gains `envelope`, `envelope_used`, `envelope_revoked` and
  `controller` rows that the chain checks. `alarm_gates_retrain` stays off
  because the alarm design has not met its detection-delay gates, so an alarm
  still asks a person.
- The coverage claim is tested on the sealed rows `C`
  (`UncertaintySpec.coverage_on_confirmation`, charter format 11,
  `scmle.evaluation.coverage_confirm`). It is a registered claim about one
  served interval or set named in advance (the `residual` interval for
  regression, the pooled set for classification, level 0.9, tolerance 0.02),
  `C` stays sealed until the bundle and its calibration are frozen, and the one
  opening runs an exact one-sided binomial test of "coverage is at most level
  less tolerance" at 0.05 (fixed in the module, not drawn from the error
  budget). The status is `confirmed` when it rejects and `unverified`
  otherwise; `claims verify` recomputes the p-value from the counts. In a small
  test run (120 rows of `C`) 110 were covered, p = 0.134, so the claim was
  `unverified`. A format 10 charter keeps its hash and signature.
- The coverage claim is registered before it is read
  (`coverage_run.claim_document`, `coverage_claim.json`): task, levels (0.5,
  0.8, 0.9, 0.95), possible methods, tolerance (0.02), the exact-lower-limit
  rule, the split fingerprint and the charter hash. Its hash is announced by a
  `coverage.registered` event and, in a charter run, appended to the custody
  log; `claims verify` gains the step "coverage claim: registered before the
  rows were read", skipped for a run without a custody log.
- Coverage and set size per declared slice column
  (`scmle.evaluation.coverage_slices`, `UncertaintySpec.slice_columns`,
  `declared_slices` in `coverage_report.json`): per level of the column (a
  numeric column is cut into four bands at quantiles of the development rows),
  the rows, the share covered with its exact Clopper-Pearson interval and the
  mean width or set size. Slices under 30 rows are counted and not shown, and a
  column the rows lack or with more than 12 levels is `not_assessable`.
- A certified slice flag for coverage: each slice of at least 30 rows is tested
  against the nominal level less a margin of 0.05, with Holm over the family
  (`n_cal = 500`, R = 2000). Familywise false-flag rate is 0.0005 against the
  bound 0.0597; detection of a slice at 3x noise (true coverage about 0.42) is
  0.992 at 30 rows and 1.0 from 50 rows (Wilson lower 0.986), so it is
  certified from 30 rows. Detection at 1.5x noise (about 0.73) is 0.17, 0.38,
  0.71, 0.93 and 1.0 at 30, 50, 100, 200 and 400 rows, and at 1.25x (about
  0.81) 0.02 to 0.37; these are filed, not gated, so the flag cannot see a
  shortfall under about 0.05. Records carry `flag`, `flag_certified` and
  `flag_gate`, which is `gating` only while the stored certificate is current
  and passing.
- The CV+ size cap is a charter field (charter format 10;
  `UncertaintySpec.cv_plus_max_bytes` in the optional `uncertainty` block). A
  model set above the cap is not kept and `cv_plus.skipped` says so; without
  the field the run uses `cv_plus.MAX_BYTES`. Charters of format 1 to 9 keep
  their hash and signature.
- The conformal score source is chosen by a tested rule
  (`cv_plus.plan_scores`) and recorded in a `conformal.scores` event. From 1000
  calibration rows (`cv_plus.MIN_ROWS_FOR_SPLIT`) the bundle serves the pooled
  out-of-fold scores alone; below that, with partitioning held-out folds and
  models within the size cap, the K fold models are also kept for CV+
  intervals, and otherwise the pooled scores are used with the reason in
  `cv_plus.skipped`. The behaviour is the one the run already had.
- The coverage monitor feeds the closed loop
  (`scmle.governance.uncertainty_feed.UncertaintyFeed`). For a classification
  bundle, `observe` sends the loop each case's conformal set size and
  out-of-distribution flag, and `mature` counts labels outside the set as
  coverage violations, chains an `observe_summary` ledger row and lets
  `LoopPolicy.coverage_target` escalate `coverage_below_nominal`. It gates
  nothing, labels for cases the feed did not serve are refused, and regression
  bundles are not covered.
- Adversarial validation reports odds (`AdversarialValidation.odds`):
  `p / (1 - p)` corrected for the sample sizes and clipped at
  `shift_weights.DEFAULT_CLIP` (20), as mean, 95th percentile, maximum, clipped
  share and Kish effective size, and says whether weighted conformal would be
  `possible` or `refused` (effective size against `MIN_EFFECTIVE_SIZE`, 50).
  The numbers are descriptive and do not replace the bundle's shift weights.
- Certificate for the Holm family of subgroup hypotheses
  (`benchmarks/qualification/subgroup_certify.py`,
  `subgroup_certificates.json`, `certified.subgroup_gate`). With eight equal
  levels (R = 2000), family-wise error is 0.0555 with no effect (Wilson 0.046
  to 0.066) and 0.0480 with two real effects of 0.5 sd (0.039 to 0.058), both
  within the floor 0.0597. Power for one effect of 0.5 sd (R = 500) is 0.420 at
  100 rows per level, 0.764 at 200 (Wilson lower 0.725, below the 0.80 floor)
  and 0.986 at 400 (lower 0.971), so it is certified from 400 rows per level;
  at 0.3 sd and 0.8 sd the numbers are filed, not gated. Unequal levels and
  overlapping columns gave 0.039, 0.044, 0.0535 and 0.022 against 0.0597
  (`subgroup_extended_measurements.json`).
- The subgroup gate decides only the word in the report: the `subgroups` key of
  `identification_report.json` has `gate` (`gating` with a current passing
  certificate, otherwise `descriptive`), `statement_gate` and per hypothesis
  `certified_size`, and `causal_claims.json` repeats the gate. The tier is
  unchanged whatever the gate, and `python -m benchmarks.qualification --check`
  now includes `subgroup_problems` and `subgroup_descriptive_only`.
- Source-side reconciliation in the charter-run ingest stage
  (`declared.reconcile_source`). For each view the stage asks the source for the
  row count, declared column sums and key checksum and compares them with the
  extracted frame (counts and checksums exactly, sums within `delta_rec`; a
  source total of zero is compared as an absolute gap). A mismatch stops the
  run with a `BlockedError` naming each quantity and side, and
  `extract_reconciliation.json` gains `source_checks` and `source_passed`.
  Offsetting row errors pass, because only aggregates are compared.
- The `known_entity_future` unit of prediction has a protocol
  (`ProfileStage.check_known_entity`, `custody.pipeline.apply_custody`). The
  carve of `C` ignores the group column, so an entity's rows may sit in `X` and
  `C`, and the final test rows are the latest period of the time-ordered split.
  The run refuses the charter without `data.group_column` or `data.time_column`
  or with a validation that is not temporal or rolling, and emits
  `unit.known_entity_future`. It is not measured on a real panel.
- Ingestion in a charter run (charter format 7; `scmle.core.ingest_stage`). The
  optional `ingest` block (`IngestSpec`) holds the source budget (queries, rows
  scanned, seconds), `B_feat`, windows, lag, `delta_rec`, tolerances and
  `strict`. With `data.ingest_plan` (CLI `--ingest-plan`,
  `MLProject(ingest_plan=)`), a stage before the split runs the ingest plan
  under the charter budget, stops with questions when the unit verdict is
  `coarser` or `unknown`, synthesizes the feature family at the cutoff
  `moment - lag`, joins it to the table and writes `feature_manifest.json` and
  `ingest_field_provenance.json`. It was tested on SQLite only.
- Extract reconciliation and entity resolution in the charter (charter format
  8). The optional `IngestSpec.reconcile` sub-block states per view a row
  count, column sums and a SHA-256 checksum of a key column; a mismatch raises
  a `BlockedError`, a pass writes `extract_reconciliation.json`. The optional
  `IngestSpec.entity` sub-block (comparison fields, blocking passes, group
  column name) makes the stage resolve the run table's rows, set
  `data.group_column` to the resolved entity (a conflicting caller value is a
  `ConfigError`) and write `entity_resolution.json`. On a table where each
  customer from 250 up repeats the email of another key, the default run splits
  250 duplicate pairs across development and test and the run with the block
  splits none; the result is `unverified`, not measured on labelled benchmarks.
- Decision reports in a charter run (`scmle.core.decision_reports`). With a
  decision block on a binary task, the final evaluation writes once, from the
  locked rows: `policy_value.json` (self-normalised IPW of the frozen rule with
  the charter's clip, support and effective-sample-size gates, from
  `data.logged_action_column` and `data.propensity_column`; `not_assessable`
  without both), the fairness satisfaction test for every declared constraint,
  `rollout_design.json` with a one-row hash-chained `rollout_decisions.jsonl`,
  `decision_threshold.json`, `net_benefit.json` and `decision_report.json`. The
  two columns (CLI `--logged-action-column`, `--propensity-column`) are never
  features.
- `claims verify` checks the decision files: a `decision_files` event in the
  custody log holds the hash of each, and the step "decision files: hashes
  match the custody log" fails by file name when a file differs, is missing,
  appeared without registration or was registered twice, and is `skipped` for
  an older run with no registration. The custody log is tamper-evident, not
  adversary-proof.
- Propensity-logged assignment for a decision rollout (`RolloutHook.assign`,
  deployment ledger kind `rollout_assignment`): units are assigned by the plan's
  salted hash and each batch is chained with the unit ids, the arm of each and
  the probability that arm had, so an off-policy estimate can be made from the
  ledger. The ledger refuses a batch for an experiment that was not started, a
  list of different length, an arm other than 0 or 1 and a probability outside
  (0, 1).
- Rollout decisions through the lifecycle manager and the deployment ledger
  (`scmle.governance.rollout_hook.RolloutHook`,
  `LifecycleManager.restore_previous`). The hook starts an experiment with the
  level taken from the plan's `DecisionBudget`, writes each batch decision as a
  chained `rollout_decision` row, and a `promote` decision as a
  `promotion_proposal`; the alias moves only through `promote_challenger`,
  which still needs the registry's independent signed approval, and moves back
  only through `restore_previous`.
- Once a rollout hook is bound (`LifecycleManager.defer_to_rollout`), the
  bootstrap promotion and rollback rules are descriptive: `review` stages the
  retrained challenger whatever the fixed-sample comparison says and audits it
  as `lifecycle.bootstrap_diagnostic` (`descriptive: true`, `decides: false`),
  and `check_rollback` never moves the alias or asks for approval. Only
  `RolloutHook.promote`, `watch` and `roll_back` decide. A manager with no hook
  behaves as before. The rollout certificate was re-issued with the same seeds
  and floors plus an amendment written before its first run: 400 of 400 worlds
  agree with the vectorised e-process (false promotion 0.0015 and 0.001, false
  rollback 0.0005, true-loss rollback share 0.008); no hook world reached a
  decision in the null scenarios.
- The baseline claim can be about the selected model
  (`--baseline-subject selected_model`, `MLProject(baseline_subject=...)`,
  `DataSpec.baseline_subject`, default `frozen_rule`). The run explores on `X`,
  holds `C` sealed until the selection is frozen, registers the claims by hash
  (model name, strategy, parameters, trial id, split fingerprint, seed) and
  opens `C` once. The logistic rule is reported in `baseline_report.json` as
  `reference_rule`, with its action agreement with the model on `C`.
  `governed.run_governed` is now `explore_governed` followed by
  `confirm_explored`.
- Attestors in the charter and signed attestations for causal claims (charter
  format 9; `CausalSpec.attestors`, `CausalSpec.attestations`,
  `scmle.data.identification.attest`). A statement counts for an assumption
  only if its attestor is listed for it, its scope is this charter's hash, the
  run date is within its dates and its HMAC-SHA256 signature verifies under the
  signing key; `causal_claims.json` records the accepted statements and why the
  others do not count. `claims verify` gains the step "causal attestations
  verify", skipped without the signing key. The signature is a shared-key HMAC,
  so it does not identify the person.
- Subgroup hypotheses in a causal claim (charter format 9; `CausalSpec.subgroups`,
  `identification.run.subgroup_report`): columns and levels fixed before any
  data is read and part of the registration hash. `identify` estimates the
  effect at each declared level on `C` and adjusts the family by Holm's
  step-down; a level that cannot be estimated stays in the family at p = 1. It
  does not change the tier. A test with 20 null subgroups and five seeds sees
  at least three unadjusted p < 0.05 and no Holm rejection.
- The causal estimate on `C` in a charter run (`scmle.core.causal_claim`). The
  primary hypothesis `theta = 0` is registered by the hash of the causal block,
  strategy, DAG hash, confirmation level and exploration row count, and
  `identify` runs on the frame `C` releases in the same opening; the five
  causal report files are hashed into the confirmation row. A `dag` path loads
  a stored DAG (`ConfigError` on a bad file). `qualify.run` applies
  `rules.decide_causal` to causal estimands with a graph (descendants of the
  treatment and colliders are quarantined), reports declared adjustment members
  the graph refused, tests the graph's implied independences on `X` and says
  when more than 16 candidates and no declared set leave the claim `attested`
  at best.
- Baseline claim in a charter run (`scmle.data.framing.claim`). With
  `data.baseline_column` (CLI `--baseline-column`, `MLProject(baseline_column=)`)
  on a binary task whose charter has a `cost_model`, a rule is frozen on the
  exploration rows (standardised ridge logistic regression acting at the cost
  model's implied threshold), and the claim that it beats the current process
  over the cost range is registered by hash and tested at both ends of the
  range on the one opening of `C` (`baseline_report.json`). The column is never
  a feature; selective labels flagged on `X` make the claim `not_issued`.
  `data.framing_draft` (`--framing-draft`) lets `provenance.claim_tier` cap the
  claim: an unsigned charter with an LLM-filled field, or a cost from data or a
  template, leaves it `unverified` (`framing_provenance.json`).
- Registered claims in the ledger (`scmle.data.ledger`). A `claim` row registers
  a claim that is not a single-feature statistic by the hash of its
  registration, before the survivors are frozen; it spends no wealth.
  `confirm.confirm` takes a `ClaimRunner` and gives it the one frame `C`
  releases after the bivariate family is tested. The confirmation row records
  each claim's status and the hash of each report file, and the ledger refuses a
  seal that omits a registered claim. `scmle claims list` prints the registered
  claims and identification status, and `claims verify` checks the report file
  hashes.
- Causal identification (`scmle/data/identification/`, charter format 4). An
  estimand may carry a causal block (`CausalSpec`, `NegativeControl`,
  `SensitivitySpec` in `scmle.data.charter.causal`): treatment, outcome,
  estimand, design columns, attested assumptions from a closed vocabulary,
  strategy priority, negative controls, strata, sensitivity benchmark and
  trimming rule. Charters of format 1 to 3 keep their hash and signature, and
  `DecisionAction` gained `refuse_claim`. The selector depends only on the
  charter and an optional DAG; when nothing is eligible the claim is
  `not_identified` with the missing assumptions and no estimate. A charter no
  strategy identifies still gets the refusal files from `X`, and `C` is not
  opened for it.
- Identification strategies (numpy and scipy only): `aipw` (nuisances fitted on
  `X`, score on `C`; ATE and ATT, overlap report, declared trimming),
  `randomized_diff`, `did` (group-time ATT against never-treated units, event
  study, a simplified relative-magnitudes sensitivity bound, not HonestDiD),
  `iv` (just-identified 2SLS, robust first-stage F, Anderson-Rubin set as
  headline below F = 10), `rd` (local linear with a bias-corrected interval,
  cross-validated bandwidth, density check, placebo cutoffs, balance) and
  `synth` (simplex weights, placebo-in-space, refusal below 19 donors at 0.05;
  at most `attested`). `battery` (Holm at `alpha_falsify`, one fifth of the
  confirmation level) and stratum replication (sign and Cochran's Q) can only
  falsify; `sensitivity` has the E-value, the Cinelli-Hazlett robustness value
  and benchmarked partial-R2 bounds. `tier.decide` applies the five conditions
  of the tier rule as a pure function of the report facts, and a gate counts
  only with a current passing certificate. `run.identify` writes
  `identification_plan.json`, `identification_report.json`,
  `falsification_report.json`, `sensitivity_report.json` and
  `causal_claims.json`.
- Causal identification certificates
  (`benchmarks/qualification/f10_certify.py`, `f10_certificates.json`), on
  simulated worlds only: refusal worlds 200/200 and 200/200; 95 percent
  interval coverage 0.944 (AIPW ATE, linear), 0.948 (ATT), 0.945 (difference in
  means), 0.944 and 0.951 (DiD single and staggered), 0.957 (IV, strong) and
  0.953 (RD); weak-instrument headline set 0.950, weak flag recall 0.982, false
  weak 0; placebo-in-space size 0.040 (bound 0.089); the overlap gate fires
  0.006 at minimum propensity 0.1 and 1.0 at 0.02 and 0.005. At
  `alpha_falsify` = 0.01 the smallest planted violations with power of at least
  0.8 (Wilson lower) are a shared hidden confounder of 0.5 (negative-control
  outcome), a trend slope of 0.3 per period (pre-trend Wald), a sorting share
  of 0.5 (density) and a covariate jump of 0.3 (balance and placebo cutoffs).
  The sensitivity bound covers 100 percent for a confounder at half the
  strongest observed covariate's strength, 80 percent at the benchmark and 0
  percent at twice it (reported, not gated).
- Limits of those certificates: the coverage gate for AIPW with boosted
  learners on a nonlinear surface missed its pre-registered floor (469 of 500
  covered, Wilson lower 0.913 against 0.93) and was not retuned, so that gate
  is descriptive and the package certifies no back-door claim as `confirmed`
  until it is re-measured; the two linear worlds passed. The end-to-end
  false-confirm gate passed (0 of 300 runs `confirmed` with a hidden confounder
  shared with the negative control), but with a confounder the negative control
  does not share and a DAG that omits it, 97.7 percent of runs reached
  `confirmed` (clean world 97.0 percent). Not measured: worlds on real
  covariates, public gold-standard datasets, frozen reference outputs of
  `rdrobust`, `sensemakr`, `EValue`, `did` and `HonestDiD`, Python 3.11 versus
  3.14 bit-identity, an outside-party replication. The Cinelli-Hazlett
  conversions and the literature claims behind the DiD, IV and RD methods are
  from memory and unverified against the papers.
- Decision layer (2026-11-04; `scmle.decision`; charter format 5 adds an
  optional `decision` block in `scmle.data.charter.decision` that needs a
  `cost_model`): cost-optimal thresholds with a calibration gate, net benefit of
  a frozen rule against treat none, treat all and the incumbent with a
  simultaneous cluster-bootstrap band, off-policy evaluation by IPW, SNIPS and
  doubly robust estimators that refuse on a failed support or
  effective-sample-size gate, fairness constraints with an upper-bound
  satisfaction test and group thresholds, and an anytime-valid betting-mixture
  e-process. `design` (salted-hash assignment, fixed-horizon size, simulated
  expected sample number, exact and anytime sample-ratio checks) and `rollout`
  (promote, hold, `hold_final`, `hold_invalid` by intersection-union over the
  primary and guardrail e-processes, a rollback monitor per champion, a
  hash-chained decision log) sit on top. `DecisionBudget` keeps exact `Fraction`
  levels; promotion levels use the discount-only rule, since the e-LOND
  rejection boost's guarantee for overlapping experiments was not checked. A
  charter without the block hashes, signs and loads as before; with it, the
  threshold comes from the chartered cost model and `evaluation.json` carries
  the calibration path and rule hash under a `decision` entry.
- Decision-layer certificates (`benchmarks/qualification/f11_certify.py`,
  `f11_certificates.json`): 19 of 20 gates met their floors. The net benefit
  band missed its floor in one setting (independent rows, n = 2000: coverage
  0.9375, Wilson interval 0.926 to 0.947 against 0.93) and is descriptive.
  Ever-promote under peeking at every unit at the boundary null is 37 in 10000;
  false rollback 19 in 10000; `abtest.compare_arms` at 6 looks under no effect
  adopted in 10 of 100 runs (Wilson 0.055 to 0.174), published, not gated. The
  expected sample number to promote at twice the minimum effect is 1.15 times a
  null-shifted fixed-horizon size for a binary metric and 1.68 times for a
  bounded metric. The sample-ratio checks rejected 1 (anytime) and 2 (exact) of
  1000 correct allocations at level 0.001 and all of 1000 with a 2 percent
  distortion at n = 100000. Net benefit power is low: at n = 2000 the band calls
  a rule useful in 91 percent of worlds for a score separation of 1.5 and in
  none at 1.0. On Hillstrom e-mail data (randomized, hash recorded, not
  committed) the A/A false promotion rate was 2 in 1000 and the real Men's
  e-mail arm was promoted in all 200 stream orders. Not done: outside-party
  reproduction, the Criteo and Open Bandit referents, and reference outputs
  from R `dcurves` and Open Bandit Pipeline.
- Closed loop and escalation (2026-11-04): a hash-chained deployment ledger
  (`scmle.governance.deployment_ledger`), anytime-valid monitors
  (`scmle.monitoring.sequential_tests`: betting alarm mixed over restarts,
  paired e-processes, e-LOND across challengers), label maturity,
  Horvitz-Thompson losses and audit and bypass slices
  (`scmle.monitoring.labels`), selective prediction with a fixed-sequence
  threshold (`scmle.governance.escalation`), the eleven never-list guards
  (`scmle.governance.never_list`) and `scmle.governance.loop.Loop` with `scmle
  loop`. `ModelMonitor.performance` takes optional `propensity` and `maturity`,
  and fresh retraining rows are held out by time or group instead of a random
  shuffle.
- Closed loop certificates (`benchmarks/qualification/loop_certify.py`), on
  simulated streams only: the false-alarm, false-promotion, false-rollback,
  selection, maturity, feedback and threshold gates passed. The alarm's
  detection delay did not (a rise of twice the tolerance is found in a median
  76 batches, the pre-registered limit was 50), so a policy does not let an
  alarm retrain by itself (`alarm_gates_retrain`). Not measured: agreement with
  a domain owner's signed audit, replay of recorded streams, Python 3.11 ledger
  replay.
- Closed loop wiring (2026-11-05). A halt writes an `incident` ledger row
  (`Loop.record_incident`). `Loop.check_feedback` runs `labels.bypass_test` on
  next-period features of the bypass slice, records it in the ledger and
  `label_mechanism.json`, and escalates on a detected loop.
  `ModelRegistryStore(guard=, scope=)` and `attach_guard` (also
  `Loop.guard_registry`) put the never-list guard in front of every champion
  move, so `promote_challenger`, `check_rollback` and
  `register_run(promote=True)` need an attestation first; without a guard
  nothing changes.
- Uncertainty wiring: fold models stored in the bundle with CV+ intervals
  (`predict_interval(method="cv_plus")`), shift weights attached to a bundle
  (`attach_shift_weights`, `shift=True`), `coverage_report.json` measured on the
  final test rows, and the charter field `unit_of_prediction` (charter format
  6; charters without it keep their hash) with entity-wise calibration for
  `new_entity`.
- Uncertainty and problem-type breadth in `scmle.evaluation`: per-class
  (Mondrian) conformal margins (`ConformalCalibrator.class_margins`,
  `sets(..., conditional="class")`, `ModelBundle.predict_set(...,
  conditional="class")`; a class with fewer than `ceil(1/(1-coverage)) - 1`
  calibration rows is kept in every set and listed by `thin_classes`); weighted
  conformal for covariate shift (`weighted_quantile`, `weighted_margins`,
  `shift_weights`, refusing below an effective size of 50); `cv_plus_bounds`;
  `coverage` (Clopper-Pearson, Holm-corrected slice flags, class coverage);
  `proper_scores` (pinball, interval score, WIS, CRPS from a quantile grid,
  MASE with the in-sample scale); `ipcw` (Cox censoring weights, Uno-style
  truncated concordance); `reconcile` (bottom-up and MinT with a shrunk
  covariance); `ranking.query_bootstrap_ndcg`; and `task_gate` with
  `task_gate_status.json` (every task type is `passed`, `failed` or
  `no_referent`; only `passed` types may keep a `confirmed` tier).
- Uncertainty gates in `benchmarks/qualification/uncertainty_certificates.json`
  (`uncertainty_certify.py`; `--check` fails when stale): split conformal
  (coverage against `Beta(451, 50)`), CV+ with 1-NN and linear learners at K =
  10 and 5, class-conditional sets (the pooled margin covers the 2 percent
  class 0 percent of the time), weighted conformal (known ratio passes,
  unweighted covers 0.73), MinT against bottom-up, covariate-dependent IPCW
  (unbiased; the marginal Kaplan-Meier version is biased by -0.017) and
  entity-wise calibration. Two gates missed their pre-registered floors and are
  published as misses with the methods descriptive: CQR in the highest noise
  decile covers 0.864 (floor 0.87; constant width 0.72), and the familywise
  false-flag rate of the earlier Holm slice flag is 0.0755 against 0.0597.
  Entity-wise calibration did not show row-wise calibration to be
  anti-conservative in this world. The CQR miss is about four standard errors,
  not noise, and more model capacity makes it larger
  (`benchmarks/qualification/cqr_analysis.py`). Not run, for lack of outside
  data or tools: real-data coverage, M4/M5, the MAPIE, scikit-survival and R
  differentials, a public ranking set, multilabel, near-duplicate grouping and
  encoder baselines. Intervals for survival, ordinal, ranking, multilabel,
  multioutput, forecast and uplift remain `None`.
- Problem framing and the decision-cost model (`scmle.data.framing`).
  `DecisionCostModel` (four costs, a threshold range, a source per entry;
  refuses a model where one action dominates) with net benefit, treat-all,
  treat-none, expected cost and re-thresholded decision curves; the identity
  `Cost = const - c_FN NB(t)` is tested to 1e-9 over 10,000 random cost
  matrices. `AnalysisCharter` gains an optional `cost_model` (charter format 2;
  format 1 files load and charters without one keep their hash).
- Framing analyses: `baseline` (the model against the current process as an
  intersection-union test at both ends of the cost range with a cluster-robust
  t test, registered by hash before `C` is read and evaluated once; a
  selective-label detector refuses the claim when outcome observation depends
  on the baseline decision), `feasibility` (verdict `inconclusive` when it
  changes across a chi-square interval) and `questions` (a tipping-point test,
  so only unknowns that change the verdict are asked).
- `scmle frame` writes `framing_draft.json`, `decision_cost_model.json`,
  `baseline_report.json`, `framing_questions.json` and `framing_refusal.json`
  using a deterministic slot-filling drafter, data-side checks and an LLM
  slot-filler interface with recorded-response replay (no client ships); `tau`
  is set by an upper Clopper-Pearson bound with its risk-coverage curve. A
  charter run writes a descriptive `framing_link.json`.
- Framing certificates (`benchmarks/qualification/framing_certificates.json`):
  empirical size of the baseline claim at most 0.0597 at 20, 50 and 200
  clusters in three null worlds (so `MIN_CLUSTERS = 20`), power 0.97 where the
  closed form says 0.95, selective labels detected in 500 of 500 planted and 0
  of 500 clean worlds (strong plants only), feasibility agreeing with the oracle
  in 98.6% of 1000 runs away from the boundary, and 1000 malformed charter
  records all rejected. Plants are synthetic and Gaussian-like, so these are
  optimistic. Not built, blocked on outside parties: the expert-written gold
  corpus (so no `tau` is on file and the LLM slot-filler is descriptive), the
  comparison with the R decision-curve packages and a domain owner's sign-off on
  a cost range.
- Raw-data ingestion and feature construction (`scmle.data.ingest`, `scmle
  ingest`, `IngestPlan`): `profile` (table catalog by aggregate queries run in
  the source, and `consistent_read`, which rejects a torn read), `grain` (a
  level-wise key search to `L = 3` by the `g3` error and a `match`, `finer`,
  `coarser` or `unknown` verdict on the charter unit), `joins` (exact inclusion
  by anti-join counts; many-to-many is reported and never joined; the search is
  charged to the ingestion budget), `entities` (Fellegi-Sunter with multi-pass
  blocking and EM; unlabelled results are `unverified`), `asof_view` and
  `synthesize` (a row is visible only if its recorded time is at or before
  `moment - lag`; ten primitives in declared windows; every kept feature is
  registered in the ledger as a `synthesized` hypothesis and carries the weakest
  tier of the tables it reads), `reconcile` and `budget`. Artifacts are
  `ingest_catalog.json`, `grain_report.json`, `join_graph.json` and
  `ingest_budget.json`.
- Ingestion measurements (`benchmarks/qualification/ingest_certificates.json`;
  `--check` fails when stale): torn reads detected in 500 of 500 mutated worlds
  and 0 of 500 static ones; planted fan-out, dropped rows and torn extracts
  detected in 500 of 500 each and 0 of 500 clean extracts flagged; 0 rows leaked
  by the view over 500 late-arrival and backfill worlds where an event-time-only
  filter leaked in all 500; equality to the independent reference
  `ingest_naive.py` (relative 1e-9) on 1000 random schemas and 0 failures in
  10,000 metamorphic rewrites; 200 of 200 synthetic unit verdicts correct; 0
  spurious auto-accepted links in 200 surrogate-key schemas; pair completeness
  and link recall 0.9986 on ten synthetic registries. Plants are the project's
  own, so the key, join and entity gates stay descriptive; real-schema key
  precision and recall and entity-resolution precision on a public benchmark are
  unmeasured.
- Breadth audit and benchmark definitions ([`docs/BREADTH.md`](docs/BREADTH.md),
  `benchmarks/breadth/`, `tests/unit/test_breadth.py`). Clustering is a library
  function with no run, packaging or monitoring wiring, and association and
  dimensionality reduction do not exist. Fourteen definition files give each
  further slice a robustness, drift, explanation and monitoring benchmark with
  every parameter unset and nothing run, and guard tests fail if an unsupported
  algorithm, family, modality or streaming path enters the production registry.
  Exactly-once streaming is not claimed: the telemetry spool is at least once.
- Business trail (`scmle.trail`, `scmle trail build|verify|show`,
  `scripts/trace_model.py`, [`docs/TRAIL.md`](docs/TRAIL.md)). One folder joins
  the signed charter, the framing and decision-cost model, the claim ledger, the
  registry and its signed approval, the release state and the deployment ledger
  with the stakeholder's KPI-to-metric mapping, acceptance criteria, the
  reviewer's thresholds and the observed cost per case. The verifier re-derives
  every fact and reports a missing link, a broken hash chain, an unsigned or
  unapproved step and an outcome outside the expectation. The three roles of the
  committed trail were played by the author with three keys, so no independent
  reviewer has traced it.
- `python -m benchmarks.qualification --check all` runs the framing, ingestion,
  causal, decision and loop certificate checks beside the default one, in CI
  and in the release preflight (neither ran them before).
- Smaller additions (2026-11-07): `curation_report.json` and `dataset_card.json`
  are rewritten after quarantine; a charter run writes `eda_report.html` from
  the stability and shift reports; an attestation can be spent once
  (`Loop.open(single_use_attestations=True)`, an `attestation_used` ledger row);
  `CITATION.cff`; the coverage claim's status is capped by the task type's gate
  (`coverage_confirm.capped`); the confirmation document says whether released
  scores look exchangeable with the calibration scores
  (`evaluation/exchangeability.py`, no status); the metric interval of a
  `known_entity_future` run resamples whole entities; the betting capital's
  martingale and Ville properties are checked by exact enumeration.
- More additions (2026-11-07): `decision/missing.py` checks anytime-validly for
  differential missingness of outcomes by arm and labels a flag
  `missingness_not_handled`; `constraints.cluster_satisfaction_test` resamples
  whole clusters (on 400 simulated clustered worlds with a true gap above the
  bound the row-wise test called the rule satisfied in 22 percent, the
  cluster-aware test in none); `evaluation/balance.py` judges "looks
  randomized" for uplift treatment by an exact permutation test instead of an
  AUC cutoff; the JWKS, OIDC discovery document and revocation list can be
  fetched with a cache and fail-closed staleness (`scmle.security.remote`); a
  hashed holdback assignment with a concurrent holdback check and a sample-ratio
  guard; the generated service feeds the operations signals from its real
  requests and scored rows (`SCMLE_SIGNALS_FILE`); Sargan and Hansen J
  overidentification tests for several instruments
  (`identification/overid.py`, descriptive).
- Continuous evaluation (`scmle.operations.continuous`). Delayed labels and
  business outcomes become losses for `Loop.mature`, for the serving model and
  each challenger. `RetrainPolicy` recommends by default or triggers, both
  through `Loop.escalate`, and a recommended retrain runs only after a person's
  signed reply. Dataset versions, training runs, proposals and replacements are
  in a hash-chained history; `replace` refuses without the loop's promotion
  proposal, the promotion gate's admission and the release registry's runtime
  confirmation; `post_promotion_check` reports `helped`, `harmed` or
  `inconclusive` with intervals per metric.
- Observability and incidents (`scmle.operations.metrics`, `logs`, `signals`,
  `classify`, `notify`, `respond`, `cost`;
  [`docs/OPERATIONS.md`](docs/OPERATIONS.md)): metrics with a Prometheus text
  exposition, JSON logs with a trace id and the model version and digest,
  windowed monitors for service health, feature quality, drift and
  delayed-label loss, HMAC-signed pager and ticket webhooks with retry and an
  idempotency key, a classifier that reports `mixed` rather than ranking causes,
  cost by service and version from declared rates, and a response desk that
  retains telemetry and writes operator actions to the audit ledger.
- Recovery (`scmle.operations.recovery`, `faults`, `spool`): a catalogue of
  thirteen faults with declared RTO and RPO objectives per stateful component,
  eleven of which run locally (process kill, an unready replacement, a clean
  rolling restart, a wrong key, lost object-store access, a metrics outage with
  a file-backed spool, a tampered bundle, a schema the service does not serve, a
  bad release rolled back through the release registry, a killed workflow run
  continued from its state, a killed ledger writer, a rotated key). A rollback
  to a managed revision and a zone or region loss are listed as not run. Not
  covered: machine crash, power loss, full disk, partitions.
- Release registry bound to the runtime (`scmle.governance.release_state`,
  `scmle.deploy.control`, `observation`). `ReleaseRegistry` keeps the states
  candidate, challenger, approved, production and retired beside
  `registry.json`. `promote` deploys and moves traffic in steps through a
  `ControlPlane` the user supplies, reads the platform back after each request
  and changes the state, the champion alias and the ledger only after it
  confirms the revision, image digest, readiness and the full share; otherwise
  the previous revision is restored. `rollback` follows the same rule,
  `reconcile` lists every disagreement between registry, ledger and an
  observation, and the deployment ledger gains a `deployment` row kind.
- Release gate (`scmle.deploy.gate`, `record`, `livetests`, `release`; `scmle
  deploy live | gate | verify-record | model-digest | reconcile`). The pure
  promotion gate (`admit`, `certify`) refuses on an unsigned or unapproved
  model, a missing or failed scan, an approver who is the requester, a missing
  or failed live check, evidence for another build, a platform that runs
  another image, and production without a matching signed staging record, and
  returns every reason. A signed deployment record binds commit, model digest,
  container digest, configuration digest and approver. The live-check responses
  are assembled from the service contract, not captured.
- Orchestration (`scmle.workflow`,
  [`docs/ORCHESTRATION.md`](docs/ORCHESTRATION.md)): one `Workflow` type for
  ingest, train, evaluate, monitor, batch-predict and approval steps with
  dependencies, per-step retries with backoff, timeouts, cost budgets,
  concurrency groups, a schedule with a declared catch-up policy, and a rule
  that an `approved` action must sit behind an approval step. `LocalExecutor`
  runs it against an `ObjectStore` with durable run ids, idempotency keys per
  step, leases, resume of only what has not succeeded, bounded parallelism and
  signed approvals from a listed approver who did not request the run;
  `Scheduler` handles missed slots.
- Portable training (`scmle.runners`,
  [`docs/PORTABLE_TRAINING.md`](docs/PORTABLE_TRAINING.md)): one `JobSpec` and a
  `Runner` template that stores the spec and a manifest under a durable run id,
  retries interrupted and retryably failed attempts under a `RetryPolicy`,
  resumes from digest-protected checkpoints, stops a run whose estimated cost
  passes its budget, enforces the timeout, refuses a changed spec for a known id
  and a job that would run as the serving identity, and keeps separate training
  and serving roles (`RoleStore`). `LocalRunner` is the adapter that ships, a
  GPU requires a declared reason, `run_project` runs an unchanged `MLProject`
  and `reproducibility.compare` reports results against tolerances declared in
  advance.
- Governed data (`scmle.data.store`, `scmle.core.retry`,
  [`docs/GOVERNED_DATA.md`](docs/GOVERNED_DATA.md)): an `ObjectStore` interface
  with a local backend and one conformance suite (`run_conformance`).
  `SnapshotReader` records for every read the object's URI, content hash, size,
  modification time, schema version and digest, freshness and lineage.
  `QualityGate` applies rules whose failure policy (`fail`, `quarantine`,
  `warn`) is declared in advance, and `IncrementalProcessor` processes new
  objects in key order with deduplication, a digest-protected checkpoint,
  retries and crash recovery: at-least-once work with idempotent output, not
  exactly-once.
- Token security (`scmle.security`,
  [`docs/TENANT_ISOLATION.md`](docs/TENANT_ISOLATION.md)). Bearer-token
  verification with the standard library only: JWT with HS256 or RS256 (public
  keys only), a required `kid`, key rotation by replacement, per-key `nbf` and
  `exp` windows, `iss`, `aud`, `exp`, `nbf` and `iat` checks with bounded
  leeway, the `none` algorithm and algorithm confusion refused, and revocation
  by token id, subject or key id (an unreadable revocation file refuses every
  token). `Gateway` applies scopes per route (`model:read`, `model:predict`),
  the deployment's tenant, a per-subject token bucket and a limit on refused
  attempts; refusals are a fixed 401, 403 or 429 body that never names the
  failed check. Activation, access and key rotation are `audit` rows of the
  deployment ledger.
- Local environment ([`docs/LOCAL.md`](docs/LOCAL.md), `compose.yaml`,
  `Makefile`, `deploy/`, `scmle.deploy.demo`, `scmle.deploy.mocks`). `make demo`
  trains and evaluates a good model and a bad release from datasets in
  `deploy/local/data/`, packages and registers them, serves them under the
  production rules, shows that a corrupted and an unsigned copy never become
  ready, monitors the bad release, rolls back with a signed approval and ships
  metrics to a mock sink. `compose.yaml` runs the lifecycle in containers on a
  read-only root, non-root, capability-free, with memory and CPU limits and
  `SCMLE_PLATFORM` selecting amd64 or arm64. `deploy/requirements.lock` is the
  hashed export of `uv.lock` (`make lock`, `make lock-check`), and CI gained the
  `local-demo` job. The container path was written and its compose file
  validated by `docker compose config`, but Docker's daemon was not available,
  so no image was built and no container started.
- Hardened service package (`scmle.inference.packaging`, `scmle.deploy`,
  `scmle.governance.sbom`). `scmle package` and
  `ModelBundle.export_service(destination, signing_key=None)` write a Dockerfile
  whose base image carries a placeholder digest that the build refuses until
  `build.sh` resolves it (`python -m scmle.deploy.base_image`), a locked install
  (`requirements.lock`, runtime closure only, with `--no-deps` and `pip check`),
  a non-root user, `SCMLE_ENV=production` as the image default, a
  `.dockerignore` for credentials, a CycloneDX SBOM that also lists the bundle
  and the base image, a signed provenance statement, a vulnerability-gate policy
  and `build.sh`. The generated `serve.py` calls `scmle.deploy.runtime`: it
  listens first, verifies the bundle signature, the checksum of every file and
  the provenance before deserializing anything, then loads and warms up.
- Vulnerability gate (`scmle.deploy.vuln_gate`, `scripts/vuln_gate.py`): fails
  closed on an empty, unreadable or uncovered scan; the policy may waive a
  component the scanner cannot see only through an `unscanned_exceptions` entry
  with a reason and an expiry. The image was not built and nothing was scanned
  against a real registry or vulnerability database; the attestations are
  HMAC-signed, not public-key or transparency-log signatures.
- Serving contract (`scmle.inference.serving`, `guard`, `parity`,
  `scmle.contracts.serving`, [`docs/SERVICE_SLO.md`](docs/SERVICE_SLO.md)):
  request and response schema version 1 with a `/v1/predict` alias. Every
  response carries `schema_version`, `request_id`, `model_version`,
  `artifact_digest` and `schema_digest`, plus one `rows` entry per record
  listing null features and unseen categories. `GET /livez` and `GET /readyz`
  are new: ready only after verify, load and warm-up, otherwise 503 with the
  phase and reason. Limits: requests in flight (429), scoring time (504), body
  size and record count (413). `on_null` and `on_unseen` accept `reject` (422
  `data_policy`); a record without a required key is a 422;
  `expected_schema_digest` and an unknown `schema_version` are a 409
  `schema_change`. The modality service gained `/livez`, `/readyz`, the same
  limits (`set_limits`) and the digest and version.
- `scmle batch` shares `respond_frame` with the online path and writes a
  manifest with the version and digest. The golden-record suite
  (`deploy/local/golden/`, hand-written model, closed-form expectations,
  tolerances declared in advance) is checked in process, online and in batch,
  and by `scripts/check_serving_parity.py` against any running service. No
  load, restart or scaling test exists.
- Deployment contract (`scmle.deploy.contract`): one typed, versioned
  `DeploymentSpec` (image by digest, bundle by manifest digest, configuration,
  secret references, resources, health, version) that a provider adapter must
  satisfy, and `conformance_problems`, which compares a platform's
  `RuntimeReport` with the spec. A secret is a reference and never a value; a
  secret-looking configuration key, key material, `SCMLE_TRUSTED=1` in
  production and an unsigned production bundle are refused. It is an extension
  surface of `tests/fixtures/public_api.json`, as is `scmle.security.*`; no
  adapter for a real platform exists.
- Qualification foundations (opt-in; runs without a charter are unchanged):
  record types `Check`, `Claim` and `Certificate` with evidence tiers
  (`scmle.contracts.qualification`), an exact-arithmetic error budget that keeps
  exploration wealth and the confirmation level as separate types
  (`scmle.data.ledger.budget`), keyed order-independent random streams
  (`scmle.core.randomness`), and a verification harness in
  `benchmarks/qualification` (defect injector, semi-synthetic simulator, frozen
  scipy-generated reference oracles, a corpus manifest, and a certificate store
  that refuses stale or missing certificates for gating detectors).
- Analysis charter and custody. `scmle charter init|check|sign` and
  `MLProject(charter=...)` / `--charter` fix the question, an ICH E9(R1)-style
  estimand, the decision map, set sizes and the error budget before profiling,
  with a SHA-256 hash and an optional HMAC signature. A charter run carves a
  confirmation set `C` out of the development rows (group- and
  duplicate-aware, time blocks with an embargo for temporal data) and seals it
  in a separate vault process, even with `enforcement="cooperative"`. Closed-form
  power gives a `feasible`, `downgrade` or `infeasible` verdict; when the set is
  too small, or the carve would leave a validation fold empty, the run
  completes as `exploratory_unconfirmed` with no `C`. New artifacts:
  `analysis_charter.json`, `error_budget.json`, `feasibility.json`,
  `split_manifest.json` (v2, full 64-hex SHA-256) and the hash-chained
  `custody_log.jsonl`. Custody is tamper-evident on one machine, not
  adversary-proof; see [`docs/SECURITY.md`](docs/SECURITY.md).
- Claim ledger and confirmation (charter runs): a hash-chained append-only
  `hypothesis_ledger.jsonl`, online FDR rules (e-LOND default, valid under
  arbitrary dependence; LORD++ and SAFFRON labelled independence-only),
  permuted-target negative controls, and one-time confirmation of exploration
  survivors by BY (default), BH, Holm or Bonferroni. New artifacts are
  `negative_control.json` and `confirmed_claims.json`; `--check fdr` runs the
  simulation gate. Thresholdout is not implemented.
- Charter format 3 and a pre-registered second stage: `AnalysisCharter.rules`
  (`scmle.data.charter.rules.RunRules`) fixes the ledger rule, the confirmation
  rule, the failure policy, `stability_filter` and the stability thresholds
  under the charter hash and signature. A run specification that contradicts a
  chartered rule is refused, and charters of format 1 and 2 keep their hash
  and signature. With `stability_filter` true, survivors the stability rule
  calls `fragile` are withdrawn from the family tested on `C` before it opens
  (`confirm.Withdrawal`, `robustness.prefilter.StabilityStage`); the ledger
  still hashes the full survivor set and the withdrawal is recorded as
  `stage_two` in the custody log and the sealed confirmation row. The filter is
  a function of the exploration set only, so the confirmation rule keeps its
  guarantee over the remaining family; it spends no alpha and can only cost
  power, and it acts only while the stability rule has a current certificate for
  the chartered thresholds (the certificate covers `DEFAULT_RULE` only). In a
  charter run the robustness stage runs between the negative control and the
  opening of `C`, and its stability checks, the two overall shift checks and the
  per-column shift p-values are wealth-free ledger diagnostics.
- `scmle claims list|show|verify RUN_DIR` (`scmle.artifacts.claims_audit`).
  `verify` checks the ledger chain and replay, that the claims file agrees with
  the sealed ledger, the multiplicity rule replayed on the p-values on `C`, the
  second-stage family and hash, the custody log order and single open of `C`,
  the charter and chartered rules, and the negative control. A missing input is
  `skipped`, never `pass`, and the exit code is 1 on any failure.
- Custody checks: canary and `gc` tests (`tests/leakage/test_custody_canary.py`,
  `test_custody_memory.py`) and an import-graph test
  (`test_exploration_imports.py`) fail the build if a sealed row or the custody
  modules become reachable from `detectors/`, `admissibility/` or `robustness/`.
  A sealed set's single release is bound to a claims hash and a ledger hash
  (`SealedSet.commit`; the vault refuses any other pair, and a set bound to
  another ledger state is refused), and the confirmation stage calls `commit`
  before it opens `C`. Every custody log entry carries an HMAC when a signing
  key is available (`SCMLE_SIGNING_KEY` or an explicit key); without a key the
  log is a checksum.
- The split manifest records a Merkle root over the canonical row content of
  `X`, `C` and the final set (`row_roots`; omitted when empty, so older
  manifests keep their hash), and `verify_rows` checks a table against it. For
  temporal splits the embargo is raised to the number of leading lags of the
  target's autocorrelation outside the white-noise band when that exceeds the
  chartered `embargo_rows`, and the manifest records the value used. Two
  fixtures have pinned manifest hashes (`test_custody_golden.py`), observed only
  on macOS arm64. The custody, charter, vault and attestation modules are in the
  mutation scope (`tests/unit/MUTANTS_custody.md`).
- Instrument checks (2026-11-04): the MMD shift test is certified on 5000 null
  worlds (259 false alarms, upper bound 0.0583 against 0.075; the dated
  amendment precedes the re-run, floors unchanged). A frozen, hash-checked
  Titanic extract (`tests/fixtures/titanic_extract.csv`) shows the shipped
  screen blocks `boat` and `body`. The data detectors were measured on wine,
  iris and digits (`f5_corpora.py`): label noise misses the floor on the two
  small tables and the sentinel detector flags clean rows on all three, so those
  two certificates are not to be read as general. Of the oracles in
  [`docs/ORACLES.md`](docs/ORACLES.md), six have a scmle producer
  (`benchmarks.qualification.oracle_table`) and six have none yet.
- Leakage attestations: a certified suspect that is a strong legitimate
  predictor is kept only by an explicit per-column attestation
  (`scmle.data.admissibility.attestations`, option `--leakage-attestations`,
  `MLProject(leakage_attestations=...)`): a named attester, evidence, a validity
  window and the charter hash. It yields tier `attested`, an `admit_flagged`
  decision and a ledger row `leakage.attestation:<column>`; expired,
  wrong-charter and unflagged attestations are ignored and reported in the
  run's notes.
- Leakage screen with error control for runs with and without a charter
  (`scmle.data.admissibility.screen`). For every candidate column of the
  training rows it computes a cross-fitted held-out skill (AUC, or R^2) with an
  exact permutation p-value, Cramer's V of missingness with Fisher or
  chi-square, and a record-purity test for a value captured only for some
  outcomes, all in one Benjamini-Yekutieli family at 0.05 in exact arithmetic.
  A column is a suspect only if it is significant and the lower confidence
  bound of its effect size exceeds the pre-registered ceiling (skill 0.95,
  association 0.9). Under the null the chance of any false alarm is at most 0.05
  and the FDR of the significant list at most 0.05; it cannot tell a strong
  legitimate predictor from a leak without provenance.
- The screen counts the cross-fitted AUC within folds, because pooling
  predictions across folds orders same-level rows against their labels (on the
  first version, a 2 percent noisy copy was found in 144 of 200 plants with the
  pooled AUC and 199 of 200 within folds). Binary classification is certified
  (`scmle/data/admissibility/screen_certificates.json`,
  `benchmarks/qualification/screen_certify.py`, `screen_measurements.json`);
  regression and multiclass targets get warnings with p-values and never block.
- Data detectors with certificates (`scmle.data.detectors`): missingness, label
  noise, sentinels, duplicates and outlier triage. The qualification CLI
  regenerates `detector_certificates.json` and `--check` verifies it.
  Certificates cover one corpus, and label-noise detection is descriptive.
- Admissibility and provenance (`scmle.data.admissibility`): field provenance,
  as-of reconstruction, admissibility rules by estimand, falsify-only DAG
  checks, leakage tests and reconciliation, with certificates in
  `benchmarks/qualification/f4_certificates.json`.
- Thread safety: the estimator and metric registries, experiment store, model
  registry, key registry, monitoring journals, shadow evaluator and bundle
  history are safe to use from several threads in one process (see
  [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), Concurrency). Separate
  processes sharing one registry or run directory are still not coordinated.
- Mutation scope and results (2026-11-05). `setup.cfg` now includes
  `scmle/decision`, the governance modules `loop`, `escalation`, `never_list`
  and `deployment_ledger`, the monitoring modules `sequential_tests` and
  `labels`, `scmle/data/identification`, `framing/draft.py` and
  `ingest/{entities,plan}.py`, with `timeout_multiplier = 2`. Scores run from 64
  percent (`identification/run.py`) to 99 percent (`sequential_tests.py`);
  results are in `tests/unit/MUTANTS_{loop,decision,identification,evaluation,
  framing_ingest}.md`. Golden-snapshot tests (`tests/unit/golden.py`, fixtures
  in `tests/fixtures/*_golden.json`) pin the decision layer, identification,
  evaluation and the loop; framing and ingestion have no new pins.
- Citations re-read (2026-11-05,
  [`docs/LITERATURE_CHECK.md`](docs/LITERATURE_CHECK.md)). Docstrings of the
  betting monitors, the selective threshold, CV+ and weighted conformal, MinT,
  net benefit, the robustness value and entity resolution now say what the
  sources support: the plug-in bet is the paper's empirical-Bernstein plug-in
  used one-sided here, Learn then Test assumes an i.i.d. calibration set, the
  CV+ worst-case coverage is `1 - 2 alpha - sqrt(2/n)`, and the paper's rule for
  `RV_{q,a} <= 1 - 1/f_q^2` is not implemented. No code changed.

### Changed

- A run holds the table once less. The ingest stage shares the project's frame
  instead of deep-copying it (pandas 3 is copy-on-write, so a write to either
  side still copies first); outputs are identical. On a 30-column table the
  peak memory slope fell from 4.0 to 3.7 KB per row (cache-free allocator: 2.2
  to 1.9). The measured memory per row, where it goes, and a test that bounds
  it are in [`docs/WHAT_SCMLE_CANNOT_DO.md`](docs/WHAT_SCMLE_CANNOT_DO.md).
- Lint and types are strict. `ruff` selects every rule (16 ignores, each with a
  reason in `pyproject.toml`) and `mypy` runs in strict mode with
  `warn_unreachable` and the extra error codes. `scmle` and `benchmarks` pass
  both; `# type: ignore` appears only with an error code and a reason (almost
  all on scikit-learn and torch base classes, which ship no types).
- The model adapters are plain classes configured by keyword arguments, in
  place of five unrelated adapter hierarchies; the 44 built-in adapters
  register in the same order and build the same estimators.
- `check_remote` rejects URLs that are not `http` or `https` (a `ValueError`
  before the golden files are read).
- A timestamp is read by `datetime.fromisoformat` as written, without first
  replacing a trailing `Z`: a full timestamp ending in `Z` still parses, but a
  date-only text such as `"2026-01-01Z"` now raises `ValueError`. This applies
  to the evidence, observability and performance contracts, the incremental
  and streaming readers, inference evidence, and monitoring alerts,
  observability and SLO.
- The serving warm-up and load failures are logged with a traceback
  (`LOGGER.exception`) instead of only the error text; the traceback can show
  file paths.
- About twelve unused internal parameters were removed (a caller that passed
  one gets a `TypeError`).
- A fit that has standard errors but no interval errors now builds its
  intervals from the standard errors, as documented, instead of raising
  `TypeError`.
- The bootstrap band on the winner's exam score (`metric_uncertainty`) draws up
  to 2,000 resamples on small evaluation sets (200 on large ones, within a
  budget of 4 million rows scored) instead of always 200. In 200 simulated
  null runs its coverage of the true value rose from 88 to 92 percent
  (`benchmarks/statistics/null_selection.py`); it is still below the 95
  percent promise in the point estimate, and the run takes about twice as long.
- The coverage ledger reports the exact `interval_coverage` and
  `p95_interval_achieved_confidence` in place of `percentile_interval`, which
  covered only 0.626 at 20 reruns.

- Breaking: `ServingAdapter.respond_locked` is gone (use `respond_frame`), the
  serving error body carries `schema_version` and `request_id`, and
  `HttpServing(bundle)` warms up in `start()` and answers 503 before that
  finishes.
- Breaking: a service that fails verification used to exit; it now stays alive
  with `/livez` 200, `/readyz` 503 and the reason, so that it never takes
  traffic.
- Breaking: `HttpServing` gained the keyword-only `gateway` and refuses to
  listen beyond loopback without one; `ModalityHttpServing.start` refuses the
  same (it gained `set_gateway`, its constructor is unchanged). `scmle serve`
  and the generated service read `SCMLE_AUTH_*`, so the generated image, which
  binds every interface, no longer starts without them. The compose services get
  the published demo identity settings and the test runner mints a demo token.
- Breaking: model bundles checksum `schema.json` and use bundle format 2. Legacy
  format 1 bundles are refused because their schema was not authenticated.
- Breaking: `interval_coverage` in `benchmarks/reliability.py` now needs
  observed coverage of at least 0.88 (nominal minus the 0.02 tolerance) instead
  of 0.85 and records the Clopper-Pearson interval.
- `ModelBundle.predict_set` gained `conditional`, `conformal.json` gains
  `class_scores` (old bundles load; class-conditional sets then raise), and
  `ForecastMetric` gained `scale` and the kinds `mase_same_rows` and
  `mase_insample`. `mase` remains the same-rows score, which divides by the
  no-change error on the evaluation rows and is not comparable with published
  MASE.
- The leakage screen replaces the profiler's fixed cutoffs and the `"leakage"`
  string matches in `core/stages.py` and `intelligence/recommendations.py`.
  `leakage_policy="block"` blocks on a typed result
  (`DatasetProfile.blocking_leaks`) and only for mechanisms with a current
  passing certificate. `dataset_profile.json` has a `leakage` object; Titanic
  `body` is now flagged by record purity (it is recorded only for passengers who
  died); `risk.leakage_hypotheses` in the fingerprint counts screen suspects.
- Charter runs use the certified leakage screen: `admissibility.qualify` no
  longer calls the earlier `leakage.scan` but `screen.screen_table` on the
  exploration set, so runs with and without a charter flag the same columns with
  one test. The screen's certificates decide per mechanism whether a suspect may
  quarantine, and every tested column is a wealth-free ledger diagnostic with its
  permutation p-value. `rules.build_table` accepts a leakage gate per field, and
  the `MLProject` snapshot was regenerated for the new keyword.
- The leakage screen runs inside the declared time budget and is loud when it
  does not run. The cap is `time_budget_seconds * SCREEN_SHARE (0.15) *
  FITS_PER_SECOND (25,000)` cross-fitted fits, deterministic (arithmetic on the
  budget, never the clock); 72,000 fits of a breast-cancer sized table took 1.4
  s, about 51,000 per second. The default 1800 s allows 6.75 million fits and
  the CI benchmark's 120 s allows 450,000. `screen_table` and `screen` take
  `time_budget_seconds`; omitted, nothing is capped beyond `MAX_PERMUTATIONS`.
- A table the screen declined (work or permutation limit) sets
  `ScreenReport.unchecked` and `warning`. The last line of `RunResult.summary`
  ends `WARNING leakage screen NOT RUN for N columns (...): these columns are
  unchecked, not clean; ...`, the same sentence is the first plan and profile
  warning, the first finding of `report.html` and a `leakage_screen` entry of
  the decision log, and the test score is labelled `PROVISIONAL: leakage screen
  incomplete`. `leakage_policy="block"` refuses an unchecked screen. The screen
  used the whole profiling time before (digits: 7 trials); the statistics did
  not change and `screen_certificates.json` was rewritten with the same seeds
  and floors.
- `leakage_policy` stays `"warn"` by default. A flagged column is a statistical
  suspicion, not a finding: from the data alone a leaked column and a strong
  legitimate predictor are indistinguishable (see
  [`docs/LEAKAGE.md`](docs/LEAKAGE.md)), and the screen flagged `worst radius`,
  `worst perimeter` and `worst area` on the breast-cancer table. The run summary
  now ends with a `WARNING N leakage suspect(s) kept in: ...` line and
  `report.html` opens with the findings; `block` and per-column leakage
  attestations remain the way to act on a suspicion.
- The mutation gate no longer runs in CI. Its scope is about 42,000 mutants,
  more than six hours on one hosted runner, so it could not finish inside the
  time limit even when split into four jobs. It is now the manual `mutation`
  workflow (Actions, Run workflow) and `uv run mutmut run` locally. The last
  partial run (32,010 of 41,746 mutants) killed 78.6 percent of the scored
  mutants, above the 70 percent floor; the floor is not enforced on every push
  or night, so run the workflow before a release.
- Text artifacts are written with LF on every platform. The oracle table's
  reported difference is floored at 1e-14 instead of 1e-16, which rows crossed
  between platforms. The stability certificates were re-issued with the same
  seeds and floors; no measured number changed.
- Registered TabPFN engines refuse `sample_weight` instead of ignoring it, and a
  skipped conformal calibration is recorded as a `conformal.skipped` event.
- The `cloud_deployment` lifecycle key of `benchmarks/breadth/` is now
  `deployment`.

### Removed

- Cloud infrastructure and everything built on it, to focus on the package: the
  Terraform for AWS and GCP and its reader, policy check and orphan finder; the
  ECS and Cloud Run runtime adapters and `scmle deploy observe`; the S3 and GCS
  object stores (the `ObjectStore` interface and the local backend remain); the
  AWS and GCP training runners (the `Runner` interface and the local runner
  remain); the Step Functions and Cloud Workflows emitters (the workflow
  definition and the local executor remain); the CloudWatch and Cloud
  Monitoring exporters; the production-claim checks
  (`scmle.operations.claim`, `scripts/check_production_claim.py`); the
  evidence-generation scripts for the cloud work and their evidence folders; and
  the cloud release workflows. Release gating, live checks, signed deployment
  records and the release registry remain, driven by a control plane the user
  supplies. No cloud-provider or hosted-compute SDK is a dependency
  ([`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md)). The cloud release and
  infrastructure documentation pages and their site pages are removed, and the
  remaining pages no longer describe a cloud.
- The planning and audit documents (the plan and product folders and the plan
  file) and the two local audit files, none of which were part of the package.
  The earlier files are in `git log`.
- The profiler constants `LEAK_ASSOCIATION` and `MISSINGNESS_LEAK`,
  `DatasetProfiler.leakage_findings` and the `MAX_WORK = 500_000` cap of the
  leakage screen, replaced by the screen above and a cap derived from the time
  budget.

### Fixed

- Windows: text artifacts were written with CRLF, so the framing draft's files
  differed in bytes while its charter hash did not
  (`tests/unit/test_replay_identity.py` failed on Windows 3.11 and 3.14); the
  trail and evidence digests hash raw bytes. A test scans `scmle/` for any
  writer without `newline="\n"`.
- A writer killed between the temp file and the rename left a
  `.<name>.<random>.tmp` file that made `LocalObjectStore.list_objects` raise,
  so a killed workflow run could not be continued. Listing now ignores such
  files.
- `setup.cfg` listed `scmle/data/charter/rules.py` twice, which mutmut rejects.
  28 tests that check certificate currency are skipped while mutmut has
  rewritten the hashed sources.
- Connector defects: `DbApiConnector` batches are checked against a running
  digest of the rows before each offset (`source_changed`) instead of silently
  overlapping or skipping, and `snapshot_token()` / `snapshot_query` give a
  before-and-after token. Read-only is enforced (a textual statement allow-list
  at construction, a query-only session on sqlite3, an optional
  `read_only_statement`) instead of stated in a docstring.
  `WarehouseConnector` accepts a `snapshot_reader` callback, and
  `IncrementalEvent` gains an optional `ingest_time` (omitted when empty, so
  stored events load unchanged).
- The `bandit` B101 asserts in `stages.py` and `identification/run.py` are
  explicit raises.
- The L1 feature selector no longer fails on multiclass targets.
- The public API snapshot (`tests/fixtures/public_api.json`) was regenerated
  after the `scmle` rename, which had left it stale; the set of exports is
  unchanged. `scripts/check_api_compatibility.py --write` regenerates it.
- Replay verification reports `different` when a trial's status changed between
  the original run and the rerun; it used to report `exact`.

### Security

- Breaking: `SCMLE_TRUSTED=1` and `trusted=True` are refused when
  `SCMLE_ENV=production`, in `ModelBundle.load` and in the service, and the
  variable is gone from the generated README and Dockerfile. It still works
  outside production for a deliberate local run.

## [0.1.0b1]

Not yet published: the package is not on PyPI and the repository has no release
tag for this version. The version string was set in commit `967921e`
(2026-10-15), the first day of the repository's history, and the contents below
have changed since; no release date is recorded until one is published. This
section merges two earlier sections that carried the same heading.

### Added

- Distribution name `scmle`; import name and command `scmle`. Environment
  variables are `SCMLE_SIGNING_KEY` and `SCMLE_TRUSTED`; the base exception is
  `ScmleError`.
- Windows, macOS and Linux are supported and tested in CI, with Python 3.11 or
  newer (tested 3.11 to 3.14). Helper processes (the test-set vault and the
  PyTorch worker) connect over an authenticated local channel instead of an
  inherited file descriptor, trial supervision uses `spawn` and `psutil` on
  Windows, and all text files are read and written as UTF-8 with LF line endings
  so signed bundles verify on every platform.
- A typed, versioned project specification with a deterministic config hash;
  CLI commands `plan`, `run`, `resume` and `predict`; `scmle.predict` and
  `scmle.load_model`; a workflow compiler (stages declare dependencies).
- Leakage-safe workflow: locked test set, fold-local fitting, split strategies
  (stratified, random, group, temporal) and leakage hypotheses including
  outcome-dependent recording. Under hard enforcement the locked test rows live
  in a separate vault process. Profiling, feature selection and leakage checks
  use the development partition only; trials record what each fold fit and
  scored on and the ledger verifies it against the split manifest.
- Duplicate rows (equal features) are kept on one side of the holdout and of
  every fold, and the manifest notes how many groups were kept together. Group-
  and time-aware inner cross-validation is used for calibration and stacking.
  Rolling-origin validation, `data.feature_delays`, a feature-availability
  contract and a temporal embargo are available.
- Models: scikit-learn, LightGBM, XGBoost, CatBoost, SVM and kNN (size-limited)
  adapters, cross-fitted stacking, greedy weighted blending searched alongside
  stacking, pruned hyperparameter search (random or Optuna), a grid optimizer,
  Hyperband brackets (`search.pruning`, `search.eta`),
  `models.exclude_models` / `models.exclude_families`, and
  `scmle.register_estimator` for any scikit-learn-compatible estimator class
  (conformance-checked, picklable for trial processes). Adapter capabilities
  drive preprocessing and planning (native categorical handling, no imputation
  for models that handle missing values, sparse input), with policies for
  high-cardinality columns, regression rare targets and short budgets.
- Data strategies: class weighting, resampling, SMOTE-family and opt-in jitter
  augmentation, probability calibration, log and interaction features,
  missing-value indicators and iterative imputation, an arcsinh target transform
  for long-tailed regression targets, automatic feature construction and L1
  selection as tested strategies, threshold optimization and serving-constraint
  selection.
- Selection rules: tie-breaker, tolerance, one-standard-error rule, hardware
  safety factor, Nadeau-Bengio corrected confidence interval, paired Bayesian
  tie handling (`uncertainty="bayesian"`), Pareto selection
  (`selection.pareto`, which reports the score / latency / size front and picks
  the fastest model within a score tolerance), nested cross-validation
  (`selection.nested_cv`, reporting the optimism of the winning score), and
  scoring of validation folds with the weight column (`selection.weighted`).
- Fit-health verdicts (healthy, overfit, underfit, unstable) per trial, a
  learning curve, and post-selection diagnostics bounded by a share of the time
  budget. Diagnostics and explainability include slices, robustness stress
  tests, adversarial validation, partial dependence, per-prediction attribution,
  a decision log, an HTML report with a reliability section, profile-driven
  recommendations and a run audit.
- PyTorch model families (MLP, residual, TabM-style ensemble, FT-Transformer,
  RealMLP-style) computed in a dedicated worker process, with warm-up and cosine
  schedule, automatic batch size, gradient clipping and a training-health report
  (`deeplearning` extra). Optimizers `adam`, `adamw`, `sgd` with momentum,
  `rmsprop` and full-batch `lbfgs`, Huber and focal losses, label smoothing and
  gradient accumulation. A Keras network (`keras_mlp`; Keras 3 on the PyTorch
  backend, in the same worker). Bagging adapters and early stopping for
  LightGBM, XGBoost and CatBoost. An optional TabPFN adapter (`tabpfn` extra;
  needs an accepted license).
- Enforced resource limits: each trial runs in a supervised process that is
  killed past its time ceiling or memory limit. Also a resource governor, policy
  checkpoints, a data-access ledger and audit events, resumable runs, SQLite
  (and MLflow) tracking with lineage and artifact checksums, parallel trials
  (process pool or Ray), and meta-learning from run history.
- Signed, checksummed bundles; unauthenticated loading is refused by default and
  the registry authenticates bundles. The saved model carries per-entity
  history, so lag / rolling features work on a single new record, and serving
  adapters keep the history current. Plugin discovery is gated by the
  conformance suite; save-and-reload parity verification and optional
  probability calibration at inference are available.
  `bundle.environment_lock()` writes the build environment as pinned
  requirements, and a dependency mismatch at load carries it in the error.
- Monitoring (drift with a sampling-noise floor, delayed labels, advisor),
  champion-challenger, a model registry, batch and HTTP serving and a lifecycle
  manager with shadow evaluation (`ShadowEvaluator`), A/B analysis with an
  independent bootstrap (`compare_arms`), rollback criteria
  (`LifecycleManager.check_rollback`, `previous` registry alias), and a
  retraining policy with one retrain per drift event (data digest), a row
  cooldown, a minimum of labeled rows and an audit record for every review,
  skip, rejection and promotion with its evidence. `RollbackPolicy.tolerance`
  defaults to 0.01 so routine retraining noise does not undo promotions.
- A full dataset fingerprint (shape, types, missingness, target, features,
  relationships, structure, quality, operational cost, risk), structure-aware
  bootstrap intervals, explicit inference column mapping and replay
  verification.
- A declarative, versioned feature-transformation graph (18 operations,
  fold-fitted statistics, datetime parts, per-entity lag / diff / rolling
  features) packaged in the bundle. Datetime columns become calendar and trend
  features and free-text columns are encoded as fold-local TF-IDF terms;
  `text_encoding="embedding"` encodes free text with a sentence-embedding model
  in the PyTorch worker (`embeddings` extra), each distinct string embedded once,
  falling back to TF-IDF with a warning.
- Arrow-compatible interchange (Arrow, Polars, Feather), Gaussian-copula
  synthetic data with a fidelity evaluator, and a neural-network family.
- Data quality: label-noise detection (`find_label_noise`), representativeness
  against a reference sample (`compare_to_reference`), and a missingness report
  with per-column tests of whether gaps relate to the target (development rows,
  Bonferroni-corrected); missing-value flags are planned and built only for the
  flagged columns. The profile stage reports label noise and, given
  `reference=`, a gap between the development rows and a reference sample.
- Per-row training weights (`data.weight_column`, `--weight-column`), used
  directly by models that accept `sample_weight` and otherwise by a weighted
  draw of training rows; test metrics are also reported weighted
  (`weighted_test_metrics`). `data.time_as_feature` offers the time column to
  the model (opt-in), `scmle.register_metric` registers user-defined metrics,
  and monotonic constraints are set with `data.monotonic_constraints`.
- Fairness: a report (`data.sensitive_columns`) and mitigation by
  `FairnessSpec(mitigation="group_thresholds")` (per-group thresholds judged on
  validation, reported on the locked rows and logged in the decision log) or
  `mitigation="reweighing"` (`reweighing_weights`; weights development rows so
  group and label are independent in training, leaves locked rows untouched, and
  is used through `weight_column`). Constrained fitting is not built.
- Out-of-distribution flags in the bundle; conformal prediction intervals and
  sets stored in the bundle; conformalized quantile intervals
  (`predict_interval(method="quantile")`, widths follow the row's noise; the
  default residual method is unchanged); Dirichlet calibration
  (`bundle.calibrate(..., method="dirichlet")`) for multiclass models. A
  multiclass reliability report (top-label calibration error), calibration error
  before and after in `bundle.calibrate`, and a quantile-interval coverage curve
  on the locked rows (`evaluation["quantile_coverage"]`).
- Sensitivity analysis (`scmle.evaluation.sensitivity`): training (seed,
  resample), evaluation (split) and performance (input noise) sensitivity, run
  on the selected model by `selection.sensitivity_repeats` and stored in
  `evaluation["sensitivity"]`. A bias-variance report
  (`selection.bias_variance_replicates`) gives a bootstrap decomposition of the
  selected model's error with a nearest-neighbour noise estimate, a verdict and
  advice; it reproduces resampling variance only, and the module and the report
  say so.
- Evaluation completeness: MRR and precision/recall@10 for ranking, a warning
  for exactly equal classes, and a note that validation and test keep the
  natural class mix.
- Unlabeled rows: `data.unlabeled` (`block`, `ignore`, `pseudo_label`);
  self-training is judged in cross-validation and used only if it wins, and
  `bundle.rank_for_labeling` picks the rows to label next.
- Survival analysis: `task="survival"` with `event_column`; Cox and
  discrete-time boosted models, concordance index, `predict_survival`,
  event-stratified splits, and censoring-weighted integrated Brier score and
  dynamic AUC. Competing risks (`CauseSpecificCox`, Aalen-Johansen
  `cumulative_incidence`) and time-varying covariates (`TimeVaryingCox`,
  counting-process rows) are included.
- Other task types: ordinal targets (`task="ordinal"` with `ordinal_order`;
  thresholded regressors give ordered labels and probabilities, scored by
  quadratic weighted kappa and mean absolute class error), learning to rank
  (`task="ranking"` with `query_column`; pointwise, LightGBM and XGBoost
  rankers, NDCG@10 and MAP@10, query-disjoint splits, `bundle.rank`),
  multi-label and multi-output targets (a list of target columns with
  `task="multilabel"` or `"multioutput"`; iterative stratification;
  `bundle.predict_frame` returns one column per target) and clustering
  (`scmle.unsupervised.clustering.find_clusters`: k-means, Gaussian mixture and
  agglomerative at several counts, scored by silhouette, Davies-Bouldin,
  Calinski-Harabasz and bootstrap stability).
- Forecasting: `task="forecast"` with `horizon`, `time_column` and optional
  `group_column` and `seasonal_period`; lag and rolling features are added
  automatically, the split embargoes the horizon, MASE against no-change is the
  default metric, and `bundle.forecast(history)` forecasts any series. The
  backtest report gives per-step errors against last-value and seasonal-naive
  baselines and per-origin scores.
- Uplift modelling: `task="uplift"` with `treatment_column`; T-, S- and
  X-learners, Qini coefficient and uplift@30%, treatment/outcome-stratified
  splits, a propensity check that flags non-randomized assignment, and the
  no-unmeasured-confounding assumption stated in the report.
- `bundle.counterfactual`: the smallest change to actionable columns that gives
  a wanted class or value range; unneeded changes are pruned and other columns
  never change.
- Files larger than memory: `memory_budget_gb` streams a Parquet or CSV file in
  batches. Locked rows are chosen by hashing row positions, development uses a
  bounded sample, and every locked row is scored by streaming the file.
  `training="stream"` keeps incremental models (`sgd_logistic`,
  `sgd_regression`) and updates the final one over the whole development
  partition; other models are rejected with a stated reason.
- Multi-node Ray: `executor="ray"` with `cluster_address`; trial results match
  a single-node run, hard limits hold on workers, and a lost worker is retried
  once and then recorded as a failed trial instead of ending the run.
- `ProblemSpec`: state the decision, population and success criteria; the
  criteria are checked on the locked rows and shown in the summary
  (`result.meets_criteria`).
- ONNX export (`bundle.export_onnx`, `scmle export-onnx`, `onnx` extra) with a
  parity check; unsupported models are refused by name. ONNX also covers
  LightGBM and XGBoost with `onnxmltools`, and refusals point to the service
  package. `scmle serve` runs the HTTP server; schema violations return 422 and
  `/health` reports the model version and checksum. `bundle.export_service` /
  `scmle package` write a service folder (bundle, pinned requirements, build
  environment, server script, Dockerfile) for any model.
- Benchmarks on nine open datasets with a regression gate; `--seeds N` repeats
  each dataset and reports the advantage over the reference with a 95%
  interval, and the verdict and the regression gate use the interval, so noise
  alone neither wins nor fails. A Hugging Face source and suites (`--suite`) are
  available.
- Robustness and shift (`scmle.data.robustness`; runs without a charter are
  unchanged). A charter run re-runs each exploration survivor over the
  multiverse the charter declared (imputation, outlier rule, inclusion; other
  declared choices are reported as not run) with bootstrap perturbation
  intervals, a survival rule (sign, size band, interval) and a permutation test
  of the whole specification curve. It compares `X` with the `reference` sample
  by a classifier two-sample test and a kernel MMD test with permutation
  p-values, per-feature KS or chi-square under BH, and subgroup coverage. New
  artifacts are `stability_report.json` and `shift_report.json`; without a
  deployment sample the shift status is `unverifiable`, never `pass`.
- Robustness certificates (`benchmarks/qualification/f6_certificates.json`,
  `f6_certify.py`): the classifier test is certified (false alarms 24 in 1000
  nulls, power 0.94 at a planted AUC of 0.60 with 2000 rows a side); the MMD
  test was first measured as descriptive because its false-alarm upper bound
  (0.0765) was just over the pre-registered 0.075 (it is certified on 5000 null
  worlds, see Unreleased); the stability rule is certified on semi-synthetic
  plants only. The forward-in-time check on the holdout (`forward.forward_check`,
  opens the set once) is implemented but a run reports it as skipped, because
  the final test set is opened only by the final evaluation. Per-period drift
  and calibrated PSI are not implemented, the per-feature p-values are adjusted
  within the report and not entered in the ledger, and the old `0.65`
  representativeness constant is unchanged.
- Admissibility and detectors in charter runs (runs without a charter are
  unchanged): leakage suspicion tests, the admissibility table and the data
  detector pipeline run on the exploration set `X` only, and write
  `admissibility_table.json` and `detector_certificates.json` (the certificates
  the run used). Each leakage check is chained into the ledger as a
  `diagnostic` row that spends no wealth. Certificates are read from
  `benchmarks/qualification/`, which the installed package does not ship; when
  the files are missing every detector is descriptive.
- Project hygiene: `SECURITY.md`, a code of conduct, issue and pull-request
  templates, Dependabot, and a release workflow that checks the tag against the
  package version and changelog and creates a GitHub release.

### Changed

- Modules carry their real names in a flat `scmle/` package, with no
  compatibility aliases. Logging replaces `print` in the library, CLI,
  benchmarks and examples.

### Fixed

- The forecasting backtest crashed under hard isolation; the seasonal baseline
  now comes from split time. The tight-budget rule rejected the boosted learners
  of the newer tasks and no longer does.

### Security

- The HTTP serving adapter refuses request bodies above 16 MiB with 413.
