# Changelog

## Unreleased

* The charter-run ingest stage queries the source database for the quantities
  a `reconcile` block declares (F9, WP-9.9; `declared.reconcile_source`,
  built on `reconcile.reconcile_extract`). For each declared view it asks the
  source for the row count, the declared column sums and the key checksum and
  compares them with the frame it extracted (counts and checksums exactly,
  sums within `delta_rec`; a source total of zero is compared as an absolute
  gap instead of being skipped). A mismatch against either the declared
  numbers or the source stops the run with a `BlockedError` naming each
  quantity and which side differs; `extract_reconciliation.json` gains
  `source_checks` and `source_passed`. A test seeds a corrupted extract (rows
  dropped, a sum shifted) while the declared totals are the source's. This is
  agreement of aggregates, so offsetting row errors pass. No charter field,
  hashed source or certificate changed.

* The `known_entity_future` unit of prediction has a protocol (F12, WP-12.7;
  `ProfileStage.check_known_entity`, `custody.pipeline.apply_custody`). A
  charter with `unit_of_prediction: known_entity_future` predicts a later row
  of an entity seen in training, so the entity is no longer tied to one side:
  the carve of `C` (contiguous time blocks with an embargo) ignores the
  group column and an entity's rows may sit in `X` and in `C`, while the
  final test rows are the latest period of the time-ordered split. The run
  refuses the charter without `data.group_column` or `data.time_column` or
  with a validation that is not temporal or rolling (an entity-disjoint split
  would test new entities), and emits `unit.known_entity_future`. Without
  the field, or with `new_entity`, nothing changes (a test pins the
  entity-disjoint carve). Not measured on a real panel; the hash of a charter
  with the field is unchanged (format 6). No certified source changed.

* The coverage monitor feeds the closed loop (F12 and F13;
  `scmle.governance.uncertainty_feed.UncertaintyFeed`). For a classification
  bundle, `observe` sends the loop each served case's conformal set size and
  out-of-distribution flag (the inputs `Loop.observe` already routes on), and
  `mature` counts the cases whose label fell outside the set as coverage
  violations, chains one `observe_summary` row (`source`, cases, violations,
  nominal, `gates_retrain: false`) in the deployment ledger and passes the
  coverage flags to `Loop.mature`, so the loop's existing coverage monitor
  (`LoopPolicy.coverage_target`) can escalate `coverage_below_nominal`. It
  gates nothing: nothing is retrained from it and `alarm_gates_retrain` is
  neither read nor changed (it stays False by default). Labels for cases the
  feed did not serve are refused. Regression bundles are not covered (their
  intervals have no size to route on). No certified source changed.

* Adversarial validation reports the odds (F12, WP-12.3;
  `AdversarialValidation.odds`, `odds` in `adversarial_validation.json`).
  From the same out-of-fold probabilities it already computes, the report
  gives the odds `p / (1 - p)` corrected for the sample sizes and clipped at
  `shift_weights.DEFAULT_CLIP` (20), as mean, 95th percentile, maximum,
  clipped share and Kish effective size, for the reference and the evaluation
  rows, and says whether weighted conformal would be `possible` or `refused`
  (effective size against `MIN_EFFECTIVE_SIZE`, 50). The discriminator is the
  report's boosted classifier scored out of fold, not the logistic one
  `attach_shift_weights` fits, so the numbers describe the shift and do not
  replace the bundle's weights; the sample is the report's capped one.
  Descriptive; the verdict and the AUC are unchanged. No certified source
  changed.

* Coverage and set size per declared slice column (F12, WP-12.2;
  `scmle.evaluation.coverage_slices`, `UncertaintySpec.slice_columns`,
  `declared_slices` in `coverage_report.json`). The charter's `uncertainty`
  block lists columns (distinct, non-blank; omitted when empty, so the format
  10 pin is unchanged); they are part of the hashed coverage claim. For each
  method and level the report gives, per level of the column (a numeric column
  in four bands cut at quantiles of the development rows, never the final
  rows), the rows, the share covered with its exact Clopper-Pearson interval
  and the mean width or set size. Slices under 30 rows are counted and not
  shown; a column the rows lack or with more than 12 levels is
  `not_assessable`. Descriptive only: the Holm under-coverage flag is not
  used, because its familywise false-flag rate missed its bound (G5, 0.0755
  against 0.0597), and the report says so in a note. No certified source
  changed.

* The leakage screen now runs inside the declared time budget and is loud when
  it does not run. `MAX_WORK = 500_000` is replaced by a cap derived from the
  budget: `time_budget_seconds * SCREEN_SHARE (0.15) * FITS_PER_SECOND
  (25,000)` cross-fitted fits, deterministic (arithmetic on the budget, never
  the clock). Measured: 72,000 fits of a breast-cancer sized table take 1.4 s
  (about 51,000 per second); the declared rate is about half. The default
  1800 s allows 6.75 million fits, the CI benchmark's 120 s allows 450,000.
  `screen_table` and `screen` take `time_budget_seconds`; the profiler passes
  the project's; omitted, nothing is capped beyond `MAX_PERMUTATIONS`, so the
  certification paths behave as before. A declined table (work or permutation
  limit) sets `ScreenReport.unchecked` and `warning`: the last line of
  `RunResult.summary` ends `WARNING leakage screen NOT RUN for N columns
  (...): these columns are unchecked, not clean; ...`, the same sentence is the
  first plan and profile warning, the first finding of `report.html` and a
  `leakage_screen` entry of the decision log, and the test score is labelled
  `PROVISIONAL: leakage screen incomplete`. `leakage_policy="block"` refuses
  an unchecked screen; `warn` proceeds loudly. The run clock already starts
  when the run context is created, so profiling time was always deducted from
  the trial budget; the screen simply used it all (digits: 7 trials). The
  statistics did not change; the screen's code hash did, so
  `screen_certificates.json` was rewritten with the same seeds and floors and
  `screen_measurements.json` is byte-identical.

* The coverage claim is registered before it is read (F12, WP-12.2;
  `coverage_run.claim_document`, `coverage_claim.json`, `claims verify`).
  Before `FinalEvaluationStage` opens the final test rows it writes
  `coverage_claim.json`: task, levels, the methods the bundle may hold,
  tolerance, the exact-lower-limit rule, the split fingerprint and the charter
  hash. The document's hash (`ledger.digest`) is announced by a
  `coverage.registered` event and, in a charter run, appended to the custody
  log (`coverage_claim_registered`); `coverage_report.json` carries it as
  `claim_sha256`. The new `claims verify` step "coverage claim: registered
  before the rows were read" fails by file name when the claim file differs
  from the registered hash or the report names another claim, when the custody
  log holds no or several registrations, and is skipped for a run without a
  custody log. The levels and the tolerance are the ones the report always
  used (0.5, 0.8, 0.9, 0.95; 0.02), so no number changed. A task with no
  stored interval or set has no claim. Limit: the claim is fixed before the
  rows are read but the bundle is built after, so the claim names methods the
  bundle may hold and the report says which it did hold. No certified source
  changed.

* The CV+ size cap is a charter field (F12, charter format 10;
  `scmle.data.charter.uncertainty.UncertaintySpec`,
  `AnalysisCharter.uncertainty`, `PackageStage.cv_plus_cap`). The optional
  `uncertainty` block holds `cv_plus_max_bytes`, the most bytes of fold models
  a regression bundle keeps for CV+; the run reads it from the verified
  charter and otherwise uses `cv_plus.MAX_BYTES`. A model set above the cap is
  not kept and `cv_plus.skipped` says so. The block is omitted when unset, so
  every charter of format 1 to 9 keeps its hash and signature (the pin of
  `make_charter()` and the format 7, 8 and 9 pins still pass, and a new test
  pins the hash and signature of a format 10 charter); a wrapper below format
  10 that carries `uncertainty` is refused. Nothing a certificate hashes
  changed.

* Conformal score source is chosen by a tested rule and recorded (F12,
  WP-12.1; `cv_plus.plan_scores`, `conformal.scores` event). From 1000
  calibration rows (`cv_plus.MIN_ROWS_FOR_SPLIT`) the bundle serves the pooled
  out-of-fold scores alone, each scored by the fold model that did not see its
  row, and keeps no fold model; below 1000 rows, with held-out folds that
  partition the rows and models within the size cap, the K fold models are
  also kept for CV+ intervals; overlapping folds or an oversize set fall back
  to the pooled scores with the reason in `cv_plus.skipped`. The behaviour is
  the one the run already had; what is new is the pure function that decides
  it (unit tests pin each branch and the cap as a parameter) and the event
  that names the source. No separate refit was added: the pooled scores
  already come from models that did not see their rows. No certified source
  changed.

* `claims verify` checks the four decision files (F11, W2). A charter run with
  a decision block now writes `decision_threshold.json` and `net_benefit.json`
  next to `policy_value.json` and `rollout_design.json`, and appends one
  `decision_files` event to the hash-chained custody log with the hash of each
  (`decision_reports.file_digests`, the same digest the ledger uses; a file
  that does not exist, for example the design without a rollout plan, is
  registered as absent). The new step "decision files: hashes match the
  custody log" fails by file name when a registered file differs
  ("differs from the registered hash"), is gone ("missing") or appeared
  although none was registered ("not registered"), or when the files were
  registered twice. It is absent from a run with no decision files and
  `skipped` for an older run that has the files but no registration. The
  hashes sit in the custody log and not in the hypothesis ledger because the
  ledger is sealed at the confirmation, before the locked rows these files
  come from are opened. Limit: the custody log is tamper-evident, not
  adversary-proof (canon P6). No certified source changed. The two tests that
  counted `claims verify` steps now expect ten (the causal attestation step of
  the previous change had made them nine and wrong).

* Propensity-logged assignment rows for a decision rollout (F11, W2;
  `RolloutHook.assign`, deployment ledger kind `rollout_assignment`). When a
  rollout runs, units are assigned by the plan's salted hash
  (`design.assign`) and each batch is chained in the deployment ledger with
  the unit ids, the arm of each and the probability that arm had (`q` for the
  challenger, `1 - q` for the champion), so an off-policy estimate can be made
  from the ledger. The ledger refuses a batch for an experiment that was not
  started, a list of different length, an arm other than 0 or 1 and a
  probability outside (0, 1). A charter run itself runs no experiment, so it
  still writes only the registration row. No certified source changed.

* Rollout decisions through the lifecycle manager and the deployment ledger
  (F11, W1; `scmle.governance.rollout_hook.RolloutHook`,
  `LifecycleManager.restore_previous`). The hook starts an experiment with the
  level `assign_experiment` takes from the plan's `DecisionBudget` (spent
  whether or not it promotes) and writes the level and the budget after it to
  the deployment ledger (`rollout_experiment`); each batch's decision is a
  chained `rollout_decision` row and a row of the hash-chained decision log. A
  `promote` decision writes a `promotion_proposal` and moves nothing; the alias
  moves only through `LifecycleManager.promote_challenger`, which still needs
  the registry's independent signed approval (and the attestation of a guarded
  registry). A promoted champion gets its `RollbackMonitor` once, at the plan's
  rollback level; a rollback decision writes a `rollback_proposal` and the
  alias moves back only through `restore_previous`, which `check_rollback`
  now also uses (its behaviour is unchanged). The deployment ledger gained the
  row kinds `rollout_experiment` and `rollout_decision`, with checks that a
  decision names a started experiment; an older ledger loads as before. Not
  done: the bootstrap rules of `review` and `check_rollback` are still the
  decision paths when the hook is not used. No certified source changed.

* The baseline claim can be about the selected model (F8, wiring;
  `--baseline-subject selected_model`, `MLProject(baseline_subject=...)`,
  `DataSpec.baseline_subject`, default `frozen_rule`). The default is
  unchanged: a logistic rule frozen on `X`. With `selected_model` the charter
  run explores on `X` and registers everything that does not need the model,
  then holds `C` sealed (`ledger.deferred`) until `SelectStage` has frozen the
  selection (`selection.frozen`). Only then are the claims registered by hash
  (the registration covers the model name, strategy, parameters, trial id,
  the split fingerprint and the seed) and `C` is opened, once, for the
  bivariate family and the claims together. The selected model acts where its
  positive-class probability reaches the cost model's implied threshold. The
  logistic rule is reported in `baseline_report.json` as `reference_rule`,
  with its action agreement with the model on `C` (descriptive). The
  selection used `X` only, so the intersection-union test keeps its level; a
  selection that read `C` would void it. `governed.run_governed` is now
  `explore_governed` followed by `confirm_explored` (same behaviour when
  called as one). No certified source changed.

* Attestors in the charter and signed attestations for causal claims (F10,
  charter format 9; `CausalSpec.attestors`, `CausalSpec.attestations`,
  `scmle.data.identification.attest`, `claims verify`). The causal block
  names who may vouch for which assumption and the file of signed statements.
  Before `C` opens the run loads the file; a statement counts for an
  assumption only if its attestor is listed for it, its scope is this
  charter's hash, the run date is within its dates and its HMAC-SHA256
  signature verifies under the signing key (no key, nothing counts).
  `causal_claims.json` records the accepted statements, the reason each other
  assumption has none, and the date checked. A claim whose assumptions are not
  all so attested still reports `unverified` ("no named attestor"), exactly as
  before; a charter without the fields keeps its hash and signature (pinned),
  and the older wrapper that carries them is refused. `tier.decide` is
  unchanged (the attestors are derived upstream of it), so the
  `false_confirm` certificate and every other certified source stand and none
  was re-issued. `claims verify` gained the step "causal attestations
  verify": it fails when a claim above `unverified` has no charter-named
  attestors, an accepted statement is by an unlisted attestor, for another
  charter, outside its dates or with a signature that does not verify, and it
  is skipped (not passed) without the signing key. The glossary entry
  "attestation (attestor)" and `docs/START_HERE.md` explain the role. Limit:
  the signature is a shared-key HMAC, so it does not identify the person.

* Subgroup hypotheses in a causal claim (F10, charter format 9;
  `CausalSpec.subgroups`, `identification.run.subgroup_report`). The causal
  block gained the optional `subgroups` field: columns with the levels to
  test, fixed before any data is read, so they are part of the registration
  hash the ledger stores before `C` opens. `identify` estimates the
  strategy's effect among the rows of `C` at each declared level and adjusts
  the whole family by Holm's step-down at the primary level (the rule the
  falsification battery already uses; `multiplicity.adjust` gives the adjusted
  p-values); a level that cannot be estimated stays in the family at p = 1,
  so a missing subgroup never shrinks the correction. The result is in
  `identification_report.json` and summarised in `causal_claims.json`; it does
  not change the tier. Omitted from the record when empty, so a format 4
  causal charter keeps its hash and signature (both pinned). `tier.decide` and
  every certified source are unchanged, so no certificate was re-issued. A
  test with 20 null subgroups and five seeds sees at least three unadjusted
  p < 0.05 and no Holm rejection; with one real effect of 2.0 among three
  levels only that level is rejected. Not measured: power at real subgroup
  sizes, or a certificate for the family.

* Extract reconciliation in a charter run (F9, charter format 8;
  `IngestSpec.reconcile`, `scmle.data.ingest.declared`). The `ingest` block
  gained the optional `reconcile` sub-block: per view a row count, column sums
  and a checksum of a key column (SHA-256 of the sorted values as text), as the
  source system's owner states them. The stage compares them with the frames
  it read, before synthesizing anything: counts and checksums exactly, sums
  within `delta_rec`. A mismatch raises a `BlockedError` that names each
  quantity; a pass writes `extract_reconciliation.json`. Omitted when unset, so
  a format 7 ingest charter keeps its hash and signature; an older wrapper
  that carries it is refused. Tests drop a row from the declared count and see
  the run stop. Not done: the stage does not call `reconcile_extract`, which
  queries the source, and ingest beyond SQLite is still open (DuckDB is not a
  dependency, so no second backend was added).

* The resolved entity becomes the split's group column (F9, charter format 8;
  `IngestSpec.entity`, `scmle.core.ingest_stage.resolve_entities`). The
  `ingest` block gained the optional `entity` sub-block (comparison fields,
  blocking passes, group column name). It is omitted when unset, so a format 7
  ingest charter keeps its hash and signature (both pinned by a test); a
  wrapper below format 8 that carries it is refused. In a run, the stage
  resolves the run table's rows, adds the group column, sets
  `data.group_column` to it (a conflicting caller value is a `ConfigError`),
  and writes `entity_resolution.json`. On a table where each customer from 250
  up repeats the email of another key, the default run splits 250 duplicate
  pairs across development and test, and the run with the block splits none
  (`tests/integration/test_ingest_charter_run.py`). The result is
  `unverified` (no clerical sample in a run) and is not measured on labelled
  benchmarks.

* Breadth audit and benchmark definitions (F15 item 15, local half;
  `docs/BREADTH.md`, `benchmarks/breadth/`, `tests/unit/test_breadth.py`).
  The audit of clustering, association and dimensionality reduction found
  clustering is a library function with no run, packaging or monitoring
  wiring, and association and dimensionality reduction do not exist. Fourteen
  definition files give each further slice (those three, forecasting, ranking,
  documents and NLP, image, audio, video, geospatial, graph, pretrained and
  multimodal, accelerated training, streaming) a robustness, drift,
  explanation and monitoring benchmark with every parameter unset and nothing
  run. Guard tests fail if an unsupported algorithm, family, modality or
  streaming path enters the production registry. No slice is qualified, and
  exactly-once streaming is not claimed: the only delivery path, the telemetry
  spool, is at least once.
* Business trail (F15 item 14, local half; `scmle.trail`, `scmle trail
  build|verify|show`, `scripts/trace_model.py`, `docs/TRAIL.md`,
  `plan/evidence/phase-18/local/`). One folder joins the signed charter, the
  framing and decision-cost model, the claim ledger, the registry and its
  signed approval, the release state and the deployment ledger of a model, with
  the stakeholder's KPI-to-metric mapping, acceptance criteria and approved
  expectation, the reviewer's domain thresholds and evidence, a deterministic
  exploratory report and experiment comparison, and the observed cost per case
  with an interval against the approved range. The verifier re-derives every
  fact from the retained records and reports a missing link, a broken hash
  chain, an unsigned or unapproved step and an outcome outside the expectation,
  each with a seeded test. The committed trail follows a real run on the
  Titanic extract. The three roles were played by the author with three keys,
  the observed outcomes replay held-out test rows, and the release state is
  `approved`, not `production`: no independent reviewer has traced it, and the
  gate stays open. `scmle.trail` is a new extension surface (additions only to
  `tests/fixtures/public_api.json`).
* `leakage_policy` stays `"warn"` by default. A flagged column is a
  statistical suspicion, not a finding: from the data alone a leaked column
  and a strong legitimate predictor are indistinguishable (see
  `docs/LEAKAGE.md`), so the screen cannot be made to block without refusing
  good columns (it flagged `worst radius`, `worst perimeter` and `worst area`
  on the breast-cancer table). Instead the run summary now ends with a
  `WARNING N leakage suspect(s) kept in: ...` line and `report.html` opens
  with the findings; `block` and per-column leakage attestations remain the
  way to act on a suspicion.

* Phase 17 local evidence (`plan/evidence/phase-17/local/`,
  `scripts/make_phase17_evidence.py`, `docs/OPERATIONS.md`, `plan/phase-17-...`,
  F14). A killed local service classified as infrastructure and a shifted
  feed as data quality, then mixed once labels worsened, with signed webhooks
  to recording transports, retained telemetry and audit rows for the operator;
  a simulated degradation that reaches the escalation path, a challenger, the
  loop's proposal, four seeded gate refusals and a replacement on a scripted
  ECS control plane; eleven recovery faults against the real local service
  and emulators with measured RTO and RPO. The phase plan and F14 list only
  what is open. None of it ran on AWS or GCP, no real pager or ticket
  endpoint was used, and the production claim has not been made.
* Production claim (F14 release gate; `scmle.operations.claim`,
  `scripts/check_production_claim.py`, `docs/RELEASING.md`). The claim is one
  fixed sentence in the releasing guide, which also says it has not happened.
  The checker refuses unless `plan/evidence/phase-17/aws/` and `gcp/` each
  hold signed acceptance, deployment, recovery, observability and
  continuous-evaluation records from staging for one commit, all passing
  within the declared RTO and RPO, and a sign-off from a reviewer who is not
  the requester, signed with a key the pipeline does not hold, that covers
  the current digest of each file. The committed provider folders are empty,
  so it refuses; tests seed each missing or spoiled record.
* Recovery (F14 item 13, local half; `scmle.operations.recovery`, `faults`,
  `spool`). A catalogue of thirteen faults with declared RTO and RPO
  objectives per stateful component (service, ledger, registry state, object
  store, secrets, telemetry spool, job state). Eleven run locally: process
  kill, an unready replacement and a clean rolling restart, a wrong key,
  lost object-store access, a metrics outage with a file-backed spool, a
  tampered bundle, a schema the service does not serve, a bad release rolled
  back through the release registry against revisions that are local
  processes, a killed workflow run continued from its state, a killed ledger
  writer, and a rotated key. The rollback to a managed revision and a zone or
  region loss are listed as not run, as is every provider. The runner found a
  real defect: a writer killed between the temp file and the rename left a
  `.<name>.<random>.tmp` file that made `LocalObjectStore.list_objects` raise,
  so a killed workflow run could not be continued. Listing now ignores such
  files. Not covered: machine crash, power loss, full disk, partitions.
* Continuous evaluation (F14 item 12, local half;
  `scmle.operations.continuous`). Delayed labels and business outcomes become
  losses for `Loop.mature`, for the serving model and each challenger.
  `RetrainPolicy` recommends by default or triggers; both go through
  `Loop.escalate` and neither sets `alarm_gates_retrain`, which stays False.
  A recommended retrain runs only after a person's signed reply. Dataset
  versions (content digest, preserved copy), training runs, proposals and
  replacements are in a hash-chained history. `replace` refuses without the
  loop's promotion proposal, the promotion gate's admission and the release
  registry's runtime confirmation. `post_promotion_check` gives intervals per
  metric and `helped`, `harmed` or `inconclusive`, and says it cannot
  attribute the change to the model.
* Observability and incidents (F14 item 11, local half;
  `scmle.operations.metrics`, `logs`, `signals`, `classify`, `exporters`,
  `notify`, `respond`, `cost`). Metrics with a Prometheus text exposition,
  JSON logs with a trace id and the model version and digest, windowed
  monitors for service health, feature quality, drift and delayed-label
  loss with configurable window, sampling and thresholds, CloudWatch
  (embedded metric format and `PutMetricData`) and Cloud Monitoring payloads
  checked against shapes written from the API references, HMAC-signed pager
  and ticket webhooks with retry and an idempotency key, a classifier that
  reports `mixed` rather than ranking causes, cost by service and version
  from declared rates, and a response desk that retains telemetry and writes
  operator actions to the audit ledger. Extension surfaces added to
  `tests/fixtures/public_api.json`.

* Phase 16 local evidence (`plan/evidence/phase-16/`,
  `scripts/make_phase16_evidence.py`, `plan/phase-16-...`, F14). The policy
  check over all Terraform and eight seeded violations, the seeded orphan
  listings, the live checks over recorded responses and twelve seeded
  failures, the gate's decision for a complete release and each seeded failed
  gate, reconcile reports after a success, a failed release, a rollback and a
  seeded disagreement on both providers, the workflow controls and the
  results of the Phase 16 tests. The phase plan and F14 now list only what is
  open: Terraform `fmt`, `validate` and `plan` (the tool was not installed),
  every apply and destroy, the workflows on GitHub, real ECS and Cloud Run
  answers, and the cloud gate. None of those ran; AWS and GCP accounts, a
  spending cap and a named approver are needed.
* Registry bound to the runtime (F14 item 10, local half;
  `scmle.governance.release_state`, `scmle.deploy.control`, `ecs`,
  `cloudrun`, `observation`, `docs/CLOUD_RELEASE.md`). `ReleaseRegistry` keeps
  the states candidate, challenger, approved, production and retired beside
  `registry.json`. `promote` deploys and moves traffic in steps through a
  `ControlPlane`, reads the platform back after each request and changes the
  state, the champion alias and the ledger only after it confirms the revision,
  its image digest, readiness and the full share; otherwise the previous
  revision is restored and nothing in the registry changes. `rollback`
  restores the previous known-good revision under the same rule. `reconcile`
  lists every disagreement between registry, ledger and an observation, and
  `deployed_version` refuses to call an alias deployed when the runtime
  disagrees. The deployment ledger gains a `deployment` row kind. ECS and
  Cloud Run adapters read the documented response shapes; ECS cannot split
  traffic. CLI: `scmle deploy observe` and `scmle deploy reconcile`. Not done:
  the adapters have read no real ECS or Cloud Run response, and no promotion
  or rollback ran on a cloud.
* CI/CD release gates (F14 item 9, local half; `.github/workflows/cloud-release.yml`,
  `cloud-deploy.yml`, `scmle.deploy.gate`, `record`, `livetests`, `release`).
  A manual workflow that does nothing unless `SCMLE_CLOUD_RELEASE` is set:
  build and scan the image, deploy to staging and then production in
  protected environments with the run's OIDC token (no stored key), push by
  digest, read the platform, run live health, schema, inference and security
  checks and write a signed deployment record binding commit, model digest,
  container digest, configuration digest and approver. The pure promotion
  gate (`admit`, `certify`) refuses on an unsigned or unapproved model, a
  missing or failed scan, an approver who is the requester, a missing or
  failed live check, evidence for another build, a platform that runs another
  image, and production without a matching signed staging record, and returns
  every reason. CLI: `scmle deploy live`, `gate`, `verify-record`,
  `model-digest`. `scripts/verify_hosted_controls.py` also checks the two new
  workflows (manual trigger, guards, environment, no long-lived key). Not
  done: the workflows never ran; the live-check responses are assembled from
  the service contract, not captured.
* Infrastructure as code (F14 item 3, local half; `deploy/terraform/`,
  `docs/INFRASTRUCTURE.md`). Shared Terraform modules for AWS (ECR, ECS
  Fargate behind an internal load balancer and a JWT-authorized API Gateway,
  S3, KMS, IAM with separate execution, serve, train and deploy roles,
  Secrets Manager, CloudWatch, a budget) and GCP (Artifact Registry, Cloud
  Run with IAM-only invocation, Cloud Storage with a customer key, service
  accounts with resource-level grants, workload identity federation, Secret
  Manager, Cloud Monitoring, a billing budget), with dev, staging and
  production configurations per provider. `scripts/check_terraform_policy.py`
  (`scmle.deploy.hcl`, `iac_policy`) parses the files and refuses public
  ingress without authentication, missing encryption, wildcard IAM, missing
  budgets, images by tag and secrets in files; `scripts/terraform_checks.sh`
  runs `fmt` and `validate` when Terraform or OpenTofu is installed;
  `scripts/check_destroy_orphans.py` lists what a destroy left from saved
  listings. A CI job runs them. Not done: Terraform was not available here, so
  `fmt`, `validate` and `plan` never ran on these hand-written files, and
  nothing was applied or destroyed.
* Phase 15 local evidence (`plan/evidence/phase-15/`,
  `scripts/make_phase15_evidence.py`, `plan/phase-15-...`, F14). The conformance
  suite on the local directory and on S3 and GCS in-memory clients, an
  interrupted job resumed to the local reference's result, duplicate
  invocation, partial failure and missed-schedule logs, a lineage trace, the
  emitted workflow definitions with their structural check, and the results of
  229 scenario tests. The phase plan and F14 now list only what is open: every
  cloud run (AWS and GCP accounts, a spending cap and a named approver are
  needed), MinIO and the GCS emulator, the container path, and the
  reproducibility-tolerance report. None of those ran.
* Orchestration (F14 item 8, local half; `scmle.workflow`,
  `docs/ORCHESTRATION.md`). One `Workflow` type for ingest, train, evaluate,
  monitor, batch-predict and approval steps with dependencies, per-step
  retries with backoff, timeouts, cost budgets, concurrency groups, a schedule
  with a declared catch-up policy, and a rule that an `approved` action must
  sit behind an approval step. `LocalExecutor` runs it against an
  `ObjectStore` with durable run ids from trigger keys, idempotency keys per
  step, leases (a second executor is refused, a dead one is taken over),
  skip-on-failure and resume of only what has not succeeded, run and step cost
  budgets, bounded parallelism and group limits, and signed approvals from a
  listed approver who did not request the run (rejection and expiry end the
  run). `Scheduler` handles missed slots; `runner_handler` runs a training
  step through a `Runner`. `emit.to_step_functions` and
  `emit.to_cloud_workflows` produce the AWS and GCP definitions as JSON data
  with structural validators; limits the languages cannot express are returned
  as `controls`. Not done: neither definition was deployed or accepted by a
  service, no scheduler job exists, no scenario ran on a cloud engine, and
  the cloud definitions do not enforce cost budgets or concurrency limits.
* Portable training (F14 item 6, local half; `scmle.runners`,
  `docs/PORTABLE_TRAINING.md`). One `JobSpec` and a `Runner` template that
  stores the spec and a manifest under a durable run id, retries interrupted
  and retryably failed attempts under a `RetryPolicy` and resumes them from
  digest-protected checkpoints, stops a run whose estimated cost passes its
  budget, enforces the timeout, refuses a changed spec for a known id and a job
  that would run as the serving identity, and keeps separate training and
  serving roles (`RoleStore`). Adapters: `LocalRunner`,
  `AwsContainerJobRunner` (ECS `RunTask`, task role override, spot
  interruption mapping) and `GcpJobRunner` (Cloud Run job executions, job
  service account and resources checked against the spec). A GPU requires a
  declared reason. `scmle.runners.entrypoint` is what the container runs;
  `run_project` runs an unchanged `MLProject` and mirrors its run directory to
  the store so a cancelled or killed job resumes. `reproducibility.compare`
  reports results against tolerances declared in advance. Not done: the AWS and
  GCP adapters ran only against scripted responses of the documented shapes,
  `RunV2Client` and the SDK factories not at all; no image, container, cloud
  job, billing measurement or cross-place result comparison exists.
* Governed data (F14 item 7, local half; `scmle.data.store`,
  `scmle.core.retry`, `docs/GOVERNED_DATA.md`). An `ObjectStore` interface with
  local, S3 and GCS backends (the SDKs are imported only by `s3_store` and
  `gcs_store`; a client you built can be passed instead; hashes kept as object
  metadata and verified on read) and one conformance suite
  (`run_conformance`, `scripts/run_object_store_conformance.py`) that the tests
  run on all three. `SnapshotReader` records, for every read, the object's
  URI, content hash, size, modification time, declared schema version,
  observed schema digest and match, freshness against the contract, and
  lineage to parent snapshots (`ReadLog.trace`); `connector_reader` plugs a
  store into the Phase 9 `ObjectStoreConnector`. `QualityGate` checks data
  before training and inference with rules whose failure policy (`fail`,
  `quarantine`, `warn`) is declared in advance; quarantine writes rows or holds
  the whole snapshot. `IncrementalProcessor` processes new objects in key order
  with deduplication, a digest-protected checkpoint, retries with reproducible
  backoff, a lock, and recovery after a crash between output and checkpoint:
  at-least-once work with idempotent output, not exactly-once. Not done: MinIO,
  the GCS emulator and real buckets were not run (the backends ran against
  in-memory clients), no warehouse was chosen or connected, and nothing was
  ingested, trained on or traced in a cloud account.
* Security (F14 item 4, local half; `scmle.security`, `docs/TENANT_ISOLATION.md`).
  Bearer-token verification with the standard library only: JWT with HS256 or
  RS256 (RS256 is the RFC 8017 PKCS#1 v1.5 check, so only public keys are
  held), a required `kid`, a key set that rotates by replacement and per-key
  `nbf`/`exp` windows, `iss`, `aud`, `exp`, `nbf`, `iat` checks with a bounded
  leeway, the `none` algorithm and algorithm confusion refused, and revocation
  by token id, subject or key id (a revocation file that cannot be read
  refuses every token). `Gateway` applies scopes per route (`model:read`,
  `model:predict`), the deployment's tenant, a per-subject token bucket and a
  limit on refused attempts; refusals are a fixed 401, 403 or 429 body that
  never names the failed check. Activation, access and key rotation are
  `audit` rows of the existing hash-chained deployment ledger (new row kind).
  Behaviour changes: `HttpServing` gained the keyword-only `gateway` and
  refuses to listen beyond loopback without one; `ModalityHttpServing` gained
  `set_gateway` (its pinned constructor is unchanged) and its `start` refuses
  the same; `scmle serve` and the generated service read `SCMLE_AUTH_*`, so
  the generated image, which binds every interface, no longer starts without
  them; the compose services get the published demo identity settings and the
  test runner mints a demo token. `scmle.security.*` is pinned as an extension
  surface in `tests/fixtures/public_api.json`. Not done: no TLS, workload
  identity or managed keys in any cloud, no OIDC discovery or JWKS fetch over
  the network (the key set is a file), and the compose path was not run.
* Local environment (F14 item 1; `compose.yaml`, `Makefile`, `deploy/`,
  `scmle.deploy.demo`, `scmle.deploy.mocks`, `docs/LOCAL.md`). `make demo` trains
  and evaluates a good model and a bad release from explicit datasets in
  `deploy/local/data/`, packages and registers them, round-trips the service
  folder through a mock object store by digest, serves it under the production
  rules, predicts online and in batch, checks the golden records, shows that a
  corrupted and an unsigned copy never become ready, monitors the bad release
  on fresh labelled data, rolls back with a signed approval, serves the
  restored model and ships metrics to a mock sink. `compose.yaml` runs the same
  lifecycle in containers (API, object store, metrics sink, test runner) on a
  read-only root, non-root, capability-free, with memory and CPU limits and
  `SCMLE_PLATFORM` selecting amd64 or arm64. `deploy/requirements.lock` is the
  hashed export of `uv.lock` (`make lock`, `make lock-check`); the image's base
  digest is a placeholder the build refuses until `make image` resolves it. CI
  gained the `local-demo` job. Not done: the container path
  (`make demo-compose`, the Dockerfiles, `build.sh`) was written and its
  compose file validated by `docker compose config`, but Docker's daemon was
  not available, so no image was built and no container started; the arm64
  build, the registry lookup and the scan against a live database have not
  run.
* Hardened service package (F14 item 2; `scmle.inference.packaging`,
  `scmle.deploy`, `scmle.governance.sbom`). `scmle package` and
  `ModelBundle.export_service(destination, signing_key=None)` now write a
  Dockerfile whose base image carries a placeholder digest that the build
  refuses until `build.sh` resolves it (`python -m scmle.deploy.base_image`),
  a locked install (`requirements.lock`, the runtime closure only, with
  `--no-deps` and `pip check`), a non-root user, `SCMLE_ENV=production` as the
  image default, a `.dockerignore` for credentials, a CycloneDX SBOM that also
  lists the bundle and the base image (`sbom.build_service`), a signed
  provenance statement, a vulnerability-gate policy and the `build.sh` that
  ties them together. The generated `serve.py` now calls
  `scmle.deploy.runtime`: it listens first, verifies the bundle signature, the
  checksum of every file and the provenance before deserializing anything,
  then loads and warms up. `SCMLE_TRUSTED=1` and `trusted=True` are refused
  when `SCMLE_ENV=production`, in `ModelBundle.load` and in the service, and
  the variable is gone from the generated README and Dockerfile (it still works
  outside production for a deliberate local run). Behaviour change: a service
  that fails verification used to exit; it now stays alive with `/livez` 200,
  `/readyz` 503 and the reason, so that it never takes traffic. New
  `scmle.deploy.vuln_gate` (and `scripts/vuln_gate.py`) fails closed on an
  empty, unreadable or uncovered scan; the policy may waive a component the
  scanner cannot see (the framework itself, until it is published) only
  through an `unscanned_exceptions` entry with a reason and an expiry. Not done: the image was not built and
  nothing was scanned or attested against a real registry or vulnerability
  database in this change; the attestations are HMAC-signed, not public-key
  or transparency-log signatures; the locked install pins versions without
  hashes (the repository's own image uses hashes).
* Serving contract (F14 item 5; `scmle.inference.serving`, `guard`, `parity`,
  `scmle.contracts.serving`, `docs/SERVICE_SLO.md`). Request and response
  schema version 1 with a `/v1/predict` alias; every response carries
  `schema_version`, `request_id`, `model_version`, `artifact_digest` (digest of
  the canonical bundle manifest) and `schema_digest`, plus one `rows` entry per
  record listing null features and unseen categories. New `GET /livez` and
  `GET /readyz`: ready only after verify, load and warm-up, otherwise 503 with
  the phase and reason (`HttpServing(loader=...)` keeps the process alive and
  never ready when verification fails). Limits: requests in flight (429),
  scoring time (504), body size and record count (413). Policies `on_null` and
  `on_unseen` accept `reject` (422 `data_policy`); a record without a required
  key is a 422; `expected_schema_digest` and an unknown `schema_version` are a
  409 `schema_change`. New `scmle batch` shares `respond_frame` with the online
  path and writes a manifest with the version and digest. The modality service
  gained `/livez`, `/readyz`, the same limits (`set_limits`; its pinned constructor is unchanged) and the digest and version.
  Behaviour changes: the error body now carries `schema_version` and
  `request_id`; `ServingAdapter.respond_locked` is gone (use `respond_frame`);
  `HttpServing(bundle)` warms up in `start()` and answers 503 before that
  finishes. The golden-record suite (`deploy/local/golden/`, hand-written model,
  closed-form expectations, tolerances declared in advance) is checked in
  process, online and in batch, and by `scripts/check_serving_parity.py` against
  any running service. Not done: no load, restart or scaling test, no cloud
  result.
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
