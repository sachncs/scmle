# Changelog

All notable changes to SCMLE are recorded in this file.

The format follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/)
and the project aims to follow [Semantic Versioning](https://semver.org/).
SCMLE is a beta (`0.1.0b1`): the public API may change between `0.x` releases
(see [`docs/API_STABILITY.md`](docs/API_STABILITY.md)). Breaking changes are
marked `Breaking:` and listed first in their group.

## [Unreleased]

### Breaking changes

* **ONNX export is verified by default.** `Bundle.export_onnx(path)` and
  `scmle export-onnx BUNDLE` score at least 299 rows held out from training
  (`check=` / `--check`) with both the graph and the pipeline, and raise
  `BlockedError` (exit 2), removing the graph, when none are given or the
  graph diverges (gate `export.onnx.divergence`). `unverified=True` /
  `--unverified` skips the check; `<path>.manifest.json` records the graph's
  digest and whether, and on how many rows, it was verified.
* **A model plugin may not shadow a built-in model.** `Catalog.load_plugins`
  refuses an entry-point adapter that takes a built-in model's name and lists
  it in `plugin_failures`; before, it silently replaced the built-in.

* **Splitter classes renamed, old names removed.** The splitters are single
  nouns in `scmle.data.splitting.splitters.splits` (`Random`, `Stratified`,
  `Events`, `Treatment`, `Group`, `Rolling`, `Chronology`, ...) and the sealed
  test set is `lockbox.Resident`; `RandomSplit`, `GroupSplit`,
  `ResidentTestSet` and the rest are gone. See `docs/RENAMES.md`.
* **Resume skips finished stages.** A stage's typed product (planning,
  training, selection) is saved, signed, beside its stage record and loaded on
  resume instead of re-running the stage; a product whose record or inputs
  changed, or that was saved without the signing key, is re-run. Stages that
  hold source rows, the locked test set or the budget reserve always re-run.
  The engine takes dependency matching and tie-breaking as parameters and
  exposes `ready(done, remaining)`, the seams the workflow path needs.
* **Stored record keys renamed.** The multi-word keys of the records scmle
  writes (charters, custody manifests, ledgers, contracts, catalogs, detector
  certificates, governance, deployment, monitoring and run records) are now
  one word or a nested object, for example `data_hash` -> `digests.data`,
  `charter_hash` -> `digests.charter`, `row_count` -> `rows` and `run_id` ->
  `identifier` (own id) or `run` (reference). Every renamed record kind has a
  higher `schema_version` and a migration, so scmle reads files written
  earlier, and anchored hashes and signatures still verify. Scripts, queries
  and notebooks of your own that read the JSON directly must use the new keys:
  see [`docs/RECORD_KEYS.md`](docs/RECORD_KEYS.md) and the machine-readable
  map [`docs/record_keys.json`](docs/record_keys.json). To rewrite a folder
  on disk at the new keys, run `scmle records upgrade FOLDER [--dry-run]`: it
  keeps a `.bak` of each file, verifies seals, signatures and chains before
  and after, and restores every file and exits 1 if a check fails (see
  [`docs/CLI.md`](docs/CLI.md)).
* **Write-only record keys renamed.** The same renames now cover the records
  scmle writes and does not read back. Assurance: detector certificates
  (duplicates, label noise, missingness), framing (feasibility, questions,
  refusal calibration, baseline report, framing draft `open_items` ->
  `pending`), ingest (budget, grain, joins, profile, reconciliation, entity
  resolution, synthesis selection), admissibility and reconciliation reports
  and the robustness reports (coverage, forward, shift, stability). Causal and
  decision: the AIPW, DiD, IV, RD and synthetic-control estimates, the
  identification plan and battery, sensitivity, subgroup gate, constraint
  satisfaction, net benefit, off-policy and rollout decisions and the
  thresholding report; `decision_report.json` is now `schema_version` 2.
  Contracts: capability declarations and the capability registry, data
  profiles, curation reports and datasheets, trial records (and so the
  `result.leaderboard()` and `leaderboard.parquet` columns: `trial_id` ->
  `identifier`, `model_name` -> `model`, `score_mean` -> `score`,
  `score_std` -> `deviation`, `fit_seconds` -> `fit`), modality, observability
  and performance contracts, release policies and explanation evidence
  (`privacy_decision` -> `clearance`, `evidence_links` -> `links`).
  `confirmed_claims.json` now carries a `schema_version` (2), with `alpha`,
  `digests.survivors`, `heads.{ledger, custody}` and `second`. scmle reads
  files of either version and anchored digests and signatures still verify;
  scripts of your own that read these files must use the new keys, listed in
  [`docs/RECORD_KEYS.md`](docs/RECORD_KEYS.md).
* **More write-only record keys renamed.** The second pass covers data,
  evaluation, model, governance and operations records. Data: modality
  curation reports and dataset cards (`dataset_hash` -> `digests.dataset`,
  `quality_findings`/`risk_findings` -> `findings.{quality, risk}`,
  `release_ready` -> `ready`), document, image, graph, geospatial and audio
  evidence, the source catalog (`source_identity` -> `identity`,
  `catalog_digest` -> `digest`), reader compatibility, data-store
  processing, quality and snapshot records, and the validation split manifest
  (`train_ids`/`test_ids` -> `training`/`test`). Evaluation and models: fit
  health, the learning curve (`last_gain`/`total_gain` ->
  `gains.{last, total}`), mitigation, resources (`model_mb` -> `size`,
  `p95_latency_ms` -> `latency`), release verdicts (`blocking_reasons` ->
  `blockers`), longitudinal, sequential and parameter-estimation
  declarations and reports, explanation quality, and the document and image
  model explanation and drift records. Governance and operations: the model
  card, registry approvals (`content_id` -> `content`, `approved_at` ->
  `approved`; the signature still covers the version-1 fields), escalation
  thresholds, deployment observations and parity, A/B comparisons, alerts,
  observability evaluations, SLOs and their assessments, retrain decisions,
  notifications, recovery, incident responses, jobs and workflows. New
  `schema_version`s: modality curation report and dataset card 2, document
  curation report 2, performance `Baseline` 3, catalog 3, `Ack` and
  `Consumption` 3, parameter-estimation, longitudinal and sequential reports
  3, the run's `parameter_estimation_declaration.json` 2, image package
  manifest 2 and the model card 2. scmle reads files of every earlier
  version; dataset, model, job, workflow and alert digests are anchored at
  version 1 and do not change. Scripts of your own that read these files must
  use the new keys, listed in [`docs/RECORD_KEYS.md`](docs/RECORD_KEYS.md).
* **Configuration keys, CLI flags and `gates.toml` variables renamed.**
  `config.json` (and `Project.config.model_dump()`) stores single-word keys:
  for example `feature_graph` -> `derivation`, `leakage_policy` ->
  `leakage`, `data.group_column` -> `data.group`, `data.time_column` ->
  `data.time`, `data.forecast_lags` -> `data.forecast.lags`,
  `validation.horizon_rows` -> `validation.rolling.horizon`,
  `resources.time_budget_seconds` -> `resources.budget`,
  `resources.memory_limit_gb` -> `resources.limits.memory`,
  `search.top_models` -> `search.leaders`, `models.exclude_models` ->
  `models.exclude.models` and `decision.false_positive_cost` ->
  `decision.costs.fp` (70 keys; all in
  [`docs/RECORD_KEYS.md`](docs/RECORD_KEYS.md)). Old `config.json` files
  and `--config` overlays in either spelling still load, and `config_hash`
  is taken on the old layout, so a configuration keeps its hash. The Python
  field names and `Project` keywords are unchanged. 29 `scmle run` flags
  take the new names: `--time-budget` -> `--budget`, `--memory-budget-gb` ->
  `--memory`, `--id-columns` -> `--identifiers`, `--group-column` ->
  `--group`, `--time-column` -> `--time`, `--time-as-feature` ->
  `--temporal`, `--weight-column` -> `--weight`, `--event-column` ->
  `--event`, `--query-column` -> `--query`, `--treatment-column` ->
  `--treatment`, `--baseline-column` -> `--baseline`, `--baseline-subject`
  -> `--comparator`, `--framing-draft` -> `--draft`, `--propensity-column`
  -> `--propensity`, `--logged-action-column` -> `--action`,
  `--ingest-plan` -> `--ingest`, `--seasonal-period` -> `--period`,
  `--ordinal-order` -> `--order`, `--sensitive-columns` -> `--sensitive`,
  `--feature-delay` -> `--delay`, `--unavailable-at-prediction` ->
  `--unavailable`, `--test-fraction` -> `--test`, `--embargo-rows` ->
  `--embargo`, `--horizon-rows` -> `--rolling-horizon`, `--window-rows` ->
  `--rolling-window`, `--top-models` -> `--leaders`, `--configs-per-model`
  -> `--configurations`, `--leakage-policy` -> `--leakage` and
  `--leakage-attestations` -> `--attestations`. The `variables` of
  `gates.toml` use the new configuration paths (`resources.budget`,
  `data.group`, `selection.size`, `validation.rolling.horizon`, ...).
  **Deprecated:** the old flag spellings are hidden aliases that warn once,
  and the gate registry still reads an old variable spelling with a warning;
  both are removed in the next release.
* **`evaluation.json` schema 2.** The file now carries `schema_version` 2
  and single-word keys: the selected model's fields are grouped as
  `selected.{model, trial, params, strategy, score, deviation}` (from
  `selected_model`, `selected_trial_id`, `selected_params`,
  `selected_strategy`, `selected_cv_score`, `selected_cv_std`), the row
  counts as `rows.{development, test}`, the release records as
  `release.{attestation, policy, evaluation, evidence}`,
  `split_fingerprint` -> `digests.split`, `test_metrics` -> `metrics`,
  `weighted_test_metrics` -> `weighted`, `fit_health` -> `health`,
  `cv_confidence` -> `confidence`, `baseline_cv_score` -> `baseline`,
  `run_id` -> `run`, `split_strategy` -> `validation`, `model_card` ->
  `card`, and `metric_greater_is_better` (a bool) -> `direction` (`higher`
  or `lower`); 44 keys in all. A run result's `evaluation` has the same
  layout. `scmle.artifacts.results.evaluation()` reads a file of either
  version at version 2.
* **Six more write-only records renamed.** The non-tabular capability
  `Scope` (`decoder_policy` -> `decoder`, `model_family` -> `family`), the
  bundle provenance `Manifest` (`model_name` -> `model`, `model_version` ->
  `version`, `scmle_version` -> `scmle`, `model_params` -> `params`,
  `decision_threshold` -> `threshold`), the claims-audit `Verification`
  (`run_dir` -> `directory`), the inference evidence `Link` (`event_id` ->
  `event`, the fingerprints and digests under `digests.{payload,
  prediction, bundle, graph}`; the node digest stays anchored at version
  1), the modality model `Package` (`model_sha256` -> `digests.model`)
  and the retry policy `Retries` (`max_attempts` -> `attempts`, the waits
  under `delays.{base, cap}`). The modality curation evidence writes
  `content_digest` as `digest`. scmle reads the old layouts.
* **Modules moved by the package layering (DES-13).** Each package now imports
  only the layers below it; the moves are listed, `old -> new`, under "Module
  moves" in [`docs/RENAMES.md`](docs/RENAMES.md). Names inside a module are
  unchanged; replace the module path to migrate. Pinned public modules moved so
  far: `scmle.core.retry` -> `scmle.runtime.retry`,
  `scmle.data.store.objectstore` -> `scmle.storage.objectstore`,
  `scmle.api.results` -> `scmle.artifacts.results`,
  `scmle.models.modality_lifecycle` -> `scmle.pipeline.modality_lifecycle`,
  `scmle.inference.modality_serving` -> `scmle.serving.modality`,
  `scmle.governance.release_state` -> `scmle.deploy.release_state`, and
  `scmle.inference.packaging` -> `scmle.deploy.packaging` (the module that
  writes a service folder, `scaffold`). A service folder generated from now on
  runs `python -m scmle.serving.runtime`; folders generated earlier pin the
  scmle version that built them. Logger names follow the module path, so a log
  filter on an old path must change.
* **Single-word names in `scmle.pipeline`, `scmle.runtime`,
  `scmle.workflow`, `scmle.operations` and `scmle.monitoring` (phase 6).**
  Classes, functions, methods and attributes of these packages take
  single-word names; the full old-to-new list is in
  [`docs/RENAMES.md`](docs/RENAMES.md) under "Name renames". Stored record
  keys and artifact file names do not change, so files written earlier still
  read. The pinned surface changes: `scmle.operations.continuous`
  `ContinuousEvaluation`, `RetrainDecision`, `RetrainPolicy`,
  `TrainingHistory` and `post_promotion_check` are `Evaluation`, `Decision`,
  `Policy`, `History` and `reassess`; `scmle.operations.notify`
  `WebhookSender` and `verify_signature` are `Sender` and `verify`;
  `scmle.operations.signals` `SignalMonitor` and `SignalThresholds` are
  `Monitor` and `Thresholds`; `scmle.operations.spool.MetricsSpool` is
  `Spool`; `scmle.pipeline.modality_lifecycle` `ExtensionCurator` and
  `LifecycleCancelledError` are `Extension` and `AbortedError`;
  `scmle.workflow.definition` `CatchUp`, `StepKind` and `standard_workflow`
  are `Backfill`, `Kind` and `scaffold`; `scmle.workflow.executor.StepContext`
  is `Context`. The fields and keywords of the workflow `Step` and
  `Workflow` (`needs`, `budget`, `timeout`, `expiry`, `concurrency`,
  `limits`, `parallel`), `Retries` (`attempts`) and the operations classes
  follow the list (the migration table is in
  [`docs/API_STABILITY.md`](docs/API_STABILITY.md)). A stage product kept by
  an earlier version for a resumed run is not loaded when its fields were
  renamed; that stage runs again. Replace the old name with the new one to
  migrate; no alias is kept.
* **Single-word names in `scmle.evaluation`, `scmle.models`,
  `scmle.inference` and `scmle.serving` (phase 6).** Classes, functions,
  methods and attributes of these packages take single-word names; the full
  old-to-new list is in [`docs/RENAMES.md`](docs/RENAMES.md) under "Name
  renames". Stored record keys do not change (they were renamed in phase
  5), so files written earlier still read. The pinned surface changes:
  `scmle.evaluation.parameter_estimation.BayesianPrior`,
  `ParameterEstimate` and `estimate_parameters` are `Prior`, `Estimate`
  and `infer`; `scmle.evaluation.longitudinal.PeriodATE` is `Period`;
  `scmle.evaluation.sequential.PeriodDiagnostics` and `RegimeEstimate` are
  `Diagnostics` and `Estimate`; `scmle.evaluation.metrics.register_metric`
  is `register`; `scmle.models.custom.register_estimator` and
  `unregister_estimator` are `register` and `unregister`;
  `scmle.serving.modality.ModalitySampleDecoder` is `Decoder`. The fields
  and keywords of the parameter-estimation, longitudinal and sequential
  `Config`, `Estimand` and result classes follow the list, and
  `ModalityHttpServing` takes `payload`, `batch` and `audited` (was
  `max_request_bytes`, `max_samples` and `require_lifecycle_report`);
  `Bundle`'s `quantile` argument is typed `Intervals`. The
  constructor keywords of the boosting `Booster`, the scikit-learn
  `Learner` and the deep-learning `Network` adapters and of the HTTP
  `Service` take the attribute names (`classifier`, `regressor`,
  `relabel`, `predictions`, `fixed`, `fitting`, `rounds`, `contract`;
  `payload`, `batch`, `timeout`, `drain`). Replace the old name with the new one
  to migrate; no alias is kept.
* **Bundle format 6.** The single-word names of phase 6 (lane `models`)
  rename classes pickled inside a bundle, among them
  `scmle.evaluation.cv_plus.CvPlusEnsemble` (`Ensemble`),
  `scmle.evaluation.quantile.QuantileIntervals` (`Intervals`),
  `scmle.evaluation.shift_weights.StoredShiftWeights` (`Weights`),
  `scmle.evaluation.dirichlet.DirichletCalibrator` (`Dirichlet`) and the
  ranking, uplift, survival, ordinal, multi-target and bagging estimators.
  `BUNDLE_FORMAT` goes from 5 to 6 and `Bundle.load` refuses a format 5
  bundle with the existing message: retrain and save it again.
  Deep-learning checkpoints pickle no renamed class (the concrete
  estimator classes keep their names) and keep their version.
* **Single-word names in `scmle.data` and `scmle.assurance` (phase 6).**
  Classes, functions, methods and attributes of the data and assurance
  packages take single-word names; the full old-to-new list is in
  [`docs/RENAMES.md`](docs/RENAMES.md) under "Name renames". Stored record
  keys do not change (they were renamed in phase 5), so files written
  earlier still read. The pinned governed-data surface changes:
  `scmle.data.store.snapshot.DataContract`, `ReadLog`, `SnapshotReader` and
  `SnapshotRecord` are `Contract`, `Log`, `Reader` and `Record`;
  `scmle.data.store.quality.QualityGate` and `QualityGateError` are `Gate`
  and `GateError`; `scmle.data.store.processing.IncrementalProcessor` is
  `Processor`, and the fields and keywords of `Processing`, `Assessment` and
  the processor follow the list. The constructor keywords of
  `DbApiConnector`, `Bucket`, `StreamingObjectStoreConnector`,
  `incremental.Stream` and the document `Reader` take the attribute names
  (`factory`, `guarded`, `preamble`, `probe`, `limit`, `codec`, `chunk`,
  `budget`, `pdf`, `reader`). Replace the old name with the new one to
  migrate; no alias is kept.
* **Bundle format 5.** The single-word names of phase 6 (lane
  `data_assurance`) rename two classes pickled inside a bundle:
  `scmle.data.transformers.SkewedLogTransformer` is now `Logarithm` and
  `scmle.data.transformers.AsinhTarget` is now `Asinh`. `BUNDLE_FORMAT` goes
  from 4 to 5 and `Bundle.load` refuses a format 4 bundle with the existing
  message: retrain and save it again. The renamed names are listed in
  `docs/RENAMES.md`.
* **Bundle format 4.** The package layering (register phase 3) moves
  modules whose classes are pickled inside a bundle, so `BUNDLE_FORMAT` goes
  from 3 to 4 and `Bundle.load` refuses a format 3 bundle with the existing
  message: retrain and save it again. A bundle whose pickled members name a
  module or class this build does not have (one written partway through the
  moves) is now refused with `ArtifactError`, its cause attached, instead of
  a raw `ModuleNotFoundError` or `AttributeError`. Deep-learning
  checkpoints and modality, image and document packages pickle no moved
  class and keep their versions.
* **`Bundle.export_service` is removed.** It only passed through to
  `scmle.inference.packaging.scaffold`, which is now the one way to write a
  service folder: replace `bundle.export_service(folder, signing_key=key)`
  with `scmle.inference.packaging.scaffold(bundle, folder, signing_key=key)`.
  `scmle package` is unchanged. This removes the import that put
  `artifacts` above `inference` (DES-13, layering task 3.3).
* **A signed bundle that cannot be verified is refused, even with
  `trusted=True`.** Loading a bundle that carries `signature.json` while no
  signing key is set (`SCMLE_SIGNING_KEY` or `signing_key=`) now raises
  `ArtifactError`: a signature that cannot be checked may be an invalid one.
  `trusted=True` (and `--trusted`) only vouches for a bundle with no
  signature at all. Anyone loading signed bundles without the key must now
  supply it (C8).
* **Single-word names for 19 more uniquely defined names.** Among them
  public classes `CorruptObjectError` -> `MismatchError`, `EvidenceOutcome`
  -> `Outcome`, `LifecycleOutcome` -> `Verdict`, `ReleaseReadinessSummary`
  -> `Scorecard` and `ReplyRefused` -> `RefusalError`; each is listed in
  [`docs/RENAMES.md`](docs/RENAMES.md). The remaining multi-word names are
  kept for a stated reason: members of a parallel family in one module
  (`RandomSplit`, `GroupSplit`, `IngestStage`, `TrainStage` ...) that one
  word cannot tell apart until the family gets its own module, record keys
  and fields, classes pickled into bundles, and names other libraries also
  define.
* **Single-word names for 97 names defined in several places.** Methods
  shared by several classes and helpers defined in several modules are
  renamed to one word each, every definition to the same word, so the
  method stays polymorphic; each rename and its modules is listed in
  [`docs/RENAMES.md`](docs/RENAMES.md). Five public classes are renamed:
  `ObjectStore` -> `Depot`, `KeySet` -> `Keystore`, `RunState` ->
  `Progress` (in `scmle.runners.base` and `scmle.workflow.executor`),
  `RetryExhaustedError` -> `RetryError`, and 15 more public classes
  changed contract because a method they expose was renamed; the API
  snapshot is regenerated. Kept on purpose: names that are also defined by
  pandas, NumPy, scikit-learn or the standard library (`to_dict`,
  `predict_proba`, `do_GET` ...), the `run_<action>` methods the command
  line finds by name, and names that are dataclass fields,
  properties stored in records, or keys of a serialized record
  (`artifact_digest`, `dataset_hash`, `source_identity` ...), whose rename
  would change a stored format and belongs with versioned records.
* **Single-word names for 1,209 cross-module names.** Functions, methods
  and classes used across modules are renamed to single words; every rename
  is listed in [`docs/RENAMES.md`](docs/RENAMES.md). Names pinned in the
  public API snapshot keep their names, but 17 of those classes changed
  contract because a method or argument type they expose was renamed; the
  snapshot is regenerated. Kept on purpose: protocol and framework hooks,
  names dispatched by string or written to records, names listed in
  `gates.toml`, classes pickled into bundles, and 46 names that installed
  packages or the standard library also define as attributes (`to_bytes`,
  `from_records`, `is_loopback` ...), where a word-level rename would also
  rewrite the other library's attribute.
* **Saved bundles from before these renames may not load.** A bundle
  pickles its estimator classes by module and name; the file-local renames
  (for example `EarlyStoppingClassifier` -> `Classifier`, `WeightedRegressor`
  -> `Regressor`) change those names. Retrain to migrate; this is a beta
  without released bundles.
* **Single-word names for 564 file-local names.** Functions, methods and
  classes used only inside their own module are renamed to single words
  (for example `EarlyStoppingClassifier` is `Classifier`, `warm_up` is
  `warm`). None is in the public API snapshot; every rename is listed in
  [`docs/RENAMES.md`](docs/RENAMES.md).
* **Single-word names (owner's naming rule).** Names are renamed to single
  words module by module; each rename is listed here. `scmle.core.retry`:
  `call_with_retries` is `call`, `Retries.delay_after` is `Retries.delay`.
  Also: `compiler.compile_workflow` is `assemble`, `strategies.get_strategy`
  is `lookup`, `serving.parse_request` is `parse`, `outcomes.optimal_threshold`
  is `optimise`, `reports.with_hash` is `stamp`, `context.included_files` is
  `include`, `synthesis.estimate_correlation` is `correlate`,
  `builder.copy_present` is `collect`, `classify.caveats_for` is `caveat`.
  Also: `comparison.compare_paired` is `pair`, `shift.most_shifted` is `rank`
  (the report key stays `most_shifted`), `policies.release_policy_from_criteria`
  is `derive`, `missing.differential_missingness` is `contrast`,
  `iterative.iterative_assignment` is `assign`, `feasibility.verdict_at` is
  `judge`, `provenance.claim_tier` is `grade`, `worker.ensure_running` is
  `start`, `forecasting.add_targets` is `augment`,
  `reweighing.reweighing_weights` is `weigh`.
  Also: `images` model `occlusion_explanation` is `occlude`,
  `mlflow_store.artifact_location` is `locate`, `find_parent` is `climb`,
  `ledger.build_ledger` is `compose`, `keyring.can_sign` is `signs`,
  `fairness.binary_summary` is `summarise`, `shift_weights.estimate_weights`
  is `weigh`, `nested.mean_cv_score` is `average`, `nested.nested_score` is
  `nest`, `registry.compare_schemas` is `diff`, `uplift.split_arms` is
  `split`, `ipc.start_helper` is `launch`, `verify.refuse_published_key` is
  `refuse`, `verify.check_listed_files` is `audit`,
  `performance.operation_regression` is `regress`.
  Also: `capabilities.bind_runtime_environment` is `bind`,
  `find_model_family` is `find`, `numeric_resource_diagnostics` is
  `quantify`, `undeclared_diagnostics` is `reject`,
  `with_declared_model_families` is `seed`; `data.normalize_quality_evidence`
  is `normalise`; `lineage.add_edge` is `connect`; `metrics.to_natural` is
  `naturalise`; `modalities.modality_from_value` is `restore`,
  `modality_value` is `serialise`; `qualification.require_fields` is
  `require`; `signing.canonical_payload` is `canonicalise`;
  `tasks.expand_forecast` is `expand`; `context.allocate_trial_ids` is
  `allocate`.
  Migration: replace the old names; the signatures are unchanged.
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

* **Stages return typed products.** `Stage.execute` returns the stage's
  frozen product (`products.Product`), which the engine applies to the
  context and keeps with the stage record, or None for a stage that writes
  the context itself. Every tabular stage now returns one: ingestion
  `Intake`, reservation `Reserve` (was `Allowance`), labelling `Resolution`,
  screening `Screen`, surveying `Portrait`, profiling `Partition`, custody
  `Custody`, claims `Claims`, planning `Blueprint`, training `Search`,
  selection `Choice`, final evaluation `Exam`, and packaging `Package`. A
  stage of your own that overrides `execute` and calls the stock one must
  return what it returns (`return super().execute(context)`), or the
  stock stage's product is lost. `ProfileStage.impound` and
  `run_claim_ledger` are gone: the profiling stage applies what
  `sequester` and `ledger` return.

### Added

- C3 reference clauses: `benchmarks/evidence/statistics.py` compares product
  statistics with independent implementations, within `reference_rel_tol`
  (1e-8; error `|product - reference| / max(|reference|, 1)`). Reported
  statistics: every metric a run reports (`metrics.CATALOG.report_names`,
  all ten tasks, through `Repertoire.evaluate`) against textbook formulas
  written here, on real tables bundled with scikit-learn (breast cancer,
  wine, digits, diabetes, linnerud) and simulated worlds with ties and an
  absent ordinal level. Gate-deciding statistics (`DECIDING`, every other
  `gates.toml` gate classified in `PREDICATES`) need two references beside
  the product: the leakage screen's statistics, permutation p-values
  (recounted on the screen's own permutation stream), Benjamini-Yekutieli
  step and lower bounds; the missingness test (chi-square and Fisher);
  Clopper-Pearson limits; the thresholdout terms; `MIN_TEST_CLASS_ROWS`;
  the ONNX tolerance and row count; the p95 latency of
  `resources.measure` that `select.serving_limits` and `select.pareto_front`
  decide on, replayed through the product on tied, one-row and two-row
  timing vectors and a recorded real run (type-7 quantile by hand and
  `scipy.stats.mstats.mquantiles`). On the current code 1,145 gate values
  agree to 1.6e-13. One reported value disagrees: `qwk`
  (`scmle/evaluation/ordinal.py`) weights misses by the position of a level
  among the levels present, not by its rank, so a level absent from the
  scored rows shrinks the distances (0.710 against 0.695). Not compared yet,
  and listed in the document's `residue`: the screen's encoding and
  duplication p-values, the paired margin of a skill suspect, the ONNX
  probability gap, metric intervals and the diagnostic reports.
  `python -m benchmarks.evidence.statistics --show [--statistic NAME]`,
  `--write`, `--check`.

- `benchmarks/evidence/calibration.py` measures C3's clause "every interval
  and test nominal by simulation-based calibration". Each of 30 scenarios
  simulates a world with a known truth. It then calls one product interval or
  test the way a user would: the bootstrap metric interval (three families),
  split-conformal intervals, pooled and class-conditional conformal sets,
  weighted conformal under covariate shift, CV+, the Nadeau-Bengio CV
  interval, the RD effect with its balance, placebo and density tests, the
  DiD ATT with its pre-trend test, the randomized difference in means, the
  Holm subgroup family, IPW and AIPW effects and p-values, linear (HC3),
  logistic and conjugate-Bayesian coefficients, the leakage skill
  permutation test (with and without groups), the missingness test, the
  leakage screen under the global null, the slice coverage flags, the
  frozen-nuisance AIPW ATE and ATT, the IV headline, 2SLS and
  Anderson-Rubin sets under a strong and a weak instrument, Sargan and
  Hansen J under valid instruments, and the synthetic-control placebo
  p-value under no effect.
- The module checks 64 quantities: coverage, size and rank uniformity, each
  against its exact nominal value. Those values include the finite-sample
  conformal rank, the class-size mixture for Mondrian sets, and the exact
  familywise rate of a Holm family. Each check gets a two-sided
  Clopper-Pearson bound at `alpha / 64`, so a registry of nominal procedures
  is flagged with chance at most `alpha`. Each scenario's sample is planned
  by an exact binomial search, so every check sees a 0.025 miscalibration
  with chance `power`.
- A check outside its bound is `liberal`, `conservative` or `nonuniform`. It
  is listed under `miscalibrated` together with the product location that
  produced the number. Run it with
  `python -m benchmarks.evidence.calibration --write | --check`.
- The verdict is `partial`, never `pass`, while `NOT_MEASURED` lists a
  product interval or test no scenario calibrates (18, each by
  `path:name`). A judged field a trial did not report counts against the
  product and is named in the scenario's messages. Bootstrap trials keep
  their scikit-learn settings inside a context, so every other scenario
  runs under the defaults, whatever the worker count.

- C3 null clause: `benchmarks/evidence/nulls.py` measures how often the
  product claims skill on tasks that have none. It runs 577 null tasks end to
  end through `Project` (60 s budget, the C4 benchmark's): nine permutations
  of the target of each of the 35 pinned C4 tables (315 real tasks) and 262
  simulated tables whose target is drawn apart from the features (three
  worlds, binary, multiclass and regression, 300 to 3,000 rows), seeded from
  the registered stream `coverage.null.validation`. A run claims skill when
  its CV or test interval lies wholly better than its `dummy` baseline (the
  cross-validation baseline it reports, and the `dummy` fitted on its own
  development rows and scored on its own test rows), or when its own fit
  verdict is `healthy`; a crash counts as a claim, a refusal does not, and an
  unknown metric direction is refused. The pooled one-sided upper bound at
  `alpha` must be at most `null_skill_max` (0.05). The count is planned from
  `acceptance.toml`: the smallest sample whose exact chance of passing
  reaches `power` when the true rate is `null_skill_max / (1 + size_margin)`
  = 0.0296, and never fewer than `null_tasks_min`.
  `python -m benchmarks.evidence.nulls --write` measures it; `--check`
  checks the stored result and refuses one measured with other code (the
  module, the C4 catalog, the acceptance rules or the product), other pinned
  data, or another bound, alpha, power or margin.

- `benchmarks/evidence/crashes.py` measures C7's kill-and-resume clause at
  every kill point it can find, without sampling. Four subjects run in child
  processes: the pipeline engine (`Project.run`/`Project.resume`), the
  workflow executor (leases and fenced records on a local object store), the
  batch scorer's progress journal and the erasure journal. A fault hook in the
  harness traces each stage record, completion marker, event, trial
  checkpoint, custody entry, lease put/delete, executor record, batch chunk,
  progress-journal write, atomic swap, erasure plan, erasure step and receipt.
  It records the moment before and after each operation. At each traced point,
  a fresh run gets SIGKILL to its whole process group and then resumes in a new
  process. Its final outputs are compared byte for byte with an uninterrupted
  reference, and the module also checks that no record durable at the kill is
  written again. Stage products are unsealed and the leaderboard read into
  canonical JSON, so they compare field by field; each stage record's digests
  are replaced by whether they hold (its own body, its product file, the
  products it needs). Executor records are checked for their fencing token: a
  record the resume wrote must carry one more than the lease the killed run
  left. Three control runs (`CONTROLS`) find what varies between
  uninterrupted runs; only the exact JSON fields that differ in some control
  are masked, and every other field is still compared. Outputs that are not
  JSON and that differ as a whole are compared for presence only. An
  essential output (the model, stage records and products, leaderboard,
  trial records, predictions, receipt) fails the subject if it is volatile or
  has a masked field that is not a timing. Masked fields and presence-only
  outputs are listed in the document. Run it with
  `python -m benchmarks.evidence.crashes --write | --check | --measure OUT`.

- `benchmarks/evidence/protocols.py` measures C7's "lease and journal
  protocols model-checked" clause. It holds executable models of three
  protocols: the workflow run lease with fencing tokens, the custody journal
  and the erasure journal. A small explicit-state model checker explores each
  model breadth first over every interleaving, with a crash allowed before any
  step. The lease is checked with distinct owner names and with the shared
  default owner name. The journals are checked under process kills and under
  power loss, where a line can be cut. The checker reports the shortest
  counterexample of each failing invariant. `Link` ties each model action to
  the function it models by text that function must contain. A gated object
  store drives the real implementation through every counterexample and 20
  sample paths per model, one store or file call per step. Each journal's
  client resumes as its real caller does: `Erasure.execute` skips every step
  the journal shows, so a step recorded twice breaks the `single` invariant;
  the orchestrator's gate review decides every gate again on resume, so
  repeated custody verdicts are by design and are outside the model.
  After each step it
  compares the real state with the model's. Run it with
  `python -m benchmarks.evidence.protocols --show | --write | --check`.
  The clause fails on these counterexamples, which the real code reproduces:
  - Fence reuse in the run lease. `Leases.relinquish` deletes the lease, so the
    next `acquire` takes fence 1 again. After a takeover at fence 2, the run's
    records carry fence 2, and every later valid holder is refused
    (`EvictionError`) for as long as the run lasts. With the default owner name
    (`Executor(owner="executor")`), one executor's `relinquish` can delete
    another's fresh lease, which has the same content and so the same store
    version. Two holders can then write the run record interleaved.
  - The custody journal's reader
    (`scmle.assurance.custody.journal.Journal.read`) cannot read a journal
    whose last line was cut by a power loss. It raises a raw
    `json.JSONDecodeError`, so the run's custody record, and with it the run,
    cannot be resumed. The erasure journal survives both crash kinds.

- `benchmarks/evidence/service.py` measures C7's three service clauses
  against the real `scmle serve` command in a child process over real HTTP
  (client side in `benchmarks/evidence/traffic.py`). Objectives: 1-record and
  100-record load profiles against the `docs/SERVICE_SLO.md` objectives, each
  until zero failures bound the failure rate below 0.001 at alpha (2995
  requests), the p95 decided by its distribution-free binomial upper bound;
  startup and resident memory checked too. Hung request: a `Stall` step hangs
  inside the model's predict for one request while load and slow-body clients
  run; every other request must succeed, then every worker is hung and
  `/livez` must fail so the platform restarts the replica, and SIGTERM must
  still end the process. SIGTERM: 29 rounds under load behind a router that
  stops routing on readiness, with `SCMLE_LINGER_SECONDS` at least the time
  the router takes to stop, plus one round with the product's default
  environment (no linger); zero resets, a router still routing at the
  signal and requests answered after it are required in every one of them.
  Three rounds of clients that ignore readiness are reported, not deciding.
  `--soak SECONDS` runs the
  soak for any duration (`python -m benchmarks.evidence.service --soak 86400
  --output soak.json`). Run it with `--write | --check`.
- `scmle serve` keeps serving out of rotation for `SCMLE_LINGER_SECONDS`
  (default 0: close at once) after SIGTERM (`/readyz` answers 503
  `draining`) before it closes the socket. Set it to the probe failure
  threshold times (period + timeout) to get the zero-reset drain, so callers
  the platform still routes there are answered instead of reset. A second
  stop signal ends the wait.

- `tests/redteam/test_redteam_attempts.py` runs every attempt in
  `benchmarks/evidence/bypass.py` end to end as its own test. An attempt
  passes only when the control it attacks refuses it, or a control it lists
  as defence in depth (`Attempt.also`). An attempt that names a documented
  override may instead go through that override. The record names what
  stopped the attack: a registered gate, the raising site of an
  unregistered control (`envelope.authenticate`, `bundle.authenticate`,
  `charter.verify`), or for the CLI the label the refusal printed
  (`cli.blocked`, `cli.artifact`). An attack that raises `Broken` because
  it never reached its control is recorded as `error`, not as blocked.
  Nothing may get through without an override, and no attempt may fail to
  set up. An attempt that got through when it was added names the product
  code at fault (`Attempt.defect`). It is a strict xfail that expects only
  a bypass, so a broken or misdirected attack fails the suite.
- `pickle_products` now reaches the control it tests. It resumes with a
  checkpoint that approves selection, and it watches `envelope.read`, so
  the record shows the swapped stage products were read and refused.
- The search grew from 47 to 88 attempts. The new ones cover custody journal
  edits, custody token reuse in process and in the vault, pickle payloads in
  bundles, deep-learning checkpoints and stage products, a stripped charter
  signature, a plugin stage, `SCMLE_TRUSTED`, budgets of NaN and infinity,
  `pandas.NA` targets, and charter reruns into another output folder or
  after deleting `.confirmations`. A pickle payload or a plugin counts as
  through when it took effect (`witnessed`), even if the call was refused
  afterwards.
- Ten attempts got through. Each is a product defect, still open:
  - `scmle/envelope.py`: `read` accepts a signed envelope with no key to
    check it when `trusted=True`, and the attacker's pickle runs
    (`pickle_checkpoint_trusted_nokey`). `Bundle.load` refuses the same
    case (`pickle_bundle_trusted_nokey` is blocked), although the module
    docstring says both follow one rule.
  - `scmle/assurance/custody/journal.py`: the custody journal's hash chain
    has no key and no anchored head. Removing its last line
    (`journal_truncated`), recomputing the chain (`journal_rechained`,
    `checkpoint_rechained`) or appending a `started` entry, which
    `Journal.current` treats as a fresh start (`journal_restarted`,
    `checkpoint_restarted`), lets a completed run score its locked test set
    again. The same edits, or a forged approval (`checkpoint_approved`), let
    a run that a checkpoint rejected resume.
  - `scmle/assurance/ledger/usage.py`: the record of spent confirmation sets
    is unsigned files in the output folder. A charter rerun into another
    folder (`charter_rerun_elsewhere`), or after `.confirmations` is deleted
    (`charter_rerun_forgotten`), confirms on the same rows a second time.
  - `scmle/pipeline/stages/defaults.py`: a plugin stage gets the raw locked
    test set from the board and can bind and open it before selection
    (`plugin_peek`). Only `FinalEvaluationStage` wraps the set in the
    journal's `Custodian`.
- The C8 row stays `not measured`, or becomes `fail` once the lead
  re-measures, because one bypass disproves "cannot be bypassed".
  `bypass.json` must be re-measured (`--write`) for the new attempts.

- `scmle.security.deserialisation` finds every call that deserialises
  (pickle, joblib, cloudpickle, dill, marshal, shelve, `pandas.read_pickle`,
  `torch.load` without `weights_only=True`, unsafe `yaml` loads, `numpy.load`
  with `allow_pickle=True`, `keras` with `safe_mode=False`) in a package's
  source. It does not import the package. Each call gets a verdict on where its
  bytes come from: `authenticated`, `forwarded` (a helper whose every caller
  passes authenticated bytes), `roundtrip` (a buffer the same function
  dumped) or `unauthenticated`. `python -m scmle.security.deserialisation
  scmle` exits 1 on an unauthenticated site. `scmle/` has 8 sites, none of
  them unauthenticated.
- `tests/unit/test_deserialisation_paths.py` attacks every site. It writes an
  unsigned tripwire pickle and one signed with the wrong key, and calls the
  public entry point. Each attack must be refused and the tripwire must never
  run. A new site with no attack fails the test. A positive control signs
  the same tripwire with the right key and requires it to run, so an attack
  that is refused before it reaches the deserialiser cannot pass. The round
  trip, which reads no input, has one attack (it must open no file) and no
  control.
- `scmle.security.leaks` scans everything a run wrote for the values of its
  input table. It looks for:
  - string cells outside the declared categorical levels (names,
    identifiers, free text);
  - category levels seen in fewer than 5 rows (k-anonymity);
  - raw-row fragments (two precise numeric cells of one row in one file),
    float cells with 12 significant digits and integer cells with 12 digits
    (account numbers, long identifiers);
  - given secrets, and environment secrets.
  It searches UTF-8, UTF-16, UTF-32, JSON- and HTML-escaped forms and
  unsalted SHA-256, SHA-1 and MD5 digests, and integers as decimal text and
  int64 and float64 bytes. It opens zlib, gzip, bz2 and xz streams, scmle
  envelopes, zip, Parquet and SQLite, and never unpickles. The report names
  values by an HMAC-SHA256 prefix under a random salt drawn per scan, so a
  label cannot be reversed by a dictionary search. `python -m scmle.security.leaks
  RUN --data train.csv --schema RUN/model/schema.json` exits 1 on any leak.
- `release.yml` now:
  - attaches SLSA level 3 provenance (`slsa-github-generator`
    `generator_generic_slsa3` v2.1.0). The generator accepts only a full
    release tag, so it is referenced by tag, with its commit recorded beside
    it and checked;
  - signs every distribution with Sigstore before the GitHub release
    (`gh-action-sigstore-python`, pinned by commit);
  - runs `pip-audit` over every locked dependency before upload, judged by
    `scmle.deploy.vuln_gate` (fail closed);
  - refuses to release unless an active ruleset protects the release tag
    from deletion and update.
- `benchmarks/evidence/supply.py` measures the three clauses:
  deserialisation (static scan plus attacks), artifacts (real runs on a
  PII-planted corpus, scanned; a crashed validation run makes the clause
  `not measured`, never a clean scan), and release (checks on the parsed
  workflow, so a comment cannot satisfy one: pip-audit, the gate, upload,
  Sigstore and the GitHub release found as steps and in order, none
  skippable by `if` or `continue-on-error`; the generator exempt from
  commit pinning only at an exact `vX.Y.Z` tag; and live checks of the
  generator tag, the tag rulesets and pip-audit). A run refused for too
  small a budget is retried once at twice the budget it names.

* **A bundle's trace is verified, not only present.** `scmle.trail.origin.trace(bundle)`
  (and `python -m scmle.trail trace BUNDLE`) checks that the four C6 trace
  fields are recorded and that each verifies against records other than the
  manifest: the data file rehashes to the data hash (raw bytes and, for a
  tabular file read whole, the parsed frame); the run's `config.json`
  rehashes to the configuration hash; the commit exists in the repository and
  its `pyproject.toml` declares the scmle version the bundle records; the
  pins agree with the manifest's own scikit-learn, pandas, scmle and library
  versions and with the run's provenance lock. The manifest is trusted only
  when every stored file matches its digest and the signature verifies.
* **Rebuild a bundle from its manifest alone.** `scmle.trail.replica.rebuild(bundle)`
  (and `python -m scmle.trail rebuild BUNDLE`) refuses a bundle that does not
  trace, then creates a fresh `uv` environment at the manifest's Python with
  exactly its pins (`--no-deps`), installs scmle from a clean clone at the
  manifest's commit, retrains the stored configuration in an isolated
  process with the manifest's thread counts, and compares every member byte
  for byte and the locked-test metrics within `replica.TOLERANCE` (1e-6
  relative). Exit codes: 0 traced/reproduced, 1 not, 2 on an error.
* **C6 evidence for the trace and clean-runner clauses.**
  `benchmarks/evidence/traceability.py` trains each task with the real
  command line, traces every bundle, applies seven tamperings (including a
  removed signature) that must each be caught by exactly the field they
  break, and rebuilds each bundle in a temporary environment. It refuses to
  build bundles from a working tree with uncommitted changes, counts an
  unsigned bundle as not intact, records a bundle that cannot be rebuilt
  instead of stopping (so `--rebuild` always writes its record before
  exiting 1), and fails the clean-runner clause when a bundle does not trace
  on that runner. The `reproducibility` workflow gains a `bundle` job and a
  `rebuild` job that rebuilds those bundles on a fresh runner; its record is
  stored with `--merge`.

- C12: the remaining unlinked guarantees of the pre-registrations (C3 to C14)
  and of the docs site pages (causal decision, evidence, extensions,
  migrations, modelling, security, workflows) each name an enforcing test, in
  `tests/docs/test_doc_protocols.py` and `tests/docs/test_doc_pages.py`.
  Two statements were corrected to the truth: the migrations checklist now
  sends readers to the capability matrix for status and resource limits and
  to the changelog's breaking changes and `docs/RENAMES.md` for renamed
  fields and newly blocking findings (the matrix records neither), and the C10
  registration records that `benchmarks/evidence/mutation.py` pools a
  stratified sample of modules rather than deciding each gated module, and
  classifies no crash.
- C7, C13, C14: three registrations now state what their modules do. C14's
  real tables are placed by the custody record into development (40), the
  deciding validation split (53) and acceptance (54), with 7 unplaced and a
  3000 s window; the module runs every task, so the acceptance split is not
  sealed. C7's compliance is decided on real tables with size-scaled budgets
  and no validation split, and its 400 seeded practice runs decide nothing.
  C13 records that no performance target has an upper-bound rule yet.

- Forecasting and panels (backlog item 3). Panel data with an entity and a
  time column (`scmle.models.forecast.Panel`) now has three forecast
  families: the seasonal-naive baseline (`Seasonal`), simple exponential
  smoothing with its weight fitted per series (`Smoothing`), and direct
  multi-step gradient boosting on lag and rolling features scaled per
  series, one model shared by every series (`Boosted`).
  `scmle.evaluation.forecasting.backtest` compares them by rolling-origin
  evaluation: origins one horizon apart back from the end, every family
  refitted on the history before each origin, MAE and in-sample-scaled MASE
  at each step ahead, and a one-sided Wilcoxon test against the baseline
  that is left unassessed below the 5 pairs that can reach p < 0.05. A
  history too short for a family to learn from is a `BlockedError`, not a
  `DataError`. `forecasting.run(config, frame)` runs it from a project
  configuration: the panel is read from `data.value_column`, the families
  take the configured lags, windows, period and seed
  (`forecasting.contenders`), and `validation.horizon_rows` is converted
  from table rows to time points per series.
- What the comparison shows: against the weekly seasonal-naive baseline,
  boosting wins (p < 0.05) on seasonal series with a trend, which the
  baseline trails by a season's rise. On a pure season with noise it is not
  shown to beat seasonal naive (p = 0.44 in the test panel).
- Two new split strategies. `grouped_temporal` (`temporal.Succession`)
  keeps each group whole and on one side of the test split and of every
  fold, orders groups by first timestamp, and keeps every training row
  earlier than the rows it is checked against. With a forecast horizon its
  embargo covers the horizon for the groups running at one time, not for
  every group. `spatial_block` (`spatial.Tiling`) keeps square blocks of
  space whole; the block side is derived from the label's semivariogram
  and reported beside its range. The manifest records
  `validation.entities` only for `temporal` and `rolling`;
  `splitter.TIMED` names the three time-ordered strategies.

- `scmle.evaluation.calibration` (a package: `verdict`, `labels`, `hazard`,
  `relevance`, `treatment`) tests the calibration of held-out predictions
  for the structured tasks and states the claim each supports
  (`STATEMENTS`): ordinal cumulative probabilities `P(level <= k)` at every
  threshold (`ordinal`), each label's probability (`multilabel`), event-free
  probabilities at horizons against the Kaplan-Meier estimate with
  Greenwood's variance within predicted-risk groups (`survival`; default
  horizons are the event-time `quartiles`), and for ranking both the order
  and the scale of the scores (`ranking`): the mean NDCG gain over a random
  order of the same rows, with a query bootstrap interval that must lie
  above zero (`gain`, `chance`), and the probability of relevance that an
  increasing Platt map learnt on other query folds gives each score
  (`crossfit`, `Platt`); a reversed scorer, whose map decreases, never
  supports the claim. Every test is a grouped Hosmer-Lemeshow chi-square
  with one degree of freedom per group (deciles, merged until each cell
  expects 5 counts, Cochran's rule); a claim holds only when every test ran
  and none rejects at 5% after a Bonferroni split, so an untestable check
  never supports a claim.
- `calibration.uplift` gives the Qini coefficient and the area under the
  uplift curve (`area`) with arm-stratified bootstrap percentile intervals
  (2,000 draws), and tests predicted effects against the treated-minus-
  control difference within groups of at least 30 rows per arm. The uplift
  claim needs the Qini interval above zero and the effect test not
  rejected.
- Measured size: `test_calibrated_predictions_are_rejected_at_most_at_the_level`
  draws 1,000 seeded calibrated simulations per task and asserts that the
  rate at which some check rejects stays within 5% plus a one-sided 99.9%
  binomial margin (2.1 points). The measured rates are 5.1% ordinal, 4.8%
  multilabel, 4.9% survival, 2.3% ranking and 6.1% uplift; uplift over
  8,000 seeds is 5.35%, slightly above the nominal 5% because its group
  test is a two-sample normal approximation with 30 rows per arm.
- A linear family for ranking: `pairwise_linear_ranker`
  (`scmle.models.ranking.estimators.Pairwise`), a logistic RankSVM on
  within-query preference pairs, capped at 50,000 sampled pairs.
- A linear family for uplift: `uplift_linear_t_learner`, a T-learner on
  logistic or ridge outcome models. Every uplift learner takes
  `kind="boosting" | "linear"` (`scmle.models.uplift.estimators.KINDS`); an
  arm whose label has one class gets a prior classifier. A linear
  S-learner's effect is homogeneous: one treatment coefficient for every
  row.

- Related tables (`scmle.data.relational`): declare tables (event time,
  recorded time, recording delay) and links (`many_to_one`, `one_to_many`)
  in code or as a JSON schema (`relational.parse`, which refuses an entry of
  the wrong type with a `ConfigError`). Read them from a SQLite file, a
  Parquet folder or a CSV set through one `Store` interface
  (`stores.locate` recognises the kind). A folder table can be one file or a
  directory of part files, either `orders/` or the Spark/pyarrow layout
  `orders.parquet/`. A table held by two entries is refused.
  `Engine.features` joins each target row to the latest version visible at
  its cutoff (moment less lag). It aggregates `one_to_many` rows with
  `count`, `sum`, `mean`, `min`, `max`, `last` and `trend` over windows
  (default 7, 30 and 90 days and all rows). A row is visible only once its
  event and recorded times are at or before the cutoff, so no feature reads
  the future. A target row with no moment is refused with an `InputError`
  rather than given a count of 0. On the five real relational datasets the
  C5 evidence caches, the engine's aggregates match both the row-by-row
  naive reference (`benchmarks/qualification/ingest_naive.py`) and the
  as-of synthesizer, for every primitive, window, rule and lag. Its
  `many_to_one` join matches the naive lookup on the four of those datasets
  whose rows carry an identifier (baseball's do not).
- The ingest stage (`--ingest-plan`) now runs the relational engine. After
  synthesis it computes every kept feature again with `Engine.features`
  from the same extracts. A feature that differs beyond 1e-8 stops the run
  with a `DataError`. The run event's summary gains an `engine` entry: the
  number of features checked, and each feature the engine cannot compute,
  with the reason. Those are `std`, `nunique` and `time_since_last`,
  depth-2 nests, and views that are `non_versioned` but have an event time.
  The synthesizer is still what produces the run's features. A run-table row
  with no moment now stops the run with an `InputError`. Before this change
  it silently got a count of 0.
- Ingest plans may name a `store` (`{"kind": "sqlite" | "parquet" | "csv",
  "path": ...}`) instead of a `database`, but not both. A folder is mirrored
  once into a SQLite file for the run, and its row counts are checked. A
  view may declare `cardinality`: `one_to_many` sets the `child` role and
  `many_to_one` the `parent` role, and a contradicting role is refused. A
  `many_to_one` table whose rows repeat a (key, event time, recorded time)
  value stops the run with a `SchemaError` before any feature is
  synthesized.

- `scmle.data.events` turns an append-only event log of (entity, time,
  type, payload) records into point-in-time features. Events come from
  whole files of any tabular format (`File`), JSON lines on a text stream
  such as standard input (`Lines`), or a JSON-lines file another process
  appends to, read as it grows and resumable from a byte offset (`Tail`); a
  strict source names the bad record, a lenient one skips it and keeps the
  reason. A `Store` takes events one at a time in any arrival order and
  answers `snapshot(entity, cutoff)` (each `kind.field`'s newest value and
  every window's value) and `frame(queries)` from events with
  `time <= cutoff` only. Windows are written `KIND[.FIELD]:STATISTIC:SPAN`
  (`purchase.amount:total:7d`, `*:count:1h`) over `(cutoff - span,
  cutoff]`, with `count`, `total`, `mean`, `min`, `max`, `last`,
  `distinct` and `recency`; a new `Statistic` subclass is found by its name
  with no registry. Events out of order within the declared lateness bound
  are placed where they belong; older ones are dropped, rejected
  (`LatenessError`) or accepted and counted. Under `drop` and `reject` a
  snapshot whose cutoff lies before the watermark is marked `final`; under
  `accept` no snapshot is final, since an event of any age may revise it. `lateness.bound` estimates
  the bound as a distribution-free 95% upper confidence bound on the 0.99
  quantile of observed delays (binomial order statistic; needs 299
  events). Tests check the incremental store against a from-scratch
  recompute over random disordered streams under every policy, and that
  final snapshots never change under every policy.
- `python -m scmle.data.events features|follow|lateness`: features at query
  cutoffs from files or stdin (`-`); `follow` answers each query from a
  growing log the moment the watermark passes its cutoff (under
  `--policy accept`, only once the log goes idle, marked not final);
  `--lateness auto`
  uses the estimated bound. Exit 2 on an input error, with a JSON summary of
  arrivals, bound, watermark and skipped records on standard error.
- Every data kind on the shared stages, experimental except `tabular`. Each
  kind is registered in `scmle.pipeline.kinds.KINDS` and declares its own
  capability records (`Kind.declare()`); `roster.enrol` enrols them by
  iterating `KINDS`. Every kind runs ingestion (`kind.read`), profiling
  (`kind.survey`), the split (`kind.constrain`), time-budget admission
  (`kind.cost`), training (`kind.featurise`) and packaging (`kind.decode`) as
  a table does: candidate search over the shared model families,
  cross-validation, the locked test set, explanation and the signed bundle.
  One shared splitter (`rows.Cohorts`) keeps copies, near-copies, declared
  groups and spatial blocks on one side of every split and fold; with a
  time-ordered protocol it purges development rows that share a cohort with
  a test row, an exchangeable protocol becomes the group split with a note,
  a declared group with missing values is refused, and when no cohort holds
  two rows the declared protocol applies. A served model of a non-tabular
  kind takes described rows (descriptor or feature rows), not raw files.
  - `document`: class folders or a listing table of text, Markdown or PDF
    (PDF needs `pypdf`), the text column declared as text (so a corpus
    under 30 documents keeps it), TF-IDF fitted inside the folds, chunks and
    near-duplicates kept together. MM-13 and MM-17 are fixed on this path
    only, not in the older document lifecycle.
  - `graph`: a node table and an edge file, node ids read as text; a node's
    structural features count only edges known when it is first seen, and a
    timed graph holds out the latest-seen nodes. With `document` this closes
    MM-1, MM-7, MM-8, MM-10, MM-11, MM-16 (documents) and MM-22.
  - `image` (Pillow, `multimodal` extra): one row of numeric descriptors per
    file. Closes MM-10, MM-12, MM-13 (occlusion keeps the original class) and
    MM-17 (standard signed bundle; a manifest naming an outside file is
    refused).
  - `video` (`ffmpeg` and `ffprobe` on the PATH): evenly spaced frames
    decoded through an FFmpeg pipe into descriptor rows. Closes MM-3 (a
    time-ordered split finishes with folds), MM-6, MM-10 and MM-12.
  - `audio` (FLAC needs `soundfile`) and `raster` (GeoTIFF needs
    `rasterio`): clips read in bounded chunks and tiles in bounded strips;
    raster keeps spatial blocks anchored at the CRS origin together. Close
    MM-4, MM-6, MM-10, MM-11, MM-12, MM-13, MM-16 and MM-18.
  - `geospatial` (vector): GeoJSON and line-by-line
    GeoJSON (other formats need `geopandas`) read as one row per feature
    with geometry measures, split by spatial blocks sized from the label's
    spatial range. Closes MM-2, MM-9 and MM-11. Besides the shared reason
    below, a served model takes the geometry columns, not raw GeoJSON, and
    the held-out test interval resamples rows, not spatial blocks, so it can
    be too narrow.

  Every non-tabular kind is declared `experimental`, not `supported`, for
  binary, multiclass and regression: supported means the whole lifecycle is
  proven, and deployment, monitoring and recovery are not proven for these
  kinds (`rows.LIFECYCLE` in each record's limitations). A run needs the
  opt-in `models.experimental_models` in the `--config` file; without it the
  run is refused with an `UnsupportedError`. No breadth slice is qualified
  (`docs/BREADTH.md`). A9 split-leakage property tests cover each kind.
- `scmle run DATA --target Y --modality KIND` picks the data kind (`tabular`,
  `document`, `graph`, `image`, `video`, `audio`, `raster`, `geospatial`).
  Any kind but `tabular` needs `--config` with
  `{"models": {"experimental_models": [...]}}` naming model families the
  task accepts.
- The `documents` extra (`pypdf`), for PDF input to the `document` kind.
- Citation guard `tests/docs/test_doc_citations.py`: fails when a test's
  citation of a doc line is stale or cites a line with no statement.
- C2 duplicate test: `records.exact`, an exact conditional test on the counts
  of singles, pairs and triples. It has exact size where the old p-value was
  too generous when triples are rare (in simulation 1.9% rejected at 0.01
  and 12% at 0.05), and is weaker on some tables (OpenML 1464: 0.011 to
  0.018; 1067: 0.018 to 0.024).

- ONNX divergence gate (`export.onnx.divergence`, claim C1):
  `onnx_export.verify` blocks a graph whose probabilities (scaled values for
  regression) differ from the pipeline's by more than gamma_1024 for float32
  (about 6.1e-5, was a fixed 1e-4), or whose label differs on a row whose
  native margin exceeds twice that, on fewer than `ROWS` = 299 rows (a 1%
  divergence rate ruled out at 95%). `scmle export-onnx --tolerance` is the
  override. Infinite inputs reach the graph as missing, as in the pipeline,
  and an unknown `FunctionTransformer` is refused instead of exported as the
  identity. C1 faults inject it on simulated data and on OpenML 31 credit-g.
- `scmle.pipeline.kinds`: the `Kind` protocol (`read`, `survey`,
  `constrain`, `featurise`, `decode`, `cost`), the one seam between the shared
  stages and a kind of data; `Tabular` is its first implementation. Stage
  products are held on the run context by type (`Context.get`). One golden
  per modality (`tests/fixtures/modality/`) pins each modality's run.
- C12 snippet check (`python -m benchmarks.evidence.snippets`): every code
  block of the README, `docs/` and the site runs as a reader would follow its
  page, in a fresh folder and process, against the working tree. Usage
  patterns marked `synopsis` are listed and not run. CI checks the stored
  result and the scorecard reports it as a C12 clause.

- `Dataset.known` (`Project(known=...)`): per column, `prediction` or
  `outcome`, when the column becomes known relative to the prediction. The
  leakage screen blocks a feature declared `outcome`, also when the
  permutation scan is declined for its cost (`records.declared`). Empty by
  default and then left out of the stored configuration, so existing
  configuration hashes do not change.
- The `thresholdout` failure policy is available: a charter with
  `failure="thresholdout"` asks each survivor's agreement again on the single
  opening of `C` through a journalled Thresholdout reusable holdout (Dwork et
  al. 2015, `scmle.assurance.ledger.confirm.thresholdout`), within its derived
  tolerance and budget. The journal at `<run_dir>/thresholdout_journal.json`
  lets an interrupted run resume with its budget spent. `Procedure.failure`
  accepts `thresholdout`. The policy without a reuse session is refused with
  a `ConfigError` before exploration starts.
- Near-duplicate detection by MinHash (`scmle.assurance.detectors.minhash`)
  is part of the f5-duplicates detector, whose certificate is stale until it
  is measured again.

- Two more runners on this machine: `scmle.runners.local.Process` runs each
  attempt as `python -m scmle.runners.entrypoint` in a child process, and
  `local.Container` runs the same entrypoint in an image through the Docker
  CLI (bind-mounted store, read-only root, every capability dropped, the
  job's resources as limits). Both read the attempt's end from the
  entrypoint's exit status (`scmle.runners.base.Exit`: 0, 1, 75 transient,
  143 interrupted, 137 killed), stop a job with SIGTERM and then SIGKILL
  after a grace period, and keep its output at `runs/<id>/logs/<n>.txt`.
  `python -m scmle.runners.local --runner process|container` runs a spec
  from the command line, and `Job.read` rebuilds a stored spec. See
  [`docs/PORTABLE_TRAINING.md`](docs/PORTABLE_TRAINING.md).
- A training image: `Dockerfile` at the repository root installs the wheel
  on `deploy/requirements.lock` with `--require-hashes`, on a base pinned by
  digest, as uid 10001. `python -m scmle.deploy.packaging image --wheel
  dist/<wheel>` builds it and smoke-tests `scmle --help` as non-root. The
  release workflow's `image` job builds it, runs a tiny job through the
  container runner and the runner tests against it, and `publish` now waits
  for it.
- Base images are pinned: `scmle.deploy.base_image.PINNED` records the
  index digest of each Python slim base scmle builds on, so a generated
  service Dockerfile defaults to a real digest instead of the placeholder,
  which stays refused for a base with no pin.
- `scmle store adopt --root R --tenant T` moves every object a store holds
  outside the tenant namespaces into `tenants/<T>/` (`Tenancy.adopt`), the
  migration of a store written before namespacing; re-running it finishes
  an interrupted move.

- Clustering has a user path: `scmle cluster DATA --output DIR` and
  `scmle.api.clustering.cluster`. It searches k-means, Gaussian mixture and
  agglomerative clustering over `--counts` (default 2 to 8) with
  `--rounds` bootstrap refits for the stability score, labels every row
  and saves a bundle of task `clustering`, so `scmle predict` and
  `Bundle.predict` return cluster labels. The numeric columns, less
  `--exclude` and `--truth`, are the features. `clustering.json` beside the
  bundle holds every candidate, the chosen one, its silhouette,
  Davies-Bouldin, Calinski-Harabasz and stability, and, with `--truth`, the
  adjusted Rand index and normalized mutual information against the known
  groups. `clustering.evaluate` scores a saved bundle on other rows. The
  save is staged and renamed, so an interrupted save leaves no partial
  bundle, and an existing non-empty folder is refused. Clustering is still
  not qualified (see [`docs/BREADTH.md`](docs/BREADTH.md)).
- The pinned public API (`tests/fixtures/public_api.json`) now includes
  `scmle.api.clustering` (`Segmentation`, `cluster`, `evaluate`) and
  `scmle.api.erasure` (`Forgetting`, `Retirement`, `Deletion`, `Subject`,
  `Receipt`, `RetiredError`); `Project` (the `entities` and `cutoff`
  keywords) and the `scmle.operations.signals` `Window` and `SignalMonitor`
  signatures were re-pinned.
- Erasure with signed receipts (`scmle.api.erasure`): `scmle forget
  dataset ROOTS --data FILE | --digest D` forgets every run, model, sealed
  trail and prediction log derived from a dataset; `scmle forget subject
  ROOTS --data FILE --column COL --value V | --values-file F` also purges a
  data subject's rows from the source and, with `--retrain`, trains each
  affected run again without them; `scmle retire TARGET` retires one bundle
  or registered version (`--model`, `--version`); `scmle delete TARGETS`
  deletes runs or project folders; `scmle forget verify RECEIPT` checks a
  receipt's signature. Each erasure is planned once, journaled step by step
  under `--receipts` (`<id>/plan.json`, `<id>/journal.jsonl`) and closed by
  an HMAC-signed receipt `<id>.json`; running the same request again
  resumes it, and `--dry-run` prints the plan and changes nothing. The
  custody records of a run (provenance, attestation, evidence graph,
  dataset manifest, events, charter, ledger, claims, custody log) are kept
  and their digests listed. A subject's identifiers are kept only as
  SHA-256 hashes. Without a signing key (`--key-file` or
  `SCMLE_SIGNING_KEY`) nothing is changed.
- Retired models are refused. Retiring writes `retired.json` into the
  bundle directory (outside the checksummed members, so the content address
  is unchanged), and `Bundle.load` refuses such a directory with
  `RetiredError` (a `BlockedError`, exit code 2), so `predict`, `batch`,
  `serve`, `package`, `export-onnx` and the registry all refuse it. A
  registered version's release state becomes `retired`.
- `scmle estimate DATA --estimation FILE [--treatment COL] [--output-dir
  DIR]` estimates a declared parameter without training a model: it writes
  `parameter_estimation.json` (each parameter's estimate and interval,
  digest-checked on read-back) or, when the declaration cannot be estimated
  on the data, `parameter_estimation_refusal.json` with the reason, its
  category and details, and exits 2 (gate `estimate.refusal`). The file is a
  JSON object with `estimand` and `policy`, in the stored layout or the
  version-1 layout; a run's own `parameter_estimation_declaration.json` is
  accepted. `scmle run --estimation FILE` makes the same estimate beside
  model selection and keeps it in the run; model selection never reads it.
- `scmle run` and `plan` take `--mitigation {none,reweighing,group_thresholds}`
  for the first sensitive column. Reweighing now also mitigates regression:
  the outcome is cut into equal-count bins (Sturges' rule, at most ten) and
  the bins play the part of the labels, so group and outcome level become
  independent in the weighted development rows. Group thresholds remain
  binary only.
- Per-period drift. With a time column, a charter run's
  `shift_report.json` has a `periods` record: the exploration rows are cut
  into contiguous periods of equal size (at least 50 rows each, at most
  ten), every later period is compared with the first column by column
  (Kolmogorov-Smirnov or chi-square) and Benjamini-Hochberg runs once over
  every period and column. Status is `pass`, `fail` or `unverifiable` (no
  time column); reports written earlier read as `not_run`. Like the other
  shift tests it detects covariate drift only, and a pass is a failure to
  detect.
- `scmle serve` runs the packaged service's loop: SIGTERM or SIGINT stops
  accepting connections and drains the requests in flight
  (`--drain-seconds`, default the longest an admitted request can take);
  SIGHUP reloads the bundle directory while the old model keeps serving,
  and a failed reload keeps the old one. The operations signals over the
  last `--signals-window` seconds (default 300), drift judged against the
  served bundle's training reference, are evaluated every
  `--signals-interval` seconds (default 60) and appended as JSON lines to
  `--signals-file`, or to the log.
- `--trace` (or `SCMLE_TRACING`) exports OpenTelemetry spans of requests,
  scoring calls, scored files and pipeline stages over OTLP; it needs the
  `tracing` extra and is a no-op without it.
- `scmle batch` streams its input: `--chunk-rows` rows (default 10,000) are
  read and scored at a time, so memory holds one chunk; delimited text is
  read record by record (`scmle.data.readers.stream`). Each chunk is
  appended to `<output>.partial` and `<output>.progress.json` records the
  rows and bytes written; running the same command again resumes after the
  last scored chunk, and `--restart` scores from the first row.
- `validation.entities` (`known`, the default, or `unseen`) for temporal
  and rolling splits with `data.group_column`: `known` forecasts the
  entities trained on, so an entity's past trains and its future is held
  out; `unseen` keeps every entity wholly on one side of the test split and
  of each fold. `unseen` without `data.group_column` is refused. A
  configuration at the default hashes as it did before the field existed.
- C14 runs on 90 more public OpenML tables (`benchmarks/evidence/refusal.py`
  `SUITE`, listed in `benchmarks/evidence/data/refusal/README.md`). The
  selection rule was fixed before any was run: every dataset of
  OpenML-CC18, the AutoML benchmark classification and regression suites
  and OpenML-CTR23, less those already in the catalog, copies of a table
  another evidence set uses (`COPIES`), tables over 20,000,000 cells and
  tables with no stated licence. Every OpenML table is pinned by the
  SHA-256 of its ARFF file (`PINS`). No product change.
- Gate `split.test_support`: a binary or multiclass split whose test rows
  hold fewer than `MIN_TEST_CLASS_ROWS` (5) rows of a class is refused. A
  binary run with two minority rows used to complete with both in
  development, test ROC AUC NaN and F1 1.0 on a single class. Five is the
  fewest rows of a class on which even a perfect score has an exact upper
  bound of its error rate, at `EVIDENCE_ALPHA` 0.05, below one half. The
  stratified split now reserves that many test rows for every class with at
  least `MIN_TEST_CLASS_ROWS + MIN_STRATUM_ROWS` (7) rows, drawn with the
  validation seed; a split whose classes already reach it is unchanged.
- One plugin registry for every kind of extension (`scmle.runtime.plugins`).
  `plugins.REGISTRY` enrols, withdraws, lists and finds items by a
  `plugins.Kind` (`ESTIMATORS`, `METRICS`, `READERS`, `DETECTORS`, `STAGES`,
  `TRANSFORMERS`) and loads a kind's entry-point group with `discover`, which
  reports a broken plugin instead of raising. The groups are `scmle.models`
  (unchanged, so installed model plugins keep loading), `scmle.metrics`,
  `scmle.readers`, `scmle.detectors`, `scmle.stages` and the new
  `scmle.transformers`. Estimators enrolled here join the model catalog,
  metrics the metric catalog and stages the standard stage sequence, where
  their `requires` and state place them.
- One state machine for every run (`scmle.runtime.state.Machine`): a state
  and a transition table given as data. The tabular run (`state.TABLE`,
  through `Tracker`, which still emits an event per transition), the
  modality lifecycle (`modality_lifecycle.state.TABLE`) and the workflow
  executor (`workflow.executor.status.TABLE`) are three tables on it, and
  `state.chain` builds a forward table. A workflow run now refuses a move its
  table does not allow; `Executor.resume` returns a run whose state has no
  way out (succeeded, rejected, expired, budget-stopped) as it is, as before.
- The workflow executor keeps a run's records through
  `workflow.executor.records.Bucket`, the object-store implementation of the
  engine's `Archive` protocol (`scmle.pipeline.engine.Archive`, now generic
  over what a stage makes): `record` writes `steps/<step>.json`, `restore`
  hands back a succeeded step's outputs, `keep` writes `run.json` and
  approvals. Keys and fields are unchanged, so runs stored by earlier
  versions load and resume (`tests/fixtures/workflow_records.json`); leases,
  fencing, the cost budget and its append-only `spend` list, approvals and
  resume behave as before.
- Stage records and a completion marker (CON-3). After each stage finishes and
  its approval gates pass, a run writes `stages/NN-name.json` atomically with
  the stage, its state, its seconds, the run's elapsed seconds, the
  configuration hash and its own digest. After the run reaches COMPLETED it
  writes `complete.json` last of all; a run directory without it is not
  complete, whatever else it holds.
- Leakage screen: dates (as day counts), free text (indicators for its 50
  commonest words), pairs of two-level columns (a joint skill test, run
  only while there are no more pairs than columns; otherwise the notes say
  the pairs were not screened) and echoes of the target (a column read as
  the target with no fitting: agreement with the label, or R^2 for a
  number, held to the existing 0.9 association ceiling) are now screened.
  Profiling passes date and text columns to the screen, each suspect is
  listed once, and the permutation count is rounded down. C8 `leak_text`,
  `leak_date` and `leak_xor` are caught; C2 post-outcome columns 6 of 16 ->
  16 of 16 at screen level. Strong legitimate predictors still block.
- Charter: a second run of the same signed charter on the same data is
  refused before it reopens the confirmation set; the use is recorded in
  `scmle/data/ledger/usage.py` (C8 `charter_rerun`). Not yet covered: a run
  written to a different output directory, or a different split of the
  same data.
- Intervals: binary ROC AUC and RMSE get a studentized (bootstrap-t)
  interval with a standard error clustered by unit (DeLong for ROC AUC,
  the delta method for RMSE). Coverage of the 95 percent band: grouped ROC
  AUC 816 -> 836 of 882, RMSE 818 -> 827 of 882 (C3).
- Logging: `--log-json` and `SCMLE_LOG_JSON` (for the generated service)
  write one JSON object per line; every record carries the run id and
  request id. Under-sized oversampling classes, uncast synthetic columns and
  a missing resource module now log `WARNING` and raise a `Degraded`
  warning (2.1, ERR-22, ERR-24). Run as `python -m scmle.cli`, the CLI's
  own messages (for example the charter hash from `charter check`) are kept:
  the entry point runs the imported `scmle.cli`, whose logger is inside the
  package the handler is installed on.
- Tests: 18 new mutation-targeted test files. On the redrawn 19-module
  sample, after packages were split, 423 of 540 mutants are killed (0.78);
  the sample decided only 18 modules, so C10 mutation is not measured yet.
- C8: the automated bypass search now finds 0 bypasses in 47 attempts (43
  blocked, 4 through a documented override), down from 9. C8 stays not
  measured: only a red-team exercise by people can pass it.
- Measured scorecard evidence for C1, C2, C4 and C5, and C14 decided on
  65 real tables (stage A; no product change):
  - C1 passes: all 43 registered gates block their injected fault and all
    9 overrides leave a record (simulated inputs).
  - C2 fails on 146 real tables: 11 of 68 leaks caught (64 planted on real
    data, 4 documented real leaks) and 8 of 78 clean tables blocked.
  - C4 fails on 15 real OpenML tasks against a pre-registered gradient
    boosting baseline: median normalised gap 0.016 binary, 0.026
    multiclass, 0.027 regression (limit 0.02); tabular only.
  - C5 fails: 9 of 34 awkward files read silently wrong (`;` and `|`
    delimiters, duplicate headers, leading zeros, big integers, short last
    row); point-in-time features match the naive reference in 1000 of 1000
    simulated worlds.
  - C14: 0 of 65 real runs refused, so the rate passes; precision is not
    yet decidable (too few justified refusals).
- Measured scorecard evidence for C3, C6, C7, C12 and C14.
  `benchmarks/evidence/` holds one module per claim; each stores its raw
  results next to it, refuses to report a pass on fewer trials than the
  `acceptance.toml` rule supports, and goes stale when the code it measured
  changes (CI checks this). The scorecard reads them:
  - C3 fails: the 95% bootstrap metric interval covers 818/882 for RMSE and
    816/882 for grouped ROC AUC, lower bounds 0.911 and 0.909 against 0.93.
  - C6 fails: bundles are byte-identical and metrics equal on this
    platform, but the manifest records no commit, configuration or data
    hash (ML-10); cross-platform is not measured.
  - C7 passes on an unloaded laptop: 400 of 400 runs finish within a 20 s
    budget plus 2 s grace, and 3 of 3 kill-and-resume runs match an
    uninterrupted run. On a loaded machine most runs overran, and two
    defects were found: profiling (`label_noise.contradict`,
    `fingerprint.ml_difficulty`) runs outside the time budget and thread
    cap, and a small budget leaves the first trial a 1 s ceiling so the
    whole run fails.
  - C12 fails: 14 of 318 documented guarantees are linked to a test.
  - C14 is not measured: 0 of 24 real and 0 of 22 simulated tasks were
    refused and 12 of 12 planted ones were, but 59 real tasks are needed
    and only 34 are cached offline.
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

- `scmle.models.survival.estimators.kaplan_meier` is vectorised (one sort
  and a cumulative product instead of a pass over the rows per event time);
  its values are unchanged.

- Wide tables: `columns.floating`, the numeric cast step of every pipeline,
  converts an all-numeric block in one vectorised step, mapping infinities
  and nullable `NA` to NaN, and keeps the column-by-column `scrub` coercion
  for any block with a non-numeric column. New unit tests in
  `tests/unit/test_columns.py` pin its output to the per-column `scrub`
  result for infinities, nullable `Int64`/`Float64`/`boolean` with `NA`, a
  plain array, a mixed block with unparseable text and a block with no
  columns, and check that the result is a writable copy. On Bioresponse
  (3,751 x 1,776) one cast went from 0.154 s to 0.003 s.
  Admission is unchanged: gina, Bioresponse, Internet-Advertisements and
  fabert are still refused at `budget.minimum_run` (estimates 31.4, 41.8,
  37.3 and 30.4 s). The shipped pipeline therefore does not train them at a
  30 s budget. In a local measurement that bypassed admission through a site
  override (not the shipped pipeline), gina at 30 s ended with
  `logistic_regression` (AUC 0.90) in 26.4 s, where before it ended with
  `dummy` in 31.1 s.

- Phase 6 rename maps (`scripts/renames/*.toml`, not applied yet): resolved
  the `[review]` tables, then corrected the mapped entries against the
  naming rule. The four lanes now map 1,675 names (data_assurance 564,
  models 243, pipeline 278, rest 590) and keep 737 in `[review]`, each with
  two or three candidates and a reason. Mapped functions take verbs (for
  example `cramers_v` to `associate`, `mde_mean` to `detect`, `metric_values`
  to `centre`). Mapped classes take nouns (`FinalEvaluationStage` to
  `Assessment`, `FinalNetBenefit` to `Examiner`). Mapped exceptions keep the
  `Error` suffix (`RuntimeUnconfirmed` to `UnconfirmedError`). These go back
  to `[review]` with a reason: eponyms and abbreviations (`kish`, `ipw`,
  `mae`, `se`, `qini`, `lbfgs`), named-metric families, `is_*` predicates,
  and every `scmle/cli` `run_*` handler that is found by
  `getattr(self, f"run_{action}")`. rename.py cannot follow that computed
  name, so renaming those handlers would break their subcommands.
  Stored-record attributes take the written key that `docs/RECORD_KEYS.md`
  records (for example `Draft.open_items` becomes `pending`).
- Validation: every lane passes rename.py's dry run with rc=0 on the live
  working tree on 2026-11-20. The runs used `--map` on a copy of each map
  with its `[review]` table removed. `scripts/rename.py`'s `read()` now
  drops the `[review]` table before flattening, so the plain lane command
  (`scripts/rename.py <lane> --dry-run`) no longer raises `ValueError` on
  it.
- New `tests/unit/test_rename_kinds.py` finds each mapped definition in the
  source tree and checks three things. A function's new name must be in a
  reviewed verb vocabulary. An exception's new name must end in `Error`. No
  method whose class finds it by a computed `getattr` name may be mapped.
  Run against the maps from before these corrections, it flags 215
  non-verb function names, 3 exceptions without the suffix and 20 handlers.
- After the `--write` apply, docstrings that cite a `Migration.renames`
  source key need the legacy key restored by hand. The rename otherwise
  turns "version 2 spells `n_calibration` as `calibration`" into
  "spells `calibration` as `calibration`". The map headers list the 12
  sites (11 in data_assurance, `contracts/evaluation/parts.py:461` in
  rest).
- `scripts/recite.py` follows line moves in every scanned doc (including
  `website/src`, `docs/preregistration` and `#L` citations), moves both ends
  of a range, and follows a line reworded in place (similarity threshold
  `KIN = 0.59`, the 99th percentile for unrelated doc lines). `--history`
  repairs citations already committed stale; `--base REF --docs` handles a
  second round of edits before a commit.
- C12 snippet check: each page's code blocks run in their own fresh
  environment built from the working-tree snapshot, not in whichever
  environment and `PATH` started the check.
- C1 faults: a path whose extra is missing from the running interpreter runs
  in an isolated interpreter with that extra (as the CI onnx job does), and
  the record names the interpreter used.
- C10 mutation: a module whose run crashes under mutmut's default runner is
  retried under mutmut's fork-server runner; a module still undecided counts
  as undecided.
- Time-budget admission prices each kind with its own `kind.cost`, so a
  media kind needs a larger budget than before for the same row count.
- Label noise has a single cost model: the profiler's noise check is priced
  by `label_noise.cost`, the model that sets how many rows it gets.
- C2 corpus: tecator (OpenML 505) declares `moisture` and `protein` as
  outcomes; its description lists them as outputs determined by analytic
  chemistry. `docs/preregistration/c2.md` records that this was changed after
  `protein` was seen missed.

- Selection is evidence-based: a variant (a data strategy, tuned
  hyperparameters or an ensemble) competes only when a paired test on the
  shared folds gives it at least a 0.95 probability of beating its default
  pipeline (`strategy_search.odds`, `selecting.earn`, `tuning.shortlist`);
  each one dropped is rejected on record. Before, the best cross-validation
  mean of whichever variants finished in time won.
- Budget admission (C14): the first trial's estimated cost grows with the
  column count (`cheapest(rows, columns)`), so a wide table is admitted with
  a linear model first instead of refused at `budget.minimum_run`; the
  bootstrap holds a 10% share of the budget, between the cost of its 200
  minimum resamples and its full cost.

- The leakage screen does not test duplicated records when the screened
  columns' value space holds no more records than the table has rows (an
  enumerated design, such as monks); screens a table with fewer than
  `min_rows` rows with a low-power note instead of leaving it unchecked;
  and judges record purity against the target's base rate, so an
  imbalanced table no longer raises false purity suspects (a rare-class
  record leak on such a table can now be missed, and then warns only).
- C2 corpus: Tecator's (OpenML 505) `moisture` and `protein`, documented
  lab measurements made with the target, are removed from the clean table
  and are two documented real leak cases; bank marketing `V12` and NYC
  green taxi `total_amount` are screened with their documented
  availability declared (`known`). The C2 store is stale until measured
  again.
* **One rule for every size gate.** A gate that measures a test of nominal
  level `L` (the falsification battery, the placebo-in-space test, the Holm
  subgroup families, the intersection-union test, the two-sample shift
  tests, the net-benefit band and the rollout e-process) passes when the
  exact upper bound of its rejection rate is at most `L (1 + size_margin)`.
  A ceiling of `L` itself fails a correctly sized test most of the time.
  `size_margin` (0.69, `acceptance.toml`) is the largest margin any
  registered gate needs, at its replicate count, `alpha` and `power`, for a
  correctly sized test to pass with chance `power`; a test of twice its
  level still fails. `falsification_alarm_max`, `familywise_error_max`,
  `shift_false_alarm_max` (0.075) and `false_useful_max` are removed.
* **Regression-discontinuity tests use Student's t.** The jump, its
  placebos and its covariate balance are referred to t with
  Welch-Satterthwaite degrees of freedom of the HC3 variance (each record
  carries `dof`) instead of the normal. With the normal the falsification
  battery rejected 0.0119 of 30000 development null worlds at level 0.01
  (exact lower bound 0.0109), and rejects 0.0097 with t (upper bound
  0.0107); intervals are slightly wider.
* **The served model is refit on every row.** After the one-time test, the
  selected configuration (with its pseudo-labels, decision threshold and
  group thresholds) is refit on development and test rows together and the
  bundle serves that refit; `evaluation.json` records it under `shipped`
  (`refit`, `rows`, `reason`). The test metrics stay those of the fit that
  never saw the test rows. The refit is skipped, on record, under a charter,
  for streamed training, or when it would not end before the packaging
  reserve.
* **Repeated records are exchangeable draws.** The random, stratified and
  event-stratified splits no longer move copies of one record to one side of
  every split; as in scikit-learn's k-fold and OpenML's splits, a future case
  may repeat a past one, so repeats may fall on both sides. The split notes
  count the test rows that repeat a development record. Copies of one entity
  are declared with `data.group_column`; a charter run's custody carve still
  keeps copies together. `duplicates.cohere`, `locate` and `majority` and
  `Splitter.gather` are removed (`Splitter.disclose` counts instead).
* **C5 split hygiene amended (owner decision 2026-11-19).** Identical rows
  are independent draws and may cross a split, as in scikit-learn and
  OpenML; copies of one declared entity (`data.group_column`, or
  `validation.entities = unseen`) never cross. C5 now requires no declared
  entity and no tied timestamp on both sides of any split; duplicate rows
  that cross are reported and are a violation only when they belong to one
  declared entity. As registered, any duplicate on both sides failed the
  row ([`docs/preregistration/c5.md`](docs/preregistration/c5.md)). Copies
  beyond chance are leakage for C2 instead: the leakage screen's
  `duplication` hypothesis flags a duplicate rate above the table's own
  natural rate (the exact pairs its singletons and triples allow).
* **Gradient boosting stops early at a smaller learning rate.** The
  untuned histogram gradient boosting model (classifier and regressor)
  runs at learning rate
  `SHRINKAGE` 0.05 (was scikit-learn's 0.1) with up to `ROUNDS` 500
  rounds, stopping after `PATIENCE` 20 rounds without improvement on a
  held-out tenth of the training rows; the search still ranges over 0.02
  to 0.3. A fit whose rarest class has a single row runs every round
  instead, and records whether it was allowed to stop (`stopped_`).
* **Classification trees split the same on every CPU architecture (C6).**
  The decision tree, random forest and extra trees classifiers are now
  `scmle.models.sklearn.trees.Tree`, `Forest` and `Extra`. scikit-learn's
  Gini split proxy, fused into one multiply-add on arm64, could break an
  exact tie differently from x86-64; these grow the equivalent regression
  trees on the class indicators, whose ties are decided by exactly rounded
  arithmetic, and store them as stock classification trees, so prediction,
  persistence and ONNX export see ordinary scikit-learn trees.
  `oob_score` and `warm_start` are refused. Bundles now pickle
  `scmle.models.sklearn.trees`, recorded in the pickle census of bundle
  format 4 (`tests/fixtures/pickled_modules.json`).

- Deprecated, warning with `DeprecationWarning` and refused in the next
  release: a plain string as a plugin kind (`plugins.REGISTRY.enrol("metrics",
  ...)`; pass `plugins.Kind.METRICS`), and `scmle.models.registry.adopt`,
  `extensions` and `deregister` (use `plugins.REGISTRY.enrol`, `entries` and
  `withdraw` with `plugins.Kind.ESTIMATORS`).
- The modality lifecycle runs on the pipeline engine. Its mutable `Context`
  is replaced by `modality_lifecycle.Board` (the run's fixed inputs and the
  frozen product of each finished stage) and `clocked` by
  `lifecycle.measure`, which builds the performance observations from the
  engine's stage timings. Reports, checkpoints and their digests are
  unchanged, and checkpoints signed earlier still resume.
- `benchmarks/profile.py` reads each stage's seconds and peak memory from
  the engine (`Pipeline.timings`) instead of wrapping the stages' `execute`,
  so it profiles every stage of the run, enrolled ones included.
- Resume (`Project.resume`, `scmle resume`) keeps the time a run already
  spent: the largest elapsed seconds in its stage records start the time
  budget and the run clock, so the budget is not restarted (CON-4). Resume
  verifies every stage record and refuses (`BlockedError`) one that was
  edited or written under another configuration or record format, asks the
  custody journal first, and treats a run as complete when it has
  `complete.json` (before: when it had `evaluation.json`). Finished stages are
  still derived again from the same data and seeds.
- Non-tabular modalities: each built-in modality's declaration (text,
  document, image, audio, video, graph, geospatial) now states in its
  limitations why it is experimental: it runs the bounded lifecycle, not the
  shared pipeline engine, so it has no candidate search, cross-validation,
  baseline comparison, prediction intervals, locked test set or analysis
  charter. Extension modalities are unchanged. See
  [`docs/WHAT_SCMLE_CANNOT_DO.md`](docs/WHAT_SCMLE_CANNOT_DO.md).
- Restructure: `scmle.core.stages`, `scmle.models.modality_lifecycle`,
  `scmle.evaluation.parameter_estimation`, the six `scmle.data.modalities`
  modules (audio, document, geospatial, graph, image, video) and
  `scmle.workflow.executor` are now packages at the same import paths; no
  caller changes. Modules over 800 lines fell from 54 to about 44.
- Wrong-kind inputs raise `errors.KindError` and calls made out of order
  (predicting before fitting, a stage before the stage it needs) raise
  `errors.OrderError` (ERR-20): 223 former `TypeError` and 27 former
  `RuntimeError` raises. `KindError` subclasses `InputError` and `TypeError`;
  `OrderError` subclasses `InputError` and `RuntimeError`, so existing
  `except TypeError` and `except RuntimeError` still catch them, and they
  now get the same input-error handling as `InputError`. Kept as built-ins:
  `TypeError` that Python protocols require, the HTTP request parser's
  errors (malformed requests stay 400), and `RuntimeError` that marks a
  defect (a report or package that was not produced, an invalid internal
  stage order).
- Input problems raise `errors.InputError` instead of a bare `ValueError`
  (ERR-20): 1,874 raises across the package. `InputError` subclasses both
  `DataError` and `ValueError`, so `except ValueError` still catches them.
  What callers see: the CLI reports bad input as an input error with exit 1
  (it was "internal error (a defect)", exit 70); a run that fails on bad
  input is FAILED with category `input` rather than `internal`; a rejected
  rollback now restores the previous plane; recorded failure kinds read
  `InputError`. Request-shape errors in the HTTP server stay `ValueError` so
  malformed requests keep answering 400. About ten raises that guard internal
  invariants stay `ValueError`.

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
- The cross-validation band of the winner (`cv_confidence.interval`) is moved
  down by the estimated flattery of choosing the best of several candidates
  (`selection_optimism`, new, with `candidates`; `confidence.cv_interval` takes
  the fold scores of every finished candidate and `greater_is_better`). Measured
  on pure-noise labels (1,000 rows, 600 runs per setting, three model families
  with hyperparameter trials added): the band contained the true value in 95.0
  percent of runs at 3 candidates and 90.3 percent at 30 (87.7 to 92.4), and
  lay wholly above the baseline in 4.5 and 9.7 percent. With the move, 30
  candidates gave 95.0 percent (91.0 to 97.3, 200 runs; 89.5 before on the same
  runs) and 5.0 percent above the baseline (10.5 before), 3 candidates were
  unchanged, and the width is the same. A Bonferroni widening covered 99.8
  percent at 30 candidates with bands 2.7 times as wide and a max-over-
  candidates critical value 97.8 percent at 1.6 times, so neither was used.
  The exam-score band (`metric_uncertainty`) is not changed: in the same runs
  it contained the true value in 96.7 percent at 3 candidates and 95.7 percent
  at 30 and was above chance in 0.8 percent at both, because the exam rows are
  opened after the choice. Still uncorrected: the estimate treats the winner as
  the best mean (ignoring tie-breaking and serving limits), and it was checked
  only on these synthetic settings. `run.evaluation` and the trail/model-card
  copies of `cv_confidence` change, so stored certificates and fixtures that
  embed it must be regenerated. `benchmarks/statistics/null_selection.py`
  takes `--configs-per-model` and `--signal`; the cell summaries are in
  `benchmarks/statistics/multiplicity_summaries.json`.
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

- `scmle.evaluation.calibration_uplift` is removed: its uplift check is
  `scmle.evaluation.calibration.uplift`, and `calibration.horizons` is now
  `calibration.quartiles`.
- Gates `ingest.multiplicity` and `robustness.assessment_present`, which no
  input could reach (`tests/unit/test_unreachable_gates.py`). The declared
  family size and the synthesized columns are counted from one dictionary;
  the split keeps every column but the target; the exploration plan puts
  every column among its hypotheses or its skipped list and registers every
  hypothesis; and the governed exploration always runs the stability stage,
  which sets its assessment or raises before the report is written.
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

- A request whose scoring hangs no longer wedges a replica. The server kept
  refusing every request with 503 while one timed-out call was still running,
  and a call that never returned kept it unready and alive forever. Now the
  replica keeps serving on its other workers. It is unready only when no
  call can run (every worker holds a timed-out call, or a series model's lock
  is held), and `/livez` fails once that has lasted `STRIKES` times the scoring
  timeout.
- A hung scoring call no longer keeps a drained server process from exiting
  after SIGTERM: scoring runs on daemon threads (`workers.Crew`).
- A client that sends its body slowly no longer holds a scoring slot. The body
  is read before admission, and a body not received within the request I/O
  timeout gets a 408.
- A caller that reset its connection while sending the body no longer
  escapes the server's handling: it is logged and the connection dropped.
- Connections the kernel had queued when the server stopped were reset. The
  listener now answers the backlog before it closes. Its listen backlog is
  `socket.SOMAXCONN` instead of 5, so bursts of callers were refused or reset
  before.
- C2 target-encoding plant bins by value, so tied values share a level
  (OpenML 1464 now blocks).

- A charter run's leakage screen reads the declared `data.cutoff` and
  `data.known` (admission passes them to `qualify.run`), so a column known
  only after the outcome is quarantined there as in a run without a charter.
- `scripts/recite.py` is idempotent: a citation moves only while it still
  holds its committed line number, so a second run changes nothing.

- MinHash near-duplicates (`duplicates.resemble`) no longer raise
  `TypeError` on a missing text value under pandas 3, whose `astype(str)`
  keeps NaN as a float; it reads as `nan`, as under pandas 2. This broke
  the custody vault bind, vault open and confirmation single-use gates on
  real data.
- `opening.evaluate` logs and re-raises a `LeakageError` (a foreign or
  tampered Thresholdout journal) instead of recording the confirmation as
  `failed`.
- The custody vault process no longer leaks when carving the confirmation
  set, writing its records or an interrupt fails after the set is sealed:
  `sequester` closes it before re-raising.
- C1 fault injection: the stand-in context of the admissibility fault uses
  the run's services and journal, and the reuse-journal fault is injected
  on a real charter run (bank marketing, OpenML 1461).

- On Linux the memory limit no longer undercounts: a process whose name holds
  non-ASCII bytes failed to decode in `/proc/<pid>/stat` and was skipped from
  the resident-memory sum. Only the numeric fields after the name are read,
  and undecodable bytes in the name are replaced.
- Resume: a run whose test set was opened, or that has no intact custody
  journal, cannot be resumed (`BlockedError`), and a gate a checkpoint
  rejected stays rejected until a checkpoint passed to
  `Project.resume(..., checkpoints=())` approves it. Start, checkpoint
  verdicts and the test-set opening are recorded in a hash-chained custody
  journal outside the run directory (`runs/custody/<run>.jsonl`), so
  deleting or editing run files (`evaluation.json`, `config.json`,
  `events.jsonl`) cannot erase them (C8).
- Readers: CSV detects the encoding (BOM, UTF-8, cp1252) and delimiter,
  and refuses duplicate headers, ragged rows, ambiguous decimal or
  thousands commas and numbers that would not read exactly; zero-padded ids
  stay text, and integers above 2**53 become exact `Int64` with gaps (C5: 0
  of 34 awkward files read silently wrong, was 9).
- Crashes: native-categorical codes are capped at 255 levels, rarer levels
  pooled and recorded; calibration folds follow the rarest class instead of
  crashing (OpenML 315, 363711).
- Serving: a hung scoring call no longer blocks other requests. Tabular
  serving drops its global scoring lock for models without series features;
  modality serving answers 503 until a timed-out call returns and fails
  liveness after 3 consecutive timeouts (SRV-1, SRV-2).
- Workflow leases: every record written under a leased run carries its
  fencing token and is written by compare-and-swap, so a writer that lost
  its lease gets `EvictionError`; step timeouts use a monotonic timer and a
  renewal never moves the expiry earlier (CON-1, CON-20).
- Security: deep-learning checkpoints and document and image packages
  refuse `trusted=True` under `SCMLE_ENV=production`, as bundles do
  (SEC-1, SEC-2).
- A release attestation that cannot be checked is logged with its reason
  (the release still blocks), and a deployment-ledger amendment that is not
  a mapping is refused instead of silently changing the policy (ERR-4,
  ERR-23).
- A run no longer exceeds its time budget. Seven of fifteen real tasks
  overran a 60 s budget (up to 129.5 s) because only trials were bounded:
  preparation ran unclocked (the leakage screen's cap assumed 25,000 fits
  per second whatever the rows, and applied it twice), the strategy-search
  batch fixed every job's ceiling at its start, trials could run 10 percent
  past the budget, and diagnostics took their own 30 s after it. Now:
  - before training, the run reserves time for the final refit,
    diagnostics and packaging from fitted cost models
    (`Governor.until("trials")`, `until("finish")`); trials, strategy
    search and tuning end by the trial deadline, with no grace past it;
  - preparation runs inside a stated share of the budget: the screen's
    cost scales with rows, the real scan and the planted probe share one
    15 percent cap, and the difficulty and label-noise checks subsample or
    skip to fit 5 percent each;
  - diagnostics get only the time left; the metric bootstrap sizes its
    resamples to fit and skips (with no interval) below its minimum;
  - work that does not fit is reduced or skipped and recorded in
    `evaluation.skipped_for_time` and a `budget.skipped` event, never run
    late;
  - a budget too small for even the minimum run is refused before any work
    with the estimate and the smallest budget that fits. This replaces the
    earlier rule that gave the first trial a whole budget, which allowed an
    overrun. The only remaining excess is `limits.SLACK_SECONDS` (poll
    interval plus kill time).
  The seven tasks now finish in 40.8 to 58.6 s, and 0 of 167 runs on 46
  real tables overran (5 crashed for other reasons: a categorical column
  with over 255 values, and 3-fold CV on a class with fewer than 3 rows).
  Trade-off: the screen's cost is the one-thread worst case, so large
  multi-class tables are screened less often (digits now needs
  `--time-budget 2800`). Under `block` a screen that does not run blocks,
  so C2's false blocks rose from 8 to 12 of 78 clean real tables (digits,
  OpenML 14, 16 and 28); leak recall is unchanged at 11 of 68.
- The move to `InputError`, `KindError` and `OrderError` (ERR-20) changed
  what some handlers did, because a handler for `ScmleError` now also
  caught input problems that used to pass it as plain `ValueError`,
  `TypeError` or `RuntimeError`. Each such handler now does something
  chosen on purpose, pinned by a test:
  - Restored: batch processing (`IncrementalProcessor.run`) raises on bad
    input again instead of returning a partial result; the claims audit
    raises on an empty signing key instead of recording failed steps; the
    service answers an input problem while scoring with 400 again (500 for
    a call made in the wrong order) instead of an undocumented 422, and its
    public readiness reason shows only an input error's type, not its
    message; a failed workflow step shortens an input error's text as
    before.
  - Kept: the command line exits 1 ("input error", as `docs/CLI.md` says)
    rather than 70 ("internal error, a defect"); a run that fails on bad
    input is marked `input` rather than `internal`; a rollback rejected by
    an input problem puts the previous revision back, as `rollback`
    promises, instead of leaving the platform out of step.
- A bundle now traces back to what built it (ML-10). The manifest records
  the run's `seed`, the `configuration` hash, the `dataset` hash, the
  `split` fingerprint, the scmle `commit` (empty when scmle runs from an
  installed wheel, whose version names the code), and the math-library
  `threads` and `blas` in use. Older bundles load with these fields empty.
  The C6 scorecard row no longer fails on the manifest; it is "not
  measured" until a second platform is compared.
- A small time budget no longer ends the run with "all trials failed". The
  first trial always runs so that a result exists, but when preparation had
  used up the budget it got the one-second minimum ceiling, too short to
  start a worker, and was killed. It now gets a whole budget plus the grace
  share, so a run whose preparation fills its budget can take up to about
  twice the budget; later trials still get only what is left. A 2 s budget
  on the quick project now finishes in about 6 s; a 1 s budget is still too
  short to start a worker.
- `cpu_limit` now also caps the math-library threads of the main process
  while the stages run. Profiling, diagnostics and evaluation fits run
  there and used every core whatever the limit said; only trial workers
  were capped.
- Every text file the package and its tests open now names its encoding
  (`utf-8`), and ruff enforces it (`PLW1514`). Windows defaults to a legacy
  code page, so reading a key, policy or attestation file with non-ASCII text
  could fail or read differently there. Not run on Windows; the README now says
  which Windows scenarios CI covers and that the rest has only run on Linux
  and macOS.
- On Windows, `scripts/admit.py` now reads available memory with `psutil` (it
  treated memory as unlimited, so the gate never held work back) and the
  deployment demo reads a child's memory with `psutil` instead of `ps`, which
  does not exist there. Not run on Windows.

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

- **A bundle member swapped after its checksum is checked no longer loads.**
  `Bundle.load` checksummed each pickled member (model, calibrator, novelty
  detector, quantile and CV+ models, shift weights) and then opened the file
  again to unpickle it, and checked the manifest signature over a second
  read of the manifest file. A file replaced between the two reads ran
  unauthenticated code. Each member is now read once, checked against the
  signed manifest and loaded from those bytes (`archive.resolve` returns
  bytes, `archive.unpickle(data, name)`). The signature and the schema
  checksum cover the bytes that were parsed (`archive.Snapshot`,
  `seal.verify_signature(root, key, text)`). The JSON and Parquet members
  are read the same way. `tests/unit/test_bundle_swap.py` covers it.
- Tenant isolation at the storage layer: `scmle.storage.tenancy.Tenancy`
  confines every operation to `tenants/<tenant>/`, refuses crafted keys and
  keys naming the namespace root, and refuses a backend listing outside the
  namespace; tenant names are lowercase DNS labels. The job entrypoint
  confines its store to `SCMLE_TENANT`, `Gateway.confine` gives a verified
  caller only its own tenant's namespace, an experiment store keeps each
  tenant's runs in a database of its own, and a local store refuses any path
  through a symbolic link (`EscapeError`). A JWK `tenant` field binds a key
  to one tenant, so it verifies only that tenant's tokens. A link swapped in
  between the check and the open is not caught; see
  [`docs/TENANT_ISOLATION.md`](docs/TENANT_ISOLATION.md).
- Tenant confinement reaches local runner jobs: `python -m
  scmle.runners.local` confines its store to `SCMLE_TENANT`, `Process` and
  `Container` accept a `Tenancy` over a local store, and each job's
  environment sets `SCMLE_TENANT` to the runner's tenant (empty when the
  store is not confined), overriding the spec's own, so the job opens the
  namespace its runner was given. The job entrypoint confines by the job's
  environment, not the process's.

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
