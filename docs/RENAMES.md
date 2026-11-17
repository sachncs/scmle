# Renames

Names renamed under the single-word naming rule: classes are nouns, functions
are verbs, attributes are adjectives, and snake_case is kept only where no
single word fits. Each entry is `old -> new`, grouped by module. Signatures
and behaviour are unchanged; replace the old name to migrate. Renames of
names in the public API snapshot are also listed in the changelog.

## `scmle.api.non_tabular`

- `raster_scope` -> `delimit`
- `fixed_scope` -> `invariant`
- `capability_for_adapter` -> `entitle`
- `check_observed_adapter` -> `conform`

## `scmle.api.project`

- `build_parameter_estimation_declaration` -> `declare`
- `read_declaration_envelope` -> `unwrap`
- `restore_parameter_estimation_declaration` -> `restore`
- `bind_parameter_estimation_evidence` -> `bind`
- `persist_parameter_estimation_evidence` -> `persist`
- `require_treatment_column` -> `locate`
- `estimate_parameters_for_project` -> `estimate`
- `verify_existing_curation` -> `verify`
- `check_manifest_matches_adapter` -> `conform`
- `publish_curation_artifact` -> `emplace`
- `parameter_estimation_declaration_digest` -> `fingerprint`

## `scmle.api.results`

- `unattested_suspects` -> `suspect`
- `feature_importance` -> `salience`
- `health_label` -> `diagnose`
- `meets_criteria` -> `accomplished`
- `claim_tiers` -> `tally`

## `scmle.api.sequential`

- `resolve_tabular_input` -> `tabulate`

## `scmle.artifacts.bundle`

- `installed_versions` -> `detect`
- `minor_version` -> `truncate`
- `sign_manifest` -> `sign`
- `verify_core_files` -> `authenticate`
- `verified_member_path` -> `resolve`
- `shifted_interval` -> `reweight`
- `installed_distributions` -> `inventory`
- `prepare_frame` -> `prepare`
- `require_conformal` -> `require`
- `verify_round_trip` -> `roundtrip`
- `file_sha256` -> `shasum`
- `novelty_of` -> `atypical`
- `read_manifest_and_schema` -> `unpack`
- `series_builder` -> `assemble`
- `attach_shift_weights` -> `tilt`
- `load_monitoring_members` -> `hydrate`
- `write_monitoring_members` -> `persist`
- `load_interval_members` -> `restore`
- `rank_for_labeling` -> `prioritize`
- `compatibility_problems` -> `incompatibilities`
- `Bundle.export_service(destination, signing_key)` -> `scmle.inference.packaging.scaffold(bundle, destination, signing_key=signing_key)` (the method was a pass-through and is removed)

## `scmle.artifacts.claims_audit`

- `coverage_status_problems` -> `reconcile`
- `evaluate_attestation` -> `assess`
- `baseline_report_problems` -> `misnamed`
- `check_replay` -> `rederive`
- `decision_file_problems` -> `tampered`
- `show_claim` -> `inspect`
- `read_claims` -> `recall`
- `read_ledger` -> `rehydrate`
- `check_charter` -> `authenticate`
- `check_stage_two` -> `tally`
- `check_control` -> `witness`
- `check_coverage_claim` -> `corroborate`
- `custody_open_finding` -> `openings`
- `evaluate_attestations` -> `vet`
- `ledger_header_mismatches` -> `compare`
- `list_claims` -> `inventory`
- `registered_rows` -> `enrolled`
- `check_claims_against_ledger` -> `crosscheck`
- `unregistered_coverage_finding` -> `orphaned`
- `check_attestations` -> `vouch`
- `check_coverage_confirmation` -> `reconfirm`
- `check_custody` -> `custodial`
- `is_confirmed` -> `upheld`

## `scmle.artifacts.reproducibility`

- `trial_identity` -> `identify`
- `structural_differences` -> `compare`
- `load_run` -> `reread`
- `compare_runs` -> `judge`
- `ReplayVerdict` -> `Comparison`

## `scmle.artifacts.store`

- `RunStore` -> `Workspace`

## `scmle.cli`

- `PlanCommand` -> `Plan`
- `RunCommand` -> `Run`
- `ResumeCommand` -> `Resumption`
- `CatalogCommand` -> `Cataloging`
- `read_raw_bytes` -> `read`
- `PredictCommand` -> `Prediction`
- `BatchCommand` -> `Batch`
- `ExportOnnxCommand` -> `Export`
- `PackageCommand` -> `Packaging`
- `CharterCommand` -> `Charter`
- `LoopCommand` -> `Loop`
- `OperateCommand` -> `Operation`
- `DeployCommand` -> `Deployment`
- `read_key` -> `read`
- `ClaimsCommand` -> `Claims`
- `FrameCommand` -> `Frame`
- `TrailCommand` -> `Trail`
- `build_project` -> `scaffold`
- `add_validation_arguments` -> `validators`
- `ServeCommand` -> `Serve`

## `scmle.contracts.artifacts`

- `BundleSchema` -> `Interface`
- `pipeline_columns` -> `consumed`

## `scmle.contracts.capabilities`

- `ModelFamilyDeclaration` -> `Declaration`
- `register_model_family` -> `enroll`
- `resource_diagnostics` -> `audit`
- `ResourceAssumptions` -> `Resources`
- `UnsupportedCapabilityError` -> `UnsupportedError`
- `list_capabilities` -> `catalogue`

## `scmle.contracts.data`

- `check_findings` -> `vet`
- `FindingDisposition` -> `Disposition`
- `ModalityManifest` -> `Passport`
- `blocking_leaks` -> `blockers`
- `CurationFinding` -> `Finding`
- `FeaturePlan` -> `Blueprint`
- `unchecked_columns` -> `untested`
- `SourceMetadata` -> `Origin`
- `DatasetProfile` -> `Portrait`
- `screen_warning` -> `admonition`
- `LineageMetadata` -> `Ancestry`
- `derived_names` -> `engineered`
- `from_curation` -> `distill`
- `QualityObservation` -> `Observation`
- `PrivacyMetadata` -> `Confidentiality`
- `input_columns` -> `sourced`
- `columns_for` -> `pick`

## `scmle.contracts.evaluation`

- `validate_evidence` -> `vet`
- `validate_finite` -> `quantify`
- `check_open_unit_interval` -> `confine`
- `check_explanation_threshold` -> `qualify`
- `CalibrationMethod` -> `Procedure`
- `MetricDeclaration` -> `Declaration`
- `validate_fairness` -> `scope`
- `FairnessMethod` -> `Technique`
- `MetricGate` -> `Hurdle`
- `ExplanationAvailability` -> `Explicability`
- `FairnessGate` -> `Equity`
- `validate_metric_gates` -> `gatekeep`
- `DiagnosticAction` -> `Remedy`
- `SmallGroups` -> `Subgroups`
- `validate_release` -> `audit`
- `MetricDirection` -> `Orientation`
- `UnsupportedEvaluationPolicyError` -> `UnsupportedError`
- `DiagnosticGate` -> `Requirement`
- `UndeclaredEvaluationPolicyError` -> `UndeclaredError`
- `validate_uncertainty` -> `enforce`

## `scmle.contracts.events`

- `utc_timestamp` -> `stamp`

## `scmle.contracts.experiments`

- `TrialStatus` -> `Termination`
- `HistoricalRun` -> `Precedent`

## `scmle.contracts.explanations`

- `validate_fingerprint` -> `require`
- `validate_name` -> `require`
- `validate_parameters` -> `normalise`
- `validate_reason` -> `require`
- `validate_redacted_fields` -> `trim`
- `validate_uri` -> `screen`
- `validate_json_fields` -> `normalise`
- `validate_measurement_object` -> `require`
- `EvidenceLink` -> `Link`
- `validate_json_value` -> `sanitise`
- `DatasetSnapshot` -> `Snapshot`
- `validate_evidence_links` -> `linked`
- `FeatureGraphReference` -> `Lineage`
- `PrivacyDecision` -> `Clearance`

## `scmle.contracts.lineage`

- `topological_order_without_validation` -> `linearise`
- `path_exists` -> `reachable`

## `scmle.contracts.metrics`

- `FunctionMetric` -> `Formula`

## `scmle.contracts.modalities`

- `check_no_blank_values` -> `blanks`
- `UnsupportedModalityError` -> `UnsupportedError`
- `ExtensionModality` -> `Bespoke`
- `list_declarations` -> `catalog`
- `LifecycleStage` -> `Phase`
- `StageEvidence` -> `Proof`

## `scmle.contracts.observability`

- `InferenceEvent` -> `Inference`
- `SignalKind` -> `Indicator`
- `MetricSample` -> `Sample`

## `scmle.contracts.performance`

- `sample_count_violations` -> `tally`
- `relative_change` -> `rise`

## `scmle.contracts.qualification`

- `SetId` -> `Subset`
- `require_tuple` -> `tupled`

## `scmle.contracts.reliability`

- `FaultScenario` -> `Contingency`
- `production_beta` -> `prerelease`
- `record_evidence` -> `log`
- `VerificationStatus` -> `Verification`
- `validate_complete` -> `insist`
- `check_evidence_requirements` -> `require`
- `default_fault_injection_plans` -> `drills`

## `scmle.contracts.search`

- `SearchSpace` -> `Space`
- `FloatParameter` -> `Real`
- `IntParameter` -> `Integer`
- `ChoiceParameter` -> `Choice`

## `scmle.contracts.serving`

- `SchemaChangeError` -> `MismatchError`
- `ScoreRequest` -> `Inquiry`

## `scmle.contracts.signing`

- `key_from_environment` -> `lookup`
- `resolve_key` -> `keying`
- `sign_payload` -> `mac`

## `scmle.contracts.tasks`

- `series_node` -> `derive`
- `forecast_nodes` -> `featurize`
- `FrozenModel` -> `Schema`
- `check_structure` -> `verify`
- `expand_feature_delays` -> `lag`
- `check_validation_rules` -> `validate`
- `check_target_column_rules` -> `exclusive`
- `effective_strategies` -> `effective`
- `horizon_names` -> `horizons`
- `FeatureNode` -> `Node`
- `FeatureGraph` -> `Derivation`
- `fast_budget` -> `brief`
- `effective_search` -> `budgeted`

## `scmle.core.causal_claim`

- `load_dag` -> `unseal`
- `verified_attestors` -> `credit`

## `scmle.core.compiler`

- `WorkflowStage` -> `Node`

## `scmle.core.config`

- `load_sections` -> `overlay`

## `scmle.core.context`

- `add_outcome` -> `enter`
- `supported_tabular_capability` -> `advertise`
- `tabular_beta_capability_registry` -> `enrol`
- `resource_governor` -> `regulator`
- `fairness_options` -> `fairnesses`
- `persist_capability_observation` -> `observe`

## `scmle.core.decision_reports`

- `file_digests` -> `hashsums`

## `scmle.core.ingest_stage`

- `read_view` -> `load`
- `load_plan` -> `recipe`
- `rule_of` -> `decode`

## `scmle.core.ipc`

- `wait_for_helper` -> `rendezvous`
- `connect_to_parent` -> `attach`

## `scmle.core.limits`

- `kill_descendants` -> `reap`
- `memory_exceeded` -> `exceeds`
- `enforcement_text` -> `explain`
- `forkserver_environment` -> `preforked`
- `process_tree` -> `footprint`
- `child_main` -> `execute`
- `proc_table` -> `tabulate`
- `thread_limit` -> `threadcap`
- `trial_ceiling_seconds` -> `deadline`
- `TrialLimitError` -> `TrialError`
- `process_table` -> `pids`

## `scmle.core.orchestrator`

- `load_history` -> `recall`
- `apply_plan` -> `adopt`
- `load_checkpoints` -> `restore`
- `for_run_directory` -> `reload`

## `scmle.core.policies`

- `GateReview` -> `Review`

## `scmle.core.randomness`

- `validate_key` -> `vet`
- `name_words` -> `derive`

## `scmle.core.resource_observation`

- `observe_extras` -> `optionals`
- `observe_cpu_cores` -> `processors`
- `is_installed` -> `present`
- `observe_disk_gb` -> `freespace`

## `scmle.core.scheduler`

- `create_backend` -> `select`

## `scmle.core.stages`

- `must_stream` -> `overflows`
- `persist_source_snapshot` -> `enrol`
- `refuse_stale_resume` -> `forbid`
- `PendingLedger` -> `Escrow`
- `resolve_task` -> `infer`
- `check_role_columns` -> `vet`
- `profile_development` -> `survey`
- `record_curation` -> `curate`
- `assign_entity_groups` -> `regroup`
- `complete_claim_ledger` -> `conclude`
- `check_ingest_family` -> `audit`
- `write_registered_claims` -> `publish`
- `run_robustness` -> `stress`
- `assess_framing` -> `ground`
- `causal_runner` -> `hypothesise`
- `assess_identification` -> `identify`
- `apply_reweighing` -> `reweigh`
- `apply_charter_custody` -> `impound`
- `make_split` -> `divide`
- `attach_feature_graph` -> `graft`
- `selected_rule` -> `codify`
- `chartered_threshold` -> `freeze`
- `chartered_net_benefit` -> `appraise`
- `diagnostic_seconds` -> `ration`
- `selected_health` -> `diagnose`
- `mitigate_fairness` -> `mitigate`
- `stream_refit` -> `relearn`
- `streamed_locked_metrics` -> `tally`
- `mitigation_on_locked_rows` -> `corroborate`
- `self_train` -> `pseudolabel`
- `register_coverage_claim` -> `pledge`
- `provenance_record` -> `notarise`
- `sbom_record` -> `inventory`
- `conformal_calibration` -> `conformalise`
- `keep_cv_plus` -> `retain`
- `complete_held_ledger` -> `settle`
- `quantile_intervals` -> `bracket`
- `build_evidence_graph` -> `trace`
- `measure_coverage` -> `gauge`
- `chartered_rule` -> `governing`
- `pareto_report` -> `frontier`
- `synthesize_features` -> `synthesize`
- `build_plan` -> `devise`
- `reliability_plan_record` -> `persist`
- `capability_key` -> `requisition`
- `register_decision_files` -> `notarize`
- `PlanStage` -> `Planning`
- `declared_slice_columns` -> `slicers`
- `note_conformal_skipped` -> `waive`
- `write_decision_reports` -> `decide`
- `chartered_decision_model` -> `decisional`
- `add_quality_notes` -> `annotate`
- `entity_calibration_column` -> `grouping`
- `cv_plus_cap` -> `allowance`

## `scmle.core.state`

- `InvalidTransitionError` -> `TransitionError`

## `scmle.data.admissibility.asof`

- `RowStatus` -> `Status`
- `AsOfAudit` -> `Audit`
- `classify_rows` -> `classify`
- `reconstruction_mismatch` -> `disagree`

## `scmle.data.admissibility.attestations`

- `LeakageAttestation` -> `Affidavit`

## `scmle.data.admissibility.dag`

- `require_nodes` -> `require`
- `children_map` -> `successors`
- `parents_map` -> `ancestry`
- `backdoor_valid` -> `adjusts`

## `scmle.data.admissibility.independence`

- `is_discrete` -> `nominal`
- `CIKind` -> `Procedure`
- `implied_statements` -> `entail`
- `CITest` -> `Independence`
- `partial_correlation` -> `correlate`

## `scmle.data.admissibility.leakage`

- `SkillTest` -> `Skill`
- `decision_for` -> `decide`
- `ConditionalTest` -> `Conditional`
- `MissingnessTest` -> `Missingness`
- `FeatureAssessment` -> `Assessment`
- `assess_feature` -> `assess`
- `skill_of` -> `predictiveness`
- `skill_test` -> `probe`
- `cross_fit_skill` -> `crossfit`
- `within_strata` -> `scramble`
- `bootstrap_skill` -> `rescore`
- `conditional_test` -> `condition`
- `make_splits` -> `carve`
- `permutation_p` -> `significance`
- `binned_target` -> `bucket`
- `binary_auc` -> `auc`
- `enough_rows` -> `sufficient`

## `scmle.data.admissibility.provenance`

- `is_known` -> `given`
- `WrittenAt` -> `Stamp`
- `valid_on` -> `effective`
- `AsOfKind` -> `Cutoff`
- `AsOfSource` -> `Provenance`
- `AsOf` -> `Lead`
- `LogCheck` -> `Reconstruction`

## `scmle.data.admissibility.qualify`

- `causal_admissibility` -> `admit`
- `LoadedStore` -> `Shelf`
- `finding_check` -> `screened`
- `attestation_check` -> `waiver`

## `scmle.data.admissibility.reconciliation`

- `reconcile_item` -> `settle`

## `scmle.data.admissibility.rules`

- `timing_failure` -> `diagnose`
- `gate_of` -> `gate`
- `FieldDecision` -> `Decision`
- `decide_timing` -> `clock`
- `decide_predictive` -> `forecastable`
- `decide_causal` -> `adjudicate`

## `scmle.data.admissibility.screen`

- `bootstrap_v` -> `resample`
- `empty_report` -> `abstain`
- `build_problem` -> `encode`
- `limit_report` -> `halt`
- `finding_bound` -> `floor`
- `auc_rows` -> `aucs`
- `missingness_statistic` -> `cramer`
- `work_cap` -> `allowance`
- `strata_of` -> `stratify`
- `batch_permutations` -> `permute`
- `observed_predictions` -> `predict`
- `purity_bound` -> `dominance`
- `ceilings_of` -> `caps`
- `fold_splits` -> `crossfit`
- `missingness_hypotheses` -> `hypothesize`
- `required_permutations` -> `resamples`
- `sample_skill` -> `replicate`
- `scan_frame` -> `scan`
- `chi_square_rows` -> `chisquare`
- `fold_concordance` -> `concordance`
- `shipped_store` -> `bundled`

## `scmle.data.catalog`

- `frame_shape` -> `measure`
- `canonical_scalar` -> `normalise`
- `frame_for_array` -> `tabulate`
- `normalize_uri` -> `normalize`
- `schema_for_frame` -> `describe`
- `MissingnessSummary` -> `Missingness`
- `missingness_for_frame` -> `tally`
- `DuplicateSummary` -> `Duplication`
- `frame_for_data` -> `coerce`
- `duplicates_for_frame` -> `dedupe`
- `content_dict` -> `contents`
- `LabelSummary` -> `Labeling`
- `frame_for_other` -> `fallback`
- `label_for_frame` -> `summarize`
- `is_sequence_value` -> `sequential`
- `canonical_value` -> `canonicalize`
- `observe_data` -> `examine`

## `scmle.data.charter.causal`

- `check_assumption` -> `vet`
- `check_attestors` -> `reconcile`
- `covariates_of` -> `adjusters`
- `NegativeControl` -> `Control`

## `scmle.data.charter.contracts`

- `check_decision_map` -> `cover`
- `IntercurrentEvent` -> `Event`
- `read_charter` -> `peek`
- `DecisionRow` -> `Trigger`
- `charter_from_record` -> `parse`

## `scmle.data.charter.decision`

- `positive_number` -> `check`
- `DecisionLayer` -> `Layer`
- `DecisionModel` -> `Policy`
- `FairnessConstraint` -> `Constraint`
- `RolloutPlan` -> `Experiment`

## `scmle.data.charter.feasibility`

- `power_mean` -> `detectability`
- `check_k` -> `countable`
- `mde_correlation` -> `resolution`
- `wilson_half_width` -> `halfwidth`

## `scmle.data.charter.ingest`

- `EntityField` -> `Attribute`
- `EntityPass` -> `Pass`
- `ColumnTotal` -> `Total`
- `non_negative` -> `nonnegative`

## `scmle.data.charter.sealing`

- `check_amendment` -> `permit`
- `VerifiedCharter` -> `Verified`
- `write_charter` -> `inscribe`

## `scmle.data.curation.embedding`

- `TextEmbedder` -> `Embedder`

## `scmle.data.curation.preprocessing`

- `split_flagged` -> `partition`
- `split_high_cardinality` -> `isolate`
- `with_weights` -> `weigh`
- `build_front_end` -> `frontend`
- `preprocessing_steps` -> `preprocess`
- `wrap_estimator` -> `wrap`
- `monotonic_signs` -> `monotonicity`
- `build_pipeline` -> `instantiate`
- `monotonic_layout_problem` -> `misalignment`

## `scmle.data.curation.selection`

- `FeatureConstructor` -> `Constructor`

## `scmle.data.curation.transformers`

- `TextVectorizer` -> `Vectorizer`

## `scmle.data.custody.carve`

- `restrict_folds` -> `restrict`
- `unit_labels` -> `unitize`
- `pick_units` -> `pick`
- `check_inputs` -> `validate`
- `pick_blocks` -> `choose`
- `carve_confirmation` -> `divide`
- `close_over_units` -> `enclose`
- `derived_seed` -> `derive`
- `embargo_mask` -> `quarantine`
- `acf_embargo` -> `acf`
- `link_equal_codes` -> `link`

## `scmle.data.custody.manifest`

- `load_manifest` -> `restore`
- `set_ids` -> `rowids`

## `scmle.data.custody.merkle`

- `row_hashes` -> `fingerprints`
- `frame_root` -> `merkleize`

## `scmle.data.custody.pipeline`

- `empty_carve` -> `forgo`
- `apply_custody` -> `sequester`
- `minority_rows` -> `scarce`
- `classes_complete` -> `complete`
- `measured_embargo` -> `gap`

## `scmle.data.custody.sealed`

- `ExplorationView` -> `View`
- `entry_hash` -> `hashed`
- `entry_mac` -> `authenticate`
- `OpenLog` -> `Journal`
- `SealedSet` -> `Locker`

## `scmle.data.detectors.certificates`

- `DefectOutcome` -> `Outcome`
- `StaleCertificateError` -> `StaleError`
- `GateStatus` -> `Authority`
- `mark_gating` -> `authorise`
- `docstring_lines` -> `documentation`

## `scmle.data.detectors.duplicates`

- `assess_duplicates` -> `assess`
- `exact_flags` -> `verbatim`

## `scmle.data.detectors.labels`

- `NoiseRate` -> `Rate`
- `mean_margins` -> `integrate`
- `noise_rate` -> `noisiness`
- `class_thresholds` -> `cutoffs`
- `hoeffding_bound` -> `hoeffding`
- `confident_assignment` -> `assign`
- `confident_learning_flags` -> `mislabeled`
- `margin_flags` -> `margins`
- `ppi_mean` -> `ppi`

## `scmle.data.detectors.missingness`

- `row_patterns` -> `group`
- `expect_block` -> `expect`
- `little_statistic` -> `measure`
- `oof_linear_scores` -> `crossfit`
- `TippingPoint` -> `Sensitivity`
- `auc_from_ranks` -> `ranksum`
- `imputation_errors` -> `benchmark`
- `numeric_columns` -> `numerics`
- `assess_mechanism` -> `investigate`
- `em_normal` -> `em`
- `fold_imputers` -> `imputers`

## `scmle.data.detectors.outliers`

- `error_flags` -> `misrecorded`

## `scmle.data.detectors.sentinels`

- `SentinelFinding` -> `Suspect`
- `numeric_spikes` -> `spot`
- `text_findings` -> `scour`
- `sentinel_mask` -> `coded`
- `recode_missing` -> `nullify`
- `sentinel_rows` -> `placeholders`
- `grid_spacing` -> `pitch`
- `expected_share` -> `expect`
- `find_sentinels` -> `hunt`

## `scmle.data.framing.baseline`

- `profile_on_exploration` -> `gauge`
- `rule_digest` -> `fingerprint`
- `ClusterTest` -> `Clustered`
- `paired_difference` -> `advantage`
- `cluster_test` -> `ttest`

## `scmle.data.framing.claim`

- `freeze_rule` -> `solidify`

## `scmle.data.framing.costs`

- `CostSource` -> `Basis`
- `unverified_entries` -> `unreviewed`
- `DominatedActionError` -> `DominatedError`
- `DecisionCostModel` -> `Tariff`
- `net_benefit_curve` -> `benefits`
- `binary_vector` -> `dichotomous`

## `scmle.data.framing.draft`

- `column_refusal` -> `mismatch`
- `consult_sampler` -> `consult`
- `SourceDescription` -> `Offer`
- `baseline_and_questions` -> `interview`
- `DataCheck` -> `Probe`
- `write_artifacts` -> `emit`
- `deterministic_slots` -> `prefill`
- `link_report` -> `linkage`
- `check_draft` -> `vet`
- `draft_hash` -> `sha`

## `scmle.data.framing.feasibility`

- `BaselineFeasibility` -> `Verdict`

## `scmle.data.framing.provenance`

- `FieldSource` -> `Filling`

## `scmle.data.framing.questions`

- `HumanQuestion` -> `Question`
- `gain_sign_verdict` -> `judge`

## `scmle.data.framing.refusal`

- `population_ids` -> `members`
- `clopper_pearson_upper` -> `ceiling`
- `SlotSampler` -> `Sampler`
- `prompt_for` -> `instruct`
- `ReplaySampler` -> `Replayer`

## `scmle.data.identification.aipw`

- `overlap_report` -> `propensities`
- `make_learners` -> `spawn`
- `fit_nuisances` -> `prefit`
- `weighted_smd` -> `imbalance`

## `scmle.data.identification.battery`

- `wald_test` -> `wald`

## `scmle.data.identification.certified`

- `subgroup_method_hash` -> `fingerprint`
- `certified_gates` -> `certificated`

## `scmle.data.identification.did`

- `pretrend_p_value` -> `pretest`
- `max_pre_event_change` -> `pretrend`
- `wide_panel` -> `widen`
- `event_arms` -> `arms`
- `event_covariance` -> `vcov`
- `overall_att` -> `aggregate`

## `scmle.data.identification.iv`

- `exclusion_bound` -> `exclude`

## `scmle.data.identification.overid`

- `two_stage` -> `estimate`

## `scmle.data.identification.rd`

- `cv_bandwidth` -> `crossvalidate`
- `chosen_bandwidth` -> `choose`
- `local_fit` -> `polyfit`

## `scmle.data.identification.run`

- `IvEstimator` -> `Instrument`
- `balance_columns` -> `pretreatment`
- `sensitivity_of` -> `fragility`
- `wald_from` -> `waldify`
- `SynthEstimator` -> `Synth`
- `closing_notes` -> `caveats`
- `BackdoorEstimator` -> `Backdoor`
- `refusal_outcome` -> `refuse`
- `subgroup_report` -> `stratify`
- `needed_assumptions` -> `prerequisites`
- `DidEstimator` -> `DiD`
- `label_subgroups` -> `certify`

## `scmle.data.identification.select`

- `check_adjustment` -> `verify`

## `scmle.data.identification.sensitivity`

- `OlsFit` -> `Regression`
- `robustness_value` -> `rvalue`
- `bias_bound` -> `ceiling`
- `evalue_interval` -> `evalues`
- `adjusted_interval` -> `robustify`
- `benchmark_strength` -> `calibrate`
- `partial_r2` -> `rsquared`
- `ols_fit` -> `regress`

## `scmle.data.identification.synth`

- `min_donors` -> `quorum`
- `fit_weights` -> `weigh`

## `scmle.data.identification.tier`

- `tier_conditions` -> `assess`
- `design_shortfalls` -> `shortfalls`

## `scmle.data.ingest.asof_view`

- `KnownTime` -> `Timing`
- `AsOfView` -> `View`
- `NoAsOfRuleError` -> `HistoryError`

## `scmle.data.ingest.budget`

- `StopBoundary` -> `Stop`

## `scmle.data.ingest.declared`

- `source_totals` -> `compare`
- `key_checksum` -> `sha`

## `scmle.data.ingest.entities`

- `fit_em` -> `em`
- `measure_quality` -> `grade`
- `BandEstimate` -> `Band`
- `geometric_bands` -> `stratify`
- `jaro_winkler` -> `jarowinkler`
- `lowest_weight` -> `weakest`
- `candidate_pairs` -> `pairing`
- `thresholds_from_bands` -> `cutoffs`

## `scmle.data.ingest.grain`

- `bound_ok` -> `feasible`
- `match_columns` -> `align`
- `name_tokens` -> `tokenize`
- `scan_single_columns` -> `sweep`
- `subset_verdict` -> `shrink`
- `unit_components` -> `grains`
- `level_candidates` -> `layer`
- `KeyCandidate` -> `Candidate`
- `key_like` -> `identifies`
- `measure_level` -> `levels`
- `KeyScan` -> `Scan`
- `infer_keys` -> `keysearch`

## `scmle.data.ingest.joins`

- `distinct_of` -> `count`
- `kinds_match` -> `match`
- `measure_link` -> `measure`
- `resolve_competition` -> `arbitrate`
- `name_supports` -> `corroborated`
- `single_column_pairs` -> `couple`
- `DeclaredKey` -> `Reference`
- `JoinGraph` -> `Joins`

## `scmle.data.ingest.plan`

- `write_artifact` -> `persist`
- `infer_grain` -> `infer`
- `discover_joins` -> `link`
- `sqlite_factory` -> `opener`
- `declared_from_json` -> `declarations`

## `scmle.data.ingest.profile`

- `ConsistentRead` -> `Reading`
- `SqlSource` -> `Sql`
- `TornReadError` -> `TornError`
- `value_kind` -> `classify`
- `has_total_order` -> `deterministic`
- `ColumnProfile` -> `Aggregates`
- `read_once` -> `slurp`
- `TableProfile` -> `Census`
- `consistent_read` -> `untorn`
- `profile_table` -> `survey`
- `quote_all` -> `quoting`

## `scmle.data.ingest.reconcile`

- `ExtractReconciliation` -> `Report`
- `reconcile_extract` -> `crosscheck`
- `JoinCheck` -> `Cardinality`
- `FeatureTotal` -> `Sum`

## `scmle.data.ingest.synthesize`

- `candidate_columns` -> `sift`
- `reduce_numeric` -> `aggregate`
- `FeatureSet` -> `Selection`
- `window_label` -> `caption`
- `feature_provenance` -> `lineage`
- `aggregate_features` -> `roll`
- `register_family` -> `enlist`
- `target_frame` -> `outcomes`
- `reduce_rows` -> `fold`

## `scmle.data.ledger.bivariate`

- `complete_pairs` -> `prune`

## `scmle.data.ledger.budget`

- `discount_weight` -> `gammas`
- `DetectorTolerance` -> `Tolerance`
- `assign_experiment` -> `enroll`
- `BudgetExhaustedError` -> `ExhaustedError`
- `split_budget` -> `apportion`

## `scmle.data.ledger.confirm`

- `EventLog` -> `Log`
- `unverified_claims` -> `disclaim`
- `withdrawn_claims` -> `withdraw`
- `run_confirmation` -> `verify`
- `open_and_test` -> `unseal`
- `family_block` -> `block`
- `failure_reason` -> `phrase`
- `ClaimRun` -> `Output`
- `head_of` -> `tip`
- `ClaimRunner` -> `Docket`
- `merge_runners` -> `merge`
- `ConfirmationSource` -> `Source`
- `RegisteredClaim` -> `Commitment`
- `log_selection` -> `preregister`

## `scmle.data.ledger.controls`

- `shuffle_columns` -> `scramble`
- `shuffle_blocks` -> `jumble`
- `percentile_interval` -> `bracket`
- `permuted_target_control` -> `placebo`

## `scmle.data.ledger.explore`

- `plan_exploration` -> `hypothesize`
- `count_discoveries` -> `tally`

## `scmle.data.ledger.governed`

- `GovernedRun` -> `Governance`
- `explore_governed` -> `scout`
- `confirm_explored` -> `finalise`

## `scmle.data.ledger.ledger`

- `SurvivorSet` -> `Selection`
- `survivors_hash` -> `fingerprint`
- `record_not_tested` -> `abstain`
- `commit_result` -> `settle`
- `apply_confirmation` -> `conclude`
- `check_calibration` -> `audit`
- `register_claim` -> `enroll`
- `apply_control` -> `benchmark`
- `require_registered` -> `pending`
- `apply_diagnostic` -> `diagnose`
- `record_control` -> `negatives`
- `require_rule` -> `tester`
- `require_unfrozen` -> `guard`
- `apply_claim` -> `lodge`
- `apply_register` -> `enrol`
- `apply_survivors` -> `survive`

## `scmle.data.ledger.multiplicity`

- `step_up` -> `monotone`
- `harmonic_exact` -> `harmonics`

## `scmle.data.ledger.online_fdr`

- `RuleInvariantError` -> `InvariantError`
- `require_p` -> `read`
- `RuleConfig` -> `Settings`
- `exact_alpha` -> `rational`
- `ELond` -> `Lond`
- `LordPlusPlus` -> `Lord`

## `scmle.data.modalities.audio`

- `wav_metadata` -> `parse`
- `wav_signal_features` -> `extract`
- `AudioValidationError` -> `ValidationError`
- `AudioDependencyError` -> `DecoderError`
- `TranscriptSegment` -> `Segment`
- `wav_pcm_values` -> `pcm`

## `scmle.data.modalities.curation`

- `check_governance` -> `govern`
- `build_finding` -> `flaw`
- `check_card_text` -> `proofread`
- `normalized_curation` -> `roundtrip`
- `check_split_fractions` -> `proportions`
- `partition_group_ids` -> `allocate`

## `scmle.data.modalities.document`

- `collect_quality_findings` -> `inspect`
- `collect_risk_findings` -> `assess`
- `format_for` -> `sniff`
- `read_pdf` -> `extract`
- `default_document_id` -> `fingerprint`
- `language_heuristic` -> `detect`
- `provenance_for_content` -> `segment`
- `document_finding` -> `flaw`
- `DocumentValidationError` -> `DocumentError`
- `DocumentDependencyError` -> `DependencyError`
- `sensitive_summary` -> `sensitivity`
- `sensitive_matches` -> `pii`
- `decode_utf8` -> `unicode`
- `label_key` -> `bucketize`
- `read_layout` -> `overlay`

## `scmle.data.modalities.geospatial`

- `check_digest` -> `verify`
- `validate_coordinates` -> `flatten`
- `geometry_coordinates` -> `unpack`
- `moran_statistic` -> `autocorrelate`
- `masked_coordinates` -> `overwrite`
- `masked_geometry` -> `mask`
- `SpatialJoin` -> `Join`
- `SpatialBoundaryLeakage` -> `Leakage`
- `SpatialSplit` -> `Partition`
- `SpatialAutocorrelation` -> `Autocorrelation`
- `spatial_neighbors` -> `connect`
- `check_join_request` -> `vet`
- `moran_values` -> `collect`
- `spatial_block_envelopes` -> `envelop`
- `block_for` -> `locate`
- `check_nodata` -> `normalize`
- `GeospatialValidationError` -> `GeometryError`
- `RasterSample` -> `Tile`
- `GeospatialTemporalError` -> `TemporalError`
- `boundary_leakage` -> `straddle`
- `mask_geometry_axis` -> `substitute`
- `masked_geometry_axis` -> `substituted`
- `split_spatial` -> `tile`
- `GeoFeature` -> `Placemark`
- `block_ids` -> `tiles`
- `SpatialJoinMatch` -> `Match`
- `pixel_window_bounds` -> `extent`
- `evaluate_predicate` -> `probe`

## `scmle.data.modalities.graph`

- `GraphTemporalError` -> `TemporalError`
- `canonical_event_time` -> `normalize`
- `PreparedFeatures` -> `Features`
- `edge_key` -> `canonicalize`
- `edge_key_counts` -> `tally`
- `GraphInputError` -> `ParseError`
- `read_edges` -> `edgelist`
- `GraphValidationError` -> `ValidationError`
- `edges_from_records` -> `link`
- `node_partitions` -> `divisions`
- `profile_nodes` -> `census`
- `NodeProfile` -> `Degrees`
- `inductive_node_split` -> `induct`
- `from_edges` -> `connect`
- `EdgeProfile` -> `Connectivity`
- `structural_degrees` -> `valence`
- `InductiveNodeSplit` -> `Partition`
- `feature_values` -> `featurize`
- `require_valid` -> `ensure`

## `scmle.data.modalities.image`

- `perceptual_hash` -> `fingerprint`
- `validate_limits` -> `validate`
- `check_dimensions` -> `bounds`
- `neighbor_components` -> `islands`
- `prepare_tabular_features` -> `flatten`
- `validate_feature_options` -> `configure`
- `ImageDependencyError` -> `DependencyError`
- `image_content_hash` -> `imprint`
- `require_pillow` -> `pillow`

## `scmle.data.modalities.video`

- `prepare_frame_features` -> `summarize`
- `frame_indices_for` -> `stride`
- `VideoValidationError` -> `VideoError`
- `synchronization_status` -> `synchrony`
- `prepare_metadata_features` -> `numerize`

## `scmle.data.profiling.label_noise`

- `LabelNoise` -> `Noise`
- `find_label_noise` -> `contradict`
- `assess_label_noise` -> `inspect`

## `scmle.data.profiling.missingness`

- `MissingnessFinding` -> `Finding`
- `class_association` -> `chisquare`
- `value_association` -> `mannwhitney`

## `scmle.data.profiling.profiler`

- `summarize_target` -> `summarize`
- `role_columns` -> `exclude`
- `inspect_columns` -> `inspect`
- `datetime_nodes` -> `derive`
- `describe_associations` -> `describe`
- `column_exclusion_reason` -> `disqualify`
- `balance_findings` -> `balance`
- `duplicate_findings` -> `dupes`

## `scmle.data.profiling.representativeness`

- `ColumnGap` -> `Gap`
- `column_gap` -> `contrast`
- `compare_to_reference` -> `distinguish`

## `scmle.data.readers.connectors`

- `ObjectPayload` -> `Payload`
- `bind_content` -> `seal`
- `retrieve_schema` -> `introspect`
- `retrieve_token` -> `fingerprint`
- `retrieve_rows` -> `fetch`
- `schema_from_description` -> `translate`
- `skip_rows` -> `skip`
- `close_resources` -> `release`
- `validate_schema` -> `vet`
- `load_frame` -> `load`
- `decode_object` -> `deserialize`
- `object_size` -> `measure`
- `read_record_lines` -> `scan`
- `decode_records` -> `interpret`
- `json_type_name` -> `classify`
- `LocalFileConnector` -> `Filesystem`
- `RowBatch` -> `Page`
- `canonical_source_uri` -> `canonicalize`
- `cursor_for` -> `bookmark`
- `execute_with_retries` -> `persevere`
- `decode_cursor` -> `unwrap`
- `refuse_writes` -> `readonly`
- `WarehouseConnector` -> `Lakehouse`
- `SourceCheckpoint` -> `Bookmark`
- `digest_rows` -> `feed`
- `read_tabular_bytes` -> `unpack`
- `UnsupportedUriSchemeError` -> `SchemeError`
- `SourceColumn` -> `Field`
- `SourceConnector` -> `Adapter`
- `ObjectStoreConnector` -> `Bucket`
- `RetryBudget` -> `Retries`
- `SourceSchema` -> `Columns`
- `file_content_sha256` -> `checksum`

## `scmle.data.readers.incremental`

- `EventSource` -> `Source`
- `EventStreamSource` -> `Stream`
- `LateData` -> `Lateness`
- `EventCheckpoint` -> `Checkpoint`
- `IngestionLedger` -> `Ledger`
- `EventBatch` -> `Batch`
- `EventConsumer` -> `Ingestor`
- `EventCommitReceipt` -> `Ack`

## `scmle.data.readers.interchange`

- `is_tabular` -> `tablelike`

## `scmle.data.readers.registry`

- `SchemaCompatibility` -> `Compatibility`
- `SourceSnapshot` -> `Observation`
- `SourceRegistry` -> `Sources`
- `SourceRegistryStore` -> `Registry`
- `from_manifest` -> `restore`

## `scmle.data.readers.streaming`

- `watermark_value` -> `advance`
- `CommitReceipt` -> `Receipt`
- `JsonLinesChunks` -> `Ndjson`
- `keep_smallest` -> `retain`
- `ChunkSource` -> `Feed`
- `CsvChunks` -> `Csv`
- `SinkCommitReceipt` -> `Voucher`
- `source_for` -> `locate`
- `validate_batch` -> `validate`
- `TransactionalSink` -> `Sink`
- `validate_acknowledgement` -> `vouch`
- `CommitEvidence` -> `Commitment`
- `sample_caps` -> `budgets`
- `sample_pass` -> `sweep`
- `ParquetChunks` -> `Chunked`
- `estimated_bytes` -> `footprint`

## `scmle.data.readers.tabular`

- `CsvReader` -> `Csv`
- `JsonReader` -> `Json`
- `JsonLinesReader` -> `Jsonl`
- `ExcelReader` -> `Excel`
- `ParquetReader` -> `Parquet`
- `FeatherReader` -> `Feather`
- `TsvReader` -> `Tsv`
- `reader_for` -> `pick`
- `uri_semantics` -> `semantic`
- `DatasetLoader` -> `Loader`
- `snapshot_identity` -> `identify`
- `build_manifest` -> `manifested`
- `replace_infinities` -> `sanitize`
- `canonical_uri` -> `canonicalize`
- `supported_suffixes` -> `extensions`

## `scmle.data.robustness.coverage`

- `LevelCoverage` -> `Cell`
- `subgroup_coverage` -> `represent`

## `scmle.data.robustness.forward`

- `ForwardFinding` -> `Finding`
- `evaluate_finding` -> `evaluate`
- `ForwardCheck` -> `Forward`
- `label_for` -> `verdict`

## `scmle.data.robustness.html_report`

- `stability_section` -> `section`

## `scmle.data.robustness.multiverse`

- `GridRun` -> `Sweep`
- `tier_after_declaration` -> `downgrade`
- `enumerate_grid` -> `universe`
- `ResultCache` -> `Memo`

## `scmle.data.robustness.prefilter`

- `feature_checks` -> `shifts`
- `diagnostic_rows` -> `gather`
- `StabilityStage` -> `Prefilter`

## `scmle.data.robustness.report`

- `validate_stability` -> `conform`

## `scmle.data.robustness.shift`

- `check_pair` -> `align`
- `FeatureShift` -> `Drift`
- `categorical_test` -> `contrast`
- `median_bandwidth` -> `sigma`
- `gaussian_kernel` -> `gaussian`
- `auc_delong` -> `delong`
- `mmd_statistics` -> `mmds`
- `pooled_codes` -> `pool`
- `holdout_split` -> `holdout`
- `feature_tests` -> `compare`

## `scmle.data.robustness.stability`

- `estimate_spec` -> `estimate`
- `CurveTest` -> `Curve`
- `curve_statistics` -> `summarize`
- `rank_effect` -> `rankcorr`
- `SpecEffect` -> `Variant`
- `curve_test` -> `specify`
- `applicable_choices` -> `actionable`

## `scmle.data.splitting.duplicates`

- `fold_of_held_rows` -> `locate`
- `keep_together` -> `cohere`

## `scmle.data.splitting.inner_cv`

- `StructuredKFold` -> `Splitter`

## `scmle.data.splitting.splitters`

- `to_ids` -> `tuplify`
- `require_rows_per_class` -> `require`
- `MultilabelStratifiedSplit` -> `Multilabel`
- `label_matrix` -> `stack`
- `PresetSplit` -> `Preset`
- `keep_duplicates_together` -> `gather`
- `LockedTestSet` -> `Lockbox`
- `resolve_strategy_name` -> `scheme`

## `scmle.data.store.conformance`

- `check_head_matches_get` -> `headcheck`
- `check_pagination` -> `paginate`

## `scmle.data.store.objectstore`

- `check_key` -> `validate`
- `meta_of` -> `describe`
- `create_exclusive` -> `claim`
- `NotFoundError` -> `AbsentError`
- `is_temporary` -> `partial`
- `AlreadyExistsError` -> `ExistsError`
- `refused_as_existing` -> `occupied`
- `unlink_tolerantly` -> `discard`
- `path_of` -> `locate`
- `sha256_of` -> `checksum`

## `scmle.data.store.processing`

- `run_locked` -> `sweep`
- `require_match` -> `matches`
- `CheckpointCorruptError` -> `CorruptError`
- `ChangedInputError` -> `ChangedError`
- `ConcurrentRunError` -> `LockError`

## `scmle.data.store.quality`

- `dataset_finding` -> `judge`
- `row_finding` -> `pinpoint`
- `marker_key` -> `locate`
- `NoNullsCheck` -> `Complete`
- `AllowedValuesCheck` -> `Membership`
- `FreshnessCheck` -> `Freshness`

## `scmle.data.store.snapshot`

- `observed_schema` -> `dtypes`

## `scmle.data.validation.columns`

- `usable_rows` -> `align`
- `encoded_with` -> `encode`
- `NumericKind` -> `Quantitative`
- `to_finite` -> `scrub`
- `to_text` -> `stringify`
- `CategoricalKind` -> `Categorical`
- `purity_association` -> `determinacy`
- `threshold_association` -> `stump`
- `ColumnKind` -> `Family`

## `scmle.data.validation.target`

- `is_numeric_target` -> `quantitative`

## `scmle.decision.anytime`

- `rescaled_null` -> `orient`
- `bet_grid` -> `bets`
- `running_center` -> `centers`
- `log_wealth` -> `grow`
- `centered_stream` -> `deviations`
- `centered_paths` -> `recentre`
- `EProcess` -> `Martingale`
- `difference_stream` -> `contrasts`
- `positive_level` -> `fractional`
- `bound_stream` -> `clamped`

## `scmle.decision.constraints`

- `SatisfactionTest` -> `Satisfaction`
- `largest_gap` -> `spread`
- `GroupThresholds` -> `Thresholds`
- `InfeasibleConstraintsError` -> `InfeasibleError`
- `rate_tables` -> `tabulate`
- `fit_group_thresholds` -> `equalize`
- `satisfaction_test` -> `satisfy`
- `price_of_constraint` -> `toll`

## `scmle.decision.design`

- `srm_rejects` -> `srm`
- `simulate_stopping` -> `simulate`
- `srm_p_value` -> `mismatch`

## `scmle.decision.netbenefit`

- `NetBenefit` -> `Benefit`
- `net_benefit_band` -> `lowerband`
- `SecondEvaluationError` -> `RepeatError`
- `resample_means` -> `redraw`
- `useful_span` -> `stretch`

## `scmle.decision.ope`

- `ValueEstimate` -> `Estimate`
- `EffectThreshold` -> `Cutoff`
- `difference_value` -> `contrast`
- `fit_outcome_model` -> `regress`
- `LoggedBandit` -> `Bandits`
- `fit_threshold_policy` -> `tune`
- `support_violations` -> `unsupported`
- `mean_interval` -> `wald`
- `estimate_value` -> `evaluate`

## `scmle.decision.rollout`

- `DecisionLog` -> `Journal`
- `RollbackMonitor` -> `Watchdog`

## `scmle.decision.threshold`

- `CalibrationCheck` -> `Calibration`
- `ThresholdDecision` -> `Thresholding`
- `bayes_threshold` -> `breakeven`
- `probabilities_and_labels` -> `pair`
- `stable_span` -> `span`
- `calibration_check` -> `hosmer`
- `choose_threshold` -> `operate`
- `fit_recalibrator` -> `calibrator`
- `raw_threshold` -> `uncalibrated`
- `cross_fitted` -> `crossfit`

## `scmle.deploy.attest`

- `verify_document` -> `authenticate`
- `verify_image` -> `vouch`
- `read_document` -> `parse`
- `write_provenance` -> `stamp`
- `material_problems` -> `discrepancies`
- `sign_statement` -> `envelope`
- `write_image_attestation` -> `bind`

## `scmle.deploy.base_image`

- `fetch_registry_digest` -> `query`
- `split_image` -> `decompose`
- `read_lock` -> `peruse`

## `scmle.deploy.contract`

- `secret_problems` -> `mounts`
- `isolation_problems` -> `diff`
- `check_config_entries` -> `nonsecret`
- `require_name` -> `simple`
- `check_production_rules` -> `harden`
- `identity_problems` -> `divergences`

## `scmle.deploy.demo`

- `start_service` -> `launch`
- `install_bearer` -> `authenticate`
- `stop_service` -> `halt`
- `predict_checks` -> `compare`
- `corrupted_and_unsigned_stay_unready` -> `tamper`
- `rollback_drill` -> `rollback`
- `drill_steps` -> `drill`
- `measure_latency` -> `measure`
- `push_metrics` -> `push`
- `round_trip_object_store` -> `mirror`
- `check_golden_parity` -> `corroborate`
- `verify_remote` -> `verify`
- `approval_for` -> `sanction`
- `download_folder` -> `download`
- `get_json` -> `retrieve`
- `records_of` -> `serialize`
- `process_memory_mb` -> `footprint`
- `fail_not_ready` -> `unready`
- `wait_ready` -> `settle`
- `free_port` -> `unused`
- `measure_and_ship_metrics` -> `instrument`
- `upload_folder` -> `sync`

## `scmle.deploy.gate`

- `model_problems` -> `veto`
- `staging_problems` -> `audit`
- `admit_problems` -> `objections`
- `runtime_problems` -> `absences`

## `scmle.deploy.livetests`

- `check_health` -> `probe`
- `check_schema` -> `inspect`
- `check_security` -> `breach`
- `check_inference` -> `infer`
- `NoRedirect` -> `Stopper`

## `scmle.deploy.lockfile`

- `runtime_closure` -> `transitive`
- `marker_applies` -> `applies`

## `scmle.deploy.mocks`

- `server_file` -> `append`
- `put_object` -> `deposit`
- `safe_key` -> `sanitize`
- `MockServer` -> `Stub`
- `post_metrics` -> `transmit`
- `ObjectStoreResponder` -> `Handler`
- `MetricsSinkResponder` -> `Sink`

## `scmle.deploy.observation`

- `share_of` -> `portion`

## `scmle.deploy.release`

- `approval_from_dict` -> `revive`
- `scan_result` -> `triage`
- `load_candidate` -> `gather`

## `scmle.deploy.runtime`

- `signal_settings` -> `cadence`
- `emit_signals` -> `emit`
- `signal_loop` -> `pulse`
- `make_loader` -> `prime`

## `scmle.deploy.verify`

- `verify_service` -> `inspect`

## `scmle.deploy.vuln_gate`

- `active_entries` -> `prune`
- `parse_trivy` -> `parse`
- `parse_pip_audit` -> `advisories`

## `scmle.durability`

- `replace_file` -> `swap`
- `path_lock` -> `latch`
- `sync_file` -> `sync`

## `scmle.evaluation.backtest`

- `per_step_errors` -> `score`

## `scmle.evaluation.balance`

- `balance_test` -> `randomisation`

## `scmle.evaluation.concordance`

- `concordance_index` -> `harrell`
- `ConcordanceMetric` -> `Cindex`

## `scmle.evaluation.conformal`

- `finite_sample_quantile` -> `cutoff`
- `for_classification` -> `classify`
- `cv_plus_bounds` -> `cvplus`
- `thin_classes` -> `scarce`
- `one_row_per_entity` -> `representatives`

## `scmle.evaluation.coverage`

- `coverage_tier` -> `grade`
- `holm_rejections` -> `stepdown`
- `undercoverage_p_value` -> `deficit`
- `coverage_summary` -> `summarise`
- `slice_flag` -> `undercovered`
- `class_coverage` -> `classwise`

## `scmle.evaluation.coverage_confirm`

- `evaluate_document` -> `assess`
- `primary_choice` -> `principal`
- `registration_body` -> `prefreeze`
- `floor_p_value` -> `shortfall`
- `inside_of` -> `held`

## `scmle.evaluation.coverage_run`

- `regression_reports` -> `regressions`
- `classification_reports` -> `classwise`

## `scmle.evaluation.coverage_slices`

- `flag_method_hash` -> `fingerprint`
- `labels_for` -> `bucket`

## `scmle.evaluation.cv_plus`

- `ScorePlan` -> `Plan`
- `partition_folds` -> `holdouts`
- `plan_scores` -> `strategize`

## `scmle.evaluation.diagnostics`

- `MulticlassCalibration` -> `Reliability`
- `ClassifierReplay` -> `Replay`
- `ConfusionMatrix` -> `Contingency`
- `resample_rows` -> `redraw`
- `MetricUncertainty` -> `Uncertainty`
- `ThresholdSweep` -> `Sweep`
- `ErrorAnalysis` -> `Errors`
- `out_of_time` -> `expired`
- `resample_count` -> `replicates`

## `scmle.evaluation.dirichlet`

- `log_probabilities` -> `featurise`

## `scmle.evaluation.exchangeability`

- `released_scores` -> `conformity`

## `scmle.evaluation.fit_health`

- `FitHealth` -> `Health`
- `assess_trial` -> `sanity`

## `scmle.evaluation.forecast`

- `truth_matrix` -> `actuals`

## `scmle.evaluation.ipcw`

- `marginal_at` -> `survive`
- `cox_at` -> `survive`
- `CensoringWeights` -> `Censoring`
- `censoring_weights` -> `reweight`
- `truncated_concordance` -> `uno`

## `scmle.evaluation.learning_curve`

- `LearningCurve` -> `Curve`

## `scmle.evaluation.longitudinal`

- `prepare_longitudinal_frame` -> `prepare`
- `fit_logistic_propensity` -> `regress`
- `require_strictly_increasing` -> `monotone`
- `influence_covariance` -> `sandwich`
- `build_period_result` -> `periodise`
- `fit_period` -> `wave`

## `scmle.evaluation.metrics`

- `MetricCatalog` -> `Repertoire`

## `scmle.evaluation.mitigation`

- `group_cuts` -> `cutoffs`

## `scmle.evaluation.multitarget`

- `MultiTargetMetric` -> `Joint`

## `scmle.evaluation.ordinal`

- `mean_absolute_class_error` -> `distance`

## `scmle.evaluation.parameter_estimation`

- `EstimatorFit` -> `Fit`
- `check_estimation_request` -> `admit`
- `update_input_digest` -> `absorb`
- `prepare_rows` -> `prepare`
- `fit_linear` -> `regress`
- `center_influence` -> `center`
- `project_random_effect_covariance` -> `project`
- `bootstrap_replicate_estimate` -> `rerun`
- `check_choices` -> `vet`
- `check_hierarchical_policy` -> `nesting`
- `fit_aipw_ate` -> `dr`
- `fit_linear_estimator` -> `ols`
- `hajek_ipw_statistics` -> `hajek`
- `hierarchical_random_effect_precision` -> `precisions`
- `reml_profile_objective` -> `deviance`
- `validate_covariance` -> `condition`
- `check_bayesian_treatment` -> `dichotomous`
- `fit_estimator` -> `compute`
- `fit_continuous_ipw_effect` -> `dosage`
- `deterministic_fold_assignments` -> `interleave`
- `validate_bayesian_prior_shape` -> `conform`
- `fit_hierarchical_random_slope_reml` -> `optimise`
- `hierarchical_gls_system` -> `gls`
- `reference_hierarchical_linear` -> `crosscheck`
- `reference_logistic` -> `corroborate`
- `draw_hierarchical_bootstrap_sample` -> `clone`
- `check_reml_bounds` -> `fence`
- `bernoulli_log_likelihood` -> `loglik`
- `check_field_support` -> `relevant`
- `fit_bayesian_estimator` -> `bayes`
- `fit_logistic` -> `logit`
- `cluster_uncertainty_diagnostics` -> `clustering`
- `aipw_limitations` -> `caveats`
- `prior_sensitivity` -> `reprior`
- `check_bayesian_policy` -> `bayesian`
- `influence_variance` -> `dispersion`
- `shift_sensitivity` -> `transport`

## `scmle.evaluation.proper_scores`

- `insample_scale` -> `naivescale`
- `interval_score` -> `winkler`

## `scmle.evaluation.quantile`

- `predicted_band` -> `corridor`
- `QuantileBand` -> `Band`
- `coverage_curve` -> `covering`
- `raw_band` -> `native`

## `scmle.evaluation.ranking`

- `query_groups` -> `group`
- `ranked_hits` -> `sort`
- `recall_at` -> `recall`
- `per_query_ndcg` -> `ndcgs`
- `query_bootstrap_ndcg` -> `cumulate`
- `precision_at` -> `topk`

## `scmle.evaluation.release`

- `ReleaseStatus` -> `Readiness`
- `apply_reliability_gate` -> `faults`
- `check_kind_coverage` -> `exhaustive`
- `diagnostic_gate_finding` -> `adjudge`
- `apply_attestation_gate` -> `certify`
- `normalize_measure` -> `quantify`
- `ReleaseEvaluation` -> `Verdict`
- `apply_evidence_index_gate` -> `countersign`
- `fairness_gate_finding` -> `audit`
- `ReleaseFinding` -> `Finding`
- `blocked_input_finding` -> `unreadable`
- `evidence_reference_finding` -> `referenced`
- `required_evidence_references` -> `citations`
- `FindingCounts` -> `Tally`
- `fairness_metric_values` -> `disparities`
- `metric_gate_finding` -> `judge`
- `evaluate_release` -> `assess`
- `FindingKind` -> `Kind`
- `check_count_values` -> `tallies`
- `diagnostic_verdict` -> `ruling`
- `explanation_quality_value` -> `grade`
- `readiness_summary` -> `tally`

## `scmle.evaluation.reliability`

- `top_label_curve` -> `topclass`

## `scmle.evaluation.replay`

- `PredictionReplay` -> `Replay`
- `RegressorPredictions` -> `Replayed`

## `scmle.evaluation.resources`

- `ResourceUsage` -> `Usage`
- `ResourceLimits` -> `Constraints`

## `scmle.evaluation.robustness`

- `GaussianNoise` -> `Noise`
- `MissingValues` -> `Blanking`

## `scmle.evaluation.selection`

- `tie_key` -> `tiebreak`
- `MeanSelector` -> `Averager`
- `probably_equivalent` -> `equivalent`
- `by_score` -> `merit`

## `scmle.evaluation.sensitivity`

- `spread_of` -> `summarise`
- `evaluation_sensitivity` -> `resplit`
- `performance_sensitivity` -> `jitter`
- `mean_score` -> `collapse`
- `training_sensitivity` -> `variability`

## `scmle.evaluation.sequential`

- `prepare_sequential_frame` -> `prepare`
- `ordered_periods` -> `order`
- `build_matrices` -> `tabulate`
- `fit_propensity` -> `regress`
- `build_regime_estimate` -> `average`
- `check_propensity_support` -> `overlap`
- `regime_covariance` -> `covary`
- `check_contrast` -> `scrutinise`
- `require_distinct_neighbours` -> `nonrepeating`

## `scmle.evaluation.shift`

- `AdversarialValidation` -> `Adversary`

## `scmle.evaluation.shift_weights`

- `WeightRefusedError` -> `RefusedError`
- `ShiftWeights` -> `Ratios`

## `scmle.evaluation.slices`

- `slice_score` -> `grade`

## `scmle.evaluation.suite`

- `full_suite` -> `lineup`

## `scmle.evaluation.survival`

- `integrated_brier` -> `brier`
- `dynamic_auc` -> `auc`
- `censoring_curve` -> `censor`
- `SurvivalMetric` -> `Survival`
- `censoring_probability` -> `survivor`
- `time_grid` -> `horizons`

## `scmle.evaluation.uplift`

- `uplift_at` -> `lift`
- `check_propensity` -> `randomised`
- `UpliftMetric` -> `Incrementality`

## `scmle.experimentation.ensemble_search`

- `EnsembleSearch` -> `Stacker`

## `scmle.experimentation.stores`

- `add_column` -> `alter`
- `NullExperimentStore` -> `Null`
- `ExperimentStore` -> `Store`

## `scmle.experimentation.strategy_search`

- `search_model` -> `tune`
- `StrategySearch` -> `Strategist`

## `scmle.experimentation.trials`

- `access_record` -> `summarise`
- `fold_weights` -> `weigh`
- `lost_trial` -> `lose`
- `TrialJob` -> `Job`
- `guarded_evaluate` -> `supervise`
- `TrialPayload` -> `Payload`
- `TrialRunner` -> `Runner`
- `runtime_requirements` -> `requirements`

## `scmle.experimentation.tuning`

- `promising_candidates` -> `shortlist`
- `tune_model` -> `optimize`
- `run_bracket` -> `ascend`
- `survivor_ids` -> `promote`
- `rung_schedule` -> `ladder`

## `scmle.explainability.counterfactual`

- `candidate_values` -> `propose`
- `scale_of` -> `measure`
- `CounterfactualSearch` -> `Recourse`
- `n_changed` -> `edits`

## `scmle.explainability.dependence`

- `PartialDependence` -> `Dependence`
- `numeric_target` -> `numerify`

## `scmle.explainability.importance`

- `PermutationImportance` -> `Importance`

## `scmle.explainability.interaction`

- `sort_key` -> `rank`
- `PairwiseInteraction` -> `Pairwise`

## `scmle.explainability.local`

- `OcclusionExplainer` -> `Occlusion`

## `scmle.explainability.quality`

- `validate_score` -> `validate`
- `QualityEvaluation` -> `Evaluation`
- `GroundTruthEvaluation` -> `Comparison`
- `GroundTruthSuiteEvaluation` -> `Suite`
- `explanation_ranking` -> `rank`
- `ranking_stability` -> `topk`
- `QualityThresholds` -> `Thresholds`
- `ExplanationGroundTruth` -> `Truth`
- `evaluate_ground_truth` -> `truth`
- `evaluate_ground_truth_suite` -> `referee`
- `linear_contribution_report` -> `decompose`

## `scmle.explainability.suite`

- `build_evidence` -> `substantiate`

## `scmle.features.feature_graph`

- `IsMissing` -> `Missingness`
- `DatePart` -> `Calendar`
- `RollingStd` -> `Volatility`
- `get_op` -> `lookup`
- `entity_keys` -> `identify`
- `compute_with_history` -> `backfill`
- `history_columns` -> `historical`
- `split_nodes` -> `segregate`
- `history_rows` -> `depth`
- `build_history` -> `retain`
- `FeatureOp` -> `Operation`
- `required_raw_inputs` -> `roots`
- `OrderedOp` -> `Sequential`
- `SeriesFeatureBuilder` -> `Temporal`
- `FeatureGraphTransformer` -> `Transformer`
- `RollingMean` -> `Rolling`

## `scmle.governance.attestation`

- `canonical_digest` -> `checksum`
- `required_text` -> `mandatory`
- `ReleaseAttestation` -> `Attestation`

## `scmle.governance.deployment_ledger`

- `apply_free` -> `accept`
- `apply_challenger` -> `replay`
- `apply_escalation` -> `escalate`
- `apply_attestation` -> `attest`
- `apply_used` -> `spend`
- `apply_envelope` -> `issue`
- `apply_envelope_used` -> `consume`
- `apply_envelope_revoked` -> `revoke`
- `apply_controller` -> `journal`
- `apply_experiment` -> `start`
- `apply_assignment` -> `assign`
- `apply_rollout_decision` -> `confirm`
- `apply_halt` -> `halt`
- `require_promotion` -> `elond`
- `DeploymentLedgerError` -> `MisuseError`
- `ReplyDecision` -> `Reply`
- `require_object_body` -> `mapped`
- `key_fingerprint` -> `thumbprint`

## `scmle.governance.envelope`

- `signature_ok` -> `verifies`
- `EnvelopeAction` -> `Action`
- `EnvelopeRefused` -> `RefusedError`
- `EnvelopeGuard` -> `Guard`

## `scmle.governance.escalation`

- `RiskCoverage` -> `Curve`
- `ChowCheck` -> `Comparison`
- `select_threshold` -> `select`
- `chow_check` -> `adjudicate`
- `everything_escalates` -> `saturated`
- `set_size_score` -> `ambiguity`

## `scmle.governance.keyring`

- `can_verify` -> `verifies`
- `SignatureEnvelope` -> `Wrapper`
- `SigningKeyRegistry` -> `Keyring`
- `KeyState` -> `State`
- `SigningKeyRecord` -> `Credential`

## `scmle.governance.lifecycle`

- `note_rows` -> `tally`
- `defer_to_rollout` -> `delegate`
- `restore_previous` -> `revert`
- `read_state` -> `recall`
- `write_state` -> `persist`
- `split_fresh` -> `bisect`
- `RollbackOutcome` -> `Outcome`
- `promote_challenger` -> `elevate`
- `review_locked` -> `locked`

## `scmle.governance.local_faults`

- `stop_child` -> `halt`
- `write_events` -> `accept`
- `reject_over_budget` -> `cap`
- `reject_over_memory_budget` -> `ration`
- `fail_connector` -> `unreachable`
- `LocalFaultExecutor` -> `Executor`

## `scmle.governance.loop`

- `ReplyOutcome` -> `Consequence`
- `json_ready` -> `jsonify`
- `build_monitors` -> `arm`
- `load_state` -> `restore`
- `process_batch` -> `feed`
- `update_challengers` -> `challenge`
- `time_out` -> `expire`
- `check_feedback` -> `performativity`
- `promo_level` -> `alpha`
- `register_challenger` -> `enroll`
- `rewrite_ledger` -> `rewrite`
- `watch_rollback` -> `surveil`
- `verify_ledger` -> `recheck`
- `estimate_signed` -> `centred`
- `guard_registry` -> `intercept`
- `silence_alert` -> `silence`
- `retrain_challenger` -> `refit`
- `enter_halt` -> `suspend`
- `decide_rollback` -> `propose`
- `execute_rollback` -> `revert`
- `decide_queue` -> `drain`

## `scmle.governance.model_card`

- `ModelCard` -> `Card`

## `scmle.governance.never_list`

- `pending_for` -> `outstanding`
- `NeverAction` -> `Forbidden`
- `question_for` -> `interrogate`
- `valid_attestation` -> `vouch`

## `scmle.governance.registry`

- `promote_locked` -> `relabel`
- `validate_target` -> `validate`
- `PromotionDecision` -> `Verdict`
- `attach_guard` -> `attach`
- `PromotionApproval` -> `Sanction`
- `VersionRecord` -> `Release`
- `content_id_of` -> `identify`
- `register_locked` -> `enroll`
- `check_promotion` -> `vet`

## `scmle.governance.release_state`

- `record_event` -> `log`
- `last_ledger_event` -> `recall`
- `recorded_findings` -> `compare`
- `state_of` -> `lookup`
- `commit_promotion` -> `finalize`
- `commit_rollback` -> `rewind`
- `deployed_version` -> `live`
- `stop_rollout` -> `halt`
- `require_previous` -> `predecessor`

## `scmle.governance.reliability`

- `FaultArtifact` -> `Emission`
- `FaultInjectionHarness` -> `Harness`

## `scmle.governance.review`

- `IndependentReleaseReview` -> `Signoff`
- `ReviewDecision` -> `Verdict`
- `ReviewClaimKind` -> `Kind`
- `ReviewClaim` -> `Recomputation`
- `require_independence` -> `separation`
- `require_claims` -> `vet`

## `scmle.governance.rollout_hook`

- `RolloutHook` -> `Hook`

## `scmle.governance.sbom`

- `build_service` -> `catalogue`

## `scmle.inference.guard`

- `NotReadyError` -> `UnreadyError`
- `AdmissionControl` -> `Admission`
- `new_request_id` -> `mint`

## `scmle.inference.mapping`

- `ColumnMapping` -> `Mapper`

## `scmle.inference.modality_serving`

- `new_server_request_id` -> `mint`
- `RejectedRequestError` -> `RejectedError`
- `request_identity` -> `identify`
- `admit_request` -> `screen`
- `read_payload` -> `receive`
- `set_gateway` -> `protect`

## `scmle.inference.novelty`

- `far_out_fences` -> `fence`
- `NoveltyDetector` -> `Strangeness`

## `scmle.inference.onnx_export`

- `register_booster_converters` -> `enrol`
- `is_token_imputer` -> `imputes`
- `without_casts` -> `uncast`

## `scmle.inference.packaging`

- `write_service` -> `scaffold`

## `scmle.inference.parity`

- `GoldenSet` -> `Goldens`
- `load_golden` -> `goldens`
- `reference_bundle` -> `handwrite`
- `expected_outputs` -> `expect`
- `check_remote` -> `crosscheck`

## `scmle.inference.serving`

- `synthetic_record` -> `synthesise`
- `warm_up` -> `warm`
- `enforce_policy` -> `enforce`
- `load_in_background` -> `boot`
- `observe_request` -> `observe`
- `observe_rows` -> `feed`
- `ReferenceServer` -> `Listener`
- `read_body` -> `receive`
- `answer_prediction` -> `answer`
- `RequestObserver` -> `Observer`
- `respond_frame` -> `infer`
- `BatchServing` -> `Batcher`

## `scmle.inference.validation`

- `row_report` -> `describe`
- `InputValidator` -> `Validator`

## `scmle.integrations.ray_backend`

- `RayBackend` -> `Remote`

## `scmle.intelligence.fingerprint`

- `missingness_patterns` -> `summarise`
- `scale_range` -> `magnitudes`
- `build_fingerprint` -> `imprint`
- `redundant_pairs` -> `redundant`
- `order_drift` -> `ordering`
- `text_like_columns` -> `texts`

## `scmle.intelligence.knowledge`

- `MetaLearner` -> `Recommender`

## `scmle.intelligence.planner`

- `MemoryFilter` -> `Memory`
- `BudgetFilter` -> `Budget`
- `IncrementalFilter` -> `Streaming`
- `SerializableFilter` -> `Persistence`
- `CategoricalPrioritizer` -> `Cardinality`
- `ExclusionFilter` -> `Exclusion`
- `ApprovalFilter` -> `Approval`
- `HistoryPrioritizer` -> `History`
- `StrategyRecorder` -> `Strategy`
- `TuningRecorder` -> `Tuning`
- `EnsembleRecorder` -> `Ensemble`
- `PlanningContext` -> `Situation`

## `scmle.intelligence.recommendations`

- `ImbalanceAdvice` -> `Imbalance`
- `MissingnessAdvice` -> `Missingness`
- `LeakageAdvice` -> `Leakage`
- `DuplicateAdvice` -> `Duplication`
- `SmallDataAdvice` -> `Scarcity`

## `scmle.models.boosting.early_stopping`

- `stopping_arguments` -> `arm`
- `recorded_iteration` -> `recall`
- `EarlyStoppingClassifier` -> `Classifier`
- `EarlyStoppingRegressor` -> `Regressor`
- `HoldoutEarlyStopping` -> `Holdout`

## `scmle.models.conformance`

- `check_conformance` -> `comply`
- `toy_problem` -> `toy`
- `check_declared_capabilities` -> `smoke`
- `check_search_space` -> `searchable`

## `scmle.models.deeplearning.architectures`

- `MlpArchitecture` -> `Mlp`
- `ResidualArchitecture` -> `Residual`
- `EnsembleArchitecture` -> `Ensemble`
- `TransformerArchitecture` -> `Transformer`
- `SmoothClip` -> `Saturation`
- `RobustMlpArchitecture` -> `Robust`
- `EnsembleLinear` -> `Affine`

## `scmle.models.deeplearning.contracts`

- `CheckpointContract` -> `Checkpoint`
- `ExplanationContract` -> `Explainability`
- `FeatureExplanation` -> `Contribution`
- `ModelFamilyContract` -> `Boundary`
- `model_family_contract` -> `resolve`
- `validate_features` -> `validate`
- `InferenceContract` -> `Contract`

## `scmle.models.deeplearning.engines`

- `UpdatePlan` -> `Plan`
- `plan_updates` -> `apportion`
- `training_arrays` -> `prepare`
- `set_learning_rate` -> `adjust`
- `check_options` -> `vet`
- `new_network` -> `construct`
- `new_optimizer` -> `instantiate`

## `scmle.models.deeplearning.estimators`

- `TabPfnEstimator` -> `TabPfn`
- `KerasEstimator` -> `Keras`

## `scmle.models.deeplearning.jepa`

- `JepaEstimator` -> `Embedder`

## `scmle.models.deeplearning.jepa_engine`

- `JepaNetwork` -> `Network`
- `validated_matrix` -> `validate`
- `check_hyperparameters` -> `vet`
- `train_batch` -> `learn`

## `scmle.models.deeplearning.keras_engine`

- `load_keras` -> `load`
- `build_network` -> `build`

## `scmle.models.deeplearning.training`

- `resolve_batch_size` -> `batching`
- `learning_rate_at` -> `anneal`

## `scmle.models.documents`

- `mean_characters` -> `measure`
- `term_ablation_explanation` -> `ablate`

## `scmle.models.ensembles`

- `GreedyBlender` -> `Blender`
- `members_of` -> `enlist`

## `scmle.models.forecasting.estimators`

- `ForecastEstimator` -> `Forecaster`
- `LastValueForecaster` -> `Naive`

## `scmle.models.modality_lifecycle`

- `FeatureBatch` -> `Batch`
- `DatasetPartitions` -> `Partitions`
- `check_report_identifies_package` -> `identify`
- `parse_signature` -> `unseal`
- `require_string_list` -> `require`
- `mapping_field` -> `pluck`
- `check_stage_progress` -> `sequence`
- `check_content_digest` -> `attest`
- `raw_span_occlusion_scores` -> `mask`
- `raw_pixel_occlusion_terms` -> `occlude`
- `raw_waveform_window_scores` -> `silence`
- `raw_frame_occlusion_scores` -> `blank`
- `masked_decoder` -> `blackout`
- `pixel_feature_values` -> `summarise`
- `record_performance` -> `clock`
- `IngestionStage` -> `Ingestion`
- `ValidationStage` -> `Validation`
- `ProfilingStage` -> `Profiling`
- `SplittingStage` -> `Splitting`
- `FeaturePreparationStage` -> `Preparation`
- `ModellingStage` -> `Modelling`
- `EvaluationStage` -> `Evaluation`
- `RepeatedExplanation` -> `Stability`
- `zero_ablation_consistency` -> `ablate`
- `repeated_explanations` -> `perturb`
- `linear_decision_fidelity` -> `replicate`
- `ground_truth_evidence` -> `corroborate`
- `PackagingStage` -> `Packaging`
- `InferenceStage` -> `Inference`
- `MonitoringStage` -> `Monitoring`
- `DocumentationStage` -> `Documentation`
- `RecoveryAndResourceStage` -> `Recovery`
- `RunPlan` -> `Plan`
- `ensure_not_cancelled` -> `poll`
- `remove_package` -> `discard`
- `check_observed_modality` -> `conform`
- `check_checkpoint_presence` -> `expect`
- `load_resume_checkpoint` -> `reload`
- `verified_completed_report` -> `verify`
- `persist_checkpoint` -> `persist`
- `evaluate_performance` -> `benchmark`
- `normalize_json_object` -> `normalise`
- `VideoPreparer` -> `Footage`
- `check_package_signature` -> `authenticate`
- `parse_package` -> `unpack`
- `perturbed_contribution_reports` -> `sensitivities`
- `check_run_option_types` -> `typecheck`
- `check_package_classifier` -> `vet`
- `RasterPreparer` -> `Imagery`
- `TextPreparer` -> `Plaintext`
- `check_checkpoint_options` -> `coherent`
- `one_hop_subgraph` -> `neighbourhood`
- `CurationStage` -> `Curating`
- `execute_with_performance` -> `clocked`
- `is_sha256` -> `hashlike`
- `pixel_feature_names` -> `headers`
- `DataExplainabilityStage` -> `Composition`
- `parse_stage_evidence` -> `rehydrate`

## `scmle.models.multitarget.adapters`

- `MultilabelLogistic` -> `Logistic`
- `MultiForest` -> `Forest`

## `scmle.models.multitarget.estimators`

- `NativeMultiOutput` -> `Ensemble`

## `scmle.models.multitarget.wrapper`

- `MultiTargetEstimator` -> `Multitarget`
- `MultiOutputEstimator` -> `Multioutput`

## `scmle.models.ordinal.estimators`

- `level_probabilities` -> `apportion`
- `best_scale` -> `tune`

## `scmle.models.ranking.wrapper`

- `QueryAwareEstimator` -> `Ranker`

## `scmle.models.registry`

- `with_builtins` -> `stock`
- `user_adapters` -> `extensions`
- `add_user_adapter` -> `adopt`
- `remove_user_adapter` -> `deregister`

## `scmle.models.sklearn.bagging`

- `BaggedTree` -> `Bagger`

## `scmle.models.sklearn.classical`

- `SupportVector` -> `Svm`

## `scmle.models.survival.adapters`

- `DiscreteTime` -> `Hazard`

## `scmle.models.survival.estimators`

- `interval_of` -> `locate`
- `step_lookup` -> `stair`
- `MarginalSurvival` -> `Marginal`
- `CoxRegressor` -> `Hazards`

## `scmle.models.thresholds`

- `optimize_threshold` -> `argmax`
- `GroupAssigner` -> `Grouper`
- `F1Objective` -> `F1`
- `GroupClassifier` -> `Segmenter`
- `AccuracyObjective` -> `Accuracy`
- `around_fitted` -> `around`
- `CostObjective` -> `Costs`

## `scmle.models.uplift.estimators`

- `ZeroEffect` -> `Inert`
- `outcome_model` -> `booster`
- `TwoModelLearner` -> `TLearner`

## `scmle.models.weighting`

- `WeightedRegressor` -> `Regressor`
- `WeightedEstimator` -> `Reweighter`
- `WeightedClassifier` -> `Classifier`

## `scmle.models.wrappers`

- `FitParamsRegressor` -> `Curried`
- `LabelEncodedClassifier` -> `Encoded`

## `scmle.monitoring.abtest`

- `score_arm` -> `tally`

## `scmle.monitoring.alerts`

- `AlertDeliveryError` -> `DeliveryError`
- `AlertDelivery` -> `Delivery`
- `AlertSink` -> `Sink`
- `successful_alerts` -> `delivered`
- `dispatch_locked` -> `relay`
- `JsonlAlertSink` -> `Jsonl`
- `WebhookAlertSink` -> `Pager`
- `AlertDispatcher` -> `Dispatcher`

## `scmle.monitoring.champion`

- `ChampionChallenger` -> `Referee`

## `scmle.monitoring.events`

- `enforce_retention_locked` -> `prune`
- `to_plain` -> `unwrap`
- `PredictionEvent` -> `Served`
- `log_predictions` -> `chronicle`
- `OutcomeEvent` -> `Outcome`
- `inference_timestamp` -> `moment`
- `enforce_retention` -> `purge`

## `scmle.monitoring.incidents`

- `event_digest` -> `digest`
- `require_record` -> `require`
- `IncidentStatus` -> `Phase`
- `check_transition` -> `proceed`
- `IncidentWorkflow` -> `Incidents`
- `append_event` -> `log`

## `scmle.monitoring.labels`

- `RandomizationCheck` -> `Uniformity`
- `permits_action` -> `actionable`
- `stream_mechanism` -> `worst`
- `BatchEstimate` -> `Estimate`
- `estimate_batch` -> `weigh`
- `check_randomization` -> `randomization`
- `bypass_test` -> `circularity`
- `feedback_verdict` -> `assessment`

## `scmle.monitoring.monitor`

- `model_output` -> `infer`
- `ModelMonitor` -> `Watcher`
- `PerformanceAdvisor` -> `Degradation`
- `UnseenCategoryAdvisor` -> `Novelty`

## `scmle.monitoring.observability`

- `ObservabilityEvaluation` -> `Evaluation`

## `scmle.monitoring.reference`

- `DistributionReference` -> `Reference`
- `reference_from_dict` -> `rebuild`
- `adjusted_psi` -> `adjust`
- `population_stability_index` -> `divergence`
- `NumericReference` -> `Deciles`
- `ReferenceProfile` -> `Baseline`

## `scmle.monitoring.sequential_tests`

- `start_term` -> `spawn`
- `PairedEProcess` -> `Comparison`
- `BettingTest` -> `Betting`
- `update_gap` -> `ingest`
- `MixtureTest` -> `Mixture`
- `ConfidenceSequence` -> `Confidence`
- `RiskAlarm` -> `Tripwire`

## `scmle.monitoring.shadow`

- `ShadowEvaluator` -> `Understudy`

## `scmle.monitoring.slo`

- `SLOEvaluation` -> `Assessment`
- `SLOStatus` -> `Status`
- `SLOTarget` -> `Slo`

## `scmle.operations.continuous`

- `canonical_csv` -> `serialize`
- `pending_id` -> `find`
- `start_challenger` -> `train`
- `check_after_promotion` -> `followup`
- `judge_retraining` -> `decide`
- `of_kind` -> `select`
- `bootstrap_difference` -> `contrast`
- `feed_labels` -> `absorb`
- `feed_outcomes` -> `feed`
- `holdback_assignment` -> `reserve`
- `promotion_proposals` -> `nominations`

## `scmle.operations.controller`

- `authorized_retrain` -> `train`

## `scmle.operations.cost`

- `from_allocation` -> `allocate`

## `scmle.operations.faults`

- `golden_spec` -> `specify`
- `inverted_spec` -> `invert`
- `ready_after` -> `boot`
- `corrupt_copy` -> `corrupt`
- `model_file_digest` -> `digest`
- `work_folder` -> `provision`
- `kill_after` -> `interrupt`
- `state_writer_trials` -> `stress`
- `rolling_restart` -> `restart`
- `register_image` -> `enroll`
- `detect_performance_loss` -> `degradation`
- `secret_rotation` -> `rekey`
- `metrics_destination_outage` -> `blackout`
- `audit_continuity` -> `resilience`
- `await_unready` -> `failing`
- `ProcessPlane` -> `Processes`

## `scmle.operations.logs`

- `traceparent_for` -> `traceheader`
- `trace_for` -> `propagate`
- `StructuredLog` -> `Journal`
- `new_trace_id` -> `traceid`

## `scmle.operations.metrics`

- `render_labels` -> `stringify`
- `merged_histogram` -> `merge`
- `render_histogram` -> `tabulate`
- `ServiceMetrics` -> `Instrumentation`
- `record_resources` -> `poll`
- `format_number` -> `spell`
- `MetricsRegistry` -> `Registry`
- `cpu_seconds` -> `cpu`

## `scmle.operations.notify`

- `urllib_transport` -> `courier`

## `scmle.operations.respond`

- `ResponseDesk` -> `Desk`
- `telemetry_path` -> `archive`

## `scmle.operations.signals`

- `service_signals` -> `assess`
- `data_signals` -> `gauge`
- `performance_signals` -> `degradation`
- `zero_one_losses` -> `misses`

## `scmle.optimization.optimizers`

- `suggest_choice` -> `choose`
- `GridOptimizer` -> `Grid`
- `OptunaOptimizer` -> `Parzen`
- `create_optimizer` -> `instantiate`
- `RandomOptimizer` -> `Uniform`

## `scmle.runners.base`

- `cost_of` -> `estimate`
- `confirm_stopped` -> `confirm`
- `manifest_key` -> `locate`
- `JobHandle` -> `Ticket`
- `JobStatus` -> `Status`
- `JobState` -> `State`
- `RunConflictError` -> `ConflictError`

## `scmle.runners.entrypoint`

- `spec_from_store` -> `respec`
- `open_store` -> `mount`
- `load_workload` -> `resolve`

## `scmle.runners.local`

- `JobStoppedError` -> `StoppedError`
- `JobInterruptedError` -> `PreemptedError`
- `JobContext` -> `Context`

## `scmle.runners.project`

- `project_options` -> `configure`
- `WorkdirSync` -> `Mirror`
- `saved_run` -> `locate`

## `scmle.runners.reproducibility`

- `for_metric` -> `lookup`

## `scmle.runners.roles`

- `allows_write` -> `permits`
- `need_write` -> `permit`

## `scmle.sampling.pseudo_labeling`

- `SelfTrainingOutcome` -> `Outcome`
- `confident_rows` -> `assured`
- `augmented_fit` -> `selftrain`

## `scmle.sampling.resampling`

- `JitterOverSampler` -> `Jittered`
- `IndexResampler` -> `Indexed`

## `scmle.sampling.strategies`

- `ClassWeighting` -> `Weighting`
- `FoldResampling` -> `Resampling`
- `BorderlineSmote` -> `Borderline`
- `LogTransform` -> `Logarithm`
- `RareTarget` -> `Rarity`
- `TargetEncoding` -> `Encoding`
- `FeatureConstruction` -> `Construction`
- `FeatureSelection` -> `Selection`
- `MissingIndicators` -> `Indicators`
- `TargetTransform` -> `Transformation`

## `scmle.sampling.synthesis`

- `SyntheticGenerator` -> `Synthesizer`
- `GaussianCopulaGenerator` -> `Copula`

## `scmle.security.access`

- `from_environment` -> `load`

## `scmle.security.remote`

- `fetch_json` -> `retrieve`
- `revocation_list_of` -> `revoked`

## `scmle.security.tokens`

- `int_from_segment` -> `unpack`
- `check_window` -> `validate`
- `optional_float` -> `coerce`
- `parse_object` -> `parse`
- `check_claims` -> `admit`
- `number_claim` -> `quantify`
- `scopes_of` -> `collect`
- `current_revocations` -> `denylist`
- `decode_segment` -> `unbase`
- `encode_segment` -> `pack`
- `from_jwk` -> `jwk`
- `from_jwks` -> `jwks`

## `scmle.trail.chain`

- `manifest_of` -> `inventory`
- `signature_for` -> `sign`
- `check_chain` -> `linkage`
- `check_steps` -> `rederive`

## `scmle.trail.inputs`

- `section_payload` -> `cover`
- `missing_fields` -> `lack`
- `signature_problem` -> `diagnose`
- `structure_problems` -> `lint`

## `scmle.trail.records`

- `sha256_file` -> `checksum`

## `scmle.trail.show`

- `facts_lines` -> `describe`

## `scmle.trail.trace`

- `check_charter_signature` -> `verify`
- `report_step` -> `recompute`
- `reconcile_feed` -> `reconcile`
- `outcomes_step` -> `observe`

## `scmle.unsupervised.clustering`

- `find_clusters` -> `discover`
- `external_scores` -> `compare`
- `score_candidate` -> `attempt`

## `scmle.workflow.definition`

- `check_name` -> `vet`
- `group_limit` -> `quota`
- `has_approval_upstream` -> `gated`

## `scmle.workflow.executor`

- `StepState` -> `Phase`
- `active_runs` -> `scan`
- `saved_progress` -> `restore`
- `skip_if_upstream_failed` -> `skip`
- `can_start` -> `admit`
- `schedule_ready_steps` -> `dispatch`
- `collect_finished_steps` -> `collect`
- `close_run` -> `close`
- `final_state` -> `derive`
- `recorded_cost` -> `tally`
- `approval_payload` -> `compose`
- `settle_approval` -> `settle`
- `RunInFlightError` -> `BusyError`
- `save_step` -> `stash`
- `take_lease` -> `acquire`
- `sign_approval` -> `decide`
- `StepTimeoutError` -> `DeadlineError`
- `submit_approval` -> `ratify`
- `release_lease` -> `relinquish`
- `StepInterruptedError` -> `PreemptedError`
- `settle_approval_step` -> `adjudicate`

## `scmle.workflow.handlers`

- `runner_handler` -> `delegate`

## `scmle.workflow.schedule`

- `last_slot` -> `load`

## Defined in several modules

These names are defined in more than one place (the same method on several
classes, or the same helper in several modules); every definition was
renamed to the same word. The modules are listed after each entry.

- `accepts_weights` -> `weightable` (`scmle.contracts.metrics`, `scmle.evaluation.concordance`, `scmle.evaluation.forecast`, `scmle.evaluation.multitarget`, `scmle.evaluation.ranking`, `scmle.evaluation.survival`, `scmle.evaluation.uplift`, `scmle.models.weighting`)
- `available_events` -> `unconsumed` (`scmle.data.readers.incremental`)
- `BudgetMeter` -> `Meter` (`scmle.data.ingest.budget`, `scmle.runners.base`)
- `build_runner` -> `rig` (`scmle.core.causal_claim`, `scmle.data.framing.claim`, `scmle.evaluation.coverage_confirm`)
- `canonical_json` -> `canonize` (`scmle.contracts.capabilities`, `scmle.contracts.evaluation`, `scmle.contracts.evidence`, `scmle.contracts.explanations`, `scmle.contracts.lineage`, `scmle.contracts.observability`, `scmle.contracts.tasks`, `scmle.data.catalog`, `scmle.deploy.contract`, `scmle.evaluation.release`, `scmle.governance.review`)
- `categorical_params` -> `nominals` (`scmle.contracts.models`, `scmle.models.boosting.boosters`, `scmle.models.sklearn.classical`)
- `check_counts` -> `census` (`scmle.data.catalog`, `scmle.data.modalities.geospatial`, `scmle.data.modalities.graph`, `scmle.evaluation.sequential`)
- `check_covariance` -> `invertible` (`scmle.evaluation.longitudinal`, `scmle.evaluation.parameter_estimation`, `scmle.evaluation.sequential`)
- `check_declared_columns` -> `conform` (`scmle.api.non_tabular`, `scmle.api.project`, `scmle.data.robustness.report`, `scmle.evaluation.longitudinal`, `scmle.evaluation.parameter_estimation`, `scmle.evaluation.sequential`, `scmle.models.modality_lifecycle`)
- `check_manifest` -> `audit` (`scmle.contracts.capabilities`, `scmle.contracts.evaluation`, `scmle.core.stages`, `scmle.data.detectors.certificates`, `scmle.data.ledger.ledger`, `scmle.data.modalities.curation`, `scmle.deploy.gate`, `scmle.deploy.verify`, `scmle.evaluation.release`, `scmle.governance.registry`, `scmle.models.modality_lifecycle`, `scmle.trail.chain`)
- `check_multiplicity` -> `plurality` (`scmle.core.ingest_stage`, `scmle.data.ingest.synthesize`)
- `check_policy` -> `enforce` (`scmle.contracts.evaluation`, `scmle.data.ledger.confirm`, `scmle.deploy.vuln_gate`, `scmle.inference.serving`)
- `check_rollback` -> `revertible` (`scmle.governance.lifecycle`, `scmle.governance.rollout_hook`)
- `check_row_accounting` -> `conserve` (`scmle.evaluation.longitudinal`, `scmle.evaluation.parameter_estimation`)
- `check_subject_and_time` -> `situate` (`scmle.evaluation.longitudinal`, `scmle.evaluation.sequential`)
- `checkpoint_manifest` -> `inventory` (`scmle.artifacts.bundle`, `scmle.artifacts.claims_audit`, `scmle.core.stages`, `scmle.models.deeplearning.estimators`, `scmle.models.deeplearning.jepa`, `scmle.trail.chain`)
- `clear_pending_acknowledgement` -> `retract` (`scmle.data.readers.incremental`, `scmle.data.readers.streaming`)
- `column_of` -> `locate` (`scmle.api.project`, `scmle.artifacts.claims_audit`, `scmle.data.ingest.declared`, `scmle.data.modalities.geospatial`, `scmle.data.readers.streaming`, `scmle.data.splitting.duplicates`, `scmle.data.store.objectstore`, `scmle.data.store.quality`, `scmle.integrations.mlflow_store`, `scmle.models.survival.estimators`, `scmle.runners.base`, `scmle.runners.project`)
- `commit_with_writer` -> `finalize` (`scmle.data.readers.incremental`, `scmle.data.readers.streaming`, `scmle.governance.release_state`)
- `CommittedState` -> `Settlement` (`scmle.data.readers.incremental`, `scmle.data.readers.streaming`)
- `convert_numeric_columns` -> `numericize` (`scmle.evaluation.longitudinal`, `scmle.evaluation.sequential`)
- `DatasetCard` -> `Datasheet` (`scmle.contracts.data`, `scmle.data.modalities.curation`)
- `DecisionAction` -> `Edict` (`scmle.contracts.experiments`, `scmle.data.charter.contracts`)
- `default_rules` -> `ruleset` (`scmle.intelligence.planner`, `scmle.intelligence.recommendations`)
- `default_suite` -> `panoply` (`scmle.evaluation.diagnostics`, `scmle.explainability.suite`)
- `deliver_pending_acknowledgement` -> `handoff` (`scmle.data.readers.incremental`, `scmle.data.readers.streaming`)
- `digest_of` -> `distill` (`scmle.contracts.data`, `scmle.deploy.observation`, `scmle.governance.ledger`, `scmle.workflow.executor`)
- `encode_target` -> `vectorize` (`scmle.data.admissibility.leakage`, `scmle.data.ledger.explore`, `scmle.models.deeplearning.engines`, `scmle.models.deeplearning.keras_engine`)
- `estimate_for` -> `reckon` (`scmle.data.identification.run`)
- `ever_crossed` -> `surpassed` (`scmle.decision.anytime`, `scmle.monitoring.sequential_tests`)
- `excludes_zero` -> `conclusive` (`scmle.data.identification.run`, `scmle.data.robustness.stability`)
- `exclusion_reason` -> `rationale` (`scmle.data.validation.columns`)
- `extra_state` -> `surplus` (`scmle.models.deeplearning.foundation`)
- `FieldProvenance` -> `Pedigree` (`scmle.data.admissibility.provenance`, `scmle.data.detectors.outliers`)
- `fingerprint_frame` -> `imprint` (`scmle.data.modalities.image`, `scmle.evaluation.longitudinal`, `scmle.evaluation.sequential`, `scmle.intelligence.fingerprint`)
- `from_observation` -> `glean` (`scmle.contracts.data`, `scmle.data.catalog`)
- `gate_status` -> `adjudication` (`scmle.data.admissibility.leakage`, `scmle.data.detectors.pipeline`, `scmle.data.robustness.stability`)
- `global_explanation` -> `overview` (`scmle.models.documents`, `scmle.models.images`)
- `gpu_usable` -> `accelerated` (`scmle.contracts.models`, `scmle.models.deeplearning.networks`)
- `group_ids` -> `cohorts` (`scmle.data.modalities.audio`, `scmle.data.modalities.image`, `scmle.data.modalities.video`, `scmle.models.modality_lifecycle`)
- `has_future_leakage` -> `lookahead` (`scmle.data.modalities.geospatial`, `scmle.data.modalities.graph`, `scmle.data.modalities.video`)
- `independent_reference` -> `referee` (`scmle.evaluation.longitudinal`, `scmle.evaluation.sequential`, `scmle.explainability.quality`)
- `is_valid_request_id` -> `wellformed` (`scmle.inference.guard`, `scmle.inference.modality_serving`)
- `issue_token` -> `mint` (`scmle.data.custody.sealed`, `scmle.data.ledger.confirm`, `scmle.data.splitting.splitters`, `scmle.evaluation.reconcile`, `scmle.governance.vault`, `scmle.inference.guard`, `scmle.inference.modality_serving`)
- `KeySet` -> `Keystore` (`scmle.security.remote`, `scmle.security.tokens`)
- `kind_of` -> `categorize` (`scmle.data.store.snapshot`, `scmle.data.validation.columns`)
- `load_checkpoint` -> `restore` (`scmle.api.project`, `scmle.artifacts.bundle`, `scmle.contracts.modalities`, `scmle.core.orchestrator`, `scmle.data.custody.manifest`, `scmle.data.readers.registry`, `scmle.data.store.processing`, `scmle.governance.loop`, `scmle.governance.release_state`, `scmle.models.deeplearning.estimators`, `scmle.models.deeplearning.jepa`, `scmle.runners.project`, `scmle.workflow.executor`)
- `load_json` -> `deserialize` (`scmle.data.charter.contracts`, `scmle.data.readers.connectors`, `scmle.deploy.vuln_gate`)
- `load_pending_acknowledgement` -> `reclaim` (`scmle.data.readers.incremental`, `scmle.data.readers.streaming`)
- `load_result` -> `recall` (`scmle.api.non_tabular`, `scmle.api.sequential`, `scmle.artifacts.claims_audit`, `scmle.contracts.qualification`, `scmle.core.orchestrator`, `scmle.evaluation.ranking`, `scmle.governance.lifecycle`, `scmle.governance.release_state`, `scmle.models.boosting.early_stopping`)
- `local_explanation` -> `spotlight` (`scmle.models.documents`, `scmle.models.images`)
- `lock_path` -> `lockfile` (`scmle.data.readers.incremental`, `scmle.data.readers.streaming`)
- `loss_name` -> `objective` (`scmle.evaluation.parameter_estimation`, `scmle.models.deeplearning.keras_engine`, `scmle.models.survival.estimators`)
- `meta_learner` -> `stacker` (`scmle.models.ensembles`)
- `modality_notes` -> `remarks` (`scmle.models.modality_lifecycle`)
- `modality_specific_explanation` -> `elucidate` (`scmle.models.modality_lifecycle`)
- `ObjectStore` -> `Depot` (`scmle.data.store.objectstore`, `scmle.deploy.mocks`)
- `observe_features` -> `sense` (`scmle.inference.serving`, `scmle.operations.signals`)
- `out_of_fold_probabilities` -> `heldout` (`scmle.data.detectors.labels`, `scmle.models.thresholds`)
- `output_units` -> `dimensionality` (`scmle.models.deeplearning.engines`, `scmle.models.deeplearning.keras_engine`)
- `period_label` -> `periodize` (`scmle.evaluation.longitudinal`, `scmle.evaluation.sequential`)
- `predicted_class_score` -> `confidence` (`scmle.models.documents`, `scmle.models.images`)
- `raise_unsupported` -> `refuse` (`scmle.contracts.capabilities`, `scmle.contracts.modalities`, `scmle.data.framing.refusal`, `scmle.data.identification.run`, `scmle.deploy.verify`, `scmle.security.access`)
- `record_request` -> `tally` (`scmle.api.results`, `scmle.artifacts.claims_audit`, `scmle.contracts.performance`, `scmle.core.stages`, `scmle.data.catalog`, `scmle.data.ledger.explore`, `scmle.data.modalities.graph`, `scmle.evaluation.release`, `scmle.governance.lifecycle`, `scmle.inference.serving`, `scmle.monitoring.abtest`, `scmle.operations.metrics`, `scmle.operations.signals`, `scmle.workflow.executor`)
- `record_trial` -> `transcribe` (`scmle.core.context`, `scmle.experimentation.stores`, `scmle.integrations.mlflow_store`)
- `rejection_reason` -> `objection` (`scmle.contracts.models`, `scmle.models.deeplearning.networks`)
- `require_fitted` -> `preflight` (`scmle.models.documents`, `scmle.models.images`)
- `resident_bytes` -> `residency` (`scmle.core.limits`, `scmle.operations.metrics`)
- `resolve_device` -> `place` (`scmle.models.deeplearning.engines`, `scmle.models.deeplearning.jepa_engine`)
- `RetryExhaustedError` -> `RetryError` (`scmle.core.retry`, `scmle.data.readers.connectors`)
- `row_losses` -> `penalize` (`scmle.models.deeplearning.engines`)
- `row_weights` -> `weighting` (`scmle.models.deeplearning.engines`)
- `RunState` -> `Progress` (`scmle.core.state`, `scmle.runners.base`, `scmle.workflow.executor`)
- `samples_for` -> `sieve` (`scmle.data.modalities.audio`, `scmle.data.modalities.image`, `scmle.data.modalities.video`)
- `save_checkpoint` -> `stash` (`scmle.data.store.processing`, `scmle.models.deeplearning.estimators`, `scmle.models.deeplearning.jepa`, `scmle.workflow.executor`)
- `send_json` -> `transmit` (`scmle.deploy.mocks`, `scmle.inference.modality_serving`, `scmle.inference.serving`)
- `split_temporal` -> `cutover` (`scmle.data.modalities.geospatial`, `scmle.data.modalities.video`)
- `start_run` -> `commence` (`scmle.experimentation.stores`, `scmle.integrations.mlflow_store`)
- `state_path` -> `statefile` (`scmle.governance.lifecycle`, `scmle.governance.loop`)
- `stratify_labels` -> `stratifier` (`scmle.data.splitting.splitters`)
- `supports_modality` -> `accommodates` (`scmle.contracts.capabilities`, `scmle.models.deeplearning.contracts`)
- `TemporalSplit` -> `Chronology` (`scmle.data.modalities.geospatial`, `scmle.data.modalities.graph`, `scmle.data.modalities.video`, `scmle.data.splitting.splitters`)
- `tipping_point` -> `tipover` (`scmle.data.detectors.missingness`, `scmle.data.framing.questions`)
- `to_float` -> `floating` (`scmle.data.validation.columns`, `scmle.inference.parity`)
- `to_object` -> `objectify` (`scmle.data.validation.columns`, `scmle.inference.parity`)
- `unsigned_dict` -> `preimage` (`scmle.contracts.evidence`, `scmle.deploy.record`, `scmle.governance.attestation`, `scmle.governance.deployment_ledger`, `scmle.governance.registry`, `scmle.governance.review`, `scmle.models.modality_lifecycle`)
- `update_run` -> `revise` (`scmle.experimentation.stores`, `scmle.integrations.mlflow_store`)
- `update_spent` -> `expend` (`scmle.data.ledger.online_fdr`)
- `validate_baseline_covariates` -> `pretreat` (`scmle.evaluation.longitudinal`, `scmle.evaluation.sequential`)
- `validate_calibration` -> `temper` (`scmle.contracts.evaluation`, `scmle.governance.escalation`)
- `validate_checkpoint` -> `rebind` (`scmle.data.readers.connectors`, `scmle.data.readers.incremental`, `scmle.data.readers.streaming`)
- `validate_digest` -> `assay` (`scmle.contracts.explanations`, `scmle.data.catalog`, `scmle.data.readers.streaming`)
- `validate_identity` -> `identify` (`scmle.artifacts.reproducibility`, `scmle.contracts.explanations`, `scmle.core.stages`, `scmle.data.identification.run`, `scmle.data.readers.tabular`, `scmle.features.feature_graph`, `scmle.governance.registry`, `scmle.inference.modality_serving`, `scmle.inference.serving`, `scmle.models.modality_lifecycle`)
- `validate_text` -> `sanitize` (`scmle.contracts.evaluation`, `scmle.contracts.explanations`, `scmle.data.readers.tabular`, `scmle.deploy.mocks`)
- `validate_time_order` -> `chronologize` (`scmle.evaluation.longitudinal`, `scmle.evaluation.sequential`)
- `verdict_of` -> `arbitrate` (`scmle.data.identification.run`, `scmle.data.ingest.joins`, `scmle.evaluation.bias_variance`)
- `write_pending_acknowledgement` -> `earmark` (`scmle.data.readers.incremental`, `scmle.data.readers.streaming`)

## Later single-definition renames

- `CachedPredictions` -> `Cache` (`scmle.evaluation.diagnostics`)
- `check_prepared_rows` -> `screen` (`scmle.evaluation.parameter_estimation`)
- `check_problems` -> `vet` (`scmle.deploy.gate`)
- `CommitLedger` -> `Journal` (`scmle.data.readers.streaming`)
- `CorruptObjectError` -> `MismatchError` (`scmle.data.store.objectstore`)
- `EvidenceOutcome` -> `Outcome` (`scmle.contracts.reliability`)
- `LifecycleOutcome` -> `Verdict` (`scmle.governance.lifecycle`)
- `observation_from_dict` -> `rebuild` (`scmle.deploy.release`)
- `parse_report` -> `interpret` (`scmle.deploy.vuln_gate`)
- `parse_traceparent` -> `extract` (`scmle.operations.logs`)
- `profile_catalog` -> `inventory` (`scmle.data.ingest.profile`)
- `register_run` -> `enlist` (`scmle.governance.lifecycle`)
- `ReleaseReadinessSummary` -> `Scorecard` (`scmle.evaluation.release`)
- `ReplyRefused` -> `RefusalError` (`scmle.governance.loop`)
- `require_keys` -> `require` (`scmle.data.robustness.report`)
- `roll_back` -> `rewind` (`scmle.governance.rollout_hook`)
- `side_of` -> `allot` (`scmle.data.readers.streaming`)
- `validate_sink_receipt` -> `attest` (`scmle.data.readers.streaming`)
- `write_document` -> `persist` (`scmle.deploy.attest`)

## Module moves

Modules moved between packages by the layer map (`scripts/layers.toml`, register phase 3). Each entry is `old -> new`; the names inside a module are unchanged.
- `scmle.core.randomness` -> `scmle.runtime.randomness`
- `scmle.core.retry` -> `scmle.runtime.retry`
- `scmle.core.limits` -> `scmle.runtime.limits`
- `scmle.core.ipc` -> `scmle.runtime.ipc`
- `scmle.core.resource_observation` -> `scmle.runtime.resource_observation`
- `scmle.core.state` -> `scmle.runtime.state`
- `scmle.core.config` -> `scmle.runtime.config`
- `scmle.core.scheduler` -> `scmle.runtime.scheduler`
- `scmle.core.budget` -> `scmle.runtime.budget`
- `scmle.core.logs` -> `scmle.runtime.logs`
- `scmle.integrations.ray_backend` -> `scmle.runtime.ray_backend`
- `scmle.models.deeplearning.worker` -> `scmle.runtime.worker`
- `scmle.data.store.objectstore` -> `scmle.storage.objectstore`
- `scmle.data.store.conformance` -> `scmle.storage.conformance`
- `scmle.experimentation.stores` -> `scmle.storage.stores`
- `scmle.integrations.mlflow_store` -> `scmle.storage.mlflow_store`
- `scmle.features.feature_graph` -> `scmle.data.feature_graph`
- `scmle.monitoring.reference` -> `scmle.data.reference`
- `scmle.data.curation.transformers` -> `scmle.data.transformers`
- `scmle.data.curation.embedding` -> `scmle.data.embedding`
- `scmle.data.curation` -> `scmle.preparation`
- `scmle.sampling.pseudo_labeling` -> `scmle.preparation.pseudo_labeling`
- `scmle.sampling.resampling` -> `scmle.preparation.resampling`
- `scmle.sampling.strategies` -> `scmle.preparation.strategies`
- `scmle.sampling.synthesis` -> `scmle.preparation.synthesis`
- `scmle.models.weighting` -> `scmle.preparation.weighting`
- `scmle.models.wrappers` -> `scmle.preparation.wrappers`
- `scmle.models.forecasting.estimators` -> `scmle.preparation.forecasters`
- `scmle.models.multitarget.wrapper` -> `scmle.preparation.multitarget`
- `scmle.models.ranking.wrapper` -> `scmle.preparation.ranking`
- `scmle.models.survival.wrapper` -> `scmle.preparation.survival`
- `scmle.models.uplift.wrapper` -> `scmle.preparation.uplift`
- `scmle.data.ledger` -> `scmle.assurance.ledger`
- `scmle.data.charter` -> `scmle.assurance.charter`
- `scmle.data.detectors` -> `scmle.assurance.detectors`
- `scmle.data.admissibility` -> `scmle.assurance.admissibility`
- `scmle.data.custody` -> `scmle.assurance.custody`
- `scmle.governance.vault` -> `scmle.assurance.custody.vault`
- `scmle.data.framing` -> `scmle.assurance.framing`
- `scmle.data.ingest` -> `scmle.assurance.ingest`
