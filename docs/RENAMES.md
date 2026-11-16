# Renames

Names renamed under the single-word naming rule: classes are nouns, functions
are verbs, attributes are adjectives, and snake_case is kept only where no
single word fits. Each entry is `old -> new`, grouped by module. Signatures
and behaviour are unchanged; replace the old name to migrate. Renames of
names in the public API snapshot are also listed in the changelog.

## `scmle.api.project`

- `build_parameter_estimation_declaration` -> `declare`
- `read_declaration_envelope` -> `unwrap`
- `restore_parameter_estimation_declaration` -> `restore`
- `bind_parameter_estimation_evidence` -> `bind`
- `persist_parameter_estimation_evidence` -> `persist`
- `require_treatment_column` -> `locate`
- `estimate_parameters_for_project` -> `estimate`
- `verify_existing_curation` -> `verify`

## `scmle.api.results`

- `unattested_suspects` -> `suspect`

## `scmle.artifacts.bundle`

- `installed_versions` -> `detect`
- `minor_version` -> `truncate`
- `sign_manifest` -> `sign`
- `verify_core_files` -> `authenticate`
- `verified_member_path` -> `resolve`
- `shifted_interval` -> `reweight`

## `scmle.artifacts.claims_audit`

- `coverage_status_problems` -> `reconcile`
- `evaluate_attestation` -> `assess`

## `scmle.artifacts.reproducibility`

- `trial_identity` -> `identify`
- `structural_differences` -> `compare`

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

## `scmle.contracts.capabilities`

- `ModelFamilyDeclaration` -> `Declaration`
- `register_model_family` -> `enroll`
- `resource_diagnostics` -> `audit`

## `scmle.contracts.data`

- `check_findings` -> `vet`

## `scmle.contracts.evaluation`

- `validate_evidence` -> `vet`
- `validate_finite` -> `quantify`
- `check_open_unit_interval` -> `confine`
- `check_explanation_threshold` -> `qualify`

## `scmle.contracts.explanations`

- `validate_fingerprint` -> `require`
- `validate_name` -> `require`
- `validate_parameters` -> `normalise`
- `validate_reason` -> `require`
- `validate_redacted_fields` -> `trim`
- `validate_uri` -> `screen`
- `validate_json_fields` -> `normalise`
- `validate_measurement_object` -> `require`

## `scmle.contracts.lineage`

- `topological_order_without_validation` -> `linearise`

## `scmle.contracts.performance`

- `sample_count_violations` -> `tally`

## `scmle.contracts.tasks`

- `series_node` -> `derive`
- `forecast_nodes` -> `featurize`
- `FrozenModel` -> `Schema`
- `check_structure` -> `verify`
- `expand_feature_delays` -> `lag`
- `check_validation_rules` -> `validate`

## `scmle.core.compiler`

- `WorkflowStage` -> `Node`

## `scmle.core.ingest_stage`

- `read_view` -> `load`

## `scmle.core.limits`

- `kill_descendants` -> `reap`
- `memory_exceeded` -> `exceeds`
- `enforcement_text` -> `explain`

## `scmle.core.orchestrator`

- `load_history` -> `recall`
- `apply_plan` -> `adopt`
- `load_checkpoints` -> `restore`

## `scmle.core.randomness`

- `validate_key` -> `vet`

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

## `scmle.data.admissibility.asof`

- `RowStatus` -> `Status`
- `AsOfAudit` -> `Audit`

## `scmle.data.admissibility.dag`

- `require_nodes` -> `require`

## `scmle.data.admissibility.leakage`

- `SkillTest` -> `Skill`
- `decision_for` -> `decide`
- `ConditionalTest` -> `Conditional`
- `MissingnessTest` -> `Missingness`
- `FeatureAssessment` -> `Assessment`

## `scmle.data.admissibility.qualify`

- `causal_admissibility` -> `admit`

## `scmle.data.admissibility.rules`

- `timing_failure` -> `diagnose`
- `gate_of` -> `gate`

## `scmle.data.admissibility.screen`

- `bootstrap_v` -> `resample`
- `empty_report` -> `abstain`
- `build_problem` -> `encode`
- `limit_report` -> `halt`
- `finding_bound` -> `floor`

## `scmle.data.catalog`

- `frame_shape` -> `measure`
- `canonical_scalar` -> `normalise`

## `scmle.data.charter.causal`

- `check_assumption` -> `vet`
- `check_attestors` -> `reconcile`

## `scmle.data.charter.contracts`

- `check_decision_map` -> `cover`

## `scmle.data.charter.decision`

- `positive_number` -> `check`

## `scmle.data.curation.preprocessing`

- `split_flagged` -> `partition`
- `split_high_cardinality` -> `isolate`
- `with_weights` -> `weigh`

## `scmle.data.custody.pipeline`

- `empty_carve` -> `forgo`

## `scmle.data.detectors.certificates`

- `DefectOutcome` -> `Outcome`

## `scmle.data.detectors.labels`

- `NoiseRate` -> `Rate`
- `mean_margins` -> `integrate`

## `scmle.data.detectors.missingness`

- `row_patterns` -> `group`
- `expect_block` -> `expect`
- `little_statistic` -> `measure`
- `oof_linear_scores` -> `crossfit`
- `TippingPoint` -> `Sensitivity`

## `scmle.data.detectors.sentinels`

- `SentinelFinding` -> `Suspect`
- `numeric_spikes` -> `spot`
- `text_findings` -> `scour`

## `scmle.data.framing.feasibility`

- `BaselineFeasibility` -> `Verdict`

## `scmle.data.identification.did`

- `pretrend_p_value` -> `pretest`

## `scmle.data.identification.overid`

- `two_stage` -> `estimate`

## `scmle.data.identification.sensitivity`

- `OlsFit` -> `Regression`

## `scmle.data.ingest.budget`

- `StopBoundary` -> `Stop`

## `scmle.data.ingest.declared`

- `source_totals` -> `compare`

## `scmle.data.ingest.joins`

- `distinct_of` -> `count`
- `kinds_match` -> `match`
- `measure_link` -> `measure`
- `resolve_competition` -> `arbitrate`

## `scmle.data.ingest.plan`

- `write_artifact` -> `persist`
- `infer_grain` -> `infer`
- `discover_joins` -> `link`

## `scmle.data.ingest.profile`

- `ConsistentRead` -> `Reading`

## `scmle.data.ingest.reconcile`

- `ExtractReconciliation` -> `Report`

## `scmle.data.ingest.synthesize`

- `candidate_columns` -> `sift`
- `reduce_numeric` -> `aggregate`

## `scmle.data.ledger.bivariate`

- `complete_pairs` -> `prune`

## `scmle.data.ledger.confirm`

- `EventLog` -> `Log`
- `unverified_claims` -> `disclaim`
- `withdrawn_claims` -> `withdraw`
- `run_confirmation` -> `verify`
- `open_and_test` -> `unseal`
- `family_block` -> `block`
- `failure_reason` -> `phrase`

## `scmle.data.ledger.ledger`

- `SurvivorSet` -> `Selection`
- `survivors_hash` -> `fingerprint`
- `record_not_tested` -> `abstain`
- `commit_result` -> `settle`

## `scmle.data.ledger.online_fdr`

- `RuleInvariantError` -> `InvariantError`
- `require_p` -> `read`

## `scmle.data.modalities.audio`

- `wav_metadata` -> `parse`
- `wav_signal_features` -> `extract`

## `scmle.data.modalities.curation`

- `check_governance` -> `govern`

## `scmle.data.modalities.document`

- `collect_quality_findings` -> `inspect`
- `collect_risk_findings` -> `assess`
- `format_for` -> `sniff`
- `read_pdf` -> `extract`
- `default_document_id` -> `fingerprint`
- `language_heuristic` -> `detect`
- `provenance_for_content` -> `segment`

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

## `scmle.data.modalities.graph`

- `GraphTemporalError` -> `TemporalError`
- `canonical_event_time` -> `normalize`
- `PreparedFeatures` -> `Features`
- `edge_key` -> `canonicalize`
- `edge_key_counts` -> `tally`

## `scmle.data.modalities.image`

- `perceptual_hash` -> `fingerprint`
- `validate_limits` -> `validate`

## `scmle.data.profiling.label_noise`

- `LabelNoise` -> `Noise`

## `scmle.data.profiling.missingness`

- `MissingnessFinding` -> `Finding`

## `scmle.data.profiling.profiler`

- `summarize_target` -> `summarize`
- `role_columns` -> `exclude`
- `inspect_columns` -> `inspect`
- `datetime_nodes` -> `derive`
- `describe_associations` -> `describe`

## `scmle.data.profiling.representativeness`

- `ColumnGap` -> `Gap`

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

## `scmle.data.readers.incremental`

- `EventSource` -> `Source`

## `scmle.data.readers.registry`

- `SchemaCompatibility` -> `Compatibility`

## `scmle.data.readers.streaming`

- `watermark_value` -> `advance`
- `CommitReceipt` -> `Receipt`
- `JsonLinesChunks` -> `Ndjson`
- `keep_smallest` -> `retain`

## `scmle.data.readers.tabular`

- `CsvReader` -> `Csv`
- `JsonReader` -> `Json`
- `JsonLinesReader` -> `Jsonl`
- `ExcelReader` -> `Excel`
- `ParquetReader` -> `Parquet`
- `FeatherReader` -> `Feather`

## `scmle.data.robustness.forward`

- `ForwardFinding` -> `Finding`
- `evaluate_finding` -> `evaluate`

## `scmle.data.robustness.multiverse`

- `GridRun` -> `Sweep`

## `scmle.data.robustness.shift`

- `check_pair` -> `align`
- `FeatureShift` -> `Drift`
- `categorical_test` -> `contrast`

## `scmle.data.robustness.stability`

- `estimate_spec` -> `estimate`
- `CurveTest` -> `Curve`
- `curve_statistics` -> `summarize`

## `scmle.data.splitting.splitters`

- `to_ids` -> `tuplify`
- `require_rows_per_class` -> `require`
- `MultilabelStratifiedSplit` -> `Multilabel`
- `label_matrix` -> `stack`
- `PresetSplit` -> `Preset`
- `keep_duplicates_together` -> `gather`

## `scmle.data.store.objectstore`

- `check_key` -> `validate`
- `meta_of` -> `describe`
- `create_exclusive` -> `claim`

## `scmle.data.store.processing`

- `run_locked` -> `sweep`

## `scmle.data.store.quality`

- `dataset_finding` -> `judge`
- `row_finding` -> `pinpoint`
- `marker_key` -> `locate`

## `scmle.data.validation.columns`

- `usable_rows` -> `align`
- `encoded_with` -> `encode`

## `scmle.decision.anytime`

- `rescaled_null` -> `orient`

## `scmle.decision.constraints`

- `SatisfactionTest` -> `Satisfaction`
- `largest_gap` -> `spread`
- `GroupThresholds` -> `Thresholds`

## `scmle.decision.netbenefit`

- `NetBenefit` -> `Benefit`

## `scmle.decision.ope`

- `ValueEstimate` -> `Estimate`
- `EffectThreshold` -> `Cutoff`

## `scmle.decision.threshold`

- `CalibrationCheck` -> `Calibration`

## `scmle.deploy.base_image`

- `fetch_registry_digest` -> `query`

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

## `scmle.deploy.gate`

- `model_problems` -> `veto`
- `staging_problems` -> `audit`

## `scmle.deploy.livetests`

- `check_health` -> `probe`
- `check_schema` -> `inspect`
- `check_security` -> `breach`

## `scmle.deploy.mocks`

- `server_file` -> `append`

## `scmle.deploy.vuln_gate`

- `active_entries` -> `prune`

## `scmle.evaluation.backtest`

- `per_step_errors` -> `score`

## `scmle.evaluation.coverage_confirm`

- `evaluate_document` -> `assess`

## `scmle.evaluation.cv_plus`

- `ScorePlan` -> `Plan`

## `scmle.evaluation.diagnostics`

- `MulticlassCalibration` -> `Reliability`

## `scmle.evaluation.dirichlet`

- `log_probabilities` -> `featurise`

## `scmle.evaluation.fit_health`

- `FitHealth` -> `Health`

## `scmle.evaluation.ipcw`

- `marginal_at` -> `survive`
- `cox_at` -> `survive`

## `scmle.evaluation.learning_curve`

- `LearningCurve` -> `Curve`

## `scmle.evaluation.longitudinal`

- `prepare_longitudinal_frame` -> `prepare`
- `fit_logistic_propensity` -> `regress`

## `scmle.evaluation.parameter_estimation`

- `EstimatorFit` -> `Fit`
- `check_estimation_request` -> `admit`
- `update_input_digest` -> `absorb`
- `prepare_rows` -> `prepare`
- `fit_linear` -> `regress`
- `center_influence` -> `center`
- `project_random_effect_covariance` -> `project`
- `bootstrap_replicate_estimate` -> `rerun`

## `scmle.evaluation.ranking`

- `query_groups` -> `group`
- `ranked_hits` -> `sort`

## `scmle.evaluation.replay`

- `PredictionReplay` -> `Replay`

## `scmle.evaluation.selection`

- `tie_key` -> `tiebreak`

## `scmle.evaluation.sensitivity`

- `spread_of` -> `summarise`

## `scmle.evaluation.sequential`

- `prepare_sequential_frame` -> `prepare`
- `ordered_periods` -> `order`
- `build_matrices` -> `tabulate`
- `fit_propensity` -> `regress`
- `build_regime_estimate` -> `average`

## `scmle.evaluation.slices`

- `slice_score` -> `grade`

## `scmle.experimentation.stores`

- `add_column` -> `alter`

## `scmle.experimentation.strategy_search`

- `search_model` -> `tune`

## `scmle.experimentation.trials`

- `access_record` -> `summarise`

## `scmle.experimentation.tuning`

- `promising_candidates` -> `shortlist`
- `tune_model` -> `optimize`
- `run_bracket` -> `ascend`
- `survivor_ids` -> `promote`

## `scmle.explainability.counterfactual`

- `candidate_values` -> `propose`
- `scale_of` -> `measure`

## `scmle.explainability.interaction`

- `sort_key` -> `rank`

## `scmle.explainability.quality`

- `validate_score` -> `validate`
- `QualityEvaluation` -> `Evaluation`
- `GroundTruthEvaluation` -> `Comparison`
- `GroundTruthSuiteEvaluation` -> `Suite`
- `explanation_ranking` -> `rank`

## `scmle.features.feature_graph`

- `IsMissing` -> `Missingness`
- `DatePart` -> `Calendar`
- `RollingStd` -> `Volatility`
- `get_op` -> `lookup`
- `entity_keys` -> `identify`
- `compute_with_history` -> `backfill`

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

## `scmle.governance.envelope`

- `signature_ok` -> `verifies`

## `scmle.governance.escalation`

- `RiskCoverage` -> `Curve`
- `ChowCheck` -> `Comparison`

## `scmle.governance.keyring`

- `can_verify` -> `verifies`

## `scmle.governance.lifecycle`

- `note_rows` -> `tally`

## `scmle.governance.local_faults`

- `stop_child` -> `halt`
- `write_events` -> `accept`

## `scmle.governance.loop`

- `ReplyOutcome` -> `Consequence`
- `json_ready` -> `jsonify`
- `build_monitors` -> `arm`
- `load_state` -> `restore`
- `process_batch` -> `feed`
- `update_challengers` -> `challenge`
- `time_out` -> `expire`

## `scmle.governance.release_state`

- `record_event` -> `log`
- `last_ledger_event` -> `recall`
- `recorded_findings` -> `compare`

## `scmle.inference.modality_serving`

- `new_server_request_id` -> `mint`
- `RejectedRequestError` -> `RejectedError`
- `request_identity` -> `identify`
- `admit_request` -> `screen`
- `read_payload` -> `receive`

## `scmle.inference.novelty`

- `far_out_fences` -> `fence`

## `scmle.inference.onnx_export`

- `register_booster_converters` -> `enrol`
- `is_token_imputer` -> `imputes`

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

## `scmle.intelligence.fingerprint`

- `missingness_patterns` -> `summarise`

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

## `scmle.models.deeplearning.architectures`

- `MlpArchitecture` -> `Mlp`
- `ResidualArchitecture` -> `Residual`
- `EnsembleArchitecture` -> `Ensemble`
- `TransformerArchitecture` -> `Transformer`
- `SmoothClip` -> `Saturation`

## `scmle.models.deeplearning.contracts`

- `CheckpointContract` -> `Checkpoint`

## `scmle.models.deeplearning.engines`

- `UpdatePlan` -> `Plan`
- `plan_updates` -> `apportion`
- `training_arrays` -> `prepare`
- `set_learning_rate` -> `adjust`
- `check_options` -> `vet`
- `new_network` -> `construct`

## `scmle.models.deeplearning.estimators`

- `TabPfnEstimator` -> `TabPfn`
- `KerasEstimator` -> `Keras`

## `scmle.models.deeplearning.jepa_engine`

- `JepaNetwork` -> `Network`
- `validated_matrix` -> `validate`
- `check_hyperparameters` -> `vet`
- `train_batch` -> `learn`

## `scmle.models.deeplearning.keras_engine`

- `load_keras` -> `load`
- `build_network` -> `build`

## `scmle.models.documents`

- `mean_characters` -> `measure`

## `scmle.models.ensembles`

- `GreedyBlender` -> `Blender`
- `members_of` -> `enlist`

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

## `scmle.models.multitarget.adapters`

- `MultilabelLogistic` -> `Logistic`
- `MultiForest` -> `Forest`

## `scmle.models.multitarget.estimators`

- `NativeMultiOutput` -> `Ensemble`

## `scmle.models.multitarget.wrapper`

- `MultiTargetEstimator` -> `Multitarget`

## `scmle.models.ordinal.estimators`

- `level_probabilities` -> `apportion`
- `best_scale` -> `tune`

## `scmle.models.sklearn.classical`

- `SupportVector` -> `Svm`

## `scmle.models.survival.estimators`

- `interval_of` -> `locate`

## `scmle.models.weighting`

- `WeightedRegressor` -> `Regressor`

## `scmle.monitoring.alerts`

- `AlertDeliveryError` -> `DeliveryError`
- `AlertDelivery` -> `Delivery`
- `AlertSink` -> `Sink`
- `successful_alerts` -> `delivered`

## `scmle.monitoring.events`

- `enforce_retention_locked` -> `prune`

## `scmle.monitoring.incidents`

- `event_digest` -> `digest`
- `require_record` -> `require`

## `scmle.monitoring.labels`

- `RandomizationCheck` -> `Uniformity`

## `scmle.monitoring.monitor`

- `model_output` -> `infer`

## `scmle.monitoring.observability`

- `ObservabilityEvaluation` -> `Evaluation`

## `scmle.monitoring.reference`

- `DistributionReference` -> `Reference`
- `reference_from_dict` -> `rebuild`

## `scmle.monitoring.sequential_tests`

- `start_term` -> `spawn`

## `scmle.operations.continuous`

- `canonical_csv` -> `serialize`
- `pending_id` -> `find`
- `start_challenger` -> `train`

## `scmle.operations.controller`

- `authorized_retrain` -> `train`

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

## `scmle.operations.metrics`

- `render_labels` -> `stringify`
- `merged_histogram` -> `merge`
- `render_histogram` -> `tabulate`

## `scmle.optimization.optimizers`

- `suggest_choice` -> `choose`
- `GridOptimizer` -> `Grid`

## `scmle.runners.base`

- `cost_of` -> `estimate`
- `confirm_stopped` -> `confirm`

## `scmle.runners.local`

- `JobStoppedError` -> `StoppedError`

## `scmle.runners.reproducibility`

- `for_metric` -> `lookup`

## `scmle.sampling.pseudo_labeling`

- `SelfTrainingOutcome` -> `Outcome`

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

## `scmle.security.access`

- `from_environment` -> `load`

## `scmle.security.tokens`

- `int_from_segment` -> `unpack`
- `check_window` -> `validate`
- `optional_float` -> `coerce`
- `parse_object` -> `parse`
- `check_claims` -> `admit`
- `number_claim` -> `quantify`
- `scopes_of` -> `collect`

## `scmle.trail.inputs`

- `section_payload` -> `cover`
- `missing_fields` -> `lack`

## `scmle.trail.trace`

- `check_charter_signature` -> `verify`
- `report_step` -> `recompute`
- `reconcile_feed` -> `reconcile`
- `outcomes_step` -> `observe`

## `scmle.workflow.definition`

- `check_name` -> `vet`

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

## `scmle.workflow.schedule`

- `last_slot` -> `load`
