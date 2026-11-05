# API stability

The project is currently `0.1.0b1`. The supported top-level Python API is the
surface exported by `scmle.__all__`:

* `MLProject`, `RunResult`, and `ModelBundle` for lifecycle and artifacts;
* `load_model` and `predict` for inference;
* `register_estimator`, `unregister_estimator`, and `register_metric` for
  extension.

The top-level exports are covered by `tests/unit/test_api_surface.py`.
Removing or changing one requires a changelog entry, a migration note, and a
deliberate versioning decision. Modules below `scmle.core`, `scmle.data`,
`scmle.models`, and other package namespaces are extension surfaces for
advanced users, but their concrete internals are not stable until promoted to
the top-level API or documented as a contract.

## Qualification types

The data-qualification records and helpers live below the pinned top-level API
and are extension surfaces until promoted: `scmle.contracts.qualification`
(`Check`, `Claim`, `Certificate`, `Tier`, `Decision`, `SetId`),
`scmle.data.ledger.budget` (`ErrorBudget` and the budget functions) and
`scmle.core.randomness` (keyed random streams). Rule: new qualification names
are added without touching any entry in `tests/fixtures/public_api.json`, and
a name is promoted to `scmle.__all__` only deliberately, with a changelog
entry and a regenerated snapshot (`scripts/check_api_compatibility.py
--write`). `tests/unit/test_qualification_api_rule.py` enforces both halves.

One deliberate change to the pinned surface: `MLProject` gained the optional
keyword `charter` (default `None`, so every existing call is unchanged) and the
snapshot was regenerated for that single entry. The charter and custody
modules (`scmle.data.charter`, `scmle.data.custody`) are extension surfaces
like the other qualification modules.

## Leakage screen

The profiler's fixed leakage cutoffs (`LEAK_ASSOCIATION`, `MISSINGNESS_LEAK`,
the `leakage_findings` method and the `"leakage"` string match that drove
`leakage_policy="block"`) are gone. `leakage_policy` keeps its values and its
meaning at the top level; its default changed from `"warn"` to `"block"`
(see the changelog), so the `MLProject` signature in
`tests/fixtures/public_api.json` was regenerated once for that default; what blocks is now a typed
result, `DatasetProfile.blocking_leaks`, from the permutation screen in
`scmle.data.admissibility.screen`. Behaviour changes that users may notice:
blocking needs a passing certificate for the mechanism (binary
classification only; regression and multiclass targets get warnings with
p-values), the screen also reports a value recorded only for some outcomes
when most rows are blank, and `DatasetProfile` gained the optional field
`leakage` (default empty) and its `to_dict` key. The fingerprint keeps its
keys: `target_association` is still computed as a descriptive statistic with
no cutoff, and `risk.leakage_hypotheses` now counts screen suspects instead of
cutoff hits, so stored fingerprints of data that crossed a retired cutoff can
differ in that one count.

`MLProject` gained the optional keyword `leakage_attestations` (default
`None`, so every existing call is unchanged), `ProjectSpec` the field of the
same name, and the CLI the flag `--leakage-attestations`; the snapshot in
`tests/fixtures/public_api.json` was regenerated for the `MLProject` entry
alone. In a charter run the leakage test is now the screen above instead of
the earlier cross-fitted skill scan, so a charter run and a run without one
flag the same columns, and a certified suspect is quarantined unless a valid
attestation keeps it.

`MLProject` gained the optional keywords `baseline_column` and `framing_draft`
(both default `None`, so every existing call is unchanged), `DataSpec` the
fields of the same names, and the CLI the flags `--baseline-column` and
`--framing-draft` on `plan`, `run` and `resume`; `tests/fixtures/public_api.json`
was regenerated for the `MLProject` entry alone. `scmle claims list` prints
claims registered by hash after the bivariate ones, and `scmle claims verify`
gained one step, present only when a claim was registered.

`MLProject` gained the optional keyword `ingest_plan` (default `None`),
`DataSpec` the field of the same name, and the CLI the flag `--ingest-plan`;
the charter gained the optional field `ingest` (format 7) and a charter
without it keeps its hash. `tests/fixtures/public_api.json` was regenerated
for the `MLProject` entry alone.

`MLProject` gained the optional keywords `logged_action_column` and
`propensity_column` (default `None`), `DataSpec` the fields of the same names,
and the CLI the flags `--logged-action-column` and `--propensity-column`;
`tests/fixtures/public_api.json` was regenerated for the `MLProject` entry
alone. A charter run with a decision block now also writes `decision_report.json`,
`policy_value.json` and, with a rollout plan, `rollout_design.json` and
`rollout_decisions.jsonl`.

Uncertainty (F12). `ModelBundle.predict_set` gained the optional argument
`conditional` (default `None`, the previous behaviour). `conformal.json` gained
`class_scores`; bundles saved earlier load and answer pooled queries, and
`conditional="class"` raises for them. `ForecastMetric` gained the field
`scale` and the kinds `mase_same_rows` and `mase_insample`. The metric named
`mase` is unchanged and is the same-rows score (the model's error divided by
the no-change error on the evaluation rows); it is not the usual MASE and is
not comparable with published values. Use `proper_scores.mase` or
`mase_insample` for that. The `interval_coverage` reliability check now
requires coverage of at least 0.88 instead of 0.85.

Closed loop (F13). No entry of the pinned surface changed.
`ModelMonitor.performance` gained the optional arguments `propensity` and
`maturity` (default `None`, the previous behaviour). With a propensity, the
labelled rows are resampled with probability proportional to `1 / propensity`
before the metrics are computed, a seeded approximation of a weighted metric;
with a maturity mask only matured rows count. The retraining step of
`LifecycleManager` now holds fresh rows out by time (the schema's time column)
or by whole groups when the champion's schema has them, instead of a random
shuffle; plain tables are shuffled as before. `scmle.monitoring.sequential_tests`,
`scmle.monitoring.labels`, `scmle.governance.escalation`,
`scmle.governance.deployment_ledger`, `scmle.governance.never_list`,
`scmle.governance.loop` and the command `scmle loop` are extension surfaces
like the other qualification modules. A `LoopPolicy` does not let an alarm
retrain unless `alarm_gates_retrain` is set, and it should be set only when
`python -m benchmarks.qualification --check loop` lists the detection-delay
gate H2 as passed for the policy's settings (today it is descriptive).
`ModelRegistryStore` gained the optional `guard` and `scope` arguments, the
method `attach_guard` and the optional `guard_detail` of `promote`; without a
guard behaviour is unchanged.

Causal identification (F10). No entry of the pinned surface changed.
`scmle.data.charter.contracts.AnalysisEstimand` gained the optional field
`causal` (`scmle.data.charter.causal.CausalSpec`, with `NegativeControl` and
`SensitivitySpec`); it is omitted from the record when unset, so charters of
format 1 to 3 keep their hash and signature, and a charter that carries it is
format 4 (an older wrapper carrying it is refused). `DecisionAction` gained the
member `refuse_claim`. `scmle.data.identification` (`select`, `aipw`, `did`,
`iv`, `rd`, `synth`, `sensitivity`, `battery`, `tier`, `run`, `certified`) is
an extension surface like the other qualification modules.

Decision layer (F11). `scmle.decision` and `scmle.data.charter.decision` are
extension surfaces like the other qualification modules; nothing was added to
`scmle.__all__` and `tests/fixtures/public_api.json` is unchanged. The charter
gained the optional field `decision` (format 5); a charter without it keeps
its hash. `choose_decision_threshold` takes the threshold from the charter's
cost model when the charter has a decision block and the task is binary, even
if `decision.threshold_objective` is `none`; without a block it behaves as
before. `evaluation.json` gained the keys `net_benefit` (null without a block)
and, inside `decision`, `f11`. `scmle.data.ledger.budget` gained
`DecisionBudget`, `assign_experiment` and `start_rollback_monitor`.

## Deployment contract

`scmle.deploy.contract` is an extension surface, not a top-level export. It
holds the typed spec every provider adapter must satisfy (`DeploymentSpec`
with `ImageRef`, `BundleRef`, `SecretRef`, `Resources`, `HealthSpec`),
`RuntimeReport`, `conformance_problems` and the `ProviderAdapter` base class.
Because it sits below `scmle.__all__`, no entry of `exports` in
`tests/fixtures/public_api.json` changed; the snapshot gained a separate
`extension_surfaces` block that pins the signature of each name above, and
`scripts/check_api_compatibility.py` reports `extension added`, `extension
removed` and `extension contract changed` for it. The spec format is
versioned (`SPEC_FORMAT`); `from_dict` refuses any other format.

## Security surface

`scmle.security.tokens`, `scmle.security.access` and `scmle.security.audit`
are extension surfaces like `scmle.deploy.contract`: token verification
(`TokenVerifier`, `KeySet`, `VerificationKey`, `RevocationList`, `Principal`),
the request gateway (`Gateway`, `RateLimiter`, `AccessDenied`) and the audit
log (`AuditLog`) are pinned in the snapshot's `extension_surfaces` block, and
no `exports` entry changed. Behaviour changes to existing classes:
`HttpServing` gained the keyword-only `gateway` and refuses a non-loopback
`host` without one; `ModalityHttpServing` kept its constructor, gained
`set_gateway`, and `start` refuses a non-loopback `host` without a gateway.
The deployment ledger gained the row kind `audit`.

## Governed-data surface

`scmle.data.store` (`objectstore`, `cloud`, `snapshot`, `quality`,
`processing`) and `scmle.core.retry` are extension surfaces pinned in the
snapshot's `extension_surfaces` block; no `exports` entry changed. Pinned
signatures avoid function-valued defaults (a `None` default is resolved in the
body), because a function's repr carries an address and would make the
snapshot differ between runs.

## Training runners

`scmle.runners` (`base`, `local`, `aws`, `gcp`, `roles`, `checkpoint`,
`reproducibility`) is an extension surface pinned like the others; the
entrypoint, project workload and fakes are not pinned.

## Workflow surface

`scmle.workflow.definition` (`Workflow`, `Step`, `StepKind`, `Schedule`,
`CatchUp`), `executor` (`LocalExecutor`), `schedule` (`Scheduler`) and `emit`
(the two emitters and validators) are extension surfaces pinned like the
others.

## Infrastructure and release surfaces

`scmle.deploy.iac_policy` (`check_tree`, `Violation`), `orphans` (`find_orphans`),
`gate` (`admit`, `certify`, `Candidate`, `GateDecision`, `ImageScan`,
`model_digest_of`), `record` (`DeploymentRecord`), `livetests` (`run_checks`,
`Target`, `CheckResult`, `Response`, `http_fetch`), `observation`
(`Observation`, `Traffic`), `control` (`ControlPlane`, `confirm`,
`expecting`, `ConfirmPolicy`), `ecs` (`EcsControlPlane`, `observe_services`),
`cloudrun` (`CloudRunControlPlane`, `observe_service`) and
`scmle.governance.release_state` (`ReleaseRegistry`, `ReleaseState`,
`Outcome`, `ReconcileReport`, `Finding`, `RuntimeDisagrees`) are extension
surfaces pinned like the others. The Terraform reader (`hcl`) and the release
manifest loader (`release`) are not pinned. The `deployment` row kind of the
deployment ledger and the `release_state.json` file are additions; existing
ledgers and registries load unchanged.

## Operations surfaces

`scmle.operations.signals` (`SignalMonitor`, `Signal`, `SignalThresholds`,
`WindowPolicy`, `Category`, `Status`), `classify` (`classify`,
`Classification`, `Verdict`), `notify` (`WebhookSender`, `Notifier`,
`Request`, `Delivery`, `sign`, `verify_signature`), `exporters`
(`CloudWatchExporter`, `CloudMonitoringExporter` and the two client
protocols) and `continuous` (`ContinuousEvaluation`, `RetrainPolicy`,
`RetrainDecision`, `MetricSpec`, `Replacement`, `TrainingHistory`,
`post_promotion_check`), `recovery` (`Fault`, `Objective`, `Measurement`,
`Result`, `Runner`), `spool` (`MetricsSpool`, `FlushResult`) and `claim`
(`check_claim`, `ClaimVerdict`, `claim_text_problems`) are extension surfaces
pinned like the others. The metrics
registry, the log writer, the cost report and the response desk are not
pinned.

The project will not claim a stable v1 API until its release, compatibility,
security, and operational evidence is complete.

## Bundle migration

Bundle format 2 authenticates `schema.json`. Format 1 bundles are refused by
default. To migrate a format 1 bundle from a source you explicitly trust,
load it with `trusted=True, allow_legacy_format=True` and immediately call
`save()` to write a format 2 bundle. The migrated bundle should then be
verified with the normal signed loading path.
