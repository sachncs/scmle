# API stability

The project is currently `0.1.0b1`. The package root exports nothing but
`__version__`: every public name is imported from the module that defines it,
for example `scmle.api.project.Project`. There is one name and one import path
for each thing.

The pinned names, by module, are:

* `scmle.api.project.Project`, `scmle.artifacts.results.Run` and
  `scmle.artifacts.bundle.Bundle` for lifecycle and artifacts;
* `scmle.api.predict.load_model` and `scmle.api.predict.predict` for inference;
* `scmle.api.clustering` (`Segmentation`, `cluster`, `evaluate`) for
  clustering, and `scmle.api.erasure` (`Forgetting`, `Retirement`,
  `Deletion`, `Subject`, `Receipt`, `RetiredError`) for erasure;
* `register` and `unregister` in `scmle.models.custom` and `register` in
  `scmle.evaluation.metrics`, for extension.

`tests/fixtures/public_api.json` (schema version 2) lists every pinned name
under its module with the digest of its signature, and
`scripts/check_api_compatibility.py` and `tests/unit/test_api_surface.py` check
it. Removing or changing a pinned name requires a changelog entry, a migration
note, and a deliberate versioning decision. Other names in `scmle.pipeline`,
`scmle.data`, `scmle.models` and the other packages are extension surfaces for
advanced users, but their concrete internals are not stable until they are
pinned or documented as a contract.

## Qualification types

The data-qualification records and helpers live below the pinned top-level API
and are extension surfaces until promoted: `scmle.contracts.qualification`
(`Check`, `Claim`, `Certificate`, `Tier`, `Decision`, `Subset`),
`scmle.assurance.ledger.budget` (`Alpha`, the error budget, and the budget functions) and
`scmle.runtime.randomness` (keyed random streams). Rule: new qualification names
are added without touching any entry in `tests/fixtures/public_api.json`, and
a name is pinned only deliberately, with a changelog entry (a review rule) and
a regenerated snapshot (`scripts/check_api_compatibility.py --write`).
`tests/unit/test_qualification_api_rule.py::test_no_previously_pinned_name_changed` enforces the first half, and `tests/unit/test_qualification_api_rule.py::test_new_types_are_not_package_root_names` that no qualification name is pinned at the package root.

One deliberate change to the pinned surface: `Project` gained the optional
keyword `charter` (default `None`, so every existing call is unchanged) and the
snapshot was regenerated for that single entry. The charter and custody
modules (`scmle.assurance.charter`, `scmle.assurance.custody`) are extension surfaces
like the other qualification modules.

## Leakage screen

The profiler's fixed leakage cutoffs (`LEAK_ASSOCIATION`, `MISSINGNESS_LEAK`,
the `leakage_findings` method and the `"leakage"` string match that drove
`leakage_policy="block"`) are gone. `leakage_policy` keeps its values and its
meaning at the top level (`Project(...)` and the snapshot in
`tests/fixtures/public_api.json` are unchanged); what blocks is now a typed
result, `Portrait.blockers`, from the permutation screen in
`scmle.assurance.admissibility.screen`. Behaviour changes that users may notice:
in a run every suspect may block, for binary, multiclass and regression
targets alike, whatever the state of the certificates, which are evidence
and decide nothing (an earlier version blocked only on a mechanism with a
passing certificate, which was binary classification alone), the screen
also reports a value recorded only for some outcomes
when most rows are blank, and `Portrait` gained the optional field
`leakage` (default empty) and its `to_dict` key. The fingerprint keeps its
keys: `target_association` is still computed as a descriptive statistic with
no cutoff, and `risk.leakage_hypotheses` now counts screen suspects instead of
cutoff hits, so stored fingerprints of data that crossed a retired cutoff can
differ in that one count.

`Project` gained the optional keyword `leakage_attestations` (default
`None`, so every existing call is unchanged), `Config` the field of the
same name, and the CLI the flag `--attestations`; the snapshot in
`tests/fixtures/public_api.json` was regenerated for the `Project` entry
alone. In a charter run the leakage test is now the screen above instead of
the earlier cross-fitted skill scan, so a charter run and a run without one
flag the same columns, and a certified suspect is quarantined unless a valid
attestation keeps it.

`Project` gained the optional keywords `baseline_column` and `framing_draft`
(both default `None`, so every existing call is unchanged), `Dataset` the
fields of the same names, and the CLI the flags `--baseline` and
`--draft` on `plan`, `run` and `resume`; `tests/fixtures/public_api.json`
was regenerated for the `Project` entry alone. `scmle claims list` prints
claims registered by hash after the bivariate ones, and `scmle claims verify`
gained one step, present only when a claim was registered.

`Project` gained the optional keyword `ingest_plan` (default `None`),
`Dataset` the field of the same name, and the CLI the flag `--ingest`;
the charter gained the optional field `ingest` (format 7) and a charter
without it keeps its hash. `tests/fixtures/public_api.json` was regenerated
for the `Project` entry alone. Charter format 8 adds the optional
`entity` and `reconcile` sub-blocks of `ingest`
(`scmle.assurance.charter.ingest.Entity`, `Expectation`, extension
surfaces); a format 7 ingest charter keeps its hash and signature.
Charter format 9 adds the optional `subgroups` field of the causal block
(`scmle.assurance.charter.causal.Subgroup`, an extension surface) and the
optional `attestors` and `attestations` fields
(`scmle.assurance.charter.causal.Attestor`, `scmle.causal.identification.attest`);
a format 4 causal charter keeps its hash and signature. The `subgroups` key
of `identification_report.json` gained `gate`, `statement_gate` and a per
hypothesis `certified_size`, and the summary in `causal_claims.json` a `gate`;
`scmle.causal.identification.certified.subgroup_gate` and `SubgroupGate` are
additions (extension surfaces).

Charter format 10 adds the optional `uncertainty` block
(`scmle.assurance.charter.uncertainty.Uncertainty`, an extension surface; the
snapshot is unchanged); a charter of format 9 or older keeps its hash and
signature.

Charter format 11 adds the optional `coverage_on_confirmation` field (stored
as `confirmatory` since Stage B) of the `uncertainty` block (an extension
surface, `scmle.assurance.charter.uncertainty.Uncertainty`; a format 10 charter
keeps its hash and signature, both pinned). `scmle.assurance.coverage_confirm`,
`coverage.undercovered`, `coverage_slices.flag` and `FlagGate` are additions (extension
surfaces); the per-slice records of `coverage_report.json` gained `flag`,
`flag_p_value` and `flag_certified`, and the slice summary `flagged`,
`flag_margin` and `flag_gate`. `claims verify` gained the step "coverage claim
on C: registered, recomputed, status matches". The snapshot is unchanged.

`scmle.governance.uncertainty_feed.UncertaintyFeed` is an addition (an
extension surface; the snapshot is unchanged).

`scmle.operations.rollout_hook.Hook` and
`Maintainer.revert` are additions (extension surfaces), as
are `Maintainer.delegate` (called by `Hook`; it makes
the bootstrap rules of `review` and `revertible` descriptive, and a
manager without a bound hook behaves as before) and
`Hook.review` and `Hook.revertible`; the
deployment ledger gained the row kinds `rollout_experiment`,
`rollout_assignment` and `rollout_decision`, and a ledger without them loads unchanged.

`Project` gained the optional keyword `baseline_subject` (`frozen_rule`,
the default, or `selected_model`), `Dataset` the field of the same name, and
`plan`, `run` and `resume` the flag `--comparator`; `tests/fixtures/public_api.json`
was regenerated for the `Project` entry alone.

`Project` gained the optional keywords `logged_action_column` and
`propensity_column` (default `None`), `Dataset` the fields of the same names,
and the CLI the flags `--action` and `--propensity`;
`tests/fixtures/public_api.json` was regenerated for the `Project` entry
alone. A charter run with a decision block now also writes `decision_report.json`,
`policy_value.json` and, with a rollout plan, `rollout_design.json` and
`rollout_decisions.jsonl`.

Uncertainty (F12). `Bundle.predict_set` gained the optional argument
`conditional` (default `None`, the previous behaviour). `conformal.json` gained
`conditional` (stored as `class_scores` before Stage B); bundles saved earlier
load and answer pooled queries, and `conditional="class"` raises for them. `ForecastMetric` gained the field
`scale` and the kinds `mase_same_rows` and `mase_insample`. The metric named
`mase` is unchanged and is the same-rows score (the model's error divided by
the no-change error on the evaluation rows); it is not the usual MASE and is
not comparable with published values. Use `proper_scores.mase` or
`mase_insample` for that. The `interval_coverage` reliability check now
requires coverage of at least 0.88 instead of 0.85.

Closed loop (F13). No entry of the pinned surface changed.
`Watcher.performance` gained the optional arguments `propensity` and
`maturity` (default `None`, the previous behaviour). With a propensity, the
labelled rows are resampled with probability proportional to `1 / propensity`
before the metrics are computed, a seeded approximation of a weighted metric;
with a maturity mask only matured rows count. The retraining step of
`Maintainer` now holds fresh rows out by time (the schema's time column)
or by whole groups when the champion's schema has them, instead of a random
shuffle; plain tables are shuffled as before. `scmle.monitoring.sequential_tests`,
`scmle.monitoring.labels`, `scmle.governance.escalation`,
`scmle.governance.deployment_ledger`, `scmle.governance.never_list`,
`scmle.governance.loop` and the command `scmle loop` are extension surfaces
like the other qualification modules. A `Preregistration` does not let an alarm
retrain unless `retrain` is set, and it should be set only when
`python -m benchmarks.qualification --check loop` lists the detection-delay
gate H2 as passed for the policy's settings (today it is descriptive).
`ModelRegistryStore` gained the optional `guard` and `scope` arguments, the
method `attach` and the optional `guard_detail` of `promote`; without a
guard behaviour is unchanged.

Causal identification (F10). No entry of the pinned surface changed.
`scmle.assurance.charter.contracts.Estimand` gained the optional field
`causal` (`scmle.assurance.charter.causal.Question`, with `Control` and
`Confounding`); it is omitted from the record when unset, so charters of
format 1 to 3 keep their hash and signature, and a charter that carries it is
format 4 (an older wrapper carrying it is refused). `Edict` gained the
member `refuse_claim`. `scmle.causal.identification` (`select`, `aipw`, `did`,
`iv`, `rd`, `synth`, `sensitivity`, `battery`, `tier`, `run`, `certified`) is
an extension surface like the other qualification modules.

Decision layer (F11). `scmle.decision` and `scmle.assurance.charter.decision` are
extension surfaces like the other qualification modules; none of their
names is pinned in `tests/fixtures/public_api.json`. The charter
gained the optional field `decision` (format 5); a charter without it keeps
its hash. `choose_decision_threshold` takes the threshold from the charter's
cost model when the charter has a decision block and the task is binary, even
if `decision.threshold_objective` is `none`; without a block it behaves as
before. `evaluation.json` gained the keys `net_benefit` (null without a block)
and, inside `decision`, `f11`. `scmle.assurance.ledger.budget` gained
`Decision` (then `DecisionBudget`), `enroll` and `watch` (then
`start_rollback_monitor`).

## Deployment contract

`scmle.deploy.contract` is an extension surface, not a top-level export. It
holds the typed deployment spec (`Deployment` with `ImageRef`,
`BundleRef`, `SecretRef`, `Resources`, `Probes`), `Runtime`,
`conformance_problems` and the `Provider` base class that a runtime
adapter written by a user implements.
The snapshot's `surfaces` block pins the signature of each name above, and
`scripts/check_api_compatibility.py` reports `surface added`, `surface
removed` and `surface contract changed` for it. The spec format is
versioned (`SPEC_FORMAT`); `from_dict` refuses any other format.

## Security surface

`scmle.security.tokens`, `scmle.security.access` and `scmle.security.audit`
are extension surfaces like `scmle.deploy.contract`: token verification
(`Verifier`, `Keystore`, `Key`, `RevocationList`, `Principal`),
the request gateway (`Gateway`, `RateLimiter`, `DeniedError`) and the audit
log (`Log`) are pinned in the snapshot's `surfaces` block. Behaviour changes to existing classes:
`HttpServing` gained the keyword-only `gateway` and refuses a non-loopback
`host` without one; `ModalityHttpServing` kept its constructor, gained
`protect`, and `start` refuses a non-loopback `host` without a gateway.
The deployment ledger gained the row kind `audit`.

## Governed-data surface

`scmle.data.store` (`objectstore`, `snapshot`, `quality`,
`processing`) and `scmle.runtime.retry` are extension surfaces pinned in the
snapshot's `surfaces` block. Pinned
signatures avoid function-valued defaults (a `None` default is resolved in the
body), because a function's repr carries an address and would make the
snapshot differ between runs.

Phase 6 renamed the pinned data-store names to single words, with no alias:

| Old name | New name |
|---|---|
| `snapshot.DataContract` | `snapshot.Contract` |
| `snapshot.ReadLog` | `snapshot.Log` |
| `snapshot.SnapshotReader` | `snapshot.Reader` |
| `snapshot.SnapshotRecord` | `snapshot.Record` |
| `quality.QualityGate` | `quality.Gate` |
| `quality.QualityGateError` | `quality.GateError` |
| `processing.IncrementalProcessor` | `processing.Processor` |

The fields of `processing.Processing` and `quality.Assessment` changed with
them; [RENAMES](RENAMES.md) lists every renamed member.

Phase 6 lane `models` renamed these pinned names, with no alias:

| Old name | New name |
|---|---|
| `parameter_estimation.BayesianPrior` | `parameter_estimation.Prior` |
| `parameter_estimation.ParameterEstimate` | `parameter_estimation.Estimate` |
| `parameter_estimation.estimate_parameters` | `parameter_estimation.infer` |
| `longitudinal.PeriodATE` | `longitudinal.Period` |
| `sequential.PeriodDiagnostics` | `sequential.Diagnostics` |
| `sequential.RegimeEstimate` | `sequential.Estimate` |
| `metrics.register_metric` | `metrics.register` |
| `custom.register_estimator` | `custom.register` |
| `custom.unregister_estimator` | `custom.unregister` |
| `modality.ModalitySampleDecoder` | `modality.Decoder` |

Phase 6 lane `pipeline` renamed these pinned names, with no alias:

| Old name | New name |
|---|---|
| `continuous.ContinuousEvaluation` | `continuous.Evaluation` |
| `continuous.RetrainDecision` | `continuous.Decision` |
| `continuous.RetrainPolicy` | `continuous.Policy` |
| `continuous.TrainingHistory` | `continuous.History` |
| `continuous.post_promotion_check` | `continuous.reassess` |
| `notify.WebhookSender` | `notify.Sender` |
| `notify.verify_signature` | `notify.verify` |
| `signals.SignalMonitor` | `signals.Monitor` |
| `signals.SignalThresholds` | `signals.Thresholds` |
| `spool.MetricsSpool` | `spool.Spool` |
| `modality_lifecycle.ExtensionCurator` | `modality_lifecycle.Extension` |
| `modality_lifecycle.LifecycleCancelledError` | `modality_lifecycle.AbortedError` |
| `definition.CatchUp` | `definition.Backfill` |
| `definition.StepKind` | `definition.Kind` |
| `definition.standard_workflow` | `definition.scaffold` |
| `executor.StepContext` | `executor.Context` |

The fields, methods and keywords of `Step`, `Workflow`, `Retries`, `Window`,
`Monitor`, `Thresholds`, `Sender`, `Delivery`, `Measurement`, `Outcome`,
`Runner`, `Extension` and `Predictor` changed with them;
[RENAMES](RENAMES.md) lists
every renamed member.

## Training runners

`scmle.runners` (`base`, `local`, `roles`, `checkpoint`,
`reproducibility`) is an extension surface pinned like the others; the
entrypoint, project workload and fakes are not pinned.

## Workflow surface

`scmle.workflow.definition` (`Workflow`, `Step`, `Kind`, `Schedule`,
`Backfill`), `executor` (`Executor`) and `schedule` (`Scheduler`) are
extension surfaces pinned like the others.

## Release surfaces

`scmle.deploy.gate` (`admit`, `certify`, `Candidate`, `GateDecision`, `ImageScan`,
`model_digest_of`), `record` (`Record`), `livetests` (`run_checks`,
`Target`, `CheckResult`, `Response`, `http_fetch`), `observation`
(`Observation`, `Traffic`), `control` (`ControlPlane`, `confirm`,
`expecting`, `Polling`) and `scmle.deploy.release_state` (`ReleaseRegistry`, `ReleaseState`,
`Outcome`, `Reconciliation`, `Finding`, `RuntimeDisagrees`) are extension
surfaces pinned like the others. The release manifest loader (`release`) is
not pinned. The `deployment` row kind of the
deployment ledger and the `release_state.json` file are additions; existing
ledgers and registries load unchanged.

## Operations surfaces

`scmle.operations.signals` (`Monitor`, `Signal`, `Thresholds`,
`Window`, `Category`, `Status`), `classify` (`classify`,
`Classification`, `Verdict`), `notify` (`Sender`, `Notifier`,
`Request`, `Delivery`, `sign`, `verify`) and `continuous` (`Evaluation`,
`Policy`, `Decision`, `Indicator`, `Replacement`, `History`,
`reassess`), `recovery` (`Fault`, `Objective`, `Measurement`,
`Result`, `Runner`) and `spool` (`Spool`, `Flush`) are extension
surfaces pinned like the others. The metrics
registry, the log writer, the cost report and the response desk are not
pinned.

## Trail surface

`scmle.trail` (`build`, `seal`, `verify`, `render`, `Report`, `Finding`) and
the command `scmle trail` are an extension surface pinned like the others.
The trail file format (`format` 1 in `trail.json`), the finding kinds and the
layout of `inputs.json` are documented in [TRAIL](TRAIL.md); a change to them
needs a new `format`. The submodules (`records`, `inputs`, `outcomes`,
`reports`, `trace`, `chain`) are not pinned.

The project will not claim a stable v1 API until its release, compatibility,
security, and operational evidence is complete.

## Bundle migration

Bundle format 5 is the current format. The single-word class names of
phase 6 (format 5), the package layering (format 4) and the class renames
before it (format 3) changed the module path and class name stored inside
every bundle, so a bundle written by an earlier build (format 4, 3, 2 or 1)
cannot be loaded or migrated: `Bundle.load` refuses it
with an error that says to retrain and save again. A bundle whose pickled
members name a module or class this build does not have is refused the same
way, with `ArtifactError`. The
`allow_legacy_format` option remains in the loader but can only read a format 1
bundle whose stored objects still carry the current class names, which no real
earlier bundle does; it is a candidate for removal.
