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
meaning at the top level (`MLProject(...)` and the snapshot in
`tests/fixtures/public_api.json` are unchanged); what blocks is now a typed
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

The project will not claim a stable v1 API until its release, compatibility,
security, and operational evidence is complete.

## Bundle migration

Bundle format 2 authenticates `schema.json`. Format 1 bundles are refused by
default. To migrate a format 1 bundle from a source you explicitly trust,
load it with `trusted=True, allow_legacy_format=True` and immediately call
`save()` to write a format 2 bundle. The migrated bundle should then be
verified with the normal signed loading path.
