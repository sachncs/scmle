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

The project will not claim a stable v1 API until its release, compatibility,
security, and operational evidence is complete.

## Bundle migration

Bundle format 2 authenticates `schema.json`. Format 1 bundles are refused by
default. To migrate a format 1 bundle from a source you explicitly trust,
load it with `trusted=True, allow_legacy_format=True` and immediately call
`save()` to write a format 2 bundle. The migrated bundle should then be
verified with the normal signed loading path.
