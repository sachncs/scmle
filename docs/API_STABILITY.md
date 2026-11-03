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

The project will not claim a stable v1 API until its release, compatibility,
security, and operational evidence is complete.

## Bundle migration

Bundle format 2 authenticates `schema.json`. Format 1 bundles are refused by
default. To migrate a format 1 bundle from a source you explicitly trust,
load it with `trusted=True, allow_legacy_format=True` and immediately call
`save()` to write a format 2 bundle. The migrated bundle should then be
verified with the normal signed loading path.
