# API stability

The project is currently `0.1.0b1`. The supported top-level Python API is the
surface exported by `scml.__all__`:

* `MLProject`, `RunResult`, and `ModelBundle` for lifecycle and artifacts;
* `load_model` and `predict` for inference;
* `register_estimator`, `unregister_estimator`, and `register_metric` for
  extension.

The top-level exports are covered by `tests/unit/test_api_surface.py`.
Removing or changing one requires a changelog entry, a migration note, and a
deliberate versioning decision. Modules below `scml.core`, `scml.data`,
`scml.models`, and other package namespaces are extension surfaces for
advanced users, but their concrete internals are not stable until promoted to
the top-level API or documented as a contract.

The project will not claim a stable v1 API until the release, compatibility,
security, and operational gates in the production-readiness plan are green.
