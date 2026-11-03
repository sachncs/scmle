# What a run writes

Every `MLProject.run()` (or `scmle run`) creates `runs/run_<date>_<n>/` under
the output directory. Nothing is hidden in memory: this directory is the
evidence for the result and is what `resume` and `verify_replay` read.
Sibling file `runs/experiments.sqlite` tracks run and trial rows.

Start with `report.html`, then `diagnostics/decision_log.json`.

## Read first

| File | Contents |
| --- | --- |
| `report.html` | Human-readable summary: leaderboard, test result, diagnostics. |
| `leaderboard.parquet` | Every candidate with status and cross-validation score. |
| `evaluation.json` | The one-time locked-test result and confidence intervals. |
| `diagnostics/decision_log.json` | The evidence behind each choice (what was included, rejected, deferred). |
| `events.jsonl` | Append-only state transitions and audit events (`split.created`, `selection.frozen`, `test.opened`, `policy.*`). |

## Configuration and inputs

| File | Contents |
| --- | --- |
| `config.json` | The complete project specification; `scmle resume` and `scmle run --config` read the same format. |
| `execution_plan.json` | The plan: resolved task, metric, validation strategy, candidate models. |
| `dataset_manifest.json`, `dataset_profile.json`, `dataset_card.json` | Row and column counts, hashes, resolved column kinds and findings. |
| `source_registry.json` | Source identities and the snapshots seen. |
| `curation_report.json` | What curation found and changed (missingness, duplicates, drift, representativeness). |
| `validation_manifest.json` | Which rows went to development folds and to the locked test set. |
| `analysis_charter.json`, `error_budget.json`, `feasibility.json` | With `--charter`: the verified charter (hash and optional signature), its error budget, and whether the confirmation set supports the promised power. |
| `split_manifest.json` | With `--charter`: the exploration, confirmation, embargo and final-test ids, isolation level per sealed set, and a full 64-hex SHA-256 over all of it. Version 1 manifests (`validation_manifest.json`) stay readable. |
| `custody_log.jsonl` | With `--charter`: a hash-chained record of every seal, token issue, open and refused attempt. It holds ids and counts, never row values. |
| `admissibility_table.json` | With `--charter` and a predictive estimand: one decision per feature (`admit`, `admit_flagged`, `quarantine`) with tier and reasons. Fields carry no declared provenance in a charter, so they are decided in permissive mode; a field is quarantined only by a leakage suspicion that a current, passing certificate lets gate. Quarantined fields leave the feature plan (`dropped`). |
| `detector_certificates.json` | With `--charter`: the F5 certificate store the run used, copied from `benchmarks/qualification/`. It is read from a source checkout; when absent, every detector is descriptive and the `admissibility.completed` event says so. |

## Search and results

| Path | Contents |
| --- | --- |
| `trials/trial_*.json` | One record per trial; finished trials are reused on resume. |
| `diagnostics/*.json` | Calibration, thresholds, classification report, error analysis, robustness, slice performance, adversarial validation, fairness, permutation importance, partial dependence, recommendations. |
| `diagnostics/data_access.json` | What every fitted operation could see, verified against the split. |
| `explanations/` | Permutation importance, partial dependence and pairwise interaction as JSON and HTML. |

## The model

`model/` is the signed, checksummed bundle:
`pipeline.joblib` (fitted preprocessing and estimator), `schema.json`,
`reference.json` (monitoring reference profile), `conformal.json`,
`novelty.joblib` and `model_manifest.json` (checksums and signature).
Bundles are pickle-based; read the loading rules in [`SECURITY.md`](SECURITY.md).
Use `result.save(directory, signing_key=...)` to export one for deployment.

## Provenance and governance

| File | Contents |
| --- | --- |
| `provenance.json` | Run id, software version, config hash, dataset snapshot. |
| `release_attestation.json`, `evidence_graph.json` | Digests binding the bundle, provenance and evidence together. |
| `model_card.json` | Intended use, metrics, limitations. |
| `release_evaluation.json` | Whether declared success criteria were met. With none declared it reports `policy_not_declared` and blocks; thresholds are never invented from observed metrics. |
| `sbom.json`, `environment.lock` | Dependency inventory and the exact installed versions. |
| `capability_registry.json` | Model families available, their extras and resource needs. |
| `fault_injection_plan.json` | Failure scenarios and the evidence each needs ([`OPERATIONS_REHEARSAL.md`](OPERATIONS_REHEARSAL.md)). |
| `parameter_estimation_declaration.*` | The declaration (or its absence) for parameter-estimation runs. |

## Cleanup

`runs/` is gitignored and safe to delete once you no longer need to resume or
replay a run. Keep the directory of any run whose model you deployed.
