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
| `dataset_manifest.json`, `dataset_profile.json`, `dataset_card.json` | Row and column counts, hashes, resolved column kinds and findings. `dataset_profile.json` also holds `leakage`: the leakage screen's level, family size, suspects and the suspects that may block a run (only mechanisms with a passing certificate can). |
| `source_registry.json` | Source identities and the snapshots seen. |
| `curation_report.json` | What curation found and changed (missingness, duplicates, drift, representativeness). |
| `validation_manifest.json` | Which rows went to development folds and to the locked test set. |
| `analysis_charter.json`, `error_budget.json`, `feasibility.json` | With `--charter`: the verified charter (hash and optional signature), its error budget, and whether the confirmation set supports the promised power. |
| `split_manifest.json` | With `--charter`: the exploration, confirmation, embargo and final-test ids, isolation level per sealed set, and a full 64-hex SHA-256 over all of it. Version 1 manifests (`validation_manifest.json`) stay readable. |
| `custody_log.jsonl` | With `--charter`: a hash-chained record of every seal, token issue, open and refused attempt. It holds ids and counts, never row values. |
| `admissibility_table.json` | With `--charter` and a predictive estimand: one decision per feature (`admit`, `admit_flagged`, `quarantine`) with tier and reasons. Fields carry no declared provenance in a charter, so they are decided in permissive mode; a field is quarantined only by a leakage suspicion (the same screen as a run without a charter) that a current, passing certificate lets gate, and only unless a valid per-column attestation keeps it (tier `attested`, recorded as a ledger diagnostic). Quarantined fields leave the feature plan (`dropped`). |
| `detector_certificates.json` | With `--charter`: the F5 certificate store the run used, copied from `benchmarks/qualification/`. It is read from a source checkout; when absent, every detector is descriptive and the `admissibility.completed` event says so. |
| `stability_report.json` | With `--charter`: each exploration survivor re-run over the grid the charter declared (imputation, outlier rule, inclusion; other choices are listed as not run). Per finding: the specification curve (effect and bootstrap interval per specification), the verdict (`stable`, `fragile` or `not_evaluable`) with the numbers behind it, a permutation test of the whole curve, and which choice moves the estimate. Stability is relative to the declared set. A fragile verdict is a warning unless a current passing certificate for the chartered thresholds lets it gate. Only when the charter's `rules.stability_filter` is true and it can gate are fragile survivors withdrawn from the family tested on `C` (second stage; accounting under `stage_two` in `confirmed_claims.json` and the confirmation row of the ledger). The stability and shift results, including per-column shift p-values, are also written as wealth-free diagnostic rows of the ledger. |
| `shift_report.json` | With `--charter`: the exploration set against the deployment sample named by `reference`. Status is `pass` (no shift detected), `fail` or `unverifiable`; without a deployment sample it is `unverifiable`, never `pass`. Holds the classifier and kernel two-sample tests (permutation p-values), per-feature KS or chi-square under Benjamini-Hochberg, subgroup coverage for `sensitive_columns`, and the forward check, which a run reports as skipped because the final-test set is opened only by the final evaluation. Per-period drift is not implemented. |

| `framing_link.json` | With `--charter`: whether the charter's target, prediction moment and unit name real columns (`resolved`) or not (`unresolved`). Names are matched to column names only; free text that matches nothing is reported, never rejected, and nothing is gated on it. Reads column names, never values. |

### Raw-data ingestion (`scmle ingest`)

`scmle ingest --sqlite FILE --tables a,b` (read-only connection) or `IngestPlan` writes into `--output`:

| Path | Contents |
| --- | --- |
| `ingest_catalog.json` | Per table: rows and, per column, kind, non-null and distinct counts, minimum and maximum, all from aggregate queries run in the source, plus a fingerprint of those aggregates. `partial` when the budget stopped profiling. |
| `grain_report.json` | Minimal approximate keys per table with `g3`, null rate, `small_n` flag and ties, and the verdict on the charter unit (`match`, `finer`, `coarser`, `unknown`) with the columns it matched and any questions. A `coarser` or `unknown` verdict exits 2. |
| `join_graph.json` | Measured links with containment, cardinality, whether declared or name-supported, and the decision (`accepted`, `candidate`, `many_to_many`, `declared_violated`); candidates are listed as questions. |
| `ingest_budget.json` | Queries, rows scanned and seconds spent against the limits, source queries counted, and where each stage stopped. |

Entity resolution (`entity_resolution.json`), the feature manifest (`feature_manifest.json`) and the reconciliation (`reconciliation.json`, extended) come from `scmle.data.ingest.entities`, `synthesize` and `reconcile` and are written by the caller; a charter run does not produce them yet.

### Problem framing (`scmle frame`)

`scmle frame` reads a request, a source description and exploration rows only, and writes into `--output`:

| Path | Contents |
| --- | --- |
| `framing_draft.json` | The draft: the unsigned charter and its hash, the source of every field (`template`, `llm`, `data` or `human`), the data-side checks, the slots left as placeholders, and, if recorded drafts were replayed, the model id, prompt, raw samples, their hash, the ambiguity score and `tau`. The charter hash covers the charter, not how it was drafted. |
| `charter.json` | The draft charter, unsigned, readable by `scmle charter check` and signable by `scmle charter sign`. Absent when the draft is refused. |
| `decision_cost_model.json` | With `--cost-model`: the four costs and the plausible threshold range, each with its source. The same model is inside the charter. |
| `baseline_report.json` | With `--cost-model` and `--baseline-column`: the current process measured on the exploration rows (net benefit at five thresholds against treat-all and treat-none) and the selective-label finding. When the outcome is observed depending on the baseline decision the baseline's net benefit is withheld. The claim against the baseline on `C` is a library call (`BaselineGuard`), not part of the run yet. |
| `framing_questions.json` | Questions that change the verdict (with the tipping value), unknowns whose range leaves the verdict unchanged, and the open items. |
| `framing_refusal.json` | When refused: the reason (a named column the source lacks, a blocking data check, an unfillable critical slot, or LLM ambiguity above `tau`). |

No LLM client ships; `--replay` serves recorded drafts. Every LLM-filled field is `unverified` until a person signs the charter, and there is no calibrated `tau` or gold corpus yet, so a replayed LLM draft without `tau` is always refused.

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
`reference.json` (monitoring reference profile), `conformal.json` (pooled
and per-class calibration scores),
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
