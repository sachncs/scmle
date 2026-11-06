# Completion audit

Written 2026-11-06 against `master` at `9d4bc04` (CI green on that commit,
run 42227276889). Scope: every `plan/phase-*.md` (6 to 18) and every
`product/features/*.md`. Each remaining open item is listed as the documents
state it, with a class and the evidence for it.

Classes:

* **A**: buildable here and verifiable here (local code, existing
  dependencies, this machine's tools: Python 3.11 and 3.14, OpenTofu, mutmut).
* **B**: needs outside people or data: labelled corpora, gold standards,
  reference implementations (R, MAPIE, scikit-survival), a red team,
  replication, a domain owner, a prospective head-to-head, clerical samples,
  real panels, external benchmarks, a reviewer other than the author, or
  something that exists only after a push (CI observation, a published
  release).
* **C**: needs an AWS or GCP account, a spending cap, an approver, or a
  container daemon / scanner that is not on this machine (the Docker daemon
  is not running here and `trivy`, `skopeo` and MinIO are not installed).
* **D**: needs a design decision from the owner (a version number, a new
  dependency, a protocol's thresholds, a business metric).

"Status" says what was done in this pass: `built` (commit named in
`CHANGELOG.md` Unreleased), `closed` (the item was already satisfied and the
document was stale), `open` (stays in the documents with its class letter).

## Plan statements that contradicted the code (fixed in this pass)

| Document | Statement | Reality |
|---|---|---|
| `plan/README.md` phase table | rows 7 to 12 say "run wiring open", "partly built", "in progress" | the charter runs for 8 to 13 are wired (`git log`: framing link, ingest stage, causal claim, decision files, coverage claim, loop feed); the rows now list only the open items with classes |
| `plan/phase-16` WP-16.1, `F14` item 3, `plan/evidence/phase-16/terraform-tool.json` | `fmt` and `validate` "never ran", Terraform "never run through terraform" | OpenTofu 1.13.5 passes `fmt -check` and `init -backend=false` + `validate` on both modules and all six environments (commits `44e6889`, `d5ab5ff`, `e552bb7`; lock files committed; CI job "infrastructure policy, fmt and validate" green); re-run in this pass. `plan`, `apply` stay C |
| `plan/phase-6` entry criterion 2, `plan/phase-7` entry criteria and header | "the uncommitted leakage screen (R1 A7)", "this phase is not yet in the table" | the screen is committed, certified, wired and in `setup.cfg`; Phase 7 is in the table |
| `F7` "State at the start" | mutation scope omits custody, charter, vault, screen; "screen once committed" | all four are in `setup.cfg` `source_paths`; survivors per module are in `tests/unit/MUTANTS_*.md` |
| `F7` W7 / Phase 6 | Windows 15 to 17 minutes of 20 | the cross-platform job has a 60 minute limit and Windows 3.11 now takes 23 minutes, 3.14 29 minutes (run 42227276889); the "under 14 minutes" acceptance fails and is open (B) |
| `R1` C11 | "not observed on other platforms" | the `cross-platform` and `minimal-dependencies` jobs, which run `tests/unit/test_custody_golden.py`, passed on Windows, macOS and Python 3.11 lowest dependencies in run 42227276889; C11 closed |
| `Phase 9` smaller items | "152 of 762 mutants survive" in `asof_view.py` and `synthesize.py` | `tests/unit/MUTANTS_framing_ingest.md` records 41 + 95 = 136 survivors of 769 scored mutants |
| `Phase 12` WP-12.7, `F12` | "open: cluster-bootstrap metric intervals" | the generic mechanism exists: `evaluation/diagnostics.py` `resample_rows` resamples whole groups or time blocks and `MetricUncertainty` uses it (`tests/unit/test_diagnostics.py`); what is open is feeding the entity column of a `known_entity_future` run into it and the real-panel check |
| `F12` acceptance 3 | "ModelBundle round trip preserves fold-model hashes; tamper fails" listed open | `tests/integration/test_cv_plus_bundle.py` `test_cv_plus_survives_saving_and_loading` and `test_a_changed_fold_model_file_is_refused` exist |
| `Phase 13` WP-13.5a | "`set_size` and `novel` columns are supplied by the caller today" | `governance/uncertainty_feed.py` `UncertaintyFeed` builds them from a classification bundle (`tests/unit/test_uncertainty_feed.py`); open: regression bundles and a certified `novelty_bound` |
| `R1` B10 | `curation_report.json` stale after quarantine | built in this pass (`stages.py` rewrites both files; `test_the_curation_report_and_card_list_the_final_feature_plan`) |

Claims of "built" that were spot-checked and found true (file, symbol or test
exists): the deployment contract, `scmle.security`, `scmle.data.store`,
`scmle.runners` (`LocalRunner`, `AwsContainerJobRunner`, `GcpJobRunner`),
`scmle.workflow`, `deploy/terraform`, `scmle.deploy.{iac_policy,orphans,gate,
record}`, `cloud-release.yml`, `scmle.governance.release_state`,
`scmle.operations`, `scripts/check_production_claim.py`, `scmle.trail`,
`scripts/trace_model.py`, `benchmarks/breadth/`, the twelve docs named in
phases 14 to 18, `plan/evidence/phase-14..18` (phase-17 `aws/` and `gcp/` hold
only a README, which is the stated state), `fairness`/`reweighing` code, the
CV+ and coverage-claim modules, `RiskAlarm`, `escalation`, `never_list`.

## Items

### Phase 6 / F7 (external validation and release)

| ID | Item as documented | Class | Evidence / reason |
|---|---|---|---|
| 6.1 | `benchmarks/qualification/PROTOCOL.md` committed and tagged `qualification-protocol-v1` before any result | D | file and tag do not exist; the claim, corpus rule, thresholds and stop rules are the owner's pre-registration |
| 6.2 | corpora: at least 15 datasets, 5 per defect family, SHA-256 verified | B | `corpora.json` entries are `candidate`; needs retrieval and licence checks of outside data |
| 6.3 | backtest (F7 W1), `backtest.py` | B | needs at least 15 real datasets with a time axis; no module exists |
| 6.4 | red team (F7 W2) | B | an analyst who did not write F1-F6 |
| 6.5 | independent replication (F7 W3) | B | a third party |
| 6.6 | TabArena campaign (F7 W4) | B | about 9 machine-days, AutoGluon baselines not in the dependencies, OpenML downloads; no result committed under `benchmarks/sota/` |
| 6.7a | R oracle fixtures `onlineFDR`, `knockoff`, `dagitty` (F7 W5) | B | R is not installed (`which R` empty) |
| 6.7b | mutation floor 80 with a committed `mutation_survivors.json` (F7 W6) | A | mutmut 3.8.0 is installed; scope is already in `setup.cfg`; per-module survivors exist in `tests/unit/MUTANTS_*.md`; the repo-wide run takes many hours, see Status |
| 6.7c | Windows jobs under 14 minutes on three consecutive master runs (F7 W7) | B | needs pushes and CI observation; currently 23 and 29 minutes |
| 6.8 | domain-owner pilot | B | a team that owns a dataset |
| 6.9-1 | version choice (`0.1.0b1` reuse or `0.2.0b1`) | D | owner's decision |
| 6.9-2 | changelog moved under the new version | D | follows 6.9-1 |
| 6.9-3 | annotated tag `v<version>` | D | follows 6.9-1; tagging is not pushing |
| 6.9-5 | `SECURITY.md` supported-versions text | D | follows 6.9-1 |
| 6.9-6 | `CITATION.cff` validated | A | file absent; `cffconvert` is not installed so a schema test is written instead |
| 6.9-7 | docs describe each artifact a charter run writes | A | `docs/ARTIFACTS.md` exists; a test that every file a charter run writes is named there is buildable |
| 6.9-8 | `scripts/release_preflight.sh` exits 0 | C | needs `npm ci` of the website, `uv sync --all-extras` on 3.14, pip-audit; not run here, CI covers its pieces |
| 6.9-9..12 | publish to PyPI, site snippet, operational rehearsal by a non-author, independent review | B | outside people and a PyPI project |
| 6.GNG | go/no-go rows 1 to 13 (signers other than the author) | B | statisticians, red team, replicating party, domain owner |

### Phase 7 / R1 (instrument residuals)

| ID | Item as documented | Class | Evidence / reason |
|---|---|---|---|
| R1-A5 | label-noise detectors: recall lower bounds below 0.80 on small tables | A | `detector_certificates.json` `f5-label-noise`; closure is a limit stated in `docs/ARTIFACTS.md` (type and table size that pass) with floors unchanged |
| R1-A6 | sentinel detector flags clean rows of natural tables (29/150, 34/178, 1797/1797); one-corpus certificates | A | `f5_corpora_measurements.json`; fix or limit the detector, re-issue its certificate with the same seeds and floors |
| R1-B2 | per-period drift with a permutation null and calibrated PSI | A | `robustness/report.py` says `not_run` |
| R1-B3 | forward check always skipped | D | `robustness/qualify.py` builds `forward.skipped(...)`; `F` is the final test set and opens once (`forward_check` opens its own source); letting the final evaluation hand its rows to the forward check, or running it as a separate step, changes who owns the single opening, which is the owner's custody decision |
| R1-B4 | Thresholdout not implemented | A | `confirm.py` raises `ConfigError`; closure is to delete the option from F3 and the docs or implement it |
| R1-B5 | resampling negative control for a feature-only finding family | A | blocked by B2 (needs a ledgered per-period drift family) |
| R1-B7 | multiverse covers three choices | A | `stability.py` `IMPLEMENTED`; needs the stability certificate re-issued with the same seeds and floors |
| R1-B8 | multiclass stability | A | `qualify.py`; a new pre-registered estimator and certificate |
| R1-B10 | `curation_report.json` stale after quarantine | A | **built** in this pass |
| R1-B12 | MinHash text near-duplicates | A | `duplicates.py:13` states it is not implemented |
| R1-B13 | instance-dependent label noise not modelled | A | no such plant in `inject.py`; closure by a plant and a measurement, or a stated limit |
| R1-B14 | static `eda_report.html` | A | no string `eda_report` in `scmle/` or `docs/` |
| R1-C11 | identical manifest hash on three platforms | closed | run 42227276889: Windows 3.11 and 3.14, macOS 3.11 and 3.14, lowest dependencies 3.11 all green, `test_custody_golden.py` included |
| R1-D1 | sample run signed by an independent holder | B | a person who holds their own key |
| R1-D2 | red-team log | B | the red-team analyst |
| R1-D4 | six oracles have no scmle producer (`chi2_contingency`, `fisher_exact`, `kendalltau`, `ks_2samp`, `mannwhitneyu`, `pearsonr`); `onlineFDR` and `knockoff` references pending R | A (producers) / B (R references) | `oracle_table.WIRED` has six of twelve; scmle code that computes chi-square, Fisher, KS and Mann-Whitney exists (`admissibility/independence.py`, `leakage.py`, `robustness/shift.py`, `profiling/missingness.py`); R is absent |
| R1-E1 | e-LOND venue (AISTATS 2024) | A | readable from the web; `online_fdr.py:16` |
| R1-E2 | C2ST, DeLong variance sources | B | DeLong et al. 1988 is not openly available |
| R1-E3 | Gretton et al. 2012 | A | JMLR, open access |
| R1-E4 | PCS, specification curve | A | open access |
| R1-E5 | Holm 1979, BH 1995, BY 2001 original statements | B | original papers are paywalled; textbook worked examples can be tested but are not the source |
| R1-acc | acceptance: a test that each row ID still exists in its named file | A | listed as "proposed" |

### Phase 8 / F8

| ID | Item | Class | Evidence / reason |
|---|---|---|---|
| F8-4 | gold corpus, at least 100 problems, 20 double-charted | B | domain experts |
| F8-5b | LLM adapter behind `refusal.SlotSampler`, risk-coverage curve and `tau` certificate | B | needs the gold corpus and a model account; the interface, `ReplaySampler` and `calibrate_tau` exist |
| F8-c7,c8 | gold-corpus gate and LLM-vs-deterministic paired bootstrap | B | gold corpus |
| F8-c9c | planted-omission test on gold requests | B | gold corpus (the synthetic version passes) |
| F8-c10 | charter hash identical on Python 3.11 and 3.14 | A | `tests/unit/test_framing_draft.py` compares only inside one interpreter; both interpreters are available here |
| F8-8a | `net_benefit` vs R `rmda`/`dcurves` and Vickers' code | B | R absent |
| F8-8b | Berger citation, 0.90 factor review, weaker selection recall, owner sign-off on cost range, outcome check | B | statistician, domain owner, deployed problem; the weaker-selection recall (outcome observed for 70 percent of rows with `b = 0`) is a simulation and is A |
| F8-sel | selective-label detector recall under weaker selection (simulation) | A | `framing_measurements.json` measured strong plants only |
| F8-w | charter `unit`, `deployment_population` free text only checked by column names | D | a charter-language decision |

### Phase 9 / F9

| ID | Item | Class | Evidence / reason |
|---|---|---|---|
| F9-1 | non-SQLite source in a charter run (DuckDB or another DB-API driver) | D | a new dependency or a second backend; `plan.py` `sqlite_factory` only |
| F9-2,4,5 | key, foreign-key and entity-resolution gates on declared-key and labelled ER corpora | B | outside corpora |
| F9-3 | entity resolution unlabelled in a run | B | a clerical sample |
| F9-5b | as-of reference rewritten by a person outside the project; data-engineering reviewer | B | people |
| F9-6 | serving-time feature logs compared with the view | B | a deployment |
| F9-14 | charter run identical on Python 3.11 and 3.14; non-SQLite source | A (interpreters) / D (source) | `tests/integration/test_ingest_charter_run.py` runs one interpreter |
| F9-m | `--check ingest` full measurement only by `ingest_certify --write` | A | by design; recorded |
| F9-mut | survivor bound in the build configuration for `asof_view.py`, `synthesize.py` (136 of 769 survive) | A | `setup.cfg` has no per-module bound |
| F9-bud | `many_to_many` and entity resolution do not charge the budget per step | A | `ingest/joins.py:418`, `entities.py:488` reference no meter |
| F9-ro | read-only is a textual allow-list and query-only session; a read-only role cannot be verified | C | needs a real database role |
| F9-w | wall-time stops repeat only with a fixed clock; budget unit is rows | D | DB-API reports no bytes; accepted limit |

### Phase 10 / F10

| ID | Item | Class | Evidence / reason |
|---|---|---|---|
| F10-7a | corpora NSW (CPS, PSID), STAR, ACIC, IHDP in `corpora.json` | B | retrieval, licence, hash |
| F10-7c | NSW comparison table, frozen `rdrobust`, `sensemakr`, `EValue`, `did`, `HonestDiD`, `Synth` outputs, real-covariate worlds | B | R absent, data absent |
| F10-aipw | re-measure `coverage_aipw` (boosted learner world missed its floor) | A | `f10_certificates.json`; a new pre-registration and run |
| F10-gold | `--f10-gold`, outside-party run, literature claims "from memory" (Callaway-Sant'Anna etc.) | B | people and papers |
| F10-11 | reports bit-identical on Python 3.11 and 3.14 | A | both interpreters available |
| F10-sub | subgroups: unequal or overlapping levels, strategies other than the difference in means | A | `subgroup_certify.py` covers equal levels only |
| F10-att | per-person or asymmetric attestation keys; revocation list; attestor identity check | D | needs `cryptography` (not a dependency) for asymmetric signing; the HMAC key is project-shared |
| F10-dag | `Dag.from_artifact` loads the DAG the charter names, no check it is the intended file | D | an owner's decision on what to check |
| F10-bal | balance check on randomized data with a measured false-alarm rate replacing the AUC rule in `uplift.py` | A | `evaluation/uplift.py` `check_propensity` |
| F10-seq | register `evaluation/sequential.py` as strategy `sequential_ipw` | A | not registered |
| F10-did | DiD: Rambachan-Roth conditional procedure, covariates, not-yet-treated group, simultaneous bands | A | `identification/did.py` |
| F10-iv | IV: Sargan/Hansen J, effective F for several instruments, exclusion sensitivity, battery | A | no such code |
| F10-rd | RD: MSE-optimal bandwidth, McCrary or Cattaneo-Jansson-Ma density test | A | `identification/rd.py:24` |
| F10-syn | synthetic control: predictors, battery | A | `identification/synth.py` |
| F10-sens | sensitivity: Cinelli-Hazlett conversions are from memory, Rosenbaum bounds | B | needs the papers (sensemakr output) for the conversions; Rosenbaum bounds are A but cut unless chartered |
| F10-nc | negative-control exposures beyond `backdoor_aipw` | A | `identification/run.py:1102` refuses others |

### Phase 11 / F11

| ID | Item | Class | Evidence / reason |
|---|---|---|---|
| F11-W2 | charter run runs no experiment; ledger rows only through `RolloutHook` | D | an experiment needs live units; the hook is built |
| F11-W3 | `fairness.py`, `mitigation.py` clients of `decision.constraints`; solve `reweighing`; cluster-aware satisfaction test | A | neither file mentions `constraints` |
| F11-W4 | Criteo, Open Bandit, R `dcurves`/`rmda`, OBP, outside reproduction of A6, B1, B7 | B | data, R, a person |
| F11-W5a | re-register and re-measure the net benefit band (A3 gate) | A | `f11_certificates.json` A3 descriptive |
| F11-W5b | read the confidence-sequence papers; e-LOND overlapping tests; compare with `confseq` | B | `confseq` absent; papers read from abstracts |
| F11-W5c | rule for differential missingness of outcomes | A | `decision/rollout.py:67` |
| F11-W5d | policy trees of depth two | A | only `fit_threshold_policy` |
| F11-W5e | three incompatible fairness pairs checked against sources | B | literature |
| F11-ebl | empirical-Bernstein lower bounds for OPE | A | not in `decision/` |
| F11-mig | `DecisionSpec` migration: remove or deprecate the run spec's cost fields | D | API change |
| F11-py | ledger replay identical on 3.11 and 3.14 | A | one interpreter measured |

### Phase 12 / F12

| ID | Item | Class | Evidence / reason |
|---|---|---|---|
| F12-0 | read Romano, Uno, Wickramasuriya (JASA), Gneiting-Raftery, M4/M5; confirm corpora licence and hash; MAPIE version | B | papers, data, MAPIE absent |
| F12-1a | CQR base levels tied to the requested level | A | `quantile.py:21-22` fixes 0.05 and 0.95 |
| F12-1b | differential test against MAPIE (G2, G4) | B | MAPIE is not installed and not an extra |
| F12-1c | intervals or sets for ordinal, multioutput, multilabel, forecast | A | `stages.py:443-448` skips them |
| F12-2 | metric slicer has no coverage | D | the two reports are separate by design |
| F12-3 | shift weights: wire refusal to F13 escalation; run has no target covariates | A (wiring) | `ModelBundle.attach_shift_weights`; real target covariates are B |
| F12-4 | forecast: per-horizon calibrated quantiles, WIS in the run, `mase` scale from training history, official M4 scores | A (WIS and scale) / B (M4 data) | `proper_scores.weighted_interval_score` is not called in the run |
| F12-5 | hierarchy: charter hierarchy and summing matrix, WIS, `hts`/`fable`, M5 | A (charter field, WIS) / B (M5, R) | |
| F12-6 | survival: charter `tau`, IPCW inside Brier and AUC, scikit-survival and R differential, GBSG2/METABRIC | A (charter field, IPCW use) / B (references, data) | |
| F12-7a | cluster-bootstrap metric intervals for panel runs | A | generic resampler exists; entity column not fed |
| F12-7b | real-panel check that entity-wise calibration does not under-cover | B | a real panel |
| F12-7c | G10 NDCG@10 vs sklearn on frozen lists | A | `test_ranking.py` compares one query at k=5; G10 published figures on a public LTR set are B |
| F12-7d | G12 per-label support rule, thresholds and sets; Yeast, Scene | A (rule) / B (data) | |
| F12-8 | near-duplicate grouper, one frozen image and one text encoder (G14, G15) | A (grouper, planted duplicates) / B (encoder weights, model card) | |
| F12-9a | `task_gate.tier_ceiling` applied to run outputs (acceptance 8) | A | no module in `scmle/` imports `task_gate` |
| F12-9b | G7 over real datasets | B | datasets |
| F12-9c | G3 at `R = 2000` after re-registration; analysis of the miss | A | published miss 0.864 vs 0.87 |
| F12-xch | exchangeability checks of `C` and of calibration rows | A | no such check |
| F12-G6 | real covariate-shift data | B | data |
| F12-G13 | entity-level calibration wired; claim "row-wise anti-conservative" unsupported | A | simulation shows no under-coverage; wire statement only |
| F12-3m | acceptance 3 fold-model hash round trip | closed | `test_cv_plus_bundle.py` |
| F12-py | every certificate replays bit-identical on 3.11 | A | certificates produced on 3.14 only |

### Phase 13 / F13

| ID | Item | Class | Evidence / reason |
|---|---|---|---|
| F13-0a | read Begg-Greenes 1983, Chow 1957/1970; replace "from memory" in `labels.py`, `escalation.py` | B | papers not openly available |
| F13-0b | independent betting-bound implementation (library) or brute-force derivation | A | `test_sequential_tests.py` has a same-recurrence loop only; a derivation by exact enumeration is buildable |
| F13-0c | verify replay streams' existence, licence, SHA-256 | B | outside data |
| F13-3a | re-registered design for the L1 and selective alarms so that H2 and H10 pass | A | simulation; a new pre-registration |
| F13-5a1 | novelty and set size for regression bundles; certified `novelty_bound` | A | `UncertaintyFeed` handles classification; `loop.py:160` default 0.05 |
| F13-5a2 | exact or conservative weighted threshold test, certified | A | `escalation.py:36,122,237` Kish and rounding |
| F13-6a1 | custody API routed through the guards | D | architecture decision (documented limit) |
| F13-6a2 | single-use marker for an attestation | A | `never_list.py:18` |
| F13-6a3 | `check_feedback` needs next-period features from the caller | D | interface decision |
| F13-6a4 | `escalation_audit.json` (H11) | B | needs the owner's audit |
| F13-7 | replay and domain-owner audit (H11) | B | owner |
| F13-8a | halt connected to `monitoring/incidents.py` | D | a halt has no SLO evaluation; which SLO is the owner's |
| F13-8b | ledger replay identical on 3.11 and 3.14 | A | |
| F13-8c | `loop_certificates.json` in the release evidence index | A | not in `docs/RELEASING.md` or `scripts/` |

### Phase 14 / F14 items 1, 2, 5

| ID | Item | Class | Evidence / reason |
|---|---|---|---|
| 14-1 | container path: build image, start compose | C | `docker info`: daemon not running; no image built |
| 14-2 | image reproducibility (two builds, same digest) | C | needs a daemon |
| 14-3 | arm64 build and run | C | needs a daemon |
| 14-4 | `image-attestation.json` bound to a real digest; asymmetric or keyless signing | C | real image; Phase 16 for the verifier |
| 14-5 | scan gate against a built image (Trivy in CI) | C | `trivy` not installed; no image |
| 14-g | CI builds the image from pinned inputs and verifies its own signature | C | follows 14-1 |
| 14-5b | item 5 restart, load and scaling tests on each cloud | C | cloud |

### Phase 15 / F14 items 4, 6, 7, 8

| ID | Item | Class | Evidence / reason |
|---|---|---|---|
| 15.1a | TLS ingress, workload identity, managed keys on AWS and GCP | C | accounts |
| 15.1b | OIDC discovery and a network-fetched JWKS (the key set is a file) | A | stdlib `urllib` and a local HTTP server verify it; a real identity provider is C |
| 15.1c | revocation distribution to every replica | A (list format, pull, cache, tests) / C (replicas in a cloud) | |
| 15.1d | pen-test style review of the gateway by someone other than the author | B | a reviewer |
| 15.2a | conformance suite against MinIO, GCS emulator, real S3 and GCS | C | none installed; no daemon |
| 15.2b | one warehouse per cloud; ingest, train, trace on a cloud dataset | C | accounts (and an owner's choice) |
| 15.3 | training in each cloud: ECS and Cloud Run runs, IAM identities, interruption, billing check, `RunV2Client`/`ecs_client`, reproducibility tolerance report | C | accounts |
| 15.4a | deploy emitted Step Functions and Cloud Workflows, schedulers, notification channel, duplicate/partial/missed/interruption on each engine | C | accounts |
| 15.4b | concurrency and cost budgets enforced in the infrastructure | C | the definitions return `controls`; enforcement is a cloud setting |

### Phase 16 / F14 items 3, 9, 10

| ID | Item | Class | Evidence / reason |
|---|---|---|---|
| 16.1a | `tofu fmt`, `validate`, lock files | closed | re-run in this pass: all eight directories valid; lock files tracked |
| 16.1b | `plan` per environment | C | needs provider credentials |
| 16.1c | apply, test, destroy, orphans on real listings, cost vs bill | C | accounts, spending cap |
| 16.1d | known gaps: CMEK for Artifact Registry and Secret Manager, custom domain/certificate, API Gateway access-log permissions, WAF or Cloud Armor | D | each is a design and cost decision |
| 16.2 | protected environments, OIDC trust, run `cloud-release.yml` once to success and once refused, real image scan, reviewer sign-off, asymmetric deployment record | C | accounts, approver |
| 16.3 | runtime adapters against real ECS and Cloud Run; traffic splitting; ECS staged traffic; rollback; `reconcile` | C | accounts. The ECS weighted-target-group planning logic is A but without ECS it only checks our own assumptions, so it is not built |

### Phase 17 / F14 items 11, 12, 13

| ID | Item | Class | Evidence / reason |
|---|---|---|---|
| 17.1a | CloudWatch / Cloud Monitoring, real pager and ticket endpoint, injected failure and drift on staging, operator response | C | accounts, operator |
| 17.1b | signals wired into the generated service | A | `docs/OPERATIONS.md`: a deployment calls them, the generated service does not |
| 17.1c | cost report with real rates vs a bill; classifier tuned on real incidents | C | bill, incidents |
| 17.2a | controlled degradation in staging with real delayed labels | C | staging |
| 17.2b | F13 alarm gates retraining after H2 passes | A | follows F13-3a |
| 17.2c | business metrics and margins `post_promotion_check` watches | D | per product |
| 17.2d | holdback so the post-promotion answer can be attributed | A | `operations/continuous.py` notes it is missing; a simulated-traffic design |
| 17.3a | every catalogued fault in each provider's staging; RTO/RPO there | C | accounts |
| 17.3b | machine crash or power loss; zone or region loss | A (torn-write simulation) / C (real) | the ledger and state files are shown to survive a killed process only |
| 17.3c | revisit `recovery.OBJECTIVES` with the owner | D | |
| 17.4 | all P0 acceptance on both providers, `signoff.json` by a reviewer with a key the pipeline does not hold, `check_production_claim.py` | C | accounts, reviewer; the claim has not been made |

### Phase 18 / F15

| ID | Item | Class | Evidence / reason |
|---|---|---|---|
| 18.1a | independent trace of one deployed model; real outcomes after real delay; stakeholder criteria signed before deployment | B | reviewer, stakeholder, deployment (also C) |
| 18.1b | written explanations of data-quality, performance trade-offs and deployment decisions in the trail | A (record type, signature, verify, show) / B (the reasoning) | `scmle.trail` records facts only |
| 18.1c | independent confirmation of a released analytical claim | B | |
| 18.1d | point-in-time and source-data steps in the trail | A (record types and a synthetic source) / B (a real source system) | Titanic extract has no source |
| 18.2 | breadth slices (clustering, association, dimensionality reduction, forecasting, ranking, documents, image/audio/video, geospatial, pretrained, GPU, streaming) | D | each needs the deployment owner's declared requirement and parameters in `benchmarks/breadth/`; then a public benchmark and replication (B) and both clouds (C) |

