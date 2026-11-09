# Item register: everything still open

State on 2026-11-08, `master` at `9356b89`. `ci.yml` is green on this head (run
42471349728, 23 of 23 jobs). The scheduled `full-suite.yml` passes its test
step but its mutation step has failed since 4 Nov; a fix is committed and
unverified (item 1.4).

This file lists only open items, grouped under the four stages of
[`README.md`](README.md). A closed or built item is deleted from it; the record
is `CHANGELOG.md` and `git log`. Each row has one class. An item that needs
both local work and outside input is split into two rows (suffix `a` and `b`).

Classes:

* **A**: buildable here and verifiable here (local code, existing
  dependencies, this machine's tools: Python 3.11 and 3.14, mutmut).
* **B**: needs outside people or data: labelled corpora, gold standards,
  reference implementations (R, MAPIE, scikit-survival), a red team,
  replication, a domain owner, a prospective head-to-head, clerical samples,
  real panels, external benchmarks, a reviewer other than the author, or
  something that exists only after a push (a CI observation, a published
  release).
* **C**: needs a container daemon or scanner that is not on this machine
  (the Docker daemon is not running here; `trivy` and `skopeo` are not
  installed), a real deployment, a real database role or an approver.
* **D**: needs a design decision from the owner (a version number, a new
  dependency, a protocol's thresholds, a business metric).

Placement rule: class A items from the old instrument, framing, ingestion,
causal, decision, uncertainty and loop work are in stage 3; class B and D
items of that work, and the container, real-deployment, trail and breadth
items, are in stage 4.

## Counts

| Class | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Total |
|---|---|---|---|---|---|
| A, buildable here | 1 | 4 | 43 | 5 | 53 |
| B, outside people or data | 2 | 0 | 0 | 50 | 52 |
| C, container daemon, real deployment or approver | 0 | 1 | 0 | 10 | 11 |
| D, owner decision | 0 | 0 | 0 | 23 | 23 |
| Total | 3 | 5 | 43 | 88 | 139 |

## Stage 1: trustworthy evidence ([`1-trustworthy-evidence.md`](1-trustworthy-evidence.md))

| ID | Item | Class | Evidence / reason |
|---|---|---|---|
| 1.3 | the published quickstart (`docs/START_HERE.md`, `website/src/pages/docs/quickstart.astro`) runs verbatim in a clean environment on macOS, Linux and Windows | A | the 5 Nov product audit (`issues.md`) reported a four-row fixture failing the stratified split and `predict` refusing an unsigned local bundle; the page now explains `--trusted` and signing, but nothing re-ran it |
| 1.4 | one `full-suite.yml` run on the head with every step green and read | B | `ci.yml` is green on `9356b89` (run 42471349728, 23 of 23 jobs, after four jobs hit GitHub 504s and were re-run). The full-suite run 42476834640 passed its test step, including `test_oracle_table` on Linux 3.14, then failed "Enforce critical mutation score": mutmut instruments the hashed detector sources, so `qualification --check` reports them stale in its tree (failing since 4 Nov, run 41805832562). A skip under mutmut is committed; it needs one more run to confirm |
| 1.5 | Windows jobs under 14 minutes on three consecutive master runs | B | 24 and 27 minutes in run 42471349728 (Windows 3.14 and 3.11); needs pushes and observation |

## Stage 2: autonomous operation ([`2-autonomous-operation.md`](2-autonomous-operation.md))

| ID | Item | Class | Evidence / reason |
|---|---|---|---|
| 2.2 | alarm-gated retraining: a re-registered L1 and selective-alarm design so that H2 and H10 pass, then `alarm_gates_retrain` allowed | A | `alarm_gates_retrain` defaults to False (`scmle/governance/loop.py`, `scmle/operations/continuous.py`); until H2 passes the controller only recommends. This is the former row F13-3a |
| 2.6 | a per-day action cap, a `start` action that loops ticks, and a workflow handler so `Schedule.tick` or a workflow engine drives `scmle operate tick` | A | `scmle operate` has `tick`, `status`, `grant`, `revoke` and `halt`; nothing calls `Schedule.tick`; the only cap is an envelope's use limit |
| 2.7 | a mutation survivor record for `scmle/governance/envelope.py` and `scmle/operations/controller.py`, both added to `setup.cfg` `source_paths`, with the envelope's refusal floors written down | A | `tests/unit/test_envelope.py`, `tests/unit/test_controller.py` and `tests/simulation/` exist; no `tests/unit/MUTANTS_*.md` covers them |
| 2.8 | further envelope actions: shadow, canary and rollback to the previously approved champion (only `train_challenger` can be put in an envelope today) | A | `EnvelopeAction` has one member; rollback stays behind `Guard` and the registry |
| 2.5 | the controller in a real staging deployment with real label delay and a real business outcome | C | a staging deployment; also needs 4.29 |

## Stage 3: methods and instrument ([`3-methods-and-instrument.md`](3-methods-and-instrument.md))

### Instrument (R1)

| ID | Item | Class | Evidence / reason |
|---|---|---|---|
| R1-A5 | label-noise detectors: recall lower bounds below 0.80 on small tables | A | `detector_certificates.json` `f5-label-noise`; closure is a stated limit in `docs/ARTIFACTS.md` with floors unchanged |
| R1-A6 | sentinel detector flags clean rows of natural tables (29/150, 34/178, 1797/1797); one-corpus certificates | A | `f5_corpora_measurements.json`; fix or limit the detector, re-issue its certificate with the same seeds and floors |
| R1-B2 | per-period drift with a permutation null and calibrated PSI | A | `scmle/data/robustness/report.py` reports `periods` as `not_run` |
| R1-B5 | resampling negative control for a feature-only finding family | A | blocked by B2 (needs a ledgered per-period drift family) |
| R1-B7 | multiverse covers three choices | A | `scmle/data/robustness/stability.py` `IMPLEMENTED`; the stability certificate must be re-issued with the same seeds and floors |
| R1-B8 | multiclass stability | A | `scmle/data/robustness/qualify.py`; a new pre-registered estimator and certificate |
| R1-B12 | MinHash text near-duplicates | A | `scmle/data/detectors/duplicates.py` states it is not implemented |
| R1-B13 | instance-dependent label noise not modelled | A | no such plant in `benchmarks/qualification/inject.py`; closure by a plant and a measurement, or a stated limit |
| R1-D4a | six oracles have no scmle producer: `chi2_contingency`, `fisher_exact`, `kendalltau`, `ks_2samp`, `mannwhitneyu`, `pearsonr` | A | `benchmarks/qualification/oracle_table.py` `WIRED` has six of twelve; scmle code that computes chi-square, Fisher, KS and Mann-Whitney exists (`scmle/data/admissibility/independence.py`, `leakage.py`, `scmle/data/robustness/shift.py`) |
| R1-E1 | e-LOND venue (AISTATS 2024) not re-read | A | `scmle/data/ledger/online_fdr.py` says it was not checked; readable from the web |
| R1-E3 | Gretton et al. 2012 (kernel MMD) not re-read | A | `scmle/data/robustness/shift.py`; JMLR, open access |
| R1-E4 | PCS and specification curve not re-read | A | `stability.py`, `multiverse.py` in `scmle/data/robustness/`; open access |
| R1-acc | a test that each row ID of this register that names a file still resolves | A | does not exist |

### Framing (F8) and ingestion (F9)

| ID | Item | Class | Evidence / reason |
|---|---|---|---|
| F8-sel | selective-label detector recall under weaker selection (outcome observed for 70 percent of rows with `b = 0`) | A | `framing_measurements.json` measured strong plants only; a simulation |
| F9-14a | ingestion charter run identical on Python 3.11 and 3.14 | A | `tests/integration/test_ingest_charter_run.py` runs one interpreter |
| F9-mut | survivor bound in the build configuration for `asof_view.py` and `synthesize.py` | A | 136 of 769 scored mutants survive (`tests/unit/MUTANTS_framing_ingest.md`); `setup.cfg` has no per-module bound |

### Causal identification (F10)

| ID | Item | Class | Evidence / reason |
|---|---|---|---|
| F10-aipw | re-measure `coverage_aipw` (the boosted-learner world missed its floor) | A | `benchmarks/qualification/f10_certificates.json`; a new pre-registration and run |
| F10-11 | causal reports bit-identical on Python 3.11 and 3.14 | A | both interpreters available; acceptance 11 |
| F10-sub | subgroups: strategies other than the difference in means; heteroscedastic outcomes | A | `subgroup_certify.py` and `subgroup_extended.py` cover the difference in means only |
| F10-seq | register `scmle/evaluation/sequential.py` as strategy `sequential_ipw` | A | not registered |
| F10-did | DiD: Rambachan-Roth conditional procedure, covariates, not-yet-treated group, simultaneous bands | A | `scmle/data/identification/did.py` |
| F10-iv | IV: 2SLS with several instruments, effective F, exclusion sensitivity, battery | A | `overid.py` has Sargan and Hansen J as descriptive tests only |
| F10-rd | RD: MSE-optimal bandwidth, McCrary or Cattaneo-Jansson-Ma density test | A | `scmle/data/identification/rd.py` |
| F10-syn | synthetic control: predictor covariates, battery | A | `scmle/data/identification/synth.py` |
| F10-nc | negative-control exposures beyond `backdoor_aipw` | A | `scmle/data/identification/run.py` refuses others |

### Decision layer (F11)

| ID | Item | Class | Evidence / reason |
|---|---|---|---|
| F11-W3 | `evaluation/fairness.py` and `mitigation.py` become clients of `decision.constraints`; solve the `reweighing` mechanism | A | neither file mentions `constraints` |
| F11-W5a | re-register and re-measure the net benefit band (A3 gate) | A | `f11_certificates.json` A3 is descriptive |
| F11-W5d | policy trees of depth two | A | only `fit_threshold_policy` exists |
| F11-ebl | empirical-Bernstein lower bounds for OPE | A | not in `scmle/decision/` |

### Uncertainty and problem types (F12)

| ID | Item | Class | Evidence / reason |
|---|---|---|---|
| F12-1c | intervals or sets for ordinal, multioutput, multilabel and forecast runs | A | `scmle/pipeline/stages.py` skips them |
| F12-3a | shift weights: wire the refusal to F13 escalation | A | `ModelBundle.attach_shift_weights`; a run has no target covariates |
| F12-4a | forecast: per-horizon calibrated quantiles, WIS in the run, `mase` scale from training history | A | `proper_scores.weighted_interval_score` is not called in the run |
| F12-5a | hierarchy: charter hierarchy and summing matrix, WIS | A | simulation used squared error |
| F12-6a | survival: charter `tau`, IPCW inside Brier and AUC, calibration at `tau` | A | |
| F12-7c | G10: NDCG@10 against `sklearn.metrics.ndcg_score` on frozen lists | A | `tests/unit/test_ranking.py` compares one query at k=5 |
| F12-7d | G12: per-label support rule, thresholds and sets | A | |
| F12-8a | near-duplicate grouper (hash plus embedding cosine) and a group split, tested on planted duplicates (G14) | A | not started |
| F12-9c | G3 certified at `R = 2000` after a re-registered design | A | published miss 0.864 against the floor 0.87; analysis in `cqr_analysis.py` |
| F12-G13 | entity-level calibration wired; the claim "row-wise is anti-conservative" is unsupported | A | the 30-entity simulation shows no under-coverage; wiring statement only |
| F12-py | every certificate replays bit-identical on Python 3.11 | A | certificates were produced on 3.14 only |

### Closed loop (F13)

| ID | Item | Class | Evidence / reason |
|---|---|---|---|
| F13-0b | an independent implementation of the betting bound | A | `test_sequential_tests.py` checks the martingale and Ville properties by exact enumeration; no library comparison exists |
| F13-5a1 | novelty and set size for regression bundles; a certified `novelty_bound` | A | `scmle/governance/uncertainty_feed.py` handles classification only; `loop.py` default 0.05 |
| F13-5a2 | exact or conservative weighted threshold test, certified | A | `scmle/governance/escalation.py` uses Kish sizes and a rounded-up error count; H9 measured the unweighted case only |

## Stage 4: external validation and release ([`4-external-validation-and-release.md`](4-external-validation-and-release.md))

### Protocol, corpora and the external gates

| ID | Item | Class | Evidence / reason |
|---|---|---|---|
| 4.1 | `benchmarks/qualification/PROTOCOL.md` committed and tagged `qualification-protocol-v1` before any result | D | file and tag do not exist; claim, corpus rule, thresholds and stop rules are the owner's pre-registration |
| 4.2 | corpora: at least 15 datasets, 5 per defect family, SHA-256 verified | B | `corpora.json` entries are `candidate`; retrieval and licence checks needed |
| 4.3 | backtest (`backtest.py`) | B | at least 15 real datasets with a time axis; no module exists |
| 4.4 | red team | B | an analyst who did not write F1 to F6 |
| 4.5 | independent replication | B | a third party |
| 4.6 | TabArena campaign | B | about 9 machine-days, AutoGluon baselines not in the dependencies, OpenML downloads; no result committed under `benchmarks/sota/` |
| 4.7 | R oracle fixtures: `onlineFDR`, `knockoff`, `dagitty` | B | R is not installed |
| 4.8 | mutation floor 80 with a committed `mutation_survivors.json` | A | mutmut 3.8.0 installed; scope in `setup.cfg`; per-module survivors in `tests/unit/MUTANTS_*.md`; the repo-wide run takes many hours |
| 4.9 | domain-owner pilot | B | a team that owns a dataset |
| 4.10 | version choice (`0.1.0b1` reuse or `0.2.0b1`) | D | owner's decision |
| 4.11 | changelog moved under the new version | D | follows 4.10 |
| 4.12 | annotated tag `v<version>` | D | follows 4.10; tagging is not pushing |
| 4.13 | `SECURITY.md` supported-versions text | D | follows 4.10 |
| 4.14 | a test that every file a charter run writes is named in `docs/ARTIFACTS.md` | A | the document exists; the test does not |
| 4.15 | `scripts/release_preflight.sh` exits 0 | C | needs `npm ci` of the website, `uv sync --all-extras` on 3.14, pip-audit; CI covers its pieces |
| 4.16 | publish to PyPI, site snippet, operational rehearsal by a non-author, independent review | B | outside people and a PyPI project |
| 4.17 | go/no-go rows 1 to 13 signed by someone other than the author | B | statisticians, red team, replicating party, domain owner |
| 4.18 | `CITATION.cff` validated with `cffconvert` | B | not installed here; a key-and-version test exists |

### Instrument (R1)

| ID | Item | Class | Evidence / reason |
|---|---|---|---|
| R1-B3 | forward check always skipped | D | `scmle/data/robustness/qualify.py` builds `forward.skipped(...)`; `F` opens once, so changing who owns the opening is a custody decision |
| R1-B4 | Thresholdout not implemented | D | `scmle/data/ledger/confirm.py` raises `ConfigError`; implement or delete the option |
| R1-D1 | sample run signed by an independent key holder | B | a person who holds their own key |
| R1-D2 | red-team log | B | the red-team analyst |
| R1-D4b | `onlineFDR` and `knockoff` references | B | R is absent |
| R1-E2 | C2ST and DeLong variance sources | B | DeLong et al. 1988 is not openly available |
| R1-E5 | Holm 1979, BH 1995, BY 2001 original statements | B | paywalled; textbook examples can be tested but are not the source |

### Framing and ingestion (F8, F9)

| ID | Item | Class | Evidence / reason |
|---|---|---|---|
| F8-4 | gold corpus, at least 100 problems, 20 double-charted | B | domain experts |
| F8-5b | LLM adapter behind `refusal.SlotSampler`, risk-coverage curve and `tau` certificate | B | needs the gold corpus and a model account; `ReplaySampler` and `calibrate_tau` exist |
| F8-c7 | gold-corpus gate and LLM-versus-deterministic paired bootstrap | B | gold corpus |
| F8-c9c | planted-omission test on gold requests | B | gold corpus (the synthetic version passes) |
| F8-8a | `net_benefit` against R `rmda` or `dcurves` and Vickers' code | B | R absent |
| F8-8b | Berger citation, 0.90 factor review, owner sign-off on a cost range, outcome check | B | statistician, domain owner, a deployed problem |
| F8-w | charter `unit` and `deployment_population` are free text checked only by column names | D | a charter-language decision |
| F9-1 | non-SQLite source in a charter run | D | a new dependency or a second backend; `plan.py` has `sqlite_factory` only |
| F9-2 | key, foreign-key and entity-resolution gates on declared-key and labelled ER corpora | B | outside corpora |
| F9-3 | entity resolution unlabelled in a run | B | a clerical sample |
| F9-5b | as-of reference rewritten by a person outside the project; data-engineering reviewer | B | people |
| F9-6 | serving-time feature logs compared with the view | B | a deployment |
| F9-bud | entity resolution does not charge the budget per step | D | its unit would be candidate pairs, not rows scanned |
| F9-w | wall-time stops repeat only with a fixed clock; budget unit is rows | D | DB-API reports no bytes; accepted limit |

### Causal and decision (F10, F11)

| ID | Item | Class | Evidence / reason |
|---|---|---|---|
| F10-7a | corpora NSW (CPS, PSID), STAR, ACIC, IHDP in `corpora.json` | B | retrieval, licence, hash |
| F10-7c | NSW comparison table; frozen `rdrobust`, `sensemakr`, `EValue`, `did`, `HonestDiD`, `Synth` outputs; real-covariate worlds | B | R absent, data absent |
| F10-gold | `--f10-gold`, outside-party run, literature claims written from memory | B | people and papers |
| F10-sens | Cinelli-Hazlett conversions checked against `sensemakr` output | B | needs the papers |
| F10-att | per-person or asymmetric attestation keys, revocation list, attestor identity check | D | needs `cryptography` (not a dependency); the HMAC key is project-shared |
| F10-dag | `Dag.from_artifact` does not check the file is the intended one | D | the owner decides what to check |
| F11-W2 | a charter run runs no experiment; ledger rows only through `RolloutHook` | D | an experiment needs live units |
| F11-W4 | Criteo, Open Bandit, R `dcurves` or `rmda`, OBP; outside reproduction of A6, B1, B7 | B | data, R, a person |
| F11-W5b | read the confidence-sequence papers; e-LOND with overlapping tests; compare with `confseq` | B | `confseq` absent; papers read from abstracts |
| F11-W5e | three incompatible fairness pairs checked against sources | B | literature |
| F11-mig | `DecisionSpec` migration: remove or deprecate the run spec's cost fields | D | API change |

### Uncertainty (F12) and loop (F13)

| ID | Item | Class | Evidence / reason |
|---|---|---|---|
| F12-0 | read Romano, Uno, Wickramasuriya (JASA), Gneiting-Raftery, M4/M5; confirm corpora licence and hash; MAPIE version | B | papers, data, MAPIE absent |
| F12-1a | CQR base levels tied to the requested level | D | `scmle/evaluation/quantile.py` fixes 0.05 and 0.95; a run has no requested level, so this needs a charter field and a format bump |
| F12-1b | differential test against MAPIE (G2, G4) | B | MAPIE is not installed and not an extra |
| F12-2 | metric slicer has no coverage | D | the two reports are separate by design |
| F12-3b | real target covariates for shift weights; G6 real covariate-shift data | B | data |
| F12-4b | official M4 scores reproduced (G8) | B | M4 data release |
| F12-5b | M5 hierarchy and `hts`/`fable` differential (G9) | B | M5 data, R |
| F12-6b | scikit-survival and R `survival`/`pec` differential on GBSG2/Rotterdam and METABRIC (G11 ii) | B | references, data |
| F12-7b | real-panel check that entity-wise calibration does not under-cover | B | a real panel |
| F12-7e | G10 against published figures on a public LTR set; G12 on Yeast and Scene | B | public datasets |
| F12-8b | one frozen image and one text encoder behind extras; G15 against the model card | B | encoder weights, model card |
| F12-9b | G7 over real datasets | B | datasets |
| F13-0a | read Begg-Greenes 1983 and Chow 1957/1970; replace "from memory" in `scmle/monitoring/labels.py` and `escalation.py` | B | papers not openly available |
| F13-0c | verify replay streams' existence, licence and SHA-256 | B | outside data |
| F13-6a1 | custody API routed through the guards | D | architecture decision (documented limit in `never_list.py`) |
| F13-6a3 | `Loop.check_feedback` needs next-period features from the caller | D | interface decision |
| F13-6a4 | `escalation_audit.json` (H11) | B | needs the owner's audit |
| F13-7 | replay and domain-owner audit (H11) | B | the owner |
| F13-8a | halt connected to `scmle/monitoring/incidents.py` | D | a halt has no SLO evaluation; which SLO is the owner's |

### Trail, security review and independent confirmation

| ID | Item | Class | Evidence / reason |
|---|---|---|---|
| 4.19 | pen-test style review of the security gateway by someone other than the author | B | a reviewer |
| 4.20 | independent trace of one deployed model; real outcomes after a real delay; stakeholder criteria signed before deployment | B | reviewer, stakeholder, a real deployment |
| 4.21 | independent confirmation of a released analytical claim | B | |
| 4.22 | the reasoning behind trail explanations and a real source system for the trail | B | follows 4.33 and 4.34 |

### Container path, real deployment, trail and breadth

| ID | Item | Class | Evidence / reason |
|---|---|---|---|
| 4.23 | container path: build the image, start compose | C | `docker info`: daemon not running; no image built |
| 4.24 | image reproducibility (two builds, same digest) | C | needs a daemon |
| 4.25 | arm64 build and run | C | needs a daemon |
| 4.26 | `image-attestation.json` bound to a real digest; asymmetric or keyless signing and an independent verifier | C | needs a real image |
| 4.27 | scan gate against a built image (Trivy in CI) | C | `trivy` not installed; no image |
| 4.28 | CI builds the image from pinned inputs and verifies its own signature | C | follows 4.23 |
| 4.29 | controlled degradation in a real deployment with real delayed labels | C | a deployment; also what item 2.5 needs |
| 4.30 | machine crash or power loss | C | the ledger and state files are shown to survive a killed process only |
| 4.31 | read-only database role cannot be verified by the instrument | C | `scmle/data/ingest` uses a textual allow-list and a query-only session; needs a real database role |
| 4.32 | torn-write simulation of the ledger and state files | A | local half of 4.30 |
| 4.33 | trail: record types, signature, verify and show for written explanations of data-quality findings, performance trade-offs and deployment decisions | A | `scmle.trail` records facts only |
| 4.34 | trail: record types for point-in-time and source-data steps, on a synthetic source | A | the Titanic extract has no source system |
| 4.35 | business metrics and margins `post_promotion_check` watches | D | per product |
| 4.36 | revisit `recovery.OBJECTIVES` with the owner | D | declared for a local run |
| 4.37 | breadth slices (clustering, association, dimensionality reduction, forecasting, ranking, documents, image/audio/video, geospatial, pretrained, GPU, streaming) | D | each needs the owner's declared requirement and parameters in `benchmarks/breadth/`; then a public benchmark and replication (B) |
