# Features still to build

Names and principles: [`../plan/00-canon.md`](../plan/00-canon.md). Where each
item is scheduled: [`../plan/README.md`](../plan/README.md). Item IDs, classes
(A buildable here, B outside people or data, C cloud account or daemon, D owner
decision) and reasons: [`../plan/COMPLETION_AUDIT.md`](../plan/COMPLETION_AUDIT.md).

This file lists only what is not built. A built part is deleted from it and
recorded in `CHANGELOG.md`. Each feature ends with its unmet acceptance
criteria and an external referent that is neither the author nor the user.
Rules that hold for every feature: a gate has a pre-registered floor, measured
operating characteristic and certificate before it may gate (P1); a floor is
never lowered after a result; a gate that misses stays descriptive and the miss
is published.

| ID | Feature | Stage | Open |
|---|---|---|---|
| R1 | Instrument residuals | 3, 5 | uncertified detectors, unbuilt checks, citations, independent artifacts |
| F7 | Verification harness | 5 | backtest, red team, replication, TabArena, R fixtures, mutation floor |
| F8 | Problem framing | 3, 5 | gold corpus and the LLM gate |
| F9 | Raw-data ingestion | 3, 4, 5 | outside corpora, interpreters, a second backend |
| F10 | Causal identification | 3, 5 | method gaps, gold standards, reference outputs |
| F11 | Decision layer | 3, 5 | fairness integration, policy classes, referents |
| F12 | Uncertainty and breadth | 3, 5 | other task types, real-data gates |
| F13 | Closed loop and escalation | 2, 3, 5 | alarm design, owner audit |
| F14 | Cloud portability and production operation | 4 | everything that needs a cloud |
| F15 | Business trail and qualified breadth | 4, 5 | a real trace, explanations, breadth slices |
| F16 | Autonomous operation | 2 | alarm gate, more envelope actions, driving the tick, staging run |

## R1: Instrument residuals

Open items of the built instrument (F1 to F6). While one is open, the
behaviour named is the stated limit (P6) and documents may not claim more.

* **Detectors measured but not certified** (they stay descriptive). A5:
  label-noise recall lower bounds are below 0.80 on small tables. A6: the
  sentinel detector flags the clean rows of natural tables and its
  certificates use one corpus.
* **Wiring and policy gaps.** B2: per-period drift with a permutation null and
  calibrated PSI (the report states `periods` is `not_run`). B3: the forward
  check is always skipped because the final test set opens once (owner
  decision). B4: Thresholdout raises `ConfigError` (implement or delete; owner
  decision). B5: the resampling negative control waits for a ledgered
  feature-only finding family (B2 or B12). B7: the multiverse covers three
  choices. B8: no stability for multiclass targets. B12: MinHash text
  near-duplicates. B13: instance-dependent label noise.
* **Evidence artifacts.** D1: the sample run is signed with a published
  demonstration key; an independent holder is needed. D2: the attacker is a
  generator script, not a red team. D4: six of twelve oracles have no scmle
  producer (`chi2_contingency`, `fisher_exact`, `kendalltau`, `ks_2samp`,
  `mannwhitneyu`, `pearsonr`) and the `onlineFDR` and `knockoff` references
  are pending R.
* **Citations from memory.** E1 e-LOND venue; E2 C2ST and DeLong variance;
  E3 Gretton et al. 2012; E4 PCS and the specification curve; E5 Holm 1979,
  Benjamini-Hochberg 1995 and Benjamini-Yekutieli 2001. No user-facing
  document may cite one of these as verified until its row is closed.

Closing criteria are in `../plan/3-methods-and-instrument.md` (A items) and
`../plan/5-external-validation-and-release.md`.

Acceptance (unmet): every ID is closed (criterion met, row deleted) or accepted
(limit recorded in `docs/SECURITY.md` or `docs/ARTIFACTS.md` with the date);
`python -m benchmarks.qualification --check all` exits 0 with the A-group
certificates passing or the detector listed as descriptive; a test that each
register row naming a file still resolves (R1-acc).

External referent: frozen scipy and R fixtures, the labelled corpora in
[`../plan/validation-referents.md`](../plan/validation-referents.md), the
primary sources of the citation group and the red-team analyst.

## F7: Verification harness

Supplies the evidence the instrument cannot produce about itself: whether its
verdicts predict real outcomes, whether an outsider can defeat it, whether
someone else reproduces its numbers, how it compares with the strongest
alternatives, and whether the code that computes its statistics is protected
against silent faults. Nothing here adds user-facing analysis.

Decisions it serves: whether the project may state "block means worse out of
time" and at what strength; whether `confirmed` may be described as resistant
to a motivated analyst; whether a release may be signed off; whether "close to
the state of the art" in `benchmarks/sota` is supported.

* **W1 Predictive-validity backtest** (5.3). Do datasets the instrument calls
  `block` show a larger out-of-time gap than those it calls `pass`? The gap
  (earliest-period validation score minus latest-period score under a frozen
  reference model and metric) is fixed before any verdict. The verdict may not
  use the later period or any model score used in the gap; a charter run must
  keep `F` (or a sealed later block) out of the instrument's reach and the
  harness must assert that the run's artifacts contain no value from it.
  Corpus membership is fixed by a written inclusion rule and hash list before
  verdicts exist; datasets are not dropped for failing; report how many lacked
  a time axis. Compare within strata (size tercile by task type) with an
  adjusted effect. One pre-registered primary contrast; per-check contrasts are
  secondary, BH-adjusted and labelled exploratory. A wide interval is reported
  as "unsupported", not softened. Open design risk: TabArena tasks are mostly
  not temporal; if fewer than 15 datasets qualify the outcome is "unsupported".
* **W2 Red team** (5.4). An analyst who did not write F1 to F6 and has not read
  the detector code gets 3 datasets and a rules sheet and tries to push a
  planted-null feature to `confirmed` or get a planted leak admitted using only
  legal inputs. Reading sealed sets or editing code is a custody finding, not
  an attack. Attack budget and runs per attempt are fixed in advance. Each
  attempt is a trial; report successes out of attempts with an exact binomial
  interval. Pre-registered pass: zero successes with an upper bound consistent
  with `alpha_confirm` when attempts allow. Every success becomes a regression
  test and a ledger rule. Attack surface to put on the sheet: re-running with
  different seeds or charters, the caller-held copy of the data, the
  non-charter path (no custody).
* **W3 Independent replication** (5.5). A third party reruns the qualification
  campaign (`--smoke --full`, `--check`, the backtest, the SOTA Lite tier) from
  a tag on a fresh machine using only the committed protocol; statistics match
  within the pre-stated tolerance or a dated amendment explains the gap.
* **W4 TabArena campaign** (5.6). Execute the written campaign
  (`benchmarks/sota/campaign.py`, `scmle_runner.py`, `baselines.py`,
  `analyze.py`, 51 pinned tasks, development 31 and hold-out 20): the
  calibration step, the development run, then the hold-out once. A claim about
  the SCMLE modelling stages, not the instrument. About 9 machine-days.
* **W5 R oracle fixtures** (5.7). `onlineFDR`, `knockoff`, `dagitty`, frozen
  with generator script, versions and content hash; decisions exact, levels to
  1e-9.
* **W6 Mutation floor** (5.8). mutmut over the full configured scope; committed
  per-module survivors in `benchmarks/qualification/mutation_survivors.json`;
  score at least 80; all comparison, constant or sign mutants in `ledger/` and
  p-value code killed or justified.
* **W7 CI headroom** (1.5). Windows jobs under 14 minutes on three consecutive
  master runs without raising `timeout-minutes`.

Acceptance (unmet): the backtest command exits 0 and writes effect, interval,
strata, n and the number of datasets without a time axis; `redteam/report.json`
lists attempts, successes and the upper bound with every success mapped to a
test; a replication record exists within tolerance; `benchmarks/sota` hold-out
result files exist with SHA-256 checksums and the claim is reported met or not
met; R fixtures exist and the differential tests pass;
`./scripts/check_mutation_score.sh 80` exits 0 with custody, charter and vault
in scope; Windows jobs under 14 minutes.

Limits: certificates describe injected defect distributions only; differential
tests show agreement with a reference, not truth; a backtest gives association
across a finite corpus; a red team with few attempts supports only a loose
bound; replication by one party shows reproducibility, not correctness.

External referent: the red-team analyst, the replicating party, later-period
outcomes in the backtest, R reference outputs, TabArena's published results and
the reference systems.

## F8: Problem framing

Built: the deterministic drafter, cost algebra, baseline claim and
selective-label detector. An LLM-drafted field has no error rate the instrument
can derive, so the LLM path is descriptive and non-gating: no `tau` is on file
and `frame` refuses every LLM draft. Every LLM-filled field is `unverified`
until a named human signs.

* **Gold corpus** (F8-4, class B). At least 100 real problems, each with a
  request, a source description and a charter plus cost model written by a
  domain expert who has not seen the drafter's output; at least 20 charted
  independently by two experts (their field agreement is the ceiling); split
  into calibration and a held-out test used once per drafter version. Public
  competition descriptions give target, unit and metric but not cost models;
  if cost gold cannot be obtained, cost-model drafting stays human-supplied.
  Adjudicated and signed (`../plan/validation-referents.md`, sections 1 and 5)
  and recorded in `corpora.json` as `candidate` until verified. Scoring per
  field: target, prediction moment and unit by exact match; population by
  Jaccard of selected `X` row ids at least 0.95; metric by class; cost ratio
  within a factor of 2 or overlapping range; baseline column exact; questions
  by precision and recall against the experts' "needed" items.
* **LLM gate** (F8-5b, F8-c7). A real adapter behind `refusal.SlotSampler`,
  the risk-coverage curve and `tau` written to a certificate.
* **Weaker selective labels** (F8-sel, class A): recall where the outcome is
  observed for 70 percent of rows with `b = 0`.
* **Referent checks** (F8-8a, F8-8b): `net_benefit` against R `rmda` or
  `dcurves` and Vickers' code; the Berger 1982 and El-Yaniv and Wiener 2010
  citations re-read (Vickers and Elkin 2006 was read on 2026-11-05); a statistician's review of the 0.90 factor; a domain
  owner's recorded sign-off on a cost range for one deployed problem; the
  outcome check (does the verdict on `C` agree with realised cost).
* **Free text** (F8-w, D): the charter's `unit`, `deployment_population` and
  estimand text are checked only against column names.

Acceptance (unmet): (7) held-out gold gate, once per drafter version: Wilson
lower bound of the end-to-end rate among accepted drafts at least 0.90 times the
inter-expert ceiling, else non-gating; (8) the LLM is kept only if its
end-to-end rate minus the deterministic drafter's has a 95% paired-bootstrap
interval above zero; (9) planted omission on gold requests: question recall at
least 0.8 and false-question rate at most 0.1; test problems never appear in a
prompt (hashes checked against prompt text).

External referent: expert-written charters, the R decision-curve packages, a
domain owner's sign-off.

## F9: Raw-data ingestion

Built: the library, `scmle ingest`, the charter-run stage on SQLite, two
certified detectors and synthetic floors for the rest. Nothing ran on a real
warehouse.

* **Second backend** (F9-1, D): only SQLite is exercised in a charter run;
  DuckDB or another DB-API driver is a new dependency.
* **External referents for three gates** (F9-2): key and foreign-key precision
  and recall on schemas with declared constraints (constraints stripped), and
  entity-resolution precision and recall on labelled benchmarks. Until
  measured, the grain, join and entity gates are descriptive. Entity
  resolution in a run is unlabelled (F9-3) and errs toward keeping records
  together.
* **Real known-time** (F9-6): no serving-time feature log has been compared
  with the view's output.
* **Review** (F9-5b): the as-of reference was written separately by an agent,
  not by a person outside the project; no data-engineering reviewer signed.
* **Smaller items:** identity of a charter run across Python 3.11 and 3.14
  (F9-14a); a survivor bound for `asof_view.py` and `synthesize.py` (F9-mut);
  entity resolution does not charge the budget per step (F9-bud, D);
  wall-time stops repeat only with a fixed clock and the budget unit is rows
  (F9-w, D); read-only is a textual allow-list and a query-only session, and a
  read-only role cannot be verified (4.24, C).

Acceptance (unmet): (2) key inference recall and precision Wilson lower bounds
at least 0.9 at `L = 3` on a declared-key corpus, targets re-set after the first
run and frozen; (4) auto-accepted foreign-key precision at least 0.95; (5)
pair completeness and cross-split link recall at least 0.95 on each labelled
benchmark, else grouping stays `unverified`; (14) a charter run end to end,
identical on Python 3.11 and 3.14: expected columns, provenance table with
correct tiers, passing reconciliation.

External referent: public relational databases with declared keys, labelled
entity-resolution benchmarks, an as-of reference rewritten by an outside
person, source-system totals owned by the data owner, a data-engineering
reviewer.

## F10: Causal identification

Built: the charter block, selector and refusal, five strategies, battery,
replication, sensitivity, tier rule and the estimate on `C`. A gate that missed
its floor stays descriptive and cannot count toward `confirmed`.

* **Method gaps** (stage 3, each needs its own certificate): difference in
  differences (Rambachan-Roth conditional procedure, covariates,
  not-yet-treated group, simultaneous bands); IV (2SLS with several
  instruments, Olea-Pflueger effective F, exclusion sensitivity, battery);
  regression discontinuity (MSE-optimal bandwidth, McCrary or
  Cattaneo-Jansson-Ma density test); synthetic control (predictors, battery);
  negative-control exposures beyond `backdoor_aipw`; `sequential_ipw`
  registered; subgroup strategies other than the difference in means;
  re-measure `coverage_aipw`.
* **Sensitivity** (F10-sens, B): the Cinelli-Hazlett conversions were written
  from memory and apply to the linear-model coefficient, not the AIPW estimate;
  Rosenbaum bounds for a chartered matched design are not implemented.
* **Attestations** (F10-att, D): signed with the project's shared HMAC key, so
  a signature shows a holder of the key, not which person; per-person or
  asymmetric keys, a revocation list and an attestor identity check are not
  built. `Dag.from_artifact` does not check the file is the intended one
  (F10-dag, D).
* **Literature claims unverified:** Callaway and Sant'Anna 2021;
  Goodman-Bacon 2021; de Chaisemartin and D'Haultfoeuille 2020; Roth 2022;
  Rambachan and Roth 2023; Imbens and Angrist 1994; Olea and Pflueger 2013;
  Calonico, Cattaneo and Titiunik 2014; McCrary 2008; Cattaneo, Jansson and Ma
  2020; Abadie, Diamond and Hainmueller 2010; Crump, Hotz, Imbens and Mitnik
  2009; Rosenbaum 2002; LaLonde 1986; Dehejia and Wahba 1999; Krueger 1999;
  Hill 2011. An unverified tag that survives to the exit gate fails it.

Acceptance (unmet): (3) worlds built on real covariates; (9) frozen outputs of
`rdrobust`, `sensemakr`, `EValue`, `did`, `HonestDiD` and `Synth` under
`tests/fixtures/oracles/` with versions and hashes; (10) experimental gold
standards, command `python -m benchmarks.qualification --f10-gold` (not built):
on NSW with each declared observational comparison group no `confirmed` tier
when the interval excludes the experimental estimate, the full table published
including failures; (11) reports bit-identical on Python 3.11 and 3.14;
replication of the gold-standard table by an outside party.

External referent: NSW with CPS and PSID comparison groups (the experimental
estimate read from the data and recorded in the manifest); Tennessee STAR with
an observational sample built by a covariate selection rule; the ACIC 2016
generators (Dorie et al., Statistical Science 34(1), 2019, arXiv:1707.02641)
and IHDP; R `grf` or `DoubleML`, `did`, `HonestDiD`, `rdrobust`, `rddensity`,
`AER` or `ivmodel`, `Synth`, `EValue`, `sensemakr`; an outside party's
replication.

## F11: Decision layer

Built: the charter decision block, `DecisionBudget`, thresholds with a
calibration gate, net benefit, off-policy evaluation, fairness constraints, the
anytime-valid engine and the rollout rule. Costs are inputs, not facts.

* **Not connected:** the run spec's `threshold_objective`, `false_positive_cost`
  and `false_negative_cost` remain the source when no charter exists (F11-mig,
  D); a charter run runs no experiment, only `RolloutHook` writes assignment
  rows (F11-W2, D); `evaluation/fairness.py`, `mitigation.py` and
  `reweighing.py` are not clients of `decision.constraints` and only
  `group_thresholds` is solved (F11-W3); only a one-parameter threshold policy
  is fitted, no trees of depth two (F11-W5d).
* **Statistical gaps:** the net benefit band missed its coverage floor
  (0.9375 against 0.93 floor at `n = 2000`) and is conservative (F11-W5a);
  empirical-Bernstein lower bounds for OPE are not implemented (F11-ebl); the
  mixture e-process is the author's construction and the Waudby-Smith and
  Ramdas, Howard et al. and Ramdas et al. papers were cited from their
  abstracts, with no `confseq` comparison (F11-W5b); whether the e-LOND boost is
  valid with overlapping experiments was not checked (the boost is not used);
  bounds cover the running average of the effect, not detection under a
  changing one; `decision/missing.py` detects differential missingness but the
  rollout rule analyses observed units only; fairness bounds treat rows as
  independent and the three incompatible fairness pairs are recalled, not
  verified (F11-W5e).
* **Referents not obtained** (F11-W4): Criteo uplift (CC BY-NC-SA 4.0), Open
  Bandit Dataset, R `dcurves` or `rmda`, Open Bandit Pipeline output, `confseq`
  sequences. Hillstrom was read from the path in `SCMLE_HILLSTROM` and has no
  corpus manifest entry; its fitted policy treats 98 percent of rows, so it
  barely tests a policy difference. An outside reproduction of A6, B1 and B7
  has not happened.

Acceptance (unmet): the gates pass in `--check f11` with A3 either gating under a
re-registered floor or descriptive on purpose; a gate that cannot run (data
absent) is `not_run` and descriptive; the outside reproduction is recorded in
the certificate file.

External referent: Criteo and Open Bandit data, R decision-curve packages,
`confseq`, a person outside the author who reproduces A6, B1 and B7.

## F12: Uncertainty and problem-type breadth

Built: split, class-conditional, weighted and CV+ conformal, coverage reports
and the coverage claim on `C`, proper scores, IPCW, MinT, shift weights and the
task gate.

* **Published misses (not retuned):** G3, CQR coverage in the highest-noise
  decile 0.864 against the floor 0.87 (the filed analysis shows the miss is not
  sampling noise and more capacity does not repair it; a locally weighted or
  asymmetric design is a new method needing its own pre-registration, F12-9c);
  G5, familywise false-flag 0.0755 against 0.0597 (the redesign G5b passed);
  G13, row-wise calibration did not under-cover a new entity in the 30-entity
  world, so the claim that it is anti-conservative is unsupported (F12-G13).
* **Intervals or sets** for ordinal, multioutput, multilabel and forecast
  (F12-1c); survival, ranking and uplift stay without intervals. CQR base levels
  are fixed at 0.05 and 0.95 (F12-1a, D, needs a charter field and a format
  bump). The metric slicer has no coverage (F12-2, D).
* **Shift weights** (F12-3a, F12-3b): refusal not wired to F13 escalation; a
  run has no target covariates; G6 needs real covariate-shift data.
* **Forecast** (F12-4a, F12-4b): per-horizon calibrated quantiles, WIS in the
  run, `mase` scale from training history; official M4 Naive and Seasonal-naive
  scores (G8).
* **Hierarchy** (F12-5a, F12-5b): charter hierarchy and summing matrix, WIS,
  M5 and `hts`/`fable` (G9).
* **Survival** (F12-6a, F12-6b): charter `tau`, IPCW inside Brier and dynamic
  AUC, scikit-survival and R differential on GBSG2/Rotterdam and METABRIC
  (G11 ii).
* **Ranking, multilabel, panel** (F12-7b, F12-7c, F12-7d, F12-7e): NDCG@10
  against sklearn on frozen lists and published LTR figures (G10); per-label
  support rule, Yeast and Scene (G12); a real-panel check that entity-wise
  calibration does not under-cover.
* **Near-duplicates and encoders** (F12-8a, F12-8b): duplicate grouper and group
  split (G14); one frozen image and one text encoder against the model card
  (G15).
* **Real data** (F12-9b): G7, 5 regression and 3 classification datasets with
  time or group structure, every violation published as a miss.
* **Reproducibility** (F12-py): every certificate replays bit-identical on
  Python 3.11 (produced on 3.14 only).
* **References not read** (F12-0): the source of the Beta coverage law, the
  impossibility of distribution-free conditional coverage, Uno et al. 2011,
  Wickramasuriya et al. 2019 (JASA text), Gneiting-Raftery 2007, M4 and M5
  papers; licence and hash of M4, M5, one LTR set, GBSG2 or Rotterdam, METABRIC,
  Yeast and Scene; MAPIE version (not installed).

Acceptance (unmet): G2 and G4 differentials against a reference implementation
or the documented tie-handling difference (F12-1b); G8 reproduces the official
Naive and Seasonal-naive scores on an M4 subset, mismatch fails the build; G10,
G11(ii) and G12 differential tests pass; G14 planted duplicates recovered at
`>= 0.95` with false-merge `<= 0.01`, leak inflation shown then removed by the
grouped split; G15 within the stated margin of the model card.

External referent: the closed-form coverage law, MAPIE or another independent
conformal implementation, the datasets of the jackknife+, CQR and
weighted-conformal papers, M4 and M5 data and tables, public LTR sets,
survival data with published analyses, scikit-survival and R `survival`/`pec`,
the encoder's model card; a domain owner's attestation for the unit of
prediction and the censoring horizon `tau`.

## F13: Closed loop and escalation

Built: the stream simulator, deployment ledger, betting alarm, paired
e-processes with e-LOND, label maturity and Horvitz-Thompson losses, selective
prediction with a fixed-sequence threshold, the eleven never-list guards and
`Loop`. Gates H1, H3 to H9 passed on simulated streams.

* **Published misses (not retuned):** H2, a rise to `r_ref + 2 Delta` is found
  within `D_max = 50` batches in 0.35 percent of streams (median delay 76); H10,
  the selective-risk alarm finds a rise to `2 r*` within `D_max` in 13 percent.
  So `alarm_gates_retrain` stays False. A re-registered design is item 2.2
  (stage 2). H4 is satisfied trivially in the world measured; its power at a
  weaker advantage is not measured.
* **Escalation inputs** (F13-5a1, F13-5a2): `UncertaintyFeed` feeds a
  classification bundle only; `novelty_bound` is a charter input, not a
  certified value; weighted threshold selection uses Kish sizes and a
  rounded-up count.
* **References** (F13-0a, F13-0b, F13-0c): Begg-Greenes 1983 and Chow 1957/1970
  not read; no independent library implementation of the betting bound; replay
  streams not verified.
* **Reach of the guards** (F13-6a1, F13-6a3, F13-8a, D): the custody API is not
  routed through the guards; `check_feedback` needs next-period features from
  the caller; a halt writes an `incident` row only, because it has no SLO
  evaluation.
* **Replay and audit** (F13-6a4, F13-7): `escalation_audit.json` is not written.

Acceptance (unmet): H2 and H10 at their pre-registered `D_max`; H11, recall of
owner-`yes` steps at least 0.90 (lower bound) at an escalation rate at most
`c_max`, and recall at least 0.80 on a held-out incident set fixed by hash
before any tuning, reported with the owner's record hash, else the policy is
`unverified`.

External referent: a domain owner's signed escalation audit (two auditors for
`confirmed`, kappa reported; one auditor gives `attested`); a held-out incident
set; public streams with a documented regime change (names unverified); R
`onlineFDR` and an independent confidence-sequence implementation.

## F14: Cloud portability and production operation

One ML lifecycle with portable execution and provider adapters: local, AWS and
GCP run the same bundle, API and contracts. The local half of items 1 to 13 is
built; what is open needs a container daemon or an account, a spending cap and
a named approver.

| # | Item | Open |
|---|---|---|
| 1 | Reproducible local environment | Run the container path with Docker; arm64 (4.1 to 4.3) |
| 2 | Immutable, secure artifacts | Build and scan a real image, attest its digest, measure two-build reproducibility; prediction parity and the vulnerability gate on both clouds (4.2, 4.4 to 4.6) |
| 3 | Infrastructure as code | `plan` with credentials; apply, test, destroy, recorded cost; the known gaps (4.14, 4.15, 4.28) |
| 4 | Authentication and security | An independent review; a real identity provider; TLS, workload identity and managed keys in each cloud (4.8, 5.19) |
| 5 | Online and batch inference | Restart, load and scaling tests on each cloud (4.7) |
| 6 | Portable training | The AWS and GCP adapters against real services; reproducibility-tolerance report (4.11) |
| 7 | Governed data sources | The conformance suite against MinIO, the GCS emulator and real buckets; one warehouse per cloud (4.9, 4.10) |
| 8 | Orchestration | Services accept the emitted definitions; duplicate, partial, missed and interruption runs (4.12, 4.13) |
| 9 | CI/CD release gates | Run the workflows; build and scan a real image (4.16) |
| 10 | Registry bound to the runtime | Real revision from ECS and Cloud Run, traffic split, staged traffic on ECS, previous-revision rollback (4.17) |
| 11 | Observability and incidents | CloudWatch and Cloud Monitoring, a real pager, classifier tuned on real incidents, cost against a bill (4.18, 4.19) |
| 12 | Continuous evaluation | Controlled degradation in staging; the F13 alarm does not gate retraining (4.20, 2.2) |
| 13 | Failure recovery and DR | Machine crash, zone or region loss, every fault in staging, objectives revisited with the owner (4.21, 4.22, 4.25, 4.30) |

Acceptance (unmet), per provider: the same approved bundle passes signature
verification, image build, vulnerability gate and prediction parity; a new
environment is provisioned, deployed, tested and destroyed from version control
with infrastructure and cost recorded; a permitted release proceeds from commit
to staging to production with a complete evidence chain and a failed gate
refuses promotion and changes nothing; registry and live state reconcile after a
successful and a failed release; staging passes the declared recovery scenarios
within the documented RTO and RPO; a clean-runner CI run builds the image and
runs the demo including a rollback drill.

Release gate: "production-qualified on AWS and GCP" only when items 1 to 13 pass
on both providers; every release identifies the application, model, data and
configuration versions, the cloud environment tested, the evidence and the
responsible approver. Differences between providers in latency, cost and scaling
are reported, not equalised.

External referent: the cloud runtime itself (revision, digest, traffic split,
metrics), an independent scanner and attestation verifier, and a reviewer other
than the author who signs the staging evidence.

## F15: Business trail and qualified breadth

* **Part A, a trail on a real deployment** (4.26, 4.27, 5.20 to 5.22): a
  runtime-confirmed `production` release with real predictions and real business
  outcomes that arrive after a real delay; stakeholder-owned acceptance criteria,
  costs and expected range signed before deployment by a person who holds their
  own key; reviewable written explanations of data-quality findings,
  performance trade-offs and deployment decisions (the trail records facts, not
  reasoning); independent confirmation of released claims where required; source
  data, event logs and schema history behind the model (the committed trail
  starts from a flat extract).
* **Part B, breadth** (4.31, D): no slice is qualified. One independent vertical
  slice at a time, never by adding unsupported algorithms to the production
  registry: clustering, association analysis and dimensionality reduction
  (library functions only today); forecasting and ranking (task paths exist in
  the tabular beta); documents and NLP; image, audio and video; geospatial and
  graph; pretrained transformers, multimodal models and JEPA representations only
  where a use case is defined; distributed or GPU training only for workloads
  that justify it; streaming with its delivery semantics stated and recovery
  proved, never end-to-end exactly-once without evidence over the whole
  pipeline. The robustness, drift, explanation and monitoring benchmark
  definitions in `benchmarks/breadth/` are unset and unrun.

Acceptance (unmet): a reviewer who did not build the model traces one deployed
model from the business question through source data, model selection, approval,
production predictions and observed outcomes using only retained records
(`scripts/trace_model.py`); each new task or modality completes ingestion to
serving, including evaluation, packaging, cloud deployment, monitoring and
recovery, before it is labelled production-supported. Part B starts only after
F14 passes on both providers.

External referent: a stakeholder's recorded sign-off and a reviewer who did not
build the model; a public or domain benchmark per slice and an independent
replication.

## F16: Autonomous operation

The lifecycle after the first deployment runs without a person typing each
command and without a person present for the actions a person authorised in
advance, while every action stays signed, ledgered and replayable. Decision
recorded by the owner on 2026-11-08: **a human approves every promotion**; the
envelope covers everything up to the proposal, and rollback to the previously
approved champion stays behind the existing guard.

Built (see `docs/AUTONOMY.md`, `CHANGELOG.md`): the envelope for
`train_challenger`, the controller and `scmle operate tick | status | grant |
revoke | halt`, and the simulated worlds.

Still to build:

* **Alarm-gated retraining** (2.2, the F13 alarm design): until H2 and H10 pass
  `alarm_gates_retrain` stays off and the controller only recommends.
* **Further envelope actions** (2.8): shadow, canary and rollback to the
  previously approved champion. Human only, inexpressible in an envelope: moving
  the champion forward, reopening a sealed set, changing the charter, editing
  the ledger or registering late, relaxing a threshold, acting on an
  unverified stream, answering a flagged high-impact case, suppressing its own
  alert, changing the labelling policy, deleting or sending externally,
  creating or signing an attestation.
* **Driving the controller** (2.6): a per-day action cap, a `start` action and
  a workflow handler so `Schedule.tick` or a workflow engine runs the tick.
* **Mutation records** (2.7) for the envelope and the controller.
* **Real deployment** (2.5, C): staging with real label delay and a real
  business outcome.

Acceptance (unmet): a mutation survivor record exists; the controller has run
in a staging deployment, reviewed by someone other than the author.

Not claimed: autonomous operation in production, or anything beyond the
simulated worlds and the declared envelope.

External referent: for item 2.5, a real staging deployment reviewed by someone
other than the author.
