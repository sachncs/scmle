# Production-beta execution plan

This document is the execution contract for taking `super-charge-ml` from a
broad tabular beta to an independently evidenced production beta.

It is intentionally deeper rather than merely larger. The goal is not to add
the most model families. The goal is to make every supported path explainable,
reproducible, safe to operate, and honest about its limits.

## 1. Decision to make before implementation

The target release is a **library-first, local-first production beta**. It is
not a managed cloud platform and it is not a promise of unlimited scale.

The beta will support the following modality families through one lifecycle:

- tabular
- text and documents
- image
- audio
- video
- graph and network
- geospatial vector and raster

"Supported" has a strict meaning. A modality is supported only when its
complete lifecycle works and has evidence. A reader, a model wrapper, or a
demo notebook alone does not qualify.

The project must retain explicit labels for:

| Label | Meaning |
| --- | --- |
| Implemented | Code exists and has a focused test. |
| Tested | The contract passes the declared test matrix. |
| Supported | The full lifecycle and documentation pass the release gates. |
| Production-proven | An independent reviewer reproduced the evidence from a clean checkout and exercised failure paths. |
| Extension-only | Available behind an explicit integration boundary, but excluded from the supported guarantee. |

No release note, README, model card, or API response may use a stronger label
than the evidence supports.

### Current evidence ledger — 20 October 2026

This is the implementation boundary at the time of writing. “Implemented”
means the code path exists; it does not mean the capability is supported.

| Area | Current label | Evidence and boundary |
| --- | --- | --- |
| Tabular ingestion, manifest, structured curation | Implemented / Tested | Unit and integration tests cover deterministic snapshots, local raw-byte hashes, findings, persisted manifests, machine-generated dataset cards, schema-evolution decisions, an atomic durable source registry that survives across runs, and fail-closed blocking-finding disposition before training. Object-store/warehouse/event-stream transports remain incomplete. |
| Local, DB-API, bounded object, and range-backed JSONL source connectors | Implemented / Tested | Connector contracts cover local files, injected DB-API factories, bounded injected object readers for CSV/JSONL/Parquet, and range-backed JSONL/NDJSON reads with byte checkpoints, retry budgets, bounded records, and schema-drift failures. Broker/warehouse transports and arbitrary binary/object formats remain extension work. Evidence: `tests/unit/test_connectors.py`. |
| Incremental ingestion semantics | Implemented / Tested | Transport-neutral event contracts bind event IDs to ordered checkpoints and watermarks, classify duplicates, make late-data policy explicit, serialize multi-writer commits with durable atomic state, and provide an idempotent sink commit boundary. Atomic state, run evidence, source registries, observability retention, fault evidence, and signing-key rotation now flush both file contents and the containing directory after replacement (`b420ee3`). The sink documents its crash window and requires downstream event-ID idempotency; broker/warehouse transports and distributed exactly-once guarantees remain incomplete. |
| Tabular library training paths | Implemented / Tested | The pre-training capability gate admits only explicitly declared tabular library paths. Dataset manifests preserve the caller-declared modality, and non-tabular declarations fail closed before model fitting. This is the current beta boundary, not all-modality support. |
| Non-tabular curation and data understanding | Implemented / Tested | `DocumentDataset.curate()` and the shared `ModalityCurationReport` emit JSON-native evidence for documents, images, audio, video, graphs, and geospatial vectors: bounded profiles, identities, composition, duplicate/shape/encoding/CRS/graph findings, label state, and blocking spatial or conflicting-label risks where applicable. The experimental lifecycle now persists a `ModalityDatasetCard` containing the complete curation report plus explicit provenance and privacy status; missing governance metadata is visible rather than inferred. These paths do not provide OCR, production-scale representation models, or a supported `MLProject` non-tabular training path. Evidence: `scml/data/modalities/curation.py`, `scml/models/modality_lifecycle.py`, `tests/unit/test_modality_lifecycle.py`. |
| Complete experimental modality baseline lifecycle | Implemented / Tested | `ModalityLifecycle` runs all 15 required stages for text/documents, images, WAV audio, probed video metadata, inductive graph nodes, CRS-declared geospatial vectors, and metadata-only rasters with caller-owned label sidecars. Concrete adapters own modality-specific split and feature policies; the shared stages produce curation, evaluation, data/model explanations, signed atomic packages, verified inference, drift evidence, documentation, and tamper/resource recovery evidence. Stage payloads are validated JSON objects rather than stringified notes, so evidence can be queried without reparsing human text. These are experimental baselines, not production-supported claims or replacements for `MLProject`/`ModelBundle`. Evidence: `scml/contracts/modalities.py`, `scml/models/modality_lifecycle.py`, `tests/unit/test_modality_contracts.py`, `tests/unit/test_modality_lifecycle.py`. |
| Lifecycle performance budgets | Implemented / Tested | `PerformanceObservation` and `PerformanceBudget` record machine-, software-, and timestamp-bound elapsed-time and observed process-peak evidence for ingest, fit, package, load, and inference. `PerformanceBaseline` compares a current run with the median of at least two schema-consistent reference runs and blocks declared relative time or memory regressions. Missing operations, memory evidence, or environment identity drift fail closed. The experimental modality lifecycle emits the five observations, persists the baseline comparison, and rejects a run before returning a green report when a gate fails. Its memory value is the existing cross-platform process peak observed at each operation boundary, which is conservative because it is a lifetime peak rather than an isolated allocation trace. Evidence: `scml/contracts/performance.py`, `scml/models/modality_lifecycle.py`, `tests/unit/test_performance_contracts.py`, `tests/unit/test_modality_lifecycle.py`; focused result at `f43b969`: 23 passed under `-W error::ResourceWarning`. |
| Experimental document baseline lifecycle | Implemented / Tested | `DocumentClassifier` provides bounded TF-IDF feature preparation, logistic modelling, held-out evaluation, linear global/local term explanations, an atomic package with manifest and payload/model digest verification, and a profile-drift monitor. This remains experimental: it is not integrated with `MLProject` run evidence, the signed `ModelBundle`, reference HTTP serving, OCR/layout/table provenance, or the full document 15-stage release evidence. Evidence: `scml/models/documents.py`, `tests/unit/test_document_model.py`. |
| Experimental image baseline lifecycle | Implemented / Tested | `ImageClassifier` provides bounded resized-pixel preparation, logistic modelling, held-out evaluation, global/local pixel contribution evidence, an atomic trusted package with digest verification, and mean-pixel/duplicate drift monitoring. It remains experimental: it is not integrated with `MLProject` run evidence, the signed `ModelBundle`, reference serving, saliency/occlusion validation, localization, or the full image 15-stage release evidence. Evidence: `scml/models/images.py`, `tests/unit/test_image_model.py`. |
| Image, document, graph, audio, video, and geospatial data foundations | Implemented / Tested | Validation, identity, bounded profiling, splitting, feature-preparation, and now complete experimental baseline lifecycle slices exist with focused tests. Audio and video require explicit codec/probe boundaries; geospatial vectors require validated GeoJSON geometry and CRS provenance, while rasters remain data-foundation-only and require dimensions, dtype, CRS, and transform metadata. These remain experimental because the baselines are not integrated with the supported `MLProject` run evidence, reference serving, external fault execution, or independent release review. Evidence: `scml/models/modality_lifecycle.py`, `tests/unit/test_modality_lifecycle.py`, `tests/unit/test_document_data.py`, `tests/unit/test_image_data.py`, `tests/unit/test_graph_data.py`, `tests/unit/test_media_modalities.py`. |
| Release evaluation | Implemented / Tested | Explicit success criteria produce a task-aware policy, release findings, artifact digests, and a persisted evidence index. Runs without thresholds are explicitly `policy_not_declared`, and any run with an unexecuted or unsafe fault plan is independently blocked from release. |
| Dataset and model cards | Implemented / Tested | Curated runs persist machine-generated cards tied to dataset snapshots, model-manifest digests, evaluation context, intended use, limitations, privacy, and review status. Exact-bundle promotion approvals are required at the registry boundary, and champion/rollback approvals are HMAC-signed and verified against the exact target. Independent release review remains incomplete. |
| Content-linked evidence graph | Implemented / Tested | Completed packages persist `evidence_graph.json`, a deterministic acyclic graph linking raw source, snapshot, curation, feature graph, split, model, evaluation, explanations, and bundle identities. `InferenceEvidenceLinker` verifies the persisted graph digest and exact bundle digest before atomically adding an idempotent inference-event node and serving edge. Conflicting nodes, dangling edges, cycles, and invalid serialized order fail closed. Independent release review remains incomplete. Evidence: `scml/contracts/lineage.py`, `scml/inference/evidence.py`, `tests/unit/test_lineage_contract.py`, `tests/unit/test_inference_evidence.py`, `tests/integration/test_end_to_end.py`. |
| Portable model explanation evidence | Implemented / Tested | Existing permutation-importance and partial-dependence reports are packaged with identity, privacy, limitation, and source-report links. Generic explanation quality measures now provide explicit top-k stability and normalized fidelity scores, and release policies can block below-threshold evidence. Modality-specific explanations and production-scale repeated explanation runs remain incomplete. |
| Promotion and rollback governance | Implemented / Tested | Champion promotion and rollback require an independent, HMAC-signed approval bound to model, version, bundle content, reviewer decision, evidence, and timestamp; invalid, missing, or tampered approvals are rejected and audit records preserve the signature. Lifecycle retraining stages challengers until approval. Independent release sign-off remains incomplete. |
| Redacted inference observability | Implemented / Tested | `PredictionLog` emits a redacted sidecar, policy evaluation checks required signals, freshness, future timestamps, and upper-bound threshold breaches, retention enforcement atomically removes expired sidecar events without touching the raw prediction log, deterministic routing produces idempotent alert IDs, `SLOMonitor` evaluates worst-sample windows with minimum-data checks, error-budget evidence, owner, runbook, and alert route, `AlertDispatcher` persists idempotent local or bounded HTTPS webhook delivery outcomes, and `IncidentWorkflow` records chained open/acknowledge/resolve transitions with run and rollback identity. Serving defaults and full external paging/ticketing evidence remain incomplete. Evidence: `scml/monitoring/alerts.py`, `scml/monitoring/slo.py`, `scml/monitoring/incidents.py`, `tests/unit/test_alerts.py`, `tests/unit/test_slo.py`, `tests/unit/test_incidents.py`. |
| Fault injection and recovery | Implemented / Tested (local boundary) | `LocalFaultExecutor` now executes all 15 declared scenarios through real bounded process, timeout, atomic-write, artifact, signature, checkpoint, snapshot, retry, duplicate-event, lineage, and approval boundaries; `FaultInjectionHarness` persists digest-bound evidence and fail-closed status transitions. Controlled disk and memory cases use deterministic local capacity guards, not physical disk exhaustion or host-wide OOM. Cloud transport, container, kernel-resource, dependency-installation, connector, and independent external-fault execution evidence remain incomplete. Evidence: `scml/governance/local_faults.py`, `tests/unit/test_local_faults.py`, `tests/unit/test_reliability_harness.py`. |
| Security, provenance, SBOM, and release CI | Implemented / Tested | Bundle signing, run provenance artifacts, deterministic CycloneDX SBOM generation, a deterministic runtime license inventory, CI artifact publication, immutable CI action pins, a canonical HMAC-signed release attestation, a versioned HMAC-signed release evidence index, an exact-SHA clean-clone rehearsal job, a durable review-artifact writer, an executable exact-commit review verifier, a signed `IndependentReleaseReview` contract, and a metadata-only `SigningKeyRegistry` for identity, expiry, retirement, replacement, and immediate revocation are implemented. The review contract requires an independent reviewer, exact source commit, clean-clone and environment digests, reproducible commands, recomputed metric/fairness/explanation claims, and signed artifact digests. Secret custody remains external, existing raw bundle signatures still require operator key-policy integration, and an external reviewer signature and a full clean-release execution remain incomplete. The all-extras audit gate is now executable in CI but its first successful run is still required before optional-extra license evidence can be accepted. Evidence: `scripts/check_dependency_licenses.py`, `.github/workflows/ci.yml`, `scripts/verify_release_review.py`, `scml/contracts/evidence.py`, `scml/governance/keyring.py`, `scml/governance/review.py`, `tests/unit/test_keyring.py`, `tests/unit/test_evidence_contracts.py`, `tests/unit/test_independent_review.py`. |
| Astro developer site | Partial | Standalone docs routes, base-path links, generated accessibility smoke checks, browser smoke coverage for all 15 public routes plus keyboard navigation, a capability matrix, cards/explanations, security, recovery, migration guidance, pinned Pages artifact/deploy jobs, and post-deploy route smoke coverage build successfully. The requested public URL currently returns HTTP 404 because the GitHub repository is private; repository visibility/Pages configuration must be resolved by the owner before deployment can be verified. Exhaustive modality/operations content remains incomplete. |

### Verification snapshot — 20 October 2026

The last complete repository regression run completed with `1927 passed, 103
skipped, 6 warnings` in 29 minutes 30 seconds using `uv run pytest -q` on the
local Python 3.14t environment. After subsequent data, observability,
security, reliability, governance, and incremental-ingestion changes, the
complete unit suite completed with `1577 passed, 87 skipped, 6 warnings` in 2
minutes 24 seconds. The bounded critical-path suite (unit tests plus CLI,
end-to-end, deployment, monitoring, prediction, and serving integrations)
completed with `1612 passed, 89 skipped, 8 warnings` in 3 minutes 52 seconds;
its declared core scope (`contracts`, `data`, `evaluation`, `governance`,
`inference`, and `monitoring`) measured 86% coverage with the enforced
`uv run coverage report --include=... --fail-under=85` gate;
the configured critical mutation scope generated 1,733 mutants, of which
1,251 were killed, 481 survived, and 1 timed out (72% of scored mutants),
passing the 70% minimum while leaving survivor triage as evidence debt;
the Astro build produced 11 pages, the accessibility smoke passed all 11
HTML files, and the Playwright browser smoke passed both route and keyboard
navigation tests;
the pushed exact-SHA clean-clone acceptance slice passed `1649 passed, 80
skipped, 6 warnings` in 8 minutes 39 seconds and `scml --help` succeeded;
the complete local release preflight at commit `74b913c` then passed `1715
passed, 89 skipped, 2 runtime warnings` in 8 minutes 18 seconds, with 86%
total coverage and core scope coverage at 86%, mutation testing at
`1253/1735` killed (72%, above the 70% gate), four offline benchmarks and zero
regressions, successful wheel/sdist checks, an Astro build of 11 pages, and no
known Python dependency vulnerabilities;
the two warnings are nondeterministic SQLite connection teardown warnings in
curriculum run 3 and remain release debt; the latest critical mutation run
killed `1253/1735` mutants (72%, above the 70% gate); mutation survivors and
timeout cases remain triage debt;
the latest modality/capability slice passed `27` focused tests, and the full
integration suite completed with `334 passed, 16 skipped` in 24 minutes 2
seconds. The remaining leakage, property, specification, and benchmark tiers
completed with `85 passed` in 3 minutes 27 seconds. The prior all-repository
acceptance run therefore provided `1982 passed, 103 skipped, 6 warnings`; the
subsequent signed-governance, fault-harness, incremental-sink,
directory-durability, and undefined-score slices are included in the new-head
preflight above.
The final preflight at pushed commit `020ec74` passed `1723 passed, 89
skipped, 2 warnings` in 8 minutes 10 seconds; total coverage was 86%, the
critical mutation gate killed `1253/1735` mutants (72%, above the 70% minimum),
all four offline benchmarks had zero regressions, both distribution checks
passed, and the Astro build produced 11 pages. The document baseline focused
tests passed 4 tests in isolation and are included in this repository run.
The current preflight at pushed commit `3e28a11` passed `1728 passed, 89
skipped, 2 warnings` in 8 minutes 18 seconds; total coverage was 86%, the
critical mutation gate killed `1252/1735` mutants (72%, above the 70% minimum),
all four offline benchmarks had zero regressions, both distribution checks
passed, the dependency audit found no known vulnerabilities, and the Astro
build produced 11 pages. The two warnings remain nondeterministic SQLite
teardown warnings in curriculum run 3. The atomic model-package failure-path
suite passed 6 focused tests; the latest label-state curation increment passed
10 image tests and 24 graph/media tests.
The current image-model preflight at pushed commit `632853d` passed `1736
passed, 89 skipped, 2 warnings` in 8 minutes 12 seconds; total coverage was
86%, the critical mutation gate killed `1254/1735` mutants (72%), all four
offline benchmarks had zero regressions, both distribution checks passed, and
the Astro build produced 11 pages. The two warnings remain the known
curriculum SQLite teardown warnings.
The latest non-tabular data-foundation slice at `b15c701` passed 8 focused
media/geospatial tests, the existing modality and capability integration set
passed 46 tests, and Ruff plus mypy passed for the new modules. The document
curation slice now passes 15 focused tests; the cross-modality curation slice
passes 31 focused tests, with Ruff and mypy clean. These
slices expand what can be validated and profiled; they do not expand the
supported modelling boundary. The experimental document baseline focused suite
passes 4 tests, with Ruff and mypy clean; it does not expand the supported
modelling boundary. The latest label-state curation increment passed 10 image
tests and 24 graph/media tests, with Ruff, mypy, and the convention gate clean;
it does not expand the supported modelling boundary. The clean-clone acceptance slice is verified, but an independent reviewer
must still reproduce the package and sign the release decision. Skipped tests
and runtime warnings are not release evidence by themselves: each must be
classified as
environment-required, intentionally unsupported, or a release blocker before
a supported-production label is appropriate.

The audio feature-preparation slice at `29c8d15` passes 14 focused media
tests, Ruff, and mypy. It adds bounded WAV PCM statistics with explicit
observed-frame and truncation evidence; it does not add codec support,
audio modelling, serving, or a supported capability claim.
The video metadata-feature slice at `cd1da09` passes 16 focused media tests,
Ruff, formatting, and mypy. It adds numeric evidence from explicit probe
metadata only; it does not add frame decoding, temporal sampling,
synchronization, video modelling, serving, or a supported capability claim.
The audio grouped-split slice at `6ce8ad2` and video grouped-split slice at
`46d8790` add deterministic identity-aware partitioning with focused tests;
they do not infer speaker, subject, session, or camera identity when the
caller has not supplied a `group_id`, and they do not expand the supported
modelling boundary.
The image grouped-split slice at `006f7b2` adds the same capture-session or
subject boundary with 11 focused image-data tests and does not expand the
supported modelling boundary.
The geospatial spatial-block slice at `1a0ed48` passes 19 focused
media/geospatial tests, Ruff, and mypy. It requires an explicit CRS-unit
block size and keeps blocks together; centroid assignment for large or
cross-block geometries remains a documented limitation and does not expand
the supported modelling boundary.
The geospatial feature-preparation slice at `c65b26d` passes 23 focused
media/geospatial tests with Ruff and mypy clean. It requires CRS provenance
and emits only bounds, centroids, and coordinate counts; it does not claim
reprojection, spatial joins, rasterization, distances, or a supported model
path.
The complete experimental modality lifecycle slice passes 10 focused tests
across document, image, audio, video, graph,
geospatial, and raster adapters. Every adapter emits all 15 lifecycle-stage records,
signs an atomic baseline package, reproduces predictions after verified load,
records modality-specific monitoring and documentation evidence, and rejects a
tampered manifest during recovery testing. This does not promote any modality
to supported status: the path is intentionally separate from the tabular
`MLProject` run, `ModelBundle`, reference HTTP serving, independent review,
and external resource-fault evidence.
The structured stage-evidence increment passes 20 focused contract and
lifecycle tests with Ruff and mypy clean; each lifecycle record now preserves
JSON-native stage output for machine verification as well as human notes.
The standalone-docs increment adds cards/explanations, security, recovery,
and migration routes; Astro builds 15 pages, the accessibility smoke passes
all 15 HTML files, and Playwright passes the expanded route and keyboard
navigation suite. Public deployment remains unverified while repository
visibility and Pages configuration are unresolved.
The duplicate-label curation slice at `df5739d` passes 47 focused document,
image, and media tests with Ruff and mypy clean. Image, audio, and video now
emit blocking risk findings when duplicate content carries incompatible known
labels; this remains curation evidence, not a modelling-support claim.
The image privacy slice at `f80b8fe` adds EXIF tag capture and a blocking
`risk.image-sensitive-metadata` finding for GPS, author, device, software,
copyright, and capture-time metadata. The focused image suite passes 17 tests;
the contract preserves source bytes and requires an explicit redaction or
privacy approval rather than silently mutating the evidence.

The complete release preflight at pushed commit `642806b` passed `1750 passed,
89 skipped, 2 warnings`; the bounded suite measured 81.53% whole-package
coverage and the declared core scope measured 86%. The mutation gate passed
the configured 70% minimum, all four offline benchmarks reported zero
regressions, wheel and source distributions passed checks, the dependency
audit found no known vulnerabilities, and the Astro build produced 11 pages.
The remaining warnings are ResourceWarnings for unclosed SQLite connections
reported while scanning `scml/core/stages.py` in the convention test; they are
tracked cleanup work, not silently treated as release evidence. Generated
mutation artifacts were kept outside the repository after the run; no
generated evidence is being treated as a tracked release claim.
The local fault-matrix slice at `4a45acb` executes all 15 declared scenarios
and passes 40 focused reliability tests with Ruff, mypy, and the convention
gate clean. It is evidence for the local failure boundary only; it does not
turn controlled capacity guards into physical disk-full or host-wide OOM
evidence, and it does not close the independent external-fault gap.

The current complete release preflight at pushed commit `a10e693` passed
`1780 passed, 89 skipped, 2 warnings` in 8 minutes 28 seconds. Total coverage
was 81.75%, the declared core scope measured 86%, the mutation gate killed
`1252/1735` mutants (72%, above the 70% minimum), all four offline benchmarks
had zero regressions, wheel/sdist and Twine checks passed, `pip-audit` and npm
audit reported no known vulnerabilities, and Astro built all 11 routes. The
two warnings remain the known SQLite resource-cleanup warnings from the
convention scan; mutation survivors and external-fault evidence remain open
engineering debt.

The reproducible-artifact gate at `f689d45` builds the wheel and source
distribution twice with a fixed `SOURCE_DATE_EPOCH`, compares their SHA-256
digests, installs the wheel with `--no-index --no-deps`, rebuilds the source
distribution with `uv build --offline`, and installs that rebuilt wheel with
the same network isolation. This proves artifact reproducibility and source
archive rebuildability; it does not prove runtime behavior without the
declared dependency set.

The independent-review contract at `5fd2eea` passes 12 focused governance and
attestation tests with Ruff, mypy, and diff checks clean. This proves the
review record and fail-closed verifier; it is not evidence that an external
reviewer has yet signed this release.

The incident workflow at `79c6211` passes 4 focused tests plus the convention
gate. It records hash-chained, durable open/acknowledge/resolve transitions
with owner, run ID, rollback target, runbook, and resolution evidence. It is a
local operator contract; external paging, ticketing, and on-call delivery
remain outside the current release claim.

The transition-hardening fix at `843b693` adds valid-event-type, legal-state,
identity, and timestamp-order verification while replaying the journal; its
focused incident, SLO, and convention slice passes 824 tests.

The evidence-index signing slice at `9007b8a` passes the release-contract,
problem-spec, evidence, and convention tests. `release_evidence.json` now has
an explicit schema version and signed canonical content when
`SCML_SIGNING_KEY` is configured; the run lineage includes its digest. The
signing key remains an HMAC deployment secret, so custody, rotation, and
independent release review are still separate requirements.

The current-head complete release preflight at `ae57eec` passed `1807 passed,
89 skipped, 2 warnings` in 8 minutes 8 seconds. Total coverage was 81.86%,
the declared core scope was 86%, and the critical mutation gate killed
`1295/1794` mutants (72%, above the 70% minimum). The five-seed benchmark gate
reported zero regressions with intervals of tie, tie, tie, and ahead across the
four offline datasets; Twine, reproducible wheel/sdist verification, offline
artifact installation, dependency audits, and the 11-route Astro build all
passed. The two SQLite resource-cleanup warnings remain a release-debt item;
the unit/property/leakage slice separately passes with
`-W error::ResourceWarning`.

The tag-publish alignment at `2673ec1` applies the same five-seed benchmark
and reproducible, network-isolated artifact gate to the release workflow. A
tagged publish has not been executed; the workflow change is verified by
static diff review and the local gate, not by a PyPI or GitHub Release side
effect.

The complete release preflight at pushed commit `51c5afd` passed `1803 passed,
89 skipped, 2 warnings` in 8 minutes 11 seconds. Total coverage was 81.86% and
the declared core scope remained 86%; mutation testing killed `1253/1735`
mutants (72%), all four offline benchmarks reported zero regressions, package
and offline artifact verification passed, dependency audits found no known
vulnerabilities, and Astro built all 11 routes. The two SQLite resource
warnings remain cleanup debt. That preflight used the historical one-seed
benchmark command; the stricter five-seed gate was then enabled by `d493b58`
and passed separately across all four offline datasets with zero regressions.
The resulting 95% intervals were tie, tie, tie, and ahead respectively, so the
evidence now satisfies the five-seed statistical criterion while preserving the
honest conclusion that three of four comparisons are indistinguishable from
the reference.

The current-head preflight at `e25f3ce` passed `1824 passed, 89 skipped`. The
bounded contract suite reported two ordinary
`ResourceWarning` summaries, but the same suite rerun with
`-W error::ResourceWarning` also passed `1824 passed, 89 skipped`; therefore
the exercised API paths do not leave SQLite connections open. Total coverage
was 82.25%, the declared core scope was 86%, and the critical mutation gate
killed `1294/1794` mutants (72%, above the 70% minimum). The five-seed
benchmark gate checked breast cancer, wine, digits, and diabetes progression
with zero regressions; the verdicts were tie, tie, tie, and ahead, with
95%-interval evidence retained in the benchmark artifact. Twine checks,
reproducible wheel and source-distribution verification, offline artifact
installation, dependency audit, and the Astro site build all passed. This is
local release evidence, not independent release sign-off: the external
reviewer signature, optional-extra license audit, public Pages deployment, and
external transport/resource-fault rehearsal remain open.

The multimodal dataset-card slice at `213b6c6` passed 83 focused curation,
contract, data-foundation, and lifecycle tests with ResourceWarnings promoted
to errors; Ruff, mypy, and diff checks were clean. Each experimental adapter
now carries its curation report in the documentation stage, while provenance
and privacy are explicitly marked as unassessed until caller-supplied
governance evidence is attached. This improves data understanding evidence but
does not change the supported modelling boundary.

The signer-lifecycle slice at `4b38cc0` passed 84 focused governance, attestation,
registry, bundle, and keyring tests with ResourceWarnings promoted to errors;
Ruff and mypy were clean. `SigningKeyRegistry` stores no secrets, blocks
signing with expired or retired keys, permits explicitly requested historical
verification for pre-retirement evidence, and rejects revoked keys even in
historical mode. This is lifecycle-policy evidence, not proof of deployment
secret-manager custody or an independent release signature.

The optional-extra security gate at `99f4998` is statically verified with
`actionlint` and installs every declared extra on Python 3.13 before running
`pip-audit` and the all-extras license inventory. It is intentionally not
counted as passed until the hosted CI job produces its artifact; local base
environment evidence cannot substitute for an optional-dependency closure.

The lifecycle-performance slices at `8758a1f`, `3018453`, `3e24a08`, and
`f43b969` measure ingest, fit, package, load, and inference with machine,
software, UTC timestamp, and observed process-peak context. They enforce time
and memory budgets, compare a current run with the median of at least two
reference runs, and fail closed when environment identity drifts. The memory
value is a conservative process lifetime peak at each boundary, not an
isolated allocation trace. This slice does not make the non-tabular lifecycle
production-supported.

The only currently defensible supported boundary is the explicitly gated
tabular library path. All other modality rows must remain experimental or
extension-only until their complete lifecycle evidence is attached.

The security job now also traverses the installed base-runtime dependency
closure, rejects missing license evidence and forbidden copyleft licenses,
and publishes a deterministic `dependency-licenses.json` artifact. Optional
extras remain a separate audit scope because they are not installed by the
base security job.

The remaining work is therefore not “add more models.” It is to connect each
new data foundation to the same contracts, independent evaluation, signed
packaging, safe inference, monitoring, recovery tests, and developer-facing
documentation before changing its support label.

## 2. What a principal practitioner adds beyond the request

Before adding capabilities, a principal ML platform engineer would establish:

1. **A product boundary.** Define supported workloads, non-goals, trust
   assumptions, resource limits, and the point at which a connector or model
   becomes the user's responsibility.
2. **An evidence model.** Every claim must point to a raw input, a command or
   test, an artifact, a reviewer, and a timestamp. Summary documents are not
   primary evidence.
3. **A threat model.** Treat datasets, plugins, serialized models, generated
   services, credentials, dependencies, and user-supplied files as separate
   trust boundaries.
4. **Failure-first design.** Test process death, stale checkpoints, disk-full,
   corrupt artifacts, malformed inputs, worker loss, dependency mismatch,
   signature failure, and partial writes before celebrating the happy path.
5. **Independent proof.** Require a clean clone, clean environment, fixed
   inputs, independent reference calculations, hidden fixtures, and a reviewer
   who did not implement the change.
6. **A release decision record.** Record what passed, what was waived, who
   approved it, why the waiver is safe, and when it expires.

The professional starting point is not the README or an existing benchmark
summary. It is the rawest available material:

- original source datasets and immutable snapshots;
- data dictionaries, units, timestamps, identifiers, sampling frames, and
  ownership records;
- collection, transformation, and label-generation code;
- annotation instructions, disagreements, adjudication, and corrections;
- raw prediction events, delayed labels, rejected inputs, drift events, and
  resource logs;
- original run directories, checkpoints, event logs, bundles, signatures, and
  failed artifacts;
- dependency locks, package metadata, vulnerability records, license records,
  and SBOM inputs;
- crash, timeout, OOM, interruption, retry, rollback, and corruption traces;
- versioned benchmark corpora with licenses and hashes;
- clean-install transcripts for each supported Python, OS, and extra;
- fresh-clone onboarding transcripts;
- CI and Pages execution records.

## 3. Contract architecture

### 3.1 Modality-neutral dataset manifest

Add a versioned manifest contract containing at least:

- dataset ID, immutable snapshot ID, and schema version;
- modality and task type;
- source owner, acquisition method, extraction query or object prefix;
- source timestamp, timezone, sampling frame, units, and semantic definitions;
- partition/object identifiers and content hashes;
- label source, annotator or generator version, adjudication status;
- license, consent, PII classification, retention, deletion, and access policy;
- schema evolution relation to the prior snapshot;
- lineage parents and transformation graph;
- quality findings and their dispositions.

The manifest must be serializable as JSON and Arrow-compatible, validated at
the boundary, and forward-compatible through an explicit schema policy.

### 3.2 Capability registry

Create a capability registry keyed by:

`modality × task × model family × evaluation × uncertainty × fairness × explainability × inference`

Each combination must declare `supported`, `experimental`, or
`extension-only`, plus required extras, resource assumptions, known failure
modes, and evidence links. Unsupported combinations must fail before training
with a structured diagnostic rather than failing deep inside a model adapter.

### 3.3 Evidence graph

Every run should connect these immutable identities:

`raw source → snapshot → manifest → curation → split → feature graph → model → evaluation → explanation → bundle → inference event`

The graph must be queryable from a run result and must survive packaging. A
human should be able to answer “which data and transformation produced this
prediction?” without reading logs line by line.

## 4. Workstreams and order

The phases are dependency-ordered. Do not implement modality-specific models
before the contracts and evidence paths exist.

### Phase 0 — Baseline and release boundary

**Deliverables**

- capability matrix and explicit non-goals;
- stable, experimental, extension-only, and internal API inventory;
- supported Python/OS/extra/resource matrix;
- versioning, deprecation, migration, and compatibility policy;
- release decision template and evidence index;
- a current-state ledger marking every claim as implemented, tested,
  supported, or production-proven.

**Exit criteria**

- no public documentation makes an unqualified claim that the ledger cannot
  support;
- all future work has an owner, dependency, evidence artifact, and exit test;
- the beta release scope is approved before implementation expands.

### Phase 1 — Data engineering and data understanding

**Deliverables**

- local-file, SQL, object-store, warehouse/lakehouse, and event-stream source
  adapters behind one reader contract;
- source snapshot identity and dataset registry;
- schema evolution and compatibility checks;
- offsets, watermarks, late-data policy, deduplication, retry, and checkpoint
  semantics for incremental sources;
- partition/object lineage and row or sample lineage where feasible;
- structured curation findings;
- dataset cards with provenance, quality, privacy, limitations, and intended
  use.

Every curation finding must contain:

`id, rule_version, severity, confidence, affected_scope, evidence, remediation, owner, reviewer, disposition, blocking, observed_at`

**Required raw fixtures**

- valid, empty, malformed, truncated, duplicated, corrupted, oversized, and
  schema-evolved inputs;
- missing labels, conflicting labels, time leakage, duplicate entities,
  train/test contamination, and shifted distributions;
- PII and sensitive-field examples with redaction assertions;
- source retry, duplicate event, late event, changed snapshot, and checkpoint
  recovery cases.

**Exit criteria**

- a dataset card and manifest are generated from source evidence, not hand
  filled after the fact;
- every blocking finding stops the run before training;
- non-blocking findings remain visible and require an explicit disposition;
- repeated ingestion of the same snapshot is idempotent.

### Phase 2 — Complete modality lifecycles

Implement each modality through the same 15-stage contract:

1. ingestion
2. validation
3. profiling
4. curation
5. deduplication and splitting
6. feature or representation preparation
7. modelling
8. evaluation
9. data explainability
10. model explainability
11. signed packaging
12. inference
13. monitoring
14. documentation
15. recovery and resource tests

**Modality-specific minimums**

- **Text/documents:** document boundaries, encoding, language, OCR confidence,
  layout, tables, redaction, page-level provenance, chunking leakage, and
  citation/source traceability.
- **Images:** dimensions, channels, color space, corrupt files, EXIF/PII,
  near-duplicate detection, augmentation provenance, saliency/occlusion, and
  localization limits.
- **Audio:** sample rate, channels, duration, clipping, silence, codec failure,
  speaker leakage, transcript provenance, time-local explanations, and
  background-noise robustness.
- **Video:** container/codec validity, frame sampling, temporal boundaries,
  identity leakage, audio/video synchronization, temporal explanations, and
  resource budgets.
- **Graphs:** node/edge identity, directionality, self-loops, components,
  temporal edges, graph leakage, inductive/transductive split policy, and
  subgraph explanations.
- **Geospatial:** CRS, geometry validity, units, spatial joins, spatial
  autocorrelation, boundary leakage, spatial/temporal splits, raster nodata,
  and map-based explanations.

**Exit criteria**

- each modality has all 15 stages passing on deterministic fixtures;
- invalid, empty, corrupted, oversized, shifted, and adversarial fixtures have
  defined behavior;
- a modality cannot be advertised as supported if any stage is missing or
  extension-only;
- resource and recovery evidence is attached to the modality capability row.

### Phase 3 — Modelling and evaluation maturity

**Deliverables**

- task-specific evaluation policies for classification, regression, survival,
  ranking, forecasting, uplift, ordinal, multilabel, and multioutput;
- uncertainty policies covering intervals, conformal coverage, calibration,
  abstention, and shift limitations;
- subgroup and intersectional fairness reports with uncertainty and small-group
  power warnings;
- release-blocking quality policies;
- model cards and model-family conformance tests;
- independent reference implementations for critical metrics.

**Methods**

- nested cross-validation;
- grouped and time-blocked resampling;
- repeated seeds with confidence intervals;
- calibration and conformal coverage checks;
- slice and intersectional analysis;
- sensitivity analysis for thresholds and missingness;
- leakage and distribution-shift checks;
- reference calculations using independent implementations such as Fairlearn
  where applicable.

**Exit criteria**

- unsupported task/metric/uncertainty combinations fail before fit;
- diagnostics are classified as blocking, warning, or informational;
- warnings cannot silently become a green release;
- every promoted model has an evaluation policy, model card, and limitations;
- benchmark claims include seed count, intervals, data version, license, and
  resource measurements.

### Phase 4 — Data and model explainability

Create one portable explanation contract containing:

- model fingerprint and bundle digest;
- input fingerprint and dataset snapshot ID;
- preprocessing or feature-graph fingerprint;
- method, parameters, output, timestamp, and software version;
- uncertainty or stability information;
- known limitations and causal-versus-associational warning;
- redaction and privacy decision;
- links to the evidence package and model card.

Implement and test:

- global, local, counterfactual, interaction, and error explanations;
- modality-specific explanations for image, audio, video, document, graph,
  and geospatial inputs;
- stability tests under resampling and small perturbations;
- fidelity tests against an independent reference or controlled synthetic
  ground truth;
- privacy/redaction tests;
- JSON and human-readable HTML evidence packages;
- optional inference references to explanation artifacts without making
  explanations a hidden side effect of prediction.

**Exit criteria**

- explanations are reproducibly tied to the exact model, input, data, and
  method;
- all known limitations are machine-readable and visible in the model card;
- generic explanation stability and fidelity thresholds are declared per
  method and enforced by the release evaluator; modality-specific repeated
  perturbation and independent-ground-truth suites remain required;
- no causal claim is emitted by an associational method.

### Phase 5 — Engineering and production reliability

Prove these behaviors with fault-injection tests:

- deterministic replay and documented numeric tolerances;
- idempotent retries and duplicate-event handling;
- safe resume and cancellation;
- worker loss, timeout, memory exhaustion, disk-full, and partial writes;
- corrupt artifact, stale checkpoint, dependency mismatch, and invalid
  signature;
- connector retry exhaustion, changed source snapshot, incomplete lineage, and
  rejected approval.

Add engineering gates:

- mutation testing for critical contracts;
- property-based and parser fuzzing;
- performance budgets for ingest, fit, package, load, and inference;
- API compatibility tests and previous-release migration tests;
- test-flake tracking with an explicit quarantine policy;
- network-isolated wheel and sdist installation;
- reproducible build comparison;
- clean-clone release rehearsal.

**Exit criteria**

- every injected failure ends in a safe terminal state or a documented,
  recoverable state;
- no partial artifact is mistaken for a completed artifact;
- critical mutations are killed by tests;
- no unexplained flaky or skipped test is included in the release evidence;
- the same source and lockfile produce equivalent package and evidence hashes
  within declared nondeterminism.

### Phase 6 — Security, governance, and observability

**Security**

- pin third-party actions to immutable SHAs;
- run dependency, license, secret, static, and container scans;
- produce SBOM and provenance attestations;
- document plugin trust boundaries;
- define signer identity, custody, expiry, revocation, emergency rotation, and
  verification behavior;
- keep unsigned-bundle bypasses out of production defaults;
- define PII redaction, retention, deletion, and access review.

**Observability**

- request/run IDs;
- structured redacted logs;
- latency, error, throughput, memory, and resource metrics;
- data quality, drift, novelty, calibration, fairness, and coverage metrics;
- retention and alert thresholds;
- documented SLOs and incident response ownership.

**Exit criteria**

- no unresolved critical or high security finding without a signed waiver;
- security and privacy decisions are attached to the affected dataset, bundle,
  or service;
- monitoring can distinguish data failure, model failure, infrastructure
  failure, and policy failure;
- an operator can identify the affected run and rollback target from an event
  record.

### Phase 7 — Site, documentation, and release evidence

The Astro site must be the standalone developer-facing documentation surface.
It must not reroute readers to repository Markdown files for core onboarding.

Required content:

- a truthful “start here” path;
- installation and clean-clone quickstart;
- data manifest and curation guide;
- modality support matrix;
- architecture and lifecycle diagrams;
- API and extension contracts;
- data, model, and explanation card guides;
- operations, security, troubleshooting, and recovery guides;
- release evidence and known limitations;
- versioned migration guidance.

Use a developer-advocate voice: concrete, direct, explanatory, and honest
about trade-offs. Use diagrams to explain lifecycle, evidence lineage, trust
boundaries, and failure recovery. Use real copy and runnable examples, not
generic marketing language.

**Site acceptance**

- every published route builds under the `/super-charge-ml/` base path;
- every internal link resolves without a repository `.md` redirect;
- navigation, keyboard focus, reduced motion, contrast, and semantic headings
  pass accessibility checks;
- Playwright smoke tests cover landing, onboarding, architecture, data,
  operations, and troubleshooting;
- the deployment workflow produces a verified public URL and runs a post-deploy
  route smoke test;
- the site clearly distinguishes beta support from extension-only capability.

## 5. Exact release acceptance criteria

The production-beta release is accepted only when all criteria below have
authoritative evidence in the release evidence index.

### Correctness and coverage

- 100% of required contract tests pass; no unexplained skips or xfails.
- Core package contract coverage is at least 85%; critical paths have explicit
  mutation coverage with a target of at least 70% killed mutations.
- Every advertised modality passes all 15 lifecycle stages.
- Every advertised task/model combination appears in the capability registry.
- Invalid, empty, malformed, corrupted, oversized, shifted, and adversarial
  fixtures have deterministic, documented outcomes.

### Statistical validity

- Release benchmarks use versioned, licensed datasets and at least five seeds
  or an explicitly justified resampling design.
- Results include point estimates, uncertainty intervals, runtime, memory,
  failure rate, and hardware/software identity.
- No quality regression greater than 5% or runtime regression greater than 10%
  is accepted without a signed release decision explaining the trade-off.
- Calibration, uncertainty coverage, subgroup fairness, and slice behavior are
  evaluated wherever the capability registry declares them applicable.

### Reliability and recovery

- All required fault-injection scenarios pass.
- Resume, retry, cancellation, rollback, and recovery are idempotent where
  promised.
- Partial, corrupt, unsigned, stale, or mismatched artifacts are rejected
  safely.
- Repeated runs from the same snapshot and lockfile meet declared replay
  tolerances.

### Security and governance

- No unresolved critical/high vulnerability, secret, or license violation.
- Workflows use immutable action references or a documented equivalent control.
- SBOM, dependency/license inventory, provenance, signatures, and verification
  results are published with the release evidence.
- Dataset cards, model cards, explanation metadata, retention, privacy, and
  approval records exist for every promoted artifact.

### Independent proof

- An independent reviewer reproduces the package from a clean clone.
- The reviewer runs the clean-install, benchmark, fault-injection, and site
  smoke suites without relying on developer-local state.
- At least one critical metric, fairness result, and explanation result is
  recomputed independently.
- The reviewer signs the release decision or records a specific blocking issue.

## 6. Evidence index format

For every criterion, record:

```text
criterion_id:
  claim:                         exact claim being made
  scope:                         modality/task/platform/release
  raw_inputs:                    hashes and source identities
  command_or_test:               reproducible command
  artifact:                      path, digest, and schema version
  result:                        pass/fail/waived
  reviewer:                      independent identity
  observed_at:                   UTC timestamp
  limitations:                   known uncertainty or excluded scope
  decision:                      release-blocking disposition
```

The evidence index is itself versioned, signed, and shipped with the release.

## 7. Operating cadence

Measure, observe, evaluate, and evolve in that order:

1. **Measure:** collect raw quality, resource, statistical, security, and
   user-facing signals.
2. **Observe:** connect signals to run, dataset, model, bundle, and deployment
   identities with redaction and retention controls.
3. **Evaluate:** compare against declared thresholds, uncertainty intervals,
   prior releases, and independent references.
4. **Evolve:** change the contract, implementation, documentation, or support
   label; then rerun the affected evidence, not just the unit tests.

Each release should end with a short retrospective: what failed, what was
hard to observe, which threshold was noisy, which capability should be
demoted, and which new evidence is required next time.

## 8. Atomic implementation commits

Keep commits small enough that each one has one reviewable reason to exist.
Use this sequence unless dependency discovery changes it:

1. `docs: define production beta boundary and evidence model`
2. `feat(data): add versioned dataset manifest and lineage contracts`
3. `feat(data): add structured curation findings and dataset cards`
4. `feat(data): add source connectors and incremental ingestion semantics`
5. `feat(capabilities): register modality and task support boundaries`
6. `feat(modalities): implement one modality lifecycle at a time`
7. `feat(evaluation): add task-specific uncertainty fairness and release gates`
8. `feat(explainability): add portable explanation evidence contract`
9. `test(reliability): add fault injection fuzzing mutation and replay gates`
10. `feat(security): add provenance SBOM signing and governance evidence`
11. `feat(observability): add redacted events metrics drift and SLO checks`
12. `docs(site): publish standalone developer onboarding and operations guides`
13. `ci(release): add clean-clone matrix and post-deploy verification`
14. `release: publish production beta evidence package`

- Recent evidence: `4a45acb` and `3317bbf` execute and document the bounded
  local form of all 15 fault scenarios; external transport, container,
  kernel-resource, and independent-fault evidence remain outside that claim.
- Recent evidence: `96f7c6a` adds exact-bundle, digest-checked, idempotent
  inference nodes to persisted run lineage. This joins serving evidence to the
  package graph but does not constitute independent release review or a
  production deployment rehearsal.

Do not combine a modality implementation, a release-policy change, and a site
rewrite in one commit. Every commit must leave the repository buildable and
must state its evidence command in the PR description.

## 9. Stop conditions

Stop and revise the plan if any of these occurs:

- a modality requires a different provenance or safety model that the common
  contract cannot express;
- a “pass” depends on silently dropping data, hiding small groups, or treating
  an advisory diagnostic as success;
- an external dependency cannot provide the license, version, or security
  evidence required by the capability claim;
- the all-modalities scope would make the beta impossible to independently
  review;
- a performance or privacy trade-off changes the product boundary.

In those cases, demote the capability to extension-only with a documented
reason. Do not weaken the acceptance criteria to preserve the label.
