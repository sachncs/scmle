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

### Historical evidence ledger — 20 October 2026

This is the implementation boundary at the time of writing. “Implemented”
means the code path exists; it does not mean the capability is supported.

| Area | Current label | Evidence and boundary |
| --- | --- | --- |
| Tabular ingestion, manifest, structured curation | Implemented / Tested | Unit and integration tests cover deterministic snapshots, local raw-byte hashes, findings, persisted manifests, machine-generated dataset cards, schema-evolution decisions, an atomic durable source registry that survives across runs, and fail-closed blocking-finding disposition before training. Broker delivery and distributed exactly-once semantics remain incomplete. |
| Local, DB-API, bounded object, range-backed JSONL, and warehouse/lakehouse source connectors | Implemented / Tested | Connector contracts cover local files, injected DB-API factories, bounded injected object readers for CSV/JSONL/Parquet, caller-owned versioned decoders for arbitrary bounded object payloads, raw local-file and object-byte SHA-256 evidence bound to resumable checkpoints with changed-snapshot rejection, range-backed JSONL/NDJSON reads with byte checkpoints, and `WarehouseConnector` callback boundaries with query-bound identity, retries, bounded rows, schema validation, and resumable checkpoints. Broker-specific transports and decoder-owned semantics remain extension work. Evidence: `tests/unit/test_connectors.py`. |
| Incremental ingestion semantics | Implemented / Tested | Transport-neutral event contracts bind event IDs to ordered checkpoints and watermarks, classify duplicates, make late-data policy explicit, serialize multi-writer commits with durable atomic state, and provide an idempotent sink commit boundary. `EventStreamSource` adds bounded, retryable callback adapters for Kafka-, Pulsar-, Kinesis-, Pub/Sub-, Event Hubs-, NATS-, and Redpanda-style URI boundaries. `EventConsumer` binds a consumer group, stages acknowledgement evidence after the sink commit, retries a pending acknowledgement after restart, and never reruns the sink for that retry. Atomic state, run evidence, source registries, observability retention, fault evidence, and signing-key rotation flush both file contents and the containing directory after replacement (`b420ee3`). Broker SDK leases, acknowledgements, transactions, and distributed exactly-once guarantees remain transport-owner evidence. |
| Tabular library training paths | Implemented / Tested | The pre-training capability gate admits only explicitly declared tabular library paths. Dataset manifests preserve the caller-declared modality, and non-tabular declarations fail closed before model fitting. This is the current beta boundary, not all-modality support. |
| Non-tabular curation and data understanding | Implemented / Tested | `DocumentDataset.curate()` and the shared `ModalityCurationReport` emit JSON-native evidence for documents, images, audio, video, graphs, and geospatial vectors: bounded profiles, identities, composition, duplicate/shape/encoding/CRS/graph findings, label state, and blocking spatial or conflicting-label risks where applicable. `ModalityCurationProject` now depends only on the curation contract, and `ExtensionCurationAdapter` accepts arbitrary caller-owned records without forcing split, feature, label, or modelling hooks; both persist signed dataset-card evidence with identity/count binding. These paths do not provide OCR, production-scale representation models, or a supported `MLProject` non-tabular training path. Evidence: `scml/data/modalities/curation.py`, `scml/models/modality_lifecycle.py`, `scml/api/project.py`, `tests/unit/test_modality_project.py`, `tests/unit/test_modality_lifecycle.py`. |
| Complete experimental modality baseline lifecycle | Implemented / Tested | `ModalityLifecycle` runs all 15 required stages for text/documents, images, WAV audio, probed video metadata, inductive graph nodes, CRS-declared geospatial vectors, and rasters with caller-owned label sidecars. Raster is represented as its own capability identity rather than being reported as vector geospatial data; it can remain metadata-only or accept a named caller-owned decoder for bounded pixel summaries and measured nodata evidence. Concrete adapters own modality-specific split and feature policies; the shared stages produce curation, evaluation, data/model explanations, bounded modality-specific evidence where a decoder or geometry policy exists, signed atomic packages, verified inference, drift evidence, documentation, and tamper/resource recovery evidence. `ModalityProject` is now the public typed entry point for these adapters, with atomic lifecycle-report persistence and an explicit experimental declaration. Stage payloads are validated JSON objects rather than stringified notes, so evidence can be queried without reparsing human text. These are experimental baselines, not production-supported claims or replacements for `MLProject`/`ModelBundle`. Evidence: `scml/contracts/modalities.py`, `scml/contracts/capabilities.py`, `scml/contracts/tasks.py`, `scml/models/modality_lifecycle.py`, `scml/api/project.py`, `tests/unit/test_modality_contracts.py`, `tests/unit/test_modality_lifecycle.py`, `tests/unit/test_modality_project.py`, `tests/unit/test_media_modalities.py`. |
| Lifecycle performance budgets | Implemented / Tested | `PerformanceObservation` and `PerformanceBudget` record machine-, software-, and timestamp-bound elapsed-time and observed process-peak evidence for ingest, fit, package, load, and inference. `PerformanceBaseline` compares a current run with the median of at least two schema- and workload-consistent reference runs and blocks declared relative time or memory regressions. Missing operations, memory evidence, environment identity drift, or sample-count drift fail closed. The experimental modality lifecycle emits the five observations, persists the baseline comparison, and rejects a run before returning a green report when a gate fails. Its memory value is the existing cross-platform process peak observed at each operation boundary, which is conservative because it is a lifetime peak rather than an isolated allocation trace. Evidence: `scml/contracts/performance.py`, `scml/models/modality_lifecycle.py`, `tests/unit/test_performance_contracts.py`, `tests/unit/test_modality_lifecycle.py`; focused result at `6162923`: 24 passed under `-W error::ResourceWarning`. |
| Experimental document baseline lifecycle | Implemented / Tested | `DocumentClassifier` provides bounded TF-IDF feature preparation, logistic modelling, held-out evaluation, linear global/local term explanations, source-backed citations when the reader owns raw-source identity, an atomic package with manifest and payload/model digest verification, and a profile-drift monitor. `DocumentReader` also exposes an explicit caller-owned layout/OCR/table extraction seam with source-digest validation. This remains experimental: it is not integrated with `MLProject` run evidence, the signed `ModelBundle`, production OCR/layout/table quality governance, semantic citation grounding, or the full document 15-stage release evidence. The generic `ModalityHttpServing` boundary is available when the caller supplies and owns a decoder. Evidence: `scml/data/modalities/document.py`, `scml/models/documents.py`, `scml/inference/modality_serving.py`, `tests/unit/test_document_data.py`, `tests/unit/test_document_model.py`, `tests/unit/test_modality_serving.py`. |
| Experimental image baseline lifecycle | Implemented / Tested | `ImageClassifier` provides bounded resized-pixel preparation, logistic modelling, held-out evaluation, global/local pixel contribution evidence, black-baseline occlusion, an atomic trusted package with digest verification, and mean-pixel/duplicate drift monitoring. It remains experimental: it is not integrated with `MLProject` run evidence, the signed `ModelBundle`, reference serving, raw-image saliency fidelity, localization, or the full image 15-stage release evidence. Evidence: `scml/models/images.py`, `tests/unit/test_image_model.py`, `tests/unit/test_modality_lifecycle.py`. |
| Image, document, graph, audio, video, and geospatial data foundations | Implemented / Tested | Validation, identity, bounded profiling, splitting, feature-preparation, and now complete experimental baseline lifecycle slices exist with focused tests. Audio and video require explicit codec/probe boundaries; geospatial vectors require validated GeoJSON geometry and CRS provenance, explicit caller-owned metadata-preserving reprojection, conservative envelope-boundary leakage evidence, bounded same-CRS spatial joins through a caller-owned predicate, while rasters have a distinct capability identity and an explicit caller-owned decoder boundary for bounded pixel summaries, with dimensions, dtype, CRS, and transform metadata always required. These remain experimental because the baselines are not integrated with the supported `MLProject` run evidence, reference serving, external fault execution, or independent release review. Evidence: `scml/contracts/modalities.py`, `scml/contracts/capabilities.py`, `scml/contracts/tasks.py`, `scml/data/modalities/geospatial.py`, `scml/models/modality_lifecycle.py`, `tests/unit/test_modality_lifecycle.py`, `tests/unit/test_document_data.py`, `tests/unit/test_image_data.py`, `tests/unit/test_graph_data.py`, `tests/unit/test_media_modalities.py`. |
| Release evaluation | Implemented / Tested | Explicit success criteria produce a task-aware policy, release findings, artifact digests, and a persisted evidence index. Runs without thresholds are explicitly `policy_not_declared`, and any run with an unexecuted or unsafe fault plan is independently blocked from release. Repeated benchmark records now include observed runtime, peak memory, completed-seed count, failure rate, machine, and interpreter identity; optional baseline ceilings fail closed when evidence is missing or exceeded. Evidence: `benchmarks/run_benchmarks.py`, `benchmarks/check_regression.py`, `tests/benchmarks/test_benchmark_gate.py`, `tests/benchmarks/test_benchmark_resources.py`; focused result at `5ec2e7e`: 10 passed under `-W error::ResourceWarning`. |
| Dataset and model cards | Implemented / Tested | Curated runs persist machine-generated cards tied to dataset snapshots, model-manifest digests, evaluation context, intended use, limitations, privacy, and review status. Exact-bundle promotion approvals are required at the registry boundary, and champion/rollback approvals are HMAC-signed and verified against the exact target. Independent release review remains incomplete. |
| Content-linked evidence graph | Implemented / Tested | Completed packages persist `evidence_graph.json`, a deterministic acyclic graph linking raw source, snapshot, curation, feature graph, split, model, evaluation, explanations, and bundle identities. `InferenceEvidenceLinker` verifies the persisted graph digest and exact bundle digest before atomically adding an idempotent inference-event node and serving edge. Conflicting nodes, dangling edges, cycles, and invalid serialized order fail closed. Independent release review remains incomplete. Evidence: `scml/contracts/lineage.py`, `scml/inference/evidence.py`, `tests/unit/test_lineage_contract.py`, `tests/unit/test_inference_evidence.py`, `tests/integration/test_end_to_end.py`. |
| Portable model explanation evidence | Implemented / Tested | Permutation-importance, partial-dependence, and bounded pairwise-interaction reports are packaged as privacy-aware JSON and self-contained human-readable HTML artifacts with identity, limitation, and source-report links; bounded error-analysis records held-out failure rows without copying raw feature values. The explicit `ModelBundle.counterfactual()` capability is tested for actionable-column and minimal-change invariants, while run-level counterfactual generation remains opt-in because it requires caller-owned actionability and background policy. Generic explanation quality measures provide top-k stability, normalized fidelity, and feature-space perturbation stability, and release policies can block below-threshold evidence. Experimental modality lifecycles now gate repeated feature-space perturbations across the held-out partition with a declared 128-sample bound. Raw-modality semantic fidelity and production-scale repeated explanation runs remain incomplete. Evidence: `scml/artifacts/explanation_report.py`, `scml/models/modality_lifecycle.py`, `tests/unit/test_explanation_report.py`, `tests/integration/test_problem_spec.py`, `tests/unit/test_modality_lifecycle.py`. |
| Experimental modality explanation quality | Implemented / Tested | The shared experimental lifecycle independently reconstructs each linear baseline's decision output from coefficients and intercepts, checks zero-ablation direction consistency for a held-out row, and gates deterministic top-k ranking stability across every held-out sample up to a declared 128-sample bound. Adapter-owned evidence now covers raw document span occlusion, bounded raw and transformed-image region occlusion, bounded raw WAV waveform windows plus deterministic additive-noise score probes at declared SNRs, bounded decoded-video frame occlusion, raw CRS-coordinate axis occlusion, raw graph node-feature occlusion, bounded deterministic graph one-hop subgraph context with explicit node/edge limits and truncation evidence, and bounded decoded-raster pixel-region occlusion with source-CRS window envelopes when a named decoder is supplied. Metadata-only rasters remain explicitly unavailable for pixel evidence; encoded-payload occlusion remains outside the decoder-only contract. The quality result is thresholded and persisted in model-explanation evidence across document, image, audio, video, graph, geospatial, and raster fixtures. This validates bounded representations only; real-environment audio robustness, perceptual quality, semantic localization/saliency, raw graph-topology attribution, causal claims, and production-scale repeated perturbation suites remain incomplete. Evidence: `scml/models/modality_lifecycle.py`, `scml/data/modalities/audio.py`, `scml/data/modalities/geospatial.py`, `scml/explainability/quality.py`, `tests/unit/test_modality_lifecycle.py`, `tests/unit/test_explanation_quality.py`; graph bound regression at `30d51df`. |
| Promotion and rollback governance | Implemented / Tested | Champion promotion and rollback require an independent, HMAC-signed approval bound to model, version, bundle content, reviewer decision, evidence, and timestamp; invalid, missing, or tampered approvals are rejected and audit records preserve the signature. Lifecycle retraining stages challengers until approval. Independent release sign-off remains incomplete. |
| Redacted inference observability | Implemented / Tested | `PredictionLog` emits a redacted sidecar, policy evaluation checks required signals, freshness, future timestamps, and upper-bound threshold breaches, retention enforcement atomically removes expired sidecar events without touching the raw prediction log, deterministic routing produces idempotent alert IDs, `SLOMonitor` evaluates worst-sample windows with minimum-data checks, error-budget evidence, owner, runbook, and alert route, `AlertDispatcher` persists idempotent local or bounded HTTPS webhook delivery outcomes, and `IncidentWorkflow` records chained open/acknowledge/resolve transitions with run and rollback identity. Serving defaults and full external paging/ticketing evidence remain incomplete. Evidence: `scml/monitoring/alerts.py`, `scml/monitoring/slo.py`, `scml/monitoring/incidents.py`, `tests/unit/test_alerts.py`, `tests/unit/test_slo.py`, `tests/unit/test_incidents.py`. |
| Fault injection and recovery | Implemented / Tested (local boundary) | `LocalFaultExecutor` now executes all 15 declared scenarios through real bounded process, timeout, atomic-write, artifact, signature, checkpoint, snapshot, retry, duplicate-event, lineage, and approval boundaries; `FaultInjectionHarness` persists digest-bound evidence and fail-closed status transitions. Controlled disk and memory cases use deterministic local capacity guards, not physical disk exhaustion or host-wide OOM. Cloud transport, container, kernel-resource, dependency-installation, connector, and independent external-fault execution evidence remain incomplete. Evidence: `scml/governance/local_faults.py`, `tests/unit/test_local_faults.py`, `tests/unit/test_reliability_harness.py`. |
| Test stability and quarantine policy | Implemented / Tested | `check_test_stability.py` runs pytest repeatedly, parses durable JUnit outcomes, compares test inventories, records runner return codes, and fails closed for unlisted failures, errors, skips, inventory changes, and expired quarantines. A quarantine entry requires an owner, reason, and expiry; active quarantines remain visible as `passed_with_quarantine`. The scheduled full-suite workflow exercises the policy three times against the API and reliability contract smoke set and uploads the JSON evidence. It does not replace the full test suite or prove that optional-dependency skips are supported. Evidence: `scripts/check_test_stability.py`, `tests/quarantine.json`, `tests/unit/test_test_stability.py`, `.github/workflows/full-suite.yml`; local three-run result at `65b0676`: 22 test cases per run, all passed. |
| Security, provenance, SBOM, and release CI | Implemented / Tested | Bundle signing, run provenance artifacts, deterministic CycloneDX SBOM generation, a deterministic runtime license inventory, CI artifact publication, immutable CI action pins, a canonical HMAC-signed release attestation, a versioned HMAC-signed release evidence index, an exact-SHA clean-clone rehearsal job, a durable review-artifact writer, an executable exact-commit review verifier, a signed `IndependentReleaseReview` contract, and a metadata-only `SigningKeyRegistry` for identity, expiry, retirement, replacement, and immediate revocation are implemented. The review contract requires an independent reviewer, exact source commit, clean-clone and environment digests, reproducible commands, recomputed metric/fairness/explanation claims, and signed artifact digests. Secret custody remains external, existing raw bundle signatures still require operator key-policy integration, and an external reviewer signature and a full clean-release execution remain incomplete. The all-extras audit gate passed locally on Python 3.13 at `c85118e`; the hosted Linux job remains the cross-platform authority. Evidence: `scripts/check_dependency_licenses.py`, `.github/workflows/ci.yml`, `scripts/verify_release_review.py`, `scml/contracts/evidence.py`, `scml/governance/keyring.py`, `scml/governance/review.py`, `tests/unit/test_keyring.py`, `tests/unit/test_evidence_contracts.py`, `tests/unit/test_independent_review.py`. |
| Astro developer site | Partial | Standalone docs routes, `/super-charge-ml/` base-path link resolution, generated accessibility smoke checks, browser smoke coverage for all 17 public routes plus keyboard navigation, a capability matrix, a dedicated 15-stage modality-lifecycle guide, cards/explanations, security, recovery, migration guidance, pinned Pages artifact/deploy jobs, and post-deploy route smoke coverage build successfully. The requested public URL currently returns HTTP 404 because the GitHub repository is private; repository visibility/Pages configuration must be resolved by the owner before deployment can be verified. Hosted deployment, independent review, and exhaustive production modality claims remain incomplete. |

### Historical verification snapshot — 20 October 2026

The snapshot below is retained as an audit trail. The dated evidence records
later in this document supersede it for the current release decision; the
latest strict current-head result is recorded at `ae6b1f2` and the latest
workflow-control changes at `e825655`.

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

The lifecycle-performance slices at `8758a1f`, `3018453`, `3e24a08`,
`f43b969`, and `6162923` measure ingest, fit, package, load, and inference
with machine, software, UTC timestamp, and observed process-peak context. They
enforce time and memory budgets, compare a current run with the median of at
least two reference runs, and fail closed when environment or sample-count
identity drifts. The memory value is a conservative process lifetime peak at
each boundary, not an isolated allocation trace. This slice does not make the
non-tabular lifecycle production-supported.

The test-stability slice at `65b0676` passed three local attempts of the
CI-shaped API and reliability smoke set with 22 cases per attempt and no
quarantined failures. The policy is intentionally not a green-label bypass:
an active quarantine is reported as `passed_with_quarantine`, while an
unlisted failure, skip, inventory change, runner error, or expired exception
fails the command. The scheduled workflow still needs a hosted run before it
counts as external CI evidence.

The modality explanation-quality slice at `50c3e8b` passed 37 focused
explanation and lifecycle tests with warning failures promoted to errors. It
adds an independent linear-reference reconstruction and zero-ablation check
to the experimental model-explanation stage. The evidence is deliberately
scoped to feature-space linear baselines; it does not close the raw-modality
explanation or causal-validity gaps.

The benchmark resource-evidence slice at `5ec2e7e` records runtime, peak
memory, failure rate, completed repetitions, machine, and interpreter identity
even when a dataset run fails. Declared runtime, memory, and failure-rate
ceilings now fail closed when measurements are missing or exceed policy. The
checked-in historical result artifacts predate these fields, so a fresh
five-seed benchmark run is still required before resource gates can be
accepted as release evidence.

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

## 10. Current local preflight record

At pushed commit `ef48335`, the current-head strict regression gate passed on
2026-10-21 with:

- `uv run pytest -q tests -W error::ResourceWarning`: 2,247 passed and 88
  skipped in 28 minutes;
- no failure, error, or warning promoted to a failure by the strict warning
  policy.

This verifies the current source tree's contract and reliability suite, but it
is not a complete production release decision: coverage, mutation, benchmark,
clean-clone, hosted CI/Pages, external-fault, and independent-review evidence
remain separate gates below.

At pushed commit `2a7b939`, the post-manifest current-head strict regression
gate also passed on 2026-10-21:

- `uv run pytest -q tests -W error::ResourceWarning`: 2,249 passed and 88
  skipped in 27 minutes 53 seconds;
- no failure, error, or warning promoted to a failure by the strict warning
  policy after the modality manifest changes.

At pushed commit `1e5ee03`, the current-head strict regression gate passed
again after the verified modality predictor implementation:

- `uv run pytest -q tests -W error::ResourceWarning`: 2,251 passed and 88
  skipped in 28 minutes 4 seconds;
- no failure, error, or warning promoted to a failure by the strict warning
  policy.

The latest complete local release preflight at pushed commit `2eb767e` passed
on 2026-10-21:

- bounded contract suite: 1,897 passed, 75 skipped, 82.85% whole-package
  coverage; the declared core scope measured 86%;
- mutation gate: 1,358 of 1,898 mutants killed, for a 71% score against the
  configured 70% minimum;
- five-seed benchmark regression gate: all four offline datasets passed with
  zero failures and zero failed repetitions; breast cancer, wine, and digits
  were statistically tied with their independent references, while diabetes
  progression was ahead;
- static, dependency, API, artifact, and site build gates passed; wheel and
  source distribution reproducibility and network-isolated installation were
  verified; the Astro build produced all 15 static pages and npm audit found
  no vulnerabilities.

The two SQLite `ResourceWarning` summaries and scikit-learn deprecation
warnings remain explicit hygiene debt. The preflight is local evidence and
does not replace the independent reviewer, hosted Pages verification, or
external fault execution.

The latest complete release preflight at pushed commit `357e392` passed on
2026-10-21:

- bounded contract suite: 1,891 passed, 75 skipped, 82.82% whole-package
  coverage; the declared core scope measured 86%;
- mutation gate: 1,358 of 1,898 mutants killed, for a 71% score against the
  configured 70% minimum;
- five-seed benchmark regression gate: all four offline datasets passed with
  zero failures and zero failed repetitions; breast cancer, wine, and digits
  were statistically tied with their independent references, while diabetes
  progression was ahead;
- static and dependency gates: Ruff, format check, mypy, Bandit, and
  pip-audit passed;
- artifact gate: wheel and source distribution checks passed, both builds were
  reproducible, and network-isolated wheel and source-distribution installs
  passed;
- site gate: Astro built all 15 static pages, accessibility smoke passed, and
  npm audit reported zero vulnerabilities.

This remains local evidence only. The two existing SQLite `ResourceWarning`
summaries and scikit-learn deprecation warnings remain hygiene follow-ups. The
run does not replace clean-clone execution, hosted optional-dependency audits,
external/container/kernel fault injection, public GitHub Pages verification,
or independent review and signature.

At commit `29d1913` (`docs(evaluation): record benchmark resource gates`),
the local release preflight passed on 2026-10-21:

- bounded contract suite: 1,866 passed, 89 skipped, 82.34% coverage;
- mutation gate: 1,294 of 1,794 mutants killed, for a 72% score against a
  70% minimum;
- benchmark regression gate: 4 of 4 datasets passed with zero recorded
  failures, including runtime and peak-memory evidence;
- static and dependency gates: Ruff, format check, mypy, Bandit, and
  pip-audit passed;
- artifact gate: reproducible wheel and source distribution builds passed,
  followed by network-isolated wheel and source-distribution installation;
- site gate: Astro built all 15 static pages and npm audit reported zero
  vulnerabilities.

The test run emitted two non-failing `ResourceWarning` instances from the
Python 3.14 free-threaded AST/convention test path. The benchmark run emitted
scikit-learn deprecation warnings. These are recorded as follow-up hygiene
items, not suppressed as release evidence. This record is local evidence only:
it does not replace clean-clone execution, hosted optional-dependency audits,
external/container/kernel fault injection, public GitHub Pages verification,
or independent review and signature.

Subsequent focused evidence:

- The exact-SHA clean-clone rehearsal for `959fd2f` checked out a clean
  detached tree, installed the release extras on Python 3.14, and passed the
  workflow acceptance command with `1,868 passed, 73 skipped` in 6 minutes 46
  seconds. `scml --help` exited successfully from that clone. This is local
  reproduction of the hosted rehearsal, not independent reviewer evidence.
- `0de10f5` adds deterministic image occlusion evidence over the declared
  resized feature representation. Five focused image-model tests pass under
  `-W error::ResourceWarning`, with Ruff and mypy clean. This is bounded
  perturbation evidence with an explicit black baseline; it does not claim
  raw-image saliency fidelity, localization, or production image support.
- `ad4e434` adds bounded raw-text term-ablation evidence to the experimental
  document baseline. Seven focused document-model tests pass under
  `-W error::ResourceWarning`, with Ruff and mypy clean. The result is tied to
  the exact input and model digest but does not claim OCR, citation grounding,
  causal attribution, or supported document modelling.
- `87945a8` adds deterministic feature-space perturbation stability to the
  shared experimental model-explanation stage. It evaluates repeated local
  rankings with a declared relative step, retains the existing independent
  linear reconstruction and zero-ablation evidence, and gates the lifecycle
  at a 0.9 stability minimum. The focused explanation and modality lifecycle
  suite passes 23 tests under `-W error::ResourceWarning`; this does not claim
  raw-modality saliency, temporal localization, citation grounding, or causal
  validity.
- `8d8326e` adds bounded time-local WAV window score evidence through the
  polymorphic modality explanation hook. The decoder boundary, frame offsets,
  observed counts, and feature-space-only limitation are persisted; non-WAV
  samples remain explicitly unavailable without a decoder adapter. The focused
  media and lifecycle suite passes 37 tests under
  `-W error::ResourceWarning`; this does not claim raw waveform attribution,
  transcript grounding, or causal audio-event explanations.
- `fc5aa3d` adds an explicit caller-owned video frame decoder boundary,
  deterministic evenly spaced frame sampling, per-frame pixel limits, frame
  statistics, sampled-index provenance, and temporal frame-statistics
  occlusion evidence. The media and lifecycle slice passes 44 tests under
  `-W error::ResourceWarning`, with Ruff and mypy clean. This remains
  experimental: it does not claim raw pixel saliency, motion attribution,
  audio-video synchronization, causal temporal localization, or production
  video serving.
- `9a1b4b6` moves shared HMAC signing primitives into the contracts layer so
  release evidence no longer imports governance from a lower-level contract.
  The architecture rule, attestation, evidence-index, Ruff, and mypy focused
  tests pass (`12 passed` under `-W error::ResourceWarning`).
- `1e7ef01` adds a modality-neutral `ModalityManifest` for non-tabular
  datasets, binds its content hash and sample count before lifecycle
  execution, and persists source, lineage, privacy, task, and finding
  evidence in the generated dataset card. The focused modality contract and
  lifecycle suite passes 31 tests under `-W error::ResourceWarning`, with
  Ruff and mypy clean. Omitted governance remains explicitly unassessed.
- `b091df3` adds a verified `ModalityPredictor` library boundary that loads a
  signed package, checks modality and feature-schema identity, and predicts
  on new adapter-owned samples. The focused modality lifecycle suite passes
  33 tests under `-W error::ResourceWarning`, with Ruff and mypy clean. This
  is library inference evidence, not reference HTTP serving or raw transport
  support.
- `84193bf`, `8cb1672`, `65dc5b6`, and `c295e9a` add lifecycle-bound
  image region occlusion, graph one-hop context, CRS-bound geospatial
  geometry/feature evidence, and document span provenance respectively. The
  modality lifecycle test slice passes 17 tests under
  `-W error::ResourceWarning`; all four paths retain representation,
  decoder, topology, CRS, and causal limitations in machine-readable output.
- `c85118e` runs the optional-extra security gate on Python 3.13 with every
  declared extra installed. The license inventory covers 149 packages and
  `pip-audit --skip-editable` reports no known vulnerabilities after raising
  the Pillow floor to 13.0. The checker now recognizes standard license files
  when wheel metadata omits license fields. Python 3.14 free-threaded macOS
  remains an intentionally unsupported audit environment for the Ray extra;
  the hosted Linux CI job remains the cross-platform authority.
- `b59459d` adds a deterministic public-API signature snapshot and executes
  it in CI and release preflight; the gate passed on Python 3.11 and 3.14.
  The existing format-1-to-format-2 migration test remains part of the unit
  contract suite.
- `1ca350a` and `a84d1dc` add and export `WarehouseConnector`. Its focused
  connector and incremental-ingestion slice passed 25 tests, with Ruff and
  mypy clean. The connector supports caller-owned warehouse/lakehouse query
  engines through bounded callback reads; broker delivery and distributed
  exactly-once semantics remain explicitly outside the claim.
- `5167935` adds and exports `EventStreamSource`. Its focused incremental
  suite passed 10 tests, with bounded callback polling, retry exhaustion,
  checkpoint binding, over-limit rejection, and no-acknowledgement semantics
  covered. This establishes transport boundaries; it does not claim broker
  SDK acknowledgement, consumer-group coordination, or distributed
  exactly-once delivery.
- The Astro site build and accessibility smoke passed all 15 routes after the
  warehouse documentation was added. Public GitHub Pages verification and
  independent review remain external release gates.

Hosted-state observation on 2026-10-21:

- The repository is private and currently has no `main` branch; its default
  branch is `feat/v0.1-foundation`. The Pages workflow previously triggered
  deployment only from `main`, so the requested public URL could never be
  produced from the current repository state. Commit `405ca1b` changes the Pages
  trigger and deploy/verify guards to include the actual default branch while
  retaining the `main` path for a later branch migration.
- GitHub Actions run `39506512353` for commit `6b34516` reported all 17 CI
  jobs as failed within seconds and exposed no job logs. Local preflight
  success therefore cannot be promoted to hosted CI evidence; the run needs
  an owner-side Actions/billing/runner diagnosis before release approval.
- After `405ca1b` enabled the correct branch trigger, Pages runs `39507068972`
  and `39507078394` reached the `build` job but failed before producing logs;
  `deploy` and `verify` were skipped. This confirms the workflow trigger fix
  while leaving the hosted runner/environment diagnosis open.
- The public URL `https://sachncs.github.io/super-charge-ml/` still returns
  HTTP 404. This remains an external deployment gate, not evidence that the
  static site build is broken locally.

Local preflight evidence captured on 2026-10-21:

- The release preflight launched from `6bd4341` completed successfully through
  the benchmark, package, artifact, and marketing-site gates. The bounded
  contract suite passed with 1,916 tests and 75 skips, total coverage was
  82.83% with 86% in the declared core scope, and the mutation gate killed
  1,341 of 1,863 mutants (71%, against the 70% minimum).
- The four offline, five-seed benchmark datasets completed with zero failed
  repetitions. Breast cancer, wine, and digits were statistically tied with
  their independent references; diabetes progression was ahead. Runtime,
  peak-memory, failure-rate, and software identity were recorded for each
  dataset.
- Wheel and source-distribution checks, reproducible artifact hashes, and
  network-isolated wheel and source-distribution installations passed. The
  Astro build generated all 15 routes and npm audit reported zero
  vulnerabilities.
- Commit `e7edf2e` subsequently replaces the deprecated scikit-learn sparse
  logistic parameter with the equivalent `l1_ratio=1.0` API. Its focused
  warning-strict test slice passed 32 tests, with Ruff and mypy clean. This
  focused change still needs a fresh clean-head full preflight before it can
  be used as independent release evidence.
- The preflight remains local evidence only. It does not close the hosted
  Linux, public Pages, external secret custody, external-fault, or
  independent-review gates listed above.
- `8845e2d` adds an explicit intersectional fairness report across every
  declared sensitive column. It reuses the deterministic numeric bands and
  minimum-cell guard, keeps the existing per-column reports, and is covered
  by classification and regression tests (28 diagnostics/release tests plus
  7 focused fairness tests under `-W error::ResourceWarning`). This closes a
  local reporting gap in Phase 3; it does not establish causal fairness or
  statistical power for small groups.

Clean-head preflight evidence at `f691997` on 2026-10-21:

- Static checks passed: Ruff, format, mypy, Bandit, pip-audit, and public API
  compatibility. Bandit emitted existing comment-parser warnings only; no
  finding failed the gate.
- The bounded contract suite passed with 1,918 tests and 75 skips. Total
  coverage was 82.84%; the declared core scope reached 86%. The run emitted
  two existing SQLite `ResourceWarning` instances from the lifecycle extras
  path; they remain hygiene debt and are not suppressed.
- The mutation gate killed 1,340 of 1,863 mutants (71%, against the 70%
  minimum), with one timeout and surviving mutants retained for triage.
- Five-seed offline benchmarks passed for breast cancer, wine, digits, and
  diabetes progression with zero failed repetitions and zero regressions.
  Runtime, peak memory, failure rate, and independent-reference comparisons
  were recorded in the benchmark evidence artifact.
- Wheel and source-distribution checks passed, both artifacts were
  reproducible, and network-isolated wheel and source-distribution installs
  passed. The Astro site built all 15 routes and npm audit reported zero
  vulnerabilities.
- This is clean local evidence for `f691997`, not hosted or independent
  release proof. GitHub Pages remains HTTP 404, hosted Actions remain
  unavailable before job logs, and external key custody, external-fault
  execution, and independent review/signature remain release blockers.
- `5601af9` makes small-cell fairness limitations explicit. Per-column and
  intersectional reports now retain excluded group sizes and a machine-
  readable power warning instead of silently presenting only the surviving
  groups. The focused fairness and release-evaluation slice passes 20 tests
  under `-W error::ResourceWarning`; this is a reporting safeguard, not a
  substitute for a domain-specific power analysis.
- `f14f1a6` adds the same fairness interpretation to the standalone Astro
  documentation, including the evidence shape and the distinction between
  disparity flags, statistical power, and causal claims. The site build and
  accessibility smoke checks pass all 15 routes.

Clean-head release preflight evidence at `f94a74a` on 2026-10-21:

- Static checks passed: Ruff, format, mypy, Bandit, pip-audit, and public API
  compatibility. Bandit emitted the existing comment-parser and documented
  `B608` nosec warnings only; no finding failed the gate.
- The bounded contract suite passed with 1,918 tests and 75 skips. Total
  coverage was 82.85%; the declared core scope reached 86%. Two existing
  SQLite `ResourceWarning` instances remain in the lifecycle extras path and
  are recorded as hygiene debt rather than suppressed.
- The mutation gate killed 1,340 of 1,863 mutants (71%, against the 70%
  minimum), with one timeout and surviving mutants retained for triage.
- Five-seed offline benchmarks passed for breast cancer, wine, digits, and
  diabetes progression with zero failed repetitions and zero regressions.
  The run recorded 91/66/633/60 seconds, peak memory of
  0.3390/0.3469/0.4368/0.4405 GB, and zero failure rate respectively; the
  independent-reference verdicts were tie, tie, tie, and ahead.
- Wheel and source-distribution checks passed, both artifacts were
  reproducible, and network-isolated wheel and source-distribution installs
  passed. The Astro site built all 15 routes and npm audit reported zero
  vulnerabilities.
- This is clean local evidence for `f94a74a`, not hosted or independent
  release proof. GitHub Pages remains HTTP 404, hosted Actions remain
  unavailable before job logs, and external key custody, external-fault
  execution, and independent review/signature remain release blockers.

Reliability hygiene evidence at `1ebec45` on 2026-10-21:

- `RunContext.close()` now owns cleanup for both executing and preview
  contexts. `MLProject.plan()` and `MLProject.recommendations()` close the
  prepared context in a `finally` block, so the sealed-test vault subprocess
  cannot survive until garbage collection; executing runs use the same context
  boundary in their finalizer.
- The complete unit suite passed warning-strict with 1,826 tests passed and 73
  optional tests skipped: `uv run pytest -q tests/unit
  -W error::ResourceWarning`. The targeted curriculum preview and missingness
  curation cases also passed under the same policy.
- This closes the previously observed local subprocess-resource warning. It
  does not replace a fresh full release preflight from this new head.

Incremental acknowledgement evidence at `61d921f` on 2026-10-21:

- `EventAcknowledgement`, `EventConsumerResult`, and `EventConsumer` add a
  typed consumer-group boundary around the existing event source and ledger.
  The sink commit is durable before an acknowledgement is staged; a failed
  broker callback leaves an atomic `.ack` sidecar, and the next consume cycle
  retries that acknowledgement before polling and without replaying the sink.
- Twelve incremental-reader tests pass under
  `-W error::ResourceWarning`, including commit-before-ack ordering, writer
  failure, acknowledgement failure, durable retry, source mismatch, late data,
  duplicate events, and bounded polling. Ruff and mypy are clean.
- This closes the local acknowledgement/retry contract. It does not provide
  broker SDK leases, partition assignment, broker transactions, or distributed
  exactly-once evidence; those remain transport-specific and externally
  reviewable release work.

Clean-head release preflight evidence at `32f9fd2` on 2026-10-21:

- Static checks passed: Ruff, format, mypy, Bandit, pip-audit, and public API
  compatibility. Bandit retained the existing comment-parser and documented
  `B608` nosec warnings only; no finding failed the gate.
- The bounded contract suite passed with 1,921 tests and 75 skips. Total
  coverage was 82.82%; the declared core scope reached 86%. Two SQLite
  `ResourceWarning` instances remain in the lifecycle-extras path and are
  recorded as hygiene debt rather than suppressed.
- The critical mutation gate killed 1,495 of 2,049 mutants, for a 72% score
  against the 70% minimum. Surviving mutants and one timeout remain available
  for triage; the threshold passed.
- Five-seed offline benchmarks passed for breast cancer, wine, digits, and
  diabetes progression with zero failed repetitions and zero regressions.
  Runtime was 91/66/632/60 seconds, peak memory was 0.3557/0.3756/0.4251/
  0.4290 GB, and failure rate was zero for each dataset. The independent-
  reference verdicts were tie, tie, tie, and ahead.
- Wheel and source-distribution checks passed, both artifacts were reproducible
  (`wheel` SHA-256
  `d0942ff38efec1fcd95627cfe4f34e4fdd6b80e7d6d955bfa45bdb02456ba175`, `sdist`
  SHA-256
  `8860e9b43aca4f01108f6add0c19ab26d2f85cc3f28eaf0c4156b077482ddb48`), and
  network-isolated wheel and source-distribution installs passed.
- The Astro site built all 15 routes and npm audit reported zero
  vulnerabilities. This is clean local evidence for `32f9fd2`, not hosted or
  independent release proof.
- GitHub Pages remains HTTP 404, hosted Actions remain unavailable before job
  logs, and external key custody, external-fault execution, and independent
  review/signature remain release blockers. The two SQLite warnings, 75
  unexplained-by-default skips, and surviving/timeout mutants also remain
  engineering follow-up rather than production-readiness evidence.

Warning-strict bounded-suite evidence at `4c0d079` on 2026-10-21:

- The same bounded release suite passed with `-W error::ResourceWarning`:
  1,921 tests passed and 75 tests were skipped in 486.51 seconds. No
  `ResourceWarning` was promoted to a failure.
- This validates that the earlier two SQLite warnings are nondeterministic
  full-run hygiene observations rather than a deterministic warning-strict
  failure. They remain release debt until the source is identified and the
  default preflight is clean; warning promotion is not being used to hide
  them.

Standalone diagnostic documentation evidence at `1904cc7` on 2026-10-21:

- The Astro evidence guide now documents the bounded `error_analysis` and
  `pairwise_interaction` diagnostics, including their output fields, row and
  feature budgets, truncation interpretation, privacy boundary, and
  associational—not causal—limitations.
- The shared documentation layout adds responsive diagnostic evidence cards;
  `npm run build && npm run check:a11y` passes all 15 static routes.
- This closes the site explanation gap for the latest evaluation evidence. It
  does not make either diagnostic a root-cause, fairness, or causal claim, and
  it does not close the independent-review or hosted deployment gates.

Clean-head local release preflight at `d1c35f4` on 2026-10-21:

- Static checks passed: Ruff, format, mypy, Bandit, pip-audit, and public API
  compatibility. Bandit retained the existing comment-parser and documented
  `B608` warnings only; no finding failed the gate.
- The bounded contract suite passed with 1,933 tests and 75 optional skips.
  Total coverage was 82.97%; the declared core scope reached 86%. The run
  emitted the two existing SQLite `ResourceWarning` summaries in the
  lifecycle-extras path; they remain hygiene debt rather than suppressed
  release evidence.
- The critical mutation gate killed 1,495 of 2,049 mutants, for a 72% score
  against the configured 70% minimum. One timeout and surviving mutants remain
  available for triage.
- Five-seed offline benchmarks passed for breast cancer, wine, digits, and
  diabetes progression with zero failed repetitions and zero regressions.
  Runtime was 98/70/632/63 seconds, peak memory was 0.3734/0.3764/0.4431/
  0.4469 GB, and failure rate was zero for each dataset. Independent-reference
  verdicts were tie, tie, tie, and ahead.
- Wheel and source-distribution checks passed, reproducible hashes were
  `wheel: 1a4ea2a9edc921ee8e1a186d78c7816436debb5a980045df5fa1d57f1c72be38`
  and `sdist: 12ece6887ca3a8574d342523ef6f3744c4309f1b01afff1f1dd405669c88134a`,
  and network-isolated wheel and source-distribution installations passed.
- The Astro marketing site built all 15 routes and `npm audit` reported zero
  vulnerabilities. This is clean local evidence for `d1c35f4`, not hosted or
  independent release proof. GitHub Pages, hosted Linux, external-fault,
  external key custody, and independent-review gates remain open.

Standalone site smoke evidence at the post-preflight working head on
2026-10-21:

- `npm run check:a11y` passed for all 15 generated HTML files.
- `npm run test:e2e` passed both Playwright checks: every public route rendered
  under the `/super-charge-ml/` base path with no `.md` links, and the
  documentation navigation exposed a keyboard focus path.
- This verifies the local site acceptance surface. It does not verify the
  unavailable public GitHub Pages URL or hosted deployment workflow.

Generated-site link-gate evidence at `9dfdfd6` on 2026-10-21:

- `website/scripts/check-links.mjs` now resolves same-site links against the
  `/super-charge-ml/` base path, checks route and asset targets, and validates
  same-page anchors without making external network calls. The gate is wired
  into both the ordinary site workflow and the Pages build workflow.
- `npm run build && npm run check:links` passed for all 15 generated pages and
  310 same-site links. The same run also passed `npm run check:a11y` and both
  Playwright smoke tests.
- This turns the internal-link acceptance criterion into an executable local
  and hosted-build check; it does not verify the currently unavailable public
  Pages deployment.

Standalone data and operations documentation evidence at `faa9c2a` on
2026-10-21:

- The data guide now explains the raw-material hand-off for documents, images,
  audio/video, graphs, and geospatial data, including what curation evidence
  must survive into representation preparation and what each baseline does not
  claim.
- The operations guide now maps data, model, infrastructure, and policy
  failures to distinct operator actions, preserving the rule that only a
  verified digest-bound state may move forward.
- `npm run build`, `npm run check:links`, `npm run check:a11y`, and the two
  Playwright smoke tests passed after the content change. This improves the
  standalone developer surface but does not close hosted deployment or
  independent-review evidence.

Current-head clean-clone rehearsal at `494a297` on 2026-10-21:

- A fresh detached clone of the exact pushed commit was clean before test
  execution. All declared Python extras installed from the lockfile, and the
  acceptance slice passed with 1,906 tests passed and 73 optional skips in
  8 minutes 23 seconds.
- The package entry point executed successfully from the clone with
  `scml --help`.
- This is reproducible local clean-clone evidence, not independent reviewer
  evidence or hosted Linux proof.

SQLite warning cleanup at `b8e5653` on 2026-10-21:

- Tracemalloc identified the two nondeterministic `ResourceWarning` instances
  in connector test fixtures using `with sqlite3.connect(...)`; that context
  commits or rolls back but does not close the connection.
- The fixtures now compose `contextlib.closing` with the transaction context.
  The combined connector and lifecycle-extras regression slice passes 29 tests
  under `-W error::ResourceWarning`, with Ruff and formatting clean.
- This removes a test-owned resource leak. External transport, container, and
  kernel fault evidence remain separate reliability requirements.

Warning-strict bounded release suite at `a6dfc29` on 2026-10-21:

- The complete bounded contract suite passed with `-W error::ResourceWarning`:
  1,933 tests passed and 75 optional tests were skipped in 8 minutes 8
  seconds.

Standalone modality lifecycle guide evidence at `f0b13d7` on 2026-10-21:

- The Astro site now publishes a dedicated `/docs/modalities/` route. It
  explains the complete 15-stage modality contract, modality-specific raw
  material and curation requirements for documents, images, audio, video,
  graphs, and geospatial data, typed manifest hand-off, data/model/release
  evidence boundaries, contributor checks, and the distinction between the
  supported tabular beta path and experimental modality foundations.
- The route is present in the standalone navigation, onboarding cards, browser
  route inventory, and both local/Pages generated-route smoke loops. It does
  not redirect readers to repository Markdown files.
- The visual review passed at desktop and 390px mobile widths: hierarchy,
  navigation wrapping, lifecycle diagrams, code block overflow, and the
  experimental/support callout were visibly usable. The preview was inspected
  at `http://127.0.0.1:4322/super-charge-ml/docs/modalities/`.
- `npm run build`, `npm run check:links`, `npm run check:a11y`, and
  `npm run test:e2e` passed: 16 pages generated, 346 same-site links passed,
  all 16 HTML files passed the accessibility smoke gate, and both Playwright
  tests passed. This is local artifact evidence; it does not prove the
  currently unavailable public Pages URL or independent release review.

Post-change full regression evidence at `ddc73e1` on 2026-10-21:

- The complete local pytest suite passed with 2,274 tests passed and 88
  optional-dependency skips in 28 minutes 52 seconds. Total coverage was
  89%; the new `scml/artifacts/explanation_report.py` renderer reached 100%,
  and `scml/core/stages.py` remained at 96%.
- The focused explanation, product-spec, resume, and lineage slice passed 55
  tests before the full suite. Static checks, formatting, mypy, and diff
  hygiene passed for every changed Python file.
- The new per-explanation HTML artifacts are written atomically beside their
  JSON records, included in model-card evidence and run lineage, and withheld
  when the explanation privacy decision is `denied`. The full suite therefore
  verifies the artifact contract through the package stage, not only through
  an isolated renderer test.
- This is strong local evidence for the current commit. The hosted Linux
  matrix, external fault execution, independent review, and external signing
  key custody remain separate release requirements.

Explanation report discoverability evidence at `190a0a4` on 2026-10-21:

- The generated `report.html` now links to every explanation HTML artifact
  recorded in `evaluation.json`, so a reviewer can move from the run summary
  to the detailed evidence without guessing a path. Names are escaped and
  links remain relative to the run directory.
- The problem-spec integration test asserts the link for
  `explanations/permutation_importance.html`; the focused five-test slice,
  Ruff, formatting, and mypy all passed. This closes the navigation gap for
  the local HTML evidence package but does not change the raw-modality or
  independent-review boundaries.

Raw image explanation evidence at `23d4f76` on 2026-10-21:

- `ImageAdapter` now applies bounded zero-valued masks to validated source
  pixels before the declared 16x16 RGB resize, and persists the resulting
  source-space region score deltas beside the existing transformed-feature
  occlusion evidence. The record includes source dimensions, the masking
  baseline, feature-space boundary, and explicit limits on semantic
  localization and saliency claims.
- The focused modality lifecycle and explanation-quality slice passed 32
  tests; the complete modality-focused set passed 101 tests. Ruff, formatting,
  mypy, and diff checks passed. This closes the bounded raw-image occlusion
  gap; raw audio/video/document/graph/geospatial explanations, semantic
  saliency/localization, causal claims, and production-scale repeated
  perturbation suites remain incomplete.

Raw audio explanation evidence at `7b9bbe2` on 2026-10-21:

- `AudioAdapter` now masks source PCM frames with the format-correct silence
  value before rebuilding a bounded WAV sample and scoring the declared
  feature pipeline. The evidence retains frame and time boundaries, sample
  rate, replacement baseline, and the distinction between raw waveform
  occlusion and aggregate feature-space windows.
- The focused modality lifecycle and explanation-quality slice passed 32
  tests; the complete modality-focused set passed 101 tests. Ruff, formatting,
  mypy, and diff checks passed. The result does not claim transcript,
  speaker, semantic event, saliency, or causal attribution.

Raw video explanation evidence at `1eaafe5` on 2026-10-21:

- When a caller supplies the explicit bounded frame decoder, `VideoAdapter`
  now replaces each sampled decoded frame with zeros before rebuilding the
  frame-statistics representation and scoring it. The evidence preserves
  frame indices, time positions, decoder ownership, and the boundary between
  decoded-pixel occlusion and motion or semantic localization claims.
- The focused modality lifecycle and explanation-quality slice passed 32
  tests; the complete modality-focused set passed 101 tests. Ruff, formatting,
  mypy, and diff checks passed. Metadata-only video remains explicitly
  unavailable for raw-frame evidence until a decoder is supplied.

Raw document explanation evidence at `ba15305` on 2026-10-21:

- `DocumentAdapter` now masks each declared page or section span in source
  character space, preserves the span offsets, and records the score delta
  plus masked-content digest without copying document text. A punctuation
  sentinel handles a full-document span while preserving the non-empty
  document contract.
- The focused modality lifecycle and explanation-quality slice passed 32
  tests; the complete modality-focused set passed 101 tests. Ruff, formatting,
  mypy, and diff checks passed. The evidence remains bounded lexical
  perturbation; it does not claim OCR confidence, citation grounding,
  semantic attribution, or causal effect.

Raw geospatial explanation evidence at `9e6117a` on 2026-10-21:

- `GeospatialAdapter` now replaces source geometry x and y coordinates with
  the feature centroid in the declared CRS before feature preparation. The
  evidence preserves geometry structure, CRS identity, masked geometry hashes,
  and score deltas without claiming map attribution or spatial causality.
- The modality lifecycle and explanation-quality slice passed 32 tests; the
  complete modality-focused set passed 74 tests after adding nested geometry
  masking coverage. Ruff, formatting, mypy, and diff checks passed. Raster
  remains metadata-only without a caller-owned decoder, and graph topology
  remains contextual because the baseline does not consume edges.

Raw graph explanation evidence at `8248367` on 2026-10-21:

- `GraphAdapter` now masks each declared source node feature with a bounded
  zero baseline before scoring the node, while retaining one-hop edge context
  and explicitly recording that topology is not a model input. The evidence
  records feature names, indexes, score deltas, and the 64-feature bound.
- The modality lifecycle and explanation-quality slice passed 74 tests under
  `-W error::ResourceWarning`, with Ruff, formatting, mypy, and diff checks
  clean. This closes raw node-feature evidence; it does not claim topology
  attribution, GNN support, subgraph causality, or semantic graph saliency.

Repeated explanation evidence at `8957c65` on 2026-10-21:

- `ModelExplainabilityStage` now evaluates local contribution rankings under
  both directions of a declared small feature-space perturbation for every
  held-out sample up to `MAX_REPEATED_EXPLANATION_SAMPLES = 128`. The minimum
  stability across evaluated samples is the release-gated score, and the
  evidence records the total and evaluated counts plus the resource bound.
- The modality lifecycle slice passed 74 tests under
  `-W error::ResourceWarning`, with Ruff, formatting, mypy, and diff checks
  clean. This closes the single-sample quality gate; it does not claim
  unbounded production-scale repetition, raw-modality saliency, or causal
  explanation validity.

Current-head complete release preflight at `c9ec4c9` on 2026-10-21:

- Static, formatting, mypy, Bandit, dependency, and public-API checks passed.
- The bounded release suite passed 1,941 tests with 75 explicit optional
  dependency skips. Declared core coverage was 86% and whole-package coverage
  was 83.09%.
- The critical mutation gate killed 1,494 of 2,049 mutants (72%) against the
  70% minimum.
- All four five-seed offline benchmarks passed with zero regressions and zero
  failures. Runtime, peak-memory, completed-trial, and failure-rate evidence
  was present for every dataset.
- Reproducible wheel and source-distribution builds, network-isolated installs,
  and package verification passed. The Astro site built all 16 static pages
  and `npm audit --omit=dev --audit-level=high` reported zero vulnerabilities.
- This is current-host local evidence. It does not replace hosted Linux
  execution, public GitHub Pages verification, external fault rehearsal,
  secret-manager custody, or independent release review/signature.

Strict full-suite evidence at `c22d294` on 2026-10-21:

- `uv run --python 3.14 pytest -q tests -W error::ResourceWarning` passed
  2,276 tests with 88 optional-dependency skips in 29 minutes. No warning was
  promoted to a failure, and the worktree remained unchanged.
- This is stronger current-head regression evidence than the bounded preflight
  suite, but the skips still represent uninstalled optional capability paths;
  they are not evidence for those optional integrations.

Standalone site verification at `2eb37a9` on 2026-10-21:

- `npm run check:links` passed for all 16 generated HTML pages and 346
  internal links; no repository Markdown redirect was required.
- `npm run check:a11y` passed for all 16 pages, and `npm run test:e2e` passed
  both the public-route and keyboard-navigation browser checks.
- The local build and route smoke remain green, but
  `https://sachncs.github.io/super-charge-ml/` still returns HTTP 404 because
  hosted repository visibility and Pages state are unresolved.

Current-head test-stability evidence at `f955873` on 2026-10-21:

- `scripts/check_test_stability.py --runs 3` passed all 22 selected API,
  performance, and stability cases on each of three attempts.
- The JSON policy result reported no inventory changes, expired quarantines,
  unquarantined failures, or quarantined failures. This is local repeatability
  evidence; the scheduled hosted Linux job remains the cross-platform authority.

Exact-SHA clean-clone rehearsal at `0d5affd` on 2026-10-21:

- A fresh local clone checked out the exact pushed SHA with a clean detached
  worktree, installed the declared boosting, optimization, and multimodal
  extras, and passed the acceptance slice with 1,914 tests and 73 explicit
  optional-dependency skips in 8 minutes 30 seconds.
- `uv run --directory <clone> scml --help` exited successfully, proving the
  packaged entry point resolves from the clean clone. This remains local
  reproduction evidence, not hosted Linux or independent reviewer evidence.

Hosted release-gate observation at `e287642` on 2026-10-21:

- Authenticated repository metadata reports `sachncs/magicml` is private, its
  default branch is `feat/v0.1-foundation`, and no Pages site is configured.
- The workflow-based Pages creation request was rejected with HTTP 422 because
  the current GitHub plan does not support Pages for this private repository;
  the public URL therefore remains unverified and returns HTTP 404.
- The CI run for the pushed SHA created every job in a terminal failure before
  any runner step. GitHub annotations identify the cause as failed recent
  account payments or an insufficient spending limit. This is an account
  billing/plan blocker, not a source or workflow test result.

Standalone metadata evidence at `61059af` on 2026-10-21:

- Package metadata now points `Documentation` to
  `https://sachncs.github.io/super-charge-ml/` rather than the repository's
  Markdown `docs/` tree. A public-API contract test prevents that onboarding
  link from regressing, and the built wheel contains the same URL.
- The metadata test, Ruff, mypy, and wheel build passed. The URL is the
  requested production destination, but remains externally unavailable until
  the repository Pages and visibility constraints are resolved.

- No resource warning was promoted to a failure after the connector fixture
  cleanup. This is current-head local evidence for the warning gate; it does
  not replace hosted Linux execution or independent review.

Optional-skip inventory at `ac1d68d` on 2026-10-21:

- The bounded suite was rerun with `-q --tb=no -rs` and passed with 1,933
  tests and 75 skips in 8 minutes 5 seconds.
- Every skip had an explicit optional-dependency reason: CatBoost, Torch or
  Torch/Keras, sentence-transformers, ONNX tooling, openpyxl, or imbalanced-
  learn. No skip was caused by an unlisted failure, an error, or an unknown
  runtime condition.
- These are environment capability boundaries, not silently accepted failures;
  the corresponding optional tests remain outside the core release claim until
  their extras are installed and audited in the hosted matrix.

Caller-owned object decoder evidence at `9136e8e` on 2026-10-21:

- `ObjectStoreConnector` now accepts a caller-owned decoder for bounded object
  payloads outside the built-in CSV, JSONL, and Parquet formats. A required
  decoder name/version is included in the source identity, so checkpoints and
  lineage cannot silently mix decoding semantics for the same object URI.
- Decoder exceptions and invalid return types become structured connector
  failures. The focused connector suite passed 21 tests under
  `-W error::ResourceWarning`; Ruff, formatting, and mypy passed.
- This expands ingestion to arbitrary caller-defined bounded DataFrame
  representations without claiming that the library owns binary format
  decoding, streaming scale, or broker delivery semantics.

Python naming-convention evidence at `d88a9a9`, revalidated on 2026-10-21:

- `tests/unit/test_conventions.py` parses every package module, test module,
  and Python filename and fails on underscore-prefixed identifiers used as
  semi-private names. Python dunder protocol names remain allowed; even a
  bare `_` is rejected so the convention cannot drift between modules.
- The convention test passed under `-W error::ResourceWarning`; Ruff,
  formatting, and mypy passed. Ruff's configured Google docstring convention
  remains the source-level style gate for package code.

Current-head coding-standard verification at `d007f7e` on 2026-10-22:

- The repository-wide naming guard passed `851` tests under
  `-W error::ResourceWarning`; no underscore-prefixed semi-private identifier
  or non-dunder private filename was found in package or test Python files.
- Ruff passed across `scml`, `tests`, `scripts`, and `benchmarks`; the format
  check reported all 437 Python files already formatted. Mypy passed for 239
  source files. These checks are evidence for the current head, not a waiver
  for future additions.

Full strict regression at `ae6b1f2` on 2026-10-21:

- `uv run --python 3.14 pytest -q tests -W error::ResourceWarning` passed
  `2,247` tests with `124` explicit optional-dependency skips in 27 minutes
  52 seconds. No failure, error, or resource warning was promoted to a
  failure.
- This run includes the generic object-decoder contract, the package naming
  convention guard, and the existing data, modality, reliability, governance,
  and site-adjacent Python tests. The optional skips remain environment
  capability boundaries and do not prove the corresponding integrations.

Strict warning policy enforcement at `b65d931` on 2026-10-21:

- The local release preflight, hosted CI matrix, scheduled full suite, and
  exact-SHA clean-clone acceptance slice now all pass
  `-W error::ResourceWarning` to pytest. Resource leaks therefore cannot be
  downgraded when the same release gates run in another environment.
- `actionlint .github/workflows/*.yml`, YAML parsing, `bash -n
  scripts/release_preflight.sh`, and `git diff --check` passed before the
  workflow change was committed.

Hosted blocker re-confirmed at `ef1ae56` on 2026-10-21:

- The Pages workflow for the pushed head was created but its build job failed
  before a runner step (`steps: []`, no runner assigned). GitHub's check-run
  annotation states: “The job was not started because recent account payments
  have failed or your spending limit needs to be increased.” Deploy and public
  route verification were skipped.
- This is the same account billing/plan blocker previously observed. No source
  or workflow failure was observed, and repository visibility was not changed.

Strict stability-harness evidence at `34c0f26` on 2026-10-21:

- `check_test_stability.py` now injects `-W error::ResourceWarning` into every
  repeated pytest attempt instead of inheriting a weaker ambient warning
  policy.
- The three-run API, performance, and stability policy check passed all 26
  collected test cases on each attempt with no inventory changes, skips,
  quarantined failures, or unquarantined failures. Ruff, formatting, and mypy
  passed for the changed files.

Resampling contract correction at `ca6ab97` on 2026-10-21:

- `Resampler` now owns the general polymorphic `resample()` contract, while
  `IndexResampler` contains the shared row-index implementation used by
  random and jitter samplers. Synthetic generator sampling therefore provides
  a real `resample()` implementation without a fake index method that raises
  at runtime.
- The focused sampling and component-contract suite passed 40 tests with one
  optional skip under `-W error::ResourceWarning`; Ruff, formatting, mypy, and
  `git diff --check` passed. The source and test change was pushed as one
  atomic commit.

Spatial-dependence curation evidence at `992ee1c` on 2026-10-21:

- Geospatial curation now records deterministic k-nearest-neighbor topology
  and global Moran's I evidence for numeric labels, including a bounded
  permutation p-value, seed, feature count, and edge count. The topology and
  statistic are exposed as typed JSON-native contracts so the result can be
  independently recomputed from the feature order and CRS coordinates.
- Non-numeric labels, constant values, insufficient features, and missing CRS
  produce explicit non-applicable, non-estimable, or blocked evidence rather
  than an invented statistic. This detects spatial dependence; it does not
  claim causality or replace a domain-specific spatial model.
- The combined geospatial, modality-lifecycle, naming-convention, and
  component-contract slice passed 920 tests with 5 optional skips under
  `-W error::ResourceWarning`; Ruff, formatting, mypy, and diff checks passed.

Audio transcript provenance evidence at `4aac37a` on 2026-10-21:

- `AudioTranscript` and `AudioTranscriptSegment` now retain the transcript
  source, provider and provider version, language, confidence, ordered
  time-local segments, and JSON-native provenance. `AudioSample` rejects
  segments outside the decoded audio duration, so transcript claims cannot
  silently refer to nonexistent audio.
- Audio curation reports transcript coverage, providers, languages, segment
  counts, confidence bounds, and a visible non-blocking finding when a
  transcript has no confidence evidence. This records transcript provenance;
  it does not claim that the library owns speech recognition or codec
  decoding.
- The focused media, modality-lifecycle, and naming-convention suite passed
  896 tests with 4 optional skips under `-W error::ResourceWarning`; Ruff,
  formatting, mypy, and diff checks passed.

Video synchronization evidence at `7e52e92` on 2026-10-21:

- `VideoMetadata` now carries optional measured audio duration, synchronization
  offset, and tolerance. Its typed status distinguishes no audio, unknown
  measurement, within-tolerance timing, and measured drift. Existing
  metadata-only clips remain explicit rather than inheriting an assumption.
- Video curation records synchronization states, warns when an audio track
  lacks timing evidence, and blocks measured out-of-tolerance drift. This
  validates an input-quality boundary; it does not decode codecs or claim
  temporal event localization.
- The focused media, modality-lifecycle, and naming-convention suite passed
  898 tests with 4 optional skips under `-W error::ResourceWarning`; Ruff,
  formatting, mypy, and diff checks passed.

Document layout provenance evidence at `b3aca01` on 2026-10-21:

- `DocumentLayout` and `DocumentElement` now preserve caller-owned OCR,
  layout, heading, image, and table extraction evidence with provider/version,
  page number, source offsets, confidence, and table dimensions. A sample
  rejects any element whose offsets exceed decoded document content.
- Document curation reports layout coverage, providers, element and table
  counts, OCR confidence bounds, and an explicit warning when OCR confidence
  is absent. This makes extraction provenance reviewable without claiming
  that the core package owns OCR, PDF layout parsing, or table recognition.
- The document, modality-lifecycle, and naming-convention suite passed 880
  tests with 4 optional skips under `-W error::ResourceWarning`; Ruff,
  formatting, mypy, and diff checks passed.

Parser-fuzzing evidence at `7887e12` on 2026-10-21:

- The Hypothesis property tier now feeds bounded arbitrary byte payloads to
  `DocumentReader`, accepting only a valid sample or a structured document
  validation error. It also uses typed NumPy index arrays for the existing
  split invariants, so the complete property module passes its static type
  gate rather than carrying pandas-index typing debt.
- The property suite passed 10 tests under `-W error::ResourceWarning`; Ruff,
  formatting, mypy, and diff checks passed.

Temporal graph evidence at `a302727` on 2026-10-21:

- Graph edges now accept an explicit timezone-aware event time, normalize it to
  UTC, preserve it in graph identity and duplicate detection, and ingest it
  from CSV or JSONL through an explicit time column. A temporal split assigns
  edges at the cutoff to training and later edges to future evaluation; mixed
  or incomplete timestamp state fails closed instead of inventing ordering.
- Graph curation exposes absent, mixed, or complete temporal state with bounds,
  and emits a deterministic quality finding for mixed timestamp coverage. This
  establishes temporal graph data evidence; it does not claim a temporal graph
  model, dynamic topology semantics, or production stream watermark guarantees.
- The graph, modality-lifecycle, and naming-convention suite passed 879 tests
  with 4 optional skips under `-W error::ResourceWarning`; Ruff, formatting,
  mypy, and `git diff --check` passed. The source and test change was pushed as
  one atomic commit.

Temporal geospatial evidence at `dd1e201` on 2026-10-21:

- `GeoFeature` now preserves a timezone-aware observation time, normalizes it
  to UTC, and binds it into dataset identity. `GeospatialDataset.split_temporal`
  assigns features at the cutoff to training and later features to future
  evaluation; missing observation times fail closed rather than inventing
  temporal order.
- Geospatial curation exposes absent, mixed, or complete temporal coverage with
  time bounds and a deterministic warning for mixed state. This closes the
  temporal data-integrity boundary; it does not claim a domain-specific
  spatiotemporal model, reprojection policy, or streaming watermark guarantee.
- The media, modality-lifecycle, and naming-convention suite passed 900 tests
  with 4 optional skips under `-W error::ResourceWarning`; Ruff, formatting,
  mypy, and `git diff --check` passed. The source and test change was pushed as
  one atomic commit.

Raster nodata evidence at `5f5eb99` on 2026-10-21:

- `RasterSample` now validates caller-owned nodata sentinel and measured
  nodata-fraction metadata, exposes an explicit unknown/declared/measured
  status, and binds those values into the raster dataset identity. `RasterAdapter`
  persists the nodata profile and emits a visible warning when no nodata
  evidence is supplied.
- This closes the raster data-understanding and curation boundary without
  pretending that encoded bytes have been decoded: pixel masking, window
  sampling, reprojection, and raster-specific semantics still require an
  explicit codec/decoder policy.
- The media, modality-lifecycle, and naming-convention suite passed 901 tests
  with 4 optional skips under `-W error::ResourceWarning`; Ruff, formatting,
  mypy, and `git diff --check` passed. The source and test change was pushed as
  one atomic commit.

Document sensitive-content evidence at `0ba2f38` on 2026-10-21:

- Document curation now detects bounded email, phone, and SSN patterns without
  copying matched values into evidence, reports only type counts and affected
  document IDs, and emits a blocking sensitive-content finding. The public
  redaction function replaces detected values with type-only placeholders for
  caller-owned downstream handling.
- This is a conservative detector, not a complete DLP or legal-compliance
  system; organizations must add domain-specific patterns, review false
  positives, and record consent, retention, and access policy in governance
  metadata before promotion.
- The document, modality-lifecycle, and naming-convention suite passed 882
  tests with 4 optional skips under `-W error::ResourceWarning`; Ruff, formatting,
  mypy, and `git diff --check` passed. The source and test change was pushed as
  one atomic commit.

Current-head strict regression evidence at `1836d61` on 2026-10-21:

- `uv run pytest -q tests -W error::ResourceWarning` passed with 2,263 tests
  passed and 124 optional skips in 30 minutes. No failure, error, or warning
  promoted to a failure by the strict warning policy.
- This is current local regression evidence for the pushed source tree. It
  does not replace the hosted Linux, coverage/mutation, clean-clone, public
  Pages, external-fault, or independent-review gates defined above.

Local release preflight evidence at `f713a5c` on 2026-10-21:

- `./scripts/release_preflight.sh` passed all local gates: Ruff, formatting,
  mypy over 239 source files, Bandit, pip-audit, and public API compatibility.
- The bounded release suite passed 1,964 tests with 75 optional skips; total
  coverage was 83.14% and the declared core scope was 86%, above the 85%
  threshold. Mutation testing killed 1,494 of 2,049 mutants (72%), above the
  70% release threshold.
- The five-seed offline benchmark regression completed for breast cancer, wine,
  digits, and diabetes progression with 0 regressions and 0% failure rate for
  every dataset. Results were ties for the first three and ahead for diabetes
  progression; this is benchmark evidence, not a claim of universal model
  superiority.
- Source and wheel artifacts passed `twine check`, reproduced byte-for-byte
  with wheel hash
  `40de2311132f88a63a490786e0f620605dc94faff775763d8c4642604fc506c7` and
  sdist hash
  `2e2dd5f370509adeb3b75b5d961364146451beb6f4dd31cb3e2bfb42cd7c8d1f`, and
  installed successfully in network-isolated environments.
- The production site dependency audit found 0 vulnerabilities and Astro
  built all 16 static pages. This is local release evidence; hosted Linux and
  Pages, external transport/resource-fault tests, secret custody, and
  independent reviewer evidence remain separate release gates.

Current-head release preflight evidence at `ded9b4f` on 2026-10-21:

- `./scripts/release_preflight.sh` passed all local gates after publishing
  `ModalityProject`: Ruff, formatting, mypy over 239 source files, Bandit,
  pip-audit, public API compatibility, and explicit JUnit skip-reason
  auditing. The bounded release suite passed 1,976 tests with 75 optional
  dependency skips; all 75 skips had explicit reasons. Total coverage was
  83.17%; the declared core scope was 86%.
- Mutation testing killed 1,495 of 2,049 mutants (72%), above the 70% release
  threshold. One mutation timed out and 553 survived; the threshold is met,
  and the surviving-mutant inventory remains follow-up evidence.
- The five-seed offline benchmark regression completed for breast cancer,
  wine, digits, and diabetes progression with zero regressions and zero failed
  repetitions. The verdicts were tie, tie, tie, and ahead; peak memory was
  0.3604, 0.3832, 0.4457, and 0.4570 GB respectively. This is benchmark
  evidence, not a claim of universal model superiority.
- Wheel and source-distribution checks passed, artifacts reproduced
  byte-for-byte with wheel hash
  `4816e6de8f4f31c4107537997d923d0fc4fd50039d12cacd02321c92ed148cd7` and
  sdist hash
  `3d3be35a6053132e24a7e35eef4fe7f17642ea364084fc675c861d6a9dfb21a0`, and
  both network-isolated installation paths passed.
- The production site dependency audit found 0 vulnerabilities and Astro
  built all 16 static pages. This remains local evidence; hosted Linux and
  Pages, external transport/resource-fault tests, secret custody, and
  independent reviewer evidence remain separate release gates.

Evidence-artifact durability at `386201d` on 2026-10-21:

- Modality lifecycle JSON reports now use the shared durable atomic writer:
  complete bytes are flushed and synced before replacement, the containing
  directory is synced after replacement, and temporary files are removed on
  failure. A previous report remains intact when replacement fails.
- The modality lifecycle, durability, and naming-convention slice passed 872
  tests under `-W error::ResourceWarning`; Ruff, formatting, mypy, and
  `git diff --check` passed. This closes the local partial-report-write gap;
  physical filesystem failure and external storage semantics remain outside
  the local evidence boundary.

Explicit text lifecycle evidence at `935b887` on 2026-10-21:

- The exhaustive 15-stage modality contract now includes a distinct
  `TextAdapter` fixture in addition to the document, image, audio, video,
  graph, geospatial, and raster adapters. This prevents the text declaration
  from inheriting coverage by assumption from the document subclass.
- The focused lifecycle and naming-convention slice passed 870 tests under
  `-W error::ResourceWarning`; Ruff, formatting, and `git diff --check`
  passed. Text remains experimental and does not inherit supported status from
  the test result alone.

Current-head release preflight evidence at `d1c3737` on 2026-10-21:

- `./scripts/release_preflight.sh` passed on the source head after the atomic
  report-write and explicit text-lifecycle changes. Static checks, formatting,
  mypy over 239 source files, Bandit, pip-audit, and public API compatibility
  all passed.
- The bounded release suite passed 1,967 tests with 75 optional dependency
  skips. Total coverage was 83.15%; the declared core scope was 86%. Mutation
  testing killed 1,494 of 2,049 mutants (72%), above the 70% threshold.
- The five-seed offline benchmark regression completed for breast cancer, wine,
  digits, and diabetes progression with zero regressions and zero failed
  repetitions. The verdicts were tie, tie, tie, and ahead; peak memory was
  0.3746, 0.3776, 0.4329, and 0.4449 GB respectively.
- Wheel and source-distribution checks passed, artifacts reproduced with wheel
  hash `f586cf667580b996b5d2abea7b9ed7959cf6f9d36d8a0872b9e384b92084da9f`
  and sdist hash
  `4741e22cc39a138f88d9d5b10e89a3a7802d42c47fc67091686fd2d340f8427e`, and
  both network-isolated installation paths passed.
- The production site dependency audit found 0 vulnerabilities and Astro
  built all 16 static pages. This remains local evidence; hosted Linux and
  Pages, external transport/resource-fault execution, secret custody, and
  independent reviewer evidence remain separate release gates.

Complete local release preflight at `b12fe54` on 2026-10-23:

- Static checks, formatting, mypy over 242 source files, Bandit, pip-audit,
  and public API compatibility passed. The bounded release suite passed
  `2,078` tests with `76` explained optional-dependency skips; the skip audit
  found no unexplained skips or xfails. Total coverage was 82.75%, while the
  declared core contract scope reached 86%.
- The three-run stability gate passed with no quarantined failures, inventory
  changes, expired exceptions, or unlisted skips. Mutation testing killed
  `1,503/2,057` mutants (73%) against the 70% minimum; one timeout and 553
  survivors remain recorded engineering debt rather than being hidden.
- Five-seed offline benchmarks reported zero regressions and zero failed
  repetitions. Verdicts were tie, tie, tie, and ahead for breast cancer,
  wine, digits, and diabetes progression; runtime, peak memory, failure rate,
  and confidence intervals were retained in the benchmark artifact.
- Wheel and source-distribution builds reproduced byte-for-byte with wheel
  digest `c236f12dbe1a4ee8ab45af1102c5dd8becf9737ab56d43c35518f3f09342aa21`
  and sdist digest
  `23482efe278d178b805fbc7c6db82214f45a3f724674d318a9da1afa74af52d9`.
  Network-isolated wheel installation and source-distribution rebuild and
  installation both passed.
- The standalone Astro site passed npm audit with zero vulnerabilities,
  built 18 pages, passed accessibility checks for all 18 HTML files and 427
  internal links, and passed both browser smoke tests. This is complete local
  evidence; hosted GitHub Actions/Pages, external fault execution, secret
  custody, and independent reviewer sign-off remain external release gates.

Current-head complete strict regression at `a2d973b` on 2026-10-23:

- `uv run pytest -q tests -W error::ResourceWarning` passed `2,405` tests
  with `89` declared optional-dependency skips in 27 minutes 24 seconds.
  There were no failures, errors, or resource-warning failures. The run
  passed after the no-semi-private-name repair and is the authoritative full
  repository regression for this pushed head.
- This closes the current local correctness regression gate. Coverage,
  mutation, benchmark, package, security, hosted CI/Pages, external fault,
  secret-custody, and independent-review gates remain separately scoped and
  are not implied by this test result.

Parameter-estimation evidence hardening at `48f7537` on 2026-10-23:

- `ParameterEstimate` and `EstimationResult` now reject non-finite
  coefficients, negative standard errors, invalid p-values, reversed
  intervals, non-square or non-symmetric covariance, and unreconciled row
  accounting before a report can be written or consumed.
- Digest verification now occurs before semantic deserialization, and the
  parsed result must reproduce the same canonical digest. A modified report
  therefore fails as tampered evidence rather than being reinterpreted as a
  different valid report.
- The focused parameter-estimation suite passed 7 tests under
  `-W error::ResourceWarning`; Ruff and mypy passed. This strengthens the
  bounded frequentist artifact contract; it does not make the estimators
  causal, Bayesian, modality-specific, or an integrated `MLProject` release
  gate.

Parameter-estimand alignment at `4b41269` on 2026-10-23:

- The estimation boundary now fails before fitting when the declared
  likelihood does not match the selected estimator (`gaussian` for linear or
  `bernoulli` for logistic). The result no longer silently rewrites a
  scientific declaration after the computation has started.
- The focused parameter-estimation suite passed 8 tests under
  `-W error::ResourceWarning`; Ruff and mypy passed. This is a contract
  correctness improvement within the bounded frequentist layer, not evidence
  for causal interpretation or broader model support.

Dataset-manifest identity hardening at `7f9a7ed` on 2026-10-23:

- `DatasetManifest` now validates non-empty dataset, schema, data, target,
  and modality identities; non-negative row and column counts; positive schema
  versions; typed provenance components; and typed curation findings before a
  manifest can enter a run or artifact.
- The focused data-evidence, source-registry, and capability-use slice passed
  32 tests under `-W error::ResourceWarning`; Ruff and mypy passed. This
  strengthens phase-1 data identity and curation evidence without claiming
  that arbitrary external sources or distributed broker semantics are solved.

Modality-manifest count hardening at `ace10f9` on 2026-10-23:

- `ModalityManifest.sample_count` now accepts only non-negative integers,
  rejecting booleans and fractional counts before lifecycle evidence is
  persisted. This keeps dataset cards, split evidence, and resource budgets
  numerically trustworthy across arbitrary modality adapters.
- The focused modality contract, project, and lifecycle slice passed 64 tests
  under `-W error::ResourceWarning`; Ruff and mypy passed. This improves the
  data contract only and does not promote experimental modality modelling to
  production support.

Cross-phase regression at `a79d263` on 2026-10-23:

- The combined data-evidence, modality contract/project/lifecycle,
  parameter-estimation, local fault, reliability-harness, capability-gate,
  and tabular end-to-end suites passed 155 tests in 2 minutes 1 second under
  `-W error::ResourceWarning`.
- This is current-head focused regression evidence for the changed contract
  boundaries. It is not a substitute for the complete repository suite,
  mutation gate, hosted CI, public Pages verification, external fault
  execution, or independent release review.

Reliability CI evidence plumbing at `a76d232` on 2026-10-23:

- `.github/workflows/ci.yml` now has a dedicated pinned-action reliability
  job for the fault contracts, local fault matrix, independent-review
  verifier, and incident workflow. It emits a JUnit artifact and rejects
  unexplained skips and xfails independently of the broad quality suite.
- `actionlint` passed, and the exact CI-shaped command passed locally with 52
  tests, zero skips, and zero xfails under `-W error::ResourceWarning`.
- This makes the reliability gate observable when hosted CI is available. It
  does not count as hosted execution until GitHub Actions produces a real
  run, and it does not replace cloud, container, kernel, broker, or external
  reviewer evidence.

Convention-gate regression repair at `20d2950` on 2026-10-23:

- The first complete strict run at the preceding head reached 2,403 passed
  tests and 89 declared skips, then exposed two failures in the repository's
  no-semi-private-name convention gate: placeholder loop identifiers in the
  parameter-estimation bootstrap path and tabular JEPA engine.
- Both placeholders were replaced with explicit public loop state while
  preserving deterministic iteration. The convention gate plus the affected
  parameter-estimation and JEPA contract tests then passed 870 tests with one
  explained optional skip, and Ruff passed.
- A fresh complete strict run at the corrected head is still required before
  this entry can be treated as final full-suite evidence.

Fault-evidence publication hardening at `32ecfc2` on 2026-10-23:

- `FaultInjectionHarness` now validates duplicate and unexpected evidence
  types against the declared scenario contract before writing any artifact.
  Invalid executor output therefore cannot overwrite an evidence file or
  leave orphaned artifacts that look like a partial release record. Missing
  evidence remains explicitly inconclusive, preserving the distinction
  between incomplete coverage and an invalid contract.
- The focused local fault and harness slice passed 24 tests under
  `-W error::ResourceWarning`; Ruff and mypy passed. This strengthens the
  local evidence boundary only. It does not provide external broker,
  container, cloud, kernel, or independent-review evidence.

Standalone modelling and estimation documentation at `80ea0eb` on 2026-10-23:

- The Astro site now has a standalone `/docs/modeling/` page and navigation
  entry. It explains the supported tabular boundary, optional FT-Transformer
  and sentence-transformer paths, the extension-only TabPFN boundary, and the
  fact that JEPA is not implemented. It also distinguishes hyperparameter
  search, predictive uncertainty, model-selection uncertainty, and the still
  missing first-class statistical parameter-estimation contract.
- The page records the acceptance bar for future parameter estimation:
  explicit estimand and population, identification assumptions, resolved
  estimator and convergence evidence, a named uncertainty method,
  reproducibility from raw snapshot to artifact digest, and sensitivity and
  subgroup checks. It does not imply that current calibration or conformal
  prediction is causal or statistical parameter estimation.
- Route inventories and the public-route browser test include the new page.
  Local Astro build produced `18` pages; internal-link checks passed for
  `427` links, accessibility smoke checks passed for all `18` HTML files, and
  both Playwright tests passed. The change was pushed to
  `feat/v0.1-foundation`.
- This improves onboarding and truthfulness; it does not add JEPA,
  production multimodal modelling, or the missing parameter-estimation API.

Parameter-estimation evidence at `e8e32dd` on 2026-10-23:

- The public API now exposes `EstimandSpec`, `EstimationPolicy`,
  `ParameterEstimate`, `EstimationResult`, and `estimate_parameters`.
  The initial estimators cover declared Gaussian linear and Bernoulli logistic
  likelihoods with IID, HC3, or clustered covariance, confidence intervals,
  explicit reject-or-complete-case missingness, bootstrap sampling
  sensitivity, target-population standardized-mean-shift diagnostics, and a
  second reference calculation (`numpy.linalg.lstsq` or SciPy BFGS).
- Four focused tests cover known coefficients, robust and clustered
  uncertainty, independent logistic verification, JSON serialization, target
  shift, bootstrap evidence, and fail-closed missingness. The focused API and
  estimation slice passed `13` tests with strict resource warnings; Ruff,
  mypy, and `git diff --check` passed.
- This is a bounded frequentist estimation layer. It does not claim causal
  identification, Bayesian posterior inference, prior sensitivity, or
  automatic integration into every `MLProject` release artifact yet.

Bounded tabular JEPA evidence at `57ac21c` on 2026-10-23:

- The optional PyTorch worker now contains a real tabular JEPA trainer:
  numeric feature masking, context encoder, EMA target encoder,
  embedding-space prediction, variance regularization, deterministic seed,
  worker isolation, serialized NumPy weights, and a `transform` representation
  surface. It is deliberately not registered as a supervised model-search
  candidate.
- The contract test passed locally; the optional functional test is skipped
  because torch is not installed in the current environment. A torch-enabled
  run is still required for performance, checkpoint replay, resource-fault,
  representation-transfer, and independent benchmark evidence.
- The site now calls this “tabular JEPA” and explicitly denies multimodal
  foundation-model coverage. Documents, images, audio, video, graphs,
  geospatial data, and raster JEPA encoders remain separate work.

Estimation artifact reliability at `784a5a1` on 2026-10-23:

- `EstimationResult.write()` now persists a schema-versioned report through an
  atomic temporary-file replacement and binds the canonical report to a
  SHA-256 digest. `EstimationResult.read()` reconstructs the typed result and
  fails closed on schema or digest mismatch.
- The policy now rejects unknown estimator, covariance, and missingness values
  before data access. Clustered covariance uses cluster-level score sums with
  finite-sample correction rather than silently reusing row-level HC3 scores.
- The focused estimation suite passed `5` tests, including round-trip,
  tamper, robust covariance, bootstrap, shift, logistic reference, and
  missingness cases; Ruff and mypy passed.

Signed modality checkpoint recovery at `3831565` on 2026-10-22:

- The non-tabular lifecycle now supports an explicit, signed, atomically
  persisted checkpoint path. Each stage boundary records the ordered
  lifecycle prefix and a digest of its JSON evidence; resume replays completed
  stages against the caller-provided adapter and rejects any divergence,
  changed dataset, changed manifest, wrong seed, tampered checkpoint, or
  mismatched package identity before continuing.
- Recovery does not deserialize an arbitrary adapter or estimator from the
  checkpoint. A completed package is verified through its existing signature,
  and a replayed packaging stage verifies the package model, feature schema,
  dataset identity, and deterministic coefficients before reusing it.
- A simulated worker failure after packaging resumed successfully without
  republishing the package. Completed-checkpoint resume returned the signed
  report idempotently; tampered and changed-snapshot checkpoints failed closed.
- The modality lifecycle, public project, API compatibility, and API surface
  slice passed `56` tests under strict resource warnings. Ruff, formatting,
  mypy, and diff checks passed. This closes the local modality resume boundary;
  external process/container/kernel fault execution and independent release
  review remain separate gates.

Current-head release preflight at `a56492a` on 2026-10-22:

- `./scripts/release_preflight.sh` passed static analysis, formatting, mypy,
  Bandit, pip-audit, API compatibility, the bounded contract suite, three-run
  stability, mutation, benchmark regression, package validation,
  reproducibility, and network-isolated installation gates. The bounded suite
  passed `2,042` tests with `75` explicitly explained optional-dependency
  skips; total coverage was `83.14%` and the declared core scope was `86%`.
- Mutation testing killed `1,505/2,057` mutants (`73%`), above the `70%`
  release threshold; one mutation timed out and `551` survived for follow-up
  triage. The four five-seed offline benchmarks recorded zero regressions and
  zero failed repetitions, with verdicts tie, tie, tie, and ahead.
- Reproducible packages and network-isolated wheel/source installs passed.
  The wheel hash was
  `06fdb1dcbfaf7773ea331bed6234c13fd85bccfed1258432efd812728a0e64dc` and
  the source-distribution hash was
  `32e8cb1cb84530f9f5119d2798266752c79f8d718ba642a7c652ad01301a7c39`.
- The site gate found `0` npm audit vulnerabilities, Astro built `17` static
  pages, accessibility checks passed for all `17` HTML files, all `384`
  internal links resolved, and both Playwright smoke tests passed. Hosted CI,
  public Pages deployment, external transport/resource-fault execution, secret
  custody, and independent reviewer evidence remain open release gates.

Latest hosted gate observation at `009cb06` on 2026-10-22:

- CI run `39756648421` failed across its matrix before any job step executed;
  every job reported an empty step list. Documentation deployment run
  `39756648363` failed in its build job before any step executed, leaving its
  deploy and post-deploy verification jobs skipped. These are hosted runner or
  account failures, not source-test results; the same behavior has now been
  observed across multiple pushed heads.
- The repository remains private with default branch
  `feat/v0.1-foundation`, and the Pages API still returns `404` because no
  Pages site is configured. Public deployment requires owner-controlled
  repository visibility, Pages activation, and runner/billing changes.

Cooperative modality cancellation at `aed4d0b` on 2026-10-22:

- `LifecycleCancellation` provides a thread-safe public signal and
  `ModalityProject.cancel()` requests cancellation without introducing a
  modality-specific implementation. `ModalityLifecycle` checks the signal at
  stage boundaries and before/after report publication; a cancelled run
  removes the package and lifecycle report created by that run, so partial
  evidence cannot be mistaken for a completed lifecycle.
- Pre-run cancellation and cancellation after the first completed stage are
  covered by focused tests. The API, lifecycle, project, and explanation slice
  passed `67` tests under `-W error::ResourceWarning`; Ruff, formatting, mypy,
  API compatibility, and `git diff --check` passed.
- This closes the cooperative local cancellation contract. Process death,
  timeout, and external resource-fault scenarios remain covered by the local
  fault harness; hosted transport and infrastructure failure evidence remain
  external release gates.

Public cancellation API at `49ffa98` on 2026-10-22:

- `LifecycleCancellation` and `LifecycleCancelledError` are now exported from
  the top-level `scml` API, so callers do not need to depend on an internal
  module path to control or handle a modality run. The versioned public API
  snapshot and surface contract include both symbols.
- API compatibility, Ruff, formatting, mypy, and the focused lifecycle,
  project, explanation, and API slice passed `67` tests under strict resource
  warnings.

Structured cancellation taxonomy at `bedb671` on 2026-10-22:

- `LifecycleCancelledError` now derives from the framework's structured
  `CancelledError` contract and carries `category="cancelled"` with a
  machine-readable `scope="modality_lifecycle"`. Modality callers can handle
  cancellation through the same error taxonomy as the tabular orchestrator.
- The focused API, lifecycle, project, and explanation slice remained green
  at `67` tests under strict resource warnings, with Ruff, formatting, mypy,
  API compatibility, and diff checks passing.

Fail-closed performance publication at `38f6736` on 2026-10-22:

- A modality run that violates its declared performance budget or repeated
  baseline now rejects before lifecycle-report persistence and removes the
  package created by that run. A failed performance gate can no longer leave
  a signed-looking model package or report that an operator might mistake for
  a releasable result.
- The regression test verifies both package and report absence after a budget
  failure. The focused modality lifecycle, project, and explanation slice
  passed `58` tests under strict resource warnings; Ruff, formatting, mypy,
  and diff checks passed.

Baseline failure regression evidence at `22fc083` on 2026-10-22:

- A deterministic repeated-baseline fixture now exercises the failed baseline
  branch as well as the failed budget branch. Both paths prove that the
  package and lifecycle report are absent after rejection.
- The focused modality lifecycle, project, and explanation slice passed `58`
  tests under strict resource warnings; mypy, Ruff, formatting, and diff
  checks remained clean.

Signed lifecycle evidence at `7dc754f` on 2026-10-22:

- Lifecycle reports are now schema version 2 artifacts signed over the exact
  modality, dataset hash, stage evidence, performance evidence, package
  identity, inference, monitoring, documentation, and recovery payload. The
  report can be read and verified with the same signing key contract used by
  the package manifest; wrong-key and post-write dataset tampering fail closed.
- The lifecycle runner signs the report before atomic persistence and returns
  the signed object to callers. Round-trip, wrong-key, and tamper-rejection
  coverage passed in the 56-test public modality, lifecycle, and explanation
  slice under `-W error::ResourceWarning`; Ruff, formatting, mypy, and
  `git diff --check` passed.
- This closes report-integrity at rest for the local lifecycle contract. It
  does not provide external secret custody, independent reviewer signatures,
  hosted CI authority, or a production support claim for built-in modalities.

Exact-head full-suite evidence at `9c10883` on 2026-10-22:

- The complete repository suite passed `2,369` tests with `88` optional
  dependency skips in `31m24s` under `-W error::ResourceWarning`. The explicit
  JUnit skip/xfail audit reported `unexplained: []` and `xfails: []`; every
  skip carried a concrete missing-dependency reason.
- The API compatibility snapshot was refreshed for the already-public
  `ModalityProject` constructor contract. The compatibility verifier and API
  surface tests passed `9` tests before the full-suite rerun, and the full
  suite passed at the fixed head.
- This is stronger local correctness evidence, but it does not replace the
  unavailable hosted CI runner, public Pages deployment, independent review,
  external secret custody, or external transport/resource-fault evidence.

Current-head optional-boundary release preflight evidence at `eb163a4` on
2026-10-22:

- `./scripts/release_preflight.sh` passed the local static, packaging, site,
  and bounded-test gates. The bounded release suite passed `2,025` tests with
  `75` explicitly explained optional-dependency skips. Total coverage was
  `83.03%` and the declared core scope was `86%`.
- The enforced stability gate passed three attempts over the same `25` test
  cases with no inventory changes, quarantined failures, or expired entries.
  Mutation testing killed `1,503/2,057` mutants (`73%`), above the `70%`
  threshold; one mutation timed out and `553` survived for follow-up triage.
- Five-seed offline benchmark regression completed with zero regressions and
  zero failed repetitions. Verdicts were tie, tie, tie, and ahead for breast
  cancer, wine, digits, and diabetes progression; peak memory was `0.3681`,
  `0.3915`, `0.4530`, and `0.4577` GB respectively.
- Reproducible package checks and network-isolated wheel/source installs
  passed. The wheel hash was
  `57f15957aadbd3739617fbab4dfca707cf43e6cdae25b2ef7102da75d29cbad1` and
  the source-distribution hash was
  `b70c135ec5bd8fc8c1071292dc9f19a25f7bb054de5ead1a72086e1b97546971`.
- The site gate found `0` npm audit vulnerabilities, Astro built `17` static
  pages, accessibility checks passed for all `17` HTML files, all `384`
  internal links resolved, and both Playwright smoke tests passed. This is
  local evidence only; hosted Linux and Pages, external transport/resource-
  fault execution, secret custody, and independent reviewer evidence remain
  open release gates.

Hosted-test skip enforcement at `fbeaca3` on 2026-10-22:

- The primary CI bounded suite, scheduled full suite, and exact-SHA clean-clone
  rehearsal now publish JUnit results and run `scripts/check_pytest_skips.py`.
  A hosted green run therefore fails closed when a collected test is skipped
  without an explicit reason, instead of treating the skip as invisible.
- The workflow YAML passed `actionlint` and Ruby YAML parsing locally. The
  project stability policy still passed three attempts over `25` test cases,
  and the focused JUnit skip-audit contract passed `10` tests with zero
  unexplained skips. This verifies workflow structure and local policy
  behavior; hosted runner execution and public Pages deployment remain open
  external gates.

Skip and xfail release enforcement at `ed3e3c5` on 2026-10-22:

- `scripts/check_pytest_skips.py` now rejects both unexplained skips and any
  xfail or xpass outcome in a JUnit report. This closes the gap between the
  plan's no-unexplained-skips-or-xfails criterion and the executable gate.
- The verifier's focused suite and the repository naming-convention suite
  passed `854` tests under `-W error::ResourceWarning`; Ruff, formatting,
  mypy, and `actionlint` were clean. A synthetic xfail report was rejected
  fail-closed, while a report with an explicit skip reason remained accepted.

All-extras CI capability matrix at `151bc4d` on 2026-10-22:

- CI now installs every declared optional extra on Python 3.13 and exercises
  the optional boosting, ONNX, sampling, spreadsheet, deep-learning, Keras,
  embedding, MLflow, Ray, text-embedding, reader, interchange, and curriculum
  paths. The matrix is JUnit-backed and applies the skip/xfail release gate.
- The same clean all-extras environment passed `124` tests with `0` skips and
  one known upstream MLflow SQLAlchemy `SADeprecationWarning`; no
  `ResourceWarning` was emitted. The JUnit verifier reported empty
  unexplained-skip and xfail sets, and `actionlint` passed. Hosted execution
  remains the cross-platform authority because the local run is macOS.

Tagged-release outcome enforcement at `5ce4f73` on 2026-10-22:

- The release tag workflow now promotes `ResourceWarning` to a failure and
  runs the same JUnit skip/xfail verifier for both its platform acceptance
  slice and its full publish suite. A release cannot pass by silently
  accepting a skipped or resource-leaking test path.
- The release workflow passed `actionlint` and Ruby YAML parsing locally; the
  actual tag workflow remains unverified because hosted GitHub runners are
  currently unavailable for this repository.

Exact-head local release preflight at `6e37fa5` on 2026-10-22:

- `./scripts/release_preflight.sh` passed all local gates. The bounded suite
  passed `2,026` tests with `75` explicitly explained optional-dependency
  skips; the JUnit outcome audit found zero unexplained skips and zero
  xfail/XPASS outcomes. Total coverage was `83.02%` and the declared core
  scope was `86%`.
- The three-attempt stability policy passed with no inventory changes,
  quarantined failures, or expired entries. Mutation testing killed
  `1,504/2,057` mutants (`73%`), above the `70%` threshold; one mutation
  timed out and `552` survived for follow-up triage.
- Five-seed offline benchmarks recorded zero regressions and zero failed
  repetitions. Verdicts were tie, tie, tie, and ahead for breast cancer,
  wine, digits, and diabetes progression. Peak memory was `0.3624`, `0.3837`,
  `0.4512`, and `0.4552` GB; runtimes were `105`, `75`, `632`, and `70`
  seconds respectively.
- Reproducible packages and network-isolated wheel/source installs passed.
  The wheel hash was
  `57f15957aadbd3739617fbab4dfca707cf43e6cdae25b2ef7102da75d29cbad1` and
  the source-distribution hash was
  `0153d5c6f392cc00208f4d0c365f014bb25c49e442eb8e403e7f6f3bb5ff4d5c`.
- The site gate found `0` npm audit vulnerabilities, Astro built `17` static
  pages, accessibility checks passed for all `17` HTML files, all `384`
  internal links resolved, and both Playwright smoke tests passed. This is
  the strongest local evidence at the current head; hosted Linux and Pages,
  external transport/resource-fault execution, secret custody, and
  independent reviewer evidence remain open release gates.

Workflow-wide test outcome enforcement at `2b75123` on 2026-10-22:

- The remaining CI matrix jobs for cross-platform, lowest-direct, deep
  learning, integrations, base install, and data formats now emit JUnit
  results and run the same skip/xfail verifier as the primary and all-extras
  jobs. Every pytest job in the CI workflow therefore has an explicit outcome
  audit rather than relying on a passing exit code alone.
- The workflow passed `actionlint` and Ruby YAML parsing locally; the focused
  stability and outcome-verifier tests passed `11` tests under strict resource
  warnings. Hosted execution remains unavailable for this repository.

Release-evidence integration coverage at `09d8e7e` on 2026-10-22:

- The bounded release preflight and primary CI quality job now include the
  governance audit and problem-spec integration suites. These suites exercise
  audit-event ordering, data-access verification, signed release attestations,
  versioned evidence indexes, release-policy blocking, SBOM/provenance output,
  and tamper detection before the release evidence gate is considered green.
- The focused governance/problem-spec slice passed `10` tests with zero
  skips, zero xfail/XPASS outcomes, and strict resource warnings enabled;
  `bash -n`, `actionlint`, and diff checks were clean.

Independent-review gate hardening at `d82bfc1` on 2026-10-22:

- `scripts/verify_release_review.py` now requires every critical review claim
  kind by default: metric, fairness, explanation, reliability, and security.
  A caller may still request a narrower diagnostic subset explicitly, but the
  production default cannot silently omit reliability or security claims.
- The focused independent-review and convention suites passed `856` tests
  under `-W error::ResourceWarning`; Ruff, formatting, mypy, and diff checks
  were clean. A signed review missing reliability and security claims is now
  rejected by the default CLI gate.

Exact-head release preflight at `3ef6f59` on 2026-10-22:

- `./scripts/release_preflight.sh` passed static analysis, formatting, mypy,
  security, API compatibility, bounded tests, stability, mutation, benchmark,
  packaging, reproducibility, and site gates. The bounded suite passed
  `2,037` tests with `75` explicitly explained optional-dependency skips; the
  JUnit outcome audit found zero unexplained skips and zero xfail/XPASS
  outcomes. Total coverage was `83.19%` and the declared core scope was `86%`.
- The three-attempt stability policy passed with no inventory changes,
  quarantined failures, or expired entries. Mutation testing killed
  `1,503/2,057` mutants (`73%`), above the `70%` threshold; one mutation
  timed out and `553` survived for follow-up triage.
- Five-seed offline benchmarks recorded zero regressions and zero failed
  repetitions for breast cancer, wine, digits, and diabetes progression. The
  verdicts were tie, tie, tie, and ahead; runtimes were `105`, `76`, `632`,
  and `70` seconds, with peak memory of `0.3723`, `0.3757`, `0.4378`, and
  `0.4443` GB respectively.
- Reproducible packages and network-isolated wheel/source installs passed.
  The wheel hash was
  `57f15957aadbd3739617fbab4dfca707cf43e6cdae25b2ef7102da75d29cbad1` and
  the source-distribution hash was
  `368fcd6af5423114e85506578db5d0d924cc2c5d60a2d5b65aa7a12b8e2aba07`.
- The site gate found `0` npm audit vulnerabilities, Astro built `17` static
  pages, accessibility checks passed for all `17` HTML files, all `384`
  internal links resolved, and both Playwright smoke tests passed. This is
  local evidence; hosted Linux and Pages, external transport/resource-fault
  execution, secret custody, and independent reviewer evidence remain open
  release gates.

Current hosted gate observation at `6c11cdc` on 2026-10-22:

- The `ci` run `39730492994` and documentation deployment run `39730493042`
  both completed as failures before any job step executed. This is consistent
  with the account/runner availability failure observed on earlier pushes, not
  a source-test result.
- GitHub reports `private: true` and `has_pages: false` for
  `sachncs/magicml`; the requested public Pages URL therefore remains
  unverified. Repository visibility, Pages activation, and the account
  runner/billing condition require owner-controlled external changes.

Exact-head evidence-integration release preflight at `09d8e7e` on 2026-10-22:

- `./scripts/release_preflight.sh` passed the static, formatting, mypy,
  security, API-compatibility, bounded-test, stability, mutation, benchmark,
  packaging, reproducibility, and site gates. The bounded suite passed `2,036`
  tests with `75` explicitly explained optional-dependency skips; the JUnit
  outcome audit found zero unexplained skips and zero xfail/XPASS outcomes.
  Total coverage was `83.19%` and the declared core scope was `86%`.
- The three-attempt stability policy passed with no inventory changes,
  quarantined failures, or expired entries. Mutation testing killed
  `1,504/2,057` mutants (`73%`), above the `70%` threshold; one mutation
  timed out and `552` survived for follow-up triage.
- Five-seed offline benchmarks recorded zero regressions and zero failed
  repetitions for breast cancer, wine, digits, and diabetes progression. The
  verdicts were tie, tie, tie, and ahead; runtimes were `105`, `76`, `634`,
  and `70` seconds, with peak memory of `0.3681`, `0.3918`, `0.4560`, and
  `0.4618` GB respectively.
- Reproducible packages and network-isolated wheel/source installs passed.
  The wheel hash was
  `57f15957aadbd3739617fbab4dfca707cf43e6cdae25b2ef7102da75d29cbad1` and
  the source-distribution hash was
  `f9e37ee8346eba2c41d3788e9681b2698c23f98a35b8ba03fa9e97bac9fa681a`.
- The site gate found `0` npm audit vulnerabilities, Astro built `17` static
  pages, accessibility checks passed for all `17` HTML files, all `384`
  internal links resolved, and both Playwright smoke tests passed. This is
  local evidence at the code commit; the documentation-only follow-up is
  `d023ef2`. Hosted Linux and Pages, external transport/resource-fault
  execution, secret custody, and independent reviewer evidence remain open
  release gates.

Current-head release preflight evidence at `9a6e82e` on 2026-10-22:

- `./scripts/release_preflight.sh` passed Ruff, formatting, mypy over 239
  source files, Bandit, pip-audit, public API compatibility, and explicit
  JUnit skip-reason auditing. The bounded suite passed `2,024` tests with
  `75` explicitly explained optional-dependency skips; total coverage was
  `83.02%` and the declared core scope was `86%`.
- Mutation testing killed `1,504/2,057` mutants (`73%`), above the `70%`
  release threshold. One mutation timed out and `552` survived for follow-up
  triage; no survivor was converted into a hidden pass claim.
- The five-seed offline benchmark gate completed for breast cancer, wine,
  digits, and diabetes progression with zero regressions and zero failed
  repetitions. Verdicts were tie, tie, tie, and ahead. The run recorded
  runtime, peak memory, failure rate, seed/trial count, and hardware/software
  identity for each dataset.
- Wheel and source-distribution checks passed, artifacts reproduced
  byte-for-byte with wheel hash
  `3c345324bf11250193547afd58dafac5ec96e7468d3fb65848842a96039d9be2` and
  source-distribution hash
  `5268acac4f69dff0e65e80dc0c8123ac58c7544d191383af37f599c069369e6b`, and
  both network-isolated installation paths passed.
- The site dependency audit found `0` vulnerabilities; Astro built `17`
  static pages, accessibility checks passed for all `17` HTML files, all
  `384` internal links resolved, and both Playwright smoke tests passed.
  This is current local evidence. Hosted Linux and Pages, external
  transport/resource-fault execution, secret custody, and independent
  reviewer evidence remain separate release gates.

Current-head release preflight with stability enforcement at `ed02bf2` on
2026-10-22:

- `./scripts/release_preflight.sh` now runs the three-attempt stability and
  quarantine policy as part of the release command. The complete preflight
  passed `2,024` bounded tests with `75` explicitly explained optional
  dependency skips; all three stability attempts passed the same `25` test
  cases with no inventory changes, quarantined failures, or expired entries.
  Total coverage was `83.02%` and the declared core scope was `86%`.
- Mutation testing killed `1,503/2,057` mutants (`73%`), above the `70%`
  threshold. One mutation timed out and `553` survived for follow-up triage.
- The five-seed offline benchmark gate recorded zero regressions and zero
  failed repetitions for breast cancer, wine, digits, and diabetes
  progression. Verdicts were tie, tie, tie, and ahead, with peak memory of
  `0.3665`, `0.3900`, `0.4526`, and `0.4577` GB respectively.
- Package checks, byte-for-byte reproducible builds, and network-isolated
  wheel/source installs passed. The wheel hash was
  `3c345324bf11250193547afd58dafac5ec96e7468d3fb65848842a96039d9be2` and
  the source-distribution hash was
  `69493416d342f0d7f8e5a40cc6e8a2616f34bbeec1aaa1800805f68e5be64f20`.
- The site audit found `0` vulnerabilities; Astro built `17` pages, the
  accessibility checks passed for all pages, all `384` internal links
  resolved, and both Playwright smoke tests passed. This remains local
  release evidence; hosted CI and Pages, external fault execution, secret
  custody, and independent review remain open gates.

All-extras capability boundary evidence at `9b4b6f3` on 2026-10-22:

- A clean Python 3.13 environment installed every declared optional extra.
  The all-extras `pip-audit` found no known vulnerabilities and the license
  inventory resolved `149` packages. The editable project itself is excluded
  from PyPI vulnerability lookup, as expected for a local checkout.
- The all-extras optional matrix initially exposed that installing TabPFN
  made the planner select its `foundation` family even though that family has
  no supported beta capability declaration. The adapter now rejects automatic
  planning as extension-only until accepted weight licensing, token custody,
  and complete release evidence exist; missing-package diagnostics remain
  unchanged. Its focused foundation/capability tests passed `14` tests, and
  the all-extras deep-learning suite passed `26` tests.
- The complete all-extras optional matrix then passed `124` tests under
  `-W error::ResourceWarning`. MLflow emitted the known upstream SQLAlchemy
  `SADeprecationWarning`; no project resource warning failed the run. This
  closes the discovered optional-planner defect without changing the honest
  supported boundary.

Current-head serving lifecycle preflight at `a27a3a8` on 2026-10-22:

- `HttpServing.start()` now rejects a second start on the same adapter with a
  structured library error. The focused serving suite passed 11 tests with 2
  explicit optional-dependency skips; Ruff, formatting, and mypy passed.
- The bounded release suite passed `2,021` tests with `75` explicit optional
  dependency skips. Total coverage was 83.01%; the declared core scope was
  86%. Mutation testing killed `1,503/2,057` mutants (73%), above the 70%
  gate; one mutation timed out and 553 survived for follow-up triage.
- The five-seed offline benchmark regression completed with zero regressions
  and zero failed repetitions. Verdicts were tie, tie, tie, and ahead for
  breast cancer, wine, digits, and diabetes progression; peak memory was
  0.3677, 0.3846, 0.4522, and 0.4560 GB respectively.
- Wheel and source-distribution checks passed, including network-isolated
  installation and source-distribution rebuild installation. Reproducible
  hashes were wheel
  `93121d3da9bc81f1bc698210f0f7040b850dc835a8904ae762ace73b638e52ea` and
  sdist
  `3d3608c778b74fbf2f812ee8ac4052efa90f4592ae40d1c6548992383b14be77`.
- The site gate reached `npm audit` after dependency installation, but the
  npm advisory endpoint timed out and returned an error. Astro build,
  accessibility, link, and Playwright results are therefore not claimed for
  this head; the last successful local site evidence remains recorded above.
  Hosted Linux and Pages, external fault execution, secret custody, and
  independent reviewer evidence remain separate release gates.

Instance-scoped serving policy at `6cbecf7` on 2026-10-22:

- `HttpServing` now accepts positive per-instance `max_request_bytes` and
  `max_records` bounds, matching the modality reference server and removing
  the need for process-global test or deployment mutation. Invalid bounds fail
  at construction before a socket can be started.
- The tabular and modality serving slice passed `13` tests with `2` explicit
  optional-dependency skips under `-W error::ResourceWarning`; Ruff,
  formatting, and mypy passed. This is a reference-server reliability
  boundary; authentication, TLS, gateway policy, and external fault evidence
  remain deployment-owner gates.

Empty-request serving boundary at `f704eeb` on 2026-10-22:

- Tabular `HttpServing` now rejects an empty `records` list before model
  scoring, returning the same structured malformed-request class used for
  other invalid HTTP payloads. This prevents an empty batch from reaching a
  model-specific scoring path with undefined operational meaning.
- The tabular and modality serving slice passed `14` tests with `2` explicit
  optional-dependency skips under `-W error::ResourceWarning`; Ruff,
  formatting, and mypy passed. This is a local input-contract guarantee and
  does not change the deployment-owner authentication, TLS, or paging gates.

Optional integration matrix evidence on 2026-10-22:

- A separate Python 3.13 environment installed the declared boosting,
  CatBoost, ONNX, sampling, spreadsheet, and Polars extras from the lockfile.
  Under `-W error::ResourceWarning`, the installable optional matrix passed
  `122` tests with `2` intentionally isolated cases, and the isolated
  CatBoost conformance, string-label, and end-to-end cases each passed. This
  verifies `125` optional integration tests across boosting, ONNX export,
  imbalanced sampling, spreadsheet and Polars interchange, and component
  contracts.
- Torch/Keras, sentence-transformers, MLflow, and Ray remain explicit
  environment-bound optional skips because their heavyweight or platform/
  service requirements were not installed. Their absence is not counted as
  support evidence; the hosted optional matrix remains required for those
  integrations.

Deep-learning capability declaration at `489c943` on 2026-10-22:

- The tabular capability registry now declares `deep_learning` for binary,
  multiclass, and regression paths and records the required `deeplearning`
  extra. Previously, Torch candidates reached planning and were rejected
  because their complete capability key was absent; the registry now makes
  the optional installation boundary explicit before training.
- In the isolated Python 3.13 optional environment, the Torch and Keras unit
  suites passed `31` tests with 1 explicit skip; the remaining deep-learning,
  embedding, MLflow, curriculum, and integration suite passed `52` tests with
  1 explicit skip; and the Ray cluster suite passed `5` tests. All runs used
  `-W error::ResourceWarning`. MLflow emitted one upstream SQLAlchemy
  deprecation warning, which is not a resource-warning failure.
- Together with the preceding optional matrix, `182` optional integration
  tests now have local evidence. Hosted cross-platform execution and
  heavyweight model/service environment review remain separate release gates.

Optional deep-learning skip boundary at `3024892` on 2026-10-22:

- A repeat of the remaining optional suite with `-rs` passed `52` tests and
  reported one intentional skip at `tests/unit/test_deeplearning.py:221`:
  `xgboost` was not installed in that isolated environment. The test is a
  cross-framework process-sharing check, so the skip is an environment
  boundary rather than evidence that the deep-learning adapter is broken.
- The same run emitted one MLflow upstream `SADeprecationWarning` from
  SQLAlchemy's `noload` loader strategy. It did not violate the configured
  `ResourceWarning` policy, but remains a third-party dependency signal to
  monitor in hosted dependency updates.

Current-head release preflight at `8651c3a` on 2026-10-22:

- `./scripts/release_preflight.sh` passed Ruff, formatting, mypy over 239
  source files, Bandit, pip-audit, public API compatibility, and explicit
  JUnit skip auditing. The bounded suite passed `2,018` tests with `75`
  explicitly explained optional-dependency skips; total coverage was `83.05%`
  and the enforced core scope was `86%`.
- Mutation testing killed `1,503/2,057` mutants (`73%`) against the configured
  `70%` minimum. One mutation timed out and `553` survived; the threshold was
  met and the survivor inventory remains follow-up evidence.
- Five-seed offline regression completed with zero failures and zero
  regressions. The four verdicts were tie, tie, tie, and ahead; peak memory
  was `0.3674`, `0.3898`, `0.4520`, and `0.4571` GB respectively.
- Wheel and source-distribution checks passed, both artifacts reproduced
  byte-for-byte, and network-isolated wheel and source-distribution installs
  passed. The reproducible wheel hash was
  `c3a0b809f6c51b9d8479a4968b6d44436ff9b5aa30d9587751d66bc8d947fc38` and
  the source-distribution hash was
  `eccf15201725b486466f3ba77be2d85a1a8a19bfc9aa1e499383b5fcd700bec1`.
- The marketing-site audit found zero npm vulnerabilities; Astro built `17`
  pages, accessibility checks passed for `17` HTML files, internal-link
  validation passed for `384` links, and both Playwright browser tests
  passed. Hosted Linux and Pages, external transport/resource-fault
  execution, secret custody, and independent reviewer evidence remain
  separate release gates.

Reference-serving boundary hardening at `e53f9a0` on 2026-10-22:

- `ModalityHttpServing` now requires JSON content, validates the declared
  body length and complete read, classifies malformed JSON as a request error,
  and converts unexpected decoder/model exceptions into a bounded structured
  internal response. Authentication, TLS, tracing, and operational paging
  remain deployment-owner responsibilities.
- The focused serving suite passed 3 tests under
  `-W error::ResourceWarning`; Ruff, formatting, and mypy passed. The change
  preserves the reference-server boundary and does not claim production
  service readiness.

Current-head release preflight at `e53f9a0` on 2026-10-22:

- `./scripts/release_preflight.sh` passed Ruff, formatting, mypy over 239
  source files, Bandit, pip-audit, public API compatibility, and explicit
  JUnit skip auditing. The bounded suite passed `2,019` tests with `75`
  explicitly explained optional-dependency skips; total coverage was `83.03%`
  and the enforced core scope was `86%`.
- Mutation testing killed `1,503/2,057` mutants (`73%`) against the configured
  `70%` minimum. One mutation timed out and `553` survived; the threshold was
  met and the survivor inventory remains follow-up evidence.
- Five-seed offline regression completed with zero failures and zero
  regressions. The four verdicts were tie, tie, tie, and ahead; peak memory
  was `0.3692`, `0.3863`, `0.4541`, and `0.4595` GB respectively.
- Wheel and source-distribution checks passed, both artifacts reproduced
  byte-for-byte, and network-isolated wheel and source-distribution installs
  passed. The reproducible wheel hash was
  `0a992516c41a96a2f83db97257a358dca5bf8fe2dbea76e898e3333d8548f424` and
  the source-distribution hash was
  `03c441fd20b664e703a8f87dec65cfa052994ec8f5c71919a4dbe843eb5f6851`.
- The marketing-site audit found zero npm vulnerabilities; Astro built `17`
  pages, accessibility checks passed for `17` HTML files, internal-link
  validation passed for `384` links, and both Playwright browser tests
  passed. Hosted Linux and Pages, external transport/resource-fault
  execution, secret custody, and independent reviewer evidence remain
  separate release gates.

Tabular reference-serving hardening at `bd43262` on 2026-10-22:

- `HttpServing` now requires JSON content, validates the declared body length
  and complete read, classifies malformed JSON as a request error, and turns
  unexpected model failures into a bounded structured internal response. This
  aligns the tabular and experimental modality reference boundaries while
  leaving authentication, TLS, tracing, and operational paging to the
  deployment owner.
- The serving integration suite passed `7` tests with `2` explicit optional
  skips under `-W error::ResourceWarning`; Ruff, formatting, and mypy passed.
  This improves the reference boundary but does not claim production service
  readiness.

Current-head release preflight at `bd43262` on 2026-10-22:

- `./scripts/release_preflight.sh` passed Ruff, formatting, mypy over 239
  source files, Bandit, pip-audit, public API compatibility, and explicit
  JUnit skip auditing. The bounded suite passed `2,020` tests with `75`
  explicitly explained optional-dependency skips; total coverage was `83.01%`
  and the enforced core scope was `86%`.
- Mutation testing killed `1,503/2,057` mutants (`73%`) against the configured
  `70%` minimum. One mutation timed out and `553` survived; the threshold was
  met and the survivor inventory remains follow-up evidence.
- Five-seed offline regression completed with zero failures and zero
  regressions. The four verdicts were tie, tie, tie, and ahead; peak memory
  was `0.3683`, `0.3917`, `0.4543`, and `0.4588` GB respectively.
- Wheel and source-distribution checks passed, both artifacts reproduced
  byte-for-byte, and network-isolated wheel and source-distribution installs
  passed. The reproducible wheel hash was
  `5f7f1865f6f51748526ed634059a198d7a513122e50379a21f292fee7739a0b5` and
  the source-distribution hash was
  `04b8254c59889686a7fb3d34e35a02c3af9a7d135523a006c66bf1d4267a1fd9`.
- The marketing-site audit found zero npm vulnerabilities; Astro built `17`
  pages, accessibility checks passed for `17` HTML files, internal-link
  validation passed for `384` links, and both Playwright browser tests
  passed. Hosted Linux and Pages, external transport/resource-fault
  execution, secret custody, and independent reviewer evidence remain
  separate release gates.

Video temporal boundary at `e9e623d` on 2026-10-22:

- Video samples now accept an explicit timezone-aware observation time,
  normalize it to UTC, bind it into dataset identity, and expose complete,
  mixed, or absent temporal coverage in curation. `VideoDataset.split_temporal`
  assigns clips at or before a UTC cutoff to training and later clips to future
  evaluation, and fails closed when any clip is untimed. The new temporal
  split/error contracts are exported from `scml.data.modalities`.
- The media, lifecycle, API surface, and API compatibility suites passed 90
  tests under `-W error::ResourceWarning`; Ruff, formatting, and mypy passed.
  This closes the video temporal-boundary evidence gap but does not claim
  temporal event semantics, watermark handling, causal ordering, production
  video modelling, or external release proof.

Image duplicate and augmentation evidence at `8cb5ae4` on 2026-10-22:

- Image curation now computes bounded quantized-luminance signatures and
  reports likely near-duplicate groups separately from exact decoded-content
  duplicates. It also accepts immutable caller-declared `ImageAugmentation`
  provenance containing parent source, operation, parameters, and tool; the
  provenance is part of image dataset identity and the curation profile.
- The image, lifecycle, public API, and compatibility suites passed 55 tests
  under `-W error::ResourceWarning`; Ruff, formatting, and mypy passed. The
  signature is review evidence, not semantic similarity, object identity,
  localization, or saliency evidence. Production-scale image quality and
  independent review remain separate gates.

Current-head release preflight at `04c301f` on 2026-10-22:

- `./scripts/release_preflight.sh` passed Ruff, formatting, mypy over 239
  source files, Bandit, pip-audit, public API compatibility, and explicit
  JUnit skip auditing. The bounded suite passed `2,018` tests with `75`
  explicitly explained optional-dependency skips; total coverage was `83.04%`
  and the enforced core scope was `86%`.
- Mutation testing killed `1,503/2,057` mutants (`73%`) against the configured
  `70%` minimum. One mutation timed out and `553` survived; the threshold was
  met and the survivor inventory remains follow-up evidence.
- Five-seed offline regression completed with zero failures and zero
  regressions. The four verdicts were tie, tie, tie, and ahead; peak memory
  was `0.3672`, `0.3849`, `0.4506`, and `0.4570` GB respectively.
- Wheel and source-distribution checks passed, both artifacts reproduced
  byte-for-byte, and network-isolated wheel and source-distribution installs
  passed. The reproducible wheel hash was
  `46fd2d177174d76e9305f5b1f5ddd3f55f8227a67930ef080b1ddd8ef27b1471` and
  the source-distribution hash was
  `877a82f98cc213f2fadf6160db15124a4079e3deafcdd5e24cf3a7dd0670bed2`.
- The marketing-site audit found zero npm vulnerabilities; Astro built `17`
  pages, accessibility checks passed for `17` HTML files, internal-link
  validation passed for `384` links, and both Playwright browser tests passed.
  Hosted Linux and Pages, external transport/resource-fault execution, secret
  custody, and independent reviewer evidence remain separate release gates.

Video lifecycle temporal policy at `03fe1ec` on 2026-10-22:

- `VideoAdapter` now accepts an explicit `temporal_cutoff`. When supplied,
  lifecycle partitioning uses the video dataset's UTC-normalized past/future
  split, puts future clips in the test partition, leaves validation empty, and
  does not let the random seed override the declared policy. Without the
  cutoff, the existing group-aware split remains the explicit default.
- The modality, media, API surface, and compatibility suites passed 91 tests
  under `-W error::ResourceWarning`; Ruff, formatting, and mypy passed. This
  closes lifecycle wiring for temporal boundaries but does not infer event
  time, provide watermarks, or establish causal ordering.

Image identity correction at `818dc34` on 2026-10-22:

- The image adapter now derives its declared dataset identity from the same
  source-and-content tuple used by image curation. A reversed-source-order
  regression test verifies that the persisted curation hash and lifecycle
  dataset hash remain identical. The modality lifecycle and media suites pass
  79 tests under `-W error::ResourceWarning`; Ruff, formatting, and mypy pass.
  This closes a local evidence-binding defect but does not change the
  experimental image support boundary or the external production gates.

Curation-only extension boundary at `d05bad1` on 2026-10-22:

- `ModalityCurationProject` now depends on a curation-specific polymorphic
  contract rather than the full modelling adapter. `ExtensionCurationAdapter`
  binds arbitrary caller-owned records to an `ExtensionModality`, immutable
  dataset hash, sample count, and a structured curation callback. It exposes no
  split, feature, label, or model hooks, so curation can remain the primary job
  for unlabeled or domain-specific data.
- The public API compatibility snapshot was updated. API, curation-project,
  lifecycle, naming, Ruff, formatting, and mypy checks pass; the focused
  project/lifecycle suite passed 37 tests under
  `-W error::ResourceWarning`. Full modelling support for the new family,
  independent quality evidence, serving, external fault rehearsal, and
  production promotion remain intentionally extension-only.

The empty extension-snapshot regression at `e190540` also passes: a
curation-only adapter can publish a signed zero-sample snapshot with its
identity and profile evidence, while modelling remains unavailable by design.

Media curation identity correction at `ad179d9` on 2026-10-22:

- Audio and video curation reports now reuse the dataset's source-order-bound
  `content_hash` instead of recomputing a differently ordered digest. This
  prevents a report from detaching from its adapter when source order is not
  lexicographically sorted.
- Reversed-order audio/video regression fixtures pass in the focused 78-test
  media/lifecycle suite under `-W error::ResourceWarning`; Ruff, formatting,
  and mypy remain clean.

Capability-matrix alignment at `2325150` on 2026-10-22:

- The standalone Astro capability matrix now names the implemented image
  near-duplicate and caller-declared augmentation evidence, and the video
  observation-time identity and fail-closed temporal split evidence. Its
  limitations continue to state that these boundaries do not establish raw
  saliency, semantic localization, temporal event understanding, or a
  production modality claim.
- The site rebuilt 17 static pages; internal-link validation passed for 384
  links, accessibility smoke checks passed for 17 HTML files, and both
  Playwright browser checks passed. This is a documentation alignment change;
  it does not alter the capability labels or external Pages deployment status.

Modality-package signature hardening at `09ca664` on 2026-10-22:

- `load_package` now verifies the expected and recorded HMAC values with
  constant-time comparison, matching the security posture of the bundle,
  attestation, registry, keyring, and independent-review verifiers.
- The valid-package and same-length wrong-key regression tests passed under
  `-W error::ResourceWarning`; Ruff, formatting, and mypy passed for the
  changed module. This closes a local verifier inconsistency but does not
  provide external key custody or an independent release signature.

Audio robustness evidence at `ab21c49` on 2026-10-22:

- `AudioAdapter` now records a bounded, deterministic additive-background-noise
  probe at 20, 10, and 0 dB SNR for the experimental linear baseline. Each
  scenario preserves the WAV parameters and transcript metadata, records the
  observed SNR, score delta, and post-mix clipping fraction, and uses an
  explicit seed. Non-WAV inputs and silent signals return `not_available`
  evidence rather than implying robustness.
- The probe is a repeatable stressor, not perceptual speech-quality evidence,
  real-environment noise coverage, denoising validation, or causal attribution.
  Focused media and lifecycle tests passed 76 tests under
  `-W error::ResourceWarning`; Ruff, formatting, and mypy passed. Hosted
  Linux, production-scale noise suites, and independent review remain
  separate release gates.

Current-head focused contract verification at `7c9850c` on 2026-10-22:

- The modality lifecycle, public modality-project, and media-modality slice
  passed 72 tests under `-W error::ResourceWarning`.
- `uv run ruff check scml tests scripts benchmarks`,
  `uv run ruff format --check scml tests scripts benchmarks`, and
  `uv run mypy scml benchmarks` all passed; 437 files were already formatted
  and mypy reported no issues in 239 source files.
- The worktree and pushed branch are identical at `7c9850c`. This focused
  evidence confirms the local experimental modality contract and coding
  standards, but it does not replace the complete release preflight,
  hosted Linux and Pages verification, external fault execution, secret
  custody, or independent reviewer evidence.

Geospatial spatial-join boundary at `40c9ca5` with site documentation at
`d38fe6e` on 2026-10-22:

- `GeospatialDataset.spatial_join()` now records same-CRS dataset hashes,
  versioned caller-predicate identity, comparison and match limits, matched
  feature IDs, and unmatched counts. CRS mismatch, excessive Cartesian work,
  excessive matches, predicate failures, and non-boolean results fail closed.
- The geospatial/media/modality focused suite passed 75 tests under
  `-W error::ResourceWarning`; Ruff, formatting, and mypy passed for the
  changed Python files.
- The standalone Astro site built 17 pages, passed 384 internal-link checks
  and accessibility smoke checks, and passed both Playwright browser tests.
  The join remains a bounded same-CRS evidence boundary; reprojection,
  indexed production geometry engines, and semantic spatial inference remain
  caller-owned or outside the experimental support claim.

Geospatial identity hardening at `fa21d4c` on 2026-10-22:

- `GeospatialDataset.content_hash` now includes the canonical CRS and sorts
  feature identities before hashing. Equivalent feature order is stable, but
  identical coordinate bytes in different CRSs produce different dataset
  identities.
- The media-modality slice passed 42 tests under
  `-W error::ResourceWarning`; Ruff, formatting, and mypy passed for the
  changed Python files. This closes a provenance correctness defect without
  changing the production support label.

Geospatial feature identity hardening at `6b21fd6` on 2026-10-22:

- `GeoFeature` now validates JSON-native properties and labels, freezes the
  properties mapping, and derives a feature content hash from feature ID,
  geometry, properties, label, and observation time. The dataset identity
  incorporates those feature hashes and the canonical CRS.
- The media-modality slice passed 43 tests under
  `-W error::ResourceWarning`; Ruff, formatting, and mypy passed for the
  changed Python files. A label or property correction therefore creates a
  new curation identity instead of silently reusing an old snapshot.

Geospatial reprojection boundary at `fe5a961` on 2026-10-22:

- `GeospatialDataset.reproject()` now accepts an explicit caller-owned,
  versioned coordinate transformer and returns a transformed dataset paired
  with source/target hash evidence. It preserves feature IDs, properties,
  labels, and observation times, and rejects transformer failures, invalid
  results, and metadata changes.
- The media-modality and lifecycle slice passed 79 tests under
  `-W error::ResourceWarning`; Ruff, formatting, and mypy passed for the
  changed Python files. This is an explicit extension boundary, not a claim
  that the core selects a CRS database or provides production geometry
  engine quality.

Geospatial boundary-leakage evidence at `b277032` on 2026-10-22:

- `GeospatialDataset.boundary_leakage()` now counts conservative feature
  envelope crossings in CRS-unit blocks with a bounded per-feature count.
  `GeospatialAdapter.curate()` persists this evidence and emits a visible,
  non-blocking `quality.geospatial-boundary-leakage` finding when a geometry
  may span multiple centroid-split blocks.
- The geospatial/media/modality focused suite passed 81 tests under
  `-W error::ResourceWarning`; Ruff, formatting, and mypy passed for the
  changed Python files. Exact geometry intersection remains caller-owned.

Current-head release preflight after source-backed document citations at
`8f26bfe` on 2026-10-22:

- `./scripts/release_preflight.sh` passed static checks, formatting, mypy over
  239 source files, Bandit, pip-audit, public API compatibility, explicit JUnit
  skip-reason auditing, the bounded suite, mutation testing, benchmarks,
  packaging, reproducible builds, isolated installs, and the Astro site gate.
- The bounded release suite passed `1,994` tests with `75` optional
  dependency skips. Every skip had an explicit reason. Total coverage was
  `83.12%`; the declared core scope was `86%`.
- Mutation testing killed `1,504` of `2,057` mutants (`73%`), above the 70%
  threshold. One mutation timed out and 552 survived for follow-up triage.
- The five-seed offline benchmark regression checked breast cancer, wine,
  digits, and diabetes progression with zero regressions and zero failed
  repetitions. Verdicts were tie, tie, tie, and ahead; peak memory was
  0.3661, 0.3828, 0.4405, and 0.4547 GB respectively, with elapsed seconds
  of 119, 76, 634, and 70.
- Wheel and source-distribution checks passed, artifacts reproduced
  byte-for-byte with wheel hash
  `a90c536a5e07b58fbb62b040d0371f8b3fa45aa8406a2b15e2b1fec8dc93f482` and
  sdist hash
  `8e1d21b49d648e52247e6386ab3947b82bfdc829c81a67d4d2785bc7c40b4db2`.
  Both network-isolated installation paths passed.
- The site audit found 0 npm vulnerabilities, built all 17 static pages,
  passed accessibility checks for all 17 HTML files, passed 384 internal-link
  checks, and passed both Chromium Playwright smoke tests. These are local
  results; hosted Linux and Pages, external transport/resource-fault
  execution, secret custody, and independent reviewer evidence remain
  separate release gates.

Document source-traceability hardening at `8c5b5ef` on 2026-10-22:

- `DocumentReader` now records a SHA-256 digest of the raw source bytes before
  decoding. `DocumentSample.citations()` emits immutable page or section
  locators only when both a source URI and raw-source digest are present; it
  never substitutes the decoded-content digest for a source digest.
- `DocumentCurationReport` records source-traceability coverage and citation
  count, and emits a visible non-blocking finding when a sample supplies only
  one half of its source identity. Document local explanations and raw-span
  occlusion carry the same citation records into JSON evidence.
- The document data, model, and modality-lifecycle focused suite passed 56
  tests under `-W error::ResourceWarning`; Ruff, formatting, and mypy passed.
- The Astro site build passed with 17 pages, 384 internal links, accessibility
  checks, and 2 Playwright smoke tests. The site now documents the distinction
  between source-backed locators and semantic citation grounding; the latter
  remains outside the experimental baseline.

Current-head curation-boundary verification at `beb4568` on 2026-10-22:

- `uv run --python 3.14 pytest -q tests -W error::ResourceWarning` passed
  `2,326` tests with `88` explicit optional-dependency skips in 31 minutes 15
  seconds. No failure, error, or promoted resource warning occurred. This run
  includes the curation-only task boundary: a modality curation project does
  not infer a modelling task when no manifest is supplied, and writes the
  explicit `not_declared` task value into the dataset card.
- `./scripts/release_preflight.sh` passed the local release gates. The bounded
  suite passed `1,991` tests with `75` optional-dependency skips; every skip
  had an explicit reason. Total coverage was `83.13%`; the declared core
  scope was `86%`.
- Mutation testing killed `1,503` of `2,057` mutants (`73%`), above the 70%
  threshold. One mutation timed out and 553 survived; the surviving-mutant
  inventory remains follow-up evidence rather than a hidden quality claim.
- The five-seed offline benchmark regression checked breast cancer, wine,
  digits, and diabetes progression with zero regressions and zero failed
  repetitions. The verdicts were tie, tie, tie, and ahead. Peak memory was
  0.3677, 0.3858, 0.4452, and 0.4513 GB respectively; elapsed seconds were
  105, 77, 631, and 70. This is bounded benchmark evidence, not a claim of
  universal model superiority.
- Wheel and source-distribution checks passed; artifacts reproduced
  byte-for-byte with wheel hash
  `4a3722924474614c3e7e4948a3dc2a6587a75574926d62d0d5f442c635873897` and
  sdist hash
  `3abb8fd0eb98d4e94a6aa65cb19d167336f264b2cc5f3a4b4a8c50e962192d48`.
  Network-isolated installation from both the wheel and rebuilt source
  distribution passed.
- The Astro site audit found 0 npm vulnerabilities, built all 17 static pages,
  passed accessibility checks for all 17 HTML files, passed 384 internal-link
  checks, and passed both Chromium Playwright smoke tests. This remains local
  evidence; hosted Linux and Pages, external transport/resource-fault
  execution, secret custody, and independent reviewer evidence remain
  separate release gates.

Document extraction boundary at `26bc8a0` on 2026-10-22:

- `DocumentReader` now accepts an explicit caller-owned `LayoutReader` for
  layout, OCR, and table extraction. The callback receives raw bytes, the
  stable document ID, and the resolved format; its result is validated as a
  `DocumentLayout` contract before it enters the sample.
- Optional parser absence, invalid extractor results, parser failures, and a
  layout/source SHA-256 mismatch fail closed with structured document errors.
  Existing direct layout injection remains explicit and is not silently
  replaced by a parser default.
- The focused document, model, and modality-lifecycle suite passed 60 tests
  under `-W error::ResourceWarning`; Ruff, formatting, and mypy passed. The
  Astro site passed 17-page build, accessibility, 384-link, and 2-test
  Playwright checks after the API guide was updated.
- This is an optional extraction seam, not a claim that the core package ships
  an OCR, layout, or table parser. Parser licensing, version pinning, raw
  extraction quality, semantic citation grounding, and production-scale
  recovery remain release evidence owned by the caller or deployment.

Current-head release preflight after the layout extraction boundary at
`b38dbbf` on 2026-10-22:

- `./scripts/release_preflight.sh` passed static checks, formatting, mypy over
  239 source files, Bandit, pip-audit, public API compatibility, explicit JUnit
  skip-reason auditing, the bounded suite, mutation testing, benchmarks,
  packaging, reproducible builds, isolated installs, and the Astro site gate.
- The bounded release suite passed `1,998` tests with `75` optional
  dependency skips. Every skip had an explicit reason. Total coverage was
  `83.13%`; the declared core scope was `86%`.
- Mutation testing killed `1,503` of `2,057` mutants (`73%`), above the 70%
  threshold. One mutation timed out and 553 survived for follow-up triage.
- The five-seed offline benchmark regression checked breast cancer, wine,
  digits, and diabetes progression with zero regressions and zero failed
  repetitions. Verdicts were tie, tie, tie, and ahead; peak memory was
  0.3659, 0.3897, 0.4528, and 0.4595 GB respectively, with elapsed seconds
  of 105, 76, 633, and 70.
- Wheel and source-distribution checks passed, artifacts reproduced
  byte-for-byte with wheel hash
  `344a8d8705708cd119f75a141db9bc95f3d6ccbf9b303a2c4078929fa54bdaaf` and
  sdist hash
  `7f5e9e8e79e5448a3c09ff0c977a8c418e8b22879e29802b3142bbd6391edc5f`.
  Both network-isolated installation paths passed.
- The site audit found 0 npm vulnerabilities, built all 17 static pages,
  passed accessibility checks for all 17 HTML files, passed 384 internal-link
  checks, and passed both Chromium Playwright smoke tests. Hosted Linux and
  Pages, external transport/resource-fault execution, secret custody, and
  independent reviewer evidence remain separate release gates.

Extension dataset-family contract at `17a9417` on 2026-10-22:

- `ExtensionModality` gives caller-defined dataset families a stable,
  namespaced identity such as `extension:acme/point-cloud`. The identity is
  accepted by modality declarations, registries, lifecycle adapters, signed
  package manifests, verified inference, and reference serving without
  changing the built-in modality enum.
- A complete extension lifecycle is still declared `extension-only`; the
  declaration rejects any attempt to promote an `ExtensionModality` directly
  to `supported`. This preserves the curation-first and review boundary for
  unfamiliar raw data rather than treating a generic adapter as domain proof.
- The focused contract, project, lifecycle, and serving suite passed 49 tests
  under `-W error::ResourceWarning`; Ruff, formatting, mypy, and diff checks
  passed. The change was pushed as an atomic commit.
- The standalone Astro extension-contract guide is the developer-facing
  onboarding surface for raw fixtures, adapter responsibilities, governance,
  lifecycle evidence, and promotion criteria. It does not redirect readers to
  repository Markdown files.

Extension contract full verification at `18cddfd` on 2026-10-22:

- `uv run pytest -q tests -W error::ResourceWarning` passed `2,324` tests
  with `88` explicit optional-dependency skips in 32 minutes 43 seconds. No
  failure, error, or resource warning was promoted to a failure.
- `./scripts/release_preflight.sh` passed static checks, formatting, mypy over
  239 source files, Bandit, pip-audit, API compatibility, and JUnit skip
  auditing. The bounded release suite passed `1,989` tests with `75`
  explicitly explained skips; total coverage was `83.15%` and the declared
  core scope was `86%`.
- Mutation testing killed `1,503/2,057` mutants (`73%`) against the configured
  `70%` minimum. One mutation timed out and `553` survived; this meets the
  configured gate and the survivor inventory remains follow-up evidence.
- Five-seed offline regression passed for breast cancer, wine, digits, and
  diabetes progression with zero failures and zero regressions. Peak memory
  was `0.3635`, `0.3851`, `0.4400`, and `0.4537` GB respectively.
- Reproducible wheel and source-distribution builds plus network-isolated
  installs passed. The wheel hash was
  `108076828c59b5c123ce64cceddf5724653d10c66af1d624bb2e2bb526e84e84` and the
  source-distribution hash was
  `f7f059e545fd1a9841935349d9dc5b5fce10bbd1f40a0994a81366b88fdf0df1`.
- The updated marketing site dependency audit found zero vulnerabilities;
  Astro built `17` pages, accessibility checks passed for `17` HTML files,
  internal-link validation passed for `384` links, and both Playwright smoke
  tests passed. Hosted Linux and Pages, external transport/resource-fault
  execution, secret custody, and independent reviewer evidence remain
  separate release gates.

Curation-only modality path full verification at `021b13c` on 2026-10-22:

- `ModalityCurationProject` now accepts unlabeled, unsupervised, and empty
  modality snapshots without entering the classification lifecycle. It
  validates source hash, sample count, modality manifest identity, and typed
  curation-report output, then writes an atomic signed curation record and
  dataset card. The modelling path remains strict and still rejects missing
  labels rather than inventing a task.
- `uv run pytest -q tests -W error::ResourceWarning` passed `2,326` tests with
  `88` explicit optional-dependency skips in 32 minutes 32 seconds. No
  failure, error, or resource warning was promoted to a failure.
- `./scripts/release_preflight.sh` passed all local gates. The bounded suite
  passed `1,991` tests with `75` explained skips; total coverage was `83.12%`
  and declared core coverage was `86%`. The public API compatibility check
  passed with `14` exports.
- Mutation testing killed `1,506/2,057` mutants (`73%`) against the `70%`
  minimum. One mutation timed out and `550` survived; the configured gate
  passed and the survivor inventory remains follow-up evidence.
- Five-seed offline benchmarks passed with zero regressions and zero failures.
  Peak memory was `0.3625`, `0.3844`, `0.4286`, and `0.4341` GB for breast
  cancer, wine, digits, and diabetes progression respectively.
- Reproducible artifacts and network-isolated installs passed. The wheel hash
  was `ead66f5d5f6420d03cddff61ec4f4521049f54ca60784fd09549c9ae62161a51` and
  the source-distribution hash was
  `4a6d6e0e2f674b90ebdde20bb7863b019e8ebebe242a2e9d6bb40542388f15d9`.
- The Astro site audit found zero vulnerabilities; it built `17` pages,
  passed accessibility checks for `17` files and `384` links, and passed both
  Playwright smoke tests. Hosted Linux and Pages, external fault execution,
  secret custody, and independent reviewer evidence remain separate gates.

Current-head strict and release preflight evidence at `cd10719` on 2026-10-22:

- `uv run pytest -q tests -W error::ResourceWarning` passed `2,321` tests with
  `88` explicit optional-dependency skips in 31 minutes 36 seconds. No test,
  error, or resource warning was promoted to a failure. This run includes the
  bounded graph subgraph explanation evidence and the complete repository
  test inventory.
- `./scripts/release_preflight.sh` passed the static, formatting, mypy,
  Bandit, pip-audit, API-compatibility, and JUnit skip-audit gates. The bounded
  release suite passed `1,986` tests with `75` explicitly explained skips;
  total coverage was `83.15%` and the declared core scope was `86%`.
- Mutation testing killed `1,504/2,057` mutants (`73%`) against the configured
  `70%` minimum. One mutation timed out and `552` survived; the threshold was
  met and the survivor inventory remains follow-up evidence.
- Five-seed offline regression passed for breast cancer, wine, digits, and
  diabetes progression with zero failures and zero regressions. Verdicts were
  tie, tie, tie, and ahead; peak memory was `0.3657`, `0.3817`, `0.4372`, and
  `0.4440` GB respectively.
- Reproducible artifacts and network-isolated wheel/source installations
  passed. The wheel hash was
  `744172aca87f7329b6a07f05edd7a97d5a8862e4c9066468823691de285ec0ad` and
  the source-distribution hash was
  `6cc4b00195c8ed0218770c0a822731eba2afcb1b96d66d3031b6d19a1fa6e58f`.
- The site dependency audit found zero vulnerabilities; Astro built `16`
  pages, accessibility checks passed for `16` HTML files, internal-link
  validation passed for `346` links, and both Playwright smoke tests passed.
  This remains local evidence; hosted Linux and Pages, external
  transport/resource-fault execution, secret custody, and independent
  reviewer evidence remain separate release gates.

Graph explanation boundary at `30d51df` on 2026-10-22:

- `GraphAdapter.one_hop_subgraph` now emits deterministic, bounded topology
  context with explicit node and edge limits, observed counts, and truncation
  flags. The model explanation includes this subgraph evidence while keeping
  `model_uses_topology: false`; no graph-neural-network or causal attribution
  claim is introduced.
- The graph lifecycle and graph-data focused suite passed 44 tests under
  `-W error::ResourceWarning`; Ruff, formatting, mypy, and `git diff --check`
  passed. This is a focused evidence slice and does not replace the full
  release preflight or the still-required external topology/model evidence.

Current-head strict regression at `0c97ceb` on 2026-10-22:

- `uv run --python 3.14 pytest -q tests -W error::ResourceWarning` passed
  `2,320 passed, 88 skipped` in 31 minutes 20 seconds. No failure, error, or
  warning was promoted to a failure. The run includes CRS-bound raster-window
  evidence and the complete current test inventory.

Current-head release preflight evidence at `0c97ceb` on 2026-10-21:

- `./scripts/release_preflight.sh` passed Ruff, formatting, mypy over 239
  source files, Bandit, pip-audit, public API compatibility, and explicit
  JUnit skip-reason auditing. The bounded release suite passed 1,985 tests
  with 75 optional dependency skips; all skips had explicit reasons. Total
  coverage was 83.15%; the declared core scope was 86%.
- Mutation testing killed 1,504 of 2,057 mutants (73%), above the configured
  70% release threshold. One timeout and 552 survivors remain visible in the
  mutation inventory.
- The five-seed offline benchmark regression completed for breast cancer,
  wine, digits, and diabetes progression with zero regressions and zero
  failed repetitions. Verdicts were tie, tie, tie, and ahead; peak memory
  was 0.3673, 0.3911, 0.4433, and 0.4527 GB respectively.
- Reproducible wheel and source-distribution builds, network-isolated wheel
  installation, and source-distribution rebuild and installation passed.
  The preflight hashes were wheel
  `ba46067f1f4f2844e5aadf05e36847e4f1ac7ca8705e429b66bd9809e35ad9f2` and
  sdist `664680f4ecbc57ab692f02ae4b70d891dfcd126bad53a95614f661c756358069`.
- npm audit found 0 vulnerabilities; Astro built all 16 pages, accessibility
  checks passed for 16 HTML files, internal link checks passed for 346 links,
  and both Playwright smoke tests passed. Hosted Linux and Pages, external
  transport/resource-fault execution, secret custody, and independent
  reviewer evidence remain separate release gates.

Current-head strict regression and release preflight at `af2f2ca` on
2026-10-21:

- `uv run --python 3.14 pytest -q tests -W error::ResourceWarning` passed
  `2,319 passed, 88 skipped` in 31 minutes 16 seconds. The added decoded
  raster explanation lifecycle and metadata-only boundary tests are included
  in this result.
- `./scripts/release_preflight.sh` passed Ruff, formatting, mypy over 239
  source files, Bandit, pip-audit, public API compatibility, and explicit
  JUnit skip-reason auditing. The bounded release suite passed 1,984 tests
  with 75 optional dependency skips; all skips had explicit reasons. Total
  coverage was 83.16%; the declared core scope was 86%.
- Mutation testing killed 1,503 of 2,057 mutants (73%), above the configured
  70% release threshold. One timeout and 553 survivors remain visible in the
  mutation inventory.
- The five-seed offline benchmark regression completed for breast cancer,
  wine, digits, and diabetes progression with zero regressions and zero
  failed repetitions. Verdicts were tie, tie, tie, and ahead; peak memory
  was 0.3680, 0.3852, 0.4415, and 0.4512 GB respectively.
- Reproducible wheel and source-distribution builds, network-isolated wheel
  installation, and source-distribution rebuild and installation passed.
  The preflight hashes were wheel
  `ff2ca81927ca8a09176d5436f14574ebcdf7fba3913b018d5c353a2646d2a849` and
  sdist `33add11a155825acd0f9547a2b6dabf866c464a19e96da43d73deb27364f0c68`.
- npm audit found 0 vulnerabilities; Astro built all 16 pages, accessibility
  checks passed for 16 HTML files, internal link checks passed for 346 links,
  and both Playwright smoke tests passed. Hosted Linux and Pages, external
  transport/resource-fault execution, secret custody, and independent
  reviewer evidence remain separate release gates.

Decoded raster explanation evidence at the current working change on
2026-10-21:

- `RasterAdapter` now reuses one canonical per-band summary calculation for
  training features and bounded four-by-four decoded-pixel region occlusion.
  The explanation preserves the named decoder, decoded dimensions, window
  count, score deltas, and the boundary between decoded pixels and encoded
  payloads.
- Metadata-only rasters return an explicit `not_available` explanation rather
  than implying pixel attribution. The evidence does not claim semantic
  localization, geospatial causality, codec governance, or raw-byte
  explanations. Focused modality lifecycle tests passed 27 tests under
  `-W error::ResourceWarning`; formatting, lint, typing, and full release
  preflight remain to be refreshed after this change.

Current-head strict regression after bounded raster decoding at `bd7ad8c` on
2026-10-21:

- `uv run --python 3.14 pytest -q tests -W error::ResourceWarning` passed
  2,317 tests with 88 explicit skips in 31 minutes 6 seconds. No failure or
  warning was promoted to a failure.

Current-head release preflight evidence at `5a35c27` on 2026-10-21:

- `./scripts/release_preflight.sh` passed static checks, the public API gate,
  the bounded suite, mutation testing, benchmark regression, package checks,
  and the marketing-site checks. The bounded suite passed 1,979 tests with 75
  optional dependency skips; every skip had an explicit reason. Total
  coverage was 83.19%; the declared core scope was 86%.
- Mutation testing killed 1,504 of 2,057 mutants (73%), above the configured
  70% threshold. One mutation timed out and 552 survived; the threshold is
  met, but the survivor inventory remains follow-up work.
- The five-seed offline benchmark regression completed for breast cancer,
  wine, digits, and diabetes progression with zero regressions and zero failed
  repetitions. The verdicts were tie, tie, tie, and ahead; peak memory was
  0.3678, 0.3867, 0.4434, and 0.4549 GB respectively.
- Reproducible wheel and source-distribution builds, network-isolated wheel
  installation, and source-distribution rebuild and installation passed. The
  generated artifacts were byte-identical within the preflight run, with
  wheel hash `eafce738153f69866f136eea027a38f6f0087717872e2f846406bffbe45d45ea`
  and sdist hash
  `dfa0b8b766411f216fc089f4b3e57010825564e566bc2f5a27a111f37c36fd5f`.
- npm audit found 0 vulnerabilities; Astro built all 16 pages, accessibility
  checks passed for 16 HTML files, internal link checks passed for 346 links,
  and both Playwright smoke tests passed. Hosted Linux and Pages, external
  transport/resource-fault execution, secret custody, and independent review
  remain separate release gates.

Bounded raster decoder evidence at `de52b77` on 2026-10-21:

- `RasterSample.decode()` now validates caller-owned decoded arrays for
  shape, finite numeric values, declared dimensions and bands, and a
  per-sample pixel budget. `RasterAdapter` binds a required decoder name and
  budget into dataset identity, emits decoded pixel range and measured nodata
  evidence, and exposes bounded per-band summary features.
- The media and modality lifecycle suites passed 66 tests under
  `-W error::ResourceWarning`; Ruff, formatting, mypy, and diff checks
  passed. Production codec governance, reprojection, window sampling,
  spatial joins, raw-pixel explanations, external faults, and independent
  review remain incomplete.

Current-head release preflight after bounded raster decoding at `7c1f88d`
on 2026-10-21:

- `./scripts/release_preflight.sh` passed static checks, the public API gate,
  the bounded suite, mutation testing, benchmark regression, package checks,
  and marketing-site checks. The bounded suite passed 1,982 tests with 75
  optional dependency skips; every skip had an explicit reason. Total
  coverage was 83.17%; the declared core scope was 86%.
- Mutation testing killed 1,504 of 2,057 mutants (73%), above the configured
  70% threshold. One mutation timed out and 552 survived.
- The five-seed offline benchmark regression completed for breast cancer,
  wine, digits, and diabetes progression with zero regressions and zero failed
  repetitions. The verdicts were tie, tie, tie, and ahead; peak memory was
  0.3673, 0.3907, 0.4441, and 0.4538 GB respectively.
- Reproducible wheel and source-distribution builds, network-isolated wheel
  installation, and source-distribution rebuild and installation passed. The
  preflight hashes were wheel
  `e018f0c85a04d7232971bdc53900a7ee6d9cfb8ce308dfb0ab3e1d83c76cfeee` and
  sdist
  `83ab342a80acf77090c9b060bb4e0664cadc5e37fd54f605958d54101be06110`.
- npm audit found 0 vulnerabilities; Astro built all 16 pages, accessibility
  checks passed for 16 HTML files, internal link checks passed for 346 links,
  and both Playwright smoke tests passed. Hosted Linux and Pages, external
  transport/resource-fault execution, secret custody, and independent review
  remain separate release gates.

Raster capability identity at `dc6ef61` on 2026-10-21:

- Raster is now a distinct value in the modality, capability, and dataset
  contracts. `RasterAdapter` emits `raster` instead of masquerading as vector
  geospatial data, and the site capability matrix documents the boundary as
  metadata-only and experimental.
- The modality contract, lifecycle, and project tests passed 39 tests under
  `-W error::ResourceWarning`; Ruff, formatting, mypy, and `git diff --check`
  passed.

Current-head strict regression at `d1a3294` on 2026-10-21:

- `uv run --python 3.14 pytest -q tests -W error::ResourceWarning` passed
  2,314 tests with 88 explicit skips in 31 minutes 9 seconds. No failure or
  warning was promoted to a failure. The API compatibility fixture now records
  the public `MLProject` signature after the raster modality contract change.

Public modality inference boundary at `ce9f747` on 2026-10-21:

- `ModalityPredictor` is now exported from the package root, so callers can
  load a verified modality package without importing an internal lifecycle
  module. The public API snapshot, API-surface contract, project integration
  test, and API documentation all cover the new boundary.
- The focused API, compatibility, and modality-project slice passed 12 tests
  under `-W error::ResourceWarning`; Ruff, formatting, mypy, and the API
  compatibility check passed. The site built 16 pages, passed accessibility
  checks, and passed 346 internal-link checks.
- This closes the local public-entry-point gap. Modality support remains
  experimental until modality-specific fidelity, external serving and fault
  evidence, secret custody, hosted CI/Pages, and independent review gates are
  complete.

Public modality serving boundary at `06775ea` on 2026-10-21:

- `ModalityHttpServing` and `ModalitySampleDecoder` are now exported from the
  package root alongside `ModalityPredictor`. The documented reference HTTP
  path no longer requires callers to import an internal module; decoder,
  transport, authentication, and TLS ownership remain explicit.
- The focused API, compatibility, serving, and recovery slice passed 11 tests
  under `-W error::ResourceWarning`; Ruff, formatting, mypy, and the API
  compatibility check passed. The site built 16 pages, passed accessibility
  checks, and passed 346 internal-link checks.
- This makes the experimental serving contract publicly reachable and tested.
  It does not claim production service readiness: authentication, TLS,
  gateway policy, external resource-fault rehearsal, paging, and independent
  review remain release gates.

Current-head strict regression at `23c3975` on 2026-10-21:

- `uv run --python 3.14 pytest -q tests -W error::ResourceWarning` passed
  `2,311` tests with `88` explicit optional-dependency skips in 31 minutes 22
  seconds. No failure, error, or resource warning was promoted to a failure.
- The run includes the public modality predictor and serving exports, API
  compatibility snapshot, reference HTTP serving contract, all modality
  lifecycle tests, reliability contracts, naming-convention guard, and the
  complete repository test inventory.
- This is current local regression evidence. It does not replace bounded
  coverage and mutation gates, hosted Linux and Pages verification, external
  transport/resource-fault execution, secret custody, or independent review.

Current release preflight at `c81a2b3` on 2026-10-21:

- `./scripts/release_preflight.sh` passed static checks, formatting, mypy over
  239 source files, Bandit, pip-audit, API compatibility, and explicit skip
  auditing. The bounded contract suite passed `1,976` tests with `75`
  optional-dependency skips, all with recorded reasons; total coverage was
  83.18% and the declared core scope was 86%.
- Mutation testing killed `1,503/2,057` mutants (73%), above the configured
  70% gate; one mutation timed out and 553 survived for follow-up triage.
- The five-seed offline benchmark regression checked breast cancer, wine,
  digits, and diabetes progression with zero failures and zero regressions.
  Verdicts were tie, tie, tie, and ahead; peak memory was 0.3612, 0.3840,
  0.4429, and 0.4487 GB respectively, with 104, 76, 635, and 69 seconds.
- Wheel and source-distribution checks passed, builds were reproducible with
  wheel hash `5b480f812efb9ae9388f2af9f25eb4503898a71996700898915196a3b74937b`
  and sdist hash
  `bdd52fb079c047fffadc5b0b2a1ae12efe1e4c616dbf78a3372fa3dba10fef02`, and
  both network-isolated installation paths passed.
- The complete marketing-site gate passed: npm audit found 0 vulnerabilities,
  Astro built 16 pages, accessibility and 346 internal-link checks passed, and
  both Playwright browser smoke tests passed. This remains local evidence;
  hosted Linux and Pages, external fault execution, secret custody, and
  independent review remain separate release gates.

Raw object snapshot evidence at `92a6fa3` on 2026-10-21:

- `ObjectStoreConnector` now emits the observed raw payload SHA-256 on each
  bounded `RowBatch` and carries that digest in the resume checkpoint. A
  resumed read with changed object bytes fails closed with the structured
  `source_snapshot_changed` connector error instead of mixing rows from two
  snapshots.
- The focused connector and checkpoint integration slice passed 29 tests under
  `-W error::ResourceWarning`; Ruff, formatting, mypy, and `git diff --check`
  passed. The refreshed site built 16 pages, passed accessibility checks, and
  passed 346 internal-link checks.
- The digest boundary applies to bounded object reads. Range-backed streams,
  warehouse callbacks, broker transport semantics, and external snapshot
  authorities remain explicitly transport-owned.

Local file snapshot evidence at `6f5b520` on 2026-10-21:

- `LocalFileConnector` now hashes the raw file without loading the entire file
  into memory, binds that digest into each returned checkpoint, and compares
  the digest before and after parsing. A changed file during a read or before a
  resumed batch fails closed with `source_snapshot_changed`.
- The focused connector and checkpoint integration slice passed 30 tests under
  `-W error::ResourceWarning`; Ruff, formatting, mypy, and `git diff --check`
  passed. The site contract remained green with 16 pages, accessibility checks,
  and 346 internal links passing.
- This closes local-file snapshot consistency. Range-backed streams,
  warehouse callbacks, broker transport semantics, and external snapshot
  authorities remain explicitly transport-owned.

Current-head strict regression at `6f5b520` on 2026-10-21:

- `uv run --python 3.14 pytest -q tests -W error::ResourceWarning` passed
  `2,313` tests with `88` explicit optional-dependency skips in 31 minutes 23
  seconds. No failure, error, or resource warning was promoted to a failure.
- This run includes local-file and object snapshot binding, changed-snapshot
  rejection, the complete connector/checkpoint inventory, all modality
  lifecycles, API compatibility, reliability contracts, and the full
  repository test suite.
- This is current local regression evidence. It does not replace the bounded
  coverage and mutation gates, hosted Linux and Pages verification, external
  transport/resource-fault execution, secret custody, or independent review.

Current-head strict regression at `c71af80` on 2026-10-21:

- `uv run --python 3.14 pytest -q tests -W error::ResourceWarning` passed
  `2,312` tests with `88` explicit optional-dependency skips in 31 minutes 19
  seconds. No failure, error, or resource warning was promoted to a failure.
- This run includes the raw-object digest/checkpoint contract, changed-snapshot
  rejection, the full connector and checkpoint inventory, all modality
  lifecycles, API compatibility, reliability contracts, and the complete
  repository test suite.
- This is current local regression evidence. It does not replace the bounded
  coverage and mutation gates, hosted Linux and Pages verification, external
  transport/resource-fault execution, secret custody, or independent review.

Post-hardening strict regression evidence at `2f1e63a` on 2026-10-21:

- `uv run --python 3.14 pytest -q tests -W error::ResourceWarning` passed
  `2,311` tests with `88` explicit optional-dependency skips in 31 minutes 06
  seconds. No failure, error, or resource warning was promoted to a failure.
- This rerun includes both constant-time signature-verification hardenings in
  the independent-review and release-attestation paths, plus the complete
  repository test inventory. The optional skips remain explicit dependency
  boundaries rather than unexplained failures.
- This is current local evidence and does not replace hosted CI/Pages,
  external fault execution, secret custody, or independent review.

Current release preflight evidence at `c2504b9` on 2026-10-21:

- `./scripts/release_preflight.sh` passed all local gates: Ruff, formatting,
  mypy over 239 source files, Bandit, pip-audit, public API compatibility,
  and explicit JUnit skip-reason auditing. The bounded contract suite passed
  `1,976` tests with `75` explicit optional-dependency skips; total coverage
  was 83.17% and the declared core scope was 86%.
- Mutation testing killed `1,503/2,057` mutants (73%), above the 70% gate;
  one mutation timed out and 553 survived for follow-up triage.
- The five-seed offline benchmark regression completed for breast cancer,
  wine, digits, and diabetes progression with zero regressions and zero
  failed repetitions. Verdicts were tie, tie, tie, and ahead; peak memory
  was 0.3611, 0.3847, 0.4481, and 0.4547 GB respectively.
- Wheel and source-distribution checks passed, builds were reproducible, and
  network-isolated wheel and source-distribution installs passed. The
  reproducible wheel hash was
  `a35a6ee0ee98a730df562ee7f35f4541da8668f13648849653d657026640f176` and
  the source-distribution hash was
  `26f3d5f7fb890b7b62868df26f04a538669ccdb8493b554f085b95c2ac6304ba`.
- The marketing-site dependency audit found 0 vulnerabilities and Astro
  built all 16 static pages. This remains local evidence; hosted Linux and
  Pages, external fault execution, secret custody, and independent reviewer
  evidence remain separate release gates.

Release-site gate enforcement at `5a8d5c9` on 2026-10-21:

- `scripts/release_preflight.sh` now runs the complete local site contract:
  dependency audit, Astro build, accessibility checks, internal-link checks,
  Chromium provisioning, and Playwright smoke tests. This closes the gap where
  the release command previously checked only build and dependency health.
- The updated site portion passed with 16 built pages, 346 internal links,
  accessibility checks, and 2 browser smoke tests. The full Python preflight
  evidence immediately before this script-only change remains recorded at
  `c2504b9` above.

Current-head strict regression evidence at `fad7703` on 2026-10-21:

- `uv run --python 3.14 pytest -q tests -W error::ResourceWarning` passed
  `2,311` tests with `88` explicit optional-dependency skips in 29 minutes 32
  seconds. No failure, error, or resource warning was promoted to a failure.
- This run includes the public `ModalityProject` entry point, its API snapshot
  and focused tests, all current modality lifecycle contracts, the naming
  convention guard, and the complete repository test inventory. The skips are
  explicit optional-dependency boundaries and do not count as unexplained
  failures.
- This is current-head local regression evidence. It does not replace the
  bounded coverage and mutation gates, hosted Linux and Pages verification,
  external transport/resource-fault execution, secret custody, or independent
  reviewer evidence required by the release criteria.

Current external gate observation at `b05519a` on 2026-10-21:

- GitHub Actions created the `ci` matrix and `Deploy documentation site` runs
  for the pushed commit, but every job was rejected before a runner step. The
  GitHub check annotations state that recent account payments failed or the
  spending limit must be increased. These are unavailable hosted-runner
  results, not source-test failures.
- The requested public URL `https://sachncs.github.io/super-charge-ml/`
  returned HTTP 404. The Pages deployment therefore remains unverified until
  the account/plan and repository visibility or Pages configuration are
  resolved by the owner.

Security hardening at `43bf72e` on 2026-10-21:

- Independent release-review HMAC verification now uses constant-time
  comparison, matching the bundle, registry, evidence-index, and keyring
  verification paths. The review contract and public CLI behavior are
  unchanged.
- The independent-review, evidence-contract, and keyring tests passed 18
  tests under `-W error::ResourceWarning`; Ruff, formatting, and mypy passed.

Release-attestation hardening at `6532447` on 2026-10-21:

- Release-attestation verification now compares the non-secret binding
  separately and uses constant-time comparison for the HMAC value. Missing,
  unsigned, wrong-key, and tampered attestations remain fail-closed.
- The attestation, release-evaluation, and problem-spec integration tests
  passed 23 tests under `-W error::ResourceWarning`; Ruff, formatting, and
  mypy passed.

Current-head release preflight at `8ee7c87` on 2026-10-22:

- Static analysis, formatting, mypy, security, API compatibility, the bounded
  suite, three-run stability, mutation, four-dataset benchmark regression,
  reproducible package builds, and network-isolated wheel/source installs
  passed. The bounded suite passed `2,039` tests with `75` explicitly
  explained optional-dependency skips; total coverage was `83.17%`, and the
  declared core scope was `86%`. Mutation testing killed `1,504/2,057`
  mutants (`73%`); the benchmark gate recorded zero regressions and zero
  failed repetitions.
- The site gate did not complete because `npm audit` timed out against the
  npm security-advisory endpoint. This is an external network failure, not a
  reported vulnerability or Astro build failure; the audit/build/accessibility
  and browser checks require a network-available rerun before this head has a
  complete local preflight record.

Independent explanation reference contract at `735d647` on 2026-10-22:

- The explainability contract now accepts an adapter-owned, independently
  authored reference ranking and evaluates transparent top-k overlap against
  the persisted local explanation. A supplied reference is recorded as
  passed or failed evidence; it cannot silently become a semantic-validity
  claim.
- Every built-in modality adapter now emits an explicit
  `independent_ground_truth.status = not_available` result until a caller
  provides an annotation, benchmark, or synthetic oracle. This makes the
  remaining raw-modality validation gap visible in every lifecycle artifact.
- The focused explanation and modality lifecycle slice passed `45` tests
  with strict resource warnings; Ruff, formatting, mypy, and `git diff
  --check` passed. Production modality annotations, independent human or
  benchmark adjudication, and raw-modality causal validation remain open.

Independent explanation reference suites at `bdb9b25` on 2026-10-22:

- The lifecycle now evaluates an adapter-owned reference across every held-out
  sample within the declared 128-sample explanation bound. A complete suite
  reports per-sample overlap, minimum and mean scores, and a passing result;
  missing or partial references fail closed as `not_available` or `partial`.
- A synthetic adapter integration test proves the complete-reference path,
  while built-in adapters continue to report unavailable until their domain
  annotations or controlled oracles are supplied. The focused slice passed
  `47` tests under strict resource warnings; Ruff, formatting, mypy, and
  `git diff --check` passed. Production modality annotations and independent
  human or benchmark adjudication remain open.

Production explanation-evidence gate at `828e8c5` on 2026-10-22:

- `ModalityProject` and `ModalityLifecycle` now expose an explicit
  `require_independent_ground_truth=True` release setting. When enabled, a
  missing, partial, or failed held-out reference suite stops before package
  or lifecycle-report publication, so an experimental adapter cannot be
  mistaken for a production-evidenced release.
- The public project, lifecycle, and explanation-quality slice passed `56`
  tests under strict resource warnings; Ruff, formatting, mypy, and
  `git diff --check` passed. Built-in modality adapters intentionally remain
  experimental until they supply domain annotations or controlled oracles.

Current-head release preflight evidence at `edf9d39` on 2026-10-21:

- `./scripts/release_preflight.sh` passed all local gates: Ruff, formatting,
  mypy over 239 source files, Bandit, pip-audit, public API compatibility,
  and explicit JUnit skip-reason auditing. The bounded release suite passed
  1,971 tests with 75 optional dependency skips; all 75 skips had explicit
  reasons. Total coverage was 83.15%; the declared core scope was 86%.
- Mutation testing killed 1,495 of 2,049 mutants (72%), above the 70% release
  threshold. One mutation timed out and 553 survived; the threshold is met,
  but the surviving-mutant inventory remains useful follow-up evidence.
- The five-seed offline benchmark regression completed for breast cancer,
  wine, digits, and diabetes progression with zero regressions and zero failed
  repetitions. The verdicts were tie, tie, tie, and ahead; peak memory was
  0.3729, 0.3754, 0.4277, and 0.4313 GB respectively. This is benchmark
  evidence, not a claim of universal model superiority.
- Wheel and source-distribution checks passed, artifacts reproduced
  byte-for-byte with wheel hash
  `f586cf667580b996b5d2abea7b9ed7959cf6f9d36d8a0872b9e384b92084da9f` and
  sdist hash
  `9612e6910c46d44c445467f6df904c7d632201db4d0521d24d75456c4379df22`, and
  both network-isolated installation paths passed.
- The production site dependency audit found 0 vulnerabilities and Astro
  built all 16 static pages. This remains local evidence; hosted Linux and
  Pages, external transport/resource-fault execution, secret custody, and
  independent reviewer evidence remain separate release gates.
# Package-boundary hardening at `f8cfea7` on 2026-10-23

- The modality package loader now validates the signed manifest schema,
  modality and dataset identity, non-empty unique feature names, canonical
  model digest, evaluation object, and estimator/feature-shape agreement
  before deserializing the joblib payload. `LifecyclePackage` applies the
  same digest, signature, path, and schema checks to in-memory evidence.
- The focused modality lifecycle and serving slice passed `43` tests under
  `-W error::ResourceWarning`; Ruff, formatting, mypy, and `git diff --check`
  passed. Regression coverage includes unsupported manifest versions,
  duplicate feature names, and a correctly re-signed manifest whose feature
  schema disagrees with the serialized estimator.
- This closes a local package-structure and pre-deserialization validation
  gap. It does not make non-tabular lifecycles production-supported, provide
  external fault evidence, establish hosted CI or Pages deployment, or
  replace independent release review and secret custody.

## Curation evidence verification at `d74e04c` on 2026-10-23

- `ModalityCurationResult` now has a versioned read/parse contract,
  dataset/card identity checks, canonical JSON normalization, and
  `read_verified()` signature verification. The curation artifact now records
  its path and uses the same signature envelope as the in-memory result,
  making the persisted data-understanding evidence independently consumable.
- The result is available from the documented top-level `scml` API and is
  covered by the versioned public API snapshot. The focused API, curation,
  lifecycle, and package slice passed `59` tests under
  `-W error::ResourceWarning`; Ruff, formatting, mypy, and
  `git diff --check` passed.
- This closes the local curation-artifact verification gap. It does not
  claim that curation establishes consent, ownership, retention, semantic
  quality, or production support; those remain explicit manifest and review
  responsibilities.

## Lifecycle-report completeness at `4cf3997` on 2026-10-23

- `LifecycleReport` now validates its package identity, JSON-native evidence
  sections, and exact ordered coverage of every required lifecycle stage. A
  report with a valid signature but missing, duplicated, or reordered stage
  evidence is rejected during parsing before it can be used as release
  evidence.
- The focused lifecycle, curation, API, and compatibility slice passed `60`
  tests under `-W error::ResourceWarning`; Ruff, formatting, mypy, and
  `git diff --check` passed. Regression coverage includes a correctly
  re-signed report with an incomplete stage sequence.
- This closes a local report-structure gap. It does not establish that the
  evidence itself is scientifically sufficient, independent, externally
  reproduced, or production support for non-tabular modalities.

## Lifecycle evidence-link enforcement at `c652ef1` on 2026-10-23

- A lifecycle report now rejects stages that have a valid stage name but no
  evidence link. This prevents a signed record from presenting a complete
  stage list while omitting the artifact or test reference that makes the
  stage reviewable.
- The focused lifecycle, curation, API, and compatibility slice passed `61`
  tests under `-W error::ResourceWarning`; Ruff, formatting, mypy, and
  `git diff --check` passed.
- This strengthens evidence completeness at the local report boundary. It
  does not validate the scientific adequacy of the linked artifact or provide
  independent reviewer, hosted, cloud, broker, container, or kernel evidence.

## Tabular JEPA checkpoint persistence at `8dd09df` on 2026-10-23

- The tabular JEPA contract now has an atomic checkpoint writer and a
  versioned loader for `scml.tabular-jepa.v1`. The writer flushes and syncs a
  temporary file before replacement, and the loader rejects an unknown
  checkpoint format before reconstruction.
- Deterministic same-seed round trips and invalid-format rejection passed in
  `5` focused tests with the deep-learning extra enabled; Ruff, formatting,
  mypy, and `git diff --check` passed.
- This is a reliable persistence boundary for the existing tabular JEPA
  contract. It is not evidence for image, audio, video, graph, geospatial,
  multimodal, or distributed JEPA production support; torch-enabled training
  and end-to-end functional testing remain open.

## Project parameter-estimation evidence at `596764b` on 2026-10-23

- `MLProject` now accepts an explicit `EstimandSpec` and
  `EstimationPolicy`, runs parameter estimation as a separate scientific
  track, persists the signed `parameter_estimation.json` artifact inside the
  run, and exposes the verified result on `RunResult.estimation`. Partial
  declarations fail at configuration time; missing columns, unsupported
  modalities, and incompatible likelihoods fail closed before model
  selection.
- The project integration, estimator, API surface, and compatibility slice
  passed `17` tests under `-W error::ResourceWarning`; Ruff, formatting,
  mypy, and `git diff --check` passed.
- This establishes a reproducible linear/logistic frequentist evidence path;
  causal, Bayesian, hierarchical, modality-specific, and resume/replay
  estimation contracts remain open and are not implied by this API.

## Lifecycle-bound modality serving at `4cd16ce` on 2026-10-23

- `ModalityPredictor.from_package()` can now require and verify the signed
  lifecycle report for the exact modality, dataset identity, package path,
  model digest, and package signature. `ModalityHttpServing` exposes an
  explicit `require_lifecycle_report=True` release-bound mode and publishes
  the verified report digest in health and prediction evidence.
- Package-only predictors are rejected by release-bound serving, and tampered
  lifecycle evidence is rejected before serving starts. The focused modality,
  API, and compatibility slice passed `67` tests under
  `-W error::ResourceWarning`; Ruff, formatting, mypy, and
  `git diff --check` passed.
- This closes the local package-to-serving evidence gap. It does not make
  OCR/layout, image, audio/video, graph, geospatial, or other non-tabular
  modalities production-supported, and it does not substitute for hosted,
  cloud, broker, container, kernel, or independent-review evidence.

## Scientific estimator expansion at `16bb2f9` on 2026-10-23

- The parameter-estimation contract now includes explicit Bayesian Gaussian,
  binary-treatment Hájek IPW ATE, and grouped random-intercept estimators in
  addition to linear and logistic estimation. Each new path requires its
  prior or identification assumptions, records method-specific diagnostics,
  and runs an independent reference calculation before its report is valid.
- Bayesian evidence includes posterior covariance, credible intervals, and
  prior-strength sensitivity. IPW evidence includes treatment counts,
  propensity clipping and overlap diagnostics, effective sample sizes, and an
  influence-function variance. Hierarchical evidence includes moment-based
  variance components, ICC, cluster sizes, and the explicit fact that
  variance-component uncertainty is not propagated.
- The focused estimator suite passed `15` tests during implementation review;
  syntax compilation, Ruff, formatting, mypy, and `git diff --check` passed.
  AIPW/cross-fitting, time-varying causal effects, REML/ML, random slopes,
  nested effects, and fully Bayesian multilevel inference remain unsupported.

## Durable local streaming boundary at `b063658` on 2026-10-23

- Chunked CSV, JSONL, and Parquet sources now expose snapshot-bound
  checkpoints, watermark policy, bounded batches, deterministic batch
  idempotency keys, an atomic local ledger, and durable acknowledgement retry
  state. The sink receives the batch key so a caller-owned sink can deduplicate
  retries.
- Source changes, malformed state, mismatched checkpoints, mismatched
  consumer groups, invalid watermark values, and acknowledgement failures
  fail closed. The transport seam remains explicitly at-least-once: broker
  leases, partition ownership, transactions, and distributed exactly-once
  semantics remain the external transport owner's responsibility.
- The focused streaming contract passed `11` tests during implementation
  review; Ruff, formatting, mypy, syntax compilation, and `git diff --check`
  passed.

## Capability registry hardening at `d59d53a` on 2026-10-23

- The capability registry now serializes deterministic versioned evidence,
  catalogs transformer-family boundaries, and declares JEPA as experimental
  and tabular-only. A complete capability still requires all eight dimensions
  rather than a model-family name alone.
- Pre-training diagnostics now report missing declarations, disallowed status,
  missing extras, unknown or insufficient CPU/memory/disk/GPU/network facts,
  and row-limit violations without starting model work. This keeps partial
  transformer and multimodal paths from being represented as production
  support.
- The focused capability suite passed `21` tests during implementation
  review; Ruff, mypy, and `git diff --check` passed. Runtime facts remain
  caller-supplied and no transformer or JEPA family is promoted by the catalog.

## Public contract exposure at `811d7b7` on 2026-10-23

- `BayesianPrior` is now available from the top-level `scml` API, and the
  durable streaming contracts are available from `scml.data.readers`.
  The versioned public API fixture records the intentional estimator change.
- The public surface remains honest about optional extras and experimental
  status; exposing a contract does not promote it to supported production
  status.

## Full regression correction at `9d7b993` on 2026-10-23

- The first post-implementation full repository run completed
  `2490 passed, 38 skipped` in `42m20s` and exposed one convention-only
  failure in the new IPW reference calculation: two discard-variable names
  violated the repository's no-leading-private-name rule.
- `9d7b993` replaces those names with explicit evidence variables and keeps
  the reference calculation unchanged. The focused convention, estimator,
  project-integration, API-compatibility, and API-surface slice then passed
  `893` tests; Ruff and formatting passed.
- The full-suite gate must be rerun after this correction before release
  approval. The prior `2490/38` result is not recorded as a passing full
  suite because it predates this fix.

