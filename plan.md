# Production-readiness task: the deeper version

This is the task definition for making `super-charge-ml` production-ready.
It deliberately narrows the question before expanding the implementation:
production-ready means that every advertised path has a bounded contract,
reproducible evidence, safe failure behaviour, and an independent release
decision. It does **not** mean claiming support for every modality or model
family at once.

## What a top professional would do that this request does not yet ask for

They would first define the product boundary and the failure cost. Before
adding another estimator or model, they would identify the supported user
journeys, the data owner, the decision being automated, the unacceptable
failure modes, the required latency/cost envelope, and the explicit reasons a
run must be blocked. They would turn those decisions into a versioned support
matrix rather than treating a working class or passing test as a production
claim.

They would also establish independent evidence paths. The implementation,
the test author, and the release approver should not be the only three roles
that decide whether a result is correct. A professional release process would
include a threat model, a data-provenance review, a statistical reference
implementation, a clean-environment reproduction, failure-injection
rehearsals, performance budgets, rollback criteria, security and privacy
review, and a written decision record for every unsupported capability.

The missing professional questions are therefore:

1. What is the smallest supported product surface that can be defended?
2. What evidence would falsify each capability, not merely demonstrate its
   happy path?
3. Which claims require a domain expert, statistician, security reviewer, or
   operator rather than another software test?
4. What is the release, rollback, incident, and deprecation policy after the
   library leaves the developer workstation?
5. Can a new person reproduce the result from raw inputs and a clean checkout
   without relying on undocumented local state?

## The rawest material to start from

Do not begin with the existing summaries, cards, benchmark tables, or
documentation. Begin with an immutable evidence bundle:

- the original source bytes or source-system extracts, including event
  timestamps, source versions, schema history, and access policy;
- the label/annotation instructions, annotator identities or pseudonyms,
  disagreement records, adjudication decisions, and known measurement error;
- the data dictionary and business meaning of every field, including units,
  missing-value semantics, censoring, delayed availability, and legal/privacy
  constraints;
- the exact train/validation/test or temporal/group split inputs and the
  reason each row was eligible for each split;
- the raw run configuration, random seeds, environment lockfile, interpreter,
  hardware, dependency and container digests, source commit, and command
  line;
- the untouched predictions, outcomes, weights, residuals, calibration
  records, uncertainty samples, error slices, and resource traces;
- real deployment traces: request distributions, latency, memory, retries,
  duplicate events, incidents, rollbacks, drift, and operator actions;
- independent reference outputs from a second implementation or a trusted
  statistical package; and
- the release artifacts themselves: wheel, sdist, SBOM, signatures, attestations,
  review decision, and clean-checkout reproduction log.

The first deliverable from this material is a provenance-bound dataset and
run manifest. If the raw source, label process, or environment cannot be
recovered, the corresponding capability remains `unverified`; it is not
silently upgraded by a polished report.

## Specific methods and tools

Use the smallest tool that produces auditable evidence:

| Question | Method | Tools / evidence |
| --- | --- | --- |
| Can the data be understood and reproduced? | Data contract, schema/version checks, duplicate and leakage checks, missingness and label audits, slice and shift analysis | `DatasetCatalog`, Pandera-style contracts, DuckDB/Polars probes, raw SHA-256 manifests, OpenLineage-compatible lineage, immutable dataset cards |
| Is the estimator correct? | Estimand-first specification, identifiability review, simulation with known truth, independent reference calculation, sensitivity and uncertainty analysis | NumPy/SciPy/statsmodels reference calculations, bootstrap or sandwich covariance, property tests, seeded simulation fixtures |
| Is model behaviour correct? | Nested or temporal evaluation, calibration, subgroup/slice analysis, error taxonomy, explanation fidelity and stability | scikit-learn baselines, Torch functional tests where applicable, held-out data, permutation/occlusion checks, model cards |
| Does the package survive change? | Unit, integration, contract, property, mutation, API-compatibility, and clean-clone tests | pytest, Hypothesis, Ruff, mypy, coverage, mutation testing, API snapshots, `uv` lockfile, reproducible wheel/sdist install |
| Does it fail safely? | Fault-injection matrix across process, disk, memory, timeout, retry, duplicate, checkpoint, transport, container, and kernel boundaries | Local fault harness, subprocess limits, Toxiproxy or an injected transport, Docker/OCI tests, cgroups, Kubernetes job probes, signed recovery evidence |
| Can it be operated? | SLOs, error budgets, redacted observability, alert routing, incident rehearsal, rollback and key rotation | OpenTelemetry, Prometheus/Grafana-compatible metrics, structured audit logs, runbooks, release attestations, signed approvals |
| Can it be released publicly? | Hosted CI matrix, Pages smoke checks, artifact digest verification, clean external reproduction, independent review | GitHub Actions with pinned actions, GitHub Pages route checks, public release/PyPI digest checks, a fresh machine/container, external reviewer signature |

The tools are not the acceptance criteria. Each tool must emit a durable,
versioned artifact that says what was measured, on which input, under which
assumptions, with what limitations, and whether the gate passed.

## What other than you or me would show the result is right

Correctness must be triangulated:

- a deterministic independent reference implementation agrees within a
  declared tolerance;
- simulation tests recover known parameters and expose bias, coverage, and
  failure behaviour under violated assumptions;
- a clean checkout on a fresh environment reproduces the same digests,
  metrics, explanations, and release decision;
- hosted CI runs the declared Python, OS, dependency, and optional-integration
  matrix on the exact commit;
- a fault harness and an operator rehearse interruption, retry, duplicate,
  rollback, and recovery paths with observable safe outcomes;
- a statistician or domain expert validates estimands, labels, uncertainty,
  and interpretation; a security/privacy reviewer validates threat and data
  handling controls; and an operator validates runbooks and alerts;
- a user or downstream integration exercises the public API from the package,
  not from the repository checkout; and
- a named independent release reviewer signs the evidence bundle and records
  unresolved limitations.

If these witnesses disagree, the release is blocked and the disagreement is
preserved as evidence. A passing internal test suite alone is not sufficient.

## Rewritten task

Audit and bring `super-charge-ml` to a defensible production-beta release,
using `docs/PRODUCTION_BETA_EXECUTION_PLAN.md` as the source of truth.

1. Establish a versioned capability matrix. For every data modality, model
   family, estimator, connector, serving path, and explanation path, label it
   `implemented`, `tested`, `supported`, `production-proven`, or
   `extension-only`. Fail closed when a claim has no matching evidence. Keep
   multimodal Transformer/JEPA, OCR/layout, image/audio/video/graph/geospatial
   foundation models, broker exactly-once semantics, and hosted infrastructure
   explicitly unsupported until their full evidence exists.
2. Reconstruct provenance from raw source bytes, labels, annotations, split
   inputs, event logs, environment locks, and deployment traces. Produce
   digest-bound dataset, lineage, run, model, explanation, SBOM, and release
   manifests. Do not treat existing summaries as source evidence.
3. Complete data curation before modelling: schema and semantic contracts,
   source identity, missingness, duplicates, label quality, leakage, group and
   temporal boundaries, privacy findings, shift/representativeness slices,
   modality-specific decoding, and explicit blocking findings. Support new
   datasets through a documented adapter contract without claiming that
   unknown formats are automatically understood.
4. Complete only the modelling paths that can be supported end to end. Each
   path must have training, checkpoint integrity, verified load, inference,
   evaluation, calibration/uncertainty, explanation, drift, resource limits,
   rollback, and documented limitations. Transformer and JEPA paths must have
   real functional tests in an available Torch environment; multimodal paths
   must fail closed if their representation, decoder, or evidence contract is
   absent.
5. Complete parameter estimation as a first-class product surface. Every
   estimand must declare likelihood, identifiability, assumptions, sampling
   unit, missingness policy, sensitivity analysis, estimate, covariance,
   confidence/credible interval, effective sample size, and an independent
   reference calculation. Cover the supported binary, Bayesian, hierarchical,
   longitudinal, continuous-treatment, and modality-specific contracts; do
   not confuse predictive model selection with parameter estimation.
6. Prove engineering reliability at the declared boundary. Test atomic
   publication, resume, idempotency, retries, duplicate events, schema
   evolution, signing-key rotation, resource exhaustion, timeout, rollback,
   and observability. Add actual transport/container/kernel evidence where
   claimed; otherwise mark the gate open. Do not infer distributed exactly-once
   semantics from a local idempotency ledger.
7. Run the release evidence loop once implementation is complete: focused
   tests and coverage, static checks, mutation/property/contract checks,
   benchmark and performance gates, hosted CI, package install from wheel and
   sdist, site and Pages route checks, clean-clone reproduction, and the full
   regression suite. Record exact commands, commit, environment, counts,
   skips, warnings, failures, and artifacts in the execution plan. Do not
   repeatedly run the full suite between implementation batches.
8. Obtain independent review. A reviewer who did not implement the change
   must recompute key metrics and estimates, inspect raw provenance and
   limitations, exercise failure paths, verify artifact digests, and sign the
   release decision. A clean external reproduction must succeed without local
   untracked state.
9. Use atomic commits grouped by one concern, following the Google Python
   style guide, with no semi-private names, no compatibility shims, and no
   thin wrappers. Every commit must have focused evidence; the final plan
   commit must record the release decision and all unresolved gates.

### Acceptance criteria

The task is complete only when all of the following are true:

- the capability matrix contains no unsupported claim presented as supported;
- every supported path has a public contract, tests, documentation, limits,
  provenance, and a reproducible artifact;
- focused, static, API, coverage, benchmark, reliability, package, site, and
  full-regression evidence is green or explicitly classified;
- all skipped tests and warnings have an owner, reason, expiry or release
  decision;
- the exact release commit passes hosted CI and clean external reproduction;
- the public site and standalone developer docs resolve at the configured
  deployment path;
- independent technical, statistical/domain, security/privacy, and operational
  review records are signed; and
- the release decision states either `production-beta approved` or `blocked`,
  with the remaining risks and next owner named. No “production-ready” label
  is allowed while a required gate is merely planned or locally inferred.
