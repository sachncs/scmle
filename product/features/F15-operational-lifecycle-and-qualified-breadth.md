# F15: Operational data science lifecycle and qualified breadth

Vocabulary and principles: [`../../plan/00-canon.md`](../../plan/00-canon.md).
Built in [Phase 18](../../plan/phase-18-lifecycle-evidence-and-breadth.md).
Starts after the F14 items have passed on both providers.

## Part A: integrated problem framing and business evidence (backlog 14, P1)

Connect the built pieces (charter, framing, decision-cost model, ledger,
registry, loop) into one reviewable trail. Requirements:

- Business KPIs mapped to evaluation metrics; stakeholder-owned acceptance
  criteria, decision costs and expected value.
- Signed problem-framing and deployment approval records.
- Reproducible exploratory analysis, visual reports and experiment
  comparisons.
- Dataset qualification with domain thresholds and reviewer evidence.
- Observed business outcomes tracked against the approved expectation.
- Reviewable explanations of data-quality findings, performance trade-offs
  and deployment decisions.
- Independent confirmation of released analytical claims where required.

Acceptance: a reviewer traces one deployed model from the business question
through source data, model selection, approval, production predictions and
observed outcomes, using only retained records.

## Part B: qualify further task families and modalities (backlog 15, P2)

Expand through independent vertical slices, never by adding unsupported
algorithms to the production registry. Scope:

- Finish the audit of clustering, association analysis, dimensionality
  reduction and task-specific evaluation.
- Qualify forecasting, ranking and other task families against declared
  deployment requirements.
- Documents and NLP: ingestion, OCR confidence, citation grounding, serving.
- Image, audio and video: decoding, preprocessing, versioning, runtime
  controls.
- Geospatial: reprojection, raster operations, spatial joins, graph
  processing where required.
- Pretrained transformers, multimodal models and JEPA representations only
  where a use case is defined.
- Distributed or GPU training and serving only for workloads that justify it.
- Streaming: state the delivery semantics precisely and prove recovery; do
  not claim end-to-end exactly-once without evidence over the whole pipeline.
- Domain robustness, drift, explanation and monitoring benchmarks.

Acceptance: each new task or modality completes ingestion to serving,
including evaluation, packaging, cloud deployment, monitoring and recovery,
before it is labelled production-supported.

## External referent

For Part A, a stakeholder's recorded sign-off and a reviewer who did not build
the model. For Part B, a public or domain benchmark per slice and an
independent replication, as for F12.
