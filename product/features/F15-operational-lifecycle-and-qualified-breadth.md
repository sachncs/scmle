# F15: Operational data science lifecycle and qualified breadth

Vocabulary and principles: [`../../plan/00-canon.md`](../../plan/00-canon.md).
Built in [Phase 18](../../plan/phase-18-lifecycle-evidence-and-breadth.md).
Part B starts after the F14 items have passed on both providers.

Built: the local half of Part A (`scmle.trail`, `scmle trail`,
`scripts/trace_model.py`, [`docs/TRAIL.md`](../../docs/TRAIL.md)) and the audit,
the benchmark definitions and the registry guard of Part B
([`docs/BREADTH.md`](../../docs/BREADTH.md), `benchmarks/breadth/`). This file
lists only what is open.

## Part A: integrated problem framing and business evidence (backlog 14, P1)

Open:

- A trail on a real deployment: a runtime-confirmed `production` release,
  production predictions and real business outcomes that arrive after a real
  delay, not a replay of held-out rows.
- Stakeholder-owned acceptance criteria, decision costs and the expected range
  signed before deployment, by a person who holds their own key.
- Reviewable written explanations of data-quality findings, performance
  trade-offs and deployment decisions (the trail records facts, not reasoning).
- Independent confirmation of released analytical claims where required.
- Source data, event logs and schema history behind the model (the committed
  trail starts from a flat extract).

Acceptance: a reviewer who did not build the model traces one deployed model
from the business question through source data, model selection, approval,
production predictions and observed outcomes, using only retained records. The
local trail passes `trace_model.py` when run by its author.

## Part B: qualify further task families and modalities (backlog 15, P2)

No slice is qualified. Open, one independent vertical slice at a time, never by
adding unsupported algorithms to the production registry:

- Clustering, association analysis and dimensionality reduction: wiring into
  runs and packaging (clustering is a library function only), an estimator
  where none exists, and task-specific evaluation.
- Forecasting and ranking: qualification against a declared deployment
  requirement (the task paths exist in the tabular beta).
- Documents and NLP: OCR confidence, citation grounding, serving.
- Image, audio and video: decoding, preprocessing, versioning, runtime
  controls.
- Geospatial: reprojection, raster operations, spatial joins, graph processing
  where required.
- Pretrained transformers, multimodal models and JEPA representations only
  where a use case is defined.
- Distributed or GPU training and serving only for workloads that justify it.
- Streaming: state the delivery semantics precisely and prove recovery; do not
  claim end-to-end exactly-once without evidence over the whole pipeline.
- Domain robustness, drift, explanation and monitoring benchmarks: the
  definitions exist and are unset and unrun; running them needs the owner's
  parameters and a public or domain benchmark per slice.

Acceptance: each new task or modality completes ingestion to serving,
including evaluation, packaging, cloud deployment, monitoring and recovery,
before it is labelled production-supported.

## External referent

For Part A, a stakeholder's recorded sign-off and a reviewer who did not build
the model. For Part B, a public or domain benchmark per slice and an
independent replication, as for F12.
