# Phase 18: Lifecycle evidence and qualified breadth

Feature: [F15](../product/features/F15-operational-lifecycle-and-qualified-breadth.md)
(backlog items 14 and 15). Starts, for item 15, only after the F14 items have
passed on both providers (P2 rule: no expansion before P0 is proven). Item 14
needed no new infrastructure.

The local half is built and verified without a cloud account.

* **Trail (item 14).** `scmle.trail`, `scmle trail build|verify|show` and
  `scripts/trace_model.py` join the charter, framing, decision-cost model,
  claim ledger, registry, release state and deployment ledger into one
  hash-chained, signed trail, and detect a missing link, a broken chain, an
  unsigned or unapproved step and an outcome outside the approved expectation
  ([`docs/TRAIL.md`](../docs/TRAIL.md)). Evidence: `plan/evidence/phase-18/local/`
  (regenerate with `scripts/make_phase18_evidence.py`): a real run on the
  Titanic extract through `frame`, `run`, `package`, the local registry and
  `scmle loop`.
* **Breadth (item 15), audit and definitions only.**
  [`docs/BREADTH.md`](../docs/BREADTH.md) states per slice what exists and what
  is missing, `benchmarks/breadth/` defines each slice's robustness, drift,
  explanation and monitoring benchmark before it is built, and guard tests keep
  unsupported algorithms out of the production registry. No slice is qualified.

What the local evidence is not: the analyst, stakeholder and reviewer were the
author with three keys; the observed outcomes replay held-out rows; the release
state is `approved`, not `production`; nothing ran on AWS or GCP.

## Open

- **WP-18.1 Independent trace, on a real deployment.** A reviewer who did not
  build the model traces one deployed model with `trace_model.py` and signs
  with a key the builder does not hold. The model runs behind a runtime that
  confirms `production` (Phase 16), serves real traffic, and its observed
  outcomes are real business outcomes that arrive after a real delay. The
  stakeholder's acceptance criteria and expected range are signed before
  deployment, not after the held-out evaluation. Still missing in the trail:
  reviewable written explanations of data-quality findings, performance
  trade-offs and deployment decisions (it records facts, not the reasoning);
  independent confirmation of a released analytical claim where required;
  point-in-time and source-data steps, because the Titanic extract has no
  source system, event log or schema history (canon: raw material).
- **WP-18.2 Breadth slices, one at a time.** For each of clustering,
  association, dimensionality reduction (audit finished: see `docs/BREADTH.md`),
  forecasting, ranking, documents and NLP, image, audio and video, geospatial
  and graph, pretrained and multimodal models, GPU and distributed training
  for a justified workload, and streaming: the deployment owner declares the
  requirement and fixes the parameters in the slice's definition file, the
  slice is built, its four benchmarks run against a public or domain benchmark
  with an independent replication, and it completes ingestion, evaluation,
  packaging, cloud deployment, monitoring and recovery on both providers.
  Streaming also needs its delivery semantics stated and proved over the whole
  pipeline.

## Gate (still open)

A reviewer who did not build the model traces one deployed model from the
business question to observed outcomes using only retained records. A slice is
production-supported only after it completes ingestion, evaluation, packaging,
cloud deployment, monitoring and recovery. The local half of the first gate
passes for the committed trail, run by its author.

## Evidence

`plan/evidence/phase-18/`.

## Not claimed

Unsupported algorithms are not added to the production registry. Exactly-once
streaming is not claimed without evidence over the whole pipeline. No breadth
slice is qualified, and the cross-provider P0 gate that item 15 waits on has
not passed.
