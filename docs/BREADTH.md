# Breadth: what is qualified, what is partial, what is not supported

No slice is qualified. This page says, for each task family and modality,
what exists in the repository, what does not, and what a slice needs before
it may be called production-supported. It is the audit F15 item 15 asks for,
and it is a status table, not a roadmap.

## The rule

A slice is **qualified** only when all three hold:

1. its robustness, drift, explanation and monitoring benchmark was defined
   before it was built (the files in
   [`benchmarks/breadth/`](../benchmarks/breadth/__init__.py) do that) and
   was then run, with a public or domain benchmark and an independent
   replication;
2. it completes ingestion, evaluation, packaging, deployment,
   monitoring and recovery;
3. its deployment requirement was declared by a named owner.

**Partial** means some code exists and is tested but not all of that. **Not
supported** means nothing in the repository does the job. An unsupported
algorithm is never added to the production registry; the guard tests in
`tests/unit/test_breadth.py` fail if one is.

## Status

`Built` is the word in the definition file. `Qualified` is `no` for every
slice. All fourteen benchmark definitions are `not_run`, and every threshold
in them is unset until the owner fixes it.

| Slice | Built | Qualified | What exists | What is missing |
| --- | --- | --- | --- | --- |
| `clustering` | `library_only` | no | `scmle.unsupervised.clustering.find_clusters`: k-means, Gaussian mixture and agglomerative at several counts, scored by silhouette, Davies-Bouldin and Calinski-Harabasz, with bootstrap stability (mean adjusted Rand index) and `external_scores`; unit tests on planted blobs and uniform noise | Nothing runs it: not in `Project`, the CLI, the charter run, a bundle, serving or monitoring. No packaging, no deployment, no drift, explanation or monitoring benchmark, no declared use |
| `association` | `none` | no | Nothing | Everything |
| `dimensionality_reduction` | `none` | no | Nothing as an estimator or task. The `select_features` search strategy picks columns for a supervised model; it is not an unsupervised representation | Everything |
| `forecasting` | `tabular_beta_path` | no | `Task.FORECAST`, forecasting estimators, the feature graph's lag and rolling operations, a rolling validation strategy; the tabular-beta capability registry lists baseline, linear, tree and boosting paths as supported with library inference only and no portable explanation or uncertainty contract; `benchmarks/real_tasks.py` compares one real series with last-value and seasonal-naive | Robustness, drift, explanation and monitoring benchmarks; a declared horizon and decision; deployment of a forecaster; an independent replication |
| `ranking` | `tabular_beta_path` | no | `Task.RANKING`, pointwise and LightGBM and XGBoost rankers, NDCG evaluation, the same registry entries and library-only limits | The same list as forecasting |
| `document_nlp` | `experimental_lifecycle` | no | Document and text adapters run the fifteen lifecycle stages locally with a signed report and a regularized linear baseline on character and token counts. Extracted elements and citations are data contracts | The baseline does no OCR and no semantic reading. No OCR confidence gate, no citation-grounding test, no benchmark, no served deployment |
| `image` | `experimental_lifecycle` | no | Image adapter and the same local lifecycle with a linear baseline on resized pixels | Decoding and perturbation benchmarks, versioned preprocessing, runtime controls proved on a deployment, explanation and drift benchmarks |
| `audio` | `experimental_lifecycle` | no | Audio adapter and the same local lifecycle with a baseline on bounded WAV statistics | No transcript, speaker or event claim; no noise-robustness or drift benchmark; no deployment |
| `video` | `experimental_lifecycle` | no | Video adapter and the same local lifecycle on evenly sampled frames or probed metadata | The same list as audio, plus decode failure handling at scale |
| `geospatial` | `experimental_lifecycle` | no | Vector and raster data contracts and adapters in the local lifecycle | Reprojection invariance, raster operations and spatial joins at scale, spatial-block evaluation as a gate, benchmarks, deployment |
| `graph` | `experimental_lifecycle` | no | Graph contracts, validation, inductive node splits and a node-feature baseline in the local lifecycle | The baseline ignores topology. No graph model, explainer or serving; no benchmark |
| `pretrained_multimodal` | `none` | no | Model-family declarations (transformer, text, vision, audio, video, multimodal, JEPA) that name a boundary without support, and an experimental tabular FT-Transformer and tabular JEPA behind an explicit opt-in | No pretrained weights, no multimodal model, no contamination check, no use case |
| `accelerated_training` | `none` | no | A job specification that requires a stated `gpu_reason`, and a Ray backend test | No workload that justifies a GPU or a cluster, no equivalence test across devices |
| `streaming` | `none` | no | A reader that streams a table in chunks within a memory limit, and a telemetry spool that delivers at least once | No event-streaming pipeline, no stated delivery semantics for one, no recovery proof |

The "experimental" modality lifecycles are baselines that prove the lifecycle
holds together on the author's machine. Their own reports say so and carry
the limits quoted above; the declaration they produce has the status
`experimental`, never `supported`, and only the contract module can name a
supported modality (a guard test checks this).

## Streaming and delivery semantics

Exactly-once streaming is not claimed. SCMLE has no event-streaming pipeline.
The word "streaming" in the code means reading a table that is larger than
memory in chunks. The one delivery path that exists, the telemetry spool,
delivers at least once, so a destination can see duplicates. Before any
streaming slice, the owner must state the semantics the consumer needs, and the
evidence must be a kill at every stage of the whole pipeline with the effects
compared with a batch computation; per-stage guarantees do not add up to an
end-to-end one.

## Benchmark definitions

Each file in `benchmarks/breadth/` defines four benchmarks (robustness,
drift, explanation, monitoring) with the damage or signal, the ground truth,
the metric and the pass rule, plus the deployment requirement to be declared,
the external referent and the lifecycle stages required. Parameters are null:
the owner fixes them before building, and never fits them to a result.
`tests/unit/test_breadth.py` checks that each file is well formed, that
nothing in it is run, fixed or declared, and that the cited evidence exists.

## Registry guard

`tests/unit/test_breadth.py` fails if the production registry gains a
supported capability outside tabular library inference, a supported
pretrained, multimodal or JEPA family, a supported streaming path, a
clustering, association, dimensionality-reduction, forecasting-specific or
deep modality model name in the model registry, a clustering or association
task type, or a supported modality declaration outside the contract module.
An adapter a caller registers is `extension-only`.
